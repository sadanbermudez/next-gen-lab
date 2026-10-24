"""Hard Sudoku: chat models against NCM on the same boards.

Every model gets the same prompt for the first N boards of the NCM hard test set and must return the 81 digits.
The boards come from Hugging Face (hazyresearch/ncm-datasets, sudoku/) into ./data/sudoku/<level>/.
An answer counts as solved only if it equals the reference solution (the puzzles have unique solutions).
Each answer is appended to results/<model>.jsonl as it arrives, so a run resumes where it stopped.

  python bench_llm.py --backend openrouter --model nvidia/nemotron-3-ultra-550b-a55b:free --n 50
  python bench_llm.py --backend claude --model sonnet --n 50
  python bench_llm.py --backend openai --base-url http://127.0.0.1:18434/v1 --model local --n 50
"""
import argparse, json, os, pickle, re, subprocess, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROW, BLANK = 10, 0

PROMPT = """Solve this Sudoku. Each row, column and 3x3 box must contain the digits 1-9 exactly once. Dots are empty cells.

{grid}

Give your final answer as the 9 completed rows, one row of 9 digits per line, between the lines ANSWER and END, like this:
ANSWER
123456789
...
END"""


def board(tokens):
    """NCM puzzle tokens (11 start/end, 10 row break, 0 blank) to nine strings of digits and dots."""
    cells = [t for t in tokens if t not in (ROW, 11)]
    assert len(cells) == 81
    return ["".join(str(c) if c else "." for c in cells[r * 9:r * 9 + 9]) for r in range(9)]


def solution(tokens):
    cells = [t for t in tokens if t not in (ROW, 11)]
    assert len(cells) == 81
    return "".join(map(str, cells))


def parse(text):
    """The 81 digits between ANSWER and END, or the last nine 9-digit lines of the reply."""
    m = re.findall(r"ANSWER\s*(.*?)\s*END", text or "", re.S)
    body = m[-1] if m else (text or "")
    rows = re.findall(r"^[\s|]*((?:[1-9][\s|,]*){9})\s*$", body, re.M)
    rows = ["".join(re.findall(r"[1-9]", r)) for r in rows]
    return "".join(rows[-9:]) if len(rows) >= 9 else ""


def openrouter_key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if not k:
        raise SystemExit("set OPENROUTER_API_KEY for the openrouter backend")
    return k


def ask_openai(base, key, model, prompt, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    if "choices" not in j:
        raise RuntimeError(str(j.get("error", j))[:300])
    u = j.get("usage") or {}
    return j["choices"][0]["message"].get("content") or "", u.get("prompt_tokens"), u.get("completion_tokens")


def ask_claude(model, prompt, timeout):
    # The user's own Claude subscription through the CLI; no tools, no API key (so billing cannot switch to the API).
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    p = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", "--tools", ""],
                       input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    j = json.loads(p.stdout)
    u = j.get("usage") or {}
    return j.get("result") or "", u.get("input_tokens"), u.get("output_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openrouter", "claude", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--level", default="hard")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between requests (free-tier rate limits)")
    a = ap.parse_args()

    puzzles = pickle.load(open(os.path.join(HERE, "data", "sudoku", a.level, "puzzles.pickle"), "rb"))[:a.n]
    refs = pickle.load(open(os.path.join(HERE, "data", "sudoku", a.level, "solutions.pickle"), "rb"))[:a.n]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{a.level}-{a.backend}-{re.sub(r'[^A-Za-z0-9.-]+', '_', a.model)}.jsonl")
    # Answers that came back count as done; failed requests (rate limits, timeouts) are tried again on the next run.
    kept = [l for l in open(out, encoding="utf-8") if not json.loads(l)["error"]] if os.path.exists(out) else []
    if os.path.exists(out):
        open(out, "w", encoding="utf-8").writelines(kept)
    done = {json.loads(l)["i"] for l in kept}
    key = openrouter_key() if a.backend == "openrouter" else ""

    for i, (pz, ref) in enumerate(zip(puzzles, refs)):
        if i in done:
            continue
        prompt = PROMPT.format(grid="\n".join(board(pz)))
        t0, err, text, tin, tout = time.time(), None, "", None, None
        for attempt in range(3):
            try:
                if a.backend == "claude":
                    text, tin, tout = ask_claude(a.model, prompt, a.timeout)
                else:
                    base = a.base_url or "https://openrouter.ai/api/v1"
                    text, tin, tout = ask_openai(base, key, a.model, prompt, a.timeout)
                err = None
                break
            except Exception as e:  # rate limits and timeouts: wait and retry, then record the failure
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(20 * (attempt + 1))
        got = parse(text)
        rec = {"i": i, "model": a.model, "backend": a.backend, "solved": got == solution(ref),
               "cells": sum(x == y for x, y in zip(got, solution(ref))) / 81 if got else 0.0,
               "parsed": bool(got), "seconds": round(time.time() - t0, 2), "tokens_in": tin, "tokens_out": tout, "error": err,
               "reply_tail": (text or "")[-400:]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{i:3d} {'SOLVED' if rec['solved'] else 'wrong ' if got else 'noparse'} {rec['seconds']:7.1f}s" + (f"  {err}" if err else ""), flush=True)
        time.sleep(a.pause)

    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    s = sum(r["solved"] for r in rows)
    print(f"{a.model}: solved {s}/{len(rows)}  median {sorted(r['seconds'] for r in rows)[len(rows) // 2]:.1f}s")


if __name__ == "__main__":
    main()
