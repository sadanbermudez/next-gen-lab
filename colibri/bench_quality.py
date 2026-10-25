"""Quality of a 284B model run from the SSD (colibri) against the laptop's local model and the chat leaders.

The same GSM8K problems (OpenAI grade-school maths, MIT licence; the first N of the test set) go to every model with
the same prompt; an answer counts if the number after "ANSWER:" equals the reference. Every result is appended to
results/<backend>.jsonl as it arrives, with the time, tokens and whether the laptop was on mains power, and a rerun
skips what is already there.

  python bench_quality.py --backend openai --url http://127.0.0.1:18434/v1 --model Qwen3.6-35B-A3B-UD-Q4_K_M --key-file <file> --name qwen36
  python bench_quality.py --backend openai --url http://127.0.0.1:8796/v1 --model deepseek-v4-colibri --name deepseek-v4-colibri
  python bench_quality.py --backend claude --model opus --name claude-opus

Thinking: off for the local models (the colibri server's default for DeepSeek V4; enable_thinking false for Qwen),
and `claude -p` as it runs by default.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = ("Solve this maths word problem. Show brief working, then give the final answer as a number on its own last "
          "line in the form ANSWER: <number>\n\n{q}")


def on_mains() -> bool | None:
    class S(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte), ("BatteryLifePercent", ctypes.c_byte),
                    ("Reserved1", ctypes.c_byte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = S()
    return bool(s.ACLineStatus == 1) if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)) else None


def reference(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def extract(reply: str) -> str | None:
    m = re.findall(r"ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", reply or "")
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def ask_openai(url, model, key, q, max_tokens, timeout):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": PROMPT.format(q=q)}]}
    if "qwen" in model.lower():
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", (d.get("usage") or {}).get("completion_tokens")


def ask_claude(model, q, timeout):
    # The user's own subscription through the CLI; no tools, and no API key, so billing cannot switch to the API.
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    out = subprocess.run([exe, "-p", "--model", model, "--output-format", "json", "--tools", ""], input=PROMPT.format(q=q),
                         capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    d = json.loads(out.stdout)
    return d.get("result", ""), (d.get("usage") or {}).get("output_tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("openai", "claude"), required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    key = Path(a.key_file).read_text(encoding="utf-8").strip() if a.key_file else ""
    problems = [json.loads(l) for l in (HERE / "gsm8k-test.jsonl").read_text(encoding="utf-8").splitlines()[:a.n]]
    out = HERE / "results" / f"{a.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["i"] for l in out.read_text(encoding="utf-8").splitlines() if '"error": null' in l} if out.exists() else set()
    for i, p in enumerate(problems):
        if i in done:
            continue
        ref, mains_before, t = reference(p["answer"]), on_mains(), time.time()
        try:
            if a.backend == "claude":
                reply, tokens = ask_claude(a.model, p["question"], a.timeout)
            else:
                reply, tokens = ask_openai(a.url, a.model, key, p["question"], a.max_tokens, a.timeout)
            err = None
        except Exception as e:
            reply, tokens, err = "", None, f"{type(e).__name__}: {str(e)[:200]}"
        got = extract(reply)
        rec = {"i": i, "name": a.name, "model": a.model, "ref": ref, "got": got, "correct": same(got, ref),
               "seconds": round(time.time() - t, 1), "tokens_out": tokens, "on_mains": mains_before and on_mains(),
               "error": err, "reply_tail": reply[-300:]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{a.name} {i}: {'right' if rec['correct'] else 'wrong'} ({got} vs {ref}) {rec['seconds']}s {tokens} tok"
              + (f" ERROR {err}" if err else ""), flush=True)


if __name__ == "__main__":
    main()
