"""Make Sudoku training boards in NCM's token format: a random solved grid, then cells removed down to --givens while
the puzzle keeps a unique solution. Boards whose solution appears in NCM's test sets are dropped.

  python sudoku_gen.py --n 200000 --givens 30 --out data/train-hard.pickle --exclude <ncm data dir>/sudoku

Format (as NCM's): puzzle = [11] + 9 rows of 9 joined by 10 + [11] (91 tokens, 0 = blank);
solution = 9 rows of 9 joined by 10 (89 tokens).
"""
import argparse, glob, os, pickle, random
from multiprocessing import Pool

ROWS = [[9 * r + c for c in range(9)] for r in range(9)]
COLS = [[9 * r + c for r in range(9)] for c in range(9)]
BOXES = [[9 * (3 * br + r) + 3 * bc + c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
PEERS = [sorted({p for u in (ROWS[i // 9], COLS[i % 9], BOXES[3 * (i // 27) + (i % 9) // 3]) for p in u} - {i}) for i in range(81)]


def count_solutions(g, limit=2):
    """Solutions of grid g (list of 81, 0 = blank), stopping at `limit`. Fewest-candidates-first DFS."""
    best, cand = -1, None
    for i in range(81):
        if g[i] == 0:
            used = {g[p] for p in PEERS[i]}
            c = [d for d in range(1, 10) if d not in used]
            if not c:
                return 0
            if cand is None or len(c) < len(cand):
                best, cand = i, c
                if len(c) == 1:
                    break
    if best < 0:
        return 1
    n = 0
    for d in cand:
        g[best] = d
        n += count_solutions(g, limit - n)
        if n >= limit:
            break
    g[best] = 0
    return n


def solved_grid(rng):
    g = [0] * 81

    def fill(i):
        if i == 81:
            return True
        used = {g[p] for p in PEERS[i]}
        digits = [d for d in range(1, 10) if d not in used]
        rng.shuffle(digits)
        for d in digits:
            g[i] = d
            if fill(i + 1):
                return True
        g[i] = 0
        return False

    fill(0)
    return g


def make(args):
    seed, givens = args
    rng = random.Random(seed)
    while True:
        sol = solved_grid(rng)
        puz = sol[:]
        order = list(range(81))
        rng.shuffle(order)
        filled = 81
        for i in order:
            if filled == givens:
                break
            keep, puz[i] = puz[i], 0
            if count_solutions(puz[:]) != 1:
                puz[i] = keep
            else:
                filled -= 1
        if filled == givens:                                 # some grids cannot get this low with a unique solution
            return puz, sol


def to_tokens(g):
    out = []
    for r in range(9):
        out += g[9 * r:9 * r + 9] + ([10] if r < 8 else [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="", help="NCM sudoku data dir: drop boards whose solution is in a test set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()

    banned = set()
    for f in glob.glob(os.path.join(a.exclude, "*", "solutions.pickle")) if a.exclude else []:
        banned |= {tuple(s) for s in pickle.load(open(f, "rb"))}
    puzzles, solutions, dropped = [], [], 0
    with Pool(a.procs) as pool:
        for k, (puz, sol) in enumerate(pool.imap_unordered(make, [(a.seed * 10_000_000 + i, a.givens) for i in range(a.n)], chunksize=64)):
            s = to_tokens(sol)
            if tuple(s) in banned:
                dropped += 1
                continue
            puzzles.append([11] + to_tokens(puz) + [11])
            solutions.append(s)
            if (k + 1) % 10000 == 0:
                print(f"{k + 1}/{a.n}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump({"puzzles": puzzles, "solutions": solutions, "givens": a.givens}, open(a.out, "wb"))
    print(f"{len(puzzles)} boards -> {a.out}; {dropped} dropped as test-set solutions")


if __name__ == "__main__":
    main()
