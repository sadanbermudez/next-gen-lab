# Training a small NCM from scratch

Hazy Research released NCM's models and evaluation, not its training code. `train.py` follows the paper: the released tokenizer stays frozen, and a new generator learns to predict each level of codes from the puzzle and the coarser levels, with 10% of the conditioning codes corrupted so it learns to recover from its own mistakes.

Checked first: fed the same inputs, the released 25M generator gets 94.8% of codes right teacher-forced and solves 84.5% of hard boards, so the inputs match how it was trained.

## First run: 3.3M parameters, 2 hours
4 layers, width 256, 4 heads. 200,000 generated hard boards (`sudoku_gen.py`, 30 givens, none from NCM's test sets), batch 128, 16,000 steps, on the laptop's integrated AMD GPU (ROCm, gfx1151) at about 285 boards a second. Scored every 2,000 steps on 200 boards of Stanford's hard test set.

| Step | Loss | Cells right | Boards solved |
|---|---|---|---|
| 2,000 | 0.76 | 64.0% | 0% |
| 4,000 | 0.66 | 69.0% | 0% |
| 8,000 | 0.56 | 75.0% | 0% |
| 12,000 | 0.51 | 76.4% | 0% |
| 16,000 | 0.50 | 77.0% | 0% |

It learned a good part of Sudoku (a guess gets about 11% of cells) but solved no board. A board counts only if all of its roughly 50 blanks are right, so per-cell accuracy compounds: whole solves appear only near the top of the curve. The loss was still falling when the learning rate reached zero, so this run was limited by its budget, not stuck. The released model is 8 times larger, and was trained far longer.

Full log: `results/small-4x256.jsonl`.
