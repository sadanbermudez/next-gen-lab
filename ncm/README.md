# Next-concept modelling against chat models, on hard Sudoku

[NCM](https://github.com/HazyResearch/ncm) (Stanford / Hazy Research, MIT) writes a whole answer coarse to fine. A frozen tokenizer turns an answer into 12 levels of codes, from 1 at the top to 89 at the bottom, and a transformer predicts the levels top-down, every position in a level at once. A Sudoku board takes 12 passes through a 25M-parameter model.

## Results: the first 50 boards of the hard test set
| Model | Where it ran | Solved | Time per board |
|---|---|---|---|
| Claude Opus 5.5 | Anthropic (`claude -p`, subscription) | **46 / 50** | 53 s (median) |
| NCM Sudoku-hard, 25M | laptop CPU, one board at a time | 40 / 50 | **0.47 s** |
| Claude Sonnet 5.5 | Anthropic (`claude -p`, subscription) | 32 / 50 | 54 s (median) |

Free-tier models (Nemotron 3 Ultra 550B, Inkling, Gemma 4 31B through OpenRouter) take minutes per board; their rows are added as their runs finish.

NCM on 500 boards per level, laptop CPU: easy 99.2%, medium 97.6%, hard 85.2%. In batches of 100 that is 0.26 s per board; one board at a time, 0.47 s after an 11 s one-off load. All timings on mains power, AMD Ryzen AI Max+ 392, 64 GB, 24 October 2026.

Read it fairly:
- NCM was trained only on Sudoku. The chat models are general, and had no tools: given a code interpreter, any of them could write a solver.
- 50 boards is a small sample.

## Files
- `bench_llm.py`: gives each chat model the same prompt, checks the answer against the solution, and appends every result to `results/hard-<backend>-<model>.jsonl`. Failed requests are retried on the next run.
- `results/ncm-<level>-500.json`: NCM's 500 answers per level, with the references.
- `ncm-cpu-port.patch`: the change to HazyResearch/ncm (on c702742) that makes evaluation run on any device and loads every checkpoint with `torch.load(weights_only=True)`. All released Sudoku and TinyStories checkpoints load that way, and the dataset pickles contain only plain data.

## Reproduce
```
git clone https://github.com/HazyResearch/ncm && cd ncm && git am ../ncm-cpu-port.patch
# checkpoints and boards: hazyresearch/ncm-checkpoints (sudoku/*), hazyresearch/ncm-datasets (sudoku/*)
NCM_DEVICE=cpu python scripts/eval_shard.py --config config/eval/sudoku-hard.yaml --checkpoint-dir <ckpt>/sudoku/hard --data-dir <data>/sudoku/hard --n 500 --bs 100 --out eval/shard_0.pkl
python scripts/eval_merge.py --config config/eval/sudoku-hard.yaml --glob "eval/shard_*.pkl"

python bench_llm.py --backend claude --model opus --n 50
OPENROUTER_API_KEY=... python bench_llm.py --backend openrouter --model nvidia/nemotron-3-ultra-550b-a55b:free --n 50
```
