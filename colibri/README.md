# colibri: a 284B model from the SSD, against the laptop's own model and the chat leaders

[colibri](https://github.com/JustVugg/colibri) runs DeepSeek V4 Flash (284B parameters, mixture of experts) on a machine with far less memory than the model: it keeps 16 GB of it in RAM and reads the experts each token needs from the SSD. The question here is what that buys over a model that fits.

Same test for every model: the first 10 problems of the GSM8K test set (grade-school maths, MIT licence), the same prompt, and an answer counts if the number after "ANSWER:" equals the reference. Thinking off for the local models. AMD Ryzen AI Max+ 392, Radeon 8060S, 64 GB, on mains power, 24 and 25 October 2026. Every reply is in `results/`.

## Results
| Model | Where it runs | Right | Median time | Speed |
|---|---|---|---|---|
| Qwen3.6 35B-A3B (Q4) | laptop GPU, fits in memory | **10 / 10** | **4.2 s** | 64 tok/s |
| Claude Sonnet 5.5 | cloud | 10 / 10 | 4.3 s | n/a |
| Claude Opus 5.5 | cloud | 10 / 10 | 5.7 s | n/a |
| DeepSeek V4 Flash 284B, colibri | laptop CPU, experts from the SSD | 10 / 10 | 362 s | 0.27 tok/s |

Cloud times include the network and the Claude Code CLI, so they are times to an answer, not model speed.

## What it shows
- **colibri works.** A model about eight times larger than the laptop's 16 GB budget for it answered every problem correctly, on a laptop.
- **On this test, size bought nothing.** The 35B local model was as accurate and about 85 times faster. Grade-school maths is too easy to separate them; the difference, if any, is on harder problems.
- **The cost is the SSD.** Each token needs experts that are not in RAM, so speed is set by disk reads: 0.27 tok/s here, against 0.48 tok/s on shorter prompts in the earlier RAM sweep (16, 24 and 32 GB were the same speed).
- **Use it when nothing else fits:** an overnight or "think harder" job where waiting minutes is acceptable and the question needs a larger model. In Argus it runs on demand as the "Think harder" service, which first unloads the local model.

## Reproduce
```
python bench_quality.py --backend openai --url http://127.0.0.1:18434/v1 --model Qwen3.6-35B-A3B-UD-Q4_K_M --key-file <file> --name qwen36
python bench_quality.py --backend openai --url http://127.0.0.1:8796/v1 --model deepseek-v4-colibri --name deepseek-v4-colibri --timeout 7200
python bench_quality.py --backend claude --model opus --name claude-opus
```
`run_colibri.py` does the colibri run end to end: it starts Argus's Think harder service, runs the benchmark and stops the service. A rerun skips the problems already answered.
