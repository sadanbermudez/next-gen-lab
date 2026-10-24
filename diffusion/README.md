# dlm-lab: diffusion language models against a normal one, on a laptop

Same prompt for every model: "Explain in three sentences why a mixture-of-experts model needs less compute per token than a dense model of the same size." Greedy decoding (temperature 0), seed 1, Q8_0 weights, llama.cpp (Vulkan) on an AMD Radeon 8060S (Ryzen AI Max+ 392, 64 GB), 24 October 2026. Full outputs and logs are in `runs/`.

## Results
| Model | How it writes | Steps | Time | Answer |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | one token at a time | 105 tokens | **3.8 s** (27.9 tok/s) | Correct and fluent |
| Dream-v0-Instruct-7B (from Qwen2.5-7B) | diffusion, entropy order | 32 | 11 s | Word salad |
| | | 64 | 24 s | Fragments |
| | | 128 | 48 s | Mostly readable, stutters |
| | | 256 | 124 s | Fluent, partly wrong |
| LLaDA-8B-Instruct | diffusion, 32-token blocks | 32 | 8 s | Broken |
| | | 64 | 16 s | Readable, drifts off topic |
| | | 128 | 32 s | Readable, small errors |
| | | 256 | 64 s | Fluent, partly wrong |

## What it shows
- **Steps are the quality knob.** A diffusion model fills a whole answer in parallel, but with fewer refinement steps than the answer has tokens, the text falls apart. Fluent text needed about one step per token, which removes the speed advantage.
- **Block scheduling degrades more gently.** LLaDA, which finishes the answer 32 tokens at a time, stays readable at 64 steps; Dream, refining the whole answer at once, does not.
- **On this hardware the normal model wins** on both speed and accuracy, by 17 to 33 times in time to a fluent answer.
- **Caveat:** llama.cpp's diffusion code is a reference implementation that recomputes the whole sequence at every step with no cache. Optimised diffusion serving on datacenter GPUs is much faster, so the ratio here is not a verdict on the method. The trade-off between steps and quality is.

This is the gap next-concept modelling (NCM) aims at: a few coarse-to-fine levels instead of hundreds of refinement steps. See [../ncm](../ncm).

## Reproduce
```
llama-diffusion-cli -m Dream-org_Dream-v0-Instruct-7B-Q8_0.gguf -p "<prompt>" -ub 320 --diffusion-eps 0.001 --diffusion-algorithm 1 --diffusion-steps N -ngl 99 --temp 0 --seed 1
llama-diffusion-cli -m LLaDA-8B-Instruct.Q8_0.gguf -p "<prompt>" -ub 256 --diffusion-block-length 32 --diffusion-steps N -ngl 99 --temp 0 --seed 1
llama-completion -m Qwen2.5-7B-Instruct-Q8_0.gguf -p "<prompt>" -cnv -st -n 216 -ngl 99 --temp 0 --seed 1
```
With block scheduling, N must divide by (max length / block length); `-ub 256` gives 8 blocks.
