# next-gen-lab

Next-generation ways of generating text, run and measured on a laptop: an AMD Ryzen AI Max+ 392 with a Radeon 8060S and 64 GB, on Windows 11.

- **[ncm/](ncm/)**: Stanford's next-concept model, which writes a Sudoku solution coarse to fine in 12 parallel passes. On the same 50 hard boards, Claude Opus 5.5 solved 46, NCM 40 (0.47 s each on the CPU) and Claude Sonnet 5.5 32 (about 54 s each).
- **[diffusion/](diffusion/)**: Dream 7B and LLaDA 8B, which refine a whole answer over many steps, against Qwen2.5 7B, which writes one token at a time. Fluent diffusion text needed about one step per token; the ordinary model answered well 17 to 33 times sooner.
- **[colibri/](colibri/)**: DeepSeek V4 Flash (284B) run from the SSD with 16 GB of RAM, against the laptop's Qwen3.6 35B and Claude on 10 GSM8K problems. All four got 10 of 10; the 284B model took a median 362 s per answer (0.27 tok/s), the local model 4.2 s.

Written up at [sergioadan.dev](https://sergioadan.dev) under Cutting edge. The companion engine work is in [moe-stream](https://github.com/sadanbermudez/moe-stream).

MIT licence for this repository's code. NCM is MIT (Hazy Research); Dream, LLaDA and Qwen2.5 keep their own licences.
