# zerollm

A zero-dependency, pure-code local LLM inference engine in Node.js (plus, later, hand-emitted WebAssembly SIMD kernels) that runs GGUF models on a plain laptop CPU: no admin rights, no native addons, no cloud, no Ollama/llama.cpp.

**Status:** M0–M9 done and verified on real Qwen3 0.6B / 4B / 8B files (doctor, GGUF reader/`info`, tokenizer + chat templates, reference forward pass verified on real Qwen3-0.6B, sampler, streaming `run`/`chat`, multi-threaded worker pool, hand-emitted WebAssembly SIMD kernels for Q8_0/Q4_0, batched prefill, on-disk system-prompt cache, Q5_0 + K-quants Q4_K/Q5_K/Q6_K so standard Q4_K_M / Q5_K_M files load). SIMD attention with per-worker KV caches, optional f16 KV cache, RAM estimates before loading. Next: real 8B measurements on the target laptop, then optional server/context shifting (M10). See `docs/BRIEF.md` for the plan and `docs/DECISIONS.md` / `docs/BENCHMARKS.md` for the running log.

## Usage

Download a `.gguf` (e.g. `Qwen3-0.6B-Q8_0.gguf`) from Hugging Face into `models/` (gitignored). Supported: architectures `qwen3`, `qwen2`, `llama`; tensor types F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K (so Q4_K_M, Q5_K_M, Q6_K, Q8_0 files work; Q2_K/Q3_K/IQ*/Q4_1/Q5_1 do not).

```
node bin/zerollm.js doctor                              # environment + feature probe
node bin/zerollm.js info models/x.gguf                  # hyperparameters, quant histogram, RAM estimate with f32/f16 KV (--tensors, --metadata)
node bin/zerollm.js tokenize models/x.gguf "some text"  # token ids + pieces
node bin/zerollm.js run models/x.gguf -p "Why is the sky blue?" --n 200
node bin/zerollm.js chat models/x.gguf --system "Be brief."
node bin/zerollm.js bench models/x.gguf --kernels js,wasm --threads 1,2,4,6,8,12   # prefill/decode tok/s
npm test                                                # node --test
```

`run` / `chat` options: `--n`, `--ctx 2048`, `--threads N` (default: logical processors / 2), `--kernels auto|wasm|js` (`--threads 1 --kernels js` = the plain single-thread reference path), `--temp 0.7` (0 = greedy), `--top-k 40`, `--top-p 0.9`, `--min-p`, `--repeat-penalty`, `--seed`, `--stop "str"`, `--batch 16` (prompt tokens per pass), `--kv f32|f16` (KV cache precision: f16 halves its memory), `--no-cache` / `--cache-dir DIR` (system-prompt KV cache, on by default for system prompts >= 32 tokens), `--think|--no-think` (Qwen3; default off), `--raw` (no chat template), `--template qwen|llama3|raw`. Ctrl+C stops a reply cleanly. `chat` commands: `/reset`, `/system <text>`, `/think`, `/nothink`, `/exit`.

## Measured on the target laptop (Ryzen 5 7530U, 16 GB, Windows, Node 26, 6 threads, plain `node`, no installs)

| model | file | decode | prefill | RAM |
|---|---|---|---|---|
| Qwen3 0.6B Q8_0 | 0.6 GB | ~28 tok/s | ~30 tok/s | 0.8 GB |
| Qwen3 4B Q4_K_M | 2.3 GB | ~5-6 tok/s | ~8 tok/s | 2.7 GB |
| Qwen3 8B Q4_K_M | 4.7 GB | ~3 tok/s | ~7 tok/s | 5.0 GB |

Laptop speed varies about +-15% from run to run; details and method in `docs/BENCHMARKS.md`.

## Next: the app

Plan for a ChatGPT/Claude-style local chat app (pure Node + HTML, zero dependencies, on this engine) with the functionality of Skibeti37: `docs/APP-PLAN.md`.

## Rules

Node built-ins only. No npm dependencies, no native addons, no external binaries. Works on Node >= 22.
