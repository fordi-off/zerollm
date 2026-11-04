# zerollm

A zero-dependency, pure-code local LLM inference engine in Node.js (plus, later, hand-emitted WebAssembly SIMD kernels) that runs GGUF models on a plain laptop CPU: no admin rights, no native addons, no cloud, no Ollama/llama.cpp.

**Status:** M0–M8 done (doctor, GGUF reader/`info`, tokenizer + chat templates, reference forward pass verified on real Qwen3-0.6B, sampler, streaming `run`/`chat`, multi-threaded worker pool, hand-emitted WebAssembly SIMD kernels for Q8_0/Q4_0, batched prefill, on-disk system-prompt cache, Q5_0 + K-quants Q4_K/Q5_K/Q6_K so standard Q4_K_M / Q5_K_M files load). Next: 8B tuning (M9). See `docs/BRIEF.md` for the plan and `docs/DECISIONS.md` / `docs/BENCHMARKS.md` for the running log.

## Usage

Download a `.gguf` (e.g. `Qwen3-0.6B-Q8_0.gguf`) from Hugging Face into `models/` (gitignored). Supported: architectures `qwen3`, `qwen2`, `llama`; tensor types F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K (so Q4_K_M, Q5_K_M, Q6_K, Q8_0 files work; Q2_K/Q3_K/IQ*/Q4_1/Q5_1 do not).

```
node bin/zerollm.js doctor                              # environment + feature probe
node bin/zerollm.js info models/x.gguf                  # hyperparameters, quant histogram, RAM estimate (--tensors, --kv)
node bin/zerollm.js tokenize models/x.gguf "some text"  # token ids + pieces
node bin/zerollm.js run models/x.gguf -p "Why is the sky blue?" --n 200
node bin/zerollm.js chat models/x.gguf --system "Be brief."
node bin/zerollm.js bench models/x.gguf --kernels js,wasm --threads 1,2,4,6,8,12   # prefill/decode tok/s
npm test                                                # node --test
```

`run` / `chat` options: `--n`, `--ctx 2048`, `--threads N` (default: logical processors / 2), `--kernels auto|wasm|js` (`--threads 1 --kernels js` = the plain single-thread reference path), `--temp 0.7` (0 = greedy), `--top-k 40`, `--top-p 0.9`, `--min-p`, `--repeat-penalty`, `--seed`, `--stop "str"`, `--batch 16` (prompt tokens per pass), `--no-cache` / `--cache-dir DIR` (system-prompt KV cache, on by default for system prompts >= 32 tokens), `--think|--no-think` (Qwen3; default off), `--raw` (no chat template), `--template qwen|llama3|raw`. Ctrl+C stops a reply cleanly. `chat` commands: `/reset`, `/system <text>`, `/think`, `/nothink`, `/exit`.

## Rules

Node built-ins only. No npm dependencies, no native addons, no external binaries. Works on Node >= 22.
