# zerollm

A zero-dependency, pure-code local LLM inference engine in Node.js (plus, later, hand-emitted WebAssembly SIMD kernels) that runs GGUF models on a plain laptop CPU: no admin rights, no native addons, no cloud, no Ollama/llama.cpp.

**Status:** M0–M4 done (doctor, GGUF reader/`info`, tokenizer + chat templates, single-thread reference forward pass, sampler, streaming `run`/`chat`). The engine works but is single-threaded plain JS; workers (M5) and WASM SIMD (M6) are next. See `docs/BRIEF.md` for the plan and `docs/DECISIONS.md` / `docs/BENCHMARKS.md` for the running log.

## Usage

Download a `.gguf` (e.g. `Qwen3-0.6B-Q8_0.gguf`) from Hugging Face into `models/` (gitignored). Supported now: architectures `qwen3`, `qwen2`, `llama`; tensor types F32, F16, BF16, Q8_0, Q4_0 (K-quants come in M8).

```
node bin/zerollm.js doctor                              # environment + feature probe
node bin/zerollm.js info models/x.gguf                  # hyperparameters, quant histogram, RAM estimate (--tensors, --kv)
node bin/zerollm.js tokenize models/x.gguf "some text"  # token ids + pieces
node bin/zerollm.js run models/x.gguf -p "Why is the sky blue?" --n 200
node bin/zerollm.js chat models/x.gguf --system "Be brief."
npm test                                                # node --test
```

`run` / `chat` options: `--n`, `--ctx 2048`, `--temp 0.7` (0 = greedy), `--top-k 40`, `--top-p 0.9`, `--min-p`, `--repeat-penalty`, `--seed`, `--stop "str"`, `--think|--no-think` (Qwen3; default off), `--raw` (no chat template), `--template qwen|llama3|raw`. Ctrl+C stops a reply cleanly. `chat` commands: `/reset`, `/system <text>`, `/think`, `/nothink`, `/exit`.

## Rules

Node built-ins only. No npm dependencies, no native addons, no external binaries. Works on Node >= 22.
