# zerollm

A zero-dependency, pure-code local LLM inference engine in Node.js (plus hand-emitted WebAssembly SIMD kernels) that runs GGUF models on a plain laptop CPU: no admin rights, no native addons, no cloud, no Ollama/llama.cpp.

Status: early development (M0-M3 done: doctor, GGUF reader, tokenizer, reference forward pass). See `docs/BRIEF.md` for the plan.

## Quick start

```
node bin/zerollm.js doctor                 # environment + feature probe
node bin/zerollm.js info models/x.gguf     # GGUF metadata, quant histogram, RAM estimate
npm test                                   # node --test
```

Download `.gguf` files manually (e.g. from Hugging Face) into `models/` (gitignored).

## Rules

Node built-ins only. No npm dependencies, no native addons, no external binaries.
