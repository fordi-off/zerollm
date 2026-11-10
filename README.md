# zerollm

A local LLM engine and chat app in plain Node.js. It runs GGUF models on an ordinary laptop CPU with **zero npm dependencies, no native addons, no external programs, no admin rights and no cloud**. The fast paths are WebAssembly SIMD kernels that the project emits itself; a plain-JS reference path is kept as the oracle for them.

Two parts:

- **The engine** (`src/`, `bin/zerollm.js`): GGUF reader, tokenizers, chat templates, reference and multi-threaded forward passes, samplers, an on-disk system-prompt cache, and an embedding encoder.
- **Zerochat** (`app/`): a ChatGPT/Claude-style chat page served from `127.0.0.1` and opened in an app window. Norwegian UI by default, English switchable.

Requires **Node >= 22**.

## Zerochat, the app

```
node app/main.js
```

A window opens (Edge in app mode if found, otherwise the default browser); closing it quits the program after ~20 s. Options: `--data DIR`, `--port N`, `--no-open`, `--keep-alive`, `--quiet`. By default the data folder is the zerollm folder itself (wherever the command is run from). Put `.gguf` files in `models/`, or download one from Settings. Windows shortcut without a `.bat` file, folder layout and troubleshooting: [docs/APP-GUIDE.md](docs/APP-GUIDE.md) (Norwegian + English).

What it does:

- **Chat** with streaming, markdown, code highlighting and math (KaTeX: `\( \)`, `\[ \]`, `$$ $$`, `$ x $`); stop / continue / regenerate / copy, edit-and-resend, a context meter with exact token counts, light and dark theme.
- **Saved chats** in the same JSON format as Skibeti37 (`chats/`): list by date, rename, delete, restore the model; long chats are summarized automatically when the context window fills up.
- **Models**: model picker, a small download catalog with resumable downloads, per-model settings (display name, f16/f32 KV cache), RAM check before loading, the last used model is loaded in the background at start-up (switchable).
- **Thinking mode** for Qwen3 models ("Tenk først"): the reasoning streams in a foldable block above the answer and is stored with the chat, but not sent back to the model.
- **Attached files**: paperclip or drag-and-drop a text file; its whole text goes into the message, and because it then lives in the conversation the model reads it only once. The document has to fit the context window (the app says so before sending).
- **Slash commands** (`/humanize`, `/shorten`, `/norwegian`, `/proofread`, ...): plain `.md` files in `skills/`, same format as Skibeti37; add your own in Settings.
- **Speed tricks that matter on a CPU**: the prompt only ever grows at the end, so the model's memory (KV cache) of the earlier conversation stays valid and a follow-up only reads the new message; the system prompt is cached on disk across restarts.
- **Diagnostics tab**: system, WebAssembly features, models and the log tail, with a "copy report" button.

Security: the server binds to 127.0.0.1 only, checks the Host and Origin headers, needs a per-launch random token for every API call, and serves static files only from `app/public`. Third-party code in `app/public/vendor/` (marked, highlight.js, KaTeX) is copied unmodified with its licenses.

Manual test checklist for a real laptop with real models: [docs/APP-CHECKS.md](docs/APP-CHECKS.md).

## The engine from the command line

Download a `.gguf` (e.g. `Qwen3-0.6B-Q8_0.gguf`) from Hugging Face into `models/` (gitignored).

**Supported:** architectures `qwen3`, `qwen2`, `llama` (Llama 3.x, including fine-tunes such as the NB Norwegian ones); tensor types F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K (so Q4_K_M, Q5_K_M, Q6_K and Q8_0 files work; Q2_K/Q3_K/IQ*/Q4_1/Q5_1 do not). Not supported yet: Qwen3.5 (hybrid recurrent layers), Gemma, Mistral-style SentencePiece tokenizers.

```
node bin/zerollm.js doctor                              # environment + feature probe
node bin/zerollm.js info models/x.gguf                  # hyperparameters, quant histogram, RAM estimate with f32/f16 KV (--tensors, --metadata)
node bin/zerollm.js tokenize models/x.gguf "some text"  # token ids + pieces
node bin/zerollm.js run models/x.gguf -p "Why is the sky blue?" --n 200
node bin/zerollm.js chat models/x.gguf --system "Be brief."
node bin/zerollm.js bench models/x.gguf --kernels js,wasm --threads 1,2,4,6,8,12   # prefill/decode tok/s
node bin/zerollm.js embed models/nomic-embed-text-v1.5.Q4_K_M.gguf --selftest      # embedding model sanity check
npm test                                                # node --test (about 230 tests, no models needed)
```

`run` / `chat` options: `--n`, `--ctx 2048`, `--threads N` (default: logical processors / 2), `--kernels auto|wasm|js` (`--threads 1 --kernels js` = the plain single-thread reference path), `--temp 0.7` (0 = greedy), `--top-k 40`, `--top-p 0.9`, `--min-p`, `--repeat-penalty`, `--seed`, `--stop "str"`, `--batch 16` (prompt tokens per pass), `--kv f32|f16` (KV cache precision: f16 halves its memory), `--no-cache` / `--cache-dir DIR` (system-prompt KV cache, on by default for system prompts >= 32 tokens), `--think|--no-think` (Qwen3; default off), `--show-prompt`, `--raw` (no chat template), `--template qwen|llama3|raw`. Ctrl+C stops a reply cleanly. `chat` commands: `/reset`, `/system <text>`, `/think`, `/nothink`, `/exit`.

### As a library

`src/index.js` exports `loadModel`, `inspectModel`, `loadTokenizer`, `Engine` (`engine.generate({messages, sampling, stop, think, signal, stats})` is an async generator of text), the RAM estimator and the error types; the app uses only this surface.

### Embedding model (engine level)

`src/embed/` runs `nomic-bert` embedding models (nomic-embed-text v1/v1.5): WordPiece tokenizer, a reference encoder, and a data-parallel SIMD worker pool (`loadEmbedder`). It is **not used by the app** at the moment (the document search built on it was removed in favour of attaching whole files); the code, `zerollm embed`, `tools/embed-bench.mjs` and the tests are kept. It was written from llama.cpp's graph, with no reference output to compare against. Checked so far: it agrees with the plain-JS reference encoder on synthetic models, and a self-test on a real nomic-embed-text v1.5 Q4_K_M ranked related sentences above unrelated ones with clear margins (English clearly; Norwegian paraphrases only weakly, as expected from a mostly English model). The exact check against another implementation, `zerollm embed --compare reference.json` with `tools/ollama-reference.mjs`, has not been run.

## Measured on the target laptop (Ryzen 5 7530U, 16 GB, Windows, Node 26, 6 threads, plain `node`, no installs)

| model | file | decode | prefill | RAM |
|---|---|---|---|---|
| Qwen3 0.6B Q8_0 | 0.6 GB | ~28 tok/s | ~30 tok/s | 0.8 GB |
| Qwen3 4B Q4_K_M | 2.3 GB | ~5-6 tok/s | ~8 tok/s | 2.7 GB |
| Qwen3 8B Q4_K_M | 4.7 GB | ~3 tok/s | ~7 tok/s | 5.0 GB |

Laptop speed varies about +-15% from run to run; more models, the spin experiment, kernel micro-benchmarks and the embedding encoder in `docs/BENCHMARKS.md`. Reading the prompt (prefill) is the slow part of everything on large models, which is why the app is built to avoid re-reading.

## Tools (`tools/`)

`make-tiny-gguf.js` / `make-tiny-embed-gguf.js` (synthetic models so tests need no downloads), `kbench.mjs` (per-kernel speed of the extmul / wide / relaxed-SIMD integer dot forms, decode and batched), `dotbench.mjs` (interleaved model-level A/B of `ZEROLLM_DOT=extmul|relaxed|relaxed-add`, the opt-in relaxed-SIMD kernels, see `docs/BENCHMARKS.md`) and `abbench.mjs` (A/B of a CLI flag), `embed-bench.mjs`, `think-eval.mjs` (do Qwen3's plain and thinking modes get a set of maths questions right? run on a real model), `ollama-reference.mjs`, and `ui-smoke.mjs` (drives the app in headless Chromium and writes screenshots; needs Playwright installed globally, it is not a dependency of the project).

## Docs

| file | what |
|---|---|
| `docs/BRIEF.md` | goal, hard constraints, target hardware, design decisions, milestone plan of the engine |
| `docs/DECISIONS.md` | running log of every design decision, including what was tried and removed (calculator and code tools, document search) |
| `docs/BENCHMARKS.md` | all measured numbers |
| `docs/REAL-MODEL-CHECK.md` | how real-model correctness was checked |
| `docs/APP-PLAN.md` | the app's plan and per-milestone status |
| `docs/APP-GUIDE.md` | user guide (Norwegian + English) |
| `docs/APP-CHECKS.md` | manual test checklist for the app on a real machine |

## Rules

Node built-ins only. No npm dependencies, no native addons, no external binaries. ES modules, JSDoc types, no build step, tests with `node --test`. The JS reference path stays forever as the oracle for the fast path. Nothing about speed is claimed without a measurement.
