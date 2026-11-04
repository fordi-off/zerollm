# App plan: a ChatGPT/Claude-style local chat app on top of zerollm

Status: plan (nothing built yet). Written 2026-11-04 after reading the Skibeti37 repository (fordi-off/Skibeti37, Python + pywebview + llama-cpp-python, with a browser/wllama fallback).

## 1. Goal

A simple, clean local chat app that looks and feels like ChatGPT / Claude.ai in a browser window, with the functionality of Skibeti37 (chats, documents, skills, model management), running on **zerollm** only:

- **Node.js only**, zero npm dependencies, no native addons, no Python, no admin rights, no internet at runtime (model downloads excepted, explicit and user-triggered).
- One process: `node app/main.js` starts a local HTTP server (127.0.0.1) + the zerollm engine and opens a window. UI = plain HTML/CSS/JS.
- Norwegian first (UI strings, default prompts, NB Llama as the headline model), English available.
- Works on the locked-down target machine (confirmed: `node.exe` runs there).

Non-goals (for now): GPU, multi-user, remote access, mobile layout, plugins/MCP, image input, anything that needs a model architecture zerollm does not implement.

## 2. What we reuse from Skibeti37 (and what we do not)

Reuse (it is the owner's own MIT-licensed code): the **data formats** (`chats/*.json`, `documents/`, `skills/*.md`, `models/*.gguf`, `config.json`, so both apps can share one folder), the **skill file format**, the **prompts** (English instructions, language-matching rule), the **generation safeguards** (loop detection, context fitting, continuation prompt, sampling defaults), the vendored **marked / highlight.js / KaTeX** assets, the **download catalog idea**, the Norwegian UI strings.

Do not reuse: the pywebview bridge, the Python process, `wasm-engine.js` (530 lines of duplicated generation/RAG logic), the three-column layout (chats+documents / chat / skills) which is more cluttered than a ChatGPT-style page.

## 3. Feature parity checklist

| Feature in Skibeti37 | New app | Milestone |
|---|---|---|
| Chat with streaming, markdown, code highlighting, math (KaTeX), copy buttons | yes | A1 |
| Stop, continue (cut off / stopped), regenerate, edit-and-resend with confirmation | yes | A1-A2 |
| Model picker in the header, last model remembered | yes | A1 |
| Context meter (real tokenizer counts), fit-to-context trimming | yes (exact counts: the tokenizer is ours) | A1, A5 |
| Chats: auto-save after first reply, list, rename, delete, restore model + documents | yes | A2 |
| First-run: no model -> friendly download card; guided model downloader (queue, resume, progress, ETA) | yes (Node https + Range, no curl needed) | A3 |
| Settings: models (rename, active, delete, KV precision), performance (context size, threads), reset | yes | A3 |
| RAM check before loading with clear Norwegian message | yes (zerollm estimator) | A3 |
| Slash commands / skills (markdown files, autocomplete, chips under messages) | yes | A4 |
| Rolling conversation summary after 10 messages (background model) | yes | A5 |
| Documents: upload .txt/.md, chunk, search, attach/detach per chat, "Kilder" (sources) under replies, library + preview + delete | yes (retrieval method: see section 6) | A6 |
| Document map-reduce summaries (background model) | yes, optional | A6 |
| Windowless launch, shortcut instructions | Edge `--app` window via a shortcut | A0, A7 |
| Browser/WASM fallback mode | **not needed**: there is only one mode | - |
| GPU toggle | dropped | - |
| "Drop any .gguf into models/ and it works" | **only architectures zerollm implements** (llama, qwen2, qwen3); the app says so clearly when a file is unsupported | A3 |
| Qwen3.5 models, nomic-embed | blocked on engine work (section 6) | A8 |
| PDF/DOCX upload, export to Markdown/PDF (Skibeti37 "future improvements") | later | A8 |

## 4. Architecture

```
 browser window (Edge --app / default browser)  <-- plain HTML/CSS/JS, no build step
        |  fetch() JSON  +  Server-Sent Events (token stream, downloads, status)
        v
 app/server.js   zero-dependency node:http server, bound to 127.0.0.1
        |   routes = the 36 methods of Skibeti37's api.py (same names, JSON in/out)
        v
 app/core/*.js   chats, documents, skills, config, downloader, context (pure JS, unit tested)
        |
        v
 app/engine.js   thin wrapper over zerollm: model residency, generation, tokenize, cancel
        |
        v
 zerollm (src/)  PoolModel (worker_threads + WASM SIMD), tokenizer, sampler, prefix reuse
```

Key decisions:

- **Repo layout:** `app/` inside this repo for now (I can only push to fordi-off/zerollm); it can be split into its own repo later because it only imports zerollm's public entry (`src/index.js`, built in A1).
- **Transport:** requests are plain `fetch` POSTs; streamed events use one SSE connection (`/api/events`), which replaces pywebview's `evaluate_js` pushes and is not throttled in background tabs. Generation = POST returns immediately, chunks arrive as `chunk` / `phase` / `progress` / `done` / `error` events.
- **Model residency:** at most one big chat model resident; the background (utility) model is loaded only while summarizing and freed afterwards (same policy as Skibeti37, important for 16 GB machines). A model switch = `close()` the pool, then create the new one. The RAM estimator gates every load.
- **Prompt layout for speed:** the system prompt + skills + conversation history come first and retrieved document excerpts last, so zerollm's KV prefix reuse keeps the history cached between turns; only new tokens are prefilled. The on-disk system-prompt cache stays.
- **Prefill progress in the UI:** zerollm already reports prefill progress; the UI shows "Leser teksten... 340/2700" with an ETA instead of a silent wait (Skibeti37's main UX complaint).
- **Security of a local server:** bind to 127.0.0.1 only; random per-launch token required on every API call (header) and checked `Host`/`Origin`, so other websites open in the same browser cannot call the API; no CORS; static files served from an allow-list; path traversal-safe model/document names.
- **Launcher:** `app/main.js` picks a free port, starts the server, opens `msedge.exe --app=http://127.0.0.1:PORT/?t=TOKEN` (falls back to the default browser), single-instance lock file, exits shortly after the window closes (heartbeat). Because `.bat/.cmd` are blocked on some managed PCs, the documented launcher is a **Windows shortcut** to `node.exe app\main.js` (Run: minimized), exactly like Skibeti37's `.pyw` shortcut instructions.
- **Config/data:** same folder layout and JSON shapes as Skibeti37; new fields are additive so both apps keep working on one data folder.

## 5. UI design (ChatGPT/Claude-like, "really easy")

```
+----------------+---------------------------------------------------+
| + Ny samtale   |  [ NB Llama 3.1 8B v ]                      gear  |
|                +---------------------------------------------------+
| I dag          |                                                   |
|  . Brunost...  |        (conversation: wide readable column)       |
|  . Oppgave...  |   user message                       (right, soft)|
| I gar          |   assistant reply, markdown, code, math           |
|  . ...         |   [copy] [regenerate] [kilder: 3 utdrag v]        |
|                |                                                   |
| Dokumenter     +---------------------------------------------------+
| Kommandoer     | [clip] fil.pdf x      +-------------------------+ |
| Innstillinger  |  /  for kommandoer    | Skriv en melding...  >  | |
|                |  ====---- 3400 / 8192 +-------------------------+ |
+----------------+---------------------------------------------------+
```

- One sidebar (collapsible): new chat, chats grouped by date, and three small entries (Dokumenter, Kommandoer, Innstillinger) that open panels/modals. No right-hand column.
- Header: model picker + settings. Everything advanced lives in Settings (tabs Modeller / Dokumenter / Skills / Ytelse, as in Skibeti37).
- Input: auto-growing textarea, Enter sends, Shift+Enter newline, clip button attaches a document to this chat (chips above the input), `/` opens the command menu, the context meter is a thin bar under the input.
- Empty state: a few clickable Norwegian starter prompts; if no model is installed, a single card "Last ned en modell" with the recommended model for the detected free RAM.
- Visual: system font stack, light/dark (follows OS), generous whitespace, one accent colour, no icon fonts (inline SVG). Plain CSS, no framework.
- Norwegian strings in one `i18n/nb.json` (+ `en.json`); no string literals in the code.
- I can render the page headlessly (Playwright is available in my environment, dev only) and iterate on screenshots; the final look is judged by the owner on the laptop.

## 6. Engine work in zerollm that the app needs

| # | Item | Needed by | Size |
|---|---|---|---|
| E1 | **Public library entry** `src/index.js`: `loadModel(path, {ctx, threads, kv})` -> `{generate(messages \| rawPrompt, {sampling, stop, signal, onProgress}), tokenize, countTokens, close, info}`; async-iterable token stream; cancel via AbortSignal | A1 | S |
| E2 | Sampling parity with Skibeti37: temperature, top_p, repeat_penalty over a 320-token window, presence/frequency penalty, stop strings (all exist; expose + test) | A1 | S |
| E3 | Chat templates: keep the hand-written Qwen/ChatML and Llama 3 renderers, add an optional per-model template override and a "raw prompt" mode (the app primes `<think></think>` itself for Qwen) | A1 | S |
| E4 | Model residency helper (load/unload with RAM gate, safe concurrent requests = queue) | A1/A3 | S |
| E5 | Retrieval without an embedding model: BM25 over word + character n-gram tokens (handles Norwegian compounds), pure JS | A6 | M |
| E6 | **Embedding model support** (nomic-embed-text = BERT-style encoder, WordPiece tokenizer, mean pooling) for semantic search | A8 | M-L |
| E7 | **Qwen3.5** (Skibeti37's default family): needs inspection of a real GGUF first, may require new layer types; unknown size | A8 | ? |
| E8 | Throughput work for RAG-sized prompts (K-quant kernels, SIMD activation quantizer); measured, not guessed | ongoing | M |

Honest performance note: on the target laptop zerollm does ~3 tok/s decode and ~7 tok/s prefill for an 8B Q4_K_M. A RAG prompt of 5 chunks x 300 Norwegian words is ~2,700 tokens = about 6 minutes of prefill. So the plan keeps retrieval small by default (3 chunks of ~200 words), puts it last in the prompt, shows progress, and recommends a 4B/1B-class model for document questions. Measuring llama.cpp on the same laptop (the `done: ... tok/s` lines in Skibeti37's log) tells us how much speed is being traded for "no native code".

Why BM25 first: it needs no model, no extra RAM, works on the locked-down machine today, and for a personal document library with Norwegian text it is a reasonable baseline; semantic embeddings (E6) become an optional upgrade, not a blocker.

## 7. Milestones

**Status (2026-11-04):** A0 done. A1 built and tested against the synthetic model (server, API, streaming chat, stop/continue/regenerate, context meter, prefill progress, settings, light/dark, phone width); what is still open for A1 is the manual check on the laptop with the NB Llama 8B (`docs/APP-CHECKS.md`). Math (KaTeX) is deferred.
**A2 built (2026-11-04):** chats auto-save after every reply in Skibeti37's file format (system message first, extras and `attached_docs` preserved, epoch-second `updated_at`), sidebar list grouped by today/yesterday/earlier, open, rename, two-click delete (also removes the summary cache), edit-and-resend with a warning when later replies will be removed, switching chats stops a running reply first. Tested with a Skibeti37-style file and a scripted browser flow (`tools/ui-smoke.mjs`); still to check by hand: opening a chat file written by the real Python app. Start with `npm run app` (or `node app/main.js --data <folder>`).

Every milestone leaves the app runnable and tested. S/M/L = relative size.

- **A0 Foundations (S).** `app/` skeleton; zero-dep HTTP server with token + Origin checks; static serving; SSE channel; config + data folder; launcher (Edge `--app`, free port, single instance, heartbeat shutdown); `node --test` setup; a synthetic tiny model fixture so tests never need real models. *Done when:* `node app/main.js` opens a blank styled page that talks to the API; tests pass.
- **A1 Chat vertical slice (M).** E1-E4; model scan + load with RAM gate; streaming chat with the new UI (single sidebar, header, input), markdown/code/math, stop, continue, regenerate; context meter; prefill progress. No persistence yet. *Done when:* a Norwegian conversation with the NB Llama 8B streams smoothly on the laptop and every control works.
- **A2 Chats (S).** Auto-save, list grouped by date, rename, delete, restore, edit-and-resend with confirmation; compatibility test against real Skibeti37 chat files. *Done when:* chats created by Skibeti37 open correctly and vice versa.
- **A3 Models, downloader, settings (M).** Curated catalog (NB Llama 3.1 8B, Qwen3 8B/4B, Llama 3.2 1B/3B, Qwen3 1.7B; each with size, RAM need, Norwegian note), queue with resume/progress/ETA over Node https, unsupported-file detection with a clear message, per-model settings (display name, active, KV f32/f16), performance (context size, threads), reset, first-run card. *Done when:* a fresh folder can download, select and chat with a model without touching the file system.
- **A4 Skills (S).** Port the skill parser (`skills.py`, 298 lines), the `/` menu, chips, the three bundled skill files, hot reload. *Done when:* `/humanize`, `/norwegian` etc. behave like in Skibeti37.
- **A5 Context management (M).** Fit-to-context trimming, rolling summary via the background model (loaded on demand, freed after), loop detection, continuation prompt, sampling defaults. *Done when:* a 40-message conversation stays within the window and remembers early facts.
- **A6 Documents (M).** Upload .txt/.md, chunking, BM25 retrieval, attach/detach per chat (chips), library and preview in Settings, "Kilder" under replies, optional map-reduce summary; small default retrieval; prefix-friendly prompt layout. *Done when:* questions about an uploaded Norwegian document are answered from it with visible sources, and the time-to-first-token is acceptable (measured and recorded).
- **A7 Polish & packaging (S-M).** Error messages in Norwegian for every failure path, diagnostics panel (zerollm `doctor`, RAM, log file), keyboard shortcuts, accessibility pass, English strings, README (Norwegian + English), portable zip layout, shortcut instructions, a clean-machine test checklist.
- **A8 Optional upgrades.** E6 embeddings, E7 Qwen3.5 (after inspecting a real file), PDF/DOCX import, export to Markdown/PDF, more architectures.

Suggested order of attention: A0 -> A1 (earliest real value, validates the whole stack and the UI) -> A2 -> A3 -> A4 -> A5 -> A6 -> A7. E8 (speed) runs alongside, driven by measurements.

## 8. Testing strategy

- Pure-JS core modules (chats, skills, config, downloader queue, context fitting, BM25) get `node:test` unit tests; API routes are tested end to end against the synthetic tiny GGUF so CI-style runs need no real model.
- Compatibility tests use fixtures copied from real Skibeti37 chat/document/skill files.
- UI smoke tests run in Playwright in my environment (dev only, never a runtime dependency) with screenshots for design iteration.
- Real-model checks are manual and written down per milestone (a short checklist in `docs/APP-CHECKS.md`, same style as `docs/REAL-MODEL-CHECK.md`), run on the target laptop.
- Performance numbers go into `docs/BENCHMARKS.md`, never claimed without measurement.

## 9. Risks and open decisions

1. **NB Llama 8B quality in zerollm**: first real check passed (correct, fluent one-sentence Norwegian answer, instruction followed; see BENCHMARKS.md); keep testing longer and harder Norwegian prompts as the app takes shape.
2. **Prefill speed for document questions** (see section 6): may force "small model for documents" as a product decision.
3. **Qwen3.5 / embeddings**: unknown engine effort; the plan deliberately does not depend on them.
4. **Edge `--app` window on the managed machine**: if blocked, the app falls back to the default browser tab (same server), only the chrome differs.
5. **Two models resident (chat + utility)** on 16 GB: handled by on-demand utility loading; the RAM gate must account for it.
6. **Scope creep**: parity is defined by the table in section 3; anything else is A8.

Decisions needed from the owner: app name; default UI language (proposal: Norwegian, English switchable); curated model list for v1; whether the new app should read/write the same data folder as Skibeti37 (proposal: yes, additive formats); retrieval default (proposal: BM25 first).
