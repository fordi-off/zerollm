# Decisions log

Decisions already made in the brief are in `docs/BRIEF.md` section 3. This file records decisions made while implementing.

## 2026-11-04 — M0/M1
- Development container has Node 22 (target laptop has Node 26). Code targets `node >=22`; nothing relies on Node 26 only features. Feature probes in `doctor` report what the running Node supports.
- CLI commands live in `src/cli/<name>.js` and are lazily imported by `bin/zerollm.js`.
- GGUF reader uses a buffered file cursor (never reads the whole file); uint64 values are returned as `Number` when safe, `BigInt` otherwise.
- `Q8_1` block size is 36 bytes (f16 d + f16 s + 32 int8), not 40; derived from ggml's `block_q8_1`.

## 2026-11-04 — M2/M3
- Pre-tokenizer regexes spell out contractions with explicit case classes (`'[sS]`, `'[rR][eE]`, ...) instead of `(?i:...)`. Equivalent, and works on every Node (the Node 22 dev container lacks inline modifiers; `doctor` still reports the probe).
- Special tokens (GGUF token_type 3 control and 4 user-defined) are matched literally in text only when `special: true`. Chat templates encode scaffolding with special parsing ON and message content with it OFF, so user text such as `<|im_end|>` cannot inject control tokens.
- Control tokens are hidden when decoding for display (`skipControl`), user-defined ones (e.g. `<think>`) are shown.
- Qwen3 chat defaults to **non-thinking** (pre-filled empty `<think></think>` block) because thinking would burn tokens at ~2-5 tok/s; `--think` enables it. This differs from the model's own default.
- Llama 3.1/3.2 system header ("Cutting Knowledge Date...") uses a fixed date string by default so prompts are deterministic.
- Reference path uses a Q8_0-style activation quantization with **f32** scales (as in the brief); ggml itself uses f16 scales, so logits differ very slightly from llama.cpp.
- Reference kernels support F32/F16/BF16/Q8_0/Q4_0 only; K-quants arrive in M8.
- Tests use an independent naive whole-sequence forward pass (`test/naive-forward.js`, double precision, no KV cache) as the oracle for the optimized-style reference path. Mutation-checked: GQA mapping, attention scale, RoPE on K, QK-norm, SiLU, causal range and residual mistakes are all caught.

## 2026-11-04 — M4
- `generate()` is an async generator that yields to the event loop after every token **including during prefill**, so Ctrl+C is responsive even on long prompts (setImmediate cost is negligible vs ~0.5 s/token). It also reuses the longest common prefix between `model.tokens` and the new prompt (multi-turn chat only prefills the new turn), and always recomputes the last prompt token to obtain logits.
- Sampler order: penalties (in place, restored afterwards) -> top-k (O(V log k) heap) -> temperature softmax -> min-p -> top-p -> sample. `--temp 0` is plain argmax (after penalties).
- Stop strings are matched on the decoded text stream with hold-back of partial matches (`StopFilter`); the matched text is never printed.
- In `chat`, readline owns Ctrl+C (raw mode), so the REPL listens to `rl.on('SIGINT')`; `run` uses `process.on('SIGINT')`. Second Ctrl+C force-exits (130).
- `--threads` is accepted but ignored (with a warning) until M5.

## 2026-11-04 — M5 (worker pool)
- **Partitioning:** every weight matrix is split by output rows (multiples of 4, last worker takes the remainder); attention is split by head (`floor(w*count/n)` ranges). Each worker stores only its slices, in its own private `WebAssembly.Memory` (64-byte-aligned offsets) so M6's SIMD kernels can address them directly and the 4 GB wasm32 limit applies per worker.
- **Per layer, 6 barriers:** (A) q/k/v row slices -> shared; (B) per-KV-head k-norm + RoPE + write K/V cache; (C) per-query-head q-norm + RoPE + attention; (D) o-proj + residual on own rows; (E) gate/up row slices -> SwiGLU slice; (F) down-proj + residual on own rows. Plus one after the embedding lookup. Norms and activation quantization are recomputed redundantly by every worker (cheaper than a barrier). ~170 barriers/token for a 28-layer model.
- **No postMessage per token:** the main thread issues a command through a shared `Int32Array` (`Atomics.add` + `notify`) and awaits completion with `Atomics.waitAsync`, so the event loop (SIGINT etc.) stays live. The KV cache lives in one `SharedArrayBuffer` (layout `[layer][k|v][pos][kvDim]`), written by the worker that owns that KV head.
- **Bit-exactness:** workers use the same kernel functions as the reference path and every output row/head is computed by exactly one worker with identical arithmetic, so logits are *bit-identical* to the single-thread reference regardless of worker count. Tests assert exact equality (not tolerance). This is only true while the JS kernels are used; the WASM kernels (M6) will need tolerance-based parity tests.
- **Head-aligned q/k/v row splits were rejected:** with 8 KV heads over 6 workers (2,2,1,1,1,1) the busiest worker would carry 25% of the qkv work instead of 16.7%. Even row splits plus one extra barrier balance better.
- **Process lifetime:** workers are `unref()`'d (a forgotten `close()` must not hang the process) -- but `unref()` must be called *after* attaching the `'message'` listener, because adding the listener re-refs the port (found by test: unref-before-listener hung the process). The pool therefore holds its own keep-alive interval while loading and during each `forward()`.
- **Failure handling:** a worker that throws posts an error and the pool terminates all workers (others would otherwise wait forever at a barrier); `forward()` polls with a 200 ms `waitAsync` timeout, so an externally killed worker rejects the call instead of hanging.
- `--threads N`: default = logical processors / 2 (physical-core guess); `--threads 1` uses the single-thread reference path (kept as the oracle). `zerollm bench` compares thread counts and verifies identical greedy output.
- `node --test` runs every `.js` file in a directory named `test`; worker fixtures live in `test-fixtures/` for that reason.
