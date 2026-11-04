# Decisions log

Decisions already made in the brief are in `docs/BRIEF.md` section 3. This file records decisions made while implementing.

## 2026-11-04 — M0/M1
- Development container has Node 22 (target laptop has Node 26). Code targets `node >=22`; nothing relies on Node 26 only features. Feature probes in `doctor` report what the running Node supports.
- CLI commands live in `src/cli/<name>.js` and are lazily imported by `bin/zerollm.js`.
- GGUF reader uses a buffered file cursor (never reads the whole file); uint64 values are returned as `Number` when safe, `BigInt` otherwise.
- `Q8_1` block size is 36 bytes (f16 d + f16 s + 32 int8), not 40; derived from ggml's `block_q8_1`.
