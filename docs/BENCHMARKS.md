# Benchmarks

No benchmarks yet. Record: date, commit, model, quant, threads, ctx, prefill tok/s, decode tok/s, RAM.

| date | commit | model | quant | threads | ctx | prefill tok/s | decode tok/s | RAM |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | M3 | **synthetic** random-weight model with Qwen3-0.6B shapes (1024/28 layers/16h/8kv/hd128/ffn3072/vocab151936) | Q8_0 | 1 (reference JS) | 512 | n/a (per-token) 1.69 w/o logits | 1.28 (with full-vocab logits) | 0.71 GB RSS |

Notes on the M3 row: measured in the Linux dev container (Intel Xeon @ 2.1 GHz, 4 vCPU, Node 22.23), **not** on the target Ryzen 5 7530U, and with random weights (speed does not depend on weight values, correctness is not exercised). Real-model numbers on the target laptop are still needed. Generated with `tools/make-tiny-gguf.js` (`padVocabTo: 151936`).
