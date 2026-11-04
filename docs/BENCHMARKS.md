# Benchmarks

No benchmarks yet. Record: date, commit, model, quant, threads, ctx, prefill tok/s, decode tok/s, RAM.

| date | commit | model | quant | threads | ctx | prefill tok/s | decode tok/s | RAM |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | M3 | **synthetic** random-weight model with Qwen3-0.6B shapes (1024/28 layers/16h/8kv/hd128/ffn3072/vocab151936) | Q8_0 | 1 (reference JS) | 512 | n/a (per-token) 1.69 w/o logits | 1.28 (with full-vocab logits) | 0.71 GB RSS |

Notes on the M3 row: measured in the Linux dev container (Intel Xeon @ 2.1 GHz, 4 vCPU, Node 22.23), **not** on the target Ryzen 5 7530U, and with random weights (speed does not depend on weight values, correctness is not exercised). Real-model numbers on the target laptop are still needed. Generated with `tools/make-tiny-gguf.js` (`padVocabTo: 151936`).
| 2026-11-04 | 046ccbd | **Qwen3-0.6B Instruct (real)** | Q8_0 | 1 (reference JS) | 2048 | 1.41-1.47 | 1.20-1.26 | 738 MB RSS |

Note: row above measured by the user on the target laptop (Ryzen 5 7530U, Node 26.5.1, Windows), 6.2 GB RAM free at the time; prompt 20-24 tokens, `--temp 0`. Prefill is per-token in the reference path (batched prefill is M7).

### M5 scaling (worker pool, JS kernels), dev container, synthetic Qwen3-0.6B-shaped Q8_0 model (random weights)

Linux VM, Intel Xeon @ 2.1 GHz, 4 vCPU, Node 22.23. `zerollm bench --threads 1,2,3,4 --prompt-tokens 8 --n 10 --ctx 256`. Noisy VM: repeated 4-thread runs gave 3.7-4.2 tok/s. NOT the target machine; run `bench` on the laptop for real numbers.

| threads | prefill tok/s | decode tok/s | speedup | efficiency |
|---|---|---|---|---|
| 1 (reference) | 1.54 | 1.23 | 1.00x | 100% |
| 2 | 2.59 | 2.36 | 1.92x | 96% |
| 3 | 3.61 | 3.10 | 2.52x | 84% |
| 4 | 3.68 | 4.17 | 3.39x | 85% |

Greedy output is bit-identical across all thread counts (and the pool's logits are bit-identical to the reference path in the test suite). Fresh-process RSS with 4 workers: 761 MB (single thread 698-738 MB), i.e. no weight duplication. Barrier spin sensitivity at 4 threads (decode tok/s): spin=0 -> 2.89, 2000 -> 3.69, 20000 -> 3.85, 200000 -> 3.72.
