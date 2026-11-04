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

### M5 on the target laptop (user-measured): Ryzen 5 7530U, Node 26.5.1, Windows; Qwen3-0.6B-Q8_0 (real), JS kernels

`zerollm bench --threads 1,2,4,6,8,12 --prompt-tokens 32 --n 32` (ctx 512). RSS column is cumulative (models are created in one process).

| threads | load s | prefill tok/s | decode tok/s | eff. GB/s | speedup |
|---|---|---|---|---|---|
| 1 | 0.7 | 0.97 | 0.74 | 0.47 | 1.00x |
| 2 | 0.4 | 1.80 | 1.50 | 0.95 | 2.03x |
| 4 | 0.3 | 3.64 | 2.79 | 1.77 | 3.79x |
| 6 | 0.3 | 3.75 | 3.28 | 2.08 | 4.46x |
| 8 | 0.3 | 3.55 | 2.98 | 1.89 | 4.05x |
| 12 | 0.3 | 4.18 | 3.89 | 2.46 | 5.28x |

Greedy output identical across all thread counts. Observations: effective weight bandwidth is ~2.5 GB/s vs a ~25-35 GB/s ceiling, so the JS kernels (not RAM) are the bottleneck; 12 threads beat 6 because the workload is compute-bound (hyperthreads help). The 1-thread number (0.74) is lower than the earlier `run` measurements (1.20-1.27 decode), so the speedup column is inflated by a slow baseline (power state / background load?); compare absolute tok/s.

### M6 (WASM SIMD kernels), dev container, synthetic Qwen3-0.6B-shaped Q8_0 (random weights)

Linux VM, Xeon @ 2.1 GHz, 4 vCPU, Node 22.23, `zerollm bench --kernels js,wasm --threads 1,4 --prompt-tokens 8 --n 24 --ctx 256`. NOT the target machine.

| kernels | threads | prefill tok/s | decode tok/s | eff. GB/s | vs js x1 |
|---|---|---|---|---|---|
| js | 1 | 1.59 | 1.23 | 0.78 | 1.00x |
| js | 4 | 3.43 | 3.31 | 2.10 | 2.69x |
| wasm-simd | 1 | 8.46 | 6.61 | 4.19 | 5.4x |
| wasm-simd | 4 | 17.28 | 17.87 | 11.32 | 14.5x |

Per-token profile of worker 0 (`ZEROLLM_PROFILE=1`), ms/token:

| config | total | matmul kernels | act quantization (JS) | attention+rope | barrier waits | other |
|---|---|---|---|---|---|---|
| wasm 1 thread | 149.9 | 135.2 (90%) | 4.4 | 6.1 | 0.4 | 3.8 |
| wasm 4 threads | 57.0 | 34.6 (61%) | 4.6 | 2.0 | 13.9 (24%) | 2.0 |
| js 4 threads | 283.4 | 221.2 | 0.2 | 2.0 | 57.4 | 2.6 |

Takeaways: the kernel is ~90% of single-thread time at ~4.5 GB/s of weights per thread (instruction-bound, not memory-bound, so far below the DRAM ceiling); kernel time scales ~3.9x on 4 threads; barrier waits grow to ~14 ms/token (169 barriers, ~80 us each: load imbalance + VM scheduling jitter). Next levers: relaxed-SIMD dot / hoisting the f16 scale conversion in the kernel, fewer/cheaper barriers, quantizing activations once instead of per worker.

### M7 (batched prefill), dev container, synthetic Qwen3-0.6B-shaped Q8_0 (random weights), wasm-simd, 96-token prompt

`zerollm bench --kernels wasm --threads 1,4 --batch 1,8,16,32 --prompt-tokens 96 --n 8 --ctx 256`. 4-vCPU Xeon VM, NOT the target machine.

| threads | batch | prefill tok/s | decode tok/s | prefill/decode |
|---|---|---|---|---|
| 1 | 1 (token by token) | 8.34 | 6.44 | 1.29x |
| 1 | 8 | 19.41 | 6.49 | 2.99x |
| 1 | 16 | 19.84 | 6.75 | 2.94x |
| 1 | 32 | 18.23 | 6.62 | 2.75x |
| 4 | 1 | 24.20 | 14.83 | 1.63x |
| 4 | 8 | 42.33 | 13.61 | 3.11x |
| 4 | 16 | 41.17 | 14.03 | 2.93x |
| 4 | 32 | 39.36 | 14.91 | 2.64x |

Batch 8-16 is the sweet spot (default 16). Matmul kernel time per prompt token drops from 135 ms (decode kernel) to 35 ms (1 thread). Profile of the batched path (ms/token): 1 thread: total 54.1 = kernels 34.9 + act-quantization 4.0 + attention 12.7 + other 2.4; 4 threads: total 23.4 = kernels 9.1 + quantization 4.1 + attention 3.9 + barriers 5.2 + other 1.1. Follow-ups: attention is still plain JS and grows with position (23% of 1-thread prefill at <100 tokens), and activation quantization (JS) is redone by every worker.

### M8 (K-quants), dev container, synthetic Qwen3-0.6B-shaped models with random raw blocks (wasm-simd, batch 16)

4-vCPU Xeon VM, NOT the target machine. "Q4_K_M-style" = Q4_K everywhere except Q6_K for attn_v / ffn_down (like llama.cpp's Q4_K_M mix). The numbers show kernel speed only; the weights are random.

| model file | size | threads | prefill tok/s | decode tok/s | eff. GB/s |
|---|---|---|---|---|---|
| Q8_0 | 604 MB | 1 | 20.43 | 6.32 | 4.00 |
| Q8_0 | 604 MB | 4 | 39.53 | 16.31 | 10.33 |
| Q4_K_M-style | 363 MB | 1 | 5.47 | 4.30 | 1.64 |
| Q4_K_M-style | 363 MB | 4 | 16.99 | 13.72 | 5.22 |

Profile, Q4_K_M-style, 1 thread decode: 197 ms/token, 94% in the matmul kernels. The first-version K-quant kernels are compute-bound (per-sub-block scale unpacking and mins handling in scalar code) and process ~2.6 G weights/s/thread vs ~3.8 for Q8_0, so on this small model the smaller file is *slower* per token. They also have no batched variant yet (prefill runs the decode kernel once per token: prefill/decode = 1.27x). Both are M9 work (e.g. hoisting the activation block sums out of the kernel, repacking scales at load time, tiled batch kernels).
