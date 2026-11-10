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

### M8 on the target laptop (user-measured): Ryzen 5 7530U, Node 26.5.1, Windows; **Qwen3-4B Q4_K_M (real file, 2.33 GB)**, wasm-simd, 6 threads

`zerollm bench models/Qwen3-4B-Q4_K_M.gguf --threads 6 --batch 1,16 --prompt-tokens 64 --n 16` (ctx 512). 6.6 GB RAM was free. `info`: Q4_K x216 tensors (1.64 GB, 4.50 bits/weight), Q6_K x37 (692 MB, 6.56 bpw), F32 x145; 4.02B parameters; tied embeddings.

| kernels | threads | batch | load s | prefill tok/s | decode tok/s | eff. GB/s | RSS |
|---|---|---|---|---|---|---|---|
| wasm-simd | 6 | 1 | 0.6 | 5.50 | 5.77 | 14.37 | 2.57 GB |
| wasm-simd | 6 | 16 | 0.6 | 5.96 | 5.74 | 14.29 | 2.65 GB |

`run ... "What is the capital of France? Answer in one sentence." --temp 0`: "The capital of France is Paris." (prompt 24 tok, decode 4.30 tok/s over only 7 generated tokens incl. warm-up, RSS 2.61 GB). First real-model confirmation of the Q4_K/Q6_K layouts and SIMD kernels.

Observations: decode 5.8 tok/s clears the brief's ">= ~4 tok/s for 4B Q4" target; ~14.4 GB/s of the ~25-35 GB/s ceiling (so some headroom). Prefill is NOT faster than decode for K-quants (1.04x): they have no batched kernels yet, so a 64-token prompt takes ~11 s. Projection (not a measurement): 8B Q4_K_M (~4.7 GB) at ~14 GB/s effective would decode at roughly 3 tok/s.

### Kernel micro-benchmark (tools/kbench.mjs), dev container: compute-bound speed per kernel, weights cache-resident

`node tools/kbench.mjs` (1024 x 2048 weights, batch 16, 1 thread, 4-vCPU Xeon VM; G weights/s, batch column = G weight-token products/s). Shows what each format costs when memory is not the limit.

| format | decode | batched x16 | notes |
|---|---|---|---|
| Q8_0 | 11.4-13.1 | 15.6-17.3 | |
| Q4_0 | 9.6-10.8 | 14.5-16.5 | |
| Q5_0 | 7.1-7.3 | 12.5 | |
| Q4_K | 5.4-5.5 (was 4.6) | 5.8-5.9 | |
| Q5_K | 4.3-4.5 (was 3.7) | 5.5 | |
| Q6_K | 2.6 | 4.4 | slowest; 21% of the weights but ~35% of the matmul time in a Q4_K_M model |

Experiment that shaped the design: replacing widen+dot with `i16x8.extmul` + pairwise adds (no widening) made decode 25-35% faster for Q4_K/Q5_K but batched kernels 20-50% slower (there the weights are widened once per tile and reused for 4 tokens) and Q5_0/Q6_K decode no better, so the final kernels are a hybrid (extmul for Q4_K/Q5_K/Q6_K decode; widened tiles for every batched kernel; Q5_0 decode as before). Wasm op count is not machine instruction count: on x64 V8 an extmul costs several instructions.

Model-level effect on the Q4_K_M-style 0.6B-shaped file (before -> after, decode / prefill tok/s): 1 thread 4.30/5.47 -> 4.85/7.32; 4 threads 13.7/17.0 -> 15.5/22.0. Smaller than the kernel gains because attention (JS, grows with context) and activation quantization (JS, repeated by every worker) are still outside the kernels.

### SIMD attention + private per-worker KV cache, dev container, Q8_0 0.6B-shaped synthetic model, wasm-simd

4-vCPU Xeon VM, NOT the target machine. Attention was plain JS over a shared KV cache before; now each worker owns a set of KV heads (with the query heads that share them), keeps their K/V in its own wasm memory, and runs SIMD q.K / p.V kernels (softmax still JS).

1 thread, batch 16, ms per prompt token (profile of worker 0):

| prompt tokens | attention before | attention after | total/token before | total/token after |
|---|---|---|---|---|
| 32 | 10.2 | 1.8 | 55.0 | 48.1 |
| 128 | 13.9 | 2.4 | 53.8 | 43.3 |
| 384 | 36.1 | 5.2 | 75.2 | 43.9 |

4 threads, 384-token prompt, 16 generated tokens:

| | prefill tok/s | decode tok/s |
|---|---|---|
| before | 33.0 | 10.4 |
| after | 55.8 | 18.1 |

Before, attention cost grew linearly with context and at a 384-token prompt was larger than all matmuls combined (decode fell ~40% below its short-context speed); now it is ~12% of the token time at that length. One barrier per layer fewer (5 instead of 6: K/V no longer cross workers).

### M9 memory measurements (dev container, linux): 8B-shaped model and KV precision

**8B-shaped synthetic Q4_K_M-style model** (random raw blocks; hidden 4096, 36 layers, 32/8 heads, head_dim 128, FFN 12288, vocab 151936, untied output; 8.19B parameters; 5.43 GB file because this mix uses Q6_K for every attn_v / ffn_down; a real Q4_K_M is ~5.0 GB). `zerollm run ... --threads 4 --ctx 2048 --no-cache`, 65-token prompt, 4 generated tokens, VmHWM sampled from /proc:

| --kv | peak RSS | load | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| f32 | 5.27 GiB | 7.1 s (cold file cache) | 1.45 | 1.38 |
| f16 | 5.27 GiB | 1.7 s | 1.90 | 1.83 |

RSS is identical because KV pages are only resident once touched (65 tokens touch ~20 MB of the 576 MB reserved) and matches the estimator (weights 4.71 GiB + runtime ~0.4 GiB). Speeds are for a 4-vCPU 2.1 GHz VM and say nothing about the target laptop (the f32/f16 speed difference is noise plus the cold first read). `info` predicts for ctx 2048 / 6 threads: 6.15 GB total with f32 KV, 5.86 GB with f16, busiest worker 1.0 GB.

**KV cache growth, measured** (Q8_0 0.6B-shaped model, 4 threads, batch 16, 3000-token prompt, ctx 3100; KV = 229 KB/token f32, 115 KB/token f16):

| --kv | baseline (16-token prompt) | with 3000-token context | growth | predicted |
|---|---|---|---|---|
| f32 | 850 MiB | 1485 MiB | +635 MiB | +640 MiB |
| f16 | 850 MiB | 1158 MiB | +308 MiB | +320 MiB |

With the f16 cache at that 3000-token context: prefill 29.3 tok/s, decode 18.5 tok/s (4 threads).

### Target laptop (user-measured), after the K-quant / SIMD-attention / f16-KV work: Ryzen 5 7530U, Node 26.5.1, Windows

`node tools/kbench.mjs` (1 thread, weights cache-resident; G weights/s, batch column = G weight-token products/s):

| format | decode | batched x16 | batch speedup |
|---|---|---|---|
| Q8_0 | 12.7-13.4 | 22.0-24.5 | 1.7-1.8x |
| Q4_0 | 10.0-10.6 | 20.6-22.2 | 2.1x |
| Q5_0 | 7.2-7.7 | 16.6-17.0 | 2.2-2.3x |
| Q4_K | 8.8-9.4 (wide form: 8.6) | 15.3-16.2 | 1.7x |
| Q5_K | 7.2 (wide: 6.9) | 12.3-12.4 | 1.7x |
| Q6_K | 4.8-5.0 (wide: 4.7) | 10.8-11.1 | 2.3x |

The extmul and wide integer-dot forms for K-quant decode are within 2-4% of each other on this CPU (extmul marginally ahead), unlike the dev VM where extmul was 15-30% ahead: default stays extmul.

**Qwen3-4B Q4_K_M, 6 threads** (`bench --threads 6 --batch 1,16 --prompt-tokens 64 --n 16`): prefill 4.80 (batch 1) / **8.23** (batch 16) tok/s, decode 4.40 / 4.81 tok/s. Earlier run before the K-quant batch kernels: prefill 5.96, decode 5.77. Repeated runs of essentially the same code gave decode 4.40, 4.81, 5.31 and 5.77 tok/s: **run-to-run variation on this laptop is about +-15%** (power state, background load), so single runs cannot show differences below that.

Profile of worker 0, decode, 4B Q4_K_M, 6 threads (`ZEROLLM_PROFILE=1`, ms/token):

| dot form | total | matmul kernels | act quantization | attention | barrier waits | other |
|---|---|---|---|---|---|---|
| extmul | 209.2 | 157.8 (75%) | 12.4 (6%) | 0.8 | 35.3 (17%) | 2.9 |
| wide | 201.6 | 154.5 (77%) | 12.5 (6%) | 0.8 | 31.0 (15%) | 2.8 |

Interpretation: the cache-resident kernel speeds predict ~89 ms of matmul per worker per token, observed 155 ms, so the kernels stall on DRAM (~15 GB/s seen by the kernels vs a 25-35 GB/s ceiling); ~180 barriers per token average ~190 us of waiting each (imbalance and/or SMT contention from spinning workers); JS activation quantization costs 6%. Projection (not a measurement) for 8B Q4_K_M at ~2.1x the weight bytes: roughly 2.5-2.7 tok/s decode.

### Barrier wait strategy A/B on the target laptop (user-measured, `tools/abbench.mjs`, 5 interleaved rounds, 15 s pauses)

Qwen3-4B Q4_K_M, 6 workers, batch 1, 16-token prompt, 24 generated tokens:

| spin | decode tok/s per round | median (min-max) | prefill median |
|---|---|---|---|
| 0 (sleep at once) | 5.90, 4.55, 4.80, 6.04, 6.06 | **5.90** (4.55-6.06) | 4.48 |
| 20000 (spin first) | 4.28, 5.64, 4.57, 4.55, 4.55 | 4.55 (4.28-5.64) | 4.36 |

spin=0 won 4 of 5 rounds, median +30% (ranges overlap, so not airtight); results cluster in two modes (~6.0 vs ~4.55), consistent with spinning workers landing on the hyperthread siblings of busy cores. On the 4-vCPU dev VM (no spare hyperthreads) spinning was ~20% faster. The pool therefore defaults to `spin = 0` when logical processors >= 2 x workers, else 20000 (`--spin N` overrides).

### M9 acceptance: Qwen3-8B Q4_K_M on the target laptop (user-measured): Ryzen 5 7530U, Node 26.5.1, Windows, 7.0 GB RAM free at the start

File: Qwen3-8B-Q4_K_M.gguf, 4.68 GB (Q4_K x217 = 3.45 GB, Q6_K x37 = 1.23 GB, F32 x145), 8.19B parameters, 36 layers, 32/8 heads, untied output. `info` estimated 5.76 GB (f32 KV) / 5.48 GB (f16 KV) at ctx 2048 with 6 threads. All runs: 6 threads, wasm-simd kernels, `--kv f16`.

| run | prompt | prefill tok/s | decode tok/s | RSS |
|---|---|---|---|---|
| `run` "capital of France" (`--temp 0`) | 24 tok | 5.33 | 2.86 | 5.04 GB |
| `bench` batch 1 | 64 tok | 3.68 | 2.95 | 5.02 GB |
| `bench` batch 16 | 64 tok | 6.96 | 3.17 | 4.98 GB |
| `chat`, turn 1 | 24 tok | 6.35 | 3.10 | 4.97 GB |
| `chat`, turn 2 (20 tokens of history reused from the KV cache) | 71 tok | 7.16 | 3.13 | 4.92 GB |

- Load time 2.4-2.7 s (file in the OS cache), effective weight bandwidth 14.8-15.9 GB/s.
- Output: "The capital of France is Paris." and coherent multi-turn chat. (The second chat answer contains a factual error about European traffic; that is model quality for a 4-bit 8B in non-thinking mode, the text itself is fluent and on topic.)
- **Brief M9 criterion ("Qwen3 8B Q4 runs in <= ~9 GB RAM at >= ~2 tok/s decode, documented"): met** -- ~5.0 GB RSS (estimator: 5.48 GB at full 2048-token context), 2.9-3.2 tok/s decode. The projection made earlier from the 4B result (2.5-2.7 tok/s) was slightly pessimistic.

### Llama 3.2 1B Instruct Q8_0 on the target laptop (user-measured): first real Llama file

`info`: architecture llama, 16 layers, hidden 2048, 32/8 heads, head_dim 64, FFN 8192, vocab 128256, RoPE adjacent pairs base 500000 + `rope_freqs.weight`, tied embeddings, 1.24B parameters, Q8_0 (1.22 GB), template = the simple Llama 3 one (no "Cutting Knowledge Date" header, so none is added), BOS 128000 `<|begin_of_text|>`, EOS 128009 `<|eot_id|>`. `tokenize "Hello, world! 12345"` -> ids `9906 11 1917 0 220 4513 1774` (pieces Hello , Ġworld ! Ġ 123 45), exactly the tiktoken/cl100k ids. `run --show-prompt` rendered `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n` (22 tokens); answer "The capital of France is Paris.", stopped by `<|eot_id|>`.

6 threads, wasm-simd: prefill 39.8-53.2 tok/s, decode 15.3-16.8 tok/s (~19.5 GB/s effective weight bandwidth, the best of any model so far), RSS 1.6 GB (estimate 1.87 GB). Multi-turn `chat` reused the history from the KV cache (22, 66, 100 tokens cached). Speed across models tracks file size: Qwen3 0.6B Q8_0 (0.6 GB) ~28, Llama 3.2 1B Q8_0 (1.2 GB) ~16, Qwen3 4B Q4_K_M (2.3 GB) ~5.5, Qwen3 8B Q4_K_M (4.7 GB) ~3 tok/s.

### NB Llama 3.1 8B Instruct Q4_K_M on the target laptop (user-measured): first Norwegian model

File nb-llama-3.1-8b-q4_k_m.gguf (4.58 GB, Q4_K x193 + Q6_K x33, 8.03B parameters, 32 layers, hidden 4096, FFN 14336, vocab 128256, untied output, rope_freqs.weight, BOS `<|begin_of_text|>` 128000, GGUF EOS `<|end_of_text|>` 128001, **no embedded chat template** -> the engine fell back to the Llama 3 framing and also stops on `<|eot_id|>`). `run --system "Du er en hjelpsom assistent som alltid svarer på norsk." -p "Hva er hovedstaden i Norge? Svar med én setning." --temp 0 --kv f16`: **"Oslo er hovedstaden i Norge."** (one sentence as asked; the generic Llama 3.2 1B answered with two sentences and a spelling slip). Prompt 51 tokens, prefill 6.37 tok/s, decode 3.24 tok/s, RSS 4.94 GB (estimate 5.35 GB with f16 KV).

## Embedding encoder (nomic-embed-text-shaped synthetic Q4_K_M model, 768 wide, 12 layers, 300-token sequences), dev VM 4 vCPU
| workers | tokens/s | ms per 300-token text |
|---|---|---|
| 1 | 30 | 10158 |
| 2 | 59 | 5107 |
Profile of one worker: 86 % K-quant matmul kernels, 4 % activation quantization, 6 % attention, 3 % other. (Laptop numbers pending: `node tools/embed-bench.mjs`.)

## Speculative decoding (draft model + batched verification)

Added 2026-11-10. **Nothing in this section has been measured on the target laptop yet.** The dev-VM numbers below only show that the mechanism works and what it costs; they say nothing about the laptop. Design and guarantees: `docs/DECISIONS.md`, "Speculative decoding".

### What to expect (arithmetic, not a measurement)

Speculative decoding wins when `tokens per step > (k * t_draft + t_verify(k+1)) / t_decode`. From the numbers measured on the laptop: Qwen3-8B Q4_K_M decodes at ~3.1 tok/s (t_decode ~ 320 ms) and prefills at ~7 tok/s in batches of 16 (so a verified extra token costs ~143 ms, about 45% of a decode step); Qwen3-0.6B Q8_0 decodes at ~28 tok/s (t_draft ~ 36 ms). If verifying B tokens costs `t_mem + B * t_cmp` with t_mem ~ 190 ms and t_cmp ~ 131 ms (fitted to those two points), and a drafted token is accepted with probability 0.7, the tokens/s ratio against plain decoding comes out around 1.1x for k = 1-2 and ~1.0x for k = 3, and about 1.4x if the batched pass overlaps memory and compute perfectly (`max(t_mem, B * t_cmp)`). So the realistic range is **roughly 1.0-1.5x for this pair on this laptop, with acceptance as the deciding factor**; the hard ceiling is the prefill/decode ratio (~2.2x). `--draft-k` defaults to 3 because k + 1 = 4 tokens fill exactly one tile of the batched kernels (see the dev-VM table below); the real optimum for the laptop is what the procedure below finds. The measurement replaces this guess.

### Procedure for the laptop (PowerShell or cmd, repo root, nothing else running; ~6.5 GB must be free: 8B Q4_K_M ~5.0 GB + 0.6B Q8_0 ~0.9 GB)

Files: `models/Qwen3-8B-Q4_K_M.gguf` and `models/Qwen3-0.6B-Q8_0.gguf` (same tokenizer; any other same-family pair works the same, e.g. Llama-3.1-8B + Llama-3.2-1B).

1. Compatibility and memory (no weights are read for the check; a mismatch stops with a clear message):
   `node bin/zerollm.js run models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3-0.6B-Q8_0.gguf -p "Hi" --n 8 --threads 6 --kv f16`
   Note the RSS in the stats line (plain 8B run was ~5.0 GB).
2. Correctness on the real pair (greedy must give identical text; `fc` is the Windows file compare):
   `node bin/zerollm.js run models/Qwen3-8B-Q4_K_M.gguf -p "Explain how a hash table works." --n 150 --temp 0 --threads 6 --kv f16 > plain.txt`
   `node bin/zerollm.js run models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3-0.6B-Q8_0.gguf -p "Explain how a hash table works." --n 150 --temp 0 --threads 6 --kv f16 > spec.txt`
   `fc plain.txt spec.txt`
   (The two texts can differ in a rare near-tie: the batched verification pass and the single-token pass are bit-identical on every synthetic model tried, but this has not been confirmed on a real Q4_K_M file. If they differ, send both files and the first differing token.)
3. Speed, paired and interleaved (5 rounds, medians; takes ~10-15 minutes; `--chat` wraps the prompts in the chat template like the app does):
   `node tools/specbench.mjs models/Qwen3-8B-Q4_K_M.gguf models/Qwen3-0.6B-Q8_0.gguf --threads 6 --draft-threads 2,4,6 --k 1,2,3,4 --n 128 --rounds 5 --chat --kv f16 --ctx 2048`
   Paste the whole table. Columns: plain tok/s, speculative tok/s, speed-up (median of paired ratios and min-max), accepted (share of drafted tokens that were kept), tok/step (1 + accepted drafts per verification pass), draft ms/step and verify ms/step.
4. Same with sampling (chat default is temperature 0.7; texts differ, so only speed and acceptance are compared):
   `node tools/specbench.mjs models/Qwen3-8B-Q4_K_M.gguf models/Qwen3-0.6B-Q8_0.gguf --threads 6 --draft-threads 4 --k 2,3 --n 128 --rounds 5 --chat --temp 0.7 --kv f16`
5. A real conversation, the number that matters in the app (look at "decode X tok/s" and the "speculative k=..." part of the stats line after each answer; compare with the same chat without `--draft`):
   `node bin/zerollm.js chat models/Qwen3-8B-Q4_K_M.gguf --draft models/Qwen3-0.6B-Q8_0.gguf --draft-k 2 --threads 6 --kv f16 --system "Be brief."`
6. The one-shot variant of step 3 for a single configuration, with the built-in prompt (not representative, only for comparing runs of the same file): `node bin/zerollm.js bench models/Qwen3-8B-Q4_K_M.gguf --threads 6 --kv f16 --n 64 --prompt-file prompt.txt --draft models/Qwen3-0.6B-Q8_0.gguf --draft-k 1,2,3 --draft-threads 4`.

What to look at: (a) is `speculative tok/s` above `plain tok/s` for some k and draft thread count? (b) what is the acceptance on chat-style text (and on code, if you ask for code)? (c) `verify ms/step` against the plain decode time: that is the real cost of verifying k+1 tokens on this machine; (d) does the draft slow the target (compare plain tok/s here with the earlier 3.1 tok/s; if it is lower, the two pools are fighting for cores or memory: try fewer draft threads); (e) RSS with both models loaded. Draft and target workers never run at the same time; the default is min(`--threads`, 4) draft threads.

### Dev VM (4 vCPU, noisy), synthetic models, indicative only

All models below are **synthetic random-weight GGUFs** made with `tools/make-tiny-gguf.js` (Qwen3 architecture, Q8_0, width 1024, 16/8 heads, head_dim 128, FFN 3072). A draft that agrees with the target is built by giving both the same seed and making the target's layers beyond the draft's depth contribute little (`fadeAfter` / `fade` options: target = 28 layers, draft = its first 3 or 7 layers, same embeddings and output head). **The dev VM is noisy**: plain decode of one file varied by up to 2x between runs at times, so `tools/specbench.mjs` runs plain and speculative decoding interleaved, in the same process, and the speed-up column is the median of the paired ratios (min-max in brackets). Greedy, 64 tokens per run, 5 rounds, 4 target workers, kernels wasm-simd, f32 KV. Everything here: *synthetic model on a 4-vCPU dev VM, indicative only*.

**1. Cost of one verification pass** (`forwardBatch(B tokens, {logits: 'all'})`) divided by the cost of one decode step, pair "cheap" target (28 layers, vocabulary 8000, 455 MB), 4 workers, median of 12 repetitions, ranges over 2-3 runs. B = k + 1 drafted-plus-pending tokens; "B sequential steps" is what plain decoding would pay for the same tokens (B x 1.0).

| B | before the kernel change | after (tiles of 3 and 2 tokens added) |
|---|---|---|
| 1 | 1.00 | 1.00 |
| 2 | 1.75-1.85 | 1.23-1.39 |
| 3 | 2.59-2.63 | 1.33-1.47 |
| 4 | 1.42-1.67 | 1.42-1.65 |
| 5 | 2.12-2.41 | 2.11-2.53 |
| 6 | 2.91-3.01 | 2.26-2.86 |
| 7 | 3.62-3.85 | 2.49-3.10 |
| 8 | 2.60-3.02 | 2.53-3.35 |

The batched kernels work in tiles of 4 tokens and used to do a remainder of 1-3 tokens as lone-token tiles (each re-widening the weight tile), so B = 3 cost more than B = 4 and about as much as three decode steps. B = 4 (k = 3) fills one tile, which is why `--draft-k` defaults to 3; B = 5 needs a second pass over the tile and costs ~50% more. (Also measured on a Qwen3-0.6B-shaped Q8_0 target with the full 151936-token vocabulary, old kernels, single run: B = 2/3/4/5/6/8 cost 1.53/1.87/1.78/2.09/2.44/3.05 decode steps.) A run on an 8B-shaped synthetic Q4_K_M model was too noisy to use (non-monotonic times) and is not reported.

**2. Speculative vs plain decoding** (`node tools/specbench.mjs target draft --threads 4 --draft-threads 1,2 --k 2,3,4,6 --n 64 --rounds 5`; after the kernel change):

Pair "cheap": target 28 layers / vocabulary 8000 (455 MB), draft = its first 3 layers (56 MB, draft pass ~0.1-0.2 of a target pass). Greedy output identical to plain in every run.

| draft threads | k | plain tok/s | speculative tok/s | speed-up (median paired, min-max) | accepted | tok/step | draft ms/step | verify ms/step |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 14.44 | 19.62 | 1.34x (1.30-1.49) | 81% | 2.56 | 30.5 | 99.7 |
| 1 | 3 | 14.44 | 20.86 | 1.48x (1.39-1.56) | 77% | 3.20 | 42.4 | 114.9 |
| 1 | 4 | 14.44 | 18.37 | 1.28x (1.16-1.30) | 75% | 4.00 | 52.2 | 161.6 |
| 1 | 6 | 14.44 | 19.06 | 1.32x (1.29-1.36) | 66% | 4.57 | 62.7 | 188.3 |
| 2 | 2 | 14.57 | 21.92 | 1.50x (1.42-1.67) | 81% | 2.56 | 19.2 | 100.3 |
| 2 | 3 | 14.57 | 23.89 | 1.62x (1.51-1.86) | 77% | 3.20 | 25.5 | 112.6 |
| 2 | 4 | 14.57 | 20.59 | 1.39x (1.18-1.55) | 75% | 4.00 | 29.5 | 161.9 |
| 2 | 6 | 14.57 | 20.95 | 1.46x (1.37-1.65) | 66% | 4.57 | 38.8 | 182.3 |

(Before the kernel change the same pair gave 0.87-1.58x with a dip at k = 2 (B = 3): k = 2/3/4/6 at 1 draft thread: 0.87x/1.58x/1.30x/1.02x.)

Pair "0.6B-shaped": target 28 layers with the real 151936-token vocabulary (604 MB), draft = its first 7 layers (269 MB: the big output head makes it only ~2.2x cheaper than the target per token, a poor draft), 2 draft threads. Draft/target cost ratio ~0.45 and 77-81% acceptance:

| k | plain tok/s | speculative tok/s | speed-up | accepted | tok/step | draft ms/step | verify ms/step |
|---|---|---|---|---|---|---|---|
| 2 | 12.53 | 13.89 | 1.11x (1.02-1.15) | 81% | 2.56 | 69.2 | 112.0 |
| 3 | 12.53 | 13.67 | 1.05x (1.00-1.14) | 77% | 3.20 | 111.9 | 130.4 |
| 4 | 12.53 | 12.12 | 0.98x (0.88-1.04) | 77% | 4.00 | 139.8 | 176.5 |

Worst cases, to show the cost when speculation does not pay (pair "0.6B-shaped" with a target whose extra layers matter more, so only 18-31% of the drafts are accepted; measured BEFORE the kernel change, so the verify column is pessimistic): 0.47x (0.34-0.67) at k = 2 with 31% accepted, 0.41x at k = 4 with 18% accepted. And an "oracle" draft (an identical copy of the 455 MB target as draft, 100% accepted, k = 3, 4 draft threads): 0.77x (0.74-0.79), 4.00 tok/step, with the loop's own bookkeeping (sampling, acceptance test, rollback, event-loop turns) costing **0.2 ms per step** of 361 ms: the machinery adds no measurable overhead, the loss is the draft pass costing as much as a target pass.

Rule of thumb from these tables: speculation pays when the draft pass is much cheaper than a target pass (<= ~0.25) AND at least ~70% of drafted tokens are accepted; the dev VM gives 1.3-1.6x in that regime and ~1.0-1.1x with a draft that costs 0.45 of the target.

**3. Default (non-speculative) path unchanged** (`zerollm bench cheap-target.gguf --threads 4 --n 40 --prompt-tokens 48`, 8 interleaved runs of the previous revision and this one, medians): prefill 38.8 vs 39.1 tok/s, decode 13.66 vs 13.53 tok/s, i.e. within noise.
