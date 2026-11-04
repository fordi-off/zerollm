# zerollm — Project Brief

> A zero-dependency, pure-code local LLM inference engine in Node.js (+ hand-built WebAssembly SIMD kernels) that runs GGUF models on a plain Windows laptop CPU with **no admin rights, no native addons, no cloud, no Ollama/llama.cpp**.

Working name: **zerollm** (can be renamed; alternatives considered: cairn, trickle, nodelm, pebble).
Brief written: 2026-11-04. License plan: MIT.

---

## 0. How to use this document (instructions for Claude Code)

1. Read this whole file before writing code. It records the goal, hard constraints, the exact target hardware, every design decision made so far and why, and a milestone plan with acceptance criteria.
2. Work **milestone by milestone** (section 20). Every milestone must leave the repo in a runnable, tested state. Commit after each one with a clear message.
3. Do **not** add npm dependencies. Do **not** add native addons. Do **not** shell out to external binaries. If you think you need a dependency, stop and ask the user first.
4. Prefer correctness first (simple JS reference path), then speed (workers, then WASM SIMD). Keep the reference path forever as the test oracle for the fast path.
5. If you can run commands on the user's laptop, run the benchmarks yourself. If not, give the user exact commands and ask them to paste the output.
6. Where this brief says "verify", it means: do not trust numbers or opcodes from this document blindly. Check against the GGUF/ggml/WebAssembly specs or against runtime behavior (`WebAssembly.validate`, unit tests).
7. Keep a running `docs/DECISIONS.md` and `docs/BENCHMARKS.md` as you go.

---

## 1. Goal and hard constraints

### Goal
Run real instruction-tuned LLMs (up to **8B parameters**, quantized) locally on a laptop CPU at *usable* speed, using only code that runs inside **Node.js** (Python is acceptable only for optional tooling). Usable means: a 4B model chats at a few tokens/second, an 8B model works in a patient "batch / long answer" mode at roughly 2–3 tokens/second.

### Hard constraints (non-negotiable)
| Constraint | Detail |
|---|---|
| **No admin rights** | Everything must run as a normal user. No installers, no services, no registry edits, no driver stuff. Node is a portable install. |
| **No Ollama / llama.cpp / cloud engines** | The whole point is a self-written engine ("pure code"). No calling out to other inference runtimes or APIs. |
| **No modules compiled in C** | No native addons (`.node` files), no node-gyp, no npm packages that ship native binaries. |
| **Available languages** | JavaScript (Node.js) and Python only. |
| **Zero npm dependencies** | Use only Node built-ins (`node:fs`, `node:worker_threads`, `node:os`, `node:http`, `node:test`, ...). |
| **No external compilers** | No emscripten, no clang, no wabt/binaryen as a *runtime or build requirement*. WebAssembly modules are produced by our own small JS emitter (see 13.3). |
| **Offline** | No network access needed at runtime. The user downloads `.gguf` model files manually (browser) into `models/`. |
| **Windows + PowerShell** | Target OS is Windows. Use `node:path`, avoid shell-specific scripts, no bash-only tooling. |

### Explicit non-goals (for now)
GPU/iGPU acceleration (Node has no WebGPU and native addons are banned), training, fine-tuning, multimodal models, MoE models, Mamba/RWKV, batching multiple concurrent users.

---

## 2. Target machine (measured by the user)

| Item | Value |
|---|---|
| CPU | **AMD Ryzen 5 7530U** with Radeon Graphics (Zen 3 "Barcelo-R", 15 W class mobile part) |
| Cores / threads | **6 physical cores / 12 logical processors**, max clock reported 2000 MHz (base; boosts higher) |
| SIMD | AVX2 available natively. No AVX-512. WebAssembly SIMD is 128-bit regardless. |
| RAM | **16 GB installed, 14.8 GB reported usable** (rest reserved for the iGPU) |
| RAM config | **Dual channel**: 2 × 8 GB, DDR4, **3200 MT/s**, configured 3200 (Channel A + Channel B) |
| Theoretical bandwidth | 2 × 8 B × 3200 MT/s = **51.2 GB/s**; realistic sustained **~25–35 GB/s** |
| GPU | AMD Radeon (TM) Graphics (integrated, unusable from plain Node) |
| OS | Windows, PowerShell |
| Node | **v26.5.1**, `WebAssembly` SIMD validate = **true**, `os.cpus().length` = **12** |

Consequences:
* LLM **decoding is memory-bandwidth-bound**: every generated token streams the entire weight set from RAM once. Speed ≈ effective bandwidth ÷ model bytes.
* Hyperthreads usually do **not** help a bandwidth-bound workload. Default to **6 workers**; benchmark 4 / 6 / 8 / 12.
* Free RAM is the real ceiling: Windows + a browser eat ~5–7 GB, leaving roughly **8–9 GB** for the engine if the user closes heavy apps. 8B Q4 (~4.6–5 GB) plus KV cache and buffers fits; 8B Q8 (~8.7 GB) is too tight to be a default.
* It's a 15 W U-series chip. Expect thermal / power-limit throttling under sustained all-core load. Benchmark plugged in with Windows power mode set to Best performance.

### Probe commands the user already ran (reuse them in a `zerollm doctor` command)
```powershell
Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
"{0:N1} GB total" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB); Get-CimInstance Win32_PhysicalMemory | Format-List BankLabel,Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType
```
```
node -e "console.log('node',process.version,'SIMD:',WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11])),'threads:',require('os').cpus().length)"
```

---

## 3. Decisions made so far (and why)

1. **Node.js is the primary language, not Python.**
   * Pure Python (no numpy) is not viable: ~0.01 tok/s even for small models. numpy is compiled C and violates the rules.
   * Node gives us: V8 JIT with typed arrays (`Float32Array`, `Int8Array`, `Uint8Array`, `DataView`), `worker_threads` + `SharedArrayBuffer` + `Atomics` for real multi-core parallelism, and **WebAssembly with SIMD built in** with zero installs.
   * Python may be used only for optional dev tooling (e.g. generating test vectors), never in the runtime path.
2. **WebAssembly SIMD is the biggest speed lever** (expected ~3–5x over plain JS for the matmul inner loop). Plain JS is the correctness reference and the fallback.
3. **Model file format: GGUF** (the llama.cpp ecosystem format). It's a single file with metadata, tokenizer and tensors, with many ready-made quantizations on Hugging Face.
4. **Quantized weights stay quantized in RAM**; dequantize inside the dot-product loop. Memory traffic (4–8x less than f32) matters more than raw FLOPs.
5. **Activations are quantized to int8 (Q8_0-style blocks) per matmul** so the inner loop is an integer dot product (like llama.cpp does). Float scales applied once per 32-weight block.
6. **Parallelism = SPMD workers with barriers.** One worker per physical core, each owning a *fixed row-slice of every weight matrix*, so each core streams its own memory region. Workers run the same control flow in lockstep, synchronized with `Atomics`.
7. **Each worker holds its slice of the weights inside its own private `WebAssembly.Memory`.** This sidesteps the 4 GB wasm32 limit entirely: 8B Q4 ≈ 4.6 GB ÷ 6 workers ≈ 0.8 GB per worker. Small activation vectors (a few KB to ~50 KB) are exchanged through a `SharedArrayBuffer` and copied into/out of each worker's memory. The copy cost is negligible compared to streaming ~0.8 GB of weights.
   * (Earlier in the planning conversation, "shard across several SharedArrayBuffers / multi-memory" was floated. The per-worker private memory design above supersedes it and is simpler.)
8. **Node cannot load WAT text.** Node/V8 only accepts the binary `.wasm` format; there is no built-in text-format parser. So we write a **tiny WebAssembly binary emitter in JS** (LEB128 encoding, sections, function bodies, `0xFD`-prefixed SIMD opcodes) and generate the kernel module at startup. This keeps the project dependency-free and auditable. (Fallback: also commit a prebuilt `.wasm` generated by the same emitter.)
9. **Read weights straight from disk into worker memory.** Node has no `mmap`. Each worker opens the GGUF itself and `fs.readSync`s only its own row range directly into a `Uint8Array` view of its wasm memory (rows are contiguous in GGUF, so a row range is one contiguous byte range). No giant main-thread buffer, no copies. Chunk reads to ≤ 256 MB (single reads are limited to ~2 GB and huge reads are unfriendly).
10. **Start simple on formats:** `Q8_0` first (simplest, accurate), then `Q4_0`, then the K-quants (`Q4_K`, `Q5_K`, `Q6_K`) because the most widely published "Q4_K_M" files mix Q4_K + Q6_K (+ sometimes Q5_K). Official Qwen3 GGUF repos ship Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0 (no plain Q4_0); community repos (e.g. bartowski) often publish Q4_0 too. So K-quant support is needed to use the most common files. A load-time **repack** step may convert any supported format into a SIMD-friendly internal layout.
11. **Model choices:**
    * Daily-driver: **Qwen3 4B** (Q4) — best quality/speed balance on this laptop.
    * "Patient mode": **Qwen3 8B** or **Llama 3.1 8B Instruct** (Q4).
    * Dev/test models: Qwen3 0.6B (Q8_0), Qwen3 1.7B, Llama 3.2 1B Instruct; and Karpathy's `stories15M` (llama2.c format) only for an optional first sanity check.
12. **Name:** zerollm (zero dependencies / zero admin / zero native code).

---

## 4. Performance expectations (estimates, to be validated by benchmarks)

Decode speed ceiling ≈ `effective_bandwidth / model_bytes`. With ~25 GB/s effective:

| Model | Quant | Weights in RAM | Bandwidth ceiling | Realistic target (6 workers, WASM SIMD) |
|---|---|---|---|---|
| Qwen3 8B (8.2B) | Q4 (~4.5 bpw) | ~4.6–5.0 GB | ~5 tok/s | **~2–3 tok/s** |
| Qwen3 8B | Q8_0 | ~8.7 GB | ~3 tok/s | ~1.2–1.8 tok/s (RAM too tight) |
| Qwen3 4B | Q4 | ~2.3–2.6 GB | ~10 tok/s | **~4–7 tok/s** |
| Qwen3 1.7B | Q4 | ~1.1 GB | ~22 tok/s | **~10–16 tok/s** |
| Qwen3 0.6B | Q8_0 | ~0.64 GB | ~39 tok/s | **~25+ tok/s** |

Notes:
* A first plain-JS implementation will be roughly **3–4x slower** than these targets. The WASM SIMD kernel + workers is what gets there.
* **Prefill (prompt processing) is compute-bound** (~16 GFLOP/token for 8B). Naive per-token prefill is about decode speed (a 500-token prompt = minutes on 8B). **Batched prefill** (process N prompt tokens per pass over each weight row so weights are read once per batch) is a required optimization (milestone 7). Keep system prompts short and support caching the KV state of a fixed prefix on disk.
* With a *single* memory channel numbers would halve; this machine is dual-channel, so that's the good case.
* These are estimates, not promises. The plan is to measure, then tune (section 19).

---

## 5. Architecture overview

```
CLI / HTTP server
      │
   Engine (model.generate(), streaming tokens)
      │
 ┌────┴───────────────────────────────────────────────────────────┐
 │ Tokenizer (byte-level BPE)   Chat templates   Sampler          │
 └────┬───────────────────────────────────────────────────────────┘
      │ token ids
 Transformer forward pass (SPMD across N worker threads)
      │
 ┌────┴────────────────────────┐   ┌──────────────────────────────┐
 │ Worker 0..N-1               │   │ Shared state (SharedArrayBuf)│
 │  - private wasm Memory with │◄─►│  - activation vectors        │
 │    its weight row-slices    │   │  - barrier / sync words      │
 │  - SIMD kernels (WASM)      │   │  - KV cache (or per-worker)  │
 │  - JS fallback kernels      │   │  - logits                    │
 └─────────────────────────────┘   └──────────────────────────────┘
      ▲
 GGUF reader (header, metadata, tensor infos) → each worker reads its own row ranges from disk
```

---

## 6. Proposed repository layout

```
zerollm/
  package.json            # "type": "module", NO dependencies
  README.md
  CLAUDE.md               # short pointer: "read docs/BRIEF.md (this file)"
  LICENSE                 # MIT
  .gitignore              # models/, *.gguf, bench-out/, node_modules/
  bin/zerollm.js          # CLI entry (info | doctor | run | chat | bench | serve)
  src/
    gguf/
      reader.js           # header + metadata KV + tensor infos (BigInt-safe offsets)
      types.js            # ggml type ids, block sizes, bytes-per-row helpers
    model/
      config.js           # arch detection + hyperparameters from GGUF metadata
      loader.js           # slice planning, per-worker weight loading
      forward.js          # reference (single-thread JS) forward pass
      kvcache.js
    tokenizer/
      bpe.js              # byte-level BPE (GPT-2 style) + special tokens
      chat-templates.js   # qwen (ChatML), llama3
    sampling/sampler.js
    compute/
      ref-kernels.js      # plain JS dequant + dot products (the oracle)
      quant.js            # Q8_0 activation quantization, block helpers
      wasm/
        builder.js        # minimal wasm binary emitter (LEB128, sections, SIMD ops)
        kernels.js        # generates matmul/dot kernels as wasm bytes
        features.js       # runtime probes: simd, relaxed simd, memory64, threads...
      pool.js             # worker pool, SPMD dispatch
      worker.js           # worker entry
      barrier.js          # Atomics-based sense-reversing barrier
    server/http.js        # optional OpenAI-compatible server on node:http
    util/{log.js,args.js,timer.js}
  test/                   # node --test (built-in runner)
  tools/
    make-tiny-gguf.js     # random tiny model for tests
    bench.js
  docs/{BRIEF.md,DECISIONS.md,BENCHMARKS.md}
  models/                 # gitignored; user drops .gguf files here
```

Code style: ES modules, plain JS with JSDoc types (no TypeScript/build step), `node:test` for tests, small files, no global state.

---

## 7. GGUF reader — format details (verify against the spec while implementing)

Little-endian throughout.

```
magic        : 4 bytes  "GGUF"
version      : uint32   (expect 3)
tensor_count : uint64
metadata_kv_count : uint64
metadata KVs : repeated { key: string, value_type: uint32, value }
tensor infos : repeated { name: string, n_dims: uint32, dims: uint64[n_dims], type: uint32 (ggml type), offset: uint64 }
padding to alignment (general.alignment, default 32)
tensor data  : tensors at (data_start + offset)
```
* `string` = `uint64 length` + UTF-8 bytes (no terminator).
* Value types: 0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32, 6 float32, 7 bool, 8 string, 9 array (`elem_type: uint32`, `len: uint64`, elems), 10 uint64, 11 int64, 12 float64.
* Tensor `offset` is relative to the start of the data section. `dims` are in ggml order (dims[0] = innermost/row length). A weight matrix `[in, out]` in ggml terms is stored as `out` rows of `in` elements, row-major, each row consisting of whole quant blocks. So **a row range is a contiguous byte range** — this is what makes per-worker slice loading trivial.
* Files can be many GB: use `fs.openSync` + `fs.readSync(fd, buffer, off, len, position)` and **never read the whole file into one Buffer**. Positions beyond 2^31 must be passed as Number (safe up to 2^53) or BigInt. Parse header/metadata/tokenizer by reading in chunks (the tokenizer vocab for these models is ~150k tokens, a few MB).
* Use `BigInt` (or careful Number) for uint64 fields; convert to Number only after range checking.
* `zerollm info model.gguf` must print: architecture, all hyperparameters, quant types histogram, total bytes, tokenizer summary, chat template (if present), estimated RAM need.

### ggml type ids (verify)
`0 F32, 1 F16, 2 Q4_0, 3 Q4_1, 6 Q5_0, 7 Q5_1, 8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 15 Q8_K, 30 BF16`.

---

## 8. Quantization formats to implement

| Type | Weights/block | Bytes/block | Bits/weight | Layout |
|---|---|---|---|---|
| F32 / F16 / BF16 | 1 | 4 / 2 / 2 | 32 / 16 / 16 | norm weights, biases, maybe embeddings |
| **Q8_0** | 32 | 34 | 8.5 | f16 scale `d`, then 32 × int8; `w = d * q` |
| **Q4_0** | 32 | 18 | 4.5 | f16 `d`, then 16 bytes of nibbles; low nibbles = elements 0–15, high nibbles = elements 16–31; `w = d * (nibble - 8)` |
| Q5_0 | 32 | 22 | 5.5 | f16 `d`, 4-byte high-bit field `qh`, 16 bytes `qs`; `w = d * (q5 - 16)` |
| Q4_K | 256 | 144 | 4.5 | f16 `d`, f16 `dmin`, 12 bytes packed 6-bit scales/mins (8 sub-blocks of 32), 128 bytes nibbles |
| Q5_K | 256 | 176 | 5.5 | like Q4_K plus 32 bytes high bits |
| Q6_K | 256 | 210 | 6.56 | 128 bytes `ql`, 64 bytes `qh`, 16 × int8 scales, f16 `d` |

Verify every layout against the ggml source/spec knowledge before trusting the table. Write **unit tests that dequantize synthetic blocks** and compare to a straightforward independent reference.

### Dot product scheme (decode inner loop)
* For each matmul `y = W x`: quantize `x` once into Q8_0 blocks (f32 scale per 32 elements + int8 values). Share this quantized `x` across all matmuls that consume the same input (Q,K,V; gate,up).
* For every 32-weight block: `sum_i w_q[i] * x_q[i]` as int32, then `acc += float(sum) * d_w * d_x`.
* For Q4_0 the `-8` offset can be handled as `sum(q*xq) - 8*sum(xq)` with a precomputed per-block `sum(xq)`, or by subtracting 8 after unpacking.
* Accumulate in f32 per output row; write f32 outputs.

---

## 9. Architectures to support (all read from GGUF metadata, never hard-code sizes)

Priority: **qwen3**, **llama** (3.x), **qwen2**. Detect via `general.architecture`.

### Metadata keys (`{arch}` = `llama` / `qwen2` / `qwen3`)
`{arch}.context_length`, `{arch}.embedding_length`, `{arch}.block_count`, `{arch}.feed_forward_length`, `{arch}.attention.head_count`, `{arch}.attention.head_count_kv`, `{arch}.attention.key_length` (explicit head_dim; **Qwen3 has head_dim 128 which is NOT hidden/heads**), `{arch}.attention.layer_norm_rms_epsilon`, `{arch}.rope.freq_base`, `{arch}.rope.dimension_count`, `general.alignment`, `tokenizer.*` (see 10).

### Tensor names
```
token_embd.weight
output_norm.weight
output.weight                 # may be ABSENT → tied embeddings: use token_embd.weight as the output matrix
blk.N.attn_norm.weight
blk.N.attn_q.weight  blk.N.attn_k.weight  blk.N.attn_v.weight  blk.N.attn_output.weight
blk.N.attn_q.bias / attn_k.bias / attn_v.bias         # Qwen2 only
blk.N.attn_q_norm.weight  blk.N.attn_k_norm.weight    # Qwen3 QK-norm (size = head_dim)
blk.N.ffn_norm.weight
blk.N.ffn_gate.weight  blk.N.ffn_up.weight  blk.N.ffn_down.weight
rope_freqs.weight             # Llama 3.1 GGUFs: per-dimension RoPE frequency factors (use if present)
```

### Approximate shapes (for sizing only; always read from metadata)
| Model | hidden | layers | heads | kv heads | head_dim | ffn | vocab | tied emb |
|---|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | yes |
| Qwen3 1.7B | 2048 | 28 | 16 | 8 | 128 | 6144 | 151936 | yes |
| Qwen3 4B | 2560 | 36 | 32 | 8 | 128 | 9728 | 151936 | yes |
| Qwen3 8B | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | no |
| Llama 3.1 8B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | no |
| Llama 3.2 1B | 2048 | 16 | 32 | 8 | 64 | 8192 | 128256 | yes |

### RoPE conventions (important, easy to get wrong)
* **`llama` arch**: llama.cpp's converter permutes Q/K weights so RoPE rotates **adjacent pairs** (dims `2i, 2i+1`). Use the adjacent-pair ("normal") variant on the stored weights.
* **`qwen2` / `qwen3` arch**: **NEOX / split-half** variant — pairs are `(i, i + head_dim/2)`.
* `freq_i = freq_base^(-2i/head_dim)`; if `rope_freqs.weight` exists, divide by those factors. Precompute cos/sin tables for `max_ctx × head_dim/2`.
* Qwen3 uses `freq_base = 1e6`; Llama 3.x uses `5e5` (+ the `rope_freqs.weight` scaling). No YaRN/long-context scaling needed initially; cap context at what the user requests (default 2048–4096).

---

## 10. Tokenizer (byte-level BPE, written from scratch)

Read from GGUF: `tokenizer.ggml.model` (`gpt2` = byte-level BPE for Qwen/Llama3), `tokenizer.ggml.pre` (pre-tokenizer id, e.g. `qwen2`, `llama-bpe`), `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type` (3 = control, 4 = user-defined/added), `tokenizer.ggml.merges` (strings `"a b"`, order = rank), `tokenizer.ggml.bos_token_id`, `eos_token_id`, `tokenizer.chat_template`.

Pipeline:
1. **Split out special tokens** first (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|start_header_id|>`, ...) — they map directly to ids and are never BPE'd.
2. **Pre-tokenize** the remaining text with the model's regex (JS `u`-flag regexes support `\p{L}`/`\p{N}`):
   * Qwen2/Qwen3 (digits split individually):
     `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
   * Llama 3 (digits grouped 1–3): same but `\p{N}{1,3}` instead of `\p{N}`.
   * If the Node build lacks inline `(?i:...)` modifiers, expand to explicit case classes (`[sS]`, ...). Probe in `doctor`.
3. **Byte → unicode** mapping (GPT-2 `bytes_to_unicode`), then **apply merges by rank** (repeatedly merge the lowest-rank adjacent pair; use a rank `Map` keyed by `"a b"`, and a small priority structure so long words stay fast). Cache results per pre-token.
4. Map resulting strings to ids with a `Map`.
5. **Decode**: ids → token strings → unicode→byte → UTF-8. Use `TextDecoder` with `{stream: true}` so multi-byte characters split across tokens print correctly while streaming.
6. Add BOS only if the model/template requires it (Qwen: no BOS; Llama 3: `<|begin_of_text|>`).

Tests: round-trip `decode(encode(s)) === s` over ASCII, code, CJK, emoji, whitespace-heavy text, and strings containing special-token text. Compare token counts against known-good values the user can supply from another tool if available.

---

## 11. Transformer forward pass (per token / per batch)

```
x = embed[token]                                  # dequantize one row of token_embd
for each layer l:
    h  = rmsnorm(x, attn_norm[l], eps)
    q  = Wq h (+bq)   k = Wk h (+bk)   v = Wv h (+bv)
    reshape q → [n_head, head_dim], k,v → [n_kv_head, head_dim]
    (Qwen3) q = rmsnorm_per_head(q, attn_q_norm); k = rmsnorm_per_head(k, attn_k_norm)   # BEFORE RoPE
    apply RoPE to q and k at position pos
    write k, v into KV cache at pos
    for each head h:  kvh = h / (n_head / n_kv_head)            # GQA
        scores[t] = dot(q_h, K[kvh][t]) / sqrt(head_dim)  for t in 0..pos
        p = softmax(scores)                                      # f32, numerically stable
        o_h = sum_t p[t] * V[kvh][t]
    a  = Wo concat(o_h)
    x  = x + a
    h  = rmsnorm(x, ffn_norm[l], eps)
    f  = Wdown( silu(Wgate h) * (Wup h) )                        # SwiGLU
    x  = x + f
x = rmsnorm(x, output_norm, eps)
logits = Wout x                                   # (or token_embd if tied) — only for the LAST token when prefilling
```
* `rmsnorm(x, w) = x / sqrt(mean(x²) + eps) * w`. `silu(z) = z / (1 + e^-z)`.
* Keep residual stream, norms, softmax, RoPE in **f32**.
* The `Wout` matmul (e.g. 151936 × 4096 for Qwen3 8B) is ~10%+ of per-token cost: split its rows across workers too; skip it for all but the last prompt token.

### KV cache memory
Per token = `2 × n_layer × n_kv_head × head_dim` values.
* Qwen3 8B: 2×36×8×128 = 73,728 values → **~288 KB/token in f32**, so 2048 ctx ≈ 590 MB, 4096 ctx ≈ 1.2 GB.
* Llama 3.1 8B: 2×32×8×128 = 65,536 values → ~256 KB/token in f32.
* Default max context **2048** (configurable). Add an **f16 KV cache** option later (halves it). Preallocate once; the user is RAM-constrained.

---

## 12. Memory budget (8B Q4 on this laptop)

| Item | Size |
|---|---|
| Weights (Q4-class) | ~4.6–5.0 GB, split ≈ 0.8 GB per worker memory |
| KV cache (2048 ctx, f32) | ~0.6 GB |
| Activations / scratch / tables | < 0.1 GB |
| Node + V8 + workers overhead | ~0.3–0.5 GB |
| **Total** | **~6–6.5 GB** (needs ~8–9 GB free RAM — user should close heavy apps) |

`zerollm doctor` and `info` should estimate RAM needed vs `os.freemem()` and warn before loading. Large `WebAssembly.Memory` allocations can fail if the OS can't commit memory — handle the error with a clear message.

---

## 13. Compute kernels

### 13.1 JS reference kernels (milestone 3)
Plain JS dequant + dot products over `Uint8Array`/`DataView`/`Float32Array`/`Int8Array`. Slow but obviously correct. **Keep forever as the oracle** for every optimized kernel.

### 13.2 WASM SIMD kernels (milestone 6)
* Core kernel: `matvec_q8_0(w_ptr, x_ptr, y_ptr, rows, blocks_per_row)` and the Q4_0 equivalent; later Q4_K/Q6_K.
* Use 128-bit lanes: load weights (`v128.load`), widen int8→int16 (`i16x8.extend_low/high_i8x16_s`), multiply-accumulate pairs with `i32x4.dot_i16x8_s`, add into `i32x4` accumulators, convert to f32 (`f32x4.convert_i32x4_s`), multiply by block scales, accumulate.
* **Optional fast path:** *relaxed SIMD* integer dot (`i32x4.relaxed_dot_i8x16_i7x16_add_s`) if Node 26's V8 supports it (probe with `WebAssembly.validate`). Gate behind feature detection; keep the plain SIMD path as default.
* **Register blocking:** process 2–4 weight rows at once sharing the loaded activation vector; unroll across blocks; keep several independent accumulators to hide latency.
* Optional **row-interleaved repack** at load time (like llama.cpp's interleaved Q4_0 variants) so the 4-row kernel reads contiguous memory.
* For **batched prefill**: kernel variant that dots each weight row against N (e.g. 4–16) activation vectors, so weights are loaded once per N tokens.
* Look up opcodes in the WebAssembly SIMD spec's opcode table; **do not trust memory**. Validate every emitted module with `WebAssembly.validate` and test against the JS oracle with random data.

### 13.3 Building WASM without a compiler (`src/compute/wasm/builder.js`)
Node cannot parse `.wat`, so write a small emitter that outputs the binary format:
* Module header `\0asm` + version 1; sections: type, import, function, memory (or import memory), export, code.
* LEB128 (unsigned/signed) encoders; helper to emit `local.get/set`, `i32.const`, `i32.add/mul/shl`, loops/blocks/`br_if`, `v128.load/store` (with alignment/offset immediates), and the `0xFD`-prefixed SIMD opcode (opcode number encoded as LEB128).
* Provide a tiny structured API so kernels read like assembly in JS (e.g. `f.loop(() => {...})`), plus a `disassemble-lite` debug printer if useful.
* Generate the module at startup (fast), and optionally cache the bytes to `dist/kernels.wasm`.
* Test harness: build a trivial module (add two numbers), then a SIMD one, then the real kernels, comparing against `ref-kernels.js`.

### 13.4 Feature probes (`features.js`, shown by `zerollm doctor`)
`WebAssembly.validate` probes for: SIMD, relaxed SIMD, threads/atomics, multi-memory, memory64, bulk memory; plus: `SharedArrayBuffer` + `Atomics.wait` availability, regex inline-modifier support, max single `WebAssembly.Memory` allocation succeeded (try 1 GB, 2 GB, 3 GB), `os.availableParallelism()`, free RAM.

---

## 14. Threading / worker design

* **Pool:** `N` workers (default = physical cores = 6; configurable). Use `worker_threads` with `workerData` carrying the `SharedArrayBuffer`s and config.
* **SPMD lockstep:** every worker runs the same forward-pass control flow for a token (or batch). Phases are separated by **barriers**. No per-op message passing (`postMessage` is far too slow per matmul).
* **Barrier:** sense-reversing barrier on a shared `Int32Array`: `Atomics.add` arrival counter, last arriver resets and flips the generation word and `Atomics.notify`; others **spin briefly** (`Atomics.load` loop, ~1–5 µs) then fall back to `Atomics.wait`. Tune spin length in benchmarking. Expect ~250–400 barriers per token; at ~10–30 µs each that's a few ms — fine against a 300–500 ms token.
* **Work split:** every matmul `y = W x` is split by **output rows**: worker `i` computes rows `[r_i, r_{i+1})` (round to multiples of 4 for the blocked kernel). Elementwise ops (norms, RoPE, SiLU, residual add) are tiny: either do them redundantly on every worker over the shared vector, or split by element range.
* **Attention:** split by head (ceil-division across workers; 32 heads / 6 workers is uneven — fine). Later optimization: split by (head, position chunk) with a reduction.
* **Data flow per matmul:** (1) one worker (or all, redundantly) quantizes the activation vector to Q8_0 blocks into shared memory; (2) barrier; (3) each worker copies the quantized vector into its private wasm memory (a few KB–16 KB) and runs the kernel over its rows; (4) each worker writes its output slice into the shared f32 vector; (5) barrier.
* **Loading:** the main thread parses the GGUF header, computes the row split per tensor, tells each worker which (tensor, row range, file offset, byte length) to load; each worker `readSync`s directly into its wasm memory. Load time from a slow disk can be 10–30+ s; show a progress bar.
* **Process priority:** optionally raise priority with `os.setPriority` (HIGH is normally permitted for a normal user; do **not** use anything needing admin). Node cannot pin threads to cores on Windows, so benchmark worker counts empirically (4/6/8/12).
* **Shutdown:** terminate workers cleanly on exit/Ctrl+C.

---

## 15. Sampling

Greedy (temperature 0), temperature, top-k, top-p, min-p, repetition/presence penalty, seeded PRNG (e.g. mulberry32/xoshiro) for reproducibility. Stop on EOS ids (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`), stop strings, `max_tokens`. Apply the sampler on one worker (or main) after the logits are assembled. Avoid sorting all 150k logits per token when top-k is small (use a partial selection).

---

## 16. Chat templates

Implement the two families directly in JS (don't try to execute Jinja):
* **Qwen (ChatML):** `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. Qwen3 has a thinking mode: for non-thinking replies append `<think>\n\n</think>\n\n` after the assistant header (or support `/no_think`); expose `--think` / `--no-think`.
* **Llama 3.x:** `<|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`, BOS `<|begin_of_text|>` first.
* Detect from `general.architecture`/`tokenizer.chat_template` contents; allow `--template` override and raw (no template) completion mode.

---

## 17. CLI and API

```
zerollm doctor                       # environment + feature probe + RAM estimate
zerollm info  models/x.gguf          # metadata, tensors, quant histogram, RAM estimate
zerollm run   models/x.gguf -p "..." [--n 256] [--temp 0.7] [--top-k 40] [--top-p 0.9] [--threads 6] [--ctx 2048] [--seed 1]
zerollm chat  models/x.gguf [--system "..."] [--no-think]     # interactive, streaming
zerollm bench models/x.gguf [--threads 4,6,8,12]               # prints prefill + decode tok/s
zerollm serve models/x.gguf [--port 8080]                      # optional OpenAI-compatible /v1/chat/completions, SSE streaming, node:http only
```
Stream tokens as they're produced. Print timing: load time, prefill tok/s, decode tok/s, RAM used.

---

## 18. Testing and validation

Use `node --test` (built-in). Tests to write:
1. **GGUF reader:** parse a tiny synthetic GGUF built by `tools/make-tiny-gguf.js` (all value types, arrays, alignment).
2. **Dequant/quant:** every supported block type vs an independent straightforward reference on random blocks.
3. **Kernel parity:** every WASM kernel vs JS oracle on random data; tolerance on f32 accumulation order.
4. **Tokenizer:** round-trips + special tokens + unicode edge cases.
5. **Forward pass:** consistency between single-thread JS path and worker/WASM path on a random tiny model (logits within tolerance, identical greedy tokens).
6. **Real-model sanity:** a documented manual check — Qwen3 0.6B answers a simple question coherently; greedy output is deterministic across runs and across worker counts.
7. **If a reference is available** (user has another machine with HF transformers or similar): compare top-5 logits for a fixed prompt to catch RoPE/permutation/norm mistakes. Most "garbled output" bugs come from: wrong RoPE variant, missing Qwen3 QK-norm, wrong head_dim, wrong tokenizer pre-regex, wrong template.

---

## 19. Benchmarking and tuning plan

Record in `docs/BENCHMARKS.md` (date, commit, model, quant, threads, ctx, prefill tok/s, decode tok/s, RAM). Knobs to explore:
* Worker count 4 / 6 / 8 / 12 (and spin-vs-wait barrier thresholds).
* Rows per worker rounding, 1/2/4-row register blocking, loop unrolling, int8 vs relaxed-dot path.
* Interleaved repack on/off.
* Batched prefill batch size (4, 8, 16, 32).
* f32 vs f16 KV cache; context length impact on attention cost.
* Power plan: plugged in, Best performance; watch for thermal throttling (re-run after a few minutes of load).
* Sanity bound: compare decode tok/s against `effective_bandwidth / model_bytes`. If far below (e.g. < 40%), the kernel or sync overhead is the bottleneck, not RAM.

---

## 20. Milestones (with acceptance criteria)

| # | Milestone | Done when |
|---|---|---|
| **M0** | Scaffold repo, `package.json` (no deps), CLI skeleton, `doctor` command, README, .gitignore, MIT license, copy this brief into `docs/BRIEF.md` | `node bin/zerollm.js doctor` prints Node version, cores, RAM, all feature probes |
| **M1** | GGUF reader + `info` | `info` works on a real GGUF (e.g. Qwen3 0.6B) and prints correct hyperparameters, tokenizer summary, quant histogram, RAM estimate; unit tests pass |
| **M2** | Tokenizer + chat templates | Round-trip tests pass; special tokens handled; Qwen and Llama templates render correctly |
| **M3** | Reference forward pass in JS (single thread, Q8_0 then Q4_0), KV cache, greedy decode | Qwen3 0.6B Q8_0 produces coherent text for a simple prompt; ≥ ~1 tok/s single thread |
| **M4** | Sampler + streaming `run`/`chat` CLI | Temperature/top-k/top-p work; seeded runs reproducible; Ctrl+C clean |
| **M5** | Worker pool + barriers + per-worker row-sliced weight loading (still JS kernels) | Identical greedy output to M3; near-linear scaling to ~4–6 workers on 0.6B; load time reported |
| **M6** | WASM module emitter + SIMD Q8_0/Q4_0 kernels with per-worker private memories | Kernel parity tests pass; 0.6B Q8 ≥ ~15–25 tok/s; 1.7B Q4 ≥ ~8–12 tok/s; 4B Q4 ≥ ~4 tok/s |
| **M7** | Batched prefill + last-token-only logits + KV prefix cache (disk) | Prefill clearly faster than decode (≥ 2x on small models); repeated system prompt skipped on later runs |
| **M8** | K-quants (Q4_K, Q5_K, Q6_K) + Q5_0 (+ optional repack) | Standard Q4_K_M GGUFs from Hugging Face load and produce correct output |
| **M9** | 8B tuning (thread count, interleave, f16 KV, memory checks) | Qwen3 8B Q4 / Llama 3.1 8B Q4 run in ≤ ~9 GB RAM at ≥ ~2 tok/s decode, documented in BENCHMARKS.md |
| **M10** | Optional: OpenAI-compatible HTTP server, context shifting, GGUF metadata-driven stop tokens, polish/README/docs | `curl`/any OpenAI client can chat against it |

Do M0–M4 first; they give a working (slow) engine and a trustworthy oracle before any optimization.

---

## 21. Risks and gotchas

* **Correctness traps:** RoPE pair convention (adjacent vs split-half), Qwen3's QK-norm, explicit `head_dim` ≠ hidden/heads, tied embeddings (`output.weight` missing), Qwen2 biases, `rope_freqs.weight` for Llama 3.1, Llama Q/K permutation assumptions, tokenizer pre-regex differences (digit grouping).
* **GGUF quirks:** alignment padding before tensor data; tensor offsets relative to data start; uint64 values; models can have mixed quant types per tensor (Q4_K_M mixes Q4_K/Q6_K/Q5_K) — dispatch per tensor.
* **Memory:** big `WebAssembly.Memory` allocations can fail; KV cache can quietly eat >1 GB; 8 GB machines would swap (this one has 16 GB). Check `os.freemem()` before load.
* **Node/V8 limits:** single `readSync` ≤ ~2 GB; wasm32 memory ≤ 4 GB (avoided via per-worker memories); `Atomics.wait` wakeup latency (use brief spinning); no mmap.
* **Windows power/thermal:** battery mode or Balanced plan can cut speed dramatically; sustained load throttles a 15 W chip.
* **WASM SIMD is 128-bit only:** it will not match native AVX2 llama.cpp speeds; expect roughly a fraction of it. That's accepted.
* **Don't claim speeds without measuring.** Put real numbers in BENCHMARKS.md.

---

## 22. Open questions for the user (ask early, don't block on them)

1. Is the final repo name **zerollm** (or one of the alternatives)? Public or private repo?
2. Which model files does the user already have downloaded (name + quant)? If none: suggest Qwen3-0.6B Q8_0 first (testing), then a Qwen3-4B Q4-class and a Qwen3-8B / Llama-3.1-8B Q4-class file.
3. Is Claude Code running on the target laptop (so it can run benchmarks) or elsewhere?
4. Preference for CLI-only vs also an HTTP server / simple web UI later?
5. Max context the user actually needs (default 2048)?

---

## 23. Appendix: quick reference

**Model files** (download manually from Hugging Face into `models/`): Qwen3 0.6B / 1.7B / 4B / 8B GGUF, Llama 3.2 1B Instruct, Llama 3.1 8B Instruct. Prefer Q8_0 for small test models and Q4_0 / Q4_K_M for 4B–8B. Always check that the file size fits in free RAM.

**Back-of-envelope:**
* Bytes per token (decode) ≈ model file size.
* tok/s ≈ effective_GB_per_s / model_GB; this laptop ≈ 25–35 GB/s peak, expect 40–60% of that in practice.
* FLOPs per token ≈ 2 × parameters (8B → ~16 GFLOP).

**Summary of the plan in one paragraph:** build a dependency-free Node.js GGUF inference engine; get a slow-but-correct single-thread JS reference working on a tiny model; add a tokenizer, chat templates and sampler; parallelize with SPMD worker threads that each own a row-slice of every weight matrix in their own WebAssembly memory; replace the matmul inner loop with hand-emitted WebAssembly SIMD int8 dot-product kernels generated by our own tiny wasm emitter; add batched prefill and K-quant support; tune for the Ryzen 5 7530U (6 workers, dual-channel DDR4-3200) until Qwen3 4B Q4 chats at a few tok/s and 8B Q4 runs at ~2–3 tok/s.
