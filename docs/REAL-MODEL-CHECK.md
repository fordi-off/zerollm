# Checking a real model (needs a machine with internet; the dev container could not reach Hugging Face)

Everything so far is validated against synthetic tiny models and an independent naive oracle. The following has **not** been run against a real GGUF yet. Please run it on the laptop and paste the output back.

1. Download `Qwen3-0.6B-Q8_0.gguf` (Qwen/Qwen3-0.6B-GGUF on Hugging Face) into `models/`.
2. `node bin/zerollm.js doctor` — paste all of it.
3. `node bin/zerollm.js info models/Qwen3-0.6B-Q8_0.gguf` — expect: arch qwen3, 28 layers, hidden 1024, 16 heads / 8 KV, head dim 128 (!= 64), vocab 151936, tied embeddings, QK-norm yes, Q8_0 + F32 tensors.
4. `node bin/zerollm.js tokenize models/Qwen3-0.6B-Q8_0.gguf "Hello, world! 123"` — expect pieces like `Hello`, `,`, `Ġworld`, `!`, `Ġ`, `1`, `2`, `3`.
5. `node bin/zerollm.js run models/Qwen3-0.6B-Q8_0.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60` — expect a coherent answer mentioning Paris, roughly 1-2 tok/s single-threaded. Run it twice: greedy output must be identical.
6. If the answer is garbled, the usual suspects (brief section 21) are RoPE variant, QK-norm, head_dim, tokenizer pre-regex, chat template. Send the exact output of steps 3-5 and I will bisect with `tokenize` and per-layer checks.

If you have another machine with HF transformers / llama.cpp: compare the top-5 next-token logits for a fixed prompt (brief section 18, item 7).

## M5: worker pool (after `git pull`)

```
node bin/zerollm.js bench models/Qwen3-0.6B-Q8_0.gguf --threads 1,2,4,6,8,12 --prompt-tokens 32 --n 32
```

Close the browser and other heavy apps first, plug in the charger and set the Windows power mode to Best performance. Paste the whole table. Expected: `greedy output identical across thread counts: yes`, decode speed rising with threads (likely flattening around 6 = physical cores), load time under a few seconds.

## M6: WASM SIMD kernels (after `git pull`)

1. Correctness on the real model (the SIMD path is the new default):

```
node bin/zerollm.js run models/Qwen3-0.6B-Q8_0.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60
node bin/zerollm.js run models/Qwen3-0.6B-Q8_0.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60 --threads 1 --kernels js
```

Both should say Paris. The second is the untouched reference path. The words may differ slightly after a few tokens (float summation order) but both must be coherent.

2. Speed:

```
node bin/zerollm.js bench models/Qwen3-0.6B-Q8_0.gguf --kernels js,wasm --threads 1,4,6,8,12 --prompt-tokens 32 --n 32
```

(`bench` may print "NO (differs ...)" for the output check: see docs/DECISIONS.md, M6 numerics; compare the text from step 1 instead.)

3. Where the time goes (optional): set `ZEROLLM_PROFILE=1` (PowerShell: `$env:ZEROLLM_PROFILE=1`) before a `bench --kernels wasm --threads 6` run.

## M7: batched prefill + system-prompt cache (after `git pull`)

```
node bin/zerollm.js bench models/Qwen3-0.6B-Q8_0.gguf --kernels wasm --threads 6 --batch 1,8,16,32 --prompt-tokens 128 --n 16
```

Expect prefill tok/s to be roughly 2-3x the decode tok/s at batch 8-16 (the "prefill/decode" column). Then the cache (use a system prompt of at least ~30 words; run the same command twice):

```
node bin/zerollm.js run models/Qwen3-0.6B-Q8_0.gguf --system "You are a careful assistant who always answers in complete sentences and explains each step of its reasoning clearly before giving a final answer." -p "What is the capital of France?" --temp 0 --n 30
```

First run ends with `system-prompt cache: saved`, the second with `system-prompt cache: hit` and the same answer; a `models\.zerollm-cache` folder appears (safe to delete).

## M8: K-quants (after `git pull`) -- the first real-model test of these formats

The K-quant layouts are verified against the official gguf-py implementation on synthetic blocks, but no real Q4_K_M file has been run yet, so this is the important check. Download a Q4_K_M file into `models\` (for example Qwen3-4B-Q4_K_M.gguf, about 2.5 GB, or a smaller Qwen3 Q4_K_M if you prefer to test quickly) and run:

```
node bin/zerollm.js info models/Qwen3-4B-Q4_K_M.gguf
node bin/zerollm.js run models/Qwen3-4B-Q4_K_M.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60
node bin/zerollm.js bench models/Qwen3-4B-Q4_K_M.gguf --threads 6 --batch 1,16 --prompt-tokens 64 --n 16
```

`info` should list Q4_K and Q6_K (and show 4.5 / 6.56 bits per weight) with every type marked supported. The answer must be coherent and mention Paris; if the text is garbled, send the exact output (a wrong bit layout gives gibberish immediately). Compare with `--kernels js --threads 1` for a few tokens if you want to cross-check the SIMD path against the reference (slow on a 4B model: use `--n 8`).

Note: these first-version K-quant kernels are slower per weight than Q8_0 and have no batched prefill yet (see docs/BENCHMARKS.md); speed work is M9.

## M9: 8B models and memory (after `git pull`)

`info` now prints a RAM estimate for both KV precisions and warns if the model will not fit in the free RAM; `run`/`chat`/`bench` accept `--kv f16` (half-size KV cache).

For the Qwen3-8B Q4_K_M file (about 5 GB; download into `models\`), close the browser and other heavy apps first, then:

```
node bin/zerollm.js info models/Qwen3-8B-Q4_K_M.gguf --threads 6
node bin/zerollm.js run models/Qwen3-8B-Q4_K_M.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60 --kv f16
node bin/zerollm.js bench models/Qwen3-8B-Q4_K_M.gguf --threads 6 --batch 1,16 --prompt-tokens 64 --n 16 --kv f16
```

Paste the three outputs, and tell me the RAM figure Task Manager shows for node while it runs. Also compare long-context speed: `bench ... --prompt-tokens 1000 --ctx 1100` (needs a few minutes).

Per-kernel numbers for your CPU (shows which formats are slow on your machine): `node tools/kbench.mjs`.

## Performance diagnosis on the target laptop (decode got slower after the kernel changes)

After the K-quant/attention work, Qwen3-4B Q4_K_M prefill improved (5.96 -> 8.23 tok/s) but decode went 5.77 -> 4.4-4.8 tok/s, while `tools/kbench.mjs` shows fast kernels (Q4_K 9.4 Gw/s/thread). Three runs tell where the time goes (PowerShell):

```
node tools/kbench.mjs
```

(compares the two integer-dot forms of the K-quant decode kernels: `extmul` is the default, `wide` is the older form; which is faster depends on the CPU)

```
$env:ZEROLLM_PROFILE=1
node bin/zerollm.js bench models/Qwen3-4B-Q4_K_M.gguf --threads 6 --batch 1 --prompt-tokens 16 --n 24
```

(prints, for worker 0, ms per token spent in matmul kernels / activation quantization / attention / barrier waits / other)

```
$env:ZEROLLM_KDOT="wide"
node bin/zerollm.js bench models/Qwen3-4B-Q4_K_M.gguf --threads 6 --batch 1 --prompt-tokens 16 --n 24
Remove-Item Env:ZEROLLM_KDOT
Remove-Item Env:ZEROLLM_PROFILE
```

Run the plain bench twice if the numbers look odd: laptop power state, background apps and thermal limits move decode speed by 10-20% between runs.

## A/B benchmarks that resolve small differences (`tools/abbench.mjs`)

The laptop's decode speed varies +-15% from run to run, so compare settings by alternating them over several rounds and looking at medians and spreads:

```
node tools/abbench.mjs models/Qwen3-4B-Q4_K_M.gguf --vary spin=0,20000 --rounds 5 --pause 15
```

(plugged in, Windows power mode "Best performance", nothing else running; `--vary` takes any `bench` flag, e.g. `threads=4,6,8` or `kernels=js,wasm`).
