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
