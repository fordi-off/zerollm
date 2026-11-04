# Checking a real model (needs a machine with internet; the dev container could not reach Hugging Face)

Everything so far is validated against synthetic tiny models and an independent naive oracle. The following has **not** been run against a real GGUF yet. Please run it on the laptop and paste the output back.

1. Download `Qwen3-0.6B-Q8_0.gguf` (Qwen/Qwen3-0.6B-GGUF on Hugging Face) into `models/`.
2. `node bin/zerollm.js doctor` — paste all of it.
3. `node bin/zerollm.js info models/Qwen3-0.6B-Q8_0.gguf` — expect: arch qwen3, 28 layers, hidden 1024, 16 heads / 8 KV, head dim 128 (!= 64), vocab 151936, tied embeddings, QK-norm yes, Q8_0 + F32 tensors.
4. `node bin/zerollm.js tokenize models/Qwen3-0.6B-Q8_0.gguf "Hello, world! 123"` — expect pieces like `Hello`, `,`, `Ġworld`, `!`, `Ġ`, `1`, `2`, `3`.
5. `node bin/zerollm.js run models/Qwen3-0.6B-Q8_0.gguf -p "What is the capital of France? Answer in one sentence." --temp 0 --n 60` — expect a coherent answer mentioning Paris, roughly 1-2 tok/s single-threaded. Run it twice: greedy output must be identical.
6. If the answer is garbled, the usual suspects (brief section 21) are RoPE variant, QK-norm, head_dim, tokenizer pre-regex, chat template. Send the exact output of steps 3-5 and I will bisect with `tokenize` and per-layer checks.

If you have another machine with HF transformers / llama.cpp: compare the top-5 next-token logits for a fixed prompt (brief section 18, item 7).
