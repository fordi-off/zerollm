// Models offered for download. Only architectures zerollm runs (llama, qwen2, qwen3); everything is a plain
// Q4_K_M/Q8_0 GGUF. `sizeBytes` is approximate (the real size comes from the server and is verified after the download).
// `ramGB` is the rough RAM need at a 8192 context with an f16 KV cache (the model list shows the exact estimate once the file is there).
const HF = 'https://huggingface.co';

export const CATALOG = [
  {
    id: 'nomic-embed', kind: 'embed', filename: 'nomic-embed-text-v1.5.Q4_K_M.gguf', displayName: 'nomic-embed-text v1.5 (søkemodell)',
    sizeBytes: 84_000_000, ramGB: 0.4, languages: 'en',
    url: `${HF}/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf`,
  },
  {
    id: 'nb-llama-3.1-8b', filename: 'nb-llama-3.1-8b-instruct-q4_k_m.gguf', displayName: 'NB Llama 3.1 8B (norsk)',
    sizeBytes: 4_920_000_000, ramGB: 6.5, recommended: true, languages: 'no',
    url: `${HF}/NbAiLab/nb-llama-3.1-8B-Instruct-Q4_K_M-GGUF/resolve/main/nb-llama-3.1-8b-instruct-q4_k_m.gguf`,
  },
  {
    id: 'qwen3-8b', filename: 'Qwen_Qwen3-8B-Q4_K_M.gguf', displayName: 'Qwen3 8B',
    sizeBytes: 5_027_783_040, ramGB: 6.7, languages: 'multi',
    url: `${HF}/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf`,
  },
  {
    id: 'qwen3-4b', filename: 'Qwen_Qwen3-4B-Q4_K_M.gguf', displayName: 'Qwen3 4B',
    sizeBytes: 2_500_000_000, ramGB: 4, languages: 'multi',
    url: `${HF}/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf`,
  },
  {
    id: 'llama-3.2-3b', filename: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', displayName: 'Llama 3.2 3B',
    sizeBytes: 2_020_000_000, ramGB: 3.5, languages: 'multi',
    url: `${HF}/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf`,
  },
  {
    id: 'qwen3-1.7b', filename: 'Qwen_Qwen3-1.7B-Q4_K_M.gguf', displayName: 'Qwen3 1.7B',
    sizeBytes: 1_280_000_000, ramGB: 2.3, languages: 'multi',
    url: `${HF}/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf`,
  },
  {
    id: 'llama-3.2-1b', filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', displayName: 'Llama 3.2 1B',
    sizeBytes: 808_000_000, ramGB: 1.6, languages: 'multi',
    url: `${HF}/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf`,
  },
];
