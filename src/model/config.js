// Architecture detection + hyperparameters from GGUF metadata. Nothing here is hard-coded per model size.

import { GGML_TYPE, typeName } from '../gguf/types.js';

export const SUPPORTED_ARCHS = ['llama', 'qwen2', 'qwen3'];

/** Tensor types the engine can compute with (reference + SIMD kernels). */
export const SUPPORTED_TENSOR_TYPES = new Set([
  GGML_TYPE.F32, GGML_TYPE.F16, GGML_TYPE.BF16, GGML_TYPE.Q4_0, GGML_TYPE.Q5_0, GGML_TYPE.Q8_0,
  GGML_TYPE.Q4_K, GGML_TYPE.Q5_K, GGML_TYPE.Q6_K,
]);

/**
 * Throw a clear error if the model uses a quantization format we cannot compute.
 * @param {import('../gguf/reader.js').GGUFFile} g
 */
export function assertSupportedTensorTypes(g) {
  const bad = new Map();
  for (const t of g.tensors) if (!SUPPORTED_TENSOR_TYPES.has(t.type)) bad.set(typeName(t.type), (bad.get(typeName(t.type)) ?? 0) + 1);
  if (bad.size) {
    const list = [...bad].map(([n, c]) => `${n} (${c} tensors)`).join(', ');
    const ok = [...SUPPORTED_TENSOR_TYPES].map(typeName).join(', ');
    throw new Error(`this model uses quantization formats the engine cannot run yet: ${list}. Supported: ${ok}. Try another file of the same model (e.g. Q4_K_M, Q5_K_M, Q6_K, Q8_0 or Q4_0).`);
  }
}

/**
 * @typedef {Object} ModelConfig
 * @property {string} arch
 * @property {number} nCtxTrain
 * @property {number} nEmbd
 * @property {number} nLayer
 * @property {number} nFF
 * @property {number} nHead
 * @property {number} nHeadKv
 * @property {number} headDim         explicit key_length when present (Qwen3: 128, NOT nEmbd/nHead)
 * @property {number} nRot            rope dimension count (<= headDim)
 * @property {number} rmsEps
 * @property {number} ropeBase
 * @property {'norm'|'neox'} ropeType 'norm' = adjacent pairs (llama), 'neox' = split-half (qwen2/qwen3)
 * @property {number} vocabSize
 * @property {boolean} qkNorm         Qwen3 per-head q/k RMSNorm
 * @property {boolean} qkvBias        Qwen2 attention biases
 * @property {boolean} tiedEmbeddings output.weight absent -> reuse token_embd.weight
 * @property {boolean} hasRopeFreqs   rope_freqs.weight tensor present (Llama 3.1)
 */

export class UnsupportedArchError extends Error {}

/**
 * @param {import('../gguf/reader.js').GGUFFile} g
 * @returns {ModelConfig}
 */
export function modelConfig(g) {
  const arch = g.get('general.architecture');
  if (typeof arch !== 'string') throw new Error('GGUF has no general.architecture');
  if (!SUPPORTED_ARCHS.includes(arch)) {
    throw new UnsupportedArchError(`architecture "${arch}" is not supported (supported: ${SUPPORTED_ARCHS.join(', ')})`);
  }
  const need = (k) => {
    const v = g.get(`${arch}.${k}`);
    if (v === undefined) throw new Error(`GGUF is missing metadata key ${arch}.${k}`);
    return v;
  };
  const nEmbd = need('embedding_length');
  const nHead = need('attention.head_count');
  const nHeadKv = g.get(`${arch}.attention.head_count_kv`, nHead);
  const headDim = g.get(`${arch}.attention.key_length`, nEmbd / nHead);
  if (!Number.isInteger(headDim)) throw new Error(`cannot determine integer head_dim (embd ${nEmbd} / heads ${nHead})`);
  const embd = g.tensor('token_embd.weight');
  const tokens = g.get('tokenizer.ggml.tokens');
  const vocabSize = g.get(`${arch}.vocab_size`) ?? (Array.isArray(tokens) ? tokens.length : embd ? embd.dims[1] : undefined);
  if (vocabSize === undefined) throw new Error('cannot determine vocab size');
  const nLayer = need('block_count');
  return {
    arch,
    nCtxTrain: need('context_length'),
    nEmbd,
    nLayer,
    nFF: need('feed_forward_length'),
    nHead,
    nHeadKv,
    headDim,
    nRot: g.get(`${arch}.rope.dimension_count`, headDim),
    rmsEps: g.get(`${arch}.attention.layer_norm_rms_epsilon`, 1e-5),
    ropeBase: g.get(`${arch}.rope.freq_base`, 10000),
    ropeType: arch === 'llama' ? 'norm' : 'neox',
    vocabSize,
    qkNorm: g.tensor('blk.0.attn_q_norm.weight') !== undefined,
    qkvBias: g.tensor('blk.0.attn_q.bias') !== undefined,
    tiedEmbeddings: g.tensor('output.weight') === undefined,
    hasRopeFreqs: g.tensor('rope_freqs.weight') !== undefined,
  };
}

/** Bytes of KV cache per token at f32 / f16. @param {ModelConfig} c */
export function kvBytesPerToken(c, bytesPerValue = 4) {
  return 2 * c.nLayer * c.nHeadKv * c.headDim * bytesPerValue;
}

/**
 * Rough RAM estimate for running a model.
 * @param {import('../gguf/reader.js').GGUFFile} g
 * @param {ModelConfig|null} cfg
 * @param {{ctx?: number, kvBytes?: number}} [opts]
 */
export function estimateRam(g, cfg, opts = {}) {
  const weights = g.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);
  const ctx = opts.ctx ?? Math.min(2048, cfg?.nCtxTrain ?? 2048);
  const kv = cfg ? kvBytesPerToken(cfg, opts.kvBytes ?? 4) * ctx : 0;
  const scratch = 100 * 1024 * 1024;
  const runtime = 400 * 1024 * 1024; // node + v8 + workers
  return { weights, kv, scratch, runtime, ctx, total: weights + kv + scratch + runtime };
}
