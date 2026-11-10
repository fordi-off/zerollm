// Architecture detection + hyperparameters from GGUF metadata. Nothing here is hard-coded per model size.

import { GGML_TYPE, typeName } from '../gguf/types.js';

/** Architectures that run on every path (reference, worker pool, WASM SIMD kernels). qwen35 is the hybrid Gated DeltaNet + gated attention family. */
export const SUPPORTED_ARCHS = ['llama', 'qwen2', 'qwen3', 'qwen35'];

/** How many recurrent-state checkpoints a hybrid model keeps for truncate() (see state-checkpoints.js). */
export const RECURRENT_CHECKPOINTS = 6;

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
 * @property {number} nAttnLayers     layers that own a KV cache (= nLayer except for hybrid models)
 * @property {boolean} hybrid         mixes full-attention layers with recurrent (Gated DeltaNet) layers (qwen35)
 * @property {boolean[]} layerIsRecurrent  per layer: true = recurrent layer (no KV cache, has conv + delta-rule state)
 * @property {boolean} attnOutputGate attn_q projects 2*headDim per head: [query | output gate] (qwen35)
 * @property {number[]|null} ropeSections M-RoPE sections from metadata (informational: text-only input reduces to plain NEOX rotary)
 * @property {SsmConfig|null} ssm     recurrent-layer hyperparameters (hybrid models)
 *
 * @typedef {Object} SsmConfig  Gated DeltaNet ("linear attention") layer shape
 * @property {number} convKernel   depthwise causal conv taps (4)
 * @property {number} headKDim     key/query head size = ssm.state_size (128)
 * @property {number} headVDim     value head size = inner_size / time_step_rank (128)
 * @property {number} nKHeads      key/query heads = ssm.group_count (16)
 * @property {number} nVHeads      value heads = ssm.time_step_rank (32); each uses key head (h % nKHeads) (llama.cpp's tiled layout)
 * @property {number} innerSize    nVHeads * headVDim (4096)
 * @property {number} convDim      conv channels = 2 * nKHeads * headKDim + innerSize (8192)
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
  // qwen35: block_count includes any trailing MTP (multi-token-prediction) blocks; the main stack is the first
  // block_count - nextn_predict_layers layers (llama.cpp: n_layer = n_layer_all - n_layer_nextn). The MTP blocks are not run.
  const nLayer = need('block_count') - (arch === 'qwen35' ? g.get(`${arch}.nextn_predict_layers`, 0) : 0);
  let hybrid = null;
  if (arch === 'qwen35') hybrid = qwen35Layout(g, arch, nLayer, headDim, g.get(`${arch}.attention.value_length`, headDim));
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
    qkNorm: hybrid ? true : g.tensor('blk.0.attn_q_norm.weight') !== undefined,
    qkvBias: g.tensor('blk.0.attn_q.bias') !== undefined,
    tiedEmbeddings: g.tensor('output.weight') === undefined,
    hasRopeFreqs: g.tensor('rope_freqs.weight') !== undefined,
    nAttnLayers: hybrid ? hybrid.layerIsRecurrent.filter((r) => !r).length : nLayer,
    hybrid: hybrid !== null,
    layerIsRecurrent: hybrid ? hybrid.layerIsRecurrent : Array.from({ length: nLayer }, () => false),
    attnOutputGate: hybrid !== null,
    ropeSections: hybrid ? hybrid.ropeSections : null,
    ssm: hybrid ? hybrid.ssm : null,
  };
}

/**
 * qwen35 layer layout + Gated DeltaNet shape from metadata (llama.cpp: llama_model_qwen35::load_arch_hparams).
 * A layer is recurrent unless (layer + 1) % full_attention_interval == 0, or the explicit
 * `<arch>.attention.recurrent_layers` boolean array says otherwise.
 */
function qwen35Layout(g, arch, nLayer, headDim, valueLen) {
  const need = (k) => {
    const v = g.get(`${arch}.${k}`);
    if (v === undefined) throw new Error(`GGUF is missing metadata key ${arch}.${k}`);
    return v;
  };
  if (valueLen !== headDim) throw new Error(`${arch}: attention.value_length ${valueLen} != key_length ${headDim} is not supported`);
  const explicit = g.get(`${arch}.attention.recurrent_layers`);
  const interval = g.get(`${arch}.full_attention_interval`, 4);
  const layerIsRecurrent = Array.isArray(explicit) && explicit.length >= nLayer
    ? explicit.slice(0, nLayer).map(Boolean)
    : Array.from({ length: nLayer }, (_, i) => (i + 1) % interval !== 0);
  const convKernel = need('ssm.conv_kernel');
  const headKDim = need('ssm.state_size');
  const nKHeads = need('ssm.group_count');
  const nVHeads = need('ssm.time_step_rank');
  const innerSize = need('ssm.inner_size');
  if (innerSize % nVHeads !== 0) throw new Error(`${arch}: ssm.inner_size ${innerSize} is not a multiple of ssm.time_step_rank ${nVHeads}`);
  const headVDim = innerSize / nVHeads;
  if (headVDim !== headKDim) throw new Error(`${arch}: value head size ${headVDim} != key head size ${headKDim} is not supported`);
  if (nVHeads % nKHeads !== 0) throw new Error(`${arch}: ssm.time_step_rank ${nVHeads} is not a multiple of ssm.group_count ${nKHeads}`);
  const sections = g.get(`${arch}.rope.dimension_sections`);
  return {
    layerIsRecurrent,
    ropeSections: Array.isArray(sections) ? Array.from(sections) : null,
    ssm: { convKernel, headKDim, headVDim, nKHeads, nVHeads, innerSize, convDim: 2 * nKHeads * headKDim + innerSize },
  };
}

/**
 * Bytes of recurrent state of ONE sequence (always f32): per recurrent layer the conv window
 * ((convKernel - 1) x convDim) and the delta-rule matrices (nVHeads x headVDim x headKDim). 0 for non-hybrid models.
 * @param {ModelConfig} c
 */
export function recurrentStateBytes(c) {
  if (!c?.ssm) return 0;
  const n = c.layerIsRecurrent.filter(Boolean).length;
  const { convKernel, convDim, nVHeads, headVDim, headKDim } = c.ssm;
  return n * 4 * ((convKernel - 1) * convDim + nVHeads * headVDim * headKDim);
}

/** Bytes of KV cache per token at f32 / f16 (only attention layers have one). @param {ModelConfig} c */
export function kvBytesPerToken(c, bytesPerValue = 4) {
  return 2 * (c.nAttnLayers ?? c.nLayer) * c.nHeadKv * c.headDim * bytesPerValue;
}

/**
 * Rough RAM estimate for running a model.
 *  weights  matrix bytes (every worker holds a row slice, so the total is the file's tensor data)
 *  kv       KV cache for `ctx` tokens at `kvBytes` per value (4 = f32, 2 = f16)
 *  scratch  per-worker activation / kernel buffers
 *  runtime  node + V8 + one isolate per worker
 *  recurrent  hybrid models: the live recurrent state (f32) of the one sequence (shared memory on the pool)
 *  checkpoints  hybrid models: room for RECURRENT_CHECKPOINTS state snapshots (truncate() support, see state-checkpoints.js)
 *  perWorker  the busiest worker's WebAssembly.Memory (must stay under the 4 GB wasm32 limit)
 * @param {import('../gguf/reader.js').GGUFFile} g
 * @param {ModelConfig|null} cfg
 * @param {{ctx?: number, kvBytes?: number, threads?: number}} [opts]
 */
export function estimateRam(g, cfg, opts = {}) {
  const threads = Math.max(1, opts.threads ?? 1);
  const ctx = opts.ctx ?? Math.min(2048, cfg?.nCtxTrain ?? 2048);
  const weights = g.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);
  const kvPerToken = cfg ? kvBytesPerToken(cfg, opts.kvBytes ?? 4) : 0;
  const kv = kvPerToken * ctx;
  const recurrent = recurrentStateBytes(cfg);
  const checkpoints = cfg?.hybrid ? recurrent * RECURRENT_CHECKPOINTS : 0;
  const scratch = threads * 8 * 1024 * 1024;
  const runtime = 250 * 1024 * 1024 + threads * 40 * 1024 * 1024;
  // busiest worker: an even share of the weights plus the KV heads it owns (ceil(nHeadKv / threads) of nHeadKv)
  const heads = cfg ? Math.ceil(cfg.nHeadKv / Math.min(threads, cfg.nHeadKv)) / cfg.nHeadKv : 1;
  const perWorker = weights / threads + kv * heads + 8 * 1024 * 1024;
  return { weights, kv, recurrent, checkpoints, scratch, runtime, ctx, perWorker, total: weights + kv + recurrent + checkpoints + scratch + runtime };
}
