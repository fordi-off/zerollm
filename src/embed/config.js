// Hyperparameters of the supported embedding encoders. Today: "nomic-bert" (nomic-embed-text v1 / v1.5).
//
// nomic-bert (as in llama.cpp's build_bert): token embedding + token-type-0 embedding -> LayerNorm; then per layer (POST-norm):
//   qkv = Wqkv x (fused rows [q | k | v], no biases) -> RoPE (split-half / "neox", base 1000) on q and k -> bidirectional attention
//   -> Wo -> + x -> LayerNorm(attn_output_norm) -> SwiGLU FFN down(silu(gate x) * up x) -> + residual -> LayerNorm(layer_output_norm)
// and the sentence vector is the MEAN over all tokens ([CLS] and [SEP] included), as llama.cpp's mean pooling.
import { GGML_TYPE } from '../gguf/types.js';
import { SUPPORTED_TENSOR_TYPES } from '../model/config.js';

export const SUPPORTED_EMBED_ARCHS = ['nomic-bert'];
export class UnsupportedEmbedModelError extends Error {}

/** @param {import('../gguf/reader.js').GGUFFile} g */
export function embedConfig(g) {
  const arch = g.get('general.architecture');
  if (!SUPPORTED_EMBED_ARCHS.includes(arch)) throw new UnsupportedEmbedModelError(`architecture "${arch}" is not a supported embedding model (supported: ${SUPPORTED_EMBED_ARCHS.join(', ')})`);
  const need = (k) => { const v = g.get(`${arch}.${k}`); if (v === undefined) throw new Error(`GGUF is missing metadata key ${arch}.${k}`); return v; };
  const nEmbd = need('embedding_length'); const nHead = need('attention.head_count');
  if (nEmbd % nHead) throw new Error('embedding_length is not divisible by head_count');
  const pooling = g.get(`${arch}.pooling_type`, 1); // 1 = mean (0 none, 2 cls)
  if (pooling !== 1 && pooling !== 2) throw new UnsupportedEmbedModelError(`pooling type ${pooling} is not supported (mean and CLS are)`);
  for (const t of g.tensors) if (!SUPPORTED_TENSOR_TYPES.has(t.type)) throw new UnsupportedEmbedModelError(`tensor ${t.name} has an unsupported type ${t.type}`);
  if (g.tensor('token_embd.weight') === undefined) throw new Error('GGUF has no token_embd.weight');
  return {
    arch, nCtxTrain: need('context_length'), nEmbd, nLayer: need('block_count'), nFF: need('feed_forward_length'), nHead, headDim: nEmbd / nHead,
    eps: g.get(`${arch}.attention.layer_norm_epsilon`, 1e-12), ropeBase: g.get(`${arch}.rope.freq_base`, 1000), pooling,
    vocabSize: g.tensor('token_embd.weight').dims[1], hasTokenTypes: g.tensor('token_types.weight') !== undefined,
  };
}

/** Is this GGUF an embedding model this module can run? (cheap: header only) */
export function isEmbedModel(g) { try { embedConfig(g); return g.get('tokenizer.ggml.model') === 'bert'; } catch { return false; } }
void GGML_TYPE;
