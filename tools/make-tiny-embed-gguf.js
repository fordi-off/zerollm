// A tiny random "nomic-bert" GGUF (WordPiece vocabulary included) for tests: node tools/make-tiny-embed-gguf.js out.gguf [--quant f32|q8_0|q4_k_m]
import fs from 'node:fs';
import { buildGGUF } from './gguf-writer.js';
import { GGUF_TYPE, GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { quantizeQ8_0 } from '../src/compute/quant.js';
import { randomQuantBlocks } from './random-blocks.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs } from '../src/util/args.js';

const SOW = '▁';
/** Vocabulary: specials, every ascii letter/digit in word-initial and continuation form, punctuation, a few whole words and suffixes. */
export function tinyWordPieceVocab() {
  const t = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'];
  for (const c of 'abcdefghijklmnopqrstuvwxyz0123456789') { t.push(SOW + c); t.push(c); }
  for (const c of '.,!?;:-\'"()/&%$#@') t.push(SOW + c);
  for (const w of ['hello', 'world', 'kommune', 'barnehage', 'the', 'og', 'er', 'ansvar', 'for', 'skole']) t.push(SOW + w);
  for (const w of ['ing', 'er', 'ene', 'en', 'hage', 'ne', 'ls', 'ed']) t.push(w);
  t.push(SOW + 'aø'); // a word containing a non-ascii letter
  t.push('ø'); t.push(SOW + 'ø'); t.push(SOW + 'æ'); t.push('æ');
  return t;
}

/** @param {{quant?: 'f32'|'q8_0'|'q4_k_m', seed?: number, nEmbd?: number, nLayer?: number, nHead?: number, nFF?: number, ctx?: number, pooling?: number}} [o] */
export function makeTinyEmbedModel(o = {}) {
  const quant = o.quant ?? 'f32';
  const rand = mulberry32(o.seed ?? 1);
  const nEmbd = o.nEmbd ?? 256; const nLayer = o.nLayer ?? 2; const nHead = o.nHead ?? 4; const nFF = o.nFF ?? 512; const ctx = o.ctx ?? 128;
  const tokens = tinyWordPieceVocab(); const vocab = tokens.length;
  const tensors = [];
  const mat = (name, ne0, ne1, std = 0.04) => {
    const n = ne0 * ne1;
    if (quant === 'q4_k_m' && ne0 % 256 === 0) {
      const rawType = /ffn_down|attn_output/.test(name) ? GGML_TYPE.Q6_K : GGML_TYPE.Q4_K;
      tensors.push({ name, dims: [ne0, ne1], type: rawType, data: randomQuantBlocks(rawType, n / GGML_TYPE_INFO[rawType].blockSize, rand) });
      return;
    }
    const f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = gaussian(rand) * std;
    if (quant === 'q8_0' && ne0 % 32 === 0) tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.Q8_0, data: quantizeQ8_0(f) });
    else tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) });
  };
  const vec = (name, n, base, spread) => { const f = new Float32Array(n); for (let i = 0; i < n; i++) f[i] = base + gaussian(rand) * spread; tensors.push({ name, dims: [n], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) }); };
  mat('token_embd.weight', nEmbd, vocab, 0.3);
  mat('token_types.weight', nEmbd, 2, 0.1);
  vec('token_embd_norm.weight', nEmbd, 1, 0.05); vec('token_embd_norm.bias', nEmbd, 0, 0.02);
  for (let l = 0; l < nLayer; l++) {
    const p = `blk.${l}.`;
    mat(`${p}attn_qkv.weight`, nEmbd, 3 * nEmbd);
    mat(`${p}attn_output.weight`, nEmbd, nEmbd);
    vec(`${p}attn_output_norm.weight`, nEmbd, 1, 0.05); vec(`${p}attn_output_norm.bias`, nEmbd, 0, 0.02);
    mat(`${p}ffn_up.weight`, nEmbd, nFF); mat(`${p}ffn_gate.weight`, nEmbd, nFF); mat(`${p}ffn_down.weight`, nFF, nEmbd);
    vec(`${p}layer_output_norm.weight`, nEmbd, 1, 0.05); vec(`${p}layer_output_norm.bias`, nEmbd, 0, 0.02);
  }
  const U32 = GGUF_TYPE.UINT32; const kv = (key, type, value, elemType) => ({ key, type, value, elemType });
  const metadata = [
    kv('general.architecture', GGUF_TYPE.STRING, 'nomic-bert'), kv('general.name', GGUF_TYPE.STRING, `tiny-nomic-bert-${quant}`),
    kv('nomic-bert.context_length', U32, ctx), kv('nomic-bert.embedding_length', U32, nEmbd), kv('nomic-bert.block_count', U32, nLayer),
    kv('nomic-bert.feed_forward_length', U32, nFF), kv('nomic-bert.attention.head_count', U32, nHead),
    kv('nomic-bert.attention.layer_norm_epsilon', GGUF_TYPE.FLOAT32, 1e-12), kv('nomic-bert.attention.causal', GGUF_TYPE.BOOL, false),
    kv('nomic-bert.pooling_type', U32, o.pooling ?? 1), kv('nomic-bert.rope.freq_base', GGUF_TYPE.FLOAT32, 1000),
    kv('tokenizer.ggml.model', GGUF_TYPE.STRING, 'bert'), kv('tokenizer.ggml.tokens', GGUF_TYPE.ARRAY, tokens, GGUF_TYPE.STRING),
    kv('tokenizer.ggml.token_type', GGUF_TYPE.ARRAY, tokens.map((_, i) => (i < 5 ? 3 : 1)), GGUF_TYPE.INT32),
    kv('tokenizer.ggml.unknown_token_id', U32, 1), kv('tokenizer.ggml.cls_token_id', U32, 2), kv('tokenizer.ggml.seperator_token_id', U32, 3), kv('tokenizer.ggml.padding_token_id', U32, 0),
  ];
  return { buffer: buildGGUF({ metadata, tensors, alignment: 32 }), tokens, cfg: { nEmbd, nLayer, nHead, nFF, vocab } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { _, flags } = parseArgs(process.argv.slice(2));
  if (!_[0]) { console.error('usage: make-tiny-embed-gguf.js out.gguf [--quant f32|q8_0|q4_k_m]'); process.exit(1); }
  const { buffer } = makeTinyEmbedModel({ quant: typeof flags.quant === 'string' ? flags.quant : undefined });
  fs.writeFileSync(_[0], buffer); console.log(`wrote ${_[0]} (${buffer.length} bytes)`);
}
