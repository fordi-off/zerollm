#!/usr/bin/env node
// Generate a tiny random GGUF model (with a small byte-level BPE tokenizer) for tests.
//   node tools/make-tiny-gguf.js out.gguf [--arch qwen3|llama|qwen2] [--quant f32|q8_0|q4_0] [--seed N] [--tied]
import fs from 'node:fs';
import { buildGGUF, writeGGUF } from './gguf-writer.js';
import { GGUF_TYPE, GGML_TYPE } from '../src/gguf/types.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { randomQuantBlocks } from './random-blocks.js';
import { GGML_TYPE_INFO } from '../src/gguf/types.js';
import { BYTE_TO_CHAR } from '../src/tokenizer/bytes.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs, num } from '../src/util/args.js';

const SP = BYTE_TO_CHAR[0x20]; // 'Ġ'
export const TINY_MERGES = [
  [SP, 't'], ['h', 'e'], [`${SP}t`, 'he'], ['i', 'n'], [SP, 'a'], ['e', 'r'], ['l', 'l'], ['he', 'll'], ['hell', 'o'], [SP, 'w'], [`${SP}w`, 'o'],
];

/**
 * @param {{arch?: string, quant?: 'f32'|'q8_0'|'q4_0'|'q5_0'|'q4_k'|'q5_k'|'q6_k'|'q4_k_m', seed?: number, tied?: boolean, nEmbd?: number, nLayer?: number,
 *          nHead?: number, nHeadKv?: number, headDim?: number, nFF?: number, ctx?: number, alignment?: number, ropeFreqs?: boolean, padVocabTo?: number, outFile?: string,
 *          std?: number, embdStd?: number, perturb?: number, perturbSeed?: number, fadeAfter?: number, fade?: number}} [o]  fadeAfter/fade: multiply the output projections of layers >= fadeAfter by `fade` (they
 *          only nudge the residual stream), so a model with `nLayer: fadeAfter` and the same seed is a cheaper draft that mostly agrees with this one.  perturb: add gaussian noise of this relative size (x std) to every float weight
 *          (separate random stream `perturbSeed`; the model keeps its shape): a related-but-different model, e.g. a draft with partial acceptance.  std: stddev of the layer weights (default 0.05; larger = more context-dependent, peakier logits),
 *          embdStd: stddev of token_embd / output (default 0.05). Two models with the same arch share the vocabulary whatever their size/seed.
 * @returns {{ buffer: Buffer, tokens: string[], cfg: object }}
 */
export function makeTinyModel(o = {}) {
  const arch = o.arch ?? 'qwen3';
  const quant = o.quant ?? 'f32';
  const rand = mulberry32(o.seed ?? 1);
  const prand = mulberry32(o.perturbSeed ?? 12345);
  const nEmbd = o.nEmbd ?? 64;
  const nLayer = o.nLayer ?? 2;
  const nHead = o.nHead ?? 4;
  const nHeadKv = o.nHeadKv ?? 2;
  // qwen3: head_dim deliberately != nEmbd/nHead to exercise the explicit key_length path
  const headDim = o.headDim ?? (arch === 'qwen3' ? 32 : nEmbd / nHead);
  const nFF = o.nFF ?? 128;
  const ctx = o.ctx ?? 256;
  const tied = o.tied ?? arch !== 'llama';

  // --- tokenizer: 256 byte tokens, merged tokens, specials
  const tokens = BYTE_TO_CHAR.slice();
  for (const [a, b] of TINY_MERGES) tokens.push(a + b);
  const types = tokens.map(() => 1);
  const addSpecial = (s) => { tokens.push(s); types.push(3); return tokens.length - 1; };
  let bos; let eos;
  const chatTemplate = arch === 'llama'
    ? "{% for m in messages %}<|start_header_id|>{{m.role}}<|end_header_id|>\n\n{{m.content}}<|eot_id|>{% endfor %}"
    : "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}";
  if (arch === 'llama') {
    bos = addSpecial('<|begin_of_text|>'); addSpecial('<|end_of_text|>');
    addSpecial('<|start_header_id|>'); addSpecial('<|end_header_id|>'); eos = addSpecial('<|eot_id|>');
  } else {
    bos = addSpecial('<|endoftext|>'); addSpecial('<|im_start|>'); eos = addSpecial('<|im_end|>');
  }
  // optional padding with unique dummy tokens (to benchmark real-model shapes)
  while (tokens.length < (o.padVocabTo ?? 0)) { types.push(1); tokens.push(`\u0100pad${tokens.length}`); }
  const vocab = tokens.length;

  // --- tensors
  const tensors = [];
  const RAW_TYPES = { q5_0: GGML_TYPE.Q5_0, q4_k: GGML_TYPE.Q4_K, q5_k: GGML_TYPE.Q5_K, q6_k: GGML_TYPE.Q6_K };
  const mat = (name, ne0, ne1, std0) => {
    const std = std0 ?? ((name === 'token_embd.weight' || name === 'output.weight' ? o.embdStd : o.std) ?? 0.05);
    const n = ne0 * ne1;
    // K-quant style per-tensor type choice. 'q4_k_m' mimics llama.cpp's Q4_K_M mix: Q6_K for attn_v / ffn_down / output,
    // Q4_K for the rest. Rows whose length is not a whole number of blocks fall back to F32 (ggml falls back too).
    let rawType = RAW_TYPES[quant];
    if (quant === 'q4_k_m') rawType = /attn_v|ffn_down|output\.weight/.test(name) ? GGML_TYPE.Q6_K : GGML_TYPE.Q4_K;
    if (rawType !== undefined && ne0 % GGML_TYPE_INFO[rawType].blockSize === 0) {
      // random raw blocks with sane scales; (the draw order differs from the float path, models are not comparable across quants)
      tensors.push({ name, dims: [ne0, ne1], type: rawType, data: randomQuantBlocks(rawType, n / GGML_TYPE_INFO[rawType].blockSize, rand) });
      return;
    }
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = gaussian(rand) * std;
    if (o.perturb) for (let i = 0; i < n; i++) f[i] += gaussian(prand) * std * o.perturb;
    const fadeLayer = /^blk\.(\d+)\.(attn_output|ffn_down)\.weight$/.exec(name);
    if (o.fadeAfter !== undefined && fadeLayer && Number(fadeLayer[1]) >= o.fadeAfter) for (let i = 0; i < n; i++) f[i] *= o.fade ?? 0.1;
    if (quant === 'q8_0' && ne0 % 32 === 0) tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.Q8_0, data: quantizeQ8_0(f) });
    else if (quant === 'q4_0' && ne0 % 32 === 0) tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.Q4_0, data: quantizeQ4_0(f) });
    else tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) });
  };
  const vec = (name, n) => {
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = 1 + gaussian(rand) * 0.05;
    tensors.push({ name, dims: [n], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) });
  };
  const bias = (name, n) => {
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = gaussian(rand) * 0.02;
    tensors.push({ name, dims: [n], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) });
  };
  const qDim = nHead * headDim;
  const kvDim = nHeadKv * headDim;
  mat('token_embd.weight', nEmbd, vocab);
  if (o.ropeFreqs) { // Llama 3.1 style per-dimension frequency factors
    const f = new Float32Array(headDim / 2).map((_, i) => 1 + 3 * (i / (headDim / 2)));
    tensors.push({ name: 'rope_freqs.weight', dims: [headDim / 2], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer) });
  }
  vec('output_norm.weight', nEmbd);
  if (!tied) mat('output.weight', nEmbd, vocab);
  for (let l = 0; l < nLayer; l++) {
    const p = `blk.${l}.`;
    vec(`${p}attn_norm.weight`, nEmbd);
    mat(`${p}attn_q.weight`, nEmbd, qDim);
    mat(`${p}attn_k.weight`, nEmbd, kvDim);
    mat(`${p}attn_v.weight`, nEmbd, kvDim);
    mat(`${p}attn_output.weight`, qDim, nEmbd);
    if (arch === 'qwen2') { bias(`${p}attn_q.bias`, qDim); bias(`${p}attn_k.bias`, kvDim); bias(`${p}attn_v.bias`, kvDim); }
    if (arch === 'qwen3') { vec(`${p}attn_q_norm.weight`, headDim); vec(`${p}attn_k_norm.weight`, headDim); }
    vec(`${p}ffn_norm.weight`, nEmbd);
    mat(`${p}ffn_gate.weight`, nEmbd, nFF);
    mat(`${p}ffn_up.weight`, nEmbd, nFF);
    mat(`${p}ffn_down.weight`, nFF, nEmbd);
  }

  const U32 = GGUF_TYPE.UINT32;
  const kv = (key, type, value, elemType) => ({ key, type, value, elemType });
  const metadata = [
    kv('general.architecture', GGUF_TYPE.STRING, arch),
    kv('general.name', GGUF_TYPE.STRING, `tiny-${arch}-${quant}`),
    kv(`${arch}.context_length`, U32, ctx),
    kv(`${arch}.embedding_length`, U32, nEmbd),
    kv(`${arch}.block_count`, U32, nLayer),
    kv(`${arch}.feed_forward_length`, U32, nFF),
    kv(`${arch}.attention.head_count`, U32, nHead),
    kv(`${arch}.attention.head_count_kv`, U32, nHeadKv),
    kv(`${arch}.attention.layer_norm_rms_epsilon`, GGUF_TYPE.FLOAT32, 1e-6),
    kv(`${arch}.rope.freq_base`, GGUF_TYPE.FLOAT32, arch === 'llama' ? 500000 : 1000000),
    kv(`${arch}.rope.dimension_count`, U32, headDim),
    ...(arch === 'qwen3' ? [kv('qwen3.attention.key_length', U32, headDim), kv('qwen3.attention.value_length', U32, headDim)] : []),
    kv('tokenizer.ggml.model', GGUF_TYPE.STRING, 'gpt2'),
    kv('tokenizer.ggml.pre', GGUF_TYPE.STRING, arch === 'llama' ? 'llama-bpe' : 'qwen2'),
    kv('tokenizer.ggml.tokens', GGUF_TYPE.ARRAY, tokens, GGUF_TYPE.STRING),
    kv('tokenizer.ggml.token_type', GGUF_TYPE.ARRAY, types, GGUF_TYPE.INT32),
    kv('tokenizer.ggml.merges', GGUF_TYPE.ARRAY, TINY_MERGES.map(([a, b]) => `${a} ${b}`), GGUF_TYPE.STRING),
    kv('tokenizer.ggml.bos_token_id', U32, bos),
    kv('tokenizer.ggml.eos_token_id', U32, eos),
    kv('tokenizer.ggml.add_bos_token', GGUF_TYPE.BOOL, arch === 'llama'),
    kv('tokenizer.chat_template', GGUF_TYPE.STRING, chatTemplate),
  ];
  let buffer = null;
  if (o.outFile) writeGGUF(o.outFile, { metadata, tensors, alignment: o.alignment ?? 32 }); // streamed: no second in-memory copy
  else buffer = buildGGUF({ metadata, tensors, alignment: o.alignment ?? 32 });
  return { buffer, tokens, cfg: { arch, nEmbd, nLayer, nHead, nHeadKv, headDim, nFF, vocab, tied, bos, eos } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { _, flags } = parseArgs(process.argv.slice(2), { boolean: ['tied'] });
  if (!_[0]) { console.error('usage: make-tiny-gguf.js out.gguf [--arch qwen3|llama|qwen2] [--quant f32|q8_0|q4_0] [--seed N] [--tied]'); process.exit(1); }
  const { buffer } = makeTinyModel({
    arch: typeof flags.arch === 'string' ? flags.arch : undefined,
    quant: typeof flags.quant === 'string' ? /** @type {any} */ (flags.quant) : undefined,
    seed: num(flags.seed, 1),
    tied: flags.tied === true ? true : undefined,
  });
  fs.writeFileSync(_[0], buffer);
  console.log(`wrote ${_[0]} (${buffer.length} bytes)`);
}
