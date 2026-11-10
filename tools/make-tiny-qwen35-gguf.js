#!/usr/bin/env node
// Generate a tiny random Qwen3.5-style ("qwen35": Gated DeltaNet + full attention hybrid) GGUF for tests, with the same
// tensor names / shapes / layouts as the real Unsloth Qwen3.5-4B file, scaled down:
//   real:  32 layers, every 4th full attention, 16 heads x 256 (+gate), 4 KV heads, rotary on 64 dims, 16 key / 32 value
//          heads x 128, conv kernel 4, embedding 2560, vocab 248320, tied output, no BOS.
//   tiny:  4 layers, every 2nd full attention, 4 heads x 16, 2 KV heads, rotary on 8 dims, 2 key / 4 value heads x 16, ...
//
// The weights are first drawn in the layout of the Hugging Face checkpoint (value heads GROUPED by key head, norm weights
// zero-centred, A_log, dt_bias) and then converted exactly like llama.cpp's convert_hf_to_gguf.py (Qwen3_5TextModel):
//   - every `*norm.weight` except linear_attn.norm.weight gets +1 (Qwen3-Next RMSNorm computes x * (1 + w));
//   - A_log -> ssm_a = -exp(A_log);  conv1d [C,1,K] -> [C,K];
//   - value heads are reordered from grouped (kh * r + j) to TILED order (j * nK + kh), in the V rows of in_proj_qkv, in_proj_z,
//     in_proj_a/b, A_log, dt_bias, the V channels of conv1d, and the columns of out_proj (_LinearAttentionVReorderBase).
// The HF-layout floats are returned (`hf`) so tests can run an independent naive implementation straight from HF formulas.
//   node tools/make-tiny-qwen35-gguf.js out.gguf [--quant f32|q8_0|q4_0|q4_k_m] [--seed N]
import fs from 'node:fs';
import { buildGGUF, writeGGUF } from './gguf-writer.js';
import { GGUF_TYPE, GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { randomQuantBlocks } from './random-blocks.js';
import { TINY_MERGES } from './make-tiny-gguf.js';
import { BYTE_TO_CHAR } from '../src/tokenizer/bytes.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs, num } from '../src/util/args.js';

/** Move head blocks of `hd` rows along the row axis: HF grouped index (kh * r + j) -> tiled index (j * nK + kh). */
export function reorderRows(m, rows, cols, nK, r, hd) {
  const out = new Float32Array(m.length);
  for (let kh = 0; kh < nK; kh++) for (let j = 0; j < r; j++) out.set(m.subarray((kh * r + j) * hd * cols, ((kh * r + j) + 1) * hd * cols), (j * nK + kh) * hd * cols);
  return out;
}

/** Same permutation along the column axis (out_proj: columns are value heads). */
export function reorderCols(m, rows, cols, nK, r, hd) {
  const out = new Float32Array(m.length);
  for (let row = 0; row < rows; row++) {
    for (let kh = 0; kh < nK; kh++) for (let j = 0; j < r; j++) out.set(m.subarray(row * cols + (kh * r + j) * hd, row * cols + (kh * r + j + 1) * hd), row * cols + (j * nK + kh) * hd);
  }
  return out;
}

/**
 * @param {{quant?: 'f32'|'q8_0'|'q4_0'|'q4_k_m', seed?: number, nEmbd?: number, nLayer?: number, interval?: number, nHead?: number, nHeadKv?: number,
 *          headDim?: number, nRot?: number, nFF?: number, nK?: number, nV?: number, headSsm?: number, convKernel?: number, ctx?: number,
 *          sections?: number[], alignment?: number, outFile?: string}} [o]
 * @returns {{buffer: Buffer|null, tokens: string[], cfg: object, hf: Record<string, Float32Array>}}
 */
export function makeTinyQwen35(o = {}) {
  const quant = o.quant ?? 'f32';
  const rand = mulberry32(o.seed ?? 1);
  const nEmbd = o.nEmbd ?? 64; const nLayer = o.nLayer ?? 4; const interval = o.interval ?? 2;
  const nHead = o.nHead ?? 4; const nHeadKv = o.nHeadKv ?? 2; const hd = o.headDim ?? 16; const nRot = o.nRot ?? 8;
  const nFF = o.nFF ?? 96; const nK = o.nK ?? 2; const nV = o.nV ?? 4; const dk = o.headSsm ?? 16; const K = o.convKernel ?? 4;
  const ctx = o.ctx ?? 256;
  const sections = o.sections ?? [nRot / 2 - 2 * Math.floor(nRot / 6), Math.floor(nRot / 6), Math.floor(nRot / 6), 0];
  const r = nV / nK;
  const keyDim = nK * dk; const inner = nV * dk; const convDim = 2 * keyDim + inner;
  const qDim = nHead * hd; const kvDim = nHeadKv * hd;

  // tokenizer: 256 byte tokens, a few merges, Qwen3.5-like specials (no BOS), <think> tags as user-defined tokens
  const tokens = BYTE_TO_CHAR.slice();
  for (const [a, b] of TINY_MERGES) tokens.push(a + b);
  const types = tokens.map(() => 1);
  const add = (s, t) => { tokens.push(s); types.push(t); return tokens.length - 1; };
  add('<|endoftext|>', 3); add('<|im_start|>', 3); const eos = add('<|im_end|>', 3); add('<think>', 4); add('</think>', 4);
  const vocab = tokens.length;

  /** HF-layout floats by HF name */
  const hf = {};
  const draw = (n, std) => Float32Array.from({ length: n }, () => gaussian(rand) * std);
  const hfNorm = (name, n) => { hf[name] = draw(n, 0.05); return hf[name]; };

  const tensors = [];
  const RAW = { q4_k_m: (name) => (/ffn_down|token_embd/.test(name) ? GGML_TYPE.Q6_K : /attn_qkv|ssm_out/.test(name) ? GGML_TYPE.Q5_K : /ssm_alpha|ssm_beta/.test(name) ? GGML_TYPE.Q8_0 : GGML_TYPE.Q4_K) };
  /** emit a matrix whose floats are already in final GGUF layout (ne0 = cols, ne1 = rows) */
  const mat = (name, ne0, ne1, f) => {
    let rawType = quant === 'q4_k_m' ? RAW.q4_k_m(name) : undefined;
    if (rawType === GGML_TYPE.Q8_0) rawType = undefined; // handled below as a quantized float matrix
    if (rawType !== undefined && ne0 % GGML_TYPE_INFO[rawType].blockSize === 0) {
      tensors.push({ name, dims: [ne0, ne1], type: rawType, data: randomQuantBlocks(rawType, (ne0 * ne1) / GGML_TYPE_INFO[rawType].blockSize, rand) });
    } else if ((quant === 'q8_0' || quant === 'q4_k_m') && ne0 % 32 === 0) tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.Q8_0, data: quantizeQ8_0(f) });
    else if (quant === 'q4_0' && ne0 % 32 === 0) tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.Q4_0, data: quantizeQ4_0(f) });
    else tensors.push({ name, dims: [ne0, ne1], type: GGML_TYPE.F32, data: new Uint8Array(f.buffer.slice(f.byteOffset, f.byteOffset + f.byteLength)) });
  };
  const vec = (name, f, dims = [f.length]) => tensors.push({ name, dims, type: GGML_TYPE.F32, data: new Uint8Array(f.buffer.slice(f.byteOffset, f.byteOffset + f.byteLength)) });
  const plus1 = (f) => f.map((v) => v + 1);

  hf['model.embed_tokens'] = draw(vocab * nEmbd, 0.05);
  mat('token_embd.weight', nEmbd, vocab, hf['model.embed_tokens']);
  vec('output_norm.weight', plus1(hfNorm('model.norm', nEmbd)));
  const isRecurrent = [];
  for (let l = 0; l < nLayer; l++) {
    const rec = (l + 1) % interval !== 0;
    isRecurrent.push(rec);
    const p = `blk.${l}.`; const hp = `layers.${l}.`;
    vec(`${p}attn_norm.weight`, plus1(hfNorm(`${hp}input_layernorm`, nEmbd)));
    vec(`${p}post_attention_norm.weight`, plus1(hfNorm(`${hp}post_attention_layernorm`, nEmbd)));
    if (rec) {
      const q = `${hp}linear_attn.`;
      hf[`${q}in_proj_qkv`] = draw(convDim * nEmbd, 0.08);
      hf[`${q}in_proj_z`] = draw(inner * nEmbd, 0.08);
      hf[`${q}in_proj_b`] = draw(nV * nEmbd, 0.08);
      hf[`${q}in_proj_a`] = draw(nV * nEmbd, 0.08);
      hf[`${q}conv1d`] = draw(convDim * K, 0.4);
      hf[`${q}A_log`] = Float32Array.from({ length: nV }, () => Math.log(0.1 + 1.4 * rand()));
      hf[`${q}dt_bias`] = draw(nV, 0.5);
      hf[`${q}norm`] = Float32Array.from({ length: dk }, () => 1 + gaussian(rand) * 0.1); // RMSNormGated weight: used as is, no +1
      hf[`${q}out_proj`] = draw(nEmbd * inner, 0.08);
      // --- converter: HF -> GGUF layout
      const qkv = hf[`${q}in_proj_qkv`];
      const qk = qkv.subarray(0, 2 * keyDim * nEmbd);
      const vPart = reorderRows(qkv.subarray(2 * keyDim * nEmbd), inner, nEmbd, nK, r, dk);
      const qkvG = new Float32Array(qkv.length); qkvG.set(qk); qkvG.set(vPart, 2 * keyDim * nEmbd);
      mat(`${p}attn_qkv.weight`, nEmbd, convDim, qkvG);
      mat(`${p}attn_gate.weight`, nEmbd, inner, reorderRows(hf[`${q}in_proj_z`], inner, nEmbd, nK, r, dk));
      mat(`${p}ssm_alpha.weight`, nEmbd, nV, reorderRows(hf[`${q}in_proj_a`], nV, nEmbd, nK, r, 1));
      mat(`${p}ssm_beta.weight`, nEmbd, nV, reorderRows(hf[`${q}in_proj_b`], nV, nEmbd, nK, r, 1));
      const conv = hf[`${q}conv1d`];
      const convG = new Float32Array(conv.length); convG.set(conv.subarray(0, 2 * keyDim * K)); convG.set(reorderRows(conv.subarray(2 * keyDim * K), inner, K, nK, r, dk), 2 * keyDim * K);
      vec(`${p}ssm_conv1d.weight`, convG, [K, convDim]);
      vec(`${p}ssm_a`, reorderRows(hf[`${q}A_log`].map((v) => -Math.exp(v)), nV, 1, nK, r, 1));
      vec(`${p}ssm_dt.bias`, reorderRows(hf[`${q}dt_bias`], nV, 1, nK, r, 1));
      vec(`${p}ssm_norm.weight`, hf[`${q}norm`]);
      mat(`${p}ssm_out.weight`, inner, nEmbd, reorderCols(hf[`${q}out_proj`], nEmbd, inner, nK, r, dk));
    } else {
      const q = `${hp}self_attn.`;
      hf[`${q}q_proj`] = draw(2 * qDim * nEmbd, 0.08); // rows per head: [query hd | gate hd]
      hf[`${q}k_proj`] = draw(kvDim * nEmbd, 0.08);
      hf[`${q}v_proj`] = draw(kvDim * nEmbd, 0.08);
      hf[`${q}o_proj`] = draw(nEmbd * qDim, 0.08);
      mat(`${p}attn_q.weight`, nEmbd, 2 * qDim, hf[`${q}q_proj`]);
      mat(`${p}attn_k.weight`, nEmbd, kvDim, hf[`${q}k_proj`]);
      mat(`${p}attn_v.weight`, nEmbd, kvDim, hf[`${q}v_proj`]);
      mat(`${p}attn_output.weight`, qDim, nEmbd, hf[`${q}o_proj`]);
      vec(`${p}attn_q_norm.weight`, plus1(hfNorm(`${q}q_norm`, hd)));
      vec(`${p}attn_k_norm.weight`, plus1(hfNorm(`${q}k_norm`, hd)));
    }
    hf[`${hp}mlp.gate_proj`] = draw(nFF * nEmbd, 0.08);
    hf[`${hp}mlp.up_proj`] = draw(nFF * nEmbd, 0.08);
    hf[`${hp}mlp.down_proj`] = draw(nEmbd * nFF, 0.08);
    mat(`${p}ffn_gate.weight`, nEmbd, nFF, hf[`${hp}mlp.gate_proj`]);
    mat(`${p}ffn_up.weight`, nEmbd, nFF, hf[`${hp}mlp.up_proj`]);
    mat(`${p}ffn_down.weight`, nFF, nEmbd, hf[`${hp}mlp.down_proj`]);
  }

  const A = 'qwen35';
  const U32 = GGUF_TYPE.UINT32;
  const kv = (key, type, value, elemType) => ({ key, type, value, elemType });
  const metadata = [
    kv('general.architecture', GGUF_TYPE.STRING, A),
    kv('general.name', GGUF_TYPE.STRING, `tiny-qwen35-${quant}`),
    kv(`${A}.context_length`, U32, ctx),
    kv(`${A}.embedding_length`, U32, nEmbd),
    kv(`${A}.block_count`, U32, nLayer),
    kv(`${A}.feed_forward_length`, U32, nFF),
    kv(`${A}.attention.head_count`, U32, nHead),
    kv(`${A}.attention.head_count_kv`, U32, nHeadKv),
    kv(`${A}.attention.key_length`, U32, hd),
    kv(`${A}.attention.value_length`, U32, hd),
    kv(`${A}.attention.layer_norm_rms_epsilon`, GGUF_TYPE.FLOAT32, 1e-6),
    kv(`${A}.rope.freq_base`, GGUF_TYPE.FLOAT32, 10000000),
    kv(`${A}.rope.dimension_count`, U32, nRot),
    kv(`${A}.rope.dimension_sections`, GGUF_TYPE.ARRAY, sections, GGUF_TYPE.INT32),
    kv(`${A}.ssm.conv_kernel`, U32, K),
    kv(`${A}.ssm.state_size`, U32, dk),
    kv(`${A}.ssm.group_count`, U32, nK),
    kv(`${A}.ssm.time_step_rank`, U32, nV),
    kv(`${A}.ssm.inner_size`, U32, inner),
    kv(`${A}.full_attention_interval`, U32, interval),
    kv('tokenizer.ggml.model', GGUF_TYPE.STRING, 'gpt2'),
    kv('tokenizer.ggml.pre', GGUF_TYPE.STRING, 'qwen35'),
    kv('tokenizer.ggml.tokens', GGUF_TYPE.ARRAY, tokens, GGUF_TYPE.STRING),
    kv('tokenizer.ggml.token_type', GGUF_TYPE.ARRAY, types, GGUF_TYPE.INT32),
    kv('tokenizer.ggml.merges', GGUF_TYPE.ARRAY, TINY_MERGES.map(([a, b]) => `${a} ${b}`), GGUF_TYPE.STRING),
    kv('tokenizer.ggml.eos_token_id', U32, eos),
    kv('tokenizer.chat_template', GGUF_TYPE.STRING, "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}"),
  ];
  let buffer = null;
  if (o.outFile) writeGGUF(o.outFile, { metadata, tensors, alignment: o.alignment ?? 32 });
  else buffer = buildGGUF({ metadata, tensors, alignment: o.alignment ?? 32 });
  return { buffer, tokens, hf, cfg: { arch: A, nEmbd, nLayer, interval, nHead, nHeadKv, headDim: hd, nRot, nFF, nK, nV, headSsm: dk, convKernel: K, keyDim, inner, convDim, vocab, eos, isRecurrent, sections } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { _, flags } = parseArgs(process.argv.slice(2), {});
  if (!_[0]) { console.error('usage: make-tiny-qwen35-gguf.js out.gguf [--quant f32|q8_0|q4_0|q4_k_m] [--seed N]'); process.exit(1); }
  const { buffer } = makeTinyQwen35({ quant: typeof flags.quant === 'string' ? /** @type {any} */ (flags.quant) : undefined, seed: num(flags.seed, 1) });
  fs.writeFileSync(_[0], buffer);
  console.log(`wrote ${_[0]} (${buffer.length} bytes)`);
}
