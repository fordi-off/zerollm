// Deliberately naive, independent implementation of Qwen3.5 (arch "qwen35") for tests, written from the Hugging Face
// formulas (Qwen3-Next / Qwen3.5 modeling code) and NOT from src/model/forward-qwen35.js:
//   - whole-sequence recomputation in double precision, plain arrays, no caches, no state objects;
//   - the Gated DeltaNet is evaluated in its PARALLEL "WY / chunk" form (HF torch_chunk_gated_delta_rule with one chunk
//     covering the whole sequence and a zero initial state: decay mask, forward substitution of (I + A)^-1, k_cumdecay),
//     which is a different algorithm from the engine's token-by-token recurrence;
//   - weights are in the HF layout (value heads GROUPED per key head, norm weights zero-centred "1 + w", A_log, dt_bias,
//     repeat_interleave of q/k): `hfFromGguf` undoes llama.cpp's converter for any quantization;
//     for F32 files the generator also hands over the original HF-layout floats directly.
import { readTensor } from '../src/gguf/reader.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { dequantizeQ8_0, dequantizeQ4_0, dequantizeQ5_0, dequantizeQ4_K, dequantizeQ5_K, dequantizeQ6_K } from '../src/compute/quant.js';
import { modelConfig } from '../src/model/config.js';

function floats(g, name) {
  const t = g.requireTensor(name);
  const raw = Uint8Array.from(readTensor(g, name));
  switch (t.type) {
    case GGML_TYPE.F32: return Float32Array.from(new Float32Array(raw.buffer));
    case GGML_TYPE.Q8_0: return dequantizeQ8_0(raw, t.nElements);
    case GGML_TYPE.Q4_0: return dequantizeQ4_0(raw, t.nElements);
    case GGML_TYPE.Q5_0: return dequantizeQ5_0(raw, t.nElements);
    case GGML_TYPE.Q4_K: return dequantizeQ4_K(raw, t.nElements);
    case GGML_TYPE.Q5_K: return dequantizeQ5_K(raw, t.nElements);
    case GGML_TYPE.Q6_K: return dequantizeQ6_K(raw, t.nElements);
    default: throw new Error('naive: unsupported type');
  }
}

/** inverse of the converter's head reorder: tiled index (j * nK + kh) -> HF grouped index (kh * r + j), along rows */
function unTileRows(m, cols, nK, r, hd) {
  const out = new Float32Array(m.length);
  for (let kh = 0; kh < nK; kh++) for (let j = 0; j < r; j++) out.set(m.subarray((j * nK + kh) * hd * cols, (j * nK + kh + 1) * hd * cols), (kh * r + j) * hd * cols);
  return out;
}
/** same along the columns */
function unTileCols(m, rows, cols, nK, r, hd) {
  const out = new Float32Array(m.length);
  for (let row = 0; row < rows; row++) for (let kh = 0; kh < nK; kh++) for (let j = 0; j < r; j++) out.set(m.subarray(row * cols + (j * nK + kh) * hd, row * cols + (j * nK + kh + 1) * hd), row * cols + (kh * r + j) * hd);
  return out;
}

/** HF-layout float weights (flat Float32Arrays by HF name) recovered from a qwen35 GGUF of any quantization. */
export function hfFromGguf(g) {
  const c = modelConfig(g);
  const { nKHeads: nK, nVHeads: nV, headKDim: dk, convKernel: K, convDim } = c.ssm;
  const r = nV / nK; const keyDim = nK * dk; const inner = nV * dk;
  const minus1 = (f) => f.map((v) => v - 1);
  const hf = {};
  hf['model.embed_tokens'] = floats(g, 'token_embd.weight');
  hf['model.norm'] = minus1(floats(g, 'output_norm.weight'));
  for (let l = 0; l < c.nLayer; l++) {
    const p = `blk.${l}.`; const hp = `layers.${l}.`;
    hf[`${hp}input_layernorm`] = minus1(floats(g, `${p}attn_norm.weight`));
    hf[`${hp}post_attention_layernorm`] = minus1(floats(g, `${p}post_attention_norm.weight`));
    hf[`${hp}mlp.gate_proj`] = floats(g, `${p}ffn_gate.weight`);
    hf[`${hp}mlp.up_proj`] = floats(g, `${p}ffn_up.weight`);
    hf[`${hp}mlp.down_proj`] = floats(g, `${p}ffn_down.weight`);
    if (c.layerIsRecurrent[l]) {
      const q = `${hp}linear_attn.`;
      const qkv = floats(g, `${p}attn_qkv.weight`);
      const nEmbd = qkv.length / convDim;
      const out = new Float32Array(qkv.length); out.set(qkv.subarray(0, 2 * keyDim * nEmbd)); out.set(unTileRows(qkv.subarray(2 * keyDim * nEmbd), nEmbd, nK, r, dk), 2 * keyDim * nEmbd);
      hf[`${q}in_proj_qkv`] = out;
      hf[`${q}in_proj_z`] = unTileRows(floats(g, `${p}attn_gate.weight`), nEmbd, nK, r, dk);
      hf[`${q}in_proj_a`] = unTileRows(floats(g, `${p}ssm_alpha.weight`), nEmbd, nK, r, 1);
      hf[`${q}in_proj_b`] = unTileRows(floats(g, `${p}ssm_beta.weight`), nEmbd, nK, r, 1);
      const conv = floats(g, `${p}ssm_conv1d.weight`);
      const convHf = new Float32Array(conv.length); convHf.set(conv.subarray(0, 2 * keyDim * K)); convHf.set(unTileRows(conv.subarray(2 * keyDim * K), K, nK, r, dk), 2 * keyDim * K);
      hf[`${q}conv1d`] = convHf;
      hf[`${q}A_log`] = unTileRows(floats(g, `${p}ssm_a`), 1, nK, r, 1).map((v) => Math.log(-v));
      hf[`${q}dt_bias`] = unTileRows(floats(g, `${p}ssm_dt.bias`), 1, nK, r, 1);
      hf[`${q}norm`] = floats(g, `${p}ssm_norm.weight`);
      hf[`${q}out_proj`] = unTileCols(floats(g, `${p}ssm_out.weight`), nEmbd, inner, nK, r, dk);
    } else {
      const q = `${hp}self_attn.`;
      hf[`${q}q_proj`] = floats(g, `${p}attn_q.weight`);
      hf[`${q}k_proj`] = floats(g, `${p}attn_k.weight`);
      hf[`${q}v_proj`] = floats(g, `${p}attn_v.weight`);
      hf[`${q}o_proj`] = floats(g, `${p}attn_output.weight`);
      hf[`${q}q_norm`] = minus1(floats(g, `${p}attn_q_norm.weight`));
      hf[`${q}k_norm`] = minus1(floats(g, `${p}attn_k_norm.weight`));
    }
  }
  return hf;
}

const rowsOf = (flat, cols) => Array.from({ length: flat.length / cols }, (_, r) => Array.from(flat.subarray(r * cols, (r + 1) * cols)));
const mv = (W, x) => W.map((row) => row.reduce((a, w, i) => a + w * x[i], 0));
const add = (a, b) => a.map((v, i) => v + b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const silu = (z) => z / (1 + Math.exp(-z));
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const softplus = (z) => Math.log(1 + Math.exp(z));
const rmsPlus1 = (x, w, eps) => { const s = 1 / Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length + eps); return x.map((v, i) => v * s * (1 + w[i])); }; // Qwen3-Next RMSNorm
const l2 = (x, eps) => { const s = 1 / Math.sqrt(dot(x, x) + eps); return x.map((v) => v * s); };
const chunks = (v, n) => Array.from({ length: v.length / n }, (_, i) => v.slice(i * n, (i + 1) * n));

/**
 * Parallel (single chunk, zero initial state) gated delta rule for ONE head, HF torch_chunk_gated_delta_rule.
 * @param {number[][]} q [T][dk] already L2-normalised and scaled  @param {number[][]} k [T][dk]  @param {number[][]} v [T][dv]
 * @param {number[]} g [T] log decay (<= 0)  @param {number[]} beta [T]
 * @returns {number[][]} [T][dv]
 */
function chunkGatedDeltaRule(q, k, v, g, beta) {
  const T = q.length;
  const gc = []; g.reduce((s, x, i) => { gc[i] = s + x; return gc[i]; }, 0);
  const decay = (i, j) => Math.exp(gc[i] - gc[j]);
  // attn = -(k_beta k^T * decay) strictly lower
  const attn = Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (j < i ? -beta[i] * dot(k[i], k[j]) * decay(i, j) : 0)));
  for (let i = 1; i < T; i++) {
    const row = attn[i].slice();
    for (let j = 0; j < i; j++) { let s = 0; for (let m = 0; m < T; m++) s += attn[i][m] * attn[m][j]; row[j] = attn[i][j] + s; }
    attn[i] = row;
  }
  for (let i = 0; i < T; i++) attn[i][i] += 1;
  const dv = v[0].length;
  const value = attn.map((row) => Array.from({ length: dv }, (_, d) => row.reduce((s, a, j) => s + a * beta[j] * v[j][d], 0)));
  // zero initial state: v_new = value, no inter-chunk term
  return q.map((qi, i) => {
    const o = new Array(dv).fill(0);
    for (let j = 0; j <= i; j++) { const a = dot(qi, k[j]) * decay(i, j); for (let d = 0; d < dv; d++) o[d] += a * value[j][d]; }
    return o;
  });
}

/**
 * @param {import('../src/model/config.js').ModelConfig} c
 * @param {Record<string, Float32Array>} hf HF-layout weights (see hfFromGguf)
 * @param {number[]} tokens
 * @returns {number[][]} logits for every input position
 */
export function naiveQwen35Logits(c, hf, tokens) {
  const eps = c.rmsEps;
  const { nKHeads: nK, nVHeads: nV, headKDim: dk, convKernel: K, convDim } = c.ssm;
  const r = nV / nK; const keyDim = nK * dk; const inner = nV * dk;
  const nEmbd = c.nEmbd; const hd = c.headDim;
  const embd = rowsOf(hf['model.embed_tokens'], nEmbd);
  let xs = tokens.map((t) => embd[t].slice());
  for (let l = 0; l < c.nLayer; l++) {
    const hp = `layers.${l}.`;
    const hs = xs.map((x) => rmsPlus1(x, hf[`${hp}input_layernorm`], eps));
    let mixed;
    if (c.layerIsRecurrent[l]) {
      const q = `${hp}linear_attn.`;
      const Wqkv = rowsOf(hf[`${q}in_proj_qkv`], nEmbd); const Wz = rowsOf(hf[`${q}in_proj_z`], nEmbd);
      const Wb = rowsOf(hf[`${q}in_proj_b`], nEmbd); const Wa = rowsOf(hf[`${q}in_proj_a`], nEmbd);
      const convW = rowsOf(hf[`${q}conv1d`], K);
      const raw = hs.map((h) => mv(Wqkv, h));
      const zs = hs.map((h) => mv(Wz, h)); const bs = hs.map((h) => mv(Wb, h)); const as = hs.map((h) => mv(Wa, h));
      // causal depthwise conv1d (zero left padding), then SiLU
      const mixedQkv = raw.map((_, t) => Array.from({ length: convDim }, (_, ch) => {
        let s = 0;
        for (let i = 0; i < K; i++) { const tt = t - (K - 1) + i; if (tt >= 0) s += convW[ch][i] * raw[tt][ch]; }
        return silu(s);
      }));
      const T = tokens.length;
      const coreByT = Array.from({ length: T }, () => new Array(inner));
      for (let hv = 0; hv < nV; hv++) {
        const kh = Math.floor(hv / r); // repeat_interleave
        const qh = mixedQkv.map((m) => l2(m.slice(kh * dk, (kh + 1) * dk), 1e-6).map((v) => v / Math.sqrt(dk)));
        const kk = mixedQkv.map((m) => l2(m.slice(keyDim + kh * dk, keyDim + (kh + 1) * dk), 1e-6));
        const vv = mixedQkv.map((m) => m.slice(2 * keyDim + hv * dk, 2 * keyDim + (hv + 1) * dk));
        const beta = bs.map((b) => sigmoid(b[hv]));
        const g = as.map((a) => -Math.exp(hf[`${q}A_log`][hv]) * softplus(a[hv] + hf[`${q}dt_bias`][hv]));
        const o = chunkGatedDeltaRule(qh, kk, vv, g, beta);
        o.forEach((ot, t) => {
          // RMSNormGated: weight * norm(x), times silu(z)
          const s = 1 / Math.sqrt(ot.reduce((a, v) => a + v * v, 0) / dk + eps);
          const z = zs[t].slice(hv * dk, (hv + 1) * dk);
          ot.forEach((v, d) => { coreByT[t][hv * dk + d] = v * s * hf[`${q}norm`][d] * silu(z[d]); });
        });
      }
      const Wo = rowsOf(hf[`${q}out_proj`], inner);
      mixed = coreByT.map((core) => mv(Wo, core));
    } else {
      const q = `${hp}self_attn.`;
      const Wq = rowsOf(hf[`${q}q_proj`], nEmbd); const Wk = rowsOf(hf[`${q}k_proj`], nEmbd); const Wv = rowsOf(hf[`${q}v_proj`], nEmbd); const Wo = rowsOf(hf[`${q}o_proj`], c.nHead * hd);
      const Q = []; const G = []; const Kc = []; const Vc = [];
      hs.forEach((h, pos) => {
        const qf = chunks(mv(Wq, h), 2 * hd); // per head [query | gate]
        const qh = qf.map((x) => rope(rmsPlus1(x.slice(0, hd), hf[`${q}q_norm`], eps), pos, c));
        G.push(qf.map((x) => x.slice(hd)));
        Q.push(qh);
        Kc.push(chunks(mv(Wk, h), hd).map((x) => rope(rmsPlus1(x, hf[`${q}k_norm`], eps), pos, c)));
        Vc.push(chunks(mv(Wv, h), hd));
      });
      mixed = hs.map((_, pos) => {
        const concat = [];
        for (let hh = 0; hh < c.nHead; hh++) {
          const kvh = Math.floor(hh / (c.nHead / c.nHeadKv));
          const sc = []; for (let t = 0; t <= pos; t++) sc.push(dot(Q[pos][hh], Kc[t][kvh]) / Math.sqrt(hd));
          const m = Math.max(...sc); const e = sc.map((s) => Math.exp(s - m)); const Z = e.reduce((a, b) => a + b, 0);
          const o = new Array(hd).fill(0);
          for (let t = 0; t <= pos; t++) for (let d = 0; d < hd; d++) o[d] += (e[t] / Z) * Vc[t][kvh][d];
          concat.push(...o.map((v, d) => v * sigmoid(G[pos][hh][d])));
        }
        return mv(Wo, concat);
      });
    }
    xs = xs.map((x, i) => add(x, mixed[i]));
    const Wg = rowsOf(hf[`${hp}mlp.gate_proj`], nEmbd); const Wu = rowsOf(hf[`${hp}mlp.up_proj`], nEmbd); const Wd = rowsOf(hf[`${hp}mlp.down_proj`], c.nFF);
    xs = xs.map((x) => {
      const h = rmsPlus1(x, hf[`${hp}post_attention_layernorm`], eps);
      const gt = mv(Wg, h); const u = mv(Wu, h);
      return add(x, mv(Wd, gt.map((z, i) => silu(z) * u[i])));
    });
  }
  return xs.map((x) => mv(embd, rmsPlus1(x, hf['model.norm'], eps)));
}

/** rotate_half-style partial rotary on the first nRot dims (text-only M-RoPE: all position streams are `pos`). */
function rope(x, pos, c) {
  const out = x.slice(); const half = c.nRot / 2;
  for (let i = 0; i < half; i++) {
    const th = pos * c.ropeBase ** (-2 * i / c.nRot);
    out[i] = x[i] * Math.cos(th) - x[i + half] * Math.sin(th);
    out[i + half] = x[i + half] * Math.cos(th) + x[i] * Math.sin(th);
  }
  return out;
}
