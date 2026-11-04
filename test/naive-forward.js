// Deliberately naive, independent implementation of the transformer from brief section 11.
// Whole-sequence recomputation (no KV cache), double precision, plain arrays. Used only as a test oracle.
import { readTensor } from '../src/gguf/reader.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { dequantizeQ8_0, dequantizeQ4_0 } from '../src/compute/quant.js';
import { modelConfig } from '../src/model/config.js';

function floats(g, name) {
  const t = g.requireTensor(name);
  const raw = Uint8Array.from(readTensor(g, name));
  if (t.type === GGML_TYPE.F32) return Array.from(new Float32Array(raw.buffer));
  if (t.type === GGML_TYPE.Q8_0) return Array.from(dequantizeQ8_0(raw, t.nElements));
  if (t.type === GGML_TYPE.Q4_0) return Array.from(dequantizeQ4_0(raw, t.nElements));
  throw new Error('naive: unsupported type');
}
/** matrix as array of rows */
function rows(g, name) {
  const t = g.requireTensor(name);
  const f = floats(g, name);
  const [cols, nrows] = t.dims;
  return Array.from({ length: nrows }, (_, r) => f.slice(r * cols, (r + 1) * cols));
}
const mv = (W, x) => W.map((row) => row.reduce((a, w, i) => a + w * x[i], 0));
const rms = (x, w, eps) => { const s = 1 / Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length + eps); return x.map((v, i) => v * s * w[i]); };
const add = (a, b) => a.map((v, i) => v + b[i]);

function rope(vec, pos, headDim, nRot, base, type, factors) {
  const out = vec.slice();
  for (let i = 0; i < nRot / 2; i++) {
    let theta = pos * Math.pow(base, -2 * i / nRot);
    if (factors) theta /= factors[i];
    const c = Math.cos(theta); const s = Math.sin(theta);
    const [a, b] = type === 'neox' ? [i, i + nRot / 2] : [2 * i, 2 * i + 1];
    out[a] = vec[a] * c - vec[b] * s;
    out[b] = vec[a] * s + vec[b] * c;
  }
  return out;
}

/** @returns {number[][]} logits for every input position */
export function naiveLogits(g, tokens) {
  const c = modelConfig(g);
  const embd = rows(g, 'token_embd.weight');
  const outW = g.tensor('output.weight') ? rows(g, 'output.weight') : embd;
  const outNorm = floats(g, 'output_norm.weight');
  const factors = g.tensor('rope_freqs.weight') ? floats(g, 'rope_freqs.weight') : null;
  const layers = [];
  for (let l = 0; l < c.nLayer; l++) {
    const p = `blk.${l}.`;
    const has = (n) => g.tensor(p + n);
    layers.push({
      an: floats(g, `${p}attn_norm.weight`), fn: floats(g, `${p}ffn_norm.weight`),
      q: rows(g, `${p}attn_q.weight`), k: rows(g, `${p}attn_k.weight`), v: rows(g, `${p}attn_v.weight`), o: rows(g, `${p}attn_output.weight`),
      bq: has('attn_q.bias') ? floats(g, `${p}attn_q.bias`) : null, bk: has('attn_k.bias') ? floats(g, `${p}attn_k.bias`) : null, bv: has('attn_v.bias') ? floats(g, `${p}attn_v.bias`) : null,
      qn: has('attn_q_norm.weight') ? floats(g, `${p}attn_q_norm.weight`) : null, kn: has('attn_k_norm.weight') ? floats(g, `${p}attn_k_norm.weight`) : null,
      gate: rows(g, `${p}ffn_gate.weight`), up: rows(g, `${p}ffn_up.weight`), down: rows(g, `${p}ffn_down.weight`),
    });
  }
  let xs = tokens.map((t) => embd[t].slice());
  for (const L of layers) {
    const qs = []; const ks = []; const vs = [];
    xs.forEach((x, pos) => {
      const h = rms(x, L.an, c.rmsEps);
      let q = mv(L.q, h); let k = mv(L.k, h); let v = mv(L.v, h);
      if (L.bq) { q = add(q, L.bq); k = add(k, L.bk); v = add(v, L.bv); }
      const heads = (vec, n) => Array.from({ length: n }, (_, i) => vec.slice(i * c.headDim, (i + 1) * c.headDim));
      let qh = heads(q, c.nHead); let kh = heads(k, c.nHeadKv);
      if (L.qn) { qh = qh.map((hh) => rms(hh, L.qn, c.rmsEps)); kh = kh.map((hh) => rms(hh, L.kn, c.rmsEps)); }
      qh = qh.map((hh) => rope(hh, pos, c.headDim, c.nRot, c.ropeBase, c.ropeType, factors));
      kh = kh.map((hh) => rope(hh, pos, c.headDim, c.nRot, c.ropeBase, c.ropeType, factors));
      qs.push(qh); ks.push(kh); vs.push(heads(v, c.nHeadKv));
    });
    xs = xs.map((x, pos) => {
      const concat = [];
      for (let h = 0; h < c.nHead; h++) {
        const kvh = Math.floor(h * c.nHeadKv / c.nHead);
        const sc = [];
        for (let t = 0; t <= pos; t++) sc.push(qs[pos][h].reduce((a, v, i) => a + v * ks[t][kvh][i], 0) / Math.sqrt(c.headDim));
        const m = Math.max(...sc);
        const e = sc.map((s) => Math.exp(s - m));
        const z = e.reduce((a, b) => a + b, 0);
        const o = new Array(c.headDim).fill(0);
        for (let t = 0; t <= pos; t++) for (let i = 0; i < c.headDim; i++) o[i] += (e[t] / z) * vs[t][kvh][i];
        concat.push(...o);
      }
      let y = add(x, mv(L.o, concat));
      const h2 = rms(y, L.fn, c.rmsEps);
      const gt = mv(L.gate, h2); const u = mv(L.up, h2);
      const a = gt.map((z, i) => (z / (1 + Math.exp(-z))) * u[i]);
      y = add(y, mv(L.down, a));
      return y;
    });
  }
  return xs.map((x) => mv(outW, rms(x, outNorm, c.rmsEps)));
}
