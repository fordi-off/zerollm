// Reference (single thread, plain JS) nomic-bert encoder: the oracle for the fast worker path.
// Linear layers use the same int8-activation arithmetic as the SIMD kernels (Act + matvec), everything else is f64-accurate JS.
import fs from 'node:fs';
import { embedConfig } from './config.js';
import { loadWeight, loadVector } from '../model/loader.js';
import { Act, matvec, dequantRow, softmax, silu, ropeTables, applyRope } from '../compute/ref-kernels.js';

export function layerNorm(out, x, w, b, eps, n = x.length, xo = 0, oo = 0) {
  let mean = 0; for (let i = 0; i < n; i++) mean += x[xo + i]; mean /= n;
  let v = 0; for (let i = 0; i < n; i++) { const d = x[xo + i] - mean; v += d * d; } v /= n;
  const inv = 1 / Math.sqrt(v + eps);
  for (let i = 0; i < n; i++) out[oo + i] = (x[xo + i] - mean) * inv * w[i] + b[i];
}

export class RefEncoder {
  /** @param {import('../gguf/reader.js').GGUFFile} g */
  constructor(g) {
    const cfg = this.cfg = embedConfig(g);
    const fd = fs.openSync(g.path, 'r');
    try {
      const W = (n) => loadWeight(fd, g.requireTensor(n)); const V = (n) => loadVector(fd, g.requireTensor(n));
      this.tokEmbd = W('token_embd.weight');
      this.tokType0 = cfg.hasTokenTypes ? (() => { const t = W('token_types.weight'); return dequantRow(t, 0, new Float32Array(cfg.nEmbd)); })() : null;
      this.embNormW = V('token_embd_norm.weight'); this.embNormB = V('token_embd_norm.bias');
      this.layers = [];
      for (let l = 0; l < cfg.nLayer; l++) {
        const p = `blk.${l}.`;
        this.layers.push({ wqkv: W(`${p}attn_qkv.weight`), wo: W(`${p}attn_output.weight`), anW: V(`${p}attn_output_norm.weight`), anB: V(`${p}attn_output_norm.bias`),
          up: W(`${p}ffn_up.weight`), gate: W(`${p}ffn_gate.weight`), down: W(`${p}ffn_down.weight`), lnW: V(`${p}layer_output_norm.weight`), lnB: V(`${p}layer_output_norm.bias`) });
      }
    } finally { fs.closeSync(fd); }
    this.kernels = 'js';
  }

  /** @param {number[]} ids token ids (with [CLS]/[SEP]) @returns {Float32Array} mean-pooled, NOT normalised */
  embedIds(ids) {
    const { cfg } = this; const L = ids.length; const { nEmbd, nHead, headDim, nFF } = cfg;
    const rope = ropeTables({ nRot: headDim, ropeBase: cfg.ropeBase }, L);
    const x = new Float32Array(L * nEmbd); const tmp = new Float32Array(nEmbd);
    for (let t = 0; t < L; t++) {
      dequantRow(this.tokEmbd, ids[t], tmp);
      if (this.tokType0) for (let i = 0; i < nEmbd; i++) tmp[i] += this.tokType0[i];
      layerNorm(x, tmp, this.embNormW, this.embNormB, cfg.eps, nEmbd, 0, t * nEmbd);
    }
    const qkv = new Float32Array(3 * nEmbd); const att = new Float32Array(L); const ao = new Float32Array(nEmbd); const proj = new Float32Array(nEmbd);
    const gate = new Float32Array(nFF); const up = new Float32Array(nFF); const hb = new Float32Array(nFF);
    const Q = new Float32Array(L * nEmbd); const K = new Float32Array(L * nEmbd); const Vv = new Float32Array(L * nEmbd); const A = new Float32Array(L * nEmbd);
    const scale = 1 / Math.sqrt(headDim);
    for (const Ly of this.layers) {
      for (let t = 0; t < L; t++) {
        matvec(Ly.wqkv, new Act(x.subarray(t * nEmbd, (t + 1) * nEmbd)), qkv);
        const q = Q.subarray(t * nEmbd, (t + 1) * nEmbd); const k = K.subarray(t * nEmbd, (t + 1) * nEmbd);
        q.set(qkv.subarray(0, nEmbd)); k.set(qkv.subarray(nEmbd, 2 * nEmbd)); Vv.set(qkv.subarray(2 * nEmbd), t * nEmbd);
        applyRope(q, nHead, headDim, headDim, 'neox', rope, t); applyRope(k, nHead, headDim, headDim, 'neox', rope, t);
      }
      for (let t = 0; t < L; t++) {
        for (let h = 0; h < nHead; h++) {
          const qo = t * nEmbd + h * headDim;
          for (let s = 0; s < L; s++) { let d = 0; const ko = s * nEmbd + h * headDim; for (let i = 0; i < headDim; i++) d += Q[qo + i] * K[ko + i]; att[s] = d * scale; }
          softmax(att, L);
          const oo = t * nEmbd + h * headDim; A.fill(0, oo, oo + headDim);
          for (let s = 0; s < L; s++) { const p = att[s]; const vo = s * nEmbd + h * headDim; for (let i = 0; i < headDim; i++) A[oo + i] += p * Vv[vo + i]; }
        }
      }
      for (let t = 0; t < L; t++) {
        matvec(Ly.wo, new Act(A.subarray(t * nEmbd, (t + 1) * nEmbd)), proj);
        for (let i = 0; i < nEmbd; i++) ao[i] = x[t * nEmbd + i] + proj[i];
        layerNorm(x, ao, Ly.anW, Ly.anB, cfg.eps, nEmbd, 0, t * nEmbd);                       // x = attn_out_norm
        const h = new Act(x.subarray(t * nEmbd, (t + 1) * nEmbd));
        matvec(Ly.gate, h, gate); matvec(Ly.up, h, up);
        for (let i = 0; i < nFF; i++) hb[i] = silu(gate[i]) * up[i];
        matvec(Ly.down, new Act(hb), proj);
        for (let i = 0; i < nEmbd; i++) ao[i] = x[t * nEmbd + i] + proj[i];
        layerNorm(x, ao, Ly.lnW, Ly.lnB, cfg.eps, nEmbd, 0, t * nEmbd);                       // x = layer_output_norm
      }
    }
    const out = new Float32Array(nEmbd);
    if (cfg.pooling === 2) out.set(x.subarray(0, nEmbd));
    else { for (let t = 0; t < L; t++) for (let i = 0; i < nEmbd; i++) out[i] += x[t * nEmbd + i]; for (let i = 0; i < nEmbd; i++) out[i] /= L; }
    return out;
  }
}
