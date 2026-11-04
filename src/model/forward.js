// Reference (single-thread, plain JS) transformer forward pass. The oracle for the worker/WASM path.
import fs from 'node:fs';
import { modelConfig } from './config.js';
import { KVCache } from './kvcache.js';
import { loadWeight, loadVector } from './loader.js';
import { Act, matvec, dequantRow, rmsnorm, softmax, silu, ropeTables, applyRope } from '../compute/ref-kernels.js';

/** Fully loaded model ready to run tokens one at a time. */
export class RefModel {
  /**
   * @param {import('../gguf/reader.js').GGUFFile} g
   * @param {{ctx?: number, onProgress?: (done: number, total: number) => void}} [opts]
   */
  constructor(g, opts = {}) {
    const cfg = this.cfg = modelConfig(g);
    this.ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
    const fd = fs.openSync(g.path, 'r');
    try {
      const names = g.tensors.length;
      let done = 0;
      const W = (n) => { const w = loadWeight(fd, g.requireTensor(n)); opts.onProgress?.(++done, names); return w; };
      const V = (n) => { const v = loadVector(fd, g.requireTensor(n)); opts.onProgress?.(++done, names); return v; };
      const opt = (n, f) => (g.tensor(n) ? f(n) : null);
      this.tokEmbd = W('token_embd.weight');
      this.outputNorm = V('output_norm.weight');
      this.output = cfg.tiedEmbeddings ? this.tokEmbd : W('output.weight');
      this.ropeFreqs = cfg.hasRopeFreqs ? V('rope_freqs.weight') : null;
      this.layers = [];
      for (let l = 0; l < cfg.nLayer; l++) {
        const p = `blk.${l}.`;
        this.layers.push({
          attnNorm: V(`${p}attn_norm.weight`),
          wq: W(`${p}attn_q.weight`), wk: W(`${p}attn_k.weight`), wv: W(`${p}attn_v.weight`), wo: W(`${p}attn_output.weight`),
          bq: opt(`${p}attn_q.bias`, V), bk: opt(`${p}attn_k.bias`, V), bv: opt(`${p}attn_v.bias`, V),
          qNorm: opt(`${p}attn_q_norm.weight`, V), kNorm: opt(`${p}attn_k_norm.weight`, V),
          ffnNorm: V(`${p}ffn_norm.weight`),
          wGate: W(`${p}ffn_gate.weight`), wUp: W(`${p}ffn_up.weight`), wDown: W(`${p}ffn_down.weight`),
        });
      }
    } finally { fs.closeSync(fd); }

    this.cache = new KVCache(cfg, this.ctx);
    /** token ids currently represented in the KV cache */
    this.tokens = [];
    this.kernels = 'js';
    this.rope = ropeTables(cfg, this.ctx, this.ropeFreqs);
    const qDim = cfg.nHead * cfg.headDim; const kvDim = cfg.nHeadKv * cfg.headDim;
    this.x = new Float32Array(cfg.nEmbd);
    this.xb = new Float32Array(cfg.nEmbd);
    this.q = new Float32Array(qDim);
    this.k = new Float32Array(kvDim);
    this.v = new Float32Array(kvDim);
    this.att = new Float32Array(this.ctx);
    this.attOut = new Float32Array(qDim);
    this.proj = new Float32Array(cfg.nEmbd);
    this.gate = new Float32Array(cfg.nFF);
    this.up = new Float32Array(cfg.nFF);
    this.hb = new Float32Array(cfg.nFF);
    this.logits = new Float32Array(cfg.vocabSize);
  }

  /** @param {import('../gguf/reader.js').GGUFFile} g @param {object} [opts] */
  static fromGGUF(g, opts) { return new RefModel(g, opts); }

  /** Number of positions currently in the KV cache. */
  get pos() { return this.cache.length; }

  /** First n positions of one layer's K (isV=0) or V (isV=1) cache, as an [n][kvDim] view (used by the prefix cache). */
  async kvGet(layer, isV, n) { return (isV ? this.cache.v[layer] : this.cache.k[layer]).subarray(0, n * this.cache.kvDim); }

  /** Write n positions of one layer's K or V cache. */
  async kvSet(layer, isV, n, data) { (isV ? this.cache.v[layer] : this.cache.k[layer]).set(data.subarray(0, n * this.cache.kvDim)); }

  /** After the KV regions were filled from a prefix cache: declare that the model now holds exactly `ids`. */
  restorePrefix(ids) { this.cache.length = ids.length; this.tokens = Array.from(ids); }

  /** Forget everything (start a new conversation). */
  reset() { this.cache.reset(); this.tokens.length = 0; }

  /** Roll the cache back to `n` positions (e.g. to reuse a shared prefix). */
  truncate(n) { if (n < this.cache.length) { this.cache.length = n; this.tokens.length = n; } }

  /**
   * Run one token at position `pos` (must equal the current cache length for append-style use).
   * @param {number} token
   * @param {{logits?: boolean}} [opts] logits:false skips the output projection (prompt prefill)
   * @returns {Float32Array|null} logits (a reused buffer: copy if you need to keep it)
   */
  forward(token, opts = {}) {
    const { cfg, cache } = this;
    const pos = cache.length;
    if (pos >= this.ctx) throw new Error(`context full (${this.ctx} tokens)`);
    if (!(token >= 0 && token < cfg.vocabSize)) throw new Error(`token id ${token} out of range`);
    const { nEmbd, nHead, nHeadKv, headDim, nRot, rmsEps, ropeType } = cfg;
    const kvDim = nHeadKv * headDim;
    const group = nHead / nHeadKv;
    const scale = 1 / Math.sqrt(headDim);
    const { x, xb, q, k, v, att, attOut, proj, gate, up, hb } = this;

    dequantRow(this.tokEmbd, token, x);

    for (let l = 0; l < cfg.nLayer; l++) {
      const L = this.layers[l];
      // --- attention
      rmsnorm(xb, x, L.attnNorm, rmsEps);
      const h = new Act(xb);
      matvec(L.wq, h, q); matvec(L.wk, h, k); matvec(L.wv, h, v);
      if (L.bq) for (let i = 0; i < q.length; i++) q[i] += L.bq[i];
      if (L.bk) for (let i = 0; i < k.length; i++) k[i] += L.bk[i];
      if (L.bv) for (let i = 0; i < v.length; i++) v[i] += L.bv[i];
      if (L.qNorm) for (let hd = 0; hd < nHead; hd++) rmsnorm(q, q, L.qNorm, rmsEps, headDim, hd * headDim, hd * headDim);
      if (L.kNorm) for (let hd = 0; hd < nHeadKv; hd++) rmsnorm(k, k, L.kNorm, rmsEps, headDim, hd * headDim, hd * headDim);
      applyRope(q, nHead, headDim, nRot, ropeType, this.rope, pos);
      applyRope(k, nHeadKv, headDim, nRot, ropeType, this.rope, pos);
      cache.k[l].set(k, pos * kvDim);
      cache.v[l].set(v, pos * kvDim);
      const K = cache.k[l]; const V = cache.v[l];
      for (let hd = 0; hd < nHead; hd++) {
        const kvh = Math.floor(hd / group);
        const qo = hd * headDim;
        for (let t = 0; t <= pos; t++) {
          const ko = t * kvDim + kvh * headDim;
          let s = 0;
          for (let i = 0; i < headDim; i++) s += q[qo + i] * K[ko + i];
          att[t] = s * scale;
        }
        softmax(att, pos + 1);
        const oo = hd * headDim;
        attOut.fill(0, oo, oo + headDim);
        for (let t = 0; t <= pos; t++) {
          const p = att[t];
          const vo = t * kvDim + kvh * headDim;
          for (let i = 0; i < headDim; i++) attOut[oo + i] += p * V[vo + i];
        }
      }
      matvec(L.wo, new Act(attOut), proj);
      for (let i = 0; i < nEmbd; i++) x[i] += proj[i];

      // --- feed-forward (SwiGLU)
      rmsnorm(xb, x, L.ffnNorm, rmsEps);
      const h2 = new Act(xb);
      matvec(L.wGate, h2, gate); matvec(L.wUp, h2, up);
      for (let i = 0; i < cfg.nFF; i++) hb[i] = silu(gate[i]) * up[i];
      matvec(L.wDown, new Act(hb), proj);
      for (let i = 0; i < nEmbd; i++) x[i] += proj[i];
    }
    cache.length = pos + 1;
    this.tokens.push(token);
    if (opts.logits === false) return null;
    rmsnorm(xb, x, this.outputNorm, rmsEps);
    matvec(this.output, new Act(xb), this.logits);
    return this.logits;
  }

  /** Feed several tokens, returning logits for the last one (per-token prefill; batched prefill comes in M7). */
  prefill(tokens, wantLogits = true) {
    let logits = null;
    for (let i = 0; i < tokens.length; i++) {
      logits = this.forward(tokens[i], { logits: wantLogits && i === tokens.length - 1 });
    }
    return logits;
  }
}
