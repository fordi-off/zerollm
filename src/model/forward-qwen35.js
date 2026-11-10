// Reference (single-thread, plain JS) forward pass of Qwen3.5 dense ("qwen35"): a hybrid of Gated DeltaNet
// (linear attention, recurrent) layers and every `full_attention_interval`-th full-attention layer.
// RefModel (forward.js) delegates here for arch "qwen35"; this is the oracle for the later pool / SIMD path.
//
// Per layer:  x += mixer(rmsnorm(x, attn_norm));  x += swiglu(rmsnorm(x, post_attention_norm))
//   full attention layer: q,gate = split(attn_q x) per head [query | gate]; per-head RMSNorm of q and k; rotary on the first
//                         nRot dims (NEOX pairs; text-only M-RoPE == plain rotary); softmax attention over the KV cache;
//                         out = attn_output( attention * sigmoid(gate) )
//   Gated DeltaNet layer: see src/compute/ref-deltanet.js; state = conv window + one [vDim x kDim] matrix per value head.
// Sources for every formula: docs/DECISIONS.md.
//
// STATE AND TRUNCATION. A KV cache is positional: dropping positions is just `length = n`. Recurrent state is not: it
// summarises the whole prefix. So the model keeps a bounded set of CHECKPOINTS (copies of the recurrent state at chosen
// positions) and truncate(n) restores the nearest checkpoint <= n and replays the tokens in between (documented below).
import fs from 'node:fs';
import { modelConfig, recurrentStateBytes } from './config.js';
import { RecurrentState } from './recurrent-state.js';
import { StateCheckpoints } from './state-checkpoints.js';
import { KVCache } from './kvcache.js';
import { loadWeight, loadVector } from './loader.js';
import { Act, matvec, dequantRow, rmsnorm, softmax, silu, ropeTables, applyRope } from '../compute/ref-kernels.js';
import { convStep, l2normHeads, deltaRuleHead, gatedRmsNormHead, sigmoid, softplus, keyHeadOfValueHead } from '../compute/ref-deltanet.js';

export { RecurrentState };

/** Scratch buffers of one forward pass (sizes are fixed by the config). */
function makeScratch(cfg) {
  const s = cfg.ssm;
  const qDim = cfg.nHead * cfg.headDim; const kvDim = cfg.nHeadKv * cfg.headDim;
  return {
    qkv: new Float32Array(s.convDim), z: new Float32Array(s.innerSize),
    beta: new Float32Array(s.nVHeads), alpha: new Float32Array(s.nVHeads), core: new Float32Array(s.innerSize),
    qfull: new Float32Array(2 * qDim), q: new Float32Array(qDim), gate: new Float32Array(qDim), k: new Float32Array(kvDim), v: new Float32Array(kvDim),
    attOut: new Float32Array(qDim),
  };
}

/**
 * One Gated DeltaNet layer for one token. Reads the normalised hidden state `h`, updates `convState` / `ssmState`
 * (this layer's slices) and writes the layer output (before the residual add) to `out`.
 * @param {import('./config.js').ModelConfig} cfg
 * @param {object} L layer weights  @param {Act} h
 * @param {Float32Array} convState @param {Float32Array} ssmState @param {ReturnType<typeof makeScratch>} sc @param {Float32Array} out
 */
export function ssmLayerStep(cfg, L, h, convState, ssmState, sc, out) {
  const { convKernel, headKDim: dk, headVDim: dv, nKHeads, nVHeads, convDim } = cfg.ssm;
  const keyDim = nKHeads * dk;
  matvec(L.wqkv, h, sc.qkv); matvec(L.wz, h, sc.z); matvec(L.wBeta, h, sc.beta); matvec(L.wAlpha, h, sc.alpha);
  convStep(convState, L.conv, sc.qkv, sc.qkv, convDim, convKernel); // causal depthwise conv + SiLU, in place
  // channels: [q: nKHeads x dk | k: nKHeads x dk | v: nVHeads x dv]
  l2normHeads(sc.qkv, 0, nKHeads, dk, cfg.rmsEps, 1 / Math.sqrt(dk)); // q: L2-normalise, then scale by 1/sqrt(dk)
  l2normHeads(sc.qkv, keyDim, nKHeads, dk, cfg.rmsEps);               // k: L2-normalise
  for (let hv = 0; hv < nVHeads; hv++) {
    const kh = keyHeadOfValueHead(hv, nKHeads, nVHeads); // tiled head layout (llama.cpp converter): hv % nKHeads; see ref-deltanet.js
    const beta = sigmoid(sc.beta[hv]);
    const g = softplus(sc.alpha[hv] + L.dtBias[hv]) * L.ssmA[hv]; // ssmA = -exp(A_log) (done by the converter): g <= 0
    deltaRuleHead(ssmState, hv * dv * dk, sc.qkv, kh * dk, sc.qkv, keyDim + kh * dk, sc.qkv, 2 * keyDim + hv * dv, sc.core, hv * dv, dk, dv, Math.exp(g), beta);
    gatedRmsNormHead(sc.core, hv * dv, sc.z, hv * dv, L.ssmNorm, dv, cfg.rmsEps);
  }
  matvec(L.wOut, new Act(sc.core), out);
}

/**
 * One full-attention layer for one token at position `pos`: appends k/v to this layer's cache arrays (K, V) and writes the
 * layer output (before the residual add) to `out`.
 * @param {import('./config.js').ModelConfig} cfg
 * @param {object} L @param {Act} h @param {Float32Array} K @param {Float32Array} V
 * @param {number} pos @param {{cos: Float32Array, sin: Float32Array, half: number}} rope
 * @param {ReturnType<typeof makeScratch>} sc @param {Float32Array} att softmax scratch (>= pos + 1) @param {Float32Array} out
 */
export function attnLayerStep(cfg, L, h, K, V, pos, rope, sc, att, out) {
  const { nHead, nHeadKv, headDim: D, nRot, rmsEps } = cfg;
  const kvDim = nHeadKv * D;
  const group = nHead / nHeadKv;
  const scale = 1 / Math.sqrt(D);
  const { qfull, q, gate, k, v, attOut } = sc;
  matvec(L.wq, h, qfull); matvec(L.wk, h, k); matvec(L.wv, h, v);
  for (let hd = 0; hd < nHead; hd++) { // attn_q rows per head: [query D | gate D]
    for (let i = 0; i < D; i++) { q[hd * D + i] = qfull[hd * 2 * D + i]; gate[hd * D + i] = qfull[hd * 2 * D + D + i]; }
    rmsnorm(q, q, L.qNorm, rmsEps, D, hd * D, hd * D);
  }
  for (let hd = 0; hd < nHeadKv; hd++) rmsnorm(k, k, L.kNorm, rmsEps, D, hd * D, hd * D);
  applyRope(q, nHead, D, nRot, 'neox', rope, pos);
  applyRope(k, nHeadKv, D, nRot, 'neox', rope, pos);
  K.set(k, pos * kvDim); V.set(v, pos * kvDim);
  for (let hd = 0; hd < nHead; hd++) {
    const kvh = Math.floor(hd / group);
    const qo = hd * D;
    for (let t = 0; t <= pos; t++) {
      const ko = t * kvDim + kvh * D;
      let s = 0;
      for (let i = 0; i < D; i++) s += q[qo + i] * K[ko + i];
      att[t] = s * scale;
    }
    softmax(att, pos + 1);
    attOut.fill(0, qo, qo + D);
    for (let t = 0; t <= pos; t++) {
      const p = att[t];
      const vo = t * kvDim + kvh * D;
      for (let i = 0; i < D; i++) attOut[qo + i] += p * V[vo + i];
    }
  }
  for (let i = 0; i < attOut.length; i++) attOut[i] *= sigmoid(gate[i]);
  matvec(L.wo, new Act(attOut), out);
}

/**
 * Every tensor the reference model reads, with its GGUF dims ([ne0 = inputs, ne1 = outputs]).
 * @param {import('./config.js').ModelConfig} cfg @returns {Array<[string, number[]]>}
 */
export function expectedTensors(cfg) {
  const { nEmbd, nFF, nHead, nHeadKv, headDim: hd, vocabSize } = cfg;
  const out = [['token_embd.weight', [nEmbd, vocabSize]], ['output_norm.weight', [nEmbd]]];
  if (!cfg.tiedEmbeddings) out.push(['output.weight', [nEmbd, vocabSize]]);
  for (let l = 0; l < cfg.nLayer; l++) {
    const p = `blk.${l}.`;
    out.push([`${p}attn_norm.weight`, [nEmbd]], [`${p}post_attention_norm.weight`, [nEmbd]],
      [`${p}ffn_gate.weight`, [nEmbd, nFF]], [`${p}ffn_up.weight`, [nEmbd, nFF]], [`${p}ffn_down.weight`, [nFF, nEmbd]]);
    if (cfg.layerIsRecurrent[l]) {
      const s = cfg.ssm;
      out.push([`${p}attn_qkv.weight`, [nEmbd, s.convDim]], [`${p}attn_gate.weight`, [nEmbd, s.innerSize]],
        [`${p}ssm_alpha.weight`, [nEmbd, s.nVHeads]], [`${p}ssm_beta.weight`, [nEmbd, s.nVHeads]],
        [`${p}ssm_conv1d.weight`, [s.convKernel, s.convDim]], [`${p}ssm_dt.bias`, [s.nVHeads]], [`${p}ssm_a`, [s.nVHeads]],
        [`${p}ssm_norm.weight`, [s.headVDim]], [`${p}ssm_out.weight`, [s.innerSize, nEmbd]]);
    } else {
      out.push([`${p}attn_q.weight`, [nEmbd, 2 * nHead * hd]], [`${p}attn_k.weight`, [nEmbd, nHeadKv * hd]], [`${p}attn_v.weight`, [nEmbd, nHeadKv * hd]],
        [`${p}attn_output.weight`, [nHead * hd, nEmbd]], [`${p}attn_q_norm.weight`, [hd]], [`${p}attn_k_norm.weight`, [hd]]);
    }
  }
  return out;
}

/** Throw ONE error naming every missing tensor / wrong shape (instead of failing on the first lookup half way through loading). */
export function checkTensors(g, cfg) {
  const problems = [];
  for (const [name, dims] of expectedTensors(cfg)) {
    const t = g.tensor(name);
    if (!t) problems.push(`missing ${name}`);
    else if (t.dims.length !== dims.length || t.dims.some((d, i) => d !== dims[i])) problems.push(`${name} has dims [${t.dims}], expected [${dims}]`);
  }
  if (problems.length) throw new Error(`this ${cfg.arch} file does not match the layout the engine expects: ${problems.slice(0, 8).join('; ')}${problems.length > 8 ? `; ... (${problems.length} problems)` : ''}`);
}

/**
 * Qwen3.5 reference model. Same interface as RefModel (forward / prefill / truncate / reset / pos / tokens / ctx).
 *
 * Checkpoint policy (recurrent state cannot be cut back by position):
 *  - state before position 0 is all zeros: always available for free;
 *  - a snapshot is taken BEFORE processing (a) the last token of a prompt (a call that wants logits right after calls
 *    that did not: exactly how generate() feeds a prompt), (b) every token in `boundaryIds` (<|im_start|>: chat turn
 *    boundaries), (c) every `checkpointEvery`-th position when that option is > 0;
 *  - at most `maxCheckpoints` snapshots (default RECURRENT_CHECKPOINTS = 6, each recurrentStateBytes(cfg), ~50 MB for
 *    Qwen3.5-4B); when full the one with the smallest position is dropped;
 *  - truncate(n): drop snapshots beyond n, restore the one with the largest position c <= n (or zeros), then replay the
 *    n - c tokens c..n-1 internally, so afterwards pos === n exactly like a KV-only model (generate() needs no change).
 *    The replay costs (n - c) forward passes; truncate(n) returns that number. truncatePlan(n) tells it in advance.
 */
export class Qwen35Model {
  /**
   * @param {import('../gguf/reader.js').GGUFFile} g
   * @param {{ctx?: number, maxCheckpoints?: number, checkpointEvery?: number, boundaryIds?: number[], onProgress?: (done: number, total: number) => void}} [opts]
   */
  constructor(g, opts = {}) {
    const cfg = this.cfg = modelConfig(g);
    if (!cfg.hybrid) throw new Error('Qwen35Model needs a qwen35 model');
    this.ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
    checkTensors(g, cfg);
    const fd = fs.openSync(g.path, 'r');
    try {
      const total = g.tensors.length;
      let done = 0;
      const W = (n) => { const w = loadWeight(fd, g.requireTensor(n)); opts.onProgress?.(++done, total); return w; };
      const V = (n) => { const v = loadVector(fd, g.requireTensor(n)); opts.onProgress?.(++done, total); return v; };
      this.tokEmbd = W('token_embd.weight');
      this.outputNorm = V('output_norm.weight');
      this.output = cfg.tiedEmbeddings ? this.tokEmbd : W('output.weight');
      this.layers = [];
      for (let l = 0; l < cfg.nLayer; l++) {
        const p = `blk.${l}.`;
        const L = {
          recurrent: cfg.layerIsRecurrent[l],
          attnNorm: V(`${p}attn_norm.weight`), postNorm: V(`${p}post_attention_norm.weight`),
          wGate: W(`${p}ffn_gate.weight`), wUp: W(`${p}ffn_up.weight`), wDown: W(`${p}ffn_down.weight`),
        };
        if (L.recurrent) {
          Object.assign(L, {
            wqkv: W(`${p}attn_qkv.weight`), wz: W(`${p}attn_gate.weight`), wAlpha: W(`${p}ssm_alpha.weight`), wBeta: W(`${p}ssm_beta.weight`),
            conv: V(`${p}ssm_conv1d.weight`), dtBias: V(`${p}ssm_dt.bias`), ssmA: V(`${p}ssm_a`), ssmNorm: V(`${p}ssm_norm.weight`), wOut: W(`${p}ssm_out.weight`),
          });
        } else {
          Object.assign(L, {
            wq: W(`${p}attn_q.weight`), wk: W(`${p}attn_k.weight`), wv: W(`${p}attn_v.weight`), wo: W(`${p}attn_output.weight`),
            qNorm: V(`${p}attn_q_norm.weight`), kNorm: V(`${p}attn_k_norm.weight`),
          });
        }
        this.layers.push(L);
      }
    } finally { fs.closeSync(fd); }
    if (this.layers.some((L) => L.recurrent && L.conv.length !== cfg.ssm.convDim * cfg.ssm.convKernel)) throw new Error('ssm_conv1d has an unexpected size');

    // the KV cache only has the attention layers: layer l uses cache index attnIndex[l]
    this.attnIndex = []; let a = 0;
    for (let l = 0; l < cfg.nLayer; l++) this.attnIndex.push(cfg.layerIsRecurrent[l] ? -1 : a++);
    this.cache = new KVCache({ nLayer: cfg.nAttnLayers, nHeadKv: cfg.nHeadKv, headDim: cfg.headDim }, this.ctx);
    this.state = new RecurrentState(cfg);
    /** token ids currently represented in the KV cache and the recurrent state */
    this.tokens = [];
    this.kernels = 'js';
    this.kv = 'f32';
    this.rope = ropeTables(cfg, this.ctx, null);
    /** the checkpoint store + policy (src/model/state-checkpoints.js, shared with the worker pool) */
    this.ck = new StateCheckpoints(cfg, { maxCheckpoints: opts.maxCheckpoints, checkpointEvery: opts.checkpointEvery, boundaryIds: opts.boundaryIds ?? findBoundaryIds(g) });
    this.prevNoLogits = false;

    this.x = new Float32Array(cfg.nEmbd);
    this.xb = new Float32Array(cfg.nEmbd);
    this.proj = new Float32Array(cfg.nEmbd);
    this.scratch = makeScratch(cfg);
    this.att = new Float32Array(this.ctx);
    this.gate = new Float32Array(cfg.nFF);
    this.up = new Float32Array(cfg.nFF);
    this.hb = new Float32Array(cfg.nFF);
    this.logits = new Float32Array(cfg.vocabSize);
  }

  /** @param {import('../gguf/reader.js').GGUFFile} g @param {object} [opts] */
  static fromGGUF(g, opts) { return new Qwen35Model(g, opts); }

  /** Number of positions currently held. */
  get pos() { return this.cache.length; }

  /** Bytes of recurrent state per snapshot / live state. */
  get stateBytes() { return recurrentStateBytes(this.cfg); }

  /** Positions of the stored checkpoints (ascending). */
  get checkpointPositions() { return this.ck.positions; }

  get maxCheckpoints() { return this.ck.max; }

  get checkpointEvery() { return this.ck.every; }

  /** Token ids that get a snapshot before them (<|im_start|>). */
  get boundaryIds() { return this.ck.boundaryIds; }

  // ---- the on-disk prefix cache (src/model/prefix-cache.js): KV of the attention layers + the recurrent state at the prefix end.
  // Layer numbers are ATTENTION-layer numbers (0 .. nAttnLayers - 1).

  /** The first n positions of attention layer #layer's K (isV = 0) or V (isV = 1) cache: an [n][kvDim] f32 view. */
  async kvGet(layer, isV, n) { return (isV ? this.cache.v : this.cache.k)[layer].subarray(0, n * this.cfg.nHeadKv * this.cfg.headDim); }

  /** Write n positions of attention layer #layer's K or V cache (data: [n][kvDim] f32). */
  async kvSet(layer, isV, n, data) { (isV ? this.cache.v : this.cache.k)[layer].set(data.subarray(0, n * this.cfg.nHeadKv * this.cfg.headDim)); }

  /** A copy of the live recurrent state: {conv, ssm} f32 arrays (async like the pool's). */
  async getRecurrentState() { return { conv: this.state.conv.slice(), ssm: this.state.ssm.slice() }; }

  /** Replace the live recurrent state. */
  async setRecurrentState({ conv, ssm }) { this.state.conv.set(conv); this.state.ssm.set(ssm); }

  /** After the KV regions and the recurrent state were filled from a prefix cache: declare that the model now holds exactly `ids`. */
  restorePrefix(ids) {
    this.tokens = Array.from(ids); this.cache.length = ids.length; this.ck.clear(); this.prevNoLogits = false;
    this.checkpoint(); // the end of the cached prefix is a natural place to come back to
  }

  /** Forget everything (start a new conversation). */
  reset() { this.cache.reset(); this.tokens.length = 0; this.state.clear(); this.ck.clear(); this.prevNoLogits = false; }

  close() { /* nothing to release */ }

  /** Take a snapshot of the current state at the current position (no-op at position 0 or if one exists). */
  checkpoint() { this.ck.save(this.pos, this.state); }

  /**
   * What truncate(n) would do: the checkpoint it restores from and how many tokens it replays.
   * @returns {{from: number, replay: number}}
   */
  truncatePlan(n) { return this.ck.plan(n, this.pos); }

  /**
   * Roll back to `n` positions (e.g. to reuse a shared prefix). Restores the nearest checkpoint <= n and replays the
   * remaining tokens (synchronously: n - from forward passes). @returns {number} tokens replayed
   */
  truncate(n) {
    if (n >= this.pos) return 0;
    if (n < 0) n = 0;
    const { from, replay } = this.truncatePlan(n);
    const tail = this.tokens.slice(from, n);
    this.ck.dropAfter(n);
    this.ck.restore(this.state, from);
    this.cache.length = from; this.tokens.length = from;
    this.prevNoLogits = false;
    for (const t of tail) this.forward(t, { logits: false });
    if (replay > 0) this.checkpoint(); // n is where the next prompt continues (and where a regenerate comes back to): keep it
    this.prevNoLogits = false;
    return replay;
  }

  /**
   * Run one token at the current position.
   * @param {number} token
   * @param {{logits?: boolean}} [opts] logits:false skips the output projection (prompt prefill)
   * @returns {Float32Array|null} logits (a reused buffer: copy if you need to keep it)
   */
  forward(token, opts = {}) {
    const { cfg, cache, state } = this;
    const pos = cache.length;
    if (pos >= this.ctx) throw new Error(`context full (${this.ctx} tokens)`);
    if (!(token >= 0 && token < cfg.vocabSize)) throw new Error(`token id ${token} out of range`);
    const wantLogits = opts.logits !== false;
    // checkpoint policy (see the class comment)
    if (this.ck.wantsBefore(pos, token, wantLogits, this.prevNoLogits)) this.checkpoint();
    this.prevNoLogits = !wantLogits;

    const { nEmbd, rmsEps } = cfg;
    const { x, xb, proj, gate, up, hb, scratch, att } = this;
    dequantRow(this.tokEmbd, token, x);
    let ri = 0;
    for (let l = 0; l < cfg.nLayer; l++) {
      const L = this.layers[l];
      rmsnorm(xb, x, L.attnNorm, rmsEps);
      const h = new Act(xb);
      if (L.recurrent) { ssmLayerStep(cfg, L, h, state.convOf(ri), state.ssmOf(ri), scratch, proj); ri++; } else {
        const ai = this.attnIndex[l];
        attnLayerStep(cfg, L, h, cache.k[ai], cache.v[ai], pos, this.rope, scratch, att, proj);
      }
      for (let i = 0; i < nEmbd; i++) x[i] += proj[i];

      rmsnorm(xb, x, L.postNorm, rmsEps); // post_attention_norm is the pre-FFN norm
      const h2 = new Act(xb);
      matvec(L.wGate, h2, gate); matvec(L.wUp, h2, up);
      for (let i = 0; i < cfg.nFF; i++) hb[i] = silu(gate[i]) * up[i];
      matvec(L.wDown, new Act(hb), proj);
      for (let i = 0; i < nEmbd; i++) x[i] += proj[i];
    }
    cache.length = pos + 1;
    this.tokens.push(token);
    if (!wantLogits) return null;
    rmsnorm(xb, x, this.outputNorm, rmsEps);
    matvec(this.output, new Act(xb), this.logits);
    return this.logits;
  }

  /** Feed several tokens, returning logits for the last one. */
  prefill(tokens, wantLogits = true) {
    let logits = null;
    for (let i = 0; i < tokens.length; i++) logits = this.forward(tokens[i], { logits: wantLogits && i === tokens.length - 1 });
    return logits;
  }
}

/** Token ids that start a chat turn (<|im_start|>): a checkpoint before each makes rewriting the last turn cheap. */
export function findBoundaryIds(g) {
  const tokens = g.get('tokenizer.ggml.tokens');
  if (!Array.isArray(tokens)) return [];
  const id = tokens.indexOf('<|im_start|>');
  return id >= 0 ? [id] : [];
}
