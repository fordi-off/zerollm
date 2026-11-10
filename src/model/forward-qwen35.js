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
import { modelConfig, recurrentStateBytes, RECURRENT_CHECKPOINTS } from './config.js';
import { KVCache } from './kvcache.js';
import { loadWeight, loadVector } from './loader.js';
import { Act, matvec, dequantRow, rmsnorm, softmax, silu, ropeTables, applyRope } from '../compute/ref-kernels.js';
import { convStep, l2normHeads, deltaRuleHead, gatedRmsNormHead, sigmoid, softplus } from '../compute/ref-deltanet.js';

/**
 * All recurrent state of one sequence: for each recurrent layer (numbered 0.. in layer order) the conv window and the
 * delta-rule matrices, in two flat f32 arrays so a snapshot is two `set()` calls and the size is explicit.
 */
export class RecurrentState {
  /** @param {import('./config.js').ModelConfig} cfg */
  constructor(cfg) {
    const s = cfg.ssm;
    this.nLayers = cfg.layerIsRecurrent.filter(Boolean).length;
    /** floats per layer */
    this.convSize = (s.convKernel - 1) * s.convDim;
    this.ssmSize = s.nVHeads * s.headVDim * s.headKDim;
    this.conv = new Float32Array(this.nLayers * this.convSize);
    this.ssm = new Float32Array(this.nLayers * this.ssmSize);
  }

  /** Total bytes (matches recurrentStateBytes(cfg)). */
  get bytes() { return (this.conv.length + this.ssm.length) * 4; }

  /** Conv window of recurrent layer #i: [channel][kernel-1], oldest first. */
  convOf(i) { return this.conv.subarray(i * this.convSize, (i + 1) * this.convSize); }

  /** Delta-rule matrices of recurrent layer #i: [vHead][vDim][kDim]. */
  ssmOf(i) { return this.ssm.subarray(i * this.ssmSize, (i + 1) * this.ssmSize); }

  clear() { this.conv.fill(0); this.ssm.fill(0); }

  /** @param {RecurrentState} o */
  copyFrom(o) { this.conv.set(o.conv); this.ssm.set(o.ssm); }
}

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
    const kh = hv % nKHeads; // tiled head layout (llama.cpp converter): value head hv reads key head hv % nKHeads
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
    this.maxCheckpoints = opts.maxCheckpoints ?? RECURRENT_CHECKPOINTS;
    this.checkpointEvery = opts.checkpointEvery ?? 0;
    this.boundaryIds = new Set(opts.boundaryIds ?? findBoundaryIds(g));
    /** @type {Array<{pos: number, state: RecurrentState}>} sorted by pos */
    this.checkpoints = [];
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
  get checkpointPositions() { return this.checkpoints.map((c) => c.pos); }

  /** The on-disk prefix cache stores KV state only; a hybrid model also needs the recurrent state. */
  async kvGet() { throw new Error('the KV prefix cache does not support Qwen3.5 (recurrent layers): its state is not a KV prefix'); }
  async kvSet() { throw new Error('the KV prefix cache does not support Qwen3.5 (recurrent layers): its state is not a KV prefix'); }
  restorePrefix() { throw new Error('the KV prefix cache does not support Qwen3.5 (recurrent layers): its state is not a KV prefix'); }

  /** Forget everything (start a new conversation). */
  reset() { this.cache.reset(); this.tokens.length = 0; this.state.clear(); this.checkpoints.length = 0; this.prevNoLogits = false; }

  close() { /* nothing to release */ }

  /** Take a snapshot of the current state at the current position (no-op at position 0 or if one exists). */
  checkpoint() {
    const pos = this.pos;
    if (pos === 0 || this.maxCheckpoints <= 0 || this.checkpoints.some((c) => c.pos === pos)) return;
    let snap;
    if (this.checkpoints.length >= this.maxCheckpoints) snap = this.checkpoints.shift().state; // drop the smallest position, reuse its buffers
    else snap = new RecurrentState(this.cfg);
    snap.copyFrom(this.state);
    this.checkpoints.push({ pos, state: snap });
    this.checkpoints.sort((a, b) => a.pos - b.pos);
  }

  /**
   * What truncate(n) would do: the checkpoint it restores from and how many tokens it replays.
   * @returns {{from: number, replay: number}}
   */
  truncatePlan(n) {
    if (n >= this.pos) return { from: this.pos, replay: 0 };
    let from = 0;
    for (const c of this.checkpoints) if (c.pos <= n && c.pos > from) from = c.pos;
    return { from, replay: n - from };
  }

  /**
   * Roll back to `n` positions (e.g. to reuse a shared prefix). Restores the nearest checkpoint <= n and replays the
   * remaining tokens (synchronously: n - from forward passes). @returns {number} tokens replayed
   */
  truncate(n) {
    if (n >= this.pos) return 0;
    if (n < 0) n = 0;
    const { from, replay } = this.truncatePlan(n);
    const tail = this.tokens.slice(from, n);
    this.checkpoints = this.checkpoints.filter((c) => c.pos <= n);
    if (from === 0) this.state.clear();
    else this.state.copyFrom(this.checkpoints.find((c) => c.pos === from).state);
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
    if (pos > 0 && ((wantLogits && this.prevNoLogits) || this.boundaryIds.has(token) || (this.checkpointEvery > 0 && pos % this.checkpointEvery === 0))) this.checkpoint();
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
