// Speculative decoding: a small DRAFT model proposes k tokens, the big TARGET model checks all of them in ONE batched
// forward pass (its weights are read once for the whole block, which on a memory-bandwidth-bound CPU costs little more
// than a single decode step), the accepted prefix is kept and the first wrong position is corrected.
//
// Guarantees
//  - greedy (temperature <= 0): the output is the same token sequence plain greedy decoding produces (the verification
//    logits are bit-identical to single-token logits in this engine, see test/spec-*.test.js);
//  - sampled: standard speculative sampling (Leviathan et al. / Chen et al. 2023). A draft token x ~ q is accepted with
//    probability min(1, p(x) / q(x)); on rejection the replacement is drawn from the normalised residual max(0, p - q); if
//    all k are accepted a bonus token is drawn from p. The output DISTRIBUTION is exactly the target's (p includes the
//    penalties, temperature, top-k, min-p and top-p of the sampler; repetition penalties see the tentative tokens of the
//    block, i.e. the history each position would have if everything before it is accepted). The random stream differs
//    from plain sampling, so a seeded run gives a different (equally distributed) text.
//
// Bookkeeping invariant at the top of every cycle: `seq` = prompt + emitted tokens; its LAST token is "pending" (chosen,
// not yet processed by any model); the target has processed exactly seq[0 .. len-1). Rolling back is just truncating
// the (positional) KV caches: a position is a pure function of the tokens up to it, so whatever matches `seq` stays valid.
import { createHash } from 'node:crypto';
import { commonPrefix, prefillTokens, argmax } from './generate.js';

const yieldToEventLoop = () => new Promise((r) => setImmediate(r));
const EMPTY = Object.freeze([]);

/** How many tokens one verification pass can carry (and return per-position logits for). Infinity: no batch support. */
export function verifyLimit(model) {
  if (typeof model.forwardBatch !== 'function') return Infinity;
  return Math.min(model.maxBatch ?? 1, model.logitsBatch ?? model.maxBatch ?? 1);
}

/** @type {WeakMap<object, Float32Array>} */
const rowScratch = new WeakMap();

/**
 * Logits of EVERY position of `tokens` (row b at b * vocab = distribution of the token after tokens[b]); the result is only
 * valid until the next call. Models with batch + per-position logits (PoolModel) do it in one pass over the weights; any
 * other model (the single-thread reference) falls back to one forward per token.
 * @returns {Promise<Float32Array>}
 */
export async function forwardRows(model, tokens) {
  const B = tokens.length;
  if (typeof model.forwardBatch === 'function' && B <= verifyLimit(model)) return model.forwardBatch(tokens, { logits: 'all' });
  if (B === 1) return model.forward(tokens[0]);
  const V = model.cfg.vocabSize;
  let buf = rowScratch.get(model);
  if (!buf || buf.length < B * V) { buf = new Float32Array(B * V); rowScratch.set(model, buf); }
  for (let b = 0; b < B; b++) buf.set(await model.forward(tokens[b]), b * V);
  return buf.subarray(0, B * V);
}

/** Feed ids[from..] (batched when the model can) and return the logits after the last one. No event-loop yields: small amounts only. */
async function feedLogits(model, ids, from) {
  let logits = null;
  const chunk = typeof model.forwardBatch === 'function' ? Math.max(1, model.maxBatch ?? 1) : 1;
  for (let i = from; i < ids.length;) {
    const end = Math.min(ids.length, i + chunk);
    const last = end === ids.length;
    logits = chunk > 1 ? await model.forwardBatch(ids.slice(i, end), { logits: last }) : await model.forward(ids[i], { logits: last });
    i = end;
  }
  return logits;
}

/** probability of token `id` under a sparse distribution (0 when it is not a candidate) */
export function probOf(d, id) {
  const i = d.ids.indexOf(id);
  return i < 0 ? 0 : d.probs[i] / d.total;
}

/**
 * Draw from the normalised residual max(0, p - q). `scratch` is an all-zero Float64Array(vocab); it is left all-zero.
 * If the residual is empty (p == q up to rounding) a token is drawn from p itself.
 * @param {import('../sampling/sampler.js').Sampler} sample
 * @param {import('../sampling/sampler.js').TokenDist} pd target distribution  @param {import('../sampling/sampler.js').TokenDist} qd draft distribution
 */
export function drawResidual(sample, pd, qd, scratch) {
  for (let i = 0; i < qd.ids.length; i++) scratch[qd.ids[i]] = qd.probs[i] / qd.total;
  const n = pd.ids.length;
  const res = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { const v = pd.probs[i] / pd.total - scratch[pd.ids[i]]; if (v > 0) { res[i] = v; sum += v; } }
  for (let i = 0; i < qd.ids.length; i++) scratch[qd.ids[i]] = 0;
  if (!(sum > 0)) return sample.draw(pd);
  let r = sample.rand() * sum;
  let last = 0;
  for (let i = 0; i < n; i++) { if (res[i] > 0) { last = i; r -= res[i]; if (r <= 0) return pd.ids[i]; } }
  return pd.ids[last];
}

/**
 * Generate tokens after `promptIds` with speculative decoding. Same contract as generate(): yields token ids (stop tokens are
 * not yielded), fills `opts.stats`, reuses the KV prefix of both models, honours abort / maxTokens / stopIds / penalizePrompt.
 * Extra stats: `stats.speculative` = {k, cycles, drafted, accepted, acceptRate, tokensPerCycle, draftMs, verifyMs, otherMs, accepted histogram}.
 * `stats.prefillMs` covers both models' prefill; the prompt's last token is processed inside the first verification pass.
 *
 * @param {any} target big model  @param {any} draft small model with the same vocabulary (PoolModel or RefModel)
 * @param {number[]} promptIds
 * @param {{k?: number, maxTokens?: number, sample?: any, stopIds?: Set<number>, signal?: {aborted: boolean}, stats?: object,
 *          onPrefillProgress?: (done: number, total: number) => void, penalizePrompt?: boolean}} [opts]
 */
export async function* generateSpeculative(target, draft, promptIds, opts = {}) {
  const sample = opts.sample ?? argmax;
  const greedy = sample === argmax || sample.greedy === true;
  if (!greedy && typeof sample.distribution !== 'function') throw new Error('speculative sampling needs a sampler made by createSampler (it must expose distribution())');
  const stop = opts.stopIds ?? new Set();
  const max = opts.maxTokens ?? 256;
  const stats = opts.stats ?? {};
  const kMax = Math.max(1, Math.floor(opts.k ?? 4));
  const n0 = promptIds.length;
  if (n0 === 0) throw new Error('empty prompt');
  if (n0 >= target.ctx) throw new Error(`prompt is ${n0} tokens but the context window is ${target.ctx}; shorten it or raise --ctx`);
  if (n0 >= draft.ctx) throw new Error(`prompt is ${n0} tokens but the draft model's context window is ${draft.ctx}; give the draft model at least the target's context`);
  const V = target.cfg?.vocabSize;
  if (V !== undefined && draft.cfg?.vocabSize !== undefined && draft.cfg.vocabSize !== V) throw new Error(`draft vocabulary (${draft.cfg.vocabSize}) differs from the target's (${V})`);
  const tLimit = verifyLimit(target);
  if (tLimit < 2) throw new Error('speculative decoding needs a target that can verify at least 2 tokens per pass (raise maxBatch)');

  // ---- prefill: everything but the last prompt token (that one is the first "pending" token), on BOTH models
  const body = promptIds.slice(0, n0 - 1);
  const reuse = Math.min(commonPrefix(target.tokens, promptIds), n0 - 1);
  await target.truncate(reuse);
  const dReuse = Math.min(commonPrefix(draft.tokens, body), body.length);
  await draft.truncate(dReuse);
  stats.promptTokens = n0;
  stats.cachedTokens = reuse;
  stats.generated = 0;
  stats.stopReason = 'length';
  const spec = stats.speculative = {
    k: kMax, cycles: 0, drafted: 0, accepted: 0, acceptRate: 0, tokensPerCycle: 0, draftMs: 0, verifyMs: 0, otherMs: 0,
    acceptedHist: new Array(kMax + 1).fill(0), draftCachedTokens: dReuse, draftPrefillMs: 0,
  };

  const t0 = performance.now();
  const tTodo = body.length - reuse; const dTodo = body.length - dReuse;
  const progress = opts.onPrefillProgress;
  let pre = await prefillTokens(target, body, { from: reuse, wantLogits: false, signal: opts.signal, onProgress: progress ? (d) => progress(d, tTodo + dTodo) : undefined });
  if (!pre.aborted) {
    const td = performance.now();
    pre = await prefillTokens(draft, body, { from: dReuse, wantLogits: false, signal: opts.signal, onProgress: progress ? (d) => progress(tTodo + d, tTodo + dTodo) : undefined });
    spec.draftPrefillMs = performance.now() - td;
  }
  stats.prefillMs = performance.now() - t0;
  if (pre.aborted) { stats.stopReason = 'aborted'; return; }
  stats.prefilled = n0 - reuse;

  // ---- decode
  const seq = promptIds.slice();
  /** history the sampler's repetition penalties see for the first `len` tokens of `full` */
  const needHist = sample !== argmax && sample.usesHistory !== false;
  const histOf = (full, len) => (!needHist ? EMPTY : opts.penalizePrompt === false ? full.slice(n0, len) : full.slice(0, len));
  const dense = greedy ? null : new Float64Array(V ?? 0);
  let generated = 0;
  const tDecode = performance.now();
  try {
    outer:
    while (generated < max) {
      if (opts.signal?.aborted) { stats.stopReason = 'aborted'; break; }
      const pending = seq[seq.length - 1];
      // how many tokens may be drafted this cycle (at most `remaining` tokens can be emitted: kk drafts + 1 correction)
      let kk = Math.min(kMax, max - generated - 1, target.ctx - seq.length, tLimit - 1, draft.ctx - seq.length + 1);
      kk = Math.max(0, kk);

      // -- draft: bring the draft model up to `seq` (pending token included) and let it propose kk tokens
      const drafted = []; const qd = [];
      if (kk > 0) {
        const td = performance.now();
        await draft.truncate(Math.min(commonPrefix(draft.tokens, seq), seq.length - 1));
        let dl = await feedLogits(draft, seq, draft.pos);
        draft.markRollback?.(); // hybrid draft: a snapshot here makes the next cycle's rollback a short replay
        for (let j = 0; j < kk; j++) {
          const h = histOf(seq.concat(drafted), seq.length + j);
          let x;
          if (greedy) x = sample(dl, h);
          else { const d = sample.distribution(dl, h); x = sample.draw(d); qd.push(d); }
          drafted.push(x);
          if (stop.has(x) || j === kk - 1) break; // nothing after a stop token matters; the last draft need not be fed
          dl = await draft.forward(x);
        }
        spec.draftMs += performance.now() - td;
        kk = drafted.length;
      }

      // -- target: ONE pass over [pending, x_1 .. x_kk]; row j = distribution after x_j (row 0: after the pending token)
      const tv = performance.now();
      const batch = [pending, ...drafted];
      target.markRollback?.(); // hybrid target: snapshot the recurrent state before the pass; a partial acceptance replays only the accepted tokens
      const rows = await forwardRows(target, batch);
      const tAfter = performance.now();
      spec.verifyMs += tAfter - tv;
      const full = seq.concat(drafted);
      const row = (j) => rows.subarray(j * (rows.length / batch.length), (j + 1) * (rows.length / batch.length));

      // -- accept / reject
      let a = 0; let tok = -1;
      for (let j = 0; j < kk; j++) {
        const h = histOf(full, seq.length + j);
        if (greedy) {
          const t = sample(row(j), h);
          if (t === drafted[j]) { a++; continue; }
          tok = t; break;
        }
        const pd = sample.distribution(row(j), h);
        const px = probOf(pd, drafted[j]); const qx = probOf(qd[j], drafted[j]);
        if (sample.rand() * qx < px) { a++; continue; } // accept with probability min(1, p/q)
        tok = drawResidual(sample, pd, qd[j], dense);
        break;
      }
      if (a === kk) { // every draft accepted (or none was drafted): the bonus / plain token comes from the last row
        tok = sample(row(kk), histOf(full, seq.length + kk));
      }
      spec.cycles++; spec.drafted += kk; spec.accepted += a; spec.acceptedHist[a]++;

      // -- roll the target back to what is really part of the text; emit
      const emit = [...drafted.slice(0, a), tok];
      let done = false;
      for (const e of emit) {
        if (opts.signal?.aborted) { stats.stopReason = 'aborted'; done = true; break; }
        if (stop.has(e)) { stats.stopReason = 'stop'; done = true; break; }
        const idx = seq.length; // number of tokens before e
        seq.push(e);
        yield e;
        generated++;
        stats.generated = generated;
        if (generated >= max) { done = true; break; }
        if (idx >= target.ctx) { stats.stopReason = 'context'; done = true; break; }
      }
      await target.truncate(Math.min(commonPrefix(target.tokens, seq), seq.length - 1));
      spec.otherMs += performance.now() - tAfter;
      if (done) break outer;
      await yieldToEventLoop();
    }
  } finally {
    // whatever happens (stop, abort, the consumer walking away): both caches keep only tokens that are really in the text
    await target.truncate(commonPrefix(target.tokens, seq));
    await draft.truncate(commonPrefix(draft.tokens, seq));
    stats.decodeMs = performance.now() - tDecode;
    spec.acceptRate = spec.drafted ? spec.accepted / spec.drafted : 0;
    spec.tokensPerCycle = spec.cycles ? generated / spec.cycles : 0;
    spec.effectiveTps = stats.decodeMs > 0 ? (generated * 1000) / stats.decodeMs : 0;
  }
}

// ---------------------------------------------------------------------------------------------- compatibility of the two models

/** Draft threads when the caller does not say: the draft is small and runs alternately with the target, so a few threads are enough (untuned guess). */
export function defaultDraftThreads(targetThreads) { return Math.max(1, Math.min(targetThreads, 4)); }

/** Default number of drafted tokens per verification pass: k + 1 = 4 tokens fill exactly one tile of the batched kernels (a 5th token needs a second pass over the weights tile, measured ~1.5x dearer on the dev VM), and on the target laptop a verified extra token costs far less than a decode step. See docs/DECISIONS.md. */
export const DEFAULT_DRAFT_K = 3;

/**
 * Fingerprint of a model's vocabulary: size, and a hash of every token string and token type in id order. Two models can
 * only share token ids (the draft proposes ids the target must understand) if these are equal.
 * @param {import('../gguf/reader.js').GGUFFile} g
 * @returns {{size: number, hash: string}}
 */
export function vocabFingerprint(g) {
  const tokens = g.get('tokenizer.ggml.tokens', []);
  const types = g.get('tokenizer.ggml.token_type', []);
  const h = createHash('sha256');
  for (let i = 0; i < tokens.length; i++) { h.update(String(tokens[i])); h.update('\0'); }
  h.update(`#${types.length}:`);
  for (let i = 0; i < types.length; i++) h.update(`${types[i]},`);
  return { size: tokens.length, hash: h.digest('hex') };
}

/**
 * Why `draft` cannot draft for `target`, or null when their vocabularies are compatible.
 * @param {import('../gguf/reader.js').GGUFFile} target @param {import('../gguf/reader.js').GGUFFile} draft
 * @returns {string|null}
 */
export function vocabMismatch(target, draft) {
  const a = vocabFingerprint(target); const b = vocabFingerprint(draft);
  if (a.size !== b.size) return `the draft model has a vocabulary of ${b.size} tokens but the target has ${a.size}: speculative decoding needs two models of the same family with the same tokenizer (e.g. Qwen3-0.6B for Qwen3-8B)`;
  if (a.hash !== b.hash) {
    const ta = target.get('tokenizer.ggml.tokens', []); const tb = draft.get('tokenizer.ggml.tokens', []);
    let i = 0; while (i < ta.length && ta[i] === tb[i]) i++;
    const detail = i < ta.length ? ` (first difference at token id ${i}: ${JSON.stringify(ta[i])} vs ${JSON.stringify(tb[i])})` : ' (the token types differ)';
    return `the draft model's tokenizer differs from the target's${detail}: speculative decoding needs the same vocabulary`;
  }
  const va = target.get(`${target.get('general.architecture')}.vocab_size`); const vb = draft.get(`${draft.get('general.architecture')}.vocab_size`);
  if (va !== undefined && vb !== undefined && va !== vb) return `the models disagree on the logits vocabulary size (${va} vs ${vb})`;
  return null;
}
