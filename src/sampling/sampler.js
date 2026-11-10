// Token sampler: repetition/presence/frequency penalties, temperature, top-k, min-p, top-p, seeded PRNG.
import { mulberry32 } from '../util/rng.js';

/**
 * @typedef {Object} SamplerOptions
 * @property {number} [temperature]       <= 0 means greedy
 * @property {number} [topK]              0 disables
 * @property {number} [topP]              1 disables
 * @property {number} [minP]              0 disables
 * @property {number} [repeatPenalty]     1 disables (llama.cpp style: divide positive / multiply negative logits)
 * @property {number} [repeatLastN]       window of recent tokens the penalties look at
 * @property {number} [presencePenalty]
 * @property {number} [frequencyPenalty]
 * @property {number} [seed]
 */

/**
 * Indices and values of the k largest logits, sorted descending. O(V log k) via a size-k min-heap.
 * @param {Float32Array} logits @param {number} k
 */
export function topKSelect(logits, k) {
  const n = logits.length;
  k = Math.min(k, n);
  const ids = new Int32Array(k); const vals = new Float64Array(k);
  let size = 0;
  const swap = (a, b) => { const ti = ids[a]; ids[a] = ids[b]; ids[b] = ti; const tv = vals[a]; vals[a] = vals[b]; vals[b] = tv; };
  const up = (i) => { while (i > 0) { const p = (i - 1) >> 1; if (vals[i] >= vals[p]) break; swap(i, p); i = p; } };
  const down = (i) => {
    for (;;) {
      const l = 2 * i + 1; const r = l + 1; let m = i;
      if (l < size && vals[l] < vals[m]) m = l;
      if (r < size && vals[r] < vals[m]) m = r;
      if (m === i) break;
      swap(i, m); i = m;
    }
  };
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (size < k) { ids[size] = i; vals[size] = v; size++; up(size - 1); } else if (v > vals[0]) { ids[0] = i; vals[0] = v; down(0); }
  }
  const order = Array.from({ length: size }, (_, i) => i).sort((a, b) => vals[b] - vals[a] || ids[a] - ids[b]);
  return { ids: Int32Array.from(order, (i) => ids[i]), vals: Float64Array.from(order, (i) => vals[i]) };
}

/** @param {Float32Array} logits */
function argmax(logits) {
  let best = 0; let bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}

/**
 * The distribution a sampler draws from, after every transformation (penalties, temperature, top-k, min-p, top-p).
 * Sparse: `ids[i]` has probability `probs[i] / total` (candidates sorted by descending probability; every token that is
 * not listed has probability 0). `probs` is deliberately NOT normalised: drawing uses `rand() * total`, exactly as the
 * sampler always did, so seeded sampling is bit-identical to what it was before this type existed.
 * @typedef {Object} TokenDist
 * @property {Int32Array} ids
 * @property {Float64Array} probs
 * @property {number} total  sum of probs
 * @property {boolean} greedy  point mass on ids[0] (temperature <= 0); drawing consumes no random number
 */

/**
 * A sampler: `sample(logits, history)` returns a token id. It also exposes its two halves, used by speculative decoding:
 * `distribution(logits, history)` (the final probability vector, no random number consumed) and `draw(dist)` (one random
 * number -> token). `sample(l, h)` is exactly `draw(distribution(l, h))`.
 * @typedef {((logits: Float32Array, history: number[]) => number) & {
 *   distribution: (logits: Float32Array, history: number[]) => TokenDist,
 *   draw: (dist: TokenDist) => number,
 *   rand: () => number,
 *   greedy: boolean,
 *   usesHistory: boolean,  // false: the token history never changes the result (no penalties)
 * }} Sampler
 */

/**
 * @param {SamplerOptions} [o]
 * @returns {Sampler}
 */
export function createSampler(o = {}) {
  const temperature = o.temperature ?? 0.8;
  const topK = o.topK ?? 40;
  const topP = o.topP ?? 0.95;
  const minP = o.minP ?? 0;
  const repeatPenalty = o.repeatPenalty ?? 1;
  const repeatLastN = o.repeatLastN ?? 64;
  const presence = o.presencePenalty ?? 0;
  const frequency = o.frequencyPenalty ?? 0;
  const rand = mulberry32(o.seed ?? ((Date.now() ^ (Math.random() * 2 ** 32)) >>> 0));
  const penalized = repeatPenalty !== 1 || presence !== 0 || frequency !== 0;

  /** @returns {TokenDist} */
  function distribution(logits, history) {
    // --- penalties (applied in place, restored before returning so callers' buffers stay untouched)
    let saved = null;
    if (penalized && history.length) {
      const counts = new Map();
      for (let i = Math.max(0, history.length - repeatLastN); i < history.length; i++) counts.set(history[i], (counts.get(history[i]) ?? 0) + 1);
      saved = [];
      for (const [t, c] of counts) {
        const l = logits[t];
        saved.push(t, l);
        let v = repeatPenalty !== 1 ? (l > 0 ? l / repeatPenalty : l * repeatPenalty) : l;
        v -= c * frequency + presence;
        logits[t] = v;
      }
    }
    const restore = () => { if (saved) for (let i = 0; i < saved.length; i += 2) logits[saved[i]] = saved[i + 1]; };

    if (temperature <= 0) { const id = argmax(logits); restore(); return { ids: Int32Array.of(id), probs: Float64Array.of(1), total: 1, greedy: true }; }

    // --- candidates sorted by logit, descending
    let ids; let vals;
    if (topK > 0 && topK < logits.length) {
      ({ ids, vals } = topKSelect(logits, topK));
    } else {
      const idx = Array.from({ length: logits.length }, (_, i) => i).sort((a, b) => logits[b] - logits[a] || a - b);
      ids = Int32Array.from(idx); vals = Float64Array.from(idx, (i) => logits[i]);
    }
    restore();

    // --- softmax with temperature
    const n = ids.length;
    const probs = new Float64Array(n);
    const max = vals[0];
    let sum = 0;
    for (let i = 0; i < n; i++) { probs[i] = Math.exp((vals[i] - max) / temperature); sum += probs[i]; }
    for (let i = 0; i < n; i++) probs[i] /= sum;

    // --- min-p, then top-p (candidates are sorted so both are prefixes)
    let keep = n;
    if (minP > 0) { const thr = minP * probs[0]; keep = 1; while (keep < n && probs[keep] >= thr) keep++; }
    if (topP < 1) {
      let c = 0; let i = 0;
      while (i < keep) { c += probs[i]; i++; if (c >= topP) break; }
      keep = Math.max(1, i);
    }
    let total = 0;
    for (let i = 0; i < keep; i++) total += probs[i];
    return { ids: ids.subarray(0, keep), probs: probs.subarray(0, keep), total, greedy: false };
  }

  /** @param {TokenDist} d */
  function draw(d) {
    if (d.greedy) return d.ids[0];
    const keep = d.ids.length;
    let r = rand() * d.total;
    for (let i = 0; i < keep; i++) { r -= d.probs[i]; if (r <= 0) return d.ids[i]; }
    return d.ids[keep - 1];
  }

  const sample = (logits, history) => draw(distribution(logits, history));
  return Object.assign(sample, { distribution, draw, rand, greedy: temperature <= 0, usesHistory: penalized });
}
