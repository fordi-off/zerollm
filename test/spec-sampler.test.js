// The sampler's split into distribution() + draw() (needed by speculative sampling) must not change what it does.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createSampler, topKSelect } from '../src/sampling/sampler.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

/** The sampler exactly as it was before the refactor (copied from the previous revision): the oracle for bit-identical seeded output. */
function legacySampler(o = {}) {
  const temperature = o.temperature ?? 0.8;
  const topK = o.topK ?? 40;
  const topP = o.topP ?? 0.95;
  const minP = o.minP ?? 0;
  const repeatPenalty = o.repeatPenalty ?? 1;
  const repeatLastN = o.repeatLastN ?? 64;
  const presence = o.presencePenalty ?? 0;
  const frequency = o.frequencyPenalty ?? 0;
  const rand = mulberry32(o.seed);
  const penalized = repeatPenalty !== 1 || presence !== 0 || frequency !== 0;
  const argmax = (l) => { let best = 0; let bv = -Infinity; for (let i = 0; i < l.length; i++) if (l[i] > bv) { bv = l[i]; best = i; } return best; };
  return function sample(logits, history) {
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
    if (temperature <= 0) { const id = argmax(logits); restore(); return id; }
    let ids; let vals;
    if (topK > 0 && topK < logits.length) {
      ({ ids, vals } = topKSelect(logits, topK));
    } else {
      const idx = Array.from({ length: logits.length }, (_, i) => i).sort((a, b) => logits[b] - logits[a] || a - b);
      ids = Int32Array.from(idx); vals = Float64Array.from(idx, (i) => logits[i]);
    }
    restore();
    const n = ids.length;
    const probs = new Float64Array(n);
    const max = vals[0];
    let sum = 0;
    for (let i = 0; i < n; i++) { probs[i] = Math.exp((vals[i] - max) / temperature); sum += probs[i]; }
    for (let i = 0; i < n; i++) probs[i] /= sum;
    let keep = n;
    if (minP > 0) { const thr = minP * probs[0]; keep = 1; while (keep < n && probs[keep] >= thr) keep++; }
    if (topP < 1) {
      let c = 0; let i = 0;
      while (i < keep) { c += probs[i]; i++; if (c >= topP) break; }
      keep = Math.max(1, i);
    }
    let total = 0;
    for (let i = 0; i < keep; i++) total += probs[i];
    let r = rand() * total;
    for (let i = 0; i < keep; i++) { r -= probs[i]; if (r <= 0) return ids[i]; }
    return ids[keep - 1];
  };
}

/** probability of `id` under a sparse distribution */
const probOf = (d, id) => { const i = d.ids.indexOf(id); return i < 0 ? 0 : d.probs[i] / d.total; };

const CONFIGS = [
  {}, { temperature: 0 }, { temperature: 1, topK: 0, topP: 1 }, { temperature: 0.7, topK: 20, topP: 0.9 }, { temperature: 1.3, topK: 5, topP: 1, minP: 0.05 },
  { temperature: 0.6, topK: 0, topP: 0.8, minP: 0.02 }, { temperature: 0.8, repeatPenalty: 1.3, repeatLastN: 16 },
  { temperature: 0.9, presencePenalty: 0.5, frequencyPenalty: 0.3, topK: 30 }, { temperature: 0, repeatPenalty: 1.5 },
  { temperature: 1, topK: 1 }, { temperature: 0.5, topK: 3, topP: 0.3, minP: 0.5, repeatPenalty: 1.1, presencePenalty: 0.1 },
];

test('seeded sampling is bit-identical to the pre-refactor sampler (every option combination, 300 draws each, with history)', () => {
  for (const [ci, cfg] of CONFIGS.entries()) {
    const seed = 100 + ci;
    const now = createSampler({ ...cfg, seed });
    const old = legacySampler({ ...cfg, seed });
    const rand = mulberry32(seed * 7);
    const V = ci % 2 ? 300 : 97;
    const history = [];
    for (let i = 0; i < 300; i++) {
      const l = Float32Array.from({ length: V }, () => gaussian(rand) * (1 + (i % 3)));
      const l2 = Float32Array.from(l);
      const a = now(l, history.slice()); const b = old(l2, history.slice());
      assert.equal(a, b, `config ${ci} draw ${i}`);
      assert.deepEqual([...l], [...l2], 'caller buffer untouched / restored identically');
      history.push(a);
    }
  }
});

test('sample(l, h) is exactly draw(distribution(l, h)); distribution() consumes no random number; the logits buffer is left untouched', () => {
  for (const [ci, cfg] of CONFIGS.entries()) {
    const a = createSampler({ ...cfg, seed: 5 });
    const b = createSampler({ ...cfg, seed: 5 });
    const rand = mulberry32(ci + 1);
    for (let i = 0; i < 60; i++) {
      const l = Float32Array.from({ length: 150 }, () => gaussian(rand));
      const copy = Float32Array.from(l);
      const hist = Array.from({ length: i % 9 }, () => Math.floor(rand() * 150));
      b.distribution(l, hist); b.distribution(l, hist); // extra calls must not disturb the random stream
      assert.deepEqual([...l], [...copy]);
      assert.equal(b.draw(b.distribution(l, hist)), a(l, hist), `config ${ci} step ${i}`);
    }
  }
});

test('distribution(): sorted support, normalisation, truncation by top-k / min-p / top-p, penalties; greedy is a point mass', () => {
  const l = Float32Array.from([2, 5, 1, 4, 3, 0, 4.5, -1]);
  let d = createSampler({ temperature: 1, topK: 0, topP: 1 }).distribution(l, []);
  assert.deepEqual([...d.ids], [1, 6, 3, 4, 0, 2, 5, 7]);
  assert.ok(Math.abs(d.probs.reduce((s, v) => s + v, 0) - d.total) < 1e-12);
  const z = [...l].reduce((s, v) => s + Math.exp(v - 5), 0);
  assert.ok(Math.abs(probOf(d, 3) - Math.exp(4 - 5) / z) < 1e-12);
  assert.equal(probOf(d, 99), 0);

  d = createSampler({ temperature: 1, topK: 3, topP: 1 }).distribution(l, []);
  assert.deepEqual([...d.ids], [1, 6, 3]);
  assert.ok(Math.abs(d.probs.reduce((s, v) => s + v, 0) / d.total - 1) < 1e-12);
  assert.equal(probOf(d, 4), 0);
  assert.ok(Math.abs(probOf(d, 1) - 1 / (1 + Math.exp(-0.5) + Math.exp(-1)) / 1) < 1e-12, 'renormalised over the kept candidates');

  d = createSampler({ temperature: 1, topK: 0, topP: 1, minP: 0.5 }).distribution(l, []); // keeps p >= 0.5 * p_max: logits 5, 4.5 (e^-0.5 = 0.607)
  assert.deepEqual([...d.ids], [1, 6]);
  d = createSampler({ temperature: 1, topK: 0, topP: 0.5 }).distribution(l, []); // top token alone has p ~ 0.42 < 0.5 -> two tokens
  assert.deepEqual([...d.ids], [1, 6]);

  d = createSampler({ temperature: 0 }).distribution(l, []);
  assert.deepEqual([[...d.ids], [...d.probs], d.total, d.greedy], [[1], [1], 1, true]);
  // penalties move the greedy choice: token 1 (5) is penalised below token 6 (4.5)
  d = createSampler({ temperature: 0, repeatPenalty: 2 }).distribution(l, [1]);
  assert.equal(d.ids[0], 6);
  const hist = createSampler({ temperature: 1, topK: 0, topP: 1, presencePenalty: 10 }).distribution(l, [1, 1, 6]);
  assert.deepEqual([...hist.ids].slice(0, 2), [3, 4], 'presence penalty applies once per distinct token in the history');
  assert.equal(createSampler({}).greedy, false);
  assert.equal(createSampler({ temperature: 0 }).greedy, true);
  assert.equal(createSampler({ repeatPenalty: 1.2 }).usesHistory, true);
  assert.equal(createSampler({}).usesHistory, false);
});
