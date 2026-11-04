import test from 'node:test';
import assert from 'node:assert/strict';
import { createSampler, topKSelect } from '../src/sampling/sampler.js';
import { StopFilter } from '../src/sampling/stop.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

const logitsOf = (arr) => Float32Array.from(arr);

test('topKSelect matches a full sort', () => {
  const rand = mulberry32(3);
  for (const [n, k] of [[1000, 40], [50, 50], [10, 40], [5000, 1], [777, 100]]) {
    const l = Float32Array.from({ length: n }, () => gaussian(rand));
    const { ids, vals } = topKSelect(l, k);
    const want = [...l.keys()].sort((a, b) => l[b] - l[a] || a - b).slice(0, k);
    assert.deepEqual([...ids], want);
    assert.deepEqual([...vals], want.map((i) => l[i]));
  }
});

test('temperature 0 is greedy', () => {
  const s = createSampler({ temperature: 0, seed: 1 });
  assert.equal(s(logitsOf([0.1, 3, 2, -1]), []), 1);
});

test('seeded sampling is reproducible, different seeds differ', () => {
  const rand = mulberry32(1);
  const l = Float32Array.from({ length: 200 }, () => gaussian(rand) * 0.5);
  const run = (seed) => { const s = createSampler({ temperature: 1, topK: 0, topP: 1, seed }); return Array.from({ length: 30 }, () => s(l, [])); };
  assert.deepEqual(run(5), run(5));
  assert.notDeepEqual(run(5), run(6));
});

test('top-k restricts the support', () => {
  const l = logitsOf([5, 4, 3, 2, 1, 0]);
  const s = createSampler({ temperature: 5, topK: 2, topP: 1, seed: 1 });
  const seen = new Set();
  for (let i = 0; i < 500; i++) seen.add(s(l, []));
  assert.deepEqual([...seen].sort(), [0, 1]);
});

test('top-p keeps the smallest set reaching p; always at least one token', () => {
  const l = logitsOf([10, 0, 0, 0]); // token 0 has ~all the mass
  const s = createSampler({ temperature: 1, topK: 0, topP: 0.5, seed: 2 });
  for (let i = 0; i < 100; i++) assert.equal(s(l, []), 0);
  const flat = logitsOf([0, 0, 0, 0]);
  const s2 = createSampler({ temperature: 1, topK: 0, topP: 0.5, seed: 2 }); // keeps exactly 2 of 4 equal tokens
  const seen = new Set();
  for (let i = 0; i < 300; i++) seen.add(s2(flat, []));
  assert.deepEqual([...seen].sort(), [0, 1]);
});

test('min-p drops tokens far below the best', () => {
  const l = logitsOf([4, 3.9, 0, 0, 0]);
  const s = createSampler({ temperature: 1, topK: 0, topP: 1, minP: 0.2, seed: 4 });
  const seen = new Set();
  for (let i = 0; i < 300; i++) seen.add(s(l, []));
  assert.deepEqual([...seen].sort(), [0, 1]);
});

test('empirical distribution matches softmax(logits / T)', () => {
  const l = logitsOf([1, 0.5, 0, -0.5]);
  const T = 0.8;
  const w = [...l].map((v) => Math.exp(v / T));
  const z = w.reduce((a, b) => a + b);
  const s = createSampler({ temperature: T, topK: 0, topP: 1, seed: 11 });
  const counts = [0, 0, 0, 0];
  const N = 40000;
  for (let i = 0; i < N; i++) counts[s(l, [])]++;
  counts.forEach((c, i) => assert.ok(Math.abs(c / N - w[i] / z) < 0.01, `token ${i}: ${c / N} vs ${w[i] / z}`));
});

test('repetition / presence / frequency penalties lower repeated tokens and leave logits untouched', () => {
  const l = logitsOf([2, 1.9, 0, 0]);
  const before = [...l];
  const rep = createSampler({ temperature: 0, repeatPenalty: 1.5 });
  assert.equal(rep(l, [0]), 1); // 2/1.5 < 1.9
  assert.deepEqual([...l], before);
  const neg = createSampler({ temperature: 0, repeatPenalty: 2 });
  assert.equal(neg(logitsOf([-1, -1.5]), [0]), 1); // negative logits are multiplied
  const pres = createSampler({ temperature: 0, presencePenalty: 0.5 });
  assert.equal(pres(l, [0, 0, 0]), 1);
  const freq = createSampler({ temperature: 0, frequencyPenalty: 0.05 });
  assert.equal(freq(l, [0]), 0); // 2 - 0.05 > 1.9
  assert.equal(freq(l, [0, 0, 0]), 1); // 2 - 0.15 < 1.9
  // window
  const win = createSampler({ temperature: 0, repeatPenalty: 1.5, repeatLastN: 1 });
  assert.equal(win(l, [0, 2]), 0);
});

test('StopFilter: holds back partial matches, cuts at the stop, flushes otherwise', () => {
  const f = new StopFilter(['STOP', '###']);
  assert.equal(f.push('hello '), 'hello ');
  assert.equal(f.push('ST'), '');
  assert.equal(f.push('OX'), 'STOX');
  assert.equal(f.push('a#'), 'a');
  assert.equal(f.push('#'), '');
  assert.equal(f.push('#tail'), '');
  assert.equal(f.stopped, true);
  assert.equal(f.push('more'), '');

  const g = new StopFilter(['END']);
  assert.equal(g.push('fooEN'), 'foo');
  assert.equal(g.flush(), 'EN');
  const h = new StopFilter(['END']);
  assert.equal(h.push('xxENDyy'), 'xx');
  assert.equal(h.stopped, true);
  assert.equal(new StopFilter([]).push('abc'), 'abc');
  // stop split across many tiny pushes
  const k = new StopFilter(['<|x|>']);
  const out = [...'ab<|x|>cd'].map((c) => k.push(c)).join('');
  assert.equal(out, 'ab');
});
