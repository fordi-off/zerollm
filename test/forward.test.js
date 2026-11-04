import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { naiveLogits } from './naive-forward.js';
import { RefModel } from '../src/model/forward.js';
import { generate, argmax } from '../src/model/generate.js';
import { Act, matvec, dequantRow, ropeTables, applyRope, softmax, rmsnorm } from '../src/compute/ref-kernels.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

const cosine = (a, b) => {
  let d = 0; let na = 0; let nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
};
const maxAbsDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

function randomWeight(type, rows, cols, seed) {
  const rand = mulberry32(seed);
  const f = new Float32Array(rows * cols).map(() => gaussian(rand) * 0.3);
  let data;
  if (type === GGML_TYPE.Q8_0) data = quantizeQ8_0(f);
  else if (type === GGML_TYPE.Q4_0) data = quantizeQ4_0(f);
  else if (type === GGML_TYPE.F32) data = new Uint8Array(f.buffer);
  else throw new Error('type');
  return { type, rows, cols, data };
}

test('matvec equals dequantize-then-float-matmul (up to activation quantization)', () => {
  const rand = mulberry32(5);
  const cols = 96; const rows = 17;
  const x = new Float32Array(cols).map(() => gaussian(rand));
  for (const [type, tol] of [[GGML_TYPE.F32, 1e-5], [GGML_TYPE.Q8_0, 0.02], [GGML_TYPE.Q4_0, 0.02]]) {
    const w = randomWeight(type, rows, cols, 11);
    const y = matvec(w, new Act(x), new Float32Array(rows));
    const row = new Float32Array(cols);
    let normY = 0;
    const want = [];
    for (let r = 0; r < rows; r++) {
      dequantRow(w, r, row);
      want.push(row.reduce((a, v, i) => a + v * x[i], 0));
      normY += want[r] ** 2;
    }
    const err = Math.sqrt(y.reduce((a, v, r) => a + (v - want[r]) ** 2, 0) / normY);
    assert.ok(err < tol, `type ${type}: relative error ${err}`);
  }
});

test('matvec row range', () => {
  const w = randomWeight(GGML_TYPE.Q8_0, 8, 64, 3);
  const x = new Float32Array(64).map((_, i) => Math.cos(i));
  const full = matvec(w, new Act(x), new Float32Array(8));
  const part = new Float32Array(8);
  matvec(w, new Act(x), part, 2, 5);
  assert.deepEqual([...part.subarray(2, 5)], [...full.subarray(2, 5)]);
  assert.equal(part[0], 0); assert.equal(part[7], 0);
});

test('RoPE: preserves norm; q.k depends only on relative position (both variants)', () => {
  for (const type of ['neox', 'norm']) {
    const cfg = { nRot: 16, ropeBase: 10000 };
    const tabs = ropeTables(cfg, 64);
    const rand = mulberry32(9);
    const q0 = new Float32Array(16).map(() => gaussian(rand));
    const k0 = new Float32Array(16).map(() => gaussian(rand));
    const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
    const at = (v, p) => { const c = v.slice(); applyRope(c, 1, 16, 16, type, tabs, p); return c; };
    assert.ok(Math.abs(dot(at(q0, 7), at(q0, 7)) - dot(q0, q0)) < 1e-4);
    assert.ok(Math.abs(dot(at(q0, 10), at(k0, 6)) - dot(at(q0, 14), at(k0, 10))) < 1e-4);
    assert.ok(Math.abs(dot(at(q0, 10), at(k0, 6)) - dot(at(q0, 11), at(k0, 6))) > 1e-3); // really position dependent
    assert.deepEqual([...at(q0, 0)].map((v) => +v.toFixed(6)), [...q0].map((v) => +v.toFixed(6)));
  }
  // pairs differ between variants
  const tabs = ropeTables({ nRot: 4, ropeBase: 10000 }, 8);
  const a = Float32Array.from([1, 0, 0, 0]); const b = a.slice();
  applyRope(a, 1, 4, 4, 'neox', tabs, 1); applyRope(b, 1, 4, 4, 'norm', tabs, 1);
  assert.ok(a[2] !== 0 && a[1] === 0, 'neox rotates (0,2)');
  assert.ok(b[1] !== 0 && b[2] === 0, 'norm rotates (0,1)');
});

test('partial RoPE leaves trailing dims untouched', () => {
  const tabs = ropeTables({ nRot: 4, ropeBase: 10000 }, 8);
  const v = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
  applyRope(v, 1, 8, 4, 'neox', tabs, 3);
  assert.deepEqual([...v.subarray(4)], [5, 6, 7, 8]);
});

test('softmax / rmsnorm basics', () => {
  const x = Float32Array.from([1000, 1001, 1002, 0]);
  softmax(x, 3);
  assert.ok(Math.abs(x[0] + x[1] + x[2] - 1) < 1e-6);
  assert.equal(x[3], 0);
  const o = new Float32Array(4);
  rmsnorm(o, Float32Array.from([3, 4, 0, 0]), Float32Array.from([1, 1, 1, 1]), 0);
  assert.ok(Math.abs(o[0] - 3 / Math.sqrt(25 / 4)) < 1e-6);
});

for (const [name, opts] of [
  ['qwen3 (QK-norm, head_dim != hidden/heads, tied)', { arch: 'qwen3' }],
  ['qwen2 (qkv bias, tied)', { arch: 'qwen2' }],
  ['llama (adjacent-pair RoPE, untied output)', { arch: 'llama' }],
  ['llama + rope_freqs.weight', { arch: 'llama', ropeFreqs: true }],
  ['qwen3 with MHA (kv heads == heads)', { arch: 'qwen3', nHeadKv: 4 }],
  ['qwen3 with 1 kv head', { arch: 'qwen3', nHeadKv: 1 }],
]) {
  test(`forward == naive oracle, f32: ${name}`, () => {
    const { g } = tinyFile({ ...opts, quant: 'f32', seed: 21 });
    const toks = [3, 70, 5, 200, 12, 9, 255, 1];
    const want = naiveLogits(g, toks);
    const m = new RefModel(g, { ctx: 64 });
    toks.forEach((t, i) => {
      const got = m.forward(t);
      assert.ok(maxAbsDiff(got, want[i]) < 2e-4 * Math.max(1, ...want[i].map(Math.abs)), `pos ${i}: diff ${maxAbsDiff(got, want[i])}`);
      assert.equal(argmax(got), want[i].indexOf(Math.max(...want[i])));
    });
  });
}

test('quantized models track the f32 model with the same underlying weights', () => {
  const toks = [3, 70, 5, 200, 12];
  const run = (quant) => {
    const { g } = tinyFile({ arch: 'qwen3', quant, seed: 33 });
    const m = new RefModel(g, { ctx: 32 });
    return toks.map((t) => Float32Array.from(m.forward(t)));
  };
  const f = run('f32'); const q8 = run('q8_0'); const q4 = run('q4_0');
  toks.forEach((_, i) => {
    assert.ok(cosine(f[i], q8[i]) > 0.995, `q8 cos ${cosine(f[i], q8[i])}`);
    assert.ok(cosine(f[i], q4[i]) > 0.85, `q4 cos ${cosine(f[i], q4[i])}`);
  });
  // q8 vs naive (dequantized weights, unquantized activations) is also close
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 33 });
  const want = naiveLogits(g, toks);
  assert.ok(cosine(q8[4], want[4]) > 0.999);
});

test('zero-layer model: logits = Wout * rmsnorm(embed[token]) (hand computation)', () => {
  const { g } = tinyFile({ arch: 'llama', nLayer: 0, quant: 'f32', seed: 2 });
  const m = new RefModel(g, { ctx: 8 });
  const logits = Float32Array.from(m.forward(17));
  const want = naiveLogits(g, [17])[0];
  assert.ok(maxAbsDiff(logits, want) < 1e-4);
});

test('KV cache: truncate + replay reproduces logits; reset starts fresh', () => {
  const { g } = tinyFile({ arch: 'qwen2', quant: 'q8_0', seed: 8 });
  const m = new RefModel(g, { ctx: 32 });
  const seq = [4, 5, 6, 7];
  const first = seq.map((t) => Float32Array.from(m.forward(t)));
  m.truncate(2);
  assert.equal(m.pos, 2);
  const again = seq.slice(2).map((t) => Float32Array.from(m.forward(t)));
  assert.deepEqual([...again[0]], [...first[2]]);
  assert.deepEqual([...again[1]], [...first[3]]);
  m.reset();
  assert.deepEqual([...m.forward(4)], [...first[0]]);
  assert.throws(() => { for (let i = 0; i < 40; i++) m.forward(1); }, /context full/);
});

test('prefill without logits equals per-token forward', () => {
  const { g } = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 4 });
  const a = new RefModel(g, { ctx: 32 }); const b = new RefModel(g, { ctx: 32 });
  const toks = [10, 20, 30, 40];
  const la = Float32Array.from(a.prefill(toks));
  let lb;
  for (const t of toks) lb = b.forward(t);
  assert.deepEqual([...la], [...lb]);
});

test('greedy generation is deterministic; honours maxTokens, stop ids, abort', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 6 });
  const collect = async (model, prompt, opts) => { const out = []; for await (const id of generate(model, prompt, opts)) out.push(id); return out; };
  const m = new RefModel(g, { ctx: 64 });
  const a = await collect(m, [1, 2, 3], { maxTokens: 12 });
  m.reset();
  const b = await collect(m, [1, 2, 3], { maxTokens: 12 });
  assert.equal(a.length, 12);
  assert.deepEqual(a, b);
  m.reset();
  const stats = {};
  const c = await collect(m, [1, 2, 3], { maxTokens: 12, stopIds: new Set([a[4]]), stats });
  assert.deepEqual(c, a.slice(0, a.indexOf(a[4])));
  assert.equal(stats.stopReason, 'stop');
  m.reset();
  const signal = { aborted: false };
  const d = [];
  for await (const id of generate(m, [1, 2, 3], { maxTokens: 50, signal, stats })) { d.push(id); if (d.length === 3) signal.aborted = true; }
  assert.equal(d.length, 3);
  assert.equal(stats.stopReason, 'aborted');
  await assert.rejects(async () => collect(m, [], {}), /empty prompt/);
});
