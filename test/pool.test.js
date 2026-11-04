import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tinyFile } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { generate } from '../src/model/generate.js';
import { splitRows, splitRange } from '../src/compute/layout.js';

const here = path.dirname(fileURLToPath(import.meta.url));

test('splitRows: contiguous, complete, 4-aligned, balanced', () => {
  for (const rows of [0, 1, 3, 4, 64, 100, 1024, 2048, 3072, 151936, 12288]) {
    for (const n of [1, 2, 3, 4, 5, 6, 8, 12]) {
      let prev = 0;
      const sizes = [];
      for (let w = 0; w < n; w++) {
        const [a, b] = splitRows(rows, n, w);
        assert.equal(a, prev, `rows=${rows} n=${n} w=${w}: gap/overlap`);
        assert.ok(b >= a);
        if (w < n - 1) assert.ok(b % 4 === 0 || b === rows, `boundary ${b} not aligned (rows=${rows})`);
        sizes.push(b - a);
        prev = b;
      }
      assert.equal(prev, rows);
      if (rows >= 64 * n) assert.ok(Math.max(...sizes) - Math.min(...sizes) <= 8 + (rows % 4), `imbalance ${sizes}`);
    }
  }
});

test('splitRange: even split of heads incl. more workers than items', () => {
  for (const count of [0, 1, 2, 8, 16, 32]) {
    for (const n of [1, 3, 6, 12]) {
      let prev = 0;
      for (let w = 0; w < n; w++) { const [a, b] = splitRange(count, n, w); assert.equal(a, prev); prev = b; }
      assert.equal(prev, count);
    }
  }
  assert.deepEqual([0, 1, 2, 3, 4, 5].map((w) => { const [a, b] = splitRange(8, 6, w); return b - a; }), [1, 1, 2, 1, 1, 2]);
});

for (const spin of [0, 2000]) {
  test(`barrier: 4 threads x 3000 rounds, no lost wakeups, plain writes visible (spin=${spin})`, async () => {
    const n = 4; const rounds = 3000;
    const shared = { ctrl: new SharedArrayBuffer(48 * 4), slots: new SharedArrayBuffer(n * 4), errors: new SharedArrayBuffer(4) };
    const workers = Array.from({ length: n }, (_, id) => new Worker(path.join(here, '..', 'test-fixtures', 'barrier-worker.js'), { workerData: { id, n, rounds, spin, ...shared } }));
    await Promise.race([
      Promise.all(workers.map((w) => new Promise((res, rej) => { w.on('message', res); w.on('error', rej); }))),
      new Promise((_, rej) => setTimeout(() => rej(new Error('barrier deadlocked')), 30000)),
    ]).finally(() => workers.forEach((w) => w.terminate()));
    assert.equal(Atomics.load(new Int32Array(shared.errors), 0), 0);
  });
}

const CASES = [
  ['qwen3 (QK-norm, head_dim 32 != 16)', { arch: 'qwen3' }],
  ['qwen2 (bias)', { arch: 'qwen2' }],
  ['llama untied + adjacent RoPE', { arch: 'llama' }],
  ['llama + rope_freqs', { arch: 'llama', ropeFreqs: true }],
  ['qwen3 1 KV head (fewer heads than workers)', { arch: 'qwen3', nHeadKv: 1 }],
];
for (const [name, opts] of CASES) {
  for (const quant of ['q8_0', 'q4_0', 'f32']) {
    if (quant !== 'q8_0' && opts.nHeadKv === 1) continue;
    test(`pool logits are bit-identical to the reference: ${name}, ${quant}, 1-5 workers`, async () => {
      const { g } = tinyFile({ ...opts, quant, seed: 17 });
      const ref = new RefModel(g, { ctx: 48 });
      const toks = [3, 70, 5, 200, 12, 9, 255, 1, 8];
      const want = toks.map((t) => Float32Array.from(ref.forward(t)));
      for (const workers of [1, 2, 3, 5]) {
        const pool = await PoolModel.create(g, { ctx: 48, workers, kernels: 'js' });
        try {
          for (let i = 0; i < toks.length; i++) {
            const got = await pool.forward(toks[i]);
            assert.deepEqual([...got], [...want[i]], `workers=${workers} pos=${i}`);
          }
        } finally { await pool.close(); }
      }
    });
  }
}

test('pool: weight slices partition the matrices; truncate/reset replay; prefill without logits', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 2 });
  const matBytes = g.tensors.filter((t) => t.dims.length === 2).reduce((s, t) => s + t.nBytes, 0);
  const pool = await PoolModel.create(g, { ctx: 32, workers: 4, kernels: 'js' });
  try {
    assert.equal(pool.sliceBytes.reduce((a, b) => a + b, 0), matBytes, 'slices cover every matrix byte exactly once');
    assert.ok(Math.max(...pool.sliceBytes) < matBytes / 4 * 1.3, `slices are balanced: ${pool.sliceBytes}`);
    const seq = [4, 5, 6, 7];
    assert.equal(await pool.forward(seq[0], { logits: false }), null);
    const a = []; for (const t of seq.slice(1)) a.push(Float32Array.from(await pool.forward(t)));
    assert.equal(pool.pos, 4);
    pool.truncate(2);
    assert.equal(pool.pos, 2);
    const b = []; for (const t of seq.slice(2)) b.push(Float32Array.from(await pool.forward(t)));
    assert.deepEqual([...b[0]], [...a[1]]);
    assert.deepEqual([...b[1]], [...a[2]]);
    pool.reset();
    assert.equal(pool.pos, 0);
    const ref = new RefModel(g, { ctx: 32 });
    assert.deepEqual([...await pool.forward(4)], [...ref.forward(4)]);
  } finally { await pool.close(); }
});

test('generate() over the pool equals generate() over the reference (with prefix reuse)', async () => {
  const { g } = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 23 });
  const collect = async (m, ids, stats = {}) => { const o = []; for await (const t of generate(m, ids, { maxTokens: 10, stats })) o.push(t); return o; };
  const ref = new RefModel(g, { ctx: 96 });
  const pool = await PoolModel.create(g, { ctx: 96, workers: 3, kernels: 'js' });
  try {
    const p1 = [5, 6, 7, 8, 9];
    const r1 = await collect(ref, p1); const q1 = await collect(pool, p1);
    assert.deepEqual(q1, r1);
    const p2 = [...p1, ...r1, 40, 41];
    const st = {};
    assert.deepEqual(await collect(pool, p2, st), await collect(ref, p2));
    assert.ok(st.cachedTokens >= p1.length);
  } finally { await pool.close(); }
});

test('pool: argument errors, use after close, idempotent close', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const pool = await PoolModel.create(g, { ctx: 8, workers: 2 });
  await assert.rejects(pool.forward(-1), /out of range/);
  await assert.rejects(pool.forward(10 ** 6), /out of range/);
  for (let i = 0; i < 8; i++) await pool.forward(1, { logits: false });
  await assert.rejects(pool.forward(1), /context full/);
  await pool.close();
  await pool.close();
  await assert.rejects(pool.forward(1), /closed/);
});

test('pool: a worker that cannot load fails creation promptly with its message', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const broken = Object.assign(Object.create(Object.getPrototypeOf(g)), g, { path: path.join(here, 'does-not-exist.gguf') });
  const t0 = Date.now();
  await assert.rejects(PoolModel.create(broken, { ctx: 8, workers: 3 }), /ENOENT|worker \d failed/);
  assert.ok(Date.now() - t0 < 10000);
});

test('pool: killing a worker mid-session makes forward() reject instead of hanging', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const pool = await PoolModel.create(g, { ctx: 16, workers: 3 });
  await pool.forward(1);
  await pool.workers[1].terminate();
  const t0 = Date.now();
  await assert.rejects(pool.forward(2), /worker|exited/);
  assert.ok(Date.now() - t0 < 5000);
  await pool.close();
});

test('forgetting close() does not keep the process alive', () => {
  const r = spawnSync(process.execPath, [path.join(here, '..', 'test-fixtures', 'leak-pool.mjs')], { encoding: 'utf8', timeout: 20000 });
  assert.equal(r.error, undefined, String(r.error));
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /main finished without close/);
});

// ------------------------------------------------------------------------------------------------ SIMD kernels in the pool
for (const [arch, quant] of [['qwen3', 'q8_0'], ['qwen2', 'q8_0'], ['llama', 'q4_0'], ['qwen3', 'q4_0']]) {
  test(`pool with WASM SIMD kernels tracks the reference (rounding-level differences only): ${arch} ${quant}, 1-4 workers`, async () => {
    const { g } = tinyFile({ arch, quant, seed: 29 });
    const ref = new RefModel(g, { ctx: 48 });
    const toks = [3, 70, 5, 200, 12, 9, 255, 1];
    const want = toks.map((t) => Float32Array.from(ref.forward(t)));
    let first = null;
    for (const workers of [1, 2, 3, 4]) {
      const pool = await PoolModel.create(g, { ctx: 48, workers, kernels: 'wasm' });
      assert.equal(pool.kernels, 'wasm');
      try {
        const got = [];
        for (let i = 0; i < toks.length; i++) {
          const l = Float32Array.from(await pool.forward(toks[i]));
          got.push(l);
          const top = Math.max(...want[i].map(Math.abs));
          const diff = l.reduce((m, v, j) => Math.max(m, Math.abs(v - want[i][j])), 0);
          assert.ok(diff <= 1e-4 * top, `workers=${workers} pos=${i}: max diff ${diff} vs top logit ${top}`);
          assert.equal(l.indexOf(Math.max(...l)), want[i].indexOf(Math.max(...want[i])), `argmax at pos ${i}`);
        }
        // the SIMD result itself must not depend on how rows are split among workers
        if (first) got.forEach((l, i) => assert.deepEqual([...l], [...first[i]], `workers=${workers} pos=${i} differs from workers=1`));
        else first = got;
      } finally { await pool.close(); }
    }
  });
}

test('kernels option: auto resolves to wasm when SIMD exists, js is honoured, bad values are rejected by the CLI layer', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const a = await PoolModel.create(g, { ctx: 16, workers: 1 });
  const b = await PoolModel.create(g, { ctx: 16, workers: 1, kernels: 'js' });
  try {
    assert.equal(a.kernels, 'wasm');
    assert.equal(b.kernels, 'js');
    assert.equal(b.workerInfo[0].kernels, 'js');
    assert.equal(a.workerInfo[0].kernels, 'wasm-simd');
  } finally { await a.close(); await b.close(); }
  const { kernelsFromFlags } = await import('../src/cli/common.js');
  assert.equal(kernelsFromFlags({}), 'auto');
  assert.throws(() => kernelsFromFlags({ kernels: 'avx512' }), /--kernels must be/);
});

test('F16/F32 weights fall back to JS kernels inside a SIMD pool (mixed-type model)', async () => {
  const { g } = tinyFile({ arch: 'llama', quant: 'f32', seed: 3 });
  const ref = new RefModel(g, { ctx: 16 });
  const pool = await PoolModel.create(g, { ctx: 16, workers: 2, kernels: 'wasm' });
  try {
    for (const t of [4, 8, 15]) assert.deepEqual([...await pool.forward(t)], [...ref.forward(t)]); // F32 path has no SIMD kernel: still exact
  } finally { await pool.close(); }
});
