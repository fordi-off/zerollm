// End to end: a synthetic model through the worker pool with ZEROLLM_DOT=relaxed gives the same logits as the default
// kernels (within the engine's usual SIMD-vs-reference tolerance). SKIPPED on a V8 without relaxed SIMD.
import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { PoolModel } from '../src/compute/pool.js';
import { probeWasmFeatures } from '../src/compute/wasm/features.js';

const RELAXED = probeWasmFeatures().relaxedSimd;
const SKIP = RELAXED ? false : 'WebAssembly relaxed SIMD is not available in this V8';

// K-quants need rows that are multiples of 256: hidden 256, 4 heads x 64, 2 KV heads, ffn 512.
const DIMS = { nEmbd: 256, nHead: 4, nHeadKv: 2, headDim: 64, nFF: 512, nLayer: 2 };
const TOKS = [3, 70, 5, 200, 12, 9, 255, 1, 44, 17, 90, 6, 31];
const cosine = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y); };
const argmax = (a) => a.reduce((m, v, i) => (v > a[m] ? i : m), 0);

/** create a pool with ZEROLLM_DOT set to `dot` (workers copy process.env when they are created) */
async function createPool(g, dot, opts) {
  const was = process.env.ZEROLLM_DOT;
  if (dot === undefined) delete process.env.ZEROLLM_DOT; else process.env.ZEROLLM_DOT = dot;
  try { return await PoolModel.create(g, { ctx: 32, kernels: 'wasm', ...opts }); } finally { if (was === undefined) delete process.env.ZEROLLM_DOT; else process.env.ZEROLLM_DOT = was; }
}

for (const mode of ['relaxed', 'relaxed-add']) for (const quant of ['q4_k_m', 'q5_k', 'q5_0', 'q4_0', 'q8_0']) {
  test(`${quant}: ZEROLLM_DOT=${mode} pool tracks the default pool; batched prefill is bit-identical to token-by-token; independent of worker count`, { skip: SKIP }, async () => {
    const { g } = tinyFile({ arch: 'qwen3', quant, seed: 9, ...DIMS });
    const base = await createPool(g, undefined, { workers: 2 });
    const rx1 = await createPool(g, mode, { workers: 1 });
    const rx3 = await createPool(g, mode, { workers: 3 });
    const rxB = await createPool(g, mode, { workers: 2, maxBatch: 16 });
    try {
      assert.ok(base.workerInfo.every((i) => i.dot === 'extmul'), 'default pool uses the default kernels');
      for (const p of [rx1, rx3, rxB]) assert.ok(p.workerInfo.every((i) => i.dot === mode), `ZEROLLM_DOT=${mode} reaches every worker`);

      let exact = 0;
      const seq = [];
      for (let i = 0; i < TOKS.length; i++) {
        const want = Float32Array.from(await base.forward(TOKS[i]));
        const got = Float32Array.from(await rx1.forward(TOKS[i]));
        seq.push(got);
        assert.deepEqual([...await rx3.forward(TOKS[i])], [...got], `relaxed: 1 worker vs 3 workers, position ${i}`);
        const top = Math.max(...want.map(Math.abs));
        const diff = got.reduce((m, v, j) => Math.max(m, Math.abs(v - want[j])), 0);
        // same bound as the other SIMD-vs-reference tests: float summation order differs (~1e-7), and an int8 activation
        // rounding flip can move a logit by a fraction of a percent on random weights (docs/DECISIONS.md, M6/M8)
        assert.ok(diff <= 2e-2 * top, `position ${i}: relaxed vs default differ by ${diff} (top logit ${top})`);
        assert.ok(cosine(got, want) > 0.9999, `cosine at ${i}`);
        if (diff <= 1e-3 * top) exact++;
        const sorted = [...want].sort((a, b) => b - a);
        if (sorted[0] - sorted[1] > 3 * diff) assert.equal(argmax(got), argmax(want), `argmax at ${i}`);
      }
      assert.ok(exact >= TOKS.length - 3, `most positions agree to rounding noise (${exact}/${TOKS.length})`);

      // batched prefill (the relaxed batched kernels, int8 activations) == token by token (the relaxed decode kernels)
      await rxB.forwardBatch(TOKS.slice(0, 5), { logits: false });
      await rxB.forwardBatch(TOKS.slice(5, 6), { logits: false });
      const last = await rxB.forwardBatch(TOKS.slice(6));
      assert.deepEqual([...last], [...seq[TOKS.length - 1]], 'batched prefill vs token-by-token');
    } finally { await Promise.all([base.close(), rx1.close(), rx3.close(), rxB.close()]); }
  });
}
