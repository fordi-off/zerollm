// Per-position logits of a batched pass (`forwardBatch(tokens, {logits: 'all'})`) and KV-cache rollback: the two
// building blocks of speculative decoding.
import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { mulberry32 } from '../src/util/rng.js';

const tokens = (n, seed = 1) => { const r = mulberry32(seed); return Array.from({ length: n }, () => Math.floor(r() * 250)); };

const CASES = [['qwen3', 'q8_0'], ['llama', 'q4_0'], ['qwen2', 'q4_k_m'], ['qwen3', 'f32']];
for (const kernels of ['js', 'wasm']) {
  for (const [arch, quant] of CASES) {
    test(`logits:'all' returns every position's logits, bit-identical to token-by-token forward (${kernels}, ${arch} ${quant}, 1-3 workers)`, async () => {
      const { g } = tinyFile({ arch, quant, seed: 11 });
      const toks = tokens(21, 3);
      const seq = await PoolModel.create(g, { ctx: 64, workers: 2, kernels });
      const want = [];
      for (const t of toks) want.push(Float32Array.from(await seq.forward(t)));
      await seq.close();
      for (const workers of [1, 3]) {
        const pool = await PoolModel.create(g, { ctx: 64, workers, kernels, maxBatch: 16 });
        try {
          const V = pool.cfg.vocabSize;
          let i = 0;
          for (const sz of [1, 5, 2, 8, 1, 4]) { // 21 tokens, odd batch sizes, one single-token 'all' batch
            const got = await pool.forwardBatch(toks.slice(i, i + sz), { logits: 'all' });
            assert.equal(got.length, sz * V, 'one vocabulary-sized row per batch token');
            for (let b = 0; b < sz; b++) assert.deepEqual([...got.subarray(b * V, (b + 1) * V)], [...want[i + b]], `${workers} workers, pos ${i + b}`);
            i += sz;
          }
          assert.equal(pool.pos, 21);
        } finally { await pool.close(); }
      }
    });
  }
}

test("logits:'all' matches the single-thread reference model (RefModel) within the pool tolerances, f32 and f16 KV", async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 4 });
  const toks = tokens(14, 9);
  const ref = new RefModel(g, { ctx: 48 });
  const want = toks.map((t) => Float32Array.from(ref.forward(t)));
  for (const kernels of ['js', 'wasm']) {
    const pool = await PoolModel.create(g, { ctx: 48, workers: 2, kernels });
    try {
      const V = pool.cfg.vocabSize;
      const got = Float32Array.from(await pool.forwardBatch(toks.slice(0, 9), { logits: 'all' }));
      const got2 = Float32Array.from(await pool.forwardBatch(toks.slice(9), { logits: 'all' }));
      for (let i = 0; i < toks.length; i++) {
        const row = i < 9 ? got.subarray(i * V, (i + 1) * V) : got2.subarray((i - 9) * V, (i - 8) * V);
        if (kernels === 'js') assert.deepEqual([...row], [...want[i]], `js kernels are bit-identical to the reference (pos ${i})`);
        else {
          const top = Math.max(...want[i].map(Math.abs));
          const diff = row.reduce((m, v, j) => Math.max(m, Math.abs(v - want[i][j])), 0);
          assert.ok(diff <= 1e-4 * top, `pos ${i}: max diff ${diff} vs top ${top}`);
          assert.equal(row.indexOf(Math.max(...row)), want[i].indexOf(Math.max(...want[i])));
        }
      }
    } finally { await pool.close(); }
  }
  // f16 KV: the 'all' rows equal the one-at-a-time rows of the same f16 pool exactly
  const a = await PoolModel.create(g, { ctx: 48, workers: 2, kv: 'f16' });
  const b = await PoolModel.create(g, { ctx: 48, workers: 2, kv: 'f16' });
  try {
    const V = a.cfg.vocabSize;
    const rows = Float32Array.from(await a.forwardBatch(toks.slice(0, 10), { logits: 'all' }));
    for (let i = 0; i < 10; i++) assert.deepEqual([...rows.subarray(i * V, (i + 1) * V)], [...await b.forward(toks[i])], `f16 pos ${i}`);
  } finally { await a.close(); await b.close(); }
});

test("logits default is unchanged (last token only, vocab floats); 'last'/'none'/false/true spellings; limits are enforced", async () => {
  const { g } = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 2 });
  const pool = await PoolModel.create(g, { ctx: 64, workers: 2, maxBatch: 8, logitsBatch: 4 });
  try {
    const V = pool.cfg.vocabSize;
    assert.equal(pool.logitsBatch, 4);
    const last = await pool.forwardBatch([1, 2, 3]);
    assert.equal(last.length, V);
    const same = await pool.forwardBatch([4, 5], { logits: 'last' });
    assert.equal(same.length, V);
    assert.equal(await pool.forwardBatch([6], { logits: 'none' }), null);
    assert.equal(await pool.forwardBatch([7], { logits: false }), null);
    assert.equal((await pool.forwardBatch([8], { logits: true })).length, V);
    assert.equal((await pool.forwardBatch([9, 10, 11, 12], { logits: 'all' })).length, 4 * V);
    await assert.rejects(pool.forwardBatch([1, 2, 3, 4, 5], { logits: 'all' }), /logitsBatch/); // 5 > logitsBatch (but <= maxBatch)
    assert.equal((await pool.forwardBatch([1, 2, 3, 4, 5])).length, V); // plain batches up to maxBatch still work
  } finally { await pool.close(); }
  const small = await PoolModel.create(g, { ctx: 16, workers: 1, maxBatch: 4, logitsBatch: 1 });
  try { await assert.rejects(small.forwardBatch([1, 2], { logits: 'all' }), /logitsBatch/); } finally { await small.close(); }
});

// ---------------------------------------------------------------------------------------------------- KV rollback
for (const kv of ['f32', 'f16']) {
  for (const kernels of ['js', 'wasm']) {
    test(`KV rollback: 50 random accept/reject cycles leave exactly the KV state of a fresh sequential run (${kernels} kernels, ${kv} cache)`, async () => {
      const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 21 });
      const rnd = mulberry32(99);
      const pool = await PoolModel.create(g, { ctx: 160, workers: 3, kernels, kv, maxBatch: 8 });
      const fresh = async (seq) => { // the oracle: same configuration, tokens fed one by one into a clean pool
        const p = await PoolModel.create(g, { ctx: 160, workers: 3, kernels, kv });
        try { let l; for (const t of seq) l = Float32Array.from(await p.forward(t)); return l; } finally { await p.close(); }
      };
      try {
        const seq = tokens(6, 17); // the "text so far"
        await pool.forwardBatch(seq.slice(0, 5), { logits: false }); // the last token stays pending
        let rollbacks = 0;
        for (let cycle = 0; cycle < 50 && seq.length < 150; cycle++) {
          // verify a block of drafted tokens (some of them garbage), keep a random-length accepted prefix, then append a corrected token
          const k = 1 + Math.floor(rnd() * 5);
          const draft = Array.from({ length: k }, () => Math.floor(rnd() * 250));
          const pending = seq[seq.length - 1];
          const base = pool.pos; // == seq.length - 1: the pending token is not fed yet
          assert.equal(base, seq.length - 1);
          await pool.forwardBatch([pending, ...draft], { logits: 'all' });
          const accepted = Math.floor(rnd() * (k + 1));
          const corrected = Math.floor(rnd() * 250);
          seq.push(...draft.slice(0, accepted), corrected);
          if (accepted < k) rollbacks++;
          pool.truncate(Math.min(base + 1 + accepted, seq.length - 1)); // what generateSpeculative does
          assert.equal(pool.pos, seq.length - 1);
          if (cycle % 12 === 11) { // the whole text continues exactly like a fresh run (logits depend on every cached position)
            const got = Float32Array.from(await pool.forward(seq[seq.length - 1]));
            assert.deepEqual([...got], [...await fresh(seq)], `after cycle ${cycle}, ${seq.length} tokens`);
            assert.equal(pool.pos, seq.length);
            pool.truncate(seq.length - 1); // drop it again, keep cycling
            // (the re-fed token must recompute the same state)
          }
        }
        assert.ok(rollbacks >= 20, `exercised plenty of rollbacks (${rollbacks})`);
        const got = Float32Array.from(await pool.forward(seq[seq.length - 1]));
        assert.deepEqual([...got], [...await fresh(seq)], 'final state');
      } finally { await pool.close(); }
    });
  }
}
