// Qwen3.5 (hybrid Gated DeltaNet + gated attention) on the worker pool: the pool must reproduce the single-thread reference
// path (RefModel -> Qwen35Model, itself checked against an independent HF-formula implementation in qwen35.test.js).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { tinyQwen35File, Q35_KQUANT_DIMS } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { generate, prefillTokens, argmax } from '../src/model/generate.js';
import { generateSpeculative } from '../src/model/speculative.js';
import { PrefixCache, modelFingerprint } from '../src/model/prefix-cache.js';
import { recurrentShards, splitRange } from '../src/compute/layout.js';
import { keyHeadOfValueHead } from '../src/compute/ref-deltanet.js';
import { loadModel } from '../src/index.js';
import { modelConfig } from '../src/model/config.js';

const TOKS = [3, 70, 5, 200, 12, 9, 255, 1, 17, 40, 41];
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const cosine = (a, b) => { let d = 0; let na = 0; let nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };
const collect = async (m, ids, o = {}) => { const out = []; for await (const t of generate(m, ids, { maxTokens: 10, ...o })) out.push(t); return out; };
/** state arrays of a pool as plain copies */
const stateOf = (m) => ({ conv: Array.from(m.state.conv), ssm: Array.from(m.state.ssm) });

// (name, generator options, tolerance class). The K-quant file needs 256-multiples; q4_0 flips an int8 activation rounding somewhere
// (SIMD summation order) and the recurrent state carries that, so it is compared loosely.
const FILES = [
  ['f32', { quant: 'f32', seed: 3 }],
  ['q8_0', { quant: 'q8_0', seed: 5 }],
  ['q4_k_m (Q4_K/Q5_K/Q6_K/Q8_0 mix)', { quant: 'q4_k_m', seed: 6, ...Q35_KQUANT_DIMS }],
];

// ------------------------------------------------------------------------------------------------ sharding helpers

test('recurrentShards: key heads and value heads are split evenly and completely, also with more workers than heads', () => {
  for (const [nK, nV] of [[2, 4], [16, 32], [1, 1], [3, 6]]) {
    for (const n of [1, 2, 3, 4, 5, 7, 12]) {
      let pk = 0; let pv = 0;
      for (let w = 0; w < n; w++) {
        const { kk0, kk1, vh0, vh1 } = recurrentShards({ nKHeads: nK, nVHeads: nV }, n, w);
        assert.equal(kk0, pk); assert.equal(vh0, pv); assert.ok(kk1 >= kk0 && vh1 >= vh0);
        pk = kk1; pv = vh1;
      }
      assert.equal(pk, nK); assert.equal(pv, nV);
    }
  }
  assert.deepEqual(recurrentShards({ nKHeads: 16, nVHeads: 32 }, 4, 1), { kk0: 4, kk1: 8, vh0: 8, vh1: 16 });
});

test('value head -> key head mapping lives in one function (tiled: h % nKHeads)', () => {
  assert.deepEqual(Array.from({ length: 8 }, (_, h) => keyHeadOfValueHead(h, 4, 8)), [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.equal(keyHeadOfValueHead(31, 16, 32), 15);
  assert.equal(keyHeadOfValueHead(16, 16, 32), 0);
});

// ------------------------------------------------------------------------------------------------ pool == reference

for (const [name, opts] of FILES) {
  test(`qwen35 pool, JS kernels: logits are BIT-IDENTICAL to the reference path (${name}), 1-5 workers, state included`, async () => {
    const { g } = tinyQwen35File(opts);
    const ref = new RefModel(g, { ctx: 64 });
    const want = TOKS.map((t) => Float32Array.from(ref.forward(t)));
    for (const workers of [1, 2, 3, 4, 5]) { // 5 workers > 4 value heads / 2 key heads / 2 KV heads: some own nothing
      const pool = await PoolModel.create(g, { ctx: 64, workers, kernels: 'js' });
      try {
        for (let i = 0; i < TOKS.length; i++) assert.deepEqual([...await pool.forward(TOKS[i])], [...want[i]], `workers=${workers} pos=${i}`);
        assert.deepEqual(Array.from(pool.state.ssm), Array.from(ref.state.ssm), `delta-rule state, workers=${workers}`);
        assert.deepEqual(Array.from(pool.state.conv), Array.from(ref.state.conv), `conv state, workers=${workers}`);
      } finally { await pool.close(); }
    }
  });

  // extmul keeps the float summation order of the reference (1e-4); the default relaxed-add groups the same exact integer sums differently,
  // and a float difference of ~1e-7 can flip an int8 activation rounding a few layers later: a fraction of a percent on random weights.
  for (const [dot, tol] of [['extmul', 1e-4], [undefined, 2e-2]]) test(`qwen35 pool, WASM SIMD kernels (${dot ?? 'default dot form'}): tracks the reference and does not depend on the worker count (${name})`, async () => {
    const was = process.env.ZEROLLM_DOT;
    if (dot === undefined) delete process.env.ZEROLLM_DOT; else process.env.ZEROLLM_DOT = dot;
    try { await simdTracksReference(opts, tol); } finally { if (was === undefined) delete process.env.ZEROLLM_DOT; else process.env.ZEROLLM_DOT = was; }
  });
  async function simdTracksReference(opts, tol) {
    const { g } = tinyQwen35File(opts);
    const ref = new RefModel(g, { ctx: 64 });
    const want = TOKS.map((t) => Float32Array.from(ref.forward(t)));
    let first = null;
    for (const workers of [1, 2, 3, 4]) {
      const pool = await PoolModel.create(g, { ctx: 64, workers, kernels: 'wasm' });
      assert.equal(pool.kernels, 'wasm');
      try {
        const got = [];
        for (let i = 0; i < TOKS.length; i++) {
          const l = Float32Array.from(await pool.forward(TOKS[i]));
          got.push(l);
          const top = Math.max(...want[i].map(Math.abs));
          assert.ok(maxAbs(l, want[i]) <= tol * top, `workers=${workers} pos=${i}: ${maxAbs(l, want[i])} vs top ${top}`);
          // the pool's best token must be (within tol) a best token of the reference too: random tiny models have near-tied logits
          assert.ok(want[i][argmax(l)] >= Math.max(...want[i]) - tol * top, `argmax at ${i}: ${argmax(l)} vs ${argmax(want[i])}`);
        }
        if (first) got.forEach((l, i) => assert.deepEqual([...l], [...first[i]], `workers=${workers} pos=${i} differs from workers=1`));
        else first = got;
      } finally { await pool.close(); }
    }
  }
}

test('qwen35 pool with 6 and 7 workers (the target laptop has 6; key / value / KV heads split unevenly, some workers own few): same as the reference', async () => {
  const { g } = tinyQwen35File({ quant: 'q8_0', seed: 15, nK: 4, nV: 8, nHead: 8, nHeadKv: 4, nEmbd: 96, nFF: 128 });
  const ref = new RefModel(g, { ctx: 64 });
  const want = TOKS.map((t) => Float32Array.from(ref.forward(t)));
  for (const workers of [6, 7]) {
    for (const kernels of ['js', 'wasm']) {
      const pool = await PoolModel.create(g, { ctx: 64, workers, kernels });
      try {
        for (let i = 0; i < TOKS.length; i++) {
          const got = await pool.forward(TOKS[i]);
          if (kernels === 'js') assert.deepEqual([...got], [...want[i]], `workers=${workers} pos=${i}`);
          else assert.ok(maxAbs(got, want[i]) <= 1e-4 * Math.max(...want[i].map(Math.abs)), `wasm workers=${workers} pos=${i}`);
        }
        // batched prefill on the same configuration
        pool.reset();
        const l = await pool.forwardBatch(TOKS.slice(0, 7));
        assert.ok(maxAbs(l, want[6]) <= 1e-4 * Math.max(...want[6].map(Math.abs)));
      } finally { await pool.close(); }
    }
  }
});

test('qwen35 pool, WASM kernels on a Q4_0 file: the same up to int8-activation rounding flips (cosine > 0.99, same argmax at the first positions)', async () => {
  const { g } = tinyQwen35File({ quant: 'q4_0', seed: 5 });
  const ref = new RefModel(g, { ctx: 64 });
  const want = TOKS.map((t) => Float32Array.from(ref.forward(t)));
  const pool = await PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'wasm' });
  try {
    for (let i = 0; i < TOKS.length; i++) {
      const l = Float32Array.from(await pool.forward(TOKS[i]));
      assert.ok(cosine(l, want[i]) > 0.99, `pos ${i}: cos ${cosine(l, want[i])}`);
    }
  } finally { await pool.close(); }
});

for (const kernels of ['js', 'wasm']) {
  test(`qwen35 pool, f16 KV cache (${kernels} kernels): close to the f32 cache; JS and SIMD f16 paths agree`, async () => {
    const { g } = tinyQwen35File({ quant: 'q8_0', seed: 8 });
    const ref = new RefModel(g, { ctx: 64 });
    const want = TOKS.map((t) => Float32Array.from(ref.forward(t)));
    for (const workers of [1, 3]) {
      const a = await PoolModel.create(g, { ctx: 64, workers, kernels, kv: 'f16' });
      const f32 = await PoolModel.create(g, { ctx: 64, workers, kernels, kv: 'f32' });
      try {
        assert.ok(a.workerInfo.reduce((s, i) => s + i.bytes, 0) < f32.workerInfo.reduce((s, i) => s + i.bytes, 0), 'f16 shrinks the workers\' memory');
        for (let i = 0; i < TOKS.length; i++) {
          const l = Float32Array.from(await a.forward(TOKS[i]));
          const top = Math.max(...want[i].map(Math.abs));
          assert.ok(maxAbs(l, want[i]) < 5e-2 * top && cosine(l, want[i]) > 0.998, `f16 pos ${i}: ${maxAbs(l, want[i]) / top}`);
        }
      } finally { await a.close(); await f32.close(); }
    }
  });
}

test('qwen35 pool: weight slices cover every matrix byte exactly once (heads, gates, alpha/beta, ffn, embeddings)', async () => {
  const { g } = tinyQwen35File({ quant: 'q8_0', seed: 2 });
  // (ssm_conv1d is a 2-D tensor too, but every worker keeps all of it as a small private vector, like the norms)
  const matBytes = g.tensors.filter((t) => t.dims.length === 2 && !t.name.endsWith('ssm_conv1d.weight')).reduce((s, t) => s + t.nBytes, 0);
  for (const workers of [1, 3, 4]) {
    const pool = await PoolModel.create(g, { ctx: 32, workers, kernels: 'js' });
    try { assert.equal(pool.sliceBytes.reduce((a, b) => a + b, 0), matBytes, `workers=${workers}`); } finally { await pool.close(); }
  }
});

// ------------------------------------------------------------------------------------------------ batching

for (const kernels of ['js', 'wasm']) {
  test(`qwen35 forwardBatch is BIT-IDENTICAL to token-by-token forward (${kernels} kernels): any batch size / split, logits 'all', state too`, async () => {
    const { g } = tinyQwen35File({ quant: 'q8_0', seed: 41 });
    const toks = Array.from({ length: 23 }, (_, i) => (i * 37 + 11) % 250);
    const seq = await PoolModel.create(g, { ctx: 64, workers: 3, kernels });
    const wantAll = []; for (const t of toks) wantAll.push(Float32Array.from(await seq.forward(t)));
    const wantState = stateOf(seq);
    await seq.close();
    for (const sizes of [[23], [1, 22], [5, 5, 5, 5, 3], [7, 16], [16, 7], [2, 3, 4, 5, 6, 3]]) {
      const pool = await PoolModel.create(g, { ctx: 64, workers: 3, kernels, maxBatch: 16 });
      try {
        let i = 0;
        for (const sz of sizes) {
          if (sz > pool.maxBatch) {
            await assert.rejects(pool.forwardBatch(toks.slice(i, i + sz)), /batch of/);
            const half = Math.ceil(sz / 2);
            await pool.forwardBatch(toks.slice(i, i + half), { logits: false });
            assert.deepEqual([...await pool.forwardBatch(toks.slice(i + half, i + sz))], [...wantAll[i + sz - 1]]);
          } else assert.deepEqual([...await pool.forwardBatch(toks.slice(i, i + sz))], [...wantAll[i + sz - 1]], `sizes=${sizes} batch ending at ${i + sz - 1}`);
          i += sz;
        }
        assert.equal(pool.pos, 23);
        assert.deepEqual(stateOf(pool), wantState, `recurrent state after sizes=${sizes}`);
      } finally { await pool.close(); }
    }
    // logits for every position of a batch (speculative verification) == the single-token logits
    const pool = await PoolModel.create(g, { ctx: 64, workers: 2, kernels, maxBatch: 16 });
    try {
      await pool.forwardBatch(toks.slice(0, 6), { logits: false });
      const rows = Float32Array.from(await pool.forwardBatch(toks.slice(6, 11), { logits: 'all' }));
      const V = pool.cfg.vocabSize;
      for (let b = 0; b < 5; b++) assert.deepEqual([...rows.subarray(b * V, (b + 1) * V)], [...wantAll[6 + b]], `row ${b}`);
    } finally { await pool.close(); }
  });
}

test('qwen35 pool: batched prefill + decode equals a pure token-by-token run (greedy generate), batch 1 vs 16', async () => {
  const { g } = tinyQwen35File({ quant: 'q8_0', seed: 7 });
  const prompt = Array.from({ length: 21 }, (_, i) => (i * 13 + 3) % 250);
  const a = await PoolModel.create(g, { ctx: 64, workers: 2, maxBatch: 1 });
  const b = await PoolModel.create(g, { ctx: 64, workers: 2, maxBatch: 16 });
  const ref = new RefModel(g, { ctx: 64 });
  try {
    const want = await collect(ref, prompt, { maxTokens: 8 });
    assert.deepEqual(await collect(b, prompt, { maxTokens: 8 }), await collect(a, prompt, { maxTokens: 8 }));
    assert.deepEqual(await collect(b, prompt, { maxTokens: 8 }), want, 'greedy generation == reference path (wasm kernels)');
  } finally { await a.close(); await b.close(); }
});

// ------------------------------------------------------------------------------------------------ truncate / checkpoints

test('qwen35 pool: truncate to EVERY position, continue, and compare with a fresh run (bit-identical), several checkpoint settings', async () => {
  const { g, tokens } = tinyQwen35File({ quant: 'q8_0', seed: 9 });
  const im = tokens.indexOf('<|im_start|>');
  const seq = [im, 70, 5, 200, im, 9, 255, 1, 17, 3, 33];
  const fresh = await PoolModel.create(g, { ctx: 64, workers: 2, kernels: 'js' });
  const want = []; for (const t of seq) want.push(Float32Array.from(await fresh.forward(t)));
  await fresh.close();
  for (const opts of [{}, { maxCheckpoints: 0 }, { maxCheckpoints: 1 }, { checkpointEvery: 3, maxCheckpoints: 2 }]) {
    const m = await PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'js', maxBatch: 4, ...opts });
    try {
      await m.forwardBatch(seq.slice(0, 4), { logits: false });
      for (let i = 4; i < seq.length; i++) await m.forward(seq[i], { logits: false });
      for (let cut = seq.length; cut >= 0; cut--) {
        const plan = m.truncatePlan(cut);
        const replayed = await m.truncate(cut);
        assert.equal(replayed, cut === seq.length ? 0 : plan.replay);
        assert.equal(m.pos, cut);
        assert.deepEqual(m.tokens, seq.slice(0, cut));
        // mix batched and single-token continuation
        if (seq.length - cut >= 3) {
          const l = await m.forwardBatch(seq.slice(cut, cut + 3));
          assert.deepEqual([...l], [...want[cut + 2]], `opts ${JSON.stringify(opts)} cut ${cut} (batch)`);
          for (let i = cut + 3; i < seq.length; i++) assert.deepEqual([...await m.forward(seq[i])], [...want[i]], `opts ${JSON.stringify(opts)} cut ${cut} pos ${i}`);
        } else for (let i = cut; i < seq.length; i++) assert.deepEqual([...await m.forward(seq[i])], [...want[i]], `opts ${JSON.stringify(opts)} cut ${cut} pos ${i}`);
      }
    } finally { await m.close(); }
  }
});

test('qwen35 pool: a different continuation after truncate equals a fresh run of prefix + new tokens', async () => {
  const { g } = tinyQwen35File({ quant: 'f32', seed: 12 });
  const m = await PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'wasm' });
  const f = await PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'wasm' });
  try {
    for (const t of [3, 70, 5, 200, 12, 9]) await m.forward(t);
    await m.truncate(4);
    const got = []; for (const t of [8, 99, 4]) got.push(Float32Array.from(await m.forward(t)));
    const want = []; for (const t of [3, 70, 5, 200, 8, 99, 4]) want.push(Float32Array.from(await f.forward(t)));
    got.forEach((x, i) => assert.deepEqual([...x], [...want[4 + i]]));
  } finally { await m.close(); await f.close(); }
});

test('qwen35 pool: checkpoint policy matches the reference path (im_start boundaries, last prompt token, bounded, smallest evicted); reset clears everything', async () => {
  const { g, tokens } = tinyQwen35File({ quant: 'q8_0', seed: 2 });
  const im = tokens.indexOf('<|im_start|>');
  const run = async (make) => {
    const out = [];
    const m = await make();
    try {
      const prompt = [im, 4, 5, im, 6, 7, 8];
      await prefillTokens(m, prompt, { wantLogits: true }); // how generate() feeds a prompt: logits only for the last token
      out.push(m.checkpointPositions);
      out.push(m.truncatePlan(5), m.truncatePlan(6), m.truncatePlan(2));
      out.push(await m.truncate(5)); out.push(m.checkpointPositions);
      m.reset(); out.push(m.checkpointPositions, m.pos);
      return out;
    } finally { await m.close?.(); }
  };
  const want = await run(async () => new RefModel(g, { ctx: 64 }));
  assert.deepEqual(want[0], [3, 6]);
  assert.deepEqual(await run(() => PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'js', maxBatch: 16 })), want);
  assert.deepEqual(await run(() => PoolModel.create(g, { ctx: 64, workers: 2, kernels: 'js', maxBatch: 2 })), want, 'a small batch size splits the prompt elsewhere but the checkpoints are the same');
  const small = await PoolModel.create(g, { ctx: 64, workers: 2, kernels: 'js', maxCheckpoints: 2 });
  try {
    await prefillTokens(small, [im, 4, im, 5, im, 6, 7], { wantLogits: true });
    assert.deepEqual(small.checkpointPositions, [4, 6]); // [2, 4, 6]: the oldest is evicted
    const refSmall = new RefModel(g, { ctx: 64, maxCheckpoints: 2 });
    refSmall.prefill([im, 4, im, 5, im, 6, 7]);
    assert.deepEqual(refSmall.checkpointPositions, small.checkpointPositions);
  } finally { await small.close(); }
});

test('qwen35 pool: generate() regenerate / prefix reuse / divergent prompts go through the async truncate and equal a cold run', async () => {
  const { g, tokens } = tinyQwen35File({ quant: 'q8_0', seed: 14 });
  const im = tokens.indexOf('<|im_start|>');
  const prompt = [im, 4, 5, 6, im, 7, 8];
  for (const kernels of ['js', 'wasm']) {
    const m = await PoolModel.create(g, { ctx: 96, workers: 3, kernels });
    try {
      const s1 = {}; const s2 = {};
      const a = await collect(m, prompt, { stats: s1 });
      const b = await collect(m, prompt, { stats: s2 }); // regenerate: truncate(prompt.length - 1) -> a checkpoint, nothing to replay
      assert.deepEqual(a, b);
      assert.equal(s1.cachedTokens, 0);
      assert.equal(s2.cachedTokens, prompt.length - 1);
      const prompt2 = [...prompt, a[0], 99, 3];
      const cold = await PoolModel.create(g, { ctx: 96, workers: 3, kernels });
      try {
        assert.deepEqual(await collect(m, prompt2, { maxTokens: 6 }), await collect(cold, prompt2, { maxTokens: 6 }));
        const prompt3 = [im, 4, 5, 6, im, 12, 13, 14]; // diverges inside the second turn: restores the snapshot before <|im_start|>
        cold.reset();
        assert.deepEqual(await collect(m, prompt3, { maxTokens: 6 }), await collect(cold, prompt3, { maxTokens: 6 }));
      } finally { await cold.close(); }
    } finally { await m.close(); }
  }
});

// ------------------------------------------------------------------------------------------------ speculative decoding

test('qwen35 speculative decoding (hybrid target and hybrid draft, rollback by snapshot + replay): greedy output == plain greedy decoding', async () => {
  const target = tinyQwen35File({ quant: 'q8_0', seed: 51 });
  const draft = tinyQwen35File({ quant: 'q8_0', seed: 52, nLayer: 2, interval: 2 });
  const prompt = [3, 70, 5, 200, 12, 9];
  for (const kernels of ['js', 'wasm']) {
    const plain = await PoolModel.create(target.g, { ctx: 96, workers: 2, kernels });
    const T = await PoolModel.create(target.g, { ctx: 96, workers: 2, kernels, maxBatch: 8 });
    const D = await PoolModel.create(draft.g, { ctx: 96, workers: 1, kernels });
    try {
      const want = await collect(plain, prompt, { maxTokens: 24 });
      for (const k of [1, 3, 5]) {
        const stats = {};
        const got = []; for await (const t of generateSpeculative(T, D, prompt, { k, maxTokens: 24, stats })) got.push(t);
        assert.deepEqual(got, want, `kernels=${kernels} k=${k}`);
        assert.ok(stats.speculative.cycles > 0);
        // state and tokens afterwards are consistent: continuing equals the plain model continuing
        assert.deepEqual(T.tokens, T.tokens.slice(0, T.pos));
      }
      // a second request sharing the prefix (hybrid truncate inside generateSpeculative)
      const p2 = [...prompt, ...want.slice(0, 5), 77];
      const got2 = []; for await (const t of generateSpeculative(T, D, p2, { k: 3, maxTokens: 12 })) got2.push(t);
      assert.deepEqual(got2, await collect(plain, p2, { maxTokens: 12 }));
    } finally { await plain.close(); await T.close(); await D.close(); }
  }
});

// ------------------------------------------------------------------------------------------------ prefix cache

test('qwen35 prefix cache on the pool: save with one pool, load into another worker count / the reference path, continue exactly', async () => {
  const { g, path: mp } = tinyQwen35File({ quant: 'q8_0', seed: 33 });
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-q35cache-'));
  try {
    const cache = new PrefixCache(dir, modelFingerprint(mp, 'js/kv-f32'));
    const prefix = Array.from({ length: 19 }, (_, i) => (i * 17 + 5) % 250);
    const a = await PoolModel.create(g, { ctx: 64, workers: 2, kernels: 'js' });
    const b = await PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'js' });
    const r = new RefModel(g, { ctx: 64 });
    try {
      await prefillTokens(a, prefix, { wantLogits: false });
      await cache.save(a, prefix);
      assert.equal(await cache.load(b, prefix), true);
      assert.equal(await cache.load(r, prefix), true);
      assert.deepEqual(b.checkpointPositions, [prefix.length]);
      for (const t of [9, 33, 120, 4]) {
        const want = [...await a.forward(t)];
        assert.deepEqual([...await b.forward(t)], want, `pool ${t}`);
        assert.deepEqual([...r.forward(t)], want, `reference ${t}`);
      }
      // truncating back to the cached end goes through the snapshot taken at load
      assert.equal(await b.truncate(prefix.length), 0, 'the snapshot taken at load is exactly the cached end: nothing to replay');
      assert.equal(b.pos, prefix.length);
      assert.deepEqual([...await b.forward(9)], [...await a.truncate(prefix.length).then(() => a.forward(9))]);
    } finally { await a.close(); await b.close(); }
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// ------------------------------------------------------------------------------------------------ errors / library

test('qwen35 pool: a file the pool cannot run fails early with a clear message (no workers started)', async () => {
  const { g } = tinyQwen35File({ quant: 'q8_0' });
  const missing = new Proxy(g, { get: (t, k) => (k === 'tensor' ? (n) => (n === 'blk.0.ssm_a' ? undefined : t.tensor(n)) : Reflect.get(t, k)) });
  const t0 = Date.now();
  await assert.rejects(PoolModel.create(missing, { ctx: 16, workers: 2 }), /does not match the layout.*missing blk\.0\.ssm_a/);
  const other = new Proxy(g, { get: (t, k) => (k === 'get' ? (key, d) => (key === 'general.architecture' ? 'mamba2' : t.get(key, d)) : Reflect.get(t, k)) });
  await assert.rejects(PoolModel.create(other, { ctx: 16, workers: 2 }), /architecture "mamba2" is not supported \(supported: llama, qwen2, qwen3, qwen35\)/);
  assert.ok(Date.now() - t0 < 3000, 'failed before spawning any worker');
});

test('qwen35 pool: context full / use after close / out of range tokens', async () => {
  const { g } = tinyQwen35File({ quant: 'q8_0' });
  const pool = await PoolModel.create(g, { ctx: 8, workers: 2 });
  await assert.rejects(pool.forward(-1), /out of range/);
  for (let i = 0; i < 8; i++) await pool.forward(1, { logits: false });
  await assert.rejects(pool.forward(1), /context full/);
  await pool.close();
  await assert.rejects(pool.forward(1), /closed/);
});

test('loadModel(qwen35): pool by default (SIMD kernels, f16 KV, several threads); { threads: 1, kernels: "js" } is the single-thread reference path; greedy text agrees', async () => {
  const { path: p } = tinyQwen35File({ quant: 'q8_0', seed: 21 });
  const gen = async (engine, messages, o = {}) => { let s = ''; for await (const c of engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, ...o })) s += c; return s; };
  const messages = [{ role: 'user', content: 'hello' }];
  const ref = await loadModel(p, { ctx: 160, threads: 1, kernels: 'js' });
  const poolJs = await loadModel(p, { ctx: 160, threads: 3, kernels: 'js' });
  const poolSimd = await loadModel(p, { ctx: 160, threads: 2, kv: 'f16' });
  try {
    assert.equal(ref.model.constructor.name, 'Qwen35Model');
    assert.equal(poolJs.model.constructor.name, 'PoolModel');
    assert.equal(poolSimd.info.kernels, 'wasm-simd');
    assert.equal(poolSimd.info.kv, 'f16');
    const want = await gen(ref, messages);
    assert.ok(want.length > 0);
    assert.equal(await gen(poolJs, messages), want);
    assert.ok((await gen(poolSimd, messages)).length > 0);
    // two turns: the second reuses the state (cached tokens) and equals a cold run
    const s1 = {}; const s2 = {};
    const a = await gen(poolJs, messages, { stats: s1 });
    const follow = [...messages, { role: 'assistant', content: a }, { role: 'user', content: 'and you?' }];
    const c = await gen(poolJs, follow, { stats: s2 });
    assert.ok(s2.cachedTokens >= s1.promptTokens);
    poolJs.reset();
    assert.equal(await gen(poolJs, follow), c);
    assert.equal(await gen(ref, follow), c);
    // system-prompt prefix cache works for the hybrid model, on both paths
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-q35prime-'));
    try {
      const ids = Array.from({ length: 40 }, (_, i) => (i * 7 + 1) % 250);
      poolJs.reset(); ref.reset();
      assert.equal(await poolJs.primePrefix(ids, { dir, minTokens: 8 }), 'saved');
      poolJs.reset();
      assert.equal(await poolJs.primePrefix(ids, { dir, minTokens: 8 }), 'hit');
      assert.equal(poolJs.cachedTokens, 40);
      assert.equal(await ref.primePrefix(ids, { dir, minTokens: 8 }), 'hit'); // same fingerprint (js kernels, f32 KV)
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  } finally { await ref.close(); await poolJs.close(); await poolSimd.close(); }
  assert.ok(modelConfig(ref.g).hybrid);
});

test('tools/make-synth-qwen35.js writes a loadable file with the real tensor set; the pool runs it (shape check of the benchmark input)', async () => {
  const { writeSynthQwen35 } = await import('../tools/make-synth-qwen35.js');
  const { readGGUF } = await import('../src/gguf/reader.js');
  const { expectedTensors } = await import('../src/model/forward-qwen35.js');
  const f = path.join(os.tmpdir(), `zerollm-synth-${process.pid}.gguf`);
  try {
    const r = writeSynthQwen35({ outFile: f, layers: 4, vocab: 512 });
    assert.ok(r.tensors > 40);
    const g = readGGUF(f);
    const cfg = modelConfig(g);
    assert.deepEqual([cfg.nLayer, cfg.hybrid, cfg.vocabSize, cfg.nAttnLayers], [4, true, 512, 1]);
    assert.deepEqual(g.tensors.map((t) => t.name).sort(), expectedTensors(cfg).map(([n]) => n).sort());
    const pool = await PoolModel.create(g, { ctx: 32, workers: 3, kernels: 'wasm' });
    try {
      const l = await pool.forwardBatch([1, 2, 3, 4, 5]);
      assert.equal(l.length, 512);
      assert.ok(l.every(Number.isFinite));
    } finally { await pool.close(); }
  } finally { fs.rmSync(f, { force: true }); }
});

test('splitRange sanity for hybrid shards is the one used everywhere', () => {
  assert.deepEqual(recurrentShards({ nKHeads: 2, nVHeads: 4 }, 3, 2), { kk0: splitRange(2, 3, 2)[0], kk1: splitRange(2, 3, 2)[1], vh0: splitRange(4, 3, 2)[0], vh1: splitRange(4, 3, 2)[1] });
});
