// Speculative decoding loop (src/model/speculative.js) on synthetic models: target and draft share the vocabulary but have
// different sizes / seeds; an "oracle" draft is an identical copy of the target (100% acceptance); a "noisy copy" is the same
// weights quantised differently (partial acceptance).
import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { generate } from '../src/model/generate.js';
import { generateSpeculative, forwardRows, drawResidual } from '../src/model/speculative.js';
import { createSampler } from '../src/sampling/sampler.js';
import { mulberry32 } from '../src/util/rng.js';

const col = async (it) => { const o = []; for await (const t of it) o.push(t); return o; };
const tokens = (n, seed = 1) => { const r = mulberry32(seed); return Array.from({ length: n }, () => Math.floor(r() * 250)); };
const greedy = () => createSampler({ temperature: 0 });
const prompts = (n, seed = 1) => { const r = mulberry32(seed); return Array.from({ length: n }, (_, i) => Array.from({ length: 1 + Math.floor(r() * 12) + (i === 0 ? 0 : 0) }, () => Math.floor(r() * 250))); };

/** target + the three kinds of draft, as model files */
function files(arch = 'qwen3', extra = {}) {
  return {
    target: tinyFile({ arch, quant: 'q8_0', seed: 5, ...extra }),
    oracle: tinyFile({ arch, quant: 'q8_0', seed: 5, ...extra }),
    noisy: tinyFile({ arch, quant: 'q4_0', seed: 5, ...extra }),
    small: tinyFile({ arch, quant: 'q8_0', seed: 9, nEmbd: 32, nLayer: 1, nFF: 64, ...extra }),
  };
}

for (const kernels of ['js', 'wasm']) {
  test(`greedy: speculative output is IDENTICAL to plain greedy decoding (${kernels} kernels; oracle / noisy / small draft; k = 1,2,4,7; many prompts)`, async () => {
    const f = files();
    const T = await PoolModel.create(f.target.g, { ctx: 128, workers: 2, kernels, maxBatch: 16 });
    const pools = { oracle: await PoolModel.create(f.oracle.g, { ctx: 128, workers: 1, kernels }), noisy: await PoolModel.create(f.noisy.g, { ctx: 128, workers: 2, kernels }), small: await PoolModel.create(f.small.g, { ctx: 128, workers: 1, kernels }) };
    try {
      const ps = prompts(10, 3);
      const want = [];
      for (const p of ps) { T.reset(); want.push(await col(generate(T, p, { maxTokens: 28, sample: greedy() }))); }
      const rates = {};
      for (const [name, D] of Object.entries(pools)) {
        for (const k of [1, 2, 4, 7]) {
          let drafted = 0; let accepted = 0;
          for (const [i, p] of ps.entries()) {
            T.reset(); D.reset();
            const st = {};
            const got = await col(generateSpeculative(T, D, p, { maxTokens: 28, k, sample: greedy(), stats: st }));
            assert.deepEqual(got, want[i], `${name} draft, k=${k}, prompt ${i} (${p.length} tokens)`);
            assert.equal(st.generated, got.length);
            assert.equal(st.stopReason, 'length');
            drafted += st.speculative.drafted; accepted += st.speculative.accepted;
          }
          rates[`${name}/${k}`] = accepted / drafted;
        }
      }
      for (const k of [1, 2, 4, 7]) assert.equal(rates[`oracle/${k}`], 1, 'an identical draft is always accepted');
      assert.ok(rates['noisy/4'] > 0.3 && rates['noisy/4'] < 1, `noisy copy partially accepted (${rates['noisy/4']})`);
      assert.ok(rates['small/4'] < rates['noisy/4'], `unrelated small draft is accepted less (${rates['small/4']})`);
    } finally { await T.close(); for (const p of Object.values(pools)) await p.close(); }
  });
}

test('greedy with repetition / presence / frequency penalties (tentative tokens of the block count in the history) and llama arch + f16 KV', async () => {
  for (const [arch, kv] of [['llama', 'f16'], ['qwen2', 'f32']]) {
    const f = files(arch);
    const T = await PoolModel.create(f.target.g, { ctx: 128, workers: 2, kv });
    const D = await PoolModel.create(f.noisy.g, { ctx: 128, workers: 1, kv });
    try {
      for (const penalizePrompt of [true, false]) {
        for (const [pi, p] of prompts(6, 8).entries()) {
          const mk = () => createSampler({ temperature: 0, repeatPenalty: 1.4, repeatLastN: 12, presencePenalty: 0.4, frequencyPenalty: 0.3 });
          T.reset(); const want = await col(generate(T, p, { maxTokens: 30, sample: mk(), penalizePrompt }));
          T.reset(); D.reset();
          const got = await col(generateSpeculative(T, D, p, { maxTokens: 30, k: 5, sample: mk(), penalizePrompt }));
          assert.deepEqual(got, want, `${arch} ${kv} penalizePrompt=${penalizePrompt} prompt ${pi}`);
        }
      }
    } finally { await T.close(); await D.close(); }
  }
});

test('the single-thread reference model (RefModel) works as target and as draft: same tokens as plain decoding', async () => {
  const f = files('qwen3');
  const refT = new RefModel(f.target.g, { ctx: 96 });
  const refD = new RefModel(f.noisy.g, { ctx: 96 });
  const poolT = await PoolModel.create(f.target.g, { ctx: 96, workers: 2, kernels: 'js' });
  const poolD = await PoolModel.create(f.noisy.g, { ctx: 96, workers: 1, kernels: 'js' });
  try {
    for (const p of prompts(4, 11)) {
      refT.reset();
      const want = await col(generate(refT, p, { maxTokens: 20, sample: greedy() }));
      for (const [T, D, name] of [[refT, refD, 'ref/ref'], [poolT, refD, 'pool/ref'], [refT, poolD, 'ref/pool']]) {
        T.reset(); D.reset();
        assert.deepEqual(await col(generateSpeculative(T, D, p, { maxTokens: 20, k: 3, sample: greedy() })), want, name);
      }
    }
  } finally { await poolT.close(); await poolD.close(); }
});

test('stop tokens (also in the middle of a drafted block), maxTokens boundaries and the context limit behave exactly like plain generate()', async () => {
  const f = files();
  const T = await PoolModel.create(f.target.g, { ctx: 40, workers: 2, kernels: 'js' });
  const D = await PoolModel.create(f.oracle.g, { ctx: 40, workers: 1, kernels: 'js' });
  try {
    const p = [5, 9, 13, 100, 7];
    const plain = async (o) => { T.reset(); const st = {}; return { out: await col(generate(T, p, { sample: greedy(), stats: st, ...o })), st }; };
    const spec = async (o, k = 4) => { T.reset(); D.reset(); const st = {}; return { out: await col(generateSpeculative(T, D, p, { sample: greedy(), stats: st, k, ...o })), st }; };
    const full = (await plain({ maxTokens: 20 })).out;
    // a stop token in the output at index i: generation must end right before it, whatever the position inside a block
    for (const i of [0, 1, 2, 3, 4, 5, 9]) {
      const stopIds = new Set([full[i]]);
      const a = await plain({ maxTokens: 20, stopIds }); const b = await spec({ maxTokens: 20, stopIds });
      assert.deepEqual(b.out, a.out, `stop at ${i}`);
      assert.equal(b.st.stopReason, a.st.stopReason);
      assert.equal(b.st.generated, a.st.generated);
    }
    for (const max of [0, 1, 2, 3, 4, 5, 6, 11]) {
      for (const k of [1, 4, 6]) {
        const a = await plain({ maxTokens: max }); const b = await spec({ maxTokens: max }, k);
        assert.deepEqual(b.out, a.out, `maxTokens ${max}, k ${k}`);
        assert.equal(b.st.stopReason, a.st.stopReason);
      }
    }
    // context: 5 prompt tokens + 80 requested in a 40-token window
    for (const k of [1, 4, 7]) {
      const a = await plain({ maxTokens: 80 }); const b = await spec({ maxTokens: 80 }, k);
      assert.equal(a.st.stopReason, 'context');
      assert.deepEqual(b.out, a.out, `context limit, k ${k}`);
      assert.equal(b.st.stopReason, 'context');
    }
    // the same with a prompt that nearly fills the window
    const long = Array.from({ length: 37 }, (_, i) => (i * 7 + 1) % 250);
    for (const k of [1, 3, 8]) {
      T.reset(); D.reset();
      const a = await col(generate(T, long, { maxTokens: 10, sample: greedy() }));
      T.reset(); D.reset();
      assert.deepEqual(await col(generateSpeculative(T, D, long, { maxTokens: 10, k, sample: greedy() })), a, `37-token prompt, k ${k}`);
    }
    // errors
    await assert.rejects(col(generateSpeculative(T, D, [], {})), /empty prompt/);
    await assert.rejects(col(generateSpeculative(T, D, new Array(40).fill(1), {})), /context window/);
  } finally { await T.close(); await D.close(); }
});

test('abort and early exit keep both KV caches consistent: the next call reuses them and is still exact; prefix reuse across turns', async () => {
  const f = files();
  const T = await PoolModel.create(f.target.g, { ctx: 160, workers: 2 });
  const D = await PoolModel.create(f.noisy.g, { ctx: 160, workers: 1 });
  const ref = await PoolModel.create(f.target.g, { ctx: 160, workers: 2 });
  try {
    const p1 = [3, 4, 5, 6, 7, 8, 9, 10];
    // abort after 5 tokens
    const sig = { aborted: false }; const st = {}; const got = [];
    for await (const t of generateSpeculative(T, D, p1, { maxTokens: 40, k: 4, sample: greedy(), signal: sig, stats: st })) { got.push(t); if (got.length === 5) sig.aborted = true; }
    assert.equal(st.stopReason, 'aborted');
    assert.equal(got.length, 5);
    const want = await col(generate(ref, p1, { maxTokens: 40, sample: greedy() }));
    assert.deepEqual(got, want.slice(0, 5));
    // both caches only hold tokens that are really in the text
    const text = [...p1, ...got];
    assert.deepEqual(T.tokens, text.slice(0, T.pos)); assert.deepEqual(D.tokens, text.slice(0, D.pos));
    assert.ok(T.pos >= p1.length && T.pos <= text.length);
    // second turn extends the text: cache hits on both models, output exact
    const p2 = [...text, 40, 41, 42];
    const st2 = {};
    const got2 = await col(generateSpeculative(T, D, p2, { maxTokens: 25, k: 4, sample: greedy(), stats: st2 }));
    ref.reset();
    assert.deepEqual(got2, await col(generate(ref, p2, { maxTokens: 25, sample: greedy() })));
    assert.ok(st2.cachedTokens >= p1.length, `target cache reused (${st2.cachedTokens})`);
    assert.ok(st2.speculative.draftCachedTokens >= p1.length, `draft cache reused (${st2.speculative.draftCachedTokens})`);
    // the consumer walks away (break) in the middle of a block
    const got3 = [];
    for await (const t of generateSpeculative(T, D, p2, { maxTokens: 25, k: 6, sample: greedy() })) { got3.push(t); if (got3.length === 3) break; }
    const text3 = [...p2, ...got3];
    assert.deepEqual(T.tokens, text3.slice(0, T.pos)); assert.deepEqual(D.tokens, text3.slice(0, D.pos));
    const got4 = await col(generateSpeculative(T, D, [...text3, 99], { maxTokens: 15, k: 6, sample: greedy() }));
    ref.reset();
    assert.deepEqual(got4, await col(generate(ref, [...text3, 99], { maxTokens: 15, sample: greedy() })));
    // an abort during the prefill reports 'aborted' and produces nothing
    const st5 = {};
    assert.deepEqual(await col(generateSpeculative(T, D, new Array(60).fill(7), { maxTokens: 5, signal: { aborted: true }, stats: st5 })), []);
    assert.equal(st5.stopReason, 'aborted');
  } finally { await T.close(); await D.close(); await ref.close(); }
});

test('prefill progress covers both models and ends at done == total; stats are internally consistent', async () => {
  const f = files();
  const T = await PoolModel.create(f.target.g, { ctx: 128, workers: 2 });
  const D = await PoolModel.create(f.noisy.g, { ctx: 128, workers: 1 });
  try {
    const p = Array.from({ length: 50 }, (_, i) => (i * 13 + 2) % 250);
    const prog = []; const st = {};
    const out = await col(generateSpeculative(T, D, p, { maxTokens: 30, k: 4, sample: greedy(), stats: st, onPrefillProgress: (d, t) => prog.push([d, t]) }));
    assert.ok(prog.length >= 4);
    assert.ok(prog.every(([d, t], i) => d <= t && (i === 0 || d >= prog[i - 1][0]) && t === prog[0][1]));
    assert.deepEqual(prog.at(-1), [98, 98], '49 prompt tokens through each model'); // the last prompt token is processed in the first verification pass
    const sp = st.speculative;
    assert.equal(st.promptTokens, 50); assert.equal(st.prefilled, 50); assert.equal(st.generated, out.length);
    assert.equal(sp.acceptedHist.reduce((a, b) => a + b, 0), sp.cycles);
    assert.equal(sp.acceptedHist.reduce((a, c, i) => a + c * i, 0), sp.accepted);
    assert.equal(out.length, sp.accepted + sp.cycles, 'every cycle emits its accepted drafts plus one more token');
    assert.ok(sp.accepted <= sp.drafted && sp.drafted <= sp.cycles * 4);
    assert.ok(Math.abs(sp.acceptRate - sp.accepted / sp.drafted) < 1e-12);
    assert.ok(Math.abs(sp.tokensPerCycle - out.length / sp.cycles) < 1e-12);
    assert.ok(sp.draftMs > 0 && sp.verifyMs > 0 && sp.draftMs + sp.verifyMs <= st.decodeMs + 1);
    assert.ok(sp.effectiveTps > 0);
    // second call, same prompt: the draft prefill is skipped as well
    const st2 = {};
    await col(generateSpeculative(T, D, p, { maxTokens: 3, k: 4, sample: greedy(), stats: st2 }));
    assert.equal(st2.cachedTokens, 49); assert.equal(st2.speculative.draftCachedTokens, 49);
  } finally { await T.close(); await D.close(); }
});

test('misuse: draft with a different vocabulary size / a target that cannot verify batches / a sampler without distribution()', async () => {
  const f = files();
  const T = await PoolModel.create(f.target.g, { ctx: 32, workers: 1, maxBatch: 1 });
  const T2 = await PoolModel.create(f.target.g, { ctx: 32, workers: 1 });
  const odd = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 5, padVocabTo: 300 });
  const D = await PoolModel.create(odd.g, { ctx: 32, workers: 1 });
  const D2 = await PoolModel.create(f.noisy.g, { ctx: 32, workers: 1 });
  try {
    await assert.rejects(col(generateSpeculative(T, D2, [1, 2, 3], {})), /at least 2 tokens per pass/);
    await assert.rejects(col(generateSpeculative(T2, D, [1, 2, 3], {})), /draft vocabulary/);
    await assert.rejects(col(generateSpeculative(T2, D2, [1, 2, 3], { sample: () => 0 })), /createSampler/);
  } finally { await T.close(); await T2.close(); await D.close(); await D2.close(); }
});

test('forwardRows: PoolModel does one batched pass; the reference model (no batch support) falls back to one forward per token with the same rows', async () => {
  const { g } = tinyFile({ arch: 'qwen2', quant: 'q8_0', seed: 6 });
  const toks = tokens(9, 5);
  const ref = new RefModel(g, { ctx: 32 });
  const pool = await PoolModel.create(g, { ctx: 32, workers: 2, kernels: 'js' });
  try {
    const V = pool.cfg.vocabSize;
    const a = Float32Array.from(await forwardRows(ref, toks.slice(0, 6)));
    const b = Float32Array.from(await forwardRows(pool, toks.slice(0, 6)));
    assert.equal(a.length, 6 * V);
    assert.deepEqual([...a], [...b]);
    const c = Float32Array.from(await forwardRows(ref, toks.slice(6, 7))); // single token goes through forward()
    assert.deepEqual([...c], [...await forwardRows(pool, toks.slice(6, 7))]);
    assert.equal(ref.pos, 7); assert.equal(pool.pos, 7);
  } finally { await pool.close(); }
});

test('drawResidual: draws from the normalised max(0, p - q); leaves its scratch buffer clean; empty residual falls back to p', () => {
  const s = createSampler({ temperature: 1, topK: 0, topP: 1, seed: 3 });
  const mk = (arr) => s.distribution(Float32Array.from(arr.map(Math.log)), []);
  const p = mk([0.5, 0.3, 0.15, 0.05]); const q = mk([0.1, 0.6, 0.2, 0.1]);
  // residual = [0.4, 0, 0, 0] -> always token 0
  const scratch = new Float64Array(4);
  for (let i = 0; i < 200; i++) assert.equal(drawResidual(s, p, q, scratch), 0);
  assert.ok(scratch.every((v) => v === 0));
  // residual with two positive entries: p = [.4,.4,.1,.1], q = [.1,.2,.4,.3] -> [.3,.2,0,0] -> 60% / 40%
  const p2 = mk([0.4, 0.4, 0.1, 0.1]); const q2 = mk([0.1, 0.2, 0.4, 0.3]);
  const c = [0, 0, 0, 0]; const N = 20000;
  for (let i = 0; i < N; i++) c[drawResidual(s, p2, q2, scratch)]++;
  assert.equal(c[2] + c[3], 0);
  assert.ok(Math.abs(c[0] / N - 0.6) < 0.015, `P(0) = ${c[0] / N}`);
  assert.ok(scratch.every((v) => v === 0));
  // p == q: nothing left, so the draw is from p
  const same = new Set(); for (let i = 0; i < 300; i++) same.add(drawResidual(s, p, p, scratch));
  assert.ok(same.size >= 3);
});

// ----------------------------------------------------------------------------------------------------------------------
// Sampling: the output distribution equals the target's. Exact reference probabilities come from the target model itself.
// ----------------------------------------------------------------------------------------------------------------------

/** chi-square statistic of observed counts (Map id -> n) against exact probabilities (sparse dist); small expected bins are merged */
function chiSquare(counts, total, dist) {
  let stat = 0; let df = -1; let poolE = 0; let poolO = 0;
  const inSupport = new Set(dist.ids);
  let outside = 0; for (const [id, n] of counts) if (!inSupport.has(id)) outside += n;
  assert.equal(outside, 0, 'a token outside the target distribution\'s support was produced');
  for (let i = 0; i < dist.ids.length; i++) {
    const e = (total * dist.probs[i]) / dist.total; const o = counts.get(dist.ids[i]) ?? 0;
    if (e < 8) { poolE += e; poolO += o; continue; }
    stat += (o - e) ** 2 / e; df++;
  }
  if (poolE > 0) { if (poolE >= 5) { stat += (poolO - poolE) ** 2 / poolE; df++; } }
  return { stat, df };
}
const critical = (df) => df + 6 * Math.sqrt(2 * df) + 6; // far in the tail (p << 1e-5): fails only for a real distribution error

test('sampling: speculative output has exactly the target distribution at every position (chi-square vs the exact probabilities; penalties, top-k, top-p)', async () => {
  const peaky = { std: 0.4, embdStd: 0.3 };
  const tf = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 5, ...peaky });
  const df = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 5, ...peaky, perturb: 0.05 }); // same shape, weights = target + 5% noise: partial acceptance
  const T = await PoolModel.create(tf.g, { ctx: 64, workers: 2, kernels: 'js' });
  const D = await PoolModel.create(df.g, { ctx: 64, workers: 1, kernels: 'js' });
  const oracle = await PoolModel.create(tf.g, { ctx: 64, workers: 1, kernels: 'js' }); // for exact reference distributions
  const prompt = [11, 22, 33, 44, 55, 66];
  const CONFIGS = [
    { temperature: 1, topK: 10, topP: 0.95, repeatPenalty: 1.3, repeatLastN: 16, presencePenalty: -4 }, // negative presence penalty = repeats are rewarded, so the tentative tokens of a block visibly change the next distribution
    { temperature: 0.8, topK: 0, topP: 0.9, minP: 0.02 },
  ];
  try {
    for (const [ci, cfg] of CONFIGS.entries()) {
      const exact = createSampler({ ...cfg, seed: 1 });
      const N = 3000; const runs = [];
      const stSum = { drafted: 0, accepted: 0 };
      for (let i = 0; i < N; i++) {
        const sample = createSampler({ ...cfg, seed: (mulberry32(i + 1)() * 2 ** 32) >>> 0 });
        const st = {};
        runs.push(await col(generateSpeculative(T, D, prompt, { maxTokens: 3, k: 4, sample, stats: st })));
        stSum.drafted += st.speculative.drafted; stSum.accepted += st.speculative.accepted;
      }
      const rate = stSum.accepted / stSum.drafted;
      assert.ok(rate > 0.05 && rate < 0.95, `config ${ci}: both acceptance and rejection paths are exercised (accept rate ${rate.toFixed(2)})`);

      // exact distribution of token 1 given the prompt
      oracle.reset();
      await oracle.forwardBatch(prompt.slice(0, -1), { logits: false });
      const p1 = exact.distribution(Float32Array.from(await oracle.forward(prompt.at(-1))), prompt);
      const c1 = new Map(); for (const r of runs) c1.set(r[0], (c1.get(r[0]) ?? 0) + 1);
      const x1 = chiSquare(c1, N, p1);
      assert.ok(x1.stat < critical(x1.df), `config ${ci} token 1: chi2 ${x1.stat.toFixed(1)} (df ${x1.df})`);

      // power check: the same data tested against the DRAFT's distribution must be rejected (so the test can tell the two apart)
      D.reset();
      await D.forwardBatch(prompt.slice(0, -1), { logits: false });
      const q1 = exact.distribution(Float32Array.from(await D.forward(prompt.at(-1))), prompt);
      const supportP = new Set(p1.ids);
      const outsideQ = [...c1.keys()].filter((id) => !new Set(q1.ids).has(id)).length;
      const xq = outsideQ > 0 ? { stat: Infinity, df: 1 } : chiSquare(c1, N, q1);
      assert.ok(xq.stat > 3 * critical(xq.df), `config ${ci}: draft distribution is distinguishable from the target (chi2 ${xq.stat.toFixed(0)}, support overlap ${[...q1.ids].filter((i) => supportP.has(i)).length})`);

      // token 2 given each of the most frequent tokens 1 (the history of that position contains the tentative token 1), token 3 given the most frequent pair
      const ranked = [...c1].sort((x, y) => y[1] - x[1]).filter(([, n]) => n >= 150).slice(0, 4);
      assert.ok(ranked.length >= 2);
      let maxHistoryEffect = 0; let best = null;
      for (const [t1, n1] of ranked) {
        const sub = runs.filter((r) => r[0] === t1);
        assert.equal(sub.length, n1);
        oracle.reset();
        await oracle.forwardBatch(prompt, { logits: false });
        const l2 = Float32Array.from(await oracle.forward(t1));
        const p2 = exact.distribution(l2, [...prompt, t1]);
        const c2 = new Map(); for (const r of sub) c2.set(r[1], (c2.get(r[1]) ?? 0) + 1);
        const x2 = chiSquare(c2, sub.length, p2);
        assert.ok(x2.stat < critical(x2.df), `config ${ci} token 2 | t1=${t1} (n=${sub.length}): chi2 ${x2.stat.toFixed(1)} (df ${x2.df})`);
        // how much would the distribution differ if the tentative token were missing from the history? (guards the test's power for penalties)
        const wrong = exact.distribution(l2, prompt);
        const pm = new Map(Array.from(p2.ids, (id, i) => [id, p2.probs[i] / p2.total])); const wm = new Map(Array.from(wrong.ids, (id, i) => [id, wrong.probs[i] / wrong.total]));
        let tv = 0; for (const id of new Set([...pm.keys(), ...wm.keys()])) tv += Math.abs((pm.get(id) ?? 0) - (wm.get(id) ?? 0)) / 2;
        maxHistoryEffect = Math.max(maxHistoryEffect, tv);
        if (!best || n1 > best.n) best = { t1, sub, n: n1 };
      }
      if (cfg.presencePenalty) assert.ok(maxHistoryEffect > 0.1, `config ${ci}: the tentative token must matter for the test to be meaningful (TV ${maxHistoryEffect.toFixed(3)})`);
      const { t1, sub } = best;
      const [t2] = [...sub.reduce((m, r) => m.set(r[1], (m.get(r[1]) ?? 0) + 1), new Map())].sort((x, y) => y[1] - x[1])[0];
      const sub2 = sub.filter((r) => r[1] === t2);
      if (sub2.length >= 60) {
        oracle.reset();
        await oracle.forwardBatch([...prompt, t1], { logits: false });
        const p3 = exact.distribution(Float32Array.from(await oracle.forward(t2)), [...prompt, t1, t2]);
        const c3 = new Map(); for (const r of sub2) c3.set(r[2], (c3.get(r[2]) ?? 0) + 1);
        const x3 = chiSquare(c3, sub2.length, p3);
        assert.ok(x3.stat < critical(x3.df), `config ${ci} token 3 | t1,t2 (n=${sub2.length}): chi2 ${x3.stat.toFixed(1)} (df ${x3.df})`);
      }
    }
  } finally { await T.close(); await D.close(); await oracle.close(); }
});

test('sampling: with an oracle draft every token is accepted at any temperature; seeded runs are reproducible; plain generate() with the same sampler stays unchanged', async () => {
  const f = files('llama', { std: 0.3, embdStd: 0.3 });
  const T = await PoolModel.create(f.target.g, { ctx: 96, workers: 2 });
  const D = await PoolModel.create(f.oracle.g, { ctx: 96, workers: 1 });
  try {
    const p = [9, 8, 7, 6];
    const mk = () => createSampler({ temperature: 0.9, topK: 20, topP: 0.9, repeatPenalty: 1.2, seed: 77 });
    const st = {};
    const a = await col(generateSpeculative(T, D, p, { maxTokens: 25, k: 4, sample: mk(), stats: st }));
    const b = await col(generateSpeculative(T, D, p, { maxTokens: 25, k: 4, sample: mk() }));
    assert.deepEqual(a, b);
    assert.equal(st.speculative.acceptRate, 1, 'q == p, so min(1, p/q) = 1 for every draft token');
    assert.equal(a.length, 25);
    // an unrelated draft changes how many tokens each step yields, never the legal support
    const small = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 3, nEmbd: 32, nLayer: 1, nFF: 64, std: 0.3, embdStd: 0.3 });
    const S = await PoolModel.create(small.g, { ctx: 96, workers: 1 });
    try {
      const st2 = {};
      const c = await col(generateSpeculative(T, S, p, { maxTokens: 25, k: 4, sample: mk(), stats: st2 }));
      assert.equal(c.length, 25);
      assert.ok(st2.speculative.acceptRate < 1 && st2.speculative.acceptRate >= 0);
    } finally { await S.close(); }
    // plain sampling is unaffected by any of this (seeded output is a pure function of the model and the seed)
    T.reset();
    assert.deepEqual(await col(generate(T, p, { maxTokens: 25, sample: mk() })), await (async () => { T.reset(); return col(generate(T, p, { maxTokens: 25, sample: mk() })); })());
  } finally { await T.close(); await D.close(); }
});
