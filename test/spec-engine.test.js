// Speculative decoding through the public library API (loadModel({draft}), Engine#useDraft) and the CLI (--draft).
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tinyFile } from './helpers.js';
import { loadModel, estimateModelRam, inspectModel, EngineMemoryError, UnsupportedModelError, vocabMismatch, vocabFingerprint, Engine } from '../src/index.js';
import { readGGUF } from '../src/gguf/reader.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const cli = (args, input) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8', input });
const collect = async (gen) => { let s = ''; for await (const c of gen) s += c; return s; };

const M = { target: { arch: 'qwen3', quant: 'q8_0', seed: 5 }, noisy: { arch: 'qwen3', quant: 'q4_0', seed: 5 }, small: { arch: 'qwen3', quant: 'q8_0', seed: 9, nEmbd: 32, nLayer: 1, nFF: 64 } };

test('vocabulary check: same family passes (any size / seed / arch variant with the same tokens); different size or different tokens fail with a clear message', () => {
  const a = tinyFile(M.target).g; const b = tinyFile(M.small).g; const c = tinyFile({ ...M.target, arch: 'qwen2' }).g;
  assert.equal(vocabMismatch(a, b), null);
  assert.equal(vocabMismatch(a, c), null, 'qwen2 and qwen3 tiny models carry the same token list');
  assert.deepEqual(vocabFingerprint(a), vocabFingerprint(b));
  const llama = tinyFile({ arch: 'llama', quant: 'q8_0' }).g;
  assert.match(vocabMismatch(a, llama), /vocabulary of 272 tokens but the target has 270/);
  // same SIZE, different token strings (llama has 272 natural tokens; pad the qwen model to 272 with different specials)
  const padded = tinyFile({ ...M.target, padVocabTo: 272 }).g;
  assert.equal(vocabFingerprint(padded).size, vocabFingerprint(llama).size);
  assert.match(vocabMismatch(padded, llama), /tokenizer differs from the target's \(first difference at token id 267/);
});

test('loadModel({draft}): greedy text is identical with and without the draft; stats report acceptance, draft/verify time and effective tok/s; per-call opt-out; reset', async () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const plain = await loadModel(t.path, { ctx: 200, threads: 2 });
  const spec = await loadModel(t.path, { ctx: 200, threads: 2, draft: { path: d.path, k: 4, threads: 1 } });
  try {
    assert.equal(plain.info.draft, null);
    assert.equal(spec.info.draft.k, 4);
    assert.equal(spec.info.draft.threads, 1);
    assert.equal(spec.info.draft.ctx, 200);
    assert.ok(spec.info.draft.estimate.total > 0);
    const messages = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hello there' }];
    const sampling = { temperature: 0 };
    const want = await collect(plain.generate({ messages, maxTokens: 24, sampling }));
    const st = {};
    const got = await collect(spec.generate({ messages, maxTokens: 24, sampling, stats: st }));
    assert.equal(got, want);
    assert.equal(st.generated, 24);
    assert.equal(st.stopReason, 'length');
    const sp = st.speculative;
    assert.ok(sp && sp.cycles > 0 && sp.k === 4);
    assert.ok(sp.acceptRate > 0 && sp.acceptRate <= 1);
    assert.ok(sp.tokensPerCycle >= 1 && sp.tokensPerCycle <= 5);
    assert.ok(sp.draftMs > 0 && sp.verifyMs > 0 && sp.effectiveTps > 0);
    assert.equal(st.promptTokens, spec.countPromptTokens(messages));
    assert.ok(st.prefillMs >= 0 && st.decodeMs > 0);

    // second call: prefix served from the KV caches of BOTH models
    const st2 = {};
    assert.equal(await collect(spec.generate({ messages, maxTokens: 24, sampling, stats: st2 })), want);
    assert.ok(st2.cachedTokens >= st2.promptTokens - 1);
    assert.ok(st2.speculative.draftCachedTokens >= st2.promptTokens - 2);

    // per-call opt-out: plain decoding, no speculative stats
    const st3 = {};
    assert.equal(await collect(spec.generate({ messages, maxTokens: 10, sampling, stats: st3, speculative: false })), want.slice(0, (await collect(plain.generate({ messages, maxTokens: 10, sampling }))).length));
    assert.equal(st3.speculative, undefined);

    // reset forgets both models
    spec.reset();
    assert.equal(spec.cachedTokens, 0);
    assert.equal(spec.draft.engine.cachedTokens, 0);
    const st4 = {};
    assert.equal(await collect(spec.generate({ messages, maxTokens: 24, sampling, stats: st4 })), want);
    assert.equal(st4.cachedTokens, 0);
  } finally { await plain.close(); await spec.close(); }
  assert.equal(spec.draft, null);
});

test('Engine.generate with a draft: stop strings, abort (signal object and AbortController), maxTokens, prefill progress, raw prompts, chat continuation; all equal the plain engine', async () => {
  for (const arch of ['qwen3', 'llama']) {
    const t = tinyFile({ ...M.target, arch }); const d = tinyFile({ ...M.small, arch });
    const plain = await loadModel(t.path, { ctx: 220, threads: 2, kv: 'f16' });
    const spec = await loadModel(t.path, { ctx: 220, threads: 2, kv: 'f16', draft: { path: d.path, k: 5 } });
    try {
      const messages = [{ role: 'user', content: 'hello hello hello' }];
      const sampling = { temperature: 0 };
      const want = await collect(plain.generate({ messages, maxTokens: 30, sampling }));
      // stop strings (cut before the match; the match is never emitted)
      const marker = want.slice(5, 7);
      const stA = {}; const stB = {};
      const a = await collect(plain.generate({ messages, maxTokens: 30, sampling, stop: [marker], stats: stA }));
      const b = await collect(spec.generate({ messages, maxTokens: 30, sampling, stop: [marker], stats: stB }));
      assert.equal(b, a);
      assert.equal(stB.stopReason, 'stop-string'); assert.equal(stA.stopReason, 'stop-string');
      // abort
      const ctl = new AbortController(); const stC = {}; let n = 0;
      for await (const _ of spec.generate({ messages, maxTokens: 50, sampling, signal: ctl.signal, stats: stC })) { if (++n === 3) ctl.abort(); }
      assert.equal(stC.stopReason, 'aborted'); assert.ok(n >= 3 && n < 50);
      // the engine is still consistent afterwards
      assert.equal(await collect(spec.generate({ messages, maxTokens: 30, sampling })), want);
      // a conversation: second turn extends the first turn's text
      const turn1 = await collect(spec.generate({ messages, maxTokens: 12, sampling }));
      const messages2 = [...messages, { role: 'assistant', content: turn1 }, { role: 'user', content: 'again please' }];
      const st5 = {};
      const t2s = await collect(spec.generate({ messages: messages2, maxTokens: 20, sampling, stats: st5 }));
      assert.equal(t2s, await collect(plain.generate({ messages: messages2, maxTokens: 20, sampling })));
      // raw prompt mode + special tokens
      const raw = arch === 'qwen3' ? '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n' : '<|begin_of_text|>hello';
      assert.equal(await collect(spec.generate({ prompt: raw, maxTokens: 12, sampling })), await collect(plain.generate({ prompt: raw, maxTokens: 12, sampling })));
      // progress reporting ends at done == total
      const long = [{ role: 'user', content: 'a b c d e f g h i j k l m n o p q r s t u v w x y z '.repeat(3) }];
      const prog = [];
      await collect(spec.generate({ messages: long, maxTokens: 2, sampling, onPrefill: (dn, tt) => prog.push([dn, tt]) }));
      assert.ok(prog.length >= 2 && prog.at(-1)[0] === prog.at(-1)[1]);
      // context limits are still enforced
      await assert.rejects(collect(spec.generate({ prompt: 'x '.repeat(500), maxTokens: 1 })), /context window/);
    } finally { await plain.close(); await spec.close(); }
  }
});

test('Engine.generate with a draft: sampled output is reproducible for a seed, penalties accepted, and stays inside a sane stats envelope', async () => {
  const t = tinyFile({ ...M.target, std: 0.3, embdStd: 0.3 }); const d = tinyFile({ ...M.target, std: 0.3, embdStd: 0.3, perturb: 0.05 });
  const spec = await loadModel(t.path, { ctx: 200, threads: 2, draft: { path: d.path, k: 4, threads: 1 } });
  try {
    const messages = [{ role: 'user', content: 'tell me something' }];
    const sampling = { temperature: 0.9, topK: 30, topP: 0.9, repeatPenalty: 1.2, seed: 42 };
    const st = {};
    const a = await collect(spec.generate({ messages, maxTokens: 30, sampling, stats: st }));
    const b = await collect(spec.generate({ messages, maxTokens: 30, sampling }));
    assert.equal(a, b);
    assert.ok(st.speculative.acceptRate > 0.2 && st.speculative.acceptRate < 1, `accept rate ${st.speculative.acceptRate}`);
    const c = await collect(spec.generate({ messages, maxTokens: 30, sampling: { ...sampling, seed: 43 } }));
    assert.notEqual(a, c);
  } finally { await spec.close(); }
});

test('useDraft on a loaded engine: attaches / replaces / detaches; k is clamped to the target batch; misuse is rejected with clear errors', async () => {
  const t = tinyFile(M.target); const d1 = tinyFile(M.noisy); const d2 = tinyFile(M.small);
  const eng = await loadModel(t.path, { ctx: 120, threads: 1, maxBatch: 4 });
  const e1 = await loadModel(d1.path, { ctx: 120, threads: 1 });
  const e2 = await loadModel(d2.path, { ctx: 120, threads: 1 });
  try {
    assert.equal(eng.useDraft(e1, { k: 10 }), eng);
    assert.equal(eng.draft.k, 3, 'k is at most maxBatch - 1');
    const st = {};
    await collect(eng.generate({ prompt: 'hello', maxTokens: 12, sampling: { temperature: 0 }, stats: st }));
    assert.equal(st.speculative.k, 3);
    eng.useDraft(e2, { k: 2 });
    assert.equal(eng.draft.engine, e2); assert.equal(eng.draft.own, false);
    assert.equal(e1.closed, false, 'a draft the engine does not own is never closed by it');
    assert.throws(() => eng.useDraft(eng), /cannot draft for itself/);
    assert.throws(() => eng.useDraft({}), /expects an Engine/);
    assert.throws(() => eng.useDraft(e1, { k: 0 }), /positive integer/);
    assert.throws(() => eng.useDraft(e1, { k: 1.5 }), /positive integer/);
    const llama = await loadModel(tinyFile({ arch: 'llama', quant: 'q8_0' }).path, { ctx: 64, threads: 1 });
    try { assert.throws(() => eng.useDraft(llama), (e) => e instanceof UnsupportedModelError && /vocabulary of 272 tokens/.test(e.message)); } finally { await llama.close(); }
    const single = await loadModel(t.path, { ctx: 64, threads: 1, maxBatch: 1 });
    try { assert.throws(() => single.useDraft(e1), /maxBatch >= 2/); } finally { await single.close(); }
    await eng.clearDraft();
    assert.equal(eng.draft, null);
    assert.equal(e2.closed, false);
    const st2 = {};
    await collect(eng.generate({ prompt: 'hello', maxTokens: 5, sampling: { temperature: 0 }, stats: st2 }));
    assert.equal(st2.speculative, undefined);
    // a closed draft is reported, not silently ignored
    eng.useDraft(e2);
    await e2.close();
    await assert.rejects(collect(eng.generate({ prompt: 'hello', maxTokens: 5 })), /draft engine was closed/);
    assert.throws(() => eng.useDraft(e2), /closed/);
  } finally { await eng.close(); await e1.close(); await e2.close(); }
});

test('loadModel({draft}): incompatible vocabulary throws UnsupportedModelError before anything is loaded; bad draft files are rejected', async () => {
  const t = tinyFile(M.target);
  const llama = tinyFile({ arch: 'llama', quant: 'q8_0' });
  await assert.rejects(loadModel(t.path, { threads: 1, draft: { path: llama.path } }), (e) => e instanceof UnsupportedModelError && e.code === 'UNSUPPORTED_MODEL' && /draft model has a vocabulary of 272 tokens but the target has 270/.test(e.message));
  const padded = tinyFile({ ...M.target, padVocabTo: 272 });
  await assert.rejects(loadModel(padded.path, { threads: 1, draft: { path: llama.path } }), /tokenizer differs from the target's/);
  await assert.rejects(loadModel(t.path, { threads: 1, draft: {} }), /draft.path is missing/);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'spec-bad-'));
  try {
    const junk = path.join(tmp, 'junk.gguf'); fs.writeFileSync(junk, 'not a model');
    await assert.rejects(loadModel(t.path, { threads: 1, draft: { path: junk } }));
    await assert.rejects(loadModel(t.path, { threads: 1, draft: { path: path.join(tmp, 'missing.gguf') } }), /ENOENT/);
  } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
});

test("loadModel's RAM gate counts the draft model's memory", async () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const et = estimateModelRam(inspectModel(t.path), { ctx: 200, threads: 1 }).total;
  const ed = estimateModelRam(inspectModel(d.path), { ctx: 200, threads: 1 }).total;
  const free = (et + 0.5 * ed) / 0.95; // enough for the target alone, not for both
  const ok = await loadModel(t.path, { ctx: 200, threads: 1, freeBytes: free });
  await ok.close();
  await assert.rejects(loadModel(t.path, { ctx: 200, threads: 1, freeBytes: free, draft: { path: d.path, threads: 1 } }),
    (e) => e instanceof EngineMemoryError && e.reason === 'ram' && /draft model/.test(e.message) && Math.abs(e.needed - (et + ed)) < 1 && e.needed > e.free);
  const both = await loadModel(t.path, { ctx: 200, threads: 1, freeBytes: (et + ed) / 0.9, draft: { path: d.path, threads: 1 } });
  try { assert.ok(both.draftEstimate.total > 0); assert.equal(both.draftEstimate.total, ed); } finally { await both.close(); }
});

test('draft context smaller than the prompt: that call silently runs without speculation; the default draft thread count is small', async () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const eng = await loadModel(t.path, { ctx: 200, threads: 2, draft: { path: d.path, ctx: 48 } });
  try {
    assert.equal(eng.draft.engine.ctx, 48);
    const st = {};
    const long = 'a b c d e f g h i j k l m n o p q r s t u v w x y z '.repeat(3);
    const text = await collect(eng.generate({ prompt: long, maxTokens: 6, sampling: { temperature: 0 }, stats: st }));
    assert.ok(st.promptTokens > 48);
    assert.equal(st.speculative, undefined);
    assert.ok(text.length > 0);
    const st2 = {};
    await collect(eng.generate({ prompt: 'hello', maxTokens: 6, sampling: { temperature: 0 }, stats: st2 }));
    assert.ok(st2.speculative);
  } finally { await eng.close(); }
  const { defaultDraftThreads } = await import('../src/index.js');
  assert.equal(defaultDraftThreads(6), 4); assert.equal(defaultDraftThreads(2), 2); assert.equal(defaultDraftThreads(1), 1);
});

test('primePrefix (on-disk system-prompt cache) works with a draft attached: the draft prefills its own prompt, output is exact', async () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'spec-prefix-'));
  const plain = await loadModel(t.path, { ctx: 220, threads: 2 });
  try {
    const system = 'You are a very careful assistant. Answer briefly and politely. '.repeat(2);
    const messages = [{ role: 'system', content: system }, { role: 'user', content: 'hello there' }];
    const sampling = { temperature: 0 };
    const want = await collect(plain.generate({ messages, maxTokens: 20, sampling }));
    for (const expected of ['saved', 'hit']) {
      const spec = await loadModel(t.path, { ctx: 220, threads: 2, draft: { path: d.path, threads: 1 } });
      try {
        const prefixIds = spec.promptIds([{ role: 'system', content: system }]);
        const ids = spec.promptIds(messages);
        let n = 0; while (n < prefixIds.length && prefixIds[n] === ids[n]) n++;
        assert.ok(n >= 32);
        assert.equal(await spec.primePrefix(ids.slice(0, n), { dir }), expected);
        assert.equal(spec.cachedTokens, n);
        assert.equal(spec.draft.engine.cachedTokens, 0, 'the draft was not primed');
        const st = {};
        assert.equal(await collect(spec.generate({ messages, maxTokens: 20, sampling, stats: st })), want);
        assert.ok(st.cachedTokens >= n);
        assert.equal(st.speculative.draftCachedTokens, 0);
        assert.ok(st.speculative.cycles > 0);
      } finally { await spec.close(); }
    }
  } finally { await plain.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

// ------------------------------------------------------------------------------------------------------------------ CLI
test('CLI run --draft: same text as without at --temp 0, speculative stats line, banner; flag validation', () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy); const llama = tinyFile({ arch: 'llama', quant: 'q8_0' });
  const plain = cli(['run', t.path, '-p', 'hello', '--n', '24', '--temp', '0', '--threads', '2']);
  const spec = cli(['run', t.path, '-p', 'hello', '--n', '24', '--temp', '0', '--threads', '2', '--draft', d.path, '--draft-k', '3', '--draft-threads', '1']);
  assert.equal(plain.status, 0, plain.stderr); assert.equal(spec.status, 0, spec.stderr);
  assert.equal(spec.stdout, plain.stdout);
  assert.match(spec.stderr, /speculative decoding: draft qwen3 2L on 1 thread, k=3/);
  assert.match(spec.stderr, /speculative k=3: accepted \d+% of \d+ drafted, [\d.]+ tok\/step, draft \d+% \+ verify \d+% of decode time/);
  assert.doesNotMatch(plain.stderr, /speculative/);
  // single-thread JS reference path (RefModel) as target and draft
  const ref = cli(['run', t.path, '-p', 'hello', '--n', '12', '--temp', '0', '--threads', '1', '--kernels', 'js', '--draft', d.path]);
  assert.equal(ref.status, 0, ref.stderr);
  assert.equal(ref.stdout, plain.stdout.slice(0, ref.stdout.length - 1) + '\n');
  // sampled + seeded: reproducible
  const s1 = cli(['run', t.path, '-p', 'hello', '--n', '16', '--seed', '9', '--threads', '2', '--draft', d.path]);
  const s2 = cli(['run', t.path, '-p', 'hello', '--n', '16', '--seed', '9', '--threads', '2', '--draft', d.path]);
  assert.equal(s1.status, 0, s1.stderr); assert.equal(s1.stdout, s2.stdout);
  // validation
  const bad = cli(['run', t.path, '-p', 'hello', '--n', '4', '--draft', llama.path]);
  assert.notEqual(bad.status, 0); assert.match(bad.stderr, /vocabulary of 272 tokens but the target has 270/);
  assert.match(cli(['run', t.path, '-p', 'hello', '--draft', d.path, '--draft-k', '0']).stderr, /--draft-k must be a positive integer/);
  assert.match(cli(['run', t.path, '-p', 'hello', '--draft-k', '4']).stderr, /need --draft/);
  assert.match(cli(['run', t.path, '-p', 'hello', '--draft', d.path, '--draft-threads', 'x']).stderr, /expected a number|positive integer/);
  assert.match(cli(['run', t.path, '-p', 'hello', '--draft', d.path, '--batch', '1', '--threads', '2']).stderr, /at least 2 tokens per pass/);
});

test('CLI chat --draft: scripted session with /reset; every reply reports speculative stats', () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const r = cli(['chat', t.path, '--n', '8', '--temp', '0', '--threads', '2', '--draft', d.path, '--draft-threads', '1'], 'hi\nagain\n/reset\nonce more\n/exit\n');
  assert.equal(r.status, 0, r.stderr);
  assert.equal((r.stderr.match(/generated 8 tok/g) ?? []).length, 3);
  assert.equal((r.stderr.match(/speculative k=3/g) ?? []).length, 3);
});

test('CLI bench --draft: baseline vs speculative table with acceptance statistics; greedy output identical', () => {
  const t = tinyFile(M.target); const d = tinyFile(M.noisy);
  const r = cli(['bench', t.path, '--threads', '2', '--n', '20', '--prompt-tokens', '16', '--draft', d.path, '--draft-k', '2,4', '--draft-threads', '1']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /plain decode tok\/s \| speculative tok\/s \| speed-up \| accepted \| tok\/step \| draft ms\/step \| verify ms\/step \| output same as plain/);
  const rows = r.stdout.split('\n').filter((l) => /^\| (wasm-simd|js) \| 2 \| 1 \| [24] \|/.test(l));
  assert.equal(rows.length, 2);
  for (const row of rows) { assert.match(row, /\| yes \|$/); assert.match(row, /\d+% \|/); }
  assert.match(r.stdout, /greedy output identical across all configurations: yes/);
  const dflt = cli(['bench', t.path, '--threads', '1', '--n', '8', '--prompt-tokens', '8']);
  assert.equal(dflt.status, 0, dflt.stderr);
  assert.doesNotMatch(dflt.stdout, /speculative/);
});

test('Engine class is exported and loadModel returns one', async () => {
  const eng = await loadModel(tinyFile(M.target).path, { ctx: 64, threads: 1 });
  try { assert.ok(eng instanceof Engine); assert.equal(readGGUF(eng.path).path, eng.path); } finally { await eng.close(); }
});
