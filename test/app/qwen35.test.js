import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { loadTokenizer } from '../../src/index.js';
import { buildMessages } from '../../app/chat-core.js';
import { splitThinking } from '../../app/public/js/thinking.js';
import { startTestApp } from './helpers.js';

// Qwen3.5 (hybrid Gated DeltaNet, arch qwen35) in the app: runs on the worker pool, "Tenk først" works as for Qwen3 (the thinking
// prompt already ends with `<think>\n`, so the server puts the opening tag in front of the stream), the prompt stays append-only,
// and the system-prompt state (KV of the attention layers + the recurrent state) is cached on disk.
const U = (content) => ({ role: 'user', content }); const A = (content) => ({ role: 'assistant', content });
const common = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };
const MODELS = [{ name: 'tiny-qwen35.gguf', arch: 'qwen35' }, { name: 'tiny-qwen3.gguf', arch: 'qwen3' }, { name: 'tiny-llama.gguf', arch: 'llama' }];
const text = (ev) => ev.filter((e) => e.type === 'chunk').map((e) => e.data.text).join('');

test('app: a qwen35 model is listed as supported and with a thinking switch (like Qwen3, unlike Llama)', async () => {
  const t = await startTestApp({ ctx: 4096, models: MODELS });
  try {
    const models = await t.call('list_models');
    const by = Object.fromEntries(models.map((m) => [m.filename, m]));
    assert.equal(by['tiny-qwen35.gguf'].arch, 'qwen35');
    assert.equal(by['tiny-qwen35.gguf'].supported, true);
    assert.equal(by['tiny-qwen35.gguf'].reason, null);
    assert.equal(by['tiny-qwen35.gguf'].thinking, true);
    assert.equal(by['tiny-qwen3.gguf'].thinking, true);
    assert.equal(by['tiny-llama.gguf'].thinking, false);
    assert.ok(by['tiny-qwen35.gguf'].ramNeed > 0);
  } finally { await t.close(); }
});

test('app: the think switch changes the qwen35 prompt (open <think> vs empty think block) and the meter counts the right one', async () => {
  const t = await startTestApp({ ctx: 4096, models: MODELS });
  try {
    const tools = loadTokenizer(path.join(t.dir, 'models', 'tiny-qwen35.gguf'));
    const off = tools.promptIds(buildMessages([U('Hei')]), { think: false });
    const on = tools.promptIds(buildMessages([U('Hei')]), { think: true });
    assert.ok(tools.tok.decode(off, { skipControl: false }).endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'));
    assert.ok(tools.tok.decode(on, { skipControl: false }).endsWith('<|im_start|>assistant\n<think>\n'));
    const e0 = await t.call('estimate_context', { model: 'tiny-qwen35.gguf', messages: [U('Hei')], think: false });
    const e1 = await t.call('estimate_context', { model: 'tiny-qwen35.gguf', messages: [U('Hei')], think: true });
    assert.equal(e0.contextUsed, off.length);
    assert.equal(e1.contextUsed, on.length);
    assert.ok(e1.contextUsed < e0.contextUsed);
    const llama0 = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [U('Hei')], think: false });
    const llama1 = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [U('Hei')], think: true });
    assert.equal(llama0.contextUsed, llama1.contextUsed, 'the switch is ignored by models without a thinking mode');
  } finally { await t.close(); }
});

test('app: a qwen35 reply streams as <think>..</think>answer when thinking (the engine puts the prompt\'s opening tag in front), plain otherwise', async () => {
  const t = await startTestApp({ ctx: 4096, models: MODELS });
  try {
    const on = await t.runChat('tiny-qwen35.gguf', [U('Hei')], { extra: { think: true } });
    assert.equal(on.at(-1).type, 'done');
    assert.equal(on.at(-1).data.thinking, true);
    const chunks = on.filter((e) => e.type === 'chunk');
    assert.ok(chunks.length > 1);
    assert.equal(chunks[0].data.text, '<think>\n', 'the first chunk is the opening tag the prompt already ended with');
    assert.equal(splitThinking(text(on)).state === 'none', false, 'the page sees a thinking block (open or closed), not an answer');
    const off = await t.runChat('tiny-qwen35.gguf', [U('Hei')], { extra: { think: false } });
    assert.equal(off.at(-1).data.thinking, false);
    assert.ok(!text(off).startsWith('<think>\n'));
    // the switch is ignored for a model without a thinking mode, as before
    const llama = await t.runChat('tiny-llama.gguf', [U('Hei')], { extra: { think: true } });
    assert.equal(llama.at(-1).data.thinking, false);
  } finally { await t.close(); }
});

test('app: qwen35 chats are append-only: turn N+1 reuses everything the model read for turn N (state, not re-reading), with and without thinking', async () => {
  // prompt level
  const t = await startTestApp({ ctx: 4096, models: MODELS });
  try {
    const tools = loadTokenizer(path.join(t.dir, 'models', 'tiny-qwen35.gguf'));
    const ids = (h, think) => tools.promptIds(buildMessages(h), { think });
    const q1 = 'Hvem har ansvar for barnehager i kommunene?'; const a1 = 'Kommunen har ansvar.';
    const p1 = ids([U(q1)], false); const p2 = ids([U(q1), A(a1), U('Og hvem betaler?')], false);
    assert.equal(common(p1, p2), p1.length, 'thinking off: the empty think block the model wrote stays in the history');
    const p3 = ids([U(q1), A(a1), U('Og hvem betaler?'), A('Staten.'), U('Takk')], false);
    assert.equal(common(p2, p3), p2.length);
    // end to end through the engine: the second message finds the first one's state (cachedTokens), no matter how it is rolled back
    const e1 = await t.runChat('tiny-qwen35.gguf', [U(q1)]);
    const d1 = e1.at(-1).data;
    assert.equal(e1.at(-1).type, 'done');
    assert.ok(d1.generated > 0);
    const e2 = await t.runChat('tiny-qwen35.gguf', [U(q1), A(text(e1)), U('Og hvem betaler?')]);
    const d2 = e2.at(-1).data;
    assert.equal(e2.at(-1).type, 'done');
    assert.ok(d2.cachedTokens >= d1.promptTokens, `turn 2 reuses at least the whole prompt of turn 1 (${d2.cachedTokens} >= ${d1.promptTokens})`);
    assert.ok(d2.promptTokens > d1.promptTokens);
    // thinking on: the history keeps the answer only (like Qwen3); the second turn still resumes from the first turn's prompt
    const th1 = await t.runChat('tiny-qwen35.gguf', [U('Hva er 2+3?')], { extra: { think: true } });
    const raw = text(th1);
    const msg = splitThinking(raw);
    const th2 = await t.runChat('tiny-qwen35.gguf', [U('Hva er 2+3?'), A(msg.answer), U('Og 3+4?')], { extra: { think: true } });
    assert.equal(th2.at(-1).type, 'done');
    assert.ok(th2.at(-1).data.cachedTokens >= 1, 'the shared prefix (system prompt + first question) is reused');
  } finally { await t.close(); }
});

test('app: the system prompt state of a qwen35 model is saved on first use and restored by a new process (KV + recurrent state)', async () => {
  const first = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-qwen35.gguf', arch: 'qwen35' }] });
  const dir = first.dir;
  try {
    const ev = await first.runChat('tiny-qwen35.gguf', [U('Hei')]);
    const done = ev.at(-1).data;
    assert.equal(ev.at(-1).type, 'done');
    assert.equal(done.prefixCache, 'saved');
    assert.ok(done.cachedTokens >= 40, `the system prompt counted as cached tokens (${done.cachedTokens})`);
    const files = fs.readdirSync(path.join(dir, 'cache')).filter((f) => f.endsWith('.kv'));
    assert.equal(files.length, 1);
    assert.equal(fs.readFileSync(path.join(dir, 'cache', files[0])).subarray(0, 8).toString('latin1'), 'ZLLMKV02');
  } finally { await first.close({ keep: true }); }

  const second = await startTestApp({ ctx: 4096, dir });
  try {
    const ev = await second.runChat('tiny-qwen35.gguf', [U('Hei')]);
    const done = ev.at(-1).data;
    assert.equal(done.prefixCache, 'hit');
    assert.ok(done.cachedTokens >= 40);
    assert.ok(done.generated > 0);
  } finally { await second.close(); }
});
