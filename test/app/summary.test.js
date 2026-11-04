import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { pickUtilityModel } from '../../app/summarizer.js';
import { chunkForSummary, summaryPrompt } from '../../app/chat-core.js';

const long = (label) => `${label} ` + 'ord '.repeat(30);
const conv = (n) => Array.from({ length: n }, (_, i) => ({ role: i % 2 === 0 ? 'user' : 'assistant', content: long(`melding ${i}`) }));

test('pickUtilityModel: configured one, else the smallest clearly-smaller supported model, else none', () => {
  const m = (filename, sizeBytes, supported = true) => ({ filename, sizeBytes, supported });
  const models = [m('big.gguf', 5e9), m('mid.gguf', 4e9), m('small.gguf', 1e9), m('tiny.gguf', 0.8e9), m('bad.gguf', 1e8, false)];
  assert.equal(pickUtilityModel(models, 'big.gguf', null), 'tiny.gguf');
  assert.equal(pickUtilityModel(models, 'big.gguf', 'mid.gguf'), 'mid.gguf');
  assert.equal(pickUtilityModel(models, 'big.gguf', 'bad.gguf'), 'tiny.gguf', 'unsupported choice ignored');
  assert.equal(pickUtilityModel(models, 'tiny.gguf', null), null, 'nothing clearly smaller than the smallest model');
  assert.equal(pickUtilityModel([m('a.gguf', 1e9)], 'a.gguf', null), null);
});

test('chunkForSummary splits into bounded passes and keeps order; summaryPrompt carries old summary and new text', () => {
  const msgs = conv(40);
  const passes = chunkForSummary(msgs, 2000);
  assert.ok(passes.length >= 3);
  assert.deepEqual(passes.flat(), msgs);
  for (const p of passes) assert.ok(p.reduce((n, m) => n + m.content.length + 12, 0) <= 2000 + 200);
  const p = summaryPrompt('GAMMEL', [{ role: 'user', content: 'Hei verden' }]);
  assert.match(p[1].content, /GAMMEL/);
  assert.match(p[1].content, /user: Hei verden/);
});

test('long chat: older messages are folded into a saved summary, newest stay verbatim; edits invalidate it', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }] });
  try {
    const history = conv(11);
    const id = (await t.call('save_chat', { name: 'Lang', model: 'tiny-llama.gguf', messages: history })).id;
    const sumFile = path.join(t.dir, 'chats', '_context_summaries', `${id}.json`);
    const plain = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: history });
    assert.equal(plain.compressed, false);

    const ev = await t.runChat('tiny-llama.gguf', history, { extra: { chatId: id } });
    assert.equal(ev.at(-1).type, 'done', JSON.stringify(ev.at(-1)));
    assert.ok(ev.some((e) => e.type === 'phase' && e.data.phase === 'summarizing'), 'a summarizing phase was announced');
    assert.equal(ev.at(-1).data.compressed, true);
    const saved = JSON.parse(fs.readFileSync(sumFile, 'utf8'));
    assert.equal(saved.summarized_count, 5, '11 messages - 6 kept verbatim');
    assert.ok(saved.summary.length > 0);

    const withSummary = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: history, chatId: id });
    assert.equal(withSummary.compressed, true);
    assert.notEqual(withSummary.contextUsed, plain.contextUsed, 'the prompt now holds the summary instead of the older messages (the tiny model writes a long garbage summary, so no size comparison)');

    // same history again: no new summarizing needed
    const ev2 = await t.runChat('tiny-llama.gguf', history, { extra: { chatId: id } });
    assert.equal(ev2.some((e) => e.type === 'phase' && e.data.phase === 'summarizing'), false);

    // history edited back to before the summarized part: the stale summary is not used
    const short = conv(3);
    const ev3 = await t.runChat('tiny-llama.gguf', short, { extra: { chatId: id } });
    assert.equal(ev3.at(-1).data.compressed, false);
  } finally { await t.close(); }
});

test('a configured utility model is loaded for the summary and closed again', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, { name: 'tiny-util.gguf', arch: 'llama' }] });
  try {
    const cfgPath = path.join(t.dir, 'config.json');
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8')); cfg.utility_model = 'tiny-util.gguf';
    fs.writeFileSync(cfgPath, JSON.stringify(cfg));
    await t.close({ keep: true });
    const t2 = await startTestApp({ ctx: 4096, dir: t.dir });
    try {
      const history = conv(11);
      const id = (await t2.call('save_chat', { name: 'Lang', model: 'tiny-llama.gguf', messages: history })).id;
      const ev = await t2.runChat('tiny-llama.gguf', history, { extra: { chatId: id } });
      assert.equal(ev.at(-1).type, 'done');
      assert.equal(ev.at(-1).data.compressed, true);
      assert.equal((await t2.call('state')).loaded, 'tiny-llama.gguf', 'only the chat model stays resident');
      const log = fs.readFileSync(path.join(t.dir, 'app.log'), 'utf8');
      assert.match(log, /summarizing with tiny-util\.gguf/);
    } finally { await t2.close(); }
  } catch (e) { fs.rmSync(t.dir, { recursive: true, force: true }); throw e; }
});
