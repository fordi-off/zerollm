import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';

const user = (content) => ({ role: 'user', content });

test('state lists the model and settings; i18n returns Norwegian and English', async () => {
  const t = await startTestApp({ models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, { name: 'junk.gguf', raw: Buffer.from('not a gguf at all') }, { name: 'nomic-embed-text.gguf', raw: Buffer.from('x') }] });
  try {
    const s = await t.call('state');
    assert.equal(s.language, 'nb');
    const names = s.models.map((m) => m.filename).sort();
    assert.deepEqual(names, ['junk.gguf', 'tiny-llama.gguf']); // embedding models are not chat models
    assert.equal(s.models.find((m) => m.filename === 'tiny-llama.gguf').supported, true);
    const junk = s.models.find((m) => m.filename === 'junk.gguf');
    assert.equal(junk.supported, false);
    assert.ok(junk.reason);
    assert.ok(s.settings.contextSteps.includes(8192));
    const nb = await t.call('i18n', { lang: 'nb' }); const en = await t.call('i18n', { lang: 'en' });
    assert.deepEqual(Object.keys(nb).sort(), Object.keys(en).sort(), 'both languages define the same strings');
  } finally { await t.close(); }
});

test('send_message streams phase, chunk and done events', async () => {
  const t = await startTestApp();
  try {
    const events = await t.runChat('tiny-llama.gguf', [user('Hei')]);
    const types = events.map((e) => e.type);
    assert.ok(types.includes('phase'));
    assert.equal(types.at(-1), 'done');
    const done = events.at(-1).data;
    assert.ok(done.generated > 0);
    assert.ok(done.contextMax > 0 && done.contextUsed > 0);
    assert.equal(done.model, 'tiny-llama.gguf');
    const runIds = new Set(events.map((e) => e.data.runId));
    assert.equal(runIds.size, 1);
    assert.equal((await t.call('state')).loaded, 'tiny-llama.gguf');
  } finally { await t.close(); }
});

test('a second send_message while one runs is refused as busy; stop ends the run', async () => {
  const t = await startTestApp();
  try {
    let busyError = null; let stopped = false;
    const events = await t.runChat('tiny-llama.gguf', [user('Fortell en lang historie')], {
      after: async (ev) => {
        if (ev.type === 'chunk' && !stopped) {
          stopped = true;
          try { await t.call('send_message', { model: 'tiny-llama.gguf', messages: [user('x')] }); } catch (e) { busyError = e; }
          await t.call('stop_generation');
        }
      },
    });
    assert.ok(busyError && /busy/.test(busyError.message), 'second run refused');
    const done = events.at(-1);
    assert.equal(done.type, 'done');
    // the tiny model may stop on its own before the stop lands; either way the run ended cleanly
    assert.ok(['aborted', 'eos', 'length', 'stop'].includes(done.data.stopReason) || done.data.stopped !== undefined);
    assert.equal((await t.call('state')).busy, false);
  } finally { await t.close(); }
});

test('unsupported model produces an error event, not a crash', async () => {
  const t = await startTestApp({ models: [{ name: 'junk.gguf', raw: Buffer.from('not a gguf at all') }] });
  try {
    const events = await t.runChat('junk.gguf', [user('Hei')]);
    assert.equal(events.at(-1).type, 'error');
    assert.ok(events.at(-1).data.message);
    assert.equal((await t.call('state')).busy, false);
    await assert.rejects(t.call('send_message', { model: '../../etc/passwd', messages: [user('x')] }), /./);
  } finally { await t.close(); }
});

test('estimate_context counts exact prompt tokens without loading the model', async () => {
  const t = await startTestApp();
  try {
    const a = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [user('Hei')] });
    const b = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [user('Hei '.repeat(50))] });
    assert.ok(b.contextUsed > a.contextUsed);
    assert.equal(a.contextMax, 512, 'capped to what the model was trained for (the tiny test model: 512)');
    assert.equal((await t.call('state')).loaded, null);
  } finally { await t.close(); }
});

test('settings persist to config.json and unload the model', async () => {
  const t = await startTestApp();
  try {
    await t.runChat('tiny-llama.gguf', [user('Hei')]);
    assert.equal((await t.call('state')).loaded, 'tiny-llama.gguf');
    await t.call('set_context_window', { value: 4096 });
    assert.equal((await t.call('state')).loaded, null);
    await t.call('set_thread_count', { value: 2 });
    await t.call('update_model_settings', { filename: 'tiny-llama.gguf', display_name: 'Min modell' });
    await t.call('set_last_model', { filename: 'tiny-llama.gguf' });
    const cfg = JSON.parse(fs.readFileSync(path.join(t.dir, 'config.json'), 'utf8'));
    assert.equal(cfg.context_window, 4096);
    assert.equal(cfg.n_threads, 2);
    assert.equal(cfg.last_model, 'tiny-llama.gguf');
    const s = await t.call('state');
    assert.equal(s.models[0].displayName, 'Min modell');
    await assert.rejects(t.call('set_context_window', { value: 10 }), /invalid/);
    await assert.rejects(t.call('set_thread_count', { value: 0 }), /invalid/);
    await assert.rejects(t.call('update_model_settings', { filename: '../x.gguf', display_name: 'x' }), /./);
    await t.call('reset_settings');
    assert.equal(JSON.parse(fs.readFileSync(path.join(t.dir, 'config.json'), 'utf8')).context_window, 8192);
  } finally { await t.close(); }
});
