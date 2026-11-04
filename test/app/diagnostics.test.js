import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { createLogger, resolvePaths } from '../../app/config.js';

test('diagnostics reports system, features, models and the log tail', async () => {
  const t = await startTestApp({ models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, { name: 'junk.gguf', raw: Buffer.from('nope') }] });
  try {
    await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'Hei' }]);
    const d = await t.call('diagnostics');
    assert.match(d.system.node, /^v\d+/);
    assert.equal(typeof d.features.simd, 'boolean');
    assert.equal(d.app.dataFolder, t.dir);
    assert.equal(d.app.loadedModel, 'tiny-llama.gguf');
    assert.deepEqual(d.models.map((m) => [m.filename, m.supported]), [['junk.gguf', false], ['tiny-llama.gguf', true]]);
    assert.match(d.logTail, /model ready: tiny-llama\.gguf/);
    assert.ok(Array.isArray(d.problems));
  } finally { await t.close(); }
});

test('a prompt that cannot fit the context gives a Norwegian explanation, not an English exception', async () => {
  const t = await startTestApp(); // tiny model with a 512-token context
  try {
    const ev = await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'ord '.repeat(400) }]);
    assert.equal(ev.at(-1).type, 'error');
    assert.match(ev.at(-1).data.message, /for lang for kontekstvinduet/);
  } finally { await t.close(); }
});

test('the log is rotated when it grows past 2 MB', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zc-log-'));
  try {
    const paths = resolvePaths(dir);
    fs.writeFileSync(paths.logPath, 'x'.repeat(2_100_000));
    const log = createLogger(paths, { quiet: true });
    log.info('hello');
    assert.ok(fs.existsSync(`${paths.logPath}.old`));
    assert.match(fs.readFileSync(paths.logPath, 'utf8'), /hello/);
    assert.ok(fs.statSync(paths.logPath).size < 1000);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
