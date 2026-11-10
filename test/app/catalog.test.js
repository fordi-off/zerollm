import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { CATALOG } from '../../app/catalog.js';
import { makeTinyQwen35 } from '../../tools/make-tiny-qwen35-gguf.js';
import { startTestApp } from './helpers.js';

test('the shipped download catalog is well formed: unique ids and file names, plain .gguf names, https Hugging Face links, sane sizes', () => {
  assert.ok(CATALOG.length >= 5);
  assert.equal(new Set(CATALOG.map((e) => e.id)).size, CATALOG.length);
  assert.equal(new Set(CATALOG.map((e) => e.filename)).size, CATALOG.length);
  for (const e of CATALOG) {
    assert.match(e.filename, /^[A-Za-z0-9._-]+\.gguf$/, `${e.id}: file name`);
    const u = new URL(e.url);
    assert.equal(u.protocol, 'https:'); assert.equal(u.hostname, 'huggingface.co');
    assert.ok(u.pathname.endsWith(`/${e.filename}`), `${e.id}: the link ends with the file name`);
    assert.ok(e.sizeBytes > 1e8 && e.sizeBytes < 2e10, `${e.id}: size`);
    assert.ok(e.ramGB > 0 && e.displayName, `${e.id}: ram and name`);
  }
});

test('the Qwen3.5 4B entry: the exact file the user tested; an installed copy is recognised, a missing one is offered', async () => {
  const e = CATALOG.find((x) => x.id === 'qwen3.5-4b');
  assert.ok(e, 'catalog has qwen3.5-4b');
  assert.equal(e.filename, 'Qwen3.5-4B-Q4_K_M.gguf');
  assert.ok(e.url.endsWith('/Qwen3.5-4B-Q4_K_M.gguf'));
  const t = await startTestApp({ models: [{ name: e.filename, arch: 'qwen35' }] }); // uses the real catalog
  try {
    const row = (await t.call('catalog')).entries.find((x) => x.id === 'qwen3.5-4b');
    assert.equal(row.installed, true);
    assert.equal(row.fits, true);
  } finally { await t.close(); }
  const t2 = await startTestApp({ models: [] });
  try {
    const row = (await t2.call('catalog')).entries.find((x) => x.id === 'qwen3.5-4b');
    assert.equal(row.installed, false);
    assert.equal(row.displayName, 'Qwen3.5 4B');
  } finally { await t2.close(); }
});

test('download -> model list -> chat with a qwen35 file: the catalog entry installs under its real file name, is supported with a thinking switch, and answers', async () => {
  const body = makeTinyQwen35({ quant: 'q8_0', seed: 3 }).buffer;
  const server = http.createServer((req, res) => { res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': body.length }); res.end(body); });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const real = CATALOG.find((x) => x.id === 'qwen3.5-4b');
  const entry = { ...real, sizeBytes: body.length, ramGB: 0.1, url: `http://127.0.0.1:${server.address().port}/${real.filename}` };
  const t = await startTestApp({ models: [], catalog: [entry] });
  try {
    await t.call('download_start', { ids: [entry.id] });
    for (let i = 0; i < 200 && (await t.call('download_status')).current; i++) await new Promise((r) => setTimeout(r, 25));
    await new Promise((r) => setTimeout(r, 100));
    assert.ok(fs.existsSync(path.join(t.dir, 'models', real.filename)));
    const m = (await t.call('list_models')).find((x) => x.filename === real.filename);
    assert.equal(m.arch, 'qwen35'); assert.equal(m.supported, true); assert.equal(m.thinking, true);
    assert.equal((await t.call('catalog')).entries[0].installed, true);
    for (const think of [false, true]) {
      const run = await t.runChat(real.filename, [{ role: 'user', content: 'Hei' }], { extra: { think } });
      assert.equal(run.at(-1).type, 'done', `think=${think}`);
    }
  } finally { await t.close(); await new Promise((r) => server.close(r)); }
});
