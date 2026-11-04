import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { makeTinyModel } from '../../tools/make-tiny-gguf.js';
import { startTestApp } from './helpers.js';

const GOOD = makeTinyModel({ arch: 'llama', quant: 'q8_0', seed: 9, ctx: 512 }).buffer;

/** Local file server: Range support, optional redirect, optional cut-off after N bytes on the first request, slow mode. */
function fileServer(routes) {
  const log = [];
  const server = http.createServer((req, res) => {
    const r = routes[req.url];
    log.push({ url: req.url, range: req.headers.range ?? null });
    if (!r) { res.writeHead(404); return res.end(); }
    if (r.redirect) { res.writeHead(302, { Location: r.redirect }); return res.end(); }
    const body = r.body;
    let start = 0; let status = 200; const headers = { 'Content-Type': 'application/octet-stream' };
    const m = /bytes=(\d+)-/.exec(req.headers.range ?? '');
    if (m && r.ranges !== false) { start = Number(m[1]); status = 206; headers['Content-Range'] = `bytes ${start}-${body.length - 1}/${body.length}`; }
    headers['Content-Length'] = body.length - start;
    res.writeHead(status, headers);
    const cutAt = r.cutFirst && log.filter((l) => l.url === req.url).length === 1 ? start + r.cutFirst : null;
    let pos = start;
    const step = () => {
      if (res.destroyed) return;
      if (cutAt !== null && pos >= cutAt) return res.destroy();
      if (pos >= body.length) return res.end();
      const n = Math.min(r.chunk ?? 4096, body.length - pos);
      res.write(body.subarray(pos, pos + n)); pos += n;
      if (r.delay) setTimeout(step, r.delay); else setImmediate(step);
    };
    step();
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, log, base: `http://127.0.0.1:${server.address().port}`, close: () => new Promise((r) => { server.close(r); server.closeAllConnections(); }) })));
}

const entry = (id, base, p, size = GOOD.length) => ({ id, filename: `${id}.gguf`, displayName: id, sizeBytes: size, ramGB: 0.1, url: `${base}${p}` });

test('download: follows a redirect, writes .part, verifies and installs; the model then shows up and chats', async () => {
  const fs_ = await fileServer({ '/a.gguf': { redirect: '/real.gguf' }, '/real.gguf': { body: GOOD } });
  const t = await startTestApp({ models: [], catalog: [entry('dl-ok', fs_.base, '/a.gguf')] });
  try {
    const events = []; const ctl = new AbortController();
    (async () => { for await (const ev of t.events(ctl.signal)) if (ev.type === 'download') events.push(ev.data); })().catch(() => {});
    const cat = await t.call('catalog');
    assert.equal(cat.entries[0].installed, false);
    await t.call('download_start', { ids: ['dl-ok'] });
    for (let i = 0; i < 200 && (await t.call('download_status')).current; i++) await new Promise((r) => setTimeout(r, 25));
    await new Promise((r) => setTimeout(r, 100));
    ctl.abort();
    assert.ok(fs.existsSync(path.join(t.dir, 'models', 'dl-ok.gguf')));
    assert.equal(fs.existsSync(path.join(t.dir, 'models', 'dl-ok.gguf.part')), false);
    assert.equal(fs.statSync(path.join(t.dir, 'models', 'dl-ok.gguf')).size, GOOD.length);
    assert.ok(events.some((e) => e.current?.received > 0), 'progress events were sent');
    assert.equal((await t.call('catalog')).entries[0].installed, true);
    const state = await t.call('state');
    assert.equal(state.models.find((m) => m.filename === 'dl-ok.gguf').supported, true);
    const run = await t.runChat('dl-ok.gguf', [{ role: 'user', content: 'Hei' }]);
    assert.equal(run.at(-1).type, 'done');
  } finally { await t.close(); await fs_.close(); }
});

test('download: a dropped connection is resumed with a Range request', async () => {
  const fs_ = await fileServer({ '/m.gguf': { body: GOOD, cutFirst: 20000 } });
  const t = await startTestApp({ models: [], catalog: [entry('dl-resume', fs_.base, '/m.gguf')] });
  try {
    await t.call('download_start', { ids: ['dl-resume'] });
    for (let i = 0; i < 400 && (await t.call('download_status')).current; i++) await new Promise((r) => setTimeout(r, 25));
    const dest = path.join(t.dir, 'models', 'dl-resume.gguf');
    assert.ok(fs.existsSync(dest), JSON.stringify(await t.call('download_status')));
    assert.ok(GOOD.equals(fs.readFileSync(dest)), 'file is byte-identical after resuming');
    const ranges = fs_.log.map((l) => l.range).filter(Boolean);
    assert.ok(ranges.length >= 1 && /^bytes=\d+-$/.test(ranges[0]) && ranges[0] !== 'bytes=0-', `resumed with ${ranges}`);
  } finally { await t.close(); await fs_.close(); }
});

test('download: continues an existing .part file; a server that ignores Range restarts cleanly', async () => {
  const fs_ = await fileServer({ '/r.gguf': { body: GOOD }, '/n.gguf': { body: GOOD, ranges: false } });
  const t = await startTestApp({ models: [], catalog: [entry('dl-part', fs_.base, '/r.gguf'), entry('dl-norange', fs_.base, '/n.gguf')] });
  try {
    const models = path.join(t.dir, 'models');
    fs.writeFileSync(path.join(models, 'dl-part.gguf.part'), GOOD.subarray(0, 30000));
    fs.writeFileSync(path.join(models, 'dl-norange.gguf.part'), GOOD.subarray(0, 30000));
    assert.equal((await t.call('catalog')).entries[0].partialBytes, 30000);
    await t.call('download_start', { ids: ['dl-part', 'dl-norange'] });
    for (let i = 0; i < 400 && (await t.call('download_status')).current; i++) await new Promise((r) => setTimeout(r, 25));
    await new Promise((r) => setTimeout(r, 100));
    assert.ok(GOOD.equals(fs.readFileSync(path.join(models, 'dl-part.gguf'))));
    assert.ok(GOOD.equals(fs.readFileSync(path.join(models, 'dl-norange.gguf'))), 'no duplicated prefix when the server sends the whole file');
    assert.equal(fs_.log.find((l) => l.url === '/r.gguf').range, 'bytes=30000-');
  } finally { await t.close(); await fs_.close(); }
});

test('download: 404, wrong content and cancel are reported; cancel keeps the partial file; unknown ids are refused', async () => {
  const slow = Buffer.concat([GOOD, GOOD, GOOD]);
  const fs_ = await fileServer({ '/bad.gguf': { body: Buffer.from('<html>not a model</html>'.repeat(50)) }, '/slow.gguf': { body: slow, chunk: 2048, delay: 5 } });
  const t = await startTestApp({ models: [], catalog: [entry('dl-404', fs_.base, '/missing.gguf'), entry('dl-bad', fs_.base, '/bad.gguf', 1200), entry('dl-slow', fs_.base, '/slow.gguf', slow.length)] });
  try {
    await assert.rejects(t.call('download_start', { ids: ['nope'] }), /unknown model/);
    await t.call('download_start', { ids: ['dl-404', 'dl-bad'] });
    for (let i = 0; i < 400 && (await t.call('download_status')).current !== null || (await t.call('download_status')).queue.length; i++) await new Promise((r) => setTimeout(r, 25));
    await new Promise((r) => setTimeout(r, 100));
    const st = await t.call('download_status');
    assert.match(st.failed['dl-404'], /404/);
    assert.match(st.failed['dl-bad'], /GGUF/);
    assert.equal(fs.existsSync(path.join(t.dir, 'models', 'dl-bad.gguf')), false);
    assert.equal(fs.existsSync(path.join(t.dir, 'models', 'dl-bad.gguf.part')), false, 'garbage is not kept');

    await t.call('download_start', { ids: ['dl-slow'] });
    for (let i = 0; i < 200 && !((await t.call('download_status')).current?.received > 20000); i++) await new Promise((r) => setTimeout(r, 10));
    await t.call('download_cancel');
    for (let i = 0; i < 200 && (await t.call('download_status')).current; i++) await new Promise((r) => setTimeout(r, 10));
    const part = path.join(t.dir, 'models', 'dl-slow.gguf.part');
    assert.ok(fs.existsSync(part) && fs.statSync(part).size > 0 && fs.statSync(part).size < slow.length, 'partial file kept for resuming');
    assert.equal(fs.existsSync(path.join(t.dir, 'models', 'dl-slow.gguf')), false);
    assert.equal((await t.call('catalog')).entries.find((e) => e.id === 'dl-slow').partialBytes, fs.statSync(part).size);
  } finally { await t.close(); await fs_.close(); }
});

test('delete_model removes the file and forgets its settings; kv precision is validated and per model', async () => {
  const t = await startTestApp();
  try {
    await t.call('update_model_settings', { filename: 'tiny-llama.gguf', kv: 'f32' });
    assert.equal((await t.call('state')).models[0].kv, 'f32');
    await assert.rejects(t.call('update_model_settings', { filename: 'tiny-llama.gguf', kv: 'f64' }), /invalid kv/);
    await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'Hei' }]);
    assert.equal((await t.call('state')).loaded, 'tiny-llama.gguf');
    await t.call('update_model_settings', { filename: 'tiny-llama.gguf', kv: 'f16' });
    assert.equal((await t.call('state')).loaded, null, 'changing the KV precision unloads the model');
    await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'Hei' }]);
    await assert.rejects(t.call('delete_model', { filename: '../x.gguf' }), /./);
    await t.call('delete_model', { filename: 'tiny-llama.gguf' });
    const s = await t.call('state');
    assert.equal(s.models.length, 0);
    assert.equal(s.loaded, null);
    assert.equal(fs.existsSync(path.join(t.dir, 'models', 'tiny-llama.gguf')), false);
  } finally { await t.close(); }
});
