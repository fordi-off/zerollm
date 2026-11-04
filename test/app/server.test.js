import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { startTestApp } from './helpers.js';

// Raw request so tests can forge Host / Origin headers (fetch forbids that).
function raw(port, { method = 'GET', path = '/', headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, method, path, headers }, (res) => {
      const chunks = []; res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString() }));
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

test('API needs the token; wrong token, GET and unknown methods are refused', async () => {
  const t = await startTestApp();
  try {
    const h = { Host: `127.0.0.1:${t.app.port}`, 'Content-Type': 'application/json' };
    assert.equal((await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: h, body: '{}' })).status, 401);
    assert.equal((await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: { ...h, 'X-Token': 'nope' }, body: '{}' })).status, 401);
    assert.equal((await raw(t.app.port, { method: 'GET', path: '/api/ping', headers: { ...h, 'X-Token': t.app.token } })).status, 405);
    const unknown = await raw(t.app.port, { method: 'POST', path: '/api/__proto__', headers: { ...h, 'X-Token': t.app.token }, body: '{}' });
    assert.equal(unknown.status, 404);
    const bad = await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: { ...h, 'X-Token': t.app.token }, body: '[1]' });
    assert.equal(bad.status, 400);
    assert.deepEqual(await t.call('ping'), { ok: true });
    await assert.rejects(t.call('ping', {}, { token: 'x' }), (e) => e.status === 401);
    assert.equal((await raw(t.app.port, { path: '/api/events', headers: { Host: `127.0.0.1:${t.app.port}` } })).status, 401);
  } finally { await t.close(); }
});

test('Host and Origin are checked (DNS rebinding / cross-site requests)', async () => {
  const t = await startTestApp();
  try {
    const good = { Host: `127.0.0.1:${t.app.port}`, 'X-Token': t.app.token, 'Content-Type': 'application/json' };
    assert.equal((await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: { ...good, Host: 'evil.example' }, body: '{}' })).status, 403);
    assert.equal((await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: { ...good, Origin: 'http://evil.example' }, body: '{}' })).status, 403);
    assert.equal((await raw(t.app.port, { method: 'POST', path: '/api/ping', headers: { ...good, Origin: `http://127.0.0.1:${t.app.port}` }, body: '{}' })).status, 200);
    assert.equal((await raw(t.app.port, { path: '/', headers: { Host: 'evil.example' } })).status, 403);
  } finally { await t.close(); }
});

test('static files: index served with CSP; traversal and dotfiles refused', async () => {
  const t = await startTestApp();
  try {
    const h = { Host: `127.0.0.1:${t.app.port}` };
    const index = await raw(t.app.port, { path: '/', headers: h });
    assert.equal(index.status, 200);
    assert.match(index.headers['content-type'], /text\/html/);
    assert.match(index.headers['content-security-policy'], /default-src 'self'/);
    assert.equal((await raw(t.app.port, { path: '/../config.js', headers: h })).status, 404);
    assert.equal((await raw(t.app.port, { path: '/%2e%2e/config.js', headers: h })).status, 404);
    assert.equal((await raw(t.app.port, { path: '/.hidden', headers: h })).status, 404);
    assert.equal((await raw(t.app.port, { path: '/nonexistent.js', headers: h })).status, 404);
  } finally { await t.close(); }
});

test('event stream says hello and delivers emitted events', async () => {
  const t = await startTestApp();
  try {
    const ctl = new AbortController(); const got = [];
    const it = t.events(ctl.signal);
    const first = await it.next();
    assert.equal(first.value.type, 'hello');
    t.app.app.emit('ping-test', { a: 1 });
    const second = await it.next();
    got.push(second.value);
    ctl.abort();
    assert.deepEqual(got, [{ type: 'ping-test', data: { a: 1 } }]);
  } finally { await t.close(); }
});
