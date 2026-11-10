import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../../tools/make-tiny-gguf.js';
import { startApp } from '../../app/main.js';

/** Start the app in a temp data folder with one synthetic model; returns helpers for calling the API and reading events. */
export async function startTestApp({ models = [{ name: 'tiny-llama.gguf', arch: 'llama' }], port = 0, ctx = 512, dir: existingDir = null, catalog, retryDelayMs = 20, preload = false } = {}) {
  const dir = existingDir ?? fs.mkdtempSync(path.join(os.tmpdir(), 'zerochat-test-'));
  fs.mkdirSync(path.join(dir, 'models'), { recursive: true });
  if (!existingDir) for (const m of models) {
    if (m.raw) fs.writeFileSync(path.join(dir, 'models', m.name), m.raw);
    else fs.writeFileSync(path.join(dir, 'models', m.name), makeTinyModel({ arch: m.arch ?? 'llama', quant: 'q8_0', seed: 4, ctx }).buffer);
  }
  const app = await startApp({ dataDir: dir, port, quiet: true, catalog, retryDelayMs, preload });
  const base = `http://127.0.0.1:${app.port}`;
  const call = async (name, params = {}, { token = app.token, headers = {} } = {}) => {
    const res = await fetch(`${base}/api/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Token': token, ...headers }, body: JSON.stringify(params) });
    const body = await res.json();
    if (!res.ok) throw Object.assign(new Error(body.error ?? `HTTP ${res.status}`), { status: res.status });
    return body.result;
  };
  /** Collect SSE events of the run until `done` or `error` arrives (or timeout). */
  const events = async function* (signal) {
    const res = await fetch(`${base}/api/events?t=${app.token}`, { signal });
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = /^event: (.*)$/m.exec(block)?.[1]; const data = /^data: (.*)$/m.exec(block)?.[1];
        if (ev) yield { type: ev, data: data ? JSON.parse(data) : null };
      }
    }
  };
  /** Run a send_message and return all events of that run. */
  const runChat = async (model, messages, { method = 'send_message', after, extra = {} } = {}) => {
    const ctl = new AbortController(); const out = []; let started;
    const ready = new Promise((r) => { started = r; });
    const reader = (async () => {
      try { for await (const ev of events(ctl.signal)) { if (ev.type === 'hello') started(); else { out.push(ev); if (after) await after(ev, out); if (ev.type === 'done' || ev.type === 'error') break; } } } catch (e) { if (e.name !== 'AbortError') throw e; }
    })();
    await ready;
    await call(method, { model, messages, ...extra });
    await Promise.race([reader, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for events')), 30000))]);
    ctl.abort();
    return out;
  };
  return { app, dir, base, call, events, runChat, async close({ keep = false } = {}) { await app.close(); if (!keep) fs.rmSync(dir, { recursive: true, force: true }); } };
}
