// Embedding model front end: loadEmbedder(path, {threads}) -> { embed(texts), embedIds(idsList), close() }.
// Texts are tokenized (WordPiece), then spread over `threads` workers (one full copy of the weights each).
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { readGGUF } from '../gguf/reader.js';
import { embedConfig } from './config.js';
import { WordPieceTokenizer } from '../tokenizer/wordpiece.js';
import { probeWasmFeatures } from '../compute/wasm/features.js';

export const DEFAULT_MAX_SEQ = 512;

export async function loadEmbedder(path, { threads, maxSeq = DEFAULT_MAX_SEQ, kernels = 'auto', stripAccents } = {}) {
  const g = readGGUF(path);
  const cfg = embedConfig(g);
  const tok = WordPieceTokenizer.fromGGUF(g, stripAccents === undefined ? {} : { stripAccents });
  const seq = Math.min(maxSeq, cfg.nCtxTrain);
  const n = Math.max(1, threads ?? Math.max(1, Math.floor((os.availableParallelism?.() ?? os.cpus().length) / 2)));
  const kern = kernels === 'auto' ? (probeWasmFeatures().simd ? 'wasm' : 'js') : kernels;
  const tensors = {};
  for (const t of g.tensors) tensors[t.name] = { name: t.name, type: t.type, dims: t.dims, absOffset: t.absOffset, nBytes: t.nBytes, nElements: t.nElements };
  const url = new URL('./embed-worker.js', import.meta.url);
  let alive = null;
  const workers = []; const idle = []; const pending = new Map(); const queue = []; let nextId = 1; let closed = false; let failure = null;
  const t0 = performance.now();
  const dispatch = () => { while (queue.length && idle.length) { const w = idle.pop(); const job = queue.shift(); pending.get(job.id).worker = w; w.postMessage({ type: 'job', id: job.id, ids: job.ids }); } };
  const failAll = (e) => { failure = failure ?? e; for (const p of pending.values()) p.reject(e); pending.clear(); queue.length = 0; };
  const ready = [];
  for (let i = 0; i < n; i++) {
    const w = new Worker(url, { workerData: { cfg, tensors, path, maxSeq: seq, kernels: kern } });
    workers.push(w);
    ready.push(new Promise((resolve, reject) => {
      w.on('message', (m) => {
        if (m.type === 'ready') { idle.push(w); resolve(m); dispatch(); }
        else if (m.type === 'result') { const p = pending.get(m.id); pending.delete(m.id); idle.push(w); p.resolve(m.vec); dispatch(); }
        else if (m.type === 'jobError') { const p = pending.get(m.id); pending.delete(m.id); idle.push(w); p?.reject(new Error(m.message)); dispatch(); }
        else if (m.type === 'error') { const e = new Error(`embedding worker failed: ${m.message}`); failAll(e); reject(e); }
      });
      w.on('error', (e) => { failAll(e); reject(e); });
      w.on('exit', (code) => { if (!closed && code !== 0) { const e = new Error(`embedding worker exited (code ${code})`); failAll(e); reject(e); } });
      w.unref();
    }));
  }
  const keep = setInterval(() => {}, 1000);
  try { await Promise.all(ready); } catch (e) { await close(); throw e; } finally { clearInterval(keep); }

  // workers are unref'd (a forgotten close() must not hang the process), so keep the event loop alive while jobs are outstanding
  const syncAlive = () => { if (pending.size > 0 && !alive) alive = setInterval(() => {}, 1000); else if (pending.size === 0 && alive) { clearInterval(alive); alive = null; } };
  function run(ids) {
    if (closed) return Promise.reject(new Error('embedder is closed'));
    if (failure) return Promise.reject(failure);
    return new Promise((resolve, reject) => { const id = nextId++; pending.set(id, { resolve: (v) => { resolve(v); setImmediate(syncAlive); }, reject: (e) => { reject(e); setImmediate(syncAlive); }, worker: null }); syncAlive(); queue.push({ id, ids }); dispatch(); });
  }
  async function close() {
    if (closed) return; closed = true; if (alive) { clearInterval(alive); alive = null; }
    await Promise.all(workers.map((w) => w.terminate()));
  }
  return {
    path, cfg, tokenizer: tok, threads: n, maxSeq: seq, kernels: kern, loadMs: performance.now() - t0, dim: cfg.nEmbd,
    /** Token ids of a text, as it is fed to the model ([CLS] ... [SEP], truncated to maxSeq). */
    tokenize: (text) => tok.encode(text, { special: true, maxLen: seq }),
    /** @param {number[][]} idsList @param {(done: number, total: number) => void} [onProgress] @returns {Promise<Float32Array[]>} mean-pooled, NOT normalised */
    async embedIds(idsList, onProgress) {
      let done = 0;
      return Promise.all(idsList.map((ids) => run(ids).then((v) => { onProgress?.(++done, idsList.length); return v; })));
    },
    async embed(texts, onProgress) { return this.embedIds(texts.map((t) => tok.encode(t, { special: true, maxLen: seq })), onProgress); },
    close,
  };
}

export const cosine = (a, b) => { let d = 0; let x = 0; let y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / (Math.sqrt(x * y) || 1e-12); };
