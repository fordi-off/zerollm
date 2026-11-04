// Main-thread side of the worker pool: spawns SPMD workers, issues one command per token via shared atomics
// (no postMessage per token), and exposes the same interface as RefModel so generate() works unchanged.
import { Worker } from 'node:worker_threads';
import { modelConfig } from '../model/config.js';
import { CTRL, CMD_TOKEN, CMD_EXIT, MAX_BATCH, allocShared, kvOffset } from './layout.js';
import { probeWasmFeatures } from './wasm/features.js';

export class PoolModel {
  /**
   * @param {import('../gguf/reader.js').GGUFFile} g
   * @param {{ctx?: number, workers?: number, spin?: number, kernels?: 'auto'|'wasm'|'js', maxBatch?: number, onProgress?: (done: number, total: number) => void}} [opts]
   * @returns {Promise<PoolModel>}
   */
  static async create(g, opts = {}) {
    const m = new PoolModel();
    await m.#init(g, opts);
    return m;
  }

  /** Current number of positions in the KV cache. */
  get pos() { return this.tokens.length; }

  async #init(g, opts) {
    const cfg = this.cfg = modelConfig(g);
    this.ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
    this.n = Math.max(1, opts.workers ?? 1);
    const want = opts.kernels ?? 'auto';
    if (want === 'wasm' && !probeWasmFeatures().simd) throw new Error('--kernels wasm requested but this Node/V8 has no WebAssembly SIMD; use --kernels js');
    this.kernels = want === 'auto' ? (probeWasmFeatures().simd ? 'wasm' : 'js') : want;
    this.tokens = [];
    this.failed = null;
    this.closed = false;
    this.issued = 0;
    /** largest prompt batch one pass may carry (weights are read once per batch) */
    this.maxBatch = Math.max(1, Math.min(opts.maxBatch ?? 16, MAX_BATCH));
    const sh = this.shared = allocShared(cfg, this.ctx, this.maxBatch);
    this.kvView = new Float32Array(sh.kv);
    this.ctrl = new Int32Array(sh.ctrl);
    this.x = new Float32Array(sh.x);
    this.logits = new Float32Array(sh.logits);

    // everything a worker needs to know about tensors (small, structured-cloneable)
    const tensors = {};
    let totalBytes = 0;
    for (const t of g.tensors) {
      tensors[t.name] = { name: t.name, type: t.type, dims: t.dims, absOffset: t.absOffset, nBytes: t.nBytes, nElements: t.nElements };
      if (t.dims.length === 2) totalBytes += t.nBytes ?? 0;
    }

    const url = new URL('./worker.js', import.meta.url);
    const progress = new Array(this.n).fill(0);
    const t0 = performance.now();
    this.workers = [];
    this.exits = [];
    const ready = [];
    for (let id = 0; id < this.n; id++) {
      const w = new Worker(url, { workerData: { id, n: this.n, cfg, tensors, path: g.path, ctx: this.ctx, shared: sh, spin: opts.spin ?? 20000, kernels: this.kernels, maxBatch: this.maxBatch } });
      this.workers.push(w);
      this.exits.push(new Promise((r) => w.once('exit', r)));
      ready.push(new Promise((resolve, reject) => {
        w.on('message', (msg) => {
          if (msg.type === 'ready') resolve(msg);
          else if (msg.type === 'progress') { progress[id] = msg.bytes; opts.onProgress?.(progress.reduce((a, b) => a + b, 0), totalBytes); }
          else if (msg.type === 'error') { const e = new Error(`worker ${msg.id} failed: ${msg.message}`); this.#fail(e); reject(e); }
        });
        w.on('error', (e) => { this.#fail(e); reject(e); });
        w.on('exit', (code) => {
          if (!this.closed && !this.failed) { const e = new Error(`worker ${id} exited unexpectedly (code ${code})`); this.#fail(e); reject(e); }
        });
        // A forgotten close() must not hang the process. unref() has to come AFTER the 'message' listener is added:
        // adding a message listener re-refs the worker's MessagePort.
        w.unref();
      }));
    }
    // workers are unref'd, so keep the event loop alive ourselves while they load
    const keepAlive = setInterval(() => {}, 1000);
    try {
      this.workerInfo = await Promise.all(ready);
    } catch (e) {
      await this.close();
      throw e;
    } finally { clearInterval(keepAlive); }
    opts.onProgress?.(totalBytes, totalBytes);
    this.loadMs = performance.now() - t0;
    this.sliceBytes = this.workerInfo.map((i) => i.sliceBytes);
  }

  #fail(e) {
    if (this.failed) return;
    this.failed = e;
    for (const w of this.workers ?? []) w.terminate();
  }

  /**
   * Run one token at the next position.
   * @param {number} token
   * @param {{logits?: boolean}} [opts] logits:false skips the output projection (prompt prefill)
   * @returns {Promise<Float32Array|null>} logits view over shared memory (valid until the next forward call)
   */
  forward(token, opts = {}) { return this.forwardBatch([token], opts); }

  /**
   * Run several tokens (at consecutive positions) in ONE pass over the weights. Logits, if requested, are for the
   * last token only.
   * @param {ArrayLike<number>} tokens at most `maxBatch` of them
   * @param {{logits?: boolean}} [opts]
   * @returns {Promise<Float32Array|null>}
   */
  async forwardBatch(tokens, opts = {}) {
    if (this.failed) throw this.failed;
    if (this.closed) throw new Error('model is closed');
    const B = tokens.length;
    if (B < 1 || B > this.maxBatch) throw new Error(`batch of ${B} tokens (max ${this.maxBatch})`);
    const pos = this.tokens.length;
    if (pos + B > this.ctx) throw new Error(`context full (${this.ctx} tokens)`);
    for (let i = 0; i < B; i++) if (!(tokens[i] >= 0 && tokens[i] < this.cfg.vocabSize)) throw new Error(`token id ${tokens[i]} out of range`);
    const { ctrl, n } = this;
    const want = opts.logits === false ? 0 : 1;
    for (let i = 0; i < B; i++) Atomics.store(ctrl, CTRL.TOKENS + i, tokens[i]);
    Atomics.store(ctrl, CTRL.BATCH, B);
    Atomics.store(ctrl, CTRL.POS, pos);
    Atomics.store(ctrl, CTRL.WANT_LOGITS, want);
    Atomics.store(ctrl, CTRL.CMD, CMD_TOKEN);
    const target = ++this.issued * n;
    Atomics.add(ctrl, CTRL.CMD_SEQ, 1);
    Atomics.notify(ctrl, CTRL.CMD_SEQ);

    // Wait for all workers. Atomics.waitAsync keeps the event loop free (SIGINT handlers keep working); the
    // interval timer keeps the process alive while we wait, and the 200 ms timeout lets us notice worker failures.
    const keepAlive = setInterval(() => {}, 1000);
    try {
      for (;;) {
        const cur = Atomics.load(ctrl, CTRL.DONE);
        if (cur >= target) break;
        if (this.failed) throw this.failed;
        const r = Atomics.waitAsync(ctrl, CTRL.DONE, cur, 200);
        if (r.async) await r.value;
      }
    } finally { clearInterval(keepAlive); }
    for (let i = 0; i < B; i++) this.tokens.push(tokens[i]);
    return want ? this.logits : null;
  }

  /** Whole [pos][kvDim] region of one layer's K (isV=0) or V (isV=1) cache, as a live view (for the prefix cache). */
  kvRegion(layer, isV) {
    const kvDim = this.cfg.nHeadKv * this.cfg.headDim;
    return this.kvView.subarray(kvOffset(layer, isV, this.ctx, kvDim), kvOffset(layer, isV, this.ctx, kvDim) + this.ctx * kvDim);
  }

  /** After the KV regions were filled from a prefix cache: declare that the model now holds exactly `ids`. */
  restorePrefix(ids) { this.tokens = Array.from(ids); }

  /** Roll the KV cache back to `n` positions (the cache itself is positional, so this is just bookkeeping). */
  truncate(n) { if (n < this.tokens.length) this.tokens.length = n; }
  reset() { this.tokens.length = 0; }

  /** Stop all workers. Safe to call more than once. */
  async close() {
    if (this.closed) return;
    this.closed = true;
    Atomics.store(this.ctrl, CTRL.CMD, CMD_EXIT);
    Atomics.add(this.ctrl, CTRL.CMD_SEQ, 1);
    Atomics.notify(this.ctrl, CTRL.CMD_SEQ);
    // Graceful first: healthy workers see CMD_EXIT and return on their own. After a failure some workers are stuck
    // at a barrier and will never look at the command, so skip the grace period then.
    if (!this.failed) {
      let timer;
      await Promise.race([Promise.all(this.exits), new Promise((r) => { timer = setTimeout(r, 2000); })]);
      clearTimeout(timer);
    }
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}
