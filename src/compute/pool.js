// Main-thread side of the worker pool: spawns SPMD workers, issues one command per token via shared atomics
// (no postMessage per token), and exposes the same interface as RefModel so generate() works unchanged.
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { modelConfig, assertSupportedTensorTypes, recurrentStateBytes } from '../model/config.js';
import { checkTensors, findBoundaryIds } from '../model/forward-qwen35.js';
import { RecurrentState } from '../model/recurrent-state.js';
import { StateCheckpoints } from '../model/state-checkpoints.js';
import { CTRL, CMD_TOKEN, CMD_EXIT, CMD_KV_EXPORT, CMD_KV_IMPORT, MAX_BATCH, DEFAULT_LOGITS_BATCH, allocShared } from './layout.js';
import { probeWasmFeatures } from './wasm/features.js';

/**
 * Barrier/command wait strategy. Spinning is fastest when every worker has a core to itself, but when the machine has
 * spare logical processors (hyperthreads) a spinning worker can sit on the sibling of a busy core and slow it down,
 * while sleeping workers let the OS spread the threads. Measured on a 6-core/12-thread laptop (5 interleaved rounds,
 * Qwen3-4B Q4_K_M, 6 workers): spin=0 median 5.90 tok/s vs spin=20000 median 4.55; on a 4-vCPU VM with 4 workers
 * spinning was ~20% faster. So: sleep immediately if there are at least 2 logical processors per worker, else spin.
 * @param {number} workers @param {number} [logical]
 */
export function defaultSpin(workers, logical = os.availableParallelism?.() ?? os.cpus().length) {
  return logical >= 2 * workers ? 0 : 20000;
}

export class PoolModel {
  /**
   * @param {import('../gguf/reader.js').GGUFFile} g
   * @param {{ctx?: number, workers?: number, spin?: number, kernels?: 'auto'|'wasm'|'js', maxBatch?: number, logitsBatch?: number, kv?: 'f32'|'f16',
   *          maxCheckpoints?: number, checkpointEvery?: number, boundaryIds?: number[], onProgress?: (done: number, total: number) => void}} [opts]
   *   logitsBatch: most positions one pass can return logits for (`forwardBatch(.., {logits: 'all'})`); default min(maxBatch, 16), 1 = off
   *   maxCheckpoints / checkpointEvery / boundaryIds: hybrid (Gated DeltaNet) models only, see src/model/state-checkpoints.js
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
    const cfg = this.cfg = modelConfig(g); // throws UnsupportedArchError for an architecture this engine does not know
    // fail early and clearly (before any worker starts) on a file the pool cannot run
    assertSupportedTensorTypes(g);
    if (cfg.hybrid) checkTensors(g, cfg);
    this.ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
    this.n = Math.max(1, opts.workers ?? 1);
    /** busy-wait iterations before a worker sleeps at a barrier / waits for a command (0 = sleep immediately) */
    this.spin = opts.spin ?? defaultSpin(this.n);
    /** precision of the KV cache: f16 halves its memory (values are rounded to half precision) */
    this.kv = opts.kv ?? 'f32';
    if (this.kv !== 'f32' && this.kv !== 'f16') throw new Error(`kv must be f32 or f16 (got "${this.kv}")`);
    const want = opts.kernels ?? 'auto';
    if (want === 'wasm' && !probeWasmFeatures().simd) throw new Error('--kernels wasm requested but this Node/V8 has no WebAssembly SIMD; use --kernels js');
    this.kernels = want === 'auto' ? (probeWasmFeatures().simd ? 'wasm' : 'js') : want;
    this.tokens = [];
    this.failed = null;
    this.prevNoLogits = false;
    this.closed = false;
    this.issued = 0;
    /** largest prompt batch one pass may carry (weights are read once per batch) */
    this.maxBatch = Math.max(1, Math.min(opts.maxBatch ?? 16, MAX_BATCH));
    /** largest batch that can return per-position logits (the shared logits buffer holds this many vocabulary rows) */
    this.logitsBatch = Math.max(1, Math.min(opts.logitsBatch ?? DEFAULT_LOGITS_BATCH, this.maxBatch));
    const sh = this.shared = allocShared(cfg, this.ctx, this.maxBatch, this.logitsBatch);
    this.kvStage = new Float32Array(sh.kvStage);
    this.ctrl = new Int32Array(sh.ctrl);
    this.x = new Float32Array(sh.x);
    this.logits = new Float32Array(sh.logits);
    if (cfg.hybrid) {
      /** the live recurrent state: the workers update it (each its own heads), the main thread snapshots / restores it while they are idle */
      this.state = new RecurrentState(cfg, { conv: new Float32Array(sh.conv), ssm: new Float32Array(sh.ssm) });
      this.ck = new StateCheckpoints(cfg, { maxCheckpoints: opts.maxCheckpoints, checkpointEvery: opts.checkpointEvery, boundaryIds: opts.boundaryIds ?? findBoundaryIds(g) });
    }

    // everything a worker needs to know about tensors (small, structured-cloneable)
    const tensors = {};
    let totalBytes = 0;
    for (const t of g.tensors) {
      tensors[t.name] = { name: t.name, type: t.type, dims: t.dims, absOffset: t.absOffset, nBytes: t.nBytes, nElements: t.nElements };
      if (t.dims.length === 2 && !t.name.endsWith('ssm_conv1d.weight')) totalBytes += t.nBytes ?? 0; // (the conv taps are a small private vector in every worker)
    }

    const url = new URL('./worker.js', import.meta.url);
    const progress = new Array(this.n).fill(0);
    const t0 = performance.now();
    this.workers = [];
    this.exits = [];
    const ready = [];
    for (let id = 0; id < this.n; id++) {
      const w = new Worker(url, { workerData: { id, n: this.n, cfg, tensors, path: g.path, ctx: this.ctx, shared: sh, spin: this.spin, kernels: this.kernels, maxBatch: this.maxBatch, logitsBatch: this.logitsBatch, kvType: this.kv } });
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
   * Run several tokens (at consecutive positions) in ONE pass over the weights. By default the logits are for the
   * last token only; `{logits: 'all'}` returns the logits of EVERY token of the batch (row b, at offset b * vocab, is the
   * distribution of the token that follows batch token b): the output weights are still read once for the whole batch.
   * @param {ArrayLike<number>} tokens at most `maxBatch` of them (`logitsBatch` with logits 'all')
   * @param {{logits?: boolean|'last'|'all'|'none'}} [opts]
   * @returns {Promise<Float32Array|null>} view over shared memory, valid until the next forward call: vocab floats, or B * vocab floats with 'all'
   */
  async forwardBatch(tokens, opts = {}) {
    if (this.failed) throw this.failed;
    if (this.closed) throw new Error('model is closed');
    const B = tokens.length;
    if (B < 1 || B > this.maxBatch) throw new Error(`batch of ${B} tokens (max ${this.maxBatch})`);
    const pos = this.tokens.length;
    if (pos + B > this.ctx) throw new Error(`context full (${this.ctx} tokens)`);
    for (let i = 0; i < B; i++) if (!(tokens[i] >= 0 && tokens[i] < this.cfg.vocabSize)) throw new Error(`token id ${tokens[i]} out of range`);
    const lg = opts.logits;
    const want = lg === false || lg === 'none' ? 0 : lg === 'all' ? 2 : 1;
    if (want === 2 && B > this.logitsBatch) throw new Error(`logits for all positions of a batch of ${B} tokens (this pool supports ${this.logitsBatch}; raise the logitsBatch option)`);
    if (this.cfg.hybrid) await this.#hybridBatch(tokens, want);
    else await this.#pass(tokens, want);
    if (want === 2) return this.logits.subarray(0, B * this.cfg.vocabSize);
    return want ? this.logits.subarray(0, this.cfg.vocabSize) : null;
  }

  /** One command: the workers run `tokens` at the next positions; afterwards the tokens are part of the sequence. */
  async #pass(tokens, want) {
    const { ctrl } = this;
    const B = tokens.length;
    for (let i = 0; i < B; i++) Atomics.store(ctrl, CTRL.TOKENS + i, tokens[i]);
    Atomics.store(ctrl, CTRL.BATCH, B);
    Atomics.store(ctrl, CTRL.POS, this.tokens.length);
    Atomics.store(ctrl, CTRL.WANT_LOGITS, want);
    await this.#issue(CMD_TOKEN);
    for (let i = 0; i < B; i++) this.tokens.push(tokens[i]);
  }

  /**
   * forwardBatch for hybrid models. The checkpoint policy (src/model/state-checkpoints.js) wants a snapshot of the recurrent
   * state BEFORE certain tokens; inside a batch the state of a position is never materialised (the recurrences run token by
   * token inside the workers), so the batch is split there: segments run as separate passes with a snapshot in between.
   * Splits happen before chat-turn boundary tokens, periodic positions and (the one that matters) the last token of a prompt,
   * which is how a regenerate / a follow-up turn can resume with nothing to replay. Batches that return logits for every
   * position (speculative verification) are never split: only the state before the first token can be snapshotted.
   */
  async #hybridBatch(tokens, want) {
    const B = tokens.length;
    const pos = this.tokens.length;
    const ck = this.ck;
    if (ck.wantsBefore(pos, tokens[0], want === 2 || (want === 1 && B === 1), this.prevNoLogits)) this.#checkpoint();
    const cuts = [0];
    if (want !== 2) for (let i = 1; i < B; i++) if (ck.wantsBefore(pos + i, tokens[i], want === 1 && i === B - 1, true)) cuts.push(i);
    cuts.push(B);
    for (let c = 0; c + 1 < cuts.length; c++) {
      if (c > 0) this.#checkpoint();
      const last = c + 2 === cuts.length;
      await this.#pass(tokens.slice(cuts[c], cuts[c + 1]), last ? want : 0);
    }
    this.prevNoLogits = want === 0;
  }

  #checkpoint() { this.ck.save(this.tokens.length, this.state); }

  /** Send a command to all workers and wait until every one of them has finished it. */
  async #issue(cmd) {
    const { ctrl, n } = this;
    Atomics.store(ctrl, CTRL.CMD, cmd);
    const target = ++this.issued * n;
    Atomics.add(ctrl, CTRL.CMD_SEQ, 1);
    Atomics.notify(ctrl, CTRL.CMD_SEQ);
    // Atomics.waitAsync keeps the event loop free (SIGINT handlers keep working); the interval timer keeps the process
    // alive while we wait, and the 200 ms timeout lets us notice worker failures.
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
  }

  /**
   * Copy the first n positions of one layer's K (isV=0) or V (isV=1) cache out of the workers: returns an
   * [n][kvDim] f32 view of the staging buffer, valid until the next KV command. (Used by the prefix cache.)
   */
  async kvGet(layer, isV, n) {
    if (this.failed) throw this.failed;
    Atomics.store(this.ctrl, CTRL.KV_LAYER, layer); Atomics.store(this.ctrl, CTRL.KV_ISV, isV); Atomics.store(this.ctrl, CTRL.KV_N, n);
    await this.#issue(CMD_KV_EXPORT);
    return this.kvStage.subarray(0, n * this.cfg.nHeadKv * this.cfg.headDim);
  }

  /** Write n positions of one layer's K or V cache into the workers (data: [n][kvDim] f32). */
  async kvSet(layer, isV, n, data) {
    if (this.failed) throw this.failed;
    this.kvStage.set(data.subarray(0, n * this.cfg.nHeadKv * this.cfg.headDim));
    Atomics.store(this.ctrl, CTRL.KV_LAYER, layer); Atomics.store(this.ctrl, CTRL.KV_ISV, isV); Atomics.store(this.ctrl, CTRL.KV_N, n);
    await this.#issue(CMD_KV_IMPORT);
  }

  /**
   * After the KV regions (and, for a hybrid model, the recurrent state) were filled from a prefix cache: declare that the
   * model now holds exactly `ids`.
   */
  restorePrefix(ids) {
    this.tokens = Array.from(ids);
    if (this.cfg.hybrid) { this.ck.clear(); this.prevNoLogits = false; this.#checkpoint(); } // the end of the cached prefix is a natural place to come back to
  }

  /** Bytes of recurrent state per snapshot / live state (0 for models without any). */
  get stateBytes() { return recurrentStateBytes(this.cfg); }

  /** Positions of the stored recurrent-state checkpoints (hybrid models; ascending). */
  get checkpointPositions() { return this.ck ? this.ck.positions : []; }

  /** What truncate(n) would do: the snapshot position it restores from and how many tokens it replays. */
  truncatePlan(n) { return this.ck ? this.ck.plan(n, this.tokens.length) : { from: Math.min(n, this.tokens.length), replay: 0 }; }

  /** A copy of the live recurrent state {conv, ssm} (hybrid models; the workers must be idle). Used by the prefix cache. */
  async getRecurrentState() { return { conv: this.state.conv.slice(), ssm: this.state.ssm.slice() }; }

  /** Replace the live recurrent state (hybrid models). */
  async setRecurrentState({ conv, ssm }) { this.state.conv.set(conv); this.state.ssm.set(ssm); }

  /**
   * Hybrid models: remember the recurrent state at the current position in the single rollback slot, so that a later
   * truncate() to a position at or after it only replays the tokens in between (speculative decoding marks the state before
   * each verification pass). No-op for models whose cache is positional.
   */
  markRollback() { if (this.cfg.hybrid) this.ck.mark(this.tokens.length, this.state); }

  /**
   * Roll back to `n` positions (e.g. to reuse a shared prefix or to drop rejected speculative tokens). A KV cache is
   * positional, so for ordinary models this is bookkeeping. A hybrid model restores the nearest recurrent-state snapshot at or
   * before `n` and replays the tokens in between (batched, as many passes as needed); the returned promise resolves to the
   * number of tokens replayed. Do not call forward while a truncate is pending.
   * @returns {Promise<number>}
   */
  async truncate(n) {
    if (n >= this.tokens.length) return 0;
    if (n < 0) n = 0;
    if (!this.cfg.hybrid) { this.tokens.length = n; return 0; }
    if (this.failed) throw this.failed;
    if (this.closed) throw new Error('model is closed');
    const { from, replay } = this.ck.plan(n, this.tokens.length);
    const tail = this.tokens.slice(from, n);
    this.ck.dropAfter(n);
    this.ck.restore(this.state, from);
    this.tokens.length = from;
    this.prevNoLogits = false;
    for (let i = 0; i < tail.length; i += this.maxBatch) await this.#hybridBatch(tail.slice(i, i + this.maxBatch), 0);
    if (replay > 0) this.#checkpoint(); // n is where the next prompt continues (and where a regenerate comes back to): keep it
    this.prevNoLogits = false;
    return replay;
  }

  reset() {
    this.tokens.length = 0;
    if (this.cfg.hybrid) { this.state.clear(); this.ck.clear(); this.prevNoLogits = false; }
  }

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
