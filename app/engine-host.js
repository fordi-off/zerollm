// Keeps at most one big chat model resident (RAM!) and serializes generations.
import os from 'node:os';
import { loadModel, EngineMemoryError } from '../src/index.js';
import { modelPath } from './models.js';

export class EngineHost {
  /** @param {{paths: object, getConfig: () => object, log: {info: Function, error: Function}}} o */
  constructor({ paths, getConfig, log }) {
    this.paths = paths; this.getConfig = getConfig; this.log = log;
    this.engine = null; this.filename = null; this.loading = null; this.busy = false;
  }

  get loadedName() { return this.engine ? this.filename : null; }

  /** Load `filename` (unloading any other model first). Concurrent calls for the same file share one load. */
  async ensure(filename, { onProgress } = {}) {
    if (this.engine && this.filename === filename) return this.engine;
    if (this.loading && this.loading.filename === filename) return this.loading.promise;
    const promise = (async () => {
      await this.unload();
      const cfg = this.getConfig();
      const full = modelPath(this.paths, filename);
      this.log.info(`loading model ${filename} (ctx ${cfg.context_window}, threads ${cfg.n_threads})`);
      const t0 = performance.now();
      const engine = await loadModel(full, { ctx: cfg.context_window, threads: cfg.n_threads, kv: 'f16', onProgress });
      this.engine = engine; this.filename = filename;
      this.log.info(`model ready: ${filename} in ${((performance.now() - t0) / 1000).toFixed(1)}s (${engine.info.kernels}, ctx ${engine.ctx}, RAM free ${(os.freemem() / 1e9).toFixed(1)} GB)`);
      return engine;
    })();
    this.loading = { filename, promise };
    try { return await promise; } finally { this.loading = null; }
  }

  async unload() {
    const e = this.engine;
    this.engine = null; this.filename = null;
    if (e) { await e.close(); this.log.info('model unloaded'); }
  }
}

export { EngineMemoryError };
