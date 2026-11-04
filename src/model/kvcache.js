// f32 KV cache for the reference path. Layout per layer: [pos][kvHead][headDim] contiguous.

export class KVCache {
  /** @param {{nLayer: number, nHeadKv: number, headDim: number}} cfg @param {number} ctx */
  constructor(cfg, ctx) {
    this.ctx = ctx;
    this.kvDim = cfg.nHeadKv * cfg.headDim;
    this.k = Array.from({ length: cfg.nLayer }, () => new Float32Array(ctx * this.kvDim));
    this.v = Array.from({ length: cfg.nLayer }, () => new Float32Array(ctx * this.kvDim));
    this.length = 0; // number of positions filled
  }
  get bytes() { return this.k.length * 2 * this.ctx * this.kvDim * 4; }
  reset() { this.length = 0; }
}
