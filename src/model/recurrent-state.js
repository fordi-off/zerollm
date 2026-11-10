// The recurrent state of one sequence of a hybrid (Gated DeltaNet) model, in two flat f32 arrays.

/**
 * For each recurrent layer (numbered 0.. in layer order) the conv window and the delta-rule matrices, so a snapshot is two
 * `set()` calls and the size is explicit. The arrays may be supplied (the worker pool keeps the live state in
 * SharedArrayBuffers that every worker updates for its own heads and the main thread can snapshot / restore).
 */
export class RecurrentState {
  /**
   * @param {import('./config.js').ModelConfig} cfg
   * @param {{conv: Float32Array, ssm: Float32Array}} [arrays] existing storage (must have the right sizes)
   */
  constructor(cfg, arrays) {
    const s = cfg.ssm;
    this.nLayers = cfg.layerIsRecurrent.filter(Boolean).length;
    /** floats per layer */
    this.convSize = (s.convKernel - 1) * s.convDim;
    this.ssmSize = s.nVHeads * s.headVDim * s.headKDim;
    this.conv = arrays?.conv ?? new Float32Array(this.nLayers * this.convSize);
    this.ssm = arrays?.ssm ?? new Float32Array(this.nLayers * this.ssmSize);
    if (this.conv.length !== this.nLayers * this.convSize || this.ssm.length !== this.nLayers * this.ssmSize) throw new Error('recurrent state arrays have the wrong size');
  }

  /** Total bytes (matches recurrentStateBytes(cfg)). */
  get bytes() { return (this.conv.length + this.ssm.length) * 4; }

  /** Conv window of recurrent layer #i: [channel][kernel-1], oldest first. */
  convOf(i) { return this.conv.subarray(i * this.convSize, (i + 1) * this.convSize); }

  /** Delta-rule matrices of recurrent layer #i: [vHead][vDim][kDim]. */
  ssmOf(i) { return this.ssm.subarray(i * this.ssmSize, (i + 1) * this.ssmSize); }

  clear() { this.conv.fill(0); this.ssm.fill(0); }

  /** @param {RecurrentState} o */
  copyFrom(o) { this.conv.set(o.conv); this.ssm.set(o.ssm); }
}
