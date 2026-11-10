// Work partitioning + shared-memory layout used by the worker pool (main thread and workers agree on these).

/** Control block (Int32Array over a SharedArrayBuffer). Hot words are spread over separate cache lines. */
export const CTRL = Object.freeze({
  CMD_SEQ: 0,   // main increments to issue a command
  DONE: 1,      // each worker increments when it finishes a command
  TOKEN: 2,
  POS: 3,
  WANT_LOGITS: 4, // 0 = none, 1 = last token of the batch, 2 = every token of the batch (speculative verification)
  ERROR: 5,     // worker id + 1 of the first failing worker
  CMD: 6,       // 0 = run tokens, 1 = exit
  BATCH: 7,     // number of tokens in this command (1 = decode)
  KV_LAYER: 8,  // KV export/import: layer (attention-layer number: only attention layers have a KV cache)
  KV_ISV: 9,    //                   0 = K, 1 = V
  KV_N: 10,     //                   number of positions
  BAR_COUNT: 16,
  BAR_GEN: 32,
  TOKENS: 48,   // token ids of the batch (up to MAX_BATCH)
  SIZE: 48 + 64,
});

/** Largest prompt batch a pool can be asked to run in one pass. */
export const MAX_BATCH = 64;

export const CMD_TOKEN = 0;
export const CMD_EXIT = 1;
export const CMD_KV_EXPORT = 2; // workers copy their private K or V cache of one layer into the shared staging buffer
export const CMD_KV_IMPORT = 3; // ... and the reverse

/** Even split of `count` items among `n` workers; returns [start, end) for worker w. */
export function splitRange(count, n, w) {
  return [Math.floor((w * count) / n), Math.floor(((w + 1) * count) / n)];
}

/**
 * Split matrix rows among workers in multiples of `align` (the SIMD kernels want 4-row groups).
 * The last worker takes the remainder; workers may get 0 rows when rows is tiny.
 */
export function splitRows(rows, n, w, align = 4) {
  const bound = (i) => (i >= n ? rows : Math.min(rows, Math.round((i * rows) / n / align) * align));
  return [bound(w), w === n - 1 ? rows : bound(w + 1)];
}

/** Round up to a multiple of `a`. */
export const alignUp = (v, a) => Math.ceil(v / a) * a;

/** Largest batch for which per-position logits (`logits: 'all'`) are available unless the pool is told otherwise. */
export const DEFAULT_LOGITS_BATCH = 16;

/**
 * How a hybrid model's Gated DeltaNet layers are sharded: worker `w` owns the value heads [vh0, vh1) (their delta-rule
 * matrices, their slice of the conv window, their gate / beta / alpha rows) and computes the q/k conv + L2 norm of the key
 * heads [kk0, kk1). The two ranges are independent: a value head reads its key head's (already normalised) q and k from
 * shared memory, whoever computed it, so the value-head -> key-head mapping (keyHeadOfValueHead in ref-deltanet.js) does
 * not influence the sharding.
 * @param {{nKHeads: number, nVHeads: number}} ssm @param {number} n workers @param {number} w
 */
export function recurrentShards(ssm, n, w) {
  const [kk0, kk1] = splitRange(ssm.nKHeads, n, w);
  const [vh0, vh1] = splitRange(ssm.nVHeads, n, w);
  return { kk0, kk1, vh0, vh1 };
}

/**
 * Allocate the SharedArrayBuffers a pool needs. `logitsBatch` = how many positions' logits one pass can return
 * (`logits: 'all'`); the shared logits buffer holds that many vocabulary-sized rows.
 * Hybrid (Gated DeltaNet) models additionally get the live recurrent state, `conv` and `ssm` (layout of RecurrentState:
 * per recurrent layer [channel][kernel-1] and [vHead][vDim][kDim], f32), updated by the workers and snapshotted by the main
 * thread, and wider `q` (attention q rows come as [query | gate] per head; the conv output of a recurrent layer) and `att`
 * (the gated-norm output of the recurrent heads) buffers.
 */
export function allocShared(cfg, ctx, maxBatch = 1, logitsBatch = 1) {
  const qDim = cfg.nHead * cfg.headDim;
  const kvDim = cfg.nHeadKv * cfg.headDim;
  const f32 = (n) => new SharedArrayBuffer(Math.max(1, n) * 4);
  const qRows = cfg.attnOutputGate ? 2 * qDim : qDim;
  const sh = {
    ctrl: new SharedArrayBuffer(CTRL.SIZE * 4),
    x: f32(cfg.nEmbd * maxBatch), q: f32(Math.max(qRows, cfg.ssm?.convDim ?? 0) * maxBatch), k: f32(kvDim * maxBatch), v: f32(kvDim * maxBatch),
    att: f32(Math.max(qDim, cfg.ssm?.innerSize ?? 0) * maxBatch),
    hb: f32(cfg.nFF * maxBatch), logits: f32(cfg.vocabSize * Math.max(1, logitsBatch)),
    kvStage: f32(ctx * kvDim), // staging area for KV export/import (the caches themselves are private to each worker)
    conv: null, ssm: null,
  };
  if (cfg.hybrid) {
    const nRec = cfg.layerIsRecurrent.filter(Boolean).length;
    const s = cfg.ssm;
    sh.conv = f32(nRec * (s.convKernel - 1) * s.convDim);
    sh.ssm = f32(nRec * s.nVHeads * s.headVDim * s.headKDim);
  }
  return sh;
}
