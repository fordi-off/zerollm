// Work partitioning + shared-memory layout used by the worker pool (main thread and workers agree on these).

/** Control block (Int32Array over a SharedArrayBuffer). Hot words are spread over separate cache lines. */
export const CTRL = Object.freeze({
  CMD_SEQ: 0,   // main increments to issue a command
  DONE: 1,      // each worker increments when it finishes a command
  TOKEN: 2,
  POS: 3,
  WANT_LOGITS: 4,
  ERROR: 5,     // worker id + 1 of the first failing worker
  CMD: 6,       // 0 = run tokens, 1 = exit
  BATCH: 7,     // number of tokens in this command (1 = decode)
  BAR_COUNT: 16,
  BAR_GEN: 32,
  TOKENS: 48,   // token ids of the batch (up to MAX_BATCH)
  SIZE: 48 + 64,
});

/** Largest prompt batch a pool can be asked to run in one pass. */
export const MAX_BATCH = 64;

export const CMD_TOKEN = 0;
export const CMD_EXIT = 1;

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

/** Float offset of a KV cache row. Layout: [layer][k=0|v=1][pos][kvDim]. */
export function kvOffset(layer, isV, ctx, kvDim, pos = 0) {
  return (((layer * 2 + isV) * ctx) + pos) * kvDim;
}

/** Allocate the SharedArrayBuffers a pool needs. */
export function allocShared(cfg, ctx, maxBatch = 1) {
  const qDim = cfg.nHead * cfg.headDim;
  const kvDim = cfg.nHeadKv * cfg.headDim;
  const f32 = (n) => new SharedArrayBuffer(Math.max(1, n) * 4);
  return {
    ctrl: new SharedArrayBuffer(CTRL.SIZE * 4),
    x: f32(cfg.nEmbd * maxBatch), q: f32(qDim * maxBatch), k: f32(kvDim * maxBatch), v: f32(kvDim * maxBatch), att: f32(qDim * maxBatch),
    hb: f32(cfg.nFF * maxBatch), logits: f32(cfg.vocabSize),
    kv: f32(cfg.nLayer * 2 * ctx * kvDim),
  };
}
