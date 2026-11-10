// Token generation loop shared by the CLI and (later) the server.

/** Greedy sampler: argmax of logits. @param {Float32Array} logits */
export function argmax(logits) {
  let best = 0; let bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}

const yieldToEventLoop = () => new Promise((r) => setImmediate(r));


/**
 * Feed tokens `ids[from..]` through the model (batched when the model supports it). Returns the logits of the last
 * token if `wantLogits`. Yields to the event loop between passes so Ctrl+C stays responsive.
 * @returns {Promise<{logits: Float32Array|null, aborted: boolean}>}
 */
export async function prefillTokens(model, ids, { from = 0, wantLogits = true, signal, onProgress } = {}) {
  const chunk = model.forwardBatch ? Math.max(1, model.maxBatch ?? 1) : 1;
  let logits = null;
  for (let i = from; i < ids.length;) {
    if (signal?.aborted) return { logits: null, aborted: true };
    const end = Math.min(ids.length, i + chunk);
    const last = wantLogits && end === ids.length;
    logits = chunk > 1 ? await model.forwardBatch(ids.slice(i, end), { logits: last }) : await model.forward(ids[i], { logits: last });
    i = end;
    onProgress?.(i - from, ids.length - from);
    await yieldToEventLoop();
  }
  return { logits, aborted: false };
}

/** Length of the common prefix of two token arrays. */
export function commonPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

/**
 * Generate tokens after `promptIds`. Yields token ids as they are produced (stop tokens are not yielded).
 *
 * The model's KV cache is reused for any prefix shared with `model.tokens` (e.g. earlier chat turns), so only the
 * new part of the prompt is processed. The loop yields to the event loop after every token (prefill included), so
 * SIGINT handlers run promptly.
 *
 * @param {{forward: Function, truncate: Function (sync or async), tokens: number[], pos: number, ctx: number}} model
 * @param {number[]} promptIds
 * @param {{maxTokens?: number, sample?: (logits: Float32Array, history: number[]) => number, stopIds?: Set<number>,
 *          signal?: {aborted: boolean}, stats?: object, onPrefillProgress?: (done: number, total: number) => void,
 *          penalizePrompt?: boolean}} [opts]  penalizePrompt=false: the sampler's repetition penalties only see the generated tokens
 *          (the prompt - e.g. document excerpts the answer should quote - is not penalised); default true like llama.cpp
 */
export async function* generate(model, promptIds, opts = {}) {
  const sample = opts.sample ?? argmax;
  const stop = opts.stopIds ?? new Set();
  const max = opts.maxTokens ?? 256;
  const stats = opts.stats ?? {};
  if (promptIds.length === 0) throw new Error('empty prompt');
  if (promptIds.length >= model.ctx) throw new Error(`prompt is ${promptIds.length} tokens but the context window is ${model.ctx}; shorten it or raise --ctx`);

  // reuse cached prefix, but always recompute at least the last prompt token (we need its logits)
  const reuse = Math.min(commonPrefix(model.tokens, promptIds), promptIds.length - 1);
  await model.truncate(reuse); // may be async (a hybrid model restores a recurrent-state checkpoint and replays tokens)
  stats.promptTokens = promptIds.length;
  stats.cachedTokens = reuse;
  stats.generated = 0;
  stats.stopReason = 'length';

  const t0 = performance.now();
  const pre = await prefillTokens(model, promptIds, { from: reuse, signal: opts.signal, onProgress: opts.onPrefillProgress });
  let logits = pre.logits;
  if (pre.aborted) { stats.stopReason = 'aborted'; stats.prefillMs = performance.now() - t0; return; }
  stats.prefillMs = performance.now() - t0;
  stats.prefilled = promptIds.length - reuse;

  const history = promptIds.slice();
  const tDecode = performance.now();
  try {
    for (let n = 0; n < max; n++) {
      if (opts.signal?.aborted) { stats.stopReason = 'aborted'; break; }
      const id = sample(logits, opts.penalizePrompt === false ? history.slice(promptIds.length) : history);
      if (stop.has(id)) { stats.stopReason = 'stop'; break; }
      yield id;
      history.push(id);
      stats.generated = n + 1;
      if (n + 1 >= max) break;
      if (model.pos >= model.ctx) { stats.stopReason = 'context'; break; }
      logits = await model.forward(id);
      await yieldToEventLoop();
    }
  } finally {
    stats.decodeMs = performance.now() - tDecode;
  }
}
