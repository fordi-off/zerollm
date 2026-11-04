// Token generation loop shared by the CLI and (later) the server.

/** Greedy sampler: argmax of logits. @param {Float32Array} logits */
export function argmax(logits) {
  let best = 0; let bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}

const yieldToEventLoop = () => new Promise((r) => setImmediate(r));

/**
 * Generate tokens after `promptIds`. Yields token ids as they are produced (stop tokens are not yielded).
 * @param {{forward: Function, prefill: Function, pos: number}} model
 * @param {number[]} promptIds
 * @param {{maxTokens?: number, sample?: (logits: Float32Array, history: number[]) => number, stopIds?: Set<number>,
 *          signal?: {aborted: boolean}, stats?: object}} [opts]
 */
export async function* generate(model, promptIds, opts = {}) {
  const sample = opts.sample ?? argmax;
  const stop = opts.stopIds ?? new Set();
  const max = opts.maxTokens ?? 256;
  const stats = opts.stats ?? {};
  if (promptIds.length === 0) throw new Error('empty prompt');
  const t0 = performance.now();
  let logits = model.prefill(promptIds);
  const t1 = performance.now();
  stats.promptTokens = promptIds.length;
  stats.prefillMs = t1 - t0;
  stats.generated = 0;
  stats.stopReason = 'length';
  const history = promptIds.slice();
  let tDecode = performance.now();
  for (let n = 0; n < max; n++) {
    if (opts.signal?.aborted) { stats.stopReason = 'aborted'; break; }
    const id = sample(logits, history);
    if (stop.has(id)) { stats.stopReason = 'stop'; break; }
    yield id;
    history.push(id);
    stats.generated = n + 1;
    if (n + 1 >= max) break;
    if (model.pos >= model.ctx) { stats.stopReason = 'context'; break; }
    logits = model.forward(id);
    await yieldToEventLoop(); // lets SIGINT handlers run between tokens
  }
  stats.decodeMs = performance.now() - tDecode;
}
