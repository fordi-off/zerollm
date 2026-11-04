// Shared CLI plumbing for run / chat / bench: engine loading, sampler flags, streaming output.
import os from 'node:os';
import { readGGUF } from '../gguf/reader.js';
import { modelConfig, estimateRam, assertSupportedTensorTypes } from '../model/config.js';
import { RefModel } from '../model/forward.js';
import { PoolModel } from '../compute/pool.js';
import path from 'node:path';
import { generate, prefillTokens } from '../model/generate.js';
import { PrefixCache, modelFingerprint } from '../model/prefix-cache.js';
import { encodeChat } from '../tokenizer/chat-templates.js';
import { Tokenizer } from '../tokenizer/bpe.js';
import { detectTemplate, stopTokenIds, TEMPLATE_KINDS } from '../tokenizer/chat-templates.js';
import { createSampler } from '../sampling/sampler.js';
import { StopFilter } from '../sampling/stop.js';
import { num } from '../util/args.js';
import { formatBytes, warn } from '../util/log.js';

/** Shorter system prompts are not worth a cache file. */
export const MIN_CACHED_PREFIX = 32;

export const SAMPLER_HELP = '--temp 0.7 --top-k 40 --top-p 0.9 --min-p 0 --repeat-penalty 1 --seed N  (--temp 0 = greedy)';

/** @param {Record<string, any>} flags */
export function samplerFromFlags(flags) {
  return createSampler({
    temperature: num(flags.temp, 0.7),
    topK: num(flags['top-k'], 40),
    topP: num(flags['top-p'], 0.9),
    minP: num(flags['min-p'], 0),
    repeatPenalty: num(flags['repeat-penalty'], 1),
    repeatLastN: num(flags['repeat-last-n'], 64),
    presencePenalty: num(flags['presence-penalty'], 0),
    frequencyPenalty: num(flags['frequency-penalty'], 0),
    seed: flags.seed === undefined ? undefined : num(flags.seed, 0),
  });
}

/** Default worker count: a guess at physical cores (logical / 2), at least 1. */
export function defaultThreads() { return Math.max(1, Math.floor((os.availableParallelism?.() ?? os.cpus().length) / 2)); }

/** @param {Record<string, any>} flags */
export function kernelsFromFlags(flags) {
  const k = flags.kernels === undefined ? 'auto' : String(flags.kernels);
  if (!['auto', 'wasm', 'js'].includes(k)) throw new Error(`--kernels must be auto, wasm or js (got "${k}")`);
  return k;
}

/** @param {Record<string, any>} flags */
export function kvFromFlags(flags) {
  const k = flags.kv === undefined ? 'f32' : String(flags.kv);
  if (k !== 'f32' && k !== 'f16') throw new Error(`--kv must be f32 or f16 (got "${k}")`);
  return k;
}

/** @param {Record<string, any>} flags */
export function threadsFromFlags(flags) {
  if (flags.threads === undefined || flags.threads === 'auto' || flags.threads === true) return defaultThreads();
  const n = num(flags.threads, 1);
  if (!Number.isInteger(n) || n < 1) throw new Error(`--threads must be a positive integer, got "${flags.threads}"`);
  return n;
}

/** Progress printer for model loading (stderr, TTY only). */
function loadProgress() {
  if (!process.stderr.isTTY) return undefined;
  let last = 0;
  return (d, n) => { const now = Date.now(); if (now - last > 100 || d === n) { last = now; process.stderr.write(`\rloading ${Math.round((100 * d) / n)}%`); } };
}

/**
 * Create the compute model: the single-thread reference path for 1 thread, the SPMD worker pool otherwise.
 * @param {import('../gguf/reader.js').GGUFFile} g
 */
export async function createModel(g, { threads, ctx, spin, kernels = 'auto', maxBatch, kv = 'f32', onProgress }) {
  // `--threads 1 --kernels js` is the plain single-thread reference path (the test oracle); everything else uses the pool
  // (the reference path keeps an f32 KV cache, so --kv f16 uses the pool even with --threads 1 --kernels js)
  const model = threads === 1 && kernels === 'js' && kv === 'f32' ? new RefModel(g, { ctx, onProgress }) : await PoolModel.create(g, { ctx, workers: threads, spin, kernels, maxBatch, kv, onProgress });
  if (process.stderr.isTTY) process.stderr.write('\r\x1b[K');
  return model;
}

/**
 * Load GGUF + tokenizer + model.
 * @param {string} modelPath
 * @param {Record<string, any>} flags
 */
export async function loadEngine(modelPath, flags) {
  if (!modelPath) throw new Error('missing model path (e.g. models/Qwen3-0.6B-Q8_0.gguf)');
  const t0 = performance.now();
  const g = readGGUF(modelPath);
  const cfg = modelConfig(g);
  assertSupportedTensorTypes(g);
  const ctx = num(flags.ctx, 2048);
  const threads = threadsFromFlags(flags);
  const kv = kvFromFlags(flags);
  const est = estimateRam(g, cfg, { ctx, threads, kvBytes: kv === 'f16' ? 2 : 4 });
  const free = os.freemem();
  if (est.perWorker > 3.8 * 2 ** 30) throw new Error(`the busiest worker would need ${formatBytes(est.perWorker)} of WebAssembly memory, over the 4 GB wasm32 limit: use more --threads, a smaller --ctx or --kv f16`);
  if (est.total > free * 0.9) {
    const alt = kv === 'f32' ? estimateRam(g, cfg, { ctx, threads, kvBytes: 2 }) : null;
    const advice = alt && alt.total <= free * 0.9 ? `--kv f16 would bring it to ${formatBytes(alt.total)}` : 'try --kv f16, a smaller --ctx, a smaller quantization, or close other apps';
    warn(`estimated RAM need ${formatBytes(est.total)} (weights ${formatBytes(est.weights)} + KV cache ${formatBytes(est.kv)} + runtime) vs ${formatBytes(free)} free: ${advice}.`);
  }
  const tok = Tokenizer.fromGGUF(g);
  const model = await createModel(g, { threads, ctx, kernels: kernelsFromFlags(flags), kv: kvFromFlags(flags), maxBatch: flags.batch === undefined ? undefined : num(flags.batch, 16), spin: flags.spin === undefined ? undefined : num(flags.spin, 20000), onProgress: loadProgress() });
  let kind = typeof flags.template === 'string' ? flags.template : flags.raw ? 'raw' : detectTemplate(cfg.arch, g.get('tokenizer.chat_template'), tok);
  if (!TEMPLATE_KINDS.includes(kind)) { await model.close?.(); throw new Error(`unknown --template "${kind}" (use ${TEMPLATE_KINDS.join(', ')})`); }
  return {
    g, cfg, tok, model, kind, ctx: model.ctx, threads, kernels: model.kernels === 'wasm' ? 'wasm-simd' : 'js',
    jinja: g.get('tokenizer.chat_template'),
    stopIds: stopTokenIds(tok, kind),
    loadMs: performance.now() - t0,
    close: async () => { await model.close?.(); },
  };
}

/** Print the exact prompt the model will see: token count, ids, and the text with control tokens visible (--show-prompt). */
export function describePrompt(eng, ids, err = process.stderr) {
  const shown = ids.length > 120 ? [...ids.slice(0, 60), null, ...ids.slice(-60)] : ids;
  err.write(`[prompt: ${ids.length} tokens, template ${eng.kind}]\n`);
  err.write(`[ids: ${shown.map((i) => (i === null ? '...' : i)).join(' ')}]\n`);
  err.write(`[text: ${JSON.stringify(eng.tok.decode(ids, { skipControl: false }))}]\n`);
}

/** Template options derived from flags + model. */
export function templateOptions(eng, flags) {
  return { arch: eng.cfg.arch, jinja: eng.jinja, think: flags.think === true };
}

/**
 * Make sure the KV state of the system-prompt prefix is in the model, from the on-disk cache when possible.
 * Only acts on a fresh model (pos 0). Returns 'hit' | 'saved' | 'off' | 'skip' | 'unsaved'.
 * @param {number[]} promptIds full prompt   @param {{role: string, content: string}[]} systemMessages usually [] or one system message
 */
export async function usePrefixCache(eng, flags, systemMessages, promptIds, tmplOpts, signal) {
  if (flags.cache === false || eng.kind === 'raw' || eng.model.pos !== 0) return 'off';
  const prefix = encodeChat(eng.tok, eng.kind, systemMessages, { ...tmplOpts, generationPrompt: false });
  if (prefix.length < MIN_CACHED_PREFIX || prefix.length >= promptIds.length || prefix.some((t, i) => t !== promptIds[i])) return 'skip';
  const dir = typeof flags['cache-dir'] === 'string' ? flags['cache-dir'] : path.join(path.dirname(eng.g.path), '.zerollm-cache');
  const cache = new PrefixCache(dir, modelFingerprint(eng.g.path, `${eng.model.kernels === 'wasm' ? 'wasm' : 'js'}/kv-${eng.model.kv ?? 'f32'}`));
  if (await cache.load(eng.model, prefix)) return 'hit';
  const r = await prefillTokens(eng.model, prefix, { wantLogits: false, signal });
  if (r.aborted) return 'skip';
  try { await cache.save(eng.model, prefix); return 'saved'; } catch (e) { warn(`could not write the system-prompt cache (${e.message}); continuing without it`); return 'unsaved'; }
}

/** Install a SIGINT handler that sets `signal.aborted`; a second Ctrl+C exits immediately. Returns a disposer. */
export function installAbort(signal, rl = null) {
  let presses = 0;
  const handler = () => {
    presses++;
    if (presses > 1) { process.stderr.write('\n'); process.exit(130); }
    signal.aborted = true;
  };
  const target = rl ?? process;
  target.on('SIGINT', handler);
  return () => { target.off('SIGINT', handler); };
}

/**
 * Generate and stream a reply to stdout.
 * @returns {Promise<{text: string, stats: any}>}
 */
export async function streamReply(eng, promptIds, { sample, maxTokens, stops = [], signal, out = process.stdout, err = process.stderr, progress = err.isTTY, stats = {} }) {
  const dec = eng.tok.streamDecoder();
  const filter = new StopFilter(stops);
  let text = '';
  // The progress line lives on stderr and is erased exactly once, BEFORE any reply text is written. Never erase
  // after text has been printed: on a terminal that would wipe the last line of the answer.
  let progressShown = false;
  const clearProgress = () => { if (progressShown) { err.write('\r\x1b[K'); progressShown = false; } };
  const emit = (s) => { if (s) { clearProgress(); out.write(s); text += s; } };
  for await (const id of generate(eng.model, promptIds, {
    maxTokens, sample, stopIds: eng.stopIds, signal, stats,
    onPrefillProgress: progress ? (d, n) => { if (n > 8) { err.write(`\rprefill ${d}/${n}`); progressShown = true; } } : undefined,
  })) {
    emit(filter.push(dec.push(id)));
    if (filter.stopped) { signal.aborted = true; stats.stopReason = 'stop-string'; }
  }
  if (!filter.stopped) { emit(filter.push(dec.flush())); emit(filter.flush()); }
  clearProgress(); // only does anything if no reply text was ever written
  return { text, stats };
}

export function formatStats(eng, stats) {
  const rate = (n, ms) => (ms > 0 && n > 0 ? `${((n * 1000) / ms).toFixed(2)} tok/s` : 'n/a');
  const parts = [
    `prompt ${stats.promptTokens} tok${stats.cachedTokens ? ` (${stats.cachedTokens} cached)` : ''}, prefill ${rate(stats.prefilled ?? 0, stats.prefillMs)}`,
    `generated ${stats.generated} tok, decode ${rate(stats.generated, stats.decodeMs)}`,
    ...(stats.prefixCache === 'hit' || stats.prefixCache === 'saved' ? [`system-prompt cache: ${stats.prefixCache}`] : []),
    `stop: ${stats.stopReason}`,
    `RSS ${formatBytes(process.memoryUsage().rss)}`,
  ];
  return `[${parts.join(' | ')}]`;
}
