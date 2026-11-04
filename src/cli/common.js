// Shared CLI plumbing for run / chat / bench: engine loading, sampler flags, streaming output.
import os from 'node:os';
import { readGGUF } from '../gguf/reader.js';
import { modelConfig, estimateRam } from '../model/config.js';
import { RefModel } from '../model/forward.js';
import { generate } from '../model/generate.js';
import { Tokenizer } from '../tokenizer/bpe.js';
import { detectTemplate, stopTokenIds, TEMPLATE_KINDS } from '../tokenizer/chat-templates.js';
import { createSampler } from '../sampling/sampler.js';
import { StopFilter } from '../sampling/stop.js';
import { num } from '../util/args.js';
import { formatBytes, warn } from '../util/log.js';

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

/**
 * Load GGUF + tokenizer + model.
 * @param {string} modelPath
 * @param {Record<string, any>} flags
 */
export function loadEngine(modelPath, flags) {
  if (!modelPath) throw new Error('missing model path (e.g. models/Qwen3-0.6B-Q8_0.gguf)');
  if (flags.threads !== undefined && num(flags.threads, 1) !== 1) warn('--threads is ignored until the worker pool lands (M5); running single-threaded.');
  const t0 = performance.now();
  const g = readGGUF(modelPath);
  const cfg = modelConfig(g);
  const ctx = num(flags.ctx, 2048);
  const est = estimateRam(g, cfg, { ctx });
  if (est.total > os.freemem() * 0.9) warn(`estimated RAM need ${formatBytes(est.total)} vs ${formatBytes(os.freemem())} free: close other apps or lower --ctx.`);
  const tok = Tokenizer.fromGGUF(g);
  const show = process.stderr.isTTY;
  let last = 0;
  const model = new RefModel(g, {
    ctx,
    onProgress: show ? (d, n) => { const now = Date.now(); if (now - last > 100 || d === n) { last = now; process.stderr.write(`\rloading ${d}/${n} tensors`); } } : undefined,
  });
  if (show) process.stderr.write('\r\x1b[K');
  let kind = typeof flags.template === 'string' ? flags.template : flags.raw ? 'raw' : detectTemplate(cfg.arch, g.get('tokenizer.chat_template'), tok);
  if (!TEMPLATE_KINDS.includes(kind)) throw new Error(`unknown --template "${kind}" (use ${TEMPLATE_KINDS.join(', ')})`);
  return {
    g, cfg, tok, model, kind, ctx: model.ctx,
    jinja: g.get('tokenizer.chat_template'),
    stopIds: stopTokenIds(tok, kind),
    loadMs: performance.now() - t0,
  };
}

/** Template options derived from flags + model. */
export function templateOptions(eng, flags) {
  return { arch: eng.cfg.arch, jinja: eng.jinja, think: flags.think === true };
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
export async function streamReply(eng, promptIds, { sample, maxTokens, stops = [], signal, out = process.stdout }) {
  const stats = {};
  const dec = eng.tok.streamDecoder();
  const filter = new StopFilter(stops);
  const show = process.stderr.isTTY;
  let text = '';
  const emit = (s) => { if (s) { out.write(s); text += s; } };
  for await (const id of generate(eng.model, promptIds, {
    maxTokens, sample, stopIds: eng.stopIds, signal, stats,
    onPrefillProgress: show ? (d, n) => { if (n > 8) process.stderr.write(`\rprefill ${d}/${n}`); } : undefined,
  })) {
    if (show && stats.generated === 0) process.stderr.write('\r\x1b[K');
    emit(filter.push(dec.push(id)));
    if (filter.stopped) { signal.aborted = true; stats.stopReason = 'stop-string'; }
  }
  if (show) process.stderr.write('\r\x1b[K');
  if (!filter.stopped) emit(filter.push(dec.flush())), emit(filter.flush());
  return { text, stats };
}

export function formatStats(eng, stats) {
  const rate = (n, ms) => (ms > 0 && n > 0 ? `${((n * 1000) / ms).toFixed(2)} tok/s` : 'n/a');
  const parts = [
    `prompt ${stats.promptTokens} tok${stats.cachedTokens ? ` (${stats.cachedTokens} cached)` : ''}, prefill ${rate(stats.prefilled ?? 0, stats.prefillMs)}`,
    `generated ${stats.generated} tok, decode ${rate(stats.generated, stats.decodeMs)}`,
    `stop: ${stats.stopReason}`,
    `RSS ${formatBytes(process.memoryUsage().rss)}`,
  ];
  return `[${parts.join(' | ')}]`;
}
