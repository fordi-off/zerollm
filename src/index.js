// Public library entry of zerollm: load a GGUF model and stream text from it.
//
//   import { loadModel, inspectModel } from 'zerollm';        // (path: src/index.js)
//   const engine = await loadModel('models/x.gguf', { ctx: 4096, threads: 6, kv: 'f16' });
//   const stats = {};
//   for await (const text of engine.generate({ messages, sampling: { temperature: 0.6 }, stop: ['###'], stats })) process.stdout.write(text);
//   await engine.close();
//
// Everything here is plain JS on Node built-ins (worker threads + hand-emitted WebAssembly SIMD kernels).
import fs from 'node:fs';
import os from 'node:os';
import { readGGUF } from './gguf/reader.js';
import { typeName } from './gguf/types.js';
import { modelConfig, estimateRam, assertSupportedTensorTypes, SUPPORTED_ARCHS, SUPPORTED_TENSOR_TYPES, UnsupportedArchError } from './model/config.js';
import { Tokenizer } from './tokenizer/bpe.js';
import { detectTemplate, encodeChat, stopTokenIds } from './tokenizer/chat-templates.js';
import { createSampler } from './sampling/sampler.js';
import { StopFilter } from './sampling/stop.js';
import { generate as generateTokens, prefillTokens } from './model/generate.js';
import { PoolModel } from './compute/pool.js';
import { PrefixCache, modelFingerprint } from './model/prefix-cache.js';

export { SUPPORTED_ARCHS, SUPPORTED_TENSOR_TYPES };

/** Thrown by loadModel when the model would not fit in RAM. `needed` / `free` are in bytes. */
export class EngineMemoryError extends Error {
  constructor(message, details) { super(message); this.name = 'EngineMemoryError'; this.code = 'ENOMEM_ESTIMATE'; Object.assign(this, details); }
}

/** Thrown when a file cannot be run by this engine (unknown architecture / quantization / tokenizer). */
export class UnsupportedModelError extends Error {
  constructor(message) { super(message); this.name = 'UnsupportedModelError'; this.code = 'UNSUPPORTED_MODEL'; }
}

/** A guess at the physical core count: logical processors / 2, at least 1. */
export function defaultThreads() { return Math.max(1, Math.floor((os.availableParallelism?.() ?? os.cpus().length) / 2)); }

/**
 * Read a model file's header (no weights are loaded) and say whether this engine can run it.
 * @param {string} path
 * @returns {{path: string, sizeBytes: number, supported: boolean, reason: string|null, arch: string|null, name: string|null,
 *            params: number, quant: string, nLayer?: number, ctxTrain?: number, cfg?: object, g?: object}}
 */
export function inspectModel(path) {
  const sizeBytes = fs.statSync(path).size;
  const out = { path, sizeBytes, supported: false, reason: null, arch: null, name: null, params: 0, quant: '' };
  let g;
  try { g = readGGUF(path); } catch (e) { out.reason = `Not a readable GGUF file: ${e.message}`; return out; }
  out.name = g.get('general.name', null);
  out.arch = g.get('general.architecture', null);
  const hist = new Map();
  for (const t of g.tensors) {
    out.params += t.nElements;
    if (t.dims.length === 2) hist.set(t.type, (hist.get(t.type) ?? 0) + (t.nBytes ?? 0));
  }
  const main = [...hist].sort((a, b) => b[1] - a[1])[0];
  out.quant = main ? typeName(main[0]) : '';
  try {
    const cfg = modelConfig(g);
    out.cfg = cfg; out.nLayer = cfg.nLayer; out.ctxTrain = cfg.nCtxTrain;
    assertSupportedTensorTypes(g);
    if (g.get('tokenizer.ggml.model') !== 'gpt2') throw new UnsupportedModelError(`unsupported tokenizer "${g.get('tokenizer.ggml.model')}"`);
    out.supported = true;
  } catch (e) {
    out.reason = e instanceof UnsupportedArchError ? `Architecture "${out.arch}" is not supported (supported: ${SUPPORTED_ARCHS.join(', ')})` : e.message;
  }
  out.g = g;
  return out;
}

/**
 * Estimated RAM for running an inspected model.
 * @param {ReturnType<typeof inspectModel>} info
 */
export function estimateModelRam(info, { ctx = 2048, threads = defaultThreads(), kv = 'f32' } = {}) {
  if (!info.g || !info.cfg) return null;
  return estimateRam(info.g, info.cfg, { ctx: Math.min(ctx, info.cfg.nCtxTrain), threads, kvBytes: kv === 'f16' ? 2 : 4 });
}

/**
 * Load a model.
 * @param {string} path
 * @param {{ctx?: number, threads?: number, kv?: 'f32'|'f16', kernels?: 'auto'|'wasm'|'js', maxBatch?: number,
 *          template?: 'qwen'|'llama3'|'raw', ramCheck?: boolean, freeBytes?: number, onProgress?: (done: number, total: number) => void}} [opts]
 * @returns {Promise<Engine>}
 */
export async function loadModel(path, opts = {}) {
  const t0 = performance.now();
  const g = readGGUF(path);
  const cfg = modelConfig(g);
  try { assertSupportedTensorTypes(g); } catch (e) { throw new UnsupportedModelError(e.message); }
  if (g.get('tokenizer.ggml.model') !== 'gpt2') throw new UnsupportedModelError(`unsupported tokenizer "${g.get('tokenizer.ggml.model')}"`);
  const threads = opts.threads ?? defaultThreads();
  const kv = opts.kv ?? 'f32';
  const ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
  const est = estimateRam(g, cfg, { ctx, threads, kvBytes: kv === 'f16' ? 2 : 4 });
  if (opts.ramCheck !== false) {
    const free = opts.freeBytes ?? os.freemem();
    if (est.perWorker > 3.8 * 2 ** 30) {
      throw new EngineMemoryError(`the busiest worker would need ${(est.perWorker / 2 ** 30).toFixed(1)} GiB of WebAssembly memory (limit 4 GiB): use more threads, a smaller context or kv=f16`, { needed: est.total, free, reason: 'worker-limit' });
    }
    if (est.total > free * 0.95) {
      throw new EngineMemoryError(`needs about ${(est.total / 1e9).toFixed(1)} GB (weights + KV cache + runtime) but only ${(free / 1e9).toFixed(1)} GB is free`, { needed: est.total, free, reason: 'ram' });
    }
  }
  const tok = Tokenizer.fromGGUF(g);
  const jinja = g.get('tokenizer.chat_template');
  const kind = opts.template ?? detectTemplate(cfg.arch, jinja, tok);
  const model = await PoolModel.create(g, { ctx, workers: threads, kernels: opts.kernels ?? 'auto', kv, maxBatch: opts.maxBatch, onProgress: opts.onProgress });
  return new Engine({ path, g, cfg, tok, kind, jinja, model, ctx, threads, kv, estimate: est, loadMs: performance.now() - t0 });
}

/** Tokenizer + chat template of a model, without its weights: exact prompt token counts for a model that is not loaded. */
export class PromptTools {
  constructor({ tok, kind, cfg, jinja }) { this.tok = tok; this.kind = kind; this.cfg = cfg; this.jinja = jinja; }

  /** Token count of plain text (special-token text is NOT parsed). */
  countTokens(text) { return this.tok.encode(text, { special: false, bos: false }).length; }

  /** Token ids of a chat prompt exactly as it will be fed to the model. `think` = let Qwen3 reason (default off). */
  promptIds(messages, { think = false } = {}) {
    if (this.kind === 'raw') return this.tok.encode(messages.map((m) => m.content).join('\n\n'));
    return encodeChat(this.tok, this.kind, messages, { arch: this.cfg.arch, jinja: this.jinja, think });
  }

  /** Number of prompt tokens a chat would take (exact). */
  countPromptTokens(messages, opts) { return this.promptIds(messages, opts).length; }

  /** Token ids of a raw prompt (special-token text such as <|im_start|> is parsed). */
  rawIds(prompt) { return this.tok.encode(prompt, { special: true }); }
}

/** Read only the tokenizer and template of a model file (about a second for a 128k vocabulary). */
export function loadTokenizer(path, opts = {}) {
  const g = readGGUF(path);
  const cfg = modelConfig(g);
  const tok = Tokenizer.fromGGUF(g);
  const jinja = g.get('tokenizer.chat_template');
  return new PromptTools({ tok, cfg, jinja, kind: opts.template ?? detectTemplate(cfg.arch, jinja, tok) });
}

/** A loaded model. One generation at a time (calls are queued by the caller / the app). */
export class Engine extends PromptTools {
  constructor(p) {
    super({ tok: p.tok, kind: p.kind, cfg: p.cfg, jinja: p.jinja });
    this.path = p.path; this.g = p.g;
    this.model = p.model; this.ctx = p.ctx; this.threads = p.threads; this.kv = p.kv; this.estimate = p.estimate; this.loadMs = p.loadMs;
    this.closed = false;
    this.stopIds = stopTokenIds(p.tok, p.kind);
  }

  get info() {
    return {
      name: this.g.get('general.name', null), arch: this.cfg.arch, template: this.kind, ctx: this.ctx, threads: this.threads, kv: this.kv,
      kernels: this.model.kernels === 'wasm' ? 'wasm-simd' : 'js', loadMs: this.loadMs, estimate: this.estimate,
      bos: this.tok.bosId, eos: this.tok.eosId, vocab: this.tok.vocabSize, hasChatTemplate: typeof this.jinja === 'string',
    };
  }

  /**
   * Stream a reply.
   * @param {object} o
   * @param {Array<{role: string, content: string}>} [o.messages]  chat mode (template applied)
   * @param {string} [o.prompt]                                      raw mode (no template; special-token text is parsed)
   * @param {boolean} [o.think]                                      Qwen3 thinking (default false)
   * @param {number} [o.maxTokens]
   * @param {import('./sampling/sampler.js').SamplerOptions} [o.sampling]
   * @param {string[]} [o.stop]                                      stop strings (not included in the output)
   * @param {{aborted: boolean}|AbortSignal} [o.signal]
   * @param {object} [o.stats]                                       filled with promptTokens, cachedTokens, prefillMs, decodeMs, generated, stopReason
   * @param {(done: number, total: number) => void} [o.onPrefill]    prompt-reading progress
   * @returns {AsyncGenerator<string>} text chunks
   */
  async *generate(o) {
    if (this.closed) throw new Error('engine is closed');
    const ids = o.prompt !== undefined ? this.rawIds(o.prompt) : this.promptIds(o.messages ?? [], { think: o.think });
    if (ids.length >= this.ctx) throw new Error(`prompt is ${ids.length} tokens but the context window is ${this.ctx}`);
    const stats = o.stats ?? {};
    let stoppedByString = false;
    const external = o.signal;
    const signal = { get aborted() { return stoppedByString || external?.aborted === true; } };
    const sample = createSampler(o.sampling ?? {});
    const filter = new StopFilter(o.stop ?? []);
    const dec = this.tok.streamDecoder();
    const gen = generateTokens(this.model, ids, {
      maxTokens: o.maxTokens ?? 512, sample, stopIds: this.stopIds, signal, stats, onPrefillProgress: o.onPrefill,
    });
    for await (const id of gen) {
      const text = filter.push(dec.push(id));
      if (filter.stopped) stoppedByString = true;
      if (text) yield text;
    }
    if (!filter.stopped) {
      const tail = filter.push(dec.flush()) + filter.flush();
      if (tail) yield tail;
    }
    if (stoppedByString && stats.stopReason === 'aborted') stats.stopReason = 'stop-string';
  }

  /**
   * Make the model hold the KV state of `prefixIds` (typically the system prompt), from the on-disk cache when it has
   * been computed before, otherwise by reading it once and saving it. Only acts on a fresh model (nothing cached yet).
   * Returns 'hit' | 'saved' | 'unsaved' | 'off' (model not fresh / too short) | 'aborted'.
   * @param {number[]} prefixIds  must be a prefix of the prompt that will be generated next
   * @param {{dir: string, minTokens?: number, signal?: {aborted: boolean}, onProgress?: (done: number, total: number) => void}} o
   */
  async primePrefix(prefixIds, { dir, minTokens = 32, signal, onProgress } = {}) {
    if (this.closed || this.model.pos !== 0 || prefixIds.length < minTokens || prefixIds.length >= this.ctx) return 'off';
    const cache = new PrefixCache(dir, modelFingerprint(this.path, `${this.model.kernels === 'wasm' ? 'wasm' : 'js'}/kv-${this.model.kv ?? 'f32'}`));
    if (await cache.load(this.model, prefixIds)) return 'hit';
    const r = await prefillTokens(this.model, prefixIds, { wantLogits: false, signal, onProgress });
    if (r.aborted) return 'aborted';
    try { await cache.save(this.model, prefixIds); return 'saved'; } catch { return 'unsaved'; }
  }

  /** Forget the conversation held in the KV cache (the next prompt is processed from scratch). */
  reset() { this.model.reset(); }

  /** Number of tokens currently held in the KV cache. */
  get cachedTokens() { return this.model.pos; }

  async close() {
    if (this.closed) return;
    this.closed = true;
    await this.model.close();
  }
}

export { PrefixCache, modelFingerprint };
