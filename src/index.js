// Public library entry of zerollm: load a GGUF model and stream text from it.
//
//   import { loadModel, inspectModel } from 'zerollm';        // (path: src/index.js)
//   const engine = await loadModel('models/x.gguf', { ctx: 4096, threads: 6, kv: 'f16' });
//   const stats = {};
//   for await (const text of engine.generate({ messages, sampling: { temperature: 0.6 }, stop: ['###'], stats })) process.stdout.write(text);
//   await engine.close();
//
// Speculative decoding (a small draft model of the same family proposes tokens, the big model verifies them in one batch):
//   const engine = await loadModel('models/Qwen3-8B-Q4_K_M.gguf', { threads: 6, draft: { path: 'models/Qwen3-0.6B-Q8_0.gguf', k: 4, threads: 4 } });
//   // engine.generate(...) now uses it transparently; stats.speculative reports acceptance, draft/verify time, effective tok/s
//
// Everything here is plain JS on Node built-ins (worker threads + hand-emitted WebAssembly SIMD kernels).
import fs from 'node:fs';
import os from 'node:os';
import { readGGUF } from './gguf/reader.js';
import { typeName } from './gguf/types.js';
import { modelConfig, estimateRam, assertSupportedTensorTypes, SUPPORTED_ARCHS, SUPPORTED_TENSOR_TYPES, UnsupportedArchError } from './model/config.js';
import { RefModel } from './model/forward.js';
import { Tokenizer } from './tokenizer/bpe.js';
import { detectTemplate, encodeChat, stopTokenIds, thinkingMode, thinkOpening } from './tokenizer/chat-templates.js';
import { createSampler } from './sampling/sampler.js';
import { StopFilter } from './sampling/stop.js';
import { generate as generateTokens, prefillTokens } from './model/generate.js';
import { generateSpeculative, verifyLimit, vocabMismatch, vocabFingerprint, defaultDraftThreads, DEFAULT_DRAFT_K } from './model/speculative.js';
import { PoolModel } from './compute/pool.js';
import { PrefixCache, modelFingerprint } from './model/prefix-cache.js';

export { SUPPORTED_ARCHS, SUPPORTED_TENSOR_TYPES };

/**
 * @typedef {Object} DraftOptions
 * @property {string} path        draft model file (same family / tokenizer as the target, e.g. Qwen3-0.6B for Qwen3-8B)
 * @property {number} [k]         tokens drafted per verification pass (default 3; at most the target's maxBatch - 1)
 * @property {number} [threads]   draft worker threads (default min(target threads, 4); draft and target never run at the same time)
 * @property {number} [ctx]       draft context (default: the target's)
 * @property {'f32'|'f16'} [kv]   draft KV precision (default: the target's)
 * @property {'auto'|'wasm'|'js'} [kernels]
 * @property {number} [maxBatch]
 */

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
 *          template?: 'qwen'|'llama3'|'raw', ramCheck?: boolean, freeBytes?: number, onProgress?: (done: number, total: number) => void,
 *          draft?: DraftOptions}} [opts]
 *   draft: also load a small draft model of the same family and use it for speculative decoding (see Engine#useDraft). Its
 *   weights count in the RAM check; UnsupportedModelError is thrown (before anything is loaded) when the vocabularies differ.
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
  // { threads: 1, kernels: 'js' } (f32 KV) is the plain single-thread reference path, the oracle of the pool; hybrid models run it
  // on request (the CLI does this for every architecture). Everything else, and every other architecture, uses the worker pool.
  const referencePath = cfg.hybrid && opts.threads === 1 && opts.kernels === 'js' && kv === 'f32';
  const ctx = Math.min(opts.ctx ?? 2048, cfg.nCtxTrain);
  const est = estimateRam(g, cfg, { ctx, threads, kvBytes: kv === 'f16' ? 2 : 4 });

  // the draft is validated BEFORE any weights are loaded (cheap, and a wrong model is reported immediately)
  const dopts = opts.draft ? { ...opts.draft } : null;
  let dg = null; let dthreads = 0; let dctx = 0; let dkv = kv; let dest = null;
  if (dopts) {
    if (typeof dopts.path !== 'string' || !dopts.path) throw new UnsupportedModelError('draft.path is missing');
    dg = readGGUF(dopts.path);
    const dcfg = modelConfig(dg);
    try { assertSupportedTensorTypes(dg); } catch (e) { throw new UnsupportedModelError(`draft model: ${e.message}`); }
    if (dg.get('tokenizer.ggml.model') !== 'gpt2') throw new UnsupportedModelError(`draft model: unsupported tokenizer "${dg.get('tokenizer.ggml.model')}"`);
    const bad = vocabMismatch(g, dg);
    if (bad) throw new UnsupportedModelError(bad);
    dthreads = dopts.threads ?? defaultDraftThreads(threads);
    dkv = dopts.kv ?? kv;
    dctx = Math.min(dopts.ctx ?? ctx, dcfg.nCtxTrain);
    dest = estimateRam(dg, dcfg, { ctx: dctx, threads: dthreads, kvBytes: dkv === 'f16' ? 2 : 4 });
  }

  if (opts.ramCheck !== false) {
    const free = opts.freeBytes ?? os.freemem();
    if (est.perWorker > 3.8 * 2 ** 30) {
      throw new EngineMemoryError(`the busiest worker would need ${(est.perWorker / 2 ** 30).toFixed(1)} GiB of WebAssembly memory (limit 4 GiB): use more threads, a smaller context or kv=f16`, { needed: est.total, free, reason: 'worker-limit' });
    }
    if (dest && dest.perWorker > 3.8 * 2 ** 30) {
      throw new EngineMemoryError(`the busiest draft worker would need ${(dest.perWorker / 2 ** 30).toFixed(1)} GiB of WebAssembly memory (limit 4 GiB): use more draft threads`, { needed: est.total + dest.total, free, reason: 'worker-limit' });
    }
    const needed = est.total + (dest?.total ?? 0);
    if (needed > free * 0.95) {
      throw new EngineMemoryError(dest
        ? `needs about ${(needed / 1e9).toFixed(1)} GB (target ${(est.total / 1e9).toFixed(1)} GB + draft model ${(dest.total / 1e9).toFixed(1)} GB; weights + KV cache + runtime) but only ${(free / 1e9).toFixed(1)} GB is free`
        : `needs about ${(est.total / 1e9).toFixed(1)} GB (weights + KV cache + runtime) but only ${(free / 1e9).toFixed(1)} GB is free`, { needed, free, reason: 'ram' });
    }
  }
  const tok = Tokenizer.fromGGUF(g);
  const jinja = g.get('tokenizer.chat_template');
  const kind = opts.template ?? detectTemplate(cfg.arch, jinja, tok);
  const model = referencePath
    ? new RefModel(g, { ctx, onProgress: opts.onProgress })
    : await PoolModel.create(g, { ctx, workers: threads, kernels: opts.kernels ?? 'auto', kv, maxBatch: opts.maxBatch, onProgress: opts.onProgress });
  const engine = new Engine({ path, g, cfg, tok, kind, jinja, model, ctx, threads, kv, estimate: est, loadMs: performance.now() - t0 });
  if (dopts) {
    try {
      const draftEngine = await loadModel(dopts.path, { ctx: dctx, threads: dthreads, kv: dkv, kernels: dopts.kernels ?? opts.kernels, maxBatch: dopts.maxBatch, template: 'raw', ramCheck: false });
      engine.useDraft(draftEngine, { k: dopts.k, own: true });
      engine.draftEstimate = dest;
    } catch (e) { await engine.close(); throw e; }
    engine.loadMs = performance.now() - t0;
  }
  return engine;
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

  /** 'model-opens' (Qwen3) | 'prompt-opens' (Qwen3.5) | null: does this model have a thinking switch, and who writes `<think>` (see chat-templates.js). */
  get thinkingMode() { return thinkingMode(this.cfg.arch); }

  /** Text to put in front of the generated reply of a thinking run so it reads `<think>..</think>answer` (Qwen3.5's prompt already opened the block). */
  thinkOpening(think) { return thinkOpening(this.cfg.arch, think === true); }

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
    /** @type {{engine: Engine, k: number, own: boolean}|null} the attached draft model (speculative decoding) */
    this.draft = null;
    /** RAM estimate of the draft model (set by loadModel when it loaded one) */
    this.draftEstimate = null;
  }

  get info() {
    return {
      name: this.g.get('general.name', null), arch: this.cfg.arch, template: this.kind, ctx: this.ctx, threads: this.threads, kv: this.kv,
      kernels: this.model.kernels === 'wasm' ? 'wasm-simd' : 'js', loadMs: this.loadMs, estimate: this.estimate,
      bos: this.tok.bosId, eos: this.tok.eosId, vocab: this.tok.vocabSize, hasChatTemplate: typeof this.jinja === 'string',
      draft: this.draft ? { name: this.draft.engine.g.get('general.name', null), path: this.draft.engine.path, k: this.draft.k, threads: this.draft.engine.threads, ctx: this.draft.engine.ctx, estimate: this.draftEstimate } : null,
    };
  }

  /**
   * Use `draftEngine` (a model of the same family, loaded with loadModel) as the draft model of speculative decoding:
   * generate() then drafts k tokens with it and verifies them in one batched pass of this model. Greedy output is
   * identical to plain decoding; sampled output has exactly the target's distribution. Calling it again replaces the draft
   * (or changes k). While attached, the draft engine must not be used for anything else.
   * @param {Engine} draftEngine
   * @param {{k?: number, own?: boolean}} [o]  k: tokens drafted per pass (default 3, at most maxBatch - 1 of this model);
   *   own: close the draft engine together with this one (loadModel's `draft` option does this)
   * @throws {UnsupportedModelError} when the vocabularies differ
   */
  useDraft(draftEngine, { k = DEFAULT_DRAFT_K, own = false } = {}) {
    if (this.closed) throw new Error('engine is closed');
    if (!(draftEngine instanceof Engine)) throw new TypeError('useDraft expects an Engine made by loadModel');
    if (draftEngine === this) throw new Error('a model cannot draft for itself (load a second, smaller model)');
    if (draftEngine.closed) throw new Error('the draft engine is closed');
    const bad = vocabMismatch(this.g, draftEngine.g);
    if (bad) throw new UnsupportedModelError(bad);
    if (draftEngine.cfg.vocabSize !== this.cfg.vocabSize) throw new UnsupportedModelError(`the models disagree on the logits vocabulary size (${draftEngine.cfg.vocabSize} vs ${this.cfg.vocabSize})`);
    if (!Number.isInteger(k) || k < 1) throw new RangeError(`draft k must be a positive integer, got ${k}`);
    const limit = verifyLimit(this.model);
    if (limit < 2) throw new UnsupportedModelError('speculative decoding needs a target loaded with maxBatch >= 2');
    if (this.draft && this.draft.engine !== draftEngine && this.draft.own) this.draft.engine.close();
    this.draft = { engine: draftEngine, k: Math.min(k, limit - 1), own };
    return this;
  }

  /** Stop using the draft model (closes it if this engine owns it). */
  async clearDraft() {
    const d = this.draft;
    this.draft = null;
    this.draftEstimate = null;
    if (d?.own) await d.engine.close();
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
   * @param {boolean} [o.penalizePrompt]                             false: repetition penalties ignore prompt tokens (default true)
   * @param {boolean} [o.speculative]                                false: do not use the draft model for this call (default: use it when attached)
   * @param {object} [o.stats]                                       filled with promptTokens, cachedTokens, prefillMs, decodeMs, generated, stopReason; with a draft
   *                                                                 model also `speculative`: {k, cycles, drafted, accepted, acceptRate, tokensPerCycle (accepted+1 per
   *                                                                 verification step), draftMs, verifyMs, otherMs, effectiveTps, acceptedHist, draftPrefillMs}
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
    const gopts = { maxTokens: o.maxTokens ?? 512, sample, stopIds: this.stopIds, signal, stats, onPrefillProgress: o.onPrefill, penalizePrompt: o.penalizePrompt };
    const d = o.speculative === false ? null : this.draft;
    if (d?.engine.closed) throw new Error('the draft engine was closed; call clearDraft() or useDraft() again');
    // a prompt the draft model's context cannot hold is generated without speculation
    const gen = d && ids.length < d.engine.ctx
      ? generateSpeculative(this.model, d.engine.model, ids, { ...gopts, k: d.k })
      : generateTokens(this.model, ids, gopts);
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
  reset() { this.model.reset(); this.draft?.engine.model.reset(); }

  /** Number of tokens currently held in the KV cache. */
  get cachedTokens() { return this.model.pos; }

  async close() {
    if (this.closed) return;
    this.closed = true;
    const d = this.draft;
    this.draft = null;
    try { if (d?.own) await d.engine.close(); } finally { await this.model.close(); }
  }
}

export { thinkingMode, thinkOpening, PrefixCache, modelFingerprint, vocabMismatch, vocabFingerprint, defaultDraftThreads, DEFAULT_DRAFT_K };
