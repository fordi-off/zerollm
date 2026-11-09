// The API the page talks to. Every method takes one params object and returns JSON; streamed output goes out as events
// ({runId, ...}) over the SSE channel: phase, progress, chunk, done, error.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import v8 from 'node:v8';
import { loadTokenizer, EngineMemoryError, UnsupportedModelError, defaultThreads } from '../src/index.js';
import { probeWasmFeatures, probeJsFeatures } from '../src/compute/wasm/features.js';
import { defaultConfig, CONTEXT_STEPS } from './config.js';
import { listModels, modelPath } from './models.js';
import { createChatStore } from './chats.js';
import { createSkills } from './skills.js';
import { createDocuments } from './documents.js';
import { createEmbeddings } from './embeddings.js';
import { createDownloader, freeDiskBytes } from './downloader.js';
import { CATALOG } from './catalog.js';
import { createSummarizer } from './summarizer.js';
import { buildMessages, fitToContext, looksLooped, stablePrefixIds, sourcesOf, RECENT_KEEP, SUMMARIZE_AT, SUMMARY_PREFIX, SAMPLING, THINK_SAMPLING, THINK_MAX_TOKENS, DEFAULT_MAX_TOKENS, CONTINUATION_PROMPT } from './chat-core.js';

const I18N_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'i18n');

export function createApi({ paths, host, getConfig, setConfig, log, emit, catalog = CATALOG, retryDelayMs }) {
  let runSeq = 0;
  let current = null; // {id, abort: {aborted}}
  const chatStore = createChatStore(paths);
  const docs = createDocuments(paths, chatStore);
  const embeddings = createEmbeddings({ paths, getConfig, log });
  let indexing = null; // {abort, docId, filename, done, total} while document chunks are being embedded
  const pendingIndex = new Set(); // documents added while a reply was being written: indexed right after it
  const skills = createSkills(paths);
  const summarizer = createSummarizer({ paths, host, getConfig, log });
  skills.seed();
  /** Instruction text for the slash commands on the last user message (they apply to that one reply). */
  const extraFor = (history) => skills.bodies([...history].reverse().find((m) => m.role === 'user')?.commands) ?? '';
  let afterDownload = () => {};
  const downloader = createDownloader({ paths, log, emit, catalog, retryDelayMs, onDone: (e) => afterDownload(e) });
  const tokenizers = new Map(); // filename -> PromptTools (weights not loaded)

  const toolsFor = (filename) => {
    if (host.engine && host.filename === filename) return host.engine;
    if (!tokenizers.has(filename)) {
      if (tokenizers.size >= 2) tokenizers.delete(tokenizers.keys().next().value);
      tokenizers.set(filename, loadTokenizer(modelPath(paths, filename)));
    }
    return tokenizers.get(filename);
  };

  const contextMax = () => getConfig().context_window;

  /** The saved rolling summary of a chat, dropped when the history was edited back past what it covers. */
  function summaryState(chatId, historyLength) {
    if (!chatId) return { summary: '', summarized: 0 };
    let c; try { c = chatStore.loadSummary(chatId); } catch { return { summary: '', summarized: 0 }; }
    if (!c.summary || c.summarized_count < 1 || c.summarized_count > historyLength - 1) return { summary: '', summarized: 0 };
    return { summary: c.summary, summarized: c.summarized_count };
  }
  /** Excerpts of the chat's active documents for the last question (short follow-ups also use the question before).
   *  With `vector` and an installed embedding model the question is embedded and the ranking fuses keywords and meaning. */
  async function excerptsFor(chatId, history, countTokens, ctxMax, { vector = false } = {}) {
    if (!chatId) return [];
    const users = history.filter((m) => m.role === 'user').map((m) => String(m.content));
    const last = users.at(-1) ?? ''; let query = last;
    if (last.split(/\s+/).length < 5 && users.length > 1) query = `${users.at(-2)} ${last}`;
    try {
      const queryVec = vector && embeddings.available() && docs.hasEmbeddings(chatId) ? await embeddings.embedQuery(query) : null;
      return docs.retrieve(chatId, query, { queryVec, k: getConfig().doc_chunks ?? 4, budgetTokens: Math.min(1500, Math.round(ctxMax * 0.25)), countTokens });
    } catch { return []; }
  }

  /** While documents are being indexed, chat generation, summaries and model changes are refused: the search model needs every core. */
  const assertNotIndexing = () => { if (indexing) throw new Error('indexing'); };

  /**
   * Document context of the LAST user message. A message that already carries `context` (regenerate, continue, an older turn)
   * keeps it unchanged; otherwise the best matching chunks are retrieved, and those not already somewhere earlier in the
   * conversation become the message's new `context` (stored by the page with the message, so the prompt stays append-only).
   * @returns {Promise<{history: any[], context: any[]|undefined, sources: any[]}>} history = with the last user message's context set
   */
  async function withLastContext(chatId, history, countTokens, ctxMax, { retrieve = true, vector = false } = {}) {
    const i = history.map((m) => m.role).lastIndexOf('user');
    if (i < 0) return { history, context: undefined, sources: [] };
    const last = history[i];
    if (Array.isArray(last.context)) return { history, context: last.context, sources: sourcesOf(last.context) };
    if (!retrieve) return { history, context: undefined, sources: [] };
    const picked = await excerptsFor(chatId, history, countTokens, ctxMax, { vector });
    const seen = new Set(history.flatMap((m) => (Array.isArray(m.context) ? m.context.map((c) => c.text) : [])));
    const fresh = picked.filter((e) => !seen.has(e.text)).map(({ filename, text, score }) => ({ filename, text, score }));
    const out = history.slice(); out[i] = { ...last, context: fresh };
    return { history: out, context: fresh, sources: sourcesOf(picked) };
  }

  /** Embed the chunks of documents that have no vectors yet, in the background; progress goes out as `docindex` events.
   *  The lock (`indexing`) is set synchronously, so nothing can start between "document added" and "indexing running". */
  function startIndexing(docIds) {
    if (indexing || current || !embeddings.available() || docIds.length === 0) return false;
    const abort = { aborted: false }; indexing = { abort, docId: docIds[0], filename: docs.preview(docIds[0])?.filename ?? '', done: 0, total: 0 };
    emit('docindex', { docId: indexing.docId, filename: indexing.filename, done: 0, total: 0, state: 'running' });
    (async () => {
      try {
        for (const docId of docIds) {
          if (abort.aborted) break;
          const texts = docs.chunkTexts(docId); if (!texts?.length) continue;
          const t0 = performance.now();
          Object.assign(indexing, { docId, filename: docs.preview(docId)?.filename ?? '', done: 0, total: texts.length });
          emit('docindex', { docId, filename: indexing.filename, done: 0, total: texts.length, state: 'running' });
          const vecs = await embeddings.embedTexts(texts, { signal: abort, threads: Math.min(getConfig().n_threads ?? 4, 8), onProgress: (done, total) => { indexing.done = done; emit('docindex', { docId, filename: indexing.filename, done, total, state: 'running' }); } });
          if (vecs && docs.setEmbeddings(docId, vecs)) log.info(`document indexed for search: ${texts.length} chunks in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
        }
      } catch (e) { log.error(`indexing failed: ${e.stack ?? e}`); emit('docindex', { state: 'error', message: String(e.message ?? e) }); }
      finally {
        indexing = null; emit('docindex', { state: 'idle' });
        if (pendingIndex.size && !current) { const ids = [...pendingIndex]; pendingIndex.clear(); startIndexing(ids); } // documents added meanwhile
      }
    })();
    return true;
  }
  afterDownload = (entry) => { if (entry.kind === 'embed') startIndexing(docs.needingEmbeddings()); };

  /** System-prompt additions after the base prompt: command instructions, then the rolling summary (the language reminder stays last). */
  const extraText = (history, summary) => [extraFor(history), summary ? SUMMARY_PREFIX + summary : ''].filter(Boolean).join('\n\n');

  function memoryMessage(e) {
    return `Ikke nok ledig minne til å laste denne modellen. Den trenger omtrent ${(e.needed / 1e9).toFixed(1)} GB (modell + kontekst), men bare ${(e.free / 1e9).toFixed(1)} GB er ledig. Lukk andre programmer, velg en mindre modell, eller senk kontekstvinduet i Innstillinger.`;
  }

  async function run(model, history, { continuation, chatId, think = false }) {
    const id = ++runSeq;
    const abort = { aborted: false };
    current = { id, abort };
    const send = (type, data = {}) => emit(type, { runId: id, ...data });
    try {
      const resident = host.loadedName === model;
      send('phase', { phase: resident ? 'preparing' : 'loading', model });
      const engine = await host.ensure(model);
      if (abort.aborted) { send('done', { stopReason: 'aborted', stopped: true, generated: 0 }); return; }
      send('phase', { phase: 'preparing', model });

      const ctxMax = engine.ctx;
      // Qwen3 can reason before it answers (the page's "Tenk først"); other models ignore the switch
      const thinking = !!think && engine.cfg.arch === 'qwen3';
      const reserve = thinking ? Math.min(THINK_MAX_TOKENS, Math.floor(ctxMax / 2)) : DEFAULT_MAX_TOKENS;
      const count = (m) => engine.countPromptTokens(m, { think: thinking });
      const withCont = (h) => (continuation ? [...h, { role: 'user', content: CONTINUATION_PROMPT }] : h);
      let { summary, summarized } = summaryState(chatId, history.length);
      const lc = await withLastContext(chatId, history, (t) => engine.countTokens(t), ctxMax, { retrieve: !continuation, vector: true });
      const build = () => buildMessages(withCont(lc.history.slice(summarized)), { extraSystem: extraText(history, summary) });

      // Long conversation: fold the older messages into the rolling summary (cached on disk), keep the newest ones verbatim.
      const budget = ctxMax - reserve - 64;
      const keepFrom = history.length - RECENT_KEEP;
      if (chatId && keepFrom > summarized && count(build()) > SUMMARIZE_AT * budget) {
        send('phase', { phase: 'summarizing', model });
        try {
          const folded = await summarizer.fold(summary, history.slice(summarized, keepFrom), { chatFilename: model, signal: abort });
          if (folded && !abort.aborted) { summary = folded; summarized = keepFrom; chatStore.saveSummary(chatId, summary, summarized); log.info(`summary updated: ${keepFrom} older messages folded in`); }
        } catch (e) { log.error(`summarizing failed, continuing without: ${e.stack ?? e}`); }
        if (abort.aborted) { send('done', { stopReason: 'aborted', stopped: true, generated: 0, model }); return; }
        send('phase', { phase: 'preparing', model });
      }
      const fitted = fitToContext(build(), count, ctxMax, reserve);
      const contextUsed = count(fitted.messages);
      if (process.env.ZEROCHAT_DEBUG_PROMPT === '1') log.info(`PROMPT (${contextUsed} tokens):\n${engine.tok.decode(engine.promptIds(fitted.messages, { think: thinking }), { skipControl: false })}`);
      log.info(`generating: ${model} | ${contextUsed}/${ctxMax} ctx tokens${fitted.trimmed ? ' | prompt trimmed to fit' : ''}`);

      // A freshly loaded model has read nothing yet: restore the system prompt's state from disk (or read it once and save it).
      let prefixCache = 'off';
      if (engine.cachedTokens === 0) {
        const prefix = stablePrefixIds(engine);
        const full = engine.promptIds(fitted.messages);
        if (prefix.length <= full.length && prefix.every((t, i) => t === full[i])) {
          prefixCache = await engine.primePrefix(prefix, { dir: paths.cacheDir, signal: abort, onProgress: (done, total) => send('progress', { done, total }) });
          if (prefixCache === 'hit' || prefixCache === 'saved') log.info(`system prompt cache: ${prefixCache} (${prefix.length} tokens)`);
        }
      }

      const stats = {};
      let answer = ''; let checkedAt = 0; let looped = false; let first = true;
      const gen = engine.generate({
        messages: fitted.messages, think: thinking, maxTokens: thinking ? Math.max(256, Math.min(reserve, ctxMax - contextUsed - 8)) : DEFAULT_MAX_TOKENS,
        sampling: thinking ? THINK_SAMPLING : SAMPLING, signal: abort, stats, penalizePrompt: false,
        onPrefill: (done, total) => send('progress', { done, total }),
      });
      for await (const text of gen) {
        if (first) { first = false; send('phase', { phase: 'generating', model }); }
        answer += text;
        send('chunk', { text });
        if (answer.length - checkedAt >= 120) {
          checkedAt = answer.length;
          if (looksLooped(answer)) { looped = true; abort.aborted = true; }
        }
      }
      const tokPerSec = stats.decodeMs > 0 ? (stats.generated * 1000) / stats.decodeMs : 0;
      log.info(`done: ${stats.generated} tokens, prefill ${(stats.prefillMs / 1000).toFixed(1)}s (${stats.cachedTokens} cached of ${stats.promptTokens}), decode ${tokPerSec.toFixed(1)} tok/s - ${looped ? 'looped' : stats.stopReason}`);
      send('done', {
        stopReason: looped ? 'looped' : stats.stopReason, truncated: stats.stopReason === 'length', stopped: abort.aborted && !looped, looped,
        thinking, trimmed: fitted.trimmed, compressed: summarized > 0, sources: lc.sources, context: lc.context, prefixCache, generated: stats.generated, promptTokens: stats.promptTokens, cachedTokens: stats.cachedTokens,
        prefillMs: stats.prefillMs, decodeMs: stats.decodeMs, tokPerSec, contextUsed, contextMax: ctxMax, model,
      });
    } catch (e) {
      log.error(`generation failed: ${e.stack ?? e}`);
      const message = e instanceof EngineMemoryError ? memoryMessage(e)
        : e instanceof UnsupportedModelError ? `Denne modellen støttes ikke av motoren (${e.message}). Velg en annen modell.`
          : /model file not found/.test(String(e.message)) ? 'Fant ikke modellfilen. Den kan ha blitt flyttet eller slettet.'
            : /prompt is \d+ tokens but the context window/.test(String(e.message)) ? 'Meldingen er for lang for kontekstvinduet. Skriv kortere, eller øk kontekstvinduet i Innstillinger.'
              : String(e.message ?? e);
      send('error', { message, code: e.code ?? null });
    } finally {
      if (current?.id === id) current = null;
      if (pendingIndex.size && !current) { const ids = [...pendingIndex]; pendingIndex.clear(); startIndexing(ids); }
    }
  }

  const methods = {
    async ping() { return { ok: true }; },

    /** Everything the page needs at start-up. */
    async state() {
      const cfg = getConfig();
      const models = listModels(paths, cfg);
      return {
        language: cfg.language, models, lastModel: cfg.last_model, loaded: host.loadedName,
        settings: { docChunks: cfg.doc_chunks ?? 4, utilityModel: cfg.utility_model, contextWindow: cfg.context_window, threads: cfg.n_threads, contextSteps: CONTEXT_STEPS, maxThreads: os.availableParallelism?.() ?? os.cpus().length, defaultThreads: defaultThreads() },
        system: { freeRam: os.freemem(), totalRam: os.totalmem() },
        busy: current !== null, indexing: indexing ? { docId: indexing.docId, filename: indexing.filename, done: indexing.done, total: indexing.total } : null, embedding: embeddings.modelFile(),
      };
    },

    async i18n({ lang }) {
      const name = lang === 'en' ? 'en' : 'nb';
      return JSON.parse(fs.readFileSync(path.join(I18N_DIR, `${name}.json`), 'utf8'));
    },

    /** Downloadable models, with what is already installed / partly downloaded and whether they fit in this PC's RAM. */
    async catalog() {
      const total = os.totalmem();
      const have = new Set(listModels(paths, getConfig()).map((m) => m.filename));
      return {
        entries: downloader.entries().map((e) => ({
          id: e.id, filename: e.filename, displayName: e.displayName, sizeBytes: e.sizeBytes, ramGB: e.ramGB, recommended: !!e.recommended, languages: e.languages,
          installed: have.has(e.filename), partialBytes: downloader.partial(e), fits: e.ramGB * 1e9 <= total * 0.8,
        })),
        status: downloader.status(), freeDisk: freeDiskBytes(paths.modelsDir), totalRam: total,
      };
    },
    async download_start({ ids }) { if (!Array.isArray(ids)) throw new Error('ids must be a list'); return downloader.start(ids); },
    async download_cancel() { return downloader.cancel(); },
    async download_status() { return downloader.status(); },

    async delete_model({ filename }) {
      assertNotIndexing();
      const full = modelPath(paths, filename);
      if (host.loadedName === filename) await host.unload();
      await embeddings.closeResident();
      fs.unlinkSync(full);
      const cfg = getConfig();
      if (cfg.models?.[filename]) { delete cfg.models[filename]; }
      if (cfg.last_model === filename) cfg.last_model = null;
      setConfig(cfg);
      return { ok: true };
    },

    /** Everything worth pasting into a bug report. */
    async diagnostics() {
      const cpus = os.cpus(); const wasm = probeWasmFeatures(); const js = probeJsFeatures(); const cfg = getConfig();
      let logTail = ''; try { logTail = fs.readFileSync(paths.logPath, 'utf8').split('\n').slice(-80).join('\n'); } catch { /* no log */ }
      const problems = [];
      if (!wasm.simd) problems.push('WebAssembly SIMD finnes ikke: motoren kan ikke bruke de raske kjernene.');
      if (!js.sharedArrayBuffer || !js.atomicsWait) problems.push('SharedArrayBuffer/Atomics.wait mangler: trådene kan ikke brukes.');
      return {
        system: { node: process.version, v8: process.versions.v8, platform: `${process.platform} ${process.arch} (${os.release()})`, cpu: cpus[0]?.model?.trim() ?? 'unknown', logicalCpus: cpus.length, ramTotal: os.totalmem(), ramFree: os.freemem(), heapLimit: v8.getHeapStatistics().heap_size_limit },
        features: { simd: wasm.simd, relaxedSimd: wasm.relaxedSimd, threads: wasm.threads, sharedArrayBuffer: js.sharedArrayBuffer, atomicsWait: js.atomicsWait },
        app: { dataFolder: paths.root, logFile: paths.logPath, contextWindow: cfg.context_window, threads: cfg.n_threads, loadedModel: host.loadedName },
        models: listModels(paths, cfg).map((m) => ({ filename: m.filename, sizeBytes: m.sizeBytes, supported: m.supported, reason: m.reason, quant: m.quant, arch: m.arch, ramNeed: m.ramNeed })),
        embedding: embeddings.modelFile(),
        problems, logTail,
      };
    },

    async list_models() { return listModels(paths, getConfig()); },

    async set_language({ lang }) { const cfg = getConfig(); cfg.language = lang === 'en' ? 'en' : 'nb'; setConfig(cfg); return { ok: true, language: cfg.language }; },

    async set_last_model({ filename }) { const cfg = getConfig(); cfg.last_model = filename; setConfig(cfg); return { ok: true }; },

    async list_commands() { return skills.commands(); },
    async list_skills() { return skills.list(); },
    async add_skill({ name, content }) { skills.add(name, content); return skills.list(); },
    async delete_skill({ id }) { skills.delete(id); return skills.list(); },

    // ---- documents
    async list_documents({ chatId }) { return docs.forChat(chatId); },
    async available_documents({ chatId }) { return docs.available(chatId); },
    async document_library() { return docs.library(); },
    async add_document({ chatId, filename, content }) {
      const r = docs.add(chatId, filename, content);
      log.info(`document added: ${filename} (${r.chunks} chunks)`);
      const started = startIndexing([r.id]);
      if (!started && embeddings.available()) pendingIndex.add(r.id); // while a reply is written or another document is indexed
      return { ...r, indexing: started, deferred: !started && embeddings.available(), documents: docs.forChat(chatId) };
    },
    async attach_document({ chatId, docId }) { docs.attach(chatId, docId); return docs.forChat(chatId); },
    async detach_document({ chatId, docId }) { docs.detach(chatId, docId); return docs.forChat(chatId); },
    async toggle_document({ chatId, docId }) { return { active: docs.toggle(chatId, docId) }; },
    async delete_document({ docId }) { docs.remove(docId); return { ok: true }; },
    async embedding_status() { return { file: embeddings.modelFile(), indexing: indexing !== null, missing: docs.needingEmbeddings().length }; },
    /** Embed every document that has no vectors yet (after the search model was installed). */
    async index_documents() { return { started: startIndexing(docs.needingEmbeddings()) }; },
    async preview_document({ docId }) { return docs.preview(docId); },
    async set_doc_chunks({ value }) {
      const v = Number(value); if (!Number.isInteger(v) || v < 1 || v > 10) throw new Error('invalid number of excerpts');
      const cfg = getConfig(); cfg.doc_chunks = v; setConfig(cfg); return { ok: true };
    },
    /** Long job (minutes): summary of a whole document, with the small model when there is one. */
    async summarize_document({ docId, model }) {
      assertNotIndexing();
      if (current) throw new Error('busy');
      const parts = docs.textParts(docId);
      if (!parts) throw new Error('document not found');
      const id = ++runSeq; const abort = { aborted: false }; current = { id, abort, kind: 'summary' };
      try {
        if (!summarizer.pick(model) && model) await host.ensure(model);
        const text = await summarizer.summarizeParts(parts, { chatFilename: model, signal: abort, onProgress: (done, total) => emit('docprogress', { docId, done, total }) });
        if (text && !abort.aborted) docs.setSummary(docId, text);
        return { summary: abort.aborted ? '' : text, aborted: abort.aborted };
      } finally { current = null; }
    },

    async list_chats() { return chatStore.list(); },
    async load_chat({ id }) { return chatStore.load(id); },
    async save_chat(p) { return { id: chatStore.save(p) }; },
    async rename_chat({ id, name }) { return { ok: chatStore.rename(id, name) }; },
    async delete_chat({ id }) { return { ok: chatStore.delete(id) }; },

    async send_message({ model, messages, chatId, think }) {
      modelPath(paths, model);
      assertNotIndexing();
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: false, chatId: chatId ?? null, think: think === true });
      await started;
      return { started: true, runId: runSeq };
    },

    async continue_message({ model, messages, chatId, think }) {
      modelPath(paths, model);
      assertNotIndexing();
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: true, chatId: chatId ?? null, think: think === true });
      await started;
      return { started: true, runId: runSeq };
    },

    async stop_generation() { if (current) current.abort.aborted = true; return { stopping: current !== null }; },

    /** Exact prompt size for the context meter (works without loading the model's weights). */
    async estimate_context({ model, messages, chatId, think }) {
      const tools = toolsFor(model);
      const { summary, summarized } = summaryState(chatId ?? null, messages.length);
      const lc = await withLastContext(chatId ?? null, messages, (t) => tools.countTokens(t), Math.min(contextMax(), tools.cfg.nCtxTrain));
      const fitted = buildMessages(lc.history.slice(summarized), { extraSystem: extraText(messages, summary) });
      return { contextUsed: tools.countPromptTokens(fitted, { think: think === true && tools.cfg.arch === 'qwen3' }), contextMax: Math.min(contextMax(), tools.cfg.nCtxTrain), compressed: summarized > 0 };
    },

    async set_context_window({ value }) {
      assertNotIndexing();
      const v = Number(value);
      if (!Number.isInteger(v) || v < 512 || v > 131072) throw new Error('invalid context size');
      const cfg = getConfig(); cfg.context_window = v; setConfig(cfg);
      await host.unload();
      return { ok: true, contextWindow: v };
    },

    async set_utility_model({ filename }) {
      const cfg = getConfig();
      if (filename) modelPath(paths, filename);
      cfg.utility_model = filename || null; setConfig(cfg);
      return { ok: true };
    },

    async set_thread_count({ value }) {
      assertNotIndexing();
      const v = Number(value);
      if (!Number.isInteger(v) || v < 1 || v > 64) throw new Error('invalid thread count');
      const cfg = getConfig(); cfg.n_threads = v; setConfig(cfg);
      await host.unload();
      return { ok: true, threads: v };
    },

    async update_model_settings({ filename, display_name, active, kv }) {
      modelPath(paths, filename);
      if (kv !== undefined) assertNotIndexing();
      const cfg = getConfig();
      const m = { ...(cfg.models[filename] ?? {}) };
      if (kv !== undefined) {
        if (kv !== 'f16' && kv !== 'f32') throw new Error('invalid kv precision');
        if (m.kv !== kv && host.loadedName === filename) await host.unload();
        m.kv = kv;
      }
      if (typeof display_name === 'string') m.display_name = display_name.trim().slice(0, 80);
      if (typeof active === 'boolean') m.active = active;
      cfg.models[filename] = m; setConfig(cfg);
      return { ok: true };
    },

    async unload_model() { assertNotIndexing(); await host.unload(); return { ok: true }; },

    async reset_settings() {
      assertNotIndexing();
      await host.unload();
      const keep = getConfig();
      setConfig({ ...defaultConfig(), last_model: keep.last_model });
      return { ok: true };
    },
  };

  return { methods, async shutdown() { downloader.cancel(); if (indexing) indexing.abort.aborted = true; await embeddings.closeResident(); if (current) current.abort.aborted = true; await host.unload(); } };
}
