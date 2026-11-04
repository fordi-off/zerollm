// The API the page talks to. Every method takes one params object and returns JSON; streamed output goes out as events
// ({runId, ...}) over the SSE channel: phase, progress, chunk, done, error.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadTokenizer, EngineMemoryError, defaultThreads } from '../src/index.js';
import { defaultConfig, CONTEXT_STEPS } from './config.js';
import { listModels, modelPath } from './models.js';
import { createChatStore } from './chats.js';
import { createSkills } from './skills.js';
import { createDownloader, freeDiskBytes } from './downloader.js';
import { CATALOG } from './catalog.js';
import { createSummarizer } from './summarizer.js';
import { buildMessages, fitToContext, looksLooped, stablePrefixIds, RECENT_KEEP, SUMMARIZE_AT, SUMMARY_PREFIX, SAMPLING, DEFAULT_MAX_TOKENS, CONTINUATION_PROMPT } from './chat-core.js';

const I18N_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'i18n');

export function createApi({ paths, host, getConfig, setConfig, log, emit, catalog = CATALOG, retryDelayMs }) {
  let runSeq = 0;
  let current = null; // {id, abort: {aborted}}
  const chatStore = createChatStore(paths);
  const skills = createSkills(paths);
  const summarizer = createSummarizer({ paths, host, getConfig, log });
  skills.seed();
  /** Instruction text for the slash commands on the last user message (they apply to that one reply). */
  const extraFor = (history) => skills.bodies([...history].reverse().find((m) => m.role === 'user')?.commands) ?? '';
  const downloader = createDownloader({ paths, log, emit, catalog, retryDelayMs });
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
  /** System-prompt additions after the base prompt: command instructions, then the rolling summary (the language reminder stays last). */
  const extraText = (history, summary) => [extraFor(history), summary ? SUMMARY_PREFIX + summary : ''].filter(Boolean).join('\n\n');

  function memoryMessage(e) {
    return `Ikke nok ledig minne til å laste denne modellen. Den trenger omtrent ${(e.needed / 1e9).toFixed(1)} GB (modell + kontekst), men bare ${(e.free / 1e9).toFixed(1)} GB er ledig. Lukk andre programmer, velg en mindre modell, eller senk kontekstvinduet i Innstillinger.`;
  }

  async function run(model, history, { continuation, chatId }) {
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
      const count = (m) => engine.countPromptTokens(m);
      const withCont = (h) => (continuation ? [...h, { role: 'user', content: CONTINUATION_PROMPT }] : h);
      let { summary, summarized } = summaryState(chatId, history.length);
      const build = () => buildMessages(withCont(history.slice(summarized)), { extraSystem: extraText(history, summary) });

      // Long conversation: fold the older messages into the rolling summary (cached on disk), keep the newest ones verbatim.
      const budget = ctxMax - DEFAULT_MAX_TOKENS - 64;
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
      const fitted = fitToContext(build(), count, ctxMax, DEFAULT_MAX_TOKENS);
      const contextUsed = engine.countPromptTokens(fitted.messages);
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
        messages: fitted.messages, maxTokens: DEFAULT_MAX_TOKENS, sampling: SAMPLING, signal: abort, stats,
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
        trimmed: fitted.trimmed, compressed: summarized > 0, prefixCache, generated: stats.generated, promptTokens: stats.promptTokens, cachedTokens: stats.cachedTokens,
        prefillMs: stats.prefillMs, decodeMs: stats.decodeMs, tokPerSec, contextUsed, contextMax: ctxMax, model,
      });
    } catch (e) {
      log.error(`generation failed: ${e.stack ?? e}`);
      const message = e instanceof EngineMemoryError ? memoryMessage(e) : String(e.message ?? e);
      send('error', { message, code: e.code ?? null });
    } finally {
      if (current?.id === id) current = null;
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
        settings: { utilityModel: cfg.utility_model, contextWindow: cfg.context_window, threads: cfg.n_threads, contextSteps: CONTEXT_STEPS, maxThreads: os.availableParallelism?.() ?? os.cpus().length, defaultThreads: defaultThreads() },
        system: { freeRam: os.freemem(), totalRam: os.totalmem() },
        busy: current !== null,
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
      const full = modelPath(paths, filename);
      if (host.loadedName === filename) await host.unload();
      fs.unlinkSync(full);
      const cfg = getConfig();
      if (cfg.models?.[filename]) { delete cfg.models[filename]; }
      if (cfg.last_model === filename) cfg.last_model = null;
      setConfig(cfg);
      return { ok: true };
    },

    async list_models() { return listModels(paths, getConfig()); },

    async set_language({ lang }) { const cfg = getConfig(); cfg.language = lang === 'en' ? 'en' : 'nb'; setConfig(cfg); return { ok: true, language: cfg.language }; },

    async set_last_model({ filename }) { const cfg = getConfig(); cfg.last_model = filename; setConfig(cfg); return { ok: true }; },

    async list_commands() { return skills.commands(); },
    async list_skills() { return skills.list(); },
    async add_skill({ name, content }) { skills.add(name, content); return skills.list(); },
    async delete_skill({ id }) { skills.delete(id); return skills.list(); },

    async list_chats() { return chatStore.list(); },
    async load_chat({ id }) { return chatStore.load(id); },
    async save_chat(p) { return { id: chatStore.save(p) }; },
    async rename_chat({ id, name }) { return { ok: chatStore.rename(id, name) }; },
    async delete_chat({ id }) { return { ok: chatStore.delete(id) }; },

    async send_message({ model, messages, chatId }) {
      modelPath(paths, model);
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: false, chatId: chatId ?? null });
      await started;
      return { started: true, runId: runSeq };
    },

    async continue_message({ model, messages, chatId }) {
      modelPath(paths, model);
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: true, chatId: chatId ?? null });
      await started;
      return { started: true, runId: runSeq };
    },

    async stop_generation() { if (current) current.abort.aborted = true; return { stopping: current !== null }; },

    /** Exact prompt size for the context meter (works without loading the model's weights). */
    async estimate_context({ model, messages, chatId }) {
      const tools = toolsFor(model);
      const { summary, summarized } = summaryState(chatId ?? null, messages.length);
      const fitted = buildMessages(messages.slice(summarized), { extraSystem: extraText(messages, summary) });
      return { contextUsed: tools.countPromptTokens(fitted), contextMax: Math.min(contextMax(), tools.cfg.nCtxTrain), compressed: summarized > 0 };
    },

    async set_context_window({ value }) {
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
      const v = Number(value);
      if (!Number.isInteger(v) || v < 1 || v > 64) throw new Error('invalid thread count');
      const cfg = getConfig(); cfg.n_threads = v; setConfig(cfg);
      await host.unload();
      return { ok: true, threads: v };
    },

    async update_model_settings({ filename, display_name, active, kv }) {
      modelPath(paths, filename);
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

    async unload_model() { await host.unload(); return { ok: true }; },

    async reset_settings() {
      await host.unload();
      const keep = getConfig();
      setConfig({ ...defaultConfig(), last_model: keep.last_model });
      return { ok: true };
    },
  };

  return { methods, async shutdown() { downloader.cancel(); if (current) current.abort.aborted = true; await host.unload(); } };
}
