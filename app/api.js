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
import { buildMessages, fitToContext, looksLooped, stablePrefixIds, SAMPLING, DEFAULT_MAX_TOKENS, CONTINUATION_PROMPT } from './chat-core.js';

const I18N_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'i18n');

export function createApi({ paths, host, getConfig, setConfig, log, emit }) {
  let runSeq = 0;
  let current = null; // {id, abort: {aborted}}
  const chatStore = createChatStore(paths);
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

  function memoryMessage(e) {
    return `Ikke nok ledig minne til å laste denne modellen. Den trenger omtrent ${(e.needed / 1e9).toFixed(1)} GB (modell + kontekst), men bare ${(e.free / 1e9).toFixed(1)} GB er ledig. Lukk andre programmer, velg en mindre modell, eller senk kontekstvinduet i Innstillinger.`;
  }

  async function run(model, history, { continuation }) {
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

      const msgs = continuation ? [...history, { role: 'user', content: CONTINUATION_PROMPT }] : history;
      const ctxMax = engine.ctx;
      const fitted = fitToContext(buildMessages(msgs), (m) => engine.countPromptTokens(m), ctxMax, DEFAULT_MAX_TOKENS);
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
        trimmed: fitted.trimmed, prefixCache, generated: stats.generated, promptTokens: stats.promptTokens, cachedTokens: stats.cachedTokens,
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
        settings: { contextWindow: cfg.context_window, threads: cfg.n_threads, contextSteps: CONTEXT_STEPS, maxThreads: os.availableParallelism?.() ?? os.cpus().length, defaultThreads: defaultThreads() },
        system: { freeRam: os.freemem(), totalRam: os.totalmem() },
        busy: current !== null,
      };
    },

    async i18n({ lang }) {
      const name = lang === 'en' ? 'en' : 'nb';
      return JSON.parse(fs.readFileSync(path.join(I18N_DIR, `${name}.json`), 'utf8'));
    },

    async list_models() { return listModels(paths, getConfig()); },

    async set_language({ lang }) { const cfg = getConfig(); cfg.language = lang === 'en' ? 'en' : 'nb'; setConfig(cfg); return { ok: true, language: cfg.language }; },

    async set_last_model({ filename }) { const cfg = getConfig(); cfg.last_model = filename; setConfig(cfg); return { ok: true }; },

    async list_chats() { return chatStore.list(); },
    async load_chat({ id }) { return chatStore.load(id); },
    async save_chat(p) { return { id: chatStore.save(p) }; },
    async rename_chat({ id, name }) { return { ok: chatStore.rename(id, name) }; },
    async delete_chat({ id }) { return { ok: chatStore.delete(id) }; },

    async send_message({ model, messages }) {
      modelPath(paths, model);
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: false });
      await started;
      return { started: true, runId: runSeq };
    },

    async continue_message({ model, messages }) {
      modelPath(paths, model);
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: true });
      await started;
      return { started: true, runId: runSeq };
    },

    async stop_generation() { if (current) current.abort.aborted = true; return { stopping: current !== null }; },

    /** Exact prompt size for the context meter (works without loading the model's weights). */
    async estimate_context({ model, messages }) {
      const tools = toolsFor(model);
      const fitted = buildMessages(messages);
      return { contextUsed: tools.countPromptTokens(fitted), contextMax: Math.min(contextMax(), tools.cfg.nCtxTrain) };
    },

    async set_context_window({ value }) {
      const v = Number(value);
      if (!Number.isInteger(v) || v < 512 || v > 131072) throw new Error('invalid context size');
      const cfg = getConfig(); cfg.context_window = v; setConfig(cfg);
      await host.unload();
      return { ok: true, contextWindow: v };
    },

    async set_thread_count({ value }) {
      const v = Number(value);
      if (!Number.isInteger(v) || v < 1 || v > 64) throw new Error('invalid thread count');
      const cfg = getConfig(); cfg.n_threads = v; setConfig(cfg);
      await host.unload();
      return { ok: true, threads: v };
    },

    async update_model_settings({ filename, display_name, active }) {
      modelPath(paths, filename);
      const cfg = getConfig();
      const m = { ...(cfg.models[filename] ?? {}) };
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

  return { methods, async shutdown() { if (current) current.abort.aborted = true; await host.unload(); } };
}
