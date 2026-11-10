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
import { createDownloader, freeDiskBytes } from './downloader.js';
import { CATALOG } from './catalog.js';
import { createSummarizer } from './summarizer.js';
import { createTitler } from './titles.js';
import { buildMessages, fitToContext, looksLooped, stablePrefixIds, renderWithContext, RECENT_KEEP, SUMMARIZE_AT, SUMMARY_PREFIX, SAMPLING, THINK_SAMPLING, THINK_MAX_TOKENS, DEFAULT_MAX_TOKENS, CONTINUATION_PROMPT } from './chat-core.js';

const I18N_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'i18n');

export function createApi({ paths, host, getConfig, setConfig, log, emit, catalog = CATALOG, retryDelayMs, titleRun }) {
  let runSeq = 0;
  let current = null; // {id, abort: {aborted}}
  const chatStore = createChatStore(paths);
  const skills = createSkills(paths);
  const summarizer = createSummarizer({ paths, host, getConfig, log });
  // Automatic chat titles (background job after the first exchange; see app/titles.js). The messages it builds for the chat model are the chat's own.
  const titler = createTitler({
    chatStore, summarizer, host, getConfig, log, emit, runActive: () => current !== null, run: titleRun,
    chatPrompt: (history) => buildMessages(history, { extraSystem: extraText(history, '') }),
  });
  /** Unload the chat model - after a running title job (which may be using it) has been stopped. */
  const unloadHost = async ({ all = false } = {}) => { await titler.stop({ all }); await host.unload(); };
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
  /** Largest attached document (in tokens): 80 % of what is left of the window after the room reserved for the reply. */
  const replyReserve = (ctxMax, thinking) => (thinking ? Math.min(THINK_MAX_TOKENS, Math.floor(ctxMax / 2)) : DEFAULT_MAX_TOKENS);
  const docLimit = (ctxMax, thinking) => Math.floor(0.8 * (ctxMax - replyReserve(ctxMax, thinking) - 64));

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

  async function run(model, history, { continuation, chatId, think = false }) {
    const id = ++runSeq;
    const abort = { aborted: false };
    current = { id, abort };
    const send = (type, data = {}) => emit(type, { runId: id, ...data });
    try {
      await titler.stop(); // a title job in the background gives way at once (the engine serves one thing at a time)
      const resident = host.loadedName === model;
      send('phase', { phase: resident ? 'preparing' : 'loading', model });
      const engine = await host.ensure(model);
      if (abort.aborted) { send('done', { stopReason: 'aborted', stopped: true, generated: 0 }); return; }
      send('phase', { phase: 'preparing', model });

      const ctxMax = engine.ctx;
      // Qwen3 / Qwen3.5 can reason before they answer (the page's "Tenk først"); other models ignore the switch
      const thinking = !!think && engine.thinkingMode !== null;
      const reserve = replyReserve(ctxMax, thinking);
      const count = (m) => engine.countPromptTokens(m, { think: thinking });
      const withCont = (h) => (continuation ? [...h, { role: 'user', content: CONTINUATION_PROMPT }] : h);
      let { summary, summarized } = summaryState(chatId, history.length);
      const build = () => buildMessages(withCont(history.slice(summarized)), { extraSystem: extraText(history, summary) });
      const budget = ctxMax - reserve - 64;
      // An attached document goes into the message whole: it has to fit in the context window together with the reply.
      const lastUser = [...history].reverse().find((m) => m.role === 'user');
      if (lastUser?.context?.length) {
        const docTokens = engine.countTokens(renderWithContext(lastUser.content, lastUser.context));
        if (docTokens > docLimit(ctxMax, thinking)) throw Object.assign(new Error('document too big'), { code: 'DOC_TOO_BIG', docTokens, budget: docLimit(ctxMax, thinking) });
      }

      // Long conversation: fold the older messages into the rolling summary (cached on disk), keep the newest ones verbatim.
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
        if (first) {
          first = false; send('phase', { phase: 'generating', model });
          // Qwen3.5's thinking prompt already ends with the opening <think>: put it in front so the stream reads <think>..</think>answer like Qwen3's
          const opening = engine.thinkOpening(thinking);
          if (opening) { answer += opening; send('chunk', { text: opening }); }
        }
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
        thinking, trimmed: fitted.trimmed, compressed: summarized > 0, prefixCache, generated: stats.generated, promptTokens: stats.promptTokens, cachedTokens: stats.cachedTokens,
        prefillMs: stats.prefillMs, decodeMs: stats.decodeMs, tokPerSec, contextUsed, contextMax: ctxMax, model,
      });
    } catch (e) {
      log.error(`generation failed: ${e.stack ?? e}`);
      const message = e.code === 'DOC_TOO_BIG' ? `Dokumentet er for stort for kontekstvinduet (omtrent ${e.docTokens.toLocaleString('nb-NO')} tokens, maks ${e.budget.toLocaleString('nb-NO')}). Bruk et kortere utdrag, eller øk kontekstvinduet i Innstillinger.`
        : e instanceof EngineMemoryError ? memoryMessage(e)
        : e instanceof UnsupportedModelError ? `Denne modellen støttes ikke av motoren (${e.message}). Velg en annen modell.`
          : /model file not found/.test(String(e.message)) ? 'Fant ikke modellfilen. Den kan ha blitt flyttet eller slettet.'
            : /prompt is \d+ tokens but the context window/.test(String(e.message)) ? 'Meldingen er for lang for kontekstvinduet. Skriv kortere, eller øk kontekstvinduet i Innstillinger.'
              : String(e.message ?? e);
      send('error', { message, code: e.code ?? null });
    } finally {
      if (current?.id === id) current = null;
    }
  }

  /** Load the last used model in the background at start-up, so the first message does not wait for "Laster modellen...". */
  async function preload() {
    const cfg = getConfig();
    if (cfg.preload_model === false || current) return;
    const usable = listModels(paths, cfg).filter((m) => m.supported);
    const m = usable.find((x) => x.filename === cfg.last_model) ?? (usable.length === 1 ? usable[0] : null);
    if (!m) return;
    emit('preload', { state: 'loading', model: m.filename });
    try { await host.ensure(m.filename); log.info(`preloaded ${m.filename}`); emit('preload', { state: 'ready', model: m.filename }); }
    catch (e) { log.error(`preload failed: ${e.message}`); emit('preload', { state: 'error', model: m.filename }); }
  }

  const methods = {
    async ping() { return { ok: true }; },

    /** Everything the page needs at start-up. */
    async state() {
      const cfg = getConfig();
      const models = listModels(paths, cfg);
      return {
        language: cfg.language, models, lastModel: cfg.last_model, loaded: host.loadedName,
        settings: { preload: cfg.preload_model !== false, autoTitles: cfg.auto_titles !== false, utilityModel: cfg.utility_model, contextWindow: cfg.context_window, threads: cfg.n_threads, contextSteps: CONTEXT_STEPS, maxThreads: os.availableParallelism?.() ?? os.cpus().length, defaultThreads: defaultThreads() },
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
      if (host.loadedName === filename) await unloadHost();
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

    async list_chats() { return chatStore.list(); },
    async load_chat({ id }) { return chatStore.load(id); },
    async save_chat(p) { const r = chatStore.save(p); titler.maybeStart(r.id); return r; },
    async rename_chat({ id, name }) { return { ok: chatStore.rename(id, name) }; },
    async delete_chat({ id }) { return { ok: chatStore.delete(id) }; },

    async send_message({ model, messages, chatId, think }) {
      modelPath(paths, model);
      if (current) throw new Error('busy');
      const started = new Promise((resolve) => setImmediate(resolve));
      run(model, messages, { continuation: false, chatId: chatId ?? null, think: think === true });
      await started;
      return { started: true, runId: runSeq };
    },

    async continue_message({ model, messages, chatId, think }) {
      modelPath(paths, model);
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
      const fitted = buildMessages(messages.slice(summarized), { extraSystem: extraText(messages, summary) });
      const thinking = think === true && tools.thinkingMode !== null;
      const max = Math.min(contextMax(), tools.cfg.nCtxTrain);
      const lastUser = [...messages].reverse().find((m) => m.role === 'user');
      return {
        contextUsed: tools.countPromptTokens(fitted, { think: thinking }), contextMax: max, compressed: summarized > 0,
        // an attached document must fit whole: how many tokens it takes and how many it may take (same rule as when the reply is made)
        docTokens: lastUser?.context?.length ? tools.countTokens(renderWithContext(lastUser.content, lastUser.context)) : 0, docLimit: docLimit(max, thinking),
      };
    },

    async set_context_window({ value }) {
      const v = Number(value);
      if (!Number.isInteger(v) || v < 512 || v > 131072) throw new Error('invalid context size');
      const cfg = getConfig(); cfg.context_window = v; setConfig(cfg);
      await unloadHost();
      return { ok: true, contextWindow: v };
    },

    async set_preload({ value }) { const cfg = getConfig(); cfg.preload_model = value === true; setConfig(cfg); return { ok: true, preload: cfg.preload_model }; },

    async set_auto_titles({ value }) {
      const cfg = getConfig(); cfg.auto_titles = value === true; setConfig(cfg);
      if (!cfg.auto_titles) await titler.stop();
      return { ok: true, autoTitles: cfg.auto_titles };
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
      await unloadHost();
      return { ok: true, threads: v };
    },

    async update_model_settings({ filename, display_name, active, kv }) {
      modelPath(paths, filename);
      const cfg = getConfig();
      const m = { ...(cfg.models[filename] ?? {}) };
      if (kv !== undefined) {
        if (kv !== 'f16' && kv !== 'f32') throw new Error('invalid kv precision');
        if (m.kv !== kv && host.loadedName === filename) await unloadHost();
        m.kv = kv;
      }
      if (typeof display_name === 'string') m.display_name = display_name.trim().slice(0, 80);
      if (typeof active === 'boolean') m.active = active;
      cfg.models[filename] = m; setConfig(cfg);
      return { ok: true };
    },

    async unload_model() { await unloadHost(); return { ok: true }; },

    async reset_settings() {
      await unloadHost();
      const keep = getConfig();
      setConfig({ ...defaultConfig(), last_model: keep.last_model });
      return { ok: true };
    },
  };

  return { methods, preload, async shutdown() { downloader.cancel(); if (current) current.abort.aborted = true; await unloadHost({ all: true }); } };
}
