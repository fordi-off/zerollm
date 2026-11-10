// Folds old messages into the rolling summary with a small model: the "utility model" from settings, else the smallest installed
// supported model that is clearly smaller than the chat model, else the chat model itself (slow, but it works).
// A separate small model is loaded only for the summary and closed again; if it does not fit in RAM the chat model does the job.
import { loadModel, EngineMemoryError } from '../src/index.js';
import { listModels, modelPath } from './models.js';
import { summaryPrompt, chunkForSummary } from './chat-core.js';

const UTILITY_CTX = 4096;

export function pickUtilityModel(models, chatFilename, configured) {
  const usable = models.filter((m) => m.supported && m.filename !== chatFilename);
  const chosen = configured ? usable.find((m) => m.filename === configured) : null;
  if (chosen) return chosen.filename;
  const main = models.find((m) => m.filename === chatFilename);
  const small = usable.filter((m) => !main || m.sizeBytes < main.sizeBytes * 0.6).sort((a, b) => a.sizeBytes - b.sizeBytes);
  return small[0]?.filename ?? null;
}

export function createSummarizer({ paths, host, getConfig, log }) {
  async function complete(engine, messages, signal, maxTokens = 500) {
    let out = '';
    for await (const text of engine.generate({ messages, maxTokens, sampling: { temperature: 0.3, topP: 0.9 }, signal })) out += text;
    return out.trim();
  }

  /**
   * Run `fn(engine, {own})` with the small model (loaded for the job and closed after; `own` = true), or the chat model when none is
   * usable (`own` = false). With `signal`, an abort while the small model was loading ends the job before anything runs (returns null),
   * and `onLoading(true|false)` brackets the load so the caller knows when the chat engine is certainly not in use.
   */
  async function withEngine(chatFilename, fn, { signal, onLoading, label = 'summarizing' } = {}) {
    const cfg = getConfig();
    const utility = pickUtilityModel(listModels(paths, cfg), chatFilename, cfg.utility_model);
    let engine = null; let own = false;
    if (utility) {
      onLoading?.(true);
      try {
        engine = await loadModel(modelPath(paths, utility), { ctx: UTILITY_CTX, threads: cfg.n_threads, kv: 'f16' });
        own = true;
        log.info(`${label} with ${utility}`);
      } catch (e) {
        log.info(`utility model ${utility} not used (${e instanceof EngineMemoryError ? 'not enough free RAM' : e.message}); ${label} with the chat model`);
      } finally { onLoading?.(false); }
    }
    if (signal?.aborted) { if (own) await engine.close(); return null; }
    if (!engine) engine = host.engine;
    if (!engine) throw new Error('no model available for summarizing');
    try { return await fn(engine, { own }); } finally { if (own) await engine.close(); }
  }

  /**
   * @param {string} existing  summary so far   @param {{role: string, content: string}[]} chunk  messages to fold in
   * @returns {Promise<string>} the new summary ('' if aborted)
   */
  async function fold(existing, chunk, { chatFilename, signal }) {
    return withEngine(chatFilename, async (engine) => {
      let summary = existing;
      for (const pass of chunkForSummary(chunk, engine.ctx >= 8192 ? 12000 : 7000)) {
        if (signal?.aborted) return '';
        const next = await complete(engine, summaryPrompt(summary, pass), signal);
        if (signal?.aborted) return '';
        if (next) summary = next;
      }
      return summary;
    });
  }

  return { fold, withEngine, pick: (chatFilename) => pickUtilityModel(listModels(paths, getConfig()), chatFilename, getConfig().utility_model) };
}
