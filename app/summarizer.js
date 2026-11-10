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

  /** Run `fn(engine)` with the small model (loaded for the job and closed after), or the chat model when none is usable. */
  async function withEngine(chatFilename, fn) {
    const cfg = getConfig();
    const utility = pickUtilityModel(listModels(paths, cfg), chatFilename, cfg.utility_model);
    let engine = null; let own = false;
    if (utility) {
      try {
        engine = await loadModel(modelPath(paths, utility), { ctx: UTILITY_CTX, threads: cfg.n_threads, kv: 'f16' });
        own = true;
        log.info(`summarizing with ${utility}`);
      } catch (e) {
        log.info(`utility model ${utility} not used (${e instanceof EngineMemoryError ? 'not enough free RAM' : e.message}); summarizing with the chat model`);
      }
    }
    if (!engine) engine = host.engine;
    if (!engine) throw new Error('no model available for summarizing');
    try { return await fn(engine); } finally { if (own) await engine.close(); }
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

  return { fold, pick: (chatFilename) => pickUtilityModel(listModels(paths, getConfig()), chatFilename, getConfig().utility_model) };
}
