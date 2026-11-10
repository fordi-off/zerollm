// Automatic chat titles: after the first exchange of a chat a short title is written in the background and the sidebar entry
// is renamed. See docs/DECISIONS.md ("Automatic chat titles") for the design; the short version:
//  - state lives in the chat file: `auto_title` = 'pending' (name is the automatic placeholder) | 'done' (name was written by this job)
//    | 'failed' (the model produced nothing usable) | 'user' (the user chose the name); absent = an old / foreign chat, never titled;
//  - a separate small "utility" model is used when there is one (loaded for the job, closed after); otherwise the chat model
//    continues the REAL conversation with one extra question at the end, so its KV cache stays a valid prefix of the next prompt;
//  - the job never runs while a reply is being written and is aborted (and awaited) the moment anything else wants the engine;
//  - nothing here may block or fail the chat: errors are logged and swallowed. No word lists: sanitising is purely structural.

/** Title wanted from the small model: system message (English), user message = start of the first exchange. */
export const TITLE_SYSTEM =
  'Write a short title for this conversation, in the same language as the conversation. Use 3 to 6 words, no quotes and no punctuation at the end. Reply with the title only.';
/** Asked as one more user turn when the CHAT model does the job (its system prompt says "reply in the language of the latest message", hence the explicit language sentence). */
export const TITLE_FOLLOWUP =
  'Write a short title for the conversation above (3 to 6 words). Use the language the conversation is written in, not the language of this instruction. No quotes and no punctuation at the end. Reply with the title only.';
export const TITLE_SAMPLING = { temperature: 0.2, topP: 0.9, topK: 40, minP: 0.05, repeatPenalty: 1, frequencyPenalty: 0, presencePenalty: 0 };
export const TITLE_MAX_TOKENS = 24;
export const TITLE_MAX_CHARS = 60;
const USER_CAP = 600; const REPLY_CAP = 400;
const MAX_ATTEMPTS = 3; // per chat and app run (a job that is aborted by the next message is tried again after the next save)

const cap = (s, n) => { const a = Array.from(String(s ?? '').trim()); return a.length > n ? a.slice(0, n).join('').trimEnd() : a.join(''); };

/** Messages for the utility model. */
export function titleMessages(userText, replyText) {
  return [
    { role: 'system', content: TITLE_SYSTEM },
    { role: 'user', content: `User message:\n${cap(userText, USER_CAP)}\n\nStart of the reply:\n${cap(replyText, REPLY_CAP)}` },
  ];
}

// ---------- sanitising
const QUOTE_ANY = /["“”„‟«»‹›「」『』〝〞＂]/u; // quote marks that are never part of a word
const QUOTE_PAIR = /['‘’‚＇]/u; // apostrophes: only removed when they wrap the whole title
const WRAPS = { '[': ']', '(': ')', '{': '}', '<': '>', '【': '】', '（': '）' };
const TRAILING = /[\s.,;:!?…。，、；：！？·–—-]+$/u;

/** True when `s` starts with an opening bracket whose partner is its last character (and the brackets inside are balanced). */
function wrappedByBrackets(s) {
  const close = WRAPS[s[0]];
  if (!close || s.at(-1) !== close || s.length < 3) return false;
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === s[0]) depth++; else if (s[i] === close) depth--;
    if (depth === 0 && i < s.length - 1) return false; // closed early: "(a) b (c)"
  }
  return depth === 0;
}

function stripEdges(line) {
  let x = line;
  for (let i = 0; i < 12; i++) {
    const before = x;
    x = x.trim();
    x = x.replace(/^(?:#{1,6}\s*|>\s*|[-•·–—]\s+)/u, '').replace(/^[*_~`]+/u, '').replace(/[*_~`]+$/u, '');
    while (x && QUOTE_ANY.test(x[0])) x = x.slice(1).trimStart();
    while (x && QUOTE_ANY.test(x.at(-1))) x = x.slice(0, -1).trimEnd();
    if (x.length > 1 && QUOTE_PAIR.test(x[0]) && QUOTE_PAIR.test(x.at(-1))) x = x.slice(1, -1);
    if (wrappedByBrackets(x)) x = x.slice(1, -1);
    x = x.replace(TRAILING, '');
    if (x === before) break;
  }
  return x;
}

/**
 * Turn a model's raw answer into a title, or '' when it is not usable. Language-neutral on purpose: the first non-empty line
 * (a line that only introduces the next one, ending in a colon, is skipped), emphasis / quotes / brackets / hashes / trailing
 * punctuation trimmed, one short leading "Label:" dropped, whitespace collapsed, cut to `maxLen` at a word boundary; empty,
 * mostly non-printable or mostly non-letter results are rejected.
 * @param {string} raw @param {{maxLen?: number}} [o]
 */
export function cleanTitle(raw, { maxLen = TITLE_MAX_CHARS } = {}) {
  let s = String(raw ?? '').replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*$/i, '').replace(/<\/?think>/gi, '');
  s = s.replace(/\r\n?/g, '\n').replace(/\t/g, ' ');
  const total = Array.from(s).length;
  if (total === 0) return '';
  const bad = Array.from(s).filter((c) => c !== '\n' && /[\p{Cc}\p{Cf}\p{Co}\p{Cn}\p{Cs}�]/u.test(c)).length;
  if (bad / total > 0.25) return ''; // mostly garbage bytes
  const lines = s.split('\n').map((l) => l.replace(/[\p{Cc}\p{Cf}\p{Co}\p{Cn}\p{Cs}�]/gu, '').replace(/\*\*+|__+|~~+/g, '')).map((l) => l.replace(/\s+/g, ' ').trim());
  const usable = lines.filter((l) => stripEdges(l));
  let line = usable[0] ?? '';
  if (/[:：]\s*$/u.test(line) && usable.length > 1) line = usable[1]; // "Here is the title:" / newline / the title
  let x = stripEdges(line);
  x = x.replace(/^\p{L}[\p{L}\p{N}]{0,14}\s*[:：]\s*(?=\S)/u, ''); // one short leading label ("Title: ...", "Tittel: ...")
  x = stripEdges(x).replace(/\s+/g, ' ');
  const chars = Array.from(x);
  if (chars.length > maxLen) {
    let cut = chars.slice(0, maxLen).join('');
    const sp = cut.lastIndexOf(' ');
    if (sp >= maxLen * 0.4) cut = cut.slice(0, sp);
    x = stripEdges(cut);
  }
  const cs = Array.from(x); const nonSpace = cs.filter((c) => !/\s/u.test(c));
  const letters = nonSpace.filter((c) => /[\p{L}\p{N}]/u.test(c)).length;
  if (letters < 2 || letters / nonSpace.length < 0.6) return '';
  const freq = new Map(); for (const c of nonSpace) freq.set(c, (freq.get(c) ?? 0) + 1);
  if (nonSpace.length >= 8 && Math.max(...freq.values()) / nonSpace.length > 0.5) return ''; // "aaaaaaaa"
  return x;
}

// ---------- the job
/** Length of the common leading run of two token arrays. */
function commonLen(a, b) { let i = 0; const n = Math.min(a.length, b.length); while (i < n && a[i] === b[i]) i++; return i; }

/** The generation itself (tests replace it to get known answers: the synthetic test models only produce noise). */
async function defaultRun(engine, messages, abort, stats) {
  let out = '';
  for await (const text of engine.generate({ messages, think: false, maxTokens: TITLE_MAX_TOKENS, sampling: TITLE_SAMPLING, signal: abort, stats, penalizePrompt: false })) out += text;
  return out;
}

/**
 * @param {{chatStore: ReturnType<import('./chats.js').createChatStore>, summarizer: ReturnType<import('./summarizer.js').createSummarizer>,
 *   host: import('./engine-host.js').EngineHost, getConfig: () => any, log: {info: Function, error: Function}, emit: Function,
 *   runActive: () => boolean, chatPrompt: (history: any[]) => {role: string, content: string}[],
 *   run?: (engine: any, messages: any[], abort: {aborted: boolean}, stats: object) => Promise<string>}} o
 *   `chatPrompt(history)` = the messages the chat itself sends for that history (system prompt incl.), so the chat-model path can extend them.
 */
export function createTitler({ chatStore, summarizer, host, getConfig, log, emit, runActive, chatPrompt, run = defaultRun }) {
  let job = null; // {chatId, abort, phase: 'loading'|'running', usesChat, promise}
  const attempts = new Map();

  async function generate(c, abort, job) {
    const utility = summarizer.pick(c.model);
    const chatUsable = c.exchanges === 1 && host.engine && host.loadedName === c.model;
    if (!utility && !chatUsable) { log.info(`title for ${c.id} skipped: no utility model, and the chat model is not loaded / the chat has moved on`); return null; }
    job.usesChat = !utility; job.phase = utility ? 'loading' : 'running'; // (a utility model that does not fit in RAM turns this around inside withEngine; usesChat is corrected there)
    return summarizer.withEngine(c.model, async (engine, { own }) => {
      job.phase = 'running'; job.usesChat = !own;
      if (abort.aborted) return null;
      let messages;
      if (own) messages = titleMessages(c.userText, c.replyText);
      else {
        if (!chatUsable || host.engine !== engine) { log.info(`title for ${c.id} skipped: utility model not usable and the chat model is not available`); return null; }
        messages = [...chatPrompt(c.history), { role: 'assistant', content: c.replyText }, { role: 'user', content: TITLE_FOLLOWUP }];
        const ids = engine.promptIds(messages, { think: false });
        if (ids.length + TITLE_MAX_TOKENS + 8 >= engine.ctx) { log.info(`title for ${c.id} skipped: conversation fills the context window`); return null; }
        // Only continue a conversation the engine actually holds. Its KV cache is a valid prefix of this prompt when it matches at least the
        // chat's own prompt (the part after it - the reply, as the model tokenized it - may differ a little); if the memory belongs to some
        // other conversation, a title run would throw that away, so it is left alone.
        const kv = engine.model?.tokens;
        const promptLen = engine.promptIds(chatPrompt(c.history), { think: false }).length;
        if (!Array.isArray(kv) || commonLen(kv, ids) < Math.min(kv.length, promptLen) - 16) { log.info(`title for ${c.id} skipped: the chat model's memory holds another conversation`); return null; }
      }
      const stats = {};
      const t0 = performance.now();
      const out = await run(engine, messages, abort, stats);
      if (abort.aborted) return null;
      log.info(`title run (${own ? 'utility model' : 'chat model'}): ${stats.promptTokens} prompt tokens, ${stats.cachedTokens} cached, ${stats.generated} generated, ${Math.round(performance.now() - t0)} ms`);
      return out;
    }, { signal: abort, label: 'titling', onLoading: (b) => { job.phase = b ? 'loading' : 'running'; } });
  }

  async function execute(c, abort, j) {
    try {
      const raw = await generate(c, abort, j);
      if (raw === null || raw === undefined || abort.aborted) { if (abort.aborted) log.info(`title for ${c.id} aborted`); return; }
      const name = cleanTitle(raw);
      if (!name) { log.info(`title for ${c.id}: model answer not usable (${JSON.stringify(String(raw).slice(0, 80))}); keeping the placeholder name`); chatStore.markAutoTitle(c.id, 'failed'); return; }
      const written = chatStore.applyAutoTitle(c.id, name, { userText: c.userText });
      if (written) { log.info(`title for ${c.id}: "${written}"`); emit('title', { chatId: c.id, name: written }); }
      else log.info(`title for ${c.id} dropped: the chat was renamed, edited or deleted meanwhile`);
    } catch (e) {
      log.error(`title for ${c.id} failed: ${e.stack ?? e}`); // silent for the user
    }
  }

  return {
    /** Start the job for `chatId` if it qualifies. Never throws; returns true when a job was started. */
    maybeStart(chatId) {
      try {
        if (getConfig().auto_titles === false || job || runActive()) return false;
        const c = chatStore.autoTitleCandidate(chatId);
        if (!c) return false;
        const n = attempts.get(chatId) ?? 0;
        if (n >= MAX_ATTEMPTS) return false;
        attempts.set(chatId, n + 1);
        const abort = { aborted: false };
        const j = { chatId, abort, phase: 'running', usesChat: false, promise: null };
        job = j;
        j.promise = execute(c, abort, j).finally(() => { if (job === j) job = null; });
        return true;
      } catch (e) { log.error(`title start failed: ${e.stack ?? e}`); return false; }
    },
    /**
     * Abort a running job and wait until it has let go of the engine. A job that is only loading its own small model is not
     * waited for (it does not touch the chat engine; it notices the abort when the load is done and closes the model again),
     * unless `all` (shutdown).
     */
    async stop({ all = false } = {}) {
      const j = job; if (!j) return;
      j.abort.aborted = true;
      if (all || j.usesChat || j.phase !== 'loading') await j.promise;
    },
    active: () => job !== null,
  };
}
