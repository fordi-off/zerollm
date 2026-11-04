// Prompt building and generation safeguards, ported from Skibeti37 (chats.py / generation.py / state.js).

export const SYSTEM_PROMPT =
  "You are a helpful assistant. Reply in the same language as the user's most recent message, and keep the whole reply " +
  'in that one language - never switch or mix languages partway through. If the user changes language, follow them from ' +
  'their next message on. Keep code, direct quotes and proper names in their original form. Match the user\'s tone, and ' +
  'think step by step when a question needs it. The chat renders Markdown plus math: use \\( \\) for inline formulas and ' +
  '\\[ \\] or $$ $$ for standalone ones; never use a bare single $ for math, since that reads as currency.';

export const CONTINUATION_PROMPT =
  'Continue your previous answer exactly where you left off. Do not repeat anything you have already written, and do not ' +
  'write any introduction - continue straight on, in the very same language as the rest of your answer.';

/** Sampling defaults tuned in Skibeti37 (repeat penalty over a 320-token window; a low frequency penalty on purpose). */
export const SAMPLING = { temperature: 0.6, topP: 0.95, topK: 40, minP: 0.05, repeatPenalty: 1.18, repeatLastN: 320, frequencyPenalty: 0.1, presencePenalty: 0.3 };
export const DEFAULT_MAX_TOKENS = 2000;

// --- a small, dependency-free Norwegian/English sniff (small models drift, naming the language last keeps them steady)
const NO_WORDS = new Set(['og', 'ikke', 'ikkje', 'jeg', 'det', 'er', 'som', 'på', 'en', 'et', 'ein', 'ei', 'å', 'har', 'kan', 'vil', 'hva', 'kva', 'hvorfor', 'hvordan', 'med', 'til', 'de', 'vi', 'du', 'meg', 'deg', 'være', 'blir', 'gjøre', 'hvis', 'eller', 'men', 'så', 'noen', 'noe', 'mye', 'mer', 'mest', 'veldig', 'bare', 'hele', 'takk', 'hei', 'kanskje', 'fordi', 'denne', 'dette', 'disse', 'slik', 'her', 'der', 'når', 'skal', 'må', 'om', 'av', 'at', 'den', 'ja', 'nei', 'mellom']);
const EN_WORDS = new Set(['the', 'and', 'is', 'are', 'you', 'what', 'why', 'how', 'this', 'that', 'of', 'to', 'in', 'it', 'for', 'with', 'can', 'could', 'would', 'should', 'please', 'thanks', 'hey', 'hello', 'make', 'want', 'need', 'because', 'there', 'their', 'about', 'which', 'when', 'where', 'does', 'do', 'explain', 'write', 'tell', 'give', 'list', 'show', 'describe', 'compare', 'summarize', 'summarise', 'into', 'terms', 'between', 'difference', 'example', 'simple', 'help']);

/** "Norwegian", "English", or null (unsure: let the model decide). */
export function detectLanguage(text) {
  if (!text) return null;
  const low = text.toLowerCase();
  const words = low.match(/[a-zà-ÿæøå]+/g) ?? [];
  if (words.length < 2) return null;
  let no = 0; let en = 0;
  for (const w of words) { if (NO_WORDS.has(w)) no++; if (EN_WORDS.has(w)) en++; }
  const hasNoChars = /[æøå]/.test(low);
  if (hasNoChars && en === 0) return 'Norwegian';
  if (no > en && (no >= 2 || hasNoChars)) return 'Norwegian';
  if (en > no && en >= 2) return 'English';
  if (hasNoChars && no >= en) return 'Norwegian';
  return null;
}

export function languageDirective(lastUserText) {
  const lang = detectLanguage(lastUserText);
  if (lang) {
    return `LANGUAGE: The user's latest message is written in ${lang}. Write your entire reply in ${lang} (from the first word to the last). Do not switch or mix languages partway through, and do not translate the user's own words back to them. Keep code, direct quotes and proper names in their original form.`;
  }
  return "LANGUAGE: Reply in the exact same language as the user's latest message (from the first word to the last). Do not switch or mix languages partway through. Keep code, direct quotes and proper names in their original form.";
}

/** The messages actually sent to the model: one system message (prompt + extras + language reminder LAST) + history. */
export function buildMessages(history, { extraSystem = '' } = {}) {
  const lastUser = [...history].reverse().find((m) => m.role === 'user');
  let system = SYSTEM_PROMPT;
  if (extraSystem) system += `\n\n${extraSystem}`;
  system += `\n\n${languageDirective(lastUser?.content ?? '')}`;
  return [{ role: 'system', content: system }, ...history.map(({ role, content }) => ({ role, content }))];
}

const CUT = '\n\n[...forkortet for å passe kontekstvinduet...]';

/**
 * Make sure the prompt leaves room for the reply: drop whole turns from the oldest end (always keeping the last message),
 * then trim an oversized system block, then an oversized last message. `count(messages)` returns the exact token count.
 * @returns {{messages: Array, trimmed: boolean}}
 */
export function fitToContext(messages, count, contextMax, responseReserve) {
  const budget = Math.max(512, contextMax - responseReserve - 64);
  if (count(messages) <= budget) return { messages, trimmed: false };
  const system = messages[0];
  const rest = messages.slice(1);
  while (rest.length > 1 && count([system, ...rest]) > budget) rest.shift();
  let msgs = [system, ...rest];
  if (count(msgs) > budget) { // the system block itself is too big: keep the head
    const over = count(msgs) - budget;
    const keep = Math.max(400, system.content.length - over * 4 - CUT.length - 400);
    msgs[0] = { role: 'system', content: system.content.slice(0, keep).trimEnd() + CUT };
  }
  if (count(msgs) > budget && msgs.length > 1) { // last resort: one giant message
    const headTokens = count(msgs.slice(0, -1));
    const roomChars = Math.max(400, (budget - headTokens) * 4 - CUT.length - 400);
    const last = msgs.at(-1);
    msgs = [...msgs.slice(0, -1), { ...last, content: last.content.slice(0, roomChars).trimEnd() + CUT }];
  }
  return { messages: msgs, trimmed: true };
}

// If the same long span keeps coming back, the model has degenerated into a loop: stop instead of filling the budget.
const LOOP_WINDOW = 180; const LOOP_THRESHOLD = 3;
export function looksLooped(text) {
  if (text.length < LOOP_WINDOW * LOOP_THRESHOLD) return false;
  const tail = text.slice(-LOOP_WINDOW);
  let count = 0; let from = 0;
  for (;;) { const i = text.indexOf(tail, from); if (i < 0) break; count++; from = i + 1; if (count >= LOOP_THRESHOLD) return true; }
  return false;
}

/** Length of the common leading run of two token arrays. */
function commonLen(a, b) { let i = 0; const n = Math.min(a.length, b.length); while (i < n && a[i] === b[i]) i++; return i; }

/**
 * Token ids at the start of every prompt this app builds (system prompt + the template header), whatever language the
 * user writes in: the common prefix of the prompts for a Norwegian, an English and an unrecognised message. These are
 * the tokens worth caching on disk.
 * @param {{promptIds: (m: any[]) => number[]}} tools
 */
export function stablePrefixIds(tools, { extraSystem = '' } = {}) {
  const variants = ['Hei, hva skjer i dag?', 'Hello, what is going on today?', 'x'];
  const all = variants.map((v) => tools.promptIds(buildMessages([{ role: 'user', content: v }], { extraSystem })));
  const len = all.reduce((m, ids) => Math.min(m, commonLen(all[0], ids)), all[0].length);
  return all[0].slice(0, len);
}
