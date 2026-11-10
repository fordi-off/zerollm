// Prompt building and generation safeguards, ported from Skibeti37 (chats.py / generation.py / state.js).

// Kept SHORT on purpose: small models (1B-8B) latch onto every topic the system prompt mentions - a sentence about math made
// Llama 3.2 1B answer "hei" with a LaTeX exercise - and every token of it costs prefill time.
export const SYSTEM_PROMPT =
  "You are a helpful assistant. Reply in the same language as the user's most recent message, and keep the whole reply " +
  'in that one language. Keep code, direct quotes and proper names in their original form. Match the user\'s tone.';

export const CONTINUATION_PROMPT =
  'Continue your previous answer exactly where you left off. Do not repeat anything you have already written, and do not ' +
  'write any introduction - continue straight on, in the very same language as the rest of your answer.';

/** Sampling defaults tuned in Skibeti37 (repeat penalty over a 320-token window; a low frequency penalty on purpose). */
export const SAMPLING = { temperature: 0.6, topP: 0.95, topK: 40, minP: 0.05, repeatPenalty: 1.18, repeatLastN: 320, frequencyPenalty: 0.1, presencePenalty: 0.3 };
export const DEFAULT_MAX_TOKENS = 2000;
/** Qwen3 thinking mode (Qwen's recommended settings): no repetition penalties - they punish the digits and phrases a long chain of reasoning must repeat. */
export const THINK_SAMPLING = { temperature: 0.6, topP: 0.95, topK: 20, minP: 0, repeatPenalty: 1, frequencyPenalty: 0, presencePenalty: 0 };
export const THINK_MAX_TOKENS = 6000;

/**
 * The messages actually sent to the model: the system prompt (+ optional extra text) and the history. NOTHING in an earlier
 * message may change from one turn to the next - the model keeps what it has already read (its KV cache) only for an unchanged
 * prefix, so a changing language reminder or re-picked document excerpts would make it re-read the whole conversation.
 * A user message that carries document context (`context`, stored with the message) renders it in front of the question.
 */
export function buildMessages(history, { extraSystem = '' } = {}) {
  const system = extraSystem ? `${SYSTEM_PROMPT}\n\n${extraSystem}` : SYSTEM_PROMPT;
  return [{ role: 'system', content: system }, ...history.map((m) => ({ role: m.role, content: m.role === 'user' ? renderWithContext(m.content, m.context) : m.content }))];
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

/** Token ids at the start of every prompt this app builds (system prompt + the template header): the part worth caching on disk. */
export function stablePrefixIds(tools, { extraSystem = '' } = {}) {
  const all = ['x', 'yy zz'].map((v) => tools.promptIds(buildMessages([{ role: 'user', content: v }], { extraSystem })));
  return all[0].slice(0, commonLen(all[0], all[1]));
}

// --- rolling summary of old messages (same wording and cache format as Skibeti37)
export const RECENT_KEEP = 6;      // newest messages always sent verbatim
export const SUMMARIZE_AT = 0.6;   // start folding old messages into the summary at this share of the prompt budget

export const SUMMARY_PREFIX =
  'Summary of earlier parts of this conversation (older messages are compressed to save space, but the user still has the full ' +
  'history saved). Keep replying in the language of the user\'s latest message:\n';

/** Messages for the small model that folds `chunk` into the running summary. */
export function summaryPrompt(existingSummary, chunk) {
  const text = chunk.map((m) => `${m.role}: ${String(m.content).slice(0, 2500)}`).join('\n');
  return [
    { role: 'system', content: 'You maintain a running summary of a conversation between a user and an AI assistant. Merge the existing summary with the new messages below into one short, concrete summary that preserves important facts, names, numbers and decisions. Keep it under 300 words - prioritise the most recent and still-relevant content, and drop small talk and details that no longer matter. Write the summary in the same language as the messages below.' },
    { role: 'user', content: `Existing summary:\n${existingSummary || '(none yet)'}\n\nNew messages to incorporate:\n${text}` },
  ];
}

/** Split messages into passes small enough for the summarizing model's context (about `maxChars` characters each). */
export function chunkForSummary(chunk, maxChars = 7000) {
  const passes = []; let cur = []; let size = 0;
  for (const m of chunk) {
    const len = Math.min(String(m.content).length, 2500) + 12;
    if (cur.length && size + len > maxChars) { passes.push(cur); cur = []; size = 0; }
    cur.push(m); size += len;
  }
  if (cur.length) passes.push(cur);
  return passes;
}

// --- attached documents. The text of a file the user attaches is stored WITH the message it was attached to (`context`: [{filename, text}])
// and rendered in front of that message for the rest of the conversation: the prompt only ever grows at the end, so the model reads the
// document once and keeps it in memory (its KV cache) for every follow-up question.
export const DOC_PREFIX =
  'The user attached the document(s) below. Use them to answer the question that follows, in the language of the question. ' +
  'If the answer is not in them, say so instead of guessing.';

/** One user message with its attached documents in front of the question (unchanged text for messages without documents). */
export function renderWithContext(content, context) {
  if (!context?.length) return content;
  const block = context.map((e) => `<document name="${String(e.filename).replace(/"/g, "'")}">\n${e.text}\n</document>`).join('\n');
  return `${DOC_PREFIX}\n\n<documents>\n${block}\n</documents>\n\nQuestion: ${content}`;
}
