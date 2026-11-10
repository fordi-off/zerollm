// Chat templates implemented directly (no Jinja). Produces token ids via the tokenizer.
//
// Scaffolding text (role markers) is encoded with special-token parsing ON; user/assistant content is encoded
// with it OFF, so text like "<|im_end|>" typed by a user cannot inject control tokens.

/** @typedef {{role: 'system'|'user'|'assistant', content: string, reasoning_content?: string}} Message */

export const TEMPLATE_KINDS = ['qwen', 'llama3', 'raw'];

/**
 * @param {string} arch general.architecture
 * @param {string|undefined} jinja tokenizer.chat_template contents
 * @param {import('./bpe.js').Tokenizer} [tok]
 * @returns {'qwen'|'llama3'|'raw'}
 */
export function detectTemplate(arch, jinja, tok) {
  if (jinja?.includes('<|im_start|>')) return 'qwen';
  if (jinja?.includes('<|start_header_id|>')) return 'llama3';
  if (arch === 'qwen2' || arch === 'qwen3' || arch === 'qwen35') return 'qwen';
  if (arch === 'llama' && tok?.idOf('<|start_header_id|>') !== undefined) return 'llama3';
  return 'raw';
}

/**
 * How an architecture's reasoning ("thinking") switch works:
 *   'model-opens'  Qwen3: with thinking ON the prompt ends at the assistant header and the model writes `<think>` itself;
 *   'prompt-opens' Qwen3.5: with thinking ON the generation prompt already ends with `<think>\n`, so the reply starts with the
 *                  reasoning (the model only writes `</think>`); with thinking OFF both end with an empty think block;
 *   null           no thinking mode (the switch is ignored).
 * @param {string} arch general.architecture @returns {'model-opens'|'prompt-opens'|null}
 */
export function thinkingMode(arch) {
  if (arch === 'qwen3') return 'model-opens';
  if (arch === 'qwen35') return 'prompt-opens';
  return null;
}

/**
 * Text that belongs in front of the generated reply so that it always reads `<think>reasoning</think>answer` (what the app
 * and the CLI show and what the history keeps): the opening tag the PROMPT already contains for 'prompt-opens' models, else ''.
 * @param {string} arch @param {boolean} think
 */
export function thinkOpening(arch, think) {
  return think && thinkingMode(arch) === 'prompt-opens' ? '<think>\n' : '';
}

const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>\s*/g, '');

/** Python's str.rstrip('\n') / lstrip('\n') as the Qwen3.5 Jinja template uses them. */
const rstripNl = (s) => s.replace(/\n+$/, '');
const lstripNl = (s) => s.replace(/^\n+/, '');

/**
 * Qwen3.5 (arch qwen35) ChatML, text only, following the model's Jinja template: contents are trimmed; the system message
 * comes first (a later one is an error, like the template's raise_exception); an assistant turn AFTER the last user query
 * is rendered `<think>\n{reasoning}\n</think>\n\n{content}` and earlier ones as bare `{content}`; the generation prompt
 * is `<|im_start|>assistant\n` + `<think>\n` (thinking on: the model continues with its reasoning and writes `</think>`
 * itself) or `<think>\n\n</think>\n\n` (thinking off, the default).
 *
 * One deliberate deviation, as for Qwen3 (see the 'qwen' branch below): with thinking OFF every assistant turn in the history is
 * rendered with the same empty think block the model generated after the generation prompt, so the prompt stays an exact
 * extension of what is already in the model's cache (the real template would drop the block of earlier turns).
 * With thinking ON the real template is followed exactly (earlier turns carry no reasoning).
 * Tool calls / tool responses / images are not implemented.
 */
function qwen35Segments(messages, opts, S, C) {
  const think = opts.think === true;
  let lastQuery = messages.length - 1;
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'user') { lastQuery = i; break; }
  messages.forEach((m, i) => {
    const content = String(m.content ?? '').trim();
    if (m.role === 'system') {
      if (i !== 0) throw new Error('System message must be at the beginning.');
      S('<|im_start|>system\n'); C(content); S('<|im_end|>\n');
    } else if (m.role === 'user') {
      S('<|im_start|>user\n'); C(content); S('<|im_end|>\n');
    } else if (m.role === 'assistant') {
      let reasoning = '';
      let text = content;
      if (typeof m.reasoning_content === 'string') reasoning = m.reasoning_content;
      else if (content.includes('</think>')) {
        const parts = content.split('</think>');
        reasoning = lstripNl(rstripNl(parts[0]).split('<think>').pop());
        text = lstripNl(parts[parts.length - 1]);
      }
      S('<|im_start|>assistant\n');
      if (!think) { S('<think>\n\n</think>\n\n'); C(typeof m.reasoning_content === 'string' ? text : stripThink(text)); } else if (i > lastQuery) {
        S('<think>\n'); C(reasoning.trim()); S('\n</think>\n\n'); C(text);
      } else C(text);
      S('<|im_end|>\n');
    } else throw new Error('Unexpected message role.');
  });
  if (opts.generationPrompt ?? true) S(think ? '<|im_start|>assistant\n<think>\n' : '<|im_start|>assistant\n<think>\n\n</think>\n\n');
}

/**
 * @typedef {Object} TemplateOptions
 * @property {boolean} [think]        Qwen3 / Qwen3.5: allow thinking. false (default) pre-fills an empty <think></think> block.
 *                                    Qwen3.5 thinking on: the prompt ends with `<think>\n` (Qwen3: the model writes <think> itself).
 * @property {string} [dateString]    Llama 3.1/3.2 system header date (fixed default keeps output deterministic).
 * @property {boolean} [generationPrompt] append the assistant header (default true)
 * @property {string} [arch]
 * @property {string} [jinja]
 */

/**
 * Build prompt segments: [{text, special}] where special=true means scaffolding.
 * @param {'qwen'|'llama3'|'raw'} kind
 * @param {Message[]} messages
 * @param {TemplateOptions} [opts]
 */
export function renderSegments(kind, messages, opts = {}) {
  const gen = opts.generationPrompt ?? true;
  /** @type {Array<{text: string, special: boolean}>} */
  const seg = [];
  const S = (text) => seg.push({ text, special: true });
  const C = (text) => seg.push({ text, special: false });

  if (kind === 'raw') {
    for (const m of messages) C(m.content);
    return seg;
  }

  if (kind === 'qwen' && opts.arch === 'qwen35') {
    qwen35Segments(messages, opts, S, C);
    return seg;
  }

  if (kind === 'qwen') {
    let msgs = messages;
    if (opts.arch === 'qwen2' && msgs[0]?.role !== 'system') {
      msgs = [{ role: 'system', content: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }, ...msgs];
    }
    for (const m of msgs) {
      S(`<|im_start|>${m.role}\n`);
      // Qwen3 in non-thinking mode generates "<think>\n\n</think>\n\n" before every reply; showing the same empty block in the history
      // keeps the prompt identical to what the model already read (its cache stays valid) instead of re-reading each earlier reply.
      if (m.role === 'assistant' && opts.arch === 'qwen3' && !opts.think) S('<think>\n\n</think>\n\n');
      C(m.role === 'assistant' ? stripThink(m.content) : m.content);
      S('<|im_end|>\n');
    }
    if (gen) {
      S('<|im_start|>assistant\n');
      if (opts.arch === 'qwen3' && !opts.think) S('<think>\n\n</think>\n\n');
    }
    return seg;
  }

  // llama3
  S('<|begin_of_text|>');
  let msgs = messages;
  if (opts.jinja?.includes('Cutting Knowledge Date')) {
    const sys = msgs[0]?.role === 'system' ? msgs[0].content.trim() : '';
    if (msgs[0]?.role === 'system') msgs = msgs.slice(1);
    S('<|start_header_id|>system<|end_header_id|>\n\n');
    C(`Cutting Knowledge Date: December 2023\nToday Date: ${opts.dateString ?? '26 Jul 2024'}\n\n${sys}`);
    S('<|eot_id|>');
  }
  for (const m of msgs) {
    S(`<|start_header_id|>${m.role}<|end_header_id|>\n\n`);
    C(m.content.trim());
    S('<|eot_id|>');
  }
  if (gen) S('<|start_header_id|>assistant<|end_header_id|>\n\n');
  return seg;
}

/**
 * Token ids for a conversation. Does NOT add BOS for qwen; llama3 scaffolding contains <|begin_of_text|>.
 * @param {import('./bpe.js').Tokenizer} tok
 * @param {'qwen'|'llama3'|'raw'} kind
 * @param {Message[]} messages
 * @param {TemplateOptions} [opts]
 * @returns {number[]}
 */
export function encodeChat(tok, kind, messages, opts = {}) {
  const ids = [];
  for (const s of renderSegments(kind, messages, opts)) {
    for (const id of tok.encode(s.text, { special: s.special, bos: false })) ids.push(id);
  }
  return ids;
}

/** Ids that end generation for a template (plus the GGUF EOS). @returns {Set<number>} */
export function stopTokenIds(tok, kind) {
  const names = kind === 'qwen' ? ['<|im_end|>', '<|endoftext|>'] : kind === 'llama3' ? ['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'] : [];
  const out = new Set();
  for (const n of names) { const id = tok.idOf(n); if (id !== undefined) out.add(id); }
  if (tok.eosId !== undefined) out.add(tok.eosId);
  return out;
}
