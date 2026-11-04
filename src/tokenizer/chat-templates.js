// Chat templates implemented directly (no Jinja). Produces token ids via the tokenizer.
//
// Scaffolding text (role markers) is encoded with special-token parsing ON; user/assistant content is encoded
// with it OFF, so text like "<|im_end|>" typed by a user cannot inject control tokens.

/** @typedef {{role: 'system'|'user'|'assistant', content: string}} Message */

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
  if (arch === 'qwen2' || arch === 'qwen3') return 'qwen';
  if (arch === 'llama' && tok?.idOf('<|start_header_id|>') !== undefined) return 'llama3';
  return 'raw';
}

const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>\s*/g, '');

/**
 * @typedef {Object} TemplateOptions
 * @property {boolean} [think]        Qwen3: allow thinking. false (default) pre-fills an empty <think></think> block.
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

  if (kind === 'qwen') {
    let msgs = messages;
    if (opts.arch === 'qwen2' && msgs[0]?.role !== 'system') {
      msgs = [{ role: 'system', content: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }, ...msgs];
    }
    for (const m of msgs) {
      S(`<|im_start|>${m.role}\n`);
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
