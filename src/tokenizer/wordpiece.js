// BERT-style WordPiece tokenizer as stored in GGUF (tokenizer.ggml.model = "bert"): word-initial pieces carry a leading "▁",
// continuation pieces are stored without the "##". Matches the HF uncased BERT tokenizer: NFD, lower-case, accents removed,
// punctuation and CJK characters become words of their own, then greedy longest-match-first.
const SOW = '▁'; // "▁"
const MAX_WORD_CHARS = 100;

const isPunct = (c) => {
  const cp = c.codePointAt(0);
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true; // ASCII symbols count as punctuation in BERT
  return /^\p{P}$/u.test(c);
};
const isCjk = (cp) => (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2EBEF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);

export class WordPieceTokenizer {
  /** @param {string[]} tokens @param {{unk?: number, cls?: number, sep?: number, pad?: number, stripAccents?: boolean, lowercase?: boolean}} [o] */
  constructor(tokens, o = {}) {
    this.tokens = tokens;
    this.index = new Map(); tokens.forEach((t, i) => { if (!this.index.has(t)) this.index.set(t, i); });
    const find = (name) => this.index.get(name);
    this.unk = o.unk ?? find('[UNK]') ?? 0;
    this.cls = o.cls ?? find('[CLS]');
    this.sep = o.sep ?? find('[SEP]');
    this.pad = o.pad ?? find('[PAD]');
    this.stripAccents = o.stripAccents ?? true;
    this.lowercase = o.lowercase ?? true;
    this.vocabSize = tokens.length;
  }

  /** @param {import('../gguf/reader.js').GGUFFile} g */
  static fromGGUF(g, o = {}) {
    if (g.get('tokenizer.ggml.model') !== 'bert') throw new Error(`not a WordPiece (bert) tokenizer: "${g.get('tokenizer.ggml.model')}"`);
    const tokens = g.get('tokenizer.ggml.tokens');
    if (!Array.isArray(tokens)) throw new Error('GGUF has no tokenizer.ggml.tokens');
    const id = (...keys) => { for (const k of keys) { const v = g.get(`tokenizer.ggml.${k}`); if (typeof v === 'number') return v; } return undefined; };
    return new WordPieceTokenizer(tokens, { unk: id('unknown_token_id'), cls: id('cls_token_id', 'bos_token_id'), sep: id('seperator_token_id', 'separator_token_id', 'eos_token_id'), pad: id('padding_token_id'), ...o });
  }

  /** Split text into BERT "words" (punctuation / CJK characters stand alone). */
  words(text) {
    let s = String(text).normalize('NFD');
    if (this.lowercase) s = s.toLowerCase();
    const out = []; let cur = '';
    const flush = () => { if (cur) { out.push(cur); cur = ''; } };
    for (const ch of s) {
      const cp = ch.codePointAt(0);
      if (cp === 0 || cp === 0xFFFD || (/^\p{Cc}$/u.test(ch) && !/\s/.test(ch))) continue;      // control characters vanish
      if (this.stripAccents && /^\p{Mn}$/u.test(ch)) continue;                                     // combining accents vanish
      if (/\s/.test(ch)) { flush(); continue; }
      if (isPunct(ch) || isCjk(cp)) { flush(); out.push(ch); continue; }
      cur += ch;
    }
    flush();
    return out;
  }

  /** Greedy longest-match-first pieces of one word, as ids ([UNK] for the whole word when any part cannot be matched). */
  wordIds(word) {
    const chars = [...word];
    if (chars.length > MAX_WORD_CHARS) return [this.unk];
    const ids = []; let start = 0;
    while (start < chars.length) {
      let end = chars.length; let found = -1;
      while (end > start) {
        const piece = (start === 0 ? SOW : '') + chars.slice(start, end).join('');
        const id = this.index.get(piece);
        if (id !== undefined) { found = id; break; }
        end--;
      }
      if (found < 0) return [this.unk];
      ids.push(found); start = end;
    }
    return ids;
  }

  /**
   * @param {string} text
   * @param {{special?: boolean, maxLen?: number}} [o] special: add [CLS] ... [SEP]; maxLen: truncate (specials included)
   * @returns {number[]}
   */
  encode(text, { special = true, maxLen = Infinity } = {}) {
    const ids = [];
    for (const w of this.words(text)) ids.push(...this.wordIds(w));
    const room = Math.max(0, maxLen - (special ? 2 : 0));
    const body = ids.length > room ? ids.slice(0, room) : ids;
    return special ? [this.cls, ...body, this.sep] : body;
  }

  decode(ids) {
    return ids.map((i) => this.tokens[i] ?? '').map((t) => (t.startsWith(SOW) ? ` ${t.slice(1)}` : t)).join('').trim();
  }
}
