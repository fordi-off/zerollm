// Byte-level BPE tokenizer (GPT-2 family: Qwen2/Qwen3, Llama 3), built from GGUF metadata.
import { BYTE_TO_CHAR, CHAR_TO_BYTE } from './bytes.js';

// Contractions written with explicit case classes so we don't depend on (?i:...) support.
const CONTRACTIONS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
const tail = (digits) => `[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|${digits}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`;

/** Pre-tokenizer regexes by kind. */
export const PRETOKENIZERS = {
  // Qwen2 / Qwen2.5 / Qwen3: digits split one at a time
  qwen2: new RegExp(`${CONTRACTIONS}|${tail('\\p{N}')}`, 'gu'),
  // Qwen3.5 (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35): like qwen2, but combining marks (\\p{M}) belong to words: letters are
  // [\\p{L}\\p{M}]+ and the punctuation run excludes \\p{M} (the optional lead character of a word still may not be \\p{L}/\\p{N}/CR/LF)
  qwen35: new RegExp(`${CONTRACTIONS}|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`, 'gu'),
  // Llama 3.x: digits grouped 1-3
  llama3: new RegExp(`${CONTRACTIONS}|${tail('\\p{N}{1,3}')}`, 'gu'),
  // original GPT-2
  gpt2: new RegExp("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", 'gu'),
};

/** Map GGUF `tokenizer.ggml.pre` to a pre-tokenizer kind. */
export function preKind(pre) {
  switch (pre) {
    case 'qwen2': case 'qwen3': return 'qwen2';
    case 'qwen35': return 'qwen35';
    case 'llama-bpe': case 'llama3': case 'smaug-bpe': return 'llama3';
    case 'gpt-2': case 'gpt2': case undefined: return 'gpt2';
    default: throw new Error(`unsupported tokenizer.ggml.pre "${pre}" (supported: qwen2, qwen35, llama-bpe, gpt-2)`);
  }
}

/** @param {string} text @param {'qwen2'|'qwen35'|'llama3'|'gpt2'} kind */
export function preTokenize(text, kind) {
  return text.match(PRETOKENIZERS[kind]) ?? [];
}

const TOKEN_CONTROL = 3;
const TOKEN_USER_DEFINED = 4;

/** Small binary min-heap on (rank, pos). */
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  static less(x, y) { return x.rank < y.rank || (x.rank === y.rank && x.left < y.left); }
  push(v) {
    const a = this.a;
    a.push(v);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1; const r = l + 1; let m = i;
        if (l < a.length && Heap.less(a[l], a[m])) m = l;
        if (r < a.length && Heap.less(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}

export class Tokenizer {
  /**
   * @param {object} p
   * @param {string[]} p.tokens
   * @param {number[]|undefined} p.types
   * @param {string[]} p.merges
   * @param {string|undefined} p.pre
   * @param {number|undefined} p.bos
   * @param {number|undefined} p.eos
   * @param {boolean} [p.addBos]
   * @param {boolean} [p.ignoreMerges] use a whole pre-token directly when the vocabulary has it (default: true for Llama 3)
   */
  constructor(p) {
    this.tokens = p.tokens;
    this.types = p.types ?? [];
    this.kind = preKind(p.pre);
    // Llama 3's tokenizer (tiktoken-derived; HF tokenizer.json: "ignore_merges": true) looks the whole pre-token up in the
    // vocabulary first; its merge list does not always reproduce every vocabulary entry. Qwen2 does not do this.
    this.ignoreMerges = p.ignoreMerges ?? this.kind === 'llama3';
    this.bosId = p.bos;
    this.eosId = p.eos;
    this.addBos = p.addBos ?? false;
    this.vocabSize = p.tokens.length;

    /** @type {Map<string, number>} */ this.tokenToId = new Map();
    for (let i = 0; i < p.tokens.length; i++) if (!this.tokenToId.has(p.tokens[i])) this.tokenToId.set(p.tokens[i], i);

    /** @type {Map<string, number>} merge pair "a b" -> rank */
    this.ranks = new Map();
    for (let i = 0; i < p.merges.length; i++) if (!this.ranks.has(p.merges[i])) this.ranks.set(p.merges[i], i);

    // special tokens: control + user-defined, matched literally in text when parsing special
    this.specialIds = new Set();
    const specials = [];
    for (let i = 0; i < p.tokens.length; i++) {
      const t = this.types[i];
      if ((t === TOKEN_CONTROL || t === TOKEN_USER_DEFINED) && p.tokens[i].length > 0) { specials.push(p.tokens[i]); this.specialIds.add(i); }
    }
    specials.sort((a, b) => b.length - a.length);
    this.specialRe = specials.length
      ? new RegExp(specials.map((s) => s.replace(/[.*+?^${}()|[\]\\\/-]/g, '\\$&')).join('|'), 'g')
      : null;

    /** @type {Map<string, number[]>} */ this.cache = new Map();
    /** @type {(Uint8Array|undefined)[]} */ this.bytesCache = new Array(p.tokens.length);
    this.encoder = new TextEncoder();
  }

  /** @param {import('../gguf/reader.js').GGUFFile} g */
  static fromGGUF(g) {
    const model = g.get('tokenizer.ggml.model');
    if (model !== 'gpt2') throw new Error(`unsupported tokenizer.ggml.model "${model}" (only byte-level BPE "gpt2" is implemented)`);
    const tokens = g.get('tokenizer.ggml.tokens');
    const merges = g.get('tokenizer.ggml.merges');
    if (!Array.isArray(tokens) || !Array.isArray(merges)) throw new Error('GGUF has no tokenizer.ggml.tokens / merges');
    return new Tokenizer({
      tokens,
      types: g.get('tokenizer.ggml.token_type'),
      merges,
      pre: g.get('tokenizer.ggml.pre'),
      bos: g.get('tokenizer.ggml.bos_token_id'),
      eos: g.get('tokenizer.ggml.eos_token_id'),
      addBos: g.get('tokenizer.ggml.add_bos_token', false),
    });
  }

  /** id of a token string, or undefined. @param {string} s */
  idOf(s) { return this.tokenToId.get(s); }

  /** BPE-merge one pre-token that is already byte-mapped (each char = one byte). @param {string} word */
  bpeWord(word) {
    const cached = this.cache.get(word);
    if (cached) return cached;
    if (this.ignoreMerges) {
      const whole = this.tokenToId.get(word);
      if (whole !== undefined) { const ids = [whole]; if (this.cache.size > 50000) this.cache.clear(); this.cache.set(word, ids); return ids; }
    }
    const n = word.length;
    let ids;
    if (n === 1) {
      ids = [this.#idOrFallback(word)];
    } else {
      const start = new Int32Array(n); const len = new Int32Array(n);
      const prev = new Int32Array(n); const next = new Int32Array(n);
      for (let i = 0; i < n; i++) { start[i] = i; len[i] = 1; prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
      const heap = new Heap();
      const tryPair = (l, r) => {
        if (l < 0 || r < 0) return;
        const rank = this.ranks.get(`${word.substr(start[l], len[l])} ${word.substr(start[r], len[r])}`);
        if (rank !== undefined) heap.push({ rank, left: l, right: r, size: len[l] + len[r] });
      };
      for (let i = 0; i + 1 < n; i++) tryPair(i, i + 1);
      while (heap.size) {
        const b = heap.pop();
        if (len[b.left] === 0 || len[b.right] === 0 || len[b.left] + len[b.right] !== b.size) continue; // stale
        len[b.left] += len[b.right];
        len[b.right] = 0;
        next[b.left] = next[b.right];
        if (next[b.right] >= 0) prev[next[b.right]] = b.left;
        tryPair(prev[b.left], b.left);
        tryPair(b.left, next[b.left]);
      }
      ids = [];
      for (let i = 0; i >= 0; i = next[i]) ids.push(this.#idOrFallback(word.substr(start[i], len[i])));
    }
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  #idOrFallback(sym) {
    const id = this.tokenToId.get(sym);
    if (id !== undefined) return id;
    throw new Error(`token "${sym}" missing from vocab (corrupt tokenizer?)`);
  }

  /** @param {string} text plain text with no special-token parsing @returns {number[]} */
  encodeOrdinary(text) {
    const out = [];
    for (const piece of preTokenize(text, this.kind)) {
      const bytes = this.encoder.encode(piece);
      let word = '';
      for (let i = 0; i < bytes.length; i++) word += BYTE_TO_CHAR[bytes[i]];
      for (const id of this.bpeWord(word)) out.push(id);
    }
    return out;
  }

  /**
   * @param {string} text
   * @param {{special?: boolean, bos?: boolean}} [opts]
   *   special: parse special-token text (e.g. "<|im_end|>") into their ids (default true).
   *   bos: prepend BOS (default: GGUF add_bos_token).
   * @returns {number[]}
   */
  encode(text, opts = {}) {
    const special = opts.special ?? true;
    const out = [];
    if (opts.bos ?? this.addBos) {
      if (this.bosId === undefined) throw new Error('model has no BOS token');
      out.push(this.bosId);
    }
    if (!special || !this.specialRe) {
      for (const id of this.encodeOrdinary(text)) out.push(id);
      return out;
    }
    let last = 0;
    this.specialRe.lastIndex = 0;
    for (let m = this.specialRe.exec(text); m; m = this.specialRe.exec(text)) {
      if (m.index > last) for (const id of this.encodeOrdinary(text.slice(last, m.index))) out.push(id);
      out.push(this.tokenToId.get(m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) for (const id of this.encodeOrdinary(text.slice(last))) out.push(id);
    return out;
  }

  /** Raw UTF-8 bytes a token stands for. */
  tokenBytes(id) {
    let b = this.bytesCache[id];
    if (b) return b;
    const s = this.tokens[id];
    if (s === undefined) throw new Error(`token id ${id} out of range`);
    if (this.specialIds.has(id)) {
      b = this.encoder.encode(s);
    } else {
      const arr = [];
      let ok = true;
      for (const ch of s) {
        const v = CHAR_TO_BYTE.get(ch);
        if (v === undefined) { ok = false; break; }
        arr.push(v);
      }
      b = ok ? Uint8Array.from(arr) : this.encoder.encode(s);
    }
    this.bytesCache[id] = b;
    return b;
  }

  isControl(id) { return this.types[id] === TOKEN_CONTROL; }

  /**
   * @param {ArrayLike<number>} ids
   * @param {{skipControl?: boolean}} [opts] skip control tokens such as <|im_end|> (default true)
   */
  decode(ids, opts = {}) {
    const skip = opts.skipControl ?? true;
    const parts = [];
    let total = 0;
    for (let i = 0; i < ids.length; i++) {
      if (skip && this.isControl(ids[i])) continue;
      const b = this.tokenBytes(ids[i]);
      parts.push(b); total += b.length;
    }
    const all = new Uint8Array(total);
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    return new TextDecoder().decode(all);
  }

  /** Streaming decoder: multi-byte characters split across tokens are held back until complete. */
  streamDecoder(opts = {}) {
    const skip = opts.skipControl ?? true;
    const dec = new TextDecoder('utf-8', { fatal: false });
    return {
      /** @param {number} id */
      push: (id) => (skip && this.isControl(id) ? '' : dec.decode(this.tokenBytes(id), { stream: true })),
      flush: () => dec.decode(),
    };
  }
}
