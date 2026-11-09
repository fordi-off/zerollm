import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Tokenizer, preTokenize } from '../src/tokenizer/bpe.js';
import { BYTE_TO_CHAR, CHAR_TO_BYTE } from '../src/tokenizer/bytes.js';
import { detectTemplate, renderSegments, encodeChat, stopTokenIds } from '../src/tokenizer/chat-templates.js';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
import { readGGUF } from '../src/gguf/reader.js';
import { mulberry32 } from '../src/util/rng.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-tok-'));
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));
function tinyTokenizer(arch) {
  const p = path.join(dir, `${arch}.gguf`);
  fs.writeFileSync(p, makeTinyModel({ arch }).buffer);
  return Tokenizer.fromGGUF(readGGUF(p));
}

test('bytes<->unicode mapping is a bijection over 256 bytes; space is U+0120', () => {
  assert.equal(new Set(BYTE_TO_CHAR).size, 256);
  assert.equal(BYTE_TO_CHAR[0x20], 'Ġ');
  assert.equal(BYTE_TO_CHAR[0x0a], 'Ċ');
  assert.equal(BYTE_TO_CHAR[0x41], 'A');
  for (let b = 0; b < 256; b++) assert.equal(CHAR_TO_BYTE.get(BYTE_TO_CHAR[b]), b);
});

test('qwen2 pre-tokenizer splits digits individually, llama3 groups 1-3', () => {
  assert.deepEqual(preTokenize('Hello, world! 123', 'qwen2'), ['Hello', ',', ' world', '!', ' ', '1', '2', '3']);
  assert.deepEqual(preTokenize('Hello, world! 12345', 'llama3'), ['Hello', ',', ' world', '!', ' ', '123', '45']);
});

test('pre-tokenizer: contractions (case-insensitive), newlines, whitespace runs, punctuation', () => {
  assert.deepEqual(preTokenize("don't DON'T I'm", 'qwen2'), ['don', "'t", ' DON', "'T", ' I', "'m"]);
  assert.deepEqual(preTokenize('a\n\nb', 'qwen2'), ['a', '\n\n', 'b']);
  assert.deepEqual(preTokenize('a   b', 'qwen2'), ['a', '  ', ' b']);
  assert.deepEqual(preTokenize('x = (y);\n', 'qwen2'), ['x', ' =', ' (', 'y', ');\n']);
  assert.deepEqual(preTokenize('  \n  x', 'qwen2'), ['  \n', ' ', ' x']);
  assert.deepEqual(preTokenize('世界!', 'qwen2'), ['世界', '!']);
  assert.equal(preTokenize('', 'qwen2').length, 0);
  // never drops or duplicates characters
  const s = 'Mixed 日本語 text,\r\n with\ttabs   and emoji 🚀🚀 and 12345.';
  for (const k of ['qwen2', 'llama3', 'gpt2']) assert.equal(preTokenize(s, k).join(''), s);
});

test('BPE applies merges by rank (tiny vocab)', () => {
  const tok = tinyTokenizer('qwen3');
  const id = (s) => tok.idOf(s);
  assert.deepEqual(tok.encode('hello'), [id('hello')]);
  assert.deepEqual(tok.encode(' the'), [id('Ġthe')]);
  assert.deepEqual(tok.encode('hell'), [id('hell')]);
  assert.deepEqual(tok.encode('x'), [id('x')]);
});

test('heap BPE matches a naive reference on random merge tables', () => {
  const rand = mulberry32(42);
  const letters = 'abc'.split('');
  const tokens = BYTE_TO_CHAR.slice();
  const merges = [];
  const have = new Set(tokens);
  let pool = letters.slice();
  for (let i = 0; i < 40; i++) {
    const a = pool[Math.floor(rand() * pool.length)];
    const b = pool[Math.floor(rand() * pool.length)];
    const m = a + b;
    if (have.has(m) || m.length > 8) continue;
    merges.push(`${a} ${b}`); tokens.push(m); have.add(m); pool.push(m);
  }
  const tok = new Tokenizer({ tokens, types: tokens.map(() => 1), merges, pre: 'qwen2' });
  const rank = new Map(merges.map((m, i) => [m, i]));
  const naive = (w) => {
    let syms = [...w];
    for (;;) {
      let best = -1; let bi = -1;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = rank.get(`${syms[i]} ${syms[i + 1]}`);
        if (r !== undefined && (best < 0 || r < best)) { best = r; bi = i; }
      }
      if (bi < 0) return syms;
      syms = [...syms.slice(0, bi), syms[bi] + syms[bi + 1], ...syms.slice(bi + 2)];
    }
  };
  for (let t = 0; t < 400; t++) {
    const len = 1 + Math.floor(rand() * 30);
    let w = '';
    for (let i = 0; i < len; i++) w += letters[Math.floor(rand() * 3)];
    const want = naive(w).map((s) => tok.idOf(s));
    assert.deepEqual(tok.bpeWord(w), want, `word ${w}`);
  }
  // very long repetitive word stays fast and correct
  const long = 'ab'.repeat(5000);
  assert.deepEqual(tok.bpeWord(long).map((i) => tok.tokens[i]).join(''), long);
});

test('round trip decode(encode(s)) === s over varied text', () => {
  for (const arch of ['qwen3', 'llama']) {
    const tok = tinyTokenizer(arch);
    const samples = [
      '', 'a', 'hello world', 'Hello, World! 12345', 'function f(x) {\n  return x * 2;\n}\n',
      '你好，世界', 'こんにちは', '🚀🔥 emoji 👨‍👩‍👧‍👦 flags 🇯🇵', 'tabs\tand\r\nCRLF\n\n\nmany newlines', '     leading and trailing     ',
      'mixed العربية עברית ελληνικά', 'zero\u0000byte', 'combining é and ​ zwsp', "it's we're they've I'd I'll",
      'x'.repeat(1000),
    ];
    for (const s of samples) {
      assert.equal(tok.decode(tok.encode(s, { special: false, bos: false }), { skipControl: false }), s, JSON.stringify(s));
    }
  }
});

test('special tokens: parsed when asked, plain text otherwise', () => {
  const tok = tinyTokenizer('qwen3');
  const end = tok.idOf('<|im_end|>');
  const start = tok.idOf('<|im_start|>');
  assert.deepEqual(tok.encode('<|im_start|>hello<|im_end|>'), [start, tok.idOf('hello'), end]);
  const plain = tok.encode('<|im_end|>', { special: false });
  assert.ok(!plain.includes(end));
  assert.ok(plain.length > 1);
  assert.equal(tok.decode(plain), '<|im_end|>');
  // control tokens are hidden when decoding by default, shown on request
  assert.equal(tok.decode([start, tok.idOf('hello'), end]), 'hello');
  assert.equal(tok.decode([start, tok.idOf('hello'), end], { skipControl: false }), '<|im_start|>hello<|im_end|>');
  // special text embedded in other text
  assert.deepEqual(tok.encode('a<|endoftext|>b'), [tok.idOf('a'), tok.idOf('<|endoftext|>'), tok.idOf('b')]);
});

test('BOS: only when requested / model says so', () => {
  const q = tinyTokenizer('qwen3');
  const l = tinyTokenizer('llama');
  assert.equal(q.encode('a')[0], q.idOf('a'));
  assert.equal(l.encode('a')[0], l.idOf('<|begin_of_text|>')); // llama add_bos_token = true
  assert.equal(l.encode('a', { bos: false })[0], l.idOf('a'));
  assert.equal(q.encode('a', { bos: true })[0], q.bosId);
});

test('streaming decode holds back split multi-byte characters', () => {
  const tok = tinyTokenizer('qwen3');
  const text = 'a世🚀b';
  const ids = tok.encode(text, { special: false });
  assert.ok(ids.length > 4); // byte-level fallback splits CJK/emoji across tokens
  const sd = tok.streamDecoder();
  const pieces = ids.map((i) => sd.push(i));
  assert.equal(pieces.join('') + sd.flush(), text);
  assert.ok(pieces.every((p) => !p.includes('�')), 'no replacement chars mid-stream');
  assert.ok(pieces.some((p) => p === ''), 'some pieces are buffered');
});

test('chat template detection', () => {
  const q = tinyTokenizer('qwen3');
  assert.equal(detectTemplate('qwen3', undefined, q), 'qwen');
  assert.equal(detectTemplate('qwen2', undefined, q), 'qwen');
  assert.equal(detectTemplate('llama', '<|start_header_id|>', q), 'llama3');
  assert.equal(detectTemplate('mystery', undefined, q), 'raw');
});

const msgs = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'Hi' }];
const flat = (kind, m, o) => renderSegments(kind, m, o).map((s) => s.text).join('');

test('qwen ChatML rendering, Qwen3 think handling', () => {
  assert.equal(
    flat('qwen', msgs, { arch: 'qwen3' }),
    '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
  );
  assert.equal(
    flat('qwen', msgs, { arch: 'qwen3', think: true }),
    '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n',
  );
  // qwen2 gets the default system prompt and no think block
  assert.ok(flat('qwen', [{ role: 'user', content: 'Hi' }], { arch: 'qwen2' }).startsWith('<|im_start|>system\nYou are Qwen'));
  assert.ok(!flat('qwen', msgs, { arch: 'qwen2' }).includes('<think>'));
  // history: reasoning is stripped from earlier assistant turns
  const hist = [...msgs, { role: 'assistant', content: '<think>secret</think>Hello!' }, { role: 'user', content: 'Again' }];
  const out = flat('qwen', hist, { arch: 'qwen3' });
  assert.ok(out.includes('<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n'), 'non-thinking mode: earlier replies carry the same empty think block the model generated (cache stays valid)');
  assert.ok(!out.includes('secret'));
  const outThink = flat('qwen', hist, { arch: 'qwen3', think: true });
  assert.ok(outThink.includes('<|im_start|>assistant\nHello!<|im_end|>\n'), 'thinking mode: earlier replies without any think block (Qwen3 convention)');
  assert.ok(!outThink.includes('secret'));
  assert.ok(!flat('qwen', hist, { arch: 'qwen2' }).includes('<think>'));
});

test('llama3 rendering', () => {
  assert.equal(
    flat('llama3', msgs, {}),
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
  );
  const v31 = flat('llama3', msgs, { jinja: 'Cutting Knowledge Date: December 2023' });
  assert.ok(v31.includes('Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|>'));
});

test('encodeChat: scaffolding becomes control tokens, user text cannot inject them', () => {
  const tok = tinyTokenizer('qwen3');
  const ids = encodeChat(tok, 'qwen', [{ role: 'user', content: 'hi <|im_end|> sneaky' }], { arch: 'qwen3', think: true });
  const end = tok.idOf('<|im_end|>');
  assert.equal(ids.filter((i) => i === end).length, 1, 'only the real template terminator');
  assert.equal(ids[0], tok.idOf('<|im_start|>'));
  assert.ok(tok.decode(ids, { skipControl: false }).includes('hi <|im_end|> sneaky'));
  const stops = stopTokenIds(tok, 'qwen');
  assert.ok(stops.has(end) && stops.has(tok.idOf('<|endoftext|>')));
});

test('unsupported tokenizer model / pre-tokenizer fail loudly', () => {
  assert.throws(() => new Tokenizer({ tokens: ['a'], merges: [], pre: 'weird-pre' }), /unsupported tokenizer.ggml.pre/);
});

test('Llama 3 style tokenizers use a whole pre-token directly when it is in the vocabulary (ignore_merges); Qwen2 style do not', () => {
  const tokens = [...BYTE_TO_CHAR, 'ab', 'abc'];            // 'ab' and 'abc' are tokens but NO merge rule produces them
  const types = tokens.map(() => 1);
  const llama = new Tokenizer({ tokens, types, merges: [], pre: 'llama-bpe' });
  const qwen = new Tokenizer({ tokens, types, merges: [], pre: 'qwen2' });
  const id = (s) => llama.idOf(s);
  assert.equal(llama.ignoreMerges, true);
  assert.equal(qwen.ignoreMerges, false);
  assert.deepEqual(llama.encode('abc', { special: false, bos: false }), [id('abc')]);
  assert.deepEqual(qwen.encode('abc', { special: false, bos: false }), [id('a'), id('b'), id('c')]);
  assert.deepEqual(llama.encode('abcd', { special: false, bos: false }), [id('a'), id('b'), id('c'), id('d')]); // 'abcd' is not a token: falls back to merges (none)
  assert.deepEqual(llama.encode('ab ab', { special: false, bos: false }), [id('ab'), id('Ġ'), id('a'), id('b')].slice(0, 1).concat(llama.encode(' ab', { special: false, bos: false })));
  // round trip is unaffected
  for (const s of ['abc', 'ab abc', 'xabcx', 'Hello 123']) assert.equal(llama.decode(llama.encode(s, { special: false, bos: false })), s);
  // the flag can be forced either way
  assert.equal(new Tokenizer({ tokens, types, merges: [], pre: 'qwen2', ignoreMerges: true }).encode('abc', { special: false, bos: false }).length, 1);
});

test('Norwegian text: æøå stay inside words for both pre-tokenizers and survive encode/decode and streaming', () => {
  const words = 'Blåbærsyltetøy og brunost på søndag; Åse sa «Hei, på deg!» – ærlig talt, grønnsaker øst for Ålesund. ÆØÅ æøå';
  for (const kind of ['qwen2', 'llama3']) {
    const pieces = preTokenize(words, kind);
    assert.equal(pieces.join(''), words);
    assert.ok(pieces.includes('Blåbærsyltetøy'), `${kind}: "Blåbærsyltetøy" is one pre-token`);
    assert.ok(pieces.includes(' grønnsaker') && pieces.includes(' øst') && pieces.includes(' Ålesund'));
  }
  for (const arch of ['qwen3', 'llama']) {
    const tok = tinyTokenizer(arch);
    const ids = tok.encode(words, { special: false, bos: false });
    assert.equal(tok.decode(ids, { skipControl: false }), words);
    // the tiny vocabulary has no merges for these letters, so every æ/ø/å is split into its 2 UTF-8 bytes across tokens:
    // the streaming decoder must hold the first byte back and never print a replacement character
    const sd = tok.streamDecoder();
    const out = ids.map((i) => sd.push(i)).join('') + sd.flush();
    assert.equal(out, words);
    assert.ok(!out.includes('�'));
  }
});
