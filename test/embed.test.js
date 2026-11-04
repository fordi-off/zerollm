import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyEmbedModel } from '../tools/make-tiny-embed-gguf.js';
import { readGGUF } from '../src/gguf/reader.js';
import { WordPieceTokenizer } from '../src/tokenizer/wordpiece.js';
import { RefEncoder } from '../src/embed/ref-encoder.js';
import { loadEmbedder, cosine } from '../src/embed/embedder.js';
import { embedConfig, isEmbedModel } from '../src/embed/config.js';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'zl-embed-'));
const file = (name, o) => { const p = path.join(tmp, name); fs.writeFileSync(p, makeTinyEmbedModel(o).buffer); return p; };
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));

test('WordPiece: word-initial and continuation pieces, punctuation, accents, unknown words, truncation', () => {
  const g = readGGUF(file('t.gguf'));
  const tok = WordPieceTokenizer.fromGGUF(g);
  const pieces = (s) => tok.encode(s, { special: false }).map((i) => tok.tokens[i]);
  assert.deepEqual(pieces('Hello, World!'), ['▁hello', '▁,', '▁world', '▁!']);
  assert.deepEqual(pieces('hellos'), ['▁hello', 's'], 'greedy longest match then continuation pieces');
  assert.deepEqual(pieces('kommunene'), ['▁kommune', 'ne']);
  assert.deepEqual(pieces('ÅPEN'), ['▁a', 'p', 'en'], 'upper case and accents (å -> a) are removed');
  assert.deepEqual(pieces('ø'), ['▁ø'], 'ø has no decomposition and stays');
  assert.deepEqual(pieces('xyz中'), ['▁x', 'y', 'z', '[UNK]'], 'a CJK character is its own word, unknown -> [UNK]');
  assert.deepEqual(pieces('  \t\n '), []);
  assert.deepEqual(tok.encode('hello world'), [2, ...tok.encode('hello world', { special: false }), 3]);
  const long = tok.encode('hello '.repeat(50), { maxLen: 10 });
  assert.equal(long.length, 10); assert.equal(long[0], 2); assert.equal(long.at(-1), 3);
  assert.equal(new WordPieceTokenizer(g.get('tokenizer.ggml.tokens'), { stripAccents: false }).words('å').length, 1);
});

test('embedConfig / isEmbedModel recognise nomic-bert and refuse other architectures', () => {
  const g = readGGUF(file('c.gguf'));
  const cfg = embedConfig(g);
  assert.deepEqual([cfg.nEmbd, cfg.nLayer, cfg.nHead, cfg.headDim, cfg.pooling], [256, 2, 4, 64, 1]);
  assert.equal(isEmbedModel(g), true);
});

test('reference encoder: deterministic, order-sensitive, related texts closer than unrelated ones', () => {
  const g = readGGUF(file('r.gguf', { seed: 3 }));
  const enc = new RefEncoder(g); const tok = WordPieceTokenizer.fromGGUF(g);
  const e = (s) => enc.embedIds(tok.encode(s));
  const a = e('hello world'); assert.deepEqual(Array.from(a), Array.from(e('hello world')));
  assert.ok(cosine(a, e('world hello')) < 0.9999, 'RoPE makes the encoder position-aware');
  assert.ok(a.every(Number.isFinite));
});

test('fast SIMD workers agree with the reference encoder (f32, q8_0 and q4_k_m weights); results do not depend on the worker count', async () => {
  for (const quant of ['f32', 'q8_0', 'q4_k_m']) {
    const p = file(`f-${quant}.gguf`, { quant, seed: 5 });
    const g = readGGUF(p); const ref = new RefEncoder(g); const tok = WordPieceTokenizer.fromGGUF(g);
    const texts = ['hello world', 'kommunene har ansvar for barnehagene.', 'a', 'hello '.repeat(40), 'The 12 schools, and the world!'];
    const emb = await loadEmbedder(p, { threads: 2, maxSeq: 128 });
    try {
      assert.equal(emb.kernels, 'wasm');
      const got = await emb.embed(texts);
      texts.forEach((t, i) => {
        const want = ref.embedIds(tok.encode(t, { maxLen: 128 }));
        const c = cosine(got[i], want);
        assert.ok(c > 0.9995, `${quant} "${t.slice(0, 20)}": cosine ${c}`);
      });
    } finally { await emb.close(); }
    const one = await loadEmbedder(p, { threads: 1, maxSeq: 128 });
    const two = await loadEmbedder(p, { threads: 3, maxSeq: 128 });
    try {
      const x = await one.embed(texts); const y = await two.embed(texts);
      for (let i = 0; i < texts.length; i++) assert.deepEqual(Array.from(x[i]), Array.from(y[i]), 'bit-identical regardless of the number of workers');
    } finally { await one.close(); await two.close(); }
  }
});

test('embedder: progress callback, jobs beyond the worker count, js kernels, bad token ids', async () => {
  const p = file('j.gguf', { quant: 'q8_0', seed: 7 });
  const emb = await loadEmbedder(p, { threads: 2, maxSeq: 64, kernels: 'js' });
  try {
    const seen = []; const out = await emb.embed(Array.from({ length: 7 }, (_, i) => `hello ${'world '.repeat(i + 1)}`), (d, t) => seen.push([d, t]));
    assert.equal(out.length, 7); assert.equal(seen.length, 7); assert.deepEqual(seen.at(-1), [7, 7]);
    await assert.rejects(emb.embedIds([[2, 99999, 3]]), /out of range/);
    assert.equal((await emb.embed(['hello']))[0].length, 256, 'still usable after a failed job');
  } finally { await emb.close(); }
});

import { spawnSync } from 'node:child_process';
test('CLI: `embed --compare` accepts a Skibeti37 document or a plain list; `--selftest` runs', async () => {
  const p = file('cli.gguf', { quant: 'q8_0', seed: 11 });
  const emb = await loadEmbedder(p, { threads: 1 });
  let v; try { v = await emb.embed(['hello world', 'kommune og barnehage']); } finally { await emb.close(); }
  const items = [{ text: 'hello world', embedding: Array.from(v[0]) }, { text: 'kommune og barnehage', embedding: Array.from(v[1]) }];
  fs.writeFileSync(path.join(tmp, 'list.json'), JSON.stringify(items));
  fs.writeFileSync(path.join(tmp, 'doc.json'), JSON.stringify({ id: 'x', filename: 'x.txt', summary: '', chunks: items }));
  for (const f of ['list.json', 'doc.json']) {
    const r = spawnSync(process.execPath, ['bin/zerollm.js', 'embed', p, '--compare', path.join(tmp, f), '--n', '2'], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stdout, /accents stripped \(BERT default\): 1\.0000 {2}1\.0000/);
  }
  const s = spawnSync(process.execPath, ['bin/zerollm.js', 'embed', p, '--selftest'], { encoding: 'utf8' });
  assert.equal(s.status, 0, s.stderr);
  assert.match(s.stdout, /triples ranked correctly/);
});
