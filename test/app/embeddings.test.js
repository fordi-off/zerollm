import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { makeTinyEmbedModel } from '../../tools/make-tiny-embed-gguf.js';
import { loadEmbedder, cosine } from '../../src/embed/embedder.js';
import { fuseRankings, vectorRank } from '../../app/retrieval.js';
import { startTestApp } from './helpers.js';

const EMBED = { name: 'tiny-embed.gguf', raw: makeTinyEmbedModel({ quant: 'q8_0', seed: 9 }).buffer };
const TEXT = 'hello world og kommune. '.repeat(30);
const until = async (fn, ms = 20000) => { const t0 = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t0 > ms) throw new Error('timeout'); await new Promise((r) => setTimeout(r, 50)); } };

test('fuseRankings: reciprocal-rank fusion lets a vector-only hit and a keyword-only hit both in', () => {
  const chunks = [{ embedding: [1, 0] }, { embedding: [0, 1] }, { embedding: [0.7, 0.7] }, {}];
  const vr = vectorRank(chunks, [1, 0]);
  assert.deepEqual(vr.map((r) => r.index), [0, 2, 1], 'cosine order; chunks without a vector are skipped');
  const fused = fuseRankings([[{ index: 3 }, { index: 1 }], vr], { scoreOf: (i) => i });
  assert.ok(fused.map((r) => r.index).includes(3) && fused.map((r) => r.index).includes(0));
  assert.equal(fused[0].index, 1, 'present in both lists -> first');
});

test('with an embedding model installed, a new document is indexed in the background and questions use the vectors', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, EMBED] });
  try {
    assert.equal((await t.call('state')).embedding, 'tiny-embed.gguf');
    assert.deepEqual((await t.call('state')).models.map((m) => m.filename), ['tiny-llama.gguf'], 'the search model is not offered as a chat model');
    const chat = (await t.call('save_chat', { name: 'E', model: 'tiny-llama.gguf', messages: [] })).id;
    const r = await t.call('add_document', { chatId: chat, filename: 'k.txt', content: TEXT.repeat(3) });
    assert.equal(r.indexing, true);
    const doc = await until(async () => { const d = (await t.call('document_library'))[0]; return d.embedded === d.chunks && d.chunks > 0 ? d : null; });
    assert.ok(doc.chunks >= 2);
    const raw = JSON.parse(fs.readFileSync(path.join(t.dir, 'documents', `${r.id}.json`), 'utf8'));
    assert.equal(raw.chunks[0].embedding.length, 256);
    // the stored vector is exactly what the embedder gives for that chunk text
    const emb = await loadEmbedder(path.join(t.dir, 'models', 'tiny-embed.gguf'), { threads: 1 });
    try { assert.ok(cosine(Float32Array.from(raw.chunks[0].embedding), (await emb.embed([raw.chunks[0].text]))[0]) > 0.99999); } finally { await emb.close(); }
    assert.equal((await t.call('embedding_status')).missing, 0);

    const ev = await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'hello world' }], { extra: { chatId: chat } });
    assert.equal(ev.at(-1).type, 'done', JSON.stringify(ev.at(-1)));
    assert.ok(ev.at(-1).data.sources.length >= 1);
    assert.match(fs.readFileSync(path.join(t.dir, 'app.log'), 'utf8'), /embedding model ready: tiny-embed\.gguf/);
  } finally { await t.close(); }
});

test('semantic match without any shared keyword is found through the vectors (Skibeti37-style chunks with embeddings)', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, EMBED] });
  try {
    const emb = await loadEmbedder(path.join(t.dir, 'models', 'tiny-embed.gguf'), { threads: 1 });
    let qv; try { qv = Array.from((await emb.embed(['hello world']))[0]); } finally { await emb.close(); }
    const far = new Array(256).fill(0).map((_, i) => (i % 2 ? 1 : -1));
    fs.mkdirSync(path.join(t.dir, 'documents'), { recursive: true });
    const filler = Array.from({ length: 6 }, (_, i) => ({ text: `fyll ${i} bil hus tre`, embedding: far.map((v) => v * (i + 1)) }));
    fs.writeFileSync(path.join(t.dir, 'documents', 'sk.json'), JSON.stringify({ id: 'sk', filename: 'sk.txt', summary: '', chunks: [...filler, { text: 'zzz qqq xxx', embedding: qv }, { text: 'hello world', embedding: far }] }));
    const chat = (await t.call('save_chat', { name: 'S', model: 'tiny-llama.gguf', messages: [] })).id;
    await t.call('attach_document', { chatId: chat, docId: 'sk' });
    const ev = await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'hello world' }], { extra: { chatId: chat } });
    const texts = ev.at(-1).data.sources.map((s) => s.snippet);
    assert.ok(texts.includes('zzz qqq xxx'), `vector-only hit present: ${JSON.stringify(texts)}`);
    assert.ok(texts.includes('hello world'), 'keyword hit present');
    // without the embedding model the same chat falls back to keywords only
    fs.unlinkSync(path.join(t.dir, 'models', 'tiny-embed.gguf'));
    const ev2 = await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'hello world' }], { extra: { chatId: chat } });
    assert.deepEqual(ev2.at(-1).data.sources.map((s) => s.snippet), ['hello world']);
  } finally { await t.close(); }
});

test('without an embedding model nothing is indexed and everything still works; index_documents embeds old documents later', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    assert.equal((await t.call('state')).embedding, null);
    const chat = (await t.call('save_chat', { name: 'N', model: 'tiny-llama.gguf', messages: [] })).id;
    const r = await t.call('add_document', { chatId: chat, filename: 'k.txt', content: TEXT });
    assert.equal(r.indexing, false);
    assert.deepEqual(await t.call('index_documents'), { started: false });
    assert.equal((await t.call('document_library'))[0].embedded, 0);
    fs.writeFileSync(path.join(t.dir, 'models', 'tiny-embed.gguf'), EMBED.raw);
    assert.equal((await t.call('embedding_status')).missing, 1);
    assert.deepEqual(await t.call('index_documents'), { started: true });
    await until(async () => { const d = (await t.call('document_library'))[0]; return d.embedded === d.chunks; });
    assert.equal((await t.call('embedding_status')).missing, 0);
  } finally { await t.close(); }
});

test('while a document is being indexed, chat, summaries and model changes are refused; they work again afterwards', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, EMBED] });
  try {
    const chat = (await t.call('save_chat', { name: 'L', model: 'tiny-llama.gguf', messages: [] })).id;
    const big = 'hello world og kommune barnehage skole. '.repeat(1200); // enough chunks to keep the encoder busy for a moment
    const r = await t.call('add_document', { chatId: chat, filename: 'big.txt', content: big });
    assert.equal(r.indexing, true);
    const st = await t.call('state');
    assert.ok(st.indexing && st.indexing.filename === 'big.txt', 'state reports the running index job');
    const msg = [{ role: 'user', content: 'Hei' }];
    await assert.rejects(t.call('send_message', { model: 'tiny-llama.gguf', messages: msg, chatId: chat }), /indexing/);
    await assert.rejects(t.call('continue_message', { model: 'tiny-llama.gguf', messages: msg }), /indexing/);
    await assert.rejects(t.call('summarize_document', { docId: r.id, model: 'tiny-llama.gguf' }), /indexing/);
    await assert.rejects(t.call('set_context_window', { value: 2048 }), /indexing/);
    await assert.rejects(t.call('unload_model'), /indexing/);
    await assert.rejects(t.call('delete_model', { filename: 'tiny-llama.gguf' }), /indexing/);
    await until(async () => (await t.call('state')).indexing === null, 60000);
    const d = (await t.call('document_library'))[0]; assert.equal(d.embedded, d.chunks);
    const ev = await t.runChat('tiny-llama.gguf', msg, { extra: { chatId: chat } });
    assert.equal(ev.at(-1).type, 'done');
  } finally { await t.close(); }
});

test('a document added while a reply is being written is indexed right after it; a second one waits for the first', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, EMBED] });
  try {
    const chat = (await t.call('save_chat', { name: 'P', model: 'tiny-llama.gguf', messages: [] })).id;
    let added = null;
    await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'Fortell' }], {
      after: async (ev) => { if (ev.type === 'chunk' && !added) added = await t.call('add_document', { chatId: chat, filename: 'late.txt', content: TEXT }); },
    });
    assert.equal(added.indexing, false); assert.equal(added.deferred, true);
    await until(async () => { const d = (await t.call('document_library'))[0]; return d?.embedded === d?.chunks && d.chunks > 0; }, 30000);
    const a = await t.call('add_document', { chatId: chat, filename: 'a.txt', content: TEXT.repeat(40) });
    const b = await t.call('add_document', { chatId: chat, filename: 'b.txt', content: TEXT });
    assert.equal(a.indexing, true); assert.equal(b.indexing, false); assert.equal(b.deferred, true);
    await until(async () => { const lib = await t.call('document_library'); return lib.length === 3 && lib.every((d) => d.embedded === d.chunks) && (await t.call('state')).indexing === null; }, 60000);
  } finally { await t.close(); }
});

test('installing the search model (download finished) indexes the documents that have no vectors yet', async () => {
  const http = await import('node:http');
  const server = http.createServer((req, res) => { res.writeHead(200, { 'Content-Length': EMBED.raw.length }); res.end(EMBED.raw); });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const catalog = [{ id: 'emb', kind: 'embed', filename: 'dl-embed.gguf', displayName: 'emb', sizeBytes: EMBED.raw.length, ramGB: 0.1, url: `http://127.0.0.1:${server.address().port}/e.gguf` }];
  const t = await startTestApp({ ctx: 4096, catalog });
  try {
    const chat = (await t.call('save_chat', { name: 'D', model: null, messages: [] })).id;
    const r = await t.call('add_document', { chatId: chat, filename: 'k.txt', content: TEXT });
    assert.equal(r.indexing, false);
    await t.call('download_start', { ids: ['emb'] });
    await until(async () => { const d = (await t.call('document_library'))[0]; return d.embedded === d.chunks && d.chunks > 0; }, 30000);
  } finally { await t.close(); server.close(); }
});
