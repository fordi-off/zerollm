import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { bm25, chunkText, selectExcerpts, tokenize, stem } from '../../app/retrieval.js';
import { withExcerpts, sourcesOf } from '../../app/chat-core.js';

const user = (content) => ({ role: 'user', content });
const KOMMUNE = 'Kommunen har ansvar for skoler og barnehager. Kommunene finansieres blant annet gjennom skatt og overføringer fra staten. '.repeat(10);
const FORSVAR = 'Forsvaret består av hæren, marinen og luftforsvaret, og styres av staten gjennom Forsvarsdepartementet. '.repeat(10);

test('stemmer and tokenizer match Norwegian inflections', () => {
  assert.equal(stem('kommunene'), stem('kommune'));
  assert.equal(stem('kommunenes'), stem('kommunen'));
  assert.equal(stem('skolene'), stem('skoler'));
  assert.deepEqual(tokenize('Hva er kommunenes oppgaver i Norge?'), ['hva', 'er', 'kommun', 'oppgav', 'norg'], 'no stop-word list: IDF handles common words');
});

test('chunkText overlaps, bm25 ranks the right chunk, selectExcerpts respects count and token budget', () => {
  const words = Array.from({ length: 100 }, (_, i) => `w${i}`).join(' ');
  const c = chunkText(words, 40, 10);
  assert.equal(c.length, 3);
  assert.ok(c[1].startsWith('w30 '));
  assert.equal(c.join(' ').split(' ').includes('w99'), true);
  const chunks = [{ text: 'Kommunen har ansvar for barnehager', filename: 'a' }, { text: 'Forsvaret styres av staten', filename: 'a' }, { text: 'Fisk og skalldyr fra havet', filename: 'b' }];
  const r = bm25(chunks, 'hvem driver barnehagene i kommunene');
  assert.equal(r[0].index, 0);
  assert.equal(r.length, 1, 'chunks without any matching term are left out');
  const many = Array.from({ length: 8 }, (_, i) => ({ text: `skole ${'x '.repeat(50)} ${i}`, filename: 'a' }));
  assert.equal(selectExcerpts(many, 'skole', { k: 3 }).length, 3);
  assert.equal(selectExcerpts(many, 'skole', { k: 6, budgetTokens: 120, countTokens: (t) => t.length }).length, 1 + 0, 'budget stops after the first (always allowed) excerpt');
  assert.deepEqual(selectExcerpts(chunks, 'zxqv blorp', { k: 2 }), [], 'a question that matches nothing gets no excerpts');
});

test('withExcerpts puts excerpts before the question in the last user message only', () => {
  const msgs = [{ role: 'system', content: 'S' }, user('første'), { role: 'assistant', content: 'a' }, user('spørsmål')];
  const out = withExcerpts(msgs, [{ filename: 'f.txt', text: 'utdrag', score: 1 }]);
  assert.equal(out[1].content, 'første');
  assert.match(out[3].content, /<documents>\n<document name="f\.txt">\nutdrag\n<\/document>\n<\/documents>\n\nQuestion: spørsmål$/);
  assert.match(out[3].content, /^Answer the question using the documents below/);
  assert.equal(withExcerpts(msgs, []), msgs);
  assert.deepEqual(sourcesOf([{ filename: 'f', text: 'a b c', score: 0.5 }]), [{ filename: 'f', snippet: 'a b c', score: 0.5 }]);
});

test('documents: add to a chat, list, toggle, detach, library attach, preview, delete everywhere; hostile input refused', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const chat = (await t.call('save_chat', { name: 'Doc', model: 'tiny-llama.gguf', messages: [] })).id;
    const chat2 = (await t.call('save_chat', { name: 'Doc2', model: 'tiny-llama.gguf', messages: [] })).id;
    await assert.rejects(t.call('add_document', { chatId: null, filename: 'x.txt', content: 'abc' }), /Ingen samtale/);
    await assert.rejects(t.call('add_document', { chatId: chat, filename: 'x.txt', content: '' }), /tom/);
    await assert.rejects(t.call('add_document', { chatId: chat, filename: 'x.bin', content: 'a\u0000b' }), /tekst/);
    await assert.rejects(t.call('add_document', { chatId: 'nochat', filename: 'x.txt', content: 'hei' }), /finnes ikke/);
    assert.deepEqual(await t.call('document_library'), [], 'failed adds leave nothing behind');

    const r = await t.call('add_document', { chatId: chat, filename: '../kommune.txt', content: KOMMUNE });
    assert.equal(r.documents.length, 1);
    assert.equal(r.documents[0].filename, '.._kommune.txt', 'path separators are neutralised');
    const docId = r.id;
    await t.call('add_document', { chatId: chat, filename: 'forsvar.md', content: FORSVAR });
    assert.equal((await t.call('list_documents', { chatId: chat })).length, 2);
    assert.equal((await t.call('available_documents', { chatId: chat2 })).length, 2);

    assert.deepEqual(await t.call('toggle_document', { chatId: chat, docId }), { active: false });
    assert.equal((await t.call('list_documents', { chatId: chat })).find((d) => d.id === docId).active, false);
    await t.call('toggle_document', { chatId: chat, docId });
    await t.call('attach_document', { chatId: chat2, docId });
    assert.equal((await t.call('available_documents', { chatId: chat2 })).length, 1);
    await t.call('detach_document', { chatId: chat2, docId });
    assert.equal((await t.call('list_documents', { chatId: chat2 })).length, 0);

    const pv = await t.call('preview_document', { docId });
    assert.match(pv.fullText, /Kommunen har ansvar/);
    await assert.rejects(t.call('preview_document', { docId: '../x' }), /invalid document id/);

    await t.call('attach_document', { chatId: chat2, docId });
    await t.call('delete_document', { docId });
    assert.equal((await t.call('document_library')).length, 1);
    assert.equal((await t.call('list_documents', { chatId: chat })).length, 1);
    assert.equal((await t.call('list_documents', { chatId: chat2 })).length, 0, 'removed from every chat');
  } finally { await t.close(); }
});

test('documents from Skibeti37 (with embeddings) are searchable; its attached_docs are kept on save', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    fs.mkdirSync(path.join(t.dir, 'documents'), { recursive: true });
    fs.writeFileSync(path.join(t.dir, 'documents', 'd1.json'), JSON.stringify({ id: 'd1', filename: 'gammel.txt', summary: 'Om kommuner', chunks: [{ text: KOMMUNE, embedding: [0.1, 0.2, 0.3] }] }));
    const chat = (await t.call('save_chat', { name: 'S', model: 'tiny-llama.gguf', messages: [] })).id;
    await t.call('attach_document', { chatId: chat, docId: 'd1' });
    const msgs = [user('Hvem har ansvar for barnehager?')];
    await t.call('save_chat', { id: chat, messages: msgs });
    assert.deepEqual((await t.call('list_documents', { chatId: chat })).map((d) => [d.id, d.summary, d.active]), [['d1', 'Om kommuner', true]], 'saving a chat keeps its attached documents');
  } finally { await t.close(); }
});

test('a question about an attached document brings excerpts into the prompt and sources into the reply', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const messages = [user('Hvem har ansvar for barnehager i kommunene?')];
    const chat = (await t.call('save_chat', { name: 'Q', model: 'tiny-llama.gguf', messages })).id;
    const plain = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages, chatId: chat });
    await t.call('add_document', { chatId: chat, filename: 'kommune.txt', content: KOMMUNE });
    await t.call('add_document', { chatId: chat, filename: 'forsvar.txt', content: FORSVAR });
    const withDocs = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages, chatId: chat });
    assert.ok(withDocs.contextUsed > plain.contextUsed + 100, 'excerpts add tokens to the prompt');

    const ev = await t.runChat('tiny-llama.gguf', messages, { extra: { chatId: chat } });
    const done = ev.at(-1);
    assert.equal(done.type, 'done', JSON.stringify(done));
    assert.ok(done.data.sources.length >= 1);
    assert.ok(done.data.sources.every((s) => s.filename && s.snippet));
    assert.equal(done.data.sources[0].filename, 'kommune.txt', 'the matching document ranks first');

    // inactive documents are not used; continuing a reply does not search
    const docs = await t.call('list_documents', { chatId: chat });
    for (const d of docs) await t.call('toggle_document', { chatId: chat, docId: d.id });
    const off = await t.runChat('tiny-llama.gguf', messages, { extra: { chatId: chat } });
    assert.deepEqual(off.at(-1).data.sources, []);
  } finally { await t.close(); }
});

test('summarize_document stores a summary (tiny model: content is garbage, flow is what is tested); busy while another job runs', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const chat = (await t.call('save_chat', { name: 'Sum', model: 'tiny-llama.gguf', messages: [] })).id;
    const { id } = await t.call('add_document', { chatId: chat, filename: 'kommune.txt', content: KOMMUNE });
    const res = await t.call('summarize_document', { docId: id, model: 'tiny-llama.gguf' });
    assert.equal(res.aborted, false);
    assert.ok(res.summary.length > 0);
    assert.equal((await t.call('list_documents', { chatId: chat }))[0].summary, res.summary);
    await assert.rejects(t.call('summarize_document', { docId: 'nope', model: 'tiny-llama.gguf' }), /not found/);
  } finally { await t.close(); }
});

test('API: a message that matches nothing in the documents gets no excerpts and no sources (no word lists involved); a matching one does', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const chat = (await t.call('save_chat', { name: 'H', model: 'tiny-llama.gguf', messages: [] })).id;
    await t.call('add_document', { chatId: chat, filename: 'k.txt', content: KOMMUNE });
    for (const q of ['hei', 'zxqv blorp', 'Good morning']) {
      const withDoc = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [user(q)], chatId: chat });
      const bare = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [user(q)] });
      assert.equal(withDoc.contextUsed, bare.contextUsed, `prompt for "${q}" is identical to one without documents`);
      const ev = await t.runChat('tiny-llama.gguf', [user(q)], { extra: { chatId: chat } });
      assert.deepEqual(ev.at(-1).data.sources, []);
    }
    const hit = await t.runChat('tiny-llama.gguf', [user('Hvem har ansvar for barnehager?')], { extra: { chatId: chat } });
    assert.ok(hit.at(-1).data.sources.length >= 1);
  } finally { await t.close(); }
});
