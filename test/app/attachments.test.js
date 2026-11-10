import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { buildMessages, renderWithContext } from '../../app/chat-core.js';

const user = (content, extra = {}) => ({ role: 'user', content, ...extra });
const DOC = 'Kommunen har ansvar for skoler og barnehager. Kommunene finansieres gjennom skatt. '.repeat(8);

test('an attached document goes into the message whole, in front of the question', () => {
  const m = buildMessages([user('Hva står det?', { context: [{ filename: 'k.txt', text: 'Innholdet.' }] })]);
  assert.match(m[1].content, /^The user attached the document\(s\) below/);
  assert.match(m[1].content, /<document name="k\.txt">\nInnholdet\.\n<\/document>/);
  assert.ok(m[1].content.endsWith('Question: Hva står det?'));
  assert.equal(renderWithContext('x', undefined), 'x');
});

test('API: the document counts towards the prompt, is answered from (done event ok), and is saved with the chat', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const plain = [user('Hva står det i filen?')];
    const withDoc = [user('Hva står det i filen?', { context: [{ filename: 'k.txt', text: DOC }] })];
    const a = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: plain });
    const b = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: withDoc });
    assert.ok(b.contextUsed > a.contextUsed + DOC.length * 0.5, 'the whole text is in the prompt');
    assert.equal(a.docTokens, 0); assert.ok(b.docTokens > DOC.length * 0.5 && b.docTokens < b.docLimit, 'estimate says how big the document is and how big it may be');
    assert.equal(b.docLimit, Math.floor(0.8 * (4096 - 2000 - 64)));
    const ev = await t.runChat('tiny-llama.gguf', withDoc);
    assert.equal(ev.at(-1).type, 'done');
    const id = (await t.call('save_chat', { name: 'D', model: 'tiny-llama.gguf', messages: [...withDoc, { role: 'assistant', content: 'Svar.' }] })).id;
    const loaded = await t.call('load_chat', { id });
    assert.equal(loaded.messages[0].context[0].filename, 'k.txt');
    assert.equal(loaded.messages[0].context[0].text, DOC);
    // follow-up: the stored document is still in the prompt, the new message adds only its own text
    const follow = [...loaded.messages, user('Og hvem betaler?')];
    const c = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: follow });
    assert.ok(c.contextUsed > b.contextUsed && c.contextUsed - b.contextUsed < 120);
  } finally { await t.close(); }
});

test('API: a document that cannot fit the context window is refused with a clear Norwegian message', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const huge = 'ord '.repeat(20000);
    const est = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [user('Oppsummer', { context: [{ filename: 'stor.txt', text: huge }] })] });
    assert.ok(est.docTokens > est.docLimit, 'the page can see before sending that it will not fit');
    const ev = await t.runChat('tiny-llama.gguf', [user('Oppsummer', { context: [{ filename: 'stor.txt', text: huge }] })]);
    assert.equal(ev.at(-1).type, 'error');
    assert.match(ev.at(-1).data.message, /Dokumentet er for stort for kontekstvinduet/);
  } finally { await t.close(); }
});

test('the old document library, search and indexing API is gone; chats keep a Skibeti37 chat\'s attached_docs untouched', async () => {
  const t = await startTestApp();
  try {
    for (const m of ['add_document', 'list_documents', 'document_library', 'index_documents', 'embedding_status', 'summarize_document', 'set_doc_chunks']) {
      await assert.rejects(t.call(m, {}), (e) => e.status === 404, m);
    }
    const id = '3f2b1c0e-1111-4222-8333-444455556666';
    fs.mkdirSync(path.join(t.dir, 'chats'), { recursive: true });
    fs.writeFileSync(path.join(t.dir, 'chats', `${id}.json`), JSON.stringify({ id, name: 'x', model: null, messages: [{ role: 'system', content: 's' }], attached_docs: { d1: true }, updated_at: 1 }));
    await t.call('save_chat', { id, messages: [user('hei')] });
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(t.dir, 'chats', `${id}.json`), 'utf8')).attached_docs, { d1: true });
    assert.equal(fs.existsSync(path.join(t.dir, 'documents')), false, 'no documents folder is created any more');
  } finally { await t.close(); }
});

test('start-up loading: the last used model is loaded in the background (setting, on by default); off means nothing is loaded', async () => {
  const first = await startTestApp({ ctx: 4096 });
  const dir = first.dir;
  try {
    assert.equal((await first.call('state')).settings.preload, true);
    await first.call('set_last_model', { filename: 'tiny-llama.gguf' });
  } finally { await first.close({ keep: true }); }
  const on = await startTestApp({ ctx: 4096, dir, preload: true });
  try {
    for (let i = 0; i < 200 && (await on.call('state')).loaded !== 'tiny-llama.gguf'; i++) await new Promise((r) => setTimeout(r, 50));
    assert.equal((await on.call('state')).loaded, 'tiny-llama.gguf');
    const ev = await on.runChat('tiny-llama.gguf', [user('Hei')]);
    assert.equal(ev.at(-1).type, 'done');
    await on.call('set_preload', { value: false });
    assert.equal((await on.call('state')).settings.preload, false);
  } finally { await on.close({ keep: true }); }
  const off = await startTestApp({ ctx: 4096, dir, preload: true });
  try {
    await new Promise((r) => setTimeout(r, 600));
    assert.equal((await off.call('state')).loaded, null, 'setting off: nothing is loaded at start');
  } finally { await off.close(); }
});
