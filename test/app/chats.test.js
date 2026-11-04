import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';

// A file exactly as Skibeti37 writes it (system message first, extras on messages, attached_docs, epoch seconds).
const skibeti = {
  id: '3f2b1c0e-1111-4222-8333-444455556666', name: 'Gammel samtale', model: 'tiny-llama.gguf',
  messages: [
    { role: 'system', content: 'Skibeti system prompt' },
    { role: 'user', content: 'Hei', commands: ['humanize'] },
    { role: 'assistant', content: 'Hei på deg!', sources: [{ doc: 'a.txt' }] },
  ],
  attached_docs: { 'doc-1': { enabled: true } }, updated_at: 1760000000.5,
};

test('chats from Skibeti37 are listed and opened; saving keeps its extras, system message and documents', async () => {
  const t = await startTestApp();
  try {
    const dir = path.join(t.dir, 'chats');
    fs.mkdirSync(path.join(dir, '_context_summaries'), { recursive: true });
    fs.writeFileSync(path.join(dir, `${skibeti.id}.json`), JSON.stringify(skibeti));
    fs.writeFileSync(path.join(dir, '_context_summaries', `${skibeti.id}.json`), '{"summary":"x","summarized_count":2}');
    fs.writeFileSync(path.join(dir, 'broken.json'), '{not json');
    fs.writeFileSync(path.join(dir, 'other.json'), JSON.stringify({ id: 'mismatch' }));

    const list = await t.call('list_chats');
    assert.deepEqual(list.map((c) => c.id), [skibeti.id]);
    assert.equal(list[0].name, 'Gammel samtale');
    assert.equal(list[0].model, 'tiny-llama.gguf');

    const chat = await t.call('load_chat', { id: skibeti.id });
    assert.deepEqual(chat.messages.map((m) => m.role), ['user', 'assistant'], 'system message is not shown');
    assert.deepEqual(chat.messages[0].commands, ['humanize']);

    chat.messages.push({ role: 'user', content: 'Mer?' });
    await t.call('save_chat', { id: skibeti.id, name: chat.name, model: chat.model, messages: chat.messages });
    const raw = JSON.parse(fs.readFileSync(path.join(dir, `${skibeti.id}.json`), 'utf8'));
    assert.equal(raw.messages[0].content, 'Skibeti system prompt');
    assert.equal(raw.messages.length, 4);
    assert.deepEqual(raw.messages[2].sources, [{ doc: 'a.txt' }]);
    assert.deepEqual(raw.attached_docs, skibeti.attached_docs);
    assert.ok(raw.updated_at > 1.7e9 && raw.updated_at < 1e11, 'epoch seconds like Skibeti37');

    await t.call('delete_chat', { id: skibeti.id });
    assert.equal(fs.existsSync(path.join(dir, `${skibeti.id}.json`)), false);
    assert.equal(fs.existsSync(path.join(dir, '_context_summaries', `${skibeti.id}.json`)), false);
  } finally { await t.close(); }
});

test('create, rename, order by recency, delete; hostile ids and bad messages are refused', async () => {
  const t = await startTestApp();
  try {
    const a = (await t.call('save_chat', { name: 'A', model: 'tiny-llama.gguf', messages: [{ role: 'user', content: 'a' }] })).id;
    await new Promise((r) => setTimeout(r, 20));
    const b = (await t.call('save_chat', { name: 'B', model: null, messages: [{ role: 'user', content: 'b' }, { role: 'assistant', content: 'B!' }] })).id;
    assert.deepEqual((await t.call('list_chats')).map((c) => c.name), ['B', 'A']);
    const first = JSON.parse(fs.readFileSync(path.join(t.dir, 'chats', `${a}.json`), 'utf8'));
    assert.equal(first.messages[0].role, 'system', 'new files get a system message first, like Skibeti37');
    await t.call('rename_chat', { id: a, name: '  Nytt navn  ' });
    await new Promise((r) => setTimeout(r, 20));
    await t.call('save_chat', { id: a, messages: [{ role: 'user', content: 'a2' }] }); // name kept when not given
    const list = await t.call('list_chats');
    assert.equal(list[0].id, a);
    assert.equal(list[0].name, 'Nytt navn');
    assert.equal((await t.call('load_chat', { id: 'does-not-exist' })), null);

    for (const id of ['../x', '..\\x', 'a/b', '', 'a'.repeat(65)]) {
      await assert.rejects(t.call('load_chat', { id }), /invalid chat id/);
      await assert.rejects(t.call('delete_chat', { id }), /invalid chat id/);
    }
    await assert.rejects(t.call('save_chat', { messages: [{ role: 'system', content: 'x' }] }), /invalid message/);
    await assert.rejects(t.call('save_chat', { messages: 'no' }), /list/);
    await t.call('delete_chat', { id: b });
    assert.deepEqual((await t.call('list_chats')).map((c) => c.id), [a]);
  } finally { await t.close(); }
});
