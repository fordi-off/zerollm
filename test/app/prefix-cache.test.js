import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { loadTokenizer } from '../../src/index.js';
import { stablePrefixIds, buildMessages } from '../../app/chat-core.js';

const user = (content) => ({ role: 'user', content });

test('the stable prefix is the system prompt and is shared by every prompt', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const tools = loadTokenizer(path.join(t.dir, 'models', 'tiny-llama.gguf'));
    const prefix = stablePrefixIds(tools);
    assert.ok(prefix.length > 100, `system prompt is long enough to be worth caching (${prefix.length})`);
    for (const text of ['Hei, kan du hjelpe meg?', 'Please help me', '12345', 'Dette er en lengre setning på norsk, med æ, ø og å.']) {
      const ids = tools.promptIds(buildMessages([user(text)]));
      assert.deepEqual(ids.slice(0, prefix.length), prefix, `prefix is a real prefix of the prompt for "${text}"`);
    }
  } finally { await t.close(); }
});

test('system prompt state is saved on first use and restored by a new process', async () => {
  const first = await startTestApp({ ctx: 4096 });
  const dir = first.dir;
  let savedTokens;
  try {
    const ev = await first.runChat('tiny-llama.gguf', [user('Hei')]);
    const done = ev.at(-1).data;
    assert.equal(ev.at(-1).type, 'done');
    assert.equal(done.prefixCache, 'saved');
    assert.ok(done.cachedTokens >= 100, 'the system prompt counted as cached tokens');
    savedTokens = done.cachedTokens;
    assert.ok(fs.readdirSync(path.join(dir, 'cache')).some((f) => f.endsWith('.kv')));
    // a second message in the same session reuses the live state, no priming needed
    const again = await first.runChat('tiny-llama.gguf', [user('Hei'), { role: 'assistant', content: 'Hei!' }, user('Hvordan går det?')]);
    assert.equal(again.at(-1).data.prefixCache, 'off');
    assert.ok(again.at(-1).data.cachedTokens >= savedTokens);
  } finally { await first.close({ keep: true }); }

  const second = await startTestApp({ ctx: 4096, dir });
  try {
    const ev = await second.runChat('tiny-llama.gguf', [user('Tell me something')]);
    const done = ev.at(-1).data;
    assert.equal(ev.at(-1).type, 'done');
    assert.equal(done.prefixCache, 'hit');
    assert.ok(done.cachedTokens >= 100);
    assert.ok(done.generated > 0);
  } finally { await second.close(); }
});
