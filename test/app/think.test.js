import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../../tools/make-tiny-gguf.js';
import { loadTokenizer } from '../../src/index.js';
import { buildMessages, THINK_SAMPLING, SAMPLING } from '../../app/chat-core.js';
import { splitThinking, fmtDuration } from '../../app/public/js/thinking.js';
import { startTestApp } from './helpers.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zc-think-'));
const file = path.join(dir, 'q.gguf'); fs.writeFileSync(file, makeTinyModel({ arch: 'qwen3', quant: 'q8_0', seed: 4, ctx: 4096 }).buffer);
const tools = loadTokenizer(file);
process.on('exit', () => fs.rmSync(dir, { recursive: true, force: true }));
const common = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };
const U = (content) => ({ role: 'user', content }); const A = (content) => ({ role: 'assistant', content });

test('Qwen3 non-thinking chats: turn N+1 starts with the whole prompt of turn N plus the empty think block the model wrote (no re-reading of replies)', () => {
  const p1 = tools.promptIds(buildMessages([U('Hei, hvem er du?')]));                    // ends with the assistant header + empty <think> block
  const reply = 'Jeg er en assistent.';
  // what the model's memory holds after turn 1: the prompt + its reply (+ <|im_end|> is not stored); turn 2 renders the reply from history
  const p2 = tools.promptIds(buildMessages([U('Hei, hvem er du?'), A(reply), U('Takk!')]));
  assert.equal(common(p1, p2), p1.length, 'everything read for turn 1 (incl. the empty think block) is still valid in turn 2');
  const p3 = tools.promptIds(buildMessages([U('Hei, hvem er du?'), A(reply), U('Takk!'), A('Bare hyggelig.'), U('Hvorfor?')]));
  assert.equal(common(p2, p3), p2.length);
});

test('thinking mode: the prompt ends at the assistant header (the model writes <think> itself); history carries answers only', () => {
  const off = tools.promptIds(buildMessages([U('Hei')]), { think: false });
  const on = tools.promptIds(buildMessages([U('Hei')]), { think: true });
  assert.ok(off.length > on.length, 'no pre-filled empty think block when thinking');
  assert.equal(common(off, on), on.length);
  assert.equal(THINK_SAMPLING.repeatPenalty, 1); assert.equal(THINK_SAMPLING.presencePenalty, 0);
  assert.ok(SAMPLING.repeatPenalty > 1, 'normal chat keeps its penalties');
});

test('app: the think switch is honoured for Qwen3 models and ignored for others; the meter counts the right prompt', async () => {
  const t = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-qwen3.gguf', arch: 'qwen3' }, { name: 'tiny-llama.gguf', arch: 'llama' }] });
  try {
    const st = await t.call('state');
    assert.deepEqual(st.models.map((m) => [m.filename, m.arch]), [['tiny-llama.gguf', 'llama'], ['tiny-qwen3.gguf', 'qwen3']]);
    const msgs = [U('Hei')];
    const off = await t.call('estimate_context', { model: 'tiny-qwen3.gguf', messages: msgs, think: false });
    const on = await t.call('estimate_context', { model: 'tiny-qwen3.gguf', messages: msgs, think: true });
    assert.ok(on.contextUsed < off.contextUsed, 'thinking prompt has no empty think block');
    const llamaOff = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: msgs, think: false });
    const llamaOn = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: msgs, think: true });
    assert.equal(llamaOn.contextUsed, llamaOff.contextUsed, 'Llama has no thinking mode');
    const ev = await t.runChat('tiny-qwen3.gguf', msgs, { extra: { think: true } });
    assert.equal(ev.at(-1).type, 'done'); assert.equal(ev.at(-1).data.thinking, true);
    const plain = await t.runChat('tiny-qwen3.gguf', msgs, { extra: { think: false } });
    assert.equal(plain.at(-1).data.thinking, false);
    const llama = await t.runChat('tiny-llama.gguf', msgs, { extra: { think: true } });
    assert.equal(llama.at(-1).data.thinking, false);
  } finally { await t.close(); }
});

test('splitThinking: streaming states, closed block, no block, leftovers', () => {
  assert.deepEqual(splitThinking('Hei der'), { thinking: '', answer: 'Hei der', state: 'none' });
  assert.deepEqual(splitThinking(''), { thinking: '', answer: '', state: 'none' });
  assert.equal(splitThinking('<thi').state, 'open', 'the tag itself is still arriving');
  assert.deepEqual(splitThinking('<think>\nJeg må regne'), { thinking: 'Jeg må regne', answer: '', state: 'open' });
  assert.deepEqual(splitThinking('<think>\nJeg regner.\n</think>\n\nSvaret er 42.'), { thinking: 'Jeg regner.', answer: 'Svaret er 42.', state: 'done' });
  assert.deepEqual(splitThinking('<think>\nx\n</think>'), { thinking: 'x', answer: '', state: 'done' });
  assert.deepEqual(splitThinking('tenker uten åpningstagg\n</think>\nSvar'), { thinking: 'tenker uten åpningstagg', answer: 'Svar', state: 'done' });
  assert.equal(fmtDuration(134000), '2:14'); assert.equal(fmtDuration(5000), '0:05'); assert.equal(fmtDuration(-1), '0:00');
});
