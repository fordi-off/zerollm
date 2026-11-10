import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../../tools/make-tiny-gguf.js';
import { loadTokenizer } from '../../src/index.js';
import { buildMessages } from '../../app/chat-core.js';

// The model only keeps what it already read (its KV cache) for an UNCHANGED prompt prefix. If anything early in the prompt
// changes between two turns, the whole conversation after that point is read again ("Leser teksten..." every time).
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zc-append-'));
const file = path.join(dir, 'm.gguf'); fs.writeFileSync(file, makeTinyModel({ arch: 'llama', quant: 'q8_0', seed: 4, ctx: 4096 }).buffer);
const tools = loadTokenizer(file);
process.on('exit', () => fs.rmSync(dir, { recursive: true, force: true }));
const common = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };
const U = (content, extra = {}) => ({ role: 'user', content, ...extra }); const A = (content) => ({ role: 'assistant', content });
const ids = (h) => tools.promptIds(buildMessages(h));

test('turn N+1 starts with the complete prompt of turn N, whatever language or content the new message has', () => {
  const q1 = 'Hvem har ansvar for barnehager i kommunene?'; const a1 = 'Kommunen har ansvar.'; const a2 = 'Staten og kommunen.';
  for (const q2 of ['Og hvem betaler for det?', 'ok', 'And who pays for it?', 'Hva er 2+3? Regn ut \\frac{1}{2}', 'x'.repeat(5)]) {
    const p1 = ids([U(q1)]);
    const p2 = ids([U(q1), A(a1), U(q2)]);
    assert.equal(common(p1, p2), p1.length, `"${q2}": everything the model read for turn 1 is still valid`);
    const p3 = ids([U(q1), A(a1), U(q2), A(a2), U('Takk!')]);
    assert.equal(common(p2, p3), p2.length, 'and again for turn 3');
  }
});

test('an attached document is stored with its question: later turns reuse it unchanged (read once, kept in memory) and a follow-up adds only its own few tokens', () => {
  const ctx1 = [{ filename: 'k.txt', text: 'Kommunen har ansvar for skoler og barnehager. '.repeat(20), score: 1 }];
  const ctx2 = [{ filename: 'k.txt', text: 'Kommunene finansieres gjennom skatt. '.repeat(20), score: 1 }];
  const turn1 = [U('Hvem har ansvar for barnehager?', { context: ctx1 })];
  const turn2 = [...turn1, A('Kommunen.'), U('Hvem betaler?', { context: ctx2 })];
  const turn3 = [...turn2, A('Skatt.'), U('Takk, og hva mer?', { context: [] })];
  const p1 = ids(turn1); const p2 = ids(turn2); const p3 = ids(turn3);
  assert.equal(common(p1, p2), p1.length);
  assert.equal(common(p2, p3), p2.length);
  assert.ok(p2.length - p1.length > 200, 'the new excerpts are the only large addition');
  assert.ok(p3.length - p2.length < 120, 'a follow-up without new excerpts adds only its own few tokens');
});
