import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { startTestApp } from './helpers.js';
import { cleanTitle, titleMessages, TITLE_SYSTEM, TITLE_FOLLOWUP, TITLE_MAX_TOKENS } from '../../app/titles.js';
import { buildMessages } from '../../app/chat-core.js';

// The synthetic test models only produce noise, so these tests check the MECHANISM (when a title job runs, locking, abort, sanitising,
// the race with the page's saves, what the chat model's memory looks like afterwards) - not the quality of titles. Where a known answer
// is needed, the generation step is replaced (`titleRun`); everything around it (engine choice, locks, writes, events) is the real code.

const U = (content, extra = {}) => ({ role: 'user', content, ...extra });
const A = (content) => ({ role: 'assistant', content });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, what, ms = 10000) {
  const t0 = Date.now();
  for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t0 > ms) throw new Error(`timeout waiting for ${what}`); await sleep(10); }
}
const readChat = (t, id) => JSON.parse(fs.readFileSync(path.join(t.dir, 'chats', `${id}.json`), 'utf8'));
const logText = (t) => { try { return fs.readFileSync(path.join(t.dir, 'app.log'), 'utf8'); } catch { return ''; } };
const nameOf = async (t, id) => (await t.call('list_chats')).find((c) => c.id === id)?.name;

/** Listen to the SSE channel; returns the collected events. */
async function listen(t) {
  const ctl = new AbortController(); const events = []; let ready; const r = new Promise((res) => { ready = res; });
  (async () => { try { for await (const ev of t.events(ctl.signal)) { if (ev.type === 'hello') ready(); else events.push(ev); } } catch (e) { if (e.name !== 'AbortError') throw e; } })();
  await r;
  return { events, stop: () => ctl.abort() };
}

/** A generation step with a known answer; records what it was asked. `answer` may be a function (engine, messages, abort, calls) => string. */
function fakeRun(answer = 'Hovedstaden i Norge') {
  const calls = [];
  const run = async (engine, messages, abort, stats) => {
    const call = { engine, messages, abort, closedWhenDone: null };
    calls.push(call);
    return typeof answer === 'function' ? answer(engine, messages, abort, calls, stats) : answer;
  };
  run.calls = calls;
  return run;
}
/** A generation step that "runs" until it is aborted (or released): lets a test act while a title job is in progress. */
function gatedRun(answer = 'Tittel fra modellen') {
  const gate = { release: null, started: null, finishedAt: 0, sawAbort: false, calls: 0 };
  gate.startedP = new Promise((r) => { gate.started = r; });
  const run = async (engine, messages, abort) => {
    gate.calls++; gate.started();
    await new Promise((r) => { gate.release = r; const iv = setInterval(() => { if (abort.aborted) { gate.sawAbort = true; clearInterval(iv); r(); } }, 5); gate.release = () => { clearInterval(iv); r(); }; });
    gate.finishedAt = performance.now();
    return abort.aborted ? '' : answer;
  };
  return { run, gate };
}

const UTIL = [{ name: 'tiny-llama.gguf', arch: 'llama' }, { name: 'tiny-util.gguf', arch: 'llama' }];
async function withUtility(titleRun, fn, opts = {}) {
  const t = await startTestApp({ ctx: 1024, models: UTIL, titleRun, ...opts });
  try { await t.call('set_utility_model', { filename: 'tiny-util.gguf' }); await fn(t); } finally { await t.close(); }
}
/** Save a first exchange the way the page does (name = the placeholder it made from the first message). */
const saveExchange = (t, { id, name = 'Hva er hovedstaden i Norge?', user = 'Hva er hovedstaden i Norge?', reply = 'Oslo er hovedstaden i Norge.', model = 'tiny-llama.gguf' } = {}) =>
  t.call('save_chat', { id, name, model, messages: [U(user), ...(reply === null ? [] : [A(reply)])] });

// ---------------------------------------------------------------- sanitising

test('cleanTitle: messy model output in several languages becomes a plain short title', () => {
  const ok = [
    ['Kommunale barnehager og ansvar', 'Kommunale barnehager og ansvar'],
    ['"Kommunale barnehager og ansvar"', 'Kommunale barnehager og ansvar'],
    ['«Åse og øya på Ærø»', 'Åse og øya på Ærø'],
    ['„Käse und Brötchen“', 'Käse und Brötchen'],
    ['Tittel: Hvorfor er himmelen blå?', 'Hvorfor er himmelen blå'],
    ['**Title:** Planning a trip to Bergen.', 'Planning a trip to Bergen'],
    ['Titre : Un voyage à Genève !', 'Un voyage à Genève'],
    ['# Oppskrift på rømmegrøt\n\nHer er litt mer tekst som ikke hører til.', 'Oppskrift på rømmegrøt'],
    ['\n\n   Fjordene   i\tNorge  \n', 'Fjordene i Norge'],
    ['Here is a title:\nBergen i regnet', 'Bergen i regnet'],
    ['```\nTrening og kosthold\n```', 'Trening og kosthold'],
    ['[Skattemelding for 2024]', 'Skattemelding for 2024'],
    ['Skattemelding (2024)', 'Skattemelding (2024)'],
    ['(Oversikt) over (alt)', '(Oversikt) over (alt)'],
    ['"Hva er en monade?"', 'Hva er en monade'],
    ["'Kort og godt'.", 'Kort og godt'],
    ["Ola's bil og Kari's hytte", "Ola's bil og Kari's hytte"],
    ['- Handleliste til helgen', 'Handleliste til helgen'],
    ['3. verdenskrig i korte trekk', '3. verdenskrig i korte trekk'],
    ['Tittel: "Bærekraftig energi i Norge"...', 'Bærekraftig energi i Norge'],
    ['你好，世界的故事。', '你好，世界的故事'],
    ['「東京の天気」', '東京の天気'],
    ['Как приготовить борщ?', 'Как приготовить борщ'],
    ['<think>\nHmm, la meg tenke litt\n</think>\nReiseplan til Lofoten', 'Reiseplan til Lofoten'],
    ['__Søvn og helse__', 'Søvn og helse'],
  ];
  for (const [raw, want] of ok) assert.equal(cleanTitle(raw), want, JSON.stringify(raw));
  const long = cleanTitle('Dette er en veldig lang tittel som fortsetter og fortsetter og fortsetter uten noen ende i sikte');
  assert.ok(long.length <= 60 && long.length >= 30, long);
  assert.ok(!/\s$/.test(long) && 'Dette er en veldig lang tittel som fortsetter og fortsetter og fortsetter uten noen ende i sikte'.startsWith(long), 'cut at a word boundary');
  assert.equal(Array.from(cleanTitle(Array.from({ length: 100 }, (_, i) => String.fromCodePoint(0x4e00 + i)).join(''))).length, 60, 'no spaces: hard cut');
  assert.ok(cleanTitle('Æ'.repeat(5) + ' ' + 'Ø'.repeat(70)).length <= 60);
  assert.equal(cleanTitle('Lang tittel her', { maxLen: 8 }), 'Lang');
});

test('cleanTitle: empty, punctuation-only, garbage and degenerate answers are rejected', () => {
  for (const raw of ['', '   ', '\n\n', '...', '???', '"" ""', '😀😀😀', '\x00\x01\x02\x03\x04\x05', '����������', 'aaaaaaaaaaaaaa', 'a', '-----\n=====', 'Hei' + '\x01'.repeat(12), null, undefined, '<think>bare tanker, ingen tittel', '()', '«»'])
    assert.equal(cleanTitle(raw), '', JSON.stringify(raw));
  // what the synthetic model really writes
  assert.equal(cleanTitle('(p�(ԑ@V�(\x1d\x18�[Eb\x19�Z(E\x1cE'), '');
});

test('titleMessages: English system message, first message capped at 600 and reply start at 400 characters', () => {
  const m = titleMessages('å'.repeat(5000), 'ø'.repeat(5000));
  assert.deepEqual(m.map((x) => x.role), ['system', 'user']);
  assert.equal(m[0].content, TITLE_SYSTEM);
  assert.match(TITLE_SYSTEM, /same language as the conversation/); assert.match(TITLE_SYSTEM, /3 to 6 words/); assert.match(TITLE_SYSTEM, /title only/);
  assert.equal((m[1].content.match(/å/g) ?? []).length, 600);
  assert.equal((m[1].content.match(/ø/g) ?? []).length, 400);
  const short = titleMessages('Hei på deg', 'Hei!');
  assert.match(short[1].content, /Hei på deg/); assert.match(short[1].content, /Hei!$/);
  assert.match(TITLE_FOLLOWUP, /not the language of this instruction/);
});

// ---------------------------------------------------------------- the job

test('utility model: a title is written after the first exchange; sidebar order, other fields and the saved file stay intact; event is sent', async () => {
  const run = fakeRun('«Hovedstaden i Norge».');
  await withUtility(run, async (t) => {
    const sse = await listen(t);
    const { id } = await saveExchange(t);
    const before = readChat(t, id);
    assert.equal(before.auto_title, 'pending');
    await waitFor(async () => (await nameOf(t, id)) === 'Hovedstaden i Norge', 'the title');
    const ev = await waitFor(() => sse.events.find((e) => e.type === 'title'), 'title event');
    assert.deepEqual(ev.data, { chatId: id, name: 'Hovedstaden i Norge' });
    const after = readChat(t, id);
    assert.equal(after.auto_title, 'done');
    assert.equal(after.auto_placeholder, 'Hva er hovedstaden i Norge?');
    assert.equal(after.updated_at, before.updated_at, 'a title that arrives later does not move the chat in the sidebar');
    assert.deepEqual(after.messages, before.messages);
    assert.equal(after.model, 'tiny-llama.gguf');
    assert.equal(run.calls.length, 1);
    assert.deepEqual(run.calls[0].messages, titleMessages('Hva er hovedstaden i Norge?', 'Oslo er hovedstaden i Norge.'));
    assert.equal(run.calls[0].engine.closed, true, 'the small model was closed after the job');
    assert.equal((await t.call('state')).loaded, null, 'the chat model was never needed');
    assert.match(logText(t), /titling with tiny-util\.gguf/);
    sse.stop();
  });
});

test('the page may be closed meanwhile: no event listeners, the job still finishes and list_chats shows the title', async () => {
  await withUtility(fakeRun('Reiseplan til Lofoten'), async (t) => {
    const { id } = await saveExchange(t, { user: 'Planlegg en tur', name: 'Planlegg en tur' });
    await waitFor(async () => (await nameOf(t, id)) === 'Reiseplan til Lofoten', 'the title');
  });
});

test('only the first exchange is titled: not when the chat is older / has more exchanges, never twice', async () => {
  const run = fakeRun('Første tittel');
  await withUtility(run, async (t) => {
    // an old chat (no auto_title field), created with several exchanges
    const old = (await t.call('save_chat', { name: 'Gammel', model: 'tiny-llama.gguf', messages: [U('a'), A('b'), U('c'), A('d')] })).id;
    assert.equal(readChat(t, old).auto_title, undefined);
    await t.call('save_chat', { id: old, name: 'Gammel', messages: [U('a'), A('b'), U('c'), A('d'), U('e'), A('f')] });
    // a new chat: titled after its first exchange, then left alone
    const { id } = await saveExchange(t);
    await waitFor(async () => (await nameOf(t, id)) === 'Første tittel', 'the title');
    await t.call('save_chat', { id, name: 'Første tittel', messages: [U('Hva er hovedstaden i Norge?'), A('Oslo.'), U('Og Sverige?'), A('Stockholm.')] });
    await sleep(150);
    assert.equal(run.calls.length, 1, 'one job in total');
    assert.equal(await nameOf(t, id), 'Første tittel');
    assert.equal(await nameOf(t, old), 'Gammel');
    assert.equal(readChat(t, id).auto_title, 'done');
  });
});

test('a chat renamed before its reply arrives is never titled; one renamed while the job runs keeps the user name', async () => {
  const first = fakeRun('Skal ikke brukes');
  await withUtility(first, async (t) => {
    const { id } = await t.call('save_chat', { name: 'Hva er hovedstaden i Norge?', model: 'tiny-llama.gguf', messages: [U('Hva er hovedstaden i Norge?')] });
    await t.call('rename_chat', { id, name: 'Mitt navn' });
    assert.equal(readChat(t, id).auto_title, 'user');
    await saveExchange(t, { id, name: 'Mitt navn' });
    await sleep(150);
    assert.equal(first.calls.length, 0);
    assert.equal(await nameOf(t, id), 'Mitt navn');
  });
  const { run, gate } = gatedRun('Maskintittel');
  await withUtility(run, async (t) => {
    const sse = await listen(t);
    const { id } = await saveExchange(t);
    await gate.startedP;
    await t.call('rename_chat', { id, name: 'Brukerens tittel' }); // while the model is "thinking"
    gate.release();
    await waitFor(() => /dropped/.test(logText(t)), 'the job to notice');
    assert.equal(await nameOf(t, id), 'Brukerens tittel');
    assert.equal(readChat(t, id).auto_title, 'user');
    assert.equal(sse.events.some((e) => e.type === 'title'), false, 'no title event for a dropped title');
    sse.stop();
  });
});

test('a chat that is deleted or whose first message was edited while the job runs gets no title', async () => {
  const { run, gate } = gatedRun('Utdatert tittel');
  await withUtility(run, async (t) => {
    const a = (await saveExchange(t)).id;
    await gate.startedP;
    await t.call('delete_chat', { id: a });
    gate.release();
    await waitFor(() => /dropped/.test(logText(t)), 'the job to notice');
    assert.equal(fs.existsSync(path.join(t.dir, 'chats', `${a}.json`)), false, 'the chat is not brought back');
  });
  const g2 = gatedRun('Utdatert tittel');
  await withUtility(g2.run, async (t) => {
    const { id } = await saveExchange(t);
    await g2.gate.startedP;
    await t.call('save_chat', { id, name: 'Hva er hovedstaden i Norge?', messages: [U('Helt nytt spørsmål om fisk'), A('Torsk.')] }); // edit + resend
    g2.gate.release();
    await waitFor(() => /dropped/.test(logText(t)), 'the job to notice');
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?');
    assert.equal(readChat(t, id).auto_title, 'pending', 'still waiting: the next save starts a new job for the new first message');
  });
});

test('Settings: auto_titles off means no job (and the state says so); switching it off stops a running job; default is on', async () => {
  const run = fakeRun('Burde ikke skje');
  await withUtility(run, async (t) => {
    assert.equal((await t.call('state')).settings.autoTitles, true, 'on by default');
    assert.equal(JSON.parse(fs.readFileSync(path.join(t.dir, 'config.json'), 'utf8')).auto_titles, true);
    assert.deepEqual(await t.call('set_auto_titles', { value: false }), { ok: true, autoTitles: false });
    assert.equal((await t.call('state')).settings.autoTitles, false);
    const { id } = await saveExchange(t);
    await sleep(150);
    assert.equal(run.calls.length, 0);
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?');
    assert.equal(readChat(t, id).auto_title, 'pending', 'it can still be titled later when the setting is turned on again');
    await t.call('set_auto_titles', { value: true });
    await t.call('save_chat', { id, name: 'Hva er hovedstaden i Norge?', messages: [U('Hva er hovedstaden i Norge?'), A('Oslo er hovedstaden i Norge.')] });
    await waitFor(async () => (await nameOf(t, id)) === 'Burde ikke skje', 'the title after switching on');
  });
  const { run: r2, gate } = gatedRun();
  await withUtility(r2, async (t) => {
    const { id } = await saveExchange(t);
    await gate.startedP;
    await t.call('set_auto_titles', { value: false });
    assert.equal(gate.sawAbort, true, 'the running job was aborted');
    await sleep(50);
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?');
  });
});

test('failure is silent: an error in the job is only logged, the chat keeps working, the placeholder stays', async () => {
  let n = 0;
  await withUtility(fakeRun(() => { n++; throw new Error('boom in title'); }), async (t) => {
    const { id } = await saveExchange(t);
    await waitFor(() => /title for .* failed: Error: boom in title/.test(logText(t)), 'the log line');
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?');
    const ev = await t.runChat('tiny-llama.gguf', [U('Hei')]);
    assert.equal(ev.at(-1).type, 'done');
    assert.equal(n, 1);
  });
});

test('an unusable answer (noise) keeps the placeholder and is not tried again', async () => {
  // the synthetic model's real output goes through the real generation step here: it is noise, so the sanitiser rejects it
  const t = await startTestApp({ ctx: 1024, models: UTIL });
  try {
    await t.call('set_utility_model', { filename: 'tiny-util.gguf' });
    const { id } = await saveExchange(t);
    await waitFor(() => readChat(t, id).auto_title === 'failed', 'the job to give up');
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?');
    assert.match(logText(t), /model answer not usable/);
    await saveExchange(t, { id });
    await sleep(100);
    assert.equal((logText(t).match(/title run/g) ?? []).length, 1);
  } finally { await t.close(); }
});

// ---------------------------------------------------------------- engine choice and locking

test('chat-model fallback: nothing is loaded, or the chat has more than one exchange -> skipped quietly (no model is loaded just for a title)', async () => {
  const run = fakeRun('X tittel');
  const t = await startTestApp({ ctx: 1024, titleRun: run });
  try {
    const { id } = await saveExchange(t);
    await waitFor(() => /skipped: no utility model/.test(logText(t)), 'the skip line');
    assert.equal(run.calls.length, 0);
    assert.equal((await t.call('state')).loaded, null, 'the chat model was not loaded for a title');
    assert.equal(readChat(t, id).auto_title, 'pending');
    // loaded, but the chat already moved on to a second exchange: the chat model is not used either
    await t.runChat('tiny-llama.gguf', [U('Hei')]);
    await t.call('save_chat', { id, name: 'Hva er hovedstaden i Norge?', messages: [U('Hva er hovedstaden i Norge?'), A('Oslo.'), U('Og Sverige?'), A('Stockholm.')] });
    await sleep(100);
    assert.equal(run.calls.length, 0);
  } finally { await t.close(); }
});

test('chat-model fallback: continues the real conversation (same system prompt and history) with the title request, think off, short and cool', async () => {
  const run = fakeRun('Hovedstaden i Norge');
  const t = await startTestApp({ ctx: 1024, titleRun: run });
  try {
    const q = [U('Hva er hovedstaden i Norge?')];
    const ev = await t.runChat('tiny-llama.gguf', q);
    const reply = ev.filter((e) => e.type === 'chunk').map((e) => e.data.text).join('');
    const { id } = await saveExchange(t, { reply });
    await waitFor(async () => (await nameOf(t, id)) === 'Hovedstaden i Norge', 'the title');
    const [call] = run.calls;
    assert.equal(call.engine, t.app.host.engine, 'the resident chat model did the job');
    assert.deepEqual(call.messages, [...buildMessages(q), A(reply), { role: 'user', content: TITLE_FOLLOWUP }]);
    assert.match(logText(t), /title run/);
  } finally { await t.close(); }
});

test('the default generation step asks for no thinking, at most 24 tokens, low temperature (Qwen3 gets its empty think block)', async () => {
  const t = await startTestApp({ ctx: 1024, models: [{ name: 'tiny-qwen3.gguf', arch: 'qwen3' }] });
  try {
    await t.runChat('tiny-qwen3.gguf', [U('Hei')]);
    const eng = t.app.host.engine; const seen = [];
    const orig = eng.generate.bind(eng);
    eng.generate = (o) => { seen.push(o); return orig(o); };
    const { id } = await saveExchange(t, { model: 'tiny-qwen3.gguf', user: 'Hei', name: 'Hei', reply: 'Hei der!' });
    await waitFor(() => seen.length > 0, 'the generation call');
    const o = seen[0];
    assert.equal(o.think, false);
    assert.ok(o.maxTokens <= 24 && o.maxTokens === TITLE_MAX_TOKENS);
    assert.ok(o.sampling.temperature <= 0.3);
    const withBlock = eng.promptIds(o.messages, { think: false }); const without = eng.promptIds(o.messages, { think: true });
    assert.ok(withBlock.length > without.length, 'the template pre-fills an empty think block when think is off');
    await waitFor(() => /title run|title for/.test(logText(t)), 'the job to end');
    await waitFor(() => readChat(t, id).auto_title !== 'pending', 'the job to settle');
  } finally { await t.close(); }
});

test('a reply that starts while the title job runs aborts it first; the reply is not delayed behind a finished title and nothing is generated at the same time', async () => {
  const { run, gate } = gatedRun('Sen tittel');
  const t = await startTestApp({ ctx: 1024, titleRun: run });
  try {
    const q = [U('Hva er hovedstaden i Norge?')];
    const ev = await t.runChat('tiny-llama.gguf', q);
    const reply = ev.filter((e) => e.type === 'chunk').map((e) => e.data.text).join('');
    const { id } = await saveExchange(t, { reply });
    await gate.startedP;
    const t0 = performance.now(); let generatingAt = 0;
    const events2 = await t.runChat('tiny-llama.gguf', [...q, A(reply), U('Og Sverige?')], { after: (e) => { if (e.type === 'phase' && e.data.phase === 'generating' && !generatingAt) generatingAt = performance.now(); } });
    assert.equal(events2.at(-1).type, 'done', JSON.stringify(events2.at(-1)));
    assert.equal(gate.sawAbort, true, 'the title job was told to stop');
    assert.ok(gate.finishedAt > 0 && gate.finishedAt <= generatingAt, 'the title job had let go of the engine before the reply began generating');
    assert.ok(generatingAt - t0 < 5000);
    assert.match(logText(t), /title for .* aborted/);
    assert.equal(await nameOf(t, id), 'Hva er hovedstaden i Norge?', 'no title written from an aborted job');
    assert.equal(readChat(t, id).auto_title, 'pending');
  } finally { await t.close(); }
});

test('no title job starts while a reply is being generated; the same save after the reply starts it', async () => {
  const run = fakeRun('Tittel etterpå');
  const t = await startTestApp({ ctx: 1024, titleRun: run });
  try {
    await t.runChat('tiny-llama.gguf', [U('Hei')]); // loads the model
    const { id } = await t.call('save_chat', { name: 'Hva er hovedstaden i Norge?', model: 'tiny-llama.gguf', messages: [U('Hva er hovedstaden i Norge?')] });
    let saved = false;
    await t.runChat('tiny-llama.gguf', [U('Fortell en lang historie')], {
      after: async (ev) => { if (ev.type === 'chunk' && !saved) { saved = true; await saveExchange(t, { id }); } },
    });
    assert.equal(run.calls.length, 0, 'a save in the middle of a reply does not start a job');
    assert.equal(readChat(t, id).auto_title, 'pending');
  } finally { await t.close(); }
});

test('the title run leaves the next reply at least as much cached memory as without it (prefix of the chat is kept)', async () => {
  const t = await startTestApp({ ctx: 4096 }); // (a window big enough that the prompt is not trimmed)
  try {
    const q = [U('Hva er hovedstaden i Norge?')];
    const ev = await t.runChat('tiny-llama.gguf', q);
    const reply = ev.filter((e) => e.type === 'chunk').map((e) => e.data.text).join('');
    const eng = t.app.host.engine;
    const kvBefore = eng.model.tokens.slice();
    const history2 = [...q, A(reply), U('Og Sverige?')];
    const next = eng.promptIds(buildMessages(history2));
    const common = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; };
    const baseline = common(kvBefore, next);
    const { id } = await saveExchange(t, { reply }); // starts the title job on the chat model (real generation: noise, so the title is rejected)
    await waitFor(() => readChat(t, id).auto_title === 'failed', 'the title job to finish');
    assert.match(logText(t), /title run \(chat model\)/);
    const kvAfter = eng.model.tokens.slice();
    assert.notDeepEqual(kvAfter, kvBefore, 'the job really used the chat engine');
    const afterTitle = common(kvAfter, next);
    assert.ok(afterTitle >= baseline, `reusable prefix after the title run (${afterTitle}) >= before (${baseline})`);
    const ev2 = await t.runChat('tiny-llama.gguf', history2);
    const done = ev2.at(-1).data;
    assert.equal(done.cachedTokens, Math.min(afterTitle, done.promptTokens - 1));
    assert.ok(done.cachedTokens >= baseline, 'the reply after a title run reads no more than it would have without one');
  } finally { await t.close(); }
});

test('the chat model\'s memory of another conversation is left alone (title skipped)', async () => {
  const run = fakeRun('Skal ikke skje');
  const t = await startTestApp({ ctx: 1024, titleRun: run });
  try {
    await t.runChat('tiny-llama.gguf', [U('Et helt annet spørsmål om fisk og fiske i Barentshavet, lenge og vel')]);
    const { id } = await saveExchange(t, { reply: 'Oslo.' });
    await waitFor(() => /memory holds another conversation/.test(logText(t)), 'the skip line');
    assert.equal(run.calls.length, 0);
    assert.equal(readChat(t, id).auto_title, 'pending');
  } finally { await t.close(); }
});

test('shutting the app down while a title job runs aborts it and closes everything', async () => {
  const { run, gate } = gatedRun();
  const t = await startTestApp({ ctx: 1024, models: UTIL, titleRun: run });
  await t.call('set_utility_model', { filename: 'tiny-util.gguf' });
  await saveExchange(t);
  await gate.startedP;
  await t.close();
  assert.equal(gate.sawAbort, true);
});

// ---------------------------------------------------------------- the page's saves

test('race: a title that arrives between two page saves is not overwritten by the stale placeholder; a real rename still wins', async () => {
  await withUtility(fakeRun('Hovedstaden i Norge'), async (t) => {
    const placeholder = 'Hva er hovedstaden i Norge?';
    // page save 1: the question alone (the reply is on its way)
    const s1 = await t.call('save_chat', { name: placeholder, model: 'tiny-llama.gguf', messages: [U(placeholder)] });
    assert.equal(s1.name, placeholder);
    // page save 2 after the reply starts the job; the title lands before the page's next save
    await t.call('save_chat', { id: s1.id, name: placeholder, model: 'tiny-llama.gguf', messages: [U(placeholder), A('Oslo.')] });
    await waitFor(async () => (await nameOf(t, s1.id)) === 'Hovedstaden i Norge', 'the title');
    // page save 3 still carries the placeholder (it never saw the event)
    const s3 = await t.call('save_chat', { id: s1.id, name: placeholder, model: 'tiny-llama.gguf', messages: [U(placeholder), A('Oslo.'), U('Og Sverige?'), A('Stockholm.')] });
    assert.equal(s3.name, 'Hovedstaden i Norge', 'the response tells the page the real name');
    assert.equal(await nameOf(t, s1.id), 'Hovedstaden i Norge');
    // a save without any name keeps it too
    assert.equal((await t.call('save_chat', { id: s1.id, messages: [U(placeholder), A('Oslo.')] })).name, 'Hovedstaden i Norge');
    // the page sending a genuinely different name (rename) wins from then on
    const s4 = await t.call('save_chat', { id: s1.id, name: 'Mitt eget navn', messages: [U(placeholder), A('Oslo.')] });
    assert.equal(s4.name, 'Mitt eget navn');
    assert.equal(readChat(t, s1.id).auto_title, 'user');
    assert.equal((await t.call('save_chat', { id: s1.id, name: placeholder, messages: [U(placeholder), A('Oslo.')] })).name, placeholder, 'after a user rename the page decides (same as before this feature)');
  });
});

test('race: the title job finishing while a page save is in flight leaves a consistent file (writes are one synchronous step)', async () => {
  const run = fakeRun('Tittel uten kollisjon');
  await withUtility(run, async (t) => {
    const placeholder = 'Spørsmål';
    const { id } = await t.call('save_chat', { name: placeholder, model: 'tiny-llama.gguf', messages: [U(placeholder)] });
    const saves = [];
    for (let i = 0; i < 12; i++) saves.push(t.call('save_chat', { id, name: placeholder, model: 'tiny-llama.gguf', messages: [U(placeholder), A(`svar ${i}`)] }));
    await Promise.all(saves);
    await waitFor(async () => (await nameOf(t, id)) === 'Tittel uten kollisjon', 'the title');
    const raw = readChat(t, id);
    assert.equal(raw.auto_title, 'done'); assert.equal(raw.messages[0].role, 'system'); assert.equal(raw.messages.length, 3);
    assert.equal(run.calls.length, 1, 'one job even though twelve saves arrived');
  });
});

test('chats from other apps are untouched: no auto_title field is added on save, names are taken as given', async () => {
  const run = fakeRun('Skal ikke brukes');
  await withUtility(run, async (t) => {
    const id = '3f2b1c0e-1111-4222-8333-444455556666';
    fs.mkdirSync(path.join(t.dir, 'chats'), { recursive: true });
    fs.writeFileSync(path.join(t.dir, 'chats', `${id}.json`), JSON.stringify({ id, name: 'Gammel samtale', model: 'tiny-llama.gguf', messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'Hei' }, { role: 'assistant', content: 'Hei!' }], attached_docs: {}, updated_at: 1760000000 }));
    const r = await t.call('save_chat', { id, name: 'Annet navn', messages: [U('Hei'), A('Hei!')] });
    assert.equal(r.name, 'Annet navn');
    await sleep(100);
    assert.equal(readChat(t, id).auto_title, undefined);
    assert.equal(run.calls.length, 0);
  });
});
