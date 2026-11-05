import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parseSkill } from '../../app/skills.js';
import { startTestApp } from './helpers.js';

test('parseSkill: heading with inline or own-line description, body until next command, name from # heading', () => {
  const s = parseSkill('x.md', `# Mine\n\nignored intro\n\n## /one  Inline desc\nBody one\nsecond line\n\n## /Two\nDescription line\n\nBody two\n\n## /empty\n\n## /Dash-3 d\nok`);
  assert.equal(s.name, 'Mine');
  assert.deepEqual(s.commands.map((c) => c.command), ['one', 'two', 'dash-3']);
  assert.equal(s.commands[0].description, 'Inline desc');
  assert.equal(s.commands[0].body, 'Body one\nsecond line');
  assert.equal(s.commands[1].description, 'Description line');
  assert.equal(s.commands[1].body, 'Body two');
  assert.equal(parseSkill('my-notes_file.md', '## /a x\nb').name, 'My notes file');
});

test('bundled skills are seeded once and match Skibeti37 commands; deleted ones stay deleted', async () => {
  const t = await startTestApp();
  try {
    const cmds = await t.call('list_commands');
    assert.deepEqual(cmds.map((c) => c.command), ['bokmal', 'humanize', 'norwegian', 'nynorsk', 'proofread', 'shorten', 'simple', 'tutor']);
    assert.ok(cmds.every((c) => c.description && c.skill));
    const skills = await t.call('list_skills');
    assert.deepEqual(skills.map((s) => s.id), ['language.md', 'tutor.md', 'writing.md']);
    await t.call('delete_skill', { id: 'tutor.md' });
    assert.equal((await t.call('list_commands')).some((c) => c.command === 'tutor'), false);
    await assert.rejects(t.call('delete_skill', { id: '../config.json' }), /invalid skill file/);
    await t.close({ keep: true });
    const again = await startTestApp({ dir: t.dir });
    assert.equal((await again.call('list_commands')).some((c) => c.command === 'tutor'), false, 'not re-seeded after a restart');
    await again.close();
  } catch (e) { fs.rmSync(t.dir, { recursive: true, force: true }); throw e; }
});

test('add_skill validates, saves, never overwrites; files dropped in the folder appear without restart', async () => {
  const t = await startTestApp();
  try {
    await assert.rejects(t.call('add_skill', { name: 'x', content: 'no command here' }), /Fant ingen kommando/);
    await assert.rejects(t.call('add_skill', { name: 'x', content: '## /empty' }), /mangler instruksjonstekst/);
    await t.call('add_skill', { name: 'Min stil', content: '## /kort  Kort svar\nSvar med én setning.' });
    await t.call('add_skill', { name: 'Min stil', content: '## /kort2  Enda kortere\nSvar med ett ord.' });
    const files = fs.readdirSync(path.join(t.dir, 'skills'));
    assert.ok(files.includes('min-stil.md') && files.includes('min-stil-1.md'));
    fs.writeFileSync(path.join(t.dir, 'skills', 'ekstern.md'), '# Ekstern\n\n## /ekstern  Fra utsiden\nGjør noe.');
    const names = (await t.call('list_commands')).map((c) => c.command);
    assert.ok(names.includes('kort') && names.includes('kort2') && names.includes('ekstern'));
  } finally { await t.close(); }
});

test('commands on the last user message add their instruction to the system prompt for that reply only', async () => {
  const t = await startTestApp({ ctx: 4096 });
  try {
    const base = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [{ role: 'user', content: 'Hei' }] });
    const withCmd = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [{ role: 'user', content: 'Hei', commands: ['humanize'] }] });
    assert.ok(withCmd.contextUsed > base.contextUsed + 200, 'the /humanize instruction is part of the prompt');
    const older = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [{ role: 'user', content: 'Hei', commands: ['humanize'] }, { role: 'assistant', content: 'Hei!' }, { role: 'user', content: 'Og så?' }] });
    const plain = await t.call('estimate_context', { model: 'tiny-llama.gguf', messages: [{ role: 'user', content: 'Hei' }, { role: 'assistant', content: 'Hei!' }, { role: 'user', content: 'Og så?' }] });
    assert.equal(older.contextUsed, plain.contextUsed, 'only the latest message\'s commands count');
    const ev = await t.runChat('tiny-llama.gguf', [{ role: 'user', content: 'Hei', commands: ['shorten', 'nonexistent'] }]);
    assert.equal(ev.at(-1).type, 'done');
  } finally { await t.close(); }
});

import { SYSTEM_PROMPT, buildMessages } from '../../app/chat-core.js';
test('the system prompt stays free of topics, and it is the same for every message (no per-message language or math reminders)', () => {
  assert.ok(!/math|formula|latex|\$/i.test(SYSTEM_PROMPT));
  const sys = (t) => buildMessages([{ role: 'user', content: t }])[0].content;
  for (const t of ['hei', 'Hva er 2+3?', 'Regn ut arealet av en sirkel', 'And who pays for it?', 'ok']) assert.equal(sys(t), SYSTEM_PROMPT, t);
});
