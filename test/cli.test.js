import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tinyFile } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { generate } from '../src/model/generate.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const cli = (args, input) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8', input });

test('doctor runs and reports required features', () => {
  const r = cli(['doctor']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /SIMD/);
  assert.match(r.stdout, /All required features present/);
});

test('info on a tiny model', () => {
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const r = cli(['info', p]);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /Architecture\s+qwen3/);
  assert.match(r.stdout, /Q8_0/);
  assert.match(r.stdout, /Head dim\s+32/);
});

test('unknown command / missing model fail with a clear message', () => {
  assert.notEqual(cli(['bogus']).status, 0);
  const r = cli(['run']);
  assert.notEqual(r.status, 0);
});

test('run: seeded output is reproducible; greedy is independent of seed', () => {
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 12 });
  const a = cli(['run', p, '-p', 'hello', '--n', '10', '--seed', '7']);
  const b = cli(['run', p, '-p', 'hello', '--n', '10', '--seed', '7']);
  assert.equal(a.status, 0, a.stderr);
  assert.equal(a.stdout, b.stdout);
  const g1 = cli(['run', p, '-p', 'hello', '--n', '10', '--temp', '0', '--seed', '1']);
  const g2 = cli(['run', p, '-p', 'hello', '--n', '10', '--temp', '0', '--seed', '2']);
  assert.equal(g1.stdout, g2.stdout);
  assert.match(a.stderr, /generated 10 tok/);
});

test('run reads the prompt from stdin', () => {
  const { path: p } = tinyFile({ arch: 'llama', quant: 'q8_0' });
  const r = cli(['run', p, '--raw', '--n', '4', '--temp', '0'], 'hello there');
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /template: raw/);
});

test('chat: scripted session, /reset and /exit', () => {
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  const r = cli(['chat', p, '--n', '5', '--seed', '1'], 'hi\n/reset\nagain\n/exit\n');
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /conversation cleared|generated 5 tok/);
  assert.equal((r.stderr.match(/generated 5 tok/g) ?? []).length, 2);
});

// Windows cannot deliver a catchable SIGINT via child.kill() (it terminates the process), so check Ctrl+C by hand there.
test('Ctrl+C (SIGINT) during generation stops cleanly with exit code 130 and stats', { skip: process.platform === 'win32' && 'child.kill(SIGINT) kills the process on Windows' }, async () => {
  // bigger tiny model so generation is slow enough to interrupt
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0', nEmbd: 256, nLayer: 6, nFF: 512, ctx: 4096, seed: 3 });
  const child = spawn(process.execPath, [BIN, 'run', p, '-p', 'hi', '--n', '3000', '--ctx', '4096', '--temp', '0'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = ''; let err = '';
  child.stdout.on('data', (d) => { out += d; if (out.length > 5 && !child.killed) child.kill('SIGINT'); });
  child.stderr.on('data', (d) => { err += d; });
  const code = await new Promise((r) => child.on('close', r));
  assert.equal(code, 130, err);
  assert.match(err, /stop: aborted/);
  assert.ok(out.length < 3000 * 4);
});

test('prefix reuse: second turn only prefills the new tokens and matches a fresh run', async () => {
  const { g } = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 9 });
  const m = new RefModel(g, { ctx: 128 });
  const collect = async (model, ids, st) => { const o = []; for await (const t of generate(model, ids, { maxTokens: 5, stats: st })) o.push(t); return o; };
  const base = [5, 6, 7, 8, 9, 10];
  const s1 = {}; const first = await collect(m, base, s1);
  assert.equal(s1.cachedTokens, 0);
  const longer = [...base, ...first, 30, 31];
  const s2 = {}; const second = await collect(m, longer, s2);
  assert.ok(s2.cachedTokens >= base.length, `cached ${s2.cachedTokens}`);
  assert.ok(s2.prefilled < longer.length);
  const fresh = new RefModel(g, { ctx: 128 });
  assert.deepEqual(second, await collect(fresh, longer, {}));
  // identical prompt again: still recomputes the last token to get logits
  const s3 = {}; await collect(m, longer, s3);
  assert.equal(s3.prefilled, 1);
  await assert.rejects(collect(m, Array.from({ length: 200 }, () => 1), {}), /context window/);
});

test('streamReply never erases printed text with the progress-line escape (terminal regression)', async () => {
  const { loadEngine, streamReply } = await import('../src/cli/common.js');
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 5 });
  const eng = await loadEngine(p, { threads: 2 });
  const events = [];
  const out = { write: (s) => events.push(['out', s]) };
  const err = { write: (s) => events.push(['err', s]) };
  // prompt longer than 8 tokens so the progress line is shown
  const ids = Array.from({ length: 20 }, (_, i) => 10 + i);
  await streamReply(eng, ids, { sample: () => 65, maxTokens: 5, signal: { aborted: false }, out, err, progress: true });
  const firstOut = events.findIndex((e) => e[0] === 'out');
  assert.ok(firstOut > 0, 'some text was printed after progress output');
  assert.ok(events.slice(0, firstOut).some((e) => e[0] === 'err' && e[1].startsWith('\rprefill')), 'progress was shown');
  assert.ok(events.slice(0, firstOut).some((e) => e[0] === 'err' && e[1] === '\r\x1b[K'), 'progress erased before text');
  assert.ok(!events.slice(firstOut).some((e) => e[0] === 'err' && e[1].includes('\x1b[K')), 'nothing erased after text');
  await eng.close();
});

test('--show-prompt prints the token count, ids and the text with control tokens visible', () => {
  for (const arch of ['qwen3', 'llama']) {
    const { path: p } = tinyFile({ arch, quant: 'q8_0', seed: 3 });
    const r = spawnSync(process.execPath, [BIN, 'run', p, '--system', 'Be brief.', '-p', 'hello', '--n', '2', '--temp', '0', '--show-prompt', '--no-cache'], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stderr, /\[prompt: \d+ tokens, template (qwen|llama3)\]/);
    assert.match(r.stderr, /\[ids: \d+( \d+)+\]/);
    assert.match(r.stderr, arch === 'llama' ? /<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>/ : /<\|im_start\|>system/);
    assert.match(r.stderr, arch === 'llama' ? /<\|start_header_id\|>assistant<\|end_header_id\|>/ : /<\|im_start\|>assistant/);
  }
});
