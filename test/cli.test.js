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

test('Ctrl+C (SIGINT) during generation stops cleanly with exit code 130 and stats', async () => {
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
