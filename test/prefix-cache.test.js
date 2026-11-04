import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { tinyFile } from './helpers.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { PrefixCache, modelFingerprint } from '../src/model/prefix-cache.js';
import { prefillTokens, generate } from '../src/model/generate.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-cache-'));
const prefix = Array.from({ length: 21 }, (_, i) => (i * 17 + 5) % 250);
const rest = [9, 33, 120, 7];

async function makers() {
  return {
    ref: async (g) => new RefModel(g, { ctx: 64 }),
    'pool-js': async (g) => PoolModel.create(g, { ctx: 64, workers: 2, kernels: 'js' }),
    'pool-wasm': async (g) => PoolModel.create(g, { ctx: 64, workers: 3, kernels: 'wasm' }),
  };
}

for (const kind of ['ref', 'pool-js', 'pool-wasm']) {
  test(`prefix cache round trip is exact (${kind}): restored state continues bit-identically`, async () => {
    const { g, path: mp } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 6 });
    const make = (await makers())[kind];
    const dir = tmpDir();
    const cache = new PrefixCache(dir, modelFingerprint(mp, kind));
    const a = await make(g);
    const b = await make(g);
    try {
      // a: prefill the prefix, save, keep going
      await prefillTokens(a, prefix, { wantLogits: false });
      await cache.save(a, prefix);
      let want; for (const t of rest) want = Float32Array.from(await a.forward(t));
      // b: load instead of computing
      assert.equal(await cache.load(b, prefix), true);
      assert.equal(b.pos, prefix.length);
      assert.deepEqual(b.tokens.slice(), prefix);
      let got; for (const t of rest) got = Float32Array.from(await b.forward(t));
      assert.deepEqual([...got], [...want]);
    } finally { await a.close?.(); await b.close?.(); fs.rmSync(dir, { recursive: true, force: true }); }
  });
}

test('prefix cache refuses mismatches and damage without throwing', async () => {
  const { g, path: mp } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 6 });
  const dir = tmpDir();
  const fp = modelFingerprint(mp, 'js');
  const cache = new PrefixCache(dir, fp);
  const src = new RefModel(g, { ctx: 64 });
  await prefillTokens(src, prefix, { wantLogits: false });
  const file = await cache.save(src, prefix);
  const fresh = () => new RefModel(g, { ctx: 64 });
  try {
    assert.equal(await cache.load(fresh(), prefix), true);
    assert.equal(await cache.load(fresh(), [...prefix.slice(0, -1), 99]), false, 'different tokens -> different key, no file');
    assert.equal(await new PrefixCache(dir, fp + 'x').load(fresh(), prefix), false, 'different model identity');
    const used = fresh(); await prefillTokens(used, [1, 2], { wantLogits: false });
    assert.equal(await cache.load(used, prefix), false, 'only into a fresh model');
    assert.equal(await cache.load(fresh(), []), false);
    const good = fs.readFileSync(file);
    fs.writeFileSync(file, good.subarray(0, good.length - 7)); // truncated
    assert.equal(await cache.load(fresh(), prefix), false);
    fs.writeFileSync(file, Buffer.concat([good, Buffer.from([1, 2, 3])])); // extended
    assert.equal(await cache.load(fresh(), prefix), false);
    const bad = Buffer.from(good); bad.write('NOTACACH', 0, 'latin1'); fs.writeFileSync(file, bad);
    assert.equal(await cache.load(fresh(), prefix), false);
    fs.writeFileSync(file, 'garbage');
    assert.equal(await cache.load(fresh(), prefix), false);
    // header says a different token list (hash collision / tampering)
    const h = Buffer.from(good); const hlen = h.readUInt32LE(8); const head = JSON.parse(h.toString('utf8', 12, 12 + hlen));
    head.tokens[3] = 249; const nh = Buffer.from(JSON.stringify(head));
    fs.writeFileSync(file, Buffer.concat([h.subarray(0, 8), Buffer.from(new Uint32Array([nh.length]).buffer), nh, h.subarray(12 + hlen)]));
    assert.equal(await cache.load(fresh(), prefix), false);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('a model whose cache load failed halfway is still usable (state not corrupted)', async () => {
  const { g, path: mp } = tinyFile({ arch: 'llama', quant: 'q8_0', seed: 2 });
  const dir = tmpDir();
  const cache = new PrefixCache(dir, modelFingerprint(mp, 'js'));
  const src = new RefModel(g, { ctx: 64 });
  await prefillTokens(src, prefix, { wantLogits: false });
  const file = await cache.save(src, prefix);
  const good = fs.readFileSync(file);
  fs.writeFileSync(file, good.subarray(0, good.length - 1));
  const m = new RefModel(g, { ctx: 64 });
  assert.equal(await cache.load(m, prefix), false);
  assert.equal(m.pos, 0);
  const ref = new RefModel(g, { ctx: 64 });
  assert.deepEqual([...await m.forward(5)], [...await ref.forward(5)]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('prune keeps only the newest files', async () => {
  const dir = tmpDir();
  for (let i = 0; i < 6; i++) { fs.writeFileSync(path.join(dir, `f${i}.kv`), 'x'); fs.utimesSync(path.join(dir, `f${i}.kv`), new Date(2020, 0, 1 + i), new Date(2020, 0, 1 + i)); }
  fs.writeFileSync(path.join(dir, 'keep.txt'), 'not a cache file');
  new PrefixCache(dir, 'x').prune(3);
  assert.deepEqual(fs.readdirSync(dir).sort(), ['f3.kv', 'f4.kv', 'f5.kv', 'keep.txt']);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('generate() after a cache hit equals generate() after a fresh prefill', async () => {
  const { g, path: mp } = tinyFile({ arch: 'qwen2', quant: 'q8_0', seed: 12 });
  const dir = tmpDir();
  const cache = new PrefixCache(dir, modelFingerprint(mp, 'wasm'));
  const prompt = [...prefix, ...rest];
  const collect = async (m, st = {}) => { const o = []; for await (const t of generate(m, prompt, { maxTokens: 8, stats: st })) o.push(t); return o; };
  const fresh = await PoolModel.create(g, { ctx: 64, workers: 2 });
  const warm = await PoolModel.create(g, { ctx: 64, workers: 2 });
  try {
    const want = await collect(fresh);
    await prefillTokens(fresh, prefix, { wantLogits: false }); // (fresh now holds extra state; use a third model to save)
    const saver = await PoolModel.create(g, { ctx: 64, workers: 2 });
    await prefillTokens(saver, prefix, { wantLogits: false }); await cache.save(saver, prefix); await saver.close();
    assert.equal(await cache.load(warm, prefix), true);
    const st = {};
    assert.deepEqual(await collect(warm, st), want);
    assert.equal(st.cachedTokens, prefix.length);
  } finally { await fresh.close(); await warm.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

test('CLI: first run saves the system-prompt cache, second run hits it with identical output; --no-cache skips it', () => {
  const { path: mp } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 3 });
  const dir = tmpDir();
  const sys = 'You are a careful assistant who always answers in complete sentences and explains each step of its reasoning clearly before giving a final answer.';
  const run = (...extra) => spawnSync(process.execPath, [BIN, 'run', mp, '--system', sys, '-p', 'hello there', '--n', '8', '--temp', '0', '--cache-dir', dir, ...extra], { encoding: 'utf8' });
  const a = run(); const b = run();
  assert.equal(a.status, 0, a.stderr); assert.equal(b.status, 0, b.stderr);
  assert.match(a.stderr, /system-prompt cache: saved/);
  assert.match(b.stderr, /system-prompt cache: hit/);
  assert.equal(a.stdout, b.stdout);
  assert.equal(fs.readdirSync(dir).filter((f) => f.endsWith('.kv')).length, 1);
  const dir2 = tmpDir();
  const c = spawnSync(process.execPath, [BIN, 'run', mp, '--system', sys, '-p', 'hello there', '--n', '8', '--temp', '0', '--cache-dir', dir2, '--no-cache'], { encoding: 'utf8' });
  assert.equal(c.status, 0, c.stderr);
  assert.doesNotMatch(c.stderr, /system-prompt cache/);
  assert.equal(c.stdout, a.stdout);
  assert.equal(fs.readdirSync(dir2).length, 0);
  fs.rmSync(dir, { recursive: true, force: true }); fs.rmSync(dir2, { recursive: true, force: true });
});

test('CLI: short system prompts are not cached', () => {
  const { path: mp } = tinyFile({ arch: 'qwen3', quant: 'q8_0', seed: 3 });
  const dir = tmpDir();
  const r = spawnSync(process.execPath, [BIN, 'run', mp, '--system', 'Be brief.', '-p', 'hi', '--n', '4', '--temp', '0', '--cache-dir', dir], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(r.stderr, /system-prompt cache/);
  assert.equal(fs.existsSync(dir) ? fs.readdirSync(dir).length : 0, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});
