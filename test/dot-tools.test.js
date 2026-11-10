// Smoke tests for the dot-form benchmark tools (tiny sizes; they only check that the tools run and print what they promise).
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tinyFile } from './helpers.js';
import { probeWasmFeatures } from '../src/compute/wasm/features.js';

const TOOLS = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'tools');
const RELAXED = probeWasmFeatures().relaxedSimd;

test('kbench prints decode and batched throughput for every format; relaxed columns are numbers, or n/a without relaxed SIMD', () => {
  const r = spawnSync(process.execPath, [path.join(TOOLS, 'kbench.mjs'), '--rows', '16', '--cols', '512', '--batch', '4', '--reps', '1'], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  for (const f of ['q8_0', 'q4_0', 'q5_0', 'q4_k', 'q5_k', 'q6_k']) {
    const line = r.stdout.split('\n').find((l) => l.trim().startsWith(f));
    assert.ok(line, `row for ${f}`);
    const cells = line.trim().split(/\s+/).filter((c) => c !== '|');
    // format, 4 decode speeds, 2 decode ratios, 4 batched speeds, 2 batched ratios
    assert.equal(cells.length, 13, line);
    const speeds = [1, 2, 3, 4, 7, 8, 9, 10].map((i) => cells[i]);
    for (const s of speeds.slice(0, 2).concat(speeds.slice(4, 6))) assert.ok(Number.isFinite(Number(s)) && Number(s) > 0, `${f}: extmul/wide speed "${s}"`);
    for (const s of [speeds[2], speeds[3], speeds[6], speeds[7]]) {
      if (RELAXED) assert.ok(Number(s) > 0, `${f}: relaxed speed "${s}"`); else assert.equal(s, 'n/a');
    }
  }
});

test('dotbench runs interleaved rounds over the modes through `zerollm bench` and prints one row per mode', () => {
  const { path: model } = tinyFile({ arch: 'qwen3', quant: 'q4_k_m', seed: 3, nEmbd: 256, nHead: 4, nHeadKv: 2, headDim: 64, nFF: 512, nLayer: 2 });
  const r = spawnSync(process.execPath, [path.join(TOOLS, 'dotbench.mjs'), model, '--modes', 'extmul,relaxed,relaxed-add', '--rounds', '2', '--pause', '0',
    '--threads', '1', '--batch', '4', '--prompt-tokens', '8', '--n', '3'], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr + r.stdout);
  const modes = RELAXED ? ['extmul', 'relaxed', 'relaxed-add'] : ['extmul'];
  for (const m of modes) assert.ok(new RegExp(`^${m}\\s+\\d`, 'm').test(r.stdout), `summary row for ${m}:\n${r.stdout}`);
  assert.equal((r.stdout.match(/^round \d\/2 /gm) ?? []).length, 2 * modes.length);
});
