// Interleaved model-level A/B of the integer-dot forms: runs `zerollm bench` several rounds, alternating ZEROLLM_DOT between the
// given modes each round (rotating the order), and reports median / min / max prefill and decode tok/s per mode.
// Laptop speed drifts with heat and power state by +-15%, so single runs (or all runs of one mode in a row) cannot resolve
// differences smaller than that. Same idea as tools/abbench.mjs, but the varied setting is an environment variable.
//
//   node tools/dotbench.mjs models/Qwen3-4B-Q4_K_M.gguf [--modes extmul,relaxed,relaxed-add] [--rounds 5] [--threads 6]
//                           [--batch 16] [--prompt-tokens 64] [--n 16] [--pause 10]
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs, num } from '../src/util/args.js';
import { probeWasmFeatures } from '../src/compute/wasm/features.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const { _, flags } = parseArgs(process.argv.slice(2));
if (!_[0]) {
  console.error('usage: node tools/dotbench.mjs <model.gguf> [--modes extmul,relaxed,relaxed-add] [--rounds 5] [--threads 6] [--batch 16] [--prompt-tokens 64] [--n 16] [--pause 10]');
  process.exit(1);
}
let modes = (typeof flags.modes === 'string' ? flags.modes : 'extmul,relaxed,relaxed-add').split(',');
if (!probeWasmFeatures().relaxedSimd && modes.some((m) => m.startsWith('relaxed'))) {
  console.error('this Node/V8 has no WebAssembly relaxed SIMD: relaxed modes skipped');
  modes = modes.filter((m) => !m.startsWith('relaxed'));
}
const rounds = num(flags.rounds, 5);
const pause = num(flags.pause, 10); // seconds between runs
const fixed = { threads: flags.threads ?? 6, batch: flags.batch ?? 16, 'prompt-tokens': flags['prompt-tokens'] ?? 64, n: flags.n ?? 16 };
const results = Object.fromEntries(modes.map((m) => [m, { decode: [], prefill: [] }]));
const sleep = (s) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, s * 1000);
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

for (let r = 0; r < rounds; r++) {
  const order = modes.map((_, i) => modes[(i + r) % modes.length]);
  for (const mode of order) {
    const args = [BIN, 'bench', _[0], ...Object.entries(fixed).flatMap(([k, v]) => [`--${k}`, String(v)])];
    const out = spawnSync(process.execPath, args, { encoding: 'utf8', env: { ...process.env, ZEROLLM_DOT: mode } });
    const row = (out.stdout.match(/^\| (?:wasm-simd|js) .*$/m) ?? [''])[0].split('|').map((s) => s.trim());
    // columns: '', kernels, threads, batch, load s, prefill, decode, ratio, GB/s, RSS
    const prefill = Number(row[5]); const decode = Number(row[6]);
    if (!Number.isFinite(decode)) { console.error(`round ${r + 1} ZEROLLM_DOT=${mode}: could not parse output\n${out.stdout}\n${out.stderr}`); process.exit(1); }
    results[mode].decode.push(decode); results[mode].prefill.push(prefill);
    process.stdout.write(`round ${r + 1}/${rounds}  ZEROLLM_DOT=${mode.padEnd(11)} prefill ${String(prefill).padStart(6)} tok/s  decode ${String(decode).padStart(6)} tok/s\n`);
    if (pause > 0) sleep(pause);
  }
}
const range = (a) => `${med(a).toFixed(2).padStart(6)} (${Math.min(...a).toFixed(2)}-${Math.max(...a).toFixed(2)})`;
console.log(`\nthreads ${fixed.threads}, batch ${fixed.batch}, prompt ${fixed['prompt-tokens']} tokens, ${fixed.n} generated, ${rounds} interleaved rounds\n`);
console.log('ZEROLLM_DOT   prefill tok/s median (min-max)   decode tok/s median (min-max)   prefill vs first | decode vs first');
for (const m of modes) {
  const pf = med(results[m].prefill) / med(results[modes[0]].prefill); const dc = med(results[m].decode) / med(results[modes[0]].decode);
  console.log(`${m.padEnd(13)} ${range(results[m].prefill).padEnd(31)} ${range(results[m].decode).padEnd(31)} ${((pf - 1) * 100).toFixed(0).padStart(5)}% | ${((dc - 1) * 100).toFixed(0).padStart(5)}%`);
}
console.log('Differences smaller than the min-max spread of a mode are not distinguishable from noise.');
