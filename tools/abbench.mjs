// Interleaved A/B benchmark: runs `zerollm bench` several rounds, alternating the settings of ONE flag each round, and
// reports median / min / max decode and prefill tok/s per setting. Laptop speed drifts with heat and power state by
// +-15%, so single runs (or runs of one setting in a row) cannot resolve differences smaller than that.
//
//   node tools/abbench.mjs models/Qwen3-4B-Q4_K_M.gguf --vary spin=0,20000 --rounds 5 [--threads 6] [--n 24] [--prompt-tokens 16] [--pause 15]
//   node tools/abbench.mjs models/x.gguf --vary threads=4,6,8
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs, num } from '../src/util/args.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const { _, flags } = parseArgs(process.argv.slice(2));
if (!_[0] || typeof flags.vary !== 'string' || !flags.vary.includes('=')) {
  console.error('usage: node tools/abbench.mjs <model.gguf> --vary <flag>=<v1,v2,...> [--rounds 5] [--threads 6] [--n 24] [--prompt-tokens 16] [--pause 15]');
  process.exit(1);
}
const [varyFlag, valueList] = flags.vary.split('=');
const values = valueList.split(',');
const rounds = num(flags.rounds, 5);
const pause = num(flags.pause, 10); // seconds between runs, lets the chip cool a little
const fixed = { threads: flags.threads ?? 6, batch: 1, 'prompt-tokens': flags['prompt-tokens'] ?? 16, n: flags.n ?? 24 };
const results = Object.fromEntries(values.map((v) => [v, { decode: [], prefill: [] }]));

const sleep = (s) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, s * 1000);
for (let r = 0; r < rounds; r++) {
  // rotate the order each round so no setting always runs first (or right after a hot run)
  const order = values.map((_, i) => values[(i + r) % values.length]);
  for (const v of order) {
    const args = [BIN, 'bench', _[0], ...Object.entries({ ...fixed, [varyFlag]: v }).flatMap(([k, val]) => [`--${k}`, String(val)])];
    const out = spawnSync(process.execPath, args, { encoding: 'utf8' });
    const row = (out.stdout.match(/^\| (?:wasm-simd|js) .*$/m) ?? [''])[0].split('|').map((s) => s.trim());
    // columns: '', kernels, threads, batch, load s, prefill, decode, ratio, GB/s, RSS
    const prefill = Number(row[5]); const decode = Number(row[6]);
    if (!Number.isFinite(decode)) { console.error(`round ${r + 1} ${varyFlag}=${v}: could not parse output\n${out.stdout}\n${out.stderr}`); process.exit(1); }
    results[v].decode.push(decode); results[v].prefill.push(prefill);
    process.stdout.write(`round ${r + 1}/${rounds}  ${varyFlag}=${v}  decode ${decode} tok/s  prefill ${prefill} tok/s\n`);
    if (pause > 0) sleep(pause);
  }
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
console.log(`\n${varyFlag}      decode median (min-max)        prefill median (min-max)`);
for (const v of values) {
  const d = results[v].decode; const p = results[v].prefill;
  console.log(`${String(v).padEnd(9)} ${med(d).toFixed(2).padStart(6)} (${Math.min(...d).toFixed(2)}-${Math.max(...d).toFixed(2)})`.padEnd(40) + `${med(p).toFixed(2).padStart(6)} (${Math.min(...p).toFixed(2)}-${Math.max(...p).toFixed(2)})`);
}
const base = med(results[values[0]].decode);
console.log(`\ndecode vs ${varyFlag}=${values[0]}: ` + values.map((v) => `${v}: ${((med(results[v].decode) / base - 1) * 100).toFixed(0)}%`).join('  |  '));
console.log('Differences smaller than the min-max spread of a setting are not distinguishable from noise.');
