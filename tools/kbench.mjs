// Per-kernel micro-benchmark (cache-resident weights => compute-bound): node tools/kbench.mjs [--rows 1024] [--cols 2048] [--batch 16] [--reps 6] [--only q4_k,q6_k]
//
// Measures, for every format, ONE thread running
//   decode  : the single-vector kernel      (G weights/s            = rows*cols / time)
//   batched : the tile (matmul) kernel      (G weight-token products/s = rows*cols*tokens / time)
// for the integer-dot forms the engine can be built with (env ZEROLLM_DOT=extmul|wide|relaxed|relaxed-add at run time):
//   extmul      default; i16x8.extmul + pairwise adds in the K-quant DECODE kernels
//   wide        older form: widen to int16 + i32x4.dot_i16x8_s in the K-quant DECODE kernels
//   relaxed     i16x8.relaxed_dot_i8x16_i7x16_s + one widening add per 32 weights (decode AND batched; int8 activations)
//   rel-add     i32x4.relaxed_dot_i8x16_i7x16_add_s per 16 weights (decode AND batched; int8 activations)
// The relaxed columns are skipped when the engine lacks relaxed SIMD.
// The batched kernels of extmul and wide are the SAME code (they only differ in the K-quant decode kernels), so those two
// batched columns differ by measurement noise only: they show the noise floor of the run. Variants are timed interleaved
// (round after round, minimum kept), so slow drift of the machine does not favour one column.
import os from 'node:os';
import { instantiateKernels } from '../src/compute/wasm/kernels.js';
import { probeWasmFeatures } from '../src/compute/wasm/features.js';
import { quantizeAct } from '../src/compute/ref-kernels.js';
import { randomQuantBlocks } from './random-blocks.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs, num } from '../src/util/args.js';

const { flags } = parseArgs(process.argv.slice(2));
const rows = num(flags.rows, 1024); const cols = num(flags.cols, 2048); const B = num(flags.batch, 16); const REPS = num(flags.reps, 6);
const only = typeof flags.only === 'string' ? new Set(flags.only.split(',')) : null;
const relaxedOk = probeWasmFeatures().relaxedSimd;
const rand = mulberry32(1);
const cases = [
  ['q8_0', GGML_TYPE.Q8_0, false, () => quantizeQ8_0(Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.1))],
  ['q4_0', GGML_TYPE.Q4_0, false, () => quantizeQ4_0(Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.1))],
  ['q5_0', GGML_TYPE.Q5_0, true, () => randomQuantBlocks(GGML_TYPE.Q5_0, rows * cols / 32, rand)],
  ['q4_k', GGML_TYPE.Q4_K, true, () => randomQuantBlocks(GGML_TYPE.Q4_K, rows * cols / 256, rand)],
  ['q5_k', GGML_TYPE.Q5_K, true, () => randomQuantBlocks(GGML_TYPE.Q5_K, rows * cols / 256, rand)],
  ['q6_k', GGML_TYPE.Q6_K, true, () => randomQuantBlocks(GGML_TYPE.Q6_K, rows * cols / 256, rand)],
];
console.log(`node ${process.version} (V8 ${process.versions.v8}), ${os.cpus()[0]?.model ?? 'unknown CPU'}, relaxed SIMD: ${relaxedOk ? 'yes' : 'NO (relaxed columns skipped)'}`);
console.log(`rows=${rows} cols=${cols} (${(rows * cols / 1e6).toFixed(1)} M weights), batch ${B}, 1 thread, ${REPS} interleaved rounds, minimum time kept`);
console.log('decode = G weights/s; batched = G weight-token products/s (rows*cols*tokens / time); x = speed relative to extmul\n');
const head = ['format', 'dec extmul', 'wide', 'relaxed', 'rel-add', 'rlx x', 'add x', '| bat extmul', 'wide', 'relaxed', 'rel-add', 'rlx x', 'add x'];
const w = [7, 11, 7, 8, 8, 6, 6, 13, 7, 8, 8, 6, 6];
const row = (cells) => cells.map((c, i) => String(c).padStart(w[i])).join(' ');
console.log(row(head));
const al = (v) => Math.ceil(v / 64) * 64;
const time = (fn, n) => { const t = performance.now(); for (let i = 0; i < n; i++) fn(); return (performance.now() - t) / n; };

for (const [name, type, useXs, mk] of cases) {
  if (only && !only.has(name)) continue;
  const info = GGML_TYPE_INFO[type]; const nb = cols / info.blockSize; const abl = cols / 32;
  const wbytes = mk();
  const wOff = 64, xq8 = al(wOff + wbytes.length), xd1 = al(xq8 + cols), xs1 = al(xd1 + abl * 4), y1 = al(xs1 + abl * 4);
  const xq16 = al(y1 + rows * 4), xqB8 = al(xq16 + B * cols * 2), xdB = al(xqB8 + B * cols), xsB = al(xdB + B * abl * 4), yB = al(xsB + B * abl * 4);
  const mem = new WebAssembly.Memory({ initial: Math.ceil((yB + B * rows * 4 + 64) / 65536) });
  new Uint8Array(mem.buffer, wOff, wbytes.length).set(wbytes);
  const x = Float32Array.from({ length: cols }, () => gaussian(rand));
  quantizeAct(x, new Int8Array(mem.buffer, xq8, cols), new Float32Array(mem.buffer, xd1, abl), new Float32Array(mem.buffer, xs1, abl));
  for (let t = 0; t < B; t++) {
    const d = new Float32Array(mem.buffer, xdB + t * abl * 4, abl); const s = new Float32Array(mem.buffer, xsB + t * abl * 4, abl);
    quantizeAct(x, new Int16Array(mem.buffer, xq16 + t * cols * 2, cols), d, s); // default batched layout: int16-widened
    quantizeAct(x, new Int8Array(mem.buffer, xqB8 + t * cols, cols), d, s);      // relaxed batched layout: int8
  }
  const mode = (dot, xqB) => {
    const k = instantiateKernels(mem, { dot });
    const dec = k[`matvec_${name}`]; const mm = k[`mm_${name}`];
    return {
      dec: () => (useXs ? dec(wOff, xq8, xd1, xs1, y1, rows, nb) : dec(wOff, xq8, xd1, y1, rows, nb)),
      bat: () => (useXs ? mm(wOff, xqB, xdB, xsB, yB, rows, nb, B) : mm(wOff, xqB, xdB, yB, rows, nb, B)),
    };
  };
  const variants = { extmul: mode('extmul', xq16), wide: mode('wide', xq16) };
  if (relaxedOk) { variants.relaxed = mode('relaxed', xqB8); variants['relaxed-add'] = mode('relaxed-add', xqB8); }
  const best = {};
  for (const [v, f] of Object.entries(variants)) { for (let i = 0; i < 4; i++) { f.dec(); f.bat(); } best[v] = { dec: Infinity, bat: Infinity }; }
  for (let r = 0; r < REPS; r++) {
    for (const [v, f] of Object.entries(variants)) {
      best[v].dec = Math.min(best[v].dec, time(f.dec, 5));
      best[v].bat = Math.min(best[v].bat, time(f.bat, 2));
    }
  }
  const gw = rows * cols / 1e9;
  const dec = (v) => (best[v] ? gw / (best[v].dec / 1000) : NaN);
  const bat = (v) => (best[v] ? gw * B / (best[v].bat / 1000) : NaN);
  const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : 'n/a');
  const ratio = (a, b) => (Number.isFinite(a) && Number.isFinite(b) ? `${(a / b).toFixed(2)}x` : 'n/a');
  // extmul vs wide only differ in the K-quant decode kernels; for the other formats both columns are the same kernel
  console.log(row([name, f2(dec('extmul')), f2(dec('wide')), f2(dec('relaxed')), f2(dec('relaxed-add')), ratio(dec('relaxed'), dec('extmul')), ratio(dec('relaxed-add'), dec('extmul')),
    `| ${f2(bat('extmul'))}`, f2(bat('wide')), f2(bat('relaxed')), f2(bat('relaxed-add')), ratio(bat('relaxed'), bat('extmul')), ratio(bat('relaxed-add'), bat('extmul'))]));
}
console.log('\nNotes: Q8_0 weights are signed int8 (not 0..127), so Q8_0 has no relaxed kernel; its relaxed-mode decode kernel is the default one and its relaxed batched kernel only reads int8 activations (widening them itself).');
console.log('Q4_0/Q5_0/Q8_0 decode: extmul and wide are the same kernel. Batched extmul and wide are the same kernel in every format (difference = noise).');
