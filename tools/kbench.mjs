// Per-kernel micro-benchmark (cache-resident weights => compute-bound): node tools/kbench.mjs
import { instantiateKernels } from '../src/compute/wasm/kernels.js';
import { quantizeAct } from '../src/compute/ref-kernels.js';
import { randomQuantBlocks } from './random-blocks.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
const rand = mulberry32(1);
const rows = 1024, cols = 2048, B = 16, REPS = 6;
const cases = [
  ['q8_0', GGML_TYPE.Q8_0, false, () => quantizeQ8_0(Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.1))],
  ['q4_0', GGML_TYPE.Q4_0, false, () => quantizeQ4_0(Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.1))],
  ['q5_0', GGML_TYPE.Q5_0, true, () => randomQuantBlocks(GGML_TYPE.Q5_0, rows * cols / 32, rand)],
  ['q4_k', GGML_TYPE.Q4_K, true, () => randomQuantBlocks(GGML_TYPE.Q4_K, rows * cols / 256, rand)],
  ['q5_k', GGML_TYPE.Q5_K, true, () => randomQuantBlocks(GGML_TYPE.Q5_K, rows * cols / 256, rand)],
  ['q6_k', GGML_TYPE.Q6_K, true, () => randomQuantBlocks(GGML_TYPE.Q6_K, rows * cols / 256, rand)],
];
console.log(`rows=${rows} cols=${cols} (${(rows * cols / 1e6).toFixed(1)} M weights), batch ${B}, 1 thread; G weights-per-token/s = rows*cols*tokens / time`);
for (const [name, type, useXs, mk] of cases) {
  const info = GGML_TYPE_INFO[type]; const nb = cols / info.blockSize; const abl = cols / 32;
  const wbytes = mk();
  const al = (v) => Math.ceil(v / 64) * 64;
  const wOff = 64, xq8 = al(wOff + wbytes.length), xd1 = al(xq8 + cols), xs1 = al(xd1 + abl * 4), y1 = al(xs1 + abl * 4);
  const xq16 = al(y1 + rows * 4), xdB = al(xq16 + B * cols * 2), xsB = al(xdB + B * abl * 4), yB = al(xsB + B * abl * 4);
  const mem = new WebAssembly.Memory({ initial: Math.ceil((yB + B * rows * 4 + 64) / 65536) });
  const k = instantiateKernels(mem);
  new Uint8Array(mem.buffer, wOff, wbytes.length).set(wbytes);
  const x = Float32Array.from({ length: cols }, () => gaussian(rand));
  quantizeAct(x, new Int8Array(mem.buffer, xq8, cols), new Float32Array(mem.buffer, xd1, abl), new Float32Array(mem.buffer, xs1, abl));
  for (let t = 0; t < B; t++) quantizeAct(x, new Int16Array(mem.buffer, xq16 + t * cols * 2, cols), new Float32Array(mem.buffer, xdB + t * abl * 4, abl), new Float32Array(mem.buffer, xsB + t * abl * 4, abl));
  const dec = k[`matvec_${name}`], mmk = k[`mm_${name}`];
  const run1 = () => (useXs ? dec(wOff, xq8, xd1, xs1, y1, rows, nb) : dec(wOff, xq8, xd1, y1, rows, nb));
  const runB = () => (useXs ? mmk(wOff, xq16, xdB, xsB, yB, rows, nb, B) : mmk(wOff, xq16, xdB, yB, rows, nb, B));
  for (let i = 0; i < 3; i++) { run1(); runB(); }
  const time = (fn, n) => { const t = performance.now(); for (let i = 0; i < n; i++) fn(); return (performance.now() - t) / n; };
  const tDec = Math.min(...Array.from({ length: REPS }, () => time(run1, 5)));
  const tB = Math.min(...Array.from({ length: REPS }, () => time(runB, 2)));
  const gw = rows * cols / 1e9;
  console.log(`${name.padEnd(5)} decode ${(gw / (tDec / 1000)).toFixed(2).padStart(6)} Gw/s | batch16 ${(gw * B / (tB / 1000)).toFixed(2).padStart(6)} Gw/s-token | batch speedup ${(gw * B / tB / (gw / tDec)).toFixed(2)}x`);
}
