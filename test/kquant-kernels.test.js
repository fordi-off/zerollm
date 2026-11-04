import test from 'node:test';
import assert from 'node:assert/strict';
import { instantiateKernels } from '../src/compute/wasm/kernels.js';
import { quantizeAct, matvec, Act, dequantRow, rowBytes } from '../src/compute/ref-kernels.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { randomQuantBlocks } from '../tools/random-blocks.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

const CASES = [
  ['Q5_0', GGML_TYPE.Q5_0, 32, 'matvec_q5_0'],
  ['Q4_K', GGML_TYPE.Q4_K, 256, 'matvec_q4_k'],
  ['Q5_K', GGML_TYPE.Q5_K, 256, 'matvec_q5_k'],
  ['Q6_K', GGML_TYPE.Q6_K, 256, 'matvec_q6_k'],
];

function runKernel(name, type, blockElems, rows, nb, wbytes, x) {
  const cols = nb * blockElems; const abl = cols / 32;
  const al = (v) => Math.ceil(v / 64) * 64;
  const wOff = 64; const xqOff = al(wOff + wbytes.length); const xdOff = al(xqOff + cols); const yOff = al(xdOff + abl * 4);
  const mem = new WebAssembly.Memory({ initial: Math.ceil((yOff + rows * 4 + 64) / 65536) });
  new Uint8Array(mem.buffer, wOff, wbytes.length).set(wbytes);
  quantizeAct(x, new Int8Array(mem.buffer, xqOff, cols), new Float32Array(mem.buffer, xdOff, abl));
  instantiateKernels(mem)[name](wOff, xqOff, xdOff, yOff, rows, nb);
  return Float32Array.from(new Float32Array(mem.buffer, yOff, rows));
}

for (const [label, type, blockElems, kname] of CASES) {
  test(`${label} SIMD kernel matches the JS reference (row counts around the 2-row group, several block counts)`, () => {
    const rand = mulberry32(13);
    for (const rows of [1, 2, 3, 4, 5, 8, 11]) {
      for (const nb of [1, 2, 3, 6]) {
        const cols = nb * blockElems;
        const wbytes = randomQuantBlocks(type, rows * nb, rand);
        const x = Float32Array.from({ length: cols }, () => gaussian(rand));
        const got = runKernel(kname, type, blockElems, rows, nb, wbytes, x);
        const want = matvec({ type, rows, cols, data: wbytes }, new Act(x), new Float32Array(rows));
        const rms = Math.sqrt(want.reduce((a, v) => a + v * v, 0) / rows) || 1;
        for (let r = 0; r < rows; r++) assert.ok(Math.abs(got[r] - want[r]) <= 1e-4 * rms, `${label} rows=${rows} nb=${nb} row ${r}: got ${got[r]} want ${want[r]} (rms ${rms})`);
      }
    }
  });

  test(`${label} SIMD kernel: extreme bytes (all 0x00, all 0xFF, extreme f16 scales) stay finite and agree`, () => {
    const rows = 3; const nb = 2; const cols = nb * blockElems;
    const info = rowBytes(type, cols) / nb; // bytes per block
    for (const fill of [0x00, 0xff, 0x80, 0x7f]) {
      const wbytes = new Uint8Array(rows * nb * info).fill(fill);
      // make the scale fields sane (all-0xFF f16 is NaN): d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
      for (let b = 0; b < rows * nb; b++) {
        const o = b * info;
        const put = (off, v) => { wbytes[o + off] = v & 255; wbytes[o + off + 1] = v >> 8; };
        if (type === GGML_TYPE.Q6_K) put(208, 0x3c00); else { put(0, 0x3c00); if (type !== GGML_TYPE.Q5_0) put(2, 0x3800); }
      }
      const x = Float32Array.from({ length: cols }, (_, i) => ((i * 7919) % 13) - 6);
      const got = runKernel(kname, type, blockElems, rows, nb, wbytes, x);
      const want = matvec({ type, rows, cols, data: wbytes }, new Act(x), new Float32Array(rows));
      const rms = Math.sqrt(want.reduce((a, v) => a + v * v, 0) / rows) || 1;
      got.forEach((g, r) => { assert.ok(Number.isFinite(g)); assert.ok(Math.abs(g - want[r]) <= 1e-4 * rms + 1e-6, `fill 0x${fill.toString(16)} row ${r}: ${g} vs ${want[r]}`); });
    }
  });
}
