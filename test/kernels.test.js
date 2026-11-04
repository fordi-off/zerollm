import test from 'node:test';
import assert from 'node:assert/strict';
import { buildKernelModule, instantiateKernels, Q8_0_BLOCK, Q4_0_BLOCK } from '../src/compute/wasm/kernels.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { quantizeAct, matvec, Act, dequantRow } from '../src/compute/ref-kernels.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { f32ToF16 } from '../src/util/f16.js';

test('kernel module validates', () => {
  assert.ok(WebAssembly.validate(buildKernelModule()));
});

/** Layout inside one memory: [weights][xq][xd][y] with 64-byte alignment. */
function setup(type, rows, nb, weightBytes) {
  const blk = type === GGML_TYPE.Q8_0 ? Q8_0_BLOCK : Q4_0_BLOCK;
  const wLen = rows * nb * blk;
  const al = (v) => Math.ceil(v / 64) * 64;
  const wOff = 64; const xqOff = al(wOff + wLen); const xdOff = al(xqOff + nb * 32); const yOff = al(xdOff + nb * 4);
  const bytes = yOff + rows * 4 + 64;
  const mem = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) });
  const k = instantiateKernels(mem);
  new Uint8Array(mem.buffer, wOff, wLen).set(weightBytes);
  return { mem, k, wOff, xqOff, xdOff, yOff };
}

function run(type, rows, nb, weightBytes, x) {
  const s = setup(type, rows, nb, weightBytes);
  const xq = new Int8Array(s.mem.buffer, s.xqOff, nb * 32); const xd = new Float32Array(s.mem.buffer, s.xdOff, nb);
  quantizeAct(x, xq, xd);
  (type === GGML_TYPE.Q8_0 ? s.k.matvec_q8_0 : s.k.matvec_q4_0)(s.wOff, s.xqOff, s.xdOff, s.yOff, rows, nb);
  const got = Float32Array.from(new Float32Array(s.mem.buffer, s.yOff, rows));
  // reference: the plain JS kernel on the same bytes and the same activation
  const w = { type, rows, cols: nb * 32, data: Uint8Array.from(weightBytes) };
  const want = matvec(w, new Act(x), new Float32Array(rows));
  return { got, want };
}

function assertClose(got, want, label) {
  const rms = Math.sqrt(want.reduce((a, v) => a + v * v, 0) / want.length) || 1;
  for (let i = 0; i < got.length; i++) assert.ok(Math.abs(got[i] - want[i]) <= 3e-5 * rms + 1e-30, `${label} row ${i}: got ${got[i]} want ${want[i]} (rms ${rms})`);
}

for (const [name, type, quantize] of [['Q8_0', GGML_TYPE.Q8_0, quantizeQ8_0], ['Q4_0', GGML_TYPE.Q4_0, quantizeQ4_0]]) {
  test(`${name} kernel matches the JS reference: row counts around the 4-row group, many block counts`, () => {
    const rand = mulberry32(7);
    for (const rows of [1, 2, 3, 4, 5, 7, 8, 13, 64]) {
      for (const nb of [1, 2, 3, 8, 17, 32]) {
        const wf = Float32Array.from({ length: rows * nb * 32 }, () => gaussian(rand) * 0.4);
        const x = Float32Array.from({ length: nb * 32 }, () => gaussian(rand));
        const { got, want } = run(type, rows, nb, quantize(wf), x);
        assertClose(got, want, `${name} rows=${rows} nb=${nb}`);
      }
    }
  });

  test(`${name} kernel: adversarial raw bytes (extreme quants, negative / denormal / tiny / huge f16 scales)`, () => {
    const rand = mulberry32(31);
    const blk = type === GGML_TYPE.Q8_0 ? Q8_0_BLOCK : Q4_0_BLOCK;
    const rows = 9; const nb = 6;
    const scales = [1, -1, 2 ** -24, -(2 ** -24), 2 ** -14, 3 ** 7, -1000, 0.0123, 0];
    const bytes = new Uint8Array(rows * nb * blk);
    const dv = new DataView(bytes.buffer);
    for (let i = 0; i < rows * nb; i++) {
      const o = i * blk;
      dv.setUint16(o, f32ToF16(scales[Math.floor(rand() * scales.length)]), true);
      for (let j = 2; j < blk; j++) bytes[o + j] = Math.floor(rand() * 256); // includes -128 quants and 0xFF/0x00 nibbles
    }
    // all-extreme blocks: every quant -128, activation +-127
    for (let j = 2; j < blk; j++) bytes[j] = type === GGML_TYPE.Q8_0 ? 0x80 : 0x00;
    const x = Float32Array.from({ length: nb * 32 }, (_, i) => (i % 2 ? 1 : -1) * 5);
    const { got, want } = run(type, rows, nb, bytes, x);
    assertClose(got, want, `${name} adversarial`);
    assert.ok(got.every(Number.isFinite));
  });

  test(`${name} kernel output also matches an independent dequantize-then-dot (within activation-quantization error)`, () => {
    const rand = mulberry32(5);
    const rows = 6; const nb = 4; const cols = nb * 32;
    const wf = Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.3);
    const x = Float32Array.from({ length: cols }, () => gaussian(rand));
    const wb = quantize(wf);
    const { got } = run(type, rows, nb, wb, x);
    const w = { type, rows, cols, data: wb };
    const row = new Float32Array(cols);
    const want = Array.from({ length: rows }, (_, r) => { dequantRow(w, r, row); return row.reduce((a, v, i) => a + v * x[i], 0); });
    const rms = Math.sqrt(want.reduce((a, v) => a + v * v, 0) / rows);
    got.forEach((g, r) => assert.ok(Math.abs(g - want[r]) < 0.02 * rms, `row ${r}: ${g} vs ${want[r]}`));
  });
}
