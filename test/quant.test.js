import test from 'node:test';
import assert from 'node:assert/strict';
import { quantizeQ8_0, dequantizeQ8_0, Q8_0_BLOCK_BYTES } from '../src/compute/quant.js';
import { f16ToF32 } from '../src/util/f16.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

test('Q8_0 layout: f16 scale then 32 int8, w = d * q', () => {
  const x = new Float32Array(32);
  for (let i = 0; i < 32; i++) x[i] = i - 16; // amax = 16
  const q = quantizeQ8_0(x);
  assert.equal(q.length, Q8_0_BLOCK_BYTES);
  const dv = new DataView(q.buffer);
  const d = f16ToF32(dv.getUint16(0, true));
  assert.ok(Math.abs(d - 16 / 127) < 1e-4);
  assert.equal(dv.getInt8(2 + 0), -127); // -16 maps to -127
  assert.equal(dv.getInt8(2 + 16), 0);
});

test('Q8_0 round trip error is bounded by half a step', () => {
  const rand = mulberry32(7);
  const x = new Float32Array(32 * 50).map(() => gaussian(rand));
  const y = dequantizeQ8_0(quantizeQ8_0(x), x.length);
  for (let b = 0; b < 50; b++) {
    let amax = 0;
    for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(x[b * 32 + i]));
    const step = amax / 127;
    for (let i = 0; i < 32; i++) assert.ok(Math.abs(x[b * 32 + i] - y[b * 32 + i]) <= step * 0.51 + 1e-3 * amax);
  }
});

test('all-zero block', () => {
  const y = dequantizeQ8_0(quantizeQ8_0(new Float32Array(32)), 32);
  assert.deepEqual([...y], new Array(32).fill(0));
});

test('length must be a multiple of 32', () => {
  assert.throws(() => quantizeQ8_0(new Float32Array(33)));
});
