import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { dequantizeQ5_0, dequantizeQ4_K, dequantizeQ5_K, dequantizeQ6_K, unpackScaleMinK4 } from '../src/compute/quant.js';

// Golden vectors produced by the independent gguf-py implementation (tools/gen-quant-vectors.py).
const vectors = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), 'data', 'quant-vectors.json'), 'utf8'));
const decode = (b64) => Uint8Array.from(Buffer.from(b64, 'base64'));

for (const [name, fn] of [['Q5_0', dequantizeQ5_0], ['Q4_K', dequantizeQ4_K], ['Q5_K', dequantizeQ5_K], ['Q6_K', dequantizeQ6_K]]) {
  test(`${name} dequantization matches gguf-py golden vectors (random + extreme blocks, odd f16 scales)`, () => {
    const v = vectors[name];
    const raw = decode(v.raw);
    const want = new Float32Array(decode(v.expected).buffer);
    const rowBytes = raw.length / v.rows;
    for (let r = 0; r < v.rows; r++) {
      const got = fn(raw.subarray(r * rowBytes, (r + 1) * rowBytes), v.cols);
      for (let i = 0; i < v.cols; i++) {
        const w = want[r * v.cols + i];
        // gguf-py computes in float32, we compute in float64 and round: allow 1 float32 ulp-ish
        assert.ok(Object.is(got[i], w) || Math.abs(got[i] - w) <= 2e-6 * Math.abs(w) + 1e-30 || (Number.isNaN(w) && Number.isNaN(got[i])),
          `${name} row ${r} elem ${i}: got ${got[i]} want ${w}`);
      }
    }
  });
}

test('unpackScaleMinK4 round-trips a hand-packed table', () => {
  const sc = [1, 63, 32, 17, 0, 45, 63, 9]; const mn = [63, 0, 5, 40, 33, 2, 61, 50];
  const q = new Uint8Array(12);
  for (let j = 0; j < 4; j++) {
    q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6);
    q[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6);
    q[j + 8] = (sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4);
  }
  const u = unpackScaleMinK4(q);
  assert.deepEqual(u.sc, sc); assert.deepEqual(u.mn, mn);
});
