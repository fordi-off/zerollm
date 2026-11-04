import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Act, matvec, dequantRow } from '../src/compute/ref-kernels.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

const vectors = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), 'data', 'quant-vectors.json'), 'utf8'));
const TYPES = { Q5_0: GGML_TYPE.Q5_0, Q4_K: GGML_TYPE.Q4_K, Q5_K: GGML_TYPE.Q5_K, Q6_K: GGML_TYPE.Q6_K };

for (const [name, type] of Object.entries(TYPES)) {
  test(`${name}: reference dequantRow == golden vectors; reference matvec == dequantize-then-dot (within activation-quantization error)`, () => {
    const v = vectors[name];
    const data = Uint8Array.from(Buffer.from(v.raw, 'base64'));
    const want = new Float32Array(Uint8Array.from(Buffer.from(v.expected, 'base64')).buffer);
    const w = { type, rows: v.rows, cols: v.cols, data };
    const row = new Float32Array(v.cols);
    for (let r = 0; r < v.rows; r++) {
      dequantRow(w, r, row);
      for (let i = 0; i < v.cols; i++) assert.ok(Object.is(row[i], want[r * v.cols + i]) || Math.abs(row[i] - want[r * v.cols + i]) <= 2e-6 * Math.abs(want[r * v.cols + i]) + 1e-30);
    }
    // vectors contain huge/denormal scales: use a well-scaled well-conditioned copy for the matvec check
    const rand = mulberry32(3);
    const x = Float32Array.from({ length: v.cols }, () => gaussian(rand));
    const bs = data.length / v.rows / (v.cols / (type === GGML_TYPE.Q5_0 ? 32 : 256));
    const tame = Uint8Array.from(data);
    // overwrite every f16 scale with a moderate value so sums are comparable (1.0 and 0.5 as f16)
    const setScale = (off, val) => { tame[off] = val & 255; tame[off + 1] = val >> 8; };
    const nblk = data.length / bs;
    for (let b = 0; b < nblk; b++) {
      if (type === GGML_TYPE.Q6_K) setScale(b * bs + 208, 0x2800);          // 0.03125
      else { setScale(b * bs, 0x2800); if (type !== GGML_TYPE.Q5_0) setScale(b * bs + 2, 0x2400); }
    }
    const tw = { type, rows: v.rows, cols: v.cols, data: tame };
    const got = matvec(tw, new Act(x), new Float32Array(v.rows));
    const wantY = Array.from({ length: v.rows }, (_, r) => { dequantRow(tw, r, row); return row.reduce((a, c, i) => a + c * x[i], 0); });
    const rms = Math.sqrt(wantY.reduce((a, c) => a + c * c, 0) / v.rows);
    got.forEach((g, r) => assert.ok(Math.abs(g - wantY[r]) < 0.02 * rms, `${name} row ${r}: ${g} vs ${wantY[r]} (rms ${rms})`));
  });
}
