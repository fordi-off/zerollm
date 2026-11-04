// Quantization helpers (Q8_0). Also used for quantizing activations in the reference path.
import { f32ToF16, f16ToF32 } from '../util/f16.js';

export const QK8_0 = 32;
export const Q8_0_BLOCK_BYTES = 34;

/** round half away from zero (matches C roundf) */
const roundf = (v) => (v < 0 ? -Math.round(-v) : Math.round(v));

/**
 * Quantize f32 values to Q8_0 blocks (f16 scale + 32 int8), ggml-compatible layout.
 * @param {Float32Array} x length must be a multiple of 32
 * @returns {Uint8Array}
 */
export function quantizeQ8_0(x) {
  if (x.length % QK8_0) throw new Error(`quantizeQ8_0: length ${x.length} not a multiple of ${QK8_0}`);
  const nb = x.length / QK8_0;
  const out = new Uint8Array(nb * Q8_0_BLOCK_BYTES);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = 0; i < QK8_0; i++) amax = Math.max(amax, Math.abs(x[b * QK8_0 + i]));
    const d = amax / 127;
    const id = d ? 1 / d : 0;
    const o = b * Q8_0_BLOCK_BYTES;
    dv.setUint16(o, f32ToF16(d), true);
    for (let i = 0; i < QK8_0; i++) dv.setInt8(o + 2 + i, roundf(x[b * QK8_0 + i] * id));
  }
  return out;
}

/** Dequantize Q8_0 blocks to f32. @param {Uint8Array} q @param {number} n element count */
export function dequantizeQ8_0(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / QK8_0; b++) {
    const o = b * Q8_0_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o, true));
    for (let i = 0; i < QK8_0; i++) out[b * QK8_0 + i] = d * dv.getInt8(o + 2 + i);
  }
  return out;
}
