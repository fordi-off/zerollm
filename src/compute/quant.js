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

export const QK4_0 = 32;
export const Q4_0_BLOCK_BYTES = 18;

/**
 * Quantize f32 to Q4_0 (ggml layout): f16 scale d, then 16 bytes; byte j holds element j (low nibble) and j+16 (high nibble).
 * The value with the largest magnitude maps to -8.  w = d * (nibble - 8)
 * @param {Float32Array} x
 */
export function quantizeQ4_0(x) {
  if (x.length % QK4_0) throw new Error(`quantizeQ4_0: length ${x.length} not a multiple of ${QK4_0}`);
  const nb = x.length / QK4_0;
  const out = new Uint8Array(nb * Q4_0_BLOCK_BYTES);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0; let max = 0;
    for (let i = 0; i < QK4_0; i++) {
      const v = x[b * QK4_0 + i];
      if (Math.abs(v) > amax) { amax = Math.abs(v); max = v; }
    }
    const d = max / -8;
    const id = d ? 1 / d : 0;
    const o = b * Q4_0_BLOCK_BYTES;
    dv.setUint16(o, f32ToF16(d), true);
    for (let j = 0; j < 16; j++) {
      const lo = Math.min(15, Math.trunc(x[b * QK4_0 + j] * id + 8.5));
      const hi = Math.min(15, Math.trunc(x[b * QK4_0 + j + 16] * id + 8.5));
      out[o + 2 + j] = lo | (hi << 4);
    }
  }
  return out;
}

/** @param {Uint8Array} q @param {number} n */
export function dequantizeQ4_0(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / QK4_0; b++) {
    const o = b * Q4_0_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o, true));
    for (let j = 0; j < 16; j++) {
      const byte = q[o + 2 + j];
      out[b * QK4_0 + j] = d * ((byte & 0x0f) - 8);
      out[b * QK4_0 + j + 16] = d * ((byte >> 4) - 8);
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ Q5_0 and K-quants
// Straightforward block dequantizers (the formulas of ggml's dequantize_row_*), validated against golden vectors from
// the independent gguf-py implementation (test/data/quant-vectors.json, tools/gen-quant-vectors.py).

export const QK_K = 256;
export const Q5_0_BLOCK_BYTES = 22;
export const Q4_K_BLOCK_BYTES = 144;
export const Q5_K_BLOCK_BYTES = 176;
export const Q6_K_BLOCK_BYTES = 210;

/**
 * Unpack the 12-byte 6-bit scale/min table of Q4_K / Q5_K into 8 scales and 8 mins.
 *   bytes 0-3: low 6 bits = scales 0-3, top 2 bits = bits 4-5 of scales 4-7
 *   bytes 4-7: low 6 bits = mins   0-3, top 2 bits = bits 4-5 of mins   4-7
 *   bytes 8-11: low nibble = low 4 bits of scales 4-7, high nibble = low 4 bits of mins 4-7
 * @param {Uint8Array} q 12 bytes @returns {{sc: number[], mn: number[]}}
 */
export function unpackScaleMinK4(q) {
  const sc = new Array(8); const mn = new Array(8);
  for (let j = 0; j < 4; j++) {
    sc[j] = q[j] & 63;
    mn[j] = q[j + 4] & 63;
    sc[j + 4] = (q[j + 8] & 0x0f) | ((q[j] >> 6) << 4);
    mn[j + 4] = (q[j + 8] >> 4) | ((q[j + 4] >> 6) << 4);
  }
  return { sc, mn };
}

/** @param {Uint8Array} q @param {number} n element count (multiple of 32) */
export function dequantizeQ5_0(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / 32; b++) {
    const o = b * Q5_0_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o, true));
    const qh = dv.getUint32(o + 2, true);
    for (let j = 0; j < 16; j++) {
      const byte = q[o + 6 + j];
      out[b * 32 + j] = d * (((byte & 0x0f) | (((qh >>> j) & 1) << 4)) - 16);
      out[b * 32 + j + 16] = d * (((byte >> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16);
    }
  }
  return out;
}

/** @param {Uint8Array} q @param {number} n element count (multiple of 256) */
export function dequantizeQ4_K(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / QK_K; b++) {
    const o = b * Q4_K_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o, true)); const dmin = f16ToF32(dv.getUint16(o + 2, true));
    const { sc, mn } = unpackScaleMinK4(q.subarray(o + 4, o + 16));
    for (let s = 0; s < 8; s++) {
      const chunk = s >> 1; const hi = s & 1;
      for (let l = 0; l < 32; l++) {
        const byte = q[o + 16 + chunk * 32 + l];
        const v = hi ? byte >> 4 : byte & 0x0f;
        out[b * QK_K + s * 32 + l] = d * sc[s] * v - dmin * mn[s];
      }
    }
  }
  return out;
}

/** @param {Uint8Array} q @param {number} n element count (multiple of 256) */
export function dequantizeQ5_K(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / QK_K; b++) {
    const o = b * Q5_K_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o, true)); const dmin = f16ToF32(dv.getUint16(o + 2, true));
    const { sc, mn } = unpackScaleMinK4(q.subarray(o + 4, o + 16));
    for (let s = 0; s < 8; s++) {
      const chunk = s >> 1; const hi = s & 1;
      for (let l = 0; l < 32; l++) {
        const byte = q[o + 48 + chunk * 32 + l];
        const v = (hi ? byte >> 4 : byte & 0x0f) | (((q[o + 16 + l] >> s) & 1) << 4);
        out[b * QK_K + s * 32 + l] = d * sc[s] * v - dmin * mn[s];
      }
    }
  }
  return out;
}

/** @param {Uint8Array} q @param {number} n element count (multiple of 256) */
export function dequantizeQ6_K(q, n) {
  const out = new Float32Array(n);
  const dv = new DataView(q.buffer, q.byteOffset, q.byteLength);
  for (let b = 0; b < n / QK_K; b++) {
    const o = b * Q6_K_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(o + 208, true));
    for (let half = 0; half < 2; half++) {
      const ql = o + 64 * half; const qh = o + 128 + 32 * half; const sc = o + 192 + 8 * half;
      for (let l = 0; l < 32; l++) {
        const h = q[qh + l];
        const q1 = ((q[ql + l] & 0x0f) | (((h >> 0) & 3) << 4)) - 32;
        const q2 = ((q[ql + l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) - 32;
        const q3 = ((q[ql + l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
        const q4 = ((q[ql + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
        const g = l >> 4; // 16-element scale group within the 32 lanes
        const e = b * QK_K + half * 128;
        out[e + l] = d * dv.getInt8(sc + g) * q1;
        out[e + l + 32] = d * dv.getInt8(sc + g + 2) * q2;
        out[e + l + 64] = d * dv.getInt8(sc + g + 4) * q3;
        out[e + l + 96] = d * dv.getInt8(sc + g + 6) * q4;
      }
    }
  }
  return out;
}
