// Plain-JS reference kernels. Slow but obviously correct: this is the oracle for every optimized kernel.
// Weight matrices are row-major: `rows` output rows, each `cols` (ne0) inputs in whole quant blocks.
import { GGML_TYPE } from '../gguf/types.js';
import { f16Table, bf16ToF32 } from '../util/f16.js';
import { dequantizeQ5_0, dequantizeQ4_K, dequantizeQ5_K, dequantizeQ6_K, unpackScaleMinK4 } from './quant.js';

const QK = 32;
// scratch for the K-quant matvecs (single-threaded use per isolate: workers each have their own module instance)
const K_SC = new Int32Array(8); const K_MN = new Int32Array(8); const K_G16 = new Float64Array(16);

/**
 * A weight matrix resident in memory.
 * @typedef {Object} Weight
 * @property {number} type    ggml type id
 * @property {number} rows
 * @property {number} cols
 * @property {Uint8Array} data  raw bytes (rows * rowBytes)
 */

/**
 * Activation vector with a lazily-built Q8_0-style quantization (f32 scale per 32 elements, int8 values),
 * shared by all matmuls that consume it.
 */
export class Act {
  /** @param {Float32Array} x */
  constructor(x) { this.x = x; this.q = null; this.d = null; this.s = null; }
  quant() {
    if (!this.q) {
      const n = this.x.length;
      this.q = new Int8Array(n);
      this.d = new Float32Array(n / QK);
      this.s = new Int32Array(n / QK); // integer sum of each block's quants (Q4_K / Q5_K min correction)
      quantizeAct(this.x, this.q, this.d, this.s);
    }
    return this;
  }
}

/**
 * Q8_0-style activation quantization with f32 scales. d = amax/127, q = round(x/d).
 * Optionally also writes the integer sum of each block's quants into `s` (used by the Q4_K/Q5_K min correction).
 * @param {Float32Array} x @param {Int8Array|Int16Array} q @param {Float32Array} d @param {Float32Array} [s]
 */
export function quantizeAct(x, q, d, s) {
  const nb = x.length / QK;
  for (let b = 0; b < nb; b++) {
    const o = b * QK;
    let amax = 0;
    for (let i = 0; i < QK; i++) { const a = Math.abs(x[o + i]); if (a > amax) amax = a; }
    const scale = amax / 127;
    d[b] = scale;
    const inv = scale ? 1 / scale : 0;
    let sum = 0;
    for (let i = 0; i < QK; i++) {
      const v = x[o + i] * inv;
      const r = v < 0 ? -Math.round(-v) : Math.round(v);
      q[o + i] = r;
      sum += r;
    }
    if (s) s[b] = sum;
  }
}

/** Bytes per row of a weight with `cols` inputs. */
export function rowBytes(type, cols) {
  switch (type) {
    case GGML_TYPE.F32: return cols * 4;
    case GGML_TYPE.F16: case GGML_TYPE.BF16: return cols * 2;
    case GGML_TYPE.Q8_0: return (cols / QK) * 34;
    case GGML_TYPE.Q4_0: return (cols / QK) * 18;
    case GGML_TYPE.Q5_0: return (cols / QK) * 22;
    case GGML_TYPE.Q4_K: return (cols / 256) * 144;
    case GGML_TYPE.Q5_K: return (cols / 256) * 176;
    case GGML_TYPE.Q6_K: return (cols / 256) * 210;
    default: throw new Error(`reference kernels: unsupported tensor type ${type} (supported: F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K)`);
  }
}

/** Dequantize one row of a weight into `out` (length cols). */
export function dequantRow(w, row, out) {
  const rb = rowBytes(w.type, w.cols);
  const o0 = row * rb;
  const u8 = w.data;
  const tbl = f16Table();
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  switch (w.type) {
    case GGML_TYPE.F32:
      for (let i = 0; i < w.cols; i++) out[i] = dv.getFloat32(o0 + i * 4, true);
      return out;
    case GGML_TYPE.F16:
      for (let i = 0; i < w.cols; i++) out[i] = tbl[dv.getUint16(o0 + i * 2, true)];
      return out;
    case GGML_TYPE.BF16:
      for (let i = 0; i < w.cols; i++) out[i] = bf16ToF32(dv.getUint16(o0 + i * 2, true));
      return out;
    case GGML_TYPE.Q8_0:
      for (let b = 0; b < w.cols / QK; b++) {
        const o = o0 + b * 34;
        const d = tbl[u8[o] | (u8[o + 1] << 8)];
        for (let i = 0; i < QK; i++) out[b * QK + i] = d * dv.getInt8(o + 2 + i);
      }
      return out;
    case GGML_TYPE.Q4_0:
      for (let b = 0; b < w.cols / QK; b++) {
        const o = o0 + b * 18;
        const d = tbl[u8[o] | (u8[o + 1] << 8)];
        for (let j = 0; j < 16; j++) {
          const byte = u8[o + 2 + j];
          out[b * QK + j] = d * ((byte & 0x0f) - 8);
          out[b * QK + j + 16] = d * ((byte >> 4) - 8);
        }
      }
      return out;
    case GGML_TYPE.Q5_0: out.set(dequantizeQ5_0(u8.subarray(o0, o0 + rb), w.cols)); return out;
    case GGML_TYPE.Q4_K: out.set(dequantizeQ4_K(u8.subarray(o0, o0 + rb), w.cols)); return out;
    case GGML_TYPE.Q5_K: out.set(dequantizeQ5_K(u8.subarray(o0, o0 + rb), w.cols)); return out;
    case GGML_TYPE.Q6_K: out.set(dequantizeQ6_K(u8.subarray(o0, o0 + rb), w.cols)); return out;
    default: throw new Error(`reference kernels: unsupported tensor type ${w.type}`);
  }
}

/**
 * y[r] = dot(W[r], x) for r in [r0, r1).
 * Quantized weights use the int8 x int8 -> int32 block dot with float scales; float weights use f32 x.
 * @param {Weight} w @param {Act} a @param {Float32Array} y
 */
export function matvec(w, a, y, r0 = 0, r1 = w.rows) {
  const u8 = w.data;
  const cols = w.cols;
  const nb = cols / QK;
  switch (w.type) {
    case GGML_TYPE.Q8_0: {
      a.quant();
      const xq = a.q; const xd = a.d;
      const tbl = f16Table();
      const i8 = new Int8Array(u8.buffer, u8.byteOffset, u8.byteLength);
      for (let r = r0; r < r1; r++) {
        let o = r * nb * 34;
        let acc = 0;
        for (let b = 0; b < nb; b++) {
          const xo = b * QK;
          let s = 0;
          for (let i = 0; i < QK; i += 4) {
            s += i8[o + 2 + i] * xq[xo + i] + i8[o + 3 + i] * xq[xo + i + 1] + i8[o + 4 + i] * xq[xo + i + 2] + i8[o + 5 + i] * xq[xo + i + 3];
          }
          acc += s * tbl[u8[o] | (u8[o + 1] << 8)] * xd[b];
          o += 34;
        }
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.Q4_0: {
      a.quant();
      const xq = a.q; const xd = a.d;
      const tbl = f16Table();
      for (let r = r0; r < r1; r++) {
        let o = r * nb * 18;
        let acc = 0;
        for (let b = 0; b < nb; b++) {
          const xo = b * QK;
          let s = 0;
          for (let j = 0; j < 16; j++) {
            const byte = u8[o + 2 + j];
            s += ((byte & 0x0f) - 8) * xq[xo + j] + ((byte >> 4) - 8) * xq[xo + j + 16];
          }
          acc += s * tbl[u8[o] | (u8[o + 1] << 8)] * xd[b];
          o += 18;
        }
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.Q5_0: {
      a.quant();
      const xq = a.q; const xd = a.d;
      const tbl = f16Table();
      for (let r = r0; r < r1; r++) {
        let o = r * nb * 22;
        let acc = 0;
        for (let b = 0; b < nb; b++) {
          const xo = b * QK;
          const qh = u8[o + 2] | (u8[o + 3] << 8) | (u8[o + 4] << 16) | (u8[o + 5] << 24);
          let s = 0;
          for (let j = 0; j < 16; j++) {
            const byte = u8[o + 6 + j];
            s += (((byte & 0x0f) | (((qh >>> j) & 1) << 4)) - 16) * xq[xo + j] + (((byte >> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16) * xq[xo + j + 16];
          }
          acc += s * tbl[u8[o] | (u8[o + 1] << 8)] * xd[b];
          o += 22;
        }
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.Q4_K: case GGML_TYPE.Q5_K: {
      // per 32-element sub-block s:  sum_i (d*sc*q_i - dmin*mn) * x_i  =  d*sc*sum(q*xq) - dmin*mn*sum(xq), times the x scale
      // (no allocation inside the loops: scales are unpacked into module-level scratch arrays; sum(xq) comes with the activation)
      a.quant();
      const xq = a.q; const xd = a.d; const xs = a.s;
      const tbl = f16Table();
      const five = w.type === GGML_TYPE.Q5_K;
      const bs = five ? 176 : 144; const qs = five ? 48 : 16;
      const nsb = cols / 256;
      const sc = K_SC; const mn = K_MN;
      for (let r = r0; r < r1; r++) {
        let acc = 0;
        for (let sb = 0; sb < nsb; sb++) {
          const o = (r * nsb + sb) * bs;
          const d = tbl[u8[o] | (u8[o + 1] << 8)]; const dmin = tbl[u8[o + 2] | (u8[o + 3] << 8)];
          for (let j = 0; j < 4; j++) { // unpackScaleMinK4
            const a0 = u8[o + 4 + j]; const a1 = u8[o + 8 + j]; const a2 = u8[o + 12 + j];
            sc[j] = a0 & 63; mn[j] = a1 & 63;
            sc[j + 4] = (a2 & 0x0f) | ((a0 >> 6) << 4);
            mn[j + 4] = (a2 >> 4) | ((a1 >> 6) << 4);
          }
          for (let c = 0; c < 4; c++) { // 64 values per chunk: sub-block 2c in the low nibbles, 2c+1 in the high nibbles
            const qo = o + qs + c * 32; const xo = sb * 256 + c * 64;
            let lo = 0; let hi = 0;
            if (five) {
              const sh = 2 * c;
              for (let l = 0; l < 32; l++) {
                const byte = u8[qo + l]; const h5 = u8[o + 16 + l];
                lo += ((byte & 0x0f) | (((h5 >> sh) & 1) << 4)) * xq[xo + l];
                hi += ((byte >> 4) | (((h5 >> (sh + 1)) & 1) << 4)) * xq[xo + 32 + l];
              }
            } else {
              for (let l = 0; l < 32; l++) {
                const byte = u8[qo + l];
                lo += (byte & 0x0f) * xq[xo + l];
                hi += (byte >> 4) * xq[xo + 32 + l];
              }
            }
            const b0 = sb * 8 + 2 * c;
            acc += xd[b0] * (d * sc[2 * c] * lo - dmin * mn[2 * c] * xs[b0]);
            acc += xd[b0 + 1] * (d * sc[2 * c + 1] * hi - dmin * mn[2 * c + 1] * xs[b0 + 1]);
          }
        }
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.Q6_K: {
      a.quant();
      const xq = a.q; const xd = a.d;
      const tbl = f16Table();
      const nsb = cols / 256;
      const gsum = K_G16;
      for (let r = r0; r < r1; r++) {
        let acc = 0;
        for (let sb = 0; sb < nsb; sb++) {
          const o = (r * nsb + sb) * 210;
          const d = tbl[u8[o + 208] | (u8[o + 209] << 8)];
          for (let half = 0; half < 2; half++) {
            const ql = o + 64 * half; const qh = o + 128 + 32 * half;
            const e = sb * 256 + half * 128;
            for (let g = 0; g < 2; g++) { // 16 lanes each; four 6-bit values per lane (positions l, l+32, l+64, l+96)
              let a0 = 0; let a1 = 0; let a2 = 0; let a3 = 0;
              for (let l = g * 16; l < g * 16 + 16; l++) {
                const h = u8[qh + l]; const lo0 = u8[ql + l]; const lo1 = u8[ql + l + 32];
                a0 += (((lo0 & 0x0f) | ((h & 3) << 4)) - 32) * xq[e + l];
                a1 += (((lo1 & 0x0f) | (((h >> 2) & 3) << 4)) - 32) * xq[e + l + 32];
                a2 += (((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32) * xq[e + l + 64];
                a3 += (((lo1 >> 4) | (((h >> 6) & 3) << 4)) - 32) * xq[e + l + 96];
              }
              gsum[half * 8 + g] = a0; gsum[half * 8 + 2 + g] = a1; gsum[half * 8 + 4 + g] = a2; gsum[half * 8 + 6 + g] = a3;
            }
          }
          for (let g = 0; g < 16; g++) {
            const scale = (u8[o + 192 + g] << 24) >> 24; // int8
            acc += xd[sb * 8 + (g >> 1)] * d * scale * gsum[g];
          }
        }
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.F32: {
      const f = new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
      const x = a.x;
      for (let r = r0; r < r1; r++) {
        let acc = 0;
        const o = r * cols;
        for (let i = 0; i < cols; i++) acc += f[o + i] * x[i];
        y[r] = acc;
      }
      return y;
    }
    case GGML_TYPE.F16: case GGML_TYPE.BF16: {
      const x = a.x;
      const tmp = new Float32Array(cols);
      for (let r = r0; r < r1; r++) {
        dequantRow(w, r, tmp);
        let acc = 0;
        for (let i = 0; i < cols; i++) acc += tmp[i] * x[i];
        y[r] = acc;
      }
      return y;
    }
    default: throw new Error(`reference kernels: unsupported tensor type ${w.type}`);
  }
}

/** out = x / sqrt(mean(x^2) + eps) * weight. `out` may alias `x`. */
export function rmsnorm(out, x, weight, eps, n = x.length, xo = 0, oo = 0) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[xo + i] * x[xo + i];
  const s = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) out[oo + i] = x[xo + i] * s * weight[i];
}

/** In-place numerically stable softmax over x[0..n). */
export function softmax(x, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(x[i] - max); x[i] = e; sum += e; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) x[i] *= inv;
}

export const silu = (z) => z / (1 + Math.exp(-z));

/**
 * Precompute RoPE cos/sin tables: tables[pos * half + i] for i in [0, nRot/2).
 * theta_i = pos * base^(-2i/nRot) / freqFactor[i]
 * @param {{nRot: number, ropeBase: number}} cfg @param {number} ctx @param {Float32Array|null} freqFactors
 */
export function ropeTables(cfg, ctx, freqFactors = null) {
  const half = cfg.nRot / 2;
  const cos = new Float32Array(ctx * half);
  const sin = new Float32Array(ctx * half);
  for (let i = 0; i < half; i++) {
    let freq = cfg.ropeBase ** (-2 * i / cfg.nRot);
    if (freqFactors) freq /= freqFactors[i];
    for (let p = 0; p < ctx; p++) {
      cos[p * half + i] = Math.cos(p * freq);
      sin[p * half + i] = Math.sin(p * freq);
    }
  }
  return { cos, sin, half };
}

/**
 * Apply RoPE in place to `nHeads` heads of size headDim stored contiguously in v.
 * 'neox': rotate pairs (i, i + nRot/2); 'norm': rotate adjacent pairs (2i, 2i+1). Dims >= nRot are untouched.
 * Heads [h0, nHeads) are rotated (h0 lets a worker handle just its own head range).
 */
export function applyRope(v, nHeads, headDim, nRot, type, tabs, pos, h0 = 0) {
  const half = tabs.half;
  const to = pos * half;
  for (let h = h0; h < nHeads; h++) {
    const o = h * headDim;
    for (let i = 0; i < half; i++) {
      const c = tabs.cos[to + i]; const s = tabs.sin[to + i];
      const i0 = type === 'neox' ? o + i : o + 2 * i;
      const i1 = type === 'neox' ? o + i + half : o + 2 * i + 1;
      const a = v[i0]; const b = v[i1];
      v[i0] = a * c - b * s;
      v[i1] = a * s + b * c;
    }
  }
}
