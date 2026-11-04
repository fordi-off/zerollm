// IEEE half-precision conversion helpers.

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

/** f32 -> f16 bits, round-to-nearest-even. @param {number} x */
export function f32ToF16(x) {
  f32buf[0] = x;
  const b = u32buf[0];
  const sign = (b >>> 16) & 0x8000;
  let exp = (b >>> 23) & 0xff;
  const mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return sign; // underflow -> 0
    const m = mant | 0x800000;
    const shift = 14 - exp;
    let half = m >>> shift;
    const rem = m & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half++;
    return sign | half;
  }
  let half = sign | (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++; // carry may bump exponent: correct
  return half;
}

/** f16 bits -> f32. @param {number} h */
export function f16ToF32Slow(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Lookup table of all 65536 half values as f32 (256 KB), built lazily. @type {Float32Array|null} */
let table = null;
export function f16Table() {
  if (!table) {
    table = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) table[i] = f16ToF32Slow(i);
  }
  return table;
}

/** @param {number} h */
export function f16ToF32(h) { return f16Table()[h]; }

/** bf16 bits -> f32. @param {number} h */
export function bf16ToF32(h) { u32buf[0] = h << 16; return f32buf[0]; }
