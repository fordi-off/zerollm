// GGUF / ggml constants and size helpers.

/** GGUF metadata value types. */
export const GGUF_TYPE = Object.freeze({
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6,
  BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
});

/** ggml tensor type ids. */
export const GGML_TYPE = Object.freeze({
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ1_S: 19, IQ4_NL: 20, IQ3_S: 21, IQ2_S: 22, IQ4_XS: 23,
  I8: 24, I16: 25, I32: 26, I64: 27, F64: 28, IQ1_M: 29, BF16: 30,
});

/**
 * blockSize = elements per block, typeSize = bytes per block.
 * Sizes follow ggml's block structs (e.g. Q8_1 = f16 d + f16 s + 32 int8 = 36).
 * Only F32/F16/BF16/Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K are planned to be computable; the rest are listed so
 * `info` can size files that contain them.
 * @type {Record<number, {name: string, blockSize: number, typeSize: number}>}
 */
export const GGML_TYPE_INFO = Object.freeze({
  0: { name: 'F32', blockSize: 1, typeSize: 4 },
  1: { name: 'F16', blockSize: 1, typeSize: 2 },
  2: { name: 'Q4_0', blockSize: 32, typeSize: 18 },
  3: { name: 'Q4_1', blockSize: 32, typeSize: 20 },
  6: { name: 'Q5_0', blockSize: 32, typeSize: 22 },
  7: { name: 'Q5_1', blockSize: 32, typeSize: 24 },
  8: { name: 'Q8_0', blockSize: 32, typeSize: 34 },
  9: { name: 'Q8_1', blockSize: 32, typeSize: 36 },
  10: { name: 'Q2_K', blockSize: 256, typeSize: 84 },
  11: { name: 'Q3_K', blockSize: 256, typeSize: 110 },
  12: { name: 'Q4_K', blockSize: 256, typeSize: 144 },
  13: { name: 'Q5_K', blockSize: 256, typeSize: 176 },
  14: { name: 'Q6_K', blockSize: 256, typeSize: 210 },
  15: { name: 'Q8_K', blockSize: 256, typeSize: 292 },
  16: { name: 'IQ2_XXS', blockSize: 256, typeSize: 66 },
  17: { name: 'IQ2_XS', blockSize: 256, typeSize: 74 },
  18: { name: 'IQ3_XXS', blockSize: 256, typeSize: 98 },
  19: { name: 'IQ1_S', blockSize: 256, typeSize: 50 },
  20: { name: 'IQ4_NL', blockSize: 32, typeSize: 18 },
  21: { name: 'IQ3_S', blockSize: 256, typeSize: 110 },
  22: { name: 'IQ2_S', blockSize: 256, typeSize: 82 },
  23: { name: 'IQ4_XS', blockSize: 256, typeSize: 136 },
  24: { name: 'I8', blockSize: 1, typeSize: 1 },
  25: { name: 'I16', blockSize: 1, typeSize: 2 },
  26: { name: 'I32', blockSize: 1, typeSize: 4 },
  27: { name: 'I64', blockSize: 1, typeSize: 8 },
  28: { name: 'F64', blockSize: 1, typeSize: 8 },
  29: { name: 'IQ1_M', blockSize: 256, typeSize: 56 },
  30: { name: 'BF16', blockSize: 1, typeSize: 2 },
});

/** @param {number} type */
export function typeName(type) { return GGML_TYPE_INFO[type]?.name ?? `type${type}`; }

/** Bytes in one row of `ne0` elements. Throws if ne0 is not a whole number of blocks. */
export function rowSize(type, ne0) {
  const t = GGML_TYPE_INFO[type];
  if (!t) throw new Error(`unknown ggml type id ${type}`);
  if (ne0 % t.blockSize !== 0) throw new Error(`row length ${ne0} is not a multiple of block size ${t.blockSize} for ${t.name}`);
  return (ne0 / t.blockSize) * t.typeSize;
}

/** Total bytes of a tensor with ggml-order dims. @param {number} type @param {number[]} dims */
export function tensorBytes(type, dims) {
  const ne0 = dims[0] ?? 1;
  let rows = 1;
  for (let i = 1; i < dims.length; i++) rows *= dims[i];
  return rowSize(type, ne0) * rows;
}
