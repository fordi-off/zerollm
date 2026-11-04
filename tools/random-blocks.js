// Random raw quantized blocks with controlled f16 scale fields (for tests and tiny synthetic models).
// Weight values come out roughly zero-mean with a std of a few hundredths, like a real model.
import { GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { f32ToF16 } from '../src/util/f16.js';

/** Per-format scale choices that make dequantized weights small and centred. */
const SCALES = {
  [GGML_TYPE.Q5_0]: { d: 0.004 },
  [GGML_TYPE.Q4_K]: { d: 2.5e-4, dminRatio: 7.5 },
  [GGML_TYPE.Q5_K]: { d: 1.3e-4, dminRatio: 15.5 },
  [GGML_TYPE.Q6_K]: { d: 2e-5 },
};

/**
 * @param {number} type ggml type (Q5_0, Q4_K, Q5_K or Q6_K)
 * @param {number} nBlocks
 * @param {() => number} rand uniform [0,1)
 * @returns {Uint8Array}
 */
export function randomQuantBlocks(type, nBlocks, rand) {
  const info = GGML_TYPE_INFO[type];
  const sc = SCALES[type];
  if (!sc) throw new Error(`randomQuantBlocks: unsupported type ${type}`);
  const out = new Uint8Array(nBlocks * info.typeSize);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nBlocks; b++) {
    const o = b * info.typeSize;
    for (let i = 0; i < info.typeSize; i++) out[o + i] = Math.floor(rand() * 256);
    const d = sc.d * (0.5 + rand());
    if (type === GGML_TYPE.Q6_K) dv.setUint16(o + 208, f32ToF16(d), true);
    else {
      dv.setUint16(o, f32ToF16(d), true);
      if (type !== GGML_TYPE.Q5_0) dv.setUint16(o + 2, f32ToF16(d * sc.dminRatio), true);
    }
  }
  return out;
}
