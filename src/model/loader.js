// Load tensors from a GGUF file into memory (single-process; the worker pool has its own sliced loader).
import fs from 'node:fs';
import { GGML_TYPE } from '../gguf/types.js';
import { readTensorBytes } from '../gguf/reader.js';
import { dequantRow, rowBytes } from '../compute/ref-kernels.js';

/** Load a matrix tensor as a Weight (raw quantized bytes). @returns {import('../compute/ref-kernels.js').Weight} */
export function loadWeight(fd, t) {
  if (t.dims.length !== 2) throw new Error(`tensor ${t.name}: expected a 2-D matrix, got dims [${t.dims}]`);
  const cols = t.dims[0]; const rows = t.dims[1];
  if (t.nBytes !== rows * rowBytes(t.type, cols)) throw new Error(`tensor ${t.name}: unexpected size`);
  const data = new Uint8Array(t.nBytes);
  readTensorBytes(fd, t, 0, data);
  return { type: t.type, rows, cols, data };
}

/** Load a 1-D tensor (norm weights, biases) as Float32Array, converting from f16/bf16 if needed. */
export function loadVector(fd, t) {
  const n = t.nElements;
  if (t.type === GGML_TYPE.F32) {
    const f = new Float32Array(n);
    readTensorBytes(fd, t, 0, new Uint8Array(f.buffer));
    return f;
  }
  const w = loadWeight(fd, { ...t, dims: [n, 1] });
  return dequantRow(w, 0, new Float32Array(n));
}

export function openModelFile(path) { return fs.openSync(path, 'r'); }
