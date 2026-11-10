import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
import { makeTinyQwen35 } from '../tools/make-tiny-qwen35-gguf.js';
import { readGGUF } from '../src/gguf/reader.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-t-'));
process.on('exit', () => { try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ } });
let n = 0;

/** Write a tiny model to a temp file and parse it. */
export function tinyFile(opts = {}) {
  const m = makeTinyModel(opts);
  const p = path.join(dir, `tiny-${n++}.gguf`);
  fs.writeFileSync(p, m.buffer);
  return { path: p, g: readGGUF(p), ...m };
}

/** Dimensions that make every K-quant block fit (hidden/ffn/inner sizes are multiples of 256), for q4_k_m synthetic qwen35 files. */
export const Q35_KQUANT_DIMS = Object.freeze({ nEmbd: 256, nFF: 256, nHead: 8, nHeadKv: 2, headDim: 32, nRot: 16, nK: 2, nV: 8, headSsm: 32, nLayer: 4, interval: 2 });

/** Write a tiny synthetic Qwen3.5 (hybrid Gated DeltaNet) model to a temp file and parse it (see tools/make-tiny-qwen35-gguf.js for the options). */
export function tinyQwen35File(opts = {}) {
  const m = makeTinyQwen35(opts);
  const p = path.join(dir, `tiny-q35-${n++}.gguf`);
  fs.writeFileSync(p, m.buffer);
  return { path: p, g: readGGUF(p), ...m };
}
