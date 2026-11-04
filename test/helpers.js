import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
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
