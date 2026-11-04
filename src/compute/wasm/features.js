// Runtime feature probes (shown by `zerollm doctor`). Everything here is detection only.
import os from 'node:os';

const MAGIC = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

/** @param {number} id @param {number[]} body */
function section(id, body) { return [id, body.length, ...body]; } // all probe sections are < 128 bytes

/** Module with one function `() -> []` whose body is `code` (instructions, no trailing end) and one memory `memBytes`. */
function probeModule(code, memBytes = [0x00, 0x01]) {
  const body = [0x00, ...code, 0x0b];
  return new Uint8Array([
    ...MAGIC,
    ...section(1, [0x01, 0x60, 0x00, 0x00]),
    ...section(3, [0x01, 0x00]),
    ...section(5, [0x01, ...memBytes]),
    ...section(10, [0x01, body.length, ...body]),
  ]);
}

const validate = (bytes) => { try { return WebAssembly.validate(bytes); } catch { return false; } };

/** v128.const 0 ; i8x16.abs? simplest: `v128.const (16 zero bytes); drop` */
const V128_ZERO = [0xfd, 0x0c, ...new Array(16).fill(0)];

export function probeWasmFeatures() {
  const simd = validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
  // relaxed SIMD: (v128 v128 v128) -> v128 using i32x4.relaxed_dot_i8x16_i7x16_add_s (0xFD 0x113)
  const relaxedSimd = validate(new Uint8Array([
    ...MAGIC,
    ...section(1, [0x01, 0x60, 0x03, 0x7b, 0x7b, 0x7b, 0x01, 0x7b]),
    ...section(3, [0x01, 0x00]),
    ...section(10, [0x01, 0x0b, 0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0x93, 0x02, 0x0b]),
  ]));
  // shared memory (flags 0x03: has max + shared) + i32.atomic.load (0xFE 0x10)
  const threads = validate(probeModule([0x41, 0x00, 0xfe, 0x10, 0x02, 0x00, 0x1a], [0x03, 0x01, 0x01]));
  // two memories in one module (multi-memory proposal)
  const multiMemory = validate(new Uint8Array([
    ...MAGIC,
    ...section(5, [0x02, 0x00, 0x01, 0x00, 0x01]),
  ]));
  // memory64: flags 0x04 (i64 index)
  const memory64 = validate(new Uint8Array([...MAGIC, ...section(5, [0x01, 0x04, 0x01])]));
  // bulk memory: memory.copy (0xFC 0x0A 0 0)
  const bulkMemory = validate(probeModule([0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x00, 0x00]));
  return { simd, relaxedSimd, threads, multiMemory, memory64, bulkMemory };
}

/** Try to allocate a WebAssembly.Memory of the given size in GB. Returns true if it worked. */
export function probeMemoryAlloc(gb) {
  try {
    const m = new WebAssembly.Memory({ initial: Math.round((gb * 1024 * 1024 * 1024) / 65536) });
    return m.buffer.byteLength > 0;
  } catch { return false; }
}

export function probeRegexInlineModifiers() {
  try { return new RegExp('(?i:a)b', 'u').test('Ab') && !new RegExp('(?i:a)b', 'u').test('AB'); } catch { return false; }
}

export function probeJsFeatures() {
  let atomicsWait = false;
  try {
    if (typeof SharedArrayBuffer !== 'undefined') {
      atomicsWait = Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 1, 0) === 'not-equal';
    }
  } catch { /* ignore */ }
  let unicodeProps = false;
  try { unicodeProps = /\p{L}/u.test('é') && /\p{N}/u.test('5'); } catch { /* ignore */ }
  return {
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    atomicsWait,
    atomicsWaitAsync: typeof Atomics.waitAsync === 'function',
    regexInlineModifiers: probeRegexInlineModifiers(),
    regexUnicodeProps: unicodeProps,
    availableParallelism: typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length,
  };
}
