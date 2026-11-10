// Relaxed-SIMD kernels (ZEROLLM_DOT=relaxed): every test is SKIPPED (not failed) on a V8 without relaxed SIMD.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildKernelModule, instantiateKernels, resolveDotMode, usesInt8Acts } from '../src/compute/wasm/kernels.js';
import { probeWasmFeatures } from '../src/compute/wasm/features.js';
import { quantizeAct, matvec, Act } from '../src/compute/ref-kernels.js';
import { quantizeQ8_0, quantizeQ4_0 } from '../src/compute/quant.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { randomQuantBlocks } from '../tools/random-blocks.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { f32ToF16 } from '../src/util/f16.js';

const RELAXED = probeWasmFeatures().relaxedSimd;
const T = { skip: RELAXED ? false : 'WebAssembly relaxed SIMD is not available in this V8' };

/** the formats the relaxed mode touches (Q8_0 only has its batched kernel changed: int8 activations) */
const FORMATS = [
  { label: 'Q8_0', type: GGML_TYPE.Q8_0, elems: 32, bytes: 34, useXs: false, dec: 'matvec_q8_0', mm: 'mm_q8_0', scaleOffs: [0], mk: (n, rand) => quantizeQ8_0(Float32Array.from({ length: n * 32 }, () => gaussian(rand) * 0.4)) },
  { label: 'Q4_0', type: GGML_TYPE.Q4_0, elems: 32, bytes: 18, useXs: false, dec: 'matvec_q4_0', mm: 'mm_q4_0', scaleOffs: [0], mk: (n, rand) => quantizeQ4_0(Float32Array.from({ length: n * 32 }, () => gaussian(rand) * 0.4)) },
  { label: 'Q5_0', type: GGML_TYPE.Q5_0, elems: 32, bytes: 22, useXs: true, dec: 'matvec_q5_0', mm: 'mm_q5_0', scaleOffs: [0], mk: (n, rand) => randomQuantBlocks(GGML_TYPE.Q5_0, n, rand) },
  { label: 'Q4_K', type: GGML_TYPE.Q4_K, elems: 256, bytes: 144, useXs: true, dec: 'matvec_q4_k', mm: 'mm_q4_k', scaleOffs: [0, 2], mk: (n, rand) => randomQuantBlocks(GGML_TYPE.Q4_K, n, rand) },
  { label: 'Q5_K', type: GGML_TYPE.Q5_K, elems: 256, bytes: 176, useXs: true, dec: 'matvec_q5_k', mm: 'mm_q5_k', scaleOffs: [0, 2], mk: (n, rand) => randomQuantBlocks(GGML_TYPE.Q5_K, n, rand) },
  { label: 'Q6_K', type: GGML_TYPE.Q6_K, elems: 256, bytes: 210, useXs: true, dec: 'matvec_q6_k', mm: 'mm_q6_k', scaleOffs: [208], mk: (n, rand) => randomQuantBlocks(GGML_TYPE.Q6_K, n, rand) },
];
const RELAXED_FORMATS = FORMATS.filter((f) => f.label !== 'Q8_0');
/** the two relaxed flavors: i16x8.relaxed_dot + widening add (`relaxed`), i32x4.relaxed_dot_add per 16 weights (`relaxed-add`) */
const MODES = ['relaxed', 'relaxed-add'];

const al = (v) => Math.ceil(v / 64) * 64;

/**
 * Runs the decode kernel once per token and the batched kernel once, in the given dot mode, on RAW int8 activations
 * (xq[t] : Int8Array(cols), xd[t] : Float32Array(cols/32); the block sums are computed here). The batched kernels get the
 * layout of their mode: int8 for relaxed, int16-widened otherwise.
 */
function runKernels(dot, fmt, rows, nb, wbytes, xq, xd) {
  const ntok = xq.length; const cols = nb * fmt.elems; const abl = cols / 32;
  const wOff = 64; const xq8 = al(wOff + wbytes.length); const xd1 = al(xq8 + cols); const xs1 = al(xd1 + abl * 4); const y1 = al(xs1 + abl * 4);
  const xqB = al(y1 + rows * 4); const xdB = al(xqB + ntok * cols * 2); const xsB = al(xdB + ntok * abl * 4); const yB = al(xsB + ntok * abl * 4);
  const mem = new WebAssembly.Memory({ initial: Math.ceil((yB + ntok * rows * 4 + 64) / 65536) });
  const k = instantiateKernels(mem, { dot });
  new Uint8Array(mem.buffer, wOff, wbytes.length).set(wbytes);
  const sums = xq.map((q) => Float32Array.from({ length: abl }, (_, b) => { let s = 0; for (let i = 0; i < 32; i++) s += q[b * 32 + i]; return s; }));
  const xqBv = usesInt8Acts(dot) ? new Int8Array(mem.buffer, xqB, ntok * cols) : new Int16Array(mem.buffer, xqB, ntok * cols);
  xq.forEach((q, t) => xqBv.set(q, t * cols));
  new Float32Array(mem.buffer, xdB, ntok * abl).set(xd.flatMap((d) => [...d]));
  new Float32Array(mem.buffer, xsB, ntok * abl).set(sums.flatMap((s) => [...s]));
  const decode = new Float32Array(ntok * rows);
  for (let t = 0; t < ntok; t++) {
    new Int8Array(mem.buffer, xq8, cols).set(xq[t]); new Float32Array(mem.buffer, xd1, abl).set(xd[t]); new Float32Array(mem.buffer, xs1, abl).set(sums[t]);
    if (fmt.useXs) k[fmt.dec](wOff, xq8, xd1, xs1, y1, rows, nb); else k[fmt.dec](wOff, xq8, xd1, y1, rows, nb);
    decode.set(new Float32Array(mem.buffer, y1, rows), t * rows);
  }
  if (fmt.useXs) k[fmt.mm](wOff, xqB, xdB, xsB, yB, rows, nb, ntok); else k[fmt.mm](wOff, xqB, xdB, yB, rows, nb, ntok);
  return { decode, batch: Float32Array.from(new Float32Array(mem.buffer, yB, ntok * rows)) };
}

/** the JS reference (the oracle) on the same raw activations */
function reference(fmt, rows, nb, wbytes, xq, xd) {
  const cols = nb * fmt.elems;
  const out = new Float32Array(xq.length * rows);
  xq.forEach((q, t) => {
    const a = new Act(new Float32Array(cols)); a.q = q; a.d = xd[t];
    out.set(matvec({ type: fmt.type, rows, cols, data: Uint8Array.from(wbytes) }, a, new Float32Array(rows)), t * rows);
  });
  return out;
}

const rmsOf = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0) / v.length) || 1;

/** quantize float activations the way the engine does (int8 in -127..127) */
function quantizedTokens(cols, ntok, rand) {
  const xq = []; const xd = [];
  for (let t = 0; t < ntok; t++) {
    const q = new Int8Array(cols); const d = new Float32Array(cols / 32);
    quantizeAct(Float32Array.from({ length: cols }, () => gaussian(rand)), q, d);
    xq.push(q); xd.push(d);
  }
  return { xq, xd };
}

// ------------------------------------------------------------------------------------------------------- the module
for (const mode of MODES) {
  test(`${mode}: kernel module validates and exports the same kernel names as the default module`, T, () => {
    const rx = new WebAssembly.Module(buildKernelModule({ dot: mode }));
    const base = new WebAssembly.Module(buildKernelModule());
    const names = (m) => WebAssembly.Module.exports(m).map((e) => e.name).sort();
    assert.deepEqual(names(rx), names(base));
  });
}

test('resolveDotMode: ZEROLLM_DOT wins, ZEROLLM_KDOT still works, default is extmul, relaxed only when supported', () => {
  assert.equal(resolveDotMode({}), 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_KDOT: 'wide' }), 'wide');
  assert.equal(resolveDotMode({ ZEROLLM_KDOT: 'extmul' }), 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'wide' }), 'wide');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'EXTMUL', ZEROLLM_KDOT: 'wide' }), 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'relaxed' }), RELAXED ? 'relaxed' : 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'relaxed-add' }), RELAXED ? 'relaxed-add' : 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'relaxed', ZEROLLM_KDOT: 'wide' }), RELAXED ? 'relaxed' : 'extmul');
  assert.equal(resolveDotMode({ ZEROLLM_DOT: 'bogus', ZEROLLM_KDOT: 'wide' }), 'wide'); // unknown value: warned about, ignored
});

test('instantiateKernels() without an explicit dot never returns relaxed kernels, even if ZEROLLM_DOT=relaxed (embedding workers stage int16)', () => {
  const was = process.env.ZEROLLM_DOT;
  process.env.ZEROLLM_DOT = 'relaxed';
  try {
    const fmt = FORMATS[3]; const rand = mulberry32(3); const rows = 3; const nb = 1; const cols = 256;
    const wbytes = fmt.mk(rows * nb, rand); const { xq, xd } = quantizedTokens(cols, 5, rand);
    // dot undefined -> the module the environment gives an int16-staging caller; it must equal the explicit default one
    const viaEnv = runKernels(undefined, fmt, rows, nb, wbytes, xq, xd);
    const explicit = runKernels('extmul', fmt, rows, nb, wbytes, xq, xd);
    assert.deepEqual([...viaEnv.batch], [...explicit.batch]);
    assert.deepEqual([...viaEnv.decode], [...explicit.decode]);
  } finally { if (was === undefined) delete process.env.ZEROLLM_DOT; else process.env.ZEROLLM_DOT = was; }
});

// ------------------------------------------------------------------------------ random data, engine-style activations
for (const mode of MODES) for (const fmt of FORMATS) {
  test(`${mode} ${fmt.label}: decode kernel matches the JS reference and the default kernel; batched is BIT-IDENTICAL to decode`, T, () => {
    const rand = mulberry32(2024);
    const shapes = [[1, 1, 1], [2, 1, 4], [3, 2, 5], [5, 1, 7], [4, 3, 8], [7, 2, 13], [2, 1, 16], [1, 2, 3], [8, 2, 4]];
    for (const [rows, nb, ntok] of shapes) {
      const cols = nb * fmt.elems;
      const wbytes = fmt.mk(rows * nb, rand);
      const { xq, xd } = quantizedTokens(cols, ntok, rand);
      const rx = runKernels(mode, fmt, rows, nb, wbytes, xq, xd);
      const base = runKernels('extmul', fmt, rows, nb, wbytes, xq, xd);
      const want = reference(fmt, rows, nb, wbytes, xq, xd);
      const rms = rmsOf(want);
      for (let i = 0; i < want.length; i++) {
        const where = `${fmt.label} rows=${rows} nb=${nb} ntok=${ntok} token ${Math.floor(i / rows)} row ${i % rows}`;
        assert.equal(rx.batch[i], rx.decode[i], `batched != decode, ${where}`); // same integer sums, same float order
        assert.ok(Math.abs(rx.decode[i] - want[i]) <= 1e-4 * rms, `vs reference, ${where}: ${rx.decode[i]} vs ${want[i]}`);
        assert.ok(Math.abs(rx.decode[i] - base.decode[i]) <= 1e-5 * rms, `vs default kernel, ${where}: ${rx.decode[i]} vs ${base.decode[i]}`);
        if (fmt.label === 'Q8_0') assert.equal(rx.decode[i], base.decode[i], `Q8_0 decode kernel is unchanged, ${where}`);
      }
    }
  });
}

// --------------------------------------------- exact-float regime: power-of-two scales make every float operation exact, so
// the relaxed kernels (whose i32 lanes group the products differently) must agree BIT FOR BIT with the default kernels and
// with the JS reference, on raw int8 activations that include -128 (which the engine's quantizer never produces).
// ONE power-of-two exponent per call (all float terms are then integer multiples of a common granule, so they stay exact
// while the sums stay below 2^24 granules), and small 6-bit / int8 sub-block scales so that holds for the extreme inputs too.
const pow2Scales = (fmt, wbytes, rand) => {
  const out = Uint8Array.from(wbytes); const dv = new DataView(out.buffer);
  const d = f32ToF16(2 ** -(3 + Math.floor(rand() * 4)));
  for (let b = 0; b < out.length / fmt.bytes; b++) {
    const o = b * fmt.bytes;
    for (const off of fmt.scaleOffs) dv.setUint16(o + off, d, true);
    if (fmt.label === 'Q4_K' || fmt.label === 'Q5_K') for (let i = 4; i < 16; i++) out[o + i] = Math.floor(rand() * 16); // all 6-bit scales and mins <= 15
    if (fmt.label === 'Q6_K') for (let i = 0; i < 16; i++) out[o + 192 + i] = (Math.floor(rand() * 16) - 8) & 0xff;
  }
  return out;
};
const rawTokens = (cols, ntok, rand, pick) => {
  const e = 2 ** -(4 + Math.floor(rand() * 4));
  return {
    xq: Array.from({ length: ntok }, () => Int8Array.from({ length: cols }, pick)),
    xd: Array.from({ length: ntok }, () => new Float32Array(cols / 32).fill(e)),
  };
};

for (const mode of MODES) for (const fmt of FORMATS) {
  test(`${mode} ${fmt.label}: exact-float regime on raw int8 activations (random, -128/127 extremes) is bit-identical to the default kernels and the reference`, T, () => {
    const rand = mulberry32(77);
    for (const [rows, nb, ntok] of [[3, 1, 5], [4, 2, 4], [5, 1, 9]]) {
      const cols = nb * fmt.elems;
      for (const pick of [
        () => Math.floor(rand() * 256) - 128,              // anything, including -128
        () => (rand() < 0.5 ? -128 : 127),                 // extremes only
        () => (rand() < 0.9 ? -128 : 127),                 // mostly -128
      ]) {
        const wbytes = pow2Scales(fmt, fmt.mk(rows * nb, rand), rand);
        const { xq, xd } = rawTokens(cols, ntok, rand, pick);
        const rx = runKernels(mode, fmt, rows, nb, wbytes, xq, xd);
        const base = runKernels('extmul', fmt, rows, nb, wbytes, xq, xd);
        const want = reference(fmt, rows, nb, wbytes, xq, xd);
        for (let i = 0; i < want.length; i++) {
          const where = `${fmt.label} rows=${rows} nb=${nb} token ${Math.floor(i / rows)} row ${i % rows}`;
          assert.equal(rx.decode[i], want[i], `relaxed decode vs reference, ${where}`);
          assert.equal(rx.batch[i], want[i], `relaxed batched vs reference, ${where}`);
          assert.equal(base.decode[i], want[i], `default decode vs reference (sanity), ${where}`);
          assert.equal(base.batch[i], want[i], `default batched vs reference (sanity), ${where}`);
        }
      }
    }
  });
}

// ------------------------------------------------------------------------------------------ worst-case input values
/** every weight at its maximum / minimum quant value (raw bytes), sane scales, nb blocks per row */
function extremeWeights(fmt, rows, nb, fill, rand) {
  const out = new Uint8Array(rows * nb * fmt.bytes).fill(fill); const dv = new DataView(out.buffer);
  for (let b = 0; b < rows * nb; b++) {
    const o = b * fmt.bytes;
    for (const off of fmt.scaleOffs) dv.setUint16(o + off, f32ToF16(2 ** -4), true);
    if (fmt.label === 'Q6_K') for (let i = 0; i < 16; i++) out[o + 192 + i] = Math.floor(rand() * 16) - 8; // int8 scales, moderate so the float sums stay exact
    if (fmt.label === 'Q4_K' || fmt.label === 'Q5_K') out.fill(fill === 0xff ? 0x3f : fill, o + 4, o + 16); // 6-bit scales/mins: 63 (0xff would spill into the high bits)
  }
  return out;
}

for (const mode of MODES) for (const fmt of RELAXED_FORMATS) {
  test(`${mode} ${fmt.label}: worst-case values (all activations -128 or +127, every weight quant at its minimum / maximum): no saturation, no sign problem`, T, () => {
    const rand = mulberry32(5);
    for (const fill of [0xff, 0x00, 0x0f, 0xf0, 0x55, 0xaa]) {
      for (const act of [-128, 127]) {
        const rows = 5; const nb = 1; const ntok = 6; const cols = nb * fmt.elems;
        const wbytes = extremeWeights(fmt, rows, nb, fill, rand);
        const { xq, xd } = rawTokens(cols, ntok, rand, () => act);
        const rx = runKernels(mode, fmt, rows, nb, wbytes, xq, xd);
        const want = reference(fmt, rows, nb, wbytes, xq, xd);
        const base = runKernels('extmul', fmt, rows, nb, wbytes, xq, xd);
        for (let i = 0; i < want.length; i++) {
          const where = `${fmt.label} fill 0x${fill.toString(16)} act ${act} row ${i % rows}`;
          assert.ok(Number.isFinite(want[i]), `finite, ${where}`);
          assert.equal(rx.decode[i], want[i], `relaxed decode vs reference, ${where}`);
          assert.equal(rx.batch[i], want[i], `relaxed batched vs reference, ${where}`);
          assert.equal(base.decode[i], want[i], `default decode (sanity), ${where}`);
        }
      }
    }
  });
}

for (const mode of MODES) {
  test(`${mode} Q8_0: the int8-activation batched kernel handles the full signed weight range and -128 activations (it widens in-kernel)`, T, () => {
    const fmt = FORMATS[0]; const rand = mulberry32(8);
    const rows = 4; const nb = 3; const ntok = 7; const cols = nb * 32;
    for (const w of [0x80, 0x7f, 0xff]) {
      const wbytes = new Uint8Array(rows * nb * 34).fill(w); const dv = new DataView(wbytes.buffer);
      for (let b = 0; b < rows * nb; b++) dv.setUint16(b * 34, f32ToF16(2 ** -5), true);
      const { xq, xd } = rawTokens(cols, ntok, rand, () => (rand() < 0.5 ? -128 : 127));
      const rx = runKernels(mode, fmt, rows, nb, wbytes, xq, xd);
      const want = reference(fmt, rows, nb, wbytes, xq, xd);
      for (let i = 0; i < want.length; i++) { assert.equal(rx.batch[i], want[i]); assert.equal(rx.decode[i], want[i]); }
    }
  });
}
