import test from 'node:test';
import assert from 'node:assert/strict';
import { ModuleBuilder, T, uleb, sleb, SIMD, OPS } from '../src/compute/wasm/builder.js';
import { mulberry32 } from '../src/util/rng.js';

const instantiate = (bytes, pages = 1) => {
  assert.ok(WebAssembly.validate(bytes), 'module must validate');
  const mem = new WebAssembly.Memory({ initial: pages });
  const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { mem } });
  return { inst, mem, u8: new Uint8Array(mem.buffer) };
};

test('LEB128 against known vectors', () => {
  assert.deepEqual(uleb(0), [0]);
  assert.deepEqual(uleb(127), [0x7f]);
  assert.deepEqual(uleb(128), [0x80, 0x01]);
  assert.deepEqual(uleb(624485), [0xe5, 0x8e, 0x26]);
  assert.deepEqual(uleb(0x113), [0x93, 0x02]); // the relaxed-simd dot sub-opcode
  assert.deepEqual(sleb(0), [0]);
  assert.deepEqual(sleb(-1), [0x7f]);
  assert.deepEqual(sleb(63), [0x3f]);
  assert.deepEqual(sleb(64), [0xc0, 0x00]);
  assert.deepEqual(sleb(-64), [0x40]);
  assert.deepEqual(sleb(-65), [0xbf, 0x7f]);
  assert.deepEqual(sleb(-123456), [0xc0, 0xbb, 0x78]);
  assert.deepEqual(sleb(2147483647), [0xff, 0xff, 0xff, 0xff, 0x07]);
  assert.throws(() => uleb(-1));
});

test('trivial module: add; i32 constants of both signs', () => {
  const m = new ModuleBuilder();
  m.func('add', [T.i32, T.i32], [T.i32]).get(0).get(1).op('i32.add');
  m.func('c', [], [T.i32]).i32(-70000);
  m.func('k', [], [T.f32]).f32(1.5);
  const { inst } = instantiate(m.build());
  assert.equal(inst.exports.add(2, 40), 42);
  assert.equal(inst.exports.c(), -70000);
  assert.equal(inst.exports.k(), 1.5);
});

test('structured control flow: loop with br_if, nested block exit', () => {
  const m = new ModuleBuilder();
  const f = m.func('sum', [T.i32], [T.i32]); // sum of 1..n
  const acc = f.local(T.i32); const i = f.local(T.i32);
  f.block((exit) => {
    f.get(0).op('i32.eqz'); f.brIf(exit);
    f.i32(1).set(i);
    f.loop((again) => {
      f.get(acc).get(i).op('i32.add').set(acc);
      f.get(i).i32(1).op('i32.add').tee(i);
      f.get(0).op('i32.le_u'); f.brIf(again);
    });
  });
  f.get(acc);
  const { inst } = instantiate(m.build());
  assert.equal(inst.exports.sum(0), 0);
  assert.equal(inst.exports.sum(1), 1);
  assert.equal(inst.exports.sum(100), 5050);
});

test('every opcode name maps to a unique encoding within its table', () => {
  assert.equal(new Set(Object.values(OPS)).size, Object.keys(OPS).length);
  assert.equal(new Set(Object.values(SIMD)).size, Object.keys(SIMD).length);
});

// ------------------------------------------------------------ SIMD ops vs JS models (guards against opcode typos)
const rand = mulberry32(99);
const randBytes = (n) => Uint8Array.from({ length: n }, () => Math.floor(rand() * 256));

/** module exporting t(a, b, out): out = op(load(a), load(b)) */
function binaryKernel(name, { shift = false, unary = false } = {}) {
  const m = new ModuleBuilder().withImportedMemory(1);
  const f = m.func('t', [T.i32, T.i32, T.i32], []);
  f.get(2);
  f.get(0).op('v128.load');
  if (shift) f.i32(4); else if (!unary) f.get(1).op('v128.load');
  f.op(name);
  f.op('v128.store');
  return instantiate(m.build());
}

function check(name, model, opts) {
  const { inst, mem } = binaryKernel(name, opts);
  const u8 = new Uint8Array(mem.buffer);
  for (let trial = 0; trial < 50; trial++) {
    const a = randBytes(16); const b = randBytes(16);
    u8.set(a, 0); u8.set(b, 16);
    inst.exports.t(0, 16, 32);
    assert.deepEqual([...u8.subarray(32, 48)], [...model(a, b)], `${name} trial ${trial}`);
  }
}
const dv = (u) => new DataView(u.buffer, u.byteOffset, 16);
const out16 = (fn) => { const o = new Uint8Array(16); fn(dv(o)); return o; };
const lanes = (kind, n, size) => (a, fn) => out16((o) => { for (let i = 0; i < n; i++) fn(o, i, dv(a)); });

test('SIMD bitwise + i8x16 ops', () => {
  check('v128.and', (a, b) => a.map((x, i) => x & b[i]));
  check('v128.or', (a, b) => a.map((x, i) => x | b[i]));
  check('v128.xor', (a, b) => a.map((x, i) => x ^ b[i]));
  check('i8x16.sub', (a, b) => a.map((x, i) => (x - b[i]) & 0xff));
  check('i8x16.add', (a, b) => a.map((x, i) => (x + b[i]) & 0xff));
  check('i8x16.shr_u', (a) => a.map((x) => x >> 4), { shift: true });
  check('i8x16.shr_s', (a) => a.map((x) => ((x << 24) >> 24) >> 4 & 0xff), { shift: true });
  check('i8x16.shl', (a) => a.map((x) => (x << 4) & 0xff), { shift: true });
});

test('SIMD swizzle and byte compares', () => {
  check('i8x16.swizzle', (a, b) => b.map((idx) => (idx < 16 ? a[idx] : 0)));
  check('i8x16.eq', (a, b) => a.map((x, i) => (x === b[i] ? 0xff : 0)));
  check('i8x16.ne', (a, b) => a.map((x, i) => (x !== b[i] ? 0xff : 0)));
  // equal-heavy data so eq/ne see both outcomes
  const { inst, mem } = binaryKernel('i8x16.eq');
  const u8 = new Uint8Array(mem.buffer);
  const a = randBytes(16); const b = Uint8Array.from(a, (x, i) => (i % 3 ? x : x ^ 1));
  u8.set(a, 0); u8.set(b, 16); inst.exports.t(0, 16, 32);
  assert.deepEqual([...u8.subarray(32, 48)], [...a].map((x, i) => (x === b[i] ? 0xff : 0)));
});

test('SIMD widening, add, dot (int8 -> int16 -> int32)', () => {
  const s8 = (x) => (x << 24) >> 24;
  check('i16x8.extend_low_i8x16_s', (a) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, s8(a[i]), true); }), { unary: true });
  check('i16x8.extend_high_i8x16_s', (a) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, s8(a[8 + i]), true); }), { unary: true });
  check('i16x8.extend_low_i8x16_u', (a) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, a[i], true); }), { unary: true });
  check('i16x8.extend_high_i8x16_u', (a) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, a[8 + i], true); }), { unary: true });
  check('i32x4.add', (a, b) => out16((o) => { const x = dv(a), y = dv(b); for (let i = 0; i < 4; i++) o.setInt32(4 * i, (x.getInt32(4 * i, true) + y.getInt32(4 * i, true)) | 0, true); }));
  check('i32x4.dot_i16x8_s', (a, b) => out16((o) => {
    const x = dv(a), y = dv(b);
    for (let i = 0; i < 4; i++) o.setInt32(4 * i, (x.getInt16(4 * i, true) * y.getInt16(4 * i, true) + x.getInt16(4 * i + 2, true) * y.getInt16(4 * i + 2, true)) | 0, true);
  }));
});

test('SIMD extended multiplies and pairwise adds (the int8 dot used by the K-quant kernels)', () => {
  const s8 = (x) => (x << 24) >> 24;
  check('i16x8.extmul_low_i8x16_s', (a, b) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, s8(a[i]) * s8(b[i]), true); }));
  check('i16x8.extmul_high_i8x16_s', (a, b) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, s8(a[8 + i]) * s8(b[8 + i]), true); }));
  check('i16x8.add', (a, b) => out16((o) => { const x = dv(a), y = dv(b); for (let i = 0; i < 8; i++) o.setInt16(2 * i, (x.getInt16(2 * i, true) + y.getInt16(2 * i, true)) << 16 >> 16, true); }));
  check('i32x4.extadd_pairwise_i16x8_s', (a) => out16((o) => { const x = dv(a); for (let i = 0; i < 4; i++) o.setInt32(4 * i, x.getInt16(4 * i, true) + x.getInt16(4 * i + 2, true), true); }), { unary: true });
  check('i16x8.extadd_pairwise_i8x16_s', (a) => out16((o) => { for (let i = 0; i < 8; i++) o.setInt16(2 * i, s8(a[2 * i]) + s8(a[2 * i + 1]), true); }), { unary: true });
});

test('SIMD zero-extend 16->32 and i32x4 shift (used to convert f16 KV values)', () => {
  check('i32x4.extend_low_i16x8_u', (a) => out16((o) => { const x = dv(a); for (let i = 0; i < 4; i++) o.setUint32(4 * i, x.getUint16(2 * i, true), true); }), { unary: true });
  check('i32x4.extend_high_i16x8_u', (a) => out16((o) => { const x = dv(a); for (let i = 0; i < 4; i++) o.setUint32(4 * i, x.getUint16(8 + 2 * i, true), true); }), { unary: true });
  check('i32x4.shl', (a) => out16((o) => { const x = dv(a); for (let i = 0; i < 4; i++) o.setInt32(4 * i, x.getInt32(4 * i, true) << 4, true); }), { shift: true });
});

test('SIMD float: convert, add, mul', () => {
  const small = (a) => { const u = new Uint8Array(16); const d = dv(u); for (let i = 0; i < 4; i++) d.setInt32(4 * i, dv(a).getInt32(4 * i, true) >> 8, true); return u; };
  const { inst, mem } = (() => {
    const m = new ModuleBuilder().withImportedMemory(1);
    const f = m.func('conv', [T.i32, T.i32], []);
    f.get(1).get(0).op('v128.load').op('f32x4.convert_i32x4_s').op('v128.store');
    const g = m.func('mul', [T.i32, T.i32, T.i32], []);
    g.get(2).get(0).op('v128.load').get(1).op('v128.load').op('f32x4.mul').op('v128.store');
    const h = m.func('add', [T.i32, T.i32, T.i32], []);
    h.get(2).get(0).op('v128.load').get(1).op('v128.load').op('f32x4.add').op('v128.store');
    const s = m.func('splat', [T.f32, T.i32], []);
    s.get(1).get(0).op('f32x4.splat').op('v128.store');
    const e = m.func('lane', [T.i32, T.i32], [T.f32]);
    e.get(0).op('v128.load').op('f32x4.extract_lane', { lane: 2 });
    const e3 = m.func('lane3', [T.i32], [T.f32]);
    e3.get(0).op('v128.load').op('f32x4.extract_lane', { lane: 3 });
    return instantiate(m.build());
  })();
  const u8 = new Uint8Array(mem.buffer); const f32 = new Float32Array(mem.buffer);
  for (let t = 0; t < 30; t++) {
    const a = small(randBytes(16));
    u8.set(a, 0);
    inst.exports.conv(0, 16);
    for (let i = 0; i < 4; i++) assert.equal(f32[4 + i], dv(a).getInt32(4 * i, true));
    const x = Float32Array.from({ length: 4 }, () => rand() * 10 - 5); const y = Float32Array.from({ length: 4 }, () => rand() * 10 - 5);
    f32.set(x, 8); f32.set(y, 12);
    inst.exports.mul(32, 48, 64); inst.exports.add(32, 48, 80);
    for (let i = 0; i < 4; i++) { assert.equal(f32[16 + i], Math.fround(x[i] * y[i])); assert.equal(f32[20 + i], Math.fround(x[i] + y[i])); }
    inst.exports.splat(3.25, 96);
    assert.deepEqual([...f32.subarray(24, 28)], [3.25, 3.25, 3.25, 3.25]);
    assert.equal(inst.exports.lane(32, 0), x[2]);
    assert.equal(inst.exports.lane3(32), x[3]);
  }
});

test('scalar helpers used by kernels: load16_u, shifts, reinterpret, f32 load/store; v128.const; i8x16.splat', () => {
  const m = new ModuleBuilder().withImportedMemory(1);
  // f16 bits at ptr -> f32 (multiply trick), the exact sequence the kernels use
  const f = m.func('f16', [T.i32], [T.f32]);
  const h = f.local(T.i32);
  f.get(0).op('i32.load16_u').set(h);
  f.get(h).i32(0x7fff).op('i32.and').i32(13).op('i32.shl')
    .get(h).i32(0x8000).op('i32.and').i32(16).op('i32.shl').op('i32.or')
    .op('f32.reinterpret_i32').f32(2 ** 112).op('f32.mul');
  const c = m.func('const', [T.i32], []);
  c.get(0).v128Const(Array.from({ length: 16 }, (_, i) => i + 1)).op('v128.store');
  const sp = m.func('splat8', [T.i32, T.i32], []);
  sp.get(1).get(0).op('i8x16.splat').op('v128.store');
  const st = m.func('fstore', [T.i32, T.f32], []);
  st.get(0).get(1).op('f32.store', { offset: 4 });
  const { inst, mem } = instantiate(m.build());
  const u8 = new Uint8Array(mem.buffer); const dv16 = new DataView(mem.buffer);
  // all finite f16 values, including denormals and negatives
  for (let bits = 0; bits < 65536; bits++) {
    if (((bits >> 10) & 0x1f) === 0x1f) continue;
    dv16.setUint16(0, bits, true);
    const exp = (bits >> 10) & 0x1f; const mant = bits & 0x3ff; const sign = bits & 0x8000 ? -1 : 1;
    const want = exp === 0 ? sign * mant * 2 ** -24 : sign * (1 + mant / 1024) * 2 ** (exp - 15);
    assert.equal(inst.exports.f16(0), want, `f16 bits 0x${bits.toString(16)}`);
  }
  inst.exports.const(16);
  assert.deepEqual([...u8.subarray(16, 32)], Array.from({ length: 16 }, (_, i) => i + 1));
  inst.exports.splat8(0x0f, 32);
  assert.deepEqual([...u8.subarray(32, 48)], new Array(16).fill(0x0f));
  inst.exports.fstore(64, 7.5);
  assert.equal(new Float32Array(mem.buffer)[17], 7.5); // byte 68
});

test('relaxed-simd dot opcode encodes and validates when the engine supports it', () => {
  const m = new ModuleBuilder();
  const f = m.func('d', [T.v128, T.v128, T.v128], [T.v128]);
  f.get(0).get(1).get(2).op('i32x4.relaxed_dot_i8x16_i7x16_add_s');
  const bytes = m.build();
  // validity depends on the Node/V8 version; just make sure our encoding equals the doctor probe's
  const probe = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 8, 1, 96, 3, 123, 123, 123, 1, 123, 3, 2, 1, 0, 10, 13, 1, 11, 0, 32, 0, 32, 1, 32, 2, 253, 147, 2, 11]));
  assert.equal(WebAssembly.validate(bytes), probe);
});
