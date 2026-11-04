// Minimal WebAssembly binary emitter (no compiler, no wabt): LEB128, sections, function bodies, SIMD opcodes.
// Node cannot parse .wat, so kernels are generated as bytes by small JS programs (see kernels.js).

export const T = Object.freeze({ i32: 0x7f, i64: 0x7e, f32: 0x7d, f64: 0x7c, v128: 0x7b });

/** Unsigned LEB128. @param {number|bigint} n */
export function uleb(n) {
  let v = BigInt(n);
  if (v < 0n) throw new Error('uleb: negative');
  const out = [];
  do { let b = Number(v & 0x7fn); v >>= 7n; if (v !== 0n) b |= 0x80; out.push(b); } while (v !== 0n);
  return out;
}

/** Signed LEB128. @param {number|bigint} n */
export function sleb(n) {
  let v = BigInt(n);
  const out = [];
  for (;;) {
    const b = Number(v & 0x7fn);
    v >>= 7n; // arithmetic shift for BigInt
    if ((v === 0n && !(b & 0x40)) || (v === -1n && (b & 0x40))) { out.push(b); return out; }
    out.push(b | 0x80);
  }
}

const f32Bytes = (x) => Array.from(new Uint8Array(new Float32Array([x]).buffer));
const f64Bytes = (x) => Array.from(new Uint8Array(new Float64Array([x]).buffer));
const str = (s) => { const b = Array.from(new TextEncoder().encode(s)); return [...uleb(b.length), ...b]; };
const vec = (items) => [...uleb(items.length), ...items.flat()];
const section = (id, body) => [id, ...uleb(body.length), ...body];

/** Plain (single-byte) opcodes. */
export const OPS = Object.freeze({
  unreachable: 0x00, nop: 0x01, return: 0x0f, drop: 0x1a, select: 0x1b,
  'i32.load': 0x28, 'i64.load': 0x29, 'f32.load': 0x2a, 'f64.load': 0x2b,
  'i32.load8_s': 0x2c, 'i32.load8_u': 0x2d, 'i32.load16_s': 0x2e, 'i32.load16_u': 0x2f,
  'i32.store': 0x36, 'f32.store': 0x38, 'i32.store8': 0x3a, 'i32.store16': 0x3b,
  'i32.eqz': 0x45, 'i32.eq': 0x46, 'i32.ne': 0x47, 'i32.lt_s': 0x48, 'i32.lt_u': 0x49, 'i32.gt_s': 0x4a, 'i32.gt_u': 0x4b,
  'i32.le_s': 0x4c, 'i32.le_u': 0x4d, 'i32.ge_s': 0x4e, 'i32.ge_u': 0x4f,
  'i32.add': 0x6a, 'i32.sub': 0x6b, 'i32.mul': 0x6c, 'i32.div_s': 0x6d, 'i32.div_u': 0x6e, 'i32.rem_s': 0x6f, 'i32.rem_u': 0x70,
  'i32.and': 0x71, 'i32.or': 0x72, 'i32.xor': 0x73, 'i32.shl': 0x74, 'i32.shr_s': 0x75, 'i32.shr_u': 0x76,
  'f32.abs': 0x8b, 'f32.neg': 0x8c, 'f32.sqrt': 0x91, 'f32.add': 0x92, 'f32.sub': 0x93, 'f32.mul': 0x94, 'f32.div': 0x95,
  'f32.min': 0x96, 'f32.max': 0x97,
  'i32.trunc_f32_s': 0xa8, 'f32.convert_i32_s': 0xb2, 'f32.convert_i32_u': 0xb3,
  'i32.reinterpret_f32': 0xbc, 'f32.reinterpret_i32': 0xbe,
});

/** SIMD opcodes, all prefixed by 0xFD with the sub-opcode LEB128-encoded. */
export const SIMD = Object.freeze({
  'v128.load': 0x00, 'v128.store': 0x0b, 'v128.const': 0x0c, 'i8x16.swizzle': 0x0e,
  'i8x16.splat': 0x0f, 'i16x8.splat': 0x10, 'i32x4.splat': 0x11, 'f32x4.splat': 0x13,
  'i32x4.extract_lane': 0x1b, 'f32x4.extract_lane': 0x1f,
  'i8x16.eq': 0x23, 'i8x16.ne': 0x24,
  'v128.not': 0x4d, 'v128.and': 0x4e, 'v128.andnot': 0x4f, 'v128.or': 0x50, 'v128.xor': 0x51,
  'i8x16.neg': 0x61, 'i8x16.shl': 0x6b, 'i8x16.shr_s': 0x6c, 'i8x16.shr_u': 0x6d, 'i8x16.add': 0x6e, 'i8x16.sub': 0x71,
  'i16x8.extadd_pairwise_i8x16_s': 0x7c, 'i32x4.extadd_pairwise_i16x8_s': 0x7e,
  'i16x8.extend_low_i8x16_s': 0x87, 'i16x8.extend_high_i8x16_s': 0x88, 'i16x8.extend_low_i8x16_u': 0x89, 'i16x8.extend_high_i8x16_u': 0x8a,
  'i16x8.add': 0x8e, 'i16x8.mul': 0x95,
  'i16x8.extmul_low_i8x16_s': 0x9c, 'i16x8.extmul_high_i8x16_s': 0x9d,
  'i32x4.add': 0xae, 'i32x4.sub': 0xb1, 'i32x4.mul': 0xb5, 'i32x4.dot_i16x8_s': 0xba,
  'f32x4.add': 0xe4, 'f32x4.sub': 0xe5, 'f32x4.mul': 0xe6, 'f32x4.div': 0xe7,
  'f32x4.convert_i32x4_s': 0xfa,
  // relaxed SIMD (optional fast path, feature-gated)
  'i32x4.relaxed_dot_i8x16_i7x16_add_s': 0x113,
});
const LANE_OPS = new Set(['i32x4.extract_lane', 'f32x4.extract_lane']);

/** Builds one function body. Values live on the wasm stack; locals are typed and allocated by `local()`. */
export class FuncBuilder {
  /** @param {number[]} params @param {number[]} results */
  constructor(params, results) {
    this.params = params; this.results = results;
    this.localTypes = [];
    this.code = [];
    this.labels = [];
  }

  /** Index of parameter i. */
  param(i) { return i; }

  /** Declare a local of type t; returns its index. */
  local(t) { this.localTypes.push(t); return this.params.length + this.localTypes.length - 1; }

  emit(...bytes) { this.code.push(...bytes); return this; }

  get(i) { return this.emit(0x20, ...uleb(i)); }
  set(i) { return this.emit(0x21, ...uleb(i)); }
  tee(i) { return this.emit(0x22, ...uleb(i)); }
  i32(v) { return this.emit(0x41, ...sleb(v)); }
  f32(v) { return this.emit(0x43, ...f32Bytes(v)); }
  v128Const(bytes16) {
    if (bytes16.length !== 16) throw new Error('v128.const needs 16 bytes');
    return this.emit(0xfd, ...uleb(SIMD['v128.const']), ...bytes16);
  }

  /** Emit a plain or SIMD op by name. For memory ops pass {offset, align} (align = log2, default 0 = unaligned ok). */
  op(name, imm = {}) {
    if (name in OPS) {
      this.emit(OPS[name]);
      if (/\.(load|store)/.test(name)) this.emit(...uleb(imm.align ?? 0), ...uleb(imm.offset ?? 0));
      return this;
    }
    if (name in SIMD) {
      this.emit(0xfd, ...uleb(SIMD[name]));
      if (name === 'v128.load' || name === 'v128.store') this.emit(...uleb(imm.align ?? 0), ...uleb(imm.offset ?? 0));
      else if (LANE_OPS.has(name)) this.emit(imm.lane ?? 0);
      return this;
    }
    throw new Error(`unknown op ${name}`);
  }

  // ---- structured control flow. Each returns after emitting; the body callback receives a label for br/brIf.
  #open(opcode, body, blockType = 0x40) {
    const label = { index: this.labels.length };
    this.labels.push(label);
    this.emit(opcode, blockType);
    body(label);
    this.emit(0x0b);
    this.labels.pop();
  }
  block(body, blockType) { this.#open(0x02, body, blockType); return this; }
  loop(body, blockType) { this.#open(0x03, body, blockType); return this; }
  /** Branch to an enclosing label. */
  br(label) { return this.emit(0x0c, ...uleb(this.labels.length - 1 - label.index)); }
  /** Pops an i32; branches if non-zero. */
  brIf(label) { return this.emit(0x0d, ...uleb(this.labels.length - 1 - label.index)); }

  /** @returns {number[]} body bytes incl. local declarations and final `end` */
  finish() {
    // run-length group consecutive locals of the same type
    const groups = [];
    for (const t of this.localTypes) {
      const last = groups[groups.length - 1];
      if (last && last[1] === t) last[0]++; else groups.push([1, t]);
    }
    const decl = [...uleb(groups.length), ...groups.flatMap(([n, t]) => [...uleb(n), t])];
    const body = [...decl, ...this.code, 0x0b];
    return [...uleb(body.length), ...body];
  }
}

/** Builds a module that imports one memory ("env"."mem") and exports functions. */
export class ModuleBuilder {
  constructor() { this.types = []; this.funcs = []; this.exports = []; this.importMemory = null; }

  /** Import a memory as env.mem with a minimum size in pages. */
  withImportedMemory(minPages = 1) { this.importMemory = minPages; return this; }

  /** Create a function; call `.finish()` is done by build(). Returns its FuncBuilder. */
  func(name, params, results = [], exported = true) {
    const f = new FuncBuilder(params, results);
    this.funcs.push({ name, f, exported });
    return f;
  }

  #typeIndex(params, results) {
    const key = JSON.stringify([params, results]);
    let i = this.types.findIndex((t) => t.key === key);
    if (i < 0) { this.types.push({ key, params, results }); i = this.types.length - 1; }
    return i;
  }

  build() {
    const typeIdx = this.funcs.map(({ f }) => this.#typeIndex(f.params, f.results));
    const out = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    out.push(...section(1, vec(this.types.map((t) => [0x60, ...vec(t.params.map((p) => [p])), ...vec(t.results.map((r) => [r]))]))));
    if (this.importMemory !== null) out.push(...section(2, vec([[...str('env'), ...str('mem'), 0x02, 0x00, ...uleb(this.importMemory)]])));
    out.push(...section(3, vec(typeIdx.map((i) => uleb(i)))));
    const exps = this.funcs.map((fn, i) => ({ fn, i })).filter(({ fn }) => fn.exported)
      .map(({ fn, i }) => [...str(fn.name), 0x00, ...uleb(i)]);
    out.push(...section(7, vec(exps)));
    out.push(...section(10, vec(this.funcs.map(({ f }) => f.finish()))));
    return new Uint8Array(out);
  }
}
