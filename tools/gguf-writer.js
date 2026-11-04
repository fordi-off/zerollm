// Minimal GGUF writer for tests and tiny synthetic models.
import { GGUF_TYPE, GGML_TYPE_INFO, tensorBytes } from '../src/gguf/types.js';

class W {
  constructor() { this.parts = []; this.len = 0; }
  push(b) { this.parts.push(b); this.len += b.length; }
  u8(v) { const b = Buffer.alloc(1); b.writeUInt8(v); this.push(b); }
  i8(v) { const b = Buffer.alloc(1); b.writeInt8(v); this.push(b); }
  u16(v) { const b = Buffer.alloc(2); b.writeUInt16LE(v); this.push(b); }
  i16(v) { const b = Buffer.alloc(2); b.writeInt16LE(v); this.push(b); }
  u32(v) { const b = Buffer.alloc(4); b.writeUInt32LE(v); this.push(b); }
  i32(v) { const b = Buffer.alloc(4); b.writeInt32LE(v); this.push(b); }
  f32(v) { const b = Buffer.alloc(4); b.writeFloatLE(v); this.push(b); }
  f64(v) { const b = Buffer.alloc(8); b.writeDoubleLE(v); this.push(b); }
  u64(v) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); this.push(b); }
  i64(v) { const b = Buffer.alloc(8); b.writeBigInt64LE(BigInt(v)); this.push(b); }
  str(s) { const b = Buffer.from(s, 'utf8'); this.u64(b.length); this.push(b); }
  pad(alignment) { const r = this.len % alignment; if (r) this.push(Buffer.alloc(alignment - r)); }
  concat() { return Buffer.concat(this.parts); }
}

function writeValue(w, type, value, elemType) {
  switch (type) {
    case GGUF_TYPE.UINT8: return w.u8(value);
    case GGUF_TYPE.INT8: return w.i8(value);
    case GGUF_TYPE.UINT16: return w.u16(value);
    case GGUF_TYPE.INT16: return w.i16(value);
    case GGUF_TYPE.UINT32: return w.u32(value);
    case GGUF_TYPE.INT32: return w.i32(value);
    case GGUF_TYPE.FLOAT32: return w.f32(value);
    case GGUF_TYPE.BOOL: return w.u8(value ? 1 : 0);
    case GGUF_TYPE.STRING: return w.str(value);
    case GGUF_TYPE.UINT64: return w.u64(value);
    case GGUF_TYPE.INT64: return w.i64(value);
    case GGUF_TYPE.FLOAT64: return w.f64(value);
    case GGUF_TYPE.ARRAY:
      w.u32(elemType);
      w.u64(value.length);
      for (const v of value) writeValue(w, elemType, v, undefined);
      return undefined;
    default: throw new Error(`writer: unsupported value type ${type}`);
  }
}

/**
 * @typedef {{key: string, type: number, value: any, elemType?: number}} MetaKV
 * @typedef {{name: string, dims: number[], type: number, data: Uint8Array}} TensorSpec
 * @param {{version?: number, alignment?: number, metadata?: MetaKV[], tensors?: TensorSpec[]}} spec
 * @returns {Buffer}
 */
export function buildGGUF({ version = 3, alignment = 32, metadata = [], tensors = [] }) {
  const meta = metadata.slice();
  if (alignment !== 32 && !meta.some((m) => m.key === 'general.alignment')) {
    meta.push({ key: 'general.alignment', type: GGUF_TYPE.UINT32, value: alignment });
  }
  const w = new W();
  w.push(Buffer.from('GGUF', 'latin1'));
  w.u32(version);
  w.u64(tensors.length);
  w.u64(meta.length);
  for (const m of meta) {
    w.str(m.key);
    w.u32(m.type);
    writeValue(w, m.type, m.value, m.elemType);
  }
  // tensor infos with aligned offsets
  let off = 0;
  const offsets = [];
  for (const t of tensors) {
    if (!GGML_TYPE_INFO[t.type]) throw new Error(`writer: unknown type ${t.type}`);
    const expect = tensorBytes(t.type, t.dims);
    if (t.data.length !== expect) throw new Error(`writer: tensor ${t.name} has ${t.data.length} bytes, expected ${expect}`);
    off = Math.ceil(off / alignment) * alignment;
    offsets.push(off);
    off += t.data.length;
    w.str(t.name);
    w.u32(t.dims.length);
    for (const d of t.dims) w.u64(d);
    w.u32(t.type);
    w.u64(offsets[offsets.length - 1]);
  }
  w.pad(alignment);
  const dataStart = w.len;
  tensors.forEach((t, i) => {
    const want = dataStart + offsets[i];
    if (w.len < want) w.push(Buffer.alloc(want - w.len));
    w.push(Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength));
  });
  return w.concat();
}
