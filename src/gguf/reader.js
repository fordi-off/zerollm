// GGUF reader: header, metadata KVs, tensor infos. Never reads the whole file into memory.
import fs from 'node:fs';
import { GGUF_TYPE, GGML_TYPE_INFO, tensorBytes } from './types.js';

const MAX_STRING = 1 << 30;
const MAX_ARRAY = 1 << 28;

/** uint64 as Number when safe, BigInt otherwise. @param {bigint} v */
function narrow(v) {
  return v >= BigInt(Number.MIN_SAFE_INTEGER) && v <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(v) : v;
}

/** Buffered sequential reader over a file descriptor. */
class Cursor {
  /** @param {number} fd @param {number} fileSize */
  constructor(fd, fileSize, chunk = 1 << 20) {
    this.fd = fd;
    this.fileSize = fileSize;
    this.chunk = chunk;
    this.pos = 0; // absolute position of next byte
    this.start = 0; // absolute position of buf[0]
    this.buf = Buffer.alloc(0);
  }

  /** Make sure `n` bytes at `pos` are in the buffer; returns offset into buf. */
  need(n) {
    const off = this.pos - this.start;
    if (off >= 0 && off + n <= this.buf.length) return off;
    if (this.pos + n > this.fileSize) throw new Error(`GGUF: unexpected end of file at byte ${this.pos} (need ${n} more, file is ${this.fileSize} bytes)`);
    const len = Math.min(Math.max(n, this.chunk), this.fileSize - this.pos);
    const buf = Buffer.allocUnsafe(len);
    let got = 0;
    while (got < len) {
      const r = fs.readSync(this.fd, buf, got, len - got, this.pos + got);
      if (r === 0) throw new Error('GGUF: short read');
      got += r;
    }
    this.buf = buf;
    this.start = this.pos;
    return 0;
  }

  u8() { const o = this.need(1); this.pos += 1; return this.buf[o]; }
  i8() { const o = this.need(1); this.pos += 1; return this.buf.readInt8(o); }
  u16() { const o = this.need(2); this.pos += 2; return this.buf.readUInt16LE(o); }
  i16() { const o = this.need(2); this.pos += 2; return this.buf.readInt16LE(o); }
  u32() { const o = this.need(4); this.pos += 4; return this.buf.readUInt32LE(o); }
  i32() { const o = this.need(4); this.pos += 4; return this.buf.readInt32LE(o); }
  f32() { const o = this.need(4); this.pos += 4; return this.buf.readFloatLE(o); }
  f64() { const o = this.need(8); this.pos += 8; return this.buf.readDoubleLE(o); }
  u64() { const o = this.need(8); this.pos += 8; return narrow(this.buf.readBigUInt64LE(o)); }
  i64() { const o = this.need(8); this.pos += 8; return narrow(this.buf.readBigInt64LE(o)); }
  /** uint64 that must fit in a Number (lengths, counts, offsets) */
  size(what) {
    const v = this.u64();
    if (typeof v !== 'number') throw new Error(`GGUF: ${what} does not fit in a safe integer`);
    return v;
  }
  string() {
    const len = this.size('string length');
    if (len > MAX_STRING) throw new Error(`GGUF: implausible string length ${len}`);
    const o = this.need(len);
    this.pos += len;
    return this.buf.toString('utf8', o, o + len);
  }
}

/** @param {Cursor} c @param {number} type */
function readValue(c, type) {
  switch (type) {
    case GGUF_TYPE.UINT8: return c.u8();
    case GGUF_TYPE.INT8: return c.i8();
    case GGUF_TYPE.UINT16: return c.u16();
    case GGUF_TYPE.INT16: return c.i16();
    case GGUF_TYPE.UINT32: return c.u32();
    case GGUF_TYPE.INT32: return c.i32();
    case GGUF_TYPE.FLOAT32: return c.f32();
    case GGUF_TYPE.BOOL: return c.u8() !== 0;
    case GGUF_TYPE.STRING: return c.string();
    case GGUF_TYPE.UINT64: return c.u64();
    case GGUF_TYPE.INT64: return c.i64();
    case GGUF_TYPE.FLOAT64: return c.f64();
    case GGUF_TYPE.ARRAY: {
      const elemType = c.u32();
      const len = c.size('array length');
      if (len > MAX_ARRAY) throw new Error(`GGUF: implausible array length ${len}`);
      const out = new Array(len);
      for (let i = 0; i < len; i++) out[i] = readValue(c, elemType);
      return out;
    }
    default: throw new Error(`GGUF: unknown metadata value type ${type}`);
  }
}

/**
 * @typedef {Object} TensorInfo
 * @property {string} name
 * @property {number[]} dims          ggml order: dims[0] is the row length (innermost)
 * @property {number} type            ggml type id
 * @property {number} offset          relative to the data section
 * @property {number} absOffset       absolute file offset
 * @property {number} nElements
 * @property {number|null} nBytes     null if the type is unknown
 */

export class GGUFFile {
  /** @param {object} p */
  constructor(p) {
    this.path = p.path;
    this.fileSize = p.fileSize;
    this.version = p.version;
    /** @type {Map<string, any>} */ this.metadata = p.metadata;
    /** @type {TensorInfo[]} */ this.tensors = p.tensors;
    /** @type {Map<string, TensorInfo>} */ this.tensorMap = new Map(p.tensors.map((t) => [t.name, t]));
    this.alignment = p.alignment;
    this.dataStart = p.dataStart;
  }

  /** @param {string} key @param {any} [dflt] */
  get(key, dflt = undefined) { return this.metadata.has(key) ? this.metadata.get(key) : dflt; }
  /** @param {string} name */
  tensor(name) { return this.tensorMap.get(name); }
  /** @param {string} name */
  requireTensor(name) {
    const t = this.tensorMap.get(name);
    if (!t) throw new Error(`tensor not found in GGUF: ${name}`);
    return t;
  }
}

/**
 * Parse the GGUF header, metadata and tensor infos of a file.
 * @param {string} path
 * @returns {GGUFFile}
 */
export function readGGUF(path) {
  const fd = fs.openSync(path, 'r');
  try {
    const fileSize = fs.fstatSync(fd).size;
    const c = new Cursor(fd, fileSize);
    const o0 = c.need(4);
    const magic = c.buf.toString('latin1', o0, o0 + 4);
    if (magic !== 'GGUF') throw new Error(`not a GGUF file (magic "${JSON.stringify(magic).slice(1, -1)}")`);
    c.pos = 4;
    const version = c.u32();
    if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version} (expected 2 or 3)`);
    const tensorCount = c.size('tensor count');
    const kvCount = c.size('metadata count');
    if (tensorCount > 1e6 || kvCount > 1e6) throw new Error('GGUF: implausible tensor/metadata count (corrupt file?)');

    const metadata = new Map();
    for (let i = 0; i < kvCount; i++) {
      const key = c.string();
      const type = c.u32();
      metadata.set(key, readValue(c, type));
    }

    const alignment = metadata.get('general.alignment') ?? 32;
    if (!Number.isInteger(alignment) || alignment <= 0) throw new Error(`GGUF: bad general.alignment ${alignment}`);

    /** @type {Array<Omit<TensorInfo,'absOffset'|'nBytes'|'nElements'> & {dims:number[]}>} */
    const raw = [];
    for (let i = 0; i < tensorCount; i++) {
      const name = c.string();
      const nDims = c.u32();
      if (nDims < 1 || nDims > 8) throw new Error(`GGUF: tensor ${name} has ${nDims} dims`);
      const dims = [];
      for (let d = 0; d < nDims; d++) dims.push(c.size('tensor dim'));
      const type = c.u32();
      const offset = c.size('tensor offset');
      raw.push({ name, dims, type, offset });
    }
    const dataStart = Math.ceil(c.pos / alignment) * alignment;

    const tensors = raw.map((r) => {
      const nElements = r.dims.reduce((a, b) => a * b, 1);
      const known = GGML_TYPE_INFO[r.type] !== undefined;
      const nBytes = known ? tensorBytes(r.type, r.dims) : null;
      const absOffset = dataStart + r.offset;
      if (nBytes !== null && absOffset + nBytes > fileSize) {
        throw new Error(`GGUF: tensor ${r.name} extends past end of file (truncated download?): needs bytes up to ${absOffset + nBytes}, file has ${fileSize}`);
      }
      return { ...r, absOffset, nElements, nBytes };
    });

    return new GGUFFile({ path, fileSize, version, metadata, tensors, alignment, dataStart });
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Read a byte range of a tensor straight from disk into `dest` (e.g. a view of a wasm memory).
 * Reads are chunked (<= 256 MB each).
 * @param {number} fd open file descriptor
 * @param {TensorInfo} t
 * @param {number} byteStart start within the tensor
 * @param {Uint8Array} dest
 * @param {number} [length] defaults to dest.length
 */
export function readTensorBytes(fd, t, byteStart, dest, length = dest.length) {
  if (t.nBytes === null) throw new Error(`tensor ${t.name}: unknown type ${t.type}`);
  if (byteStart < 0 || byteStart + length > t.nBytes) throw new Error(`tensor ${t.name}: read range out of bounds`);
  const CHUNK = 256 * 1024 * 1024;
  let done = 0;
  while (done < length) {
    const n = Math.min(CHUNK, length - done);
    const r = fs.readSync(fd, dest, done, n, t.absOffset + byteStart + done);
    if (r === 0) throw new Error(`tensor ${t.name}: unexpected end of file`);
    done += r;
  }
  return dest;
}

/** Convenience: read a whole (small) tensor into a new Buffer. @param {GGUFFile} g @param {string} name */
export function readTensor(g, name) {
  const t = g.requireTensor(name);
  const fd = fs.openSync(g.path, 'r');
  try { return readTensorBytes(fd, t, 0, Buffer.allocUnsafe(t.nBytes)); } finally { fs.closeSync(fd); }
}
