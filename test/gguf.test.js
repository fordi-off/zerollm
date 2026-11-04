import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { buildGGUF } from '../tools/gguf-writer.js';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
import { readGGUF, readTensor, readTensorBytes } from '../src/gguf/reader.js';
import { GGUF_TYPE, GGML_TYPE, GGML_TYPE_INFO, rowSize, tensorBytes } from '../src/gguf/types.js';
import { modelConfig, UnsupportedArchError, estimateRam, kvBytesPerToken } from '../src/model/config.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-test-'));
const write = (name, buf) => { const p = path.join(dir, name); fs.writeFileSync(p, buf); return p; };
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));

test('type sizes match ggml block structs', () => {
  assert.equal(rowSize(GGML_TYPE.Q8_0, 64), 68);
  assert.equal(rowSize(GGML_TYPE.Q4_0, 64), 36);
  assert.equal(rowSize(GGML_TYPE.Q4_K, 512), 288);
  assert.equal(GGML_TYPE_INFO[GGML_TYPE.Q6_K].typeSize, 128 + 64 + 16 + 2);
  assert.equal(GGML_TYPE_INFO[GGML_TYPE.Q5_K].typeSize, 2 + 2 + 12 + 128 + 32);
  assert.equal(GGML_TYPE_INFO[GGML_TYPE.Q5_0].typeSize, 2 + 4 + 16);
  assert.equal(tensorBytes(GGML_TYPE.F32, [4, 3, 2]), 96);
  assert.throws(() => rowSize(GGML_TYPE.Q8_0, 33));
});

test('parses every metadata value type, arrays, alignment and tensor data', () => {
  const t1 = new Float32Array([1, 2, 3, 4, 5, 6]);
  const t2 = Uint8Array.from({ length: 68 }, (_, i) => i); // 1 row of 64 Q8_0 = 2 blocks
  const buf = buildGGUF({
    alignment: 64,
    metadata: [
      { key: 'a.u8', type: GGUF_TYPE.UINT8, value: 200 },
      { key: 'a.i8', type: GGUF_TYPE.INT8, value: -5 },
      { key: 'a.u16', type: GGUF_TYPE.UINT16, value: 65000 },
      { key: 'a.i16', type: GGUF_TYPE.INT16, value: -30000 },
      { key: 'a.u32', type: GGUF_TYPE.UINT32, value: 4000000000 },
      { key: 'a.i32', type: GGUF_TYPE.INT32, value: -2000000000 },
      { key: 'a.f32', type: GGUF_TYPE.FLOAT32, value: 1.5 },
      { key: 'a.bool', type: GGUF_TYPE.BOOL, value: true },
      { key: 'a.str', type: GGUF_TYPE.STRING, value: 'héllo 世界 🚀' },
      { key: 'a.u64', type: GGUF_TYPE.UINT64, value: 2 ** 40 },
      { key: 'a.u64big', type: GGUF_TYPE.UINT64, value: 2n ** 63n },
      { key: 'a.i64', type: GGUF_TYPE.INT64, value: -(2 ** 40) },
      { key: 'a.f64', type: GGUF_TYPE.FLOAT64, value: Math.PI },
      { key: 'a.arr', type: GGUF_TYPE.ARRAY, elemType: GGUF_TYPE.INT32, value: [1, -2, 3] },
      { key: 'a.sarr', type: GGUF_TYPE.ARRAY, elemType: GGUF_TYPE.STRING, value: ['x', '', 'zz'] },
      { key: 'a.empty', type: GGUF_TYPE.ARRAY, elemType: GGUF_TYPE.UINT32, value: [] },
    ],
    tensors: [
      { name: 't1', dims: [3, 2], type: GGML_TYPE.F32, data: new Uint8Array(t1.buffer) },
      { name: 't2', dims: [64], type: GGML_TYPE.Q8_0, data: t2 },
    ],
  });
  const p = write('all.gguf', buf);
  const g = readGGUF(p);
  assert.equal(g.version, 3);
  assert.equal(g.alignment, 64);
  assert.equal(g.dataStart % 64, 0);
  assert.equal(g.get('a.u8'), 200);
  assert.equal(g.get('a.i8'), -5);
  assert.equal(g.get('a.u16'), 65000);
  assert.equal(g.get('a.i16'), -30000);
  assert.equal(g.get('a.u32'), 4000000000);
  assert.equal(g.get('a.i32'), -2000000000);
  assert.equal(g.get('a.f32'), 1.5);
  assert.equal(g.get('a.bool'), true);
  assert.equal(g.get('a.str'), 'héllo 世界 🚀');
  assert.equal(g.get('a.u64'), 2 ** 40);
  assert.equal(g.get('a.u64big'), 2n ** 63n); // beyond safe range stays BigInt
  assert.equal(g.get('a.i64'), -(2 ** 40));
  assert.equal(g.get('a.f64'), Math.PI);
  assert.deepEqual(g.get('a.arr'), [1, -2, 3]);
  assert.deepEqual(g.get('a.sarr'), ['x', '', 'zz']);
  assert.deepEqual(g.get('a.empty'), []);
  assert.equal(g.get('missing', 'dflt'), 'dflt');

  assert.equal(g.tensors.length, 2);
  const a = g.tensor('t1');
  assert.deepEqual(a.dims, [3, 2]);
  assert.equal(a.nBytes, 24);
  assert.equal(a.nElements, 6);
  const b = g.tensor('t2');
  assert.equal(b.offset % 64, 0);
  assert.equal(b.absOffset % 64, 0);
  const raw1 = readTensor(g, 't1');
  assert.deepEqual([...new Float32Array(Uint8Array.from(raw1).buffer)], [...t1]);
  assert.deepEqual([...readTensor(g, 't2')], [...t2]);

  // partial range read straight into a destination buffer
  const fd = fs.openSync(p, 'r');
  const dest = new Uint8Array(10);
  readTensorBytes(fd, b, 34, dest);
  fs.closeSync(fd);
  assert.deepEqual([...dest], [...t2.subarray(34, 44)]);
});

test('metadata larger than the read chunk (long strings, big arrays)', () => {
  const big = 'x'.repeat(3 * 1024 * 1024);
  const arr = Array.from({ length: 200000 }, (_, i) => `tok${i}`);
  const p = write('big.gguf', buildGGUF({
    metadata: [
      { key: 's', type: GGUF_TYPE.STRING, value: big },
      { key: 'arr', type: GGUF_TYPE.ARRAY, elemType: GGUF_TYPE.STRING, value: arr },
    ],
  }));
  const g = readGGUF(p);
  assert.equal(g.get('s').length, big.length);
  assert.equal(g.get('arr').length, 200000);
  assert.equal(g.get('arr')[199999], 'tok199999');
});

test('rejects bad magic, bad version, truncated files', () => {
  assert.throws(() => readGGUF(write('bad.gguf', Buffer.from('NOPE0000000000000000'))), /not a GGUF/);
  const v9 = buildGGUF({ version: 3 });
  v9.writeUInt32LE(9, 4);
  assert.throws(() => readGGUF(write('v9.gguf', v9)), /version/);
  const good = buildGGUF({ tensors: [{ name: 't', dims: [32], type: GGML_TYPE.F32, data: new Uint8Array(128) }] });
  assert.throws(() => readGGUF(write('trunc.gguf', good.subarray(0, good.length - 10))), /truncated|past end/);
  assert.throws(() => readGGUF(write('trunc2.gguf', good.subarray(0, 20))), /unexpected end/);
});

test('tiny model: config extraction for each architecture', () => {
  for (const arch of ['qwen3', 'qwen2', 'llama']) {
    const { buffer, cfg: tiny } = makeTinyModel({ arch, quant: 'q8_0' });
    const g = readGGUF(write(`${arch}.gguf`, buffer));
    const c = modelConfig(g);
    assert.equal(c.arch, arch);
    assert.equal(c.nEmbd, tiny.nEmbd);
    assert.equal(c.nHead, 4);
    assert.equal(c.nHeadKv, 2);
    assert.equal(c.headDim, tiny.headDim);
    assert.equal(c.vocabSize, tiny.vocab);
    assert.equal(c.ropeType, arch === 'llama' ? 'norm' : 'neox');
    assert.equal(c.qkNorm, arch === 'qwen3');
    assert.equal(c.qkvBias, arch === 'qwen2');
    assert.equal(c.tiedEmbeddings, arch !== 'llama');
    const est = estimateRam(g, c, { ctx: 100 });
    assert.equal(est.kv, kvBytesPerToken(c) * 100);
    assert.equal(kvBytesPerToken(c), 2 * c.nLayer * c.nHeadKv * c.headDim * 4);
  }
});

test('qwen3 head_dim comes from key_length, not hidden/heads', () => {
  const { buffer } = makeTinyModel({ arch: 'qwen3', nEmbd: 64, nHead: 4, headDim: 48 });
  const c = modelConfig(readGGUF(write('hd.gguf', buffer)));
  assert.equal(c.headDim, 48);
  assert.notEqual(c.headDim, c.nEmbd / c.nHead);
});

test('unsupported architecture gives a clear error', () => {
  const p = write('mamba.gguf', buildGGUF({ metadata: [{ key: 'general.architecture', type: GGUF_TYPE.STRING, value: 'mamba' }] }));
  assert.throws(() => modelConfig(readGGUF(p)), UnsupportedArchError);
});
