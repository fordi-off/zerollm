import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { naiveLogits } from './naive-forward.js';
import { RefModel } from '../src/model/forward.js';
import { PoolModel } from '../src/compute/pool.js';
import { generate } from '../src/model/generate.js';
import { typeName } from '../src/gguf/types.js';

// K-quants need rows that are multiples of 256: hidden 256, 4 heads x 64, 2 KV heads, ffn 512.
const DIMS = { nEmbd: 256, nHead: 4, nHeadKv: 2, headDim: 64, nFF: 512, nLayer: 2 };
const cosine = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y); };
const argmax = (a) => a.reduce((m, v, i) => (v > a[m] ? i : m), 0);
const TOKS = [3, 70, 5, 200, 12, 9, 255, 1, 44];

for (const quant of ['q5_0', 'q4_k', 'q5_k', 'q6_k', 'q4_k_m']) {
  for (const arch of quant === 'q4_k_m' ? ['qwen3', 'llama', 'qwen2'] : ['qwen3']) {
    test(`${quant} / ${arch}: the model really uses the format; JS forward ~ naive oracle; JS pool bit-exact; SIMD pool tracks`, async () => {
      const { g } = tinyFile({ arch, quant, seed: 9, ...DIMS });
      const types = new Set(g.tensors.map((t) => typeName(t.type)));
      const expected = { q5_0: 'Q5_0', q4_k: 'Q4_K', q5_k: 'Q5_K', q6_k: 'Q6_K', q4_k_m: 'Q6_K' }[quant];
      assert.ok(types.has(expected), `has ${expected}: ${[...types]}`);
      if (quant === 'q4_k_m') assert.ok(types.has('Q4_K'));

      const oracle = naiveLogits(g, TOKS);          // dequantized weights, unquantized activations, doubles
      const ref = new RefModel(g, { ctx: 32 });
      const refLogits = TOKS.map((t) => Float32Array.from(ref.forward(t)));
      TOKS.forEach((_, i) => {
        assert.ok(cosine(refLogits[i], oracle[i]) > 0.999, `pos ${i}: cos ${cosine(refLogits[i], oracle[i])}`);
        assert.equal(argmax(refLogits[i]), argmax(oracle[i]), `argmax at ${i}`);
      });

      for (const workers of [1, 3]) {
        const js = await PoolModel.create(g, { ctx: 32, workers, kernels: 'js' });
        const wasm = await PoolModel.create(g, { ctx: 32, workers, kernels: 'wasm' });
        try {
          let exact = 0;
          for (let i = 0; i < TOKS.length; i++) {
            assert.deepEqual([...await js.forward(TOKS[i])], [...refLogits[i]], `js pool workers=${workers} pos ${i}`);
            const w = await wasm.forward(TOKS[i]);
            const top = Math.max(...refLogits[i].map(Math.abs));
            const diff = w.reduce((m, v, j) => Math.max(m, Math.abs(v - refLogits[i][j])), 0);
            // A kernel bug produces garbage; float-order differences produce either ~1e-7 (almost always) or, when an int8
            // activation rounding flips (see docs/DECISIONS.md, M6/M8), a discrete jump of at most a percent or so.
            assert.ok(diff <= 2e-2 * top, `simd pool workers=${workers} pos ${i}: diff ${diff} vs ${top}`);
            assert.ok(cosine(w, refLogits[i]) > 0.9999, `cosine at ${i}`);
            if (diff <= 1e-3 * top) exact++;
            const sorted = [...refLogits[i]].sort((a, b) => b - a);
            if (sorted[0] - sorted[1] > 3 * diff) assert.equal(argmax(w), argmax(refLogits[i]), `argmax at ${i}`);
          }
          assert.ok(exact >= TOKS.length - 3, `most positions agree to rounding noise (${exact}/${TOKS.length})`);
        } finally { await js.close(); await wasm.close(); }
      }
    });
  }
}

test('batched prefill on K-quant models (decode-kernel fallback) is bit-identical to token-by-token', async () => {
  for (const quant of ['q4_k_m', 'q5_0']) {
    const { g } = tinyFile({ arch: 'qwen3', quant, seed: 4, ...DIMS });
    const seq = await PoolModel.create(g, { ctx: 32, workers: 2, kernels: 'wasm' });
    const want = []; for (const t of TOKS) want.push(Float32Array.from(await seq.forward(t)));
    await seq.close();
    const bat = await PoolModel.create(g, { ctx: 32, workers: 2, kernels: 'wasm', maxBatch: 16 });
    try {
      await bat.forwardBatch(TOKS.slice(0, 4), { logits: false });
      const got = await bat.forwardBatch(TOKS.slice(4));
      assert.deepEqual([...got], [...want[TOKS.length - 1]], quant);
    } finally { await bat.close(); }
  }
});

test('greedy generation on a Q4_K_M-style model is identical across thread counts and between JS/pool paths', async () => {
  const { g } = tinyFile({ arch: 'qwen3', quant: 'q4_k_m', seed: 21, ...DIMS });
  const collect = async (m) => { const o = []; for await (const t of generate(m, [5, 6, 7, 8, 9], { maxTokens: 12 })) o.push(t); return o; };
  const ref = await collect(new RefModel(g, { ctx: 64 }));
  for (const workers of [1, 2, 4]) {
    const pool = await PoolModel.create(g, { ctx: 64, workers, kernels: 'js' });
    try { assert.deepEqual(await collect(pool), ref); } finally { await pool.close(); }
  }
  const simd = [];
  for (const workers of [1, 2, 4]) {
    const pool = await PoolModel.create(g, { ctx: 64, workers, kernels: 'wasm' });
    try { simd.push(await collect(pool)); } finally { await pool.close(); }
  }
  assert.deepEqual(simd[0], simd[1]); assert.deepEqual(simd[0], simd[2]); // SIMD output is independent of worker count
});

test('unsupported quantization formats fail early with a clear message (and info flags them)', async () => {
  const { buildGGUF } = await import('../tools/gguf-writer.js');
  const { GGUF_TYPE, GGML_TYPE } = await import('../src/gguf/types.js');
  const { readGGUF } = await import('../src/gguf/reader.js');
  const { assertSupportedTensorTypes } = await import('../src/model/config.js');
  const fs = await import('node:fs'); const os = await import('node:os'); const pathMod = await import('node:path');
  const buf = buildGGUF({
    metadata: [{ key: 'general.architecture', type: GGUF_TYPE.STRING, value: 'llama' }],
    tensors: [{ name: 'blk.0.attn_q.weight', dims: [256, 2], type: GGML_TYPE.Q3_K, data: new Uint8Array(2 * 110) }],
  });
  const p = pathMod.join(os.tmpdir(), `unsupported-${process.pid}.gguf`);
  fs.writeFileSync(p, buf);
  try {
    assert.throws(() => assertSupportedTensorTypes(readGGUF(p)), /Q3_K \(1 tensors\).*Supported:.*Q4_K/s);
    const { spawnSync } = await import('node:child_process');
    const { fileURLToPath } = await import('node:url');
    const BIN = pathMod.join(pathMod.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
    const r = spawnSync(process.execPath, [BIN, 'info', p], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stdout, /Q3_K.*NOT supported/);
  } finally { fs.unlinkSync(p); }
});
