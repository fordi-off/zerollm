#!/usr/bin/env node
// Speed of a Qwen3.5 file on the worker pool, without a tokenizer (so it also runs on the synthetic real-shape file from
// tools/make-synth-qwen35.js): prompt reading in batches, then token-by-token decoding, per thread count / kernel set / KV precision.
//   node tools/bench-qwen35-pool.mjs model.gguf [--threads 1,2,4] [--kernels wasm,js] [--kv f32|f16] [--prompt 64] [--n 16] [--ctx 512]
//                                    [--batch 16] [--spin N]
// ZEROLLM_PROFILE=1 prints where worker 0's time goes (matmul kernels / act quantization / attention / conv + delta rule / barriers).
// The token ids are arbitrary (a fixed pseudo-random sequence), the decoded tokens are the argmax of the logits. With the
// synthetic file the numbers are timing only; with a real file they are real throughput on this machine.
import { readGGUF } from '../src/gguf/reader.js';
import { modelConfig } from '../src/model/config.js';
import { PoolModel } from '../src/compute/pool.js';
import { argmax } from '../src/model/generate.js';
import { parseArgs, num } from '../src/util/args.js';
import { formatBytes } from '../src/util/log.js';

const { _, flags } = parseArgs(process.argv.slice(2), {});
if (!_[0]) { console.error('usage: bench-qwen35-pool.mjs model.gguf [--threads 1,2,4] [--kernels wasm,js] [--kv f32|f16] [--prompt 64] [--n 16] [--ctx 512] [--batch 16]'); process.exit(1); }
const g = readGGUF(_[0]);
const cfg = modelConfig(g);
const threads = String(flags.threads ?? '1,4').split(',').map(Number);
const kernels = String(flags.kernels ?? 'wasm').split(',');
const P = num(flags.prompt, 64); const N = num(flags.n, 16);
const ctx = num(flags.ctx, Math.max(512, P + N + 8));
const kv = String(flags.kv ?? 'f32');
const maxBatch = flags.batch === undefined ? 16 : num(flags.batch, 16);
const weightBytes = g.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);
let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed; };
const ids = Array.from({ length: P }, () => rnd() % Math.min(cfg.vocabSize, 100000));

console.log(`${cfg.arch} ${cfg.nLayer}L (${cfg.layerIsRecurrent.filter(Boolean).length} recurrent + ${cfg.nAttnLayers} attention), ${formatBytes(weightBytes)} weights, prompt ${P} tok, decode ${N} tok, ctx ${ctx}, kv ${kv}, node ${process.version}, ${process.arch}\n`);
console.log('| kernels | threads | batch | load s | prefill tok/s | decode tok/s | ms/token decode | eff. GB/s | RSS | finite |\n|---|---|---|---|---|---|---|---|---|---|');
for (const k of kernels) for (const n of threads) {
  const t0 = performance.now();
  const m = await PoolModel.create(g, { ctx, workers: n, kernels: k, kv, maxBatch, spin: flags.spin === undefined ? undefined : num(flags.spin, 20000) });
  const loadS = (performance.now() - t0) / 1000;
  try {
    let finite = true;
    const tp = performance.now();
    for (let i = 0; i < P; i += m.maxBatch) {
      const chunk = ids.slice(i, i + m.maxBatch);
      const lg = await m.forwardBatch(chunk, { logits: i + chunk.length >= P });
      if (lg && !Number.isFinite(lg[0])) finite = false;
    }
    const prefill = (P * 1000) / (performance.now() - tp);
    let logits = await m.forward(ids[0]);
    const td = performance.now();
    for (let i = 0; i < N; i++) {
      logits = await m.forward(argmax(logits));
      if (!Number.isFinite(logits[0])) finite = false;
    }
    const decodeMs = (performance.now() - td) / N;
    console.log(`| ${m.kernels === 'wasm' ? 'wasm-simd' : 'js'} | ${n} | ${m.maxBatch} | ${loadS.toFixed(1)} | ${prefill.toFixed(2)} | ${(1000 / decodeMs).toFixed(2)} | ${decodeMs.toFixed(0)} | ${((weightBytes / 1e9) * (1000 / decodeMs)).toFixed(2)} | ${formatBytes(process.memoryUsage().rss)} | ${finite ? 'yes' : 'NO'} |`);
  } finally { await m.close(); }
}
