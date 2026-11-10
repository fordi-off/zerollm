#!/usr/bin/env node
// Times the reference path (plain JS, one thread) of ONE Qwen3.5-4B-shaped recurrent layer, ONE attention layer, the FFN and
// (a slice of) the 248320 x 2560 Q6_K output projection, using random weights of the REAL shapes and quantization types
// (from the Unsloth Q4_K_M file header). No model file needed. Prints the measured per-component times and a per-token
// ESTIMATE for the full model (24 recurrent + 8 attention layers + output): the estimate assumes the real model costs the same
// as these random-weight layers, which holds for dense quantized matvecs (no data-dependent branches) but is NOT a measurement
// of the real file.
//   node tools/bench-qwen35-ref.js [--reps 5] [--ctx 512] [--rows 24832]
import { GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { randomQuantBlocks } from './random-blocks.js';
import { quantizeQ8_0 } from '../src/compute/quant.js';
import { Act, matvec } from '../src/compute/ref-kernels.js';
import { ropeTables } from '../src/compute/ref-kernels.js';
import { ssmLayerStep, attnLayerStep } from '../src/model/forward-qwen35.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs, num } from '../src/util/args.js';

const { flags } = parseArgs(process.argv.slice(2), {});
const reps = num(flags.reps, 5);
const ctx = num(flags.ctx, 512);
const rand = mulberry32(1);

const weight = (type, rows, cols) => {
  if (type === GGML_TYPE.Q8_0) return { type, rows, cols, data: quantizeQ8_0(Float32Array.from({ length: rows * cols }, () => gaussian(rand) * 0.05)) };
  const info = GGML_TYPE_INFO[type];
  return { type, rows, cols, data: randomQuantBlocks(type, (rows * cols) / info.blockSize, rand) };
};
const vec = (n, f = () => 1) => Float32Array.from({ length: n }, f);

// real Qwen3.5-4B hyperparameters
const cfg = {
  nEmbd: 2560, nFF: 9216, nHead: 16, nHeadKv: 4, headDim: 256, nRot: 64, rmsEps: 1e-6, ropeBase: 1e7,
  ssm: { convKernel: 4, headKDim: 128, headVDim: 128, nKHeads: 16, nVHeads: 32, innerSize: 4096, convDim: 8192 },
};
const Q4 = GGML_TYPE.Q4_K; const Q5 = GGML_TYPE.Q5_K; const Q6 = GGML_TYPE.Q6_K; const Q8 = GGML_TYPE.Q8_0;
const rec = {
  wqkv: weight(Q5, 8192, 2560), wz: weight(Q4, 4096, 2560), wAlpha: weight(Q8, 32, 2560), wBeta: weight(Q8, 32, 2560), wOut: weight(Q5, 2560, 4096),
  conv: vec(8192 * 4, () => gaussian(rand) * 0.3), dtBias: vec(32, () => gaussian(rand) * 0.5), ssmA: vec(32, () => -0.1 - rand()), ssmNorm: vec(128),
};
const attn = { wq: weight(Q4, 8192, 2560), wk: weight(Q4, 1024, 2560), wv: weight(Q6, 1024, 2560), wo: weight(Q4, 2560, 4096), qNorm: vec(256), kNorm: vec(256) };
const ffn = { wGate: weight(Q4, 9216, 2560), wUp: weight(Q4, 9216, 2560), wDown: weight(Q6, 2560, 9216) };
const outRows = num(flags.rows, 24832);
const out = weight(Q6, outRows, 2560);

const x = vec(2560, () => gaussian(rand));
const h = new Act(x);
const sc = {
  qkv: new Float32Array(8192), z: new Float32Array(4096), beta: new Float32Array(32), alpha: new Float32Array(32), core: new Float32Array(4096),
  qfull: new Float32Array(8192), q: new Float32Array(4096), gate: new Float32Array(4096), k: new Float32Array(1024), v: new Float32Array(1024), attOut: new Float32Array(4096),
};
const proj = new Float32Array(2560);
const convState = new Float32Array(3 * 8192); const ssmState = new Float32Array(32 * 128 * 128);
const K = new Float32Array(ctx * 1024); const V = new Float32Array(ctx * 1024);
for (let i = 0; i < K.length; i++) { K[i] = gaussian(rand); V[i] = gaussian(rand); }
const rope = ropeTables(cfg, ctx, null);
const att = new Float32Array(ctx);
const gate = new Float32Array(9216); const up = new Float32Array(9216); const logits = new Float32Array(outRows);

function time(label, fn, n = reps) {
  fn(); // warm-up (JIT)
  const t = [];
  for (let i = 0; i < n; i++) { const t0 = performance.now(); fn(); t.push(performance.now() - t0); }
  t.sort((a, b) => a - b);
  const med = t[Math.floor(t.length / 2)];
  console.log(`${label.padEnd(58)} ${med.toFixed(1).padStart(8)} ms  (median of ${n})`);
  return med;
}

console.log(`Qwen3.5-4B-shaped random weights, plain-JS reference kernels, 1 thread, node ${process.version}\n`);
const tSsm = time('recurrent layer: Gated DeltaNet mixer (qkv,z,a,b,conv,delta,out)', () => ssmLayerStep(cfg, rec, h, convState, ssmState, sc, proj));
const tDelta = time('  of which: delta rule + norm only (32 heads, 128x128)', () => {
  // the state update alone, to show how little of the layer it is
  for (let hv = 0; hv < 32; hv++) for (let j = 0; j < 128; j++) { let s = 0; for (let i = 0; i < 128; i++) s += ssmState[hv * 16384 + j * 128 + i] * sc.qkv[i]; sc.core[hv * 128 + j] = s; }
});
const pos = ctx - 1;
const tAttn = time(`attention layer: gated GQA mixer at position ${pos}`, () => attnLayerStep(cfg, attn, h, K, V, pos, rope, sc, att, proj));
const tAttn0 = time('attention layer: gated GQA mixer at position 0', () => attnLayerStep(cfg, attn, h, K, V, 0, rope, sc, att, proj));
const tFfn = time('SwiGLU FFN (gate, up, down; per layer)', () => { matvec(ffn.wGate, h, gate); matvec(ffn.wUp, h, up); matvec(ffn.wDown, new Act(up), proj); });
const tOut = time(`output projection, ${outRows} of 248320 rows (Q6_K)`, () => matvec(out, h, logits), Math.min(reps, 3));
const tOutFull = tOut * (248320 / outRows);

const perTok = (24 * tSsm + 8 * tAttn + 32 * tFfn + tOutFull) / 1000;
const perTokShort = (24 * tSsm + 8 * tAttn0 + 32 * tFfn + tOutFull) / 1000;
const perPrefill = (24 * tSsm + 8 * tAttn + 32 * tFfn) / 1000; // no output projection while reading the prompt
console.log('\nESTIMATE for the full 4B model (24 recurrent + 8 attention layers, 32 FFNs, full output projection):');
console.log(`  output projection (extrapolated x${(248320 / outRows).toFixed(1)}): ${tOutFull.toFixed(0)} ms`);
console.log(`  decode, context ~${ctx}: ${perTok.toFixed(2)} s/token  =  ${(1 / perTok).toFixed(2)} tok/s   (context ~0: ${(1 / perTokShort).toFixed(2)} tok/s)`);
console.log(`  prompt reading (no logits): ${perPrefill.toFixed(2)} s/token  =  ${(1 / perPrefill).toFixed(2)} tok/s`);
void tDelta;
