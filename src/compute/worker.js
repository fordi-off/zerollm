// SPMD worker. Every worker runs the same control flow for a batch of tokens; phases are separated by barriers.
// Each worker owns a contiguous row slice of every weight matrix, stored in its own private WebAssembly.Memory
// (so the 4 GB wasm32 limit applies per worker, not to the whole model).
//
// Hybrid models (qwen35: Gated DeltaNet + gated attention) use the same phases. A recurrent layer is sharded by VALUE HEAD:
// a worker owns the delta-rule matrices and the conv window of its value heads (they live in SharedArrayBuffers, so the main
// thread can snapshot them for truncate()), computes the q/k rows of its share of the KEY heads (conv + L2 norm included),
// and after a barrier reads whatever q/k its value heads need from shared memory. Within a batch the conv and the delta rule
// run token by token (they are recurrences); the matmuls around them are batched like everywhere else.
//
// A command carries B tokens at consecutive positions. B = 1 is normal decoding (single-vector SIMD kernels);
// B > 1 is batched prefill: every weight tile is read once for all B tokens (matmul kernels).
import fs from 'node:fs';
import { parentPort, workerData } from 'node:worker_threads';
import { readTensorBytes } from '../gguf/reader.js';
import { GGML_TYPE, GGML_TYPE_INFO } from '../gguf/types.js';
import { f16Table, f32ToF16 } from '../util/f16.js';
import { Act, matvec, dequantRow, rmsnorm, softmax, silu, ropeTables, applyRope, rowBytes, quantizeAct } from './ref-kernels.js';
import { instantiateKernels, resolveDotMode, usesInt8Acts } from './wasm/kernels.js';
import { Barrier } from './barrier.js';
import { convStep, l2normHeads, deltaRuleHead, gatedRmsNormHead, sigmoid, softplus, keyHeadOfValueHead } from './ref-deltanet.js';
import { CTRL, CMD_EXIT, CMD_KV_EXPORT, CMD_KV_IMPORT, splitRows, splitRange, recurrentShards, alignUp } from './layout.js';

const { id, n, cfg, tensors, path, ctx, shared, spin, kernels, maxBatch, logitsBatch, kvType } = workerData;
const kvF16 = kvType === 'f16';
// opt-in profile of worker 0 (ZEROLLM_PROFILE=1): where does a token's time go?
const prof = process.env.ZEROLLM_PROFILE === '1' && id === 0;
const PT = { kernel: 0, quant: 0, attn: 0, conv: 0, delta: 0, barrier: 0, total: 0, tokens: 0, batches: 0 };

/** Fatal errors: tell the pool, then stop. */
function fail(e) {
  try { Atomics.store(new Int32Array(shared.ctrl), CTRL.ERROR, id + 1); } catch { /* ignore */ }
  parentPort.postMessage({ type: 'error', id, message: e?.stack ?? String(e) });
}

try {
  main();
} catch (e) {
  fail(e);
}

function main() {
  const t0 = performance.now();
  const { nEmbd, nHead, nHeadKv, headDim, nRot, rmsEps, ropeType, nLayer, nFF, vocabSize } = cfg;
  const qDim = nHead * headDim;
  const kvDim = nHeadKv * headDim;
  const BM = maxBatch;

  // ---------------------------------------------------------------- weight slices -> private wasm memory
  /** @type {Array<{key: string, t: any, r0: number, r1: number, off: number, bytes: number}>} */
  const plan = [];
  let total = 0;
  /** rows [r0, r1) of matrix `name` (any range: the kernels handle row counts that are not a multiple of 4) */
  const addRows = (key, name, r0, r1) => {
    const t = tensors[name];
    if (!t) throw new Error(`worker ${id}: tensor ${name} missing from plan`);
    const cols = t.dims[0];
    const bytes = (r1 - r0) * rowBytes(t.type, cols);
    plan.push({ key, t, r0, r1, off: total, bytes });
    total += alignUp(bytes, 64);
  };
  /** my even share of the rows of matrix `name` */
  const addMat = (key, name) => { const [r0, r1] = splitRows(tensors[name]?.dims[1] ?? 0, n, id); addRows(key, name, r0, r1); };
  const hybrid = cfg.hybrid === true;
  const ssm = cfg.ssm;
  const keyDim = hybrid ? ssm.nKHeads * ssm.headKDim : 0;
  const dk = hybrid ? ssm.headKDim : 0; const dv = hybrid ? ssm.headVDim : 0;
  const convDim = hybrid ? ssm.convDim : 0; const inner = hybrid ? ssm.innerSize : 0;
  const convM = hybrid ? ssm.convKernel - 1 : 0;
  const { kk0, kk1, vh0, vh1 } = hybrid ? recurrentShards(ssm, n, id) : { kk0: 0, kk1: 0, vh0: 0, vh1: 0 };
  const nOwnV = vh1 - vh0;
  const nOwnK = kk1 - kk0;
  /** recurrent layer number (index into the shared conv / ssm state) or -1; attention layer number or -1 (KV cache index) */
  const recIdx = []; const attnIdx = [];
  for (let l = 0, r = 0, a = 0; l < nLayer; l++) {
    const rec = hybrid && cfg.layerIsRecurrent[l];
    recIdx.push(rec ? r++ : -1); attnIdx.push(rec ? -1 : a++);
  }
  addMat('embd', 'token_embd.weight');
  if (!cfg.tiedEmbeddings) addMat('output', 'output.weight');
  for (let l = 0; l < nLayer; l++) {
    const p = `blk.${l}.`;
    if (recIdx[l] >= 0) {
      // attn_qkv rows: [q: nKHeads x dk | k: nKHeads x dk | v: nVHeads x dv]. My key heads' q and k rows, my value heads' v rows.
      addRows(`${l}.qkvQ`, `${p}attn_qkv.weight`, kk0 * dk, kk1 * dk);
      addRows(`${l}.qkvK`, `${p}attn_qkv.weight`, keyDim + kk0 * dk, keyDim + kk1 * dk);
      addRows(`${l}.qkvV`, `${p}attn_qkv.weight`, 2 * keyDim + vh0 * dv, 2 * keyDim + vh1 * dv);
      addRows(`${l}.z`, `${p}attn_gate.weight`, vh0 * dv, vh1 * dv);
      addRows(`${l}.alpha`, `${p}ssm_alpha.weight`, vh0, vh1);
      addRows(`${l}.beta`, `${p}ssm_beta.weight`, vh0, vh1);
      addMat(`${l}.wo`, `${p}ssm_out.weight`);
    } else {
      addMat(`${l}.wq`, `${p}attn_q.weight`); addMat(`${l}.wk`, `${p}attn_k.weight`); addMat(`${l}.wv`, `${p}attn_v.weight`);
      addMat(`${l}.wo`, `${p}attn_output.weight`);
    }
    addMat(`${l}.gate`, `${p}ffn_gate.weight`); addMat(`${l}.up`, `${p}ffn_up.weight`); addMat(`${l}.down`, `${p}ffn_down.weight`);
  }
  // This worker owns a set of KV heads (and every query head that shares them): their K/V cache is PRIVATE and lives
  // in this worker's wasm memory, so the attention kernels can read it directly and no barrier is needed for it.
  const [kh0, kh1] = splitRange(nHeadKv, n, id);
  const nOwn = kh1 - kh0;
  const group = nHead / nHeadKv;
  // scratch for the SIMD kernels, in the same memory as the weights
  const useWasm = kernels === 'wasm';
  const attnSimd = useWasm && headDim % 16 === 0;
  const qRows = cfg.attnOutputGate ? 2 * qDim : qDim; // attn_q rows; hybrid models: [query | output gate] per head
  const qStride = cfg.attnOutputGate ? 2 * headDim : headDim; // q rows per head
  const maxCols = Math.max(nEmbd, qDim, nFF, inner);
  const maxRows = Math.max(1, ...plan.map((p) => p.r1 - p.r0));
  const maxRowsBatch = Math.max(1, ...plan.filter((p) => p.key !== 'embd' && p.key !== 'output').map((p) => p.r1 - p.r0));
  const outPlan = plan.find((p) => p.key === (cfg.tiedEmbeddings ? 'embd' : 'output'));
  const outRows = outPlan.r1 - outPlan.r0;
  const XQ = total; total += alignUp(maxCols, 64);                       // int8 activation (decode)
  const XD = total; total += alignUp((maxCols / 32) * 4, 64);            // its f32 scales
  const XQ16 = total; total += alignUp(BM * maxCols * 2, 64);            // int16-widened activations (batched)
  const XDB = total; total += alignUp(BM * (maxCols / 32) * 4, 64);      // their scales
  const XS = total; total += alignUp((maxCols / 32) * 4, 64);            // per-block quant sums (decode; used by Q4_K/Q5_K)
  const XSB = total; total += alignUp(BM * (maxCols / 32) * 4, 64);      // same for batches
  const YO = total; total += alignUp(Math.max(maxRows, BM * maxRowsBatch, logitsBatch * outRows) * 4, 64); // f32 results (logits of a whole batch with logits 'all')
  const QHP = total; total += alignUp(headDim * 4, 64);                  // attention: query head
  const SCP = total; total += alignUp(ctx * 4, 64);                       // attention: scores / probabilities
  const OHP = total; total += alignUp(headDim * 4, 64);                   // attention: output head
  const kvPerLayer = 2 * nOwn * ctx * headDim;                            // floats: [k|v][owned head][pos][headDim]
  const kvShift = kvF16 ? 1 : 2;                                          // log2(bytes per cached value)
  const nKvLayers = cfg.nAttnLayers ?? nLayer;                            // hybrid models: only attention layers have a KV cache
  const KVB = total; total += alignUp((nKvLayers * kvPerLayer) << kvShift, 64);
  const pages = Math.max(1, Math.ceil(total / 65536));
  if (pages > 65536) throw new Error(`worker ${id}: weight slice is ${(total / 2 ** 30).toFixed(2)} GB, over the 4 GB wasm32 limit; use more workers or a smaller --ctx`);
  let mem;
  try {
    mem = new WebAssembly.Memory({ initial: pages });
  } catch (e) {
    throw new Error(`worker ${id}: could not allocate ${(total / 2 ** 20).toFixed(0)} MB of WebAssembly memory (${e.message}). Close other apps or use a smaller model/quant.`);
  }
  const fd = fs.openSync(path, 'r');
  /** @type {Record<string, any>} */
  const W = {};
  let loaded = 0; let lastPost = 0;
  let outputNorm; let ropeFreqs; let layers;
  try {
    for (const p of plan) {
      const view = new Uint8Array(mem.buffer, p.off, p.bytes);
      if (p.bytes) readTensorBytes(fd, p.t, p.r0 * rowBytes(p.t.type, p.t.dims[0]), view);
      W[p.key] = { type: p.t.type, rows: p.r1 - p.r0, cols: p.t.dims[0], data: view, off: p.off, kernel: null, mmKernel: null, nb: p.t.dims[0] / (GGML_TYPE_INFO[p.t.type]?.blockSize ?? 32) };
      loaded += p.bytes;
      const now = performance.now();
      if (now - lastPost > 100) { lastPost = now; parentPort.postMessage({ type: 'progress', id, bytes: loaded }); }
    }
    // small vectors: every worker keeps a full private copy
    const vec = (name) => {
      const t = tensors[name];
      const f = new Float32Array(t.nElements);
      if (t.type === 0) readTensorBytes(fd, t, 0, new Uint8Array(f.buffer));
      else { const b = new Uint8Array(t.nBytes); readTensorBytes(fd, t, 0, b); dequantRow({ type: t.type, cols: t.nElements, data: b }, 0, f); }
      return f;
    };
    outputNorm = vec('output_norm.weight');
    ropeFreqs = cfg.hasRopeFreqs ? vec('rope_freqs.weight') : null;
    layers = [];
    for (let l = 0; l < nLayer; l++) {
      const p = `blk.${l}.`;
      const opt = (nm) => (tensors[p + nm] ? vec(p + nm) : null);
      layers.push({
        attnNorm: vec(`${p}attn_norm.weight`), ffnNorm: vec(`${p}${hybrid ? 'post_attention_norm' : 'ffn_norm'}.weight`), // qwen35: post_attention_norm is the pre-FFN norm
        bq: opt('attn_q.bias'), bk: opt('attn_k.bias'), bv: opt('attn_v.bias'),
        qNorm: opt('attn_q_norm.weight'), kNorm: opt('attn_k_norm.weight'),
        conv: opt('ssm_conv1d.weight'), dtBias: opt('ssm_dt.bias'), ssmA: opt('ssm_a'), ssmNorm: opt('ssm_norm.weight'),
      });
    }
  } finally { fs.closeSync(fd); }

  const xqv = new Int8Array(mem.buffer, XQ, maxCols);
  const xdv = new Float32Array(mem.buffer, XD, maxCols / 32);
  const xq16v = new Int16Array(mem.buffer, XQ16, BM * maxCols);
  // ZEROLLM_DOT=relaxed|relaxed-add: the relaxed-SIMD batched kernels read INT8 activations (token t at byte offset t*dim), which fit in
  // the same region (it is sized for int16); every other mode keeps the int16-widened layout.
  const dot = useWasm ? resolveDotMode() : 'extmul';
  const xqBv = usesInt8Acts(dot) ? new Int8Array(mem.buffer, XQ16, BM * maxCols) : xq16v;
  const xdBv = new Float32Array(mem.buffer, XDB, (BM * maxCols) / 32);
  const xsv = new Float32Array(mem.buffer, XS, maxCols / 32);
  const xsBv = new Float32Array(mem.buffer, XSB, (BM * maxCols) / 32);
  const yv = new Float32Array(mem.buffer, YO, Math.max(maxRows, BM * maxRowsBatch, logitsBatch * outRows));
  let k = null;
  if (useWasm) {
    k = instantiateKernels(mem, { dot });
    for (const w of Object.values(W)) {
      // Q8_0 / Q4_0 kernels take (w, xq, xd, y, rows, nb[, ntok]); Q5_0 and the K-quants additionally take the per-block
      // quant sums (w, xq, xd, xs, y, rows, nb[, ntok]).
      const set = (kernel, mmKernel, useXs) => { w.kernel = kernel; w.mmKernel = mmKernel; w.useXs = useXs; };
      if (w.type === GGML_TYPE.Q8_0) set(k.matvec_q8_0, k.mm_q8_0, false);
      else if (w.type === GGML_TYPE.Q4_0) set(k.matvec_q4_0, k.mm_q4_0, false);
      else if (w.type === GGML_TYPE.Q5_0) set(k.matvec_q5_0, k.mm_q5_0, true);
      else if (w.type === GGML_TYPE.Q4_K) set(k.matvec_q4_k, k.mm_q4_k, true);
      else if (w.type === GGML_TYPE.Q5_K) set(k.matvec_q5_k, k.mm_q5_k, true);
      else if (w.type === GGML_TYPE.Q6_K) set(k.matvec_q6_k, k.mm_q6_k, true);
    }
  }

  // ---------------------------------------------------------------- shared views + private scratch
  const ctrl = new Int32Array(shared.ctrl);
  const x = new Float32Array(shared.x); const sq = new Float32Array(shared.q);
  const sk = new Float32Array(shared.k); const sv = new Float32Array(shared.v);
  const att = new Float32Array(shared.att); const hb = new Float32Array(shared.hb);
  const logits = new Float32Array(shared.logits); const kvStage = new Float32Array(shared.kvStage);
  const kvF32 = kvF16 ? null : new Float32Array(mem.buffer, KVB, nKvLayers * kvPerLayer);
  const kvU16 = kvF16 ? new Uint16Array(mem.buffer, KVB, nKvLayers * kvPerLayer) : null;
  const h2f = kvF16 ? f16Table() : null;
  const rawBar = new Barrier(ctrl, CTRL.BAR_COUNT, CTRL.BAR_GEN, n, spin);
  const bar = prof ? { wait() { const t = performance.now(); rawBar.wait(); PT.barrier += performance.now() - t; } } : rawBar;
  const rope = ropeTables(cfg, ctx, ropeFreqs);
  const xbP = new Float32Array(BM * nEmbd); // private normed copies, one per token
  const scores = attnSimd ? new Float32Array(mem.buffer, SCP, ctx) : new Float32Array(ctx);
  const qh = attnSimd ? new Float32Array(mem.buffer, QHP, headDim) : new Float32Array(headDim);
  const oh = attnSimd ? new Float32Array(mem.buffer, OHP, headDim) : new Float32Array(headDim);
  const tmpA = new Float32Array(BM * Math.max(nEmbd, nFF));
  const tmpB = new Float32Array(BM * Math.max(nEmbd, nFF));
  // recurrent layers: live state (shared; this worker only touches its own channels / heads), private per-batch scratch
  const convAll = hybrid ? new Float32Array(shared.conv) : null;
  const ssmAll = hybrid ? new Float32Array(shared.ssm) : null;
  const convSize = convM * convDim; const ssmSize = hybrid ? ssm.nVHeads * dv * dk : 0;
  const zP = new Float32Array(BM * nOwnV * dv); const betaP = new Float32Array(BM * nOwnV); const alphaP = new Float32Array(BM * nOwnV);
  // the conv channel ranges this worker updates: my key heads' q and k channels and my value heads' v channels
  const convRanges = hybrid ? [[kk0 * dk, kk1 * dk], [keyDim + kk0 * dk, keyDim + kk1 * dk], [2 * keyDim + vh0 * dv, 2 * keyDim + vh1 * dv]] : [];
  const tmpF = new Float32Array(maxRows); // fallback matvec output

  const [qr0, qr1] = splitRows(qRows, n, id);
  const [kr0, kr1] = splitRows(kvDim, n, id);
  const [er0, er1] = splitRows(nEmbd, n, id);
  const [fr0, fr1] = splitRows(nFF, n, id);
  const [vr0, vr1] = splitRows(vocabSize, n, id);
  const erN = er1 - er0; const frN = fr1 - fr0;
  const scale = 1 / Math.sqrt(headDim);
  const embd = W.embd; const outW = cfg.tiedEmbeddings ? W.embd : W.output;

  // ---- matmul plumbing: "prep" makes B vectors the current input, "mm" multiplies a weight slice with them
  let srcVec = null; let srcB = 0; let srcDim = 0;
  /** B vectors of length `dim` stored back to back in `src` become the current matmul input (quantized once) */
  function prep(src, B, dim) {
    const t = prof ? performance.now() : 0;
    srcVec = src; srcB = B; srcDim = dim;
    if (useWasm) {
      if (B === 1) quantizeAct(src.subarray(0, dim), xqv, xdv, xsv);
      else {
        const nbl = dim / 32;
        for (let b = 0; b < B; b++) quantizeAct(src.subarray(b * dim, (b + 1) * dim), xqBv.subarray(b * dim, (b + 1) * dim), xdBv.subarray(b * nbl, (b + 1) * nbl), xsBv.subarray(b * nbl, (b + 1) * nbl));
      }
    }
    if (prof) PT.quant += performance.now() - t;
  }
  /** dest[b * destStride + destOff + i] = (W_slice . vector_b)[i] */
  function mm(w, dest, destStride, destOff) {
    if (w.rows === 0) return;
    const t = prof ? performance.now() : 0;
    const B = srcB;
    if (w.kernel) {
      if (B === 1) {
        if (w.useXs) w.kernel(w.off, XQ, XD, XS, YO, w.rows, w.nb); else w.kernel(w.off, XQ, XD, YO, w.rows, w.nb);
        dest.set(yv.subarray(0, w.rows), destOff);
      } else {
        if (w.useXs) w.mmKernel(w.off, XQ16, XDB, XSB, YO, w.rows, w.nb, B); else w.mmKernel(w.off, XQ16, XDB, YO, w.rows, w.nb, B);
        for (let b = 0; b < B; b++) dest.set(yv.subarray(b * w.rows, (b + 1) * w.rows), b * destStride + destOff);
      }
    } else {
      for (let b = 0; b < B; b++) {
        matvec(w, new Act(srcVec.subarray(b * srcDim, (b + 1) * srcDim)), tmpF);
        dest.set(tmpF.subarray(0, w.rows), b * destStride + destOff);
      }
    }
    if (prof) PT.kernel += performance.now() - t;
  }

  parentPort.postMessage({ type: 'ready', id, ms: performance.now() - t0, bytes: total, sliceBytes: loaded, kernels: useWasm ? 'wasm-simd' : 'js', dot });

  // ---------------------------------------------------------------- token mixers: full attention and Gated DeltaNet
  // Both leave x updated (residual added) for my rows, after a final barrier.

  /** Full attention layer l (phases P1-P3). Hybrid models: attn_q rows are [query | gate] per head, the gate multiplies the attention output. */
  function attentionMixer(l, L, B, pos0) {
    const ai = attnIdx[l];
    // P1: norm (redundant per worker) + my rows of q, k, v for all B tokens
    for (let b = 0; b < B; b++) rmsnorm(xbP, x, L.attnNorm, rmsEps, nEmbd, b * nEmbd, b * nEmbd);
    prep(xbP, B, nEmbd);
    mm(W[`${l}.wq`], sq, qRows, qr0);
    mm(W[`${l}.wk`], sk, kvDim, kr0);
    mm(W[`${l}.wv`], sv, kvDim, kr0);
    for (let b = 0; b < B; b++) {
      if (L.bq) for (let i = qr0; i < qr1; i++) sq[b * qRows + i] += L.bq[i];
      if (L.bk) for (let i = kr0; i < kr1; i++) sk[b * kvDim + i] += L.bk[i];
      if (L.bv) for (let i = kr0; i < kr1; i++) sv[b * kvDim + i] += L.bv[i];
    }
    bar.wait();

    // P2: my KV heads. For every token: k-norm + RoPE, append K and V to my PRIVATE cache, then causal attention for
    // each query head of the group that shares this KV head (q-norm, RoPE, scores, softmax, weighted sum of V).
    const ta = prof ? performance.now() : 0;
    for (let b = 0; b < B; b++) {
      const pos = pos0 + b;
      for (let hd = kh0; hd < kh1; hd++) {
        const hl = hd - kh0;
        const o = b * kvDim + hd * headDim;
        if (L.kNorm) rmsnorm(sk, sk, L.kNorm, rmsEps, headDim, o, o);
        applyRope(sk.subarray(o, o + headDim), 1, headDim, nRot, ropeType, rope, pos);
        const kBase = ((ai * 2) * nOwn + hl) * ctx * headDim; const vBase = ((ai * 2 + 1) * nOwn + hl) * ctx * headDim;
        if (kvF16) {
          for (let i = 0; i < headDim; i++) { kvU16[kBase + pos * headDim + i] = f32ToF16(sk[o + i]); kvU16[vBase + pos * headDim + i] = f32ToF16(sv[o + i]); }
        } else {
          kvF32.set(sk.subarray(o, o + headDim), kBase + pos * headDim);
          kvF32.set(sv.subarray(o, o + headDim), vBase + pos * headDim);
        }
        for (let g = 0; g < group; g++) {
          const qhd = hd * group + g;
          const qo = b * qRows + qhd * qStride;
          qh.set(sq.subarray(qo, qo + headDim));
          if (L.qNorm) rmsnorm(qh, qh, L.qNorm, rmsEps, headDim);
          applyRope(qh, 1, headDim, nRot, ropeType, rope, pos);
          if (attnSimd) {
            (kvF16 ? k.attn_scores_f16 : k.attn_scores)(QHP, KVB + (kBase << kvShift), SCP, pos + 1, headDim, scale);
            softmax(scores, pos + 1);
            (kvF16 ? k.attn_av_f16 : k.attn_av)(SCP, KVB + (vBase << kvShift), OHP, pos + 1, headDim);
          } else {
            for (let t = 0; t <= pos; t++) {
              const ko = kBase + t * headDim;
              let s = 0;
              for (let i = 0; i < headDim; i++) s += qh[i] * (kvF16 ? h2f[kvU16[ko + i]] : kvF32[ko + i]);
              scores[t] = s * scale;
            }
            softmax(scores, pos + 1);
            oh.fill(0);
            for (let t = 0; t <= pos; t++) {
              const p = scores[t];
              const vo = vBase + t * headDim;
              for (let i = 0; i < headDim; i++) oh[i] += p * (kvF16 ? h2f[kvU16[vo + i]] : kvF32[vo + i]);
            }
          }
          if (cfg.attnOutputGate) for (let i = 0; i < headDim; i++) oh[i] *= sigmoid(sq[qo + headDim + i]); // output gate of this head
          att.set(oh, b * qDim + qhd * headDim);
        }
      }
    }
    if (prof) PT.attn += performance.now() - ta;
    bar.wait();

    // P3: output projection, residual add into my rows of x
    prep(att, B, qDim);
    mm(W[`${l}.wo`], tmpA, erN, 0);
    for (let b = 0; b < B; b++) for (let i = 0; i < erN; i++) x[b * nEmbd + er0 + i] += tmpA[b * erN + i];
    bar.wait();
  }

  /**
   * Gated DeltaNet layer l (phases P1-P3); the per-token math is ssmLayerStep in model/forward-qwen35.js (the oracle), split
   * so that it can be sharded:
   *  P1  norm; my rows of attn_qkv (q/k of my KEY heads, v of my VALUE heads), attn_gate (z), ssm_beta / ssm_alpha of my value
   *      heads; causal conv + SiLU over those channels, token by token (the conv window lives in shared state); L2 norm of my
   *      key heads' q and k. The conv output goes to the shared `q` buffer: [token][convDim].
   *  P2  (after a barrier) for each of my value heads and each token: delta-rule step against that head's key head (read from
   *      the shared buffer, wherever it was computed), then the gated RMSNorm; the result goes to the shared `att` buffer.
   *  P3  (after a barrier) ssm_out projection + residual, like attention.
   */
  function recurrentMixer(l, L, B) {
    const ri = recIdx[l];
    for (let b = 0; b < B; b++) rmsnorm(xbP, x, L.attnNorm, rmsEps, nEmbd, b * nEmbd, b * nEmbd);
    prep(xbP, B, nEmbd);
    mm(W[`${l}.qkvQ`], sq, convDim, kk0 * dk);
    mm(W[`${l}.qkvK`], sq, convDim, keyDim + kk0 * dk);
    mm(W[`${l}.qkvV`], sq, convDim, 2 * keyDim + vh0 * dv);
    mm(W[`${l}.z`], zP, nOwnV * dv, 0);
    mm(W[`${l}.beta`], betaP, nOwnV, 0);
    mm(W[`${l}.alpha`], alphaP, nOwnV, 0);
    const tr = prof ? performance.now() : 0;
    const convL = convAll.subarray(ri * convSize, (ri + 1) * convSize);
    for (const [c0, c1] of convRanges) {
      if (c1 === c0) continue;
      const st = convL.subarray(c0 * convM, c1 * convM); const cw = L.conv.subarray(c0 * ssm.convKernel, c1 * ssm.convKernel);
      for (let b = 0; b < B; b++) { const xs = sq.subarray(b * convDim + c0, b * convDim + c1); convStep(st, cw, xs, xs, c1 - c0, ssm.convKernel); }
    }
    if (nOwnK > 0) {
      for (let b = 0; b < B; b++) {
        l2normHeads(sq, b * convDim + kk0 * dk, nOwnK, dk, rmsEps, 1 / Math.sqrt(dk));            // q: L2 norm, then 1/sqrt(dk)
        l2normHeads(sq, b * convDim + keyDim + kk0 * dk, nOwnK, dk, rmsEps);                       // k: L2 norm
      }
    }
    bar.wait();

    const td = prof ? performance.now() : 0;
    if (prof) PT.conv += td - tr;
    const ssmL = ssmAll.subarray(ri * ssmSize, (ri + 1) * ssmSize);
    for (let hv = vh0; hv < vh1; hv++) {
      const hl = hv - vh0;
      const kh = keyHeadOfValueHead(hv, ssm.nKHeads, ssm.nVHeads); // THE head mapping (ref-deltanet.js)
      for (let b = 0; b < B; b++) {
        const beta = sigmoid(betaP[b * nOwnV + hl]);
        const g = softplus(alphaP[b * nOwnV + hl] + L.dtBias[hv]) * L.ssmA[hv]; // ssmA = -exp(A_log): g <= 0
        const qo = b * convDim;
        const oo = b * inner + hv * dv;
        deltaRuleHead(ssmL, hv * dv * dk, sq, qo + kh * dk, sq, qo + keyDim + kh * dk, sq, qo + 2 * keyDim + hv * dv, att, oo, dk, dv, Math.exp(g), beta);
        gatedRmsNormHead(att, oo, zP, b * nOwnV * dv + hl * dv, L.ssmNorm, dv, rmsEps);
      }
    }
    if (prof) PT.delta += performance.now() - td;
    bar.wait();

    prep(att, B, inner);
    mm(W[`${l}.wo`], tmpA, erN, 0);
    for (let b = 0; b < B; b++) for (let i = 0; i < erN; i++) x[b * nEmbd + er0 + i] += tmpA[b * erN + i];
    bar.wait();
  }

  // ---------------------------------------------------------------- the SPMD program for B tokens at pos0..pos0+B-1
  function runBatch(B, pos0, wantLogits) {
    for (let b = 0; b < B; b++) {
      const token = Atomics.load(ctrl, CTRL.TOKENS + b);
      if (token >= vr0 && token < vr1) dequantRow(embd, token - vr0, x.subarray(b * nEmbd, (b + 1) * nEmbd)); // the owner of this embedding row
    }
    bar.wait();

    for (let l = 0; l < nLayer; l++) {
      const L = layers[l];
      if (recIdx[l] >= 0) recurrentMixer(l, L, B); else attentionMixer(l, L, B, pos0);

      // P4: FFN norm + my rows of gate/up -> SwiGLU
      for (let b = 0; b < B; b++) rmsnorm(xbP, x, L.ffnNorm, rmsEps, nEmbd, b * nEmbd, b * nEmbd);
      prep(xbP, B, nEmbd);
      mm(W[`${l}.gate`], tmpA, frN, 0);
      mm(W[`${l}.up`], tmpB, frN, 0);
      for (let b = 0; b < B; b++) for (let i = 0; i < frN; i++) hb[b * nFF + fr0 + i] = silu(tmpA[b * frN + i]) * tmpB[b * frN + i];
      bar.wait();

      // P5: down projection, residual add
      prep(hb, B, nFF);
      mm(W[`${l}.down`], tmpA, erN, 0);
      for (let b = 0; b < B; b++) for (let i = 0; i < erN; i++) x[b * nEmbd + er0 + i] += tmpA[b * erN + i];
      bar.wait();
    }

    if (wantLogits === 2) { // every token of the batch: the output weights are read once for all B rows (batched kernel)
      for (let b = 0; b < B; b++) rmsnorm(xbP, x, outputNorm, rmsEps, nEmbd, b * nEmbd, b * nEmbd);
      prep(xbP, B, nEmbd);
      mm(outW, logits, vocabSize, vr0);
    } else if (wantLogits === 1) { // only for the LAST token of the batch
      rmsnorm(xbP, x, outputNorm, rmsEps, nEmbd, (B - 1) * nEmbd, 0);
      prep(xbP, 1, nEmbd);
      mm(outW, logits, 0, vr0);
    }
  }

  // ---------------------------------------------------------------- command loop (blocks this thread)
  let seen = 0;
  for (;;) {
    let spins = spin;
    while (Atomics.load(ctrl, CTRL.CMD_SEQ) === seen) {
      if (spins > 0) spins--; else Atomics.wait(ctrl, CTRL.CMD_SEQ, seen, 1000);
    }
    seen = Atomics.load(ctrl, CTRL.CMD_SEQ);
    const cmd = Atomics.load(ctrl, CTRL.CMD);
    if (cmd === CMD_EXIT) break;
    if (cmd === CMD_KV_EXPORT || cmd === CMD_KV_IMPORT) {
      const layer = Atomics.load(ctrl, CTRL.KV_LAYER); const isV = Atomics.load(ctrl, CTRL.KV_ISV); const cnt = Atomics.load(ctrl, CTRL.KV_N);
      for (let hl = 0; hl < nOwn; hl++) {
        const base = ((layer * 2 + isV) * nOwn + hl) * ctx * headDim;
        for (let t = 0; t < cnt; t++) {
          const so = t * kvDim + (kh0 + hl) * headDim; const mo = base + t * headDim;
          if (kvF16) {
            if (cmd === CMD_KV_EXPORT) for (let i = 0; i < headDim; i++) kvStage[so + i] = h2f[kvU16[mo + i]];
            else for (let i = 0; i < headDim; i++) kvU16[mo + i] = f32ToF16(kvStage[so + i]);
          } else if (cmd === CMD_KV_EXPORT) kvStage.set(kvF32.subarray(mo, mo + headDim), so);
          else kvF32.set(kvStage.subarray(so, so + headDim), mo);
        }
      }
      Atomics.add(ctrl, CTRL.DONE, 1);
      Atomics.notify(ctrl, CTRL.DONE);
      continue;
    }
    const B = Atomics.load(ctrl, CTRL.BATCH);
    const tt = prof ? performance.now() : 0;
    runBatch(B, Atomics.load(ctrl, CTRL.POS), Atomics.load(ctrl, CTRL.WANT_LOGITS));
    if (prof) { PT.total += performance.now() - tt; PT.tokens += B; PT.batches++; }
    Atomics.add(ctrl, CTRL.DONE, 1);
    Atomics.notify(ctrl, CTRL.DONE);
  }
  if (prof && PT.tokens) {
    const per = (v) => (v / PT.tokens).toFixed(2);
    const other = PT.total - PT.kernel - PT.quant - PT.attn - PT.conv - PT.delta - PT.barrier;
    process.stderr.write(`[profile worker 0, ms/token over ${PT.tokens} tokens in ${PT.batches} passes] total ${per(PT.total)} | matmul kernels ${per(PT.kernel)} | act quantization ${per(PT.quant)} | attention+rope ${per(PT.attn)} | conv+l2norm ${per(PT.conv)} | delta rule+gated norm ${per(PT.delta)} | barrier waits ${per(PT.barrier)} | other (norms, residuals, swiglu, copies) ${per(other)}\n`);
  }
}
