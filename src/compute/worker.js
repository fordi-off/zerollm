// SPMD worker. Every worker runs the same control flow for a token; phases are separated by barriers.
// Each worker owns a contiguous row slice of every weight matrix, stored in its own private WebAssembly.Memory
// (so the 4 GB wasm32 limit applies per worker, not to the whole model).
import fs from 'node:fs';
import { parentPort, workerData } from 'node:worker_threads';
import { readTensorBytes } from '../gguf/reader.js';
import { Act, matvec, dequantRow, rmsnorm, softmax, silu, ropeTables, applyRope, rowBytes } from './ref-kernels.js';
import { Barrier } from './barrier.js';
import { CTRL, CMD_EXIT, splitRows, splitRange, alignUp, kvOffset } from './layout.js';

const { id, n, cfg, tensors, path, ctx, shared, spin } = workerData;

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

  // ---------------------------------------------------------------- weight slices -> private wasm memory
  /** @type {Array<{key: string, t: any, r0: number, r1: number, off: number, bytes: number}>} */
  const plan = [];
  let total = 0;
  const addMat = (key, name) => {
    const t = tensors[name];
    if (!t) throw new Error(`worker ${id}: tensor ${name} missing from plan`);
    const rows = t.dims[1]; const cols = t.dims[0];
    const [r0, r1] = splitRows(rows, n, id);
    const bytes = (r1 - r0) * rowBytes(t.type, cols);
    plan.push({ key, t, r0, r1, off: total, bytes });
    total += alignUp(bytes, 64);
  };
  addMat('embd', 'token_embd.weight');
  if (!cfg.tiedEmbeddings) addMat('output', 'output.weight');
  for (let l = 0; l < nLayer; l++) {
    const p = `blk.${l}.`;
    addMat(`${l}.wq`, `${p}attn_q.weight`); addMat(`${l}.wk`, `${p}attn_k.weight`); addMat(`${l}.wv`, `${p}attn_v.weight`);
    addMat(`${l}.wo`, `${p}attn_output.weight`);
    addMat(`${l}.gate`, `${p}ffn_gate.weight`); addMat(`${l}.up`, `${p}ffn_up.weight`); addMat(`${l}.down`, `${p}ffn_down.weight`);
  }
  const pages = Math.max(1, Math.ceil(total / 65536));
  if (pages > 65536) throw new Error(`worker ${id}: weight slice is ${(total / 2 ** 30).toFixed(2)} GB, over the 4 GB wasm32 limit; use more workers`);
  let mem;
  try {
    mem = new WebAssembly.Memory({ initial: pages });
  } catch (e) {
    throw new Error(`worker ${id}: could not allocate ${(total / 2 ** 20).toFixed(0)} MB of WebAssembly memory (${e.message}). Close other apps or use a smaller model/quant.`);
  }
  const fd = fs.openSync(path, 'r');
  /** @type {Record<string, import('./ref-kernels.js').Weight>} */
  const W = {};
  let loaded = 0; let lastPost = 0;
  try {
    for (const p of plan) {
      const view = new Uint8Array(mem.buffer, p.off, p.bytes);
      if (p.bytes) readTensorBytes(fd, p.t, p.r0 * rowBytes(p.t.type, p.t.dims[0]), view);
      W[p.key] = { type: p.t.type, rows: p.r1 - p.r0, cols: p.t.dims[0], data: view };
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
    var outputNorm = vec('output_norm.weight');
    var ropeFreqs = cfg.hasRopeFreqs ? vec('rope_freqs.weight') : null;
    var layers = [];
    for (let l = 0; l < nLayer; l++) {
      const p = `blk.${l}.`;
      const opt = (nm) => (tensors[p + nm] ? vec(p + nm) : null);
      layers.push({
        attnNorm: vec(`${p}attn_norm.weight`), ffnNorm: vec(`${p}ffn_norm.weight`),
        bq: opt('attn_q.bias'), bk: opt('attn_k.bias'), bv: opt('attn_v.bias'),
        qNorm: opt('attn_q_norm.weight'), kNorm: opt('attn_k_norm.weight'),
      });
    }
  } finally { fs.closeSync(fd); }

  // ---------------------------------------------------------------- shared views + private scratch
  const ctrl = new Int32Array(shared.ctrl);
  const x = new Float32Array(shared.x); const sq = new Float32Array(shared.q);
  const sk = new Float32Array(shared.k); const sv = new Float32Array(shared.v);
  const att = new Float32Array(shared.att); const hb = new Float32Array(shared.hb);
  const logits = new Float32Array(shared.logits); const kv = new Float32Array(shared.kv);
  const bar = new Barrier(ctrl, CTRL.BAR_COUNT, CTRL.BAR_GEN, n, spin);
  const rope = ropeTables(cfg, ctx, ropeFreqs);
  const xb = new Float32Array(nEmbd);
  const scores = new Float32Array(ctx);
  const qh = new Float32Array(headDim);
  const oh = new Float32Array(headDim);
  const tmpA = new Float32Array(Math.max(nEmbd, nFF, qDim, kvDim));
  const tmpB = new Float32Array(Math.max(nEmbd, nFF, qDim, kvDim));

  const [qr0, qr1] = splitRows(qDim, n, id);
  const [kr0, kr1] = splitRows(kvDim, n, id);
  const [er0, er1] = splitRows(nEmbd, n, id);
  const [fr0, fr1] = splitRows(nFF, n, id);
  const [kh0, kh1] = splitRange(nHeadKv, n, id);
  const [qh0, qh1] = splitRange(nHead, n, id);
  const [vr0, vr1] = splitRows(vocabSize, n, id);
  const group = nHead / nHeadKv;
  const scale = 1 / Math.sqrt(headDim);
  const embd = W.embd; const outW = cfg.tiedEmbeddings ? W.embd : W.output;

  parentPort.postMessage({ type: 'ready', id, ms: performance.now() - t0, bytes: total, sliceBytes: loaded });

  // ---------------------------------------------------------------- the per-token SPMD program
  function runToken(token, pos, wantLogits) {
    if (token >= vr0 && token < vr1) dequantRow(embd, token - vr0, x); // the worker that owns this embedding row
    bar.wait();

    for (let l = 0; l < nLayer; l++) {
      const L = layers[l];
      // P1: norm (redundant per worker) + my rows of q, k, v
      rmsnorm(xb, x, L.attnNorm, rmsEps);
      const h = new Act(xb);
      matvec(W[`${l}.wq`], h, sq.subarray(qr0, qr1));
      matvec(W[`${l}.wk`], h, sk.subarray(kr0, kr1));
      matvec(W[`${l}.wv`], h, sv.subarray(kr0, kr1));
      if (L.bq) for (let i = qr0; i < qr1; i++) sq[i] += L.bq[i];
      if (L.bk) for (let i = kr0; i < kr1; i++) sk[i] += L.bk[i];
      if (L.bv) for (let i = kr0; i < kr1; i++) sv[i] += L.bv[i];
      bar.wait();

      // P2a: my KV heads: k-norm, RoPE, write K and V into the shared cache
      for (let hd = kh0; hd < kh1; hd++) {
        if (L.kNorm) rmsnorm(sk, sk, L.kNorm, rmsEps, headDim, hd * headDim, hd * headDim);
        applyRope(sk.subarray(hd * headDim, (hd + 1) * headDim), 1, headDim, nRot, ropeType, rope, pos);
        kv.set(sk.subarray(hd * headDim, (hd + 1) * headDim), kvOffset(l, 0, ctx, kvDim, pos) + hd * headDim);
        kv.set(sv.subarray(hd * headDim, (hd + 1) * headDim), kvOffset(l, 1, ctx, kvDim, pos) + hd * headDim);
      }
      bar.wait();

      // P2b: my query heads: q-norm, RoPE, attention over the cache
      const kBase = kvOffset(l, 0, ctx, kvDim); const vBase = kvOffset(l, 1, ctx, kvDim);
      for (let hd = qh0; hd < qh1; hd++) {
        qh.set(sq.subarray(hd * headDim, (hd + 1) * headDim));
        if (L.qNorm) rmsnorm(qh, qh, L.qNorm, rmsEps, headDim);
        applyRope(qh, 1, headDim, nRot, ropeType, rope, pos);
        const kvh = Math.floor(hd / group);
        for (let t = 0; t <= pos; t++) {
          const ko = kBase + t * kvDim + kvh * headDim;
          let s = 0;
          for (let i = 0; i < headDim; i++) s += qh[i] * kv[ko + i];
          scores[t] = s * scale;
        }
        softmax(scores, pos + 1);
        oh.fill(0);
        for (let t = 0; t <= pos; t++) {
          const p = scores[t];
          const vo = vBase + t * kvDim + kvh * headDim;
          for (let i = 0; i < headDim; i++) oh[i] += p * kv[vo + i];
        }
        att.set(oh, hd * headDim);
      }
      bar.wait();

      // P3: output projection, residual add into my rows of x
      matvec(W[`${l}.wo`], new Act(att), tmpA);
      for (let i = 0; i < er1 - er0; i++) x[er0 + i] += tmpA[i];
      bar.wait();

      // P4: FFN norm + my rows of gate/up -> SwiGLU
      rmsnorm(xb, x, L.ffnNorm, rmsEps);
      const h2 = new Act(xb);
      matvec(W[`${l}.gate`], h2, tmpA);
      matvec(W[`${l}.up`], h2, tmpB);
      for (let i = 0; i < fr1 - fr0; i++) hb[fr0 + i] = silu(tmpA[i]) * tmpB[i];
      bar.wait();

      // P5: down projection, residual add
      matvec(W[`${l}.down`], new Act(hb), tmpA);
      for (let i = 0; i < er1 - er0; i++) x[er0 + i] += tmpA[i];
      bar.wait();
    }

    if (wantLogits) {
      rmsnorm(xb, x, outputNorm, rmsEps);
      matvec(outW, new Act(xb), logits.subarray(vr0, vr1));
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
    if (Atomics.load(ctrl, CTRL.CMD) === CMD_EXIT) break;
    runToken(Atomics.load(ctrl, CTRL.TOKEN), Atomics.load(ctrl, CTRL.POS), Atomics.load(ctrl, CTRL.WANT_LOGITS) === 1);
    Atomics.add(ctrl, CTRL.DONE, 1);
    Atomics.notify(ctrl, CTRL.DONE);
  }
}
