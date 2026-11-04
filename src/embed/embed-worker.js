// One embedding worker: holds a FULL copy of the encoder weights in its own WebAssembly.Memory and embeds whole sequences
// (data parallelism: different workers embed different texts, no barriers). Linear layers run on the batched SIMD kernels
// with all tokens of the sequence as one batch; LayerNorm, RoPE, softmax and pooling are plain JS.
import fs from 'node:fs';
import { parentPort, workerData } from 'node:worker_threads';
import { readTensorBytes } from '../gguf/reader.js';
import { GGML_TYPE, GGML_TYPE_INFO } from '../gguf/types.js';
import { Act, matvec, dequantRow, softmax, silu, ropeTables, applyRope, rowBytes, quantizeAct } from '../compute/ref-kernels.js';
import { instantiateKernels } from '../compute/wasm/kernels.js';
import { alignUp } from '../compute/layout.js';
import { layerNorm } from './ref-encoder.js';

const { cfg, tensors, path, maxSeq, kernels } = workerData;
const PROF = process.env.ZEROLLM_PROFILE === '1';
const PT = { quant: 0, kernel: 0, attn: 0, copy: 0, total: 0 };
const fail = (e) => parentPort.postMessage({ type: 'error', message: e?.stack ?? String(e) });
try { main(); } catch (e) { fail(e); }

function main() {
  const t0 = performance.now();
  const { nEmbd, nHead, headDim, nFF, nLayer, eps } = cfg;
  const useWasm = kernels === 'wasm';
  const attnSimd = useWasm && headDim % 16 === 0;
  const L = maxSeq;
  const plan = []; let total = 0;
  const addMat = (key, name) => { const t = tensors[name]; const bytes = t.dims[1] * rowBytes(t.type, t.dims[0]); plan.push({ key, t, off: total, bytes }); total += alignUp(bytes, 64); };
  for (let l = 0; l < nLayer; l++) { const p = `blk.${l}.`; addMat(`${l}.qkv`, `${p}attn_qkv.weight`); addMat(`${l}.wo`, `${p}attn_output.weight`); addMat(`${l}.gate`, `${p}ffn_gate.weight`); addMat(`${l}.up`, `${p}ffn_up.weight`); addMat(`${l}.down`, `${p}ffn_down.weight`); }
  const maxCols = Math.max(nEmbd, nFF); const maxRows = Math.max(3 * nEmbd, nFF);
  const XQ16 = total; total += alignUp(L * maxCols * 2, 64);
  const XDB = total; total += alignUp(L * (maxCols / 32) * 4, 64);
  const XSB = total; total += alignUp(L * (maxCols / 32) * 4, 64);
  const YO = total; total += alignUp(L * maxRows * 4, 64);
  const KB = total; total += alignUp(nHead * L * headDim * 4, 64);
  const VB = total; total += alignUp(nHead * L * headDim * 4, 64);
  const SCP = total; total += alignUp(L * 4, 64);
  const QHP = total; total += alignUp(headDim * 4, 64);
  const OHP = total; total += alignUp(headDim * 4, 64);
  const mem = new WebAssembly.Memory({ initial: Math.max(1, Math.ceil(total / 65536)) });
  const fd = fs.openSync(path, 'r');
  const W = {}; let tokEmbd; let tokType0 = null; let embNorm; let layers;
  try {
    for (const p of plan) {
      const view = new Uint8Array(mem.buffer, p.off, p.bytes);
      readTensorBytes(fd, p.t, 0, view);
      W[p.key] = { type: p.t.type, rows: p.t.dims[1], cols: p.t.dims[0], data: view, off: p.off, nb: p.t.dims[0] / (GGML_TYPE_INFO[p.t.type]?.blockSize ?? 32), kernel: null, mmKernel: null, useXs: false };
    }
    const te = tensors['token_embd.weight'];
    const embBytes = new Uint8Array(te.nBytes); readTensorBytes(fd, te, 0, embBytes);
    tokEmbd = { type: te.type, rows: te.dims[1], cols: te.dims[0], data: embBytes };
    const vec = (name) => { const t = tensors[name]; const f = new Float32Array(t.nElements); if (t.type === GGML_TYPE.F32) readTensorBytes(fd, t, 0, new Uint8Array(f.buffer)); else { const b = new Uint8Array(t.nBytes); readTensorBytes(fd, t, 0, b); dequantRow({ type: t.type, cols: t.nElements, data: b }, 0, f); } return f; };
    if (cfg.hasTokenTypes) { const tt = tensors['token_types.weight']; const b = new Uint8Array(tt.dims[1] * rowBytes(tt.type, tt.dims[0])); readTensorBytes(fd, tt, 0, b); tokType0 = dequantRow({ type: tt.type, cols: tt.dims[0], data: b }, 0, new Float32Array(nEmbd)); }
    embNorm = { w: vec('token_embd_norm.weight'), b: vec('token_embd_norm.bias') };
    layers = [];
    for (let l = 0; l < nLayer; l++) { const p = `blk.${l}.`; layers.push({ anW: vec(`${p}attn_output_norm.weight`), anB: vec(`${p}attn_output_norm.bias`), lnW: vec(`${p}layer_output_norm.weight`), lnB: vec(`${p}layer_output_norm.bias`) }); }
  } finally { fs.closeSync(fd); }

  let k = null;
  if (useWasm) {
    k = instantiateKernels(mem);
    for (const w of Object.values(W)) {
      const set = (kernel, mmKernel, useXs) => { w.kernel = kernel; w.mmKernel = mmKernel; w.useXs = useXs; };
      if (w.type === GGML_TYPE.Q8_0) set(k.matvec_q8_0, k.mm_q8_0, false);
      else if (w.type === GGML_TYPE.Q4_0) set(k.matvec_q4_0, k.mm_q4_0, false);
      else if (w.type === GGML_TYPE.Q5_0) set(k.matvec_q5_0, k.mm_q5_0, true);
      else if (w.type === GGML_TYPE.Q4_K) set(k.matvec_q4_k, k.mm_q4_k, true);
      else if (w.type === GGML_TYPE.Q5_K) set(k.matvec_q5_k, k.mm_q5_k, true);
      else if (w.type === GGML_TYPE.Q6_K) set(k.matvec_q6_k, k.mm_q6_k, true);
    }
  }
  const xq16 = new Int16Array(mem.buffer, XQ16, L * maxCols); const xdB = new Float32Array(mem.buffer, XDB, (L * maxCols) / 32); const xsB = new Float32Array(mem.buffer, XSB, (L * maxCols) / 32);
  const yv = new Float32Array(mem.buffer, YO, L * maxRows);
  const Kv = new Float32Array(mem.buffer, KB, nHead * L * headDim); const Vv = new Float32Array(mem.buffer, VB, nHead * L * headDim);
  const scores = new Float32Array(mem.buffer, SCP, L); const qh = new Float32Array(mem.buffer, QHP, headDim); const oh = new Float32Array(mem.buffer, OHP, headDim);
  const ropeFull = ropeTables({ nRot: headDim, ropeBase: cfg.ropeBase }, L);
  const scale = 1 / Math.sqrt(headDim);

  // activations, token-major [t][dim]
  const x = new Float32Array(L * nEmbd); const A = new Float32Array(L * nEmbd); const hb = new Float32Array(L * nFF); const gateBuf = new Float32Array(L * nFF);
  const Q = new Float32Array(L * nEmbd); const tmp = new Float32Array(nEmbd); const tmpF = new Float32Array(Math.max(3 * nEmbd, nFF));

  /** y[t][r] = W.row(r) . src[t] for n tokens, into `dest` (token-major, `w.rows` per token) */
  function linear(w, src, n, dim, dest) {
    if (w.kernel) {
      const nbl = dim / 32; const t0 = PROF ? performance.now() : 0;
      for (let t = 0; t < n; t++) quantizeAct(src.subarray(t * dim, (t + 1) * dim), xq16.subarray(t * dim, (t + 1) * dim), xdB.subarray(t * nbl, (t + 1) * nbl), xsB.subarray(t * nbl, (t + 1) * nbl));
      const t1 = PROF ? performance.now() : 0;
      if (w.useXs) w.mmKernel(w.off, XQ16, XDB, XSB, YO, w.rows, w.nb, n); else w.mmKernel(w.off, XQ16, XDB, YO, w.rows, w.nb, n);
      const t2 = PROF ? performance.now() : 0;
      dest.set(yv.subarray(0, n * w.rows));
      if (PROF) { PT.quant += t1 - t0; PT.kernel += t2 - t1; PT.copy += performance.now() - t2; }
    } else {
      for (let t = 0; t < n; t++) { matvec(w, new Act(src.subarray(t * dim, (t + 1) * dim)), tmpF); dest.set(tmpF.subarray(0, w.rows), t * w.rows); }
    }
  }
  const qkvBuf = new Float32Array(L * 3 * nEmbd); const projBuf = new Float32Array(L * nEmbd);

  function embed(ids) {
    const n = ids.length;
    if (n < 1 || n > L) throw new Error(`sequence of ${n} tokens (max ${L})`);
    for (let t = 0; t < n; t++) {
      if (!(ids[t] >= 0 && ids[t] < tokEmbd.rows)) throw new Error(`token id ${ids[t]} out of range`);
      dequantRow(tokEmbd, ids[t], tmp);
      if (tokType0) for (let i = 0; i < nEmbd; i++) tmp[i] += tokType0[i];
      layerNorm(x, tmp, embNorm.w, embNorm.b, eps, nEmbd, 0, t * nEmbd);
    }
    for (let l = 0; l < nLayer; l++) {
      const Ly = layers[l];
      linear(W[`${l}.qkv`], x, n, nEmbd, qkvBuf);
      for (let t = 0; t < n; t++) {
        const o = t * 3 * nEmbd;
        const q = Q.subarray(t * nEmbd, (t + 1) * nEmbd); q.set(qkvBuf.subarray(o, o + nEmbd));
        const kk = tmp; kk.set(qkvBuf.subarray(o + nEmbd, o + 2 * nEmbd));
        applyRope(q, nHead, headDim, headDim, 'neox', ropeFull, t); applyRope(kk, nHead, headDim, headDim, 'neox', ropeFull, t);
        for (let h = 0; h < nHead; h++) {
          Kv.set(kk.subarray(h * headDim, (h + 1) * headDim), (h * L + t) * headDim);
          Vv.set(qkvBuf.subarray(o + 2 * nEmbd + h * headDim, o + 2 * nEmbd + (h + 1) * headDim), (h * L + t) * headDim);
        }
      }
      const ta = PROF ? performance.now() : 0;
      for (let t = 0; t < n; t++) {
        for (let h = 0; h < nHead; h++) {
          qh.set(Q.subarray(t * nEmbd + h * headDim, t * nEmbd + (h + 1) * headDim));
          if (attnSimd) { k.attn_scores(QHP, KB + h * L * headDim * 4, SCP, n, headDim, scale); softmax(scores, n); k.attn_av(SCP, VB + h * L * headDim * 4, OHP, n, headDim); }
          else {
            for (let s = 0; s < n; s++) { let d = 0; const ko = (h * L + s) * headDim; for (let i = 0; i < headDim; i++) d += qh[i] * Kv[ko + i]; scores[s] = d * scale; }
            softmax(scores, n); oh.fill(0);
            for (let s = 0; s < n; s++) { const p = scores[s]; const vo = (h * L + s) * headDim; for (let i = 0; i < headDim; i++) oh[i] += p * Vv[vo + i]; }
          }
          A.set(oh, t * nEmbd + h * headDim);
        }
      }
      if (PROF) PT.attn += performance.now() - ta;
      linear(W[`${l}.wo`], A, n, nEmbd, projBuf);
      for (let t = 0; t < n; t++) { for (let i = 0; i < nEmbd; i++) tmp[i] = x[t * nEmbd + i] + projBuf[t * nEmbd + i]; layerNorm(x, tmp, Ly.anW, Ly.anB, eps, nEmbd, 0, t * nEmbd); }
      linear(W[`${l}.gate`], x, n, nEmbd, gateBuf);
      linear(W[`${l}.up`], x, n, nEmbd, hb);
      for (let i = 0; i < n * nFF; i++) hb[i] = silu(gateBuf[i]) * hb[i];
      linear(W[`${l}.down`], hb, n, nFF, projBuf);
      for (let t = 0; t < n; t++) { for (let i = 0; i < nEmbd; i++) tmp[i] = x[t * nEmbd + i] + projBuf[t * nEmbd + i]; layerNorm(x, tmp, Ly.lnW, Ly.lnB, eps, nEmbd, 0, t * nEmbd); }
    }
    const out = new Float32Array(nEmbd);
    if (cfg.pooling === 2) out.set(x.subarray(0, nEmbd));
    else { for (let t = 0; t < n; t++) for (let i = 0; i < nEmbd; i++) out[i] += x[t * nEmbd + i]; for (let i = 0; i < nEmbd; i++) out[i] /= n; }
    return out;
  }

  parentPort.on('message', (m) => {
    if (m.type === 'exit') { process.exit(0); }
    if (m.type !== 'job') return;
    try { const tt = PROF ? performance.now() : 0; const vec = embed(m.ids); if (PROF) { PT.total += performance.now() - tt; process.stderr.write(`[embed profile] total ${PT.total.toFixed(0)} ms | act quant ${PT.quant.toFixed(0)} | matmul kernels ${PT.kernel.toFixed(0)} | result copy ${PT.copy.toFixed(0)} | attention ${PT.attn.toFixed(0)} | other ${(PT.total - PT.quant - PT.kernel - PT.copy - PT.attn).toFixed(0)}\n`); } parentPort.postMessage({ type: 'result', id: m.id, vec }, [vec.buffer]); }
    catch (e) { parentPort.postMessage({ type: 'jobError', id: m.id, message: e?.stack ?? String(e) }); }
  });
  parentPort.postMessage({ type: 'ready', ms: performance.now() - t0, bytes: total, kernels: useWasm ? 'wasm-simd' : 'js' });
}
