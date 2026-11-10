// Plain-JS reference kernels for the Gated DeltaNet ("linear attention") layers of Qwen3.5 (arch "qwen35").
// Slow but obviously correct, one token at a time. Formulas follow llama.cpp (src/models/qwen35.cpp,
// src/models/delta-net-base.cpp: build_delta_net_autoregressive, ggml ssm_conv) and the Hugging Face
// Qwen3-Next reference (torch_recurrent_gated_delta_rule, causal conv1d + SiLU, RMSNormGated). See docs/DECISIONS.md.
//
// Layouts (GGUF space, i.e. AFTER llama.cpp's converter reordered the value heads into "tiled" order):
//   conv state     [channel][kernel-1]      oldest input first
//   delta state    [vHead][vDim j][kDim i]  S_h[j][i]; the matrix S_h^T of the papers, so every inner loop is contiguous
//   key head used by value head h:  h % nKHeads  (NOT h / (nVHeads / nKHeads))  -> keyHeadOfValueHead() below

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** log(1 + e^x), linear above 20 like ggml_softplus. */
export const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

const silu = (x) => x / (1 + Math.exp(-x));

/**
 * THE value-head -> key-head mapping of the Gated DeltaNet layers, in one place (the reference path AND the worker pool call
 * this; nothing else encodes it). The Qwen3.5 checkpoint has `nKHeads` (16) key/query heads shared by `nVHeads` (32) value
 * heads. llama.cpp's converter re-orders the value heads of the GGUF into "tiled" order, so that value head h reads key head
 * `h % nKHeads` (ggml_repeat tiles the 16 key heads over the 32 value heads); the Hugging Face checkpoint order would be the
 * grouped `floor(h / (nVHeads / nKHeads))`.
 * NOT VERIFIED ON THE REAL Qwen3.5-4B FILE (only against the synthetic generator that applies the converter's reordering,
 * see docs/DECISIONS.md and docs/REAL-MODEL-CHECK.md). If a real file turns out to be in grouped order, change THIS function.
 * @param {number} hv value head  @param {number} nKHeads @param {number} nVHeads
 */
export function keyHeadOfValueHead(hv, nKHeads, nVHeads) {
  return hv % nKHeads;
}

/**
 * One step of the depthwise causal conv1d followed by SiLU, updating the conv window in place.
 *   y[c] = silu( sum_{t < kernel} w[c][t] * window_t[c] )  with window = [state oldest .. state newest, x]
 * so w[c][kernel-1] multiplies the current input (ggml ssm_conv: dot of the sliding window with the weight row).
 * @param {Float32Array} state  channels * (kernel - 1) values (this layer's slice), updated
 * @param {Float32Array} w      channels * kernel taps, [channel][tap]
 * @param {Float32Array} x      current input, `channels` values
 * @param {Float32Array} out    `channels` values (may alias x)
 */
export function convStep(state, w, x, out, channels, kernel) {
  const m = kernel - 1;
  for (let c = 0; c < channels; c++) {
    const so = c * m; const wo = c * kernel;
    const cur = x[c];
    let acc = 0;
    for (let t = 0; t < m; t++) acc += w[wo + t] * state[so + t];
    acc += w[wo + m] * cur;
    for (let t = 0; t + 1 < m; t++) state[so + t] = state[so + t + 1];
    if (m > 0) state[so + m - 1] = cur;
    out[c] = silu(acc);
  }
}

/**
 * v[head] *= scale / sqrt(sum(v[head]^2) + eps) for `nHeads` consecutive heads of size `dim` starting at `off`.
 * (llama.cpp build_gdn_l2_norm = rms_norm(x, eps/n) / sqrt(n) = x / sqrt(sum x^2 + eps); HF: x * rsqrt(sum x^2 + eps).)
 */
export function l2normHeads(v, off, nHeads, dim, eps, scale = 1) {
  for (let h = 0; h < nHeads; h++) {
    const o = off + h * dim;
    let ss = 0;
    for (let i = 0; i < dim; i++) ss += v[o + i] * v[o + i];
    const s = scale / Math.sqrt(ss + eps);
    for (let i = 0; i < dim; i++) v[o + i] *= s;
  }
}

/**
 * One autoregressive delta-rule step of ONE value head (llama.cpp build_delta_net_autoregressive):
 *   S <- exp(g) S                       decay
 *   d  = beta * (v - S^T k)             delta rule: what the memory fails to predict for k
 *   S <- S + k d^T                      write
 *   o  = S^T q                          read (q already scaled by 1/sqrt(headKDim))
 * @param {Float32Array} S   state array holding this head's [dv][dk] matrix at sOff (updated)
 * @param {number} sOff
 * @param {Float32Array} q @param {number} qOff  query head (dk values, L2-normalised and scaled)
 * @param {Float32Array} k @param {number} kOff  key head (dk values, L2-normalised)
 * @param {Float32Array} v @param {number} vOff  value head (dv values)
 * @param {Float32Array} out @param {number} oOff  output head (dv values)
 * @param {number} dk @param {number} dv
 * @param {number} gExp exp(g), the decay factor in (0, 1]
 * @param {number} beta write strength in (0, 1)
 */
export function deltaRuleHead(S, sOff, q, qOff, k, kOff, v, vOff, out, oOff, dk, dv, gExp, beta) {
  for (let j = 0; j < dv; j++) {
    const r = sOff + j * dk;
    let sk = 0;
    for (let i = 0; i < dk; i++) { const s = S[r + i] * gExp; S[r + i] = s; sk += s * k[kOff + i]; }
    const d = (v[vOff + j] - sk) * beta;
    let o = 0;
    for (let i = 0; i < dk; i++) { const s = S[r + i] + k[kOff + i] * d; S[r + i] = s; o += s * q[qOff + i]; }
    out[oOff + j] = o;
  }
}

/**
 * Gated RMSNorm of one head, in place on `o`: o = o / sqrt(mean(o^2) + eps) * weight * silu(z)
 * (HF RMSNormGated, llama.cpp build_norm_gated). `weight` is shared by all heads and NOT offset by +1.
 */
export function gatedRmsNormHead(o, oOff, z, zOff, weight, dim, eps) {
  let ss = 0;
  for (let i = 0; i < dim; i++) ss += o[oOff + i] * o[oOff + i];
  const s = 1 / Math.sqrt(ss / dim + eps);
  for (let i = 0; i < dim; i++) o[oOff + i] = o[oOff + i] * s * weight[i] * silu(z[zOff + i]);
}
