import test from 'node:test';
import assert from 'node:assert/strict';
import { instantiateKernels } from '../src/compute/wasm/kernels.js';
import { mulberry32, gaussian } from '../src/util/rng.js';

function setup(hd, npos) {
  const al = (v) => Math.ceil(v / 64) * 64;
  const q = 64; const k = al(q + hd * 4); const v = al(k + npos * hd * 4); const sc = al(v + npos * hd * 4);
  const out = al(sc + Math.max(npos, 1) * 4); const total = out + hd * 4 + 64;
  const mem = new WebAssembly.Memory({ initial: Math.ceil(total / 65536) });
  return { mem, k: instantiateKernels(mem), q, kk: k, v, sc, out };
}

for (const hd of [16, 32, 48, 64, 128]) {
  test(`attention kernels match a plain JS implementation (head_dim ${hd}, 0..70 cached positions)`, () => {
    const rand = mulberry32(hd);
    for (const npos of [0, 1, 2, 3, 5, 17, 70]) {
      const { mem, k, q, kk, v, sc, out } = setup(hd, npos);
      const F = new Float32Array(mem.buffer);
      const qv = Float32Array.from({ length: hd }, () => gaussian(rand));
      const K = Float32Array.from({ length: npos * hd }, () => gaussian(rand));
      const V = Float32Array.from({ length: npos * hd }, () => gaussian(rand));
      F.set(qv, q / 4); F.set(K, kk / 4); F.set(V, v / 4);
      const scale = 1 / Math.sqrt(hd);
      k.attn_scores(q, kk, sc, npos, hd, scale);
      const wantS = Array.from({ length: npos }, (_, t) => { let s = 0; for (let i = 0; i < hd; i++) s += qv[i] * K[t * hd + i]; return s * scale; });
      for (let t = 0; t < npos; t++) assert.ok(Math.abs(F[sc / 4 + t] - wantS[t]) <= 1e-5 * (1 + Math.abs(wantS[t])), `score ${t}: ${F[sc / 4 + t]} vs ${wantS[t]}`);
      // weights: any non-trivial vector (here a softmax of the scores)
      const mx = Math.max(...wantS, -Infinity);
      const e = wantS.map((s) => Math.exp(s - mx)); const z = e.reduce((a, b) => a + b, 0) || 1;
      const p = Float32Array.from(e, (x) => x / z);
      F.set(p, sc / 4);
      F.fill(7, out / 4, out / 4 + hd); // must be fully overwritten, including when npos == 0
      k.attn_av(sc, v, out, npos, hd);
      for (let i = 0; i < hd; i++) {
        let want = 0; for (let t = 0; t < npos; t++) want += p[t] * V[t * hd + i];
        assert.ok(Math.abs(F[out / 4 + i] - want) <= 1e-5 * (1 + Math.abs(want)), `av[${i}] npos=${npos}: ${F[out / 4 + i]} vs ${want}`);
      }
    }
  });
}
