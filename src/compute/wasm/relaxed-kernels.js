// Relaxed-SIMD variants of the Q4_0 / Q5_0 / Q4_K / Q5_K / Q6_K kernels (decode AND batched), selected with
// ZEROLLM_DOT=relaxed (see kernels.js). They export the SAME names and signatures as the default kernels
// (matvec_q4_k, mm_q4_k, ...), so a module is built with either set and the worker needs no per-kernel changes.
//
// Dot product: `i32x4.relaxed_dot_i8x16_i7x16_add_s(x, w, acc)` multiplies 16 signed int8 activations x with 16 weights w
// that must lie in 0..127 and adds four adjacent products to each i32 lane of acc: one instruction per 16 weights, no
// widening of either operand (the default kernels widen to int16 and use i32x4.dot_i16x8_s, 8 products per instruction).
//
// Weight ranges. The i7 operand has to be unsigned, so every format is fed its *unsigned* quants and the signed offset is
// folded into the accumulator, exactly, in the integer domain:
//   Q4_K / Q5_K   nibbles 0..15 / 0..31 are already unsigned (the min term uses the block sums xs, as before)
//   Q6_K          q (0..63) with  sum((q-32)*x) = sum(q*x) - 32*sum(x)   -> the dot accumulator starts at -32*sum(x of the group)
//   Q4_0 / Q5_0   q (0..15) / q5 (0..31) with offsets 8 / 16 -> accumulator starts at -8 (-16) * sum(x of the block)
// sum(x) for a group is computed with one more relaxed dot against a vector of ones and is shared by all rows of the tile
// (no new parameters: Q6_K needs 16-element sums, which the activation staging does not provide).
//
// Exactness. Every product is x (-128..127) * w (0..127); the pair sums stay within int16 (127*127*2 = 32258,
// 127*-128*2 = -32512), so even a lowering through pmaddubsw (V8 on x64 without AVX-VNNI) cannot saturate, and the
// i32 lane sums are exact. The i32 lanes group the 32 products differently from the default kernels (4 adjacent
// elements per lane instead of 2+2+2+2 strided ones), so results agree with the default kernels up to float summation
// order only (the integer partial sums are identical); the relaxed batched kernels are bit-identical to the relaxed
// decode kernels, because float operations are performed in the same order in both.
//
// Activations: int8 everywhere. The batched kernels read int8 activations (token t at byte offset t*cols), NOT the
// int16-widened layout of the default batched kernels; worker.js stages them accordingly when this mode is active.
import { T } from './builder.js';
import {
  splat, f16At, hsum, accumulate, scaffold, scaffoldBatch, q50Init, q50Unpack,
  q45kInit, q45kRowSetup, byteOf, q45kNibbles, q6kInit, q6kGroup,
} from './kquant-kernels.js';

const RD = 'i32x4.relaxed_dot_i8x16_i7x16_add_s';

/** stack: [] -> [relaxed_dot(a, b, c)], all three are v128 locals (a = signed int8 vector, b = weights 0..127, c = accumulator) */
const rd = (f, a, b, c) => f.get(a).get(b).get(c).op(RD);

/** stack: [] -> [dot(x1, n1, dot(x0, n0, c))] : the 32 products of one block, accumulated onto c */
function rd2(f, x0, n0, x1, n1, c) { f.get(x1).get(n1); rd(f, x0, n0, c); f.op(RD); }

/** dst = -(sum of the activations per i32 lane << shift) for the vectors `xs` (1 or 2 locals): the accumulator start value */
function negSum(f, st, xs, shift, dst) {
  f.get(st.zv);
  if (xs.length === 2) { f.get(xs[1]).get(st.ones); rd(f, xs[0], st.ones, st.zv); f.op(RD); } else rd(f, xs[0], st.ones, st.zv);
  f.i32(shift).op('i32x4.shl').op('i32x4.sub').set(dst);
}

/** locals shared by every relaxed kernel */
function baseInit(st) {
  const v = () => st.f.local(T.v128);
  Object.assign(st, { zv: v(), ones: v() });
  st.f.v128Const(splat(0)).set(st.zv); st.f.v128Const(splat(1)).set(st.ones);
}

// ------------------------------------------------------------------------------------------------------ Q4_0 / Q5_0
function genQ4_0(m) {
  const init = (st) => {
    const f = st.f; const v = () => f.local(T.v128);
    baseInit(st);
    Object.assign(st, { c0f: v(), raw: v(), n0: v(), n1: v(), x0: v(), x1: v(), neg: v(), sc: f.local(T.f32) });
    f.v128Const(splat(0x0f)).set(st.c0f);
  };
  scaffold(m, 'matvec_q4_0', {
    noXs: true, blockBytes: 18, xStride: 32, dStride: 4,
    init,
    block(st, R) {
      const f = st.f;
      f.get(st.xp).op('v128.load').set(st.x0); f.get(st.xp).op('v128.load', { offset: 16 }).set(st.x1);
      negSum(f, st, [st.x0, st.x1], 3, st.neg); // offset 8
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.get(st.dp).op('f32.load').op('f32.mul').set(st.sc);
        f.get(st.p[j]).op('v128.load', { offset: 2 }).set(st.raw);
        f.get(st.raw).get(st.c0f).op('v128.and').set(st.n0);
        f.get(st.raw).i32(4).op('i8x16.shr_u').set(st.n1);
        rd2(f, st.x0, st.n0, st.x1, st.n1, st.neg);
        accumulate(f, st.acc[j], st.sc);
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });
  scaffoldBatch(m, 'mm_q4_0', {
    noXs: true, blockBytes: 18, xBytes: 32, dBytes: 4,
    init(st) {
      init(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, { wn: [[v(), v()], [v(), v()]], ws: [f.local(T.f32), f.local(T.f32)] });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.set(st.ws[j]);
        f.get(st.p[j]).op('v128.load', { offset: 2 }).set(st.raw);
        f.get(st.raw).get(st.c0f).op('v128.and').set(st.wn[j][0]);
        f.get(st.raw).i32(4).op('i8x16.shr_u').set(st.wn[j][1]);
      }
      for (let q = 0; q < TT; q++) {
        f.get(st.xp[q]).op('v128.load').set(st.x0); f.get(st.xp[q]).op('v128.load', { offset: 16 }).set(st.x1);
        negSum(f, st, [st.x0, st.x1], 3, st.neg);
        for (let j = 0; j < R; j++) {
          f.get(st.ws[j]).get(st.dp[q]).op('f32.load').op('f32.mul').set(st.sc);
          rd2(f, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.neg);
          accumulate(f, st.acc[j][q], st.sc);
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

function genQ5_0(m) {
  const init = (st) => {
    q50Init(st); baseInit(st);
    const v = () => st.f.local(T.v128);
    Object.assign(st, { n1: v(), x0: v(), x1: v(), neg: v() });
  };
  scaffold(m, 'matvec_q5_0', {
    blockBytes: 22, xStride: 32, dStride: 4,
    init,
    block(st, R) {
      const f = st.f;
      f.get(st.xp).op('v128.load').set(st.x0); f.get(st.xp).op('v128.load', { offset: 16 }).set(st.x1);
      negSum(f, st, [st.x0, st.x1], 4, st.neg); // offset 16
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.get(st.dp).op('f32.load').op('f32.mul').set(st.sc);
        q50Unpack(st, st.p[j], st.n, st.n1, true);
        rd2(f, st.x0, st.n, st.x1, st.n1, st.neg);
        accumulate(f, st.acc[j], st.sc);
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });
  scaffoldBatch(m, 'mm_q5_0', {
    blockBytes: 22, xBytes: 32, dBytes: 4,
    init(st) {
      init(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, { wn: [[v(), v()], [v(), v()]], ws: [f.local(T.f32), f.local(T.f32)] });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.set(st.ws[j]);
        q50Unpack(st, st.p[j], st.wn[j][0], st.wn[j][1], true);
      }
      for (let q = 0; q < TT; q++) {
        f.get(st.xp[q]).op('v128.load').set(st.x0); f.get(st.xp[q]).op('v128.load', { offset: 16 }).set(st.x1);
        negSum(f, st, [st.x0, st.x1], 4, st.neg);
        for (let j = 0; j < R; j++) {
          f.get(st.ws[j]).get(st.dp[q]).op('f32.load').op('f32.mul').set(st.sc);
          rd2(f, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.neg);
          accumulate(f, st.acc[j][q], st.sc);
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

// ------------------------------------------------------------------------------------------------------ Q4_K / Q5_K
function genQ45K(m, decodeName, batchName, five) {
  const qsOff = five ? 48 : 16;
  const blockBytes = five ? 176 : 144;
  scaffold(m, decodeName, {
    blockBytes, xStride: 256, dStride: 32,
    init(st) {
      q45kInit(st); baseInit(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { xr0: v(), xr1: v(), raw1: v() });
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) q45kRowSetup(st, j, five);
      for (let s = 0; s < 8; s++) {
        f.get(st.xp).op('v128.load', { offset: 32 * s }).set(st.xr0);
        f.get(st.xp).op('v128.load', { offset: 32 * s + 16 }).set(st.xr1);
        f.get(st.dp).op('f32.load', { offset: 4 * s }).set(st.dxs);
        f.get(st.sp).op('f32.load', { offset: 4 * s }).set(st.bs);
        for (let j = 0; j < R; j++) {
          const p = st.p[j]; const base = qsOff + 32 * (s >> 1);
          q45kNibbles(st, p, base, s, five, st.hq[j][0], st.raw);
          q45kNibbles(st, p, base + 16, s, five, st.hq[j][1], st.raw1);
          byteOf(f, s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
          rd2(f, st.xr0, st.raw, st.xr1, st.raw1, st.zv);
          accumulate(f, st.acc[j], st.coef);
          byteOf(f, s < 4 ? st.mnLo[j] : st.mnHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dmj[j]).op('f32.mul').get(st.dxs).op('f32.mul').get(st.bs).op('f32.mul').set(st.coef);
          f.get(st.mins[j]).get(st.coef).op('f32.add').set(st.mins[j]);
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); st.f.get(st.mins[j]).op('f32.sub'); },
  });

  scaffoldBatch(m, batchName, {
    blockBytes, xBytes: 256, dBytes: 32,
    init(st) {
      q45kInit(st); baseInit(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, {
        raw1: v(), wn: [[v(), v()], [v(), v()]], x0: v(), x1: v(),
        arow: [f.local(T.f32), f.local(T.f32)], mrow: [f.local(T.f32), f.local(T.f32)],
      });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) q45kRowSetup(st, j, five);
      for (let s = 0; s < 8; s++) {
        for (let j = 0; j < R; j++) {
          const p = st.p[j]; const base = qsOff + 32 * (s >> 1);
          q45kNibbles(st, p, base, s, five, st.hq[j][0], st.wn[j][0]);
          q45kNibbles(st, p, base + 16, s, five, st.hq[j][1], st.wn[j][1]);
          byteOf(f, s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').set(st.arow[j]);
          byteOf(f, s < 4 ? st.mnLo[j] : st.mnHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dmj[j]).op('f32.mul').set(st.mrow[j]);
        }
        for (let q = 0; q < TT; q++) {
          f.get(st.xp[q]).op('v128.load', { offset: 32 * s }).set(st.x0);
          f.get(st.xp[q]).op('v128.load', { offset: 32 * s + 16 }).set(st.x1);
          f.get(st.dp[q]).op('f32.load', { offset: 4 * s }).set(st.dxs);
          f.get(st.sp[q]).op('f32.load', { offset: 4 * s }).set(st.bs);
          for (let j = 0; j < R; j++) {
            f.get(st.arow[j]).get(st.dxs).op('f32.mul').set(st.coef);
            rd2(f, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.zv);
            accumulate(f, st.acc[j][q], st.coef);
            f.get(st.mrow[j]).get(st.dxs).op('f32.mul').get(st.bs).op('f32.mul').set(st.coef);
            f.get(st.mins[j][q]).get(st.coef).op('f32.add').set(st.mins[j][q]);
          }
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); st.f.get(st.mins[j][q]).op('f32.sub'); },
  });
}

// ------------------------------------------------------------------------------------------------------------ Q6_K
function genQ6K(m) {
  scaffold(m, 'matvec_q6_k', {
    blockBytes: 210, xStride: 256, dStride: 32,
    init(st) {
      q6kInit(st); baseInit(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { xg: v(), neg: v() });
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) { f16At(f, st.p[j], 208, st.h); f.set(st.dj[j]); }
      for (let half = 0; half < 2; half++) {
        for (let k = 0; k < 4; k++) {
          const xblock = half * 4 + k;
          f.get(st.dp).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
          for (let g = 0; g < 2; g++) {
            f.get(st.xp).op('v128.load', { offset: xblock * 32 + 16 * g }).set(st.xg);
            negSum(f, st, [st.xg], 5, st.neg); // offset 32, per group of 16
            for (let j = 0; j < R; j++) {
              q6kGroup(st, st.p[j], half, k, g, true);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
              rd(f, st.xg, st.q, st.neg);
              accumulate(f, st.acc[j], st.coef);
            }
          }
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });

  scaffoldBatch(m, 'mm_q6_k', {
    blockBytes: 210, xBytes: 256, dBytes: 32,
    init(st) {
      q6kInit(st); baseInit(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { wq: [v(), v()], xg: v(), neg: v(), arow: [st.f.local(T.f32), st.f.local(T.f32)] });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) { f16At(f, st.p[j], 208, st.h); f.set(st.dj[j]); }
      for (let half = 0; half < 2; half++) {
        for (let k = 0; k < 4; k++) {
          const xblock = half * 4 + k;
          for (let g = 0; g < 2; g++) {
            for (let j = 0; j < R; j++) {
              q6kGroup(st, st.p[j], half, k, g, true); f.get(st.q).set(st.wq[j]);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').set(st.arow[j]);
            }
            for (let q = 0; q < TT; q++) {
              f.get(st.xp[q]).op('v128.load', { offset: xblock * 32 + 16 * g }).set(st.xg);
              f.get(st.dp[q]).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
              negSum(f, st, [st.xg], 5, st.neg);
              for (let j = 0; j < R; j++) {
                f.get(st.arow[j]).get(st.dxs).op('f32.mul').set(st.coef);
                rd(f, st.xg, st.wq[j], st.neg);
                accumulate(f, st.acc[j][q], st.coef);
              }
            }
          }
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

/**
 * Adds the relaxed-SIMD Q4_0, Q5_0, Q4_K, Q5_K and Q6_K decode + batched kernels to a module builder (same export names and
 * signatures as the default kernels; the batched ones take INT8 activations). Requires an engine with relaxed SIMD.
 */
export function genRelaxedKernels(m) {
  genQ4_0(m);
  genQ5_0(m);
  genQ45K(m, 'matvec_q4_k', 'mm_q4_k', false);
  genQ45K(m, 'matvec_q5_k', 'mm_q5_k', true);
  genQ6K(m);
}
