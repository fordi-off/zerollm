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
// Unpacking. V8 lowers an i8x16 shift by a constant to a 16-bit shift plus a byte mask that it REBUILDS at every use
// (mov imm32, movd, pshufd, pand: 5 instructions). Here byte shifts are written as i16x8 shifts followed by an explicit
// `v128.and` with a mask kept in a local (2 instructions), and Q6_K's "(qh >> 2k & 3) << 4" is a single shift plus one mask.
//
// Activations: int8 everywhere. The batched kernels read int8 activations (token t at byte offset t*cols), NOT the
// int16-widened layout of the default batched kernels; worker.js stages them accordingly when this mode is active.
import { T } from './builder.js';
import {
  splat, f16At, hsum, accumulate, scaffold, scaffoldBatch, q50Init, q45kInit, q45kRowSetup, byteOf,
} from './kquant-kernels.js';

const RD = 'i32x4.relaxed_dot_i8x16_i7x16_add_s'; // 0x113
const MA = 'i16x8.relaxed_dot_i8x16_i7x16_s';       // 0x112

/**
 * The integer dot forms of the two flavors. In both, `init` is a v128 local (the accumulator start value, st.zv for zero).
 *  'i16' (the `relaxed` mode): i16x8.relaxed_dot (one pmaddubsw on x64) for each 16 weights, the two int16 results of a
 *        32-block are added (weights <= 31: |pair| <= 7936, so the sum of two <= 15872 < 32767) and widened once with
 *        i32x4.extadd_pairwise_i16x8_s (one pmaddwd): 4 x64 instructions per 32 weights.
 *  'add' (the `relaxed-add` mode): i32x4.relaxed_dot_..._add_s per 16 weights. A single vpdpbusd with AVX-VNNI, but V8 on
 *        x64 without it emits pcmpeqd + psrlw (to rebuild the 16-bit ones) + pmaddubsw + pmaddwd + paddd for EACH use:
 *        12 instructions per 32 weights.
 */
function dotOps(flavor) {
  const add = flavor === 'add';
  return {
    /** stack: [] -> [init + the 32 products x0*w0 + x1*w1, summed per i32 lane] */
    d32(f, st, x0, w0, x1, w1, init) {
      if (add) { f.get(x1).get(w1).get(x0).get(w0).get(init).op(RD).op(RD); return; }
      f.get(x0).get(w0).op(MA).get(x1).get(w1).op(MA).op('i16x8.add').op('i32x4.extadd_pairwise_i16x8_s');
      if (init !== st.zv) f.get(init).op('i32x4.add');
    },
    /** stack: [] -> [init + the 16 products x*w, summed per i32 lane] */
    d16(f, st, x, w, init) {
      if (add) { f.get(x).get(w).get(init).op(RD); return; }
      f.get(x).get(w).op(MA).op('i32x4.extadd_pairwise_i16x8_s');
      if (init !== st.zv) f.get(init).op('i32x4.add');
    },
    /** dst = -(sum of the activations per i32 lane << shift) for the vectors `xs` (1 or 2 locals): the accumulator start value */
    neg(f, st, xs, shift, dst) {
      f.get(st.zv);
      if (add) {
        if (xs.length === 2) f.get(xs[1]).get(st.ones).get(xs[0]).get(st.ones).get(st.zv).op(RD).op(RD);
        else f.get(xs[0]).get(st.ones).get(st.zv).op(RD);
      } else {
        // pairwise sums of int8 (|.| <= 256, two vectors <= 512) fit int16; then four-element sums per i32 lane
        f.get(xs[0]).op('i16x8.extadd_pairwise_i8x16_s');
        if (xs.length === 2) f.get(xs[1]).op('i16x8.extadd_pairwise_i8x16_s').op('i16x8.add');
        f.op('i32x4.extadd_pairwise_i16x8_s');
      }
      f.i32(shift).op('i32x4.shl').op('i32x4.sub').set(dst);
    },
  };
}

/** locals shared by every relaxed kernel */
function baseInit(st) {
  const v = () => st.f.local(T.v128);
  Object.assign(st, { zv: v(), ones: v(), c0f: v() });
  st.f.v128Const(splat(0)).set(st.zv); st.f.v128Const(splat(1)).set(st.ones); st.f.v128Const(splat(0x0f)).set(st.c0f);
}

/** [v128 on the stack] -> its low nibbles (and) / high nibbles (16-bit shift, then mask), per byte; uses st.c0f */
const lowNibbles = (f, st) => f.get(st.c0f).op('v128.and');
const highNibbles = (f, st) => f.i32(4).op('i16x8.shr_u').get(st.c0f).op('v128.and');

// ------------------------------------------------------------------------------------------------------ Q4_0 / Q5_0
function genQ4_0(m, D) {
  const init = (st) => {
    const f = st.f; const v = () => f.local(T.v128);
    baseInit(st);
    Object.assign(st, { raw: v(), n0: v(), n1: v(), x0: v(), x1: v(), neg: v(), sc: f.local(T.f32) });
  };
  scaffold(m, 'matvec_q4_0', {
    noXs: true, blockBytes: 18, xStride: 32, dStride: 4,
    init,
    block(st, R) {
      const f = st.f;
      f.get(st.xp).op('v128.load').set(st.x0); f.get(st.xp).op('v128.load', { offset: 16 }).set(st.x1);
      D.neg(f, st, [st.x0, st.x1], 3, st.neg); // offset 8
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.get(st.dp).op('f32.load').op('f32.mul').set(st.sc);
        f.get(st.p[j]).op('v128.load', { offset: 2 }).set(st.raw);
        f.get(st.raw); lowNibbles(f, st); f.set(st.n0);
        f.get(st.raw); highNibbles(f, st); f.set(st.n1);
        D.d32(f, st, st.x0, st.n0, st.x1, st.n1, st.neg);
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
        f.get(st.raw); lowNibbles(f, st); f.set(st.wn[j][0]);
        f.get(st.raw); highNibbles(f, st); f.set(st.wn[j][1]);
      }
      for (let q = 0; q < TT; q++) {
        f.get(st.xp[q]).op('v128.load').set(st.x0); f.get(st.xp[q]).op('v128.load', { offset: 16 }).set(st.x1);
        D.neg(f, st, [st.x0, st.x1], 3, st.neg);
        for (let j = 0; j < R; j++) {
          f.get(st.ws[j]).get(st.dp[q]).op('f32.load').op('f32.mul').set(st.sc);
          D.d32(f, st, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.neg);
          accumulate(f, st.acc[j][q], st.sc);
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

/** n0 / n1 = the 32 UNSIGNED 5-bit quants (0..31) of the Q5_0 block at ptr p (same layout logic as the default kernels' q50Unpack) */
function q50Unsigned(st, p, n0, n1) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: 6 }).set(st.raw);
  f.get(p).op('i32.load', { offset: 2 }).op('i32x4.splat').set(st.qv);
  // 0x10 in every byte lane whose qh bit is set: swizzle the right qh byte into place, isolate the bit, compare
  const hbits = (idx) => f.get(st.qv).get(idx).op('i8x16.swizzle').get(st.mask).op('v128.and').get(st.mask).op('i8x16.eq').get(st.c10).op('v128.and');
  f.get(st.raw); lowNibbles(f, st); hbits(st.idx0); f.op('v128.or').set(n0);
  f.get(st.raw); highNibbles(f, st); hbits(st.idx1); f.op('v128.or').set(n1);
}

function genQ5_0(m, D) {
  const init = (st) => {
    q50Init(st); baseInit(st); // q50Init also sets its own c0f (same value)
    const v = () => st.f.local(T.v128);
    Object.assign(st, { n1: v(), x0: v(), x1: v(), neg: v() });
  };
  scaffold(m, 'matvec_q5_0', {
    blockBytes: 22, xStride: 32, dStride: 4,
    init,
    block(st, R) {
      const f = st.f;
      f.get(st.xp).op('v128.load').set(st.x0); f.get(st.xp).op('v128.load', { offset: 16 }).set(st.x1);
      D.neg(f, st, [st.x0, st.x1], 4, st.neg); // offset 16
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.get(st.dp).op('f32.load').op('f32.mul').set(st.sc);
        q50Unsigned(st, st.p[j], st.n, st.n1);
        D.d32(f, st, st.x0, st.n, st.x1, st.n1, st.neg);
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
        q50Unsigned(st, st.p[j], st.wn[j][0], st.wn[j][1]);
      }
      for (let q = 0; q < TT; q++) {
        f.get(st.xp[q]).op('v128.load').set(st.x0); f.get(st.xp[q]).op('v128.load', { offset: 16 }).set(st.x1);
        D.neg(f, st, [st.x0, st.x1], 4, st.neg);
        for (let j = 0; j < R; j++) {
          f.get(st.ws[j]).get(st.dp[q]).op('f32.load').op('f32.mul').set(st.sc);
          D.d32(f, st, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.neg);
          accumulate(f, st.acc[j][q], st.sc);
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

// ------------------------------------------------------------------------------------------------------ Q4_K / Q5_K
/**
 * Leave the 16 quants (nibbles, plus the 5th bit plane for Q5_K) of sub-block s, bytes at `off` of block ptr p, in local dst.
 * Q5_K: bit s of the 5th-bit plane hq goes to bit 4 with ONE 16-bit shift and the 0x10 mask (shifts of at most 4 to the left
 * and 3 to the right never move a neighbouring byte's bits into bit 4).
 */
function nibbles(st, p, off, s, five, hq, dst) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: off });
  if (s & 1) f.i32(4).op('i16x8.shr_u');
  f.get(st.c0f).op('v128.and');
  if (five) {
    f.get(hq);
    if (s < 4) f.i32(4 - s).op('i16x8.shl'); else if (s > 4) f.i32(s - 4).op('i16x8.shr_u');
    f.get(st.c10).op('v128.and').op('v128.or');
  }
  f.set(dst);
}

function genQ45K(m, D, decodeName, batchName, five) {
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
          nibbles(st, p, base, s, five, st.hq[j][0], st.raw);
          nibbles(st, p, base + 16, s, five, st.hq[j][1], st.raw1);
          byteOf(f, s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
          D.d32(f, st, st.xr0, st.raw, st.xr1, st.raw1, st.zv);
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
          nibbles(st, p, base, s, five, st.hq[j][0], st.wn[j][0]);
          nibbles(st, p, base + 16, s, five, st.hq[j][1], st.wn[j][1]);
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
            D.d32(f, st, st.x0, st.wn[j][0], st.x1, st.wn[j][1], st.zv);
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
function q6Init(st) {
  const f = st.f; const v = () => f.local(T.v128); const fl = () => f.local(T.f32);
  baseInit(st);
  Object.assign(st, { q: v(), c30: v(), dj: [fl(), fl()], coef: fl(), dxs: fl(), h: st.h });
  f.v128Const(splat(0x30)).set(st.c30);
}

/**
 * The 16 UNSIGNED weights q (0..63) of group (half, k, g) of row pointer p, left in local st.q. Low 4 bits come from ql
 * (low nibble for k < 2, high nibble for k >= 2), the 2 high bits from qh bits 2k..2k+1, moved to bits 4..5 by ONE
 * 16-bit shift (left by 4-2k, or right by 2 for k = 3) and the 0x30 mask.
 */
function q6Group(st, p, half, k, g) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: 64 * half + 16 * g + (k & 1) * 32 });
  if (k >= 2) f.i32(4).op('i16x8.shr_u');
  f.get(st.c0f).op('v128.and');
  f.get(p).op('v128.load', { offset: 128 + 32 * half + 16 * g });
  const sh = 4 - 2 * k;
  if (sh > 0) f.i32(sh).op('i16x8.shl'); else if (sh < 0) f.i32(-sh).op('i16x8.shr_u');
  f.get(st.c30).op('v128.and').op('v128.or').set(st.q);
}

function genQ6K(m, D) {
  scaffold(m, 'matvec_q6_k', {
    blockBytes: 210, xStride: 256, dStride: 32,
    init(st) {
      q6Init(st);
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
            D.neg(f, st, [st.xg], 5, st.neg); // offset 32, per group of 16
            for (let j = 0; j < R; j++) {
              q6Group(st, st.p[j], half, k, g);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
              D.d16(f, st, st.xg, st.q, st.neg);
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
      q6Init(st);
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
              q6Group(st, st.p[j], half, k, g); f.get(st.q).set(st.wq[j]);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').set(st.arow[j]);
            }
            for (let q = 0; q < TT; q++) {
              f.get(st.xp[q]).op('v128.load', { offset: xblock * 32 + 16 * g }).set(st.xg);
              f.get(st.dp[q]).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
              D.neg(f, st, [st.xg], 5, st.neg);
              for (let j = 0; j < R; j++) {
                f.get(st.arow[j]).get(st.dxs).op('f32.mul').set(st.coef);
                D.d16(f, st, st.xg, st.wq[j], st.neg);
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
 * @param {'i16'|'add'} [flavor] which relaxed dot instruction the kernels are built from (see dotOps)
 */
export function genRelaxedKernels(m, flavor = 'i16') {
  if (flavor !== 'i16' && flavor !== 'add') throw new Error(`unknown relaxed flavor "${flavor}"`);
  const D = dotOps(flavor);
  genQ4_0(m, D);
  genQ5_0(m, D);
  genQ45K(m, D, 'matvec_q4_k', 'mm_q4_k', false);
  genQ45K(m, D, 'matvec_q5_k', 'mm_q5_k', true);
  genQ6K(m, D);
}
