// WebAssembly SIMD kernels for Q5_0 and the K-quants (Q4_K, Q5_K, Q6_K).
//
//   decode : matvec_q5_0 / matvec_q4_k / matvec_q5_k / matvec_q6_k (w, xq, xd, xs, y, rows, nblocks)
//   batched: mm_q5_0 / mm_q4_k / mm_q5_k / mm_q6_k (w, xq16, xd, xs, y, rows, nblocks, ntok)   (xq16: int16-widened, token t at element t*cols)
//
// `nblocks` is cols/32 for Q5_0 and cols/256 (super-blocks) for the K-quants. The activation is the Q8_0-style vector
// the other kernels use: quants (int8 for decode, int16-widened for batches), an f32 scale per 32 elements (xd) and
// the integer sum of each 32-block's quants (xs, as f32). Every K-quant sub-block is 32 wide (Q6_K groups 16 wide), so
// each maps onto one activation block:
//   Q4_K/Q5_K  sub-block s:  d*sc*sum(q*xq) * xd  -  dmin*mn * xd * xs         (xs = sum(xq) of the block)
//   Q6_K       group g:      d * scale_g * sum((q-32)*xq) * xd
//   Q5_0       block:        d * sum((q5-16)*xq) * xd
// Integer dot products (measured with tools/kbench.mjs, see docs/BENCHMARKS.md):
//   decode  : i16x8.extmul (int8 x int8 -> int16) + pairwise adds. With weights of at most 6 bits, 4 products never
//             overflow int16, so neither weights nor activations need widening (Q4_K/Q5_K/Q6_K; Q5_0 is faster the old way).
//   batched : weights are widened to int16 ONCE per tile and reused for 4 tokens, activations are stored pre-widened
//             as int16 -> i32x4.dot_i16x8_s.  (extmul per token was 20-50% slower here.)
// The batched kernels perform every float operation in the same order as the decode kernels, so their results are
// bit-identical to running the decode kernel token by token.
import { T } from './builder.js';

const SIX = 0x3f3f3f3f; const LOW4 = 0x0f0f0f0f; const TOP2 = 0x30303030;
export const splat = (v) => new Array(16).fill(v);

/** f16 at [ptr + offset] -> f32 (arithmetic conversion; handles denormals and sign), left on the stack. */
export function f16At(f, ptr, offset, h) {
  f.get(ptr).op('i32.load16_u', { offset }).set(h);
  f.get(h).i32(0x7fff).op('i32.and').i32(13).op('i32.shl')
    .get(h).i32(0x8000).op('i32.and').i32(16).op('i32.shl').op('i32.or')
    .op('f32.reinterpret_i32').f32(2 ** 112).op('f32.mul');
}

/** Sum of the 4 f32 lanes of local v, left on the stack (fixed lane order: (l0+l1)+(l2+l3)). */
export function hsum(f, v) {
  f.get(v).op('f32x4.extract_lane', { lane: 0 }).get(v).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
  f.get(v).op('f32x4.extract_lane', { lane: 2 }).get(v).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
  f.op('f32.add');
}

/** With an i32x4 `sum` on the stack:  acc += float(sum) * splat(coef). */
export function accumulate(f, acc, coef) {
  f.op('f32x4.convert_i32x4_s').get(coef).op('f32x4.splat').op('f32x4.mul').get(acc).op('f32x4.add').set(acc);
}

/**
 * i32x4 partial sums of the dot product of 32 small signed/unsigned weights (two int8 vectors n0, n1: elements 0-15 and
 * 16-31) with 32 int8 activations (x0, x1). |weight| <= 63 so every int16 lane holds at most 4 products without overflow.
 */
function dot32(f, n0, x0, n1, x1) {
  f.get(n0).get(x0).op('i16x8.extmul_low_i8x16_s').get(n0).get(x0).op('i16x8.extmul_high_i8x16_s').op('i16x8.add');
  f.get(n1).get(x1).op('i16x8.extmul_low_i8x16_s').get(n1).get(x1).op('i16x8.extmul_high_i8x16_s').op('i16x8.add');
  f.op('i16x8.add').op('i32x4.extadd_pairwise_i16x8_s');
}

/** same for one group of 16 weights (Q6_K groups): 2 products per int16 lane */
function dot16(f, n, x) {
  f.get(n).get(x).op('i16x8.extmul_low_i8x16_s').get(n).get(x).op('i16x8.extmul_high_i8x16_s').op('i16x8.add').op('i32x4.extadd_pairwise_i16x8_s');
}

/** (old style) i32x4 dot of 16 x int8 in local `v` with the widened activation halves lo/hi */
function dot16wide(f, v, lo, hi, t) {
  f.get(v).tee(t).op('i16x8.extend_low_i8x16_s').get(lo).op('i32x4.dot_i16x8_s');
  f.get(t).op('i16x8.extend_high_i8x16_s').get(hi).op('i32x4.dot_i16x8_s').op('i32x4.add');
}

/** i32x4 dot of 4 widened weight vectors w[0..3] with 4 int16 activation vectors x[0..3] (32 elements) */
function dotWide(f, w, x) {
  f.get(w[0]).get(x[0]).op('i32x4.dot_i16x8_s');
  f.get(w[1]).get(x[1]).op('i32x4.dot_i16x8_s').op('i32x4.add');
  f.get(w[2]).get(x[2]).op('i32x4.dot_i16x8_s');
  f.get(w[3]).get(x[3]).op('i32x4.dot_i16x8_s').op('i32x4.add');
  f.op('i32x4.add');
}

/** widen the two int8 vectors a, b into w[0..3] (lo/hi of a, lo/hi of b) */
function widen4(f, a, b, w) {
  f.get(a).op('i16x8.extend_low_i8x16_s').set(w[0]); f.get(a).op('i16x8.extend_high_i8x16_s').set(w[1]);
  f.get(b).op('i16x8.extend_low_i8x16_s').set(w[2]); f.get(b).op('i16x8.extend_high_i8x16_s').set(w[3]);
}

// ---------------------------------------------------------------------------------------------------- decode scaffold
/**
 * Loops over groups of R rows (2, then 1) and blocks; the format-specific part is spec.block / spec.result.
 * spec.noXs: the kernel has no activation-block-sums parameter, signature (w, xq, xd, y, rows, nb) like matvec_q8_0/q4_0.
 */
export function scaffold(m, name, spec) {
  const f = m.func(name, new Array(spec.noXs ? 6 : 7).fill(T.i32), []);
  const [W, XQ, XD, XS, Y, ROWS, NB] = spec.noXs ? [0, 1, 2, -1, 3, 4, 5] : [0, 1, 2, 3, 4, 5, 6];
  const st = {
    f, W, XQ, XD, NB,
    rb: f.local(T.i32), r: f.local(T.i32), b: f.local(T.i32), xp: f.local(T.i32), dp: f.local(T.i32), sp: f.local(T.i32), h: f.local(T.i32),
    p: [f.local(T.i32), f.local(T.i32)],
    acc: [f.local(T.v128), f.local(T.v128)], mins: [f.local(T.f32), f.local(T.f32)],
  };
  spec.init(st);
  f.get(NB).i32(spec.blockBytes).op('i32.mul').set(st.rb);
  f.i32(0).set(st.r);
  const zero = splat(0);
  const groups = (R) => {
    f.block((done) => {
      f.loop((next) => {
        f.get(ROWS).get(st.r).op('i32.sub').i32(R).op('i32.lt_u'); f.brIf(done);
        f.get(W).get(st.r).get(st.rb).op('i32.mul').op('i32.add').set(st.p[0]);
        if (R === 2) f.get(st.p[0]).get(st.rb).op('i32.add').set(st.p[1]);
        for (let j = 0; j < R; j++) { f.v128Const(zero).set(st.acc[j]); f.f32(0).set(st.mins[j]); }
        f.get(XQ).set(st.xp); f.get(XD).set(st.dp); if (!spec.noXs) f.get(XS).set(st.sp); f.get(NB).set(st.b);
        f.loop((blk) => {
          spec.block(st, R);
          for (let j = 0; j < R; j++) f.get(st.p[j]).i32(spec.blockBytes).op('i32.add').set(st.p[j]);
          f.get(st.xp).i32(spec.xStride).op('i32.add').set(st.xp);
          f.get(st.dp).i32(spec.dStride).op('i32.add').set(st.dp);
          if (!spec.noXs) f.get(st.sp).i32(spec.dStride).op('i32.add').set(st.sp);
          f.get(st.b).i32(1).op('i32.sub').tee(st.b); f.brIf(blk);
        });
        for (let j = 0; j < R; j++) {
          f.get(Y).get(st.r).i32(j).op('i32.add').i32(2).op('i32.shl').op('i32.add');
          spec.result(st, j);
          f.op('f32.store');
        }
        f.get(st.r).i32(R).op('i32.add').set(st.r);
        f.br(next);
      });
    });
  };
  groups(2);
  groups(1);
}

// ------------------------------------------------------------------------------------------------- batched scaffold
/**
 * Tiles of R rows (2, then 1) x TT tokens (4, then 1). The int16-widened activation of token t starts at element t*cols.
 * spec.block(st, R, TT) emits the work for ONE block (all rows of the tile, all tokens of the tile).
 * spec.xBytes is the activation bytes per block and token (64 / 512 for the int16-widened activations; the relaxed-SIMD
 * kernels store int8 and use 32 / 256). spec.noXs: signature (w, xq, xd, y, rows, nb, ntok), no block-sums parameter.
 */
export function scaffoldBatch(m, name, spec) {
  const f = m.func(name, new Array(spec.noXs ? 7 : 8).fill(T.i32), []);
  const [W, XQ, XD, XS, Y, ROWS, NB, NTOK] = spec.noXs ? [0, 1, 2, -1, 3, 4, 5, 6] : [0, 1, 2, 3, 4, 5, 6, 7];
  const i = () => f.local(T.i32);
  const st = {
    f, NB,
    rb: i(), xtok: i(), dtok: i(), r: i(), tk: i(), b: i(), h: i(),
    wbase: [i(), i()], p: [i(), i()], xp: [i(), i(), i(), i()], dp: [i(), i(), i(), i()], sp: [i(), i(), i(), i()],
    acc: [[0, 1, 2, 3].map(() => f.local(T.v128)), [0, 1, 2, 3].map(() => f.local(T.v128))],
    mins: [[0, 1, 2, 3].map(() => f.local(T.f32)), [0, 1, 2, 3].map(() => f.local(T.f32))],
  };
  spec.init(st);
  const zero = splat(0);
  f.get(NB).i32(spec.blockBytes).op('i32.mul').set(st.rb);
  f.get(NB).i32(spec.xBytes).op('i32.mul').set(st.xtok);
  f.get(NB).i32(spec.dBytes).op('i32.mul').set(st.dtok);
  f.i32(0).set(st.r);
  const rowGroups = (R) => {
    f.block((rowsDone) => {
      f.loop((nextRows) => {
        f.get(ROWS).get(st.r).op('i32.sub').i32(R).op('i32.lt_u'); f.brIf(rowsDone);
        f.get(W).get(st.r).get(st.rb).op('i32.mul').op('i32.add').set(st.wbase[0]);
        if (R === 2) f.get(st.wbase[0]).get(st.rb).op('i32.add').set(st.wbase[1]);
        f.i32(0).set(st.tk);
        const tokGroups = (TT) => {
          f.block((tokDone) => {
            f.loop((nextTok) => {
              f.get(NTOK).get(st.tk).op('i32.sub').i32(TT).op('i32.lt_u'); f.brIf(tokDone);
              f.get(XQ).get(st.tk).get(st.xtok).op('i32.mul').op('i32.add').set(st.xp[0]);
              f.get(XD).get(st.tk).get(st.dtok).op('i32.mul').op('i32.add').set(st.dp[0]);
              if (!spec.noXs) f.get(XS).get(st.tk).get(st.dtok).op('i32.mul').op('i32.add').set(st.sp[0]);
              for (let q = 1; q < TT; q++) {
                f.get(st.xp[q - 1]).get(st.xtok).op('i32.add').set(st.xp[q]);
                f.get(st.dp[q - 1]).get(st.dtok).op('i32.add').set(st.dp[q]);
                if (!spec.noXs) f.get(st.sp[q - 1]).get(st.dtok).op('i32.add').set(st.sp[q]);
              }
              for (let j = 0; j < R; j++) {
                f.get(st.wbase[j]).set(st.p[j]);
                for (let q = 0; q < TT; q++) { f.v128Const(zero).set(st.acc[j][q]); f.f32(0).set(st.mins[j][q]); }
              }
              f.get(NB).set(st.b);
              f.loop((blk) => {
                spec.block(st, R, TT);
                for (let j = 0; j < R; j++) f.get(st.p[j]).i32(spec.blockBytes).op('i32.add').set(st.p[j]);
                for (let q = 0; q < TT; q++) {
                  f.get(st.xp[q]).i32(spec.xBytes).op('i32.add').set(st.xp[q]);
                  f.get(st.dp[q]).i32(spec.dBytes).op('i32.add').set(st.dp[q]);
                  if (!spec.noXs) f.get(st.sp[q]).i32(spec.dBytes).op('i32.add').set(st.sp[q]);
                }
                f.get(st.b).i32(1).op('i32.sub').tee(st.b); f.brIf(blk);
              });
              for (let q = 0; q < TT; q++) {
                for (let j = 0; j < R; j++) {
                  f.get(Y).get(st.tk).i32(q).op('i32.add').get(ROWS).op('i32.mul').get(st.r).op('i32.add').i32(j).op('i32.add').i32(2).op('i32.shl').op('i32.add');
                  spec.result(st, j, q);
                  f.op('f32.store');
                }
              }
              f.get(st.tk).i32(TT).op('i32.add').set(st.tk);
              f.br(nextTok);
            });
          });
        };
        tokGroups(4);
        tokGroups(1);
        f.get(st.r).i32(R).op('i32.add').set(st.r);
        f.br(nextRows);
      });
    });
  };
  rowGroups(2);
  rowGroups(1);
}

// ---------------------------------------------------------------------------------------------------------- Q5_0
/** locals + constants shared by the decode and batched Q5_0 kernels */
export function q50Init(st) {
  const f = st.f; const v = () => f.local(T.v128);
  Object.assign(st, {
    t: v(), raw: v(), qv: v(), n: v(),
    c0f: v(), c10: v(), c16: v(), idx0: v(), idx1: v(), mask: v(), sc: f.local(T.f32), qh: f.local(T.i32),
  });
  f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x10)).set(st.c10); f.v128Const(splat(16)).set(st.c16);
  f.v128Const([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).set(st.idx0); // bits 0..15 live in qh bytes 0,1
  f.v128Const([2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]).set(st.idx1); // bits 16..31 live in qh bytes 2,3
  f.v128Const([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]).set(st.mask);
}

/**
 * n0 / n1 = the block's two 16-element halves of (q5 - 16) as int8 vectors, from the weight block at ptr `p`.
 * unsigned: leave q5 itself (0..31) instead, for the relaxed-SIMD dot (its weight operand must be 0..127); the caller
 * then subtracts 16 * sum(x) once per block.
 */
export function q50Unpack(st, p, n0, n1, unsigned = false) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: 6 }).set(st.raw);
  f.get(p).op('i32.load', { offset: 2 }).op('i32x4.splat').set(st.qv);
  // 0x10 in every byte lane whose qh bit is set: swizzle the right qh byte into place, isolate the bit, compare
  const hbits = (idx) => f.get(st.qv).get(idx).op('i8x16.swizzle').get(st.mask).op('v128.and').get(st.mask).op('i8x16.eq').get(st.c10).op('v128.and');
  f.get(st.raw).get(st.c0f).op('v128.and'); hbits(st.idx0); f.op('v128.or'); if (!unsigned) f.get(st.c16).op('i8x16.sub'); f.set(n0);
  f.get(st.raw).i32(4).op('i8x16.shr_u'); hbits(st.idx1); f.op('v128.or'); if (!unsigned) f.get(st.c16).op('i8x16.sub'); f.set(n1);
}

function genQ5_0(m) {
  scaffold(m, 'matvec_q5_0', {
    blockBytes: 22, xStride: 32, dStride: 4,
    init(st) {
      q50Init(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { xl0: v(), xh0: v(), xl1: v(), xh1: v(), n1: v() });
    },
    block(st, R) {
      const f = st.f;
      f.get(st.xp).op('v128.load').tee(st.t).op('i16x8.extend_low_i8x16_s').set(st.xl0);
      f.get(st.t).op('i16x8.extend_high_i8x16_s').set(st.xh0);
      f.get(st.xp).op('v128.load', { offset: 16 }).tee(st.t).op('i16x8.extend_low_i8x16_s').set(st.xl1);
      f.get(st.t).op('i16x8.extend_high_i8x16_s').set(st.xh1);
      for (let j = 0; j < R; j++) {
        const p = st.p[j];
        f16At(f, p, 0, st.h); f.get(st.dp).op('f32.load').op('f32.mul').set(st.sc);
        q50Unpack(st, p, st.n, st.n1);
        dot16wide(f, st.n, st.xl0, st.xh0, st.t);
        dot16wide(f, st.n1, st.xl1, st.xh1, st.t);
        f.op('i32x4.add');
        accumulate(f, st.acc[j], st.sc);
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });

  scaffoldBatch(m, 'mm_q5_0', {
    blockBytes: 22, xBytes: 64, dBytes: 4,
    init(st) {
      q50Init(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, { n1: v(), wv: [[v(), v(), v(), v()], [v(), v(), v(), v()]], x: [v(), v(), v(), v()], ws: [f.local(T.f32), f.local(T.f32)] });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) {
        f16At(f, st.p[j], 0, st.h); f.set(st.ws[j]);
        q50Unpack(st, st.p[j], st.n, st.n1);
        widen4(f, st.n, st.n1, st.wv[j]);
      }
      for (let q = 0; q < TT; q++) {
        for (let c = 0; c < 4; c++) f.get(st.xp[q]).op('v128.load', { offset: 16 * c }).set(st.x[c]);
        for (let j = 0; j < R; j++) {
          f.get(st.ws[j]).get(st.dp[q]).op('f32.load').op('f32.mul').set(st.sc);
          dotWide(f, st.wv[j], st.x);
          accumulate(f, st.acc[j][q], st.sc);
        }
      }
    },
    result(st, j, q) { hsum(st.f, st.acc[j][q]); },
  });
}

// ---------------------------------------------------------------------------------------------------------- Q4_K / Q5_K
/** locals and the per-row scalar unpacking of the 6-bit scale/min table (same code for decode and batched) */
export function q45kInit(st) {
  const f = st.f; const v = () => f.local(T.v128); const i = () => f.local(T.i32); const fl = () => f.local(T.f32);
  Object.assign(st, {
    t: v(), raw: v(), c0f: v(), c10: v(),
    hq: [[v(), v()], [v(), v()]],
    dj: [fl(), fl()], dmj: [fl(), fl()], coef: fl(), dxs: fl(), bs: fl(),
    scLo: [i(), i()], scHi: [i(), i()], mnLo: [i(), i()], mnHi: [i(), i()], w0: i(), w1: i(), w2: i(),
  });
  f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x10)).set(st.c10);
}

export function q45kRowSetup(st, j, five) {
  const f = st.f; const p = st.p[j];
  f16At(f, p, 0, st.h); f.set(st.dj[j]);
  f16At(f, p, 2, st.h); f.set(st.dmj[j]);
  f.get(p).op('i32.load', { offset: 4 }).set(st.w0);
  f.get(p).op('i32.load', { offset: 8 }).set(st.w1);
  f.get(p).op('i32.load', { offset: 12 }).set(st.w2);
  f.get(st.w0).i32(SIX).op('i32.and').set(st.scLo[j]);
  f.get(st.w1).i32(SIX).op('i32.and').set(st.mnLo[j]);
  f.get(st.w2).i32(LOW4).op('i32.and').get(st.w0).i32(2).op('i32.shr_u').i32(TOP2).op('i32.and').op('i32.or').set(st.scHi[j]);
  f.get(st.w2).i32(4).op('i32.shr_u').i32(LOW4).op('i32.and').get(st.w1).i32(2).op('i32.shr_u').i32(TOP2).op('i32.and').op('i32.or').set(st.mnHi[j]);
  if (five) {
    f.get(p).op('v128.load', { offset: 16 }).set(st.hq[j][0]);
    f.get(p).op('v128.load', { offset: 32 }).set(st.hq[j][1]);
  }
}

/** push byte k of an i32 word local (0..255) */
export function byteOf(f, word, k) { f.get(word); if (k) f.i32(8 * k).op('i32.shr_u'); f.i32(0xff).op('i32.and'); }

/** Leave the 16 nibbles (plus the 5th bit plane for Q5_K) of sub-block s, bytes at `off` of block ptr p, in local `dst`. */
export function q45kNibbles(st, p, off, s, five, hq, dst) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: off }).set(dst);
  if (s & 1) f.get(dst).i32(4).op('i8x16.shr_u'); else f.get(dst).get(st.c0f).op('v128.and');
  if (five) { // bit s of the 5th-bit plane -> 0x10
    f.get(hq);
    if (s < 4) f.i32(4 - s).op('i8x16.shl'); else if (s > 4) f.i32(s - 4).op('i8x16.shr_u');
    f.get(st.c10).op('v128.and').op('v128.or');
  }
  f.set(dst);
}

function genQ45K(m, decodeName, batchName, five, opts) {
  const wide = opts.kdot === 'wide';
  const qsOff = five ? 48 : 16;
  const blockBytes = five ? 176 : 144;

  scaffold(m, decodeName, {
    blockBytes, xStride: 256, dStride: 32,
    init(st) {
      q45kInit(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, { xr0: v(), xr1: v(), raw1: v(), xl0: v(), xh0: v(), xl1: v(), xh1: v() });
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) q45kRowSetup(st, j, five);
      for (let s = 0; s < 8; s++) {
        f.get(st.xp).op('v128.load', { offset: 32 * s }).set(st.xr0);
        f.get(st.xp).op('v128.load', { offset: 32 * s + 16 }).set(st.xr1);
        if (wide) {
          f.get(st.xr0).op('i16x8.extend_low_i8x16_s').set(st.xl0); f.get(st.xr0).op('i16x8.extend_high_i8x16_s').set(st.xh0);
          f.get(st.xr1).op('i16x8.extend_low_i8x16_s').set(st.xl1); f.get(st.xr1).op('i16x8.extend_high_i8x16_s').set(st.xh1);
        }
        f.get(st.dp).op('f32.load', { offset: 4 * s }).set(st.dxs);
        f.get(st.sp).op('f32.load', { offset: 4 * s }).set(st.bs);
        for (let j = 0; j < R; j++) {
          const p = st.p[j]; const base = qsOff + 32 * (s >> 1);
          q45kNibbles(st, p, base, s, five, st.hq[j][0], st.raw);
          q45kNibbles(st, p, base + 16, s, five, st.hq[j][1], st.raw1);
          byteOf(f, s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
          if (wide) { dot16wide(f, st.raw, st.xl0, st.xh0, st.t); dot16wide(f, st.raw1, st.xl1, st.xh1, st.t); f.op('i32x4.add'); } else dot32(f, st.raw, st.xr0, st.raw1, st.xr1);
          accumulate(f, st.acc[j], st.coef);
          byteOf(f, s < 4 ? st.mnLo[j] : st.mnHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dmj[j]).op('f32.mul').get(st.dxs).op('f32.mul').get(st.bs).op('f32.mul').set(st.coef);
          f.get(st.mins[j]).get(st.coef).op('f32.add').set(st.mins[j]);
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); st.f.get(st.mins[j]).op('f32.sub'); },
  });

  scaffoldBatch(m, batchName, {
    blockBytes, xBytes: 512, dBytes: 32,
    init(st) {
      q45kInit(st);
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, {
        raw1: v(), wv: [[v(), v(), v(), v()], [v(), v(), v(), v()]], x: [v(), v(), v(), v()],
        arow: [f.local(T.f32), f.local(T.f32)], mrow: [f.local(T.f32), f.local(T.f32)],
      });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) q45kRowSetup(st, j, five);
      for (let s = 0; s < 8; s++) {
        for (let j = 0; j < R; j++) {
          const p = st.p[j]; const base = qsOff + 32 * (s >> 1);
          q45kNibbles(st, p, base, s, five, st.hq[j][0], st.raw);
          q45kNibbles(st, p, base + 16, s, five, st.hq[j][1], st.raw1);
          widen4(f, st.raw, st.raw1, st.wv[j]);
          byteOf(f, s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').set(st.arow[j]);
          byteOf(f, s < 4 ? st.mnLo[j] : st.mnHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dmj[j]).op('f32.mul').set(st.mrow[j]);
        }
        for (let q = 0; q < TT; q++) {
          for (let c = 0; c < 4; c++) f.get(st.xp[q]).op('v128.load', { offset: 64 * s + 16 * c }).set(st.x[c]);
          f.get(st.dp[q]).op('f32.load', { offset: 4 * s }).set(st.dxs);
          f.get(st.sp[q]).op('f32.load', { offset: 4 * s }).set(st.bs);
          for (let j = 0; j < R; j++) {
            f.get(st.arow[j]).get(st.dxs).op('f32.mul').set(st.coef);
            dotWide(f, st.wv[j], st.x);
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

// ---------------------------------------------------------------------------------------------------------- Q6_K
export function q6kInit(st) {
  const f = st.f; const v = () => f.local(T.v128); const fl = () => f.local(T.f32);
  Object.assign(st, { t: v(), q: v(), c0f: v(), c03: v(), c20: v(), dj: [fl(), fl()], coef: fl(), dxs: fl() });
  f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x03)).set(st.c03); f.v128Const(splat(0x20)).set(st.c20);
}

/**
 * the 16 weights of group (half, k, g) of row pointer p, as (q - 32) int8, left in local st.q.
 * unsigned: leave q (0..63) instead (relaxed-SIMD dot; the caller subtracts 32 * sum(x of the group)).
 */
export function q6kGroup(st, p, half, k, g, unsigned = false) {
  const f = st.f;
  f.get(p).op('v128.load', { offset: 64 * half + 16 * g + (k & 1) * 32 }); // low 4 bits
  if (k < 2) f.get(st.c0f).op('v128.and'); else f.i32(4).op('i8x16.shr_u');
  f.get(p).op('v128.load', { offset: 128 + 32 * half + 16 * g });          // 2 high bits
  if (k === 3) f.i32(6).op('i8x16.shr_u');
  else { if (k > 0) f.i32(2 * k).op('i8x16.shr_u'); f.get(st.c03).op('v128.and'); }
  f.i32(4).op('i8x16.shl').op('v128.or'); if (!unsigned) f.get(st.c20).op('i8x16.sub'); f.set(st.q);
}

function genQ6K(m, opts) {
  const wide = opts.kdot === 'wide';
  scaffold(m, 'matvec_q6_k', {
    blockBytes: 210, xStride: 256, dStride: 32,
    init(st) {
      q6kInit(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { xg: v(), xl: v(), xh: v() });
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) { f16At(f, st.p[j], 208, st.h); f.set(st.dj[j]); }
      for (let half = 0; half < 2; half++) {
        for (let k = 0; k < 4; k++) {
          const xblock = half * 4 + k;
          f.get(st.dp).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
          for (let g = 0; g < 2; g++) {
            if (wide) {
              f.get(st.xp).op('v128.load', { offset: xblock * 32 + 16 * g }).tee(st.t).op('i16x8.extend_low_i8x16_s').set(st.xl);
              f.get(st.t).op('i16x8.extend_high_i8x16_s').set(st.xh);
            } else f.get(st.xp).op('v128.load', { offset: xblock * 32 + 16 * g }).set(st.xg);
            for (let j = 0; j < R; j++) {
              q6kGroup(st, st.p[j], half, k, g);
              if (wide) dot16wide(f, st.q, st.xl, st.xh, st.t); else dot16(f, st.q, st.xg);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
              accumulate(f, st.acc[j], st.coef);
            }
          }
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });

  scaffoldBatch(m, 'mm_q6_k', {
    blockBytes: 210, xBytes: 512, dBytes: 32,
    init(st) {
      q6kInit(st);
      const v = () => st.f.local(T.v128);
      Object.assign(st, { wl: [v(), v()], wh: [v(), v()], xlo: v(), xhi: v(), arow: [st.f.local(T.f32), st.f.local(T.f32)] });
    },
    block(st, R, TT) {
      const f = st.f;
      for (let j = 0; j < R; j++) { f16At(f, st.p[j], 208, st.h); f.set(st.dj[j]); }
      for (let half = 0; half < 2; half++) {
        for (let k = 0; k < 4; k++) {
          const xblock = half * 4 + k;
          for (let g = 0; g < 2; g++) {
            for (let j = 0; j < R; j++) {
              q6kGroup(st, st.p[j], half, k, g);
              f.get(st.q).op('i16x8.extend_low_i8x16_s').set(st.wl[j]); f.get(st.q).op('i16x8.extend_high_i8x16_s').set(st.wh[j]);
              f.get(st.p[j]).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').set(st.arow[j]);
            }
            for (let q = 0; q < TT; q++) {
              f.get(st.xp[q]).op('v128.load', { offset: xblock * 64 + 32 * g }).set(st.xlo);
              f.get(st.xp[q]).op('v128.load', { offset: xblock * 64 + 32 * g + 16 }).set(st.xhi);
              f.get(st.dp[q]).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
              for (let j = 0; j < R; j++) {
                f.get(st.arow[j]).get(st.dxs).op('f32.mul').set(st.coef);
                f.get(st.wl[j]).get(st.xlo).op('i32x4.dot_i16x8_s');
                f.get(st.wh[j]).get(st.xhi).op('i32x4.dot_i16x8_s').op('i32x4.add');
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
 * Adds the Q5_0 / Q4_K / Q5_K / Q6_K decode and batched kernels to a module builder.
 * @param {{kdot?: 'extmul'|'wide'}} [opts] kdot selects the integer dot form of the Q4_K/Q5_K/Q6_K DECODE kernels
 *   (see the header comment); which one is faster depends on the CPU, tools/kbench.mjs measures both.
 */
export function genKQuantKernels(m, opts = {}) {
  genQ5_0(m);
  genQ45K(m, 'matvec_q4_k', 'mm_q4_k', false, opts);
  genQ45K(m, 'matvec_q5_k', 'mm_q5_k', true, opts);
  genQ6K(m, opts);
}
