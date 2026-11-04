// WebAssembly SIMD matvec kernels for Q5_0 and the K-quants (Q4_K, Q5_K, Q6_K). Single activation vector (decode).
//
//   matvec_q5_0 / matvec_q4_k / matvec_q5_k / matvec_q6_k (w, xq, xd, y, rows, nblocks)
//
// `nblocks` is cols/32 for Q5_0 and cols/256 (super-blocks) for the K-quants. The activation is the same Q8_0-style
// int8 + f32-scale-per-32 vector the other kernels use; K-quant sub-blocks are 32 wide (Q6_K groups 16 wide), so each
// sub-block maps onto one activation block:
//   Q4_K/Q5_K  sub-block s:  sum_i (d*sc*q_i - dmin*mn) * x_i  =  d*sc*sum(q*xq) - dmin*mn*sum(xq), times the x scale
//   Q6_K       group g:      d * scale_g * sum((q-32)*xq) * x scale
//   Q5_0       block:        d * sum((q5-16)*xq) * x scale
// Two rows are processed together so the widened activation vectors are shared.
import { T } from './builder.js';

const SIX = 0x3f3f3f3f; const LOW4 = 0x0f0f0f0f; const TOP2 = 0x30303030;
const splat = (v) => new Array(16).fill(v);

/** f16 at [ptr + offset] -> f32 (arithmetic conversion; handles denormals and sign), left on the stack. */
function f16At(f, ptr, offset, h) {
  f.get(ptr).op('i32.load16_u', { offset }).set(h);
  f.get(h).i32(0x7fff).op('i32.and').i32(13).op('i32.shl')
    .get(h).i32(0x8000).op('i32.and').i32(16).op('i32.shl').op('i32.or')
    .op('f32.reinterpret_i32').f32(2 ** 112).op('f32.mul');
}

/** Sum of the 4 f32 lanes of local v, left on the stack. */
function hsum(f, v) {
  f.get(v).op('f32x4.extract_lane', { lane: 0 }).get(v).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
  f.get(v).op('f32x4.extract_lane', { lane: 2 }).get(v).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
  f.op('f32.add');
}

/** With an i32x4 `sum` on the stack:  acc += float(sum) * splat(coef). */
function accumulate(f, acc, coef) {
  f.op('f32x4.convert_i32x4_s').get(coef).op('f32x4.splat').op('f32x4.mul').get(acc).op('f32x4.add').set(acc);
}

/** i32x4 dot of a widened-int8 vector held in local `v` (16 x int8) with the widened activation halves lo/hi. */
function dot16(f, v, lo, hi, t) {
  f.get(v).tee(t).op('i16x8.extend_low_i8x16_s').get(lo).op('i32x4.dot_i16x8_s');
  f.get(t).op('i16x8.extend_high_i8x16_s').get(hi).op('i32x4.dot_i16x8_s').op('i32x4.add');
}

/** Loops over groups of R rows (2, then 1) and blocks; the format-specific part is spec.block / spec.result. */
function scaffold(m, name, spec) {
  const f = m.func(name, [T.i32, T.i32, T.i32, T.i32, T.i32, T.i32], []);
  const [W, XQ, XD, Y, ROWS, NB] = [0, 1, 2, 3, 4, 5];
  const st = {
    f, W, XQ, XD, NB,
    rb: f.local(T.i32), r: f.local(T.i32), b: f.local(T.i32), xp: f.local(T.i32), dp: f.local(T.i32), h: f.local(T.i32),
    p: [f.local(T.i32), f.local(T.i32)],
    acc: [f.local(T.v128), f.local(T.v128)], accm: [f.local(T.v128), f.local(T.v128)],
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
        for (let j = 0; j < R; j++) { f.v128Const(zero).set(st.acc[j]); f.v128Const(zero).set(st.accm[j]); }
        f.get(XQ).set(st.xp); f.get(XD).set(st.dp); f.get(NB).set(st.b);
        f.loop((blk) => {
          spec.block(st, R);
          for (let j = 0; j < R; j++) f.get(st.p[j]).i32(spec.blockBytes).op('i32.add').set(st.p[j]);
          f.get(st.xp).i32(spec.xStride).op('i32.add').set(st.xp);
          f.get(st.dp).i32(spec.dStride).op('i32.add').set(st.dp);
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

// ---------------------------------------------------------------------------------------------------------- Q5_0
function genQ5_0(m) {
  scaffold(m, 'matvec_q5_0', {
    blockBytes: 22, xStride: 32, dStride: 4,
    init(st) {
      const f = st.f; const v = () => f.local(T.v128);
      Object.assign(st, {
        xl0: v(), xh0: v(), xl1: v(), xh1: v(), t: v(), raw: v(), qv: v(), n: v(),
        c0f: v(), c10: v(), c16: v(), idx0: v(), idx1: v(), mask: v(), sc: f.local(T.f32), qh: f.local(T.i32),
      });
      f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x10)).set(st.c10); f.v128Const(splat(16)).set(st.c16);
      f.v128Const([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).set(st.idx0); // bits 0..15 live in qh bytes 0,1
      f.v128Const([2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]).set(st.idx1); // bits 16..31 live in qh bytes 2,3
      f.v128Const([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]).set(st.mask);
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
        f.get(p).op('v128.load', { offset: 6 }).set(st.raw);
        f.get(p).op('i32.load', { offset: 2 }).op('i32x4.splat').set(st.qv);
        // 0x10 in every byte lane whose qh bit is set: swizzle the right qh byte into place, isolate the bit, compare
        const hbits = (idx) => f.get(st.qv).get(idx).op('i8x16.swizzle').get(st.mask).op('v128.and').get(st.mask).op('i8x16.eq').get(st.c10).op('v128.and');
        f.get(st.raw).get(st.c0f).op('v128.and'); hbits(st.idx0); f.op('v128.or').get(st.c16).op('i8x16.sub').set(st.n);
        dot16(f, st.n, st.xl0, st.xh0, st.t);
        f.get(st.raw).i32(4).op('i8x16.shr_u'); hbits(st.idx1); f.op('v128.or').get(st.c16).op('i8x16.sub').set(st.n);
        dot16(f, st.n, st.xl1, st.xh1, st.t);
        f.op('i32x4.add');
        accumulate(f, st.acc[j], st.sc);
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });
}

// ---------------------------------------------------------------------------------------------------------- Q4_K / Q5_K
function genQ45K(m, name, five) {
  const qsOff = five ? 48 : 16;
  scaffold(m, name, {
    blockBytes: five ? 176 : 144, xStride: 256, dStride: 32,
    init(st) {
      const f = st.f; const v = () => f.local(T.v128); const i = () => f.local(T.i32); const fl = () => f.local(T.f32);
      Object.assign(st, {
        xr0: v(), xr1: v(), xl0: v(), xh0: v(), xl1: v(), xh1: v(), sx: v(), t: v(), raw0: v(), raw1: v(), c0f: v(), c10: v(),
        hq: [[v(), v()], [v(), v()]],
        dj: [fl(), fl()], dmj: [fl(), fl()], coef: fl(), dxs: fl(),
        scLo: [i(), i()], scHi: [i(), i()], mnLo: [i(), i()], mnHi: [i(), i()], w0: i(), w1: i(), w2: i(),
      });
      f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x10)).set(st.c10);
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) {
        const p = st.p[j];
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
      const byteOf = (word, k) => { f.get(word); if (k) f.i32(8 * k).op('i32.shr_u'); f.i32(0xff).op('i32.and'); };
      for (let s = 0; s < 8; s++) {
        f.get(st.xp).op('v128.load', { offset: 32 * s }).set(st.xr0);
        f.get(st.xp).op('v128.load', { offset: 32 * s + 16 }).set(st.xr1);
        f.get(st.xr0).op('i16x8.extend_low_i8x16_s').set(st.xl0); f.get(st.xr0).op('i16x8.extend_high_i8x16_s').set(st.xh0);
        f.get(st.xr1).op('i16x8.extend_low_i8x16_s').set(st.xl1); f.get(st.xr1).op('i16x8.extend_high_i8x16_s').set(st.xh1);
        // sum of the 32 activation quants, as i32x4 (lanes add up to the total)
        f.get(st.xr0).op('i16x8.extadd_pairwise_i8x16_s').get(st.xr1).op('i16x8.extadd_pairwise_i8x16_s').op('i16x8.add').op('i32x4.extadd_pairwise_i16x8_s').set(st.sx);
        f.get(st.dp).op('f32.load', { offset: 4 * s }).set(st.dxs);
        for (let j = 0; j < R; j++) {
          const p = st.p[j];
          const base = qsOff + 32 * (s >> 1);
          const nib = (rawLocal, off, hq) => {
            f.get(p).op('v128.load', { offset: off }).set(rawLocal);
            if (s & 1) f.get(rawLocal).i32(4).op('i8x16.shr_u'); else f.get(rawLocal).get(st.c0f).op('v128.and');
            if (five) { // bit s of the 5th-bit plane -> 0x10
              f.get(hq);
              if (s < 4) f.i32(4 - s).op('i8x16.shl'); else if (s > 4) f.i32(s - 4).op('i8x16.shr_u');
              f.get(st.c10).op('v128.and').op('v128.or');
            }
            f.set(rawLocal);
          };
          nib(st.raw0, base, st.hq[j][0]); nib(st.raw1, base + 16, st.hq[j][1]);
          // coefficients: A = d*sc*dx, B = dmin*mn*dx
          byteOf(s < 4 ? st.scLo[j] : st.scHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
          dot16(f, st.raw0, st.xl0, st.xh0, st.t); dot16(f, st.raw1, st.xl1, st.xh1, st.t); f.op('i32x4.add');
          accumulate(f, st.acc[j], st.coef);
          byteOf(s < 4 ? st.mnLo[j] : st.mnHi[j], s & 3); f.op('f32.convert_i32_s').get(st.dmj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
          f.get(st.sx); accumulate(f, st.accm[j], st.coef);
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); hsum(st.f, st.accm[j]); st.f.op('f32.sub'); },
  });
}

// ---------------------------------------------------------------------------------------------------------- Q6_K
function genQ6K(m) {
  scaffold(m, 'matvec_q6_k', {
    blockBytes: 210, xStride: 256, dStride: 32,
    init(st) {
      const f = st.f; const v = () => f.local(T.v128); const fl = () => f.local(T.f32);
      Object.assign(st, { xl: v(), xh: v(), t: v(), q: v(), c0f: v(), c03: v(), c20: v(), dj: [fl(), fl()], coef: fl(), dxs: fl() });
      f.v128Const(splat(0x0f)).set(st.c0f); f.v128Const(splat(0x03)).set(st.c03); f.v128Const(splat(0x20)).set(st.c20);
    },
    block(st, R) {
      const f = st.f;
      for (let j = 0; j < R; j++) { f16At(f, st.p[j], 208, st.h); f.set(st.dj[j]); }
      for (let half = 0; half < 2; half++) {
        for (let k = 0; k < 4; k++) {
          const xblock = half * 4 + k;
          f.get(st.dp).op('f32.load', { offset: 4 * xblock }).set(st.dxs);
          for (let g = 0; g < 2; g++) {
            f.get(st.xp).op('v128.load', { offset: xblock * 32 + 16 * g }).tee(st.t).op('i16x8.extend_low_i8x16_s').set(st.xl);
            f.get(st.t).op('i16x8.extend_high_i8x16_s').set(st.xh);
            for (let j = 0; j < R; j++) {
              const p = st.p[j];
              // low 4 bits of the quant
              f.get(p).op('v128.load', { offset: 64 * half + 16 * g + (k & 1) * 32 });
              if (k < 2) f.get(st.c0f).op('v128.and'); else f.i32(4).op('i8x16.shr_u');
              // 2 high bits
              f.get(p).op('v128.load', { offset: 128 + 32 * half + 16 * g });
              if (k === 3) f.i32(6).op('i8x16.shr_u');
              else { if (k > 0) f.i32(2 * k).op('i8x16.shr_u'); f.get(st.c03).op('v128.and'); }
              f.i32(4).op('i8x16.shl').op('v128.or').get(st.c20).op('i8x16.sub').set(st.q);
              dot16(f, st.q, st.xl, st.xh, st.t);
              f.get(p).op('i32.load8_s', { offset: 192 + half * 8 + k * 2 + g }).op('f32.convert_i32_s').get(st.dj[j]).op('f32.mul').get(st.dxs).op('f32.mul').set(st.coef);
              accumulate(f, st.acc[j], st.coef);
            }
          }
        }
      }
    },
    result(st, j) { hsum(st.f, st.acc[j]); },
  });
}

/** Adds the Q5_0 / Q4_K / Q5_K / Q6_K decode kernels to a module builder. */
export function genKQuantKernels(m) {
  genQ5_0(m);
  genQ45K(m, 'matvec_q4_k', false);
  genQ45K(m, 'matvec_q5_k', true);
  genQ6K(m);
}
