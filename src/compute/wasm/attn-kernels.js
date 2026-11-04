// WebAssembly SIMD attention kernels for ONE head (one query vector against the cached keys/values of its KV head).
//
//   attn_scores(q, k, scores, npos, hd, scale)   scores[t] = scale * dot(q, K[t])      t < npos
//   attn_av(p, v, out, npos, hd)                 out[i]   = sum_t p[t] * V[t][i]
//
// All pointers are byte offsets into the worker's private memory; K and V are f32 (or f16 for the *_f16 variants),
// [pos][hd] contiguous for the head.
// hd must be a multiple of 16. The softmax between the two calls is done by the caller.
import { T } from './builder.js';

/** Emit: 8 f16 values from the v128 in local `src` -> two f32x4 left in locals dstLo / dstHi (arithmetic conversion). */
function f16x8ToF32(f, src, dstLo, dstHi, tmp, c7fff, c8000, big) {
  const half = (extend, dst) => {
    f.get(src).op(extend).set(tmp);
    f.get(tmp).get(c7fff).op('v128.and').i32(13).op('i32x4.shl')
      .get(tmp).get(c8000).op('v128.and').i32(16).op('i32x4.shl').op('v128.or')
      .get(big).op('f32x4.mul').set(dst);
  };
  half('i32x4.extend_low_i16x8_u', dstLo);
  half('i32x4.extend_high_i16x8_u', dstHi);
}

export function genAttnKernels(m) {
  genAttnKernelsF16(m);
  // ------------------------------------------------------------------------------------------------ scores
  {
    const f = m.func('attn_scores', [T.i32, T.i32, T.i32, T.i32, T.i32, T.f32], []);
    const [Q, K, S, NPOS, HD, SCALE] = [0, 1, 2, 3, 4, 5];
    const t = f.local(T.i32); const qp = f.local(T.i32); const kp = f.local(T.i32); const sp = f.local(T.i32); const n = f.local(T.i32);
    const acc = [0, 1, 2, 3].map(() => f.local(T.v128));
    const zero = new Array(16).fill(0);
    f.block((done) => {
      f.get(NPOS).op('i32.eqz'); f.brIf(done); // nothing to do for npos == 0
      f.get(K).set(kp); f.get(S).set(sp); f.i32(0).set(t);
      f.loop((next) => {
        for (const a of acc) f.v128Const(zero).set(a);
        f.get(Q).set(qp);
        f.get(HD).i32(4).op('i32.shr_u').set(n); // number of 16-float chunks
        f.loop((chunk) => {
          for (let c = 0; c < 4; c++) {
            f.get(qp).op('v128.load', { offset: 16 * c }).get(kp).op('v128.load', { offset: 16 * c }).op('f32x4.mul').get(acc[c]).op('f32x4.add').set(acc[c]);
          }
          f.get(qp).i32(64).op('i32.add').set(qp);
          f.get(kp).i32(64).op('i32.add').set(kp);
          f.get(n).i32(1).op('i32.sub').tee(n); f.brIf(chunk);
        });
        // sum the four accumulators, then the four lanes
        f.get(acc[0]).get(acc[1]).op('f32x4.add').set(acc[0]);
        f.get(acc[2]).get(acc[3]).op('f32x4.add').set(acc[2]);
        f.get(acc[0]).get(acc[2]).op('f32x4.add').set(acc[0]);
        f.get(sp);
        f.get(acc[0]).op('f32x4.extract_lane', { lane: 0 }).get(acc[0]).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
        f.get(acc[0]).op('f32x4.extract_lane', { lane: 2 }).get(acc[0]).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
        f.op('f32.add').get(SCALE).op('f32.mul').op('f32.store');
        f.get(sp).i32(4).op('i32.add').set(sp);
        f.get(t).i32(1).op('i32.add').tee(t);
        f.get(NPOS).op('i32.lt_u'); f.brIf(next);
      });
    });
  }

  // ------------------------------------------------------------------------------------------------ weighted values
  {
    const f = m.func('attn_av', [T.i32, T.i32, T.i32, T.i32, T.i32], []);
    const [P, V, OUT, NPOS, HD] = [0, 1, 2, 3, 4];
    const c = f.local(T.i32); const t = f.local(T.i32); const pp = f.local(T.i32); const vp = f.local(T.i32); const stride = f.local(T.i32);
    const nch = f.local(T.i32); const ps = f.local(T.v128);
    const acc = [0, 1, 2, 3].map(() => f.local(T.v128));
    const zero = new Array(16).fill(0);
    f.get(HD).i32(2).op('i32.shl').set(stride);
    f.get(HD).i32(4).op('i32.shr_u').set(nch);
    f.i32(0).set(c);
    f.block((allDone) => {
      f.get(nch).op('i32.eqz'); f.brIf(allDone);
      f.loop((nextChunk) => {
        for (const a of acc) f.v128Const(zero).set(a);
        f.get(P).set(pp);
        f.get(V).get(c).i32(6).op('i32.shl').op('i32.add').set(vp); // chunk c starts 64*c bytes into each row
        f.i32(0).set(t);
        f.block((noPos) => {
          f.get(NPOS).op('i32.eqz'); f.brIf(noPos);
          f.loop((nextT) => {
            f.get(pp).op('f32.load').op('f32x4.splat').set(ps);
            for (let k = 0; k < 4; k++) {
              f.get(vp).op('v128.load', { offset: 16 * k }).get(ps).op('f32x4.mul').get(acc[k]).op('f32x4.add').set(acc[k]);
            }
            f.get(vp).get(stride).op('i32.add').set(vp);
            f.get(pp).i32(4).op('i32.add').set(pp);
            f.get(t).i32(1).op('i32.add').tee(t);
            f.get(NPOS).op('i32.lt_u'); f.brIf(nextT);
          });
        });
        for (let k = 0; k < 4; k++) {
          f.get(OUT).get(c).i32(6).op('i32.shl').op('i32.add').get(acc[k]).op('v128.store', { offset: 16 * k });
        }
        f.get(c).i32(1).op('i32.add').tee(c);
        f.get(nch).op('i32.lt_u'); f.brIf(nextChunk);
      });
    });
  }
}

/** f16-cache variants: identical structure, K/V rows are 2*hd bytes of half floats converted on the fly. */
function genAttnKernelsF16(m) {
  const consts = (f) => {
    const v = () => f.local(T.v128);
    const c = { c7fff: v(), c8000: v(), big: v(), tmp: v(), raw: v(), lo: v(), hi: v() };
    f.v128Const(new Array(4).fill([0xff, 0x7f, 0, 0]).flat()).set(c.c7fff);
    f.v128Const(new Array(4).fill([0x00, 0x80, 0, 0]).flat()).set(c.c8000);
    f.f32(2 ** 112).op('f32x4.splat').set(c.big);
    return c;
  };
  const zero = new Array(16).fill(0);

  {
    const f = m.func('attn_scores_f16', [T.i32, T.i32, T.i32, T.i32, T.i32, T.f32], []);
    const [Q, K, S, NPOS, HD, SCALE] = [0, 1, 2, 3, 4, 5];
    const t = f.local(T.i32); const qp = f.local(T.i32); const kp = f.local(T.i32); const sp = f.local(T.i32); const n = f.local(T.i32);
    const acc = [0, 1, 2, 3].map(() => f.local(T.v128));
    const c = consts(f);
    f.block((done) => {
      f.get(NPOS).op('i32.eqz'); f.brIf(done);
      f.get(K).set(kp); f.get(S).set(sp); f.i32(0).set(t);
      f.loop((next) => {
        for (const a of acc) f.v128Const(zero).set(a);
        f.get(Q).set(qp);
        f.get(HD).i32(4).op('i32.shr_u').set(n);
        f.loop((chunk) => {
          for (let h = 0; h < 2; h++) { // 16 halves = 2 vectors of 8 -> 4 f32x4
            f.get(kp).op('v128.load', { offset: 16 * h }).set(c.raw);
            f16x8ToF32(f, c.raw, c.lo, c.hi, c.tmp, c.c7fff, c.c8000, c.big);
            f.get(qp).op('v128.load', { offset: 32 * h }).get(c.lo).op('f32x4.mul').get(acc[2 * h]).op('f32x4.add').set(acc[2 * h]);
            f.get(qp).op('v128.load', { offset: 32 * h + 16 }).get(c.hi).op('f32x4.mul').get(acc[2 * h + 1]).op('f32x4.add').set(acc[2 * h + 1]);
          }
          f.get(qp).i32(64).op('i32.add').set(qp);
          f.get(kp).i32(32).op('i32.add').set(kp);
          f.get(n).i32(1).op('i32.sub').tee(n); f.brIf(chunk);
        });
        f.get(acc[0]).get(acc[1]).op('f32x4.add').set(acc[0]);
        f.get(acc[2]).get(acc[3]).op('f32x4.add').set(acc[2]);
        f.get(acc[0]).get(acc[2]).op('f32x4.add').set(acc[0]);
        f.get(sp);
        f.get(acc[0]).op('f32x4.extract_lane', { lane: 0 }).get(acc[0]).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
        f.get(acc[0]).op('f32x4.extract_lane', { lane: 2 }).get(acc[0]).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
        f.op('f32.add').get(SCALE).op('f32.mul').op('f32.store');
        f.get(sp).i32(4).op('i32.add').set(sp);
        f.get(t).i32(1).op('i32.add').tee(t);
        f.get(NPOS).op('i32.lt_u'); f.brIf(next);
      });
    });
  }

  {
    const f = m.func('attn_av_f16', [T.i32, T.i32, T.i32, T.i32, T.i32], []);
    const [P, V, OUT, NPOS, HD] = [0, 1, 2, 3, 4];
    const c0 = f.local(T.i32); const t = f.local(T.i32); const pp = f.local(T.i32); const vp = f.local(T.i32); const stride = f.local(T.i32);
    const nch = f.local(T.i32); const ps = f.local(T.v128);
    const acc = [0, 1, 2, 3].map(() => f.local(T.v128));
    const c = consts(f);
    f.get(HD).i32(1).op('i32.shl').set(stride);          // bytes per row of f16
    f.get(HD).i32(4).op('i32.shr_u').set(nch);
    f.i32(0).set(c0);
    f.block((allDone) => {
      f.get(nch).op('i32.eqz'); f.brIf(allDone);
      f.loop((nextChunk) => {
        for (const a of acc) f.v128Const(zero).set(a);
        f.get(P).set(pp);
        f.get(V).get(c0).i32(5).op('i32.shl').op('i32.add').set(vp); // chunk c starts 32*c bytes into each row
        f.i32(0).set(t);
        f.block((noPos) => {
          f.get(NPOS).op('i32.eqz'); f.brIf(noPos);
          f.loop((nextT) => {
            f.get(pp).op('f32.load').op('f32x4.splat').set(ps);
            for (let h = 0; h < 2; h++) {
              f.get(vp).op('v128.load', { offset: 16 * h }).set(c.raw);
              f16x8ToF32(f, c.raw, c.lo, c.hi, c.tmp, c.c7fff, c.c8000, c.big);
              f.get(c.lo).get(ps).op('f32x4.mul').get(acc[2 * h]).op('f32x4.add').set(acc[2 * h]);
              f.get(c.hi).get(ps).op('f32x4.mul').get(acc[2 * h + 1]).op('f32x4.add').set(acc[2 * h + 1]);
            }
            f.get(vp).get(stride).op('i32.add').set(vp);
            f.get(pp).i32(4).op('i32.add').set(pp);
            f.get(t).i32(1).op('i32.add').tee(t);
            f.get(NPOS).op('i32.lt_u'); f.brIf(nextT);
          });
        });
        for (let k = 0; k < 4; k++) {
          f.get(OUT).get(c0).i32(6).op('i32.shl').op('i32.add').get(acc[k]).op('v128.store', { offset: 16 * k });
        }
        f.get(c0).i32(1).op('i32.add').tee(c0);
        f.get(nch).op('i32.lt_u'); f.brIf(nextChunk);
      });
    });
  }
}
