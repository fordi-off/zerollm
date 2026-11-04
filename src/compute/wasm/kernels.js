// WebAssembly SIMD matvec kernels, generated as bytes by builder.js (no compiler involved).
//
//   matvec_q8_0(w, xq, xd, y, rows, nb)   matvec_q4_0(w, xq, xd, y, rows, nb)
//
//   w    byte offset of the first weight row (rows * nb blocks, ggml layout: f16 scale + 32 quants)
//   xq   byte offset of the int8-quantized activation (nb * 32 values)
//   xd   byte offset of the f32 activation scales (nb values)
//   y    byte offset where `rows` f32 results are written
//
//   mm_q8_0(w, xq16, xd, y, rows, nb, ntok)   mm_q4_0(...)      batched: y[t][r] for ntok activation vectors
//
//   xq16  int16-widened quantized activations, token t at element offset t * nb * 32
//   xd    f32 scales, token t at offset t * nb
//   y     results, token-major: y[t * rows + r]
//
// Everything lives in ONE memory (the worker's private WebAssembly.Memory, imported as env.mem).
// Inner loop per 32-weight block and row: widen int8 -> int16, i32x4.dot_i16x8_s against the (shared, pre-widened)
// activation, convert to f32, scale by (d_w * d_x), accumulate in an f32x4. Four rows are processed together so the
// widened activation vectors are reused 4x and four independent accumulators hide latency.
import { ModuleBuilder, T } from './builder.js';

export const Q8_0_BLOCK = 34;
export const Q4_0_BLOCK = 18;

/** @param {'q8'|'q4'} kind */
function genMatvec(m, name, kind) {
  const blockBytes = kind === 'q8' ? Q8_0_BLOCK : Q4_0_BLOCK;
  const f = m.func(name, [T.i32, T.i32, T.i32, T.i32, T.i32, T.i32], []);
  const [W, XQ, XD, Y, ROWS, NB] = [0, 1, 2, 3, 4, 5];
  const rb = f.local(T.i32); const r = f.local(T.i32); const b = f.local(T.i32);
  const xp = f.local(T.i32); const dp = f.local(T.i32); const h = f.local(T.i32);
  const p = [0, 1, 2, 3].map(() => f.local(T.i32));
  const acc = [0, 1, 2, 3].map(() => f.local(T.v128));
  const xl0 = f.local(T.v128); const xh0 = f.local(T.v128); const xl1 = f.local(T.v128); const xh1 = f.local(T.v128);
  const wv = f.local(T.v128); const t = f.local(T.v128);
  const c0f = f.local(T.v128); const c08 = f.local(T.v128);
  const sc = f.local(T.f32);
  const zero = new Array(16).fill(0);

  f.v128Const(new Array(16).fill(0x0f)).set(c0f);
  f.v128Const(new Array(16).fill(8)).set(c08);
  f.get(NB).i32(blockBytes).op('i32.mul').set(rb);
  f.i32(0).set(r);

  /** one pass over groups of R rows; leaves `r` pointing at the first unprocessed row */
  const groups = (R) => {
    f.block((done) => {
      f.loop((next) => {
        // fewer than R rows left? -> leave
        f.get(ROWS).get(r).op('i32.sub').i32(R).op('i32.lt_u'); f.brIf(done);
        f.get(W).get(r).get(rb).op('i32.mul').op('i32.add').set(p[0]);
        for (let j = 1; j < R; j++) f.get(p[j - 1]).get(rb).op('i32.add').set(p[j]);
        for (let j = 0; j < R; j++) f.v128Const(zero).set(acc[j]);
        f.get(XQ).set(xp); f.get(XD).set(dp); f.get(NB).set(b);

        f.loop((blk) => {
          // activation chunks, widened once and shared by all R rows
          f.get(xp).op('v128.load').tee(t).op('i16x8.extend_low_i8x16_s').set(xl0);
          f.get(t).op('i16x8.extend_high_i8x16_s').set(xh0);
          f.get(xp).op('v128.load', { offset: 16 }).tee(t).op('i16x8.extend_low_i8x16_s').set(xl1);
          f.get(t).op('i16x8.extend_high_i8x16_s').set(xh1);

          for (let j = 0; j < R; j++) {
            // sc = f16_to_f32(scale of this weight block) * activation scale
            f.get(p[j]).op('i32.load16_u').set(h);
            f.get(h).i32(0x7fff).op('i32.and').i32(13).op('i32.shl')
              .get(h).i32(0x8000).op('i32.and').i32(16).op('i32.shl').op('i32.or')
              .op('f32.reinterpret_i32').f32(2 ** 112).op('f32.mul')
              .get(dp).op('f32.load').op('f32.mul').set(sc);

            // integer dot product of the 32 weights with the 32 activations -> i32x4 (partial sums)
            if (kind === 'q8') {
              f.get(p[j]).op('v128.load', { offset: 2 }).tee(wv).op('i16x8.extend_low_i8x16_s').get(xl0).op('i32x4.dot_i16x8_s');
              f.get(wv).op('i16x8.extend_high_i8x16_s').get(xh0).op('i32x4.dot_i16x8_s');
              f.op('i32x4.add');
              f.get(p[j]).op('v128.load', { offset: 18 }).tee(wv).op('i16x8.extend_low_i8x16_s').get(xl1).op('i32x4.dot_i16x8_s');
              f.get(wv).op('i16x8.extend_high_i8x16_s').get(xh1).op('i32x4.dot_i16x8_s');
              f.op('i32x4.add').op('i32x4.add');
            } else {
              // low nibbles are elements 0..15, high nibbles elements 16..31; value = nibble - 8
              f.get(p[j]).op('v128.load', { offset: 2 }).set(wv);
              f.get(wv).get(c0f).op('v128.and').get(c08).op('i8x16.sub').tee(t).op('i16x8.extend_low_i8x16_s').get(xl0).op('i32x4.dot_i16x8_s');
              f.get(t).op('i16x8.extend_high_i8x16_s').get(xh0).op('i32x4.dot_i16x8_s').op('i32x4.add');
              f.get(wv).i32(4).op('i8x16.shr_u').get(c08).op('i8x16.sub').tee(t).op('i16x8.extend_low_i8x16_s').get(xl1).op('i32x4.dot_i16x8_s');
              f.get(t).op('i16x8.extend_high_i8x16_s').get(xh1).op('i32x4.dot_i16x8_s').op('i32x4.add').op('i32x4.add');
            }
            // acc += float(sum) * sc
            f.op('f32x4.convert_i32x4_s').get(sc).op('f32x4.splat').op('f32x4.mul').get(acc[j]).op('f32x4.add').set(acc[j]);
            f.get(p[j]).i32(blockBytes).op('i32.add').set(p[j]);
          }
          f.get(xp).i32(32).op('i32.add').set(xp);
          f.get(dp).i32(4).op('i32.add').set(dp);
          f.get(b).i32(1).op('i32.sub').tee(b); f.brIf(blk);
        });

        // horizontal sums -> y[r + j]
        for (let j = 0; j < R; j++) {
          f.get(Y).get(r).i32(j).op('i32.add').i32(2).op('i32.shl').op('i32.add');
          f.get(acc[j]).op('f32x4.extract_lane', { lane: 0 }).get(acc[j]).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
          f.get(acc[j]).op('f32x4.extract_lane', { lane: 2 }).get(acc[j]).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
          f.op('f32.add').op('f32.store');
        }
        f.get(r).i32(R).op('i32.add').set(r);
        f.br(next);
      });
    });
  };
  groups(4);
  groups(1);
}


/**
 * Batched matmul: y[t][r] = dot(W[r], x[t]) for ntok tokens, reading each weight tile ONCE per token group.
 * Tiles are R=2 rows x T=4 tokens (+ remainder tiles). Weights are widened to int16 once per tile; the activations are
 * stored pre-widened as int16, so the inner loop is: 4 loads + 4 dots + 3 adds per (row, token, block).
 * Every float operation is performed in exactly the same order as in matvec_*, so the results are bit-identical to
 * the single-vector kernels (integer lane sums are exact; scale/accumulate order is unchanged).
 * @param {'q8'|'q4'} kind
 */
function genMatmul(m, name, kind) {
  const blockBytes = kind === 'q8' ? Q8_0_BLOCK : Q4_0_BLOCK;
  const f = m.func(name, [T.i32, T.i32, T.i32, T.i32, T.i32, T.i32, T.i32], []);
  const [W, XQ, XD, Y, ROWS, NB, NTOK] = [0, 1, 2, 3, 4, 5, 6];
  const rb = f.local(T.i32); const xs = f.local(T.i32); const ds = f.local(T.i32);
  const r = f.local(T.i32); const tk = f.local(T.i32); const b = f.local(T.i32); const h = f.local(T.i32);
  const wp = [0, 1].map(() => f.local(T.i32)); const wbase = [0, 1].map(() => f.local(T.i32));
  const xp = [0, 1, 2, 3].map(() => f.local(T.i32)); const dp = [0, 1, 2, 3].map(() => f.local(T.i32));
  const acc = [0, 1].map(() => [0, 1, 2, 3].map(() => f.local(T.v128)));
  const wv = [0, 1].map(() => [0, 1, 2, 3].map(() => f.local(T.v128))); // lo0, hi0, lo1, hi1 per row
  const x = [0, 1, 2, 3].map(() => f.local(T.v128));
  const t = f.local(T.v128); const raw = f.local(T.v128);
  const c0f = f.local(T.v128); const c08 = f.local(T.v128);
  const ws = [0, 1].map(() => f.local(T.f32)); const sc = f.local(T.f32);
  const zero = new Array(16).fill(0);

  f.v128Const(new Array(16).fill(0x0f)).set(c0f);
  f.v128Const(new Array(16).fill(8)).set(c08);
  f.get(NB).i32(blockBytes).op('i32.mul').set(rb);
  f.get(NB).i32(64).op('i32.mul').set(xs);
  f.get(NB).i32(4).op('i32.mul').set(ds);
  f.i32(0).set(r);

  /** load + widen the weights of row j for the current block; also computes ws[j] (f32 scale of the weight block) */
  const loadRow = (j) => {
    f.get(wp[j]).op('i32.load16_u').set(h);
    f.get(h).i32(0x7fff).op('i32.and').i32(13).op('i32.shl')
      .get(h).i32(0x8000).op('i32.and').i32(16).op('i32.shl').op('i32.or')
      .op('f32.reinterpret_i32').f32(2 ** 112).op('f32.mul').set(ws[j]);
    if (kind === 'q8') {
      f.get(wp[j]).op('v128.load', { offset: 2 }).tee(t).op('i16x8.extend_low_i8x16_s').set(wv[j][0]);
      f.get(t).op('i16x8.extend_high_i8x16_s').set(wv[j][1]);
      f.get(wp[j]).op('v128.load', { offset: 18 }).tee(t).op('i16x8.extend_low_i8x16_s').set(wv[j][2]);
      f.get(t).op('i16x8.extend_high_i8x16_s').set(wv[j][3]);
    } else {
      f.get(wp[j]).op('v128.load', { offset: 2 }).set(raw);
      f.get(raw).get(c0f).op('v128.and').get(c08).op('i8x16.sub').tee(t).op('i16x8.extend_low_i8x16_s').set(wv[j][0]);
      f.get(t).op('i16x8.extend_high_i8x16_s').set(wv[j][1]);
      f.get(raw).i32(4).op('i8x16.shr_u').get(c08).op('i8x16.sub').tee(t).op('i16x8.extend_low_i8x16_s').set(wv[j][2]);
      f.get(t).op('i16x8.extend_high_i8x16_s').set(wv[j][3]);
    }
  };

  /** row groups of R; inside, token groups of 4 then 1 */
  const rowGroups = (R) => {
    f.block((rowsDone) => {
      f.loop((nextRows) => {
        f.get(ROWS).get(r).op('i32.sub').i32(R).op('i32.lt_u'); f.brIf(rowsDone);
        f.get(W).get(r).get(rb).op('i32.mul').op('i32.add').set(wbase[0]);
        if (R === 2) f.get(wbase[0]).get(rb).op('i32.add').set(wbase[1]);
        f.i32(0).set(tk);

        const tokGroups = (TT) => {
          f.block((tokDone) => {
            f.loop((nextTok) => {
              f.get(NTOK).get(tk).op('i32.sub').i32(TT).op('i32.lt_u'); f.brIf(tokDone);
              f.get(XQ).get(tk).get(xs).op('i32.mul').op('i32.add').set(xp[0]);
              f.get(XD).get(tk).get(ds).op('i32.mul').op('i32.add').set(dp[0]);
              for (let q = 1; q < TT; q++) {
                f.get(xp[q - 1]).get(xs).op('i32.add').set(xp[q]);
                f.get(dp[q - 1]).get(ds).op('i32.add').set(dp[q]);
              }
              for (let j = 0; j < R; j++) { f.get(wbase[j]).set(wp[j]); for (let q = 0; q < TT; q++) f.v128Const(zero).set(acc[j][q]); }
              f.get(NB).set(b);

              f.loop((blk) => {
                for (let j = 0; j < R; j++) loadRow(j);
                for (let q = 0; q < TT; q++) {
                  for (let c = 0; c < 4; c++) f.get(xp[q]).op('v128.load', { offset: 16 * c }).set(x[c]);
                  for (let j = 0; j < R; j++) {
                    f.get(ws[j]).get(dp[q]).op('f32.load').op('f32.mul').set(sc);
                    f.get(wv[j][0]).get(x[0]).op('i32x4.dot_i16x8_s');
                    f.get(wv[j][1]).get(x[1]).op('i32x4.dot_i16x8_s').op('i32x4.add');
                    f.get(wv[j][2]).get(x[2]).op('i32x4.dot_i16x8_s');
                    f.get(wv[j][3]).get(x[3]).op('i32x4.dot_i16x8_s').op('i32x4.add');
                    f.op('i32x4.add');
                    f.op('f32x4.convert_i32x4_s').get(sc).op('f32x4.splat').op('f32x4.mul').get(acc[j][q]).op('f32x4.add').set(acc[j][q]);
                  }
                  f.get(xp[q]).i32(64).op('i32.add').set(xp[q]);
                  f.get(dp[q]).i32(4).op('i32.add').set(dp[q]);
                }
                for (let j = 0; j < R; j++) f.get(wp[j]).i32(blockBytes).op('i32.add').set(wp[j]);
                f.get(b).i32(1).op('i32.sub').tee(b); f.brIf(blk);
              });

              for (let q = 0; q < TT; q++) {
                for (let j = 0; j < R; j++) {
                  f.get(Y).get(tk).i32(q).op('i32.add').get(ROWS).op('i32.mul').get(r).op('i32.add').i32(j).op('i32.add').i32(2).op('i32.shl').op('i32.add');
                  const a = acc[j][q];
                  f.get(a).op('f32x4.extract_lane', { lane: 0 }).get(a).op('f32x4.extract_lane', { lane: 1 }).op('f32.add');
                  f.get(a).op('f32x4.extract_lane', { lane: 2 }).get(a).op('f32x4.extract_lane', { lane: 3 }).op('f32.add');
                  f.op('f32.add').op('f32.store');
                }
              }
              f.get(tk).i32(TT).op('i32.add').set(tk);
              f.br(nextTok);
            });
          });
        };
        tokGroups(4);
        tokGroups(1);
        f.get(r).i32(R).op('i32.add').set(r);
        f.br(nextRows);
      });
    });
  };
  rowGroups(2);
  rowGroups(1);
}

/** @returns {Uint8Array} the kernel module bytes */
export function buildKernelModule() {
  const m = new ModuleBuilder().withImportedMemory(1);
  genMatvec(m, 'matvec_q8_0', 'q8');
  genMatvec(m, 'matvec_q4_0', 'q4');
  genMatmul(m, 'mm_q8_0', 'q8');
  genMatmul(m, 'mm_q4_0', 'q4');
  return m.build();
}

/** @type {WebAssembly.Module|null} */
let compiled = null;

/** Compile once per thread, instantiate against a worker's private memory. */
export function instantiateKernels(memory) {
  compiled ??= new WebAssembly.Module(buildKernelModule());
  return new WebAssembly.Instance(compiled, { env: { mem: memory } }).exports;
}
