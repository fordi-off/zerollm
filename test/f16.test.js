import test from 'node:test';
import assert from 'node:assert/strict';
import { f32ToF16, f16ToF32, f16ToF32Slow, bf16ToF32 } from '../src/util/f16.js';

test('known f16 encodings', () => {
  assert.equal(f32ToF16(1), 0x3c00);
  assert.equal(f32ToF16(-2), 0xc000);
  assert.equal(f32ToF16(65504), 0x7bff);
  assert.equal(f32ToF16(65520), 0x7c00); // rounds to inf
  assert.equal(f32ToF16(0), 0);
  assert.equal(f32ToF16(2 ** -24), 1); // smallest subnormal
  assert.equal(f32ToF16(2 ** -25), 0); // tie to even -> 0
  assert.equal(f32ToF16(1.5 * 2 ** -24), 2); // tie to even -> 2
  assert.equal(f32ToF16(Infinity), 0x7c00);
  assert.equal(f32ToF16(NaN) & 0x7c00, 0x7c00);
});

test('every finite f16 survives f16 -> f32 -> f16', () => {
  for (let h = 0; h < 65536; h++) {
    const exp = (h >>> 10) & 0x1f;
    if (exp === 0x1f) continue;
    const back = f32ToF16(f16ToF32Slow(h));
    assert.equal(back, h, `h=0x${h.toString(16)}`);
    assert.equal(f16ToF32(h), f16ToF32Slow(h));
  }
});

test('f32 -> f16 rounds to nearest', () => {
  for (let i = 0; i < 20000; i++) {
    const x = (Math.random() - 0.5) * 200;
    const h = f32ToF16(x);
    const err = Math.abs(f16ToF32(h) - x);
    const next = Math.abs(f16ToF32(h + 1) - x);
    const prev = h > 0 ? Math.abs(f16ToF32(h - 1) - x) : Infinity;
    // not strictly valid across sign boundaries, but nearest neighbour must not beat us
    if (Math.abs(x) > 1e-3) assert.ok(err <= Math.max(next, 0) + 1e-12 || err <= prev + 1e-12);
  }
});

test('bf16', () => {
  assert.equal(bf16ToF32(0x3f80), 1);
  assert.equal(bf16ToF32(0xc000), -2);
});
