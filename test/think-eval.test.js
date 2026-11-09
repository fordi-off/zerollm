import { test } from 'node:test';
import assert from 'node:assert/strict';
import { numbersIn, grade, QUESTIONS } from '../tools/think-eval.mjs';

test('think-eval grading: Norwegian number formats, tolerance, all-of vs any-of', () => {
  assert.deepEqual(numbersIn('Du har 16 958,81 kr og 3.5 døgn, x=2 og x=3'), [16958.81, 3.5, 2, 3]);
  const q = (id) => QUESTIONS.find((x) => x.id === id);
  assert.equal(grade('Farten er omtrent 874,8 m/s', q(3)), true);
  assert.equal(grade('Farten er omtrent 880 m/s', q(3)), false, '0.6 % off is outside the 0.5 % tolerance');
  assert.equal(grade('v ≈ 872.5 m/s', q(4)), true);
  assert.equal(grade('v ≈ 874.8 m/s', q(4)), false, 'the constant-g answer is not the inhomogeneous one');
  assert.equal(grade('x = 2 eller x = 3', q(6)), true);
  assert.equal(grade('x = 2', q(6)), false);
  assert.equal(grade('Etter 12 år: 16 958,81 kr', q(5)), true);
  assert.equal(grade('391', q(1)), true);
  assert.equal(grade('Svaret er 392', q(1)), false);
  assert.ok(Math.abs(q(4).expect[0] - 872.52) < 0.05 && Math.abs(q(3).expect[0] - 874.75) < 0.05);
});
