import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { writeFileAtomic } from '../app/atomic-write.js';

test('writeFileAtomic: creates and replaces a file, leaves no temp file behind', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zc-atomic-'));
  try {
    const f = path.join(dir, 'a.json');
    writeFileAtomic(f, '{"a":1}');
    writeFileAtomic(f, '{"a":2}');
    assert.equal(fs.readFileSync(f, 'utf8'), '{"a":2}');
    assert.deepEqual(fs.readdirSync(dir), ['a.json']);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('writeFileAtomic: a transient EPERM from rename (Windows, file held open) is retried; a hard error is thrown and cleaned up', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zc-atomic-'));
  const real = fs.renameSync;
  try {
    const f = path.join(dir, 'b.json');
    let calls = 0;
    fs.renameSync = (a, b) => { if (++calls <= 3) { const e = new Error('EPERM'); e.code = 'EPERM'; throw e; } return real(a, b); };
    writeFileAtomic(f, 'ok');
    assert.equal(fs.readFileSync(f, 'utf8'), 'ok');
    assert.equal(calls, 4);
    fs.renameSync = () => { const e = new Error('EPERM'); e.code = 'EPERM'; throw e; };
    writeFileAtomic(f, 'forced'); // never succeeds: falls back to a direct write
    assert.equal(fs.readFileSync(f, 'utf8'), 'forced');
    fs.renameSync = () => { const e = new Error('ENOSPC'); e.code = 'ENOSPC'; throw e; };
    assert.throws(() => writeFileAtomic(f, 'x'), /ENOSPC/);
    assert.deepEqual(fs.readdirSync(dir), ['b.json']);
  } finally { fs.renameSync = real; fs.rmSync(dir, { recursive: true, force: true }); }
});
