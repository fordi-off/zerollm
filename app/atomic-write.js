// Write a file by writing a temp file and renaming it over the target (a crash never leaves half a file).
// On Windows the rename fails with EPERM / EBUSY / EACCES while another handle (a reader, a virus scanner, the indexer) has the
// target open for a moment. Those errors are transient, so retry briefly; if it still fails, write the target directly.
import fs from 'node:fs';

const TRANSIENT = new Set(['EPERM', 'EBUSY', 'EACCES']);
const pause = (ms) => { try { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); } catch { /* no wait available: retry at once */ } };

/** @param {string} target @param {string} text */
export function writeFileAtomic(target, text) {
  const tmp = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, text, 'utf8');
  for (let attempt = 0; ; attempt++) {
    try { fs.renameSync(tmp, target); return; } catch (e) {
      if (!TRANSIENT.has(e.code)) { try { fs.unlinkSync(tmp); } catch { /* gone */ } throw e; }
      if (attempt >= 8) { // still locked: overwrite in place (not atomic, but the data lands)
        try { fs.writeFileSync(target, text, 'utf8'); } finally { try { fs.unlinkSync(tmp); } catch { /* gone */ } }
        return;
      }
      pause(5 * (attempt + 1));
    }
  }
}
