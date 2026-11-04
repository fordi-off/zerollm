// Sequential, resumable model downloads with Node's built-in fetch (redirects, Range). No external programs.
//
// A file is written as "<name>.part" and renamed when complete and checked (size matches, starts with "GGUF"). A cancelled or
// failed download leaves the .part file, and the next attempt continues from where it stopped.
import fs from 'node:fs';
import path from 'node:path';
import { CATALOG } from './catalog.js';

const RETRIES = 4;
const wait = (ms, signal) => new Promise((resolve) => { const t = setTimeout(resolve, ms); signal?.addEventListener('abort', () => { clearTimeout(t); resolve(); }, { once: true }); });

/** Free bytes on the volume holding `dir` (null when unknown). */
export function freeDiskBytes(dir) {
  try { const s = fs.statfsSync(dir); return Number(s.bavail) * Number(s.bsize); } catch { return null; }
}

/**
 * @param {{paths: {modelsDir: string}, log: {info: Function, error: Function}, emit: (type: string, data: any) => void,
 *          catalog?: typeof CATALOG, retryDelayMs?: number}} o
 */
export function createDownloader({ paths, log, emit, catalog = CATALOG, retryDelayMs = 2000, onDone = null }) {
  const entries = new Map(catalog.map((e) => [e.id, e]));
  let queue = []; let current = null; let running = false; let abort = null;
  const failed = new Map(); // id -> message
  const done = new Set();

  const dest = (e) => path.join(paths.modelsDir, e.filename);
  const status = () => ({
    queue: queue.map((e) => e.id), failed: Object.fromEntries(failed), completed: [...done],
    current: current ? { id: current.id, received: current.received, total: current.total, speed: current.speed, eta: current.eta, state: current.state } : null,
  });
  const pushStatus = () => emit('download', status());

  async function fetchOnce(e, cur) {
    const part = `${dest(e)}.part`;
    let have = 0; try { have = fs.statSync(part).size; } catch { have = 0; }
    const headers = have > 0 ? { Range: `bytes=${have}-` } : {};
    const res = await fetch(e.url, { headers, redirect: 'follow', signal: abort.signal });
    if (res.status === 416) { // the .part is already complete (or garbage): let the final check decide
      return;
    }
    if (res.status !== 200 && res.status !== 206) throw Object.assign(new Error(res.status === 404 ? 'Filen finnes ikke på serveren (404).' : `Serveren svarte ${res.status}.`), { fatal: res.status === 404 || res.status === 401 || res.status === 403 });
    let total;
    if (res.status === 206) {
      const m = /bytes (\d+)-\d+\/(\d+)/.exec(res.headers.get('content-range') ?? '');
      total = m ? Number(m[2]) : have + Number(res.headers.get('content-length') ?? 0);
      if (m && Number(m[1]) !== have) { fs.rmSync(part, { force: true }); throw new Error('server gave an unexpected range'); }
    } else { have = 0; total = Number(res.headers.get('content-length') ?? 0); } // 200: whole file again
    cur.total = total || e.sizeBytes; cur.received = have;
    if (have === 0) fs.rmSync(part, { force: true });
    const free = freeDiskBytes(paths.modelsDir);
    if (free !== null && cur.total - have > free - 200_000_000) throw Object.assign(new Error(`Ikke nok ledig diskplass: trenger ${((cur.total - have) / 1e9).toFixed(1)} GB, ${(free / 1e9).toFixed(1)} GB er ledig.`), { fatal: true });
    const fd = fs.openSync(part, have > 0 ? 'a' : 'w');
    let lastEmit = 0; let t0 = performance.now(); let b0 = have;
    try {
      for await (const chunk of res.body) {
        fs.writeSync(fd, chunk);
        cur.received += chunk.length;
        const now = performance.now();
        if (now - lastEmit > 500) {
          const dt = (now - t0) / 1000;
          if (dt >= 1) { cur.speed = (cur.received - b0) / dt; t0 = now; b0 = cur.received; }
          cur.eta = cur.speed > 0 ? (cur.total - cur.received) / cur.speed : null;
          lastEmit = now; pushStatus();
        }
      }
    } finally { fs.closeSync(fd); }
    pushStatus(); // final 100 % update
  }

  function finalize(e, cur) {
    const part = `${dest(e)}.part`;
    const size = fs.statSync(part).size;
    if (cur.total && size !== cur.total) throw new Error(`Nedlastingen ble ufullstendig (${size} av ${cur.total} byte).`);
    const fd = fs.openSync(part, 'r'); const magic = Buffer.alloc(4);
    try { fs.readSync(fd, magic, 0, 4, 0); } finally { fs.closeSync(fd); }
    if (magic.toString('latin1') !== 'GGUF') { fs.rmSync(part, { force: true }); throw Object.assign(new Error('Filen er ikke en GGUF-modell (feil innhold fra serveren).'), { fatal: true }); }
    fs.renameSync(part, dest(e));
  }

  async function runOne(e) {
    current = { id: e.id, received: 0, total: e.sizeBytes, speed: 0, eta: null, state: 'downloading' };
    pushStatus();
    log.info(`download start: ${e.filename}`);
    let lastErr = null;
    for (let attempt = 0; attempt <= RETRIES; attempt++) {
      if (abort.signal.aborted) return 'cancelled';
      try {
        fs.mkdirSync(paths.modelsDir, { recursive: true });
        await fetchOnce(e, current);
        finalize(e, current);
        log.info(`download done: ${e.filename}`);
        return 'done';
      } catch (err) {
        if (abort.signal.aborted) return 'cancelled';
        lastErr = err;
        if (err.fatal) break;
        log.error(`download ${e.filename} failed (attempt ${attempt + 1}): ${err.message}`);
        current.state = 'retrying'; pushStatus();
        await wait(retryDelayMs * (attempt + 1), abort.signal);
        current.state = 'downloading';
      }
    }
    failed.set(e.id, lastErr?.message ?? 'unknown error');
    return 'failed';
  }

  async function pump() {
    if (running) return;
    running = true; abort = new AbortController();
    try {
      while (queue.length && !abort.signal.aborted) {
        const e = queue.shift();
        const result = await runOne(e);
        if (result === 'done') { done.add(e.id); try { onDone?.(e); } catch (err) { log.error(`onDone failed: ${err.message}`); } }
        current = null; pushStatus();
        if (result === 'cancelled') break;
      }
    } finally { running = false; current = null; queue = abort.signal.aborted ? [] : queue; pushStatus(); }
  }

  return {
    status,
    /** Queue catalog entries (ids). Already installed files are skipped. */
    start(ids) {
      for (const id of ids) {
        const e = entries.get(id);
        if (!e) throw new Error(`unknown model ${id}`);
        if (fs.existsSync(dest(e)) || queue.some((q) => q.id === id) || current?.id === id) continue;
        failed.delete(id); done.delete(id); queue.push(e);
      }
      pushStatus();
      pump(); // runs in the background
      return status();
    },
    cancel() { abort?.abort(); return status(); },
    /** Bytes already downloaded of an unfinished file (for "resume" labels). */
    partial(e) { try { return fs.statSync(`${dest(e)}.part`).size; } catch { return 0; } },
    entries: () => [...entries.values()],
    isBusy: () => running,
    /** Wait until the queue is empty (tests). */
    async idle() { while (running) await wait(20); },
  };
}
