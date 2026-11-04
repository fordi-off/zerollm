// The embedding ("search") model: nomic-embed-text-v1.5 in models/ (any file zerollm's embedding encoder supports).
// Document chunks are embedded once when a document is added (a temporary pool of workers); questions are embedded on demand
// with one resident worker that is released after a few idle minutes.
import fs from 'node:fs';
import path from 'node:path';
import { readGGUF } from '../src/gguf/reader.js';
import { loadEmbedder } from '../src/embed/embedder.js';
import { isEmbedModel } from '../src/embed/config.js';

const IDLE_MS = 10 * 60 * 1000;
/** Skibeti37 embeds raw chunk text (no "search_document: " / "search_query: " prefixes) - same here, so vectors are interchangeable. */
export const DOC_PREFIX = ''; export const QUERY_PREFIX = '';

export function createEmbeddings({ paths, getConfig, log }) {
  const seen = new Map(); // path|size|mtime -> boolean
  let resident = null; let residentPromise = null; let idleTimer = null;

  /** Filename of the installed embedding model (models/ file that the embedding encoder can run), or null. */
  function modelFile() {
    let names = []; try { names = fs.readdirSync(paths.modelsDir); } catch { return null; }
    for (const f of names.filter((n) => /\.gguf$/i.test(n) && /embed/i.test(n)).sort()) {
      const full = path.join(paths.modelsDir, f);
      let st; try { st = fs.statSync(full); } catch { continue; }
      const key = `${full}|${st.size}|${Math.round(st.mtimeMs)}`;
      if (!seen.has(key)) { let ok = false; try { ok = isEmbedModel(readGGUF(full)); } catch { ok = false; } seen.set(key, ok); }
      if (seen.get(key)) return f;
    }
    return null;
  }
  const available = () => modelFile() !== null;

  async function closeResident() {
    clearTimeout(idleTimer); idleTimer = null;
    const e = resident; resident = null; residentPromise = null;
    if (e) { await e.close(); log.info('embedding model released'); }
  }
  async function getResident() {
    const file = modelFile(); if (!file) return null;
    if (!residentPromise) {
      residentPromise = loadEmbedder(path.join(paths.modelsDir, file), { threads: 1 }).then((e) => { resident = e; log.info(`embedding model ready: ${file} (${e.kernels}, ${(e.loadMs / 1000).toFixed(1)} s)`); return e; }).catch((err) => { residentPromise = null; throw err; });
    }
    const e = await residentPromise;
    clearTimeout(idleTimer); idleTimer = setTimeout(() => { closeResident().catch(() => {}); }, IDLE_MS); idleTimer.unref?.();
    return e;
  }

  return {
    modelFile, available,
    /** Vector of a question (mean-pooled, not normalised), or null when there is no embedding model. */
    async embedQuery(text) {
      try { const e = await getResident(); if (!e) return null; return Array.from((await e.embed([QUERY_PREFIX + text]))[0]); }
      catch (err) { log.error(`embedding the question failed, using keyword search only: ${err.message}`); return null; }
    },
    /** Embed many chunk texts with a temporary pool; frees its memory afterwards. */
    async embedTexts(texts, { onProgress, signal, threads: wanted } = {}) {
      const file = modelFile(); if (!file) throw new Error('Ingen søkemodell (embedding) er installert.');
      await closeResident(); // one pool at a time
      const threads = Math.max(1, wanted ?? Math.min(getConfig().n_threads ?? 2, 4));
      const e = await loadEmbedder(path.join(paths.modelsDir, file), { threads });
      try {
        const out = []; const BATCH = threads * 4;
        for (let i = 0; i < texts.length; i += BATCH) {
          if (signal?.aborted) return null;
          const part = await e.embed(texts.slice(i, i + BATCH).map((t) => DOC_PREFIX + t));
          for (const v of part) out.push(Array.from(v));
          onProgress?.(Math.min(texts.length, i + BATCH), texts.length);
        }
        return out;
      } finally { await e.close(); }
    },
    closeResident,
  };
}
