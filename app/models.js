// Discover the .gguf files in models/ and describe them (what the engine can run, how much RAM it needs).
import fs from 'node:fs';
import path from 'node:path';
import { inspectModel, estimateModelRam } from '../src/index.js';

const cache = new Map(); // "path|size|mtime" -> inspection result

/** Pretty label from a file name: "nb-llama-3.1-8b-q4_k_m.gguf" -> "Nb llama 3.1 8b q4 k m". */
export function humanizeFilename(f) {
  const stem = f.replace(/\.gguf$/i, '').replace(/[_-]+/g, ' ').trim();
  return stem ? stem[0].toUpperCase() + stem.slice(1) : f;
}

/** Embedding models cannot chat and are not offered (they are only used for document search, later). */
const isEmbeddingFile = (f) => /embed/i.test(f);

/**
 * @param {ReturnType<import('./config.js').resolvePaths>} paths
 * @param {object} cfg loaded config
 * @param {{ctx?: number, threads?: number}} [opts]
 */
export function listModels(paths, cfg, opts = {}) {
  let files = [];
  try { files = fs.readdirSync(paths.modelsDir).filter((f) => /\.gguf$/i.test(f) && !isEmbeddingFile(f)); } catch { return []; }
  const ctx = opts.ctx ?? cfg.context_window;
  const threads = opts.threads ?? cfg.n_threads;
  const out = [];
  for (const filename of files.sort()) {
    const full = path.join(paths.modelsDir, filename);
    let st;
    try { st = fs.statSync(full); } catch { continue; }
    const key = `${full}|${st.size}|${Math.round(st.mtimeMs)}`;
    let info = cache.get(key);
    if (!info) { info = inspectModel(full); cache.set(key, info); }
    const mcfg = cfg.models?.[filename] ?? {};
    const est = info.supported ? estimateModelRam(info, { ctx, threads, kv: 'f16' }) : null;
    out.push({
      filename,
      displayName: mcfg.display_name || humanizeFilename(filename),
      active: mcfg.active !== false,
      sizeBytes: st.size,
      supported: info.supported,
      reason: info.reason,
      arch: info.arch,
      params: info.params,
      quant: info.quant,
      ctxTrain: info.ctxTrain ?? null,
      ramNeed: est ? est.total : null,
    });
  }
  return out;
}

/** Safe absolute path of a model file by name (no directory components allowed). */
export function modelPath(paths, filename) {
  if (typeof filename !== 'string' || filename !== path.basename(filename) || !/\.gguf$/i.test(filename)) throw new Error('invalid model name');
  const full = path.join(paths.modelsDir, filename);
  if (!fs.existsSync(full)) throw new Error(`model file not found: ${filename}`);
  return full;
}
