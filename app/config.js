// Paths, settings and logging. The on-disk layout and the config.json keys match Skibeti37, so both apps can share one
// data folder (models/, chats/, documents/, skills/, config.json). New keys are additive.
import fs from 'node:fs';
import path from 'node:path';
import { defaultThreads } from '../src/index.js';

export const CONTEXT_STEPS = [2048, 4096, 8192, 16384, 32768];
export const DEFAULT_CONTEXT = 8192; // Skibeti37 uses 16384; 8192 keeps an 8B model comfortably inside 16 GB of RAM

/** @param {string} root the data folder */
export function resolvePaths(root) {
  const abs = path.resolve(root);
  return {
    root: abs,
    modelsDir: path.join(abs, 'models'),
    chatsDir: path.join(abs, 'chats'),
    docsDir: path.join(abs, 'documents'),
    skillsDir: path.join(abs, 'skills'),
    configPath: path.join(abs, 'config.json'),
    logPath: path.join(abs, 'app.log'),
    lockPath: path.join(abs, 'app.lock'),
    cacheDir: path.join(abs, 'cache'),
  };
}

export function defaultConfig() {
  return { context_window: DEFAULT_CONTEXT, n_threads: defaultThreads(), utility_model: null, doc_chunks: 4, last_model: null, language: 'nb', models: {} };
}

/** Read config.json (creating it with defaults when missing). Unknown keys are preserved. */
export function loadConfig(paths) {
  let cfg = {};
  try { cfg = JSON.parse(fs.readFileSync(paths.configPath, 'utf8')); } catch { cfg = {}; }
  const merged = { ...defaultConfig(), ...cfg };
  if (typeof merged.models !== 'object' || merged.models === null) merged.models = {};
  if (!CONTEXT_STEPS.includes(merged.context_window) && !(Number.isInteger(merged.context_window) && merged.context_window >= 512)) merged.context_window = DEFAULT_CONTEXT;
  return merged;
}

export function saveConfig(paths, cfg) {
  fs.mkdirSync(paths.root, { recursive: true });
  const tmp = `${paths.configPath}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2), 'utf8');
  fs.renameSync(tmp, paths.configPath);
}

/** Create the folders the app writes to. */
export function ensureDirs(paths) {
  for (const d of [paths.modelsDir, paths.chatsDir, paths.docsDir, paths.skillsDir]) fs.mkdirSync(d, { recursive: true });
}

/** Tiny file logger (also echoes to the console unless quiet). */
export function createLogger(paths, { quiet = false } = {}) {
  const write = (level, msg) => {
    const line = `${new Date().toISOString()} ${level} ${msg}`;
    try { fs.appendFileSync(paths.logPath, `${line}\n`); } catch { /* logging must never break the app */ }
    if (!quiet) (level === 'ERROR' ? console.error : console.log)(line);
  };
  return { info: (m) => write('INFO', m), error: (m) => write('ERROR', m) };
}
