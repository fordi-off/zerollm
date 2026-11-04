#!/usr/bin/env node
// Launcher: starts the local server + engine and opens the app window.
//   node app/main.js [--data DIR] [--port N] [--no-open] [--keep-alive] [--quiet]
//
// The window is Edge in --app mode (a plain window without browser chrome; Edge ships with Windows) or, failing that,
// the default browser. When the page is closed the app exits after a few seconds (--keep-alive disables that).
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from '../src/util/args.js';
import { resolvePaths, ensureDirs, loadConfig, saveConfig, createLogger } from './config.js';
import { EngineHost } from './engine-host.js';
import { createApi } from './api.js';
import { createAppServer } from './server.js';

const here = path.dirname(fileURLToPath(import.meta.url));

/** Start everything. Returns handles (used by the tests and by main()). */
export async function startApp({ dataDir, port = 0, quiet = false, catalog, retryDelayMs } = {}) {
  const paths = resolvePaths(dataDir ?? process.cwd());
  ensureDirs(paths);
  const log = createLogger(paths, { quiet });
  let config = loadConfig(paths);
  saveConfig(paths, config);
  const getConfig = () => config;
  const setConfig = (c) => { config = c; saveConfig(paths, config); };
  const token = crypto.randomBytes(24).toString('hex');
  const host = new EngineHost({ paths, getConfig, log });
  const holder = {};
  const emit = (type, data) => holder.server?.emit(type, data);
  const api = createApi({ paths, host, getConfig, setConfig, log, emit, catalog, retryDelayMs });
  const app = createAppServer({ token, methods: api.methods, publicDir: path.join(here, 'public'), log });
  holder.server = app;
  const actualPort = await app.listen(port);
  const url = `http://127.0.0.1:${actualPort}/?t=${token}`;
  return {
    paths, log, token, port: actualPort, url, app, api, host,
    async close() { await api.shutdown(); await app.close(); },
  };
}

/** Browsers that can open an app-style window (no tabs / address bar), most preferred first. */
function appWindowBrowsers() {
  if (process.platform !== 'win32') return [];
  const roots = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean);
  const rel = ['Microsoft\\Edge\\Application\\msedge.exe', 'Google\\Chrome\\Application\\chrome.exe'];
  return rel.flatMap((r) => roots.map((root) => path.join(root, r))).filter((p) => fs.existsSync(p));
}

/** Open `url` in an app-style window (Edge/Chrome --app), else in the default browser. Returns how it was opened. */
export function openWindow(url) {
  const opts = { detached: true, stdio: 'ignore', windowsHide: true };
  try {
    const browser = appWindowBrowsers()[0];
    if (browser) { spawn(browser, [`--app=${url}`], opts).on('error', () => {}).unref(); return 'app-window'; }
    if (process.platform === 'win32') spawn('cmd', ['/c', 'start', '', url], opts).on('error', () => {}).unref();
    else if (process.platform === 'darwin') spawn('open', [url], opts).on('error', () => {}).unref();
    else spawn('xdg-open', [url], opts).on('error', () => {}).unref();
    return 'default-browser';
  } catch { return null; }
}

/** If another instance is alive in this data folder, return its URL. */
async function existingInstance(paths) {
  try {
    const lock = JSON.parse(fs.readFileSync(paths.lockPath, 'utf8'));
    process.kill(lock.pid, 0);
    const res = await fetch(`http://127.0.0.1:${lock.port}/api/ping`, { method: 'POST', headers: { 'X-Token': lock.token, 'Content-Type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(1500) });
    if (res.ok) return `http://127.0.0.1:${lock.port}/?t=${lock.token}`;
  } catch { /* no live instance */ }
  return null;
}

async function main() {
  const { flags } = parseArgs(process.argv.slice(2), { boolean: ['no-open', 'keep-alive', 'quiet'] });
  const dataDir = typeof flags.data === 'string' ? flags.data : process.cwd();
  const paths = resolvePaths(dataDir);
  const existing = await existingInstance(paths);
  if (existing) {
    console.log('zerollm app is already running - opening its window.');
    if (flags.open !== false) openWindow(existing);
    return;
  }
  const app = await startApp({ dataDir, port: flags.port ? Number(flags.port) : 0, quiet: flags.quiet === true });
  fs.writeFileSync(app.paths.lockPath, JSON.stringify({ pid: process.pid, port: app.port, token: app.token }), 'utf8');
  console.log(`\n  zerollm app: ${app.url}\n  data folder: ${app.paths.root}\n  (close the window to quit, or press Ctrl+C)\n`);
  if (flags.open !== false) openWindow(app.url);

  let closing = false;
  const quit = async (why) => {
    if (closing) return; closing = true;
    app.log.info(`shutting down (${why})`);
    try { fs.unlinkSync(app.paths.lockPath); } catch { /* already gone */ }
    await app.close();
    process.exit(0);
  };
  process.on('SIGINT', () => quit('SIGINT'));
  process.on('SIGTERM', () => quit('SIGTERM'));
  if (!flags['keep-alive']) {
    setInterval(() => {
      const s = app.app.stats;
      if (s.everConnected && app.app.clientCount() === 0 && Date.now() - s.lastDisconnect > 20000) quit('window closed');
    }, 2000);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
