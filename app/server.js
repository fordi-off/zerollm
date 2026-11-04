// Zero-dependency local web server: static files + JSON API + one Server-Sent-Events channel.
//
// A local server is reachable by every web page open in the same browser, so it defends itself:
//  - bound to 127.0.0.1 only;
//  - every request's Host header must be exactly this server (blocks DNS rebinding);
//  - every API call needs the per-launch random token (header X-Token, or ?t= for the event stream);
//  - a browser-supplied Origin must be this server (blocks cross-site requests);
//  - static files come only from the public folder (no traversal, no dotfiles).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.txt': 'text/plain; charset=utf-8', '.map': 'application/json',
};
const MAX_BODY = 25 * 1024 * 1024;
const CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'";

const safeEqual = (a, b) => {
  const x = Buffer.from(String(a ?? '')); const y = Buffer.from(String(b ?? ''));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
};

/**
 * @param {{token: string, methods: Record<string, (params: any) => Promise<any>>, publicDir: string, log: {info: Function, error: Function}}} o
 */
export function createAppServer({ token, methods, publicDir, log }) {
  let port = 0;
  const clients = new Set();
  let lastActivity = Date.now();
  const stats = { everConnected: false, lastDisconnect: 0 };

  const hostOk = (req) => { const h = req.headers.host; return h === `127.0.0.1:${port}` || h === `localhost:${port}`; };
  const originOk = (req) => { const o = req.headers.origin; return !o || o === `http://127.0.0.1:${port}` || o === `http://localhost:${port}`; };

  const send = (res, status, body, headers = {}) => {
    res.writeHead(status, { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', ...headers });
    res.end(body);
  };
  const sendJson = (res, status, obj) => send(res, status, JSON.stringify(obj), { 'Content-Type': 'application/json; charset=utf-8' });

  function readBody(req) {
    return new Promise((resolve, reject) => {
      const chunks = []; let size = 0;
      req.on('data', (c) => { size += c.length; if (size > MAX_BODY) { reject(Object.assign(new Error('request too large'), { status: 413 })); req.destroy(); } else chunks.push(c); });
      req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
      req.on('error', reject);
    });
  }

  function serveStatic(req, res, pathname) {
    let rel = decodeURIComponent(pathname);
    if (rel === '/' || rel === '') rel = '/index.html';
    const full = path.join(publicDir, path.normalize(rel));
    const base = path.resolve(publicDir) + path.sep;
    if (!path.resolve(full).startsWith(base) || rel.split('/').some((seg) => seg.startsWith('.'))) return send(res, 404, 'not found');
    fs.stat(full, (err, st) => {
      if (err || !st.isFile()) return send(res, 404, 'not found');
      const ext = path.extname(full).toLowerCase();
      res.writeHead(200, { 'Content-Type': MIME[ext] ?? 'application/octet-stream', 'Content-Length': st.size, 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': CSP, 'Referrer-Policy': 'no-referrer' });
      if (req.method === 'HEAD') return res.end();
      fs.createReadStream(full).pipe(res);
    });
  }

  function handleEvents(req, res, url) {
    if (!safeEqual(url.searchParams.get('t'), token)) return sendJson(res, 401, { error: 'unauthorized' });
    res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store', Connection: 'keep-alive', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff' });
    res.write('retry: 1500\n\n');
    clients.add(res); stats.everConnected = true;
    res.on('close', () => { clients.delete(res); stats.lastDisconnect = Date.now(); });
    res.write(`event: hello\ndata: {}\n\n`);
  }

  async function handleApi(req, res, url) {
    const name = url.pathname.slice('/api/'.length);
    if (!safeEqual(req.headers['x-token'], token)) return sendJson(res, 401, { error: 'unauthorized' });
    if (req.method !== 'POST') return sendJson(res, 405, { error: 'POST only' });
    if (!Object.hasOwn(methods, name)) return sendJson(res, 404, { error: `unknown method ${name}` });
    let params = {};
    try {
      const raw = await readBody(req);
      if (raw) params = JSON.parse(raw);
      if (params === null || typeof params !== 'object' || Array.isArray(params)) throw new Error('body must be a JSON object');
    } catch (e) { return sendJson(res, e.status ?? 400, { error: `bad request: ${e.message}` }); }
    try {
      const result = await methods[name](params);
      sendJson(res, 200, { result: result ?? null });
    } catch (e) {
      if (e.message !== 'busy') log.error(`api ${name} failed: ${e.stack ?? e}`);
      sendJson(res, 500, { error: String(e.message ?? e) });
    }
  }

  const server = http.createServer((req, res) => {
    lastActivity = Date.now();
    try {
      if (!hostOk(req)) return send(res, 403, 'forbidden host');
      if (!originOk(req)) return send(res, 403, 'forbidden origin');
      const url = new URL(req.url, `http://127.0.0.1:${port}`);
      if (url.pathname === '/api/events' && req.method === 'GET') return handleEvents(req, res, url);
      if (url.pathname.startsWith('/api/')) { handleApi(req, res, url).catch((e) => { log.error(`server error: ${e.stack ?? e}`); if (!res.headersSent) sendJson(res, 500, { error: 'internal error' }); }); return; }
      if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'method not allowed');
      serveStatic(req, res, url.pathname);
    } catch (e) {
      log.error(`server error: ${e.stack ?? e}`);
      if (!res.headersSent) send(res, 500, 'internal error');
    }
  });

  const keepAlive = setInterval(() => { for (const c of clients) c.write(': keep-alive\n\n'); }, 15000);
  keepAlive.unref();

  return {
    server,
    /** Push an event to every open page. */
    emit(type, data) {
      const payload = `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
      for (const c of clients) c.write(payload);
    },
    clientCount: () => clients.size,
    stats,
    lastActivity: () => lastActivity,
    /** @returns {Promise<number>} the port */
    listen(wantedPort = 0) {
      return new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(wantedPort, '127.0.0.1', () => { port = server.address().port; resolve(port); });
      });
    },
    close() {
      clearInterval(keepAlive);
      for (const c of clients) c.end();
      return new Promise((resolve) => { server.close(() => resolve()); server.closeAllConnections?.(); });
    },
  };
}
