// Talks to the local server: JSON calls + one Server-Sent-Events stream.
let token = null;
try { token = new URL(location.href).searchParams.get('t') || sessionStorage.getItem('zc-token'); } catch { /* storage blocked */ }
if (token) {
  try { sessionStorage.setItem('zc-token', token); } catch { /* ignore */ }
  history.replaceState(null, '', location.pathname); // keep the token out of the address bar / history
}

export const hasToken = () => !!token;

export async function call(name, params = {}) {
  const res = await fetch(`/api/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Token': token ?? '' }, body: JSON.stringify(params) });
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw Object.assign(new Error(body?.error ?? `HTTP ${res.status}`), { status: res.status });
  return body?.result;
}

/** Subscribe to server events: handlers = {chunk(data), done(data), ...}. Reconnects automatically. */
export function listen(handlers, { onConnection } = {}) {
  const es = new EventSource(`/api/events?t=${encodeURIComponent(token ?? '')}`);
  for (const [type, fn] of Object.entries(handlers)) es.addEventListener(type, (e) => fn(JSON.parse(e.data || 'null')));
  es.addEventListener('hello', () => onConnection?.(true));
  es.onerror = () => onConnection?.(false);
  return es;
}
