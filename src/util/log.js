// Tiny logging / formatting helpers. No global state.

/** @param {number} n bytes */
export function formatBytes(n) {
  if (!Number.isFinite(n)) return String(n);
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = Math.abs(n);
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  const s = i === 0 ? String(Math.round(v)) : v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2);
  return `${n < 0 ? '-' : ''}${s} ${units[i]}`;
}

/** @param {number} n */
export function formatCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** Print a "key: value" table with aligned keys. @param {Array<[string, any]>} rows */
export function printTable(rows, out = process.stdout) {
  const w = Math.max(0, ...rows.map(([k]) => k.length));
  for (const [k, v] of rows) out.write(`${k.padEnd(w)}  ${v}\n`);
}

export function section(title, out = process.stdout) {
  out.write(`\n== ${title} ==\n`);
}

/** stderr diagnostics */
export function warn(msg) { process.stderr.write(`warning: ${msg}\n`); }
export function die(msg, code = 1) { process.stderr.write(`error: ${msg}\n`); process.exit(code); }
