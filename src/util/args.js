// Minimal argv parser: positionals, --flag, --flag value, --flag=value, -p value, short bool flags.

/**
 * @param {string[]} argv
 * @param {{ alias?: Record<string,string>, boolean?: string[] }} [opts]
 *   alias: short -> long name; boolean: long names that never take a value.
 * @returns {{ _: string[], flags: Record<string, string|boolean> }}
 */
export function parseArgs(argv, opts = {}) {
  const alias = opts.alias ?? {};
  const bools = new Set(opts.boolean ?? []);
  /** @type {string[]} */ const _ = [];
  /** @type {Record<string, string|boolean>} */ const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--') { _.push(...argv.slice(i + 1)); break; }
    let name = null;
    let value;
    if (a.startsWith('--') && a.length > 2) {
      const eq = a.indexOf('=');
      if (eq >= 0) { name = a.slice(2, eq); value = a.slice(eq + 1); } else name = a.slice(2);
    } else if (a.length === 2 && a[0] === '-' && /[A-Za-z]/.test(a[1])) {
      name = alias[a[1]] ?? a[1];
    } else { _.push(a); continue; }
    if (name.startsWith('no-')) { flags[name.slice(3)] = false; continue; }
    if (value === undefined) {
      if (bools.has(name)) value = true;
      else if (i + 1 < argv.length && !(argv[i + 1].startsWith('-') && argv[i + 1].length > 1 && isNaN(Number(argv[i + 1])))) value = argv[++i];
      else value = true;
    }
    flags[name] = value;
  }
  return { _, flags };
}

/** @param {any} v @param {number} dflt */
export function num(v, dflt) {
  if (v === undefined || v === true) return dflt;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`expected a number, got "${v}"`);
  return n;
}
