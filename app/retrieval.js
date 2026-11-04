// Keyword search over document chunks (BM25) with a light Norwegian/English suffix stemmer. No embedding model needed.
// No stop-word list: BM25's inverse document frequency already makes words that occur in most chunks ("og", "er", "the") worthless,
// and for ANY language - a word that is in every chunk cannot rank one chunk above another.

// Longest suffix first; applied (up to twice, e.g. -ene-s) only when at least 3 letters remain.
const SUFFIXES = ['hetene', 'heten', 'ingene', 'ingen', 'ingar', 'ende', 'ene', 'ane', 'ere', 'est', 'het', 'ing', 'ers', 'ens', 'ets', 'ede', 'ed', 'er', 'en', 'et', 'ar', 'te', 'de', 'es', 'ly', 'a', 'e', 's'];

export function stem(w) {
  for (let pass = 0; pass < 2; pass++) {
    const s = SUFFIXES.find((x) => w.length - x.length >= 3 && w.endsWith(x));
    if (!s) break;
    w = w.slice(0, -s.length);
  }
  return w;
}

/** Lowercased word stems without stop words. */
export function tokenize(text) {
  const out = [];
  for (const m of String(text).toLowerCase().matchAll(/[\p{L}\p{N}]+/gu)) {
    const w = m[0];
    if (w.length < 2) continue;
    out.push(stem(w));
  }
  return out;
}

/** Split into chunks of about `size` words with `overlap` words shared between neighbours. */
export function chunkText(text, size = 180, overlap = 30) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let start = 0; start < words.length; start += size - overlap) {
    chunks.push(words.slice(start, start + size).join(' '));
    if (start + size >= words.length) break;
  }
  return chunks;
}

/**
 * Rank chunks for a query. @param {{text: string, filename: string}[]} chunks
 * @returns {{index: number, score: number}[]} best first; chunks with no matching term are left out
 */
export function bm25(chunks, query, { k1 = 1.5, b = 0.75 } = {}) {
  const q = [...new Set(tokenize(query))];
  if (q.length === 0 || chunks.length === 0) return [];
  const docs = chunks.map((c) => { const t = tokenize(c.text); const tf = new Map(); for (const w of t) tf.set(w, (tf.get(w) ?? 0) + 1); return { tf, len: t.length }; });
  const avg = docs.reduce((n, d) => n + d.len, 0) / docs.length || 1;
  const df = new Map(); for (const w of q) df.set(w, docs.reduce((n, d) => n + (d.tf.has(w) ? 1 : 0), 0));
  const N = docs.length; const scores = [];
  docs.forEach((d, index) => {
    let score = 0;
    for (const w of q) {
      const f = d.tf.get(w); if (!f) continue;
      const idf = Math.log(1 + (N - df.get(w) + 0.5) / (df.get(w) + 0.5));
      score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (d.len / avg))));
    }
    if (score > 0) scores.push({ index, score });
  });
  return scores.sort((a, b2) => b2.score - a.score);
}

/**
 * Choose the excerpts for a question: the best matches, in document order within the pick so they read naturally, limited by
 * count and by a token budget. A question that matches nothing gets NO excerpts (a greeting must not drag document text into the prompt).
 * @param {{text: string, filename: string}[]} chunks
 * @param {{k?: number, budgetTokens?: number, countTokens?: (text: string) => number}} o
 */
export function selectExcerpts(chunks, query, { k = 4, budgetTokens = 1200, countTokens = (t) => Math.ceil(t.length / 3.5), ranked: given = null } = {}) {
  const ranked = given ?? bm25(chunks, query);
  const picked = []; let used = 0;
  for (const r of ranked) {
    if (picked.length >= k) break;
    const cost = countTokens(chunks[r.index].text);
    if (picked.length > 0 && used + cost > budgetTokens) continue;
    picked.push(r); used += cost;
  }
  return picked.sort((a, b) => a.index - b.index);
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a)) || 1e-12;

/** Rank chunks by cosine similarity to a query vector; chunks without a (matching) embedding are skipped. @returns {{index: number, score: number}[]} */
export function vectorRank(chunks, queryVec) {
  const qn = norm(queryVec); const out = [];
  chunks.forEach((c, index) => { const e = c.embedding; if (Array.isArray(e) && e.length === queryVec.length) out.push({ index, score: dot(e, queryVec) / (norm(e) * qn) }); });
  return out.sort((a, b) => b.score - a.score);
}

/**
 * Reciprocal-rank fusion of several rankings (best first): robust against the scores being on different scales.
 * `scoreOf` optionally supplies the score to report per index (e.g. the cosine similarity).
 */
export function fuseRankings(lists, { k = 60, scoreOf = null } = {}) {
  const fused = new Map();
  for (const list of lists) list.forEach((r, rank) => fused.set(r.index, (fused.get(r.index) ?? 0) + 1 / (k + rank + 1)));
  return [...fused].map(([index, f]) => ({ index, score: scoreOf?.(index) ?? f, fused: f })).sort((a, b) => b.fused - a.fused);
}
