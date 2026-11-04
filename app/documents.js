// The shared document library: documents/<id>.json = { id, filename, summary, chunks: [{text, embedding}] } (Skibeti37's format).
// Chats link to documents through their `attached_docs` map {docId: active}. Search is BM25 over the chunk texts, so
// documents written by Skibeti37 (with embeddings) work here too; documents written here have empty embeddings.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { chunkText, selectExcerpts, bm25, vectorRank, fuseRankings } from './retrieval.js';

const MAX_CHUNKS = 800;          // a pathologically large file must not stall indexing
const MAX_CHARS = 20_000_000;
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|log|html?|xml|yml|yaml|srt|rst|tex)$/i;

export function createDocuments(paths, chatStore) {
  const dir = paths.docsDir;
  const file = (id) => { if (typeof id !== 'string' || !ID_RE.test(id)) throw new Error('invalid document id'); return path.join(dir, `${id}.json`); };
  const read = (id) => { try { return JSON.parse(fs.readFileSync(file(id), 'utf8')); } catch (e) { if (e.message === 'invalid document id') throw e; return null; } };
  const write = (id, obj) => { fs.mkdirSync(dir, { recursive: true }); const tmp = `${file(id)}.${process.pid}.tmp`; fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), 'utf8'); fs.renameSync(tmp, file(id)); };
  const embeddedCount = (d) => (Array.isArray(d.chunks) ? d.chunks.filter((c) => Array.isArray(c.embedding) && c.embedding.length > 0).length : 0);
  const meta = (d) => ({ id: d.id, filename: d.filename, summary: d.summary ?? '', chunks: Array.isArray(d.chunks) ? d.chunks.length : 0, embedded: embeddedCount(d) });
  const allIds = () => { try { return fs.readdirSync(dir).filter((f) => f.endsWith('.json') && ID_RE.test(f.slice(0, -5))).map((f) => f.slice(0, -5)); } catch { return []; } };

  return {
    isTextName: (name) => TEXT_EXT.test(String(name)),
    /** Every document in the library. */
    library() { return allIds().map(read).filter((d) => d && d.id).map(meta).sort((a, b) => a.filename.localeCompare(b.filename)); },
    /** Documents attached to a chat, with their active flag. */
    forChat(chatId) {
      if (!chatId) return [];
      return Object.entries(chatStore.attachedDocs(chatId)).map(([id, active]) => { const d = read(id); return d ? { ...meta(d), active: !!active } : null; }).filter(Boolean);
    },
    available(chatId) { const attached = new Set(Object.keys(chatId ? chatStore.attachedDocs(chatId) : {})); return this.library().filter((d) => !attached.has(d.id)); },

    /** Index a text and attach it to the chat. */
    add(chatId, filename, content) {
      if (!chatId) throw new Error('Ingen samtale å knytte dokumentet til');
      const name = String(filename ?? '').replace(/[\\/]/g, '_').slice(0, 200) || 'dokument.txt';
      if (typeof content !== 'string') throw new Error('content must be text');
      if (content.length > MAX_CHARS) throw new Error('Filen er for stor (over 20 millioner tegn).');
      if (content.includes('\u0000')) throw new Error('Filen ser ikke ut som tekst. Støtter .txt og .md.');
      let chunks = chunkText(content);
      if (chunks.length === 0) throw new Error('Filen er tom.');
      const truncated = chunks.length > MAX_CHUNKS; if (truncated) chunks = chunks.slice(0, MAX_CHUNKS);
      const id = crypto.randomUUID();
      write(id, { id, filename: name, summary: '', chunks: chunks.map((text) => ({ text, embedding: [] })) });
      if (!chatStore.editAttached(chatId, (m) => { m[id] = true; })) { fs.unlinkSync(file(id)); throw new Error('Samtalen finnes ikke (lagre den først).'); }
      return { id, chunks: chunks.length, notice: truncated ? `«${name}» er stort - bare de første ${MAX_CHUNKS} delene ble indeksert.` : null };
    },
    attach(chatId, docId) { file(docId); return chatStore.editAttached(chatId, (m) => { m[docId] = true; }); },
    detach(chatId, docId) { return chatStore.editAttached(chatId, (m) => { delete m[docId]; }); },
    toggle(chatId, docId) { let v = true; chatStore.editAttached(chatId, (m) => { v = !(m[docId] ?? true); m[docId] = v; }); return v; },
    /** Remove a document from the library and from every chat. */
    remove(docId) {
      const p = file(docId); try { fs.unlinkSync(p); } catch { /* gone */ }
      for (const id of chatStore.ids()) chatStore.editAttached(id, (m) => { delete m[docId]; });
      return true;
    },
    preview(docId) {
      const d = read(docId); if (!d) return null;
      return { id: d.id, filename: d.filename, summary: d.summary ?? '', fullText: (d.chunks ?? []).map((c) => c.text).join('\n\n') };
    },
    setSummary(docId, summary) { const d = read(docId); if (!d) return false; d.summary = summary; write(docId, d); return true; },
    /** Text sections for summarizing (non-overlapping 2000-word parts, at most `max`). */
    textParts(docId, words = 2000, max = 8) {
      const d = read(docId); if (!d) return null;
      const all = (d.chunks ?? []).map((c) => c.text).join(' ').split(/\s+/);
      const parts = []; for (let i = 0; i < all.length && parts.length < max; i += words) parts.push(all.slice(i, i + words).join(' '));
      return parts;
    },

    /** Chunk texts of a document (for embedding). */
    chunkTexts(docId) { const d = read(docId); return d ? (d.chunks ?? []).map((c) => c.text) : null; },
    /** Store one embedding per chunk (same order as chunkTexts). */
    setEmbeddings(docId, vectors) {
      const d = read(docId); if (!d || !Array.isArray(d.chunks) || d.chunks.length !== vectors.length) return false;
      d.chunks.forEach((c, i) => { c.embedding = vectors[i]; }); write(docId, d); return true;
    },
    /** Ids of library documents that still lack embeddings. */
    needingEmbeddings() { return allIds().map(read).filter((d) => d && Array.isArray(d.chunks) && d.chunks.length && embeddedCount(d) < d.chunks.length).map((d) => d.id); },
    /** Do the active documents of this chat have vectors (of this size)? */
    hasEmbeddings(chatId) {
      for (const [docId, active] of Object.entries(chatId ? chatStore.attachedDocs(chatId) : {})) { if (!active) continue; const d = read(docId); if (d && embeddedCount(d) > 0) return true; }
      return false;
    },

    /**
     * Excerpts of the active documents of a chat for a question: keyword (BM25) ranking, fused with the cosine ranking when a
     * query vector is given and chunks have embeddings of the same size. -> [{filename, text, score}]
     */
    retrieve(chatId, query, opts = {}) {
      const { queryVec = null, ...rest } = opts;
      const chunks = [];
      for (const [docId, active] of Object.entries(chatId ? chatStore.attachedDocs(chatId) : {})) {
        if (!active) continue;
        const d = read(docId); if (!d) continue;
        for (const c of d.chunks ?? []) if (typeof c.text === 'string') chunks.push({ text: c.text, filename: d.filename, embedding: c.embedding });
      }
      if (chunks.length === 0) return [];
      let ranked = null;
      if (queryVec) {
        const vr = vectorRank(chunks, queryVec);
        if (vr.length) { const cos = new Map(vr.map((r) => [r.index, r.score])); ranked = fuseRankings([bm25(chunks, query), vr], { scoreOf: (i) => cos.get(i) ?? 0 }); }
      }
      return selectExcerpts(chunks, query, { ...rest, ranked }).map((r) => ({ filename: chunks[r.index].filename, text: chunks[r.index].text, score: Math.round(r.score * 1000) / 1000 }));
    },
  };
}
