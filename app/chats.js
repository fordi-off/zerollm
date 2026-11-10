// Conversation files, in the same format Skibeti37 uses (chats/<id>.json) so both apps can share one data folder:
//   { id, name, model, messages: [{role:'system', content}, {role:'user'|'assistant', content, ...extras}], attached_docs, updated_at (seconds) }
// The system message is stored (Skibeti37 expects it first) but never shown; unknown fields are preserved on save.
// Extra fields written by zerollm for automatic titles (Skibeti37 ignores them): `auto_title` = 'pending' | 'done' | 'failed' | 'user'
// (absent = old / foreign chat: never titled) and `auto_placeholder` (the automatic name a title replaced).
// 'pending': the name is the automatic placeholder the page made from the first message; 'done': the name was written by the title job;
// 'user': the user chose the name (rename, or a save that carried a different name). A name the page sends that is only the stale
// placeholder never overwrites an automatic title; a real rename always wins.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { SYSTEM_PROMPT } from './chat-core.js';

const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const MAX_NAME = 120;

export function createChatStore(paths) {
  const summaryDir = path.join(paths.chatsDir, '_context_summaries');
  const file = (id) => {
    if (typeof id !== 'string' || !ID_RE.test(id)) throw new Error('invalid chat id');
    return path.join(paths.chatsDir, `${id}.json`);
  };
  const readRaw = (id) => {
    try { return JSON.parse(fs.readFileSync(file(id), 'utf8')); } catch (e) { if (e.message === 'invalid chat id') throw e; return null; }
  };
  const writeAtomic = (target, obj) => {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const tmp = `${target}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), 'utf8');
    fs.renameSync(tmp, target);
  };
  const cleanName = (n) => (typeof n === 'string' && n.trim() ? n.trim().slice(0, MAX_NAME) : 'Ny samtale');

  return {
    /** Newest first. Unreadable / foreign files are skipped. */
    list() {
      let names = [];
      try { names = fs.readdirSync(paths.chatsDir); } catch { return []; }
      const out = [];
      for (const f of names) {
        if (!f.endsWith('.json') || f.startsWith('_')) continue;
        const id = f.slice(0, -5);
        if (!ID_RE.test(id)) continue;
        const d = readRaw(id);
        if (!d || typeof d !== 'object' || d.id !== id) continue;
        out.push({ id, name: d.name || 'Ny samtale', model: d.model ?? null, updatedAt: Number(d.updated_at) || 0 });
      }
      return out.sort((a, b) => b.updatedAt - a.updatedAt);
    },

    /** Messages without the system message. */
    load(id) {
      const d = readRaw(id);
      if (!d || !Array.isArray(d.messages)) return null;
      return {
        id, name: d.name || 'Ny samtale', model: d.model ?? null, updatedAt: Number(d.updated_at) || 0,
        messages: d.messages.filter((m) => m && m.role !== 'system' && typeof m.content === 'string'),
      };
    },

    /**
     * Create (no id, or an unknown id) or overwrite a chat. Returns {id, name}: the stored name can differ from the one sent, because an
     * automatic title is kept when the page still sends the placeholder it made before the title arrived.
     */
    save({ id, name, model, messages }) {
      if (!Array.isArray(messages)) throw new Error('messages must be a list');
      const clean = [];
      for (const m of messages) {
        if (!m || (m.role !== 'user' && m.role !== 'assistant') || typeof m.content !== 'string') throw new Error('invalid message');
        clean.push(m);
      }
      const chatId = id ?? crypto.randomUUID();
      const existing = readRaw(chatId) ?? {};
      const sys = Array.isArray(existing.messages) && existing.messages[0]?.role === 'system' ? existing.messages[0] : { role: 'system', content: SYSTEM_PROMPT };
      const extra = {};
      let finalName = cleanName(name ?? existing.name);
      if (!existing.id) {
        // a chat the page creates with its first exchange (or just the first question) gets an automatic title later
        const users = clean.filter((m) => m.role === 'user').length;
        if (users === 1 && clean.length <= 2 && clean[0].role === 'user') extra.auto_title = 'pending';
      } else if (existing.auto_title === 'pending' || existing.auto_title === 'done') {
        const given = typeof name === 'string' && name.trim() ? cleanName(name) : null;
        if (given === null || given === existing.name || given === existing.auto_placeholder) finalName = existing.name || finalName; // unchanged, or the stale placeholder
        else extra.auto_title = 'user'; // the page really changed the name
      }
      writeAtomic(file(chatId), {
        ...existing, ...extra, id: chatId, name: finalName, model: model ?? existing.model ?? null,
        messages: [sys, ...clean], attached_docs: existing.attached_docs ?? {}, updated_at: Date.now() / 1000,
      });
      return { id: chatId, name: finalName };
    },

    /**
     * What the title job needs, or null when this chat is not waiting for a title (not 'pending', or no finished first exchange yet).
     * `exchanges` is 1 when the chat holds exactly one question and its reply; `history` is the conversation up to the reply.
     */
    autoTitleCandidate(id) {
      const d = readRaw(id);
      if (!d || d.auto_title !== 'pending' || !Array.isArray(d.messages)) return null;
      const msgs = d.messages.filter((m) => m && m.role !== 'system' && typeof m.content === 'string');
      const u = msgs[0]; const a = msgs[1];
      if (u?.role !== 'user' || a?.role !== 'assistant' || !u.content.trim() || !a.content.trim()) return null;
      return { id, name: d.name, model: d.model ?? null, userText: u.content, replyText: a.content, exchanges: msgs.length === 2 ? 1 : Math.floor(msgs.length / 2), history: [u] };
    },

    /**
     * Write an automatic title - only if the chat is still 'pending' (not renamed, not titled) and its first message is still the one the
     * title was made from. Read, check and write are one synchronous step, so a rename cannot slip in between. updated_at is NOT touched
     * (a title that arrives later must not move the chat to the top of the sidebar). Returns the stored name, or null when dropped.
     */
    applyAutoTitle(id, title, { userText } = {}) {
      const d = readRaw(id);
      if (!d || d.auto_title !== 'pending' || !Array.isArray(d.messages)) return null;
      const first = d.messages.find((m) => m && m.role === 'user');
      if (userText !== undefined && first?.content !== userText) return null;
      d.auto_placeholder = d.name; d.name = cleanName(title); d.auto_title = 'done';
      writeAtomic(file(id), d);
      return d.name;
    },

    /** Mark a still-'pending' chat 'failed' (the model produced nothing usable): it keeps its placeholder and is not tried again. */
    markAutoTitle(id, state) {
      const d = readRaw(id);
      if (!d || d.auto_title !== 'pending') return false;
      d.auto_title = state; writeAtomic(file(id), d); return true;
    },

    /** Rolling summary cache (Skibeti37 format): {summary, summarized_count}. */
    loadSummary(id) {
      file(id);
      try { const d = JSON.parse(fs.readFileSync(path.join(summaryDir, `${id}.json`), 'utf8')); return { summary: String(d.summary ?? ''), summarized_count: Number(d.summarized_count) || 0 }; } catch { return { summary: '', summarized_count: 0 }; }
    },
    saveSummary(id, summary, count) { file(id); writeAtomic(path.join(summaryDir, `${id}.json`), { summary, summarized_count: count }); },

    rename(id, name) {
      const d = readRaw(id); if (!d) return false;
      d.name = cleanName(name);
      if (d.auto_title) d.auto_title = 'user'; // the user's name wins over any automatic title, now and later
      writeAtomic(file(id), d); return true;
    },

    delete(id) {
      const p = file(id);
      try { fs.unlinkSync(p); } catch { /* already gone */ }
      try { fs.unlinkSync(path.join(summaryDir, `${id}.json`)); } catch { /* no summary */ }
      return true;
    },
  };
}
