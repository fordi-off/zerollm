// Conversation files, in the same format Skibeti37 uses (chats/<id>.json) so both apps can share one data folder:
//   { id, name, model, messages: [{role:'system', content}, {role:'user'|'assistant', content, ...extras}], attached_docs, updated_at (seconds) }
// The system message is stored (Skibeti37 expects it first) but never shown; unknown fields are preserved on save.
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

    /** Create (no id) or overwrite a chat. Returns the id. */
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
      writeAtomic(file(chatId), {
        ...existing, id: chatId, name: cleanName(name ?? existing.name), model: model ?? existing.model ?? null,
        messages: [sys, ...clean], attached_docs: existing.attached_docs ?? {}, updated_at: Date.now() / 1000,
      });
      return chatId;
    },

    rename(id, name) {
      const d = readRaw(id); if (!d) return false;
      d.name = cleanName(name); writeAtomic(file(id), d); return true;
    },

    delete(id) {
      const p = file(id);
      try { fs.unlinkSync(p); } catch { /* already gone */ }
      try { fs.unlinkSync(path.join(summaryDir, `${id}.json`)); } catch { /* no summary */ }
      return true;
    },
  };
}
