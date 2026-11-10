// Sidebar: the saved conversations, grouped by date, with rename and delete.
import { call } from './api.js';
import { t } from './i18n.js';

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

let hooks = null; let activeId = null; let cache = [];
export function initChatList(h) { hooks = h; }
export function setActive(id) { activeId = id; paint(); }

function group(updatedAt) {
  const d = new Date(updatedAt * 1000); const now = new Date();
  const start = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((start(now) - start(d)) / 86400000);
  return days <= 0 ? 'today' : days === 1 ? 'yesterday' : 'earlier';
}

/** A chat got a new name from the server (automatic title): change it in the list without a reload. Does not disturb a rename in progress. */
export function setChatName(id, name) {
  const c = cache.find((x) => x.id === id);
  if (!c) { refreshChatList(); return; } // not in the list yet (the first save may still be on its way)
  c.name = name;
  if (!document.querySelector('.chat-rename')) paint();
}

export async function refreshChatList() {
  try { cache = await call('list_chats'); } catch { cache = []; }
  paint();
}

export function paint() {
  const box = document.getElementById('chat-list'); box.textContent = '';
  if (cache.length === 0) { box.append(el('div', 'chat-group', t('noChats'))); return; }
  let lastGroup = null;
  for (const c of cache) {
    const g = group(c.updatedAt);
    if (g !== lastGroup) { box.append(el('div', 'chat-group', t(g))); lastGroup = g; }
    box.append(row(c));
  }
}

function row(c) {
  const item = el('div', `chat-row${c.id === activeId ? ' active' : ''}`);
  const open = el('button', 'chat-item', c.name); open.type = 'button'; open.title = c.name;
  open.addEventListener('click', () => hooks.open(c.id));
  const ren = el('button', 'chat-act', '✎'); ren.type = 'button'; ren.setAttribute('aria-label', t('rename')); ren.title = t('rename');
  ren.addEventListener('click', (e) => { e.stopPropagation(); startRename(item, c); });
  const del = el('button', 'chat-act', '✕'); del.type = 'button'; del.setAttribute('aria-label', t('deleteChat')); del.title = t('deleteChat');
  let armed = null;
  del.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!armed) { del.textContent = t('confirmDelete'); del.classList.add('armed'); armed = setTimeout(() => { armed = null; del.textContent = '✕'; del.classList.remove('armed'); }, 3000); return; }
    clearTimeout(armed);
    await call('delete_chat', { id: c.id });
    hooks.deleted(c.id);
    await refreshChatList();
  });
  item.append(open, ren, del);
  return item;
}

function startRename(item, c) {
  const input = el('input', 'chat-rename'); input.type = 'text'; input.value = c.name; input.maxLength = 120;
  item.textContent = ''; item.append(input); input.focus(); input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const name = input.value.trim();
    if (save && name && name !== c.name) { await call('rename_chat', { id: c.id, name }); hooks.renamed(c.id, name); }
    await refreshChatList();
  };
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); input.blur(); } else if (e.key === 'Escape') { done = false; finish(false); } });
}
