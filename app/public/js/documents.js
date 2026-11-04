// Documents in the open conversation: attach by button or drag-and-drop, chips with on/off and remove, sources under replies.
import { call } from './api.js';
import { t } from './i18n.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|log|html?|xml|ya?ml|srt|rst|tex)$/i;
const MAX_BYTES = 20 * 1024 * 1024;

let hooks = null;
export function initDocuments(h) {
  hooks = h; // {getChatId, ensureChat, toast}
  $('btn-attach').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', async (e) => { await addFiles([...e.target.files]); e.target.value = ''; });
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
  window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; $('drop-hint').hidden = false; } });
  window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; $('drop-hint').hidden = true; } });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', async (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth = 0; $('drop-hint').hidden = true; await addFiles([...e.dataTransfer.files]); });
}

export async function addFiles(files) {
  if (hooks.isLocked?.()) { hooks.toast(t('indexingBlocked')); return; }
  for (const f of files) {
    if (!TEXT_EXT.test(f.name) && !(f.type || '').startsWith('text/')) { hooks.toast(`${t('unsupportedFile')}${f.name}`); continue; }
    if (f.size > MAX_BYTES) { hooks.toast(`${f.name}: >20 MB`); continue; }
    try {
      const chatId = await hooks.ensureChat();
      const r = await call('add_document', { chatId, filename: f.name, content: await f.text() });
      hooks.toast(`${t('docAdded', { name: f.name })} (${t('docParts', { n: r.chunks })})${r.notice ? ` ${r.notice}` : ''}`);
      await refreshChips();
    } catch (e) { hooks.toast(e.message); }
  }
}

export async function refreshChips() {
  const box = $('doc-chips'); const chatId = hooks.getChatId();
  let docs = []; if (chatId) { try { docs = await call('list_documents', { chatId }); } catch { docs = []; } }
  box.textContent = ''; box.hidden = docs.length === 0;
  for (const d of docs) {
    const chip = el('span', `doc-chip${d.active ? '' : ' off'}`);
    const nm = el('button', 'nm', d.filename); nm.type = 'button'; nm.title = t('toggleDoc');
    nm.addEventListener('click', async () => { await call('toggle_document', { chatId, docId: d.id }); await refreshChips(); hooks.onChange?.(); });
    const x = el('button', 'x', '✕'); x.type = 'button'; x.title = t('detachDoc'); x.setAttribute('aria-label', t('detachDoc'));
    x.addEventListener('click', async () => { await call('detach_document', { chatId, docId: d.id }); await refreshChips(); hooks.onChange?.(); });
    chip.append(nm, x); box.append(chip);
  }
  hooks.onChange?.();
}

/** "Kilder · n utdrag" under a reply. */
export function renderSources(msgNode, sources) {
  msgNode.querySelector('.doc-sources')?.remove();
  if (!sources?.length) return;
  const d = el('details', 'doc-sources'); d.append(el('summary', '', t('sourcesCount', { n: sources.length })));
  for (const s of sources) {
    const r = el('div', 'doc-source'); const b = el('b', '', s.filename); r.append(b);
    if (typeof s.score === 'number' && s.score > 0) r.append(el('span', 'sc', s.score.toFixed(2)));
    r.append(el('div', '', s.snippet)); d.append(r);
  }
  msgNode.append(d);
}
