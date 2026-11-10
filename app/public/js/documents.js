// Attached files. The text of each file goes into the NEXT message whole (stored with the message, see app/chat-core.js): the model reads it
// once and keeps it in memory for every follow-up question. Attach with the paperclip or by dragging a text file into the window.
import { t } from './i18n.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|log|html?|xml|ya?ml|srt|rst|tex|js|mjs|ts|py|java|c|h|cpp|cs|go|rs|sql|sh|ini|toml)$/i;
const MAX_CHARS = 2_000_000; // hard cap in the page; whether it FITS the context window is checked against the real token count before sending

let hooks = null;
/** @type {{filename: string, text: string}[]} files waiting for the next message */
let pending = [];

export const getPending = () => pending.map((p) => ({ ...p }));
export function takePending() { const p = pending; pending = []; refreshChips(); return p; }
export function clearPending() { pending = []; refreshChips(); }

export function initDocuments(h) {
  hooks = h; // {toast, onChange}
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
  for (const f of files) {
    if (!TEXT_EXT.test(f.name) && !(f.type || '').startsWith('text/')) { hooks.toast(`${t('unsupportedFile')}${f.name}`); continue; }
    try {
      const text = await f.text();
      if (text.includes('\u0000')) { hooks.toast(`${t('unsupportedFile')}${f.name}`); continue; }
      if (text.length > MAX_CHARS) { hooks.toast(t('docTooBigPage', { name: f.name })); continue; }
      if (!text.trim()) { hooks.toast(`${f.name}: ${t('docEmpty')}`); continue; }
      pending = pending.filter((p) => p.filename !== f.name); pending.push({ filename: f.name, text });
      hooks.toast(t('docAttached', { name: f.name }));
    } catch (e) { hooks.toast(e.message); }
  }
  refreshChips();
}

export function refreshChips() {
  const box = $('doc-chips');
  box.textContent = ''; box.hidden = pending.length === 0;
  for (const d of pending) {
    const chip = el('span', 'doc-chip');
    const nm = el('span', 'nm', `${d.filename} · ${Math.max(1, Math.round(d.text.length / 1000))} k`);
    const x = el('button', 'x', '✕'); x.type = 'button'; x.title = t('detachDoc'); x.setAttribute('aria-label', t('detachDoc'));
    x.addEventListener('click', () => { pending = pending.filter((p) => p !== d); refreshChips(); });
    chip.append(nm, x); box.append(chip);
  }
  hooks?.onChange?.();
}
