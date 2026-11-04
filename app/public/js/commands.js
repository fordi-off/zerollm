// Slash commands: the "/" autocomplete menu above the composer and extracting leading commands from a message.
import { call } from './api.js';
import { t } from './i18n.js';

let commands = []; let matches = []; let index = 0; let input = null; let menu = null; let onInsert = null;

export const getCommands = () => commands;
export async function loadCommands() { try { commands = await call('list_commands'); } catch { commands = []; } }

/** Leading "/cmd " words that name a known command are removed from the text and returned separately. */
export function extractCommands(raw) {
  const known = new Set(commands.map((c) => c.command));
  const found = []; let text = raw; let m;
  const re = /^\s*\/([A-Za-z0-9-]+)(\s+|$)/;
  while ((m = text.match(re))) {
    const name = m[1].toLowerCase();
    if (!known.has(name)) break;
    if (!found.includes(name)) found.push(name);
    text = text.slice(m[0].length);
  }
  return { text: text.trimStart(), commands: found };
}

const fragment = () => { const before = input.value.slice(0, input.selectionStart); const m = before.match(/(?:^|\s)\/([A-Za-z0-9-]*)$/); return m ? m[1].toLowerCase() : null; };
const isOpen = () => !menu.hidden && matches.length > 0;
function hide() { menu.hidden = true; matches = []; }

function update() {
  const frag = fragment();
  if (frag === null) { hide(); return; }
  matches = commands.filter((c) => c.command.startsWith(frag));
  if (frag && matches.length === 0) matches = commands.filter((c) => c.command.includes(frag));
  if (matches.length === 0) { hide(); return; }
  index = 0; paint(); menu.hidden = false;
}

function paint() {
  menu.textContent = '';
  matches.forEach((c, i) => {
    const row = document.createElement('div'); row.className = `cmd-row${i === index ? ' active' : ''}`; row.setAttribute('role', 'option');
    const n = document.createElement('span'); n.className = 'cmd-name'; n.textContent = `/${c.command}`;
    const d = document.createElement('span'); d.className = 'cmd-desc'; d.textContent = c.description || '';
    row.append(n, d);
    row.addEventListener('mousedown', (e) => { e.preventDefault(); choose(i); });
    menu.append(row);
  });
  menu.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
}

function choose(i) {
  const c = matches[i]; if (!c) return;
  const pos = input.selectionStart;
  const before = input.value.slice(0, pos).replace(/\/([A-Za-z0-9-]*)$/, `/${c.command} `);
  input.value = before + input.value.slice(pos);
  input.setSelectionRange(before.length, before.length);
  hide(); onInsert?.();
}

/** Open the menu with every command (the "/" button). */
export function openAll() {
  matches = commands.slice(); index = 0;
  if (matches.length === 0) { return false; }
  paint(); menu.hidden = false; return true;
}

export function initCommandMenu({ inputEl, menuEl, afterInsert }) {
  input = inputEl; menu = menuEl; onInsert = afterInsert;
  input.addEventListener('input', update);
  input.addEventListener('click', update);
  input.addEventListener('blur', () => setTimeout(hide, 120));
  input.addEventListener('keydown', (e) => {
    if (!isOpen()) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); index = (index + 1) % matches.length; paint(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); index = (index - 1 + matches.length) % matches.length; paint(); }
    else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); e.stopImmediatePropagation(); choose(index); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); hide(); }
  }, true);
  window.addEventListener('focus', loadCommands); // pick up skill files changed outside the app
}
