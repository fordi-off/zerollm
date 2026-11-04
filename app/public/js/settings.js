import { call } from './api.js';
import { t, apply } from './i18n.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);

let hooks = null;

export function setTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('zc-theme', theme); } catch { /* ignore */ }
}

export function initSettings(h) {
  hooks = h;
  try { const th = localStorage.getItem('zc-theme'); if (th) setTheme(th); } catch { /* ignore */ }
  for (const tab of document.querySelectorAll('.tabs [role="tab"]')) {
    tab.addEventListener('click', () => {
      for (const o of document.querySelectorAll('.tabs [role="tab"]')) { o.setAttribute('aria-selected', String(o === tab)); $(`tab-${o.dataset.tab}`).hidden = o !== tab; }
    });
  }
}

export async function openSettings() {
  await render();
  const dlg = $('settings');
  if (!dlg.open) dlg.showModal();
}

async function render() {
  const st = await call('state');
  apply($('settings'));
  renderModels(st); renderPerformance(st);
}

function renderModels(st) {
  const box = $('tab-models'); box.textContent = '';
  if (st.models.length === 0) { box.append(el('p', 'notice', `${t('noModelTitle')}. ${t('noModelText')}`)); return; }
  for (const m of st.models) {
    const row = el('div', 'model-row');
    const top = el('div', 'top');
    const name = el('input'); name.type = 'text'; name.value = m.displayName; name.setAttribute('aria-label', t('displayName')); name.maxLength = 80;
    name.addEventListener('change', async () => { await call('update_model_settings', { filename: m.filename, display_name: name.value }); await hooks.refreshState(); });
    const lab = el('label'); const cb = el('input'); cb.type = 'checkbox'; cb.checked = m.active; cb.disabled = !m.supported;
    cb.addEventListener('change', async () => { await call('update_model_settings', { filename: m.filename, active: cb.checked }); await hooks.refreshState(); });
    lab.append(cb, document.createTextNode(` ${t('active')}`));
    top.append(name, lab);
    const sub = el('div', 'sub', m.supported
      ? `${m.filename} · ${fmtBytes(m.sizeBytes)} · ${m.quant} · ${t('ramNeeded')} ${m.ramNeed ? fmtBytes(m.ramNeed) : '?'}`
      : `${m.filename} · ${fmtBytes(m.sizeBytes)}`);
    row.append(top, sub);
    if (!m.supported) row.append(el('div', 'sub badge-bad', `${t('unsupportedModel')}: ${m.reason ?? ''}`));
    box.append(row);
  }
  box.append(el('p', 'hint', `${t('freeRam')}: ${fmtBytes(st.system.freeRam)} / ${fmtBytes(st.system.totalRam)}`));
}

function row(label, hint, control) {
  const r = el('div', 'row'); const l = el('div'); l.append(el('div', 'lbl', label)); if (hint) l.append(el('div', 'hint', hint)); r.append(l, control); return r;
}

function renderPerformance(st) {
  const box = $('tab-performance'); box.textContent = '';
  const ctx = el('select'); ctx.setAttribute('aria-label', t('contextWindow'));
  for (const v of st.settings.contextSteps) { const o = el('option', '', v.toLocaleString()); o.value = v; o.selected = v === st.settings.contextWindow; ctx.append(o); }
  ctx.addEventListener('change', async () => { await call('set_context_window', { value: Number(ctx.value) }); await hooks.refreshState(); });
  const th = el('input'); th.type = 'number'; th.min = 1; th.max = st.settings.maxThreads; th.value = st.settings.threads; th.style.width = '80px'; th.setAttribute('aria-label', t('threads'));
  th.addEventListener('change', async () => { await call('set_thread_count', { value: Number(th.value) }); await hooks.refreshState(); });
  const lang = el('select'); lang.setAttribute('aria-label', t('language'));
  for (const [v, n] of [['nb', 'Norsk (bokmål)'], ['en', 'English']]) { const o = el('option', '', n); o.value = v; o.selected = v === st.language; lang.append(o); }
  lang.addEventListener('change', async () => { await hooks.applyLanguage(lang.value); await render(); });
  const theme = el('select'); theme.setAttribute('aria-label', t('theme'));
  let cur = 'auto'; try { cur = localStorage.getItem('zc-theme') || 'auto'; } catch { /* ignore */ }
  for (const [v, k] of [['auto', 'themeAuto'], ['light', 'themeLight'], ['dark', 'themeDark']]) { const o = el('option', '', t(k)); o.value = v; o.selected = v === cur; theme.append(o); }
  theme.addEventListener('change', () => setTheme(theme.value));
  const reset = el('button', 'btn', t('reset')); reset.type = 'button';
  reset.addEventListener('click', async () => { await call('reset_settings'); await hooks.refreshState(); await render(); });
  box.append(
    row(t('contextWindow'), t('contextHint'), ctx),
    row(t('threads'), `${t('threadsHint')} (max ${st.settings.maxThreads})`, th),
    row(t('language'), '', lang),
    row(t('theme'), '', theme),
    row(t('reset'), '', reset),
  );
}
