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

export async function openSettings(tab) {
  await render();
  if (tab) document.querySelector(`.tabs [data-tab="${tab}"]`)?.click();
  const dlg = $('settings');
  if (!dlg.open) dlg.showModal();
}

async function render() {
  const st = await call('state');
  apply($('settings'));
  renderModels(st); renderPerformance(st); await renderSkills(); await renderDiagnostics();
}

const fmtSpeed = (b) => `${(b / 1e6).toFixed(1)} MB/s`;
const fmtEta = (s) => (s == null || !isFinite(s) ? '' : s > 3600 ? `${Math.round(s / 3600)} t` : s > 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);
let lastCatalog = null;

function renderModels(st) {
  const box = $('tab-models'); box.textContent = '';
  const installed = el('div'); box.append(installed);
  if (st.models.length === 0) installed.append(el('p', 'notice', `${t('noModelTitle')}. ${t('noModelText')}`));
  for (const m of st.models) {
    const r = el('div', 'model-row');
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
    r.append(top, sub);
    if (!m.supported) r.append(el('div', 'sub badge-bad', `${t('unsupportedModel')}: ${m.reason ?? ''}`));
    const tools = el('div', 'model-tools');
    if (m.supported) {
      const kv = el('select'); kv.title = t('kvHint'); kv.setAttribute('aria-label', t('kvPrecision'));
      for (const v of ['f16', 'f32']) { const o = el('option', '', `${t('kvPrecision')} ${v}`); o.value = v; o.selected = v === m.kv; kv.append(o); }
      kv.addEventListener('change', async () => { await call('update_model_settings', { filename: m.filename, kv: kv.value }); await hooks.refreshState(); await render(); });
      tools.append(kv);
    }
    const del = el('button', 'btn', t('deleteModel')); del.type = 'button';
    let armed = null;
    del.addEventListener('click', async () => {
      if (!armed) { del.textContent = t('confirmDeleteModel'); del.classList.add('danger'); armed = setTimeout(() => { armed = null; del.textContent = t('deleteModel'); del.classList.remove('danger'); }, 3000); return; }
      clearTimeout(armed); await call('delete_model', { filename: m.filename }); await hooks.refreshState(); await render();
    });
    tools.append(del);
    r.append(tools);
    installed.append(r);
  }
  box.append(el('p', 'hint', `${t('freeRam')}: ${fmtBytes(st.system.freeRam)} / ${fmtBytes(st.system.totalRam)}`));
  const dl = el('div', 'downloads'); dl.id = 'dl-box'; box.append(dl);
  call('catalog').then((c) => { lastCatalog = c; renderDownloads(c); });
}

/** (Re)draw the download list from a catalog answer; `status` events only change the progress parts. */
function renderDownloads(cat) {
  const box = $('dl-box'); if (!box) return;
  box.textContent = '';
  box.append(el('h3', '', t('downloads')), el('p', 'hint', `${t('downloadsHint')} ${cat.freeDisk != null ? `${t('freeDisk')}: ${fmtBytes(cat.freeDisk)}.` : ''}`));
  const stat = cat.status;
  for (const e of cat.entries) {
    const r = el('div', 'model-row');
    const top = el('div', 'top');
    const nm = el('div', 'dl-name'); nm.append(el('strong', '', e.displayName));
    if (e.recommended) nm.append(el('span', 'tag', t('recommended')));
    top.append(nm);
    const isCur = stat.current?.id === e.id; const queued = stat.queue.includes(e.id);
    const act = el('button', 'btn', ''); act.type = 'button';
    if (e.installed) { act.textContent = t('installed'); act.disabled = true; }
    else if (isCur || queued) { act.textContent = t('cancelDownload'); act.addEventListener('click', () => call('download_cancel')); }
    else { act.textContent = e.partialBytes > 0 ? t('resume') : t('download'); act.classList.add('primary'); act.disabled = !e.fits; act.addEventListener('click', () => call('download_start', { ids: [e.id] })); }
    top.append(act);
    r.append(top);
    r.append(el('div', 'sub', `${fmtBytes(e.sizeBytes)} · ${t('ramNeeded')} ~${e.ramGB} GB${e.fits ? '' : ` · ${t('tooBig')}`}`));
    if (stat.failed[e.id]) r.append(el('div', 'sub badge-bad', `${t('downloadFailed')}: ${stat.failed[e.id]}`));
    if (isCur) {
      const c = stat.current; const bar = el('div', 'progress-bar wide'); const fill = el('div'); fill.id = 'dl-fill'; fill.style.width = `${c.total ? Math.round((c.received / c.total) * 100) : 0}%`; bar.append(fill);
      const info = el('div', 'sub'); info.id = 'dl-info'; info.textContent = progressText(c);
      r.append(bar, info);
    } else if (queued) r.append(el('div', 'sub', t('queued')));
    else if (!e.installed && e.partialBytes > 0) r.append(el('div', 'sub', `${fmtBytes(e.partialBytes)} / ${fmtBytes(e.sizeBytes)}`));
    box.append(r);
  }
}
const progressText = (c) => `${c.state === 'retrying' ? t('retrying') : fmtBytes(c.received)} / ${fmtBytes(c.total)}${c.speed ? ` · ${fmtSpeed(c.speed)}` : ''}${c.eta ? ` · ${t('remaining', { n: fmtEta(c.eta) })}` : ''}`;

/** Called for every `download` server event. Returns true when a download just finished (the model list changed). */
export function onDownloadEvent(status) {
  if (!lastCatalog) return false;
  const before = lastCatalog.status;
  const finished = status.completed.length > before.completed.length;
  lastCatalog.status = status;
  if (finished && $('settings').open) render();
  const structural = finished || status.current?.id !== before.current?.id || status.queue.length !== before.queue.length || JSON.stringify(status.failed) !== JSON.stringify(before.failed);
  if (structural) call('catalog').then((c) => { lastCatalog = c; renderDownloads(c); });
  else if (status.current) {
    const fill = document.getElementById('dl-fill'); const info = document.getElementById('dl-info');
    if (fill && status.current.total) fill.style.width = `${Math.round((status.current.received / status.current.total) * 100)}%`;
    if (info) info.textContent = progressText(status.current);
  }
  return finished;
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
  const util = el('select'); util.setAttribute('aria-label', t('utilityModel'));
  { const o = el('option', '', t('utilityAuto')); o.value = ''; util.append(o); }
  for (const m of st.models.filter((x) => x.supported)) { const o = el('option', '', m.displayName); o.value = m.filename; o.selected = m.filename === st.settings.utilityModel; util.append(o); }
  util.addEventListener('change', async () => { await call('set_utility_model', { filename: util.value || null }); });
  const pre = el('input'); pre.type = 'checkbox'; pre.checked = st.settings.preload !== false; pre.setAttribute('aria-label', t('preloadLabel'));
  pre.addEventListener('change', async () => { await call('set_preload', { value: pre.checked }); });
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
    row(t('utilityModel'), t('utilityHint'), util),
    row(t('preloadLabel'), t('preloadHint'), pre),
    row(t('language'), '', lang),
    row(t('theme'), '', theme),
    row(t('reset'), '', reset),
  );
}

async function renderSkills() {
  const box = $('tab-skills'); box.textContent = '';
  box.append(el('p', 'hint', t('skillsHint')));
  const skills = await call('list_skills');
  for (const s of skills) {
    const b = el('div', 'skill-box'); const h = el('h4'); h.append(el('span', '', s.name));
    const del = el('button', 'btn', t('deleteSkill')); del.type = 'button';
    let armed = null;
    del.addEventListener('click', async () => {
      if (!armed) { del.textContent = t('confirmDelete'); del.classList.add('danger'); armed = setTimeout(() => { armed = null; del.textContent = t('deleteSkill'); del.classList.remove('danger'); }, 3000); return; }
      clearTimeout(armed); await call('delete_skill', { id: s.id }); await hooks.refreshCommands(); await renderSkills();
    });
    h.append(del); b.append(h);
    for (const c of s.commands) { const r = el('div', 'skill-cmd'); r.append(el('b', '', `/${c.command}`), document.createTextNode(`  ${c.description}`)); b.append(r); }
    box.append(b);
  }
  const form = el('form', 'skill-form'); form.append(el('h4', '', t('addSkill')));
  const name = el('input'); name.placeholder = t('skillName'); name.maxLength = 60;
  const content = el('textarea'); content.placeholder = t('skillContent');
  const err = el('div', 'form-error'); err.hidden = true;
  const save = el('button', 'btn primary', t('saveSkill')); save.type = 'submit';
  form.append(name, content, err, save);
  form.addEventListener('submit', async (e) => {
    e.preventDefault(); err.hidden = true;
    try { await call('add_skill', { name: name.value, content: content.value }); await hooks.refreshCommands(); await renderSkills(); }
    catch (ex) { err.textContent = ex.message; err.hidden = false; }
  });
  box.append(form);
}

async function renderDiagnostics() {
  const box = $('tab-diagnostics'); box.textContent = ''; box.classList.add('diag');
  const d = await call('diagnostics');
  const yn = (b) => (b ? 'ja' : 'NEI');
  const lines = [
    `Node ${d.system.node} (V8 ${d.system.v8}) - ${d.system.platform}`,
    `CPU: ${d.system.cpu} (${d.system.logicalCpus} logiske)`,
    `RAM: ${fmtBytes(d.system.ramFree)} ledig av ${fmtBytes(d.system.ramTotal)}`,
    `SIMD: ${yn(d.features.simd)} · relaxed SIMD: ${yn(d.features.relaxedSimd)} · tråder: ${yn(d.features.threads)} · SharedArrayBuffer: ${yn(d.features.sharedArrayBuffer)} · Atomics.wait: ${yn(d.features.atomicsWait)}`,
    `Datamappe: ${d.app.dataFolder}`, `Loggfil: ${d.app.logFile}`,
    `Kontekst: ${d.app.contextWindow} · tråder: ${d.app.threads} · lastet modell: ${d.app.loadedModel ?? '-'}`,
    '', 'Modeller:', ...d.models.map((m) => `  ${m.filename} · ${fmtBytes(m.sizeBytes)} · ${m.quant || '?'} · ${m.supported ? 'støttet' : `IKKE støttet (${m.reason})`}`),
    '', ...(d.problems.length ? ['Problemer:', ...d.problems.map((p) => `  - ${p}`)] : [t('diagOk')]),
    '', 'Logg:', d.logTail,
  ];
  const report = lines.join('\n');
  box.append(el('p', 'hint', t('diagHint')));
  const copy = el('button', 'btn primary', t('copyReport')); copy.type = 'button';
  copy.addEventListener('click', async () => { try { await navigator.clipboard.writeText(report); } catch { /* shown below anyway */ } copy.textContent = t('copied'); setTimeout(() => { copy.textContent = t('copyReport'); }, 1500); });
  box.append(copy);
  box.append(el('h3', '', t('diagSystem'))); const sys = el('pre'); sys.textContent = lines.slice(0, 9).join('\n'); box.append(sys);
  box.append(el('h3', '', t('diagModels'))); const mo = el('pre'); mo.textContent = d.models.map((m) => `${m.filename} · ${m.supported ? 'støttet' : `IKKE støttet (${m.reason})`}`).join('\n') || '-'; box.append(mo);
  box.append(el('p', d.problems.length ? 'bad' : 'ok', d.problems.length ? d.problems.join(' ') : t('diagOk')));
  box.append(el('h3', '', t('diagLog'))); const lg = el('pre'); lg.textContent = d.logTail || '-'; box.append(lg);
  box.append(el('p', 'shortcuts', `${t('shortcuts')}: ${t('shortcutsText')}`));
}
