import { call, listen, hasToken } from './api.js';
import { loadStrings, t, apply } from './i18n.js';
import { renderMarkdown } from './markdown.js';
import { initSettings, openSettings, onDownloadEvent } from './settings.js';
import { initCommandMenu, loadCommands, extractCommands, openAll } from './commands.js';
import { splitThinking, fmtDuration } from './thinking.js';
import { initDocuments, refreshChips, getPending, takePending, clearPending } from './documents.js';
import { initChatList, refreshChatList, setActive, setChatName } from './chats.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

const S = {
  models: [], model: null, settings: null, messages: [], // messages: [{role, content, ...extras from the file, node, bubble}]
  think: (() => { try { return localStorage.getItem('zc-think') === '1'; } catch { return false; } })(), // "Tenk først" (Qwen3 / Qwen3.5)
  chatId: null, chatName: null, idleWaiters: [],
  busy: false, runId: null, waiting: false, current: null, // current: {msg, bubble, status, bar}
  stickBottom: true,
};

// ---------- small helpers ----------
let toastTimer;
export function toast(text) { const n = $('toast'); n.textContent = text; n.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { n.hidden = true; }, 2200); }
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); } catch {
    const ta = el('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch { /* ignore */ } ta.remove();
  }
  toast(t('copied'));
}

// ---------- scrolling ----------
const scroller = () => $('scroller');
const nearBottom = () => scroller().scrollHeight - scroller().scrollTop - scroller().clientHeight < 80;
function scrollToBottom(force = false) { if (force || S.stickBottom) scroller().scrollTop = scroller().scrollHeight; }

// ---------- "Tenk først" (Qwen3 / Qwen3.5 reasoning) ----------
const thinkSupported = () => currentModel()?.thinking === true; // decided by the server (src/tokenizer/chat-templates.js thinkingMode)
const thinkOn = () => S.think && thinkSupported();
function updateThinkToggle() {
  const b = $('btn-think'); b.hidden = !thinkSupported();
  b.setAttribute('aria-pressed', String(S.think)); b.classList.toggle('on', S.think);
}

// ---------- model picker ----------
const currentModel = () => S.models.find((m) => m.filename === S.model) ?? null;
function renderModelPicker() {
  const usable = S.models.filter((m) => m.active || m.filename === S.model);
  const menu = $('model-menu'); menu.textContent = '';
  for (const m of usable) {
    const li = el('li'); li.setAttribute('role', 'option');
    li.setAttribute('aria-selected', String(m.filename === S.model));
    if (!m.supported) li.setAttribute('aria-disabled', 'true');
    li.append(el('span', 'mm-name', m.displayName));
    li.append(el('span', 'mm-sub', m.supported ? `${fmtBytes(m.sizeBytes)} · ${m.quant}${m.ramNeed ? ` · ${t('ramNeeded')} ${fmtBytes(m.ramNeed)}` : ''}` : `${t('unsupportedModel')}`));
    if (m.supported) li.addEventListener('click', () => { chooseModel(m.filename); closeMenu(); });
    menu.append(li);
  }
  const m = currentModel();
  $('model-name').textContent = m ? m.displayName : t('chooseModel');
  updateThinkToggle();
  const none = S.models.filter((x) => x.supported).length === 0;
  $('no-model').hidden = !none;
  if (none) {
    const n = $('no-model'); n.textContent = '';
    const go = el('button', 'btn primary', t('noModelAction')); go.type = 'button'; go.addEventListener('click', () => openSettings('models'));
    n.append(el('strong', '', t('noModelTitle')), el('br'), document.createTextNode(`${t('noModelText')} `), el('br'), go);
  }
  updateSendEnabled();
}
function closeMenu() { $('model-menu').hidden = true; $('model-btn').setAttribute('aria-expanded', 'false'); }
function chooseModel(filename) {
  S.model = filename; renderModelPicker(); call('set_last_model', { filename }).catch(() => {}); updateMeter();
}

// ---------- context meter ----------
let meterTimer;
function updateMeter() {
  clearTimeout(meterTimer);
  meterTimer = setTimeout(async () => {
    if (!S.model || S.busy) return;
    const msgs = S.messages.map(apiMsg);
    const draft = extractCommands($('input').value.trim());
    const docs = getPending();
    if (draft.text || draft.commands.length || docs.length) msgs.push({ role: 'user', content: draft.text, ...(draft.commands.length ? { commands: draft.commands } : {}), ...(docs.length ? { context: docs } : {}) });
    if (msgs.length === 0) { setMeter(0, S.settings?.contextWindow ?? 0); return; }
    try { const r = await call('estimate_context', { model: S.model, messages: msgs, chatId: S.chatId, think: thinkOn() }); setMeter(r.contextUsed, r.contextMax); } catch { /* model not readable */ }
  }, 350);
}
function setMeter(used, max) {
  const pct = max ? Math.min(100, (used / max) * 100) : 0;
  const f = $('meter-fill'); f.style.width = `${pct}%`; f.className = `meter-fill${pct > 95 ? ' full' : pct > 80 ? ' warn' : ''}`;
  $('meter-text').textContent = t('contextUsed', { used: used.toLocaleString(), max: max.toLocaleString() });
}

// ---------- messages ----------
function buildMessageNode(msg) {
  const wrap = el('div', `msg ${msg.role}`);
  const bubble = el('div', 'bubble');
  wrap.append(bubble);
  if (msg.role === 'assistant') bubble.classList.add('md');
  msg.node = wrap; msg.bubble = bubble;
  paintMessage(msg);
  $('messages').append(wrap);
  if (msg.role === 'user') { addCommandChips(msg); addUserActions(msg); }
  return wrap;
}
let paintQueued = new Set();
/** The foldable thinking block above an answer (created on demand, kept across repaints so the user's fold/unfold sticks). */
function paintThinking(msg, text, live) {
  if (!msg.thinkBox) {
    const d = el('details', 'think'); const sum = el('summary'); const body = el('div', 'think-body md');
    d.append(sum, body); msg.node.insertBefore(d, msg.bubble);
    msg.thinkBox = d; msg.thinkSum = sum; msg.thinkBody = body;
    if (live) { d.open = true; msg.thinkWasLive = true; }
  }
  if (live && msg.thinkStart === undefined) msg.thinkStart = performance.now();
  if (!live && msg.thinkWasLive) { msg.thinkWasLive = false; msg.thinkBox.open = false; if (msg.thinkEnd === undefined) msg.thinkEnd = performance.now(); } // done thinking: fold it away
  const ms = msg.thinkMs ?? (msg.thinkStart !== undefined ? (msg.thinkEnd ?? performance.now()) - msg.thinkStart : null);
  msg.thinkSum.textContent = live ? t('thinkingNow', { time: ms != null ? fmtDuration(ms) : '' }) : (ms != null ? t('thoughtFor', { time: fmtDuration(ms) }) : t('thoughtDone'));
  msg.thinkBody.innerHTML = renderMarkdown(text);
}
function paintMessage(msg) {
  if (msg.role === 'user') { msg.bubble.textContent = msg.content; return; }
  // while streaming, `content` holds the raw "<think>..</think>answer"; once finished the parts live in `thinking` / `content`
  const s = msg.thinking !== undefined ? { thinking: msg.thinking, answer: msg.content, state: 'done' } : splitThinking(msg.content);
  if (s.state !== 'none') paintThinking(msg, s.thinking, s.state === 'open');
  msg.bubble.innerHTML = renderMarkdown(s.answer);
}
/** At the end of a run: move the reasoning out of `content` into `thinking` (what is stored with the chat). */
function finalizeThinking(msg) {
  const s = splitThinking(msg.content);
  if (s.state === 'none') return;
  msg.thinking = s.thinking; msg.content = s.answer;
  msg.thinkMs = msg.thinkStart !== undefined ? (msg.thinkEnd ?? performance.now()) - msg.thinkStart : undefined;
  if (msg.thinkMs === undefined) delete msg.thinkMs;
  msg.thinkWasLive = false; if (msg.thinkBox) msg.thinkBox.open = false;
  return s.state;
}
function schedulePaint(msg) {
  if (paintQueued.has(msg)) return; paintQueued.add(msg);
  requestAnimationFrame(() => { paintQueued.delete(msg); paintMessage(msg); if (msg === S.current?.msg) msg.bubble.classList.add('caret-blink'); scrollToBottom(); });
}
function addCommandChips(msg) {
  msg.node.querySelector('.msg-cmds')?.remove();
  const files = [...new Set((msg.context ?? []).map((c) => c.filename))];
  if (!msg.commands?.length && !files.length) return;
  const chips = el('div', 'msg-cmds');
  for (const f of files) chips.append(el('span', 'cmd-chip file-chip', `📎 ${f}`));
  for (const c of msg.commands ?? []) chips.append(el('span', 'cmd-chip', `/${c}`));
  msg.bubble.after(chips);
}

function addUserActions(msg) {
  const bar = el('div', 'actions');
  const add = (label, fn) => { const b = el('button', 'act', label); b.type = 'button'; b.addEventListener('click', fn); bar.append(b); };
  add(t('copy'), () => copyText(msg.content));
  add(t('edit'), () => startEdit(msg));
  msg.node.append(bar);
}

function startEdit(msg) {
  if (S.busy) { toast(t('busy')); return; }
  const idx = S.messages.indexOf(msg);
  const later = S.messages.length - idx - 1;
  const box = el('div', 'editor');
  const ta = el('textarea'); ta.value = msg.content; ta.rows = Math.min(10, msg.content.split('\n').length + 1);
  const row = el('div', 'editor-row');
  const ok = el('button', 'btn primary', t('editResend')); ok.type = 'button';
  const no = el('button', 'btn', t('cancel')); no.type = 'button';
  row.append(ok, no);
  if (later > 0) row.append(el('span', 'hint', t('editWarn')));
  box.append(ta, row);
  msg.bubble.hidden = true; msg.node.querySelector('.actions')?.remove();
  msg.node.append(box); ta.focus();
  const close = () => { box.remove(); msg.bubble.hidden = false; addUserActions(msg); };
  no.addEventListener('click', close);
  ok.addEventListener('click', async () => {
    const text = ta.value.trim(); if (!text) return;
    box.remove(); msg.bubble.hidden = false;
    for (const m of S.messages.slice(idx + 1)) m.node.remove();
    S.messages.length = idx + 1;
    msg.content = text; paintMessage(msg); addUserActions(msg);
    const reply = { role: 'assistant', content: '' }; S.messages.push(reply); buildMessageNode(reply); updateWelcome();
    await startRun();
  });
}

function clearActions(msg) { msg.node.querySelectorAll('.actions, .meta').forEach((n) => n.remove()); }
function addActions(msg, { last, truncated }) {
  clearActions(msg);
  const bar = el('div', 'actions');
  const add = (label, fn) => { const b = el('button', 'act', label); b.type = 'button'; b.addEventListener('click', fn); bar.append(b); };
  add(t('copy'), () => copyText(msg.content));
  if (last) {
    add(t('regenerate'), regenerate);
    if (truncated) add(t('continue'), continueReply);
  }
  msg.node.append(bar);
  msg.node.classList.toggle('last', !!last);
}
function setMeta(msg, text, kind = '') { const m = el('div', `meta ${kind}`, text); msg.node.append(m); return m; }
function updateWelcome() { $('app').classList.toggle('has-messages', S.messages.length > 0); }

// ---------- status line inside a running reply ----------
function showProgress(text, fraction) {
  const c = S.current; if (!c) return;
  if (!c.status) { c.status = el('div', 'progress'); c.label = el('span', 'dots'); c.bar = el('div', 'progress-bar'); c.fill = el('div'); c.bar.append(c.fill); c.status.append(c.label, c.bar); c.msg.node.insertBefore(c.status, c.msg.bubble); }
  c.label.textContent = text;
  c.bar.hidden = fraction == null;
  if (fraction != null) c.fill.style.width = `${Math.round(fraction * 100)}%`;
}
function hideProgress(c = S.current) { if (c?.status) { c.status.remove(); c.status = null; } }

// ---------- sending ----------
function updateSendEnabled() {
  const btn = $('send');
  if (S.busy) { btn.disabled = false; btn.classList.add('stop'); btn.textContent = '■'; btn.setAttribute('aria-label', t('stop')); return; }
  btn.classList.remove('stop'); btn.textContent = '↑'; btn.setAttribute('aria-label', t('send'));
  btn.disabled = !(S.model && currentModel()?.supported && $('input').value.trim());
}
function setBusy(b) { S.busy = b; $('status').textContent = ''; updateSendEnabled(); }

function apiMessages() { return S.messages.filter((m) => m.content || m.role === 'user').map(({ role, content }) => ({ role, content })); }

async function startRun() {
  const reply = S.messages.at(-1); // the assistant message being filled
  S.current = { msg: reply };
  S.waiting = true; S.runId = null;
  setBusy(true);
  reply.node.classList.add('last');
  showProgress(t('loadingModel'), null);
  reply.bubble.classList.add('caret-blink');
  scrollToBottom(true); S.stickBottom = true;
  const history = S.messages.slice(0, -1).map(apiMsg);
  try { await call('send_message', { model: S.model, messages: history, chatId: S.chatId, think: thinkOn() }); }
  catch (e) { finishRun(); setMeta(reply, e.message === 'busy' ? t('busy') : `${t('error')}: ${e.message}`, 'err'); }
}

async function sendDraft(raw) {
  if (S.busy || !raw.trim()) return;
  const { text, commands } = extractCommands(raw);
  if (!text) { toast(t('writeAfterCommand')); return; }
  const user = { role: 'user', content: text.trim() };
  if (commands.length) user.commands = commands;
  const docs = getPending();
  if (docs.length) {
    user.context = docs;
    // does the document (read whole) fit in the context window together with the conversation and a reply?
    try {
      const r = await call('estimate_context', { model: S.model, messages: [...S.messages.map(apiMsg), apiMsg(user)], chatId: S.chatId, think: thinkOn() });
      if (r.docTokens > r.docLimit) { toast(t('docTooBig', { used: r.docTokens.toLocaleString(), max: r.docLimit.toLocaleString() })); return; }
    } catch { /* the server checks again when it runs */ }
    takePending();
  }
  const reply = { role: 'assistant', content: '' };
  S.messages.push(user, reply);
  buildMessageNode(user); buildMessageNode(reply); updateWelcome();
  $('input').value = ''; autoGrow();
  await persist(); // the chat appears in the sidebar (named after the question) right away, not only when the reply is ready
  await startRun();
}

async function regenerate() {
  if (S.busy) return;
  const last = S.messages.at(-1);
  if (!last || last.role !== 'assistant') return;
  last.node.remove(); S.messages.pop();
  const reply = { role: 'assistant', content: '' }; S.messages.push(reply); buildMessageNode(reply);
  await startRun();
}

async function continueReply() {
  if (S.busy) return;
  const reply = S.messages.at(-1);
  if (!reply || reply.role !== 'assistant') return;
  // the server appends a "continue" instruction after the cut reply
  reply.node.querySelectorAll('.meta, .actions').forEach((n) => n.remove());
  S.current = { msg: reply }; S.waiting = true; S.runId = null; setBusy(true);
  showProgress(t('loadingModel'), null); reply.bubble.classList.add('caret-blink');
  const history = S.messages.map(apiMsg);
  try { await call('continue_message', { model: S.model, messages: history, chatId: S.chatId, think: thinkOn() }); } catch (e) { finishRun(); setMeta(reply, `${t('error')}: ${e.message}`, 'err'); }
}

function finishRun() {
  const c = S.current; S.current = null; S.runId = null; S.waiting = false;
  if (c) { c.msg.bubble.classList.remove('caret-blink'); hideProgress(c); paintMessage(c.msg); }
  setBusy(false); updateMeter();
  const w = S.idleWaiters; S.idleWaiters = []; for (const r of w) r();
}

// ---------- saving chats ----------
const apiMsg = ({ role, content, commands, context }) => ({ role, content, ...(commands?.length ? { commands } : {}), ...(Array.isArray(context) ? { context } : {}) });
const UI_KEYS = new Set(['node', 'bubble', 'thinkBox', 'thinkSum', 'thinkBody', 'thinkStart', 'thinkEnd', 'thinkWasLive']);
const plain = (m) => Object.fromEntries(Object.entries(m).filter(([k]) => !UI_KEYS.has(k)));
async function persist() {
  const msgs = S.messages.filter((m) => m.role === 'user' || m.content || m.thinking).map(plain); // snapshot now: the chat may change while saving
  if (!msgs.some((m) => m.role === 'user')) return;
  if (!S.chatName) S.chatName = (msgs.find((m) => m.role === 'user')?.content ?? '').replace(/\s+/g, ' ').trim().slice(0, 45) || t('newChat');
  try {
    const r = await call('save_chat', { id: S.chatId, name: S.chatName, model: S.model, messages: msgs });
    if (!S.chatId || S.chatId === r.id) { S.chatId = r.id; S.chatName = r.name ?? S.chatName; setActive(r.id); } // the server may have kept a newer (automatic) title
    await refreshChatList();
  } catch (e) { toast(`${t('error')}: ${e.message}`); }
}
function whenIdle() { return S.busy ? new Promise((r) => S.idleWaiters.push(r)) : Promise.resolve(); }
async function stopAndWait() { if (S.busy) { call('stop_generation').catch(() => {}); await whenIdle(); } }

function resetView() { S.messages = []; $('messages').textContent = ''; updateWelcome(); }

async function openChat(id) {
  if (id === S.chatId && !S.busy) return;
  await stopAndWait();
  let data = null;
  try { data = await call('load_chat', { id }); } catch { /* handled below */ }
  if (!data) { toast(t('chatNotFound')); await refreshChatList(); return; }
  resetView();
  S.chatId = data.id; S.chatName = data.name;
  if (data.model && S.models.some((m) => m.filename === data.model && m.supported)) { S.model = data.model; renderModelPicker(); }
  S.messages = data.messages;
  for (const m of S.messages) buildMessageNode(m);
  for (const m of S.messages) if (m.role === 'assistant') { addActions(m, { last: m === S.messages.at(-1), truncated: false }); }
  updateWelcome(); setActive(S.chatId); clearPending(); updateMeter(); scrollToBottom(true); S.stickBottom = true;
  document.addEventListener('keydown', (e) => {
    if ($('settings').open) return;
    if (e.key === 'Escape' && S.busy && $('cmd-menu').hidden) { call('stop_generation').catch(() => {}); }
    else if (e.altKey && (e.key === 'n' || e.key === 'N')) { e.preventDefault(); newChat(); }
    else if (e.altKey && (e.key === 'b' || e.key === 'B')) { e.preventDefault(); $('app').classList.toggle('side-closed'); }
    else if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName ?? '')) { e.preventDefault(); $('input').focus(); }
  });
  if (window.matchMedia('(max-width: 760px)').matches) $('app').classList.add('side-closed');
}

// ---------- server events ----------
function ours(d) {
  if (!S.current) return false;
  if (S.waiting) { S.runId = d.runId; S.waiting = false; }
  return d.runId === S.runId;
}
const events = {
  preload(d) {
    if (d.state === 'loading') $('status').textContent = t('loadingModel');
    else if (!S.busy) $('status').textContent = '';
  },
  download(d) { if (onDownloadEvent(d)) refreshState().catch(() => {}); },
  phase(d) {
    if (!ours(d)) return;
    showProgress(d.phase === 'loading' ? t('loadingModel') : d.phase === 'summarizing' ? t('summarizing') : d.phase === 'preparing' ? t('readingPrompt') : t('generating'), null);
    if (d.phase === 'generating') hideProgress();
  },
  progress(d) {
    if (!ours(d)) return;
    showProgress(`${t('readingPrompt')} ${d.done.toLocaleString()} / ${d.total.toLocaleString()}`, d.total ? d.done / d.total : null);
  },
  chunk(d) {
    if (!ours(d)) return;
    hideProgress();
    S.current.msg.content += d.text; schedulePaint(S.current.msg);
  },
  done(d) {
    if (!ours(d)) return;
    const msg = S.current.msg;
    finishRun();
    const thinkState = finalizeThinking(msg); if (thinkState) paintMessage(msg);
    const truncated = d.stopReason === 'length' || d.truncated;
    addActions(msg, { last: true, truncated });
    const bits = [];
    if (d.tokPerSec > 0) bits.push(t('tokensPerSecond', { n: d.tokPerSec.toFixed(1) }));
    if (d.prefillMs + d.decodeMs > 0) bits.push(t('seconds', { n: ((d.prefillMs + d.decodeMs) / 1000).toFixed(1) }));
    if (bits.length) setMeta(msg, bits.join(' · '));
    if (d.looped) setMeta(msg, t('looped'), 'warn');
    else if (d.stopped) setMeta(msg, t('stopped'), 'warn');
    else if (truncated && thinkState === 'open') setMeta(msg, t('thinkCut'), 'warn');
    else if (truncated) setMeta(msg, t('cut'), 'warn');
    if (d.compressed) setMeta(msg, t('compressedNote'));
    if (d.trimmed) setMeta(msg, t('trimmed'), 'warn');
    if (d.contextMax) setMeter(Math.min(d.contextMax, d.contextUsed + (d.generated ?? 0)), d.contextMax);
    S.loaded = d.model;
    persist();
  },
  /** The server wrote an automatic title for a chat (also for a chat that is not the open one, and when the page was reloaded meanwhile). */
  title(d) {
    if (!d?.chatId || typeof d.name !== 'string') return;
    if (d.chatId === S.chatId) S.chatName = d.name;
    setChatName(d.chatId, d.name);
  },
  error(d) {
    if (!ours(d)) return;
    const msg = S.current.msg;
    finishRun();
    addActions(msg, { last: true, truncated: false });
    setMeta(msg, `${t('error')}: ${d.message}`, 'err');
    persist();
  },
};

// ---------- composer ----------
function autoGrow() { const i = $('input'); i.style.height = 'auto'; i.style.height = `${Math.min(i.scrollHeight, 220)}px`; }

function wireUi() {
  $('composer').addEventListener('submit', (e) => {
    e.preventDefault();
    if (S.busy) { call('stop_generation').catch(() => {}); return; }
    if (!$('send').disabled) sendDraft($('input').value);
  });
  $('input').addEventListener('input', () => { autoGrow(); updateSendEnabled(); updateMeter(); });
  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); }
  });
  $('btn-new').addEventListener('click', newChat);
  $('btn-think').addEventListener('click', () => { S.think = !S.think; try { localStorage.setItem('zc-think', S.think ? '1' : '0'); } catch { /* ignore */ } updateThinkToggle(); updateMeter(); });
  $('btn-sidebar').addEventListener('click', () => $('app').classList.toggle('side-closed'));
  $('btn-settings').addEventListener('click', () => openSettings());
  $('model-btn').addEventListener('click', (e) => { e.stopPropagation(); const m = $('model-menu'); m.hidden = !m.hidden; $('model-btn').setAttribute('aria-expanded', String(!m.hidden)); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.model-picker')) closeMenu(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
  scroller().addEventListener('scroll', () => { S.stickBottom = nearBottom(); $('jump').hidden = S.stickBottom; });
  $('jump').addEventListener('click', () => { scrollToBottom(true); });
  $('messages').addEventListener('click', (e) => {
    const b = e.target.closest('[data-copy-code]'); if (!b) return;
    copyText(b.closest('.codeblock').querySelector('code').textContent);
  });
  if (window.matchMedia('(max-width: 760px)').matches) $('app').classList.add('side-closed');
}

function renderStarters() {
  const box = $('starters'); box.textContent = '';
  for (const k of ['starter1', 'starter2', 'starter3', 'starter4']) {
    const b = el('button', 'starter', t(k)); b.type = 'button';
    b.addEventListener('click', () => { $('input').value = t(k); autoGrow(); updateSendEnabled(); $('input').focus(); });
    box.append(b);
  }
}

async function newChat() {
  await stopAndWait();
  S.chatId = null; S.chatName = null; resetView(); setActive(null); clearPending();
  setMeter(0, S.settings?.contextWindow ?? 0); $('input').value = ''; autoGrow(); updateSendEnabled(); $('input').focus();
}

export async function refreshState() {
  const st = await call('state');
  S.models = st.models; S.settings = st.settings;
  if (!S.model || !S.models.some((m) => m.filename === S.model && m.supported)) {
    S.model = S.models.find((m) => m.filename === st.lastModel && m.supported)?.filename ?? S.models.find((m) => m.supported && m.active)?.filename ?? null;
  }
  renderModelPicker();
  return st;
}

async function init() {
  if (!hasToken()) { document.body.textContent = 'Åpne appen med snarveien (mangler tilgangsnøkkel). / Open the app from its shortcut (missing access token).'; return; }
  const initial = await call('state');
  window.__lang = initial.language;
  await loadStrings(initial.language);
  window.__copyLabel = t('copy');
  initSettings({ refreshState, refreshCommands: loadCommands, applyLanguage: async (lang) => { await call('set_language', { lang }); await loadStrings(lang); refreshChatList(); window.__copyLabel = t('copy'); renderStarters(); renderModelPicker(); setMeter(0, S.settings?.contextWindow ?? 0); } });
  initChatList({ open: openChat, deleted: (id) => { if (id === S.chatId) newChat(); }, renamed: (id, name) => { if (id === S.chatId) S.chatName = name; } });
  initDocuments({ toast, onChange: () => updateMeter() });
  wireUi(); renderStarters(); refreshChatList(); loadCommands();
  initCommandMenu({ inputEl: $('input'), menuEl: $('cmd-menu'), afterInsert: () => { autoGrow(); updateSendEnabled(); updateMeter(); } });
  $('btn-slash').addEventListener('click', () => { loadCommands().then(() => { if (!openAll()) toast(t('noCommands')); }); $('input').focus(); });
  S.models = initial.models; S.settings = initial.settings;
  S.model = initial.models.find((m) => m.filename === initial.lastModel && m.supported)?.filename ?? initial.models.find((m) => m.supported && m.active)?.filename ?? null;
  renderModelPicker(); setMeter(0, initial.settings.contextWindow);
  listen(events, { onConnection: (ok) => { if (!ok) $('status').textContent = t('openFromShortcut'); else if (!S.busy) $('status').textContent = ''; } });
  updateSendEnabled();
}

window.__zc = { S, buildMessageNode, paintMessage, finalizeThinking }; // handles for the browser smoke test
window.addEventListener('unhandledrejection', (e) => {
  const m = String(e.reason?.message ?? '');
  if (m === 'busy') { toast(t('busy')); e.preventDefault(); }
});

init().catch((e) => { document.body.textContent = `${e.message}`; });
