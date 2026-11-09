// Qwen3 "thinking": the reply streams as "<think>reasoning</think>answer". This module splits it. The reasoning is shown in a foldable
// block and stored on the message (`thinking`); the history sent back to the model only carries the answer (Qwen3's own convention).

/** @returns {{thinking: string, answer: string, state: 'none'|'open'|'done'}} open = still thinking (no closing tag yet) */
export function splitThinking(raw) {
  const s = String(raw ?? '');
  const end = s.indexOf('</think>');
  if (end >= 0) return { thinking: s.slice(0, end).replace(/^\s*<think>\s*/, '').trim(), answer: s.slice(end + 8).replace(/^\s+/, ''), state: 'done' };
  const head = s.trimStart();
  if (head.startsWith('<think>')) return { thinking: head.slice(7).trimStart(), answer: '', state: 'open' };
  if (head.length > 0 && head.length < 7 && '<think>'.startsWith(head)) return { thinking: '', answer: '', state: 'open' }; // the tag itself is still arriving
  return { thinking: '', answer: s, state: 'none' };
}

/** 134000 ms -> "2:14" */
export function fmtDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
