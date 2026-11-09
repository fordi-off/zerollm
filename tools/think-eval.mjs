// Does thinking mode (and a given model) get calculations right? Runs a fixed set of maths/physics questions in the same way the app does
// (same system prompt and sampling) once without and once with "Tenk først", and checks the final numbers automatically.
//   node tools/think-eval.mjs models/Qwen_Qwen3-8B-Q4_K_M.gguf [--variants plain,think] [--only 1,3,4] [--ctx 8192] [--threads 6] [--out results.json]
// Slow on purpose (thinking = thousands of tokens): start with --only 3,4 to see the falling-stone questions.
import fs from 'node:fs';
import { loadModel } from '../src/index.js';
import { buildMessages, SAMPLING, THINK_SAMPLING, THINK_MAX_TOKENS } from '../app/chat-core.js';
import { splitThinking } from '../app/public/js/thinking.js';
import { parseArgs, num } from '../src/util/args.js';

const G = 6.6743e-11; const M = 5.972e24; const R = 6.371e6;
export const QUESTIONS = [
  { id: 1, q: 'Hva er 17 * 23?', expect: [391] },
  { id: 2, q: 'Hva er 18 % av 2450?', expect: [441] },
  { id: 3, q: 'En gjenstand slippes fra ro fra en høyde på 39,0 km over jordoverflata. Se bort fra luftmotstand og bruk konstant tyngdeakselerasjon g = 9,81 m/s². Hvor stor fart har gjenstanden når den treffer bakken?', expect: [Math.sqrt(2 * 9.81 * 39000)], tol: 0.005 },
  { id: 4, q: 'En gjenstand slippes fra ro fra en høyde på 39,0 km over jordoverflata. Se bort fra luftmotstand, men ta hensyn til at gravitasjonen avtar med avstanden fra jordas sentrum (Newtons gravitasjonslov, G = 6,6743e-11, jordas masse 5,972e24 kg, jordradius 6,371e6 m). Hvor stor fart har gjenstanden når den treffer jordoverflata?', expect: [Math.sqrt(2 * G * M * (1 / R - 1 / (R + 39000)))], tol: 0.002 },
  { id: 5, q: 'Du setter 10 000 kr i banken med 4,5 % rente per år, med rentes rente. Hvor mye har du etter 12 år?', expect: [10000 * 1.045 ** 12], tol: 0.002 },
  { id: 6, q: 'Løs ligningen x² − 5x + 6 = 0.', expect: [2, 3], all: true, tol: 0.001 },
  { id: 7, q: 'Et tog kjører 240 km på 2 timer og 30 minutter. Hva er gjennomsnittsfarten i km/t?', expect: [96] },
  { id: 8, q: 'Hvor mange sekunder er det i 3,5 døgn?', expect: [302400] },
];

/** All numbers in a text (Norwegian "16 958,81" and "16958.81" both work). */
export function numbersIn(text) {
  const t = String(text).replace(/(\d)[\s ](?=\d{3}(?!\d))/g, '$1');
  return [...t.matchAll(/\d+(?:[.,]\d+)?/g)].map((m) => Number(m[0].replace(',', '.')));
}
export function grade(answer, { expect, tol = 0.0005, all = false }) {
  const nums = numbersIn(answer);
  const hit = (x) => nums.some((n) => Math.abs(n - x) <= Math.abs(x) * tol + 1e-9);
  return all ? expect.every(hit) : expect.some(hit);
}

async function main() {
  const { _, flags } = parseArgs(process.argv.slice(2), {});
  if (!_[0]) { console.error('usage: node tools/think-eval.mjs <model.gguf> [--variants plain,think] [--only 1,3] [--ctx 8192] [--threads N] [--out file.json]'); process.exit(1); }
  const ctx = num(flags.ctx, 8192);
  const engine = await loadModel(_[0], { ctx, threads: flags.threads ? num(flags.threads, 1) : undefined, kv: 'f16' });
  const canThink = engine.cfg.arch === 'qwen3';
  const variants = (typeof flags.variants === 'string' ? flags.variants.split(',') : ['plain', 'think']).filter((v) => v === 'plain' || (v === 'think' && canThink));
  const only = typeof flags.only === 'string' ? new Set(flags.only.split(',').map(Number)) : null;
  const results = [];
  console.log(`model ${engine.info.name ?? _[0]} (${engine.cfg.arch}), ctx ${engine.ctx}, variants: ${variants.join(', ')}${canThink ? '' : '  (no thinking mode for this architecture)'}\n`);
  try {
    for (const Q of QUESTIONS.filter((x) => !only || only.has(x.id))) {
      for (const variant of variants) {
        const think = variant === 'think';
        const messages = buildMessages([{ role: 'user', content: Q.q }]);
        const t0 = performance.now(); const stats = {}; let raw = '';
        const maxTokens = think ? Math.min(THINK_MAX_TOKENS, ctx - 700) : 1500;
        engine.reset();
        for await (const text of engine.generate({ messages, think, maxTokens, sampling: think ? THINK_SAMPLING : SAMPLING, penalizePrompt: false, stats })) raw += text;
        const s = splitThinking(raw); const answer = think ? s.answer : raw;
        const ok = grade(answer, Q); const secs = (performance.now() - t0) / 1000;
        results.push({ id: Q.id, variant, ok, secs: Math.round(secs), generated: stats.generated, thinkChars: s.thinking.length, unfinishedThinking: think && s.state === 'open', answer: answer.trim(), thinking: s.thinking });
        console.log(`Q${Q.id} ${variant.padEnd(5)} ${ok ? 'OK  ' : 'FAIL'} ${String(Math.round(secs)).padStart(4)} s  ${String(stats.generated).padStart(5)} tok${think ? `  (thinking ${s.thinking.length} chars${s.state === 'open' ? ', NEVER FINISHED' : ''})` : ''}  expected ${Q.expect.map((x) => +x.toPrecision(6)).join(' & ')}`);
        console.log(`      ${answer.trim().replace(/\s+/g, ' ').slice(-160)}`);
      }
    }
  } finally { await engine.close(); }
  console.log('\nsummary:');
  for (const v of variants) { const r = results.filter((x) => x.variant === v); console.log(`  ${v.padEnd(5)} ${r.filter((x) => x.ok).length}/${r.length} correct, ${Math.round(r.reduce((a, x) => a + x.secs, 0) / Math.max(1, r.length))} s on average`); }
  const out = typeof flags.out === 'string' ? flags.out : 'think-eval-results.json';
  fs.writeFileSync(out, JSON.stringify({ model: _[0], ctx, results }, null, 2));
  console.log(`full answers and reasoning saved in ${out}`);
}
if (import.meta.url === `file://${process.argv[1]}`) main().catch((e) => { console.error(e.message ?? e); process.exit(1); });
