#!/usr/bin/env node
// Speculative-decoding benchmark: plain decoding vs draft-model decoding, paired and interleaved (so slow drift of a noisy
// machine hits both sides alike), several rounds, medians.
//
//   node tools/specbench.mjs <target.gguf> <draft.gguf> [--threads 6] [--draft-threads 2,4] [--k 2,3,4,6] [--n 96] [--rounds 5]
//        [--chat] [--prompt-file text.txt] [--prompt-tokens 48] [--temp 0] [--top-k 40] [--top-p 0.9] [--kv f32|f16] [--ctx 1024] [--kernels auto]
//
// --chat      wrap each prompt in the model's chat template (use this with instruct models; it is what `chat` / the app feed)
// --temp 0    greedy: the speculative output must equal the plain output (checked, reported in the last column)
// --temp T    sampled; the output distribution is the same, the texts differ, so only speed and acceptance are compared
// Without --prompt-file a few built-in natural-language prompts are used (one per round, cycling).
// Numbers measured on a machine that is not your target machine say little: run this on the laptop.
import fs from 'node:fs';
import { readGGUF } from '../src/gguf/reader.js';
import { modelConfig, assertSupportedTensorTypes } from '../src/model/config.js';
import { Tokenizer } from '../src/tokenizer/bpe.js';
import { detectTemplate, encodeChat } from '../src/tokenizer/chat-templates.js';
import { generate } from '../src/model/generate.js';
import { generateSpeculative, vocabMismatch, defaultDraftThreads } from '../src/model/speculative.js';
import { createSampler } from '../src/sampling/sampler.js';
import { createModel, defaultThreads } from '../src/cli/common.js';
import { parseArgs, num } from '../src/util/args.js';
import { formatBytes } from '../src/util/log.js';

const PROMPTS = [
  'Explain in a few paragraphs how a hash table works and why lookups are usually fast.',
  'Write a short story about a lighthouse keeper who discovers that the light has been answering messages from the sea.',
  'What are the main differences between TCP and UDP? Give practical examples of when to use each.',
  'Summarise the causes of the French Revolution for a high-school student.',
  'Give me a step by step recipe for a simple vegetable soup, with quantities for four people.',
];

const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

const { _, flags } = parseArgs(process.argv.slice(2), { boolean: ['chat'] });
if (_.length < 2) { console.error('usage: node tools/specbench.mjs <target.gguf> <draft.gguf> [--threads N] [--draft-threads 2,4] [--k 2,3,4] [--n 96] [--rounds 5] [--chat] [--prompt-file f] [--temp 0]'); process.exit(1); }

const g = readGGUF(_[0]); const dg = readGGUF(_[1]);
const cfg = modelConfig(g); const dcfg = modelConfig(dg);
assertSupportedTensorTypes(g); assertSupportedTensorTypes(dg);
const bad = vocabMismatch(g, dg);
if (bad) { console.error(`error: ${bad}`); process.exit(1); }
const tok = Tokenizer.fromGGUF(g);
const kind = detectTemplate(cfg.arch, g.get('tokenizer.chat_template'), tok);

const threads = flags.threads === undefined ? defaultThreads() : num(flags.threads, 1);
const dthreadsList = String(flags['draft-threads'] ?? defaultDraftThreads(threads)).split(',').map(Number);
const kList = String(flags.k ?? '2,3,4').split(',').map(Number);
const N = num(flags.n, 96); const rounds = num(flags.rounds, 5);
const temp = num(flags.temp, 0);
const kv = flags.kv === 'f16' ? 'f16' : 'f32';
const kernels = typeof flags.kernels === 'string' ? flags.kernels : 'auto';
const texts = typeof flags['prompt-file'] === 'string' ? [fs.readFileSync(flags['prompt-file'], 'utf8')] : PROMPTS;
const promptIds = (text) => {
  if (flags.chat) return encodeChat(tok, kind, [{ role: 'user', content: text }], { arch: cfg.arch, jinja: g.get('tokenizer.chat_template'), think: false });
  let ids = tok.encode(text);
  const want = num(flags['prompt-tokens'], 0);
  if (want > 0) { let all = []; while (all.length < want) all = all.concat(ids); ids = all.slice(0, want); }
  return ids;
};
const ctx = num(flags.ctx, 2048);
const mkSampler = (seed) => createSampler({ temperature: temp, topK: num(flags['top-k'], 40), topP: num(flags['top-p'], 0.9), seed });

const weights = (gg) => gg.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);
console.log(`target ${cfg.arch} ${cfg.nLayer}L ${formatBytes(weights(g))}, ${threads} threads | draft ${dcfg.arch} ${dcfg.nLayer}L ${formatBytes(weights(dg))} | ${temp > 0 ? `sampled T=${temp}` : 'greedy'}, ${N} tokens per run, ${rounds} rounds, ${flags.chat ? 'chat template' : 'raw text'}, kv ${kv}, kernels ${kernels}\n`);

const target = await createModel(g, { threads, ctx, kernels, kv });
try {
  const rows = [];
  for (const dthreads of dthreadsList) {
    const draft = await createModel(dg, { threads: dthreads, ctx, kernels, kv });
    try {
      const base = []; const per = new Map(kList.map((k) => [k, { eff: [], ratio: [], acc: [], tps: [], draftMs: [], verifyMs: [], otherMs: [], same: true }]));
      for (let r = 0; r < rounds; r++) {
        const ids = promptIds(texts[r % texts.length]);
        if (ids.length + N + 2 >= ctx) throw new Error(`prompt of ${ids.length} tokens + ${N} does not fit ctx ${ctx}; raise --ctx`);
        const seed = 1000 + r;
        target.reset(); draft.reset();
        const st = {}; const ref = [];
        for await (const id of generate(target, ids, { maxTokens: N, stats: st, sample: mkSampler(seed) })) ref.push(id);
        const b = (st.generated * 1000) / st.decodeMs; base.push(b);
        for (const k of kList) {
          target.reset(); draft.reset();
          const s2 = {}; const out = [];
          for await (const id of generateSpeculative(target, draft, ids, { maxTokens: N, stats: s2, k, sample: mkSampler(seed) })) out.push(id);
          const e = (s2.generated * 1000) / s2.decodeMs; const sp = s2.speculative; const c = per.get(k);
          c.eff.push(e); c.ratio.push(e / b); c.acc.push(sp.acceptRate); c.tps.push(sp.tokensPerCycle);
          c.draftMs.push(sp.draftMs / Math.max(1, sp.cycles)); c.verifyMs.push(sp.verifyMs / Math.max(1, sp.cycles));
          c.otherMs.push((s2.decodeMs - sp.draftMs - sp.verifyMs) / Math.max(1, sp.cycles)); // sampling, acceptance test, bookkeeping, event-loop turns
          if (temp <= 0 && out.join() !== ref.join()) c.same = false;
        }
        process.stderr.write(`round ${r + 1}/${rounds} done\r`);
      }
      for (const k of kList) {
        const c = per.get(k);
        rows.push({ dthreads, k, base: median(base), eff: median(c.eff), speed: median(c.ratio), lo: Math.min(...c.ratio), hi: Math.max(...c.ratio), acc: median(c.acc), tps: median(c.tps), d: median(c.draftMs), v: median(c.verifyMs), o: median(c.otherMs), same: c.same });
      }
    } finally { await draft.close?.(); }
  }
  process.stderr.write('\n');
  console.log('| draft threads | k | plain tok/s | speculative tok/s | speed-up (median of paired ratios, min-max) | accepted | tok/step | draft ms/step | verify ms/step | other ms/step | greedy output identical |');
  console.log('|---|---|---|---|---|---|---|---|---|---|---|');
  for (const r of rows) {
    console.log(`| ${r.dthreads} | ${r.k} | ${r.base.toFixed(2)} | ${r.eff.toFixed(2)} | ${r.speed.toFixed(2)}x (${r.lo.toFixed(2)}-${r.hi.toFixed(2)}) | ${(100 * r.acc).toFixed(0)}% | ${r.tps.toFixed(2)} | ${r.d.toFixed(1)} | ${r.v.toFixed(1)} | ${r.o.toFixed(1)} | ${temp > 0 ? 'n/a' : r.same ? 'yes' : 'NO'} |`);
  }
} finally { await target.close?.(); }
