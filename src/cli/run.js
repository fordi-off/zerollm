import fs from 'node:fs';
import { parseArgs, num } from '../util/args.js';
import { encodeChat } from '../tokenizer/chat-templates.js';
import { loadEngine, samplerFromFlags, templateOptions, installAbort, streamReply, formatStats, SAMPLER_HELP } from './common.js';

export const help = `run <model.gguf> -p "prompt" [--n 256] [--system "..."] [--ctx 2048] [--threads N] [--raw] [--think|--no-think] [--stop "str"] ${SAMPLER_HELP}`;

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { alias: { p: 'prompt', n: 'n' }, boolean: ['raw', 'think'] });
  let prompt = flags.prompt;
  if (typeof prompt !== 'string') {
    if (process.stdin.isTTY) throw new Error(`usage: zerollm ${help}`);
    prompt = fs.readFileSync(0, 'utf8');
  }
  const eng = await loadEngine(_[0], flags);
  try {
  process.stderr.write(`[loaded ${eng.cfg.arch} ${eng.cfg.nLayer}L in ${(eng.loadMs / 1000).toFixed(1)}s, ${eng.threads} thread${eng.threads > 1 ? 's' : ''}, template: ${eng.kind}, ctx ${eng.ctx}]\n`);

  /** @type {number[]} */
  let ids;
  if (eng.kind === 'raw') {
    ids = eng.tok.encode(prompt);
  } else {
    const messages = [];
    if (typeof flags.system === 'string') messages.push({ role: 'system', content: flags.system });
    messages.push({ role: 'user', content: prompt });
    ids = encodeChat(eng.tok, eng.kind, messages, templateOptions(eng, flags));
  }

  const signal = { aborted: false };
  const dispose = installAbort(signal);
  const stops = typeof flags.stop === 'string' ? [flags.stop] : [];
  const { stats } = await streamReply(eng, ids, { sample: samplerFromFlags(flags), maxTokens: num(flags.n, 256), stops, signal });
  dispose();
  process.stdout.write('\n');
  process.stderr.write(`${formatStats(eng, stats)}\n`);
  if (signal.aborted && stats.stopReason === 'aborted') process.exitCode = 130;
  } finally { await eng.close(); }
}
