import readline from 'node:readline';
import { parseArgs, num } from '../util/args.js';
import { encodeChat } from '../tokenizer/chat-templates.js';
import { loadEngine, samplerFromFlags, templateOptions, streamReply, formatStats, usePrefixCache, describePrompt, SAMPLER_HELP, DRAFT_HELP, loadedLine } from './common.js';

export const help = `chat <model.gguf> [--system "..."] [--n 512] [--ctx 2048] [--threads N] [--kernels auto|wasm|js] [--kv f32|f16] [--think|--no-think] [--no-cache] [--cache-dir DIR] [--show-prompt] ${SAMPLER_HELP} ${DRAFT_HELP}`;

const COMMANDS = '/reset  clear conversation   /system <text>  set system prompt (resets)   /think  /nothink  toggle Qwen3 thinking   /exit';

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { alias: { n: 'n' }, boolean: ['raw', 'think', 'show-prompt'] });
  const eng = await loadEngine(_[0], flags);
  const sample = samplerFromFlags(flags);
  const maxTokens = num(flags.n, 512);
  let system = typeof flags.system === 'string' ? flags.system : null;
  let think = flags.think === true;
  /** @type {Array<{role: string, content: string}>} */
  let messages = [];

  process.stderr.write(loadedLine(eng));
  if (eng.kind === 'raw') { await eng.close(); throw new Error('chat needs a chat template (this model has none we recognise); use `run --raw` for completion, or pass --template qwen|llama3'); }
  process.stderr.write(`${COMMANDS}\nCtrl+C stops a reply; Ctrl+C at the prompt exits.\n\n`);

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
  const signal = { aborted: false };
  let busy = false;
  // readline owns Ctrl+C when stdin is a TTY, so use its SIGINT event instead of process.on('SIGINT')
  rl.on('SIGINT', () => {
    if (busy) { signal.aborted = true; } else { process.stdout.write('\n'); rl.close(); }
  });
  // stdin can end (piped input, Ctrl+D) while a reply is still streaming: finish the reply, then exit.
  let closed = false;
  rl.on('close', () => { closed = true; if (!busy) { eng.close().finally(() => process.exit(0)); } });
  const prompt = () => { if (!closed) rl.prompt(); };

  prompt();
  for await (const line of rl) {
    const input = line.trim();
    if (!input) { prompt(); continue; }
    if (input === '/exit' || input === '/quit') break;
    if (input === '/reset') { messages = []; eng.reset(); process.stdout.write('(conversation cleared)\n'); prompt(); continue; }
    if (input.startsWith('/system')) { system = input.slice(7).trim() || null; messages = []; eng.reset(); process.stdout.write('(system prompt set, conversation cleared)\n'); prompt(); continue; }
    if (input === '/think' || input === '/nothink') { think = input === '/think'; process.stdout.write(`(thinking ${think ? 'on' : 'off'})\n`); prompt(); continue; }

    messages.push({ role: 'user', content: input });
    const convo = system ? [{ role: 'system', content: system }, ...messages] : messages;
    const ids = encodeChat(eng.tok, eng.kind, convo, { ...templateOptions(eng, flags), think });
    if (flags['show-prompt']) describePrompt(eng, ids);
    signal.aborted = false;
    busy = true;
    try {
      const stats = {};
      stats.prefixCache = await usePrefixCache(eng, flags, system ? [{ role: 'system', content: system }] : [], ids, { ...templateOptions(eng, flags), think }, signal);
      const { text } = await streamReply(eng, ids, { sample, maxTokens, signal, stats });
      messages.push({ role: 'assistant', content: text });
      process.stdout.write('\n');
      process.stderr.write(`${formatStats(eng, stats)}\n`);
      if (stats.stopReason === 'context') process.stderr.write('(context window full: use /reset)\n');
    } catch (e) {
      messages.pop();
      process.stderr.write(`error: ${e.message}\n`);
    }
    busy = false;
    prompt();
  }
  rl.close();
  await eng.close();
  process.exit(0);
}
