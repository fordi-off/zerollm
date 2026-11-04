#!/usr/bin/env node
// zerollm CLI entry. Commands are lazily imported from src/cli/<name>.js.
import { die } from '../src/util/log.js';

const COMMANDS = {
  doctor: 'Environment + feature probe + RAM summary',
  info: 'Print GGUF metadata, hyperparameters, quant histogram, RAM estimate',
  tokenize: 'Tokenize text with the model\'s tokenizer (debug)',
  run: 'Generate text from a prompt',
  chat: 'Interactive chat',
  bench: 'Benchmark prefill + decode',
  embed: 'Embedding model check: similarities of texts, or compare with vectors stored by another app (--compare documents/x.json)',
  serve: 'OpenAI-compatible HTTP server',
};

function usage() {
  process.stdout.write('usage: zerollm <command> [options]\n\ncommands:\n');
  for (const [k, v] of Object.entries(COMMANDS)) process.stdout.write(`  ${k.padEnd(10)} ${v}\n`);
}

const [cmd, ...rest] = process.argv.slice(2);
if (!cmd || cmd === '-h' || cmd === '--help' || cmd === 'help') { usage(); process.exit(cmd ? 0 : 1); }
if (!(cmd in COMMANDS)) { usage(); die(`unknown command: ${cmd}`); }

let mod;
try {
  mod = await import(`../src/cli/${cmd}.js`);
} catch (e) {
  if (e?.code === 'ERR_MODULE_NOT_FOUND' && String(e.message).includes(`cli/${cmd}.js`)) die(`command "${cmd}" is not implemented yet`);
  throw e;
}
try {
  await mod.run(rest);
} catch (e) {
  die(e?.message ?? String(e));
}
