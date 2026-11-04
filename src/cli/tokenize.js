import { readGGUF } from '../gguf/reader.js';
import { Tokenizer } from '../tokenizer/bpe.js';
import { parseArgs } from '../util/args.js';

export const help = 'tokenize <model.gguf> <text> [--no-special]';

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { boolean: ['special'] });
  if (_.length < 2) throw new Error('usage: zerollm tokenize <model.gguf> <text> [--no-special]');
  const tok = Tokenizer.fromGGUF(readGGUF(_[0]));
  const ids = tok.encode(_.slice(1).join(' '), { special: flags.special !== false });
  process.stdout.write(`${ids.length} tokens\n`);
  process.stdout.write(`${JSON.stringify(ids)}\n`);
  process.stdout.write(`${JSON.stringify(ids.map((i) => tok.tokens[i]))}\n`);
  process.stdout.write(`decoded: ${JSON.stringify(tok.decode(ids, { skipControl: false }))}\n`);
}
