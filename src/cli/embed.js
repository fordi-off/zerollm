// zerollm embed <model.gguf> "text 1" "text 2" ...   -> tokens + cosine-similarity matrix
// zerollm embed <model.gguf> --compare documents/x.json [--n 5]
//     re-embeds the first chunks of a document file written by Skibeti37 (llama.cpp vectors) and prints how closely OUR vectors
//     match THEIRS - the ground-truth check for this implementation (expect cosine > 0.99 for the same model file).
import fs from 'node:fs';
import { loadEmbedder, cosine } from '../embed/embedder.js';
import { parseArgs, num } from '../util/args.js';

export const help = 'embed <model.gguf> <text...> | --compare <document.json>';

const fmt = (x) => x.toFixed(3).padStart(7);

export async function run(argv) {
  const { _, flags } = parseArgs(argv, {});
  if (_.length < 1) throw new Error('usage: zerollm embed <model.gguf> "text" "text" ...   or   --compare <documents/x.json> [--n 5] [--prefix "search_document: "]');
  const prefix = typeof flags.prefix === 'string' ? flags.prefix : '';
  const emb = await loadEmbedder(_[0], { threads: flags.threads ? num(flags.threads, 1) : undefined });
  try {
    process.stdout.write(`model: ${emb.dim} dims, ${emb.cfg.nLayer} layers, ${emb.threads} workers, kernels ${emb.kernels}, loaded in ${(emb.loadMs / 1000).toFixed(1)} s\n`);
    if (typeof flags.compare === 'string') {
      const doc = JSON.parse(fs.readFileSync(flags.compare, 'utf8'));
      const chunks = (doc.chunks ?? []).filter((c) => Array.isArray(c.embedding) && c.embedding.length === emb.dim).slice(0, num(flags.n, 5));
      if (!chunks.length) throw new Error(`no chunks with ${emb.dim}-dimensional embeddings in ${flags.compare}`);
      for (const strip of [true, false]) {
        const e2 = strip === true ? emb : await loadEmbedder(_[0], { threads: emb.threads, stripAccents: false });
        try {
          const mine = await e2.embed(chunks.map((c) => prefix + c.text));
          const sims = mine.map((v, i) => cosine(v, Float32Array.from(chunks[i].embedding)));
          process.stdout.write(`accents ${strip ? 'stripped (BERT default)' : 'kept'}: ${sims.map((s) => s.toFixed(4)).join('  ')}   mean ${(sims.reduce((a, b) => a + b, 0) / sims.length).toFixed(4)}\n`);
        } finally { if (e2 !== emb) await e2.close(); }
      }
      process.stdout.write('(1.0000 = identical to the stored vectors; ~0.99+ is expected with the same quantized model file)\n');
      return;
    }
    const texts = _.slice(1);
    if (texts.length < 2) throw new Error('give at least two texts to compare');
    const ids = texts.map((t) => emb.tokenize(prefix + t));
    texts.forEach((t, i) => process.stdout.write(`[${i}] ${ids[i].length} tokens  ${JSON.stringify(t.slice(0, 70))}\n    ${JSON.stringify(emb.tokenizer.decode(ids[i]).slice(0, 90))}\n`));
    const vecs = await emb.embedIds(ids);
    process.stdout.write(`\ncosine similarity:\n${' '.repeat(5)}${texts.map((_, i) => `[${i}]`.padStart(7)).join('')}\n`);
    vecs.forEach((a, i) => process.stdout.write(`[${i}]`.padEnd(5) + vecs.map((b) => fmt(cosine(a, b))).join('') + '\n'));
  } finally { await emb.close(); }
}
