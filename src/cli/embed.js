// zerollm embed <model.gguf> "text 1" "text 2" ...   -> tokens + cosine-similarity matrix
// zerollm embed <model.gguf> --selftest
//     built-in sentence triples (anchor, paraphrase, unrelated): does the model rank them sensibly? Catches gross mistakes only.
// zerollm embed <model.gguf> --compare <file.json> [--n 5]
//     re-embeds texts whose vectors were made by ANOTHER implementation and prints how closely ours match theirs - the real
//     ground-truth check (expect cosine > 0.99 for the same model). <file.json> is a Skibeti37 document ({chunks: [{text, embedding}]})
//     or a plain list [{text, embedding}] such as tools/ollama-reference.mjs writes.
import fs from 'node:fs';
import { loadEmbedder, cosine } from '../embed/embedder.js';
import { parseArgs, num } from '../util/args.js';

export const help = 'embed <model.gguf> <text...> | --selftest | --compare <reference.json>';

const TRIPLES = [
  ['Hva er hovedstaden i Norge?', 'Oslo er Norges hovedstad.', 'Pizza bakes i en steinovn.'],
  ['How do I reset my password?', 'Steps to change your account password', 'The mountain was covered in snow'],
  ['Kommunen har ansvar for barnehager og skoler.', 'Lokale myndigheter driver grunnskolen.', 'Motoren bruker bensin og diesel.'],
  ['The cat sat on the mat.', 'A kitten is resting on a rug.', 'Quarterly revenue increased by ten percent.'],
  ['Python is a programming language.', 'Code in Python uses indentation.', 'Whales are large marine mammals.'],
  ['Hvordan søker jeg om barnehageplass?', 'Slik får du plass i barnehage for barnet ditt.', 'Fotball-VM ble spilt i november.'],
];

const fmt = (x) => x.toFixed(3).padStart(7);

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { boolean: ['selftest'] });
  if (_.length < 1) throw new Error('usage: zerollm embed <model.gguf> "text" "text" ...   or   --compare <documents/x.json> [--n 5] [--prefix "search_document: "]');
  const prefix = typeof flags.prefix === 'string' ? flags.prefix : '';
  const emb = await loadEmbedder(_[0], { threads: flags.threads ? num(flags.threads, 1) : undefined });
  try {
    process.stdout.write(`model: ${emb.dim} dims, ${emb.cfg.nLayer} layers, ${emb.threads} workers, kernels ${emb.kernels}, loaded in ${(emb.loadMs / 1000).toFixed(1)} s\n`);
    if (flags.selftest) {
      const flat = TRIPLES.flat(); const vecs = await emb.embed(flat);
      let ok = 0; let marginSum = 0;
      TRIPLES.forEach((tr, i) => {
        const [a, n, f] = [vecs[i * 3], vecs[i * 3 + 1], vecs[i * 3 + 2]];
        const near = cosine(a, n); const far = cosine(a, f); const pass = near > far + 0.02; if (pass) ok++; marginSum += near - far;
        process.stdout.write(`${pass ? 'ok  ' : 'FAIL'} related ${near.toFixed(3)}  unrelated ${far.toFixed(3)}  ${JSON.stringify(tr[0].slice(0, 44))}\n`);
      });
      process.stdout.write(`\n${ok}/${TRIPLES.length} triples ranked correctly, mean margin ${(marginSum / TRIPLES.length).toFixed(3)}\n`);
      process.stdout.write(ok === TRIPLES.length && marginSum / TRIPLES.length > 0.08 ? 'Looks like a working embedding model.\n' : 'SUSPICIOUS: a working nomic-embed-text ranks all of these correctly with clear margins (typically 0.1-0.3). Send me this output.\n');
      return;
    }
    if (typeof flags.compare === 'string') {
      const doc = JSON.parse(fs.readFileSync(flags.compare, 'utf8'));
      const chunks = (Array.isArray(doc) ? doc : doc.chunks ?? []).filter((c) => Array.isArray(c.embedding) && c.embedding.length === emb.dim).slice(0, num(flags.n, 5));
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
