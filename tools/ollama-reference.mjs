// Ground truth for the embedding encoder WITHOUT Skibeti37: ask a locally running Ollama (https://ollama.com, `ollama pull nomic-embed-text`)
// for embeddings of some texts and save them; then compare ours:
//   node tools/ollama-reference.mjs reference.json
//   node bin/zerollm.js embed models/nomic-embed-text-v1.5.Q4_K_M.gguf --compare reference.json
// Only talks to 127.0.0.1:11434 (Ollama's local API). Ollama's nomic-embed-text is v1.5 in F16, so expect ~0.99+ against the Q4_K_M file.
import fs from 'node:fs';
import { parseArgs } from '../src/util/args.js';

const TEXTS = [
  'Hva er hovedstaden i Norge?', 'Oslo er Norges hovedstad.', 'The cat sat on the mat.', 'Quarterly revenue increased by ten percent.',
  'Kommunen har ansvar for barnehager og skoler. Kommunene finansieres blant annet gjennom skatt og overføringer fra staten.',
  'Å bo på Ålesund var fint; særlig om sommeren, når solen står opp tidlig og går ned sent.',
  'Python is a programming language. Code in Python uses indentation to group statements, and functions are defined with def.',
  'SELECT name, price FROM products WHERE price > 100 ORDER BY price DESC;',
];
const { _, flags } = parseArgs(process.argv.slice(2), {});
const out = _[0] ?? 'reference.json';
const model = typeof flags.model === 'string' ? flags.model : 'nomic-embed-text';
const base = typeof flags.url === 'string' ? flags.url : 'http://127.0.0.1:11434';
const result = [];
for (const text of TEXTS) {
  const res = await fetch(`${base}/api/embeddings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, prompt: text }) });
  if (!res.ok) throw new Error(`Ollama answered ${res.status}: ${await res.text()} (is it running, and did you run "ollama pull ${model}"?)`);
  result.push({ text, embedding: (await res.json()).embedding });
  process.stdout.write('.');
}
fs.writeFileSync(out, JSON.stringify(result));
console.log(`\nwrote ${out} (${result.length} texts, ${result[0].embedding.length} dims). Now run: node bin/zerollm.js embed <model.gguf> --compare ${out} --n ${result.length}`);
