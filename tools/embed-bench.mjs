// Speed of the embedding encoder. node tools/embed-bench.mjs [model.gguf] [--threads N] [--seq 300] [--n 24]
// Without a model file a synthetic nomic-embed-text-shaped Q4_K_M model (768 wide, 12 layers, FFN 3072) is generated.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadEmbedder } from '../src/embed/embedder.js';
import { makeTinyEmbedModel } from './make-tiny-embed-gguf.js';
import { parseArgs, num } from '../src/util/args.js';

const { _, flags } = parseArgs(process.argv.slice(2));
let file = _[0]; let tmp = null;
if (!file) { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-bench-')); file = path.join(tmp, 'synthetic.gguf'); fs.writeFileSync(file, makeTinyEmbedModel({ quant: 'q4_k_m', nEmbd: 768, nLayer: 12, nHead: 12, nFF: 3072, ctx: 2048 }).buffer); console.log('(synthetic nomic-embed-text-shaped model)'); }
const seq = num(flags.seq, 300); const count = num(flags.n, 24);
for (const threads of flags.threads ? [num(flags.threads, 1)] : [1, Math.max(1, Math.floor(os.availableParallelism() / 2))]) {
  const emb = await loadEmbedder(file, { threads, maxSeq: Math.max(seq, 64) });
  try {
    const vocab = emb.cfg.vocabSize;
    const ids = Array.from({ length: count }, (_, i) => [2, ...Array.from({ length: seq - 2 }, (__, j) => 5 + ((i * 31 + j * 7) % (vocab - 5))), 3]);
    await emb.embedIds([ids[0]]);
    const t0 = performance.now(); await emb.embedIds(ids); const ms = performance.now() - t0;
    console.log(`threads ${threads}: load ${(emb.loadMs / 1000).toFixed(1)} s | ${count} x ${seq} tokens in ${(ms / 1000).toFixed(1)} s = ${((count * seq) / (ms / 1000)).toFixed(0)} tok/s, ${(ms / count).toFixed(0)} ms per text`);
  } finally { await emb.close(); }
}
if (tmp) fs.rmSync(tmp, { recursive: true, force: true });
