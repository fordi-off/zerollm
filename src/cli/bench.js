import { readGGUF } from '../gguf/reader.js';
import { modelConfig } from '../model/config.js';
import { Tokenizer } from '../tokenizer/bpe.js';
import { generate } from '../model/generate.js';
import { parseArgs, num } from '../util/args.js';
import { formatBytes } from '../util/log.js';
import { createModel, defaultThreads } from './common.js';

export const help = 'bench <model.gguf> [--threads 1,2,4,6] [--prompt-tokens 32] [--n 32] [--ctx 512] [--spin N]';

const TEXT = 'The quick brown fox jumps over the lazy dog. In 1969, humans first walked on the Moon, and computers of the era had less memory than a modern greeting card. ';

export async function run(argv) {
  const { _, flags } = parseArgs(argv);
  if (!_[0]) throw new Error(`usage: zerollm ${help}`);
  const g = readGGUF(_[0]);
  const cfg = modelConfig(g);
  const tok = Tokenizer.fromGGUF(g);
  const list = typeof flags.threads === 'string' ? flags.threads.split(',').map(Number) : [1, defaultThreads()];
  const P = num(flags['prompt-tokens'], 32);
  const N = num(flags.n, 32);
  const ctx = num(flags.ctx, Math.max(512, P + N + 8));
  let ids = [];
  while (ids.length < P) ids = ids.concat(tok.encode(TEXT, { special: false, bos: false }));
  ids = ids.slice(0, P);
  const weightBytes = g.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);

  process.stdout.write(`model ${cfg.arch} ${cfg.nLayer}L, ${formatBytes(weightBytes)} of weights, prompt ${P} tok, generate ${N} tok, ctx ${ctx}\n\n`);
  process.stdout.write('| threads | load s | prefill tok/s | decode tok/s | eff. GB/s | speedup | RSS |\n|---|---|---|---|---|---|---|\n');
  let base = null; let baseOut = null; let allSame = true;
  for (const threads of list) {
    const t0 = performance.now();
    const model = await createModel(g, { threads, ctx, spin: flags.spin === undefined ? undefined : num(flags.spin, 20000) });
    const loadS = (performance.now() - t0) / 1000;
    const stats = {}; const out = [];
    for await (const id of generate(model, ids, { maxTokens: N, stats })) out.push(id);
    const rss = process.memoryUsage().rss;
    await model.close?.();
    const prefill = (stats.prefilled * 1000) / stats.prefillMs;
    const decode = (stats.generated * 1000) / stats.decodeMs;
    base ??= decode;
    if (baseOut) { if (out.join() !== baseOut.join()) allSame = false; } else baseOut = out;
    process.stdout.write(`| ${threads} | ${loadS.toFixed(1)} | ${prefill.toFixed(2)} | ${decode.toFixed(2)} | ${((decode * weightBytes) / 1e9).toFixed(2)} | ${(decode / base).toFixed(2)}x | ${formatBytes(rss)} |\n`);
  }
  process.stdout.write(`\ngreedy output identical across thread counts: ${allSame ? 'yes' : 'NO (bug!)'}\n`);
  if (!allSame) process.exitCode = 1;
}
