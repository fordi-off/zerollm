import { readGGUF } from '../gguf/reader.js';
import { modelConfig, assertSupportedTensorTypes } from '../model/config.js';
import { Tokenizer } from '../tokenizer/bpe.js';
import fs from 'node:fs';
import { generate } from '../model/generate.js';
import { generateSpeculative, defaultDraftThreads, DEFAULT_DRAFT_K } from '../model/speculative.js';
import { createSampler } from '../sampling/sampler.js';
import { parseArgs, num } from '../util/args.js';
import { formatBytes } from '../util/log.js';
import { createModel, defaultThreads, kvFromFlags, checkDraftFile } from './common.js';

export const help = 'bench <model.gguf> [--threads 1,2,4,6] [--kernels js,wasm] [--batch 1,16] [--kv f32|f16] [--prompt-tokens 32] [--prompt-file text.txt] [--n 32] [--ctx 512] [--spin N]'
  + ' [--draft draft.gguf --draft-k 2,4,6 --draft-threads N [--temp 0.7 --seed N]]  (--draft: also measure speculative decoding against the plain decode of the same configuration)';

const TEXT = 'The quick brown fox jumps over the lazy dog. In 1969, humans first walked on the Moon, and computers of the era had less memory than a modern greeting card. ';

export async function run(argv) {
  const { _, flags } = parseArgs(argv);
  if (!_[0]) throw new Error(`usage: zerollm ${help}`);
  const g = readGGUF(_[0]);
  const cfg = modelConfig(g);
  assertSupportedTensorTypes(g);
  const tok = Tokenizer.fromGGUF(g);
  const list = typeof flags.threads === 'string' ? flags.threads.split(',').map(Number) : [1, defaultThreads()];
  const kernelList = typeof flags.kernels === 'string' ? flags.kernels.split(',') : ['auto'];
  for (const k of kernelList) if (!['auto', 'wasm', 'js'].includes(k)) throw new Error(`bad --kernels entry "${k}"`);
  const batchList = typeof flags.batch === 'string' ? flags.batch.split(',').map(Number) : [undefined];
  const P = num(flags['prompt-tokens'], 32);
  const N = num(flags.n, 32);
  const ctx = num(flags.ctx, Math.max(512, P + N + 8));
  const text = typeof flags['prompt-file'] === 'string' ? fs.readFileSync(flags['prompt-file'], 'utf8') : TEXT;
  let ids = [];
  while (ids.length < P) ids = ids.concat(tok.encode(text, { special: false, bos: false }));
  ids = ids.slice(0, P);
  // speculative decoding: compare against plain decoding of the same configuration (greedy unless --temp is given)
  const draftPath = typeof flags.draft === 'string' ? flags.draft : null;
  if (flags.draft !== undefined && !draftPath) throw new Error('--draft needs a model path');
  const draftFile = draftPath ? checkDraftFile(g, draftPath) : null;
  const kList = draftPath ? String(flags['draft-k'] ?? DEFAULT_DRAFT_K).split(',').map(Number) : [];
  for (const k of kList) if (!Number.isInteger(k) || k < 1) throw new Error(`bad --draft-k entry "${k}"`);
  const temp = num(flags.temp, 0);
  const mkSampler = () => createSampler({ temperature: temp, seed: num(flags.seed, 1), topK: num(flags['top-k'], 40), topP: num(flags['top-p'], 0.9) });
  const specRows = [];
  const weightBytes = g.tensors.reduce((s, t) => s + (t.nBytes ?? 0), 0);

  process.stdout.write(`model ${cfg.arch} ${cfg.nLayer}L, ${formatBytes(weightBytes)} of weights, prompt ${P} tok, generate ${N} tok, ctx ${ctx}\n\n`);
  process.stdout.write('| kernels | threads | batch | load s | prefill tok/s | decode tok/s | prefill/decode | eff. GB/s | RSS |\n|---|---|---|---|---|---|---|---|---|\n');
  let base = null; let baseOut = null; let allSame = true;
  for (const kernels of kernelList) for (const threads of list) for (const maxBatch of batchList) {
    const t0 = performance.now();
    const model = await createModel(g, { threads, ctx, kernels, maxBatch, kv: kvFromFlags(flags), spin: flags.spin === undefined ? undefined : num(flags.spin, 20000) });
    const kname = model.kernels === 'wasm' ? 'wasm-simd' : 'js';
    const loadS = (performance.now() - t0) / 1000;
    const stats = {}; const out = [];
    for await (const id of generate(model, ids, { maxTokens: N, stats, sample: draftPath ? mkSampler() : undefined })) out.push(id);
    const rss = process.memoryUsage().rss;
    // speculative runs of this configuration (the target stays loaded; the draft is loaded once per configuration)
    if (draftFile && (model.maxBatch ?? 2) >= 2) { // (batch 1 cannot verify several tokens per pass: skipped)
      const dthreads = flags['draft-threads'] === undefined ? defaultDraftThreads(threads) : num(flags['draft-threads'], 1);
      const dmodel = await createModel(draftFile.g, { threads: dthreads, ctx, kernels, maxBatch, kv: kvFromFlags(flags), spin: flags.spin === undefined ? undefined : num(flags.spin, 20000) });
      try {
        for (const k of kList) {
          const sst = {}; const sout = [];
          model.reset(); dmodel.reset();
          for await (const id of generateSpeculative(model, dmodel, ids, { maxTokens: N, stats: sst, k, sample: mkSampler() })) sout.push(id);
          specRows.push({ kname, threads, dthreads, k, base: (stats.generated * 1000) / stats.decodeMs, eff: (sst.generated * 1000) / sst.decodeMs, sp: sst.speculative, same: sout.join() === out.join(), generated: sst.generated, decodeMs: sst.decodeMs });
        }
      } finally { await dmodel.close?.(); }
    }
    await model.close?.();
    const prefill = (stats.prefilled * 1000) / stats.prefillMs;
    const decode = (stats.generated * 1000) / stats.decodeMs;
    base ??= decode;
    const bsz = model.maxBatch ?? 1;
    if (baseOut) { if (out.join() !== baseOut.join()) allSame = false; } else baseOut = out;
    process.stdout.write(`| ${kname} | ${threads} | ${bsz} | ${loadS.toFixed(1)} | ${prefill.toFixed(2)} | ${decode.toFixed(2)} | ${(prefill / decode).toFixed(2)}x | ${((decode * weightBytes) / 1e9).toFixed(2)} | ${formatBytes(rss)} |\n`);
  }
  if (specRows.length) {
    process.stdout.write(`\nspeculative decoding, draft ${draftPath} (${draftFile.cfg.arch} ${draftFile.cfg.nLayer}L), ${temp > 0 ? `sampled at temperature ${temp}` : 'greedy'}, same prompt as above\n\n`);
    process.stdout.write('| kernels | threads | draft threads | k | plain decode tok/s | speculative tok/s | speed-up | accepted | tok/step | draft ms/step | verify ms/step | output same as plain |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n');
    for (const r of specRows) {
      const c = Math.max(1, r.sp.cycles);
      process.stdout.write(`| ${r.kname} | ${r.threads} | ${r.dthreads} | ${r.k} | ${r.base.toFixed(2)} | ${r.eff.toFixed(2)} | ${(r.eff / r.base).toFixed(2)}x | ${(100 * r.sp.acceptRate).toFixed(0)}% | ${r.sp.tokensPerCycle.toFixed(2)} | ${(r.sp.draftMs / c).toFixed(1)} | ${(r.sp.verifyMs / c).toFixed(1)} | ${temp > 0 ? 'n/a (sampled)' : r.same ? 'yes' : 'NO'} |\n`);
    }
    process.stdout.write('\nNote: acceptance depends heavily on the text; the built-in prompt repeats a sentence and is NOT representative. Use --prompt-file with natural text and a long --n for a realistic figure.\n');
  }
  process.stdout.write(`\ngreedy output identical across all configurations: ${allSame ? 'yes' : 'NO (differs: expected only for SIMD vs JS float rounding on near-ties; see docs)'}\n`);
}
