import os from 'node:os';
import { readGGUF } from '../gguf/reader.js';
import { typeName } from '../gguf/types.js';
import { modelConfig, estimateRam, UnsupportedArchError } from '../model/config.js';
import { parseArgs, num } from '../util/args.js';
import { formatBytes, formatCount, printTable, section } from '../util/log.js';

export const help = 'info <model.gguf> [--ctx N] [--tensors] [--kv] [--full-template]';

function summarize(v) {
  if (Array.isArray(v)) {
    const head = v.slice(0, 4).map((x) => JSON.stringify(x)).join(', ');
    return `[array len ${v.length}: ${head}${v.length > 4 ? ', ...' : ''}]`;
  }
  if (typeof v === 'string') return JSON.stringify(v.length > 120 ? `${v.slice(0, 120)}...(${v.length} chars)` : v);
  return String(v);
}

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { boolean: ['tensors', 'kv', 'full-template'] });
  if (!_[0]) throw new Error('usage: zerollm info <model.gguf> [--ctx N] [--tensors] [--kv] [--full-template]');
  const g = readGGUF(_[0]);

  section('File');
  printTable([
    ['Path', g.path],
    ['Size', formatBytes(g.fileSize)],
    ['GGUF version', g.version],
    ['Tensors', g.tensors.length],
    ['Metadata keys', g.metadata.size],
    ['Alignment', g.alignment],
    ['Data starts at', g.dataStart],
    ['Name', g.get('general.name', '(none)')],
  ]);

  /** @type {ReturnType<typeof modelConfig>|null} */
  let cfg = null;
  try {
    cfg = modelConfig(g);
  } catch (e) {
    if (!(e instanceof UnsupportedArchError)) throw e;
    process.stdout.write(`\nnote: ${e.message}\n`);
  }

  if (cfg) {
    section('Hyperparameters');
    printTable([
      ['Architecture', cfg.arch],
      ['Context (trained)', cfg.nCtxTrain],
      ['Hidden size', cfg.nEmbd],
      ['Layers', cfg.nLayer],
      ['FFN size', cfg.nFF],
      ['Heads / KV heads', `${cfg.nHead} / ${cfg.nHeadKv}`],
      ['Head dim', `${cfg.headDim}${cfg.headDim !== cfg.nEmbd / cfg.nHead ? `  (!= hidden/heads = ${cfg.nEmbd / cfg.nHead})` : ''}`],
      ['RoPE', `${cfg.ropeType === 'neox' ? 'NEOX (split-half)' : 'normal (adjacent pairs)'}, base ${cfg.ropeBase}, dims ${cfg.nRot}${cfg.hasRopeFreqs ? ', + rope_freqs.weight' : ''}`],
      ['RMSNorm eps', Number(cfg.rmsEps.toPrecision(6))],
      ['Vocab size', cfg.vocabSize],
      ['QK-norm', cfg.qkNorm ? 'yes' : 'no'],
      ['QKV bias', cfg.qkvBias ? 'yes' : 'no'],
      ['Tied embeddings', cfg.tiedEmbeddings ? 'yes (no output.weight)' : 'no'],
    ]);
  }

  section('Quantization (by tensor type)');
  const hist = new Map();
  let totalBytes = 0;
  let totalParams = 0;
  for (const t of g.tensors) {
    const h = hist.get(t.type) ?? { count: 0, bytes: 0, elems: 0 };
    h.count++; h.bytes += t.nBytes ?? 0; h.elems += t.nElements;
    hist.set(t.type, h);
    totalBytes += t.nBytes ?? 0; totalParams += t.nElements;
  }
  for (const [type, h] of [...hist].sort((a, b) => b[1].bytes - a[1].bytes)) {
    const bpw = h.elems ? ((h.bytes * 8) / h.elems).toFixed(2) : '-';
    process.stdout.write(`${typeName(type).padEnd(8)} ${String(h.count).padStart(5)} tensors  ${formatBytes(h.bytes).padStart(10)}  ${((100 * h.bytes) / totalBytes).toFixed(1).padStart(5)}%  ${bpw} bits/weight\n`);
  }
  process.stdout.write(`Total: ${formatBytes(totalBytes)} of tensor data, ${formatCount(totalParams)} parameters\n`);

  section('Tokenizer');
  const tokens = g.get('tokenizer.ggml.tokens');
  const types = g.get('tokenizer.ggml.token_type');
  const merges = g.get('tokenizer.ggml.merges');
  const ctrl = Array.isArray(types) ? types.filter((t) => t === 3).length : 0;
  const user = Array.isArray(types) ? types.filter((t) => t === 4).length : 0;
  const tokStr = (id) => (id !== undefined && Array.isArray(tokens) ? `${id} ${JSON.stringify(tokens[id])}` : '(none)');
  printTable([
    ['Model', g.get('tokenizer.ggml.model', '(none)')],
    ['Pre-tokenizer', g.get('tokenizer.ggml.pre', '(none)')],
    ['Vocab', Array.isArray(tokens) ? tokens.length : '(none)'],
    ['Merges', Array.isArray(merges) ? merges.length : '(none)'],
    ['Control / user-defined tokens', Array.isArray(types) ? `${ctrl} / ${user}` : '(no token_type)'],
    ['BOS', tokStr(g.get('tokenizer.ggml.bos_token_id'))],
    ['EOS', tokStr(g.get('tokenizer.ggml.eos_token_id'))],
    ['Add BOS', g.get('tokenizer.ggml.add_bos_token', '(unspecified)')],
  ]);

  const tpl = g.get('tokenizer.chat_template');
  section('Chat template');
  if (typeof tpl !== 'string') process.stdout.write('(none)\n');
  else if (flags['full-template']) process.stdout.write(`${tpl}\n`);
  else process.stdout.write(`${tpl.length} chars: ${JSON.stringify(tpl.slice(0, 160))}${tpl.length > 160 ? '...  (use --full-template)' : ''}\n`);

  const ctx = num(flags.ctx, Math.min(2048, cfg?.nCtxTrain ?? 2048));
  const est = estimateRam(g, cfg, { ctx });
  section(`RAM estimate (ctx ${ctx}, f32 KV cache)`);
  printTable([
    ['Weights', formatBytes(est.weights)],
    ['KV cache', formatBytes(est.kv)],
    ['Scratch + runtime', formatBytes(est.scratch + est.runtime)],
    ['Total', formatBytes(est.total)],
    ['Free RAM now', formatBytes(os.freemem())],
  ]);
  if (est.total > os.freemem() * 0.9) process.stdout.write('WARNING: estimated need is close to / above free RAM. Close heavy apps or use a smaller model/context.\n');

  if (flags.kv) {
    section('Metadata');
    for (const [k, v] of g.metadata) process.stdout.write(`${k} = ${summarize(v)}\n`);
  }
  if (flags.tensors) {
    section('Tensors');
    for (const t of g.tensors) process.stdout.write(`${t.name.padEnd(32)} ${typeName(t.type).padEnd(7)} [${t.dims.join(', ')}]  ${formatBytes(t.nBytes ?? 0)}\n`);
  }
}
