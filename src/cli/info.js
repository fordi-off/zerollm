import os from 'node:os';
import { readGGUF } from '../gguf/reader.js';
import { typeName } from '../gguf/types.js';
import { modelConfig, estimateRam, UnsupportedArchError, SUPPORTED_TENSOR_TYPES, RECURRENT_CHECKPOINTS } from '../model/config.js';
import { parseArgs, num } from '../util/args.js';
import { defaultThreads } from './common.js';
import { formatBytes, formatCount, printTable, section } from '../util/log.js';

export const help = 'info <model.gguf> [--ctx N] [--threads N] [--tensors] [--metadata] [--full-template]';

function summarize(v) {
  if (Array.isArray(v)) {
    const head = v.slice(0, 4).map((x) => JSON.stringify(x)).join(', ');
    return `[array len ${v.length}: ${head}${v.length > 4 ? ', ...' : ''}]`;
  }
  if (typeof v === 'string') return JSON.stringify(v.length > 120 ? `${v.slice(0, 120)}...(${v.length} chars)` : v);
  return String(v);
}

export async function run(argv) {
  const { _, flags } = parseArgs(argv, { boolean: ['tensors', 'kv', 'metadata', 'full-template'] });
  if (!_[0]) throw new Error('usage: zerollm info <model.gguf> [--ctx N] [--threads N] [--tensors] [--metadata] [--full-template]');
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
    // still print everything else we can (quantization table, tokenizer, ...) for unsupported or incomplete files
    process.stdout.write(`\nnote: ${e instanceof UnsupportedArchError ? e.message : `could not read the hyperparameters (${e.message})`}\n`);
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

  if (cfg?.hybrid) {
    const attn = cfg.layerIsRecurrent.flatMap((r, i) => (r ? [] : [i]));
    const s = cfg.ssm;
    section('Hybrid layout');
    printTable([
      ['Layers', `${cfg.nLayer}: ${cfg.nLayer - attn.length} recurrent (Gated DeltaNet) + ${attn.length} full attention`],
      ['Full attention layers', attn.length > 10 ? `${attn.slice(0, 4).join(', ')}, ... , ${attn[attn.length - 1]}` : attn.join(', ')],
      ['Attention', `${cfg.nHead} heads x ${cfg.headDim} (+ per-head output gate), ${cfg.nHeadKv} KV heads, rotary on ${cfg.nRot} of ${cfg.headDim} dims${cfg.ropeSections ? `, M-RoPE sections [${cfg.ropeSections.join(', ')}]` : ''}`],
      ['Gated DeltaNet', `${s.nKHeads} key heads x ${s.headKDim}, ${s.nVHeads} value heads x ${s.headVDim}, conv kernel ${s.convKernel} over ${s.convDim} channels`],
      ['Value -> key head', `value head h reads key head h % ${s.nKHeads} (llama.cpp tiled layout; UNVERIFIED on real files)`],
      ['Support', 'worker pool + WASM SIMD kernels; --threads 1 --kernels js is the single-thread reference path'],
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
    process.stdout.write(`${typeName(type).padEnd(8)} ${String(h.count).padStart(5)} tensors  ${formatBytes(h.bytes).padStart(10)}  ${((100 * h.bytes) / totalBytes).toFixed(1).padStart(5)}%  ${bpw} bits/weight${SUPPORTED_TENSOR_TYPES.has(type) ? '' : '   <- NOT supported by this engine yet'}\n`);
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
  const threads = num(flags.threads, defaultThreads());
  const e32 = estimateRam(g, cfg, { ctx, kvBytes: 4, threads });
  const e16 = estimateRam(g, cfg, { ctx, kvBytes: 2, threads });
  section(`RAM estimate (ctx ${ctx}, ${threads} threads)`);
  printTable([
    ['Weights', formatBytes(e32.weights)],
    [cfg?.hybrid ? `KV cache, f32 (default; ${cfg.nAttnLayers} attention layers only)` : 'KV cache, f32 (default)', formatBytes(e32.kv)],
    ['KV cache, f16 (--kv f16)', formatBytes(e16.kv)],
    ...(cfg?.hybrid ? [
      ['Recurrent state (1 sequence, f32)', formatBytes(e32.recurrent)],
      [`State checkpoints (up to ${RECURRENT_CHECKPOINTS})`, formatBytes(e32.checkpoints)],
    ] : []),
    ['Scratch + runtime', formatBytes(e32.scratch + e32.runtime)],
    ['Total with f32 KV', formatBytes(e32.total)],
    ['Total with f16 KV', formatBytes(e16.total)],
    ['Busiest worker memory (f32 KV)', `${formatBytes(e32.perWorker)}  (wasm32 limit 4 GB)`],
    ['Free RAM now', formatBytes(os.freemem())],
  ]);
  if (e32.total > os.freemem() * 0.9) {
    process.stdout.write(e16.total <= os.freemem() * 0.9
      ? 'WARNING: tight on RAM with an f32 KV cache. --kv f16 should fit. Close heavy apps to be safe.\n'
      : 'WARNING: estimated need is close to / above free RAM even with --kv f16. Close heavy apps, lower --ctx, or use a smaller model/quant.\n');
  }

  if (flags.kv || flags.metadata) {
    section('Metadata');
    for (const [k, v] of g.metadata) process.stdout.write(`${k} = ${summarize(v)}\n`);
  }
  if (flags.tensors) {
    section('Tensors');
    for (const t of g.tensors) process.stdout.write(`${t.name.padEnd(32)} ${typeName(t.type).padEnd(7)} [${t.dims.join(', ')}]  ${formatBytes(t.nBytes ?? 0)}\n`);
  }
}
