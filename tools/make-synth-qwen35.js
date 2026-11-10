#!/usr/bin/env node
// Writes a SYNTHETIC GGUF with the exact tensor set, shapes and quantization types of the real Qwen3.5-4B-Q4_K_M file
// (test-fixtures/qwen35-4b-tensors.txt, from the real file's `info --tensors`) and random weights, for speed measurements and
// memory checks of the pool path without the real model. The numbers it produces are timing numbers: the "model" says nothing
// (random weights), and the file has no tokenizer (use tools/bench-qwen35-pool.mjs, not the `bench` CLI, which needs one).
// Identical random blocks are reused for every tensor of the same type and size (timing does not depend on the values).
//   node tools/make-synth-qwen35.js out.gguf [--layers 32] [--vocab 248320] [--seed 1]
// The hyperparameters are the real file's: 2560 hidden, 9216 FFN, 16 heads x 256 (+gate), 4 KV heads, rotary 64, 16 key / 32 value
// heads x 128, conv kernel 4, attention every 4th layer, tied output (Q6_K token_embd, 248320 rows).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeGGUF } from './gguf-writer.js';
import { GGUF_TYPE, GGML_TYPE, GGML_TYPE_INFO } from '../src/gguf/types.js';
import { quantizeQ8_0 } from '../src/compute/quant.js';
import { randomQuantBlocks } from './random-blocks.js';
import { mulberry32, gaussian } from '../src/util/rng.js';
import { parseArgs, num } from '../src/util/args.js';

const FIXTURE = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'test-fixtures', 'qwen35-4b-tensors.txt');
const TYPES = { F32: GGML_TYPE.F32, Q4_K: GGML_TYPE.Q4_K, Q5_K: GGML_TYPE.Q5_K, Q6_K: GGML_TYPE.Q6_K, Q8_0: GGML_TYPE.Q8_0 };

/**
 * @param {{layers?: number, vocab?: number, seed?: number, outFile: string}} o
 * @returns {{bytes: number, tensors: number}}
 */
export function writeSynthQwen35(o) {
  const layers = o.layers ?? 32; const vocab = o.vocab ?? 248320;
  const rand = mulberry32(o.seed ?? 1);
  const cache = new Map();
  const quantBlocks = (type, nElem) => {
    const key = `${type}:${nElem}`;
    if (!cache.has(key)) {
      if (type === GGML_TYPE.Q8_0) cache.set(key, quantizeQ8_0(Float32Array.from({ length: nElem }, () => gaussian(rand) * 0.05)));
      else cache.set(key, randomQuantBlocks(type, nElem / GGML_TYPE_INFO[type].blockSize, rand));
    }
    return cache.get(key);
  };
  const f32 = (name, n) => {
    let f;
    if (/ssm_a$/.test(name)) f = Float32Array.from({ length: n }, () => -(0.1 + rand())); // -exp(A_log)
    else if (/ssm_dt\.bias$/.test(name)) f = Float32Array.from({ length: n }, () => gaussian(rand) * 0.5);
    else if (/ssm_conv1d/.test(name)) f = Float32Array.from({ length: n }, () => gaussian(rand) * 0.3);
    else f = Float32Array.from({ length: n }, () => 1 + gaussian(rand) * 0.05); // norm weights (the converter's +1 included)
    return new Uint8Array(f.buffer);
  };
  const tensors = [];
  for (const line of fs.readFileSync(FIXTURE, 'utf8').split(/\r?\n/)) {
    if (!line || line.startsWith('#')) continue;
    const m = /^(\S+) (\S+) \[(.*)\]$/.exec(line);
    let [, name, type, dims] = m;
    dims = dims.split(',').map(Number);
    const layer = /^blk\.(\d+)\./.exec(name);
    if (layer && Number(layer[1]) >= layers) continue;
    if (name === 'token_embd.weight') dims = [dims[0], vocab];
    const nElem = dims.reduce((a, b) => a * b, 1);
    const t = TYPES[type];
    if (t === undefined) throw new Error(`unknown type ${type}`);
    tensors.push({ name, dims, type: t, data: t === GGML_TYPE.F32 ? f32(name, nElem) : quantBlocks(t, nElem) });
  }
  const A = 'qwen35'; const U32 = GGUF_TYPE.UINT32;
  const kv = (key, type, value, elemType) => ({ key, type, value, elemType });
  const metadata = [
    kv('general.architecture', GGUF_TYPE.STRING, A), kv('general.name', GGUF_TYPE.STRING, `synthetic-qwen35-4b-shapes-${layers}L`),
    kv(`${A}.context_length`, U32, 262144), kv(`${A}.embedding_length`, U32, 2560), kv(`${A}.block_count`, U32, layers),
    kv(`${A}.feed_forward_length`, U32, 9216), kv(`${A}.attention.head_count`, U32, 16), kv(`${A}.attention.head_count_kv`, U32, 4),
    kv(`${A}.attention.key_length`, U32, 256), kv(`${A}.attention.value_length`, U32, 256),
    kv(`${A}.attention.layer_norm_rms_epsilon`, GGUF_TYPE.FLOAT32, 1e-6), kv(`${A}.rope.freq_base`, GGUF_TYPE.FLOAT32, 1e7),
    kv(`${A}.rope.dimension_count`, U32, 64), kv(`${A}.rope.dimension_sections`, GGUF_TYPE.ARRAY, [11, 11, 10, 0], GGUF_TYPE.INT32),
    kv(`${A}.ssm.conv_kernel`, U32, 4), kv(`${A}.ssm.state_size`, U32, 128), kv(`${A}.ssm.group_count`, U32, 16),
    kv(`${A}.ssm.time_step_rank`, U32, 32), kv(`${A}.ssm.inner_size`, U32, 4096), kv(`${A}.full_attention_interval`, U32, 4),
    kv(`${A}.vocab_size`, U32, vocab),
  ];
  writeGGUF(o.outFile, { metadata, tensors, alignment: 32 });
  return { bytes: fs.statSync(o.outFile).size, tensors: tensors.length };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { _, flags } = parseArgs(process.argv.slice(2), {});
  if (!_[0]) { console.error('usage: make-synth-qwen35.js out.gguf [--layers 32] [--vocab 248320] [--seed 1]'); process.exit(1); }
  const t0 = performance.now();
  const r = writeSynthQwen35({ outFile: _[0], layers: num(flags.layers, 32), vocab: num(flags.vocab, 248320), seed: num(flags.seed, 1) });
  console.log(`wrote ${_[0]}: ${r.tensors} tensors, ${(r.bytes / 1e9).toFixed(2)} GB in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
