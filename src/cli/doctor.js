import os from 'node:os';
import v8 from 'node:v8';
import { probeWasmFeatures, probeMemoryAlloc, probeJsFeatures } from '../compute/wasm/features.js';
import { formatBytes, printTable, section } from '../util/log.js';

const yn = (b) => (b ? 'yes' : 'NO');

export const help = 'Environment + feature probe + RAM summary';

export async function run() {
  const cpus = os.cpus();
  const js = probeJsFeatures();
  const wasm = probeWasmFeatures();

  section('System');
  printTable([
    ['Node', process.version],
    ['V8', process.versions.v8],
    ['Platform', `${process.platform} ${process.arch} (${os.release()})`],
    ['CPU', cpus[0]?.model?.trim() ?? 'unknown'],
    ['Logical processors', cpus.length],
    ['availableParallelism', js.availableParallelism],
    ['Suggested workers', `${Math.max(1, Math.floor(cpus.length / 2))} (physical cores guess = logical/2; benchmark 4/6/8/12)`],
    ['RAM total', formatBytes(os.totalmem())],
    ['RAM free', formatBytes(os.freemem())],
    ['Heap limit', formatBytes(v8.getHeapStatistics().heap_size_limit)],
  ]);

  section('WebAssembly features');
  printTable([
    ['SIMD (128-bit)', yn(wasm.simd)],
    ['Relaxed SIMD', yn(wasm.relaxedSimd)],
    ['Threads / shared memory / atomics', yn(wasm.threads)],
    ['Multi-memory', yn(wasm.multiMemory)],
    ['Memory64', yn(wasm.memory64)],
    ['Bulk memory', yn(wasm.bulkMemory)],
  ]);

  section('JavaScript features');
  printTable([
    ['SharedArrayBuffer', yn(js.sharedArrayBuffer)],
    ['Atomics.wait (worker side)', yn(js.atomicsWait)],
    ['Atomics.waitAsync', yn(js.atomicsWaitAsync)],
    ['Regex inline modifiers (?i:...)', yn(js.regexInlineModifiers) + (js.regexInlineModifiers ? '' : '  (tokenizer will expand to explicit case classes)')],
    ['Regex \\p{L} / \\p{N}', yn(js.regexUnicodeProps)],
  ]);

  section('WebAssembly.Memory allocation');
  printTable([1, 2, 3].map((gb) => [`${gb} GB`, yn(probeMemoryAlloc(gb))]));

  const problems = [];
  if (!wasm.simd) problems.push('WebAssembly SIMD is not available: the fast kernels cannot run (JS fallback only).');
  if (!js.sharedArrayBuffer || !js.atomicsWait) problems.push('SharedArrayBuffer/Atomics.wait missing: worker pool cannot work.');
  if (problems.length) {
    section('Problems');
    for (const p of problems) process.stdout.write(`- ${p}\n`);
    process.exitCode = 1;
  } else {
    process.stdout.write('\nAll required features present.\n');
  }
}
