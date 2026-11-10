import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { makeTinyQwen35, reorderRows } from '../tools/make-tiny-qwen35-gguf.js';
import { naiveQwen35Logits, hfFromGguf } from './naive-qwen35.js';
import { readGGUF } from '../src/gguf/reader.js';
import { GGML_TYPE } from '../src/gguf/types.js';
import { modelConfig, estimateRam, kvBytesPerToken, recurrentStateBytes, RECURRENT_CHECKPOINTS, SUPPORTED_ARCHS } from '../src/model/config.js';
import { RefModel } from '../src/model/forward.js';
import { Qwen35Model, RecurrentState, expectedTensors } from '../src/model/forward-qwen35.js';
import { generate, argmax } from '../src/model/generate.js';
import { PrefixCache, modelFingerprint } from '../src/model/prefix-cache.js';
import { Tokenizer, preTokenize } from '../src/tokenizer/bpe.js';
import { renderSegments, encodeChat, detectTemplate, stopTokenIds } from '../src/tokenizer/chat-templates.js';
import { loadModel, inspectModel, UnsupportedModelError } from '../src/index.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zerollm-q35-'));
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));
let n = 0;
function tiny(opts = {}) {
  const m = makeTinyQwen35(opts);
  const p = path.join(dir, `q35-${n++}.gguf`);
  fs.writeFileSync(p, m.buffer);
  return { path: p, g: readGGUF(p), ...m };
}
const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'zerollm.js');
const cli = (args, input) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8', input });

const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const cosine = (a, b) => { let d = 0; let na = 0; let nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };
const TOKS = [3, 70, 5, 200, 12, 9, 255, 1, 17];

// ---------------------------------------------------------------- config / info / RAM

test('qwen35 config: hybrid layout, SSM shape, attention gate; a fully supported architecture (worker pool + reference path)', () => {
  const { g, path: p } = tiny();
  const c = modelConfig(g);
  assert.equal(c.arch, 'qwen35');
  assert.equal(c.hybrid, true);
  assert.ok(SUPPORTED_ARCHS.includes('qwen35'));
  assert.deepEqual(c.layerIsRecurrent, [true, false, true, false]); // (layer + 1) % interval != 0 -> recurrent
  assert.equal(c.nAttnLayers, 2);
  assert.equal(c.attnOutputGate, true);
  assert.equal(c.qkNorm, true);
  assert.equal(c.tiedEmbeddings, true);
  assert.equal(c.nRot, 8);
  assert.equal(c.headDim, 16);
  assert.deepEqual(c.ropeSections, [2, 1, 1, 0]);
  assert.deepEqual(c.ssm, { convKernel: 4, headKDim: 16, headVDim: 16, nKHeads: 2, nVHeads: 4, innerSize: 64, convDim: 128 });
  const i = inspectModel(p);
  assert.equal(i.supported, true);
  assert.equal(i.reason, null);
  assert.equal(i.arch, 'qwen35');
});

test('the tensors the reference path reads are exactly the real Qwen3.5-4B-Q4_K_M file\'s 426 tensors (names and shapes)', () => {
  const fixture = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'test-fixtures', 'qwen35-4b-tensors.txt');
  const real = new Map(fs.readFileSync(fixture, 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
    const m = /^(\S+) (\S+) \[(.*)\]$/.exec(l);
    return [m[1], m[3].split(',').map(Number)];
  }));
  assert.equal(real.size, 426);
  // hyperparameters straight from the real file header
  const cfg = {
    arch: 'qwen35', nEmbd: 2560, nFF: 9216, nHead: 16, nHeadKv: 4, headDim: 256, vocabSize: 248320, nLayer: 32, tiedEmbeddings: true,
    layerIsRecurrent: Array.from({ length: 32 }, (_, i) => (i + 1) % 4 !== 0),
    ssm: { convKernel: 4, headKDim: 128, headVDim: 128, nKHeads: 16, nVHeads: 32, innerSize: 4096, convDim: 8192 },
  };
  const mine = new Map(expectedTensors(cfg));
  assert.deepEqual([...mine.keys()].sort(), [...real.keys()].sort());
  for (const [name, dims] of mine) assert.deepEqual(real.get(name), dims, name);
  // and the synthetic generator produces the same set of names for the same layout (up to size)
  const { g } = tiny({ nLayer: 8, interval: 4 });
  assert.deepEqual(g.tensors.map((t) => t.name).sort(), [...new Map(expectedTensors(modelConfig(g))).keys()].sort());
  assert.deepEqual([...g.tensors.map((t) => t.name.replace(/blk\.\d+\./, 'blk.N.'))].filter((v, i, a) => a.indexOf(v) === i).sort(),
    [...real.keys()].map((k) => k.replace(/blk\.\d+\./, 'blk.N.')).filter((v, i, a) => a.indexOf(v) === i).sort());
});

test('a qwen35 file with a wrong tensor shape or a missing tensor gives one clear error', () => {
  const { g } = tiny();
  const broken = { ...g, tensor: (n) => (n === 'blk.0.ssm_a' ? undefined : n === 'blk.1.attn_k.weight' ? { dims: [64, 99] } : g.tensor(n)), tensors: g.tensors };
  broken.get = (k, d) => g.get(k, d); broken.requireTensor = (n) => g.requireTensor(n); broken.path = g.path;
  assert.throws(() => new Qwen35Model(broken, { ctx: 16 }), /missing blk\.0\.ssm_a; blk\.1\.attn_k\.weight has dims \[64,99\], expected \[64,32\]/);
});

test('qwen35 RAM estimate: KV cache only for attention layers + recurrent state (+ checkpoints)', () => {
  const { g } = tiny();
  const c = modelConfig(g);
  assert.equal(kvBytesPerToken(c), 2 * 2 * c.nHeadKv * c.headDim * 4); // 2 attention layers, not 4
  const perLayer = 4 * ((4 - 1) * 128 + 4 * 16 * 16); // conv window + delta matrices, f32
  assert.equal(recurrentStateBytes(c), 2 * perLayer);
  assert.equal(new RecurrentState(c).bytes, 2 * perLayer);
  const e = estimateRam(g, c, { ctx: 100 });
  assert.equal(e.kv, kvBytesPerToken(c) * 100);
  assert.equal(e.recurrent, 2 * perLayer);
  assert.equal(e.checkpoints, RECURRENT_CHECKPOINTS * 2 * perLayer);
  assert.equal(e.total, e.weights + e.kv + e.recurrent + e.checkpoints + e.scratch + e.runtime);
  // the real Qwen3.5-4B numbers (from the file header): 8 attention layers, 4 KV heads x 256 -> 64 KiB per token at f32; state 24 layers
  const real = { nAttnLayers: 8, nLayer: 32, nHeadKv: 4, headDim: 256, layerIsRecurrent: Array.from({ length: 32 }, (_, i) => (i + 1) % 4 !== 0), ssm: { convKernel: 4, convDim: 8192, nVHeads: 32, headVDim: 128, headKDim: 128 } };
  assert.equal(kvBytesPerToken(real), 2 * 8 * 4 * 256 * 4);
  assert.equal(recurrentStateBytes(real), 24 * 4 * (3 * 8192 + 32 * 128 * 128));
});

test('zerollm info prints the hybrid layout and the corrected RAM estimate', () => {
  const { path: p } = tiny();
  const r = cli(['info', p, '--ctx', '100', '--threads', '1']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /Architecture\s+qwen35/);
  assert.match(r.stdout, /Hybrid layout/);
  assert.match(r.stdout, /4: 2 recurrent \(Gated DeltaNet\) \+ 2 full attention/);
  assert.match(r.stdout, /Full attention layers\s+1, 3/);
  assert.match(r.stdout, /2 key heads x 16, 4 value heads x 16, conv kernel 4 over 128 channels/);
  assert.match(r.stdout, /KV cache, f32 \(default; 2 attention layers only\)\s+50\.0 KB/);
  assert.match(r.stdout, /Recurrent state \(1 sequence, f32\)/);
  assert.match(r.stdout, /State checkpoints \(up to 6\)/);
  assert.match(r.stdout, /worker pool \+ WASM SIMD kernels/);
  assert.match(r.stdout, /value head h reads key head h % 2/);
  assert.match(r.stdout, /Pre-tokenizer\s+qwen35/);
});

// ---------------------------------------------------------------- forward == independent naive implementation

test('HF-layout weights recovered from the GGUF equal the generator originals (converter round trip)', () => {
  const { g, hf } = tiny({ seed: 4 });
  const back = hfFromGguf(g);
  for (const name of Object.keys(hf)) {
    assert.ok(back[name], `missing ${name}`);
    assert.ok(maxAbs(back[name], hf[name]) < 1e-6, `${name} differs`);
  }
  // the converter really reorders value heads (otherwise the test above proves nothing about the layout)
  const z = g.requireTensor('blk.0.attn_gate.weight');
  assert.deepEqual(z.dims, [64, 64]);
  const gg = Float32Array.from(new Float32Array(Uint8Array.from(new Uint8Array(fs.readFileSync(g.path).subarray(g.dataStart + z.offset, g.dataStart + z.offset + z.nBytes))).buffer));
  assert.ok(maxAbs(gg, hf['layers.0.linear_attn.in_proj_z']) > 0.01);
  assert.ok(maxAbs(gg, reorderRows(hf['layers.0.linear_attn.in_proj_z'], 64, 64, 2, 2, 16)) < 1e-6);
});

test('qwen35 forward == naive HF-formula oracle (f32 weights, chunk-form delta rule, repeat_interleave heads)', () => {
  for (const seed of [3, 11]) {
    const { g, hf } = tiny({ quant: 'f32', seed });
    const c = modelConfig(g);
    const want = naiveQwen35Logits(c, hf, TOKS);
    const m = new RefModel(g, { ctx: 64 });
    assert.ok(m instanceof Qwen35Model);
    TOKS.forEach((t, i) => {
      const got = m.forward(t);
      const scale = Math.max(1, ...want[i].map(Math.abs));
      assert.ok(maxAbs(got, want[i]) < 1e-4 * scale, `seed ${seed} pos ${i}: diff ${maxAbs(got, want[i])}`);
      assert.equal(argmax(got), want[i].indexOf(Math.max(...want[i])));
    });
  }
});

test('qwen35 forward tracks the oracle with quantized weights (Q8_0; K-quant mix Q4_K/Q5_K/Q6_K/Q8_0)', () => {
  const kq = { nEmbd: 256, nFF: 256, nHead: 8, nHeadKv: 2, headDim: 32, nRot: 16, nK: 2, nV: 8, headSsm: 32, nLayer: 4, interval: 2 };
  for (const opts of [{ quant: 'q8_0', seed: 5 }, { quant: 'q4_k_m', seed: 6, ...kq }]) {
    const { g } = tiny(opts);
    if (opts.quant === 'q4_k_m') {
      const types = new Set(g.tensors.map((t) => t.type));
      for (const ty of [GGML_TYPE.Q4_K, GGML_TYPE.Q5_K, GGML_TYPE.Q6_K, GGML_TYPE.Q8_0]) assert.ok(types.has(ty), `type ${ty} present`);
    }
    const c = modelConfig(g);
    const toks = TOKS.slice(0, 6);
    const want = naiveQwen35Logits(c, hfFromGguf(g), toks);
    const m = new RefModel(g, { ctx: 64 });
    toks.forEach((t, i) => {
      const got = m.forward(t);
      assert.ok(cosine(got, want[i]) > 0.99, `${opts.quant} pos ${i}: cos ${cosine(got, want[i])}`);
    });
  }
});

test('prefill without logits == token-by-token; recurrent state is deterministic', () => {
  const { g } = tiny({ quant: 'q8_0', seed: 7 });
  const a = new RefModel(g, { ctx: 64 }); const b = new RefModel(g, { ctx: 64 });
  const la = Float32Array.from(a.prefill(TOKS));
  let lb; for (const t of TOKS) lb = b.forward(t);
  assert.deepEqual([...la], [...lb]);
  assert.equal(a.pos, TOKS.length);
  assert.deepEqual(a.state.ssm, b.state.ssm);
  assert.ok(a.state.ssm.some((v) => v !== 0) && a.state.conv.some((v) => v !== 0));
  // the KV cache only exists for attention layers
  assert.equal(a.cache.k.length, 2);
  assert.throws(() => { for (let i = 0; i < 70; i++) a.forward(1); }, /context full/);
});

// ---------------------------------------------------------------- snapshots / truncate

test('truncate to EVERY position, continue, and compare with a fresh run (bit-identical)', () => {
  const { g, tokens } = tiny({ quant: 'q8_0', seed: 9 });
  const im = tokens.indexOf('<|im_start|>');
  const seq = [im, 70, 5, 200, im, 9, 255, 1, 17, 3, 33];
  const fresh = new RefModel(g, { ctx: 64 });
  const want = seq.map((t) => Float32Array.from(fresh.forward(t)));
  for (const opts of [{}, { maxCheckpoints: 0 }, { maxCheckpoints: 1 }, { checkpointEvery: 3, maxCheckpoints: 2 }]) {
    const m = new RefModel(g, { ctx: 64, ...opts });
    for (const t of seq) m.forward(t);
    for (let cut = seq.length; cut >= 0; cut--) {
      const plan = m.truncatePlan(cut);
      const replayed = m.truncate(cut);
      assert.equal(replayed, cut === seq.length ? 0 : plan.replay);
      assert.equal(m.pos, cut);
      assert.deepEqual(m.tokens, seq.slice(0, cut));
      for (let i = cut; i < seq.length; i++) assert.deepEqual([...m.forward(seq[i])], [...want[i]], `opts ${JSON.stringify(opts)} cut ${cut} pos ${i}`);
    }
  }
});

test('truncate then a DIFFERENT continuation equals a fresh run of prefix + new tokens', () => {
  const { g } = tiny({ quant: 'f32', seed: 12 });
  const m = new RefModel(g, { ctx: 64 });
  for (const t of [3, 70, 5, 200, 12, 9]) m.forward(t);
  m.truncate(4);
  const got = [8, 99, 4].map((t) => Float32Array.from(m.forward(t)));
  const f = new RefModel(g, { ctx: 64 });
  const want = [3, 70, 5, 200, 8, 99, 4].map((t) => Float32Array.from(f.forward(t))).slice(4);
  got.forEach((x, i) => assert.deepEqual([...x], [...want[i]]));
});

test('checkpoint policy: before <|im_start|>, before the last prompt token, bounded, smallest position evicted', () => {
  const { g, tokens } = tiny({ quant: 'q8_0', seed: 2 });
  const im = tokens.indexOf('<|im_start|>');
  const seq = [im, 4, 5, im, 6, 7, 8];
  const m = new RefModel(g, { ctx: 64 });
  assert.equal(m.boundaryIds.has(im), true);
  m.prefill(seq); // logits only for the last token: a snapshot is taken right before it
  assert.deepEqual(m.checkpointPositions, [3, 6]); // pos 0 (<|im_start|> first) is the free zero state
  assert.deepEqual(m.truncatePlan(5), { from: 3, replay: 2 });
  assert.deepEqual(m.truncatePlan(6), { from: 6, replay: 0 });
  assert.deepEqual(m.truncatePlan(2), { from: 0, replay: 2 });
  assert.equal(m.truncate(5), 2);
  assert.deepEqual(m.checkpointPositions, [3, 5]); // 6 > 5 dropped, the new truncation point is kept
  m.reset();
  assert.deepEqual(m.checkpointPositions, []);
  assert.equal(m.pos, 0);
  const small = new RefModel(g, { ctx: 64, maxCheckpoints: 2 });
  small.prefill([im, 4, im, 5, im, 6, 7]);
  assert.deepEqual(small.checkpointPositions, [4, 6]); // [2, 4, 6] -> the oldest is evicted
  assert.equal(small.stateBytes, recurrentStateBytes(small.cfg));
});

test('generate(): regenerate and shared-prefix reuse go through truncate and give the same tokens as a cold run', async () => {
  const { g, tokens } = tiny({ quant: 'q8_0', seed: 14 });
  const im = tokens.indexOf('<|im_start|>');
  const collect = async (model, prompt, opts) => { const out = []; for await (const id of generate(model, prompt, opts)) out.push(id); return out; };
  const prompt = [im, 4, 5, 6, im, 7, 8];
  const m = new RefModel(g, { ctx: 96 });
  const stats1 = {};
  const a = await collect(m, prompt, { maxTokens: 10, stats: stats1 });
  const stats2 = {};
  const b = await collect(m, prompt, { maxTokens: 10, stats: stats2 }); // regenerate: truncate(prompt.length - 1) -> checkpoint hit, nothing to replay
  assert.deepEqual(a, b);
  assert.equal(stats2.cachedTokens, prompt.length - 1);
  assert.equal(stats1.cachedTokens, 0);
  // a longer prompt that shares only a prefix of the previous state (reply tokens differ) must equal a cold model
  const prompt2 = [...prompt, a[0], 99, 3];
  const c = await collect(m, prompt2, { maxTokens: 6 });
  const cold = await collect(new RefModel(g, { ctx: 96 }), prompt2, { maxTokens: 6 });
  assert.deepEqual(c, cold);
  const prompt3 = [im, 4, 5, 6, im, 12, 13, 14];
  const d = await collect(m, prompt3, { maxTokens: 6 }); // diverges inside the second turn: restores the checkpoint before <|im_start|>
  assert.deepEqual(d, await collect(new RefModel(g, { ctx: 96 }), prompt3, { maxTokens: 6 }));
});

test('the prefix cache stores KV of the attention layers + the recurrent state: a cached prefix continues exactly like a fresh prefill', async () => {
  const { g, path: mp } = tiny({ quant: 'q8_0', seed: 31 });
  const prefix = [3, 70, 5, 200, 12, 9, 255, 1];
  const cache = new PrefixCache(path.join(dir, 'pc'), modelFingerprint(mp, 'js/kv-f32'));
  const a = new RefModel(g, { ctx: 64 });
  for (const t of prefix) a.forward(t, { logits: false });
  const saved = await cache.save(a, prefix);
  assert.equal(fs.readFileSync(saved).subarray(0, 8).toString('latin1'), 'ZLLMKV02');
  const b = new RefModel(g, { ctx: 64 });
  assert.equal(await cache.load(b, prefix), true);
  assert.equal(b.pos, prefix.length);
  assert.deepEqual(b.checkpointPositions, [prefix.length]); // the end of the prefix is a rollback point
  for (const t of [9, 33, 120, 4]) assert.deepEqual([...b.forward(t)], [...a.forward(t)], `token ${t}`);
  assert.equal(await cache.load(new RefModel(g, { ctx: 64 }), prefix.slice(0, 5)), false, 'a different prefix is a miss');
  // a state can only be saved at the exact end of the prefix
  await assert.rejects(() => cache.save(a, prefix), /exact end of the prefix/);
  // damage -> miss, never an exception
  const bytes = fs.readFileSync(saved); fs.writeFileSync(saved, bytes.subarray(0, bytes.length - 100));
  assert.equal(await cache.load(new RefModel(g, { ctx: 64 }), prefix), false);
  // a non-hybrid reader / a hybrid file from other kernels: the fingerprint keeps them apart
  const other = new PrefixCache(path.join(dir, 'pc'), modelFingerprint(mp, 'wasm/kv-f32'));
  assert.equal(await other.load(new RefModel(g, { ctx: 64 }), prefix), false);
});

// ---------------------------------------------------------------- tokenizer

test('qwen35 pre-tokenizer: Norwegian / English / digits / whitespace / combining marks', () => {
  const pt = (s) => preTokenize(s, 'qwen35');
  assert.deepEqual(pt('Hva er hovedstaden i Frankrike?'), ['Hva', ' er', ' hovedstaden', ' i', ' Frankrike', '?']);
  assert.deepEqual(pt('Åpne blåbærsyltetøy, øl!'), ['Åpne', ' blåbærsyltetøy', ',', ' øl', '!']);
  assert.deepEqual(pt("It's 2024, isn't it? I'LL go"), ['It', "'s", ' ', '2', '0', '2', '4', ',', ' isn', "'t", ' it', '?', ' I', "'LL", ' go']);
  assert.deepEqual(pt('a\n\nb  c   d\t'), ['a', '\n\n', 'b', ' ', ' c', '  ', ' d', '\t']);
  assert.deepEqual(pt('x = (y);\n'), ['x', ' =', ' (', 'y', ');\n']);
  // combining marks stay inside words in qwen35 (\p{M}), but not in qwen2
  const word = 'éléphant'; // decomposed "éléphant"
  assert.deepEqual(pt(` ${word}`), [` ${word}`]);
  assert.notDeepEqual(preTokenize(` ${word}`, 'qwen2'), [` ${word}`]);
  assert.deepEqual(preTokenize('éle', 'qwen2'), ['e', '́le']);
  // a mark after punctuation: the punctuation may be the lead character of a word; no \p{M} inside the punctuation run
  assert.deepEqual(pt('!́x'), ['!́x']);
  assert.deepEqual(pt(' ́x'), [' ́x']);
  assert.deepEqual(pt('世界!'), ['世界', '!']);
});

test('qwen35 tokenizer from GGUF: kind, no BOS, specials, round trip', () => {
  const { g } = tiny();
  const tok = Tokenizer.fromGGUF(g);
  assert.equal(tok.kind, 'qwen35');
  assert.equal(tok.bosId, undefined);
  assert.equal(tok.addBos, false);
  assert.equal(tok.eosId, tok.idOf('<|im_end|>'));
  const text = 'Hva er hovedstaden i Frankrike? 2024 éléphant\n\n  Åse';
  const ids = tok.encode(text);
  assert.equal(tok.decode(ids), text);
  assert.ok(!ids.includes(tok.idOf('<|endoftext|>')));
  assert.equal(tok.encode('<think>', { special: true }).length, 1);
  assert.equal(tok.encode('<think>', { special: false }).length > 1, true);
});

// ---------------------------------------------------------------- chat template

const flat = (m, o) => renderSegments('qwen', m, { arch: 'qwen35', ...o }).map((s) => s.text).join('');
const MSGS = [{ role: 'system', content: '  Be brief.\n' }, { role: 'user', content: '\nHi there  ' }];

test('qwen35 chat template: system first, trimmed contents, generation prompt with / without thinking', () => {
  assert.equal(flat(MSGS, {}), '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n');
  assert.equal(flat(MSGS, { think: true }), '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n');
  assert.equal(flat([MSGS[1]], { generationPrompt: false }), '<|im_start|>user\nHi there<|im_end|>\n');
  assert.throws(() => flat([MSGS[1], MSGS[0]], {}), /System message must be at the beginning/);
  assert.equal(detectTemplate('qwen35', undefined, undefined), 'qwen');
  assert.equal(detectTemplate('qwen35', 'x <|im_start|> y', undefined), 'qwen');
});

test('qwen35 chat template: history. thinking ON follows the real template; thinking OFF keeps the empty think block (cache stays valid)', () => {
  const hist = [...MSGS, { role: 'assistant', content: '<think>\nsecret\n</think>\n\nHello!' }, { role: 'user', content: 'Again' }];
  const off = flat(hist, {});
  assert.ok(off.includes('<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n'));
  assert.ok(!off.includes('secret'));
  assert.ok(off.endsWith('<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'));
  const on = flat(hist, { think: true });
  assert.ok(on.includes('<|im_start|>assistant\nHello!<|im_end|>\n'), 'earlier turns: bare content (template: index <= last_query_index)');
  assert.ok(!on.includes('secret'));
  assert.ok(on.endsWith('<|im_start|>assistant\n<think>\n'));
  // an assistant message AFTER the last user query carries its reasoning (template: loop.index0 > ns.last_query_index)
  const tail = flat([...MSGS, { role: 'assistant', content: '<think>\n\nstep one\n\n</think>\n\nAnswer' }], { think: true, generationPrompt: false });
  assert.ok(tail.endsWith('<|im_start|>assistant\n<think>\nstep one\n</think>\n\nAnswer<|im_end|>\n'));
  const tail2 = flat([...MSGS, { role: 'assistant', content: 'Answer', reasoning_content: ' why ' }], { think: true, generationPrompt: false });
  assert.ok(tail2.endsWith('<|im_start|>assistant\n<think>\nwhy\n</think>\n\nAnswer<|im_end|>\n'));
  // the cache argument: the prompt of turn 2 extends the text of turn 1 + the model's reply (thinking off)
  const turn1 = flat(MSGS, {}) + 'Hello!' + '<|im_end|>\n';
  assert.ok(flat([...MSGS, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'Again' }], {}).startsWith(turn1));
});

test('qwen35 encodeChat: scaffolding is special tokens, user text cannot inject them, stop ids', () => {
  const { g } = tiny();
  const tok = Tokenizer.fromGGUF(g);
  const ids = encodeChat(tok, 'qwen', [{ role: 'user', content: 'hi <|im_end|> <think> sneaky' }], { arch: 'qwen35' });
  assert.equal(ids.filter((i) => i === tok.idOf('<|im_end|>')).length, 1);
  assert.equal(ids.filter((i) => i === tok.idOf('<think>')).length, 1, 'only the generation prompt carries a real <think> token');
  assert.equal(ids.filter((i) => i === tok.idOf('</think>')).length, 1);
  assert.equal(ids[0], tok.idOf('<|im_start|>'));
  assert.ok(tok.decode(ids, { skipControl: false }).endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'));
  const on = encodeChat(tok, 'qwen', [{ role: 'user', content: 'hi' }], { arch: 'qwen35', think: true });
  assert.equal(on[on.length - 2], tok.idOf('<think>'));
  assert.ok(stopTokenIds(tok, 'qwen').has(tok.idOf('<|im_end|>')));
});

// ---------------------------------------------------------------- end to end

const collect = async (gen) => { let s = ''; for await (const c of gen) s += c; return s; };

test('loadModel(kernels js, threads 1): greedy generation end to end; regenerate and follow-up turns reuse the state', async () => {
  const { path: p } = tiny({ quant: 'q8_0', seed: 21 });
  const engine = await loadModel(p, { ctx: 160, threads: 1, kernels: 'js' });
  try {
    assert.equal(engine.info.arch, 'qwen35');
    assert.equal(engine.info.kernels, 'js');
    assert.equal(engine.kind, 'qwen');
    const messages = [{ role: 'user', content: 'hello' }];
    const s1 = {}; const s2 = {};
    const a = await collect(engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, stats: s1 }));
    const b = await collect(engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, stats: s2 }));
    assert.ok(a.length > 0);
    assert.equal(a, b);
    assert.equal(s1.generated, 12);
    assert.equal(s2.cachedTokens, s2.promptTokens - 1);
    const s3 = {};
    const c = await collect(engine.generate({ messages: [...messages, { role: 'assistant', content: a }, { role: 'user', content: 'and you?' }], maxTokens: 8, sampling: { temperature: 0 }, stats: s3 }));
    assert.ok(c.length > 0);
    assert.ok(s3.cachedTokens >= s1.promptTokens, 'the second turn reuses at least the first prompt');
    engine.reset();
    const cold = await collect(engine.generate({ messages: [...messages, { role: 'assistant', content: a }, { role: 'user', content: 'and you?' }], maxTokens: 8, sampling: { temperature: 0 } }));
    assert.equal(c, cold);
  } finally { await engine.close(); }
});

test('CLI: run --threads 1 --kernels js --show-prompt on a qwen35 file (thinking off and on)', () => {
  const { path: p } = tiny({ quant: 'q8_0', seed: 3 });
  const off = cli(['run', p, '-p', 'Hva er hovedstaden i Frankrike?', '--n', '4', '--threads', '1', '--kernels', 'js', '--temp', '0', '--show-prompt']);
  assert.equal(off.status, 0, off.stderr);
  assert.match(off.stderr, /template qwen/);
  assert.ok(off.stderr.includes('<|im_start|>user\\nHva er hovedstaden i Frankrike?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'));
  assert.match(off.stderr, /generated 4 tok/);
  const on = cli(['run', p, '-p', 'Hei', '--n', '3', '--threads', '1', '--kernels', 'js', '--temp', '0', '--show-prompt', '--think']);
  assert.equal(on.status, 0, on.stderr);
  assert.ok(on.stderr.includes('<|im_start|>assistant\\n<think>\\n"]'));
});

test('CLI on the worker pool (the default path now): run prints the opening <think> of a thinking reply, chat reuses the state, info says so', () => {
  const { path: p } = tiny({ quant: 'q8_0', seed: 3 });
  const on = cli(['run', p, '-p', 'Hei', '--n', '4', '--threads', '2', '--kernels', 'js', '--temp', '0', '--think']);
  assert.equal(on.status, 0, on.stderr);
  assert.match(on.stderr, /2 threads, js kernels/);
  assert.ok(on.stdout.startsWith('<think>\n'), JSON.stringify(on.stdout));
  const off = cli(['run', p, '-p', 'Hei', '--n', '4', '--threads', '2', '--temp', '0']);
  assert.equal(off.status, 0, off.stderr);
  assert.match(off.stderr, /wasm-simd kernels/);
  assert.ok(!off.stdout.startsWith('<think>'));
  const r = cli(['chat', p, '--n', '4', '--temp', '0', '--threads', '3'], 'hei\nog du?\n/exit\n');
  assert.equal(r.status, 0, r.stderr);
  const cached = [...r.stderr.matchAll(/\((\d+) cached\)/g)].map((m) => Number(m[1]));
  assert.ok(cached.length === 1 && cached[0] > 5, `second turn cached tokens: ${cached}`);
  // greedy text of the pool with JS kernels equals the reference path's
  const ref = cli(['run', p, '-p', 'Hei', '--n', '6', '--threads', '1', '--kernels', 'js', '--temp', '0']);
  const pool = cli(['run', p, '-p', 'Hei', '--n', '6', '--threads', '3', '--kernels', 'js', '--temp', '0']);
  assert.equal(ref.status, 0, ref.stderr); assert.equal(pool.status, 0, pool.stderr);
  assert.equal(pool.stdout, ref.stdout);
});

test('CLI: chat session on a qwen35 file keeps the conversation (second turn reuses the cache)', () => {
  const { path: p } = tiny({ quant: 'q8_0', seed: 3 });
  const r = cli(['chat', p, '--n', '4', '--temp', '0', '--threads', '1', '--kernels', 'js'], 'hei\nog du?\n/exit\n');
  assert.equal(r.status, 0, r.stderr);
  assert.equal((r.stderr.match(/generated 4 tok/g) ?? []).length, 2);
  const cached = [...r.stderr.matchAll(/\((\d+) cached\)/g)].map((m) => Number(m[1]));
  assert.ok(cached.length === 1 && cached[0] > 5, `second turn cached tokens: ${cached}`);
});
