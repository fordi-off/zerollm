import test from 'node:test';
import assert from 'node:assert/strict';
import { tinyFile } from './helpers.js';
import { loadModel, inspectModel, estimateModelRam, EngineMemoryError, UnsupportedModelError, defaultThreads } from '../src/index.js';
import { buildGGUF } from '../tools/gguf-writer.js';
import { GGUF_TYPE } from '../src/gguf/types.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const collect = async (gen) => { let s = ''; for await (const c of gen) s += c; return s; };

test('inspectModel: supported / unsupported files, without loading weights', () => {
  const { path: p } = tinyFile({ arch: 'llama', quant: 'q8_0' });
  const i = inspectModel(p);
  assert.equal(i.supported, true);
  assert.equal(i.arch, 'llama');
  assert.equal(i.quant, 'Q8_0');
  assert.ok(i.params > 10000);
  assert.ok(estimateModelRam(i, { ctx: 256, threads: 2, kv: 'f16' }).total > 0);

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'insp-'));
  const bad = path.join(tmp, 'bad.gguf');
  fs.writeFileSync(bad, buildGGUF({ metadata: [{ key: 'general.architecture', type: GGUF_TYPE.STRING, value: 'mamba' }] }));
  const ib = inspectModel(bad);
  assert.equal(ib.supported, false);
  assert.match(ib.reason, /not supported/i);
  const junk = path.join(tmp, 'junk.gguf');
  fs.writeFileSync(junk, 'this is not a model');
  assert.equal(inspectModel(junk).supported, false);
  fs.rmSync(tmp, { recursive: true, force: true });
});

for (const arch of ['qwen3', 'llama']) {
  test(`loadModel + generate (${arch}): streaming, stats, stop strings, abort, prefix reuse across calls, raw mode`, async () => {
    const { path: p } = tinyFile({ arch, quant: 'q8_0', seed: 5 });
    const engine = await loadModel(p, { ctx: 200, threads: 2, kv: 'f16' });
    try {
      assert.equal(engine.info.arch, arch);
      assert.equal(engine.info.kv, 'f16');
      const messages = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hello' }];

      // streaming + stats; greedy and deterministic
      const st1 = {};
      const a = await collect(engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, stats: st1 }));
      assert.ok(a.length > 0);
      assert.equal(st1.generated, 12);
      assert.equal(st1.stopReason, 'length');
      assert.ok(st1.prefillMs >= 0 && st1.decodeMs >= 0);
      assert.equal(st1.promptTokens, engine.countPromptTokens(messages));

      // same prompt again: the whole prompt (but the last token) is served from the KV cache
      const st2 = {};
      const b = await collect(engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, stats: st2 }));
      assert.equal(b, a);
      assert.ok(st2.cachedTokens >= st2.promptTokens - 1, `cached ${st2.cachedTokens} of ${st2.promptTokens}`);

      // stop string: output is cut before it and never contains it
      const marker = a.slice(4, 6);
      const st3 = {};
      const c = await collect(engine.generate({ messages, maxTokens: 12, sampling: { temperature: 0 }, stop: [marker], stats: st3 }));
      assert.equal(c, a.slice(0, a.indexOf(marker)));
      assert.ok(!c.includes(marker));
      assert.equal(st3.stopReason, 'stop-string');

      // abort mid-stream (plain object and AbortController both work)
      const ctl = new AbortController();
      const st4 = {}; let n = 0;
      for await (const _ of engine.generate({ messages, maxTokens: 50, sampling: { temperature: 0 }, signal: ctl.signal, stats: st4 })) { if (++n === 2) ctl.abort(); }
      assert.equal(st4.stopReason, 'aborted');
      assert.ok(n >= 2 && n < 50);

      // raw prompt mode parses special tokens
      const raw = engine.kind === 'qwen' ? '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n' : '<|begin_of_text|>hello';
      const st5 = {};
      await collect(engine.generate({ prompt: raw, maxTokens: 4, sampling: { temperature: 0 }, stats: st5 }));
      assert.ok(st5.promptTokens < raw.length, 'special tokens were parsed, not spelled out');

      // prefill progress callback fires for long prompts
      const long = [{ role: 'user', content: 'a b c d e f g h i j k l m n o p q r s t u v w x y z '.repeat(3) }];
      const prog = [];
      await collect(engine.generate({ messages: long, maxTokens: 1, sampling: { temperature: 0 }, onPrefill: (d, t) => prog.push([d, t]) }));
      assert.ok(prog.length >= 2 && prog.at(-1)[0] === prog.at(-1)[1]);

      // prompt longer than the context is refused with a clear error
      await assert.rejects(collect(engine.generate({ prompt: 'x '.repeat(500), maxTokens: 1 })), /context window/);

      // tokenizer helpers
      assert.equal(engine.countTokens('hello hello'), engine.tok.encode('hello hello', { special: false, bos: false }).length);
    } finally { await engine.close(); }
    await assert.rejects(collect(engine.generate({ prompt: 'hi' })), /closed/);
  });
}

test('loadModel: RAM gate throws EngineMemoryError with numbers; unsupported files throw UnsupportedModelError', async () => {
  const { path: p } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
  await assert.rejects(loadModel(p, { threads: 2, freeBytes: 1000 }), (e) => e instanceof EngineMemoryError && e.needed > e.free && e.code === 'ENOMEM_ESTIMATE');
  const ok = await loadModel(p, { threads: 1, freeBytes: 1000, ramCheck: false });
  await ok.close();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'unsup-'));
  const bad = path.join(tmp, 'x.gguf');
  fs.writeFileSync(bad, buildGGUF({ metadata: [{ key: 'general.architecture', type: GGUF_TYPE.STRING, value: 'mamba' }] }));
  await assert.rejects(loadModel(bad), /not supported/i);
  fs.rmSync(tmp, { recursive: true, force: true });
  assert.ok(defaultThreads() >= 1);
  assert.ok(UnsupportedModelError);
});

test('loadTokenizer gives the same exact prompt token counts as a loaded engine, without loading weights', async () => {
  const { loadTokenizer } = await import('../src/index.js');
  const { path: p } = tinyFile({ arch: 'llama', quant: 'q8_0' });
  const tools = loadTokenizer(p);
  const engine = await loadModel(p, { ctx: 100, threads: 1 });
  try {
    const messages = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hello there' }];
    assert.equal(tools.countPromptTokens(messages), engine.countPromptTokens(messages));
    assert.deepEqual(tools.promptIds(messages), engine.promptIds(messages));
    assert.equal(tools.countTokens('hello hello'), engine.countTokens('hello hello'));
  } finally { await engine.close(); }
});
