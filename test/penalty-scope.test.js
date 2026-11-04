import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
import { readGGUF } from '../src/gguf/reader.js';
import { RefModel } from '../src/model/forward.js';
import { generate } from '../src/model/generate.js';

test('penalizePrompt:false hides the prompt tokens from the sampler (default shows them)', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'zl-pen-'));
  try {
    const f = path.join(dir, 'm.gguf'); fs.writeFileSync(f, makeTinyModel({ arch: 'llama', quant: 'f32', seed: 2 }).buffer);
    const model = new RefModel(readGGUF(f), { ctx: 64 });
    const prompt = [1, 2, 3, 4, 5, 6];
    const run = async (opts) => {
      model.reset(); const seen = [];
      const sample = (logits, history) => { seen.push(history.slice()); return 7; };
      for await (const id of generate(model, prompt, { maxTokens: 3, sample, ...opts })) void id;
      return seen;
    };
    const normal = await run({});
    assert.deepEqual(normal.map((h) => h.length), [6, 7, 8]);
    assert.deepEqual(normal[0], prompt);
    const scoped = await run({ penalizePrompt: false });
    assert.deepEqual(scoped, [[], [7], [7, 7]]);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
