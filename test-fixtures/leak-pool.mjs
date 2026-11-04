// Creates a pool and "forgets" to close it: the process must still exit by itself.
import { tinyFile } from '../test/helpers.js';
import { PoolModel } from '../src/compute/pool.js';
const { g } = tinyFile({ arch: 'qwen3', quant: 'q8_0' });
const pool = await PoolModel.create(g, { ctx: 64, workers: 2 });
await pool.forward(5);
console.log('main finished without close()');
