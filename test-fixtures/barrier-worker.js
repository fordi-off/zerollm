// Barrier stress worker: each round every worker writes its slot, hits the barrier, then checks every slot.
import { workerData, parentPort } from 'node:worker_threads';
import { Barrier } from '../src/compute/barrier.js';

const { id, n, rounds, spin, ctrl, slots, errors } = workerData;
const i32 = new Int32Array(ctrl);
const s = new Int32Array(slots);
const bar = new Barrier(i32, 16, 32, n, spin);
let bad = 0;
for (let r = 1; r <= rounds; r++) {
  s[id] = r;              // plain (non-atomic) write, must be visible to everyone after the barrier
  bar.wait();
  for (let j = 0; j < n; j++) if (s[j] !== r) bad++;
  bar.wait();             // nobody may start round r+1 before everyone has finished checking round r
}
Atomics.add(new Int32Array(errors), 0, bad);
parentPort.postMessage('done');
