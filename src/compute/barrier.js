// Sense-reversing barrier over a shared Int32Array: arrival counter + generation word.
// Waiters spin briefly (cheap when all workers arrive within microseconds) and then fall back to Atomics.wait.

export class Barrier {
  /**
   * @param {Int32Array} i32 shared
   * @param {number} countIdx index of the arrival counter
   * @param {number} genIdx index of the generation word (must be a different cache line for best speed)
   * @param {number} n number of participants
   * @param {number} [spin] busy-wait iterations before sleeping
   */
  constructor(i32, countIdx, genIdx, n, spin = 20000) {
    this.i32 = i32; this.countIdx = countIdx; this.genIdx = genIdx; this.n = n; this.spin = spin;
  }

  wait() {
    const { i32, genIdx } = this;
    const gen = Atomics.load(i32, genIdx);
    if (Atomics.add(i32, this.countIdx, 1) === this.n - 1) {
      Atomics.store(i32, this.countIdx, 0); // reset BEFORE releasing: nobody can re-enter until gen changes
      Atomics.add(i32, genIdx, 1);
      Atomics.notify(i32, genIdx);
      return;
    }
    for (let s = this.spin; s > 0; s--) if (Atomics.load(i32, genIdx) !== gen) return;
    // The timeout is only a safety net; Atomics.wait returns immediately if gen already changed.
    while (Atomics.load(i32, genIdx) === gen) Atomics.wait(i32, genIdx, gen, 1000);
  }
}
