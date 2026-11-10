// Checkpoints of the recurrent state of a hybrid (Gated DeltaNet) model, shared by the reference path (Qwen35Model) and the
// worker pool (PoolModel). A KV cache is positional: dropping positions is `length = n`. Recurrent state is not: it summarises
// the whole prefix, so rolling back means restoring a COPY taken earlier and replaying the tokens in between.
//
// Policy (identical on both paths; decided per token BEFORE it is processed, see wantsBefore):
//  - the state before position 0 is all zeros: always available for free;
//  - a snapshot is taken before (a) the last token of a prompt (a call wanting logits right after calls that did not: exactly
//    how generate() feeds a prompt, so a regenerate returns to exactly that point), (b) every token in `boundaryIds`
//    (<|im_start|>: chat turn boundaries), (c) every `checkpointEvery`-th position when that option is > 0;
//  - at most `maxCheckpoints` snapshots (default RECURRENT_CHECKPOINTS, each recurrentStateBytes(cfg)); when full the one
//    with the smallest position is dropped and its buffers are reused;
//  - plus ONE separate rollback slot (mark()), overwritten each time: speculative decoding marks the state before every
//    verification pass so a partial acceptance costs a replay of the accepted tokens only, without evicting the chat-turn
//    snapshots above.
import { RECURRENT_CHECKPOINTS } from './config.js';
import { RecurrentState } from './recurrent-state.js';

export class StateCheckpoints {
  /**
   * @param {import('./config.js').ModelConfig} cfg
   * @param {{maxCheckpoints?: number, checkpointEvery?: number, boundaryIds?: Iterable<number>}} [o]
   */
  constructor(cfg, o = {}) {
    this.cfg = cfg;
    this.max = o.maxCheckpoints ?? RECURRENT_CHECKPOINTS;
    this.every = o.checkpointEvery ?? 0;
    this.boundaryIds = new Set(o.boundaryIds ?? []);
    /** @type {Array<{pos: number, state: RecurrentState}>} sorted by pos */
    this.list = [];
    /** @type {{pos: number, state: RecurrentState}|null} */
    this.rollback = null;
  }

  /** Positions of the stored checkpoints (ascending; the rollback slot is not included). */
  get positions() { return this.list.map((c) => c.pos); }

  /** Should a snapshot be taken before `token` is processed at `pos`? @param {boolean} wantLogits @param {boolean} prevNoLogits */
  wantsBefore(pos, token, wantLogits, prevNoLogits) {
    return pos > 0 && ((wantLogits && prevNoLogits) || this.boundaryIds.has(token) || (this.every > 0 && pos % this.every === 0));
  }

  /** Copy `live` as the checkpoint of position `pos` (no-op at position 0, with maxCheckpoints 0, or if one exists). */
  save(pos, live) {
    if (pos === 0 || this.max <= 0 || this.list.some((c) => c.pos === pos)) return;
    let snap;
    if (this.list.length >= this.max) snap = this.list.shift().state; // drop the smallest position, reuse its buffers
    else snap = new RecurrentState(this.cfg);
    snap.copyFrom(live);
    this.list.push({ pos, state: snap });
    this.list.sort((a, b) => a.pos - b.pos);
  }

  /** Copy `live` into the single rollback slot for position `pos` (skipped when a checkpoint already holds that position). */
  mark(pos, live) {
    if (pos === 0 || this.list.some((c) => c.pos === pos)) { this.rollback = null; return; }
    if (!this.rollback) this.rollback = { pos, state: new RecurrentState(this.cfg) };
    this.rollback.pos = pos;
    this.rollback.state.copyFrom(live);
  }

  /** What truncating a sequence that holds `pos` tokens to `n` does: the snapshot restored (0 = zeros) and the tokens replayed. */
  plan(n, pos) {
    if (n >= pos) return { from: pos, replay: 0 };
    let from = 0;
    for (const c of this.list) if (c.pos <= n && c.pos > from) from = c.pos;
    if (this.rollback && this.rollback.pos <= n && this.rollback.pos > from) from = this.rollback.pos;
    return { from, replay: n - from };
  }

  /** Forget every snapshot beyond position `n`. */
  dropAfter(n) {
    this.list = this.list.filter((c) => c.pos <= n);
    if (this.rollback && this.rollback.pos > n) this.rollback = null;
  }

  /** Make `live` equal the state at snapshot position `from` (0: zeros). `from` must come from plan(). */
  restore(live, from) {
    if (from === 0) { live.clear(); return; }
    const c = this.list.find((x) => x.pos === from) ?? (this.rollback?.pos === from ? this.rollback : null);
    if (!c) throw new Error(`no recurrent-state snapshot at position ${from}`);
    live.copyFrom(c.state);
  }

  clear() { this.list.length = 0; this.rollback = null; }
}
