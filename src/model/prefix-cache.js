// On-disk cache of the KV state for a fixed token prefix (typically the system prompt), so repeated runs skip it.
//
// File: "ZLLMKV01" | u32 headerLength | header JSON | for each layer: K[0..n*kvDim) then V[0..n*kvDim) as f32 LE
// Hybrid models (Qwen3.5: Gated DeltaNet + attention): "ZLLMKV02", the same, but only the ATTENTION layers have K/V blocks
// (layer numbers are attention-layer numbers), followed by the recurrent state at the prefix end: the conv windows
// (header.recurrent.conv floats) and the delta-rule matrices (header.recurrent.ssm floats), f32 LE. That is ~50 MB per file for
// Qwen3.5-4B (the state does not grow with the prefix), so fewer files are kept for such models (see prune).
// The key is a hash of (model file identity, kernel flavour, exact token ids), so a hit is always the same state a
// fresh prefill would compute.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const MAGIC = 'ZLLMKV01';
const MAGIC_HYBRID = 'ZLLMKV02';

/** Number of layers that own a KV cache (every layer, except for hybrid models). */
const kvLayers = (cfg) => cfg.nAttnLayers ?? cfg.nLayer;

/** Identity of a model file + the kernel flavour that computed the state. */
export function modelFingerprint(modelPath, kernels) {
  const st = fs.statSync(modelPath);
  return `${path.basename(modelPath)}|${st.size}|${Math.round(st.mtimeMs)}|${kernels}`;
}

export class PrefixCache {
  /** @param {string} dir @param {string} fingerprint */
  constructor(dir, fingerprint) { this.dir = dir; this.fingerprint = fingerprint; }

  file(ids) {
    const h = crypto.createHash('sha1').update(this.fingerprint).update('\n').update(Int32Array.from(ids)).digest('hex');
    return path.join(this.dir, `${h}.kv`);
  }

  /**
   * Restore the KV state for exactly `ids` into a fresh model (model.pos must be 0). Returns false on any mismatch
   * or damage; never throws for a bad cache file.
   * @param {{kvSet: Function, restorePrefix: Function, setRecurrentState?: Function, pos: number, cfg: any}} model
   */
  async load(model, ids) {
    let fd = null;
    try {
      if (model.pos !== 0 || ids.length === 0) return false;
      const hybrid = model.cfg.hybrid === true;
      fd = fs.openSync(this.file(ids), 'r');
      const head = Buffer.alloc(12);
      if (fs.readSync(fd, head, 0, 12, 0) !== 12 || head.toString('latin1', 0, 8) !== (hybrid ? MAGIC_HYBRID : MAGIC)) return false;
      const hlen = head.readUInt32LE(8);
      if (hlen > 1 << 26) return false;
      const hb = Buffer.alloc(hlen);
      if (fs.readSync(fd, hb, 0, hlen, 12) !== hlen) return false;
      const h = JSON.parse(hb.toString('utf8'));
      const kvDim = model.cfg.nHeadKv * model.cfg.headDim;
      const n = ids.length;
      if (h.fingerprint !== this.fingerprint || h.n !== n || h.nLayer !== model.cfg.nLayer || h.kvDim !== kvDim) return false;
      if (h.tokens.length !== n || h.tokens.some((t, i) => t !== ids[i])) return false;
      const bytes = n * kvDim * 4;
      const nKv = kvLayers(model.cfg);
      const rec = hybrid ? { conv: model.state.conv.length, ssm: model.state.ssm.length } : { conv: 0, ssm: 0 };
      if (hybrid && (h.recurrent?.conv !== rec.conv || h.recurrent?.ssm !== rec.ssm || h.nKvLayers !== nKv)) return false;
      const expected = 12 + hlen + bytes * 2 * nKv + (rec.conv + rec.ssm) * 4;
      if (fs.fstatSync(fd).size !== expected) return false; // truncated / extended
      const tmp = Buffer.allocUnsafe(bytes);
      let pos = 12 + hlen;
      for (let l = 0; l < nKv; l++) {
        for (const isV of [0, 1]) {
          if (fs.readSync(fd, tmp, 0, bytes, pos) !== bytes) return false;
          pos += bytes;
          await model.kvSet(l, isV, n, new Float32Array(tmp.buffer, tmp.byteOffset, n * kvDim));
        }
      }
      if (hybrid) {
        const state = { conv: new Float32Array(rec.conv), ssm: new Float32Array(rec.ssm) };
        for (const a of [state.conv, state.ssm]) {
          const raw = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
          if (fs.readSync(fd, raw, 0, raw.length, pos) !== raw.length) return false;
          pos += raw.length;
        }
        await model.setRecurrentState(state);
      }
      model.restorePrefix(ids);
      return true;
    } catch {
      return false;
    } finally { if (fd !== null) try { fs.closeSync(fd); } catch { /* ignore */ } }
  }

  /**
   * Write the KV state of the first ids.length positions (the model must hold exactly that prefix; a hybrid model's recurrent
   * state cannot be extracted for an earlier position, so it must hold exactly `ids`).
   */
  async save(model, ids) {
    const hybrid = model.cfg.hybrid === true;
    if (hybrid && model.pos !== ids.length) throw new Error('a hybrid model\'s recurrent state can only be cached at the exact end of the prefix (the model holds a different number of positions)');
    const kvDim = model.cfg.nHeadKv * model.cfg.headDim;
    const n = ids.length;
    if (model.pos < n) throw new Error('model does not hold the prefix');
    fs.mkdirSync(this.dir, { recursive: true });
    const nKv = kvLayers(model.cfg);
    const state = hybrid ? await model.getRecurrentState() : null;
    const header = Buffer.from(JSON.stringify({
      fingerprint: this.fingerprint, nLayer: model.cfg.nLayer, kvDim, n, tokens: Array.from(ids),
      ...(hybrid ? { nKvLayers: nKv, recurrent: { conv: state.conv.length, ssm: state.ssm.length } } : {}),
    }), 'utf8');
    const head = Buffer.alloc(12);
    head.write(hybrid ? MAGIC_HYBRID : MAGIC, 0, 'latin1');
    head.writeUInt32LE(header.length, 8);
    const target = this.file(ids);
    const tmp = `${target}.${process.pid}.tmp`;
    const fd = fs.openSync(tmp, 'w');
    try {
      fs.writeSync(fd, head); fs.writeSync(fd, header);
      for (let l = 0; l < nKv; l++) {
        for (const isV of [0, 1]) {
          const r = await model.kvGet(l, isV, n);
          fs.writeSync(fd, Buffer.from(r.buffer, r.byteOffset, r.byteLength));
        }
      }
      if (hybrid) for (const a of [state.conv, state.ssm]) fs.writeSync(fd, Buffer.from(a.buffer, a.byteOffset, a.byteLength));
    } finally { fs.closeSync(fd); }
    fs.renameSync(tmp, target);
    this.prune(hybrid ? 4 : 16);
    return target;
  }

  /** Keep only the most recently written files (hybrid models: 4, their files hold ~50 MB of recurrent state each). */
  prune(keep = 16) {
    try {
      const files = fs.readdirSync(this.dir).filter((f) => f.endsWith('.kv')).map((f) => ({ f, t: fs.statSync(path.join(this.dir, f)).mtimeMs }));
      files.sort((a, b) => b.t - a.t);
      for (const { f } of files.slice(keep)) fs.unlinkSync(path.join(this.dir, f));
    } catch { /* best effort */ }
  }
}
