// Stop-string detection on a text stream. Holds back text that could still become a stop string.

export class StopFilter {
  /** @param {string[]} stops */
  constructor(stops = []) {
    this.stops = stops.filter((s) => s.length > 0);
    this.buf = '';
    this.stopped = false;
  }

  /** @param {string} text @returns {string} text that is safe to emit now */
  push(text) {
    if (this.stopped) return '';
    if (!this.stops.length) return text;
    this.buf += text;
    let cut = -1;
    for (const s of this.stops) {
      const i = this.buf.indexOf(s);
      if (i >= 0 && (cut < 0 || i < cut)) cut = i;
    }
    if (cut >= 0) {
      const out = this.buf.slice(0, cut);
      this.buf = '';
      this.stopped = true;
      return out;
    }
    let hold = 0;
    for (const s of this.stops) {
      for (let k = Math.min(s.length - 1, this.buf.length); k > hold; k--) {
        if (this.buf.endsWith(s.slice(0, k))) { hold = k; break; }
      }
    }
    const out = this.buf.slice(0, this.buf.length - hold);
    this.buf = this.buf.slice(this.buf.length - hold);
    return out;
  }

  /** Emit whatever is still held back (call when generation ends without a stop match). */
  flush() {
    if (this.stopped) return '';
    const out = this.buf;
    this.buf = '';
    return out;
  }
}
