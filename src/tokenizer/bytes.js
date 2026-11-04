// GPT-2 byte <-> unicode mapping used by byte-level BPE vocabularies.

/** @returns {string[]} 256 entries: the printable unicode char standing for each byte value */
export function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const out = new Array(256);
  for (let i = 0; i < bs.length; i++) out[bs[i]] = String.fromCodePoint(cs[i]);
  return out;
}

export const BYTE_TO_CHAR = bytesToUnicode();
/** @type {Map<string, number>} */
export const CHAR_TO_BYTE = new Map(BYTE_TO_CHAR.map((c, b) => [c, b]));
