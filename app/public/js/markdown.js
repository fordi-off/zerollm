// Markdown -> safe HTML. Raw HTML in the text is shown as text, links are limited to http(s)/mailto, images are not loaded.
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const safeHref = (h) => (/^(https?:|mailto:)/i.test(h ?? '') ? h : null);

let ready = false;
function setup() {
  if (ready || !window.marked) return;
  ready = true;
  window.marked.use({
    gfm: true, breaks: true,
    renderer: {
      html(token) { return esc(token.text ?? token.raw ?? ''); },
      image(token) { return esc(token.text || token.title || ''); },
      link(token) {
        const href = safeHref(token.href);
        const inner = this.parser.parseInline(token.tokens);
        return href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${inner}</a>` : inner;
      },
      code(token) {
        const lang = (token.lang || '').split(/\s+/)[0];
        let body;
        try { body = window.hljs && lang && window.hljs.getLanguage(lang) ? window.hljs.highlight(token.text, { language: lang }).value : esc(token.text); } catch { body = esc(token.text); }
        return `<div class="codeblock"><div class="codehead"><span>${esc(lang || 'text')}</span><button type="button" class="act" data-copy-code>${esc(window.__copyLabel || 'Copy')}</button></div><pre><code class="hljs">${body}</code></pre></div>`;
      },
    },
  });
}

// ---- math: \( inline \), \[ display \], $$ display $$, and single-dollar inline math in the two shapes models actually write:
// "$ x + 1 $" (spaces inside both dollars) and "$x^2$" (no spaces, but a TeX/operator character inside). "$5 og $10" stays money:
// its dollars are not symmetric (tight opening, spaced closing).
// Math is cut out before marked (which would eat the backslashes), and put back as KaTeX HTML afterwards. Code is left alone.
const MATH = /\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\$[ \t]+([^$\n]+?)[ \t]+\$|\$(?![\s$])([^$\n]*?[\\^_{}=+*\/<>()][^$\n]*?)(?<![\s$])\$/g;
const katexCache = new Map();
function renderTex(tex, display) {
  const key = `${display ? 'D' : 'I'}${tex}`;
  let html = katexCache.get(key);
  if (html === undefined) {
    try { html = window.katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false, output: 'html', strict: 'ignore', trust: false }); }
    catch { html = `<code>${esc(tex)}</code>`; }
    if (katexCache.size > 500) katexCache.clear();
    katexCache.set(key, html);
  }
  return html;
}

/** Replace math outside code with placeholders; returns the new text and the pieces. */
export function extractMath(text) {
  const pieces = [];
  // split into code / non-code parts: fenced blocks first, then inline code spans
  const parts = String(text).split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  const out = parts.map((part, i) => {
    if (i % 2 === 1) return part; // code
    return part.replace(MATH, (m, a, b, c, d, e) => {
      const display = b !== undefined || c !== undefined; const tex = a ?? b ?? c ?? d ?? e;
      pieces.push({ tex, display });
      return `ZCMATH${pieces.length - 1}X`;
    });
  });
  return { text: out.join(''), pieces };
}

export function renderMarkdown(text) {
  setup();
  if (!window.marked) return `<p>${esc(text)}</p>`;
  const hasMath = window.katex && /\\[(\[]|\$/.test(text ?? '');
  const { text: src, pieces } = hasMath ? extractMath(text) : { text: text ?? '', pieces: [] };
  let html = window.marked.parse(src);
  if (pieces.length) {
    html = html.replace(/<p>\s*ZCMATH(\d+)X\s*<\/p>/g, (m, i) => (pieces[+i]?.display ? `<div class="math-block">${renderTex(pieces[+i].tex, true)}</div>` : m));
    html = html.replace(/ZCMATH(\d+)X/g, (m, i) => (pieces[+i] ? renderTex(pieces[+i].tex, pieces[+i].display) : m));
  }
  return html;
}
