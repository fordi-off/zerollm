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

export function renderMarkdown(text) {
  setup();
  if (!window.marked) return `<p>${esc(text)}</p>`;
  return window.marked.parse(text ?? '');
}
