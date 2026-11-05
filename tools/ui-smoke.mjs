// Dev-only: drive the app UI in headless Chromium (Playwright must be installed globally - NOT a project dependency)
// and save screenshots. Usage: node tools/ui-smoke.mjs [outDir]
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
import { makeTinyModel } from '../tools/make-tiny-gguf.js';
import { makeTinyEmbedModel } from '../tools/make-tiny-embed-gguf.js';
import { startTestApp } from '../test/app/helpers.js';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const out = process.argv[2] ?? path.resolve('ui-shots');
fs.mkdirSync(out, { recursive: true });

const t = await startTestApp();
const browser = await chromium.launch();
const errors = [];
try {
  for (const [name, viewport, scheme] of [['desktop-light', { width: 1200, height: 800 }, 'light'], ['desktop-dark', { width: 1200, height: 800 }, 'dark'], ['phone', { width: 390, height: 800 }, 'light']]) {
    const ctx = await browser.newContext({ viewport, colorScheme: scheme, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`${name}: ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`${name}: ${e.message}`));
    await page.goto(t.app.url);
    await page.waitForSelector('#model-name');
    await page.screenshot({ path: path.join(out, `${name}-1-welcome.png`) });
    await page.fill('#input', 'Hei! Skriv **fet tekst** og en liste.');
    await page.click('#send');
    await page.waitForSelector('.meta', { timeout: 30000 });
    await page.screenshot({ path: path.join(out, `${name}-2-chat.png`) });
    if (name === 'desktop-light') {
      await page.click('#model-btn'); await page.screenshot({ path: path.join(out, `${name}-3-picker.png`) }); await page.keyboard.press('Escape');
      await page.click('#btn-settings'); await page.waitForSelector('#settings[open]');
      await page.screenshot({ path: path.join(out, `${name}-4-settings.png`) });
      await page.click('.tabs [data-tab="performance"]'); await page.screenshot({ path: path.join(out, `${name}-5-performance.png`) });
    }
    await ctx.close();
  }

  // ---- chat history flow (functional checks, not just screenshots)
  const check = (cond, what) => { if (!cond) errors.push(`CHECK FAILED: ${what}`); };
  const flowApp = await startTestApp({ ctx: 4096 }); // fresh data folder: the screenshot runs above already saved chats
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`flow: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !/status of 500/.test(m.text())) errors.push(`flow: ${m.text()}`); }); // 500 = the expected rejection of an invalid skill
  await page.goto(flowApp.app.url); await page.waitForSelector('#model-name');
  const send = async (text) => { const n = await page.locator('.msg.assistant .meta').count(); await page.fill('#input', text); await page.press('#input', 'Enter'); await page.waitForFunction((k) => document.querySelectorAll('.msg.assistant .meta').length > k, n, { timeout: 30000 }); };
  await send('Første spørsmål om kommuner');
  await page.waitForSelector('.chat-row');
  check(await page.locator('.chat-row').count() === 1, 'chat appears in the sidebar after the first reply');
  check((await page.locator('.chat-row .chat-item').first().textContent()).startsWith('Første spørsmål'), 'chat is named after the first message');
  await send('Andre spørsmål');
  check(await page.locator('.msg').count() === 4, 'four messages');
  // new chat, then a second chat
  await page.click('#btn-new'); check(await page.locator('.msg').count() === 0, 'new chat is empty');
  await send('Helt annen samtale');
  await page.waitForFunction(() => document.querySelectorAll('.chat-row').length === 3, null, { timeout: 5000 }).catch(() => {});
  check(await page.locator('.chat-row').count() === 2, 'two chats listed');
  await page.screenshot({ path: path.join(out, 'flow-1-two-chats.png') });
  // open the older chat
  await page.locator('.chat-row .chat-item', { hasText: 'Første spørsmål' }).click();
  await page.waitForFunction(() => document.querySelectorAll('.msg').length === 4);
  check((await page.locator('.msg.user .bubble').first().textContent()).startsWith('Første'), 'older chat restored');
  // reload the page: chats survive, token is kept for the session
  await page.reload(); await page.waitForSelector('.chat-row');
  check(await page.locator('.chat-row').count() === 2, 'chats survive a reload');
  // rename
  await page.locator('.chat-row', { hasText: 'Helt annen' }).hover();
  await page.locator('.chat-row', { hasText: 'Helt annen' }).locator('.chat-act').first().click();
  await page.fill('.chat-rename', 'Omdøpt'); await page.press('.chat-rename', 'Enter');
  await page.waitForSelector('.chat-item:text("Omdøpt")');
  // edit and resend the first user message of the open chat
  await page.locator('.chat-row .chat-item', { hasText: 'Første spørsmål' }).click();
  await page.waitForFunction(() => document.querySelectorAll('.msg').length === 4);
  await page.locator('.msg.user').first().hover(); await page.locator('.msg.user').first().locator('.act', { hasText: 'Rediger' }).click();
  await page.screenshot({ path: path.join(out, 'flow-2-edit.png') });
  await page.fill('.editor textarea', 'Redigert spørsmål');
  const metas = await page.locator('.msg.assistant .meta').count();
  await page.click('.editor .btn.primary');
  await page.waitForFunction(() => document.querySelectorAll('.msg').length === 2 && document.querySelectorAll('.msg.assistant .meta').length >= 1, null, { timeout: 30000 });
  check((await page.locator('.msg.user .bubble').first().textContent()) === 'Redigert spørsmål', 'edited text shown, later messages removed');
  // slash commands: menu, Tab to pick, chip on the sent message, saved with the chat
  await page.click('#btn-new');
  await page.fill('#input', '/hum'); await page.locator('#input').press('End');
  await page.dispatchEvent('#input', 'input');
  await page.waitForSelector('#cmd-menu:not([hidden]) .cmd-row');
  await page.screenshot({ path: path.join(out, 'flow-4-slash-menu.png') });
  await page.press('#input', 'Tab');
  check((await page.inputValue('#input')) === '/humanize ', 'Tab completes the command');
  await page.press('#input', 'End'); await page.keyboard.type('Skriv kort om kommuner');
  const before = await page.locator('.msg.assistant .meta').count();
  await page.press('#input', 'Enter');
  await page.waitForSelector('.msg.user .cmd-chip');
  check((await page.locator('.msg.user .bubble').first().textContent()) === 'Skriv kort om kommuner', 'command is not part of the message text');
  check((await page.locator('.cmd-chip').first().textContent()) === '/humanize', 'command chip shown');
  await page.waitForFunction((k) => document.querySelectorAll('.msg.assistant .meta').length > k, before, { timeout: 30000 });
  // settings: add a skill, error for a bad one
  await page.click('#btn-settings'); await page.click('.tabs [data-tab="skills"]');
  await page.fill('.skill-form input', 'Test'); await page.fill('.skill-form textarea', 'bare tekst'); await page.click('.skill-form button[type=submit]');
  await page.waitForSelector('.form-error:not([hidden])');
  await page.fill('.skill-form textarea', '## /test  Prøve\nGjør noe lurt.'); await page.click('.skill-form button[type=submit]');
  await page.waitForSelector('.skill-box:has-text("/test")');
  await page.screenshot({ path: path.join(out, 'flow-5-skills.png') });
  await page.keyboard.press('Escape');
  // documents: attach a text file, chip, question -> sources under the reply, library tab
  await page.click('#btn-new');
  await page.setInputFiles('#file-input', { name: 'kommune.txt', mimeType: 'text/plain', buffer: Buffer.from('Kommunen har ansvar for skoler og barnehager. '.repeat(40)) });
  await page.waitForSelector('.doc-chip');
  check((await page.locator('.doc-chip .nm').textContent()) === 'kommune.txt', 'document chip shown');
  const before2 = await page.locator('.msg.assistant .meta').count();
  await page.fill('#input', 'Hvem har ansvar for barnehager?'); await page.press('#input', 'Enter');
  await page.waitForFunction((k) => document.querySelectorAll('.msg.assistant .meta').length > k, before2, { timeout: 30000 });
  await page.waitForSelector('.doc-sources summary');
  await page.click('.doc-sources summary');
  await page.screenshot({ path: path.join(out, 'flow-6-sources.png') });
  check((await page.locator('.doc-source b').first().textContent()) === 'kommune.txt', 'source names the document');
  await page.click('.doc-chip .nm'); await page.waitForSelector('.doc-chip.off', { timeout: 3000 }).catch(() => errors.push('CHECK FAILED: chip toggles off'));
  await page.click('#btn-settings'); await page.click('.tabs [data-tab="documents"]');
  await page.waitForSelector('#tab-documents .model-row');
  await page.screenshot({ path: path.join(out, 'flow-7-doclib.png') });
  await page.keyboard.press('Escape');
  // reload: the chat keeps its document and the sources
  await page.reload(); await page.waitForSelector('.chat-row');
  await page.locator('.chat-row .chat-item', { hasText: 'Hvem har ansvar' }).click();
  await page.waitForSelector('.doc-chip'); await page.waitForSelector('.doc-sources');
  // math rendering (KaTeX) and its safety
  const math = await page.evaluate(async () => {
    const { renderMarkdown } = await import('/js/markdown.js');
    return { inline: renderMarkdown('Areal \\(A = \\pi r^2\\) ok'), display: renderMarkdown('\\[ \\frac{a}{b} \\]'), money: renderMarkdown('koster $5 og $10'), spaced: renderMarkdown('fart $ v \\\\approx 874.4 , \\\\text{m/s} $ og $ r $'), evil: renderMarkdown('\\(<img src=x onerror=alert(1)>\\)') };
  });
  check(/class="katex"/.test(math.inline) && !math.inline.includes('\\('), 'inline math rendered');
  check(/math-block/.test(math.display), 'display math rendered');
  check((math.spaced.match(/class="katex"/g) ?? []).length === 2, 'single-dollar math with spaces inside renders (how models write it)');
  check(!/katex/.test(math.money) && math.money.includes('$5'), 'dollar amounts are not math');
  check(!/<img/.test(math.evil), 'math cannot inject HTML');
  // delete (two clicks)
  const row = page.locator('.chat-row', { hasText: 'Omdøpt' }); await row.hover();
  await row.locator('.chat-act').nth(1).click(); await row.locator('.chat-act.armed').click();
  await page.waitForFunction(() => document.querySelectorAll('.chat-row').length === 3, null, { timeout: 5000 }).catch(async () => { await page.screenshot({ path: path.join(out, 'flow-fail-delete.png') }); errors.push(`CHECK FAILED: delete (rows: ${await page.locator('.chat-row').count()}, armed: ${await page.locator('.chat-act.armed').count()})`); });
  await page.screenshot({ path: path.join(out, 'flow-3-after-delete.png') });
  await ctx.close(); await flowApp.close();

  // ---- first run: no model -> download card -> download -> chat
  const body = makeTinyModel({ arch: 'llama', quant: 'q8_0', seed: 3, ctx: 512 }).buffer;
  const fileServer = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Length': body.length }); let pos = 0;
    const step = () => { if (res.destroyed) return; if (pos >= body.length) return res.end(); res.write(body.subarray(pos, pos + 8192)); pos += 8192; setTimeout(step, 30); };
    step();
  });
  await new Promise((r) => fileServer.listen(0, '127.0.0.1', r));
  const dlApp = await startTestApp({ models: [], catalog: [{ id: 'tiny', filename: 'tiny-dl.gguf', displayName: 'Liten testmodell', sizeBytes: body.length, ramGB: 0.1, recommended: true, url: `http://127.0.0.1:${fileServer.address().port}/x.gguf` }] });
  {
    const c2 = await browser.newContext({ viewport: { width: 1200, height: 800 } }); const pg = await c2.newPage();
    pg.on('pageerror', (e) => errors.push(`download: ${e.message}`)); pg.on('console', (m) => { if (m.type() === 'error') errors.push(`download: ${m.text()}`); });
    await pg.goto(dlApp.app.url); await pg.waitForSelector('#no-model:not([hidden])');
    await pg.screenshot({ path: path.join(out, 'dl-1-first-run.png') });
    await pg.click('#no-model .btn'); await pg.waitForSelector('#dl-box .model-row');
    await pg.click('#dl-box .btn.primary');
    await pg.waitForSelector('#dl-fill', { timeout: 5000 }).catch(() => errors.push('CHECK FAILED: no progress bar while downloading'));
    await pg.screenshot({ path: path.join(out, 'dl-2-progress.png') });
    await pg.waitForSelector('#dl-box .btn:text("Installert")', { timeout: 30000 });
    await pg.screenshot({ path: path.join(out, 'dl-3-installed.png') });
    await pg.keyboard.press('Escape');
    await pg.waitForFunction(() => document.getElementById('model-name').textContent.includes('Tiny dl') || document.getElementById('model-name').textContent.length > 0);
    check(await pg.locator('#no-model').isHidden(), 'first-run card disappears once a model is installed');
    check((await pg.textContent('#model-name')).toLowerCase().includes('tiny'), 'the downloaded model is selected');
    await c2.close();
  }
  await dlApp.close(); fileServer.close();

  // ---- indexing lock: adding a document with a search model installed locks chat + model picker until it is indexed
  const lockApp = await startTestApp({ ctx: 4096, models: [{ name: 'tiny-llama.gguf', arch: 'llama' }, { name: 'tiny-embed.gguf', raw: makeTinyEmbedModel({ quant: 'q8_0' }).buffer }] });
  {
    const c3 = await browser.newContext({ viewport: { width: 1200, height: 800 } }); const pg = await c3.newPage();
    pg.on('pageerror', (e) => errors.push(`lock: ${e.message}`)); pg.on('console', (m) => { if (m.type() === 'error') errors.push(`lock: ${m.text()}`); });
    await pg.goto(lockApp.app.url); await pg.waitForSelector('#model-name');
    await pg.setInputFiles('#file-input', { name: 'stor.txt', mimeType: 'text/plain', buffer: Buffer.from('hello world og kommune barnehage skole. '.repeat(3000)) });
    await pg.waitForSelector('#lock-banner:not([hidden])', { timeout: 10000 }).catch(() => errors.push('CHECK FAILED: lock banner did not appear'));
    check(await pg.locator('#input').isDisabled(), 'composer is disabled while indexing');
    check(await pg.locator('#model-btn').isDisabled(), 'model picker is disabled while indexing');
    await pg.screenshot({ path: path.join(out, 'lock-1-indexing.png') });
    await pg.waitForSelector('#lock-banner', { state: 'hidden', timeout: 60000 }).catch(() => errors.push('CHECK FAILED: lock never released'));
    check(!(await pg.locator('#input').isDisabled()), 'composer usable again after indexing');
    await c3.close();
  }
  await lockApp.close();
} finally { await browser.close(); await t.close(); }
console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
console.log(`screenshots in ${out}`);
