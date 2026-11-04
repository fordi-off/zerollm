// Dev-only: drive the app UI in headless Chromium (Playwright must be installed globally - NOT a project dependency)
// and save screenshots. Usage: node tools/ui-smoke.mjs [outDir]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
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
  const flowApp = await startTestApp(); // fresh data folder: the screenshot runs above already saved chats
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`flow: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`flow: ${m.text()}`); });
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
  // delete (two clicks)
  const row = page.locator('.chat-row', { hasText: 'Omdøpt' }); await row.hover();
  await row.locator('.chat-act').nth(1).click(); await row.locator('.chat-act.armed').click();
  await page.waitForFunction(() => document.querySelectorAll('.chat-row').length === 1, null, { timeout: 5000 }).catch(async () => { await page.screenshot({ path: path.join(out, 'flow-fail-delete.png') }); errors.push(`CHECK FAILED: delete (rows: ${await page.locator('.chat-row').count()}, armed: ${await page.locator('.chat-act.armed').count()})`); });
  await page.screenshot({ path: path.join(out, 'flow-3-after-delete.png') });
  await ctx.close(); await flowApp.close();
} finally { await browser.close(); await t.close(); }
console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
console.log(`screenshots in ${out}`);
