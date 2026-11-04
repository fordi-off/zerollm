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
} finally { await browser.close(); await t.close(); }
console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
console.log(`screenshots in ${out}`);
