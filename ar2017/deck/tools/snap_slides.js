// Screenshot every slide of a running `marimo run deck.py` server, cropped to the slide, controls hidden.
// Usage: node snap_slides.js http://localhost:2719 OUTDIR [N=19]
const { chromium } = require('playwright');
(async () => {
  const [,, url, out, n = "18"] = process.argv;
  const opts = process.env.HTTPS_PROXY ? { proxy: { server: process.env.HTTPS_PROXY, bypass: 'localhost,127.0.0.1' } } : {};
  const b = await chromium.launch(opts);
  const p = await (await b.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 })).newPage();
  await p.goto(url); await p.waitForTimeout(10000);
  await p.addStyleTag({ content: '.reveal .controls, .reveal .progress, .vega-embed summary, .vega-embed details { display: none !important; }' });
  await p.mouse.click(5, 5);
  for (let i = 0; i < 30; i++) { await p.keyboard.press('ArrowLeft'); await p.waitForTimeout(50); }
  await p.waitForTimeout(800);
  const box = await p.locator('.reveal').first().boundingBox();
  for (let i = 1; i <= +n; i++) {
    await p.waitForTimeout(1700);
    await p.screenshot({ path: `${out}/slide${String(i).padStart(2, '0')}.png`, clip: box });
    await p.keyboard.press('ArrowRight');
  }
  console.log('slide box', JSON.stringify(box));
  await b.close();
})();
