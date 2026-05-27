// Captures two screenshots for the case study:
//   docs/images/chat-empty-state.png   — landing page (corpus list + disclaimer + sample queries)
//   docs/images/chat-mid-response.png  — chat UI with pipeline panel firing on a sample query
//
// Run via:  npx playwright install chromium  &&  node scripts/screenshot.mjs
// Reads URL from env (REGRAG_URL), defaults to production.

import { chromium } from "playwright";
import { mkdirSync } from "fs";
import { dirname } from "path";

const URL = process.env.REGRAG_URL || "https://regrag.vercel.app";
const OUT_DIR = "docs/images";
mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();

console.log(`navigating to ${URL}`);
await page.goto(URL, { waitUntil: "networkidle", timeout: 30000 });
await page.waitForSelector("text=Demo only", { timeout: 15000 });
await page.waitForTimeout(800); // let any fade-ins settle

console.log("capturing: chat-empty-state.png");
await page.screenshot({ path: `${OUT_DIR}/chat-empty-state.png`, fullPage: false });

// Submit a multi-doc query so the pipeline panel exercises classify→decompose→retrieve_parallel→synthesize→verify.
const QUERY = "How do Orders 841 and 2222 differ in their treatment of storage versus distributed energy aggregations?";
console.log(`submitting query: ${QUERY}`);

// The chat input is a contenteditable / textarea — try a few selectors.
const inputSelectors = ['textarea', 'input[type="text"]', '[role="textbox"]', '[contenteditable="true"]'];
let input = null;
for (const sel of inputSelectors) {
  const el = await page.$(sel);
  if (el) { input = el; break; }
}
if (!input) throw new Error("could not find chat input");

await input.click();
await page.keyboard.type(QUERY, { delay: 5 });
await page.keyboard.press("Enter");

// Wait for the pipeline panel to start streaming — look for a stage label.
const stageSelectors = [
  "text=Understanding your question",
  "text=Breaking it into parts",
  "text=Finding sources",
  "text=Writing the answer",
];
let appeared = null;
for (let i = 0; i < 30; i++) {
  for (const sel of stageSelectors) {
    if (await page.$(sel)) { appeared = sel; break; }
  }
  if (appeared) break;
  await page.waitForTimeout(500);
}
if (!appeared) console.warn("pipeline panel didn't surface a stage label in 15s; screenshot may not show it");
else console.log(`pipeline panel firing (saw "${appeared}")`);

// Wait until the synthesize stage starts so the right rail has multiple stages visible
for (let i = 0; i < 30; i++) {
  if (await page.$("text=Writing the answer")) break;
  await page.waitForTimeout(500);
}
await page.waitForTimeout(800); // let layout settle

console.log("capturing: chat-mid-response.png");
await page.screenshot({ path: `${OUT_DIR}/chat-mid-response.png`, fullPage: false });

await browser.close();
console.log("done");
