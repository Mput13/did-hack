// Снимки веб-лаборатории для слайдов (запасной вариант, пока нет снимков показа в demo/shots):
//
//   node presentation/figures/lab_shots.js          # сервер лаборатории должен работать: pixi run lab
//
// Пишет PNG в presentation/assets/c2/shots/. Нужен Playwright (ищется в кэше npx) и любой Chromium
// (кэш Playwright или установленный Google Chrome). Если чего-то нет — сообщает и выходит.
const fs = require("fs");
const os = require("os");
const path = require("path");

const BASE = process.env.LAB_URL || "http://127.0.0.1:8765";
const OUT = path.join(__dirname, "..", "assets", "c2", "shots");

function findPlaywright() {
  const roots = [path.join(os.homedir(), ".npm", "_npx")];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const d of fs.readdirSync(root)) {
      const p = path.join(root, d, "node_modules", "playwright-core");
      if (fs.existsSync(p)) return require(p);
    }
  }
  try { return require("playwright-core"); } catch (e) { return null; }
}
function findBrowser() {
  const cache = path.join(os.homedir(), "Library", "Caches", "ms-playwright");
  const found = [];
  if (fs.existsSync(cache)) {
    for (const d of fs.readdirSync(cache).sort().reverse()) {
      const shell = path.join(cache, d, "chrome-headless-shell-mac-arm64", "chrome-headless-shell");
      if (d.startsWith("chromium_headless_shell") && fs.existsSync(shell)) found.push(shell);
    }
  }
  found.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
  return found.find((f) => fs.existsSync(f));
}

const PAGES = [
  { name: "lab_overview", hash: "#/", wait: 2500 },
  { name: "lab_experiment_E1", hash: "#/exp/E1", wait: 3500 },
  {
    // Проигрыватель: тот же прогон в Gazebo, что на слайде «Сценарий». Проматываем до середины.
    name: "lab_run_gazebo_hard", hash: "#/run?file=gz_loc/adaptive/hard-1.json.gz", wait: 4500, viewport: { width: 1600, height: 1400 },
    prepare: async (page) => {
      await page.click("text=16×");
      await page.click(".rp-play");
      await page.waitForTimeout(2600);
      await page.click(".rp-play");
      await page.waitForTimeout(600);
    },
    clip: { x: 20, y: 262, width: 1560, height: 880 },
  },
  { name: "lab_pilot", hash: "#/pilot", wait: 4000 },
];

(async () => {
  const pw = findPlaywright(), exe = findBrowser();
  if (!pw || !exe) { console.warn("нет Playwright или браузера: снимки не сделаны"); return; }
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await pw.chromium.launch({ executablePath: exe, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1.5, locale: "ru-RU" });
  for (const p of PAGES) {
    const page = await ctx.newPage();
    if (p.viewport) await page.setViewportSize(p.viewport);
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));
    try {
      await page.goto(`${BASE}/${p.hash}`, { waitUntil: "load", timeout: 20000 });
      await page.waitForTimeout(p.wait);
      if (p.prepare) await p.prepare(page);
      const file = path.join(OUT, `${p.name}.png`);
      await page.screenshot(p.clip ? { path: file, clip: p.clip } : { path: file });
      if (p.name === "lab_run_gazebo_hard") {     // на какой секунде прогона снят проигрыватель — для подписи на слайде
        const t = await page.evaluate(() => {
          const m = document.body.innerText.match(/Время\s+([\d.,]+)\s*с\s+из/);
          return m ? parseFloat(m[1].replace(",", ".")) : null;
        });
        fs.writeFileSync(path.join(OUT, `${p.name}.json`), JSON.stringify({ time_s: t, url: `${BASE}/${p.hash}` }, null, 1));
      }
      console.log("снимок:", file, errors.length ? `(ошибки страницы: ${errors.join(" | ")})` : "");
    } catch (e) {
      console.warn(`не удалось снять ${p.hash}: ${String(e).slice(0, 200)}`);
    }
    await page.close();
  }
  await browser.close();
})();
