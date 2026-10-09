#!/usr/bin/env node
// Ролик-показ страницы «Демонстрация» (30 секунд, 1920×1080): оператор ставит точки и опасные зоны и запускает
// маршрут, затем выбирает готовую карту и запускает автономную миссию со всеми слоями карты.
//
//   pixi run demo --fast                      # или pixi run lab: сервер интерфейса должен работать
//   node tools/demo_video.js [--url http://127.0.0.1:8765] [--seed 2279] [--agent scientist_v2] [--out demo/lab_demo.mp4]
//
// Рядом с роликом пишется гифка того же имени (1280 точек в ширину, 15 кадров в секунду).
//
// Страницей управляет безголовый Chromium: курсор едет по ней, как рука человека, кадры снимаются с отметками
// времени. Симулятор идёт в настоящем времени; при сборке участки, где оператор работает мышью, ускоряются
// немного, а участки, где робот едет, — во столько раз, чтобы весь ролик занял ровно --seconds секунд.
// Нужны playwright-core (переменная PLAYWRIGHT_CORE или кэш npx), Chromium из кэша Playwright и ffmpeg.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};
const URL_BASE = arg('url', 'http://127.0.0.1:8765');
const SEED = Number(arg('seed', 2279));
const AGENT = arg('agent', 'scientist_v2');
const OUT = path.resolve(arg('out', path.join(ROOT, 'demo', 'lab_demo.mp4')));
const SECONDS = Number(arg('seconds', 30));
const FRAMES = path.resolve(arg('frames', path.join(os.tmpdir(), `did-demo-video-${process.pid}`)));
const W = 1920, H = 1080, FPS = 30;

// Сколько секунд ролика отдано участкам, где оператор только смотрит; остальное время — работа мышью.
const ROUTE_S = 6.5, ROUTE_END_S = 0.7, MISSION_S = 8.5, MISSION_END_S = 1.8;
// Маршрут первого случая (карта «Эталонная · статичная», сценарий 3) и опасные зоны поперёк прямых между точками.
const POINTS = [[-0.55, 1.6], [1.5, 0.55], [0.55, -1.5]];
const DANGER = [[0.5, 1.1], [1.05, -0.55]];

function findPlaywright() {
  if (process.env.PLAYWRIGHT_CORE) return require(process.env.PLAYWRIGHT_CORE);
  const root = path.join(os.homedir(), '.npm', '_npx');
  if (fs.existsSync(root)) {
    for (const d of fs.readdirSync(root)) {
      const p = path.join(root, d, 'node_modules', 'playwright-core');
      if (fs.existsSync(p)) return require(p);
    }
  }
  try { return require('playwright-core'); } catch (e) { return null; }
}
function findBrowser() {
  const cache = path.join(os.homedir(), 'Library', 'Caches', 'ms-playwright');
  if (!fs.existsSync(cache)) return null;
  for (const d of fs.readdirSync(cache).sort().reverse()) {
    const shell = path.join(cache, d, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell');
    if (d.startsWith('chromium_headless_shell') && fs.existsSync(shell)) return shell;
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => Number(process.hrtime.bigint()) / 1e6;
// Свой генератор случайных чисел: два запуска ведут курсор одинаково.
let rnd = 20261009;
const rand = () => { rnd = (rnd * 1664525 + 1013904223) >>> 0; return rnd / 4294967296; };

// Курсор и отметка щелчка: в безголовом браузере настоящего курсора на снимках нет.
const CURSOR = () => {
  const make = () => {
    const el = document.createElement('div');
    el.id = 'demo-cursor';
    el.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;width:28px;height:28px;transform:translate(-100px,-100px);will-change:transform';
    el.innerHTML = '<svg viewBox="0 0 28 28" width="28" height="28"><path d="M5 3l0 19 5-4.6 3.4 7.6 3-1.3-3.4-7.5 6.8-.5z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.append(el);
    const style = document.createElement('style');
    style.textContent = '@keyframes demo-pulse{from{opacity:.55;transform:translate(-50%,-50%) scale(.35)}to{opacity:0;transform:translate(-50%,-50%) scale(1)}}'
      + '.demo-pulse{position:fixed;z-index:2147483646;pointer-events:none;width:44px;height:44px;border-radius:50%;border:3px solid #2a78d6;background:rgba(42,120,214,.18);animation:demo-pulse .42s ease-out forwards}';
    document.documentElement.append(style);
    let x = -100, y = -100;
    window.addEventListener('mousemove', (e) => { x = e.clientX; y = e.clientY; el.style.transform = `translate(${x - 5}px,${y - 3}px)`; }, true);
    window.__demoPulse = () => {
      const p = document.createElement('div');
      p.className = 'demo-pulse';
      p.style.left = `${x}px`; p.style.top = `${y}px`;
      document.documentElement.append(p);
      setTimeout(() => p.remove(), 500);
    };
    window.addEventListener('mousedown', () => window.__demoPulse(), true);
  };
  if (document.documentElement) make(); else document.addEventListener('DOMContentLoaded', make);
};

async function main() {
  const pw = findPlaywright(), exe = findBrowser();
  if (!pw || !exe) throw new Error('нет playwright-core или Chromium из кэша Playwright (переменная PLAYWRIGHT_CORE — путь к пакету)');
  fs.mkdirSync(FRAMES, { recursive: true });
  const browser = await pw.chromium.launch({ executablePath: exe, headless: true });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, locale: 'ru-RU' });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('ошибка страницы:', String(e).slice(0, 300)));
  await page.addInitScript(CURSOR);

  const api = async (url, body) => {
    const r = body ? await page.request.post(URL_BASE + url, { data: body }) : await page.request.get(URL_BASE + url);
    return r.json();
  };
  const state = () => api('/api/pilot/state');
  const until = async (what, ok, limit = 30) => {
    const t0 = now();
    for (;;) {
      const s = await state();
      if (ok(s)) return s;
      if (now() - t0 > limit * 1000) throw new Error(`не дождался: ${what}`);
      await sleep(150);
    }
  };

  // --- запись кадров -------------------------------------------------------------------------
  const cdp = await ctx.newCDPSession(page);
  const frames = [];            // [{t, file}]
  const segments = [];          // [{name, t0, out | null}]: out — сколько секунд ролика отдано участку
  let gap = 0, recording = true;
  const mark = (name, out = null, minGap = 0) => { segments.push({ name, t0: now(), out }); gap = minGap; };
  const capture = async () => {
    let last = 0;
    while (recording) {
      const wait = last + gap - now();
      if (wait > 1) await sleep(wait);
      const t0 = now();
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
      last = t0;
      const file = path.join(FRAMES, `src_${String(frames.length).padStart(5, '0')}.png`);
      fs.writeFileSync(file, Buffer.from(shot.data, 'base64'));
      frames.push({ t: (t0 + now()) / 2, file });
    }
  };

  // --- рука оператора ------------------------------------------------------------------------
  let cx = W * 0.62, cy = H * 0.34;
  async function moveTo(x, y) {
    const dist = Math.hypot(x - cx, y - cy);
    const ms = 190 + 21 * Math.sqrt(dist) + 60 * rand();
    const bend = (rand() - 0.5) * 0.16 * dist;                  // лёгкая дуга, как у руки
    const mx = (cx + x) / 2 - (y - cy) / (dist || 1) * bend, my = (cy + y) / 2 + (x - cx) / (dist || 1) * bend;
    const x0 = cx, y0 = cy, t0 = now();
    for (;;) {
      const k = Math.min(1, (now() - t0) / ms);
      const e = k * k * k * (k * (6 * k - 15) + 10);            // разгон и торможение
      cx = (1 - e) * (1 - e) * x0 + 2 * (1 - e) * e * mx + e * e * x;
      cy = (1 - e) * (1 - e) * y0 + 2 * (1 - e) * e * my + e * e * y;
      await page.mouse.move(cx, cy);
      if (k >= 1) break;
      await sleep(8);
    }
  }
  async function clickAt(x, y, pause = 190) {
    await moveTo(x, y);
    await sleep(70 + 50 * rand());
    await page.mouse.down();
    await sleep(55);
    await page.mouse.up();
    await sleep(pause);
  }
  const center = async (locator, fx = 0.5, fy = 0.5) => {
    const b = await page.locator(locator).first().boundingBox();
    if (!b) throw new Error(`нет элемента на странице: ${locator}`);
    return [b.x + b.width * fx, b.y + b.height * fy];
  };
  const click = async (locator, fx = 0.5, fy = 0.5, pause) => clickAt(...await center(locator, fx, fy), pause);
  // Список выбора в безголовом браузере не рисуется: курсор подходит, значение меняется.
  async function choose(locator, value) {
    await moveTo(...await center(locator, 0.6));
    await sleep(90);
    await page.evaluate(() => window.__demoPulse());
    await sleep(260);
    await page.selectOption(locator, value);
    await sleep(240);
  }
  // Пока робот едет, оператор смотрит: рука лежит слева от карты (над картой страница пишет координаты курсора).
  const rest = async () => {
    const b = await page.locator('.pl-map__canvas').boundingBox();
    await moveTo(b.x - 150, b.y + b.height * 0.62);
  };
  // Точка арены → точка экрана: та же рамка, что в lab/views/pilot-map.js.
  let box = null;
  const onMap = async (x, y) => {
    const b = await page.locator('.pl-map__canvas').boundingBox();
    const scale = b.width / (box.x1 - box.x0);
    return [b.x + (x - box.x0) * scale, b.y + (box.y1 - y) * scale];
  };

  // --- сценарий ------------------------------------------------------------------------------
  await api('/api/pilot/start', { backend: 'fastsim', level: 'medium', seed: 3 });
  await page.goto(`${URL_BASE}/#/pilot?fps=30`, { waitUntil: 'load' });
  await page.waitForSelector('.pl-map__canvas');
  box = await page.evaluate(async () => {
    const a = await (await fetch('/api/arena')).json();
    let c0 = a.w, c1 = -1, r1 = -1;
    for (let r = 0; r < a.h; r++) for (let c = 0; c < a.w; c++) {
      if (a.free[r].charCodeAt(c) !== 49) continue;
      c0 = Math.min(c0, c); c1 = Math.max(c1, c); r1 = Math.max(r1, r);
    }
    return { x0: a.x0 + c0 * a.res - 0.16, x1: a.x0 + (c1 + 1) * a.res + 0.16, y1: a.y0 + (r1 + 1) * a.res + 0.16 };
  });
  await until('пульт готов', (s) => s.active && s.mode === 'idle');
  await sleep(900);
  await page.mouse.move(cx, cy);

  // Случай 1: точки и опасные зоны, затем запуск маршрута.
  mark('маршрут: постановка');
  const captured = capture();
  await sleep(500);
  for (let i = 0; i < 11; i++) { await page.mouse.wheel(0, 10); await sleep(16); }      // карта и слои — в кадр
  await sleep(250);
  await click('.dm-map-details summary', 0.5, 0.5, 220);
  await click('label.pl-check:has-text("Лучи лидара")', 0.12, 0.5, 300);
  for (let i = 0; i < POINTS.length; i++) {
    await clickAt(...await onMap(...POINTS[i]), 120);
    await until(`точка ${i + 1}`, (s) => s.route.length === i + 1, 5);
  }
  await click('.dm-tool--danger', 0.5, 0.5, 200);
  for (let i = 0; i < DANGER.length; i++) {
    await clickAt(...await onMap(...DANGER[i]), 120);
    await until(`зона ${i + 1}`, (s) => s.zones.length === i + 1, 5);
  }
  await sleep(350);
  await click('.dm-start', 0.5, 0.5, 60);
  mark('маршрут: робот едет', ROUTE_S, 60);
  await sleep(500);
  await rest();
  let s = await until('маршрут пройден', (x) => x.mode === 'idle' && x.route.length && x.route.every((p) => p.done), 200);
  console.log(`маршрут пройден: ${s.t} с симуляции, ${s.distance} м`);
  mark('маршрут: итог', ROUTE_END_S);
  await sleep(ROUTE_END_S * 1000);

  // Случай 2: готовая карта с изменениями среды и автономная миссия со всеми слоями.
  mark('миссия: постановка');
  await choose('select[aria-label="Тип карты"]', 'hard');
  const seed = 'input[aria-label="Номер сценария"]';
  await click(seed, 0.4, 0.5, 120);
  await page.locator(seed).evaluate((el) => el.select());
  await sleep(140);
  await page.keyboard.type(String(SEED), { delay: 105 });
  await sleep(220);
  await click('.dm-scenario .lb-btn', 0.5, 0.5, 150);
  await until('карта применена', (x) => x.active && x.seed === SEED && x.level === 'hard' && x.mode === 'idle', 20);
  await sleep(450);
  await click('.dm-task >> nth=1', 0.4, 0.5, 260);
  await choose('select[aria-label="Агент миссии"]', AGENT);
  await click('label.pl-check:has-text("Оценки агента")', 0.1, 0.5, 220);
  await click('label.pl-check:has-text("Скрытые объекты")', 0.08, 0.5, 520);
  await click('.dm-start', 0.5, 0.5, 60);
  mark('миссия: робот едет', MISSION_S, 90);
  await sleep(500);
  await rest();
  s = await until('миссия окончена', (x) => x.mode === 'idle' && x.mission && x.mission.result, 700);
  const r = s.mission.result;
  console.log(`миссия окончена: ${r.t} с симуляции, образцов ${r.samples_collected} из ${r.samples_total}, `
    + `${r.returned ? 'вернулся на базу' : 'на базу не вернулся'}, счёт ${r.score}, въездов в опасные зоны ${r.hazard_hits}`);
  mark('миссия: итог', MISSION_END_S);
  await sleep(350);
  await moveTo(...await center('.dm-result-title', 0.75, 1.5));    // рука подведена к итогу миссии
  await sleep(MISSION_END_S * 1000);
  const end = now();
  recording = false;
  await captured;
  await api('/api/pilot/start', { backend: 'off' });
  await browser.close();

  // --- сборка: каждому кадру ролика — последний снятый к этому моменту кадр ------------------
  segments.forEach((g, i) => { g.t1 = i + 1 < segments.length ? segments[i + 1].t0 : end; g.real = (g.t1 - g.t0) / 1000; });
  const fixed = segments.filter((g) => g.out != null).reduce((a, g) => a + g.out, 0);
  const hand = segments.filter((g) => g.out == null);
  const handReal = hand.reduce((a, g) => a + g.real, 0);
  if (SECONDS - fixed <= 0) throw new Error('на работу мышью не осталось времени ролика');
  hand.forEach((g) => { g.out = g.real * (SECONDS - fixed) / handReal; });
  let o = 0;
  for (const g of segments) { g.o0 = o; o += g.out; g.speed = g.real / g.out; }
  console.log(segments.map((g) => `  ${g.name}: ${g.real.toFixed(1)} с → ${g.out.toFixed(1)} с (×${g.speed.toFixed(1)})`).join('\n'));
  const total = Math.round(SECONDS * FPS);
  let k = 0, gi = 0;
  for (let n = 0; n < total; n++) {
    const to = (n + 0.5) / FPS;
    while (gi + 1 < segments.length && to >= segments[gi + 1].o0) gi += 1;
    const g = segments[gi];
    const t = g.t0 + (to - g.o0) * g.speed * 1000;
    while (k + 1 < frames.length && frames[k + 1].t <= t) k += 1;
    fs.symlinkSync(frames[k].file, path.join(FRAMES, `out_${String(n).padStart(5, '0')}.png`));
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', path.join(FRAMES, 'out_%05d.png'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], { stdio: 'inherit' });
  const gif = OUT.replace(/\.[^.]+$/, '.gif');
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', OUT, '-vf',
    'fps=15,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];'
    + '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle', '-loop', '0', gif], { stdio: 'inherit' });
  fs.writeFileSync(path.join(FRAMES, 'take.json'), JSON.stringify({ seed: SEED, agent: AGENT, result: r, segments }, null, 1));
  const mb = (f) => `${(fs.statSync(f).size / 1e6).toFixed(1)} МБ`;
  console.log(`ролик: ${OUT} (${mb(OUT)}), гифка: ${gif} (${mb(gif)}); снято кадров ${frames.length}, кадры: ${FRAMES}`);
}

main().catch((e) => { console.error(`запись не удалась: ${e.message}`); process.exit(1); });
