// Доработка колоды команды (DID.pptx): то же оформление — тёмный титул, пастельные круги, шрифт Inter, слайд 10 × 5,625″,
// но с рисунками из записей прогонов, слайдом математики и примерами «обычный робот — наш робот».
//
//   /usr/local/bin/python3 presentation/team/figures.py     # рисунки из runs/ (один раз или после новых прогонов)
//   cd presentation/team && node build_team.js              # DID_v2.pptx, DID_v2.pdf, slides/, preview.jpg, speech.md
//
// Состав: основной поток (14 слайдов, 7 минут) → варианты на выбор → запасные слайды.
// Числа — из research/findings/*.md, runs/*/summary.json и записей прогонов (presentation/team/data/facts.json);
// у каждой константы ниже назван источник.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const MODULES = [path.join(__dirname, "node_modules"), process.env.DID_NODE_MODULES, "/Users/a/MAI/DID/presentation/node_modules"]
  .filter((p) => p && fs.existsSync(path.join(p, "pptxgenjs")));
process.env.NODE_PATH = MODULES.concat(process.env.NODE_PATH || []).join(path.delimiter);
require("module").Module._initPaths();
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL_DIR ||
  "/Users/a/.claude/skills/synced/0ec766ba-a970-423d-9985-0a1698a18ad8_57a5e5ff-4dfe-4d5a-9b46-49c5ea27665a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ROOT = path.join(__dirname, "..", "..");
const FIG = path.join(__dirname, "fig");
const LOGOS = path.join(__dirname, "..", "assets", "logos");
const NAME = "DID_v2";
const OUT = path.join(__dirname, `${NAME}.pptx`);
const PDF = path.join(__dirname, `${NAME}.pdf`);
const SLIDES = path.join(__dirname, "slides");
const PREVIEW = path.join(__dirname, "preview.jpg");
const SPEECH = path.join(__dirname, "speech.md");

// ---------- числа ----------
const COMMITS = (() => {
  const r = spawnSync("git", ["rev-list", "--count", "main"], { cwd: ROOT, encoding: "utf8" });
  return r.status === 0 && /^\d+\s*$/.test(r.stdout) ? Number(r.stdout) : 282;
})();
const F = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "facts.json"), "utf8"));
const ru = (x, d = 1) => Number(x).toFixed(d).replace(".", ",");
// Опыт E1 (runs/E1/summary.json): трудный уровень, 40 сценариев, адаптивный − план заранее.
const E1 = { diff: "+13,3", ci: "[7,4; 19,0]", lo: 7.4, hi: 19.0, mean: 13.3, wins: "29 из 40", seed: 1026 };
// Опыт P1 (research/findings/P1.md): сценарии 8001–8040, adaptive_v2 − adaptive.
const P1 = { diff: "+6,4", ci: "[2,9; 9,8]", lo: 2.9, hi: 9.8, mean: 6.4, seed: 8024 };
// Опыт E2 (runs/E2/summary.json, пересчёт 9.10): адаптивный − он же без одного механизма, трудный уровень.
const E2 = { health: [4.9, 1.7, 8.2], reserve: [3.6, 1.6, 5.6], soil: [1.0, -3.6, 5.4] };
const R10 = [0.02, -0.03, 0.07];          // research/findings/R10.md: реакция на смену грунта
const A1 = [-0.4, -2.7, 1.9];             // research/findings/A1.md: новые схемы планирования, 160 сценариев
// L3d (research/findings/L3.md): миссии словами, 48 прогонов на модель; правило текст не читает.
const L3 = { ok: "45–46 из 48", rule: "0 из 48" };
// Показ (docs/demo_script.md, runs/F1demo): шесть засчитанных прогонов Gazebo сценария 2, счёт 90,8–91,9.
const DEMO = { runs: "6 из 6", score: ru(F.scenario.result.score), t: Math.round(F.scenario.result.t) };
const FAULT = F.pair_fault;               // записи runs/_showcase/*/hard-8024.json.gz
// research/agenda.yaml (L4b, result): второй прогон в той же лаборатории с новыми образцами, робот с памятью − без памяти,
// 160 пар прогонов в 40 лабораториях 15001–15040; расстановка и расписание событий повторяются.
const L4B = [4.3, 2.4, 6.3];
// Проверок в main на 9.10 утром собирается 1072; работ с ревью — 22 (к прежним двадцати добавились B1 и L4b), 21 возвращена.
const N = { checks: "1000+", runs: "30 000+", returned: "21 из 22" };

// ---------- оформление колоды команды ----------
const THEME = {
  name: "DID Hack — команда",
  headFontFace: "Inter",
  bodyFontFace: "Inter",
  colors: {
    dk1: "182B45", lt1: "F8F9FC", dk2: "40536B", lt2: "FFFFFF",
    accent1: "DCE8FF", accent2: "E7DFF6", accent3: "D9EEE7",          // пастельные круги колоды
    accent4: "3D63DD", accent5: "7C5CD6", accent6: "2E9E73",          // наш робот, языковая модель, «собрано»
    hlink: "3D63DD", folHlink: "7C5CD6",
  },
};
const HEX = { grey: "8A94A6", amber: "C98A1B", orange: "E8590C", red: "D64545", pale: "B9C6DA", bar0: "B7C0CF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";              // 10 × 5,625″ — как в колоде команды
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Автономный ИИ-исследователь";
pres.author = "Команда DID Hack";
const C = pres.SchemeColor;
const X0 = 0.73, CW = 8.54;
// Шрифт называем явно, как в колоде команды: тогда у программ без Inter запасной шрифт — тоже без засечек.
const FONT = "Inter";

pres.defineSlideMaster({ title: "TITLE", background: { color: C.text1 } });
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.4, w: CW, h: 0.5, fontSize: 22.8, fontFace: FONT, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 0.95, w: CW, h: 0.34, fontSize: 11.5, fontFace: FONT, color: C.text2, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "DID Hack 2026", options: { x: X0, y: 5.22, w: 3.0, h: 0.2, fontSize: 9, fontFace: FONT, color: C.text2, margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: 8.77, y: 5.22, w: 0.5, h: 0.2, fontSize: 9, fontFace: FONT, color: C.text2, align: "right" },
});

let SECTION = "Основной поток";
const SPEECH_PARTS = [];
function content(title, lead, notes, seconds) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: SECTION });
  s.addText(title, { placeholder: "title" });
  s.addText(lead, { placeholder: "body" });
  s.addNotes(`≈ ${seconds} с. ${notes}`);
  SPEECH_PARTS.push({ section: SECTION, title, notes, seconds });
  return s;
}
function T(s, text, o) {
  s.addText(text, { isTextBox: true, margin: 0, fontFace: FONT, color: C.text1, fontSize: 11, valign: "top", ...o });
}
function oval(s, x, y, d, fill, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, objectName: name });
}
function box(s, x, y, w, h, fill, name, r) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: r === undefined ? Math.min(h / 2, 0.28) : r, fill: { color: fill }, objectName: name });
}
function arrow(s, x1, y, x2, name, color = C.text2) {
  s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, objectName: name, line: { color, width: 1.25, endArrowType: "triangle" } });
}
function badge(s, x, y, n, fill, name, d = 0.28) {
  s.addText(String(n), { shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color: fill }, color: "FFFFFF", fontFace: FONT, fontSize: 10, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name });
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(s, file, x, y, w, alt, name) {
  const f = path.isAbsolute(file) ? file : path.join(FIG, file);
  const z = pngSize(f), h = (w * z.h) / z.w;
  s.addImage({ path: f, x, y, w, h, altText: alt, objectName: name || path.basename(file, ".png") });
  return h;
}
// Подпись под рисунком: жирная строка и пояснение.
function caption(s, x, y, w, head, body, name, color = C.text1) {
  T(s, [{ text: head, options: { fontSize: 11.5, color, breakLine: true } }, { text: body, options: { fontSize: 10, color: C.text2 } }],
    { x, y, w, h: 0.62, objectName: name, paraSpaceAfter: 2 });
}
// Крупное число с подписями — как на слайде команды «Как использовали ИИ».
function stat(s, x, y, w, value, lines, name, color = C.text1, size = 30) {
  T(s, value, { x, y, w, h: 0.55, fontSize: size, color, valign: "middle", objectName: `${name}-value` });
  T(s, lines.map((t, i) => ({ text: t, options: { fontSize: 10, color: C.text2, breakLine: i < lines.length - 1 } })),
    { x, y: y + 0.58, w, h: 0.2 * lines.length + 0.1, objectName: `${name}-label`, paraSpaceAfter: 2 });
}
// Строка «разность с 95% интервалом»: отрезок и точка на общей шкале (рисуется фигурами, правится в PowerPoint).
function interval(s, y, [mean, lo, hi], scale, shown, name) {
  const X = (v) => scale.x + ((v - scale.lo) / (scale.hi - scale.lo)) * scale.w;
  const color = shown ? C.accent4 : HEX.grey;
  if (X(hi) - X(lo) > 0.02) s.addShape(pres.shapes.LINE, { x: X(lo), y, w: X(hi) - X(lo), h: 0, line: { color, width: 3 }, objectName: `${name}-ci` });
  s.addShape(pres.shapes.OVAL, { x: X(mean) - 0.065, y: y - 0.065, w: 0.13, h: 0.13, fill: { color }, line: { color: "FFFFFF", width: 1 }, objectName: `${name}-dot` });
}
function axis(s, y0, y1, scale, ticks, name) {
  const X = (v) => scale.x + ((v - scale.lo) / (scale.hi - scale.lo)) * scale.w;
  s.addShape(pres.shapes.LINE, { x: X(0), y: y0, w: 0, h: y1 - y0, line: { color: C.text2, width: 0.75, dashType: "dash" }, objectName: `${name}-zero` });
  ticks.forEach((v) => T(s, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v), { x: X(v) - 0.3, y: y1 + 0.03, w: 0.6, h: 0.18, fontSize: 9, color: C.text2, align: "center", objectName: `${name}-tick-${v}` }));
}
function tag(s, text) {
  s.addText(text, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.13, x: 6.55, y: 0.17, w: 2.72, h: 0.24, fill: { color: C.accent2 }, color: C.text1, fontFace: FONT, fontSize: 9, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "variant-tag" });
}
const sgn = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + ru(Math.abs(v), d);
const ciText = ([m, lo, hi], d = 1) => `${sgn(m, d)} [${sgn(lo, d)}; ${sgn(hi, d)}]`;

// =====================================================================================================
// ОСНОВНОЙ ПОТОК
// =====================================================================================================
pres.addSection({ title: SECTION });

function slideTitle(withMap) {
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: SECTION });
  oval(s, 5.72, 4.0, 1.2, C.accent2, "circle-lavender");
  oval(s, 8.72, 0.5, 0.8, C.accent3, "circle-mint");
  oval(s, 6.2, 1.1, 3.3, C.accent1, "circle-blue");
  if (withMap) img(s, "title_map.png", 6.72, 1.67, 2.26, "Карта арены с путём робота: семь образцов собраны, робот вернулся на базу", "title-map");
  T(s, "Автономный ИИ-исследователь", { x: 0.9, y: 1.15, w: 5.2, h: 1.3, fontSize: 34, color: C.background1, valign: "middle", objectName: "title" });
  T(s, "Поиск образцов и исследование среды с ограниченным запасом энергии", { x: 0.9, y: 2.72, w: 4.6, h: 0.6, fontSize: 14, color: HEX.pale, objectName: "subtitle" });
  T(s, "Ефремова Анастасия · Путиловский Михаил · Журавлева Полина", { x: 0.9, y: 4.42, w: 5.0, h: 0.22, fontSize: 10.5, color: HEX.pale, objectName: "team" });
  T(s, "DID Hack 2026", { x: 0.9, y: 4.78, w: 3.0, h: 0.2, fontSize: 10.5, color: HEX.pale, objectName: "event" });
  const notes = "Мы сделали робота, который ищет скрытые образцы, сам замечает, что среда изменилась, и возвращается на базу до разрядки.";
  s.addNotes(`≈ 10 с. ${notes}`);
  SPEECH_PARTS.push({ section: SECTION, title: withMap ? "Титул" : "Титул (без карты)", notes, seconds: 10 });
}
slideTitle(true);

// ---------- 2. задача ----------
{
  const s = content("Задача: найти скрытое и успеть вернуться",
    "Скрытые образцы, неизвестные затраты и ограниченный запас энергии",
    "Слева — как арена устроена на самом деле: семь образцов, участки дорогого грунта, опасная зона. Справа — всё, что знает робот: стены по лидару и одно число датчика близости. По одному числу можно сказать только, что образец где-то на кольце. Заряда хватает примерно на 24 метра, а среда по ходу прогона меняется без объявления.", 30);
  const h = img(s, "truth_vs_robot.png", X0, 1.4, 4.95, "Две карты арены: слева скрытая правда с образцами, грунтами и опасной зоной, справа то, что видит робот, — стены и кольцо возможных мест образца");
  const cy = 1.4 + h + 0.06;
  caption(s, X0, cy, 2.3, "Как на самом деле", "7 образцов, дорогой грунт (×2…×4), опасная зона", "cap-truth");
  caption(s, X0 + 2.5, cy, 2.75, "Что видит робот", `стены по лидару и одно число: z = ${ru(F.truth_vs_robot.z, 2)} — образец на кольце радиусом ${ru(F.truth_vs_robot.d, 2)} м`, "cap-robot");
  const rows = [[C.accent1, "Найти", "сигнал близости без координат и направления"],
    [C.accent2, "Изучить", "расход на разных грунтах и опасные зоны"],
    [C.accent3, "Вернуться", "с образцами, пока хватает заряда: 60 единиц ≈ 24 м"]];
  rows.forEach(([fill, head, body], i) => {
    const y = 1.42 + i * 0.84;
    box(s, 6.2, y, 3.07, 0.72, fill, `task-${i + 1}-pill`, 0.36);
    T(s, [{ text: head, options: { fontSize: 13, breakLine: true } }, { text: body, options: { fontSize: 10, color: C.text2 } }],
      { x: 6.45, y: y + 0.09, w: 2.65, h: 0.56, valign: "middle", objectName: `task-${i + 1}-text` });
  });
  box(s, X0, 4.38, CW, 0.66, C.accent3, "idea-pill", 0.33);
  T(s, [{ text: "Наша идея — действовать по проверяемой модели среды", options: { fontSize: 12.5, breakLine: true } },
    { text: "Наблюдение → гипотеза → короткая проверка → новая оценка → новый маршрут. Среда меняется без объявления — робот пересматривает прежние знания.", options: { fontSize: 10, color: C.text2 } }],
  { x: X0 + 0.3, y: 4.44, w: CW - 0.6, h: 0.54, valign: "middle", objectName: "idea-text" });
}

// ---------- 3. гипотеза ----------
const HYP = "Робот, который по ходу прогона сверяет свои ожидания с показаниями датчика и расходом батареи и по расхождению меняет поиск и запас на возврат, наберёт больше очков, чем робот с планом, составленным заранее.";
function hypothesisChecks(s, y) {
  const cols = [[C.accent1, "Как проверяем", "Два робота на одних и тех же 40 сценариях трудного уровня: «план заранее» и «с адаптацией». Разность счёта — на каждом сценарии."],
    [C.accent2, "Что измеряем", "Счёт судьи — главная метрика. Рядом: собранные образцы, возврат на базу, расход батареи."],
    [C.accent3, "Что её опровергнет", "95% интервал разности целиком ниже нуля. Если он включает ноль — выигрыш не показан."]];
  cols.forEach(([fill, head, body], i) => {
    const x = X0 + i * 2.9;
    oval(s, x, y, 0.34, fill, `check-${i + 1}-dot`);
    T(s, head, { x: x + 0.46, y: y + 0.02, w: 2.2, h: 0.3, fontSize: 12.5, valign: "middle", objectName: `check-${i + 1}-head` });
    T(s, body, { x, y: y + 0.45, w: 2.62, h: 0.85, fontSize: 10.5, color: C.text2, objectName: `check-${i + 1}-body` });
  });
}
{
  const s = content("Исследовательская гипотеза", "Что утверждаем, как проверяем и что считаем опровержением",
    "Гипотеза одна и проверяемая: робот, который по ходу прогона сверяет ожидания с данными и перестраивает поиск, наберёт больше очков, чем робот с планом, составленным заранее. Проверяем на одних и тех же сорока сценариях, смотрим разность счёта. Опровержение — если интервал разности целиком ниже нуля; если он включает ноль, выигрыш не показан. Итог в быстром симуляторе: плюс тринадцать очков, интервал от семи до девятнадцати.", 30);
  box(s, X0, 1.4, CW, 1.12, C.accent1, "hyp-card", 0.3);
  T(s, HYP, { x: X0 + 0.32, y: 1.47, w: CW - 0.64, h: 0.98, fontSize: 13.5, valign: "middle", objectName: "hyp-text" });
  hypothesisChecks(s, 2.78);
  box(s, X0, 4.25, CW, 0.78, C.background2, "result-card", 0.3);
  T(s, [{ text: "Итог: ", options: { color: C.text2 } }, { text: `${E1.diff} очка`, options: { color: C.accent4 } }, { text: ` ${E1.ci} — подтвердилась`, options: {} }],
    { x: X0 + 0.32, y: 4.3, w: 4.9, h: 0.36, fontSize: 13, valign: "middle", objectName: "result-text" });
  T(s, `быстрый симулятор; выше в ${E1.wins} сценариев`, { x: X0 + 0.32, y: 4.66, w: 5.2, h: 0.28, fontSize: 10, color: C.text2, objectName: "result-note" });
  const scale = { x: 6.1, w: 2.8, lo: -2, hi: 20 };
  axis(s, 4.36, 4.76, scale, [0, 10, 20], "hyp-axis");
  interval(s, 4.56, [E1.mean, E1.lo, E1.hi], scale, true, "hyp-e1");
}

// ---------- 4. математика ----------
{
  const s = content("Математическая постановка и метрики", "Поиск при частичной наблюдаемости с бюджетом энергии: среда скрыта, робот её оценивает",
    "Формально это поиск при частичной наблюдаемости с ограничением по энергии. Цель — счёт судьи: десять очков за образец, двадцать за возврат, штрафы за столкновения, ложные сборы и опасные зоны. Робот ведёт карту вероятностей по правилу Байеса, оценивает цену грунта по расходу, строит маршрут с этой ценой и едет к цели, только если хватит заряда на дорогу домой. Сравниваем всегда парно, на одних сценариях, и говорим «лучше» только если интервал выше нуля.", 40);
  box(s, X0, 1.36, CW, 1.2, C.accent1, "goal-card", 0.3);
  T(s, [{ text: "Цель:   ", options: { fontSize: 12.5, color: C.text2 } }, { text: "max E[ J ],   J = 10·N + 20·R + 0,1·B·R − 2·K − 3·F − 5·H", options: { fontSize: 14 } }],
    { x: X0 + 0.3, y: 1.43, w: CW - 0.6, h: 0.32, valign: "middle", objectName: "goal-formula" });
  T(s, "N — собрано образцов, R — вернулся на базу (0 или 1), B — остаток заряда, K — столкновения, F — ложные сборы, H — заезды в опасную зону",
    { x: X0 + 0.3, y: 1.77, w: CW - 0.6, h: 0.4, fontSize: 10, color: C.text2, objectName: "goal-legend" });
  T(s, [{ text: "Ограничения:   ", options: { fontSize: 10, color: C.text2 } }, { text: "B₀ = 60,   T ≤ 600 с,   ΔB = 2,5·m(x, y)·Δs + 0,01·Δt", options: { fontSize: 11.5 } }, { text: "   (m — цена грунта; зоны — отдельно)", options: { fontSize: 10, color: C.text2 } }],
    { x: X0 + 0.3, y: 2.2, w: CW - 0.6, h: 0.28, valign: "middle", objectName: "constraints" });
  const cards = [
    [C.accent2, 3.0, "Что робот оценивает", [
      ["Датчик: d — до ближайшего образца, шум σ = 0,05, z обрезан до [0; 1]", "z = max(0, 1 − d/2) + ε"],
      ["Карта вероятностей: Байес по клеткам (схема), учтён ближайший образец", "p ← p·L₁ / (p·L₁ + (1 − p)·L₀)"],
      ["Цена грунта: оценка на участке пути", "m̂ ≈ (ΔB − 0,01·Δt) / (2,5·Δs)"]]],
    [C.accent3, 3.0, "Как решает", [
      ["Маршрут: Дейкстра по сетке 5 см", "вес ребра = длина × цена клеток m̂, плюс запас у стен и зон"],
      ["Запас: C₁ — путь к цели, C₂ — домой", "едет, если B − C₁ − 1,1·C₂ − 4 ≥ 0"],
      ["Опыт: информация I на заряд c", "e* = argmax I(e) / (c(e) + 0,05)"]]],
    [C.background2, 2.24, "Почему эти метрики", [
      ["Счёт судьи J", "это и есть цель задачи"],
      ["Образцы, возврат, расход", "из чего сложился счёт"],
      ["Сравнение на одних сценариях", "Δᵢ = J₁(sᵢ) − J₂(sᵢ), 95% интервал; «лучше» — если он выше нуля"]]],
  ];
  let cx = X0;
  cards.forEach(([fill, w, head, rows], i) => {
    const x = cx, y = 2.7;
    cx += w + 0.15;
    box(s, x, y, w, 2.34, fill, `math-${i + 1}-card`, 0.26);
    T(s, head, { x: x + 0.22, y: y + 0.1, w: w - 0.44, h: 0.3, fontSize: 12.5, valign: "middle", objectName: `math-${i + 1}-head` });
    const runs = [];
    rows.forEach(([label, formula], k) => {
      runs.push({ text: label, options: { fontSize: 9.5, color: C.text2, breakLine: true } });
      runs.push({ text: formula, options: { fontSize: 11, breakLine: k < rows.length - 1, paraSpaceAfter: 5 } });
    });
    T(s, runs, { x: x + 0.22, y: y + 0.46, w: w - 0.4, h: 1.84, objectName: `math-${i + 1}-rows` });
  });
}

// ---------- 5. архитектура и инструменты ----------
{
  const s = content("Архитектура и инструменты", "Быстрый цикл управления и отдельный уровень планирования — один слайд вместо трёх",
    "Среда — Gazebo с TurtleBot3; для серий опытов — быстрый симулятор с тем же судьёй. Агент получает по ROS 2 лидар, одометрию, заряд и число датчика и отдаёт скорости. Внутри агента — картина мира, планировщик и исполнитель. Планировщик — правило или языковая модель: она получает текст миссии и сводку наблюдений и возвращает подцели в JSON; скоростями модель не управляет. Судья считает заряд и штрафы, истинное состояние среды от агента скрыто.", 35);
  oval(s, X0, 1.5, 2.2, C.accent1, "env-circle");
  T(s, [{ text: "Gazebo + робот", options: { fontSize: 14, breakLine: true } },
    { text: "TurtleBot3 Burger: лидар, одометрия, колёса", options: { fontSize: 10, breakLine: true } },
    { text: "для серий — быстрый симулятор", options: { fontSize: 10, color: C.text2 } }],
  { x: X0 + 0.25, y: 1.95, w: 1.7, h: 1.3, align: "center", valign: "middle", objectName: "env-text", paraSpaceAfter: 4 });
  box(s, 3.6, 1.42, 3.05, 2.6, C.accent3, "agent-card", 0.3);
  T(s, "Агент · Python + NumPy", { x: 3.8, y: 1.5, w: 2.65, h: 0.3, fontSize: 13, valign: "middle", align: "center", objectName: "agent-head" });
  [["Картина мира", "вероятности образцов, цены грунта, зоны"],
    ["Положение", "одометрия + лидар, SLAM Toolbox"],
    ["Планировщик", "правило или LLM → подцели"],
    ["Исполнитель", "Дейкстра → скорости v, ω; запас на возврат"]].forEach(([head, body], i) => {
    const y = 1.9 + i * 0.52;
    box(s, 3.78, y, 2.69, 0.44, C.background2, `agent-part-${i + 1}`, 0.22);
    T(s, [{ text: head + ": ", options: {} }, { text: body, options: { color: C.text2 } }],
      { x: 3.93, y, w: 2.42, h: 0.44, fontSize: 9.5, valign: "middle", objectName: `agent-part-${i + 1}-text` });
  });
  oval(s, 7.07, 1.5, 2.2, C.accent2, "llm-circle");
  T(s, [{ text: "LLM", options: { fontSize: 14, breakLine: true } },
    { text: "текст миссии + сводка наблюдений → план в JSON", options: { fontSize: 10, breakLine: true } },
    { text: "сервер МАИ: QWEN, DeepSeek", options: { fontSize: 10, color: C.text2 } }],
  { x: 7.3, y: 1.95, w: 1.74, h: 1.3, align: "center", valign: "middle", objectName: "llm-text", paraSpaceAfter: 4 });
  arrow(s, 2.98, 2.5, 3.55, "arrow-env-agent");
  arrow(s, 3.55, 2.78, 2.98, "arrow-agent-env");
  T(s, "ROS 2", { x: 2.86, y: 2.18, w: 0.8, h: 0.2, fontSize: 9.5, align: "center", objectName: "ros-label" });
  T(s, "датчики →\n← v, ω", { x: 2.86, y: 2.9, w: 0.8, h: 0.36, fontSize: 9, color: C.text2, align: "center", objectName: "ros-sub" });
  arrow(s, 6.68, 2.5, 7.04, "arrow-agent-llm");
  arrow(s, 7.04, 2.78, 6.68, "arrow-llm-agent");
  box(s, X0, 4.17, CW, 0.56, C.accent1, "judge-pill", 0.28);
  T(s, [{ text: "Судья (узел ROS 2): ", options: { fontSize: 11.5 } },
    { text: "выдаёт сигнал образцов, считает заряд, сбор и штрафы. Истинное состояние среды скрыто от агента.", options: { fontSize: 10, color: C.text2 } }],
  { x: X0 + 0.3, y: 4.17, w: CW - 0.6, h: 0.56, valign: "middle", objectName: "judge-text" });
  T(s, "Ресурсы: один ноутбук · прогон показа в Gazebo ≈ 2,5 мин · быстрый симулятор: 240 прогонов за 37 с · модели — сервер МАИ",
    { x: X0, y: 4.82, w: CW, h: 0.22, fontSize: 9.5, color: C.text2, valign: "middle", objectName: "resources" });
}

// ---------- 6. как агент ищет образец ----------
{
  const b = F.belief;
  const s = content("Как агент ищет образец: из числа — место", "Тот же прогон, первые девять секунд. Зелёное кольцо — настоящий образец: робот его не видит",
    "Датчик даёт одно число без направления. Одно показание — это кольцо: образец где-то на этом расстоянии, а ближе пусто. Робот отъехал, получил второе показание — кольцо сузилось. К восьмой секунде вероятность одного места — больше половины: это гипотеза «образец здесь». Робот подъезжает, пробует собрать и так её проверяет.", 35);
  const h = img(s, "belief.png", X0, 1.36, CW, "Три карты вероятностей: кольцо после первого показания, суженное кольцо после второго и пик у настоящего образца");
  const cy = 1.36 + h + 0.06, cw = CW / 3;
  caption(s, X0 + 0.1, cy, cw - 0.25, `${ru(b[0].t, 0)} с · z = ${ru(b[0].z, 2)}`, "одно показание — кольцо: образец на этом расстоянии, ближе пусто", "bel-1");
  caption(s, X0 + cw + 0.1, cy, cw - 0.25, `${ru(b[1].t, 0)} с · z = ${ru(b[1].z, 2)}`, "показание с другого места: кольцо сузилось", "bel-2");
  caption(s, X0 + 2 * cw + 0.1, cy, cw - 0.25, `${ru(b[2].t, 0)} с · вероятность ${Math.round(b[2].pmax * 100)}%`, "гипотеза «образец здесь» → подъезд → сбор на 9-й секунде", "bel-3");
  box(s, X0, 4.62, CW, 0.42, C.accent3, "chain-pill", 0.21);
  T(s, "показание z → карта вероятностей → кандидат → маршрут с ценой грунта → проверка запаса на возврат → скорости v, ω",
    { x: X0 + 0.3, y: 4.62, w: CW - 0.6, h: 0.42, fontSize: 10.5, valign: "middle", objectName: "chain-text" });
}

// ---------- 7. сценарий на карте ----------
{
  const s = content("Сценарий на карте: один прогон в Gazebo", "Трудный уровень, сценарий 2 — его же показываем вживую. Скрытое нарисовано для зрителя",
    "Это запись прогона в Gazebo. За первые полминуты робот собирает три образца. На тридцать первой секунде судья делает грунт дороже — робот замечает перерасход и ставит опыт. На сорок второй шумит датчик — робот делает вывод «сбой» и ждёт тридцать шесть секунд, потому что стоять дешевле, чем ехать вслепую. Потом собирает ещё четыре образца и возвращается: семь из семи, без штрафов.", 45);
  img(s, "scenario.png", X0, 1.36, 3.98, "Карта арены с путём робота и пятью отмеченными событиями прогона");
  const rows = [[C.text1, "0–30 с · три образца", "каждая цель — гипотеза «образец здесь»: подъехал, собрал, подтвердил"],
    [HEX.amber, "31 с · судья сделал грунт дороже", "расход 5,7 ед/м при прогнозе 2,9 → опыт «постоять 2 с» → вывод: грунт; участок внесён в карту стоимости"],
    [HEX.orange, "42 с · датчик зашумел", "вывод «сбой датчика»; ждёт 36 с: стоять стоит 0,01 ед/с, ехать вслепую — около 0,22"],
    [C.text1, "82–134 с · ещё четыре образца", "шум спал — поиск продолжается"],
    [C.text1, `${DEMO.t} с · на базе`, `7 из 7 образцов, счёт судьи ${DEMO.score}, без штрафов`]];
  rows.forEach(([fill, head, body], i) => {
    const y = 1.4 + i * 0.66;
    badge(s, 5.0, y + 0.02, i + 1, fill, `event-${i + 1}-badge`);
    T(s, [{ text: head, options: { fontSize: 11.5, breakLine: true } }, { text: body, options: { fontSize: 10, color: C.text2 } }],
      { x: 5.42, y, w: 3.85, h: 0.62, objectName: `event-${i + 1}-text`, paraSpaceAfter: 2 });
  });
  T(s, "Зелёные точки — образцы, жёлтое — дорогой грунт (×N), красное — опасные зоны, синяя линия — путь робота",
    { x: 5.0, y: 4.74, w: 4.27, h: 0.32, fontSize: 9.5, color: C.text2, objectName: "legend" });
}

// ---------- 8–10. примеры «обычный — наш» ----------
function pairSlide({ title, lead, notes, seconds, figure, alt, left, right, value, valueColor, lines, bottom }) {
  const s = content(title, lead, notes, seconds);
  const w = 5.55, h = img(s, figure, X0, 1.4, w, alt);
  const cy = 1.4 + h + 0.07;
  caption(s, X0 + 0.05, cy, w / 2 - 0.2, left[0], left[1], "cap-left");
  caption(s, X0 + w / 2 + 0.1, cy, w / 2 - 0.2, right[0], right[1], "cap-right", right[2] || C.text1);
  oval(s, 6.62, 1.48, 2.5, C.accent1, "stat-circle");
  T(s, value, { x: 6.62, y: 2.0, w: 2.5, h: 0.62, fontSize: 34, color: valueColor || C.accent4, align: "center", valign: "middle", objectName: "stat-value" });
  T(s, lines.map((t, i) => ({ text: t, options: { breakLine: i < lines.length - 1 } })),
    { x: 6.77, y: 2.64, w: 2.2, h: 0.9, fontSize: 10, color: C.text1, align: "center", objectName: "stat-lines", paraSpaceAfter: 2 });
  if (bottom) {
    box(s, X0, 4.6, CW, 0.44, C.accent3, "takeaway-pill", 0.22);
    T(s, bottom, { x: X0 + 0.3, y: 4.6, w: CW - 0.6, h: 0.44, fontSize: 10.5, valign: "middle", objectName: "takeaway-text" });
  }
  return s;
}
{
  const [f, a] = F.pair_e1;
  pairSlide({
    title: "Пример 1. План заранее против робота с адаптацией",
    lead: `Один сценарий (№${E1.seed}): слева маршрут составлен заранее, справа — поиск по карте вероятностей`,
    notes: "Один и тот же сценарий. Слева робот едет по маршруту, составленному до старта, и берёт только то, что оказалось по пути: четыре образца из семи. Справа — наш робот: он едет туда, где образец вероятнее, и собирает шесть. В среднем по сорока сценариям разница — тринадцать очков.", seconds: 25,
    figure: "pair_fixed_adaptive.png", alt: "Две карты одного сценария: путь робота с планом заранее и путь робота-исследователя",
    left: ["План заранее", `${f.samples_collected} из ${f.samples_total} образцов · счёт ${ru(f.score)}`],
    right: ["С адаптацией", `${a.samples_collected} из ${a.samples_total} образцов · счёт ${ru(a.score)}`],
    value: E1.diff, lines: ["очка в среднем", `95% интервал ${E1.ci}`, "40 трудных сценариев,", "быстрый симулятор"],
    bottom: `Тот же заряд, та же арена — больше образцов: выше в ${E1.wins} сценариев. По возврату на базу различие не показано`,
  });
}
{
  const s = content("Пример 2. Расход вырос вдвое — робот ставит опыт", "31-я секунда того же прогона: судья сделал грунт дороже и роботу об этом не сказал",
    "Заряд уходит вдвое быстрее прогноза. Робот не гадает, а выдвигает объяснения: подорожал грунт, течёт батарея, дорогие повороты. И выбирает опыт, который их различит и почти ничего не стоит: постоять две секунды. Стоя заряд не уходит — значит, не утечка. Разворот на месте дешёвый — значит, не повороты. Остаётся грунт: участок попадает в карту стоимости.", 35);
  const steps = [[C.accent1, "Странность", "расход 5,7 ед/м при прогнозе 2,9"],
    [C.accent2, "Объяснения", "грунт подорожал · батарея течёт · дорогие повороты · другая причина"],
    [C.accent3, "Опыт", "постоять 2 с: 1,07 бит за 0,02 ед. заряда. Проехать 0,3 м дало бы 0,50 бит за 1,4 ед. — не выбран"],
    [C.accent1, "Измерено", "стоя — 0,01 ед/с: утечки нет; разворот — 0,06 ед/рад: повороты дешёвые"],
    [C.accent3, "Вывод и действие", "грунт дороже в 2,1 раза; участок внесён в карту стоимости"]];
  steps.forEach(([fill, head, body], i) => {
    const y = 1.42 + i * 0.62;
    oval(s, X0, y + 0.03, 0.3, fill, `step-${i + 1}-dot`);
    T(s, String(i + 1), { x: X0, y: y + 0.03, w: 0.3, h: 0.3, fontSize: 10, align: "center", valign: "middle", objectName: `step-${i + 1}-n` });
    T(s, [{ text: head, options: { fontSize: 11.5, breakLine: true } }, { text: body, options: { fontSize: 10, color: C.text2 } }],
      { x: X0 + 0.44, y, w: 3.65, h: 0.6, objectName: `step-${i + 1}-text`, paraSpaceAfter: 2 });
  });
  T(s, "Вероятность объяснений до опытов и после них", { x: 5.15, y: 1.42, w: 4.12, h: 0.24, fontSize: 11, valign: "middle", objectName: "chart-title" });
  s.addChart(pres.charts.BAR, [
    { name: "до опытов", labels: ["грунт подорожал", "батарея течёт", "дорогие повороты", "другая причина"], values: [47.2, 22.6, 11.3, 18.9] },
    { name: "после двух замеров", labels: ["грунт подорожал", "батарея течёт", "дорогие повороты", "другая причина"], values: [99.8, 0, 0.2, 0] },
  ], {
    x: 5.05, y: 1.66, w: 4.22, h: 2.78, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [HEX.bar0, THEME.colors.accent4],
    catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMaxVal: 115, valAxisMinVal: 0, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: THEME.colors.dk1, catAxisLabelFontSize: 10, catAxisLabelFontFace: FONT, catAxisLineShow: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', dataLabelFontSize: 9.5, dataLabelColor: THEME.colors.dk1, dataLabelFontFace: FONT,
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: THEME.colors.dk2, legendFontFace: FONT,
    objectName: "inquiry-chart", altText: "Вероятности четырёх объяснений до опытов и после двух замеров: «грунт подорожал» — с 47 до 99,8 процента",
  });
  box(s, X0, 4.6, CW, 0.44, C.accent3, "takeaway-pill", 0.22);
  T(s, "В сериях 9 из 10 подтверждённых гипотез верны по скрытой правде. Выигрыш в очках от учёта грунта не показан: +1,0 [−3,6; +5,4]",
    { x: X0 + 0.3, y: 4.6, w: CW - 0.6, h: 0.44, fontSize: 10.5, valign: "middle", objectName: "takeaway-text" });
}
{
  const [a, v] = FAULT.results;
  const dur = Math.round(FAULT.fault[1] - FAULT.fault[0]);
  pairSlide({
    title: "Пример 3. Датчик зашумел: ехать или ждать",
    lead: `Сценарий №${P1.seed}: сбой датчика с ${Math.round(FAULT.fault[0])}-й по ${Math.round(FAULT.fault[1])}-ю секунду — оранжевый участок пути`,
    notes: "Датчик образцов зашумел — тоже без объявления. Слева робот продолжает ехать по шумным показаниям: за это время он проезжает четыре метра и тратит двадцать две единицы заряда из шестидесяти — и на последний образец заряда не хватает. Справа робот сбой переждал: потратил четыре единицы и собрал все семь. Сбой замечают оба — различие в том, что делать дальше. В среднем это даёт шесть очков.", seconds: 25,
    figure: "pair_sensor_fault.png", alt: "Две карты одного сценария: робот, который едет во время сбоя датчика, и робот, который пережидает сбой",
    left: ["Без пережидания сбоя", `за ${dur} с шума: ${ru(FAULT.left.dist_in_fault)} м пути и ${Math.round(FAULT.left.battery_in_fault)} ед. заряда · ${a.samples_collected} из 7 · счёт ${ru(a.score)}`],
    right: ["Пережидает сбой", `${ru(FAULT.right.dist_in_fault)} м пути и ${Math.round(FAULT.right.battery_in_fault)} ед. заряда · ${v.samples_collected} из 7 · счёт ${ru(v.score)}`],
    value: P1.diff, lines: ["очка в среднем", `95% интервал ${P1.ci}`, "40 новых сценариев,", "быстрый симулятор"],
    bottom: "Расчёт робота в этом прогоне: стоять — 0,01 ед/с, ехать по шумным показаниям — около 0,28 ед/с впустую. Поэтому он ждёт",
  });
}

// ---------- 11. языковая модель ----------
function llmSlide({ title, mission, notes, figure, alt, left, right, quoteHead, quote, plan, variant, input }) {
  const s = content(title, "Один исполнитель; подцели выбирает правило (текст миссии не читает) или языковая модель", notes, 40);
  if (variant) tag(s, variant);
  box(s, X0, 1.34, CW, 0.4, C.accent2, "mission-pill", 0.2);
  T(s, [{ text: "Миссия словами: ", options: { color: C.text2 } }, { text: mission, options: {} }],
    { x: X0 + 0.28, y: 1.34, w: CW - 0.56, h: 0.4, fontSize: 11.5, valign: "middle", objectName: "mission-text" });
  const w = 4.5, h = img(s, figure, X0, 1.86, w, alt);
  const cy = 1.86 + h + 0.06;
  caption(s, X0 + 0.05, cy, w / 2 - 0.15, left[0], left[1], "cap-left");
  caption(s, X0 + w / 2 + 0.08, cy, w / 2 - 0.1, right[0], right[1], "cap-right");
  box(s, 5.45, 1.86, 3.82, 1.66, C.background2, "quote-card", 0.22);
  T(s, quoteHead, { x: 5.67, y: 1.94, w: 3.4, h: 0.22, fontSize: 9.5, color: C.text2, valign: "middle", objectName: "quote-head" });
  T(s, [{ text: `«${quote}»`, options: { fontSize: 10, italic: true, breakLine: true } }, { text: plan, options: { fontSize: 10, color: THEME.colors.accent5 } }],
    { x: 5.67, y: 2.2, w: 3.42, h: 1.26, objectName: "quote-text", paraSpaceAfter: 4 });
  T(s, L3.ok, { x: 5.45, y: 3.62, w: 3.82, h: 0.42, fontSize: 24, color: THEME.colors.accent5, valign: "middle", objectName: "llm-value" });
  T(s, `прогонов по четырём миссиям выполняет каждая из трёх моделей МАИ; правило — ${L3.rule}`, { x: 5.45, y: 4.05, w: 3.82, h: 0.36, fontSize: 10, color: C.text2, objectName: "llm-label" });
  T(s, `Быстрый симулятор; время ответа модели не учитывается. ${input} На обычной миссии модель выбирает то же, что правило; ответ идёт 15–25 с, поэтому на показе решает правило`,
    { x: X0, y: 4.7, w: CW, h: 0.34, fontSize: 9.5, color: C.text2, objectName: "llm-note" });
  return s;
}
{
  const m = F.pair_llm_m4;
  llmSlide({
    title: "Пример 4. LLM читает миссию и меняет план",
    mission: "«После первого же штрафа немедленно возвращайся на базу»",
    notes: "Миссию задаём словами: после первого штрафа — сразу домой. На двадцать седьмой секунде оба робота задевают опасную зону. Правило текста не читает и собирает дальше — миссия нарушена. Языковая модель видит в сводке состояния штраф, сверяет с миссией и возвращает одну подцель: на базу. Это настоящий ответ модели с сервера МАИ. Всего по четырём типам миссий модель выполняет сорок пять — сорок шесть прогонов из сорока восьми, правило — ни одного. Выигрыш здесь — выполненная миссия, а не очки судьи.",
    figure: "pair_llm_m4.png", alt: "Две карты одного сценария: после штрафа робот с правилом продолжает собирать, робот с языковой моделью едет на базу",
    left: ["Правило: текст не читает", `после штрафа собрал ещё ${m.left.collect_t.filter((t) => t > 27).length} образца: миссия нарушена`],
    right: ["LLM · qwen3.8-flash-next", `сразу повернул домой, на базе на ${Math.round(m.results[1].t)}-й с: выполнена`],
    quoteHead: "Ответ модели на 27-й секунде, дословно",
    quote: "Миссия: после первого штрафа немедленно на базу. Штраф получен (hazard_hit=1, total=1). Заряд 45,7, возврат 6,9 — запас достаточен. 3 из 7 образцов собрано, но миссия требует остановки. Возврат.",
    plan: "подцели: [ return_base ]",
    input: "Модель получает текст миссии и сводку состояния со счётчиком штрафов.",
  });
}

// ---------- 12. итоги ----------
function resultsSlide() {
  const s = content("Итоги: что подтвердилось, а что нет", "Разность счёта на одних и тех же трудных сценариях, быстрый симулятор; отрезок — 95% интервал",
    "Сводим всё на одну шкалу. Главное подтвердилось: робот с адаптацией против плана заранее — плюс тринадцать. Пережидание сбоя датчика, контроль его исправности, запас на возврат по оценке пути и память о лаборатории между прогонами — тоже в плюсе. А учёт дорогого грунта и реакция на его смену выигрыша не показали: интервалы включают ноль, и мы это говорим прямо. В Gazebo показной сценарий пройден шесть раз из шести засчитанных.", 30);
  const rows = [["С адаптацией против плана заранее", [E1.mean, E1.lo, E1.hi], true],
    ["Пережидает сбой датчика", [P1.mean, P1.lo, P1.hi], true],
    ["Следит за исправностью датчика", E2.health, true],
    ["Запас на возврат по оценке пути домой", E2.reserve, true],
    ["Помнит лабораторию: второй прогон", L4B, true],
    ["Учитывает дорогой грунт", E2.soil, false],
    ["Замечает смену грунта на пути", R10, false],
    ["Новые схемы планирования", A1, false]];
  const scale = { x: 6.2, w: 2.9, lo: -5, hi: 20 }, y0 = 1.4, rh = 0.325;
  axis(s, y0, y0 + rows.length * rh, scale, [-5, 0, 5, 10, 15, 20], "res-axis");
  rows.forEach(([label, v, shown], i) => {
    const y = y0 + i * rh;
    T(s, label, { x: X0, y, w: 3.2, h: rh, fontSize: 11, valign: "middle", objectName: `res-${i + 1}-label` });
    T(s, ciText(v, v === R10 ? 2 : 1), { x: 3.95, y, w: 1.55, h: rh, fontSize: 10, color: shown ? C.text1 : C.text2, valign: "middle", objectName: `res-${i + 1}-value` });
    T(s, shown ? "" : "не показано", { x: 5.5, y, w: 0.75, h: rh, fontSize: 9, color: C.text2, valign: "middle", objectName: `res-${i + 1}-mark` });
    interval(s, y + rh / 2, v, scale, shown, `res-${i + 1}`);
  });
  T(s, "Опыты разные: числа не складывать. Смена грунта — разность разностей", { x: X0, y: y0 + rows.length * rh + 0.03, w: 5.2, h: 0.2, fontSize: 9.5, color: C.text2, objectName: "res-note" });
  const tiles = [[C.accent1, `${DEMO.runs} · Gazebo`, "засчитанных прогонов показа: 7 из 7 образцов, без штрафов"],
    [C.accent2, L3.ok, "миссий словами выполняет LLM; правило — ни одной"],
    [C.accent3, "9 из 10", "подтверждённых роботом гипотез верны (базовые правила)"]];
  tiles.forEach(([fill, value, label], i) => {
    const x = X0 + i * 2.9;
    box(s, x, 4.3, 2.74, 0.75, fill, `tile-${i + 1}`, 0.3);
    T(s, value, { x: x + 0.25, y: 4.33, w: 2.3, h: 0.3, fontSize: 14, valign: "middle", objectName: `tile-${i + 1}-value` });
    T(s, label, { x: x + 0.25, y: 4.62, w: 2.3, h: 0.38, fontSize: 9, color: C.text2, objectName: `tile-${i + 1}-label` });
  });
  return s;
}
resultsSlide();

// ---------- 13. ИИ в разработке и ресурсы ----------
{
  const s = content("Как использовали ИИ при разработке", "Человек задавал направление; агенты исследовали, писали код и проверяли друг друга",
    "С агентами мы работали как в лаборатории. Человек задаёт цель, роли и ограничения. Ведущий агент ведёт план и сводит код. Исследователи читают литературу, инженеры пишут код и ставят опыты, а независимая модель проверяет каждую работу. Почти все работы — двадцать одну из двадцати двух — ревью вернуло на доработку, и в этом смысл: качество держится на процессе, а не на вере в один ответ модели.", 30);
  const logo = (file, x, y, name) => s.addImage({ path: path.join(LOGOS, file), x, y, w: 0.2, h: 0.2, objectName: name, altText: name });
  const roles = [[C.accent1, "Человек", ["цель, роли", "и ограничения"], []],
    [C.accent3, "Ведущий агент", ["план, задания", "и сведение кода"], [["claude.png", "Claude Opus"]]],
    [C.accent2, "Исследователи\nи инженеры", ["литература;", "код и опыты"], [["gemini.png", "Gemini"], ["claude.png", "Claude"]]],
    [C.accent1, "Независимое\nревью", ["ошибки", "и исправления"], [["openai.png", "GPT Sol"]]]];
  const d = 1.62, gap = (CW - 4 * d) / 3;
  roles.forEach(([fill, head, lines, logos], i) => {
    const x = X0 + i * (d + gap), y = 1.32;
    oval(s, x, y, d, fill, `role-${i + 1}-circle`);
    T(s, [{ text: head, options: { fontSize: 11.5, breakLine: true } }, ...lines.map((t, k) => ({ text: t, options: { fontSize: 9.5, color: C.text2, breakLine: k < lines.length - 1 } }))],
      { x: x + 0.1, y: y + 0.24, w: d - 0.2, h: 0.96, align: "center", valign: "middle", objectName: `role-${i + 1}-text`, paraSpaceAfter: 2 });
    const lw = logos.length * 0.26;
    logos.forEach(([file, name], k) => logo(file, x + d / 2 - lw / 2 + k * 0.26 + 0.03, y + 1.25, `role-${i + 1}-logo-${name}`));
    if (i < roles.length - 1) arrow(s, x + d + 0.12, y + d / 2, x + d + gap - 0.12, `role-${i + 1}-arrow`);
  });
  box(s, X0, 3.06, CW, 0.42, C.accent3, "cycle-pill", 0.21);
  T(s, "Гипотеза → код и опыт → ревью → исправления → принятие", { x: X0, y: 3.06, w: CW, h: 0.42, fontSize: 11.5, align: "center", valign: "middle", objectName: "cycle-text" });
  const nums = [[String(COMMITS), ["правок в Git"]], [N.checks, ["автоматических проверок"]], [N.runs, ["записанных прогонов"]], [N.returned, ["работ ревью вернуло", "на доработку"]]];
  nums.forEach(([value, lines], i) => stat(s, X0 + i * 2.2, 3.6, 2.05, value, lines, `num-${i + 1}`, C.text1, 24));
  T(s, "Ревью нашло 7 ошибок в первых исследованиях: исправлены 2 вывода, отклонены 2 выдуманные ссылки. Инструменты: Claude Code, Codex, Antigravity; модели робота — сервер МАИ",
    { x: X0, y: 4.68, w: CW, h: 0.36, fontSize: 9.5, color: C.text2, objectName: "ai-note" });
}

// ---------- 14. команда ----------
{
  const s = content("Команда и роли", "Спасибо! Дальше — живой показ и вопросы",
    "Команда — три человека. Дальше покажем робота вживую.", 10);
  [[C.accent1, "Ефремова\nАнастасия", "ML (LLM)"], [C.accent2, "Путиловский\nМихаил", "аналитик\n(исследователь)"], [C.accent3, "Журавлева\nПолина", "разработчик\n(Back/ROS)"]]
    .forEach(([fill, name, role], i) => {
      const x = X0 + 0.25 + i * 2.85, d = 2.35;
      oval(s, x, 1.75, d, fill, `member-${i + 1}-circle`);
      T(s, [{ text: name, options: { fontSize: 14, breakLine: true } }, { text: role, options: { fontSize: 11, color: C.text2 } }],
        { x: x + 0.2, y: 2.3, w: d - 0.4, h: 1.25, align: "center", valign: "middle", objectName: `member-${i + 1}-text`, paraSpaceAfter: 4 });
    });
}

// =====================================================================================================
// ВАРИАНТЫ НА ВЫБОР
// =====================================================================================================
SECTION = "Варианты на выбор";
pres.addSection({ title: SECTION });

// ---------- гипотеза, вариант Б: что и за счёт чего ----------
{
  const s = content("Исследовательская гипотеза: что и за счёт чего", "Главное утверждение и механизм; вторая часть подтвердилась не целиком (быстрый симулятор)",
    "Гипотеза из двух частей. Первая: робот с адаптацией наберёт больше, чем робот с планом заранее, — подтвердилась. Вторая — за счёт чего: мы ожидали, что выигрыш дают все механизмы, уточняющие знание о среде. Подтвердилось для датчика и запаса на возврат; для грунта выигрыш не показан.", 35);
  tag(s, "Вариант Б · вместо слайда 3");
  box(s, X0, 1.38, 4.1, 1.5, C.accent1, "h1-card", 0.28);
  T(s, [{ text: "Г1. Что", options: { fontSize: 11, color: C.text2, breakLine: true } },
    { text: "Робот, который сверяет ожидания с датчиком и расходом батареи и перестраивает поиск, наберёт больше очков, чем робот с планом заранее.", options: { fontSize: 11.5 } }],
  { x: X0 + 0.25, y: 1.46, w: 3.6, h: 1.34, valign: "middle", objectName: "h1-text", paraSpaceAfter: 4 });
  box(s, X0, 3.02, 4.1, 1.5, C.accent2, "h2-card", 0.28);
  T(s, [{ text: "Г2. За счёт чего", options: { fontSize: 11, color: C.text2, breakLine: true } },
    { text: "Выигрыш дают механизмы, которые уточняют знание робота о среде и о себе: исправность датчика, запас на возврат, цена грунта.", options: { fontSize: 11.5 } }],
  { x: X0 + 0.25, y: 3.1, w: 3.6, h: 1.34, valign: "middle", objectName: "h2-text", paraSpaceAfter: 4 });
  T(s, "Интервал целиком ниже нуля — опровергнута; включает ноль — выигрыш не показан", { x: X0, y: 4.62, w: 4.1, h: 0.4, fontSize: 10, color: C.text2, objectName: "refute" });
  const rows = [["Г1 · против плана заранее", [E1.mean, E1.lo, E1.hi], true], ["Г2 · пережидание сбоя датчика", [P1.mean, P1.lo, P1.hi], true],
    ["Г2 · исправность датчика", E2.health, true], ["Г2 · запас на возврат", E2.reserve, true],
    ["Г2 · учёт цены грунта", E2.soil, false], ["Г2 · смена грунта (эффект смены)", R10, false]];
  const scale = { x: 7.75, w: 1.4, lo: -5, hi: 20 }, y0 = 1.5, rh = 0.46;
  axis(s, y0, y0 + rows.length * rh, scale, [0, 10, 20], "hv-axis");
  rows.forEach(([label, v, shown], i) => {
    const y = y0 + i * rh;
    T(s, [{ text: label, options: { fontSize: 10.5, breakLine: true } }, { text: ciText(v, i === 5 ? 2 : 1) + (shown ? "" : " · не показано"), options: { fontSize: 9.5, color: C.text2 } }],
      { x: 5.1, y, w: 2.6, h: rh, valign: "middle", objectName: `hv-${i + 1}-label` });
    interval(s, y + rh / 2, v, scale, shown, `hv-${i + 1}`);
  });
  T(s, "Итог: Г1 подтверждена; Г2 — для датчика и запаса на возврат; для грунта выигрыш не показан",
    { x: 5.1, y: 4.5, w: 4.17, h: 0.46, fontSize: 10.5, objectName: "hv-result" });
}
// ---------- гипотеза, вариант В: текст команды в исходной вёрстке ----------
{
  const s = content("Исследовательская гипотеза", "Формулировка из колоды команды, сокращённая до одного предложения",
    "Робот, который ведёт себя как исследователь, наберёт больше очков, чем робот с планом, составленным заранее.", 30);
  tag(s, "Вариант В · вместо слайда 3");
  T(s, "Робот, который ведёт себя как исследователь — сам замечает, что среда или датчик изменились, выдвигает объяснения, проверяет их действием и по результату перестраивает план, — наберёт больше очков, чем робот с планом, составленным заранее.",
    { x: X0, y: 1.42, w: 5.5, h: 1.5, fontSize: 13.5, objectName: "hyp-text" });
  oval(s, 6.85, 1.3, 1.9, C.accent1, "hyp-circle");
  T(s, [{ text: E1.diff, options: { fontSize: 26, color: C.accent4, breakLine: true } }, { text: `очка ${E1.ci}`, options: { fontSize: 10, breakLine: true } }, { text: "быстрый симулятор", options: { fontSize: 9, color: C.text2 } }],
    { x: 6.85, y: 1.72, w: 1.9, h: 1.1, align: "center", valign: "middle", objectName: "hyp-stat" });
  T(s, "Число измерено для робота, который обновляет карту и перестраивает поиск; расследования с опытами показаны отдельным прогоном (пример 2)",
    { x: X0, y: 2.98, w: 5.5, h: 0.5, fontSize: 10, color: C.text2, objectName: "hyp-note" });
  hypothesisChecks(s, 3.68);
}
// ---------- математика, вариант Б: на числах одного прогона ----------
{
  const s = content("Математика на одном прогоне", "Те же формулы, подставлены числа из прогона в Gazebo (трудный уровень, сценарий 2)",
    "Покажу формулы на числах одного прогона. Датчик показал ноль пятьдесят три — значит, образец примерно в девяноста пяти сантиметрах. Расход пять и семь при прогнозе два и девять — грунт дороже вдвое. Из опытов выбран тот, что даёт больше информации на единицу заряда. И счёт: семь образцов, возврат и остаток заряда — девяносто один и восемь.", 40);
  tag(s, "Вариант Б · вместо слайда 4");
  const h = img(s, "scenario.png", X0, 1.38, 3.55, "Карта прогона в Gazebo с пятью отмеченными событиями");
  T(s, "Прогон в Gazebo, из записи которого взяты числа: ② — перерасход на 31-й секунде, ⑤ — финиш", { x: X0, y: 1.38 + h + 0.05, w: 3.55, h: 0.34, fontSize: 9.5, color: C.text2, objectName: "math-cap" });
  const steps = [[C.accent1, "Датчик → расстояние", "z = 1 − d/2", `z = ${ru(F.truth_vs_robot.z, 2)}  →  d ≈ (1 − ${ru(F.truth_vs_robot.z, 2)})·2 = ${ru(F.truth_vs_robot.d, 2)} м: образец на кольце`],
    [C.accent2, "Расход → цена грунта", "m̂ ≈ (ΔB − 0,01·Δt) / (2,5·Δs)", "5,7 ед/м при прогнозе 2,9  →  грунт дороже в 2,1 раза"],
    [C.accent3, "Выбор опыта", "I(e) / (c(e) + 0,05)", "постоять 2 с: 1,07/0,07 = 15,3; проехать 0,3 м: 0,50/1,45 = 0,34"],
    [C.accent1, "Запас на возврат", "B − C(туда) − 1,1·C(домой) − 4 ≥ 0", "к цели едет, только если потом хватит на дорогу домой"],
    [C.accent2, "Счёт судьи", "J = 10·N + 20·R + 0,1·B·R − штрафы", `10·7 + 20 + 0,1·18,3 = ${DEMO.score}`]];
  steps.forEach(([fill, head, formula, example], i) => {
    const y = 1.42 + i * 0.7;
    oval(s, 4.6, y + 0.04, 0.3, fill, `mstep-${i + 1}-dot`);
    T(s, String(i + 1), { x: 4.6, y: y + 0.04, w: 0.3, h: 0.3, fontSize: 10, align: "center", valign: "middle", objectName: `mstep-${i + 1}-n` });
    T(s, [{ text: head + "   ", options: { fontSize: 11 } }, { text: formula, options: { fontSize: 10.5, breakLine: true } }, { text: example, options: { fontSize: 10, color: C.text2 } }],
      { x: 5.03, y, w: 4.24, h: 0.66, objectName: `mstep-${i + 1}-text`, paraSpaceAfter: 2 });
  });
  T(s, "Сравнение вариантов — разность счёта на одних сценариях, 95% интервал", { x: 4.6, y: 4.86, w: 4.67, h: 0.2, fontSize: 9.5, color: C.text2, objectName: "math-note" });
}
// ---------- LLM, вариант Б: «ровно два образца» ----------
{
  const m = F.pair_llm_m1;
  llmSlide({
    title: "Пример 4. LLM читает миссию и меняет план",
    mission: "«Собери ровно два образца и сразу возвращайся на базу»",
    notes: "Миссию задаём словами: собери ровно два образца и возвращайся. Правило текста не читает и собирает все семь. Языковая модель после второго образца возвращает одну подцель: на базу.",
    figure: "pair_llm_m1.png", alt: "Две карты одного сценария: робот с правилом собирает все образцы, робот с языковой моделью собирает два и едет на базу",
    left: ["Правило: текст не читает", `собрал ${m.results[0].samples_collected} из 7: миссия нарушена`],
    right: ["LLM · qwen3.8-flash-next", `собрал 2, на базе на ${Math.round(m.results[1].t)}-й с: выполнена`],
    quoteHead: "Ответ модели после второго образца, дословно",
    quote: "Миссия: ровно 2 образца и возврат. Собрано 2 из требуемых 2 — цель выполнена. Заряд 44,3, возврат 12,5, запас 31,8. Время 27,5 из 600. Возвращаюсь на базу.",
    plan: "подцели: [ return_base ]",
    input: "Модель получает текст миссии и сводку состояния.",
    variant: "Вариант Б · вместо слайда 11",
  });
}
// ---------- итоги, вариант Б: «что готово / что дальше» в исходной вёрстке ----------
{
  const s = content("Что готово и что дальше", "В вёрстке исходного слайда «Что готово / что в работе», с нынешним состоянием",
    "Готово всё, что требует условие, бонусные треки и память о лаборатории между прогонами. Дальше — гипотезы от языковой модели внутри контура и проверка на настоящем роботе.", 25);
  tag(s, "Вариант Б · вместо слайда 12");
  oval(s, 0.95, 1.38, 3.6, C.accent3, "done-circle");
  T(s, [{ text: "Готово", options: { fontSize: 16, breakLine: true } },
    ...["ROS 2 + Gazebo, карта и движение по точкам", "Поиск образцов и адаптация: +13,3 очка (быстрый симулятор)", "Научный цикл: гипотезы, опыты, журнал", "LLM читает миссии словами: 45–46 из 48", "Память о лаборатории между прогонами: +4,3", "SLAM-карта и два робота с координацией"]
      .map((t, i, a) => ({ text: t, options: { fontSize: 10, breakLine: i < a.length - 1 } }))],
  { x: 1.4, y: 1.82, w: 2.7, h: 2.75, align: "center", valign: "middle", objectName: "done-text", paraSpaceAfter: 5 });
  oval(s, 5.45, 1.38, 3.6, C.accent2, "next-circle");
  T(s, [{ text: "Дальше", options: { fontSize: 16, breakLine: true } },
    ...["Гипотезы от LLM внутри контура: сейчас их выдвигает программа", "Память при другом расписании событий лаборатории", "Не стоять, пока модель думает: режим есть, по умолчанию выключен", "Проверка на настоящем роботе"]
      .map((t, i, a) => ({ text: t, options: { fontSize: 10, breakLine: i < a.length - 1 } }))],
  { x: 5.95, y: 1.9, w: 2.6, h: 2.6, align: "center", valign: "middle", objectName: "next-text", paraSpaceAfter: 6 });
}
// ---------- титул без карты ----------
SECTION = "Варианты на выбор";
slideTitle(false);

// =====================================================================================================
// ЗАПАСНЫЕ СЛАЙДЫ
// =====================================================================================================
SECTION = "Запасные слайды";
pres.addSection({ title: SECTION });
{
  const s = content("Бонусные треки: своя карта и два робота", "SLAM измерен в Gazebo, координация — в быстром симуляторе; запись двух роботов — из Gazebo",
    "Два бонусных трека. Робот сам строит карту через SLAM Toolbox: она совпадает с готовой на девяносто шесть — девяносто девять процентов клеток. И два робота с координацией: они делят арену и обмениваются сообщениями — прогон короче на двадцать шесть секунд.", 30);
  const sm = path.join(ROOT, "presentation", "assets", "final", "slam_map.png");
  const hs = img(s, sm, X0, 1.42, 2.75, "Карта, построенная роботом через SLAM Toolbox, с отмеченными местами столбов", "slam-map");
  caption(s, X0, 1.42 + hs + 0.08, 4.0, "SLAM вместо готовой карты", "совпала с готовой на 95,8–98,7% клеток; найдены 9 столбов из 9; начало координат сдвинуто на 1,5–5,2 см (5 прогонов Gazebo)", "slam-cap");
  T(s, "95,8–98,7%", { x: 3.62, y: 1.9, w: 1.5, h: 0.4, fontSize: 15, color: C.accent4, objectName: "slam-stat" });
  const ht = img(s, "two_robots.png", 5.2, 1.42, 2.75, "Пути двух роботов на одной арене: все пять образцов собраны");
  caption(s, 5.2, 1.42 + ht + 0.08, 4.07, "Два робота с координацией", "против пары без связи: прогон 103,7 → 77,8 с, заряда на четверть меньше, оба вернулись в 80 прогонах из 80 против 75 (80 сценариев)", "team-cap");
  T(s, "−26 с", { x: 8.08, y: 1.9, w: 1.2, h: 0.4, fontSize: 15, color: C.accent5, objectName: "team-stat" });
}
{
  const s = content("Реальная реализация: Gazebo и страница показа", "Одна команда поднимает мир, робота, судью и страницу показа",
    "Так это выглядит вживую: слева мир в Gazebo, справа страница показа — карта, заряд, счёт судьи и журнал решений робота.", 20);
  const hg = img(s, "gazebo.png", X0, 1.42, 3.5, "Окно Gazebo: арена turtlebot3_world с роботом");
  caption(s, X0, 1.42 + hg + 0.08, 3.5, "Gazebo Sim + TurtleBot3", "мир turtlebot3_world без изменений; метки на полу — подсказка зрителю", "gz-cap");
  T(s, "pixi run demo --level hard --seed 2 --agent scientist_v2 --open", { x: X0, y: 4.78, w: CW, h: 0.24, fontSize: 10, color: C.text2, valign: "middle", objectName: "command" });
  const hl = img(s, "lab_mission.png", 4.5, 1.42, 4.77, "Страница показа во время миссии: карта с путём робота, заряд, счёт судьи и журнал решений");
  caption(s, 4.5, 1.42 + hl + 0.08, 4.77, "Страница показа", "путь робота, заряд, счёт судьи и последние решения агента — в реальном времени", "lab-cap");
}

// ---------- текст доклада и хронометраж ----------
function writeSpeech() {
  const lines = ["# Текст доклада к колоде DID_v2", "",
    "Собирается вместе с колодой (`node build_team.js`); тот же текст лежит в заметках к слайдам.", ""];
  let section = null, n = 0;
  for (const p of SPEECH_PARTS) {
    n += 1;
    if (p.section !== section) {
      section = p.section;
      const total = SPEECH_PARTS.filter((q) => q.section === section).reduce((a, q) => a + q.seconds, 0);
      lines.push(`## ${section}${section === "Основной поток" ? ` — ${Math.floor(total / 60)} мин ${total % 60} с` : ""}`, "");
    }
    lines.push(`**${n}. ${p.title}** (≈ ${p.seconds} с)`, "", p.notes, "");
  }
  fs.writeFileSync(SPEECH, lines.join("\n"));
}

function render() {
  const find = (names) => names.find((n) => (n.includes("/") ? fs.existsSync(n) : spawnSync("which", [n]).status === 0)) || null;
  const soffice = find(["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice", "libreoffice"]);
  const pdftoppm = find(["pdftoppm", "/opt/homebrew/bin/pdftoppm"]);
  if (!soffice || !pdftoppm) return console.warn("нет LibreOffice или pdftoppm: PDF и картинки не обновлены");
  const profile = path.join(os.tmpdir(), "did_lo_profile");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "did_pdf_"));
  const r = spawnSync(soffice, [`-env:UserInstallation=file://${profile}`, "--headless", "--convert-to", "pdf", "--outdir", tmp, OUT], { encoding: "utf8", timeout: 240000 });
  const made = path.join(tmp, `${NAME}.pdf`);
  if (r.status !== 0 || !fs.existsSync(made)) { console.error("LibreOffice не собрал PDF:\n" + (r.stderr || r.stdout || r.error)); process.exit(1); }
  fs.copyFileSync(made, PDF);
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.rmSync(SLIDES, { recursive: true, force: true });
  fs.mkdirSync(SLIDES);
  const p = spawnSync(pdftoppm, ["-jpeg", "-r", "150", PDF, path.join(SLIDES, "slide")], { encoding: "utf8" });
  if (p.status !== 0) { console.error(p.stderr); process.exit(1); }
  const files = fs.readdirSync(SLIDES).filter((f) => f.endsWith(".jpg")).sort();
  console.log(`PDF: ${PDF}\nкартинки: ${SLIDES} (${files.length})`);
  const sheet = `
import sys
from PIL import Image
files = sys.argv[2:]
ims = [Image.open(f) for f in files]
cols = 4
w, h = 500, int(500 * ims[0].height / ims[0].width)
rows = (len(ims) + cols - 1) // cols
out = Image.new('RGB', (cols * w + (cols + 1) * 8, rows * h + (rows + 1) * 8), (120, 128, 140))
for i, im in enumerate(ims):
    out.paste(im.resize((w, h)), (8 + (i % cols) * (w + 8), 8 + (i // cols) * (h + 8)))
out.save(sys.argv[1], quality=88)
`;
  const q = spawnSync("/usr/local/bin/python3", ["-c", sheet, PREVIEW, ...files.map((f) => path.join(SLIDES, f))], { encoding: "utf8" });
  if (q.status !== 0) console.warn("общий лист не собран: " + q.stderr);
  else console.log("общий лист:", PREVIEW);
}

// Шрифт Inter вшит в колоду команды (ppt/fonts/*.fntdata, только обычное начертание) — вшиваем его же, иначе там,
// где Inter не установлен, текст уйдёт в запасной шрифт. Полужирный программы показа синтезируют сами.
async function embedFont() {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  const edit = async (name, fn) => zip.file(name, fn(await zip.file(name).async("string")));
  zip.file("ppt/fonts/inter.fntdata", fs.readFileSync(path.join(__dirname, "data", "inter.fntdata")));
  await edit("[Content_Types].xml", (x) => (x.includes('Extension="fntdata"') ? x : x.replace("</Types>", '<Default Extension="fntdata" ContentType="application/x-fontdata"/></Types>')));
  await edit("ppt/_rels/presentation.xml.rels", (x) => x.replace("</Relationships>",
    '<Relationship Id="rIdFontInter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/inter.fntdata"/></Relationships>'));
  await edit("ppt/presentation.xml", (x) => {
    const m = x.match(/<p:notesSz[^>]*\/>/);
    if (!m) throw new Error("в presentation.xml нет notesSz: шрифт не вшит");
    if (!x.includes("embedTrueTypeFonts")) x = x.replace('autoCompressPictures="0"', 'autoCompressPictures="0" embedTrueTypeFonts="1"');
    if (!x.includes("embedTrueTypeFonts")) throw new Error("в presentation.xml не найден корневой элемент: шрифт не вшит");
    return x.replace(m[0], `${m[0]}<p:embeddedFontLst><p:embeddedFont><p:font typeface="${FONT}"/><p:regular r:id="rIdFontInter"/></p:embeddedFont></p:embeddedFontLst>`);
  });
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  await embedFont();
  writeSpeech();
  console.log("готово:", OUT, "· слайдов", SPEECH_PARTS.length);
  if (!process.argv.includes("--no-pdf")) render();
})();
