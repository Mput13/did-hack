// Презентация второго чекпоинта. Одна команда собирает всё заново:
//
//   cd presentation && node build_checkpoint2.js
//
// Что делает: 1) обновляет рисунки и числа из записей прогонов (figures/checkpoint2_figures.py);
// 2) собирает checkpoint2.pptx; 3) пишет текст доклада checkpoint2_speech.md (он же — заметки к слайдам);
// 4) делает checkpoint2.pdf и картинки слайдов (render_checkpoint2.js), если есть чем.
//
// Ключи: --no-refresh — не пересчитывать рисунки и числа; --tests — заодно прогнать автоматические проверки;
//        --no-pdf — только pptx и текст доклада.
//
// Числа на слайдах читаются из runs/*/summary.json, runs/llm_real/*.summary.json и data/checkpoint2.json.
// Снимки показа берутся из demo/shots/, шаги сценария — из docs/demo_script.md, если эти файлы есть:
// один снимок встаёт в середину слайда «Сценарий», два — занимают его нижнюю половину целиком.
// Пока их нет, в середине стоит снимок проигрывателя веб-лаборатории (figures/lab_shots.js).
const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const pptxgen = require("pptxgenjs");

// apply_theme.js лежит вне проекта и ищет jszip через NODE_PATH.
process.env.NODE_PATH = [path.join(__dirname, "node_modules"), process.env.NODE_PATH].filter(Boolean).join(path.delimiter);
require("module").Module._initPaths();

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/Users/a/.claude/skills/synced/0ec766ba-a970-423d-9985-0a1698a18ad8_57a5e5ff-4dfe-4d5a-9b46-49c5ea27665a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ARGS = new Set(process.argv.slice(2));
const ROOT = path.join(__dirname, "..");
const RUNS = path.join(ROOT, "runs");
const OUT = path.join(__dirname, "checkpoint2.pptx");
const SPEECH = path.join(__dirname, "checkpoint2_speech.md");
const ASSET = (name) => path.join(__dirname, "assets", "c2", name);

// ---------- 0. рисунки и числа из записей прогонов ----------
if (!ARGS.has("--no-refresh")) {
  const py = process.env.PYTHON || "python3";
  const args = [path.join(__dirname, "figures", "checkpoint2_figures.py")].concat(ARGS.has("--tests") ? ["--tests"] : []);
  const r = spawnSync(py, args, { cwd: ROOT, encoding: "utf8" });
  if (r.status !== 0) console.warn("рисунки и числа не обновлены, беру прежние:\n" + String(r.stderr || r.error).split("\n").slice(-4).join("\n"));
}

// ---------- 1. данные ----------
const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const D = readJson(path.join(__dirname, "data", "checkpoint2.json"));
const EXP = {};
for (const dir of fs.readdirSync(RUNS)) {
  const f = path.join(RUNS, dir, "summary.json");
  if (/^E\d+$/.test(dir) && fs.existsSync(f)) EXP[dir] = readJson(f);
}
const llmSummary = (name) => { const f = path.join(RUNS, "llm_real", `${name}.summary.json`); return fs.existsSync(f) ? readJson(f) : null; };
const LLM = llmSummary("gpt-6-luna");          // планировщик на большой модели
const QWEN = llmSummary("qwen2.5-3b");         // локальная модель через Ollama
// Сколько слабых мест в расследованиях нашло второе мнение другой модели (все исправлены) — со слов команды, не из файла.
const SECOND_OPINION_FIXES = 8;

const ru = (v, d = 1) => Number(v).toFixed(d).replace(".", ",").replace("-", "−");
const int = (v) => String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const pct = (v, d = 0) => ru(v * 100, d);
const r1 = (v) => Math.round(v * 1000) / 10;         // доля → проценты с одним знаком
// Согласование с числом: pl(3262, "прогон", "прогона", "прогонов") → «прогона».
const pl = (n, one, few, many) => {
  const a = Math.abs(Math.round(n)) % 100, b = a % 10;
  return a > 10 && a < 20 ? many : b === 1 ? one : b >= 2 && b <= 4 ? few : many;
};
const nRuns = (n) => `${int(n)} ${pl(n, "прогон", "прогона", "прогонов")}`;
const nSeries = (n) => `${n} ${pl(n, "серия", "серии", "серий")}`;
const nChecks = (n) => `${n} ${pl(n, "автоматическая проверка", "автоматические проверки", "автоматических проверок")}`;

function group(exp, arm, level, condition = null) {
  const g = EXP[exp].groups.find((x) => x.arm === arm && String(x.level) === level && (condition === null || x.condition === condition));
  if (!g) throw new Error(`нет группы ${exp} ${arm} ${level} ${condition}`);
  return g;
}
const mean = (exp, arm, level, metric, condition = null) => group(exp, arm, level, condition).stats[metric].mean;
function pair(exp, metric, a, b, level, condition = null) {
  const c = EXP[exp].claims.find((x) => x.metric === metric && x.a === a && x.b === b);
  const cell = c.cells.find((x) => String(x.level) === level && (condition === null || x.condition === condition));
  return { ...cell.pair, status: c.status };
}
const runsOf = (e) => (typeof EXP[e].runs === "number" ? EXP[e].runs : EXP[e].runs.length);
const EXP_IDS = Object.keys(EXP).sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
const TOTAL_RUNS = EXP_IDS.reduce((s, e) => s + runsOf(e), 0);
// Проверки: если последний полный прогон прошёл — число прошедших, иначе число найденных в проекте.
const TESTS = D.tests ? ((D.tests.ok && D.tests.passed) || D.tests.collected || null) : null;
const TESTS_PASS = Boolean(D.tests && D.tests.ok && D.tests.passed);

// E1: адаптивный против фиксированного плана
const LEVELS = ["easy", "medium", "hard"];
const E1 = {
  fixed: LEVELS.map((l) => r1(mean("E1", "fixed", l, "samples_share"))),
  adaptive: LEVELS.map((l) => r1(mean("E1", "adaptive", l, "samples_share"))),
  diff: pair("E1", "samples_share", "adaptive", "fixed", "all"),
  retA: mean("E1", "adaptive", "all", "returned"),
  retF: mean("E1", "fixed", "all", "returned"),
  ret: pair("E1", "returned", "adaptive", "fixed", "all"),
  n: group("E1", "adaptive", "easy").n,
};
// E9: стратегии поиска
const E9_ARMS = [["adaptive", "Карта вероятностей"], ["gradient", "Подъём по сигналу"], ["fixed", "Фиксированный план"], ["spiral", "Спираль"]];
const E9 = E9_ARMS.map(([arm, label]) => ({ arm, label, share: r1(mean("E9", arm, "all", "samples_share")), n: group("E9", arm, "all").n }));
// E10: исследователь
const E10 = {
  labels: ["Исследователь", "С адаптацией", "Фиксированный план"],
  ret: ["scientist", "adaptive", "fixed"].map((a) => r1(mean("E10", a, "hard", "returned"))),
  correct: mean("E10", "scientist", "all", "inq_correct"),
  retPair: pair("E10", "returned", "scientist", "adaptive", "hard"),
  scorePair: pair("E10", "score", "scientist", "adaptive", "hard"),
  n: group("E10", "scientist", "hard").n,
};
E10.home = ["scientist", "adaptive"].map((a) => Math.round(mean("E10", a, "hard", "returned") * group("E10", a, "hard").n));
// Выводы исследователя на трудном уровне, сверенные со скрытой правдой: суммы по записям прогонов.
E10.inq = (() => {
  if (!Array.isArray(EXP.E10.runs)) return null;
  const sum = { total: 0, identified: 0, insufficient: 0, correct: 0, partial: 0, wrong: 0, faults: 0, faults_found: 0 };
  for (const r of EXP.E10.runs) {
    const q = r.arm === "scientist" && r.level === "hard" && r.metrics && r.metrics.inquiries;
    if (q) for (const k of Object.keys(sum)) sum[k] += q[k] || 0;
  }
  return sum.identified ? sum : null;
})();
// Память между прогонами (E11): есть ли хоть одно подтверждённое улучшение.
const MEMORY_HELPS = Boolean(EXP.E11 && EXP.E11.claims.some((c) => c.status === "supported"));
// E8: уход одометрии
const E8 = {
  lidar: mean("E8", "lidar", "medium", "returned", "moderate"),
  odom: mean("E8", "odom", "medium", "returned", "moderate"),
};
// Слова на слайдах зависят от того, что показали данные: серии пересчитываются, статусы могут смениться.
const claimStatus = (exp, metric, a, b) => (EXP[exp].claims.find((x) => x.metric === metric && x.a === a && x.b === b) || {}).status;
const WARN = [];
const expect = (ok, msg) => { if (!ok) WARN.push(msg); };
const PENDING = fs.readdirSync(RUNS).filter((d) => /^E\d+$/.test(d) && !EXP[d]);
expect(PENDING.length === 0, `серии ${PENDING.join(", ")} сейчас пересчитываются (нет summary.json) и в подсчёт серий и прогонов не вошли — пересоберите позже.`);
E1.moreSamples = E1.diff.status === "supported";
E1.allLevels = LEVELS.every((l) => pair("E1", "samples_share", "adaptive", "fixed", l).ci[0] > 0);
E1.retWord = E1.ret.status === "inconclusive" ? "same" : (E1.ret.mean >= 0 ? "notWorse" : "worse");
E10.retBetter = E10.retPair.status === "supported";
const E9_BEST = E9.every((e) => e.share <= E9[0].share);
expect(claimStatus("E2", "score", "adaptive", "no_soil") === "inconclusive" && claimStatus("E2", "score", "adaptive", "no_hazard") === "inconclusive",
  "E2: обучение грунтам или память об опасных зонах теперь показывают вклад — поправьте блок «Не решено» на слайде 7.");
expect(Math.abs(pair("E3", "score", "adaptive", "fixed", "hard", "none").mean - pair("E3", "score", "adaptive", "fixed", "hard", "soil").mean) < 5,
  "E3: преимущество при изменениях среды заметно отличается от преимущества без них — поправьте блок «Не решено» на слайде 7.");
expect(E9_BEST, "E9: карта вероятностей больше не лучшая стратегия — поправьте слайд 9.");
expect(E1.moreSamples, "E1: утверждение «адаптивный собирает больше образцов» больше не подтверждается — поправьте слайд 8.");
const GZ = D.gazebo;
const gz = (level) => GZ.runs.find((r) => r.level === level);
const GZ_ALL_HOME = GZ.runs.every((r) => r.gazebo.returned && r.gazebo.collisions === 0);
const SERIES_LEVEL = "easy";
const R = D.rules;
const RANGE_M = R.battery_start / R.drain_per_m;

// ---------- 2. снимки показа и шаги сценария, если их уже положили ----------
const SHOTS_DIR = process.env.DEMO_SHOTS_DIR || path.join(ROOT, "demo", "shots");
// Запасной снимок интерфейса (проигрыватель веб-лаборатории): node figures/lab_shots.js при запущенном pixi run lab.
const LAB_SHOT = path.join(__dirname, "assets", "c2", "shots", "lab_run_gazebo_hard.png");
const LAB_META = fs.existsSync(LAB_SHOT.replace(/\.png$/, ".json")) ? JSON.parse(fs.readFileSync(LAB_SHOT.replace(/\.png$/, ".json"), "utf8")) : null;
const SHOTS = fs.existsSync(SHOTS_DIR)
  ? fs.readdirSync(SHOTS_DIR).filter((f) => /\.(png|jpe?g)$/i.test(f)).sort().map((f) => path.join(SHOTS_DIR, f))
  : [];
function shotCaption(file) {
  const name = path.basename(file).toLowerCase();
  if (/pult|pilot|пульт/.test(name)) return "Пульт: щелчок по карте задаёт точку, робот едет, карта строится";
  if (/gazebo|gz/.test(name)) return "Окно Gazebo: тот же прогон в симуляторе с физикой";
  if (/rviz/.test(name)) return "RViz: что видит и думает робот";
  if (/replay|player|проигр/.test(name)) return "Проигрыватель: прогон по секундам";
  return path.basename(file).replace(/\.[^.]+$/, "").replace(/^\d+[_-]?/, "").replace(/[_-]+/g, " ");
}
// Сценарий показа (docs/demo_script.md): полный путь «запуск → карта → задача → движение → результат».
// Текст шагов на слайде написан по этому файлу; из него же читаются числа и проверяется, что шаги не поменялись.
const DEMO_FILE = path.join(ROOT, "docs", "demo_script.md");
const DEMO = fs.existsSync(DEMO_FILE) ? fs.readFileSync(DEMO_FILE, "utf8") : "";
const demoNum = (re) => { const m = DEMO.match(re); return m ? m[1].replace(".", ",") : null; };
const DEMO_FIRST_MAP = demoNum(/Карта построена:\s*(\d+)\s*%/);       // сколько пола лидар видит ещё до движения
const DEMO_READY_S = demoNum(/около\s+(\d+)\s+секунд от запуска/);
const DEMO_STEP_NAMES = [...DEMO.matchAll(/^\|\s*(\d+)\.\s*([^|]+?)\s*\|/gm)].map((m) => m[2]);
if (DEMO && !["Запуск", "Маршрут", "Миссия"].every((w) => DEMO_STEP_NAMES.some((n) => n.startsWith(w)))) {
  console.warn("ВНИМАНИЕ: шаги в docs/demo_script.md изменились (" + DEMO_STEP_NAMES.join(", ") + ") — сверьте слайд «Сценарий работы робота».");
}
const STEPS = [
  ["Запуск", "одна команда поднимает мир, робота, судью и пульт"],
  ["Карта", DEMO_FIRST_MAP ? `робот ещё стоит, а лидар уже видит ${DEMO_FIRST_MAP}% пола` : "лидар строит карту с первой секунды"],
  ["Задача", "точка, маршрут щелчком по карте или миссия"],
  ["Движение", "путь в обход столбов, положение — по лидару"],
  ["Результат", "образцы собраны, робот на базе, счёт судьи"],
];
// Из снимков показа берём один с пультом и один с окном Gazebo, если по именам их можно различить.
function pickShots(files) {
  const pult = files.find((f) => /pult|pilot|пульт/i.test(path.basename(f)));
  const gzw = files.find((f) => /gazebo|gz/i.test(path.basename(f)) && f !== pult);
  const picked = [pult, gzw].filter(Boolean);
  for (const f of files) if (picked.length < 2 && !picked.includes(f)) picked.push(f);
  return picked.slice(0, 2);
}
const DEMO_SHOTS = pickShots(SHOTS);

// ---------- 3. оформление ----------
const TEAM = [
  ["Журавлёва Полина Петровна", "разработчик (Back/ROS)"],
  ["Ефремова Анастасия Михайловна", "ML (LLM)"],
  ["Путиловский Михаил Вячеславович", "аналитик (исследователь)"],
];
const THEME = {
  name: "DID Researcher",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "14181C", lt1: "FFFFFF", dk2: "2B3138", lt2: "EEF0F2",
    accent1: "EB6834", accent2: "2A78D6", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "5B6168", accent6: "D03B3B", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
// В диаграммах цвета задаются только числом. Постоянные: фиксированный план, адаптивный, исследователь.
const HEX = { fixed: "2A78D6", adaptive: "EB6834", scientist: "1BAF7A", context: "9AA0A6", ink: "14181C", muted: "5B6168", grid: "DDE0E4", soil: "E9B291" };
const CHART_FONT = "+mn-lt";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Автономный ИИ-исследователь на роботе-платформе — чекпоинт 2";
pres.author = TEAM.map((m) => m[0].split(" ")[0]).join(", ");
const C = pres.SchemeColor;
const FOOTER = "Автономный ИИ-исследователь · DID Hack 2026 · чекпоинт 2";
const X0 = 0.6, CW = 12.13, TOP = 1.85, BOTTOM = 6.8;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.2, w: 6.7, h: 2.35, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.7, w: 6.5, h: 0.8, fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.42, w: CW, h: 0.75, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 1.2, w: CW, h: 0.42, fontSize: 16, color: C.accent5, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: X0, y: 7.0, w: 8.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

const SPEECH_PARTS = [];
function content(section, title, lead, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addText(lead, { placeholder: "body" });
  s.addNotes(notes.join(" "));
  SPEECH_PARTS.push({ title, notes });
  return s;
}
function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, valign: "top", align: "left", color: C.text1, fontSize: 14, ...o });
}
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, objectName: name });
}
function hexBadge(slide, label, x, y, size, name, fill = C.accent1, fontSize = Math.round(size * 28), color = C.background1) {
  slide.addText(String(label), {
    shape: pres.shapes.HEXAGON, x, y, w: size * 1.14, h: size, fill: { color: fill }, color,
    bold: true, fontSize, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name,
  });
}
function imageSize(file) {
  const b = fs.readFileSync(file);
  if (b.readUInt32BE(0) === 0x89504e47) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  for (let i = 2; i < b.length - 9;) {            // JPEG: ищем заголовок кадра
    if (b[i] !== 0xff) { i += 1; continue; }
    const m = b[i + 1];
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { w: b.readUInt16BE(i + 7), h: b.readUInt16BE(i + 5) };
    i += 2 + b.readUInt16BE(i + 2);
  }
  throw new Error("не понял размер картинки " + file);
}
// Картинка вписывается в прямоугольник целиком, без искажения.
function image(slide, file, box, name, altText) {
  const { w: pw, h: ph } = imageSize(file);
  const k = Math.min(box.w / pw, box.h / ph);
  const w = pw * k, h = ph * k;
  const x = box.x + (box.w - w) / 2, y = box.y + (box.align === "top" ? 0 : (box.h - h) / 2);
  slide.addImage({ path: file, x, y, w, h, objectName: name, altText });
  return { x, y, w, h };
}
function arrow(slide, x1, y, x2, name, color = C.accent5) {
  const left = x2 < x1;
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y, w: Math.abs(x2 - x1), h: 0, objectName: name,
    line: { color, width: 1.5, beginArrowType: left ? "triangle" : "none", endArrowType: left ? "none" : "triangle" },
  });
}
function arrowDown(slide, x, y1, y2, name, color = C.accent5) {
  slide.addShape(pres.shapes.LINE, { x, y: y1, w: 0, h: y2 - y1, objectName: name, line: { color, width: 1.5, endArrowType: "triangle" } });
}
function statTile(slide, x, y, w, h, value, label, name, valueSize = 34) {
  card(slide, x, y, w, h, `${name}-card`);
  text(slide, value, { x: x + 0.25, y: y + 0.14, w: w - 0.5, h: valueSize / 72 * 1.25, fontSize: valueSize, bold: true, valign: "middle", objectName: `${name}-value` });
  const ly = y + 0.14 + valueSize / 72 * 1.25 + 0.04;
  text(slide, label, { x: x + 0.25, y: ly, w: w - 0.5, h: y + h - ly - 0.1, fontSize: 14, color: C.text2, objectName: `${name}-label` });
}
function bullets(slide, items, o) {
  const paragraphs = items.map((item, k) => ({ text: item, options: { bullet: true, breakLine: k < items.length - 1 } }));
  text(slide, paragraphs, { paraSpaceAfter: 6, ...o });
}
const axisText = { catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: CHART_FONT, valAxisLabelFontFace: CHART_FONT, catAxisLabelFontSize: 14, valAxisLabelFontSize: 12 };
const labelText = { showValue: true, dataLabelColor: HEX.ink, dataLabelFontFace: CHART_FONT, dataLabelFontSize: 14, dataLabelFontBold: true };
const quietFrame = { valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, catAxisLineShow: true, valAxisLineShow: false };
const legendText = { showLegend: true, legendPos: "b", legendFontSize: 14, legendColor: HEX.ink, legendFontFace: CHART_FONT };

const SEC_A = "Задача и решение", SEC_B = "Состояние и результаты", SEC_C = "Как работали";

// ---------- 1. титул ----------
pres.addSection({ title: SEC_A });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: SEC_A });
  s.addText("«Автономный ИИ-исследователь на роботе-платформе»", { placeholder: "title" });
  s.addText("Второй чекпоинт: что работает и что показали опыты", { placeholder: "body" });
  text(s, "DID Hack 2026 · 8 октября", { x: 0.7, y: 4.6, w: 6.0, h: 0.4, fontSize: 16, color: C.accent1, bold: true, objectName: "event" });
  text(s, TEAM.map(([name, role], i) => ({ text: `${name} — ${role}`, options: { breakLine: i < TEAM.length - 1 } })),
    { x: 0.7, y: 5.3, w: 6.5, h: 1.1, fontSize: 14, color: C.background2, paraSpaceAfter: 4, objectName: "team" });
  image(s, ASSET(D.title), { x: 7.5, y: 0.75, w: 5.3, h: 5.3 }, "title-run", "Арена и путь робота в настоящем прогоне в Gazebo");
  const m = gz("medium").gazebo;
  text(s, `Настоящий прогон в Gazebo: путь робота, собрано ${m.samples_collected} из ${m.samples_total} образцов, возврат на базу`,
    { x: 7.5, y: 6.15, w: 5.3, h: 0.5, fontSize: 12, color: C.background2, align: "center", objectName: "title-caption" });
  const notes = [
    "Наш проект — автономный ИИ-исследователь на роботе TurtleBot3.",
    `Покажем, что уже работает и что показали опыты. На картинке настоящий прогон в Gazebo: собрано ${m.samples_collected} образцов из ${m.samples_total}, робот вернулся на базу.`,
  ];
  s.addNotes(notes.join(" "));
  SPEECH_PARTS.push({ title: "Титул", notes });
}

// ---------- 2. задача и цель ----------
{
  const lv = D.levels;
  const s = content(SEC_A, "Задача и цель проекта", "Робот ищет то, чего не видит: образцы, дорогой грунт и опасные зоны скрыты", [
    "На арене спрятаны образцы, дорогой грунт и опасные зоны, и робот их не видит: камеры нет, а датчик говорит только, насколько близко образец.",
    `Цель — собрать больше образцов и вернуться на базу, пока не села батарея: её хватает на ${ru(RANGE_M, 0)} метра.`,
    "Гипотеза: агент, который учится по измерениям, соберёт больше, чем агент с планом, составленным заранее.",
  ]);
  const rows = [
    [C.accent4, "Образцы", `найти и собрать, подъехав ближе ${ru(R.collect_radius_m * 100, 0)} см`],
    [HEX.soil, "Дорогой грунт", `заряд уходит в ${ru(Math.min(...Object.values(lv).map((l) => l.soil_mult[0])), 0)}–${ru(Math.max(...Object.values(lv).map((l) => l.soil_mult[1])), 0)} раза быстрее`],
    [C.accent6, "Опасные зоны", "въехал — штраф и потеря заряда"],
  ];
  rows.forEach(([fill, head, body], i) => {
    const y = TOP + i * 0.88;
    hexBadge(s, i === 2 ? "!" : "", X0, y + 0.05, 0.5, `hidden-${i + 1}-badge`, fill, 16);
    text(s, head, { x: X0 + 0.8, y, w: 4.5, h: 0.32, fontSize: 16, bold: true, objectName: `hidden-${i + 1}-head` });
    text(s, body, { x: X0 + 0.8, y: y + 0.34, w: 4.5, h: 0.4, fontSize: 14, color: C.text2, objectName: `hidden-${i + 1}-body` });
  });
  card(s, X0, 4.55, 5.3, 1.05, "goal-card", C.text2);
  text(s, [
    { text: "Цель. ", options: { bold: true, color: C.accent1 } },
    { text: `Собрать больше образцов и вернуться на базу, пока не села батарея: ${ru(R.battery_start, 0)} единиц заряда — это ${ru(RANGE_M, 0)} м пути.`, options: { color: C.background1 } },
  ], { x: X0 + 0.25, y: 4.55, w: 4.8, h: 1.05, fontSize: 15, valign: "middle", objectName: "goal-text" });
  text(s, [
    { text: "Гипотеза. ", options: { bold: true } },
    { text: "Агент, который учится по измерениям и перестраивает маршрут, соберёт больше образцов, чем агент с планом, составленным до старта." },
  ], { x: X0, y: 5.8, w: 5.3, h: 1.0, fontSize: 14, color: C.text2, objectName: "hypothesis" });

  const iw = 3.1, ix = [6.25, 9.63], ih = 2.97;
  [["Как на самом деле", D.task.truth, `уровень ${D.task.level}: ${D.task.samples} ${pl(D.task.samples, "образец", "образца", "образцов")}, ${D.task.soils} ${pl(D.task.soils, "зона", "зоны", "зон")} грунта, опасные зоны`, "Арена со скрытой правдой: образцы, зоны дорогого грунта и опасная зона"],
    ["Что видит робот", D.task.robot_view, "стены по лидару и одно число датчика: образец где-то на синей окружности", "То же место глазами робота: точки лидара на стенах и окружность возможного положения образца"]]
    .forEach(([head, file, cap, alt], i) => {
      text(s, head, { x: ix[i], y: TOP, w: iw, h: 0.3, fontSize: 14, bold: true, objectName: `view-${i + 1}-head` });
      image(s, ASSET(file), { x: ix[i], y: TOP + 0.35, w: iw, h: ih }, `view-${i + 1}`, alt);
      text(s, cap, { x: ix[i], y: TOP + 0.4 + ih, w: iw, h: 0.5, fontSize: 12, color: C.accent5, objectName: `view-${i + 1}-caption` });
    });
  const nSamples = (n) => `${n} ${pl(n, "образец", "образца", "образцов")}`;
  const nZones = (n) => `${n} ${pl(n, "зона", "зоны", "зон")}`;
  const chips = [
    ["easy", `${nSamples(lv.easy.samples)}, ${nZones(lv.easy.soils)} грунта`],
    ["medium", `${nSamples(lv.medium.samples)}, ${nZones(lv.medium.soils)} грунта`],
    ["hard", `${nSamples(lv.hard.samples)}, ${nZones(lv.hard.soils)}${lv.hard.events.length ? "; среда меняется на ходу" : " грунта"}`],
  ];
  const cw = [1.9, 2.0, 2.38], gap = 0.1;
  let cx = 6.25;
  chips.forEach(([name, desc], i) => {
    card(s, cx, 5.9, cw[i], 0.9, `level-${i + 1}-card`);
    text(s, name, { x: cx + 0.15, y: 5.96, w: cw[i] - 0.3, h: 0.28, fontSize: 13, bold: true, objectName: `level-${i + 1}-name` });
    text(s, desc, { x: cx + 0.15, y: 6.25, w: cw[i] - 0.3, h: 0.5, fontSize: 12, color: C.text2, objectName: `level-${i + 1}-desc` });
    cx += cw[i] + gap;
  });
}

// ---------- 3. сценарий работы робота ----------
{
  const st = D.story, res = st.result;
  const runLine = `настоящая миссия в Gazebo на трудном уровне: собрано ${res.samples_collected} из ${res.samples_total}, робот вернулся на базу`;
  const s = content(SEC_A, "Сценарий работы робота", "Полный сценарий одной командой: запуск → карта → задача → движение → результат", [
    "Полный сценарий — пять шагов: запуск одной командой, карта по лидару, задача щелчком по карте или миссия, движение и результат.",
    `На кадрах ${runLine}. По дороге — штраф и сбой датчика: робот записал ${st.hypotheses.total} ${pl(st.hypotheses.total, "гипотезу", "гипотезы", "гипотез")}, подтвердил ${st.hypotheses.confirmed}.`,
    "ЗДЕСЬ — ЖИВОЙ ПОКАЗ (docs/demo_script.md): пульт уже открыт; показать карту из одной точки, поставить маршрут из пяти-шести точек, нажать «Ехать», затем «Домой» и «Запустить миссию».",
  ]);
  const n = STEPS.length, sw = (CW - (n - 1) * 0.2) / n;
  STEPS.forEach(([head, body], i) => {
    const x = X0 + i * (sw + 0.2);
    hexBadge(s, i + 1, x, TOP + 0.02, 0.44, `step-${i + 1}-badge`, C.text2);
    text(s, head, { x: x + 0.62, y: TOP, w: sw - 0.62, h: 0.48, fontSize: 16, bold: true, valign: "middle", objectName: `step-${i + 1}-head` });
    text(s, body, { x, y: TOP + 0.55, w: sw, h: 0.5, fontSize: 13, color: C.text2, objectName: `step-${i + 1}-body` });
  });
  const y0 = 2.98;
  const quote = (q) => {
    if (!q) return "";
    let t = q.text.split(". Проверка")[0].replace(/(\d)\.(\d)/g, "$1,$2").replace(/-(\d)/g, "−$1");
    if (t.length > 56) {                       // длинную запись режем по концу фразы или по запятой
      const cutAt = Math.max(t.lastIndexOf(". ", 56), t.lastIndexOf(", ", 56));
      t = cutAt > 20 ? t.slice(0, cutAt) : t.slice(0, 55).replace(/\s+\S*$/, "") + "…";
    }
    return t;
  };
  // Кадр прогона с подписью: секунда прогона по часам судьи (как в проигрывателе), заряд, собрано; строка журнала.
  const frame = (f, i, x, fw) => {
    const fh = fw * 4.79 / 5;
    image(s, ASSET(f.file), { x, y: y0, w: fw, h: fh }, `frame-${i + 1}`, `Кадр прогона: ${f.label}`);
    text(s, `${ru(f.t, 0)} с · заряд ${ru(f.battery, 0)} · собрано ${f.collected} из ${f.total}`,
      { x: x - 0.15, y: y0 + fh + 0.04, w: fw + 0.3, h: 0.28, fontSize: 13, bold: true, align: "center", valign: "middle", objectName: `frame-${i + 1}-state` });
    text(s, `Журнал: «${quote(f.quote)}»`, { x: x - 0.15, y: y0 + fh + 0.32, w: fw + 0.3, h: 0.36, fontSize: 11, color: C.text2, align: "center", objectName: `frame-${i + 1}-quote` });
  };
  const legend = () => text(s, `Кадры — ${runLine}. Синее — где агент ждёт образец, жёлтые кольца — где образцы на самом деле, зелёное — собрано, красное — штраф.`,
    { x: X0, y: 6.42, w: CW, h: 0.38, fontSize: 11, color: C.accent5, align: "center", valign: "top", objectName: "frames-legend" });
  const center = DEMO_SHOTS[0] || (fs.existsSync(LAB_SHOT) ? LAB_SHOT : null);
  if (DEMO_SHOTS.length >= 2) {
    // Снимки показа: два рядом, во всю ширину.
    const w = (CW - 0.3) / 2, h = 3.1;
    DEMO_SHOTS.forEach((file, i) => {
      const x = X0 + i * (w + 0.3);
      const r = image(s, file, { x, y: y0, w, h, align: "top" }, `shot-${i + 1}`, shotCaption(file));
      text(s, shotCaption(file), { x, y: r.y + r.h + 0.08, w, h: 0.4, fontSize: 13, color: C.text2, align: "center", objectName: `shot-${i + 1}-caption` });
    });
  } else if (center) {
    // Начало и конец прогона — кадрами, середина — снимком интерфейса (или первым снимком показа).
    const cw = 5.9, fw = (CW - cw - 0.5) / 2;
    frame(st.frames[0], 0, X0, fw);
    frame(st.frames[st.frames.length - 1], st.frames.length - 1, X0 + CW - fw, fw);
    const cap = DEMO_SHOTS[0] ? shotCaption(center)
      : `Веб-лаборатория: тот же прогон в проигрывателе${LAB_META && LAB_META.time_s ? `, ${ru(LAB_META.time_s, 0)} с` : ""}`;
    const r = image(s, center, { x: X0 + fw + 0.25, y: y0, w: cw, h: 3.05, align: "top" }, "scenario-shot", cap);
    text(s, cap, { x: X0 + fw + 0.25, y: r.y + r.h + 0.05, w: cw, h: 0.26, fontSize: 12, color: C.text2, align: "center", valign: "middle", objectName: "scenario-shot-caption" });
    legend();
  } else {
    const fw = 2.55, gap = (CW - 4 * fw) / 3;
    st.frames.forEach((f, i) => frame(f, i, X0 + i * (fw + gap), fw));
    legend();
  }
}

// ---------- 4. архитектура ----------
{
  const walls = GZ.runs.map((r) => r.fastsim.wall_s).filter((v) => v > 0);
  const fast = walls.length ? walls.reduce((a, b) => a + b, 0) / walls.length : null;     // секунд на прогон в быстром симуляторе
  const s = content(SEC_A, "Архитектура решения", "Агент и стенд общаются только через каналы из условия: судью можно заменить судьёй организаторов", [
    "Слева стенд: генератор сценариев, судья и симулятор — Gazebo для показа и быстрый для серий опытов. Справа агент: картина мира, планировщик, исполнитель и журнал гипотез.",
    "Между ними только каналы из условия задачи, поэтому нашего судью можно заменить судьёй организаторов, не трогая агента.",
  ]);
  // стенд
  const sx = X0, sw = 4.0, sy = TOP, sh = 3.2;
  card(s, sx, sy, sw, sh, "stand-card");
  text(s, "Стенд", { x: sx + 0.25, y: sy + 0.12, w: 2, h: 0.3, fontSize: 14, bold: true, color: C.accent5, objectName: "stand-label" });
  const boxes = [
    ["Генератор сценариев", "уровень и номер → расстановка", sy + 0.5, 0.72],
    ["Судья", "заряд, датчик, штрафы, очки", sy + 1.34, 0.82],
    ["Симулятор", "Gazebo — показ, быстрый — серии", sy + 2.28, 0.78],
  ];
  boxes.forEach(([head, sub, y, h], i) => {
    card(s, sx + 0.2, y, sw - 0.4, h, `stand-${i + 1}-box`, C.background1);
    text(s, head, { x: sx + 0.38, y: y + 0.07, w: sw - 0.76, h: 0.3, fontSize: 15, bold: true, objectName: `stand-${i + 1}-head` });
    text(s, sub, { x: sx + 0.38, y: y + 0.38, w: sw - 0.76, h: h - 0.42, fontSize: 13, color: C.text2, objectName: `stand-${i + 1}-sub` });
  });
  // агент
  const ax = 7.75, aw = X0 + CW - ax, ay = TOP, ah = 3.2;
  card(s, ax, ay, aw, ah, "agent-card", C.text1);
  text(s, "Агент", { x: ax + 0.25, y: ay + 0.12, w: 2, h: 0.3, fontSize: 14, bold: true, color: C.accent1, objectName: "agent-label" });
  const inner = [
    ["Картина мира", "образцы, грунт, опасные зоны, положение"],
    ["Планировщик", "правило или языковая модель"],
    ["Исполнитель", "путь по сетке и скорость колёс"],
    ["Журнал гипотез", "что предположил, как проверил, что вышло"],
  ];
  const bw = (aw - 0.6) / 2, bh = 1.22;
  inner.forEach(([head, sub], i) => {
    const x = ax + 0.2 + (i % 2) * (bw + 0.2), y = ay + 0.5 + Math.floor(i / 2) * (bh + 0.14);
    card(s, x, y, bw, bh, `agent-${i + 1}-box`, C.text2);
    text(s, head, { x: x + 0.18, y: y + 0.1, w: bw - 0.36, h: 0.32, fontSize: 15, bold: true, color: C.background1, objectName: `agent-${i + 1}-head` });
    text(s, sub, { x: x + 0.18, y: y + 0.45, w: bw - 0.36, h: bh - 0.5, fontSize: 13, color: C.background2, objectName: `agent-${i + 1}-sub` });
  });
  // каналы
  const lx1 = sx + sw + 0.1, lx2 = ax - 0.1, lw = lx2 - lx1;
  const chan = (label, y, dir, labelY) => {
    arrow(s, dir > 0 ? lx1 : lx2, y, dir > 0 ? lx2 : lx1, `chan-${label[0].replace(/\W/g, "")}-${dir > 0 ? "to" : "from"}`, C.accent1);
    text(s, label.map((t, i) => ({ text: t, options: { breakLine: i < label.length - 1 } })),
      { x: lx1, y: labelY, w: lw, h: 0.2 * label.length + 0.04, fontSize: 11, color: C.text1, align: "center", valign: "middle", fontFace: "Courier New", objectName: `chan-${label[0].replace(/\W/g, "")}-label` });
  };
  chan(["/did/battery  /did/score", "/did/sample_sensor  /did/events"], sy + 1.7, 1, sy + 1.22);
  chan(["/did/collect  /did/finish"], sy + 1.92, -1, sy + 1.96);
  chan(["/scan  /odom"], sy + 2.62, 1, sy + 2.36);
  chan(["/cmd_vel"], sy + 2.84, -1, sy + 2.88);
  // запись, серии, лаборатория
  const by = 5.45, bh2 = 1.2, bw2 = 3.55, bgap = (CW - 3 * bw2) / 2;
  const chain = [
    ["Запись прогона", "путь, заряд, решения и гипотезы агента"],
    ["Серии опытов", `${nSeries(EXP_IDS.length)} на одинаковых сценариях${fast ? `; ${ru(fast, 1)} с на прогон` : ""}`],
    ["Веб-лаборатория и пульт", "графики, проигрыватель прогонов, управление показом"],
  ];
  chain.forEach(([head, sub], i) => {
    const x = X0 + i * (bw2 + bgap);
    card(s, x, by, bw2, bh2, `chain-${i + 1}-box`);
    text(s, head, { x: x + 0.22, y: by + 0.12, w: bw2 - 0.44, h: 0.32, fontSize: 15, bold: true, objectName: `chain-${i + 1}-head` });
    text(s, sub, { x: x + 0.22, y: by + 0.48, w: bw2 - 0.44, h: 0.62, fontSize: 13, color: C.text2, objectName: `chain-${i + 1}-sub` });
    if (i < 2) arrow(s, x + bw2 + 0.1, by + bh2 / 2, x + bw2 + bgap - 0.1, `chain-${i + 1}-arrow`);
  });
  arrowDown(s, X0 + bw2 / 2, sy + sh + 0.06, by - 0.06, "record-arrow");
}

// ---------- 5. технологии ----------
{
  const tech = [
    ["ROS", "ROS 2 Jazzy", "каналы сообщений между программами: /scan, /odom, /cmd_vel, /did/*"],
    ["GZ", "Gazebo Sim Harmonic", "физика, колёса и лидар; мир и робот из официальных пакетов"],
    ["TB3", "TurtleBot3 Burger", "робот: два колеса и лидар на 360°, камеры нет"],
    ["pixi", "pixi и RoboStack", "окружение ставится одной командой, одинаково на Mac и Linux"],
    ["Py", "Python, NumPy, SciPy", "агент, судья, быстрый симулятор и статистика опытов"],
    ["ИИ", "Языковая модель", "выбирает подцели; ответ проверяется, при сбое решает запасное правило"],
    ["веб", "Веб-лаборатория", "графики опытов, проигрыватель прогонов и пульт в браузере"],
    ["тест", "pytest", `${TESTS ? nChecks(TESTS) : "автоматические проверки"}; запуск одной командой`],
  ];
  const s = content(SEC_A, "Используемые технологии и их назначение", "Мир и робот — из официальных пакетов; агент, судья и стенд для опытов — свои", [
    "Мир и робот — из официальных пакетов без изменений: ROS 2 Jazzy, Gazebo и TurtleBot3 Burger.",
    `Агент, судья и быстрый симулятор — свои, на Python; всё ставится одной командой и закрыто автоматическими проверками${TESTS ? ": их " + TESTS : ""}.`,
  ]);
  const cols = 4, gap = 0.25, w = (CW - (cols - 1) * gap) / cols, h = 2.35;
  tech.forEach(([badge, name, does], i) => {
    const x = X0 + (i % cols) * (w + gap), y = TOP + Math.floor(i / cols) * (h + 0.25);
    card(s, x, y, w, h, `tech-${i + 1}-card`);
    hexBadge(s, badge, x + 0.22, y + 0.22, 0.66, `tech-${i + 1}-badge`, i < 4 ? C.text2 : C.accent1, badge.length > 3 ? 12 : 14);
    text(s, name, { x: x + 0.22, y: y + 1.02, w: w - 0.44, h: 0.34, fontSize: 16, bold: true, objectName: `tech-${i + 1}-name` });
    text(s, does, { x: x + 0.22, y: y + 1.4, w: w - 0.44, h: 0.88, fontSize: 14, color: C.text2, objectName: `tech-${i + 1}-does` });
  });
}

// ---------- 6. карта и движение ----------
{
  const nav = D.nav, map = D.mapping;
  const med = `${ru(GZ.agent_median_cm[0])}–${ru(GZ.agent_median_cm[1])}`;
  const s = content(SEC_A, "Логика построения карты и движения", "Карта стен известна и перепроверяется лидаром; путь выбирается по цене клеток", [
    `Карта стен дана сеткой с клеткой ${ru(nav.grid_m * 100, 0)} сантиметров, и робот строит такую же сам по лидару: совпадение ${pct(map.agreement)} процентов.`,
    "Положение — одометрия, то есть счёт пути по колёсам, с поправкой по лидару: ошибка около двух сантиметров.",
    `Путь ищет алгоритм Дейкстры по цене клеток, ближе ${ru(nav.inflate_m * 100, 0)} сантиметров к стенам робот не едет; цену меняют карты вероятностей образцов и стоимости грунта.`,
  ]);
  const rows = [
    ["Карта", `готовая сетка арены с клеткой ${ru(nav.grid_m * 100, 0)} см и своя карта по лидару: совпадают на ${pct(map.agreement)}%`],
    ["Положение", `одометрия и поправка по лидару: скан совмещается с картой, ошибка ${med} см`],
    ["Путь", `алгоритм Дейкстры по цене клеток; ближе ${ru(nav.inflate_m * 100, 0)} см к стенам робот не едет`],
    ["Ведение", `руль на точку пути в ${ru(nav.lookahead_m * 100, 0)} см впереди; скорости колёсам — ${ru(1 / (GZ.tick_s || 0.1), 0)} раз в секунду`],
  ];
  rows.forEach(([head, body], i) => {
    const y = TOP + i * 0.95;
    hexBadge(s, i + 1, X0, y + 0.03, 0.46, `nav-${i + 1}-badge`, C.text2);
    text(s, head, { x: X0 + 0.75, y, w: 4.6, h: 0.32, fontSize: 16, bold: true, objectName: `nav-${i + 1}-head` });
    text(s, body, { x: X0 + 0.75, y: y + 0.34, w: 4.6, h: 0.55, fontSize: 14, color: C.text2, objectName: `nav-${i + 1}-body` });
  });
  card(s, X0, 5.72, 5.35, 1.08, "layers-card", C.text2);
  text(s, [
    { text: "Поверх карты стен: ", options: { bold: true, color: C.accent1 } },
    { text: "карта вероятностей образцов и карта стоимости грунта меняют цену клеток прямо в прогоне.", options: { color: C.background1 } },
  ], { x: X0 + 0.25, y: 5.72, w: 4.85, h: 1.08, fontSize: 14, valign: "middle", objectName: "layers-text" });

  const iw = 3.1, ix = [6.3, 9.63], ih = 2.97;
  [["Карта, построенная роботом", map.file, `по ${map.scans} сканам из записи прогона в Gazebo; увидено ${pct(map.coverage, 1)}% пола, оранжевое — путь робота`, "Карта занятости, построенная по сканам лидара, и путь робота"],
    ["Путь по сетке со стоимостями", nav.file, `оранжевое — путь на базу, ${ru(nav.path_m)} м; серое — запрет у стен; цветное — дорогой грунт`, "Путь на базу по сетке: запретная полоса у стен и зоны дорогого грунта"]]
    .forEach(([head, file, cap, alt], i) => {
      text(s, head, { x: ix[i], y: TOP, w: iw, h: 0.3, fontSize: 14, bold: true, objectName: `map-${i + 1}-head` });
      image(s, ASSET(file), { x: ix[i], y: TOP + 0.35, w: iw, h: ih }, `map-${i + 1}`, alt);
      text(s, cap, { x: ix[i], y: TOP + 0.4 + ih, w: iw, h: 0.75, fontSize: 12, color: C.accent5, objectName: `map-${i + 1}-caption` });
    });
}

// ---------- 7. реализовано / в работе ----------
pres.addSection({ title: SEC_B });
{
  const llmShare = LLM ? `${pct(LLM.first_ok_share)}% годных планов с первого раза` : "проверен на настоящей модели";
  const done = [
    ["Мир и робот в Gazebo из официальных пакетов", "Судья и генератор сценариев трёх уровней", "Каналы /did/* из условия задачи",
      "Быстрый симулятор с тем же судьёй", "Навигация по сетке со стоимостями", "Поправка положения по лидару"],
    ["Поиск образцов по карте вероятностей", "Оценка грунта по расходу, возврат по запасу заряда", "Журнал гипотез и расследования с выбором опыта",
      "Память между прогонами", `Планировщик на языковой модели: ${llmShare}`, "Веб-лаборатория и запуск одной командой"],
  ];
  const qwenItem = QWEN
    ? `Локальная Qwen: счёт ${QWEN.score_mean < QWEN.rule_score_mean ? "пока ниже правила" : "не ниже правила"}`
    : "Локальная модель Qwen";
  const wip = ["Пульт и показ полного сценария в Gazebo", "Роли модели «автор» и «критик» в расследованиях",
    qwenItem, "Конструктор исследования", "Построение карты через SLAM (бонус)"];
  const open = ["Адаптация к изменениям среды пока не прибавляет очков",
    MEMORY_HELPS ? "Обучение грунтам и память об опасных зонах вклада не показали" : "Обучение грунтам и память — об опасных зонах и между прогонами — вклада не показали",
    "Правила судьи — наши допущения"];
  const s = content(SEC_B, "Что уже реализовано и что в работе", "Основа работает и проверена опытами; показ в Gazebo и роли модели доделываются", [
    `Слева то, что работает и проверено: стенд, навигация, поиск образцов, журнал гипотез, расследования и планировщик на языковой модели — ${TESTS ? nChecks(TESTS) : "автоматические проверки"} и ${nRuns(TOTAL_RUNS)}.`,
    "В работе — пульт и показ в Gazebo, роли модели, локальная модель и конструктор исследования.",
    `И прямо о нерешённом: адаптация к изменениям среды пока не прибавляет очков${MEMORY_HELPS ? "" : ", память между прогонами результата не меняет"}, а правила судьи — наши допущения.`,
  ]);
  const lx = X0, lw = 7.05, lh = BOTTOM - TOP;
  card(s, lx, TOP, lw, lh, "done-card");
  hexBadge(s, done[0].length + done[1].length, lx + 0.25, TOP + 0.22, 0.5, "done-badge", C.accent3, 15);
  text(s, "Реализовано и проверено", { x: lx + 1.0, y: TOP + 0.22, w: 3.9, h: 0.5, fontSize: 18, bold: true, valign: "middle", objectName: "done-head" });
  text(s, [
    { text: TESTS ? String(TESTS) : "есть", options: { bold: true, color: C.text1 } },
    { text: ` ${pl(TESTS || 5, "проверка", "проверки", "проверок")}`, options: { breakLine: true } },
    { text: int(TOTAL_RUNS), options: { bold: true, color: C.text1 } },
    { text: ` ${pl(TOTAL_RUNS, "прогон", "прогона", "прогонов")}` },
  ], { x: lx + 5.0, y: TOP + 0.14, w: lw - 5.25, h: 0.66, fontSize: 14, color: C.text2, align: "right", valign: "middle", objectName: "done-stats" });
  done.forEach((items, i) => bullets(s, items, { x: lx + 0.25 + i * 3.35, y: TOP + 0.95, w: 3.2, h: 3.85, fontSize: 14, paraSpaceAfter: 7, objectName: `done-list-${i + 1}` }));
  const rx = lx + lw + 0.25, rw = X0 + CW - rx, h1 = 2.38, h2 = lh - h1 - 0.25;
  card(s, rx, TOP, rw, h1, "wip-card");
  hexBadge(s, wip.length, rx + 0.25, TOP + 0.2, 0.5, "wip-badge", C.accent1, 15);
  text(s, "В работе", { x: rx + 1.0, y: TOP + 0.2, w: 3, h: 0.5, fontSize: 18, bold: true, valign: "middle", objectName: "wip-head" });
  bullets(s, wip, { x: rx + 0.25, y: TOP + 0.8, w: rw - 0.5, h: h1 - 0.88, fontSize: 14, paraSpaceAfter: 2, objectName: "wip-list" });
  const y2 = TOP + h1 + 0.25;
  card(s, rx, y2, rw, h2, "open-card", C.text2);
  hexBadge(s, open.length, rx + 0.25, y2 + 0.18, 0.5, "open-badge", C.accent5, 15);
  text(s, "Не решено — говорим прямо", { x: rx + 1.0, y: y2 + 0.18, w: rw - 1.2, h: 0.5, fontSize: 18, bold: true, valign: "middle", color: C.background1, objectName: "open-head" });
  bullets(s, open, { x: rx + 0.25, y: y2 + 0.78, w: rw - 0.5, h: h2 - 0.86, fontSize: 13, paraSpaceAfter: 3, color: C.background1, objectName: "open-list" });
}

// ---------- 8. результаты: E1 ----------
{
  const d = E1.diff;
  const retLead = { same: "на базу оба возвращаются одинаково", notWorse: "и на базу возвращается не реже", worse: "но на базу возвращается реже" }[E1.retWord];
  const retNote = { same: "А на базу оба возвращаются одинаково — эта часть гипотезы не подтвердилась.",
    notWorse: "И на базу он возвращается не реже фиксированного.", worse: "Но на базу он возвращается реже — это цена более длинного поиска." }[E1.retWord];
  const retTile = { same: "возврат на базу у адаптивного и у фиксированного: разницы нет", notWorse: "возврат на базу у адаптивного и у фиксированного: адаптивный не реже",
    worse: "возврат на базу у адаптивного и у фиксированного: адаптивный реже" }[E1.retWord];
  const s = content(SEC_B, "Реальные результаты: серии опытов",
    E1.moreSamples ? `Адаптивный агент собирает больше образцов${E1.allLevels ? " на всех уровнях" : ""}; ${retLead}` : "Адаптивный агент против фиксированного плана на одинаковых сценариях", [
      "Главный опыт: два агента на одних и тех же сценариях, которых при отладке не видели.",
      `Адаптивный собирает ${d.mean >= 0 ? "больше" : "меньше"}, в среднем на ${pct(Math.abs(d.mean))} процентных пунктов: выиграл в ${d.a_higher} сценариях из ${d.n}, проиграл в ${d.b_higher}.`,
      retNote,
    ]);
  text(s, "Доля собранных образцов, %", { x: X0, y: TOP, w: 7.3, h: 0.3, fontSize: 14, bold: true, objectName: "e1-chart-title" });
  s.addChart(pres.charts.BAR, [
    { name: "Фиксированный план", labels: LEVELS, values: E1.fixed },
    { name: "С адаптацией", labels: LEVELS, values: E1.adaptive },
  ], {
    x: X0, y: TOP + 0.3, w: 7.3, h: 4.15, barDir: "col", barGrouping: "clustered", chartColors: [HEX.fixed, HEX.adaptive],
    barGapWidthPct: 110, barOverlapPct: -8, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFormatCode: "0",
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", ...labelText, ...axisText, ...quietFrame, ...legendText,
    objectName: "e1-chart", altText: "Доля собранных образцов по уровням: фиксированный план и агент с адаптацией",
  });
  text(s, `Опыт E1: ${nRuns(runsOf("E1"))} в быстром симуляторе, по ${E1.n} одинаковых сценариев на уровень; при отладке агент их не видел.`,
    { x: X0, y: 6.38, w: 7.3, h: 0.42, fontSize: 11, color: C.accent5, objectName: "e1-note" });
  const tx = 8.3, tw = X0 + CW - tx, th = 1.5, tg = (BOTTOM - TOP - 3 * th) / 2;
  statTile(s, tx, TOP, tw, th, `${d.mean >= 0 ? "+" : "−"}${pct(Math.abs(d.mean))} п. п.`, `собранных образцов в среднем; 95% интервал — от ${pct(d.ci[0])} до ${pct(d.ci[1])}`, "e1-tile-1");
  statTile(s, tx, TOP + th + tg, tw, th, `${d.a_higher} из ${d.n}`, `сценариев адаптивный агент выиграл; проиграл — ${d.b_higher}`, "e1-tile-2");
  statTile(s, tx, TOP + 2 * (th + tg), tw, th, `${pct(E1.retA)}% и ${pct(E1.retF)}%`, retTile, "e1-tile-3");
}

// ---------- 9. результаты: поиск и исследователь ----------
{
  const rp = E10.retPair;
  const retText = E10.retBetter
    ? `Исследователь возвращается чаще адаптивного на ${pct(rp.mean, 1)} п. п. (95% интервал — от ${pct(rp.ci[0], 1)} до ${pct(rp.ci[1], 1)})${E10.scorePair.status === "supported" ? "" : "; по счёту разницы нет"}.`
    : "Разница в возврате с адаптивным агентом пока в пределах погрешности.";
  const s = content(SEC_B, "Реальные результаты: поиск и исследователь",
    `${E9_BEST ? "Карта вероятностей находит больше образцов, чем простые правила" : "Стратегии поиска образцов"}; исследователь ${E10.retBetter ? "чаще возвращается на базу" : "верно называет причину сбоев"}`, [
      `Карта вероятностей находит ${ru(E9[0].share)} процента образцов${E9_BEST ? " — больше, чем спираль и подъём по сигналу из условия" : ""}.`,
      E10.inq ? `Агент-исследователь держит несколько объяснений странности и проверяет их опытом: из ${E10.inq.identified} выводов о причине ${pl(E10.inq.wrong, "неверен", "неверны", "неверны")} ${E10.inq.wrong}.`
        : `Агент-исследователь держит несколько объяснений странности и проверяет их опытом: причину он называет верно в ${pct(E10.correct, 1)} процента случаев.`,
      E10.retBetter ? `На трудном уровне он вернулся на базу в ${E10.home[0]} прогонах из ${E10.n}, адаптивный — в ${E10.home[1]}.`
        : "На базу он возвращается чаще адаптивного, но эта разница пока в пределах погрешности.",
    ]);
  const lw = 5.9;
  text(s, "Стратегии поиска: доля собранных образцов, %", { x: X0, y: TOP, w: lw, h: 0.3, fontSize: 14, bold: true, objectName: "e9-chart-title" });
  s.addChart(pres.charts.BAR, [{ name: "Доля собранных образцов, %", labels: E9.map((e) => e.label), values: E9.map((e) => e.share) }], {
    x: X0, y: TOP + 0.3, w: lw, h: 3.95, barDir: "bar", chartColors: E9.map((e) => (e.arm === "adaptive" ? HEX.adaptive : e.arm === "fixed" ? HEX.fixed : HEX.context)),
    barGapWidthPct: 70, catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 110, valAxisHidden: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", ...labelText, ...axisText, catAxisLabelFontSize: 13,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    objectName: "e9-chart", altText: "Доля собранных образцов у четырёх стратегий поиска",
  });
  text(s, `Опыт E9: уровни easy и medium, по ${E9[0].n} прогонов на стратегию. Спираль и подъём по сигналу — простые правила из условия.`,
    { x: X0, y: 6.2, w: lw, h: 0.6, fontSize: 11, color: C.accent5, objectName: "e9-note" });

  const rx = 6.95, rw = X0 + CW - rx;
  text(s, "Возврат на базу на трудном уровне, %", { x: rx, y: TOP, w: rw, h: 0.3, fontSize: 14, bold: true, objectName: "e10-chart-title" });
  s.addChart(pres.charts.BAR, [{ name: "Возврат на базу, %", labels: E10.labels, values: E10.ret }], {
    x: rx, y: TOP + 0.3, w: rw, h: 2.45, barDir: "bar", chartColors: [HEX.scientist, HEX.adaptive, HEX.fixed],
    barGapWidthPct: 70, catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 110, valAxisHidden: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", ...labelText, ...axisText, catAxisLabelFontSize: 13,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    objectName: "e10-chart", altText: "Возврат на базу на трудном уровне: исследователь, адаптивный агент, фиксированный план",
  });
  const ty = TOP + 2.9, th = 1.3;
  card(s, rx, ty, rw, th, "e10-tile-card");
  const q = E10.inq;
  text(s, q ? `${q.wrong} из ${q.identified}` : `${pct(E10.correct, 1)}%`, { x: rx + 0.25, y: ty, w: 2.3, h: th, fontSize: 34, bold: true, valign: "middle", objectName: "e10-tile-value" });
  text(s, q ? `выводов исследователя о причине неверны; ещё ${q.partial} верны частично. Найдено ${pct(q.faults_found / q.faults)}% настоящих сбоев`
    : "причин названо верно: исследователь держит несколько объяснений и проверяет их опытом",
    { x: rx + 2.6, y: ty, w: rw - 2.8, h: th, fontSize: 14, color: C.text2, valign: "middle", objectName: "e10-tile-label" });
  text(s, `Опыт E10: усложнённые правила — у одного симптома несколько причин; по ${E10.n} прогонов. ${retText}`,
    { x: rx, y: 6.2, w: rw, h: 0.6, fontSize: 11, color: C.accent5, objectName: "e10-note" });
}

// ---------- 10. результаты: Gazebo ----------
{
  const ser = GZ.series[SERIES_LEVEL];
  const med = `${ru(GZ.agent_median_cm[0])}–${ru(GZ.agent_median_cm[1])}`;
  const before = GZ.before_fix;
  const gzGap = Math.max(...GZ.runs.map((r) => Math.abs(r.gazebo.samples_collected - r.fastsim.samples_collected)));
  const same = GZ.runs.filter((r) => r.gazebo.samples_collected === r.fastsim.samples_collected).map((r) => r.level);
  const differ = GZ.runs.filter((r) => r.gazebo.samples_collected !== r.fastsim.samples_collected);
  const s = content(SEC_B, "Реальные результаты: Gazebo",
    gzGap === 0 ? "С поправкой по лидару прогоны в Gazebo повторяют быстрый симулятор"
      : gzGap === 1 ? "С поправкой по лидару прогоны в Gazebo повторяют быстрый симулятор с точностью до одного образца"
        : "Прогоны в Gazebo и в быстром симуляторе на одних и тех же сценариях", [
    "Серии считаются в быстром симуляторе, поэтому перенос проверили в Gazebo с настоящей физикой.",
    `Сначала одометрия уходила${before && before.odom_max_cm >= 100 ? " больше чем на метр" : " на десятки сантиметров"}: первые тридцать секунд робот покачивается, и манёвры её сбивают. Теперь агент ждёт и сверяет положение с лидаром: ошибка около двух сантиметров.`,
    gzGap === 0 ? "По числу собранных образцов все три уровня совпали с быстрым симулятором."
      : `Результат совпал с быстрым симулятором${gzGap === 1 ? " с точностью до одного образца" : " не полностью"}: на ${differ.map((r) => r.level).join(" и ")} собрано ${differ.map((r) => `${r.gazebo.samples_collected} против ${r.fastsim.samples_collected}`).join(", ")}.`,
  ]);
  const lw = 4.25;
  text(s, "Собрано образцов: один и тот же сценарий", { x: X0, y: TOP, w: lw, h: 0.3, fontSize: 14, bold: true, objectName: "gz-table-title" });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 14, color: HEX.ink, align: "center", valign: "middle", ...o } });
  const head = (t) => cell(t, { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 }, fontSize: 13 });
  const rows = [[head("Уровень"), head("Быстрый симулятор"), head("Gazebo")]].concat(GZ.runs.map((r) => [
    cell(r.level, { bold: true }),
    cell(`${r.fastsim.samples_collected} из ${r.fastsim.samples_total}`),
    cell(`${r.gazebo.samples_collected} из ${r.gazebo.samples_total}`, { bold: true }),
  ]));
  s.addTable(rows, { x: X0, y: TOP + 0.38, w: lw, colW: [1.05, 1.75, 1.45], rowH: 0.5, border: { type: "solid", pt: 0.75, color: HEX.grid }, fill: { color: "FFFFFF" }, objectName: "gz-table" });
  text(s, GZ_ALL_HOME ? "В Gazebo во всех трёх прогонах робот вернулся на базу, столкновений нет." : "Не во всех прогонах Gazebo робот вернулся на базу.",
    { x: X0, y: TOP + 2.48, w: lw, h: 0.5, fontSize: 12, color: C.accent5, objectName: "gz-table-note" });
  card(s, X0, 4.95, lw, BOTTOM - 4.95, "gz-cause-card", C.text2);
  text(s, [
    { text: "Причина найдена. ", options: { bold: true, color: C.accent1 } },
    { text: `Около 30 с после появления в мире робот покачивается, и манёвры в это время сбивали одометрию${before && before.odom_max_cm >= 100 ? " — больше чем на метр" : ""}. Теперь агент ждёт до ${ru(GZ.settle_until_s || 35, 0)}-й секунды.`, options: { color: C.background1 } },
  ], { x: X0 + 0.22, y: 4.95, w: lw - 0.44, h: BOTTOM - 4.95, fontSize: 14, valign: "middle", objectName: "gz-cause-text" });

  const cx = X0 + lw + 0.35, cw2 = 4.75;
  text(s, `Ошибка положения робота, см (уровень ${SERIES_LEVEL})`, { x: cx, y: TOP, w: cw2, h: 0.3, fontSize: 14, bold: true, objectName: "gz-chart-title" });
  s.addChart(pres.charts.LINE, [
    { name: "Только одометрия", labels: ser.t.map(String), values: ser.odom_cm },
    { name: "С поправкой по лидару", labels: ser.t.map(String), values: ser.agent_cm },
  ], {
    x: cx, y: TOP + 0.3, w: cw2, h: 3.95, chartColors: [HEX.context, HEX.adaptive], lineSize: 2.5, lineDataSymbol: "none",
    valAxisMinVal: 0, valAxisLabelFormatCode: "0", catAxisLabelFrequency: 3, catAxisMajorTickMark: "none", catAxisMinorTickMark: "none",
    showCatAxisTitle: true, catAxisTitle: "время от начала движения, с", catAxisTitleColor: HEX.muted, catAxisTitleFontSize: 12, catAxisTitleFontFace: CHART_FONT,
    ...axisText, catAxisLabelFontSize: 12, ...quietFrame, ...legendText, legendFontSize: 13,
    objectName: "gz-chart", altText: "Ошибка положения во времени: одометрия уходит на десятки сантиметров, с поправкой по лидару остаётся около двух",
  });
  text(s, `Серия E8, ${nRuns(runsOf("E8"))}: при таком уходе одометрии возврат на базу — ${pct(E8.lidar)}% с поправкой и ${pct(E8.odom)}% без неё.`,
    { x: cx, y: 6.2, w: cw2, h: 0.6, fontSize: 11, color: C.accent5, objectName: "gz-note" });

  const tx = cx + cw2 + 0.3, tw = X0 + CW - tx, th = 1.52, tg = (BOTTOM - TOP - 3 * th) / 2;
  const tile = (i, v, label) => {
    const y = TOP + i * (th + tg);
    card(s, tx, y, tw, th, `gz-tile-${i + 1}-card`);
    text(s, v, { x: tx + 0.18, y: y + 0.12, w: tw - 0.36, h: 0.55, fontSize: 24, bold: true, valign: "middle", objectName: `gz-tile-${i + 1}-value` });
    text(s, label, { x: tx + 0.18, y: y + 0.7, w: tw - 0.36, h: th - 0.78, fontSize: 13, color: C.text2, objectName: `gz-tile-${i + 1}-label` });
  };
  tile(0, `${med} см`, "ошибка положения с поправкой, медиана");
  tile(1, `${ru(GZ.agent_max_cm)} см`, "наибольшая ошибка за три прогона");
  tile(2, `${ru(GZ.odom_max_cm, 0)} см`, "уходила одометрия в тех же прогонах");
}

// ---------- 11. как использовали ИИ ----------
pres.addSection({ title: SEC_C });
{
  const s = content(SEC_C, "Как использовали ИИ при разработке", "Ассистент вёл разработку, вторая модель давала второе мнение, третья планирует внутри робота", [
    "Стенд написан за полтора дня в диалоге с ИИ-ассистентом: он вёл работу как ведущий инженер и раздавал параллельные ветки вспомогательным агентам.",
    `Вторая модель давала второе мнение и нашла ${SECOND_OPINION_FIXES} слабых мест в расследованиях, все исправлены.${LLM ? ` Третья планирует внутри робота: ${pct(LLM.first_ok_share)} процентов годных планов с первого раза.` : ""}`,
    "И главное: ошибки находил стенд, а не глаз — четыре из них на слайде.",
  ]);
  const cards = [
    ["Ассистент — ведущий инженер", "Стенд написан за полтора дня в диалоге с Claude Code в среде T3 Code. Работа делилась на параллельные ветки, около десяти, и раздавалась вспомогательным агентам."],
    ["Второе мнение", `По сложным вопросам запрашивали второе мнение у другой модели — GPT-6 Astra. В расследованиях она нашла ${SECOND_OPINION_FIXES} слабых мест, все исправлены.`],
    ["ИИ внутри робота", LLM ? `Модель выбирает подцели. GPT-6 Luna: ${pct(LLM.first_ok_share)}% годных планов с первого раза, счёт ${ru(LLM.score_mean, 0)} против ${ru(LLM.rule_score_mean, 0)} у правила.${QWEN ? ` Локальная Qwen: ${pct(QWEN.first_ok_share)}% годных, но счёт ${ru(QWEN.score_mean, 0)}.` : ""}` : "Языковая модель выбирает подцели верхнего уровня; ответы проверяются."],
  ];
  const w = (CW - 2 * 0.25) / 3, h = 2.4;
  cards.forEach(([head, body], i) => {
    const x = X0 + i * (w + 0.25);
    card(s, x, TOP, w, h, `ai-${i + 1}-card`);
    hexBadge(s, i + 1, x + 0.22, TOP + 0.2, 0.46, `ai-${i + 1}-badge`, C.accent1);
    text(s, head, { x: x + 0.95, y: TOP + 0.2, w: w - 1.15, h: 0.46, fontSize: 16, bold: true, valign: "middle", objectName: `ai-${i + 1}-head` });
    text(s, body, { x: x + 0.22, y: TOP + 0.82, w: w - 0.44, h: h - 0.92, fontSize: 14, color: C.text2, objectName: `ai-${i + 1}-body` });
  });
  const y2 = TOP + h + 0.22, h2 = 1.82;
  card(s, X0, y2, CW, h2, "found-card", C.text2);
  text(s, "Ошибки, которые нашёл стенд, а не глаз", { x: X0 + 0.3, y: y2 + 0.14, w: CW - 0.6, h: 0.36, fontSize: 16, bold: true, color: C.accent1, objectName: "found-head" });
  const found = [
    ["ложные следы в карте вероятностей после сбора образца", "агент бросал работу с полной батареей: цена маршрута была смешана с зарядом на дорогу"],
    ["робот покачивается после появления в Gazebo — одометрия уходила на метр", "сотня ложных сборов подряд при неверной поправке датчика"],
  ];
  found.forEach((items, i) => bullets(s, items, { x: X0 + 0.3 + i * (CW / 2 - 0.1), y: y2 + 0.58, w: CW / 2 - 0.5, h: h2 - 0.66, fontSize: 14, color: C.background1, paraSpaceAfter: 4, objectName: `found-list-${i + 1}` }));
  text(s, [
    { text: "Качество процесса: ", options: { bold: true } },
    { text: `${TESTS ? nChecks(TESTS) : "автоматические проверки"}, воспроизводимые серии прогонов, история в git. ` },
    { text: "Навыки ассистента: ", options: { bold: true } },
    { text: "построение графиков, сборка презентаций, запуск других моделей." },
  ], { x: X0, y: y2 + h2 + 0.08, w: CW, h: BOTTOM - (y2 + h2 + 0.08), fontSize: 13, color: C.text2, valign: "middle", objectName: "process-line" });
}

// ---------- текст доклада ----------
function writeSpeech() {
  const WPS = 2.2;                                   // слов в секунду при спокойной речи
  const words = (t) => t.split(/\s+/).filter(Boolean).length;
  const spoken = (p) => p.notes.filter((n) => !n.startsWith("ЗДЕСЬ — ЖИВОЙ ПОКАЗ"));
  const secs = SPEECH_PARTS.map((p) => Math.round(words(spoken(p).join(" ")) / WPS / 5) * 5);
  const total = secs.reduce((a, b) => a + b, 0);
  const mmss = (t) => `${Math.floor(t / 60)} мин ${String(t % 60).padStart(2, "0")} с`;
  const e3 = (c) => ru(pair("E3", "score", "adaptive", "fixed", "hard", c).mean, 0);
  const e2 = (b) => pair("E2", "score", "adaptive", b, "hard");
  const lines = [
    "# Текст доклада ко второму чекпоинту",
    "",
    `Слайды целиком — около ${mmss(total)} спокойной речи (время у каждого слайда посчитано по числу слов). Живой показ — отдельно, после слайда 3.`,
    "Числа в тексте подставлены из тех же файлов, что и на слайдах. Текст пересобирается вместе с презентацией",
    "(`node build_checkpoint2.js`) и лежит в заметках к слайдам; править его нужно в скрипте, а не здесь.",
    "",
  ];
  SPEECH_PARTS.forEach((p, i) => {
    lines.push(`## Слайд ${i + 1}. ${p.title} (≈ ${secs[i]} с)`, "");
    for (const n of p.notes) lines.push(n.startsWith("ЗДЕСЬ — ЖИВОЙ ПОКАЗ") ? `\n**▶ ${n}**` : n);
    lines.push("");
  });
  lines.push("## Если на всё пять минут вместе с показом", "",
    "Полный сценарий показа из `docs/demo_script.md` сам занимает около пяти минут, поэтому нужно выбрать заранее:",
    "",
    "- **Показ главный.** Слайды 2, 3, 7, 8 и 10 — по одной-две фразы (около полутора минут), затем пульт: карта из одной точки, маршрут, «Ехать», запуск миссии. Слайды 4, 5, 6, 9 и 11 оставить для ответов на вопросы.",
    "- **Слайды главные.** Весь текст выше, а показ — одна минута: маршрут из трёх точек и «Ехать». Итог миссии есть на слайдах 1 и 3.",
    "- **Показ не запустился.** Рассказывать по слайду 3: кадры и снимок проигрывателя — запись настоящего прогона в Gazebo. Запасной вариант без Gazebo описан в `docs/demo_script.md`.",
    "",
    "## К вопросам жюри", "",
    `- **Почему адаптация к изменениям среды не видна в счёте?** Преимущество адаптивного агента на трудном уровне примерно одинаково без событий и с ними: ${e3("none")} очков без событий, ${e3("soil")} при смене грунтов, ${e3("hazard")} при новой опасной зоне, ${e3("sensor")} при сбое датчика (опыт E3). Выигрыш даёт поиск образцов, а не реакция на изменения — это мы говорим прямо.`,
    `- **Какие механизмы реально помогают?** Контроль датчика (+${ru(e2("no_sensor_health").mean)} очка) и расчёт запаса на возврат (+${ru(e2("static_reserve").mean)}). Обучение грунтам и память об опасных зонах вклада не показали (опыт E2).`,
    "- **Чьи правила подсчёта?** Наши: расход заряда, формула датчика, штрафы и очки в условии не заданы. Агент видит стенд только через каналы из условия, поэтому судью можно заменить судьёй организаторов; цифры тогда нужно пересчитать.",
    LLM ? `- **Настоящая языковая модель проверена?** Да: GPT-6 Luna, ${LLM.requests} запросов на ${LLM.scenarios} сценариях, ${pct(LLM.first_ok_share)}% годных планов с первого раза, счёт ${ru(LLM.score_mean)} против ${ru(LLM.rule_score_mean)} у правила; ответ в среднем ${ru(LLM.latency_ms.median / 1000, 0)} с (медиана), робот в это время стоит.${QWEN ? ` Локальная Qwen 2.5 отвечает за ${ru(QWEN.latency_ms.median / 1000, 0)} с и даёт ${pct(QWEN.first_ok_share)}% годных планов, но счёт ${ru(QWEN.score_mean)}: слишком рано едет домой.` : ""}` : "- **Настоящая языковая модель проверена?** Сводки прогона нет в runs/llm_real.",
    `- **Насколько можно верить быстрому симулятору?** На трёх сверочных сценариях Gazebo собрал ${GZ.runs.map((r) => `${r.gazebo.samples_collected} из ${r.gazebo.samples_total}`).join(", ")}, быстрый симулятор — ${GZ.runs.map((r) => `${r.fastsim.samples_collected} из ${r.fastsim.samples_total}`).join(", ")}. Это три прогона, а не серия.`,
    "");
  fs.writeFileSync(SPEECH, lines.join("\n"));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  writeSpeech();
  console.log("готово:", OUT);
  console.log("доклад:", SPEECH);
  console.log(`снимков показа: ${SHOTS.length}${DEMO_SHOTS.length ? " (на слайде: " + DEMO_SHOTS.map((f) => path.basename(f)).join(", ") + ")" : ""}; сценарий показа: ${DEMO ? "docs/demo_script.md, шагов " + DEMO_STEP_NAMES.length : "файла нет"}; серий: ${EXP_IDS.length}, прогонов: ${TOTAL_RUNS}, проверок: ${TESTS}`);
  for (const w of WARN) console.warn("ВНИМАНИЕ: " + w);
  if (!ARGS.has("--no-pdf")) {
    const render = path.join(__dirname, "render_checkpoint2.js");
    if (fs.existsSync(render)) spawnSync(process.execPath, [render], { stdio: "inherit" });
  }
})();
