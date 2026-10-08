// Расширенная колода (15 слайдов, задел на финальный питч): node presentation/build_extended.js
// Числа берутся из data/results.json (его пишет figures/make_figures.py), рисунки — из assets/.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

// apply_theme.js лежит вне проекта и ищет jszip через NODE_PATH.
process.env.NODE_PATH = [path.join(__dirname, "node_modules"), process.env.NODE_PATH].filter(Boolean).join(path.delimiter);
require("module").Module._initPaths();

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/Users/a/.claude/skills/synced/0ec766ba-a970-423d-9985-0a1698a18ad8_57a5e5ff-4dfe-4d5a-9b46-49c5ea27665a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const R = JSON.parse(fs.readFileSync(path.join(__dirname, "data/results.json"), "utf8"));
const ASSET = (name) => path.join(__dirname, "assets", name);
const OUT = path.join(__dirname, "extended.pptx");

// Замеры уровня 0: четыре прогона tools/smoke_sim.py на VM (2 vCPU, без GPU, без окна Gazebo).
const L0 = { rtf: "0,86–0,96", scanHz: "5 Гц", rays: "327 из 360" };

const ru = (v, d = 1) => Number(v).toFixed(d).replace(".", ",").replace("-", "−");
const pct = (ours, base) => `−${Math.round((1 - ours / base) * 100)}%`;

const THEME = {
  name: "Izyskatel",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "14181C", lt1: "FFFFFF", dk2: "2B3138", lt2: "EEF0F2",
    accent1: "EB6834", accent2: "2A78D6", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "5B6168", accent6: "D03B3B", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
// Только hex: сетка графиков, цвета столбцов по точкам, статусные точки.
const HEX = { grid: "E1E0D9", context: "B4B8BD", ink: "14181C", muted: "5B6168", blue: "2A78D6", border: "D5D9DD" };
const EVIDENCE = {
  sim: ["данные симуляции", "0CA30C"],
  calc: ["расчёт на реальной карте", "2A78D6"],
  model: ["офлайн-модель", "FAB219"],
  design: ["проектное решение", "898781"],
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "«Изыскатель» — чекпоинт 1";
pres.subject = "DID Hack 2026: автономный ИИ-исследователь на TurtleBot3";
pres.author = "Команда «Изыскатель»";
const C = pres.SchemeColor;
const FOOTER = "«Изыскатель» · DID Hack 2026 · чекпоинт 1";
const X0 = 0.6, CW = 12.13; // левое поле и ширина рабочей области

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.2, w: 6.5, h: 1.25, fontSize: 60, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.6, w: 6.3, h: 1.1, fontSize: 24, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.45, w: CW, h: 0.8, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: X0, y: 7.0, w: 6.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "STATEMENT_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.75, w: 11.9, h: 1.5, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: 0.7, y: 7.0, w: 6.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "middle" } } },
  ],
});

// ---------- помощники ----------

function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, valign: "top", align: "left", color: C.text1, fontSize: 14, ...o });
}

function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, objectName: name });
}

function hexBadge(slide, label, x, y, size, name, fill = C.accent1) {
  slide.addText(String(label), {
    shape: pres.shapes.HEXAGON, x, y, w: size * 1.14, h: size, fill: { color: fill }, color: C.background1,
    bold: true, fontSize: Math.round(size * 28), align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name,
  });
}

function statTile(slide, x, y, w, h, value, label, name, valueSize = 34) {
  card(slide, x, y, w, h, `${name}-card`);
  text(slide, value, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.62, fontSize: valueSize, bold: true, valign: "middle", objectName: `${name}-value` });
  text(slide, label, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: h - 1.05, fontSize: 14, color: C.text2, objectName: `${name}-label` });
}

function evidence(slide, kind) {
  const [label, color] = EVIDENCE[kind];
  slide.addShape(pres.shapes.OVAL, { x: 9.86, y: 7.08, w: 0.14, h: 0.14, fill: { color }, objectName: "evidence-dot" });
  text(slide, label, { x: 10.08, y: 7.0, w: 2.1, h: 0.3, fontSize: 10, color: C.accent5, valign: "middle", objectName: "evidence-label" });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function image(slide, file, x, y, size, name, altText) {
  const px = pngSize(ASSET(file));
  const ar = px.w / px.h;
  const w = size.w || size.h * ar;
  const h = size.h || size.w / ar;
  slide.addImage({ path: ASSET(file), x, y, w, h, objectName: name, altText });
  return { w, h };
}

function content(section, title, kind, notes) {
  const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  slide.addText(title, { placeholder: "title" });
  if (kind) evidence(slide, kind);
  slide.addNotes(notes);
  return slide;
}

const CHART_TEXT = { color: HEX.muted, face: "+mn-lt" };

// ---------- 1. титул ----------

const S_INTRO = "Задача и замысел", S_STATE = "Состояние и решения", S_SCIENCE = "Наука и адаптация", S_PLAN = "План", S_APP = "Приложение";
pres.addSection({ title: S_INTRO });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: S_INTRO });
  s.addText("«Изыскатель»", { placeholder: "title" });
  s.addText("Автономный ИИ-исследователь на TurtleBot3", { placeholder: "body" });
  text(s, "DID Hack 2026 · чекпоинт 1", { x: 0.7, y: 5.05, w: 6.0, h: 0.4, fontSize: 16, color: C.accent1, bold: true, objectName: "event" });
  image(s, "arena_dark.png", 7.35, 0.95, { h: 5.6 }, "title-arena", "Арена turtlebot3_world и точки реального скана лидара");
  text(s, "Оранжевые точки — реальный скан лидара из симуляции", { x: 7.35, y: 6.65, w: 5.4, h: 0.3, fontSize: 11, color: C.background2, align: "center", objectName: "title-caption" });
  s.addNotes("Мы — команда «Изыскатель». Делаем агента, который сам ставит эксперименты на роботе в Gazebo. На картинке — настоящая арена и настоящий скан лидара из нашей симуляции.");
}

// ---------- 2. задача ----------
{
  const s = content(S_INTRO, "Задача: собрать образцы и вернуться на базу", null,
    "Робот не знает, где образцы и где дорогие грунты. Датчик даёт одно шумное число без направления, батарея ограничена. В сложном сценарии среда меняется без предупреждения. Требуется замкнутый научный цикл: гипотеза, действие, данные, вывод.");
  const img = image(s, "arena.png", X0, 1.45, { h: 4.55 }, "arena", "Карта арены: шестиугольник с девятью столбами, старт в точке минус 2.0, минус 0.5");
  const tx = X0 + img.w + 0.35, tw = (X0 + CW - tx - 0.2) / 2;
  const tiles = [
    ["3 · 5 · 7", "скрытых образцов в easy, medium и hard. Сбор — ближе 0,30 м"],
    ["0…1", "скалярный шумный датчик близости. Направления не даёт"],
    ["60", "единиц батареи. «Грунты» расходуют её быстрее"],
    ["hard", "среда меняется без объявления: грунты, опасная зона, сбой датчика"],
  ];
  tiles.forEach(([v, l], i) => statTile(s, tx + (i % 2) * (tw + 0.2), 1.45 + Math.floor(i / 2) * 2.35, tw, 2.15, v, l, `fact-${i + 1}`, 36));
  const steps = ["гипотеза", "действие", "данные", "вывод", "новая гипотеза"];
  const sw = 2.0, gap = (CW - steps.length * sw) / (steps.length - 1);
  steps.forEach((label, i) => {
    const x = X0 + i * (sw + gap);
    s.addText(label, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y: 6.22, w: sw, h: 0.5, fill: { color: i === 4 ? C.accent1 : C.text2 }, color: C.background1, bold: true, fontSize: 14, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `loop-${i + 1}` });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + sw + 0.08, y: 6.47, w: gap - 0.16, h: 0, line: { color: C.accent5, width: 1.5, endArrowType: "triangle" }, objectName: `loop-arrow-${i + 1}` });
  });
}

// ---------- 3. замысел ----------
{
  const s = content(S_INTRO, "Замысел: агент-экспериментатор, а не сборщик", "design",
    "Главная идея. Агент хранит убеждения о мире в виде карт с неопределённостью. Каждый ход — эксперимент, выбранный по ожидаемой пользе на единицу батареи. Каждый вывод записывается в журнал с числом. Мы делаем не минимальную версию: целимся во все уровни и бонусные треки.");
  const cards = [
    ["Убеждения вместо догадок", "Карта вероятностей образцов и карта стоимости грунтов с неопределённостью. Каждое измерение их обновляет."],
    ["Действие — это эксперимент", "Следующий ход выбирается по ожидаемой пользе на единицу батареи: узнать новое, собрать или вернуться."],
    ["Вывод — запись в журнал", "Гипотеза, прогноз, наблюдение, вердикт с числом. LLM формулирует и объясняет, статистика проверяет."],
  ];
  const w = 3.77, gap = (CW - 3 * w) / 2;
  cards.forEach(([head, body], i) => {
    const x = X0 + i * (w + gap);
    card(s, x, 1.5, w, 3.75, `idea-${i + 1}-card`);
    hexBadge(s, i + 1, x + 0.3, 1.8, 0.62, `idea-${i + 1}-badge`);
    text(s, head, { x: x + 0.3, y: 2.65, w: w - 0.6, h: 0.75, fontSize: 20, bold: true, objectName: `idea-${i + 1}-head` });
    text(s, body, { x: x + 0.3, y: 3.5, w: w - 0.6, h: 1.65, fontSize: 16, color: C.text2, objectName: `idea-${i + 1}-body` });
  });
  card(s, X0, 5.55, CW, 1.15, "ambition-card", C.text2);
  text(s, [
    { text: "Не MVP. ", options: { bold: true, color: C.accent1 } },
    { text: "Целимся во все пять уровней и бонусные треки: два робота, SLAM вместо готовой карты, журнал гипотез с вердиктами, навыки ассистента.", options: { color: C.background1 } },
  ], { x: X0 + 0.35, y: 5.55, w: CW - 0.7, h: 1.15, fontSize: 17, valign: "middle", objectName: "ambition-text" });
}

// ---------- 4. архитектура ----------
pres.addSection({ title: S_STATE });
{
  const s = content(S_STATE, "Архитектура: три контура с разной частотой", "design",
    "Слева среда: Gazebo с роботом и наш судья, который считает батарею, датчик образцов, события и счёт. Справа агент из трёх контуров. LLM работает редко и только выбирает подцели. Исполнитель держит модель мира. Навигация крутится на десяти герцах. Граница между средой и агентом — только топики из условия.");
  const lw = 3.55;
  card(s, X0, 1.5, lw, 2.0, "env-sim-card");
  text(s, "Gazebo Sim и TurtleBot3", { x: X0 + 0.25, y: 1.68, w: lw - 0.5, h: 0.4, fontSize: 16, bold: true, objectName: "env-sim-head" });
  text(s, "Мир turtlebot3_world без изменений: лидар, одометрия, IMU", { x: X0 + 0.25, y: 2.15, w: lw - 0.5, h: 0.75, fontSize: 14, color: C.text2, objectName: "env-sim-body" });
  text(s, "ROS 2 Jazzy · Gazebo Sim 8.10", { x: X0 + 0.25, y: 3.02, w: lw - 0.5, h: 0.3, fontSize: 12, color: C.accent5, objectName: "env-sim-versions" });
  card(s, X0, 3.7, lw, 2.0, "env-judge-card");
  text(s, "Судья did_judge", { x: X0 + 0.25, y: 3.88, w: lw - 0.5, h: 0.4, fontSize: 16, bold: true, objectName: "env-judge-head" });
  text(s, "Сценарий по seed, батарея, датчик образцов, события, счёт", { x: X0 + 0.25, y: 4.35, w: lw - 0.5, h: 0.75, fontSize: 14, color: C.text2, objectName: "env-judge-body" });
  text(s, "Та же логика — в стенде без Gazebo", { x: X0 + 0.25, y: 5.22, w: lw - 0.5, h: 0.3, fontSize: 12, color: C.accent5, objectName: "env-judge-twin" });
  text(s, "Граница — только топики из условия: судью организаторов можно подставить без правок агента.", { x: X0, y: 5.88, w: lw, h: 0.85, fontSize: 14, objectName: "env-note" });

  const ax = X0 + lw + 0.12, aw = 1.5;
  s.addShape(pres.shapes.LINE, { x: ax, y: 2.5, w: aw, h: 0, line: { color: C.accent5, width: 2, endArrowType: "triangle" }, objectName: "arrow-in" });
  text(s, "/scan  /odom\n/did/battery\n/did/sample_sensor", { x: ax, y: 1.62, w: aw, h: 0.8, fontSize: 11, color: C.accent5, align: "center", valign: "bottom", objectName: "arrow-in-label" });
  s.addShape(pres.shapes.LINE, { x: ax, y: 4.7, w: aw, h: 0, line: { color: C.accent5, width: 2, beginArrowType: "triangle" }, objectName: "arrow-out" });
  text(s, "/cmd_vel\n/did/collect\n/did/finish", { x: ax, y: 4.8, w: aw, h: 0.8, fontSize: 11, color: C.accent5, align: "center", objectName: "arrow-out-label" });

  const gx = ax + aw + 0.12, gw = X0 + CW - gx;
  card(s, gx, 1.5, gw, 5.2, "agent-card");
  text(s, "Агент did_agent", { x: gx + 0.25, y: 1.62, w: gw - 0.5, h: 0.4, fontSize: 16, bold: true, objectName: "agent-head" });
  const layers = [
    ["по событиям", "Стратег на LLM", "Выбирает подцели из кандидатов, формулирует гипотезы, ведёт журнал"],
    ["1 Гц", "Исполнитель и модель мира", "Карты убеждений, кандидаты-эксперименты, резерв батареи на возврат"],
    ["10 Гц", "Навигация и управление", "A* по карте стоимостей, путевые точки, /cmd_vel, стоп по лидару"],
  ];
  layers.forEach(([rate, head, body], i) => {
    const y = 2.15 + i * 1.5;
    card(s, gx + 0.25, y, gw - 0.5, 1.35, `layer-${i + 1}-card`, C.background1);
    s.addText(rate, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: gx + 0.45, y: y + 0.45, w: 1.4, h: 0.45, fill: { color: i === 0 ? C.accent1 : C.text2 }, color: C.background1, bold: true, fontSize: 12, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `layer-${i + 1}-rate` });
    text(s, head, { x: gx + 2.05, y: y + 0.15, w: gw - 2.5, h: 0.4, fontSize: 16, bold: true, objectName: `layer-${i + 1}-head` });
    text(s, body, { x: gx + 2.05, y: y + 0.58, w: gw - 2.5, h: 0.7, fontSize: 14, color: C.text2, objectName: `layer-${i + 1}-body` });
  });
}

// ---------- 5. статус ----------
{
  const L = R.lidar;
  const s = content(S_STATE, "Статус: среда работает и проверена измерением", "sim",
    `Что уже работает. Симуляция поднимается одной командой и проверяется автоматическим тестом. На картинке три реальных скана лидара, наложенные на карту: ${L.points} точек, девяносто пять процентов лежат не дальше пяти сантиметров от стен карты. Попутно сняли противоречие условия про стартовую позу. Судья, контроллер и подключение LLM сейчас в работе.`);
  const img = image(s, "lidar_on_map.png", X0, 1.45, { h: 4.9 }, "lidar", "Три скана лидара из симуляции, наложенные на карту арены");
  text(s, `Три реальных скана из Gazebo поверх карты: ${L.points.toLocaleString("ru-RU")} точки, позы по x: ${L.poses.map((p) => ru(p[0], 2)).join("; ")} м`, { x: X0, y: 6.42, w: img.w, h: 0.3, fontSize: 12, color: C.accent5, objectName: "lidar-caption" });
  const tx = X0 + img.w + 0.4, tw = (X0 + CW - tx - 0.2) / 2;
  const tiles = [
    [L0.rtf, "real-time factor без окна Gazebo: 2 vCPU, без GPU"],
    [L0.scanHz, `частота лидара; ${L0.rays} лучей валидны`],
    [`≤ ${Math.round(L.p95_cm)} см`, "расхождение лидара с картой у 95% точек"],
    ["1 команда", "pixi run sim; версии закреплены lock-файлом"],
  ];
  tiles.forEach(([v, l], i) => statTile(s, tx + (i % 2) * (tw + 0.2), 1.45 + Math.floor(i / 2) * 1.95, tw, 1.75, v, l, `status-${i + 1}`, 30));
  text(s, [
    { text: "Сняли противоречие условия. ", options: { bold: true } },
    { text: "На старте одометрия равна (0; 0), значит мир = (−2,0; −0,5) + одометрия." },
  ], { x: tx, y: 5.42, w: X0 + CW - tx, h: 0.62, fontSize: 14, objectName: "status-finding" });
  text(s, [
    { text: "В работе: ", options: { bold: true } },
    { text: "судья did_judge, контроллер пути, подключение LLM." },
  ], { x: tx, y: 6.12, w: X0 + CW - tx, h: 0.6, fontSize: 14, objectName: "status-wip" });
}

// ---------- 6. навигация ----------
{
  const A = R.astar;
  const s = content(S_STATE, "Навигация: дешёвый путь важнее короткого", "calc",
    `Уровень один. Планировщик A-звезда работает по сетке реальной карты. Запас вокруг стен запрещён, у каждой клетки есть стоимость. На примере: путь в обход дорогого грунта длиннее, но тратит ${ru(A.smart_energy)} единицы батареи вместо ${ru(A.naive_energy)}. Полный пересчёт занимает около ${Math.round(A.plan_ms_median)} миллисекунд, поэтому перепланируем при каждом изменении карты. Это офлайн-прототип, контроллер в работе.`);
  const img = image(s, "astar.png", X0, 1.45, { h: 5.3 }, "astar", "Два маршрута на карте: кратчайший через зону грунта и более длинный в обход");
  const tx = X0 + img.w + 0.4, tw = (X0 + CW - tx - 0.2) / 2;
  statTile(s, tx, 1.45, tw, 1.75, pct(A.smart_energy, A.naive_energy), `батареи на этом маршруте: ${ru(A.smart_energy)} ед. вместо ${ru(A.naive_energy)}`, "nav-saving");
  statTile(s, tx + tw + 0.2, 1.45, tw, 1.75, `${Math.round(A.plan_ms_median)} мс`, `полный пересчёт пути: ${A.free_cells.toLocaleString("ru-RU")} клеток арены`, "nav-speed");
  const rows = [
    ["Слои карты", `запрет у стен ${ru(A.inflate_m, 2)} м, стоимость грунта, опасные зоны`],
    [`Путь → ${A.waypoints} путевых точек`, "прямой отрезок берём, если он не дороже пути по клеткам"],
    ["Контроллер 10 Гц", "TwistStamped в /cmd_vel, стоп по лидару, перепланирование"],
  ];
  rows.forEach(([head, body], i) => {
    const y = 3.5 + i * 0.88;
    hexBadge(s, i + 1, tx, y + 0.05, 0.42, `nav-row-${i + 1}-badge`, C.text2);
    text(s, head, { x: tx + 0.65, y, w: X0 + CW - tx - 0.65, h: 0.32, fontSize: 15, bold: true, objectName: `nav-row-${i + 1}-head` });
    text(s, body, { x: tx + 0.65, y: y + 0.34, w: X0 + CW - tx - 0.65, h: 0.5, fontSize: 14, color: C.text2, objectName: `nav-row-${i + 1}-body` });
  });
  text(s, [
    { text: "Статус: ", options: { bold: true } },
    { text: "планировщик — офлайн-прототип, контроллер в работе." },
  ], { x: tx, y: 6.2, w: X0 + CW - tx, h: 0.55, fontSize: 14, objectName: "nav-status" });
}

// ---------- 7. LLM ----------
{
  const s = content(S_STATE, "LLM выбирает из проверенных кандидатов", "design",
    "Уровень два. Модель не выдумывает координаты и не управляет скоростью. Код готовит пять-восемь исполнимых подцелей с числами. Модель выбирает и упорядочивает их, объясняет выбор и формулирует гипотезу. Ответ проверяется по схеме. При ошибке — один повтор с текстом ошибки, затем запасной планировщик без LLM. Резерв батареи на возврат считает код.");
  const steps = [
    ["Состояние в текст", "Поза, батарея, резерв на возврат, оценки грунтов, образцы, последние события"],
    ["Кандидаты от кода", "5–8 исполнимых подцелей с числами: польза, цена в батарее, риск"],
    ["Выбор модели", "Qwen или DeepSeek упорядочивает подцели и формулирует гипотезу. Ответ — JSON"],
    ["Проверка и запуск", "Схема, достижимость, резерв. Ошибка: повтор, затем планировщик без LLM"],
  ];
  const w = 2.88, gap = (CW - 4 * w) / 3;
  steps.forEach(([head, body], i) => {
    const x = X0 + i * (w + gap);
    card(s, x, 1.5, w, 2.75, `llm-step-${i + 1}-card`);
    hexBadge(s, i + 1, x + 0.25, 1.72, 0.5, `llm-step-${i + 1}-badge`);
    text(s, head, { x: x + 0.25, y: 2.4, w: w - 0.5, h: 0.4, fontSize: 16, bold: true, objectName: `llm-step-${i + 1}-head` });
    text(s, body, { x: x + 0.25, y: 2.85, w: w - 0.5, h: 1.3, fontSize: 14, color: C.text2, objectName: `llm-step-${i + 1}-body` });
  });
  text(s, "Пример ответа модели", { x: X0, y: 4.45, w: 6.6, h: 0.28, fontSize: 12, color: C.accent5, objectName: "llm-json-caption" });
  card(s, X0, 4.78, 6.6, 1.95, "llm-json-card", C.text1);
  const json = [
    '{"hypothesis": "в секторе NE расход выше",',
    ' "subgoals": [',
    '   {"type": "explore", "region": "NE"},',
    '   {"type": "goto", "target": "candidate_S2"},',
    '   {"type": "collect"},',
    '   {"type": "return_to_base"}]}',
  ].join("\n");
  text(s, json, { x: X0 + 0.3, y: 4.78, w: 6.0, h: 1.95, fontSize: 14, fontFace: "Courier New", color: C.background1, valign: "middle", objectName: "llm-json" });
  const gx = X0 + 6.95, gw = X0 + CW - gx;
  const guards = [
    "LLM не управляет скоростью и не выдаёт координаты: только выбор из кандидатов.",
    "Резерв на возврат считает код по верхней границе цены пути. План модели его не отменяет.",
  ];
  guards.forEach((g, i) => {
    hexBadge(s, "!", gx, 4.83 + i * 0.75, 0.4, `llm-guard-${i + 1}-badge`, C.text2);
    text(s, g, { x: gx + 0.62, y: 4.78 + i * 0.75, w: gw - 0.62, h: 0.7, fontSize: 14, objectName: `llm-guard-${i + 1}` });
  });
  text(s, [
    { text: "Статус: ", options: { bold: true } },
    { text: "контракт спроектирован, подключение к ai.mai.ru — следующий шаг." },
  ], { x: gx, y: 6.3, w: gw, h: 0.45, fontSize: 14, objectName: "llm-status" });
}

// ---------- 8. поиск образца ----------
pres.addSection({ title: S_SCIENCE });
{
  const Q = R.search;
  const s = content(S_SCIENCE, "Поиск образца: карта убеждений вместо градиента", "model",
    `Уровень три. Датчик даёт только расстояние до ближайшего образца, без направления. Мы ведём карту вероятностей: первый замер исключает ближайшую окрестность, после движения остаётся дуга, затем точка. Первый эксперимент на упрощённой модели, ${Q.runs} случайных положений: медиана пути ${ru(Q.bayes.median_m)} метра против ${ru(Q.gradient.median_m)} у градиентного подъёма, девяностый перцентиль ${ru(Q.bayes.p90_m)} против ${ru(Q.gradient.p90_m)}. Это модель без Gazebo, проверка в симуляторе завтра.`);
  const fw = 10.2;
  const img = image(s, "belief.png", X0 + (CW - fw) / 2, 1.38, { w: fw }, "belief", "Три карты вероятности положения образца: сначала широкая область, затем дуга, затем точка");
  const ry = 1.38 + img.h + 0.22, rh = 1.62, cw = 4.9;
  s.addChart(pres.charts.BAR, [{
    name: "Медиана пути, м",
    labels: ["Спираль и градиент", "Градиентный подъём", "Карта убеждений"],
    values: [Q.spiral.median_m, Q.gradient.median_m, Q.bayes.median_m],
  }], {
    x: X0, y: ry, w: cw, h: rh, barDir: "bar", chartColors: [HEX.context, HEX.context, HEX.blue],
    showTitle: true, title: "Медиана пути до сбора, м", titleFontSize: 12, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 12, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: CHART_TEXT.color, valAxisLabelColor: CHART_TEXT.color, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: CHART_TEXT.face, valAxisLabelFontFace: CHART_TEXT.face,
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60, valAxisMinVal: 0,
    objectName: "search-chart",
  });
  const tx = X0 + cw + 0.2, tw = (CW - cw - 0.4) / 2;
  statTile(s, tx, ry, tw, rh, pct(Q.bayes.median_m, Q.gradient.median_m), `медиана пути против градиента: ${ru(Q.bayes.median_m)} и ${ru(Q.gradient.median_m, 2)} м`, "search-median");
  statTile(s, tx + tw + 0.2, ry, tw, rh, pct(Q.bayes.p90_m, Q.gradient.p90_m), `90-й перцентиль: ${ru(Q.bayes.p90_m)} против ${ru(Q.gradient.p90_m)} м`, "search-p90");
  text(s, `Ложных сборов у карты убеждений нет, у градиента ${ru(Q.gradient.false_collects_per_run, 2)} на прогон. Предварительно, без Gazebo: ${Q.runs} положений, один образец, датчик 1 − d / ${ru(Q.model.range_m, 0)} м, шум σ = ${ru(Q.model.sigma, 2)}, движение по прямой.`,
    { x: X0, y: ry + rh + 0.1, w: CW, h: 0.45, fontSize: 12, color: C.accent5, objectName: "search-caveat" });
}

// ---------- 9. адаптация ----------
{
  const K = R.cusum;
  const s = content(S_SCIENCE, "Адаптация: замечаем разладку модели сами", "model",
    `Уровень четыре. Изменения среды не объявляются, поэтому агент следит за невязкой между прогнозом и фактом. Для грунтов это накопленная сумма по расходу батареи. На графике грунт стал дороже на двенадцатом метре, тревога сработала на ${ru(K.demo_alarm_m, 2)}. После тревоги агент переоценивает зону, и невязка уходит. На ${K.trials} прогонах модели медианная задержка ${ru(K.median_delay_m)} метр, ложная тревога в ${ru(K.false_alarm_pct)} процента прогонов.`);
  const labels = K.distance.map((d) => (Number.isInteger(d) && d % 4 === 0 ? String(d) : ""));
  s.addChart(pres.charts.LINE, [
    { name: "Накопленная невязка", labels, values: K.stat },
    { name: "Порог тревоги", labels, values: K.distance.map(() => K.h) },
  ], {
    x: X0, y: 1.45, w: 6.6, h: 4.3, chartColors: [HEX.blue, HEX.context], lineSize: 2, lineDataSymbol: "none",
    showTitle: true, title: "Накопленная невязка расхода батареи", titleFontSize: 13, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "путь по зоне, м", catAxisTitleColor: CHART_TEXT.color, catAxisTitleFontSize: 11, catAxisTitleFontFace: CHART_TEXT.face,
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.ink, legendFontFace: "+mn-lt",
    catAxisLabelColor: CHART_TEXT.color, valAxisLabelColor: CHART_TEXT.color, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: CHART_TEXT.face, valAxisLabelFontFace: CHART_TEXT.face, catAxisLabelFrequency: 1, catAxisMajorTickMark: "none", catAxisMinorTickMark: "none",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
    objectName: "cusum-chart",
  });
  text(s, `Грунт ×${K.from_mult} стал ×${K.to_mult} на ${ru(K.change_at_m, 0)}-м метре. Тревога на ${ru(K.demo_alarm_m, 2)} м, после переоценки невязка уходит.`,
    { x: X0, y: 5.85, w: 6.6, h: 0.6, fontSize: 14, objectName: "cusum-caption" });
  const tx = X0 + 6.95, tw = (X0 + CW - tx - 0.2) / 2;
  statTile(s, tx, 1.45, tw, 1.6, `${ru(K.median_delay_m)} м`, "медианная задержка обнаружения", "cusum-delay");
  statTile(s, tx + tw + 0.2, 1.45, tw, 1.6, `${ru(K.false_alarm_pct)}%`, "прогонов с ложной тревогой", "cusum-false");
  const rows = [
    ["Смена грунта", "невязка расхода батареи: сброс уверенности в зоне, переоценка, новый маршрут"],
    ["Новая опасная зона", "штраф hazard_hit: запретная область вокруг точки"],
    ["Сбой датчика", "разброс показаний вне нормы: идём по карте убеждений, сбор только при уверенности"],
  ];
  rows.forEach(([head, body], i) => {
    const y = 3.3 + i * 1.0;
    hexBadge(s, i + 1, tx, y + 0.05, 0.42, `adapt-row-${i + 1}-badge`, C.text2);
    text(s, head, { x: tx + 0.65, y, w: X0 + CW - tx - 0.65, h: 0.32, fontSize: 15, bold: true, objectName: `adapt-row-${i + 1}-head` });
    text(s, body, { x: tx + 0.65, y: y + 0.34, w: X0 + CW - tx - 0.65, h: 0.62, fontSize: 14, color: C.text2, objectName: `adapt-row-${i + 1}-body` });
  });
  text(s, `Модель детектора: ${K.trials.toLocaleString("ru-RU")} прогонов, шум замера ${ru(K.sigma, 2)} ед. на ${ru(K.step_m, 2)} м пути.`,
    { x: tx, y: 6.42, w: X0 + CW - tx, h: 0.3, fontSize: 12, color: C.accent5, objectName: "cusum-caveat" });
}

// ---------- 10. сценарии и оракул ----------
{
  const G = R.scenario;
  const s = content(S_SCIENCE, "Генератор сценариев и оракул как мера качества", "calc",
    `Адаптивность оценивается вместе с генерацией сценариев, поэтому сценарий у нас — это seed и уровень. На картинке сгенерированный сценарий hard: семь образцов, четыре грунта, опасная зона. Оракул знает мир заранее и перебирает порядок обхода: здесь он собирает все семь за ${ru(G.oracle_energy)} единицы батареи. Отношение расхода агента к расходу оракула — наша цена незнания. Научный цикл должен её снижать.`);
  const img = image(s, "scenario.png", X0, 1.45, { h: 5.3 }, "scenario", "Сгенерированный сценарий: семь образцов, четыре зоны грунта, опасная зона и маршрут оракула");
  const tx = X0 + img.w + 0.4, tw = X0 + CW - tx;
  const head = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 } } });
  s.addTable([
    [head("Уровень"), head("Образцы"), head("Грунты"), head("События")],
    ["easy", "3", "1", "нет"],
    ["medium", "5", "3", "нет"],
    ["hard", "7", "4", "грунты, зона, датчик"],
  ], { x: tx, y: 1.45, w: tw, colW: [1.25, 1.2, 1.1, tw - 3.55], rowH: 0.42, fontSize: 14, color: C.text1, valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.border }, objectName: "levels-table" });
  const half = (tw - 0.2) / 2;
  statTile(s, tx, 3.4, half, 1.6, `${G.oracle_collected} из ${G.samples}`, "образцов собирает оракул в этом сценарии", "oracle-count", 30);
  statTile(s, tx + half + 0.2, 3.4, half, 1.6, `${ru(G.oracle_energy)} из ${ru(G.battery, 0)}`, `единиц батареи нужно оракулу: ${ru(G.oracle_length_m)} м пути`, "oracle-energy", 30);
  text(s, [
    { text: "Цена незнания ", options: { bold: true } },
    { text: "= расход агента / расход оракула. Научный цикл обязан её снижать." },
  ], { x: tx, y: 5.2, w: tw, h: 0.62, fontSize: 14, objectName: "regret" });
  text(s, [
    { text: "Статус: ", options: { bold: true } },
    { text: "генератор и оракул — офлайн-прототип, узел did_judge в работе." },
  ], { x: tx, y: 6.0, w: tw, h: 0.62, fontSize: 14, objectName: "scenario-status" });
}

// ---------- 11. гипотезы ----------
{
  const Q = R.search, K = R.cusum, A = R.astar;
  const s = content(S_SCIENCE, "Гипотезы: что проверяем и чем измеряем", "design",
    "Шесть наших гипотез. У каждой есть метрика и порог, заданный заранее. Три уже проверены на моделях. По первой гипотезе результат частичный: хвост распределения сократился вдвое, но по медиане порог пока не взят. Остальные проверяем завтра на двух стендах: быстром кинематическом и в Gazebo.");
  const head = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 } } });
  const done = (t) => ({ text: t, options: { bold: true } });
  const plan = (t) => ({ text: t, options: { color: C.accent5 } });
  s.addTable([
    [head("№"), head("Гипотеза"), head("Метрика и порог"), head("Сейчас")],
    ["H1", "Карта убеждений находит образец короче, чем градиент и спираль", "Путь: медиана −20%, P90 −30%", done(`Модель: ${pct(Q.bayes.median_m, Q.gradient.median_m)} и ${pct(Q.bayes.p90_m, Q.gradient.p90_m)}`)],
    ["H2", "Множитель грунта узнаётся за 1,5 м пути по зоне", "Ошибка оценки не больше 20%", plan("День 2")],
    ["H3", "Знание грунтов экономит батарею на миссии", "Расход на 15% ниже, чем у кратчайшего пути", done(`Один маршрут: ${pct(A.smart_energy, A.naive_energy)}`)],
    ["H4", "LLM, выбирающая из кандидатов, не даёт неисполнимых планов", "0 отказов проверки; счёт не ниже жадного правила", plan("День 2")],
    ["H5", "Смена грунта видна по невязке расхода батареи", "Задержка до 1 м, ложных тревог до 5%", done(`Модель: ${ru(K.median_delay_m)} м и ${ru(K.false_alarm_pct)}%`)],
    ["H6", "Резерв по верхней границе цены пути гарантирует возврат", "Возврат на базу в 99% прогонов", plan("День 2")],
  ], { x: X0, y: 1.5, w: CW, colW: [0.7, 5.1, 3.85, 2.48], rowH: 0.62, fontSize: 14, color: C.text1, valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.border }, objectName: "hypotheses-table" });
  text(s, "Два стенда проверки: быстрый кинематический для сотен прогонов и Gazebo для физики. Мера качества — отношение к оракулу.",
    { x: X0, y: 6.05, w: CW, h: 0.6, fontSize: 14, objectName: "hypotheses-note" });
}

// ---------- 12. план ----------
pres.addSection({ title: S_PLAN });
{
  const s = content(S_PLAN, "План: от базового агента к полному циклу", "design",
    "Где мы сейчас и что дальше. Оранжевые уровни в работе сегодня, серые — завтра. Бонусные треки идут после уровней три и четыре в таком порядке: журнал гипотез, SLAM, два робота. Главный риск — время на интеграцию, поэтому агент обязан работать и без LLM, а судья и агент связаны только топиками.");
  const levels = [
    ["Среда", "симуляция проверена, судья в работе", true],
    ["Навигация", "A* — прототип, контроллер в работе", true],
    ["LLM-планировщик", "контракт готов, идёт подключение", true],
    ["Научный цикл", "день 2, утро", false],
    ["Адаптация", "день 2", false],
  ];
  const w = 2.26, gap = (CW - 5 * w) / 4;
  levels.forEach(([head, status, now], i) => {
    const x = X0 + i * (w + gap);
    hexBadge(s, i, x, 1.55, 0.72, `level-${i}-badge`, now ? C.accent1 : C.accent5);
    text(s, head, { x, y: 2.45, w, h: 0.38, fontSize: 16, bold: true, objectName: `level-${i}-head` });
    text(s, status, { x, y: 2.86, w, h: 0.7, fontSize: 14, color: C.text2, objectName: `level-${i}-status` });
  });
  text(s, "Бонусные треки", { x: X0, y: 3.8, w: CW, h: 0.4, fontSize: 16, bold: true, objectName: "bonus-head" });
  const bonus = [
    ["Два робота", "общая карта убеждений, раздел задач аукционом"],
    ["SLAM", "SLAM Toolbox вместо готовой карты"],
    ["Научный агент", "журнал гипотез с числовыми вердиктами"],
    ["Ассистент", "навыки для ROS и Gazebo, логи разработки"],
  ];
  const bw = 2.88, bgap = (CW - 4 * bw) / 3;
  bonus.forEach(([head, body], i) => {
    const x = X0 + i * (bw + bgap);
    card(s, x, 4.3, bw, 1.4, `bonus-${i + 1}-card`);
    text(s, head, { x: x + 0.22, y: 4.44, w: bw - 0.44, h: 0.36, fontSize: 15, bold: true, objectName: `bonus-${i + 1}-head` });
    text(s, body, { x: x + 0.22, y: 4.84, w: bw - 0.44, h: 0.78, fontSize: 14, color: C.text2, objectName: `bonus-${i + 1}-body` });
  });
  text(s, [
    { text: "Главный риск — время на интеграцию. ", options: { bold: true } },
    { text: "Страховка: агент работает и без LLM, а судья и агент связаны только топиками из условия." },
  ], { x: X0, y: 5.98, w: CW, h: 0.7, fontSize: 14, objectName: "risk" });
}

// ---------- 13. итог ----------
{
  const s = pres.addSlide({ masterName: "STATEMENT_DARK", sectionTitle: S_PLAN });
  s.addText("Каждый прогон оставляет проверяемое знание", { placeholder: "title" });
  text(s, "Журнал «Изыскателя» — это гипотеза, эксперимент, вердикт с числом и seed сценария. Его может перепроверить человек или другой агент. Так ИИ становится инфраструктурой знаний, а не чёрным ящиком.",
    { x: 0.7, y: 2.75, w: 5.9, h: 3.2, fontSize: 20, color: C.background2, objectName: "closing-text" });
  card(s, 7.2, 2.75, 5.43, 3.45, "questions-card", C.text2);
  text(s, "Вопросы к организаторам", { x: 7.5, y: 2.98, w: 4.85, h: 0.45, fontSize: 18, bold: true, color: C.background1, objectName: "questions-head" });
  const questions = [
    "Будет ли готовый судья /did/* или пишем свой?",
    "Совместим ли API ai.mai.ru с форматом OpenAI?",
    "На чьём сценарии пройдёт финальный прогон?",
  ];
  questions.forEach((q, i) => {
    hexBadge(s, i + 1, 7.5, 3.68 + i * 0.8, 0.4, `question-${i + 1}-badge`);
    text(s, q, { x: 8.12, y: 3.62 + i * 0.8, w: 4.25, h: 0.7, fontSize: 15, color: C.background1, valign: "middle", objectName: `question-${i + 1}` });
  });
  s.addNotes("Связь с темой форума. Результат прогона — не только счёт, но и журнал: гипотезы, эксперименты и вердикты, которые можно перепроверить по seed. И три вопроса к организаторам, от которых зависит наша завтрашняя работа.");
}

// ---------- приложение ----------
pres.addSection({ title: S_APP });
{
  const s = content(S_APP, "Приложение: модель среды v0", "design",
    "Запасной слайд. Условие не задаёт законы расхода батареи и сигнала датчика, поэтому мы зафиксировали свою версию. Все параметры лежат в сценарии и меняются без правок агента.");
  const head = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 } } });
  const key = (t) => ({ text: t, options: { bold: true } });
  s.addTable([
    [head("Параметр"), head("Решение v0")],
    [key("Батарея"), "Старт 60. Расход 1 ед./м × множитель грунта и 0,02 ед./с на простое"],
    [key("Грунты"), "Эллипсы поперечником 0,8–1,9 м, множитель ×2, ×3 или ×4"],
    [key("Датчик образцов"), "s = 1 − d / 2 м, обрезка до 0…1, шум σ = 0,05, частота 5 Гц"],
    [key("Сбор"), "Успех при d ≤ 0,30 м, иначе событие false_collect"],
    [key("Опасная зона"), "Событие hazard_hit, датчик слепнет на 10 с"],
    [key("События hard"), "По времени симуляции, без объявления: грунт, новая зона, сбой датчика"],
    [key("Счёт"), "+100 образец, +50 возврат, −20 ложный сбор, −30 опасная зона, −10 столкновение"],
  ], { x: X0, y: 1.5, w: CW, colW: [2.6, CW - 2.6], rowH: 0.52, fontSize: 14, color: C.text1, valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.border }, objectName: "world-table" });
  text(s, "Параметры наши: условие их не задаёт. Уточним у организаторов.", { x: X0, y: 5.95, w: CW, h: 0.4, fontSize: 14, objectName: "world-note" });
}
{
  const s = content(S_APP, "Приложение: риски и ответы", "design",
    "Запасной слайд про риски. Для каждого риска есть признак, по которому мы его заметим, и готовый ответ.");
  const head = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 } } });
  s.addTable([
    [head("Риск"), head("Как заметим"), head("Ответ")],
    ["LLM недоступна или отвечает не по схеме", "Проверка ответа", "Повтор с текстом ошибки, затем планировщик без LLM"],
    ["Одометрия уплывает", "Лидар расходится с картой больше 10 см", "Коррекция позы по лидару, резерв — AMCL"],
    ["Симуляция медленная на слабой машине", "Real-time factor ниже 0,8", "Отладка на кинематическом стенде, Gazebo для проверки"],
    ["Судья организаторов отличается от нашего", "Сверка топиков и правил", "Меняем параметры сценария: агент видит только топики"],
    ["Не успеваем бонусные треки", "План на утро дня 2", "Порядок: уровни 3–4, журнал, SLAM, два робота"],
  ], { x: X0, y: 1.5, w: CW, colW: [4.0, 3.6, CW - 7.6], rowH: 0.7, fontSize: 14, color: C.text1, valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.border }, objectName: "risk-table" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("готово:", OUT);
})();
