// Презентация первого чекпоинта, 4 слайда: node presentation/build_checkpoint1.js
// Содержание — из колоды команды («преза 1 чекпоинт DID26.pptx»), исправленное и переоформленное.
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
const OUT = path.join(__dirname, "checkpoint1.pptx");
const ru = (v, d = 1) => Number(v).toFixed(d).replace(".", ",");

const TEAM = [
  { name: "Журавлёва Полина Петровна", initials: "ЖП", role: "разработчик (Back/ROS)",
    does: "ROS 2 и Gazebo, генератор сценариев, учёт батареи и штрафов, движение робота" },
  { name: "Ефремова Анастасия Михайловна", initials: "ЕА", role: "ML (LLM)",
    does: "LLM для выбора подцелей, промпты и проверка ответов модели, журнал гипотез" },
  { name: "Путиловский Михаил Вячеславович", initials: "ПМ", role: "аналитик (исследователь)",
    does: "гипотеза и метрики, эксперименты и сравнение агентов, результаты и демо" },
];

const PLAN = [
  { date: "7.10", title: "Основа системы",
    items: [
      "ROS 2, Gazebo, робот и датчики — готово",
      "Генератор сценариев; учёт батареи, образцов и штрафов",
      "Движение с обходом препятствий",
      "LLM для выбора подцелей",
    ],
    result: "робот получает миссию и выполняет подцели" },
  { date: "8.10", title: "Исследование и адаптация",
    items: [
      "Поиск образцов по сигналу датчика",
      "Оценка расхода энергии на единицу пути",
      "Журнал гипотез",
      "Реакция на смену грунта, опасные зоны и сбои датчика: обновление модели и маршрута",
    ],
    result: "робот проверяет гипотезы и адаптируется к изменениям" },
  { date: "9.10", title: "Проверка и демонстрация",
    items: [
      "Весь цикл на easy, medium и hard",
      "Сравнение работы с адаптацией и без неё",
      "Исправление ошибок, проверка возврата на базу",
      "Запуск одной командой, результаты и финальное демо",
    ],
    result: "проверенная система уровней 0–4 с воспроизводимой демонстрацией" },
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

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Автономный ИИ-исследователь на роботе-платформе — чекпоинт 1";
pres.author = TEAM.map((m) => m.name.split(" ")[0]).join(", ");
const C = pres.SchemeColor;
const FOOTER = "Автономный ИИ-исследователь · DID Hack 2026 · чекпоинт 1";
const X0 = 0.6, CW = 12.13;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.75, w: 6.7, h: 2.35, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.3, w: 6.3, h: 0.9, fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.45, w: CW, h: 0.8, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: X0, y: 7.0, w: 8.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, valign: "top", align: "left", color: C.text1, fontSize: 14, ...o });
}

function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, objectName: name });
}

function hexBadge(slide, label, x, y, size, name, fill = C.accent1, fontSize = Math.round(size * 28)) {
  slide.addText(String(label), {
    shape: pres.shapes.HEXAGON, x, y, w: size * 1.14, h: size, fill: { color: fill }, color: C.background1,
    bold: true, fontSize, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name,
  });
}

function image(slide, file, x, y, size, name, altText) {
  const b = fs.readFileSync(ASSET(file));
  const ar = b.readUInt32BE(16) / b.readUInt32BE(20);
  const w = size.w || size.h * ar, h = size.h || size.w / ar;
  slide.addImage({ path: ASSET(file), x, y, w, h, objectName: name, altText });
  return { w, h };
}

// ---------- 1. титульник ----------
{
  const s = pres.addSlide({ masterName: "TITLE_DARK" });
  s.addText("«Автономный ИИ-исследователь на роботе-платформе»", { placeholder: "title" });
  s.addText("Первый чекпоинт: гипотеза, команда и план", { placeholder: "body" });
  text(s, "DID Hack 2026 · 7 октября", { x: 0.7, y: 5.3, w: 6.0, h: 0.4, fontSize: 16, color: C.accent1, bold: true, objectName: "event" });
  image(s, "arena_dark.png", 7.55, 0.95, { h: 5.6 }, "title-arena", "Арена turtlebot3_world и точки реального скана лидара");
  text(s, "Арена задачи; оранжевые точки — скан лидара из нашей симуляции", { x: 7.3, y: 6.65, w: 5.6, h: 0.3, fontSize: 11, color: C.background2, align: "center", objectName: "title-caption" });
  s.addNotes("Наш проект — автономный ИИ-исследователь на TurtleBot3. Сегодня показываем исследовательскую гипотезу, команду и план на три дня. На картинке арена задачи и настоящий скан лидара из нашей симуляции: среда уже запущена.");
}

// ---------- 2. исследовательская гипотеза ----------
{
  const A = R.astar;
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Исследовательская гипотеза", { placeholder: "title" });
  card(s, X0, 1.45, CW, 1.5, "hypothesis-card", C.text2);
  text(s, [
    { text: "Гипотеза. ", options: { bold: true, color: C.accent1 } },
    { text: "Агент, который по данным датчиков обновляет модель среды и перестраивает маршрут, соберёт больше образцов и чаще вернётся на базу, чем агент с фиксированным планом.", options: { color: C.background1 } },
  ], { x: X0 + 0.35, y: 1.45, w: CW - 0.7, h: 1.5, fontSize: 20, valign: "middle", objectName: "hypothesis-text" });

  const rows = [
    ["Как проверим", "Сравним агента с адаптацией и агента с фиксированным планом в одинаковых сценариях easy, medium и hard, по несколько запусков на каждый."],
    ["Что измерим", "Количество собранных образцов, долю успешных возвращений на базу и расход батареи."],
    ["Что ожидаем", "Наибольшую разницу — в hard, где среда меняется во время прогона."],
  ];
  const rw = 6.75, ys = [3.2, 4.55, 5.72];
  rows.forEach(([head, body], i) => {
    const y = ys[i];
    hexBadge(s, i + 1, X0, y + 0.03, 0.46, `row-${i + 1}-badge`, C.text2);
    text(s, head, { x: X0 + 0.72, y, w: rw - 0.72, h: 0.34, fontSize: 16, bold: true, objectName: `row-${i + 1}-head` });
    text(s, body, { x: X0 + 0.72, y: y + 0.38, w: rw - 0.72, h: 0.9, fontSize: 15, color: C.text2, objectName: `row-${i + 1}-body` });
  });

  const fx = X0 + rw + 0.75;
  const img = image(s, "astar_hypothesis.png", fx + 0.55, 3.15, { h: 3.25 }, "routes", "Два маршрута к цели: напрямик через дорогой грунт и в обход него");
  text(s, `Первый расчёт на карте арены: объезд дорогого грунта тратит ${ru(A.smart_energy)} ед. батареи вместо ${ru(A.naive_energy)}`,
    { x: fx, y: 6.42, w: X0 + CW - fx, h: 0.45, fontSize: 12, color: C.accent5, align: "center", objectName: "routes-caption" });
  s.addNotes(`Гипотеза: адаптивный агент, который обновляет модель среды по данным и перестраивает маршрут, соберёт больше образцов и чаще вернётся на базу, чем агент с фиксированным планом. Проверяем сравнением двух агентов в одинаковых сценариях трёх уровней, по несколько запусков. Измеряем собранные образцы, долю возвращений и расход батареи. Ждём наибольшую разницу в hard, где среда меняется. Справа первый расчёт: на карте арены объезд дорогого грунта тратит ${ru(A.smart_energy)} единицы батареи вместо ${ru(A.naive_energy)}.`);
}

// ---------- 3. команда и роли ----------
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Команда и роли", { placeholder: "title" });
  const w = 3.77, gap = (CW - 3 * w) / 2;
  TEAM.forEach((m, i) => {
    const x = X0 + i * (w + gap);
    card(s, x, 1.55, w, 4.7, `member-${i + 1}-card`);
    hexBadge(s, m.initials, x + 0.3, 1.9, 0.95, `member-${i + 1}-badge`, C.accent1, 20);
    text(s, m.name, { x: x + 0.3, y: 3.1, w: w - 0.6, h: 0.85, fontSize: 20, bold: true, objectName: `member-${i + 1}-name` });
    s.addText(m.role, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: x + 0.3, y: 4.08, w: w - 0.6, h: 0.46, fill: { color: C.text2 }, color: C.background1, bold: true, fontSize: 14, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `member-${i + 1}-role` });
    text(s, m.does, { x: x + 0.3, y: 4.78, w: w - 0.6, h: 1.5, fontSize: 15, color: C.text2, objectName: `member-${i + 1}-does` });
  });
  s.addNotes("Нас трое. Полина — разработчик: ROS, Gazebo, генератор сценариев и движение робота. Анастасия — ML: языковая модель для выбора подцелей и журнал гипотез. Михаил — аналитик: гипотеза, метрики, эксперименты и демонстрация.");
}

// ---------- 4. план на три дня ----------
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("План реализации проекта: 7–9 октября", { placeholder: "title" });
  const w = 3.9, gap = (CW - 3 * w) / 2, y0 = 1.5, h = 5.25;
  PLAN.forEach((d, i) => {
    const x = X0 + i * (w + gap);
    card(s, x, y0, w, h, `day-${i + 1}-card`);
    hexBadge(s, d.date, x + 0.25, y0 + 0.25, 0.8, `day-${i + 1}-date`, C.accent1, 16);
    text(s, d.title, { x: x + 1.35, y: y0 + 0.25, w: w - 1.6, h: 0.8, fontSize: 17, bold: true, valign: "middle", objectName: `day-${i + 1}-title` });
    const paragraphs = d.items.map((item, k) => ({ text: item, options: { bullet: true, breakLine: k < d.items.length - 1 } }));
    text(s, paragraphs, { x: x + 0.25, y: y0 + 1.3, w: w - 0.5, h: 2.55, fontSize: 14, paraSpaceAfter: 5, objectName: `day-${i + 1}-items` });
    card(s, x + 0.2, y0 + h - 1.3, w - 0.4, 1.1, `day-${i + 1}-result-card`, C.text2);
    text(s, [
      { text: "Результат: ", options: { bold: true, color: C.accent1 } },
      { text: d.result, options: { color: C.background1 } },
    ], { x: x + 0.38, y: y0 + h - 1.3, w: w - 0.76, h: 1.1, fontSize: 14, valign: "middle", objectName: `day-${i + 1}-result` });
  });
  s.addNotes("План на три дня. Сегодня основа: симуляция уже запущена, дальше генератор сценариев, движение и языковая модель для выбора подцелей. Завтра исследование и адаптация: поиск образцов, оценка расхода, журнал гипотез, реакция на изменения среды. Девятого проверяем весь цикл на трёх уровнях, сравниваем агента с адаптацией и без неё и готовим демонстрацию.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("готово:", OUT);
})();
