// Два слайда «Как работали с ИИ-агентами» — в оформлении презентации второго чекпоинта (build_checkpoint2.js),
// чтобы их можно было перенести в итоговую. Одна команда собирает всё заново:
//
//   cd presentation && node build_ai_process.js
//
// Что делает: 1) собирает ai_process.pptx; 2) пишет текст доклада ai_process_speech.md (он же — заметки к слайдам);
// 3) делает ai_process.pdf, картинки slides_ai_process/slide-NN.jpg и общий лист ai_process_preview.jpg.
// Ключ --no-pdf — только pptx и текст доклада. Для PDF нужен LibreOffice (soffice), для картинок — pdftoppm.
//
// Содержание и числа — из docs/requirements_check.html, раздел 8, сверены с research/findings/*.md.
// Числа записаны здесь, а не читаются из файлов: журнал сессии, из которого они взяты, в репозитории не лежит.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

// pptxgenjs и jszip: свои node_modules, если есть, иначе — основного каталога проекта (в рабочих деревьях их нет).
const MODULES = [path.join(__dirname, "node_modules"), process.env.DID_NODE_MODULES, "/Users/a/MAI/DID/presentation/node_modules"]
  .filter((p) => p && fs.existsSync(path.join(p, "pptxgenjs")));
process.env.NODE_PATH = MODULES.concat(process.env.NODE_PATH || []).join(path.delimiter);
require("module").Module._initPaths();
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/Users/a/.claude/skills/synced/0ec766ba-a970-423d-9985-0a1698a18ad8_57a5e5ff-4dfe-4d5a-9b46-49c5ea27665a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ARGS = new Set(process.argv.slice(2));
const NAME = "ai_process";
const OUT = path.join(__dirname, `${NAME}.pptx`);
const PDF = path.join(__dirname, `${NAME}.pdf`);
const SLIDES = path.join(__dirname, `slides_${NAME}`);
const PREVIEW = path.join(__dirname, `${NAME}_preview.jpg`);
const SPEECH = path.join(__dirname, `${NAME}_speech.md`);

// ---------- 1. числа и содержание ----------
// Сутки работы: журнал сессии и история репозитория (страница сверки, раздел 8).
const N = { hours: 24, tasks: 45, reviews: 11, commits: 104, checks: "600+", runs: "7 500", messages: 40 };
// Итоги исследований: отчёты research/findings (V1, R13, G2, R3, R10, P1) и отчёт ветки R16.
const R = {
  reviewFindings: 7,          // V1: шесть ошибок подсчёта и одна ошибка агента
  flipped: 2,                 // V1: два утверждения R4 были неверны
  mission: "6 из 6",          // R13, миссия «собери ровно два образца»: модель 6 из 6, правило 0 из 6
  gzBefore: "8 из 9", gzAfter: "7 из 7",   // G2: возврат на базу в Gazebo
  p1: "6,4",                  // P1: +6,38 [+2,89; +9,78] на нетронутых сценариях 8001–8040, трудный уровень (после ревью)
  wait: 73,                   // R3: −73,35 [−98,52; −49,23] при 15 с ожидания, трудный уровень
  noWait: 42,                 // R16: +41,9 [+25,0; +59,5] «ехать по правилу» против «стоять и ждать»; работа на ревью
};

// Плашки: Г — обзор Gemini, И — инженер, Р — ревью, ✓ — принято, Ч — решение человека, → — одно исследование выросло из другого.
const LANES = [
  ["R1, R4. Среда и гипотезы", "ГЧРИРИРИРИ✓", `ревью нашло ${R.reviewFindings} ошибок; ${R.flipped === 2 ? "два вывода" : R.flipped + " вывода"} перевёрнуты`],
  ["R13. Миссия, заданная словами", "ГИРИ✓", `модель выполняет миссию ${R.mission}, правило — 0 из 6`],
  ["G2. Потеря положения в Gazebo", "ИРИ✓", `возврат на базу в Gazebo: ${R.gzBefore} → ${R.gzAfter}`],
  ["P1. Где робот теряет очки", "ЧИРИ✓", `пережидание сбоя датчика: +${R.p1} очка на трудном уровне`],
  ["R3 → R16. Цена ожидания модели", "ГИИ✓→И", `ждать модель — минус ${R.wait} очка → ехать, пока думает: +${R.noWait}`],
  ["R10. Смена грунта на пути", "ИРИР✓", "выигрыш не показан — честный отрицательный результат"],
];
const DECISIONS = [
  ["«У нас не MVP, делаем в самом амбициозном виде»", "не минимальная версия: два симулятора, 20+ опытов"],
  ["«Робот-учёный» — идея коллеги", "верных выводов робота — 98,5%"],
  ["«Исследователи не пишут код»", `в их коде ревью нашло ${R.reviewFindings} ошибок`],
  ["«Главная цель — результативный робот»", "разбор потерь очков: +6 очков на трудном уровне"],
  ["«Не два варианта агента, а больше»", "шесть устройств агента и потолок 92,5 очка"],
  ["«Никакого маршрутизатора моделей»", null],
];

// ---------- 2. оформление (как в build_checkpoint2.js) ----------
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
pres.title = "ИИ-исследователь на роботе-платформе — как работали с ИИ-агентами";
const C = pres.SchemeColor;
const FOOTER = "ИИ-исследователь на роботе-платформе · DID Hack 2026 · работа с ИИ-агентами";
const X0 = 0.6, CW = 12.13, TOP = 1.85, BOTTOM = 6.8;

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

const SEC = "Как работали";
const SPEECH_PARTS = [];
function content(title, lead, notes, titleSize) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: SEC });
  // Размер из образца слайдов в самом заполнителе не переопределяется — задаём его тексту.
  s.addText(titleSize ? [{ text: title, options: { fontSize: titleSize } }] : title, { placeholder: "title" });
  s.addText(lead, { placeholder: "body" });
  s.addNotes(notes.join(" "));
  SPEECH_PARTS.push({ title, notes });
  return s;
}
function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, valign: "top", align: "left", color: C.text1, fontSize: 16, ...o });
}
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, objectName: name });
}
function arrow(slide, x1, y, x2, name, color = C.accent5) {
  slide.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, objectName: name, line: { color, width: 1.5, endArrowType: "triangle" } });
}
function bullets(slide, items, o) {
  const paragraphs = items.map((item, k) => ({ text: item, options: { bullet: true, breakLine: k < items.length - 1 } }));
  text(slide, paragraphs, { paraSpaceAfter: 4, ...o });
}
// Плашка шага: буква на цветном квадрате. Цвет повторяет букву, а не заменяет её.
const CHIP = { "Г": C.accent2, "И": C.accent5, "Р": C.accent6, "✓": C.accent3, "Ч": C.accent1 };
const CHIP_S = 0.27, CHIP_GAP = 0.04;
function chip(slide, letter, x, y, name) {
  slide.addText(letter, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.04, x, y, w: CHIP_S, h: CHIP_S, fill: { color: CHIP[letter] },
    color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name,
  });
}

pres.addSection({ title: SEC });

// ---------- слайд 1: кто что делает ----------
{
  const s = content("Человек руководит лабораторией, агенты в ней работают",
    "Агенты работали часами без участия человека, но направление, роли и приоритеты задавал человек", [
      "С ИИ-агентами мы работали как в лаборатории: человек ею руководит, агенты в ней работают.",
      "Человек задаёт цель, роли и ограничения и принимает решения на развилках.",
      "Ведущий агент ведёт план, пишет задания остальным, сводит их работу в общий код и сам пересчитывает числа.",
      "Исследователи читают литературу и кода не пишут. Инженеры пишут код и ставят опыты, каждый в своей копии кода.",
      "Отдельная модель-ревьюер читает каждую правку до вливания и ищет ошибки. Советников зовём редко, на развилках.",
      "Каждое исследование идёт по одному кругу: вопрос и гипотеза, задание, код и опыт, независимое ревью, исправления, вливание, пересчёт чисел.",
      `За сутки — ${N.tasks} поручений агентам, и все ${N.reviews} ревью вернули работу на доработку: ни одна не прошла с первого раза.`,
      "Агенты работали часами без участия человека, но направление, роли и приоритеты задавал человек.",
    ], 28);

  // схема: человек → ведущий агент → четыре роли
  const y0 = TOP, h = 2.66, mid = y0 + h / 2;
  const hx = X0, hw = 2.3, lx = 3.3, lw = 2.5, rx = 6.2, rgap = 0.12;
  card(s, hx, y0, hw, h, "human-card", C.text2);
  text(s, "Человек", { x: hx + 0.22, y: y0 + 0.16, w: hw - 0.44, h: 0.34, fontSize: 18, bold: true, color: C.accent1, objectName: "human-head" });
  bullets(s, ["цель", "роли", "ограничения", "решения на развилках"], { x: hx + 0.22, y: y0 + 0.6, w: hw - 0.4, h: 1.5, color: C.background1, objectName: "human-list" });
  text(s, `около ${N.messages} сообщений за сутки`, { x: hx + 0.22, y: y0 + h - 0.52, w: hw - 0.44, h: 0.42, fontSize: 12, color: C.background2, valign: "bottom", objectName: "human-caption" });
  arrow(s, hx + hw + 0.08, mid, lx - 0.08, "arrow-human-lead", C.accent1);

  card(s, lx, y0, lw, h, "lead-card");
  text(s, "Ведущий агент", { x: lx + 0.22, y: y0 + 0.16, w: lw - 0.44, h: 0.34, fontSize: 18, bold: true, objectName: "lead-head" });
  bullets(s, ["ведёт план", "пишет задания", "сводит ветки", "пересчитывает числа"], { x: lx + 0.22, y: y0 + 0.6, w: lw - 0.4, h: 1.5, color: C.text2, objectName: "lead-list" });
  text(s, "Claude Opus 5.5", { x: lx + 0.22, y: y0 + h - 0.42, w: lw - 0.44, h: 0.3, fontSize: 12, color: C.accent5, valign: "bottom", objectName: "lead-model" });
  arrow(s, lx + lw + 0.08, mid, rx - 0.08, "arrow-lead-roles");

  const rw = (X0 + CW - rx - rgap) / 2, rh = (h - rgap) / 2;
  const roles = [
    ["Исследователи", "обзоры литературы; кода не пишут", "Gemini 3.8 Flash"],
    ["Инженеры", "код, опыт, отчёт — каждый в своей копии кода", "Claude Opus 5.5"],
    ["Ревьюер (проверяющий)", "читает разницу, ищет ошибки, ничего не меняет", "GPT-6.1 Sol"],
    ["Советники", "редко, только на развилках", "Claude Fable 5.1 и GPT-6 Astra"],
  ];
  roles.forEach(([head, body, model], i) => {
    const x = rx + (i % 2) * (rw + rgap), y = y0 + Math.floor(i / 2) * (rh + rgap);
    card(s, x, y, rw, rh, `role-${i + 1}-card`);
    text(s, head, { x: x + 0.2, y: y + 0.12, w: rw - 0.4, h: 0.3, bold: true, objectName: `role-${i + 1}-head` });
    text(s, body, { x: x + 0.2, y: y + 0.43, w: rw - 0.35, h: 0.56, color: C.text2, objectName: `role-${i + 1}-body` });
    text(s, model, { x: x + 0.2, y: y + rh - 0.29, w: rw - 0.4, h: 0.22, fontSize: 12, color: C.accent5, valign: "bottom", objectName: `role-${i + 1}-model` });
  });

  // цикл одного исследования
  const cy = y0 + h + 0.1;
  text(s, [
    { text: "Цикл одного исследования", options: { bold: true, color: C.text1 } },
    { text: "   ветка — отдельная копия кода; ревью — независимая проверка кода другой моделью" },
  ], { x: X0, y: cy, w: CW, h: 0.26, fontSize: 13, color: C.accent5, valign: "middle", objectName: "cycle-label" });
  const steps = [["вопрос", "и гипотеза"], ["задание"], ["код и опыт", "в отдельной ветке"], ["независимое", "ревью"], ["исправления"], ["вливание", "в общий код"], ["пересчёт", "чисел"]];
  const sgap = 0.24, sy = cy + 0.32, sh = 0.66;
  const need = steps.map((l) => Math.max(...l.map((t) => t.length)) + (l[0] === "независимое" ? 5 : 3));          // ширина — по самой длинной строке
  const unit = (CW - sgap * (steps.length - 1)) / need.reduce((a, b) => a + b, 0);
  let sx = X0;
  steps.forEach((lines, i) => {
    const w = need[i] * unit, hot = lines[0] === "независимое";
    card(s, sx, sy, w, sh, `cycle-${i + 1}-card`, hot ? C.accent1 : C.background2);
    text(s, lines.join("\n"), { x: sx, y: sy, w, h: sh, align: "center", valign: "middle", bold: hot, color: hot ? C.background1 : C.text1, objectName: `cycle-${i + 1}-text` });
    if (i < steps.length - 1) arrow(s, sx + w + 0.03, sy + sh / 2, sx + w + sgap - 0.03, `cycle-${i + 1}-arrow`);
    sx += w + sgap;
  });

  // четыре числа
  const ty = sy + sh + 0.14, th = BOTTOM - ty, tgap = 0.15;
  const tiles = [[2.2, `${N.hours} часа`, "одна непрерывная сессия"], [2.0, String(N.tasks), "поручений агентам"], [2.75, `${N.reviews} из ${N.reviews}`, "ревью вернули работу на доработку"]];
  let tx = X0;
  const value = (v, x, w, name) => text(s, v, { x, y: ty + 0.06, w, h: 0.46, fontSize: 28, bold: true, valign: "middle", objectName: name });
  const label = (v, x, w, name) => text(s, v, { x, y: ty + 0.53, w, h: th - 0.58, fontSize: 13, color: C.text2, objectName: name });
  tiles.forEach(([w, v, l], i) => {
    card(s, tx, ty, w, th, `num-${i + 1}-card`);
    value(v, tx + 0.22, w - 0.4, `num-${i + 1}-value`);
    label(l, tx + 0.22, w - 0.36, `num-${i + 1}-label`);
    tx += w + tgap;
  });
  const w4 = X0 + CW - tx, sub = [[1.4, String(N.commits), "правки кода (коммита)"], [1.65, N.checks, "автоматических проверок"], [1.3, N.runs, "прогонов"]];
  card(s, tx, ty, w4, th, "num-4-card");
  let qx = tx + 0.22;
  sub.forEach(([w, v, l], i) => {
    value(v, qx, w - 0.1, `num-4-${i + 1}-value`);
    label(l, qx, w - 0.1, `num-4-${i + 1}-label`);
    qx += w;
  });
}

// ---------- слайд 2: решения людей и ход исследований ----------
{
  const s = content("Что решали люди и как прошли исследования",
    "На развилках решал человек; код каждой работы до вливания читала независимая модель-ревьюер", [
      "Слева — исследования. Каждая плашка — один шаг: обзор, работа инженера, ревью или решение человека.",
      `В первых двух исследованиях ревью нашло ${R.reviewFindings} ошибок, и два вывода пришлось перевернуть.`,
      `Робот с языковой моделью выполнил миссию, заданную словами, ${R.mission}; робот с простым правилом — ни разу.`,
      `В Gazebo робот вернулся на базу в прогонах ${R.gzBefore}, после правки — ${R.gzAfter}. Разбор потерь дал плюс шесть очков на трудном уровне.`,
      `Ожидание ответа модели стоило ${R.wait} очка; если робот едет, пока модель думает, возвращается ${R.noWait} — эта работа ещё на ревью.`,
      "А один опыт выигрыша не показал, и мы говорим об этом прямо.",
      "Справа — решения людей: не минимальная версия, робот-учёный, исследователи не пишут код, главная цель — результативный робот.",
      "Качество держится на устройстве процесса, а не на вере в один ответ модели.",
    ]);

  const lw = 6.4, rxx = X0 + lw + 0.3, rww = X0 + CW - rxx, headH = 0.3, y1 = TOP + headH + 0.12, yEnd = 5.99;
  // легенда плашек
  const legend = [["Г", "Gemini"], ["И", "инженер"], ["Р", "ревью"], ["✓", "принято"], ["Ч", "решение человека"]];
  let gx = X0;
  legend.forEach(([letter, word], i) => {
    const w = word.length * 0.097 + 0.04;
    chip(s, letter, gx, TOP + 0.015, `legend-${i + 1}-chip`);
    text(s, word, { x: gx + CHIP_S + 0.07, y: TOP, w, h: headH, fontSize: 13, color: C.text2, valign: "middle", wrap: false, objectName: `legend-${i + 1}-word` });
    gx += CHIP_S + 0.07 + w + 0.11;
  });
  // дорожки исследований
  const lh = (yEnd - y1) / LANES.length;
  LANES.forEach(([name, chain, result], i) => {
    const y = y1 + i * lh;
    if (i > 0) s.addShape(pres.shapes.LINE, { x: X0, y: y - 0.03, w: lw, h: 0, line: { color: C.background2, width: 1 }, objectName: `lane-${i + 1}-rule` });
    const letters = [...chain];
    const cw = letters.length * (CHIP_S + CHIP_GAP) - CHIP_GAP;
    text(s, name, { x: X0, y, w: lw - cw - 0.15, h: 0.28, bold: true, valign: "middle", objectName: `lane-${i + 1}-name` });
    letters.forEach((letter, k) => {
      const x = X0 + lw - cw + k * (CHIP_S + CHIP_GAP);
      if (letter === "→") text(s, "→", { x, y, w: CHIP_S, h: 0.28, fontSize: 14, bold: true, color: C.accent5, align: "center", valign: "middle", objectName: `lane-${i + 1}-next` });
      else chip(s, letter, x, y + 0.005, `lane-${i + 1}-chip-${k + 1}`);
    });
    text(s, result, { x: X0, y: y + 0.275, w: lw, h: 0.27, color: C.text2, valign: "middle", objectName: `lane-${i + 1}-result` });
  });

  // решения людей
  text(s, "Решения людей", { x: rxx, y: TOP, w: rww, h: headH, bold: true, valign: "middle", objectName: "decisions-head" });
  const dx = rxx + CHIP_S + 0.12, dw = X0 + CW - dx;
  const heights = DECISIONS.map(([q, out]) => (q.length > 40 ? 0.54 : 0.28) + (out ? 0.24 : 0));
  const dgap = (yEnd - y1 - heights.reduce((a, b) => a + b, 0)) / (DECISIONS.length - 1);
  let dy = y1;
  DECISIONS.forEach(([quote, out], i) => {
    const qh = heights[i] - (out ? 0.24 : 0);
    chip(s, "Ч", rxx, dy + 0.005, `decision-${i + 1}-chip`);
    text(s, quote, { x: dx, y: dy, w: dw, h: qh, bold: true, valign: "middle", objectName: `decision-${i + 1}-quote` });
    if (out) text(s, out, { x: dx, y: dy + qh, w: dw, h: 0.24, fontSize: 13, color: C.text2, valign: "middle", objectName: `decision-${i + 1}-out` });
    dy += heights[i] + dgap;
  });

  // вывод
  const cy = 6.1;
  card(s, X0, cy, CW, BOTTOM - cy, "conclusion-card", C.text2);
  text(s, [
    { text: "Вывод. ", options: { bold: true, color: C.accent1 } },
    { text: "Качество держится на устройстве процесса — отдельная ветка, независимое ревью, пересчёт чисел,\u00A0— а не на вере в один ответ модели." },
  ], { x: X0 + 0.3, y: cy, w: CW - 0.6, h: BOTTOM - cy, color: C.background1, valign: "middle", objectName: "conclusion-text" });
}

// ---------- текст доклада ----------
function writeSpeech() {
  const WPS = 2.2;                                   // слов в секунду при спокойной речи
  const words = (t) => t.split(/\s+/).filter(Boolean).length;
  const secs = SPEECH_PARTS.map((p) => Math.round(words(p.notes.join(" ")) / WPS / 5) * 5);
  const lines = [
    "# Текст доклада к слайдам «Как работали с ИИ-агентами»",
    "",
    `Два слайда, около ${secs.reduce((a, b) => a + b, 0)} секунд спокойной речи (время посчитано по числу слов). Текст пересобирается вместе`,
    "со слайдами (`node build_ai_process.js`) и лежит в заметках к слайдам; править его нужно в скрипте, а не здесь.",
    "",
  ];
  SPEECH_PARTS.forEach((p, i) => {
    lines.push(`## Слайд ${i + 1}. ${p.title} (≈ ${secs[i]} с)`, "");
    for (const n of p.notes) lines.push(n);
    lines.push("");
  });
  lines.push("## К вопросам жюри", "",
    "- **Вы сами читали код?** Нет. Код каждой работы до вливания читала независимая модель-ревьюер, и ни одна работа не прошла ревью с первого раза. Человек задавал цель, роли и ограничения и принимал решения на развилках — около сорока содержательных сообщений за сутки.",
    "- **Агенты работали сами?** Они работали часами без участия человека, в том числе ночью, но направление, роли и приоритеты задавал человек, и несколько его сообщений развернули работу.",
    `- **Что нашло ревью?** В первых двух исследованиях — шесть ошибок подсчёта и одну ошибку самого робота: разовую потерю заряда от штрафа он принимал за дорогой грунт. Два вывода оказались неверны и исправлены. Всего ревью было ${N.reviews}, и все ${N.reviews} вернули работу на доработку.`,
    `- **Откуда +${R.noWait} очка?** Исследование R16 на трудном уровне: робот едет, пока модель думает, вместо того чтобы стоять. Измерено на имитаторе модели, работа сдана и ждёт ревью — на слайде у неё нет отметки «принято».`,
    "- **Почему показываете опыт без выигрыша?** Исследование R10 (смена грунта на пути) выигрыша в очках не показало. Правка оставлена выключенной, результат записан как отрицательный.",
    "- **Где логи разработки?** План исследований — `research/agenda.yaml`, правила — `research/PROTOCOL.md`, отчёты с числами — `research/findings/`, история правок — в git.",
    "");
  fs.writeFileSync(SPEECH, lines.join("\n"));
}

// ---------- PDF и картинки слайдов (как render_checkpoint2.js) ----------
function render() {
  const find = (names) => names.find((n) => (n.includes("/") ? fs.existsSync(n) : spawnSync("which", [n]).status === 0)) || null;
  const soffice = find(["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice", "libreoffice"]);
  const pdftoppm = find(["pdftoppm", "/opt/homebrew/bin/pdftoppm"]);
  if (!soffice) return console.warn("LibreOffice не найден: PDF и картинки не обновлены (brew install --cask libreoffice).");
  // Свой профиль: без окон первого запуска и без чужих настроек.
  const profile = path.join(os.tmpdir(), "did_lo_profile");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "did_pdf_"));
  const r = spawnSync(soffice, [`-env:UserInstallation=file://${profile}`, "--headless", "--convert-to", "pdf", "--outdir", tmp, OUT], { encoding: "utf8", timeout: 180000 });
  const made = path.join(tmp, `${NAME}.pdf`);
  if (r.status !== 0 || !fs.existsSync(made)) { console.error("LibreOffice не собрал PDF:\n" + (r.stderr || r.stdout || r.error)); process.exit(1); }
  fs.copyFileSync(made, PDF);
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log("PDF:", PDF);
  if (!pdftoppm) return console.warn("pdftoppm не найден: картинки слайдов не обновлены (brew install poppler).");
  fs.rmSync(SLIDES, { recursive: true, force: true });
  fs.mkdirSync(SLIDES);
  const p = spawnSync(pdftoppm, ["-jpeg", "-r", "110", PDF, path.join(SLIDES, "slide")], { encoding: "utf8" });
  if (p.status !== 0) { console.error(p.stderr); process.exit(1); }
  const files = fs.readdirSync(SLIDES).filter((f) => f.endsWith(".jpg")).sort().map((f) => path.join(SLIDES, f));
  console.log(`картинки: ${SLIDES} (${files.length} слайда)`);
  // Общий лист: оба слайда на одной картинке.
  const sheet = `
import sys
from PIL import Image
ims = [Image.open(f) for f in sys.argv[2:]]
w, h, gap = 960, 540, 8
out = Image.new('RGB', (len(ims) * w + (len(ims) + 1) * gap, h + 2 * gap), '#9A9A9A')
for i, im in enumerate(ims):
    out.paste(im.resize((w, h), Image.LANCZOS), (gap + i * (w + gap), gap))
out.save(sys.argv[1], quality=88)
`;
  const q = spawnSync(process.env.PYTHON || "python3", ["-c", sheet, PREVIEW, ...files], { encoding: "utf8" });
  if (q.status === 0) console.log("общий лист:", PREVIEW);
  else console.warn("общий лист не собран (нужен Python с Pillow)");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  writeSpeech();
  console.log("готово:", OUT);
  console.log("доклад:", SPEECH);
  if (!ARGS.has("--no-pdf")) render();
})();
