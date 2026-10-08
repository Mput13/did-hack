// Финальная презентация DID Hack: 8 слайдов к пятиминутному выступлению (большая часть — живой показ) и
// приложение для вопросов жюри. Одна команда собирает всё заново:
//
//   cd presentation && PYTHON=/usr/local/bin/python3 node build_final.js
//
// Что делает: 1) рисует графики из чисел ниже (matplotlib) в assets/final/; 2) собирает final.pptx;
// 3) пишет текст выступления final_speech.md (вариант на 2 минуты — он же в заметках к слайдам);
// 4) делает final.pdf, картинки slides_final/slide-NN.jpg и общий лист final_preview.jpg.
// Ключ --no-pdf — только рисунки, pptx и текст. Для PDF нужен LibreOffice (soffice), для картинок — pdftoppm.
//
// Оформление — как в build_checkpoint2.js и build_ai_process.js; слайды о работе с ИИ-агентами перенесены из
// build_ai_process.js. Числа записаны здесь, у каждого в комментарии — файл, из которого оно взято: сводки прогонов
// (runs/*/summary.json) лежат только в основном каталоге, в рабочих деревьях их нет.
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
const PY = process.env.PYTHON || "python3";
const ROOT = path.join(__dirname, "..");
const NAME = "final";
const OUT = path.join(__dirname, `${NAME}.pptx`);
const PDF = path.join(__dirname, `${NAME}.pdf`);
const SLIDES = path.join(__dirname, `slides_${NAME}`);
const PREVIEW = path.join(__dirname, `${NAME}_preview.jpg`);
const SPEECH = path.join(__dirname, `${NAME}_speech.md`);
const FIG = path.join(__dirname, "assets", "final");
const C2 = (name) => path.join(__dirname, "assets", "c2", name);

// ---------- 1. числа ----------
// Счёт на трудном уровне, быстрый симулятор: [среднее, низ и верх 95% интервала].
const SCORE = {
  fixed: [59.349, 55.46, 63.37],     // runs/E1/summary.json, groups: fixed, hard, 40 сценариев 1001–1040
  adaptive: [74.89, 70.78, 78.91],    // runs/E22b/summary.json, groups: adaptive, base, hard, 40 сценариев 8001–8040 (отчёт P1)
  waits: [81.27, 77.17, 84.75],       // runs/E22b/summary.json, groups: adaptive_v2, base, hard — те же 40 сценариев (отчёт P1)
  ceiling: [92.5, 92.44, 92.57],      // runs/E23b/summary.json, groups: oracle_all, base, hard, 160 сценариев (отчёт A1)
  adaptiveA1: 73.3,                   // research/findings/A1.md: нынешний агент на тех же 160 сценариях
  tourA1: 72.9,                       // research/findings/A1.md: план на остаток прогона
  samplesOnly: 87.8,                  // research/findings/A1.md: подсказаны только места образцов
};
// Парные разности счёта на одинаковых сценариях: [среднее, низ, верх]. Трудный уровень, быстрый симулятор.
const EFFECT = {
  adaptive: [13.3, 7.4, 19.0],        // presentation/data/results.json, e1.diff: адаптивный − план заранее (E1, hard)
  waits: [6.4, 2.9, 9.8],             // research/findings/P1.md, «Коротко»: +6,38 [+2,89; +9,78]
  hazard: [3.5, 0.4, 7.1],            // research/findings/R1.md: adaptive − no_hazard, условие «только зоны ×3»: +3,54 [+0,39; +7,08]
  tour: [-0.4, -2.7, 1.9],            // research/findings/A1.md, таблица перепроверки: adaptive_tour, базовые правила
  soil: [0.02, -0.03, 0.07],          // research/findings/R10.md, «Коротко»: разница, приходящаяся на смену пола
  review: [-1.5, -4.2, 1.0],          // research/findings/V1.md, «Коротко»: цена исправления ошибки агента
};
const GZ = {
  before: "8 из 9",                   // research/findings/G1.md: возврат на базу в девяти трудных сценариях
  after: "7 из 7",                    // research/findings/G2.md, «Коротко»: 7 прогонов из 7 на итоговом коде
  today: "5 из 5", todayScore: "75–91", // presentation/data/results.json, e7.after; счёт — runs/F1_v2/*/hard-*.json.gz (75,1–91,5)
  pair: "+0,3 [−10,4; +11,1]", gzScore: "76,2", fastScore: "75,9",   // research/findings/G1.md: Gazebo − быстрый, 9 пар
  tour: 53, mission: "5 из 5",        // docs/demo_script.md: замер показа, средний уровень, сценарий 3
};
const LLM = {
  mission: "6 из 6", rule: "0 из 6",  // research/findings/R13.md, «Коротко»: миссия «собери ровно два образца и возвращайся»
  same: 96, model: "71,3", ruleScore: "71,2",   // research/findings/L2.md и R3.md: совпадение решений с правилом, счёт
  wait: 15, cost: 73, costCi: "[−98,5; −49,2]", from: "75,7", to: "2,3",   // research/findings/R3.md: −73,35 [−98,52; −49,23]
};
const REVIEW = { found: 7, flipped: 2 };   // research/findings/V1.md: шесть ошибок подсчёта и одна ошибка агента; два вывода R4
const JOURNAL = { confirmed: "9 из 10", refutedWrong: 38, runs: 320 };
// research/findings/R4.md: верных среди подтверждённых 90,7% (адаптивный) и 89,6% (исследователь), базовые правила;
// 37,7% опровергнутых отвергнуты зря; 320 прогонов E17. После G3 серия E17 пересчитывается — сверить перед финалом.
const SLAM = { runs: 5, agree: "95,8–98,7", pillars: "9 из 9", origin: "около 5 см", worst: "9,5" };
// research/agenda.yaml (F2, result) и research/findings/F2.md: сдвиг начала координат 1,5–5,2 см, худший столб 9,5 см.
const TEAM2 = {                       // research/agenda.yaml (M1, result) — пересчёт после G3
  time: ["103,7", "77,8"], dt: "−25,9 с [−30,5; −21,2]", charge: ["100,4", "74,1"], dq: "−26,3 [−29,6; −22,8]",
  home: ["80 из 80", "75"], last: "+3,5 с [−0,6; +8,1]", n: 80,
};
// Процесс, на момент сборки (8.10 вечером): git rev-list --count main = 178; ./px python -m pytest -q --collect-only
// = 744 проверки; сумма runs по 21 сводке runs/*/summary.json основного каталога = 14 902 прогона.
const N = { commits: 178, checks: 744, runs: "14 900", series: 21, messages: 40 };
const E1_WALL = { runs: 240, seconds: 37 };   // runs/E1/summary.json: wall_s = 37,1 — серия считается за 37 секунд

// Ход исследований (слайд приложения): из build_ai_process.js, статусы сверены с research/agenda.yaml.
// Г — Gemini, И — Claude Opus (инженер), Р — GPT Sol (проверяющий), ✓ — принято; строка без ✓ — работа не принята.
const LANES = [
  ["R1, R4. Среда и гипотезы", "ГРИРИРИРИ✓", `${REVIEW.found} ошибок найдено, два вывода перевёрнуты`],
  ["K1–K4. Обзоры литературы", "ГГГГИ✓", "проверка нашла две выдуманные ссылки"],
  ["R13. Миссия, заданная словами", "ГИРИ✓", `модель — миссия ${LLM.mission}, правило — 0 из 6`],
  ["R3 → R16. Цена ожидания модели", "ГИИ✓→ИР", `ждать — минус ${LLM.cost} очка; R16 — в работе`],
  ["R10. Смена грунта на пути", "ИРИР✓", "выигрыш не показан — так и записано"],
  ["G2, G3. Потеря положения в Gazebo", "ИРИ✓", `возврат на базу в Gazebo: ${GZ.before} → ${GZ.after}`],
  ["P1. Где робот теряет очки", "ИРИ✓", "пережидает сбой датчика: +6,4 очка"],
  ["A1. Новые схемы агента", "ИРИ✓", "не лучше нынешней; потолок — 92,5 очка"],
  ["M1. Два робота с координацией", "ИРИ✓", "прогон короче на 26 с, заряда на четверть меньше"],
  ["R14. Самокалибровка датчика", "ИР", "проверка вернула работу; в работе"],
  ["F2. SLAM вместо готовой карты", "ГИРИРИ✓", "карта робота совпала с готовой на 96–99%"],
];

// ---------- 2. графики из чисел (matplotlib) ----------
function figures() {
  fs.mkdirSync(FIG, { recursive: true });
  const data = {
    out: FIG,
    bars: [
      ["План заранее", ...SCORE.fixed, "#2A78D6", "40 сценариев"],
      ["Адаптивный агент", ...SCORE.adaptive, "#F0A27F", "40 сценариев"],
      ["+ пережидает сбой датчика", ...SCORE.waits, "#EB6834", "те же 40"],
      ["Потолок: знает всё заранее", ...SCORE.ceiling, "#9AA0A6", "160 сценариев"],
    ],
    effects: [
      ["Адаптация вместо плана заранее", ...EFFECT.adaptive],
      ["Пережидание сбоя датчика", ...EFFECT.waits],
      ["Память об опасных зонах*", ...EFFECT.hazard],
      ["Новые схемы планирования", ...EFFECT.tour],
      ["Реакция на смену грунта", ...EFFECT.soil],
    ],
    slam: path.join(ROOT, "research", "findings", "F2", "main5_map.npz"),
    slamCheck: path.join(ROOT, "research", "findings", "F2", "main5_check.json"),
  };
  const code = String.raw`
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.loads(sys.argv[1])
INK, MUTED, GRID, GREEN, GREY = '#14181C', '#5B6168', '#DDE0E4', '#1BAF7A', '#9AA0A6'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 13, 'axes.edgecolor': GRID, 'text.color': INK})
ru = lambda v, d=1: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')
sign = lambda v, d=1: ('+' if v > 0 else '') + ru(v, d)

# 1. Счёт по вариантам агента и потолок. Размер рисунка равен месту на слайде: кегль на слайде тот же.
fig, ax = plt.subplots(figsize=(5.9, 3.0))
rows = D['bars'][::-1]
for i, (name, m, lo, hi, color, note) in enumerate(rows):
    ax.barh(i, m, color=color, height=0.62, zorder=2)
    if hi - lo > 1:
        ax.plot([lo, hi], [i, i], color=INK, lw=1.6, zorder=3, solid_capstyle='butt')
        for v in (lo, hi):
            ax.plot([v, v], [i - 0.13, i + 0.13], color=INK, lw=1.6, zorder=3)
    ax.text(max(hi, m) + 1.5, i, ru(m), va='center', ha='left', fontsize=17, fontweight='bold')
    ax.text(1.5, i, name, va='center', ha='left', fontsize=13, color='white', fontweight='bold', zorder=4)
    ax.text(109, i, note, va='center', ha='left', fontsize=12, color=MUTED)
ax.set_xlim(0, 141); ax.set_ylim(-0.55, len(rows) - 0.45)
ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.tick_params(axis='x', colors=MUTED, labelsize=12, length=0)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.set_xlabel('счёт судьи, очки', color=MUTED, fontsize=12, loc='left')
fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.19)
fig.savefig(D['out'] + '/scores.png', dpi=220); plt.close(fig)

# 2. Что дало каждое улучшение: парные разности с интервалами.
fig, ax = plt.subplots(figsize=(5.9, 3.0))
rows = D['effects'][::-1]
ax.axvline(0, color=INK, lw=1.2, zorder=1)
for i, (name, m, lo, hi) in enumerate(rows):
    shown = lo > 0 or hi < 0
    color = GREEN if shown else GREY
    ax.plot([lo, hi], [i, i], color=color, lw=4, zorder=2, solid_capstyle='round')
    ax.plot([m], [i], 'o', color=color, ms=10, zorder=3, markeredgecolor='white', markeredgewidth=1.5)
    ax.text(-27.5, i + 0.17, name, va='center', ha='left', fontsize=13, fontweight='bold')
    d = 2 if abs(m) < 0.1 else 1
    ax.text(-27.5, i - 0.2, f'{sign(m, d)} [{sign(lo, d)}; {sign(hi, d)}]' + ('' if shown else ' — не показано'),
            va='center', ha='left', fontsize=12, color=INK if shown else MUTED)
ax.set_xlim(-28, 20.5); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks([]); ax.set_xticks([-5, 0, 5, 10, 15, 20]); ax.set_xticklabels(['−5', '0', '+5', '+10', '+15', '+20'])
ax.tick_params(axis='x', colors=MUTED, labelsize=12, length=0)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.set_xlabel('прибавка к счёту, очки', color=MUTED, fontsize=12, loc='right')
fig.text(0.005, 0.03, '* при трёх новых опасных зонах за прогон', fontsize=12, color=MUTED)
fig.subplots_adjust(left=0.005, right=0.985, top=0.99, bottom=0.19)
fig.savefig(D['out'] + '/effects.png', dpi=220); plt.close(fig)

# 3. Карта, построенная роботом через SLAM Toolbox (пятый прогон), и настоящие места столбов мира.
z = np.load(D['slam'], allow_pickle=True)
g, res, ox, oy = z['grid'], float(z['res']), float(z['ox']), float(z['oy'])
img = np.full(g.shape + (3,), 0.80)
img[g == 0] = (1, 1, 1); img[g == 100] = (0.08, 0.09, 0.11)
fig, ax = plt.subplots(figsize=(3.3, 3.07))
ax.imshow(img, origin='lower', extent=(ox, ox + g.shape[1] * res, oy, oy + g.shape[0] * res), interpolation='nearest')
for p in json.load(open(D['slamCheck']))['pillars']:
    ax.add_patch(plt.Circle((p['x'], p['y']), 0.15, fill=False, color='#EB6834', lw=2))
ax.plot([-2.0], [-0.5], 's', color='#2A78D6', ms=8)
ax.set_xlim(-2.9, 2.6); ax.set_ylim(-2.55, 2.55); ax.set_aspect('equal'); ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(D['out'] + '/slam_map.png', dpi=220); plt.close(fig)
`;
  const r = spawnSync(PY, ["-c", code, JSON.stringify(data)], { encoding: "utf8" });
  if (r.status !== 0) { console.error("графики не построены:\n" + (r.stderr || r.error)); process.exit(1); }
}
figures();

// ---------- 3. оформление (как в build_checkpoint2.js) ----------
const TEAM = [   // как на титуле презентации второго чекпоинта (build_checkpoint2.js)
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

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Автономный ИИ-исследователь на роботе-платформе — финал";
pres.author = TEAM.map((m) => m[0].split(" ")[0]).join(", ");
const C = pres.SchemeColor;
const FOOTER = "Автономный ИИ-исследователь · DID Hack 2026 · финал";
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

// Текст выступления: two — вариант «2 минуты слайды», one — «1 минута слайды» (пусто — слайд пролистывается).
const SPEECH_PARTS = [];
function content(section, title, lead, say, titleSize) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  // Размер из образца слайдов в самом заполнителе не переопределяется — задаём его тексту.
  s.addText(titleSize ? [{ text: title, options: { fontSize: titleSize } }] : title, { placeholder: "title" });
  s.addText(lead, { placeholder: "body" });
  s.addNotes((say.two || say.ask || []).join(" "));
  SPEECH_PARTS.push({ title, ...say });
  return s;
}
function text(slide, str, o) {
  slide.addText(str, { isTextBox: true, margin: 0, valign: "top", align: "left", color: C.text1, fontSize: 16, ...o });
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
function bullets(slide, items, o) {
  const paragraphs = items.map((item, k) => ({ text: item, options: { bullet: true, breakLine: k < items.length - 1 } }));
  text(slide, paragraphs, { paraSpaceAfter: 4, ...o });
}
// Плитка с крупным числом: число, под ним пояснение, внизу — где измерено.
function numTile(slide, x, y, w, h, value, label, where, name, o = {}) {
  card(slide, x, y, w, h, `${name}-card`, o.fill || C.background2);
  const vs = o.valueSize || 28, vh = vs / 72 * 1.2;
  text(slide, value, { x: x + 0.22, y: y + 0.1, w: w - 0.4, h: vh, fontSize: vs, bold: true, valign: "middle", color: o.valueColor || C.text1, objectName: `${name}-value` });
  text(slide, label, { x: x + 0.22, y: y + 0.12 + vh, w: w - 0.4, h: h - vh - (where ? 0.5 : 0.2), fontSize: o.labelSize || 14, color: o.labelColor || C.text2, objectName: `${name}-label` });
  if (where) text(slide, where, { x: x + 0.22, y: y + h - 0.34, w: w - 0.4, h: 0.26, fontSize: 12, color: o.whereColor || C.accent5, valign: "middle", objectName: `${name}-where` });
}
// Значок шага: логотип модели на светлой плитке; «принято» — галочка. Логотипы — товарные знаки своих владельцев,
// файлы взяты из набора @lobehub/icons-static-svg и лежат в assets/logos (svg и png 256×256).
const LOGOS = path.join(__dirname, "assets", "logos");
const LOGO = { "Г": path.join(LOGOS, "gemini.png"), "И": path.join(LOGOS, "claude.png"), "Р": path.join(LOGOS, "openai.png") };
const CHIP_S = 0.28, CHIP_GAP = 0.04;
function logo(slide, letter, x, y, size, name) {
  slide.addImage({ path: LOGO[letter], x, y, w: size, h: size, objectName: name });
}
function chip(slide, letter, x, y, name) {
  if (letter === "✓") {
    slide.addText("✓", {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.04, x, y, w: CHIP_S, h: CHIP_S, fill: { color: C.accent3 },
      color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name,
    });
    return;
  }
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: CHIP_S, h: CHIP_S, rectRadius: 0.04, fill: { color: C.background2 }, objectName: `${name}-tile` });
  logo(slide, letter, x + 0.045, y + 0.045, CHIP_S - 0.09, name);
}
// Таблица из строк с разделителями: cols — [ширина, заголовок], rows — массивы ячеек (строка или {text, ...свойства}).
function table(slide, cols, rows, y0, yEnd, name, o = {}) {
  const headH = 0.34, lh = (yEnd - y0 - headH) / rows.length, gap = 0.2;
  let x = X0;
  const xs = cols.map(([w]) => { const v = x; x += w; return v; });
  cols.forEach(([w, head], j) => text(slide, head, { x: xs[j], y: y0, w: w - gap, h: headH, fontSize: 13, bold: true, color: C.accent5, valign: "middle", objectName: `${name}-head-${j + 1}` }));
  rows.forEach((row, i) => {
    const y = y0 + headH + i * lh;
    slide.addShape(pres.shapes.LINE, { x: X0, y, w: CW, h: 0, line: { color: C.background2, width: 1 }, objectName: `${name}-rule-${i + 1}` });
    row.forEach((cell, j) => {
      const c = typeof cell === "string" ? { text: cell } : cell;
      const { text: t, ...rest } = c;
      text(slide, t, { x: xs[j], y, w: cols[j][0] - gap, h: lh, fontSize: o.fontSize || 14, valign: "middle", color: j === 0 ? C.text1 : C.text2, bold: j === 0, ...rest, objectName: `${name}-${i + 1}-${j + 1}` });
    });
  });
}
const MARK = { ok: { text: "✓", color: C.accent3, bold: true, fontSize: 20, align: "center" }, part: { text: "частично", color: C.accent4, bold: true, fontSize: 13, align: "center" } };

const SEC_A = "Выступление", SEC_B = "Приложение: к вопросам жюри";

// ---------- 1. титул ----------
pres.addSection({ title: SEC_A });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: SEC_A });
  s.addText("Автономный ИИ-исследователь на роботе-платформе", { placeholder: "title" });
  s.addText("Робот сам ищет скрытые образцы, замечает, что среда изменилась, и возвращается на базу", { placeholder: "body" });
  text(s, "DID Hack 2026 · финал · 9 октября", { x: 0.7, y: 4.75, w: 6.0, h: 0.4, fontSize: 16, color: C.accent1, bold: true, objectName: "event" });
  text(s, TEAM.map(([name, role], i) => ({ text: `${name} — ${role}`, options: { breakLine: i < TEAM.length - 1 } })),
    { x: 0.7, y: 5.35, w: 6.5, h: 1.1, fontSize: 14, color: C.background2, paraSpaceAfter: 4, objectName: "team" });
  image(s, C2("title_run.png"), { x: 7.5, y: 0.75, w: 5.3, h: 5.3 }, "title-run", "Арена и путь робота в настоящем прогоне в Gazebo");
  // presentation/data/checkpoint2.json, gazebo.runs (medium): собрано 5 из 5, возврат на базу.
  text(s, "TurtleBot3 Burger в Gazebo: путь робота в настоящем прогоне, собрано 5 образцов из 5, возврат на базу",
    { x: 7.5, y: 6.15, w: 5.3, h: 0.5, fontSize: 12, color: C.background2, align: "center", objectName: "title-caption" });
  const say = {
    two: ["Наш проект — автономный ИИ-исследователь. Робот TurtleBot3 сам ищет скрытые образцы, замечает, что среда изменилась, и возвращается на базу."],
    one: ["Наш проект — автономный ИИ-исследователь: робот сам ищет скрытые образцы и возвращается на базу."],
  };
  s.addNotes(say.two.join(" "));
  SPEECH_PARTS.push({ title: "Титул", ...say });
}

// ---------- 2. задача и идея ----------
{
  const s = content(SEC_A, "Робот ищет то, чего не видит, — как исследователь",
    "Образцы, дорогой грунт и опасные зоны скрыты; камеры нет — только лидар, заряд батареи и одно число датчика", {
      two: [
        "Образцы, дорогой грунт и опасные зоны скрыты, камеры нет.",
        "Поэтому робот действует как исследователь: ставит гипотезу, проверяет её действием, смотрит на данные и делает вывод.",
        "Девять из десяти гипотез, которые он сам подтвердил, верны.",
      ],
      one: ["Всё важное на арене скрыто, поэтому робот действует как исследователь: гипотеза, действие, данные, вывод."],
    });
  const iw = 2.72, ih = iw * 958 / 1000, gap = 0.26;
  [["Как на самом деле", "task_truth.png", "образцы, зоны дорогого грунта, опасные зоны", "Арена со скрытой правдой"],
    ["Что видит робот", "task_robot_view.png", "стены по лидару и число датчика: образец где-то на окружности", "То же место глазами робота"]]
    .forEach(([head, file, cap, alt], i) => {
      const x = X0 + i * (iw + gap);
      text(s, head, { x, y: TOP, w: iw, h: 0.3, bold: true, objectName: `view-${i + 1}-head` });
      image(s, C2(file), { x, y: TOP + 0.36, w: iw, h: ih }, `view-${i + 1}`, alt);
      text(s, cap, { x, y: TOP + 0.42 + ih, w: iw, h: 0.44, fontSize: 12, color: C.accent5, objectName: `view-${i + 1}-caption` });
    });
  const gy = 5.72, lw = 2 * iw + gap;
  card(s, X0, gy, lw, BOTTOM - gy, "goal-card", C.text2);
  text(s, [
    { text: "Цель. ", options: { bold: true, color: C.accent1 } },
    { text: "Собрать больше образцов и вернуться на базу, пока не села батарея. Об изменениях среды никто не сообщает.", options: { color: C.background1 } },
  ], { x: X0 + 0.22, y: gy, w: lw - 0.44, h: BOTTOM - gy, valign: "middle", objectName: "goal-text" });

  // научный цикл: четыре шага и возврат к началу
  const rx = X0 + lw + 0.5, rw = X0 + CW - rx;
  text(s, "Научный цикл робота", { x: rx, y: TOP, w: rw, h: 0.3, bold: true, objectName: "cycle-head" });
  const steps = [
    ["Гипотеза", "«в этой точке лежит образец», «здесь грунт вдвое дороже»"],
    ["Действие", "подъехать, проехать по участку, постоять"],
    ["Данные", "показание датчика, расход заряда, штраф судьи"],
    ["Вывод", "подтверждена или опровергнута — карта и маршрут обновлены"],
  ];
  const sy = TOP + 0.4, sh = 0.72, sg = 0.14, hw = 1.75;
  steps.forEach(([head, body], i) => {
    const y = sy + i * (sh + sg);
    card(s, rx, y, rw, sh, `cycle-${i + 1}-card`);
    card(s, rx, y, hw, sh, `cycle-${i + 1}-tab`, i === 0 ? C.accent1 : C.text2);
    text(s, head, { x: rx, y, w: hw, h: sh, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `cycle-${i + 1}-head` });
    text(s, body, { x: rx + hw + 0.2, y, w: rw - hw - 0.35, h: sh, color: C.text2, valign: "middle", objectName: `cycle-${i + 1}-body` });
    if (i < steps.length - 1) arrowDown(s, rx + hw / 2, y + sh + 0.01, y + sh + sg - 0.01, `cycle-${i + 1}-arrow`);
  });
  const ty = sy + 4 * (sh + sg) + 0.06;
  const th = BOTTOM - ty;
  card(s, rx, ty, rw, th, "journal-card");
  text(s, JOURNAL.confirmed, { x: rx + 0.22, y: ty, w: 1.75, h: th, fontSize: 30, bold: true, valign: "middle", objectName: "journal-value" });
  text(s, "гипотез, которые робот сам подтвердил, верны по скрытой правде", { x: rx + 2.0, y: ty + 0.08, w: rw - 2.2, h: th - 0.42, fontSize: 14, color: C.text2, valign: "middle", objectName: "journal-label" });
  text(s, `быстрый симулятор, ${JOURNAL.runs} прогонов`, { x: rx + 2.0, y: ty + th - 0.34, w: rw - 2.2, h: 0.26, fontSize: 12, color: C.accent5, valign: "middle", objectName: "journal-where" });
}

// ---------- 3. как устроено ----------
function architecture(s, y0, h, detailed) {
  const sx = X0, sw = 4.0;
  card(s, sx, y0, sw, h, "stand-card");
  text(s, "Стенд", { x: sx + 0.25, y: y0 + 0.12, w: 2, h: 0.3, bold: true, color: C.accent5, objectName: "stand-label" });
  const stand = [
    ["Генератор сценариев", "уровень и номер → расстановка"],
    ["Судья", "заряд, датчик, штрафы, очки"],
    ["Симулятор", "Gazebo — показ, быстрый — серии"],
  ];
  const bh = (h - 0.56 - 0.2 - 2 * 0.12) / 3;
  stand.forEach(([head, sub], i) => {
    const y = y0 + 0.56 + i * (bh + 0.12);
    card(s, sx + 0.2, y, sw - 0.4, bh, `stand-${i + 1}-box`, C.background1);
    text(s, head, { x: sx + 0.38, y: y + 0.07, w: sw - 0.76, h: 0.3, bold: true, objectName: `stand-${i + 1}-head` });
    text(s, sub, { x: sx + 0.38, y: y + 0.38, w: sw - 0.76, h: bh - 0.42, fontSize: 14, color: C.text2, objectName: `stand-${i + 1}-sub` });
  });
  const ax = 7.75, aw = X0 + CW - ax;
  card(s, ax, y0, aw, h, "agent-card", C.text1);
  text(s, "Агент", { x: ax + 0.25, y: y0 + 0.12, w: 2, h: 0.3, bold: true, color: C.accent1, objectName: "agent-label" });
  const inner = [
    ["Картина мира", "образцы, грунт, зоны, положение", false],
    ["Планировщик", "правило или языковая модель", !detailed],
    ["Исполнитель", "путь по сетке и скорость колёс", false],
    ["Журнал гипотез", "что предположил и что вышло", false],
  ];
  const bw = (aw - 0.6) / 2, ih = (h - 0.56 - 0.2 - 0.14) / 2;
  inner.forEach(([head, sub, hot], i) => {
    const x = ax + 0.2 + (i % 2) * (bw + 0.2), y = y0 + 0.56 + Math.floor(i / 2) * (ih + 0.14);
    card(s, x, y, bw, ih, `agent-${i + 1}-box`, hot ? C.accent1 : C.text2);
    text(s, head, { x: x + 0.18, y: y + 0.1, w: bw - 0.36, h: 0.32, bold: true, color: C.background1, objectName: `agent-${i + 1}-head` });
    text(s, sub, { x: x + 0.18, y: y + 0.45, w: bw - 0.36, h: ih - 0.5, fontSize: 14, color: hot ? C.background1 : C.background2, objectName: `agent-${i + 1}-sub` });
  });
  // каналы между стендом и агентом
  const lx1 = sx + sw + 0.1, lx2 = ax - 0.1, lw = lx2 - lx1, mid = y0 + h / 2;
  if (detailed) {
    const chan = (label, y, dir, labelY, k) => {
      arrow(s, dir > 0 ? lx1 : lx2, y, dir > 0 ? lx2 : lx1, `chan-${k}`, C.accent1);
      text(s, label.map((t, i) => ({ text: t, options: { breakLine: i < label.length - 1 } })),
        { x: lx1, y: labelY, w: lw, h: 0.21 * label.length + 0.04, fontSize: 12, align: "center", valign: "middle", fontFace: "Courier New", objectName: `chan-${k}-label` });
    };
    chan(["/did/battery  /did/score", "/did/sample_sensor", "/did/events"], mid - 0.3, 1, mid - 1.02, 1);
    chan(["/did/collect /did/finish"], mid - 0.08, -1, mid - 0.04, 2);
    chan(["/scan /odom"], mid + 0.72, 1, mid + 0.44, 3);
    chan(["/cmd_vel"], mid + 0.94, -1, mid + 0.98, 4);
  } else {
    text(s, "заряд, датчик образцов,\nштрафы, лидар, одометрия", { x: lx1, y: mid - 0.82, w: lw, h: 0.56, fontSize: 14, align: "center", valign: "bottom", objectName: "chan-to-label" });
    arrow(s, lx1, mid - 0.16, lx2, "chan-to", C.accent1);
    arrow(s, lx2, mid + 0.16, lx1, "chan-from", C.accent1);
    text(s, "скорость колёс,\n«собрать», «финиш»", { x: lx1, y: mid + 0.26, w: lw, h: 0.56, fontSize: 14, align: "center", objectName: "chan-from-label" });
  }
}
{
  const s = content(SEC_A, "Один агент и один судья, два симулятора",
    "Агент и стенд общаются только через каналы из условия: нашего судью можно заменить судьёй организаторов", {
      two: [
        "Стенд с судьёй и агент общаются только через каналы из условия.",
        "Один и тот же агент ездит в Gazebo и в быстром симуляторе, где мы ставим серии опытов.",
        "Языковая модель читает миссию и предлагает подцели; гипотезы и движение — программа.",
      ],
      one: [],
    });
  const h = 3.2;
  architecture(s, TOP, h, false);
  const y = TOP + h + 0.2, lw = 7.35, g = 0.2;
  card(s, X0, y, lw, BOTTOM - y, "llm-card", C.text2);
  text(s, [
    { text: "Место языковой модели. ", options: { bold: true, color: C.accent1 } },
    { text: "Читает миссию, заданную словами, и предлагает подцели: разведать область, проверить место, вернуться. Ответ проверяется; при сбое решает правило. Гипотезы, путь и скорость колёс — программа.", options: { color: C.background1 } },
  ], { x: X0 + 0.25, y, w: lw - 0.5, h: BOTTOM - y, valign: "middle", objectName: "llm-text" });
  card(s, X0 + lw + g, y, CW - lw - g, BOTTOM - y, "sim-card");
  text(s, [
    { text: "Зачем два симулятора. ", options: { bold: true } },
    { text: `Gazebo — физика и показ. Быстрый — серии: ${E1_WALL.runs} прогонов за ${E1_WALL.seconds} секунд, поэтому у каждого вывода есть интервал.`, options: { color: C.text2 } },
  ], { x: X0 + lw + g + 0.25, y, w: CW - lw - g - 0.5, h: BOTTOM - y, valign: "middle", objectName: "sim-text" });
}

// ---------- 4. результаты числами ----------
{
  const s = content(SEC_A, "Трудный уровень: 81 очко при потолке 92,5",
    "Быстрый симулятор, новые сценарии; отрезки — 95% интервалы. Внизу — проверка в Gazebo", {
      two: [
        "Главные числа — трудный уровень, быстрый симулятор.",
        "Адаптивный агент набирает 75 очков, а научившись пережидать сбой датчика — 81, при потолке 92 с половиной.",
        "Справа — что дало каждое улучшение. Две идеи выигрыша не показали, и это тоже результат.",
        "В Gazebo после защиты от потери положения робот вернулся на базу семь раз из семи.",
      ],
      one: [
        "На трудном уровне в быстром симуляторе робот набирает 81 очко при потолке 92 с половиной.",
        "Две идеи выигрыша не показали — это тоже результат. В Gazebo робот вернулся на базу семь раз из семи.",
      ],
    });
  const fw = 5.9, fh = 3.0, hy = TOP, fy = TOP + 0.34, rx = X0 + CW - fw;
  text(s, "Счёт по вариантам агента", { x: X0, y: hy, w: fw, h: 0.3, bold: true, objectName: "scores-head" });
  s.addImage({ path: path.join(FIG, "scores.png"), x: X0, y: fy, w: fw, h: fh, objectName: "scores-chart", altText: "Счёт на трудном уровне: план заранее 59,3; адаптивный агент 74,9; с пережиданием сбоя 81,3; потолок 92,5" });
  text(s, "Что дало каждое улучшение", { x: rx, y: hy, w: fw, h: 0.3, bold: true, objectName: "effects-head" });
  s.addImage({ path: path.join(FIG, "effects.png"), x: rx, y: fy, w: fw, h: fh, objectName: "effects-chart", altText: "Прибавка к счёту: адаптация +13,3; пережидание сбоя +6,4; память об опасных зонах +3,5; новые схемы планирования −0,4; реакция на смену грунта +0,02" });
  const ty = fy + fh + 0.14, th = BOTTOM - ty, tg = 0.15, tw = (CW - 2 * tg) / 3;
  const tiles = [
    [`${GZ.before} → ${GZ.after}`, "возврат на базу после защиты от потери положения", "Gazebo, трудный уровень"],
    [GZ.today, `возвратов с пережиданием сбоя датчика; счёт ${GZ.todayScore}`, "Gazebo, трудный уровень"],
    [`${GZ.gzScore} и ${GZ.fastScore}`, "счёт в Gazebo и в быстром симуляторе: различие не показано", "9 пар одинаковых трудных сценариев"],
  ];
  tiles.forEach(([v, l, w], i) => {
    const x = X0 + i * (tw + tg);
    card(s, x, ty, tw, th, `gz-${i + 1}-card`);
    text(s, v, { x: x + 0.22, y: ty + 0.08, w: tw - 0.4, h: 0.46, fontSize: 26, bold: true, valign: "middle", objectName: `gz-${i + 1}-value` });
    text(s, l, { x: x + 0.22, y: ty + 0.56, w: tw - 0.4, h: th - 0.9, fontSize: 13, color: C.text2, objectName: `gz-${i + 1}-label` });
    text(s, w, { x: x + 0.22, y: ty + th - 0.32, w: tw - 0.4, h: 0.24, fontSize: 12, color: C.accent5, valign: "middle", objectName: `gz-${i + 1}-where` });
  });
}

// ---------- 5. адаптация и языковая модель ----------
{
  const s = content(SEC_A, "Изменения робот замечает сам; модель читает миссию",
    "Об изменениях среды никто не сообщает: робот узнаёт о них по датчику, расходу заряда и штрафам", {
      two: [
        "Изменения никто не объявляет: робот сам замечает сбой датчика, новые опасные зоны и потерю положения.",
        "Языковая модель полезна, когда задание меняют словами: шесть миссий из шести против нуля у правила.",
        "Но ответ идёт пятнадцать секунд, поэтому на показе модель — в записи.",
      ],
      one: ["Изменения робот замечает сам. Языковая модель нужна, когда задание меняют словами: шесть миссий из шести против нуля у правила."],
    }, 28);
  const lw = 6.6, g = 0.3, rx = X0 + lw + g, rw = CW - lw - g;
  text(s, "Что робот замечает сам", { x: X0, y: TOP, w: lw, h: 0.3, bold: true, objectName: "adapt-head" });
  const rows = [
    ["Датчик образцов врёт", "стоит и пережидает сбой", "+6,4", "очка [+2,9; +9,8]", "быстрый симулятор", C.accent3],
    ["Штрафы: новая опасная зона", "запоминает зону и объезжает", "+3,5", "очка [+0,4; +7,1]", "быстрый, три новые зоны", C.accent3],
    ["Положение потеряно после удара", "ищет себя заново, едет на базу", GZ.after, `возвратов (было ${GZ.before})`, "Gazebo", C.accent3],
    ["Грунт на пути подорожал", "замечает, но очков это не даёт", "+0,02", "[−0,03; +0,07]", "выигрыш не показан", C.accent5],
  ];
  const ry = TOP + 0.4, rh = (BOTTOM - ry - 3 * 0.12) / 4, nw = 2.3;
  rows.forEach(([sign, act, num, unit, where, color], i) => {
    const y = ry + i * (rh + 0.12);
    card(s, X0, y, lw, rh, `adapt-${i + 1}-card`);
    text(s, sign, { x: X0 + 0.22, y: y + 0.1, w: lw - nw - 0.4, h: 0.32, bold: true, valign: "middle", objectName: `adapt-${i + 1}-sign` });
    text(s, "→ " + act, { x: X0 + 0.22, y: y + 0.46, w: lw - nw - 0.4, h: rh - 0.56, color: C.text2, objectName: `adapt-${i + 1}-act` });
    text(s, num, { x: X0 + lw - nw, y: y + 0.06, w: nw - 0.2, h: 0.46, fontSize: 26, bold: true, color, valign: "middle", objectName: `adapt-${i + 1}-num` });
    text(s, unit, { x: X0 + lw - nw, y: y + 0.52, w: nw - 0.2, h: 0.24, fontSize: 13, color: C.text2, valign: "middle", objectName: `adapt-${i + 1}-unit` });
    text(s, where, { x: X0 + lw - nw, y: y + rh - 0.3, w: nw - 0.2, h: 0.24, fontSize: 12, color: C.accent5, valign: "middle", objectName: `adapt-${i + 1}-where` });
  });

  text(s, "Языковая модель: QWEN и DeepSeek", { x: rx, y: TOP, w: rw, h: 0.3, bold: true, objectName: "llm-head" });
  const my = ry, mh = 2.5;
  card(s, rx, my, rw, mh, "mission-card", C.text2);
  text(s, "Миссия словами: «собери ровно два образца и возвращайся»", { x: rx + 0.25, y: my + 0.14, w: rw - 0.5, h: 0.6, color: C.background1, objectName: "mission-text" });
  const hw = (rw - 0.5) / 2;
  [["модель", LLM.mission, C.accent1], ["правило", LLM.rule, C.background2]].forEach(([who, v, color], i) => {
    const x = rx + 0.25 + i * hw;
    text(s, v, { x, y: my + 0.82, w: hw, h: 0.75, fontSize: 40, bold: true, color, valign: "middle", objectName: `mission-${i + 1}-value` });
    text(s, who, { x, y: my + 1.56, w: hw, h: 0.3, color: C.background2, valign: "middle", objectName: `mission-${i + 1}-who` });
  });
  text(s, "модели с сервера МАИ, быстрый симулятор; 6 сценариев по одному прогону — счёт прогонов, а не доля", { x: rx + 0.25, y: my + mh - 0.52, w: rw - 0.5, h: 0.44, fontSize: 12, color: C.background2, valign: "middle", objectName: "mission-where" });
  const by = my + mh + 0.12, bh = (BOTTOM - by - 0.12) / 2;
  [[`${LLM.same}%`, "решений обычной миссии — те же, что у правила; счёт одинаков", "обычной миссии модель ничего не добавляет"],
    [`−${LLM.cost}`, `очка стоит ожидание ответа ${LLM.wait} с на трудном уровне`, "поэтому на показе модель — в записи"]]
    .forEach(([v, l, w], i) => {
      const y = by + i * (bh + 0.12);
      card(s, rx, y, rw, bh, `llm-${i + 1}-card`);
      text(s, v, { x: rx + 0.22, y, w: 1.25, h: bh, fontSize: 26, bold: true, valign: "middle", objectName: `llm-${i + 1}-value` });
      text(s, l, { x: rx + 1.5, y: y + 0.06, w: rw - 1.65, h: bh - 0.36, fontSize: 13, color: C.text2, valign: "middle", objectName: `llm-${i + 1}-label` });
      text(s, w, { x: rx + 1.5, y: y + bh - 0.3, w: rw - 1.65, h: 0.24, fontSize: 12, color: C.accent5, valign: "middle", objectName: `llm-${i + 1}-where` });
    });
}

// ---------- 6. бонусные треки ----------
{
  const s = content(SEC_A, "Бонусные треки: четыре закрыты числами",
    "У каждого числа подписано, где оно измерено: Gazebo или быстрый симулятор", {
      two: [
        "Бонусные треки. Карта, которую робот строит сам через SLAM, совпадает с готовой на 96–99 процентов.",
        "Два робота с координацией заканчивают на 26 секунд раньше пары без связи.",
        "Журнал гипотез сверен со скрытой правдой.",
      ],
      one: [],
    });
  const mw = 3.3, lw = CW - mw - 0.35, g = 0.12, rh = (BOTTOM - TOP - 3 * g) / 4, nw = 2.55;
  const rows = [
    ["SLAM вместо готовой карты", `Карта робота совпала с готовой; столбов ${SLAM.pillars}; начало координат совпадает с мировым с точностью ${SLAM.origin}`,
      `${SLAM.agree}%`, `Gazebo, ${SLAM.runs} прогонов`],
    ["Два робота, которые координируются", `Против пары без связи: прогон ${TEAM2.time[0]} → ${TEAM2.time[1]} с, заряд ${TEAM2.charge[0]} → ${TEAM2.charge[1]}; оба вернулись ${TEAM2.home[0]} против ${TEAM2.home[1]}`,
      "−26 с", `быстрый симулятор, ${TEAM2.n} сценариев`],
    ["Научный агент: журнал гипотез", `Подтверждённые роботом гипотезы верны по скрытой правде; но ${JOURNAL.refutedWrong}% опровергнутых отвергнуты зря`,
      JOURNAL.confirmed, `быстрый симулятор, ${JOURNAL.runs} прогонов`],
    ["Команда с ИИ-агентами", "Столько ошибок независимая проверка нашла в коде первых исследований; два вывода перевёрнуты",
      `${REVIEW.found} ошибок`, "план, отчёты и история правок — в репозитории"],
  ];
  rows.forEach(([head, body, num, where], i) => {
    const y = TOP + i * (rh + g);
    card(s, X0, y, lw, rh, `bonus-${i + 1}-card`);
    text(s, head, { x: X0 + 0.22, y: y + 0.1, w: lw - nw - 0.4, h: 0.32, bold: true, valign: "middle", objectName: `bonus-${i + 1}-head` });
    text(s, body, { x: X0 + 0.22, y: y + 0.44, w: lw - nw - 0.4, h: rh - 0.5, fontSize: 14, color: C.text2, objectName: `bonus-${i + 1}-body` });
    text(s, num, { x: X0 + lw - nw, y: y + 0.1, w: nw - 0.2, h: 0.52, fontSize: 26, bold: true, color: C.accent1, valign: "middle", objectName: `bonus-${i + 1}-num` });
    text(s, where, { x: X0 + lw - nw, y: y + 0.64, w: nw - 0.2, h: rh - 0.72, fontSize: 12, color: C.accent5, objectName: `bonus-${i + 1}-where` });
  });
  const mx = X0 + CW - mw;
  s.addImage({ path: path.join(FIG, "slam_map.png"), x: mx, y: TOP, w: mw, h: mw * 3.07 / 3.3, objectName: "slam-map", altText: "Карта арены, построенная роботом через SLAM Toolbox, и места девяти столбов мира" });
  text(s, `Карта, которую робот построил сам (SLAM Toolbox, Gazebo). Оранжевые кольца — настоящие места столбов; худший столб ушёл на ${SLAM.worst} см. Синий квадрат — база.`,
    { x: mx, y: TOP + mw * 3.07 / 3.3 + 0.1, w: mw, h: 1.0, fontSize: 12, color: C.accent5, objectName: "slam-caption" });
}

// ---------- 7. как работали с ИИ-агентами (из build_ai_process.js, числа обновлены) ----------
{
  const s = content(SEC_A, "Человек руководит лабораторией, агенты в ней работают",
    "Агенты работали часами без участия человека, но направление, роли и приоритеты задавал человек", {
      two: [
        "Работали как лаборатория: человек задаёт цель и роли, агенты пишут код и ставят опыты.",
        "Каждую работу до приёма читает независимая модель. Ни одна не прошла проверку с первого раза.",
      ],
      one: ["Работали как лаборатория: каждую работу агентов до приёма проверяет независимая модель."],
    }, 28);

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
  logo(s, "И", lx + 0.22, y0 + h - 0.33, 0.19, "lead-logo");
  text(s, "Claude Opus 5.5", { x: lx + 0.5, y: y0 + h - 0.42, w: lw - 0.72, h: 0.3, fontSize: 12, color: C.accent5, valign: "bottom", objectName: "lead-model" });
  arrow(s, lx + lw + 0.08, mid, rx - 0.08, "arrow-lead-roles");

  const rw = (X0 + CW - rx - rgap) / 2, rh = (h - rgap) / 2;
  const roles = [
    ["Исследователи", "обзоры литературы; кода не пишут", "Gemini 3.8 Flash", "Г"],
    ["Инженеры", "код, опыт, отчёт — каждый в своей копии кода", "Claude Opus 5.5", "И"],
    ["Ревьюер (проверяющий)", "читает разницу, ищет ошибки, ничего не меняет", "GPT-6.1 Sol", "Р"],
    ["Советники", "редко, только на развилках", "Fable 5.1 и GPT-6 Astra", "ИР"],
  ];
  roles.forEach(([head, body, model, marks], i) => {
    const x = rx + (i % 2) * (rw + rgap), y = y0 + Math.floor(i / 2) * (rh + rgap);
    card(s, x, y, rw, rh, `role-${i + 1}-card`);
    text(s, head, { x: x + 0.2, y: y + 0.12, w: rw - 0.4, h: 0.3, bold: true, objectName: `role-${i + 1}-head` });
    text(s, body, { x: x + 0.2, y: y + 0.43, w: rw - 0.35, h: 0.56, color: C.text2, objectName: `role-${i + 1}-body` });
    [...marks].forEach((m, k) => logo(s, m, x + 0.2 + k * 0.25, y + rh - 0.28, 0.19, `role-${i + 1}-logo-${k + 1}`));
    const mx = x + 0.2 + marks.length * 0.25 + 0.03;
    text(s, model, { x: mx, y: y + rh - 0.29, w: x + rw - 0.2 - mx, h: 0.22, fontSize: 12, color: C.accent5, valign: "bottom", wrap: false, objectName: `role-${i + 1}-model` });
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

  // числа процесса
  const ty = sy + sh + 0.14, th = BOTTOM - ty, tgap = 0.15;
  const tiles = [[2.35, "2 сессии", "рабочих, за двое суток"], [1.9, "60+", "поручений агентам"], [2.9, "ни одна", "работа не прошла ревью с первого раза"]];
  let tx = X0;
  const value = (v, x, w, name) => text(s, v, { x, y: ty + 0.06, w, h: 0.46, fontSize: 28, bold: true, valign: "middle", objectName: name });
  const label = (v, x, w, name) => text(s, v, { x, y: ty + 0.53, w, h: th - 0.58, fontSize: 13, color: C.text2, objectName: name });
  tiles.forEach(([w, v, l], i) => {
    card(s, tx, ty, w, th, `num-${i + 1}-card`);
    value(v, tx + 0.22, w - 0.4, `num-${i + 1}-value`);
    label(l, tx + 0.22, w - 0.36, `num-${i + 1}-label`);
    tx += w + tgap;
  });
  const w4 = X0 + CW - tx, sub = [[1.25, String(N.commits), "правок кода (коммитов)"], [1.5, String(N.checks), "автоматические проверки"], [1.45, N.runs, "записанных прогонов"]];
  card(s, tx, ty, w4, th, "num-4-card");
  let qx = tx + 0.22;
  sub.forEach(([w, v, l], i) => {
    value(v, qx, w - 0.1, `num-4-${i + 1}-value`);
    label(l, qx, w - 0.1, `num-4-${i + 1}-label`);
    qx += w;
  });
}

// ---------- 8. ИИ как инфраструктура знаний ----------
{
  const s = content(SEC_A, "Инфраструктура знаний: лаборатория, а не один робот",
    "Что из сделанного переносится на другие задачи и как такие агенты встраиваются в науку и образование", {
      two: [
        "И главное. Остаётся не робот, а лаборатория: воспроизводимые опыты, журнал гипотез, независимая проверка и страница-объяснение.",
        "Так устроены роботы-учёные и автономные лаборатории в науке; студенту такой стенд даёт метод.",
        "Теперь покажем вживую.",
      ],
      one: [
        "Главное: остаётся не робот, а лаборатория — воспроизводимые опыты, журнал гипотез и независимая проверка.",
        "Это и есть ИИ как инфраструктура знаний. Показываем вживую.",
      ],
    }, 28);
  const lw = 6.05, g = 0.3, rx = X0 + lw + g, rw = CW - lw - g;
  text(s, "Что переносится на другие задачи", { x: X0, y: TOP, w: lw, h: 0.3, bold: true, objectName: "infra-head" });
  const rows = [
    ["Воспроизводимые опыты", `одна команда → серия → разность с интервалом; ${N.series} серия, ${N.runs} прогонов`],
    ["Журнал гипотез", "предположил, проверил, что вышло — и сверка со скрытой правдой"],
    ["Независимая проверка", "код читает другая модель; отрицательные результаты записаны как результаты"],
    ["Страница-объяснение", "те же результаты простым языком и с рисунками"],
  ];
  const sy = 6.14, ry = TOP + 0.4, rh = (sy - 0.14 - ry - 3 * 0.1) / 4;
  rows.forEach(([head, body], i) => {
    const y = ry + i * (rh + 0.1);
    card(s, X0, y, lw, rh, `infra-${i + 1}-card`);
    hexBadge(s, i + 1, X0 + 0.2, y + (rh - 0.44) / 2, 0.44, `infra-${i + 1}-badge`, C.text2);
    text(s, head, { x: X0 + 0.92, y: y + 0.07, w: lw - 1.1, h: 0.3, bold: true, valign: "middle", objectName: `infra-${i + 1}-head` });
    text(s, body, { x: X0 + 0.92, y: y + 0.37, w: lw - 1.1, h: rh - 0.4, fontSize: 13, color: C.text2, objectName: `infra-${i + 1}-body` });
  });
  text(s, "Куда такие агенты встраиваются", { x: rx, y: TOP, w: rw, h: 0.3, bold: true, objectName: "embed-head" });
  const ch = (sy - 0.14 - ry - 0.12) / 2;
  card(s, rx, ry, rw, ch, "science-card", C.text2);
  text(s, "В науке", { x: rx + 0.25, y: ry + 0.12, w: rw - 0.5, h: 0.3, bold: true, color: C.accent1, objectName: "science-head" });
  text(s, "Тот же замкнутый цикл «гипотеза → опыт → данные → вывод», что у робота-учёного Adam [1, 2], автономной лаборатории A-Lab [3] и The AI Scientist [4]. Агент берёт на себя рутину опыта и оставляет проверяемый след.",
    { x: rx + 0.25, y: ry + 0.46, w: rw - 0.5, h: ch - 0.52, fontSize: 14, color: C.background1, objectName: "science-body" });
  const ey = ry + ch + 0.12;
  card(s, rx, ey, rw, ch, "edu-card");
  text(s, "В образовании", { x: rx + 0.25, y: ey + 0.12, w: rw - 0.5, h: 0.3, bold: true, objectName: "edu-head" });
  text(s, `Студент меняет гипотезу или правило агента и получает серию опытов с интервалом: ${E1_WALL.runs} прогонов считаются за ${E1_WALL.seconds} секунд на одном компьютере. Учится методу, а не только коду.`,
    { x: rx + 0.25, y: ey + 0.46, w: rw - 0.5, h: ch - 0.52, fontSize: 14, color: C.text2, objectName: "edu-body" });
  // Источники — только названные проверенными в research/notes/K4.md, блок «Проверка ведущего».
  text(s, [
    { text: "Источники: ", options: { bold: true } },
    { text: "[1] King et al., Nature, 2004 · [2] King et al., Science, 2009 · [3] Szymanski et al., Nature, 2023 · [4] Lu et al., arXiv:2408.06292, 2024. Об ИИ в науке: OECD, Artificial Intelligence in Science, 2023; The Royal Society, Science in the Age of AI, 2024." },
  ], { x: X0, y: sy, w: CW, h: BOTTOM - sy, fontSize: 12, color: C.accent5, valign: "middle", objectName: "sources" });
}

// ================= приложение =================
pres.addSection({ title: SEC_B });

// ---------- П1. ход исследований (из build_ai_process.js) ----------
{
  const s = content(SEC_B, "Приложение. Как прошли исследования",
    "Каждая строка — одно исследование, каждый значок — один шаг: какая модель работала и чем закончилось", {
      ask: [
        "Каждая строка — одно исследование, каждый значок — шаг: обзор исследователя, работа инженера или независимая проверка. Галочка — работа принята.",
        `В первых двух исследованиях проверка нашла ${REVIEW.found} ошибок, и два вывода пришлось перевернуть. В обзорах литературы нашлись две выдуманные ссылки.`,
        "Строки без галочки — работа не принята: это видно на слайде.",
      ],
    });
  const headH = 0.3, y1 = TOP + headH + 0.14, yEnd = 5.99;
  const legend = [["Г", "Gemini 3.8 Flash — исследователь"], ["И", "Claude Opus 5.5 — инженер"], ["Р", "GPT-6.1 Sol — ревьюер"], ["✓", "работа принята"]];
  let gx = X0;
  legend.forEach(([letter, word], i) => {
    const w = word.length * 0.094 + 0.04;
    chip(s, letter, gx, TOP + 0.01, `legend-${i + 1}-chip`);
    text(s, word, { x: gx + CHIP_S + 0.08, y: TOP, w, h: headH, fontSize: 13, color: C.text2, valign: "middle", wrap: false, objectName: `legend-${i + 1}-word` });
    gx += CHIP_S + 0.08 + w + 0.22;
  });
  const nameW = 3.95, chipsX = X0 + nameW + 0.1, chipsW = 10 * (CHIP_S + CHIP_GAP), resX = chipsX + chipsW + 0.15, resW = X0 + CW - resX;
  const lh = (yEnd - y1) / LANES.length;
  LANES.forEach(([name, chain, result], i) => {
    const y = y1 + i * lh, cy = y + (lh - CHIP_S) / 2;
    s.addShape(pres.shapes.LINE, { x: X0, y, w: CW, h: 0, line: { color: C.background2, width: 1 }, objectName: `lane-${i + 1}-rule` });
    text(s, name, { x: X0, y, w: nameW, h: lh, bold: true, valign: "middle", wrap: false, objectName: `lane-${i + 1}-name` });
    [...chain].forEach((letter, k) => {
      const x = chipsX + k * (CHIP_S + CHIP_GAP);
      if (letter === "→") text(s, "→", { x, y, w: CHIP_S, h: lh, fontSize: 14, bold: true, color: C.accent5, align: "center", valign: "middle", objectName: `lane-${i + 1}-next` });
      else chip(s, letter, x, cy, `lane-${i + 1}-chip-${k + 1}`);
    });
    text(s, result, { x: resX, y, w: resW, h: lh, fontSize: 14, color: C.text2, valign: "middle", wrap: false, objectName: `lane-${i + 1}-result` });
  });
  const cy = 6.1;
  card(s, X0, cy, CW, BOTTOM - cy, "conclusion-card", C.text2);
  text(s, [
    { text: "Вывод. ", options: { bold: true, color: C.accent1 } },
    { text: "Качество держится на устройстве процесса — отдельная ветка, независимое ревью, пересчёт чисел, — а не на вере в один ответ модели." },
  ], { x: X0 + 0.3, y: cy, w: CW - 0.6, h: BOTTOM - cy, color: C.background1, valign: "middle", objectName: "conclusion-text" });
}

// ---------- П2. уровни 0–4 (docs/requirements_check.html, раздел 3) ----------
{
  const s = content(SEC_B, "Приложение. Уровни 0–4: адаптация — с оговоркой",
    "Сверка с условием задачи: что есть на каждом уровне и что нужно сказать честно", {
      ask: ["Уровни с нулевого по третий выполнены. На четвёртом механизмы адаптации работают, но польза в очках доказана не для всех: смена грунта выигрыша не дала."],
    }, 28);
  table(s, [[2.7, "Уровень"], [1.05, ""], [4.8, "Что есть"], [3.58, "Оговорка"]], [
    ["0. Запуск среды", MARK.ok, "Одна команда поднимает мир, робота и судью; есть ручное управление и проверка, что данные идут", "—"],
    ["1. Навигация", MARK.ok, "Путь по сетке карты с ценой клеток, запас 17 см от стен; команды скорости — как в условии", "Nav2 не используется: условие это разрешает"],
    ["2. Планировщик на языковой модели", MARK.ok, "Модель по тексту миссии и сводке состояния возвращает подцели; исполняет их программа", "Пока модель думает, робот стоит"],
    ["3. Научный цикл", MARK.ok, "Журнал гипотез; поиск по карте вероятностей; цена грунта — по расходу заряда на метр", "Гипотезы выдвигает программа, а не модель"],
    ["4. Адаптация", MARK.part, "Робот распознаёт перерасход заряда, штрафы и шум датчика и перестраивает маршрут", "Смена грунта очков не даёт"],
  ], TOP, BOTTOM, "levels", { fontSize: 15 });
}

// ---------- П3. критерии (research/task_statement.txt, строки 112–134; docs/requirements_check.html, раздел 1) ----------
{
  const s = content(SEC_B, "Приложение. Критерии оценки: чем закрыт каждый",
    "Веса — из условия задачи; в правом столбце — слабое место, о котором мы знаем", {
      ask: ["По каждому критерию — чем он закрыт и где слабое место. Самые слабые места: адаптация к смене грунта и то, что гипотезы выдвигает программа, а не языковая модель."],
    });
  table(s, [[3.3, "Критерий и вес"], [5.05, "Чем закрыт"], [3.78, "Слабое место"]], [
    ["Работоспособность агента · 30%", `Трудный уровень: 81,3 очка при потолке 92,5 (быстрый симулятор); в Gazebo возврат ${GZ.after}`, "На перегруженной машине робот в Gazebo бьётся о столбы"],
    ["Адаптивность · 25%", "Генератор сценариев; робот сам замечает сбой датчика (+6,4), новые зоны (+3,5), потерю положения", "Реакция на смену грунта очков не даёт"],
    ["Планирование на языковой модели · 15%", `QWEN и DeepSeek с сервера МАИ; проверка ответа и запасное правило; миссия словами ${LLM.mission}`, "Обычной миссии модель ничего не добавляет; ответ идёт 15 с"],
    ["Научный подход · 15%", `Журнал гипотез сверен со скрытой правдой: верны ${JOURNAL.confirmed} подтверждённых`, "Гипотезы выдвигает программа, а не модель"],
    ["Ассистент и процесс · 5%", `${N.checks} автоматические проверки; показ и серии опытов — одной командой`, "Скиллы ассистента в общий репозиторий не выложены"],
    ["Презентация и «ИИ как инфраструктура знаний» · 10%", "Эта презентация, страница-объяснение и страница опытов", "Связь с наукой — по литературе, не по внедрению"],
  ], TOP, BOTTOM, "criteria");
}

// ---------- П4. опыты с числами и интервалами ----------
{
  const s = content(SEC_B, "Приложение. Опыты: числа и интервалы",
    "В скобках — 95% интервал разности на одних сценариях. Всё, кроме строки Gazebo, — быстрый симулятор", {
      ask: ["Все сравнения — на одинаковых сценариях, не использованных при настройке. Если интервал включает ноль, мы говорим «различие не показано»."],
    });
  const no = { color: C.accent5 }, yes = { color: C.accent3, bold: true };
  table(s, [[3.3, "Что сравнивали"], [2.55, "Где измерено"], [3.6, "Число и интервал"], [2.68, "Вывод"]], [
    ["Адаптация против плана заранее", "трудный, 40 сценариев", "72,7 против 59,3: +13,3 [+7,4; +19,0]", { text: "подтверждено", ...yes }],
    ["Пережидание сбоя датчика", "трудный, 40 сценариев", "74,9 → 81,3: +6,4 [+2,9; +9,8]", { text: "подтверждено", ...yes }],
    ["Память об опасных зонах", "трудный, три новые зоны", "+3,5 [+0,4; +7,1]", { text: "только при трёх зонах", ...yes }],
    ["Новые схемы планирования", "трудный, 160 сценариев", `${String(SCORE.tourA1).replace(".", ",")} против ${String(SCORE.adaptiveA1).replace(".", ",")}: −0,4 [−2,7; +1,9]`, { text: "различие не показано", ...no }],
    ["Реакция на смену грунта", "40 сценариев", "+0,02 [−0,03; +0,07]", { text: "выигрыш не показан", ...no }],
    ["Потолок: агент знает всё заранее", "трудный, 160 сценариев", `92,5; знает только места образцов — ${String(SCORE.samplesOnly).replace(".", ",")}`, "знание важнее порядка обхода"],
    ["Gazebo против быстрого", "Gazebo, 9 пар трудных", `${GZ.gzScore} против ${GZ.fastScore}: ${GZ.pair}`, { text: "различие не показано", ...no }],
    [`Ожидание ответа модели ${LLM.wait} с`, "трудный, 20 сценариев", `${LLM.from} → ${LLM.to}: −73,4 ${LLM.costCi}`, { text: "ждать на месте нельзя", color: C.accent6, bold: true }],
    ["Два робота: со связью и без", `${TEAM2.n} сценариев`, `прогон ${TEAM2.dt}`, { text: "подтверждено", ...yes }],
    ["То же: последний образец", `${TEAM2.n} сценариев`, TEAM2.last, { text: "различие не показано", ...no }],
  ], TOP, BOTTOM, "exp", { fontSize: 13 });
}

// ---------- П5. ограничения ----------
{
  const s = content(SEC_B, "Приложение. Ограничения и что не сделано",
    "Что мы знаем о слабых местах и о чём нельзя говорить сильнее, чем показывают данные", {
      ask: ["Серии считаются в быстром симуляторе; в Gazebo прогонов единицы — это проверка работоспособности. Модель на показе в записи. Гипотезы выдвигает программа."],
    });
  const w = (CW - 0.3) / 2, h = BOTTOM - TOP;
  const cols = [
    ["Измерения", C.background2, C.text1, C.text2, [
      "Серии опытов — в быстром симуляторе. В Gazebo — единицы прогонов (7, 5, 9 пар): это проверка работоспособности",
      "Миссии словами: 6 сценариев по одному прогону — счёт прогонов, а не доля",
      "Расход заряда и закон датчика — наши допущения: при другом законе датчика робот собирает намного меньше",
      "SLAM: счёт миссии по своей карте с готовой не сравнивался; худший столб ушёл на 9,5 см при пороге 10",
      "Два робота: в Gazebo один прогон; счёт команды со счётом одного не сравнивается",
    ]],
    ["Не сделано", C.text2, C.accent1, C.background1, [
      "Модель вживую на показе: пока она думает, робот стоит. Режим «ехать, пока модель думает» — в работе, не принят",
      "Гипотезы выдвигает и проверяет программа. Гипотезы от языковой модели — в работе",
      "Реакция на смену грунта очков не даёт; исправление оставлено выключенным",
      "Грунты — данные судьи: заряд уходит быстрее, но сам робот на них не замедляется",
      "До потолка остаётся 11 очков: следующие потери разбираются",
    ]],
  ];
  cols.forEach(([head, fill, headColor, color, items], i) => {
    const x = X0 + i * (w + 0.3);
    card(s, x, TOP, w, h, `limits-${i + 1}-card`, fill);
    text(s, head, { x: x + 0.3, y: TOP + 0.18, w: w - 0.6, h: 0.34, fontSize: 18, bold: true, color: headColor, objectName: `limits-${i + 1}-head` });
    bullets(s, items, { x: x + 0.3, y: TOP + 0.66, w: w - 0.55, h: h - 0.8, color, paraSpaceAfter: 8, objectName: `limits-${i + 1}-list` });
  });
}

// ---------- П6. архитектура подробнее (как в build_checkpoint2.js) ----------
{
  const s = content(SEC_B, "Приложение. Архитектура: каналы между стендом и агентом",
    "Имена и типы каналов — как в условии задачи; агент не видит ничего, кроме них", {
      ask: ["Слева стенд: генератор сценариев, судья и симулятор. Справа агент. Между ними только каналы из условия, поэтому нашего судью можно заменить судьёй организаторов, не трогая агента."],
    }, 28);
  const h = 3.3;
  architecture(s, TOP, h, true);
  const by = TOP + h + 0.32, bh = BOTTOM - by, bw = 3.7, bgap = (CW - 3 * bw) / 2;
  const chain = [
    ["Запись прогона", "путь, заряд, решения и гипотезы агента"],
    ["Серии опытов", `${N.series} серия на одинаковых сценариях; разности с интервалами`],
    ["Веб-лаборатория и пульт", "графики, проигрыватель прогонов, управление показом"],
  ];
  chain.forEach(([head, sub], i) => {
    const x = X0 + i * (bw + bgap);
    card(s, x, by, bw, bh, `chain-${i + 1}-box`);
    text(s, head, { x: x + 0.22, y: by + 0.12, w: bw - 0.44, h: 0.32, bold: true, objectName: `chain-${i + 1}-head` });
    text(s, sub, { x: x + 0.22, y: by + 0.48, w: bw - 0.44, h: bh - 0.55, fontSize: 14, color: C.text2, objectName: `chain-${i + 1}-sub` });
    if (i < 2) arrow(s, x + bw + 0.1, by + bh / 2, x + bw + bgap - 0.1, `chain-${i + 1}-arrow`);
  });
  arrowDown(s, X0 + bw / 2, TOP + h + 0.05, by - 0.05, "record-arrow");
}

// ---------- текст выступления ----------
function writeSpeech() {
  const WPS = 2.2;                                   // слов в секунду при спокойной речи
  const words = (t) => t.split(/\s+/).filter((w) => /[A-Za-zА-Яа-яЁё0-9]/.test(w)).length;
  const secs = (lines) => Math.round(words(lines.join(" ")) / WPS);
  const main = SPEECH_PARTS.filter((p) => p.two);
  const total = (key) => main.reduce((a, p) => a + secs(p[key]), 0);
  const variant = (key) => main.flatMap((p, i) => {
    const head = `### Слайд ${i + 1}. ${p.title}`;
    return p[key].length ? [`${head} (≈ ${secs(p[key])} с)`, "", ...p[key], ""] : [`${head} — пролистать молча`, ""];
  });
  const lines = [
    "# Финал DID Hack: текст выступления",
    "",
    "Формат из условия: 5 минут демо и 2 минуты вопросов. Большая часть времени — живой показ, слайды его обрамляют.",
    "Время посчитано по числу слов (2,2 слова в секунду). Текст слайдов пересобирается вместе с презентацией",
    "(`PYTHON=/usr/local/bin/python3 node build_final.js`); править его нужно в скрипте, а не здесь.",
    "Вариант на 2 минуты лежит и в заметках к слайдам. Слайды 9–14 — приложение, их открывают только по вопросам.",
    "",
    "Стенд поднят заранее (`pixi run demo --open`, готовность через 45 с), перед выходом нажато «Сбросить»,",
    "тяжёлые расчёты на машине остановлены. Окна: Gazebo слева, пульт справа.",
    "",
    `## Вариант А. 2 минуты слайды + 3 минуты показ (слайды ≈ ${total("two")} с)`,
    "",
    ...variant("two"),
    "### Показ, 3 минуты (180 с)",
    "",
    "Времена шагов — замер из `docs/demo_script.md` (Gazebo, средний уровень, сценарий 3): объезд 53 с, миссия 96–100 с.",
    "Шаги «Цель» и «Домой» в этом варианте пропускаются. Запаса почти нет: если слайды заняли больше двух минут,",
    "ставьте маршрут из трёх-четырёх точек вместо шести (время такого объезда не замерено — проверьте на репетиции).",
    "",
    "| Время | Что нажать | Что говорить |",
    "|---|---|---|",
    "| 0:00–0:15 | ничего; затем 5–6 щелчков по карте | «Одна команда подняла мир, робота, судью и пульт. Робот ещё стоит, а лидар уже нарисовал треть арены. Ставлю точки — пульт строит путь в обход столбов». |",
    "| 0:15–1:10 | «Ехать» | Пока робот едет (53 с): «Карта достраивается на глазах. Положение — колёса с поправкой по лидару. Готовую карту можно не давать совсем: через SLAM робот строит её сам, и она совпадает с готовой на 96–99 процентов». В конце показать «Совпадение с эталоном 98 %». |",
    "| 1:10–1:20 | «Запустить миссию» (агент «С адаптацией») | «А теперь без оператора. Образцов робот не видит: у него одно число датчика — насколько близко образец». |",
    "| 1:20–2:45 | ничего, показывать на журнал справа | Пока робот едет (около 100 с): «Зелёные пятна — где робот ждёт образец. Справа его журнал: гипотеза, проверка, вывод. Подъехал, собрал — гипотеза подтверждена. На трудном уровне к этому добавляются сбой датчика и новые опасные зоны; там робот набирает 81 очко при потолке 92 с половиной — это серии в быстром симуляторе с тем же агентом и тем же судьёй. Каждую работу агентов-разработчиков проверяла независимая модель: в первых исследованиях она нашла семь ошибок». |",
    "| 2:45–3:00 | «Показать скрытую правду» | «Собрал пять из пяти, вернулся на базу, счёт судьи 72. Вот что было скрыто. Остаётся не один робот, а лаборатория, в которой такой опыт может поставить и повторить любой. Спасибо». |",
    "",
    `## Вариант Б. 1 минута слайды + 4 минуты показ (слайды ≈ ${total("one")} с)`,
    "",
    ...variant("one"),
    "### Показ, 4 минуты (240 с)",
    "",
    "Полный сценарий из `docs/demo_script.md`: объезд 53 с, цель 16 с, домой 22 с, миссия 96–100 с — вместе с щелчками",
    "около 215 с, запас 25 с. То, что не сказано на пролистанных слайдах, говорится здесь, пока робот едет.",
    "",
    "| Время | Что нажать | Что говорить |",
    "|---|---|---|",
    "| 0:00–0:15 | ничего; затем 5–6 щелчков по карте | «Одна команда подняла мир, робота, судью и пульт. Робот стоит, а лидар уже нарисовал треть арены. Ставлю точки — пульт строит путь в обход столбов». |",
    "| 0:15–1:10 | «Ехать» | Пока робот едет (53 с): «Устроено так: стенд с судьёй и агент общаются только через каналы из условия. Тот же агент ездит в быстром симуляторе, где мы ставим серии опытов. Готовую карту можно не давать: через SLAM робот строит её сам, совпадение 96–99 процентов, начало координат совпадает с мировым с точностью около пяти сантиметров». |",
    "| 1:10–1:30 | щелчок в дальний угол, «Ехать» | «Одна точка назначения — приехал с точностью до сантиметров». |",
    "| 1:30–1:55 | «Домой» | «Одна кнопка — и робот на базе. Бонусом у нас два робота: с координацией они заканчивают на 26 секунд раньше пары без связи — это быстрый симулятор, 80 сценариев». |",
    "| 1:55–2:05 | «Запустить миссию» | «Теперь без оператора. Образцов робот не видит: у него одно число датчика». |",
    "| 2:05–3:40 | ничего, показывать на журнал | Пока робот едет (около 100 с): «Справа журнал: гипотеза, проверка, вывод. Девять из десяти гипотез, которые робот подтвердил, верны. Об изменениях среды ему никто не сообщает: сбой датчика он пережидает — это плюс шесть очков, — новые опасные зоны запоминает. Языковая модель читает миссию, заданную словами: шесть миссий из шести против нуля у правила. Но отвечает она пятнадцать секунд, поэтому сейчас решает правило, а модель мы показываем в записи. Работали мы как лаборатория: агенты пишут код и ставят опыты, каждую работу проверяет независимая модель, и ни одна не прошла проверку с первого раза». |",
    "| 3:40–4:00 | «Показать скрытую правду» | «Пять из пяти, робот на базе, счёт судьи 72. Вот что было скрыто. Остаётся не один робот, а лаборатория: опыт может поставить и повторить любой. Спасибо». |",
    "",
    "## Если показ сорвался",
    "",
    "- Робот стоит: ждать не больше 30 секунд (он обязан поехать сам или найти себя), говорить про журнал. Не поехал — «Завершить миссию», «Сбросить», миссия заново.",
    "- Gazebo завис: на пульте «Перейти на быстрый симулятор» — та же страница, тот же агент и судья; сказать об этом вслух.",
    "- Совсем запасной вариант: записи `demo/pilot_gazebo_tour.mp4` и `demo/pilot_fastsim.mp4`.",
    "- Бонусные показы (SLAM — `pixi run demo --slam-map --open`, два робота — запись в лаборатории) в пять минут не помещаются: только по вопросу жюри.",
    "",
    "## Слайды приложения (открывать по вопросам)",
    "",
  ];
  SPEECH_PARTS.filter((p) => p.ask).forEach((p, i) => lines.push(`- **Слайд ${main.length + i + 1}. ${p.title.replace("Приложение. ", "")}.** ${p.ask.join(" ")}`));
  lines.push("",
    "## Вопросы жюри и ответы",
    "",
    "1. **Вы сами читали код?** Нет. Код каждой работы до приёма читала независимая модель-проверяющий, и ни одна работа не прошла проверку с первого раза. Человек задавал цель, роли и ограничения и принимал решения на развилках.",
    `2. **Что нашла проверка?** В коде первых двух исследований — шесть ошибок подсчёта и одну ошибку самого робота: разовую потерю заряда от штрафа он принимал за дорогой грунт. Два вывода оказались неверны и исправлены. В обзорах литературы нашлись две выдуманные ссылки.`,
    "3. **Числа из Gazebo или из своего симулятора?** Серии с интервалами — из быстрого симулятора: там тот же агент и тот же судья. В Gazebo — проверка: на девяти парах одинаковых сценариев счёт 76,2 против 75,9, разность +0,3 с интервалом от −10,4 до +11,1 — различие не показано, но девять пар мало. После защиты от потери положения в Gazebo 7 возвратов из 7, с пережиданием сбоя — 5 из 5, счёт 75–91.",
    "4. **Откуда потолок 92,5?** Это робот-подсказчик, которому заранее сообщили всё скрытое: места образцов, грунт, зоны и события. 160 трудных сценариев, быстрый симулятор. Наш робот без подсказок набирает на этих же сценариях 73,3, а на сорока сценариях опыта с пережиданием сбоя — 81,3.",
    "5. **Гипотезы выдвигает языковая модель?** Нет. Гипотезы выдвигает и проверяет программа. Модель читает миссию и предлагает подцели. Гипотезы от модели — в работе, результата пока нет.",
    `6. **Зачем тогда языковая модель?** На обычной миссии она в ${LLM.same} процентах решений выбирает то же, что простое правило, и счёт одинаков: 71,3 и 71,2. Польза там, где задание меняют словами: миссию «собери ровно два образца и возвращайся» робот с моделью выполнил в 6 прогонах из 6, с правилом — ни разу. Это шесть сценариев по одному прогону, счёт прогонов, а не доля.`,
    `7. **Почему модель не работает на показе вживую?** Ответ идёт около 15 секунд, и всё это время робот стоит. На трудном уровне такое ожидание стоит 73 очка: робот стоит в опасной зоне и получает штраф за штрафом. Поэтому на показе решает правило, а модель — в записи. Режим, где робот едет, пока модель думает, — в работе и ещё не принят.`,
    "8. **Что не получилось?** Реакция на смену грунта очков не дала: +0,02 при интервале от −0,03 до +0,07. Новые схемы планирования не лучше нынешней: −0,4 при интервале от −2,7 до +1,9. Оба результата записаны как отрицательные, исправление для грунта оставлено выключенным.",
    `9. **SLAM: начало координат совпадает с мировым?** С точностью около 5 сантиметров: сдвиг 1,5–5,2 см в пяти прогонах в Gazebo, поворот до 1,3 градуса. Все девять столбов найдены, худший ушёл на 9,5 см при пороге 10. Счёт миссии по своей карте с ездой по готовой мы не сравнивали.`,
    `10. **Два робота набирают больше одного?** Так сравнивать нельзя: у команды двойной заряд и два бонуса за возврат. Честное сравнение — с парой без связи: прогон ${TEAM2.time[0]} → ${TEAM2.time[1]} с, заряд ${TEAM2.charge[0]} → ${TEAM2.charge[1]} единицы, оба вернулись в ${TEAM2.home[0]} прогонах против ${TEAM2.home[1]}. Последний образец быстрее не берётся: ${TEAM2.last}, различие не показано. Измерено в быстром симуляторе; в Gazebo — один прогон как проверка.`,
    "11. **Что будет с судьёй организаторов?** Каналы и типы сообщений — как в условии, судья заменяется без правки агента. Но расход заряда и закон датчика — наши допущения. Мы проверяли другие правила: к другому расходу робот устойчив умеренно, а при другой форме закона датчика собирает намного меньше. Самокалибровка датчика — в работе, не принята.",
    "12. **При чём здесь «ИИ как инфраструктура знаний»?** Переносится не робот, а устройство работы: опыт запускается одной командой и повторяется, гипотезы и решения записаны и сверены с правдой, код проверяет независимая модель, результаты объяснены простым языком. Тот же замкнутый цикл — у роботов-учёных (King, 2004 и 2009) и автономных лабораторий (Szymanski, 2023). Студенту такой стенд даёт серию опытов с интервалом за секунды.",
    "13. **Где логи разработки?** План исследований — `research/agenda.yaml`, правила — `research/PROTOCOL.md`, отчёты с числами и замечаниями проверки — `research/findings/`, история правок — в git.",
    "");
  fs.writeFileSync(SPEECH, lines.join("\n"));
  console.log(`доклад: ${SPEECH} (слайды: вариант А ≈ ${total("two")} с, вариант Б ≈ ${total("one")} с)`);
}

// ---------- PDF и картинки слайдов (как в build_ai_process.js) ----------
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
  console.log(`картинки: ${SLIDES} (слайдов: ${files.length})`);
  // Общий лист: все слайды на одной картинке, по четыре в ряд.
  const sheet = `
import sys
from PIL import Image
ims = [Image.open(f) for f in sys.argv[2:]]
w, h, gap, cols = 640, 360, 8, 4
rows = (len(ims) + cols - 1) // cols
out = Image.new('RGB', (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), '#9A9A9A')
for i, im in enumerate(ims):
    out.paste(im.resize((w, h), Image.LANCZOS), (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
out.save(sys.argv[1], quality=88)
`;
  const q = spawnSync(PY, ["-c", sheet, PREVIEW, ...files], { encoding: "utf8" });
  if (q.status === 0) console.log("общий лист:", PREVIEW);
  else console.warn("общий лист не собран (нужен Python с Pillow)");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  writeSpeech();
  console.log("готово:", OUT);
  if (!ARGS.has("--no-pdf")) render();
})();
