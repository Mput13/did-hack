// Один слайд «Реальные результаты: шесть опытов» — в оформлении презентации второго чекпоинта.
//
//   pixi run python presentation/figures/results_numbers.py     # числа из сводок опытов → data/results.json
//   cd presentation && node build_results_slide.js              # results_slide.pptx, текст рассказа, PDF и картинка
//
// Ключ --no-pdf — только pptx и текст. Для PDF нужен LibreOffice (soffice), для картинки — pdftoppm.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

// pptxgenjs и jszip должны находиться и из сценария оформления (apply_theme.js), который лежит вне проекта.
process.env.NODE_PATH = [path.join(__dirname, "node_modules")].concat(process.env.NODE_PATH || []).join(path.delimiter);
require("module").Module._initPaths();
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/Users/a/.claude/skills/synced/0ec766ba-a970-423d-9985-0a1698a18ad8_57a5e5ff-4dfe-4d5a-9b46-49c5ea27665a/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ARGS = new Set(process.argv.slice(2));
const NAME = "results_slide";
const OUT = path.join(__dirname, `${NAME}.pptx`);
const PDF = path.join(__dirname, `${NAME}.pdf`);
const PREVIEW = path.join(__dirname, `${NAME}_preview.jpg`);
const SPEECH = path.join(__dirname, `${NAME}_speech.md`);
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "results.json"), "utf8"));

const ru = (v, d = 1) => Number(v).toFixed(d).replace(".", ",").replace("-", "−");
const plus = (v, d = 1) => (v >= 0 ? "+" : "") + ru(v, d);
const pct = (v, d = 0) => ru(v * 100, d) + "%";
const ci = (p, d = 1, k = 1) => `[${ru(p.ci[0] * k, d)}; ${ru(p.ci[1] * k, d)}]`;

// ---------- содержание: по строке на опыт ----------
// bars: [подпись, значение текстом, доля шкалы]; синий — с чем сравниваем, оранжевый — наш вариант.
const ROWS = [
  {
    name: "Агент против готового маршрута", sub: `${D.e1.runs} прогонов · счёт на трудном уровне`,
    bars: [["готовый маршрут", ru(D.e1.fixed), D.e1.fixed / 100], ["адаптивный агент", ru(D.e1.adaptive), D.e1.adaptive / 100]],
    big: `${plus(D.e1.diff.mean)} очка`, small: ci(D.e1.diff),
    say: `Собирает ${pct(D.e1.samples[1])} образцов вместо ${pct(D.e1.samples[0])}. Возврат на базу одинаковый — ${pct(D.e1.returned[1])}.`,
  },
  {
    name: "Как искать образцы", sub: `${D.e9.runs} прогонов · доля собранных образцов`,
    bars: [["подъём по сигналу", pct(D.e9.share.gradient), D.e9.share.gradient], ["карта вероятностей", pct(D.e9.share.adaptive, 1), D.e9.share.adaptive]],
    big: `${plus(D.e9.diff.mean * 100)} п. п.`, small: ci(D.e9.diff, 0, 100),
    say: `Карта вероятностей находит почти всё и тратит на ${ru(-D.e9.battery, 0)} единиц заряда меньше. Спираль — ${pct(D.e9.share.spiral, 1)}.`,
  },
  {
    name: "Сбой датчика: ехать или ждать", sub: `${D.p1.runs} прогонов · счёт на трудном уровне`,
    bars: [["едет по показаниям", ru(D.p1.old), D.p1.old / 100], ["пережидает сбой", ru(D.p1.new), D.p1.new / 100]],
    big: `${plus(D.p1.diff.mean)} очка`, small: ci(D.p1.diff),
    say: "Заметить сбой и постоять выгоднее, чем ехать по ложным показаниям.",
  },
  {
    name: "Миссия, заданная словами", sub: `3 задания с ограничением × ${D.r13.total / 3} сценариев`,
    bars: [["правило без модели", `${D.r13.rule} из ${D.r13.total}`, D.r13.rule / D.r13.total], ["языковая модель", `${D.r13.model_every} из ${D.r13.total}`, D.r13.model_every / D.r13.total]],
    big: `${D.r13.model_every} из ${D.r13.total}`, small: "заданий выполнено",
    say: "Меняем текст задания — меняется поведение. На обычной миссии модель счёт не повышает.",
  },
  {
    name: "Журнал гипотез и скрытая правда", sub: `${D.e17.runs} прогонов · верны ли выводы о грунте`,
    bars: [["адаптивный агент", pct(D.e17.soil.adaptive.share), D.e17.soil.adaptive.share], ["исследователь", pct(D.e17.soil.scientist.share), D.e17.soil.scientist.share]],
    big: pct(D.e17.soil.scientist.share), small: "гипотез о грунте верны",
    say: `Исследователь проверяет другие объяснения и ошибается реже. Счёт почти тот же: ${ru(D.e17.score.scientist)} и ${ru(D.e17.score.adaptive)}.`,
  },
  {
    name: "Перенос в Gazebo", sub: `${D.e7.pairs} сценариев в двух симуляторах · счёт`,
    bars: [["быстрый симулятор", ru(D.e7.fast), D.e7.fast / 100], ["Gazebo", ru(D.e7.gazebo), D.e7.gazebo / 100]],
    big: `${plus(D.e7.diff)} очка`, small: "средняя разность счёта",
    say: `В ${D.e7.close} сценариях из ${D.e7.pairs} счёт сходится до 6 очков. После правки в Gazebo — ${D.e7.after.returned} возвратов из ${D.e7.after.n}.`,
  },
];
const NOT_SHOWN = `Обнаружение смены грунта: ${plus(D.e2.change.mean)} ${ci(D.e2.change)}. Обучение цене грунта: ${plus(D.e2.soil.mean)} ${ci(D.e2.soil)}. ` +
  "Различие не показано.";

const NOTES = [
  "Здесь шесть опытов, по одному на каждый критерий. Везде агенты проходят одни и те же сценарии, и мы считаем разность по парам.",
  `Первый: робот, который перестраивает маршрут по измерениям, набирает на трудном уровне на ${ru(D.e1.diff.mean, 0)} очков больше, чем робот с готовым маршрутом.`,
  `Второй: из способов поиска, названных в условии, карта вероятностей собирает ${pct(D.e9.share.adaptive, 1)} образцов против ${pct(D.e9.share.gradient)} у подъёма по сигналу.`,
  `Третий: когда датчик образцов ломается, выгоднее постоять и переждать — это даёт ${plus(D.p1.diff.mean)} очка.`,
  `Четвёртый: задание словами. Правило без модели не выполнило ни одного задания с ограничением, с языковой моделью — ${D.r13.model_every} из ${D.r13.total}.`,
  `Пятый: журнал гипотез мы сверили со скрытой правдой среды. У исследователя подтверждённые гипотезы о грунте верны в ${pct(D.e17.soil.scientist.share)} случаев, у обычного агента — в ${pct(D.e17.soil.adaptive.share)}.`,
  `Шестой: те же сценарии в Gazebo дают в среднем тот же счёт, разность ${plus(D.e7.diff)} очка.`,
  "И то, что не подтвердилось: реакция на смену грунта очков не добавляет. Мы это измерили и записали как есть.",
];

// ---------- оформление (как в build_checkpoint2.js) ----------
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
pres.title = "ИИ-исследователь на роботе-платформе — реальные результаты";
const C = pres.SchemeColor;
const X0 = 0.6, CW = 12.13;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.42, w: CW, h: 0.75, fontSize: 32, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: X0, y: 1.2, w: CW, h: 0.42, fontSize: 16, color: C.accent5, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "ИИ-исследователь на роботе-платформе · DID Hack 2026", options: { x: X0, y: 7.0, w: 8.0, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, valign: "middle" } } },
  ],
});

const s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Реальные результаты: шесть опытов", { placeholder: "title" });
s.addText("Агенты проходят одни и те же сценарии; в скобках — 95% интервал разности", { placeholder: "body" });
s.addNotes(NOTES.join(" "));
const text = (str, o) => s.addText(str, { isTextBox: true, margin: 0, valign: "middle", align: "left", color: C.text1, fontSize: 12, ...o });

// Колонки: номер · опыт · что сравнивали (две полоски) · главное число · вывод.
const COL = { num: X0, name: 1.12, label: 4.62, track: 6.12, value: 7.02, big: 7.95, say: 9.95 };
const W = { name: 3.35, label: 1.45, track: 0.82, value: 0.8, big: 1.9, say: X0 + CW - 9.95 };
const TOP = 1.84, HEAD = 0.3, ROW = 0.685;

[["Опыт", COL.name], ["Что сравнивали", COL.label], ["Главное число", COL.big], ["Вывод", COL.say]].forEach(([label, x]) =>
  text(label, { x, y: TOP, w: 3, h: HEAD - 0.06, fontSize: 11, bold: true, color: C.accent5 }));
s.addShape(pres.shapes.LINE, { x: X0, y: TOP + HEAD, w: CW, h: 0, line: { color: C.accent5, width: 1 } });

ROWS.forEach((r, i) => {
  const y = TOP + HEAD + ROW * i;
  if (i) s.addShape(pres.shapes.LINE, { x: X0, y, w: CW, h: 0, line: { color: "D5D9DE", width: 0.75 } });
  s.addShape(pres.shapes.OVAL, { x: COL.num, y: y + 0.165, w: 0.34, h: 0.34, fill: { color: C.text2 }, objectName: `Номер ${i + 1}` });
  text(String(i + 1), { x: COL.num, y: y + 0.165, w: 0.34, h: 0.34, fontSize: 12, bold: true, color: C.background1, align: "center" });
  text(r.name, { x: COL.name, y: y + 0.09, w: W.name, h: 0.28, fontSize: 14, bold: true });
  text(r.sub, { x: COL.name, y: y + 0.37, w: W.name, h: 0.22, fontSize: 10.5, color: C.accent5 });
  r.bars.forEach(([label, value, share], k) => {
    const by = y + 0.1 + 0.25 * k;
    text(label, { x: COL.label, y: by, w: W.label, h: 0.22, fontSize: 10.5, color: C.accent5 });
    s.addShape(pres.shapes.RECTANGLE, { x: COL.track, y: by + 0.05, w: W.track, h: 0.12, fill: { color: C.background2 } });
    if (share > 0) {
      s.addShape(pres.shapes.RECTANGLE, { x: COL.track, y: by + 0.05, w: Math.max(0.03, W.track * Math.min(1, share)), h: 0.12, fill: { color: k ? C.accent1 : C.accent2 } });
    }
    text(value, { x: COL.value, y: by, w: W.value, h: 0.22, fontSize: 11, bold: true });
  });
  text(r.big, { x: COL.big, y: y + 0.06, w: W.big, h: 0.34, fontSize: 20, bold: true, color: C.accent1 });
  text(r.small, { x: COL.big, y: y + 0.4, w: W.big, h: 0.2, fontSize: 10.5, color: C.accent5 });
  text(r.say, { x: COL.say, y: y + 0.04, w: W.say, h: ROW - 0.08, fontSize: 11 });
});

const yb = TOP + HEAD + ROW * ROWS.length + 0.1;
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X0, y: yb, w: CW, h: 0.42, rectRadius: 0.06, fill: { color: C.text2 }, objectName: "Что не подтвердилось" });
text([{ text: "Что не подтвердилось. ", options: { bold: true, color: C.accent1 } }, { text: NOT_SHOWN, options: { color: C.background1 } }],
  { x: X0 + 0.2, y: yb, w: CW - 0.4, h: 0.42, fontSize: 12 });

// ---------- текст рассказа ----------
function writeSpeech() {
  const lines = ["# Реальные результаты: шесть опытов — текст рассказа", "",
    "Слайд: `presentation/results_slide.pptx`. Около минуты.", "", ...NOTES.map((n) => n + "\n"),
    "## Откуда числа", "",
    "Собираются командой `pixi run python presentation/figures/results_numbers.py` из сводок опытов — тех же, что на странице `docs/experiments.html`.", ""];
  fs.writeFileSync(SPEECH, lines.join("\n"));
}

// ---------- PDF и картинка слайда ----------
function render() {
  const find = (names) => names.find((n) => (n.includes("/") ? fs.existsSync(n) : spawnSync("which", [n]).status === 0)) || null;
  const soffice = find(["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice", "libreoffice"]);
  const pdftoppm = find(["pdftoppm", "/opt/homebrew/bin/pdftoppm"]);
  if (!soffice) return console.warn("LibreOffice не найден: PDF и картинка не обновлены (brew install --cask libreoffice).");
  const profile = path.join(os.tmpdir(), "did_lo_profile");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "did_pdf_"));
  const r = spawnSync(soffice, [`-env:UserInstallation=file://${profile}`, "--headless", "--convert-to", "pdf", "--outdir", tmp, OUT], { encoding: "utf8", timeout: 180000 });
  const made = path.join(tmp, `${NAME}.pdf`);
  if (r.status !== 0 || !fs.existsSync(made)) { console.error("LibreOffice не собрал PDF:\n" + (r.stderr || r.stdout || r.error)); process.exit(1); }
  fs.copyFileSync(made, PDF);
  console.log("PDF:", PDF);
  if (!pdftoppm) return console.warn("pdftoppm не найден: картинка слайда не обновлена (brew install poppler).");
  const p = spawnSync(pdftoppm, ["-jpeg", "-r", "144", "-singlefile", PDF, path.join(tmp, "slide")], { encoding: "utf8" });
  if (p.status !== 0) { console.error(p.stderr); process.exit(1); }
  fs.copyFileSync(path.join(tmp, "slide.jpg"), PREVIEW);
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log("картинка:", PREVIEW);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  writeSpeech();
  console.log("готово:", OUT);
  if (!ARGS.has("--no-pdf")) render();
})();
