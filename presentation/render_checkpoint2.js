// PDF и картинки слайдов из checkpoint2.pptx:  node render_checkpoint2.js
//
// Нужен LibreOffice (soffice; на Mac: brew install --cask libreoffice) и pdftoppm (poppler).
// Пишет checkpoint2.pdf, slides_checkpoint2/slide-NN.jpg и общий лист checkpoint2_preview.jpg.
// Если LibreOffice нет, ничего не трогает и сообщает об этом: старый PDF остаётся как был.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const DIR = __dirname;
const PPTX = path.join(DIR, "checkpoint2.pptx");
const PDF = path.join(DIR, "checkpoint2.pdf");
const SLIDES = path.join(DIR, "slides_checkpoint2");

function find(names) {
  for (const n of names) {
    if (n.includes("/") ? fs.existsSync(n) : spawnSync("which", [n]).status === 0) return n;
  }
  return null;
}
const soffice = find(["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice", "libreoffice"]);
const pdftoppm = find(["pdftoppm", "/opt/homebrew/bin/pdftoppm"]);
if (!soffice) {
  console.warn("LibreOffice не найден: PDF и картинки не обновлены (brew install --cask libreoffice).");
  process.exit(0);
}

// Свой профиль: без окон первого запуска и без чужих настроек.
const profile = path.join(os.tmpdir(), "did_lo_profile");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "did_pdf_"));
const r = spawnSync(soffice, [`-env:UserInstallation=file://${profile}`, "--headless", "--convert-to", "pdf", "--outdir", tmp, PPTX],
  { encoding: "utf8", timeout: 180000 });
const made = path.join(tmp, "checkpoint2.pdf");
if (r.status !== 0 || !fs.existsSync(made)) {
  console.error("LibreOffice не собрал PDF:\n" + (r.stderr || r.stdout || r.error));
  process.exit(1);
}
fs.copyFileSync(made, PDF);
console.log("PDF:", PDF);

if (pdftoppm) {
  fs.rmSync(SLIDES, { recursive: true, force: true });
  fs.mkdirSync(SLIDES);
  const p = spawnSync(pdftoppm, ["-jpeg", "-r", "110", PDF, path.join(SLIDES, "slide")], { encoding: "utf8" });
  if (p.status !== 0) { console.error(p.stderr); process.exit(1); }
  const files = fs.readdirSync(SLIDES).filter((f) => f.endsWith(".jpg")).sort();
  console.log(`картинки: ${SLIDES} (${files.length} слайдов)`);
  // Общий лист, как checkpoint1_preview.jpg: все слайды на одной картинке.
  const sheet = `
import sys
from PIL import Image
files = sys.argv[2:]
ims = [Image.open(f) for f in files]
cols = 3
w, h = 640, 360
rows = (len(ims) + cols - 1) // cols
gap = 8
out = Image.new('RGB', (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), '#9A9A9A')
for i, im in enumerate(ims):
    out.paste(im.resize((w, h), Image.LANCZOS), (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
out.save(sys.argv[1], quality=88)
`;
  const s = spawnSync("python3", ["-c", sheet, path.join(DIR, "checkpoint2_preview.jpg"), ...files.map((f) => path.join(SLIDES, f))], { encoding: "utf8" });
  if (s.status === 0) console.log("общий лист:", path.join(DIR, "checkpoint2_preview.jpg"));
  else console.warn("общий лист не собран (нужен Python с Pillow)");
} else {
  console.warn("pdftoppm не найден: картинки слайдов не обновлены (brew install poppler).");
}
fs.rmSync(tmp, { recursive: true, force: true });
