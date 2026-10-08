// Графики лаборатории. Два вида:
//   stripChart — варианты агента по строкам: точки отдельных прогонов, среднее и 95% интервал;
//   diffChart  — разность на одинаковых сценариях относительно нуля, с интервалом и выводом словами.
// Цвет несёт только то, что он значит: вариант агента (постоянный цвет) или вывод (вместе со значком и словом).

import { h, s, tip, dot, badge, num, plural } from './views/common.js';

const FONT = '-apple-system, "Segoe UI", Roboto, Arial, sans-serif';

/** Круглые деления оси: {ticks, lo, hi, step}. */
export function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / Math.max(1, target);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
  const a = Math.floor(lo / step + 1e-9) * step;
  const b = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = a; v <= b + step * 1e-6; v += step) ticks.push(Number(v.toFixed(10)));
  return { ticks, lo: a, hi: b, step };
}

function domainFor(info, values) {
  if (info.kind === 'pct') return niceTicks(0, 1, 5);
  if (!values.length) return niceTicks(0, 1, 4);
  let lo = Math.min(...values);
  const hi = Math.max(...values);
  if (info.kind === 'count') {
    const top = Math.max(1, Math.ceil(hi));
    const t = niceTicks(0, top, Math.min(5, top));
    t.ticks = t.ticks.filter(Number.isInteger);
    return t;
  }
  if (lo >= 0 && lo < hi * 0.35) lo = 0;
  const pad = (hi - lo) * 0.04 || 1;
  return niceTicks(lo === 0 ? 0 : lo - pad, hi + pad, 5);
}

let measureCtx = null;
function textWidth(text, font) {
  if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d');
  measureCtx.font = font;
  return measureCtx.measureText(text).width;
}

function fit(text, font, maxW) {
  if (textWidth(text, font) <= maxW) return text;
  let t = text;
  while (t.length > 3 && textWidth(`${t}…`, font) > maxW) t = t.slice(0, -1);
  return `${t.trimEnd()}…`;
}

/** Подпись в одну или две строки: перенос по пробелу, вторая строка при нехватке места обрезается с многоточием. */
function wrap2(text, font, maxW) {
  if (textWidth(text, font) <= maxW) return [text];
  const words = String(text).split(/\s+/);
  let best = null;
  for (let i = 1; i < words.length; i++) {
    const a = words.slice(0, i).join(' ');
    const b = words.slice(i).join(' ');
    const wa = textWidth(a, font);
    if (wa > maxW) break;
    const score = Math.max(wa, textWidth(b, font));
    if (!best || score <= best.score) best = { a, b, score };
  }
  if (!best) return [fit(text, font, maxW)];
  return [best.a, fit(best.b, font, maxW)];
}

/**
 * Раскладка точек внутри строки. По горизонтали значение точное, по вертикали точки расступаются.
 * Прогоны с одинаковым значением (доли, штрафы, «вернулся / нет») собираются в плотную кучку вокруг
 * своего значения: иначе двадцать точек слились бы в одну.
 */
function dodge(points, r, half, lo, hi) {
  const step = r * 1.8;
  const maxK = Math.max(0, Math.floor(half / step));
  const offs = [0];
  for (let k = 1; k <= maxK; k++) offs.push(k * step, -k * step);
  const lim = 4 * r * r * 0.92;
  const placed = [];
  const hit = (x, y) => {
    let sum = 0;
    for (const q of placed) {
      const d2 = (x - q.x) ** 2 + (y - q.y) ** 2;
      if (d2 < lim) sum += lim - d2;
    }
    return sum;
  };
  const ties = new Map();
  for (const p of points) {
    const key = Math.round(p.px * 2);
    ties.set(key, (ties.get(key) || 0) + 1);
  }
  const order = points.map((_, i) => i).sort((a, b) => points[a].px - points[b].px || points[a].key - points[b].key);
  for (const i of order) {
    const p = points[i];
    const tied = ties.get(Math.round(p.px * 2)) > offs.length;
    const cols = [0];
    if (tied) for (let c = 1; c <= 8; c++) cols.push(c * step, -c * step);
    let best = { x: p.px, y: 0 };
    let bestScore = Infinity;
    search:
    for (const dx of cols) {
      const x = p.px + dx;
      if (dx && (x < lo || x > hi)) continue;
      for (const y of offs) {
        const score = hit(x, y);
        if (score < bestScore) { bestScore = score; best = { x, y }; }
        if (score === 0) break search;
      }
    }
    p.px = best.x;
    p.off = best.y;
    placed.push(best);
  }
}

/**
 * Сравнение вариантов: строки — варианты агента внутри групп (уровней или условий).
 * cfg: {info, groups: [{label, rows: [{label, color, stat, points: [{v, run}]}]}], rowLabels, onPick(run), describe(run)}
 */
export function stripChart(host, cfg) {
  const { info, groups } = cfg;
  const values = [];
  for (const g of groups) {
    for (const r of g.rows) {
      for (const p of r.points) values.push(p.v);
      if (r.stat) values.push(r.stat.ci[0], r.stat.ci[1]);
    }
  }
  const scale = domainFor(info, values);
  // Чем больше прогонов в строке, тем выше строка и мельче точки — чтобы кучки одинаковых значений оставались узкими.
  const most = Math.max(1, ...groups.flatMap((g) => g.rows.map((r) => r.points.length)));
  const BASE_ROW = most <= 24 ? 30 : most <= 45 ? 38 : 44;
  const R = most <= 24 ? 3.2 : most <= 45 ? 2.8 : 2.5;
  const GAP = 12;
  const PAD = 18;                // поле внутри области графика: кучке у края шкалы есть куда расти
  let lastW = 0;
  let hits = [];
  let focus = null;
  let svg = null;

  function draw() {
    const W = Math.max(300, Math.floor(host.clientWidth));
    lastW = W;
    const labelFont = `13.5px ${FONT}`;
    const headFont = `600 13.5px ${FONT}`;
    const multi = groups.length > 1;
    // Слева либо подписи вариантов (когда их больше двух), либо подписи групп.
    let gutter = 0;
    let ROW = BASE_ROW;
    if (cfg.rowLabels) {
      const longest = Math.max(...groups.flatMap((g) => g.rows.map((r) => textWidth(r.label, labelFont))));
      gutter = Math.min(Math.max(90, longest + 26), W * 0.4);
      // Длинные подписи вариантов переносятся на вторую строку — строкам нужно чуть больше высоты.
      if (longest + 26 > gutter) ROW = Math.max(ROW, 36);
    } else if (multi) {
      const longest = Math.max(...groups.map((g) => textWidth(g.label, headFont)));
      gutter = Math.min(Math.max(70, longest + 14), W * 0.34);
    }
    const valueW = 62;
    const x0 = gutter + 10;
    const x1 = W - valueW - 6;
    const span = Math.max(40, x1 - x0 - 2 * PAD);
    const x = (v) => x0 + PAD + ((v - scale.lo) / (scale.hi - scale.lo)) * span;
    const headH = multi && cfg.rowLabels ? 24 : 0;

    hits = [];
    const layers = { grid: s('g'), rows: s('g'), marks: s('g'), text: s('g') };
    let y = 6;
    const top = y;
    groups.forEach((g, gi) => {
      if (gi > 0) y += GAP;
      const gTop = y;
      if (headH) {
        layers.text.append(s('text', { x: 0, y: y + 16, class: 'lb-ch-head' }, g.label));
        y += headH;
      }
      g.rows.forEach((row) => {
        const cy = y + ROW / 2;
        if (cfg.rowLabels) {
          layers.text.append(s('circle', { cx: 6, cy, r: 5, fill: row.color }));
          const lines = wrap2(row.label, labelFont, gutter - 26);
          const cut = lines.join(' ') !== row.label;
          lines.forEach((line, li) => {
            const ty = lines.length === 1 ? cy + 4.5 : cy - 3 + li * 15;
            layers.text.append(s('text', { x: 18, y: ty, class: 'lb-ch-label' }, line, cut ? s('title', null, row.label) : null));
          });
        }
        const pts = row.points.map((p) => ({ ...p, px: x(p.v), key: p.run.seed || 0 }));
        dodge(pts, R, ROW / 2 - R - 2.5, x0 + R + 1, x1 - R - 1);
        for (const p of pts) {
          layers.rows.append(s('circle', { cx: p.px.toFixed(1), cy: (cy + p.off).toFixed(1), r: R, fill: row.color, 'fill-opacity': 0.4 }));
          hits.push({ type: 'run', x: p.px, y: cy + p.off, row, group: g, point: p });
        }
        if (row.stat) {
          const a = x(row.stat.ci[0]);
          const b = x(row.stat.ci[1]);
          const m = x(row.stat.mean);
          layers.marks.append(
            s('line', { x1: a, x2: b, y1: cy, y2: cy, stroke: '#fff', 'stroke-width': 6.5, 'stroke-linecap': 'round' }),
            s('line', { x1: a, x2: b, y1: cy, y2: cy, stroke: row.color, 'stroke-width': 2.5, 'stroke-linecap': 'round' }),
            s('circle', { cx: m, cy, r: 5.5, fill: row.color, stroke: '#fff', 'stroke-width': 2 }));
          layers.text.append(s('text', { x: W - 2, y: cy + 5, class: 'lb-ch-value', 'text-anchor': 'end' }, info.mean(row.stat.mean)));
          hits.push({ type: 'mean', x: m, y: cy, row, group: g });
        } else {
          layers.text.append(s('text', { x: W - 2, y: cy + 5, class: 'lb-ch-muted', 'text-anchor': 'end' }, '—'));
        }
        y += ROW;
      });
      if (!cfg.rowLabels && multi) {
        const lines = g.rows.length > 1 ? wrap2(g.label, headFont, gutter - 8) : [fit(g.label, headFont, gutter - 8)];
        const cut = lines.join(' ') !== g.label;
        lines.forEach((line, li) => {
          const ty = (gTop + y) / 2 + 5 + (lines.length === 1 ? 0 : -8 + li * 16);
          layers.text.append(s('text', { x: 0, y: ty, class: 'lb-ch-head' }, line, cut ? s('title', null, g.label) : null));
        });
      }
      if (gi < groups.length - 1) {
        layers.grid.append(s('line', { x1: 0, x2: W, y1: y + GAP / 2, y2: y + GAP / 2, class: 'lb-ch-sep' }));
      }
    });
    const bottom = y + 4;
    for (const t of scale.ticks) {
      const tx = x(t);
      layers.grid.append(s('line', { x1: tx, x2: tx, y1: top - 2, y2: bottom, class: 'lb-ch-grid' }));
      layers.text.append(s('text', { x: tx, y: bottom + 16, class: 'lb-ch-tick', 'text-anchor': 'middle' }, info.tick(t)));
    }
    const H = bottom + 24;
    focus = s('g', { class: 'lb-ch-focus', visibility: 'hidden' },
      s('circle', { r: 5, class: 'lb-ch-focus__dot' }), s('circle', { r: 8, class: 'lb-ch-focus__ring' }));
    const overlay = s('rect', { x: x0, y: top - 2, width: Math.max(0, x1 - x0), height: bottom - top + 2, fill: 'transparent', class: 'lb-ch-hit' });
    svg = s('svg', { class: 'lb-ch', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.aria || info.title },
      layers.grid, layers.rows, layers.marks, layers.text, focus, overlay);

    let current = null;
    const pick = (evt) => {
      const box = svg.getBoundingClientRect();
      const px = evt.clientX - box.left;
      const py = evt.clientY - box.top;
      let best = null;
      let bestD = 18 * 18;
      for (const hit of hits) {
        const d = (hit.x - px) ** 2 + (hit.y - py) ** 2 - (hit.type === 'mean' ? 20 : 0);
        if (d < bestD) { bestD = d; best = hit; }
      }
      return best;
    };
    overlay.addEventListener('pointermove', (evt) => {
      current = pick(evt);
      if (!current) {
        focus.setAttribute('visibility', 'hidden');
        overlay.style.cursor = '';
        tip.hide();
        return;
      }
      focus.setAttribute('visibility', 'visible');
      focus.setAttribute('transform', `translate(${current.x.toFixed(1)},${current.y.toFixed(1)})`);
      focus.firstChild.setAttribute('fill', current.row.color);
      focus.firstChild.setAttribute('r', current.type === 'mean' ? 5.5 : 4.2);
      overlay.style.cursor = current.type === 'run' ? 'pointer' : 'default';
      tip.show(evt.clientX, evt.clientY, current.type === 'run' ? runTip(current) : meanTip(current));
    });
    overlay.addEventListener('pointerleave', () => {
      current = null;
      focus.setAttribute('visibility', 'hidden');
      tip.hide();
    });
    overlay.addEventListener('click', (evt) => {
      const hit = pick(evt) || current;
      if (hit && hit.type === 'run' && cfg.onPick) {
        tip.hide();
        cfg.onPick(hit.point.run);
      }
    });
    host.replaceChildren(svg);
  }

  function runTip(hit) {
    const v = hit.point.v;
    const value = info.bool ? (v ? 'вернулся на базу' : 'не вернулся') : `${info.value(v)}${info.kind === 'num' && info.unitOf ? ` ${info.unitOf}` : ''}`;
    return [
      h('div', { class: 'lb-tip__value', text: value }),
      h('div', { class: 'lb-tip__row' }, h('span', { class: 'lb-tip__key', style: { background: hit.row.color } }), hit.row.label),
      h('div', { class: 'lb-tip__muted', text: cfg.describe ? cfg.describe(hit.point.run) : '' }),
      h('div', { class: 'lb-tip__hint', text: 'Щелчок — открыть прогон' }),
    ];
  }

  function meanTip(hit) {
    const st = hit.row.stat;
    const unit = info.kind === 'num' && info.unitOf ? ` ${info.unitOf}` : '';
    return [
      h('div', { class: 'lb-tip__value', text: `${info.mean(st.mean)}${unit}` }),
      h('div', { class: 'lb-tip__row' }, h('span', { class: 'lb-tip__key', style: { background: hit.row.color } }), hit.row.label),
      h('div', { class: 'lb-tip__muted', text: `среднее по ${st.n} прогонам${groups.length > 1 ? ` · ${hit.group.label}` : ''}` }),
      h('div', { class: 'lb-tip__muted', text: `95% интервал: от ${info.mean(st.ci[0])} до ${info.mean(st.ci[1])}` }),
    ];
  }

  const ro = new ResizeObserver(() => {
    const w = Math.floor(host.clientWidth);
    if (w && w !== lastW) draw();
  });
  draw();
  ro.observe(host);
  return { destroy() { ro.disconnect(); tip.hide(); } };
}

/** Подписи цветов: варианты агента. */
export function legend(items, label) {
  return h('div', { class: 'lb-legend', role: 'list', 'aria-label': label || 'Варианты агента' },
    items.map((it) => h('span', { class: 'lb-legend__item', role: 'listitem' }, dot(it.color), it.label)));
}

/** Разность словами: «на 15 п. п. больше». */
export function diffWords(info, v) {
  if (v == null) return '—';
  const shown = info.diffBare(Math.abs(v)).replace('+', '');
  if (/^0(,0+)?$/.test(shown)) return 'разницы нет';
  const unit = info.kind === 'pct' ? ' п. п.' : info.kind === 'num' && info.unitOf ? ` ${info.unitOf}` : '';
  return `на ${shown}${unit} ${v > 0 ? 'больше' : 'меньше'}`;
}

/**
 * Разность «a − b» на одинаковых сценариях по строкам (уровням или условиям).
 * cfg: {info, better, aLabel, bLabel, rows: [{label, strong, pair, verdict}]}
 */
export function diffChart(cfg) {
  const { info, better, rows } = cfg;
  const withPair = rows.filter((r) => r.pair);
  const vals = [0];
  for (const r of withPair) vals.push(r.pair.ci[0], r.pair.ci[1], r.pair.mean);
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = (hi - lo) * 0.06 || 1;
  if (lo < 0) lo -= pad;
  if (hi > 0) hi += pad;
  const scale = niceTicks(lo, hi, 5);
  const pos = (v) => `${(((v - scale.lo) / (scale.hi - scale.lo)) * 100).toFixed(2)}%`;
  const TONE = { supported: 'ok', refuted: 'bad', inconclusive: 'none', no_data: 'none' };
  const side = better === 'higher' ? 'Правее' : 'Левее';

  const grid = () => scale.ticks.map((t) => h('i', { class: `lb-forest__grid${t === 0 ? ' lb-forest__grid--zero' : ''}`, style: { left: pos(t) } }));
  const cells = [
    h('div', { class: 'lb-forest__th' }),
    h('div', { class: 'lb-forest__th lb-forest__th--plot', text: `${side} нуля — лучше у варианта «${cfg.aLabel}»` }),
    h('div', { class: 'lb-forest__th', text: 'В среднем' }),
    h('div', { class: 'lb-forest__th', text: 'На одинаковых сценариях' }),
    h('div', { class: 'lb-forest__th', text: 'Вывод' }),
  ];

  for (const r of rows) {
    const cls = r.strong ? ' lb-forest__cell--strong' : '';
    const p = r.pair;
    const plot = h('div', { class: `lb-forest__plot${cls}` }, h('div', { class: 'lb-forest__track' }, grid()));
    let numbers;
    let tally;
    if (p) {
      const tone = TONE[r.verdict] || 'none';
      const a = Math.min(p.ci[0], p.ci[1]);
      const b = Math.max(p.ci[0], p.ci[1]);
      plot.firstChild.append(
        h('i', { class: 'lb-forest__ci', data: { tone }, style: { left: pos(a), width: `${(((b - a) / (scale.hi - scale.lo)) * 100).toFixed(2)}%` } }),
        h('i', { class: 'lb-forest__pt', data: { tone }, style: { left: pos(p.mean) } }));
      const aBetter = better === 'higher' ? p.a_higher : p.b_higher;
      const bBetter = better === 'higher' ? p.b_higher : p.a_higher;
      const single = p.n < 2;        // по одной паре интервал не посчитать — так и пишем
      numbers = h('div', { class: `lb-forest__num${cls}` },
        h('div', { class: 'lb-forest__main', text: diffWords(info, p.mean) }),
        h('div', { class: 'lb-forest__sub', text: single ? 'одна пара прогонов, интервала нет' : `95% интервал: от ${info.diffBare(p.ci[0])} до ${info.diffBare(p.ci[1])}` }));
      tally = h('div', { class: `lb-forest__num${cls}` },
        h('div', { class: 'lb-forest__main', text: `лучше в ${aBetter} из ${p.n}` }),
        h('div', { class: 'lb-forest__sub', text: `хуже в ${bBetter}, поровну в ${p.ties}` }));
      const tipRows = [
        h('div', { class: 'lb-tip__value', text: `${info.diff(p.mean)}${info.kind === 'num' && info.unitOf ? ` ${info.unitOf}` : ''}` }),
        h('div', { class: 'lb-tip__muted', text: `«${cfg.aLabel}» минус «${cfg.bLabel}», ${p.n} ${plural(p.n, 'сценарий', 'сценария', 'сценариев')}` }),
        single ? h('div', { class: 'lb-tip__muted', text: 'Сравнение одной пары прогонов: это наблюдение, а не статистика' })
          : h('div', { class: 'lb-tip__muted', text: `95% интервал: от ${info.diffBare(p.ci[0])} до ${info.diffBare(p.ci[1])}` }),
        p.sign_p != null && !single ? h('div', { class: 'lb-tip__muted', text: `Будь варианты равны, такой перевес по числу сценариев выпадал бы с вероятностью ${num(p.sign_p * 100, p.sign_p < 0.01 ? 2 : 1)} %` }) : null,
      ];
      plot.addEventListener('pointermove', (e) => tip.show(e.clientX, e.clientY, tipRows.filter(Boolean)));
      plot.addEventListener('pointerleave', () => tip.hide());
    } else {
      numbers = h('div', { class: `lb-forest__num${cls}` }, h('div', { class: 'lb-forest__sub', text: 'нет пар для сравнения' }));
      tally = h('div', { class: `lb-forest__num${cls}` });
    }
    cells.push(
      h('div', { class: `lb-forest__label${cls}`, text: r.label }),
      plot, numbers, tally,
      h('div', { class: `lb-forest__verdict${cls}` }, badge(r.verdict)));
  }

  cells.push(
    h('div', { class: 'lb-forest__axis-pad' }),
    h('div', { class: 'lb-forest__axis' }, h('div', { class: 'lb-forest__track' },
      scale.ticks.map((t) => h('span', { class: 'lb-forest__tick', style: { left: pos(t) }, text: t === 0 ? '0' : info.diffTick(t) })))),
    h('div', { class: 'lb-forest__axis-note', text: `разность, ${info.diffUnit || 'ед.'}` }),
    h('div'), h('div'));

  return h('div', { class: 'lb-forest', role: 'table', 'aria-label': `Разность «${cfg.aLabel}» минус «${cfg.bLabel}»: ${info.label}` }, cells);
}
