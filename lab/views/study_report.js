// Отчёт исследования для страницы «Конструктор»: итог крупно, график замеров с погрешностями, сравнение
// объяснений, таблица замеров, вывод словами. Данные — поле study из ответа POST /api/study (did/study.py).
// Цвет на графиках значит одно: оранжевый — исследуемое место, синий — контроль. Чёрное — итоговая оценка.

import { h, s, fill, icon, tip, num, ruText, runHref, plural } from './common.js';
import { niceTicks } from '../charts.js';

const TEST = '#eb6834';
const CONTROL = '#2a78d6';
const ROLE = { test: 'исследуемое место', control: 'контроль', calibration: 'калибровка', check: 'проверка' };
const KIND = { straight: 'пробег', pause: 'пауза', spin: 'разворот', listen: 'слушает датчик' };
const STATUS = {
  done: { word: 'Точность достигнута', tone: 'ok', icon: 'check' },
  not_reached: { word: 'Точность не достигнута', tone: 'warn', icon: 'half' },
  incomplete: { word: 'Набор замеров неполный', tone: 'warn', icon: 'half' },
  failed: { word: 'Оценку получить не удалось', tone: 'bad', icon: 'cross' },
  refused: { word: 'Задание не принято', tone: 'bad', icon: 'cross' },
};
const LAW_STATUS = { done: 'Объяснение найдено', not_reached: 'Данных недостаточно', failed: 'Проверить не удалось' };
const LAWS = {
  linear: (d, r) => Math.max(0, 1 - d / r),
  faster: (d, r) => Math.max(0, 1 - d / r) ** 2,
  slower: (d, r) => Math.max(0, 1 - (d / r) ** 2),
};

/** Сколько знаков после запятой показывать при такой погрешности (две значащие цифры погрешности). */
export function digitsFor(sigma) {
  if (!sigma || !(sigma > 0)) return 2;
  return Math.min(5, Math.max(0, 1 - Math.floor(Math.log10(sigma))));
}

const cap = (t) => (t ? t[0].toUpperCase() + t.slice(1) : '');
const colorOf = (role) => (role === 'test' ? TEST : CONTROL);

function statusBadge(report) {
  const q = report.spec && report.spec.quantity;
  const st = STATUS[report.status] || { word: report.status || '—', tone: 'none', icon: 'dash' };
  const word = q === 'sensor_law' && LAW_STATUS[report.status] ? LAW_STATUS[report.status] : st.word;
  return h('span', { class: 'lb-badge lb-badge--big', data: { tone: st.tone } }, icon(st.icon, 18), h('span', { text: word }));
}

function shortLaw(hyp) {
  if (!hyp || !hyp.law) return hyp ? hyp.statement : '';
  if (hyp.law === 'linear') return `линейно до ${num(hyp.range_m, 1)} м`;
  return hyp.law === 'faster' ? 'быстрее линейного' : 'медленнее линейного';
}

// --- графики ------------------------------------------------------------------------------------

/** Раздвинуть подписи у правого края, чтобы они не наезжали друг на друга. */
function spread(labels, top, bottom, gap = 16) {
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < gap) labels[i].y = labels[i - 1].y + gap;
  const over = labels.length ? labels[labels.length - 1].y - bottom : 0;
  if (over > 0) labels.forEach((l) => { l.y = Math.max(top, l.y - over); });
  return labels;
}

function pointTip(p, unit, fmt) {
  const what = `${cap(KIND[p.kind] || p.kind)}${p.kind === 'straight' ? ` ${num(p.ds, 2)} м` : p.kind === 'spin' ? ` на ${num((p.dth * 180) / Math.PI, 0)}°` : ` ${num(p.dt, 1)} с`}`;
  return [
    h('div', { class: 'lb-tip__value', text: `${fmt(p.value)} ± ${fmt(p.err)} ${unit}` }),
    h('div', { class: 'lb-tip__row' }, h('span', { class: 'lb-tip__key', style: { background: colorOf(p.role) } }), `замер № ${p.n} · ${ROLE[p.role] || p.role}`),
    h('div', { class: 'lb-tip__muted', text: `${what} · ${num(p.t, 0)}-я секунда · ушло ${num(p.spent, 3)} ед.` }),
    p.d != null ? h('div', { class: 'lb-tip__muted', text: `до образца ${num(p.d, 2)} м` }) : null,
    p.used ? null : h('div', { class: 'lb-tip__muted', text: `не учтён: ${ruText(p.note)}` }),
  ].filter(Boolean);
}

/** Слой наведения: ближайшая точка получает кольцо и подсказку. hits: [{px, py, tip}] */
function hoverLayer(svg, hits, box) {
  const ring = s('circle', { r: 9, class: 'st-ch-focus', visibility: 'hidden' });
  const overlay = s('rect', { x: box.x, y: box.y, width: Math.max(0, box.w), height: Math.max(0, box.h), fill: 'transparent' });
  overlay.addEventListener('pointermove', (e) => {
    const r = svg.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    let best = null;
    let bd = Infinity;
    for (const hit of hits) {
      const d = (hit.px - px) ** 2 + ((hit.py - py) ** 2) * 0.25;      // по горизонтали целимся точнее, чем по вертикали
      if (d < bd) { bd = d; best = hit; }
    }
    if (!best || bd > 60 * 60) { ring.setAttribute('visibility', 'hidden'); tip.hide(); return; }
    ring.setAttribute('visibility', 'visible');
    ring.setAttribute('cx', best.px);
    ring.setAttribute('cy', best.py);
    tip.show(e.clientX, e.clientY, best.tip);
  });
  overlay.addEventListener('pointerleave', () => { ring.setAttribute('visibility', 'hidden'); tip.hide(); });
  svg.append(ring, overlay);
}

function mark(layer, px, yLo, yHi, py, p) {
  const color = colorOf(p.role);
  layer.append(
    s('line', { x1: px, x2: px, y1: yLo, y2: yHi, stroke: color, class: 'st-ch-err' }),
    s('line', { x1: px - 4, x2: px + 4, y1: yLo, y2: yLo, stroke: color, class: 'st-ch-err' }),
    s('line', { x1: px - 4, x2: px + 4, y1: yHi, y2: yHi, stroke: color, class: 'st-ch-err' }),
    p.used ? s('circle', { cx: px, cy: py, r: 5, fill: color, class: 'st-ch-dot' })
      : s('circle', { cx: px, cy: py, r: 4.5, stroke: color, class: 'st-ch-dot--open' }));
}

/**
 * Замеры по порядку: точка — значение, усы — 95% интервал одного замера; чёрная линия и серая полоса — итоговая оценка.
 * cfg: {points: [{n, t, value, err, role, used, …}], unit, fmt, est: {value, lo, hi}, estLabel, truth, ref: {value, label}, height}
 */
function measureChart(host, cfg) {
  const { points, fmt } = cfg;
  let lastW = 0;
  function draw() {
    const W = Math.max(260, Math.floor(host.clientWidth));
    lastW = W;
    const H = cfg.height || 250;
    const hasLabels = cfg.est || cfg.truth != null || cfg.ref;
    const M = { l: 56, r: hasLabels ? 132 : 14, t: 26, b: 40 };
    let lo = Infinity;
    let hi = -Infinity;
    const take = (v) => { if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
    points.forEach((p) => { take(p.value - p.err); take(p.value + p.err); });
    if (cfg.est) { take(cfg.est.lo); take(cfg.est.hi); }
    if (cfg.truth != null) take(cfg.truth);
    if (cfg.ref) take(cfg.ref.value);
    const pad = (hi - lo) * 0.14 || Math.abs(hi) * 0.05 || 1;
    const scale = niceTicks(lo - pad, hi + pad, 4);
    const x0 = M.l;
    const x1 = W - M.r;
    const y = (v) => M.t + (1 - (v - scale.lo) / (scale.hi - scale.lo)) * (H - M.t - M.b);
    const x = (i) => x0 + ((i + 0.5) * (x1 - x0)) / Math.max(1, points.length);
    const digits = Math.max(0, Math.min(4, -Math.floor(Math.log10(scale.step) + 1e-9)));

    const grid = s('g');
    const marks = s('g');
    const text = s('g');
    for (const t of scale.ticks) {
      grid.append(s('line', { x1: x0, x2: x1, y1: y(t), y2: y(t), class: 'st-ch-grid' }));
      text.append(s('text', { x: x0 - 8, y: y(t) + 4, class: 'st-ch-tick', 'text-anchor': 'end' }, num(t, digits)));
    }
    text.append(s('text', { x: 0, y: 12, class: 'st-ch-title' }, cfg.unit));
    text.append(s('text', { x: (x0 + x1) / 2, y: H - 4, class: 'st-ch-title', 'text-anchor': 'middle' }, 'номер замера, по порядку'));

    const labels = [];
    if (cfg.est) {
      grid.append(s('rect', { x: x0, y: y(cfg.est.hi), width: x1 - x0, height: Math.max(1, y(cfg.est.lo) - y(cfg.est.hi)), class: 'st-ch-band' }));
      grid.append(s('line', { x1: x0, x2: x1, y1: y(cfg.est.value), y2: y(cfg.est.value), class: 'st-ch-est' }));
      labels.push({ y: y(cfg.est.value), text: `${cfg.estLabel || 'оценка'} ${fmt(cfg.est.value)}`, cls: 'st-ch-label' });
    }
    if (cfg.ref) {
      grid.append(s('line', { x1: x0, x2: x1, y1: y(cfg.ref.value), y2: y(cfg.ref.value), class: 'st-ch-axis' }));
      labels.push({ y: y(cfg.ref.value), text: cfg.ref.label, cls: 'st-ch-label st-ch-label--mute' });
    }
    if (cfg.truth != null) {
      grid.append(s('line', { x1: x0, x2: x1, y1: y(cfg.truth), y2: y(cfg.truth), class: 'st-ch-truth' }));
      labels.push({ y: y(cfg.truth), text: `на самом деле ${fmt(cfg.truth)}`, cls: 'st-ch-label' });
    }
    for (const l of spread(labels, M.t + 4, H - M.b)) text.append(s('text', { x: x1 + 8, y: l.y + 4, class: l.cls }, l.text));

    const hits = [];
    const every = points.length > 14 ? 2 : 1;
    points.forEach((p, i) => {
      const px = x(i);
      mark(marks, px, y(p.value - p.err), y(p.value + p.err), y(p.value), p);
      if (i % every === 0) text.append(s('text', { x: px, y: H - M.b + 16, class: 'st-ch-tick', 'text-anchor': 'middle' }, String(p.n)));
      hits.push({ px, py: y(p.value), tip: pointTip(p, cfg.unitShort || cfg.unit, fmt) });
    });
    const svg = s('svg', { class: 'st-ch', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.aria || 'Замеры по порядку' }, grid, marks, text);
    hoverLayer(svg, hits, { x: x0, y: M.t, w: x1 - x0, h: H - M.t - M.b });
    host.replaceChildren(svg);
  }
  const ro = new ResizeObserver(() => { const w = Math.floor(host.clientWidth); if (w && w !== lastW) draw(); });
  draw();
  ro.observe(host);
  return () => { ro.disconnect(); tip.hide(); };
}

/** Показание датчика против расстояния до образца: точки замеров и кривые объяснений (подтвердившееся — чёрным). */
function lawChart(host, cfg) {
  const { points, hyps } = cfg;
  let lastW = 0;
  function draw() {
    const W = Math.max(280, Math.floor(host.clientWidth));
    lastW = W;
    const H = 320;
    const M = { l: 48, r: 18, t: 26, b: 42 };
    const xs = niceTicks(0, Math.max(2.5, ...points.map((p) => p.d + 0.2)), 5);
    const x = (d) => M.l + ((d - xs.lo) / (xs.hi - xs.lo)) * (W - M.l - M.r);
    const y = (v) => M.t + (1 - v) * (H - M.t - M.b);
    const grid = s('g');
    const curves = s('g');
    const marks = s('g');
    const text = s('g');
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      grid.append(s('line', { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: 'st-ch-grid' }));
      text.append(s('text', { x: M.l - 8, y: y(t) + 4, class: 'st-ch-tick', 'text-anchor': 'end' }, num(t, 2)));
    }
    for (const t of xs.ticks) text.append(s('text', { x: x(t), y: H - M.b + 16, class: 'st-ch-tick', 'text-anchor': 'middle' }, num(t, 1)));
    text.append(s('text', { x: 0, y: 12, class: 'st-ch-title' }, 'показание датчика'));
    text.append(s('text', { x: (M.l + W - M.r) / 2, y: H - 4, class: 'st-ch-title', 'text-anchor': 'middle' }, 'расстояние до образца, м'));

    const at = Math.min(1.1, xs.hi * 0.45);                    // где подписывать кривые: здесь они расходятся сильнее всего
    const labels = [];
    for (const hyp of hyps) {
      const f = LAWS[hyp.law];
      if (!f) continue;
      const pts = [];
      for (let i = 0; i <= 80; i++) {
        const d = xs.lo + ((xs.hi - xs.lo) * i) / 80;
        pts.push(`${x(d).toFixed(1)},${y(f(d, hyp.range_m)).toFixed(1)}`);
      }
      const best = hyp.id === cfg.best;
      curves.append(s('polyline', { points: pts.join(' '), class: `st-ch-curve${best ? ' st-ch-curve--best' : ''}` }));
      labels.push({ y: y(f(at, hyp.range_m)), text: shortLaw(hyp) + (cfg.truthId === hyp.id ? ' — так на самом деле' : ''), cls: `st-ch-label${best ? '' : ' st-ch-label--mute'}` });
    }
    for (const l of spread(labels, M.t + 8, H - M.b - 4, 15)) text.append(s('text', { x: x(at) + 10, y: l.y - 4, class: l.cls }, l.text));

    const hits = [];
    for (const p of points) {
      const px = x(p.d);
      mark(marks, px, y(Math.max(0, p.value - p.err)), y(Math.min(1, p.value + p.err)), y(p.value), p);
      hits.push({ px, py: y(p.value), tip: pointTip(p, '', (v) => num(v, 3)) });
    }
    const svg = s('svg', { class: 'st-ch', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Показание датчика против расстояния до образца' }, grid, curves, marks, text);
    hoverLayer(svg, hits, { x: M.l, y: M.t, w: W - M.l - M.r, h: H - M.t - M.b });
    host.replaceChildren(svg);
  }
  const ro = new ResizeObserver(() => { const w = Math.floor(host.clientWidth); if (w && w !== lastW) draw(); });
  draw();
  ro.observe(host);
  return () => { ro.disconnect(); tip.hide(); };
}

const key = (cls, text, style) => h('span', { class: 'st-key' }, h('i', { class: cls, style }), text);

function chartCard(report, truthOn, cleanup) {
  const q = report.spec.quantity;
  const est = report.estimate;
  const truth = truthOn && report.truth ? report.truth : null;
  const all = (report.measurements || []).filter((m) => m.chart && m.value != null)
    .map((m) => ({ ...m, err: 1.96 * (m.sigma || 0) }));
  if (!all.length) return null;
  const unused = all.some((p) => !p.used);
  const legend = [];
  const plot = h('div', { class: 'st-chart__plot' });
  const parts = [plot];
  let title;
  let sub;

  if (q === 'sensor_law') {
    const inq = report.inquiry;
    const hyps = (report.spec.hypotheses || []).map((x) => ({ ...x }));
    const best = inq && inq.conclusion && inq.conclusion.status === 'identified' ? inq.conclusion.best : null;
    title = 'Показание датчика на разных расстояниях от образца';
    sub = 'Точка — среднее показание за паузу, усы — его 95% интервал. Линии — что предсказывает каждое объяснение.';
    legend.push(key('st-key__dot', 'замер на расстоянии', { background: TEST }), key('st-key__dot', 'опорная точка у образца (контроль)', { background: CONTROL }));
    if (best) legend.push(key('st-key__line', 'объяснение, которое подтвердилось'));
    legend.push(key('st-key__curve', best ? 'остальные объяснения' : 'объяснения из задания'));
    cleanup.push(lawChart(plot, { points: all.filter((p) => p.d != null), hyps, best, truthId: truth ? truth.best : null }));
  } else {
    const d = digitsFor(est ? est.sigma : all[0].sigma);
    const fmt = (v) => num(v, d);
    const band = est ? { value: est.value, lo: est.ci95[0], hi: est.ci95[1] } : null;
    const both = all.some((p) => p.role === 'test') && all.some((p) => p.role === 'control');
    const unit = q === 'soil_cost' ? 'во сколько раз дороже обычного пола' : est ? est.unit : all[0].unit;
    title = { soil_cost: 'Замеры: исследуемая область против контрольного участка', turn_cost: 'Цена поворота по каждому развороту',
      idle_cost: 'Расход на месте по каждой паузе', load_effect: 'Расход на метр с грузом по каждому пробегу' }[q] || 'Замеры';
    sub = 'Точка — один замер, усы — его 95% интервал по шуму показаний батареи.';
    if (both) legend.push(key('st-key__dot', 'в исследуемом месте', { background: TEST }), key('st-key__dot', 'контроль', { background: CONTROL }));
    if (band && q !== 'load_effect') legend.push(key('st-key__band', 'итоговая оценка и её 95% интервал'));
    if (truth && q !== 'load_effect') legend.push(key('st-key__dash', 'настоящее значение из сценария'));
    const main = { points: all, unit, unitShort: q === 'soil_cost' ? '× к обычному полу' : unit, fmt, aria: title };
    if (q !== 'load_effect') Object.assign(main, { est: band, truth: truth ? truth.value : null });
    if (q === 'soil_cost') main.ref = { value: 1, label: 'обычный пол = 1' };
    cleanup.push(measureChart(plot, main));

    // Усы намного меньше разницы между участками — показываем каждый участок крупно, в своём масштабе.
    const maxErr = Math.max(...all.map((p) => p.err));
    const span = Math.max(...all.map((p) => p.value)) - Math.min(...all.map((p) => p.value));
    if (q === 'soil_cost' && both && span > 12 * maxErr) {
      const c = report.control && report.control.control;
      const lens = [
        { name: 'Крупно: контрольный участок', pts: all.filter((p) => p.role === 'control'),
          cfg: c && c.sigma ? { est: { value: 1, lo: 1 - (1.96 * c.sigma) / c.value, hi: 1 + (1.96 * c.sigma) / c.value }, estLabel: 'обычный пол' } : {} },
        { name: 'Крупно: исследуемая область. Полоса шире усов: в итоговую погрешность входит и погрешность контроля', pts: all.filter((p) => p.role === 'test'), cfg: { est: band, truth: truth ? truth.value : null } },
      ];
      parts.push(h('div', { class: 'st-lenses' }, lens.map((l) => {
        const box = h('div', { class: 'st-chart__plot' });
        const node = h('div', null, h('div', { class: 'st-chart__sub', text: l.name }), box);
        queueMicrotask(() => cleanup.push(measureChart(box, { points: l.pts, unit: '× к обычному полу', fmt, height: 200, aria: l.name, ...l.cfg })));
        return node;
      })));
      sub += ' Усы меньше самой точки, поэтому ниже каждый участок показан крупно, в своём масштабе.';
    }
  }
  if (unused) legend.push(key('st-key__open', 'замер не учтён'));
  return h('figure', { class: 'lb-card st-chartcard', style: { margin: 0 } },
    h('div', null, h('div', { class: 'st-chart__title', text: title }), h('div', { class: 'st-chart__sub', text: sub })),
    legend.length ? h('div', { class: 'st-keys' }, legend) : null,
    parts,
    h('div', { class: 'st-chart__sub', text: 'Те же числа — в таблице замеров ниже.' }));
}

// --- сравнение объяснений -------------------------------------------------------------------------

function inquiryCard(report, truthOn) {
  const inq = report.inquiry;
  if (!inq) return null;
  const best = inq.conclusion && inq.conclusion.status === 'identified' ? inq.conclusion.best : null;
  const right = truthOn && report.truth ? report.truth.best : null;
  const short = new Map((report.spec.hypotheses || []).map((x) => [x.id, x.law ? shortLaw(x) : x.statement]));
  const rows = inq.alternatives.map((a) => [
    h('div', { class: `st-alt__name${a.id === best ? ' st-alt__name--best' : ''}` },
      a.id === best ? icon('check', 16) : null, h('span', { text: cap(ruText(a.statement)) + (a.id === right ? ' — так на самом деле' : '') })),
    h('div', { class: 'st-alt__bar', role: 'img', 'aria-label': `вероятность после опытов ${num(a.posterior * 100, 0)} %, до опытов ${num(a.prior * 100, 0)} %` },
      h('i', { style: { width: `${(a.posterior * 100).toFixed(1)}%` } }), h('b', { style: { left: `${(a.prior * 100).toFixed(1)}%` } })),
    h('div', { class: 'st-alt__num' }, `${num(a.prior * 100, 0)} % → `, h('strong', { text: `${num(a.posterior * 100, 0)} %` })),
  ]);
  const done = inq.tests.filter((t) => t.measured && t.measured.value != null);
  return h('section', { class: 'lb-card st-block' },
    h('h3', { class: 'lb-h3', text: 'Сравнение объяснений' }),
    h('p', { class: 'lb-muted', text: 'Перед каждым опытом известно, что предсказывает каждое объяснение. После замера вероятности пересчитываются по правилу Байеса.' }),
    h('div', { class: 'st-keys' }, key('st-key__post', 'вероятность после опытов'), key('st-key__prior', 'до опытов')),
    h('div', { class: 'st-alts' }, rows),
    inq.conclusion ? h('p', null, h('strong', { text: 'Вывод: ' }), ruText(inq.conclusion.text) + '.') : null,
    done.length ? h('div', { class: 'lb-tablewrap' }, h('table', { class: 'lb-table' },
      h('thead', null, h('tr', null,
        h('th', { scope: 'col', text: 'Опыт' }), h('th', { scope: 'col', class: 'lb-table__num', text: 'Польза, бит' }),
        h('th', { scope: 'col', text: 'Что ожидалось по каждому объяснению' }),
        h('th', { scope: 'col', class: 'lb-table__num', text: 'Измерено' }), h('th', { scope: 'col', class: 'lb-table__num', text: '± (95 %)' }))),
      h('tbody', null, done.map((t) => {
        const d = Math.max(2, digitsFor(t.measured.sigma));
        return h('tr', null,
          h('th', { scope: 'row', text: cap(ruText(t.name)) }),
          h('td', { class: 'lb-table__num', text: num(t.gain_bits, 2) }),
          h('td', { class: 'st-pred', text: Object.entries(t.predictions).map(([id, p]) => `${short.get(id) || id}: ${num(p.mean, 2)}`).join(' · ') }),
          h('td', { class: 'lb-table__num', text: num(t.measured.value, d) }),
          h('td', { class: 'lb-table__num', text: num(1.96 * t.measured.sigma, d) }));
      })))) : null);
}

// --- таблица замеров ------------------------------------------------------------------------------

function measureTable(report) {
  const ms = report.measurements || [];
  if (!ms.length) return null;
  const amount = (m) => (m.kind === 'straight' ? `${num(m.ds, 2)} м` : m.kind === 'spin' ? `${num((m.dth * 180) / Math.PI, 0)}°` : `${num(m.dt, 1)} с`);
  return h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' }, h('h2', { class: 'lb-h2', text: 'Все замеры' }),
      h('span', { class: 'lb-muted', text: 'Каждый замер: робот стоит и усредняет показания батареи, выполняет воздействие, снова стоит и усредняет' })),
    h('div', { class: 'lb-tablewrap' }, h('table', { class: 'lb-table' },
      h('thead', null, h('tr', null,
        h('th', { scope: 'col', class: 'lb-table__num', text: '№' }), h('th', { scope: 'col', class: 'lb-table__num', text: 'Секунда' }),
        h('th', { scope: 'col', text: 'Что и где' }), h('th', { scope: 'col', class: 'lb-table__num', text: 'Сколько' }),
        h('th', { scope: 'col', class: 'lb-table__num', text: 'Длился, с' }), h('th', { scope: 'col', class: 'lb-table__num', text: 'Ушло заряда, ед.' }),
        h('th', { scope: 'col', class: 'lb-table__num', text: 'Значение' }), h('th', { scope: 'col', class: 'lb-table__num', text: '± (95 %)' }),
        h('th', { scope: 'col', text: 'В расчёте' }))),
      h('tbody', null, ms.map((m) => {
        const d = digitsFor(m.sigma);
        const dotColor = m.role === 'test' ? TEST : m.role === 'control' ? CONTROL : '#aab1b8';
        return h('tr', { class: m.used ? null : 'st-unused' },
          h('td', { class: 'lb-table__num', text: String(m.n) }),
          h('td', { class: 'lb-table__num', text: num(m.t, 0) }),
          h('td', null, h('span', { class: 'st-role' }, h('span', { class: 'lb-dot', style: { background: dotColor } }),
            `${cap(KIND[m.kind] || m.kind)}, ${ROLE[m.role] || m.role}`, h('span', { class: 'st-table-note', text: ` (${num(m.x, 1)}; ${num(m.y, 1)})` }))),
          h('td', { class: 'lb-table__num', text: amount(m) }),
          h('td', { class: 'lb-table__num', text: num(m.dt, 1) }),
          h('td', { class: 'lb-table__num', text: num(m.spent, 3) }),
          h('td', { class: 'lb-table__num', text: m.value == null ? '—' : `${num(m.value, d)} ${m.unit || ''}` }),
          h('td', { class: 'lb-table__num', text: m.sigma == null ? '—' : num(1.96 * m.sigma, d) }),
          h('td', null, m.used ? 'да' : h('span', { text: `нет: ${ruText(m.note) || '—'}` })));
      })))));
}

// --- отчёт целиком --------------------------------------------------------------------------------

async function copyText(text) {
  // Сначала старый надёжный способ: он срабатывает сразу, пока браузер считает нажатие действием человека.
  const ta = h('textarea', { style: { position: 'fixed', left: '-9999px', top: '0' }, readonly: true });
  ta.value = text;
  document.body.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  ta.remove();
  if (ok) return true;
  try {
    await Promise.race([navigator.clipboard.writeText(text), new Promise((_, no) => setTimeout(() => no(new Error('нет ответа')), 1500))]);
    return true;
  } catch {
    return false;
  }
}

function fact(label, value, sub, extra) {
  return h('div', { class: 'st-fact' }, h('div', { class: 'st-fact__label', text: label }), h('div', { class: 'st-fact__value' }, value), extra || null,
    sub ? h('div', { class: 'st-fact__sub', text: sub }) : null);
}

function heroBlock(report) {
  const q = report.spec.quantity;
  const est = report.estimate;
  const inq = report.inquiry;
  if (q === 'sensor_law') {
    const c = inq && inq.conclusion;
    const best = c && c.status === 'identified' ? inq.alternatives.find((a) => a.id === c.best) : null;
    return [
      h('div', { class: 'st-result__label', text: 'Как показание датчика зависит от расстояния' }),
      h('div', { class: 'st-hero' }, h('span', { class: 'st-hero__value st-hero__value--text', text: best ? cap(ruText(best.statement)) : 'Объяснения различить не удалось' })),
      best ? h('div', { class: 'st-result__ci', text: `вероятность ${num(c.confidence * 100, 0)} % после ${inq.tests.filter((t) => t.measured).length} ${plural(inq.tests.filter((t) => t.measured).length, 'опыта', 'опытов', 'опытов')}` }) : null,
      est ? h('div', { class: 'st-result__ci', text: `Если закон линейный, дальность датчика ${num(est.value, 2)} ± ${num(1.96 * est.sigma, 2)} м (95% интервал от ${num(est.ci95[0], 2)} до ${num(est.ci95[1], 2)})` }) : null,
    ];
  }
  if (!est) {
    return [h('div', { class: 'st-result__label', text: 'Итог' }), h('div', { class: 'st-hero' }, h('span', { class: 'st-hero__value st-hero__value--text', text: 'Оценки нет' }))];
  }
  const d = digitsFor(est.sigma);
  const soil = q === 'soil_cost';
  return [
    h('div', { class: 'st-result__label', text: { soil_cost: 'Пол в области дороже обычного', turn_cost: 'Поворот стоит', idle_cost: 'На месте уходит', load_effect: 'Каждый образец удорожает метр пути на' }[q] || est.label }),
    h('div', { class: 'st-hero' },
      h('span', { class: 'st-hero__value', text: `${soil ? 'в ' : ''}${num(est.value, d)}` }),
      h('span', { class: 'st-hero__pm', text: `± ${num(1.96 * est.sigma, d)}` }),
      h('span', { class: 'st-hero__unit', text: soil ? 'раза' : { 'ед/рад': 'ед. заряда на радиан', 'ед/с': 'ед. заряда в секунду', 'доля на образец': 'от цены метра' }[est.unit] || est.unit })),
    h('div', { class: 'st-result__ci', text: `95% интервал: от ${num(est.ci95[0], d)} до ${num(est.ci95[1], d)} · погрешность ±${num(est.rel_error * 100, 1)} % при требуемых ±${num(est.target * 100, 1)} %` }),
    est.spread > 1 ? h('div', { class: 'st-result__ci', text: `Повторы расходятся в ${num(est.spread, 1)} раза сильнее шума прибора — погрешность увеличена во столько же раз.` }) : null,
  ];
}

/**
 * Нарисовать отчёт. ctx: {summary (ответ /api/study), truth (показывать ли скрытую правду)}.
 * Возвращает функцию, которая снимает наблюдателей графиков.
 */
export function renderReport(host, ctx) {
  const report = ctx.summary.study;
  const cleanup = [];
  const e = report.energy || {};
  const ms = report.measurements || [];
  const used = ms.filter((m) => m.used);
  const truth = ctx.truth ? report.truth : null;
  const copied = h('span', { class: 'st-copied', role: 'status' });
  const copyBtn = h('button', { class: 'lb-btn', type: 'button' }, 'Скопировать отчёт');
  let copyTimer = null;
  copyBtn.addEventListener('click', async () => {
    const ok = await copyText((ctx.truth && report.markdown_truth) || report.markdown || '');
    copied.textContent = ok ? 'Отчёт в буфере обмена: это текст в разметке Markdown, его можно вставить в документ или на слайд' : 'Не получилось скопировать: браузер не дал доступ к буферу';
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { copied.textContent = ''; }, 6000);
  });

  const share = e.budget ? Math.min(1, Math.max(0, e.spent / e.budget)) : 0;
  const parts = [['дорога', e.travel], ['замеры', e.test], ['контроль и проверки', e.control], ['возврат', e.home]]
    .filter(([, v]) => v >= 0.05).map(([k, v]) => `${k} ${num(v, 1)}`).join(' · ');
  const side = [
    fact('Заряд', `${num(e.spent, 1)} из ${num(e.budget, 0)} ед.`, parts,
      h('div', { class: 'st-meter', role: 'img', 'aria-label': `потрачено ${num(share * 100, 0)} % бюджета` },
        h('i', { style: { width: `${(share * 100).toFixed(1)}%` } }))),
    fact('Время и замеры', `${num(report.time_s, 0)} с · ${used.length} ${plural(used.length, 'замер', 'замера', 'замеров')}`,
      ms.length > used.length ? `ещё ${ms.length - used.length} не ${plural(ms.length - used.length, 'учтён', 'учтены', 'учтены')} — причины в таблице` : 'все замеры учтены'),
    fact('Почему остановились', h('span', { style: { fontSize: '17px', fontWeight: 600 }, text: cap(ruText(report.stop ? report.stop.text : '—')) })),
  ];

  const texts = [];
  if (report.control && report.control.text) texts.push(h('div', { class: 'lb-card st-block' }, h('h3', { class: 'lb-h3', text: 'Сравнение с контролем' }), h('p', { text: ruText(report.control.text) })));
  if ((report.failures || []).length) {
    texts.push(h('div', { class: 'lb-card st-block' }, h('h3', { class: 'lb-h3', text: 'Чего не удалось и о чём стоит помнить' }),
      h('ul', { class: 'st-list' }, report.failures.map((f) => h('li', { text: cap(ruText(f)) })))));
  }

  fill(host,
    h('div', { class: 'lb-section-head' }, h('h2', { class: 'lb-h2', text: 'Отчёт робота' }),
      h('span', { class: 'lb-muted', text: ruText(report.task || '') })),
    h('section', { class: 'lb-card st-result' },
      h('div', { class: 'st-result__main' },
        heroBlock(report),
        h('div', { class: 'st-result__status' }, statusBadge(report)),
        h('p', { class: 'st-result__text', text: ruText(report.conclusion || '') }),
        h('div', { class: 'st-actions' },
          ctx.summary.file ? h('a', { class: 'lb-btn lb-btn--primary', href: runHref(ctx.summary.file) }, icon('play'), 'Посмотреть, как робот это делал') : null,
          copyBtn),
        copied),
      h('div', { class: 'st-result__side' }, side)),
    truth ? h('div', { class: 'st-truth', role: 'note' },
      h('div', null,
        h('div', { class: 'st-truth__title', text: 'Скрытая правда из сценария — роботу она недоступна' }),
        h('div', { text: `${cap(ruText(truth.text))}.` }),
        truth.error_pct != null ? h('div', { class: 'st-truth__row' },
          h('span', { text: `Оценка робота отличается на ${num(truth.error_pct, 1)} %.` }),
          h('span', { class: 'lb-badge', data: { tone: truth.covered ? 'ok' : 'bad' } }, icon(truth.covered ? 'check' : 'cross', 15),
            truth.covered ? 'настоящее значение внутри заявленного интервала' : 'настоящее значение вне заявленного интервала')) : null,
        truth.verdict ? h('div', { class: 'st-truth__row' },
          h('span', { class: 'lb-badge', data: { tone: truth.verdict === 'correct' ? 'ok' : truth.verdict === 'wrong' ? 'bad' : 'none' } },
            icon(truth.verdict === 'correct' ? 'check' : truth.verdict === 'wrong' ? 'cross' : 'question', 15),
            { correct: 'робот назвал верное объяснение', wrong: 'робот назвал неверное объяснение', insufficient: 'робот честно не выбрал объяснение' }[truth.verdict])) : null)) : null,
    report.status === 'refused' ? null : chartCard(report, ctx.truth, cleanup),
    inquiryCard(report, ctx.truth),
    texts.length ? h('div', { class: texts.length > 1 ? 'st-two' : null }, texts) : null,
    measureTable(report));
  return () => { for (const fn of cleanup.splice(0)) { try { fn(); } catch (err) { console.error(err); } } };
}
