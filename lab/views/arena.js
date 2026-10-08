// Арена для страницы сценариев и для неподвижной схемы маршрута: пол, столбы, грунты, опасные зоны,
// образцы, база. Ось y мира направлена вверх, поэтому по вертикали координаты переворачиваются.

import { h, s, soilColor, multLabel } from './common.js';

const U = 100;                 // единиц рисунка на метр
const MARGIN = 0.22;           // поле вокруг арены, м
const cache = new WeakMap();
let uid = 0;

function dpSimplify(pts, eps) {
  if (pts.length < 3) return pts.slice();
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a];
    const [bx, by] = pts[b];
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    let worst = -1;
    let idx = -1;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i][0] - ax) * dy - (pts[i][1] - ay) * dx) / len;
      if (d > worst) { worst = d; idx = i; }
    }
    if (worst > eps) {
      keep[idx] = 1;
      stack.push([a, idx], [idx, b]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

function simplifyLoop(loop, eps) {
  let far = 0;
  let best = -1;
  for (let i = 1; i < loop.length; i++) {
    const d = (loop[i][0] - loop[0][0]) ** 2 + (loop[i][1] - loop[0][1]) ** 2;
    if (d > best) { best = d; far = i; }
  }
  const a = dpSimplify(loop.slice(0, far + 1), eps);
  const b = dpSimplify(loop.slice(far).concat([loop[0]]), eps);
  return a.slice(0, -1).concat(b.slice(0, -1));
}

/** Контуры пола из сетки свободных клеток: внешние границы — многоугольники, столбы — круги. */
function geometry(arena) {
  if (cache.has(arena)) return cache.get(arena);
  const { w, h: rows, res, x0, y0, free } = arena;
  const isFree = (c, r) => r >= 0 && r < rows && c >= 0 && c < w && free[r].charCodeAt(c) === 49;
  let c0 = w;
  let c1 = -1;
  let r0 = rows;
  let r1 = -1;
  const next = new Map();
  const add = (ca, ra, cb, rb) => {
    const k = ca + ra * (w + 1);
    const list = next.get(k);
    if (list) list.push([cb, rb]); else next.set(k, [[cb, rb]]);
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < w; c++) {
      if (!isFree(c, r)) continue;
      if (c < c0) c0 = c;
      if (c > c1) c1 = c;
      if (r < r0) r0 = r;
      if (r > r1) r1 = r;
      if (!isFree(c, r - 1)) add(c, r, c + 1, r);
      if (!isFree(c + 1, r)) add(c + 1, r, c + 1, r + 1);
      if (!isFree(c, r + 1)) add(c + 1, r + 1, c, r + 1);
      if (!isFree(c - 1, r)) add(c, r + 1, c, r);
    }
  }
  const loops = [];
  for (const k0 of [...next.keys()]) {
    while (next.get(k0) && next.get(k0).length) {
      const start = [k0 % (w + 1), Math.floor(k0 / (w + 1))];
      const loop = [start];
      let cur = start;
      for (let guard = 0; guard < 100000; guard++) {
        const list = next.get(cur[0] + cur[1] * (w + 1));
        if (!list || !list.length) break;
        const nx = list.pop();
        if (nx[0] === start[0] && nx[1] === start[1]) break;
        loop.push(nx);
        cur = nx;
      }
      if (loop.length >= 4) loops.push(loop);
    }
  }
  const bx0 = x0 + c0 * res - MARGIN;
  const bx1 = x0 + (c1 + 1) * res + MARGIN;
  const by0 = y0 + r0 * res - MARGIN;
  const by1 = y0 + (r1 + 1) * res + MARGIN;
  const X = (x) => (x - bx0) * U;
  const Y = (y) => (by1 - y) * U;
  const floor = [];
  const pillars = [];
  const holes = [];
  for (const loop of loops) {
    let area = 0;
    let minC = Infinity;
    let maxC = -Infinity;
    let minR = Infinity;
    let maxR = -Infinity;
    for (let i = 0; i < loop.length; i++) {
      const [ax, ay] = loop[i];
      const [bx, by] = loop[(i + 1) % loop.length];
      area += ax * by - bx * ay;
      minC = Math.min(minC, ax); maxC = Math.max(maxC, ax);
      minR = Math.min(minR, ay); maxR = Math.max(maxR, ay);
    }
    area /= 2;
    const bw = maxC - minC;
    const bh = maxR - minR;
    if (area < 0 && bw <= 16 && bh <= 16 && Math.abs(bw - bh) <= 3) {
      pillars.push({ x: x0 + ((minC + maxC) / 2) * res, y: y0 + ((minR + maxR) / 2) * res, r: Math.sqrt(-area / Math.PI) * res });
      continue;
    }
    const pts = simplifyLoop(loop, 0.85).map(([c, r]) => [X(x0 + c * res), Y(y0 + r * res)]);
    const d = `M${pts.map((p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('L')}Z`;
    (area < 0 ? holes : floor).push(d);
  }
  const geo = { X, Y, W: (bx1 - bx0) * U, H: (by1 - by0) * U, floor: floor.join(''), holes: holes.join(''), pillars };
  cache.set(arena, geo);
  return geo;
}

function zoneShape(z, geo, attrs) {
  if (z.shape === 'circle') return s('circle', { cx: geo.X(z.x), cy: geo.Y(z.y), r: z.r * U, ...attrs });
  return s('rect', { x: geo.X(z.x - z.w / 2), y: geo.Y(z.y + z.h / 2), width: z.w * U, height: z.h * U, rx: 5, ...attrs });
}

/**
 * Рисунок арены. opts: {thumb} — маленькая картинка без подписей.
 * state: {samples, soils, hazards, base, ghostSoils, freshSoils (ид.), freshHazards (ид.), moves: [[from, to]],
 *         track: {x, y, color}, collected: Set индексов образцов}
 */
export function createArena(arena, opts = {}) {
  const geo = geometry(arena);
  const id = `lb-ar-${++uid}`;
  const thumb = !!opts.thumb;
  const dynamic = s('g');
  const svg = s('svg', { class: `lb-arena__svg${thumb ? ' lb-arena__svg--thumb' : ''}`, viewBox: `0 0 ${geo.W.toFixed(0)} ${geo.H.toFixed(0)}`, role: 'img', 'aria-label': opts.label || 'Арена' },
    s('defs', null,
      s('clipPath', { id: `${id}-floor` }, s('path', { d: geo.floor })),
      s('pattern', { id: `${id}-hatch`, width: 10, height: 10, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
        s('rect', { width: 10, height: 10, fill: '#d64545', 'fill-opacity': 0.14 }),
        s('line', { x1: 0, y1: 0, x2: 0, y2: 10, stroke: '#d64545', 'stroke-width': 3.2 })),
      s('marker', { id: `${id}-arrow`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
        s('path', { d: 'M0 0L10 5L0 10z', fill: '#14181c' }))),
    s('path', { d: geo.floor, class: 'lb-arena__floor' }),
    dynamic);

  function render(state) {
    const kids = [];
    const zones = s('g', { 'clip-path': `url(#${id}-floor)` });
    const fresh = new Set(state.freshSoils || []);
    for (const z of state.ghostSoils || []) {
      zones.append(zoneShape(z, geo, { class: 'lb-arena__ghost' }));
    }
    for (const z of state.soils || []) {
      zones.append(zoneShape(z, geo, { fill: soilColor(z.mult), 'fill-opacity': 0.88, class: fresh.has(z.id) ? 'lb-arena__fresh' : null }));
    }
    const freshHaz = new Set(state.freshHazards || []);
    for (const z of state.hazards || []) {
      zones.append(zoneShape(z, geo, { fill: `url(#${id}-hatch)`, class: `lb-arena__hazard${freshHaz.has(z.id) ? ' lb-arena__fresh' : ''}` }));
    }
    kids.push(zones);
    if (geo.holes) kids.push(s('path', { d: geo.holes, class: 'lb-arena__pillar' }));
    for (const p of geo.pillars) kids.push(s('circle', { cx: geo.X(p.x), cy: geo.Y(p.y), r: p.r * U, class: 'lb-arena__pillar' }));
    // Стрелка «зона переехала»: от старого места до края нового, чтобы не закрывать подпись.
    for (const [from, to] of state.moves || []) {
      const len = Math.hypot(to.x - from.x, to.y - from.y);
      const reach = to.shape === 'circle' ? to.r : Math.min(to.w, to.h) / 2;
      const k = len > reach + 0.1 ? (len - reach - 0.04) / len : 1;
      kids.push(s('line', {
        x1: geo.X(from.x), y1: geo.Y(from.y), x2: geo.X(from.x + (to.x - from.x) * k), y2: geo.Y(from.y + (to.y - from.y) * k),
        class: 'lb-arena__move', 'marker-end': `url(#${id}-arrow)`,
      }));
    }
    if (!thumb) {
      for (const z of state.soils || []) {
        const dark = z.mult >= 2.6;
        kids.push(s('text', { x: geo.X(z.x), y: geo.Y(z.y) + 6, class: `lb-arena__mult${dark ? ' lb-arena__mult--on-dark' : ''}`, 'text-anchor': 'middle' }, multLabel(z.mult)));
      }
      for (const z of state.hazards || []) {
        kids.push(s('g', { transform: `translate(${geo.X(z.x)},${geo.Y(z.y)})` },
          s('circle', { r: 12, fill: '#fff', stroke: '#d64545', 'stroke-width': 2 }),
          s('text', { y: 6, class: 'lb-arena__bang', 'text-anchor': 'middle' }, '!')));
      }
    }
    if (state.track && state.track.x && state.track.x.length > 1) {
      const pts = [];
      const step = Math.max(1, Math.floor(state.track.x.length / 1500));
      for (let i = 0; i < state.track.x.length; i += step) pts.push(`${geo.X(state.track.x[i]).toFixed(1)},${geo.Y(state.track.y[i]).toFixed(1)}`);
      const last = state.track.x.length - 1;
      pts.push(`${geo.X(state.track.x[last]).toFixed(1)},${geo.Y(state.track.y[last]).toFixed(1)}`);
      kids.push(s('polyline', { points: pts.join(' '), class: 'lb-arena__track', stroke: state.track.color || '#2a78d6' }));
      kids.push(s('circle', { cx: geo.X(state.track.x[last]), cy: geo.Y(state.track.y[last]), r: 6, fill: state.track.color || '#2a78d6', stroke: '#fff', 'stroke-width': 2 }));
    }
    if (state.base) {
      const bx = geo.X(state.base[0]);
      const by = geo.Y(state.base[1]);
      const k = thumb ? 9 : 8;
      kids.push(s('circle', { cx: bx, cy: by, r: 0.3 * U, class: 'lb-arena__base-zone' }));
      kids.push(s('rect', { x: bx - k, y: by - k, width: 2 * k, height: 2 * k, rx: 3, class: 'lb-arena__base' }));
      if (!thumb) kids.push(s('text', { x: bx, y: by - 0.3 * U - 7, class: 'lb-arena__caption', 'text-anchor': 'middle' }, 'База'));
    }
    // Образец — ромб, как в проигрывателе.
    (state.samples || []).forEach((p, i) => {
      const missed = state.collected && !state.collected.has(i);
      const k = thumb ? 10 : 7.5;
      kids.push(s('rect', {
        x: -k, y: -k, width: 2 * k, height: 2 * k, rx: 1.5, transform: `translate(${geo.X(p[0]).toFixed(1)},${geo.Y(p[1]).toFixed(1)}) rotate(45)`,
        class: `lb-arena__sample${missed ? ' lb-arena__sample--missed' : ''}`,
      }));
    });
    if (!thumb) {
      const y = geo.H - 16;
      kids.push(s('g', { class: 'lb-arena__scale' },
        s('path', { d: `M16 ${y - 5}V${y}H${16 + U}V${y - 5}` }),
        s('text', { x: 16 + U / 2, y: y - 6, 'text-anchor': 'middle' }, '1 м')));
    }
    dynamic.replaceChildren(...kids);
  }

  const el = h('div', { class: `lb-arena${thumb ? ' lb-arena--thumb' : ''}` }, svg);
  // svg и geo (X, Y — мир → рисунок) нужны страницам, которые рисуют поверх арены своё (конструктор исследования).
  return { el, render, svg, geo };
}

/** Пояснение к рисунку арены. */
export function arenaLegend(state, extra = []) {
  const mults = [...new Set((state.soils || []).map((z) => z.mult))].sort((a, b) => a - b);
  const items = [
    h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__sample' }), 'образец'),
    h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__base' }), 'база: старт и финиш'),
  ];
  if (mults.length) {
    items.push(h('span', { class: 'lb-key' },
      mults.map((m) => h('i', { class: 'lb-key__soil', style: { background: soilColor(m) }, title: multLabel(m) })),
      mults.length > 1
        ? `дорогой грунт: заряд уходит быстрее в ${multLabel(mults[0]).slice(1)}–${multLabel(mults[mults.length - 1]).slice(1)} раза`
        : `дорогой грунт: заряд уходит быстрее в ${multLabel(mults[0]).slice(1)} раза`));
  }
  if ((state.hazards || []).length) items.push(h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__hazard' }), 'опасная зона: штраф при въезде'));
  items.push(h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__pillar' }), 'столб'));
  return h('div', { class: 'lb-keys' }, items, extra);
}
