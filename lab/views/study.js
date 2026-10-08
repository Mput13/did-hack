// Конструктор исследования: слева задание простыми словами, справа карта арены с областью и участками,
// ниже — отчёт робота (lab/views/study_report.js). Сервер: GET /api/study/presets, POST /api/study/plan, POST /api/study.

import {
  h, s, fill, icon, api, loading, errorBox, select, getArena, getScenario, getTrace, href, num, ruText, LEVEL_ORDER, levelName,
} from './common.js';
import { createArena, arenaLegend } from './arena.js';
import { renderReport } from './study_report.js';

const MAX_SEED = 9999;
const Q_HINT = {
  soil_cost: 'Прямые пробеги в области и на контрольном участке',
  turn_cost: 'Развороты на месте и паузы для сравнения',
  idle_cost: 'Паузы: сколько уходит, когда робот стоит',
  load_effect: 'Нужен груз на борту — иначе робот честно откажется',
  sensor_law: 'Найти образец и слушать датчик на разных расстояниях',
};
const Q_NAME = {
  soil_cost: 'Цена пола в области', turn_cost: 'Цена поворота', idle_cost: 'Расход на месте', load_effect: 'Влияние груза',
  sensor_law: 'Закон датчика образцов',
};
const NEEDS_REGION = { soil_cost: 'нужна', turn_cost: 'по желанию', idle_cost: 'по желанию', sensor_law: 'по желанию', load_effect: null };
const ACTION_NAME = { straight: 'Прямые пробеги', pause: 'Паузы на месте', spin: 'Развороты на месте' };
const ACTION_DEFAULT = { straight: { length_m: 0.5, repeats: 2 }, pause: { seconds: 3, repeats: 1 }, spin: { angle_deg: 360, repeats: 2 } };

function addCss(url) {
  if (document.querySelector(`link[data-lb-css="${url}"]`)) return;
  document.head.append(h('link', { rel: 'stylesheet', href: url, 'data-lb-css': url }));
}

const clone = (v) => JSON.parse(JSON.stringify(v));
const round2 = (v) => Math.round(v * 100) / 100;

/** Задание с полями по умолчанию: форма всегда работает с полным объектом. */
function fullSpec(spec) {
  const out = {
    title: '', quantity: 'soil_cost', region: null, allowed: {}, controls: { max: 1, sites: [] },
    stop: { rel_error: 0.05, max_measurements: 16, time_s: 300, confidence: 0.9 }, budget: { energy: 30, reserve: 3 }, hypotheses: [],
    ...clone(spec || {}),
  };
  out.controls = { max: 1, sites: [], ...(out.controls || {}) };
  out.stop = { rel_error: 0.05, max_measurements: 16, time_s: 300, confidence: 0.9, ...(out.stop || {}) };
  out.budget = { energy: 30, reserve: 3, ...(out.budget || {}) };
  const allowed = Array.isArray(out.allowed) ? Object.fromEntries(out.allowed.map((k) => [k, {}])) : (out.allowed || {});
  out.allowed = {};
  for (const k of Object.keys(ACTION_DEFAULT)) if (allowed[k]) out.allowed[k] = { ...ACTION_DEFAULT[k], ...(allowed[k] === true ? {} : allowed[k]) };
  return out;
}

/** То, что уходит на сервер: без пустых полей. */
function wireSpec(spec) {
  const out = clone(spec);
  if (!out.region) delete out.region;
  if (!out.title) delete out.title;
  if (!out.hypotheses.length) delete out.hypotheses;
  return out;
}

function regionWords(r) {
  if (!r) return 'не задана';
  if (r.zone) return `область ${r.zone} этого сценария (робот знает только её контур)`;
  if (r.shape === 'circle') return `круг радиусом ${num(r.r, 2)} м, центр (${num(r.x, 1)}; ${num(r.y, 1)})`;
  return `прямоугольник ${num(r.w, 2)} × ${num(r.h, 2)} м, центр (${num(r.x, 1)}; ${num(r.y, 1)})`;
}

export async function render(root, ctx) {
  addCss('study.css');
  root.classList.add('st-root');
  root.append(loading('Загружаю конструктор…'));
  let arena;
  let meta;
  try {
    [arena, meta] = await Promise.all([getArena(), api.get('/api/study/presets')]);
  } catch (e) {
    root.replaceChildren(errorBox('Конструктор не открылся', `${e.message}. Если сервер лаборатории запущен давно, перезапустите его: pixi run lab`, ctx.reload));
    return;
  }
  if (!ctx.alive()) return;

  const presets = meta.presets || [];
  const first = presets.find((p) => p.id === ctx.query.preset) || presets[0];
  const state = {
    preset: first ? first.id : null,
    level: first ? first.level : 'medium',
    seed: first ? first.seed : 3,
    spec: fullSpec(first ? first.spec : null),
    scenario: null, plan: null, planError: '', truth: false, tool: null, draft: null,
    result: null, track: null, busy: false, token: 0,
  };
  let planTimer = null;
  let dropReport = null;

  // ---------- карта ----------
  const view = createArena(arena, { label: 'Арена: область исследования и участки для опытов' });
  const U = view.geo.X(1) - view.geo.X(0);
  const overlay = s('g');
  view.svg.append(overlay);
  const mapBox = h('div', { class: 'st-map' }, view.el);
  const mapHint = h('div', { class: 'st-map__hint' });
  const mapKeys = h('div');
  const stepsBox = h('div');
  const truthBox = h('input', { type: 'checkbox' });
  truthBox.addEventListener('change', () => { state.truth = truthBox.checked; paintMap(); paintReport(); });

  const toWorld = (e) => {
    const pt = view.svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const q = pt.matrixTransform(view.svg.getScreenCTM().inverse());
    return { x: (q.x - view.geo.X(0)) / U, y: (view.geo.Y(0) - q.y) / U };
  };
  const shapeNode = (r, cls) => (r.shape === 'circle'
    ? s('circle', { cx: view.geo.X(r.x), cy: view.geo.Y(r.y), r: r.r * U, class: cls })
    : s('rect', { x: view.geo.X(r.x - r.w / 2), y: view.geo.Y(r.y + r.h / 2), width: r.w * U, height: r.h * U, rx: 4, class: cls }));
  const siteMid = (site) => (site.kind === 'segment' ? [(site.a[0] + site.b[0]) / 2, (site.a[1] + site.b[1]) / 2] : [site.x, site.y]);
  /** Подпись у точки (x, y), отодвинутая прочь от точки away, чтобы подписи соседних участков не слипались. */
  const tag = (x, y, text, away) => {
    let dx = 0;
    let dy = 1;
    if (away) {
      const len = Math.hypot(x - away[0], y - away[1]) || 1;
      dx = (x - away[0]) / len;
      dy = (y - away[1]) / len;
    }
    const side = Math.abs(dx) > 0.5;
    return s('text', {
      x: view.geo.X(x) + (side ? Math.sign(dx) * 16 : 0), y: view.geo.Y(y) + (side ? 5 : dy >= 0 ? -14 : 24),
      class: 'st-tag', 'text-anchor': side ? (dx > 0 ? 'start' : 'end') : 'middle',
    }, text);
  };

  function siteNodes(site, role, label, away) {
    const cls = site.bad ? 'bad' : role;
    const [mx, my] = siteMid(site);
    if (site.kind === 'segment') {
      const [ax, ay] = site.a;
      const [bx, by] = site.b;
      const line = { x1: view.geo.X(ax), y1: view.geo.Y(ay), x2: view.geo.X(bx), y2: view.geo.Y(by) };
      return [s('line', { ...line, class: 'st-seg__ring' }), s('line', { ...line, class: `st-seg st-seg--${cls}` }),
        label ? tag(mx, my, label, away) : null];
    }
    return [s('circle', { cx: view.geo.X(site.x), cy: view.geo.Y(site.y), r: 11, class: `st-spot st-spot--${role}` }), label ? tag(mx, my, label, away) : null];
  }

  function paintMap() {
    const sc = state.scenario;
    const base = sc ? sc.base : arena.base;
    view.render(state.truth && sc ? { samples: sc.samples, soils: sc.soils, hazards: sc.hazards, base } : { base });
    const kids = [];
    const done = state.result ? state.result.study : null;
    const sites = done && done.plan ? done.plan.sites : state.plan ? state.plan.sites : null;
    const region = state.draft || (done && done.plan ? done.plan.region : state.plan ? state.plan.region : null)
      || (state.spec.region && state.spec.region.shape ? state.spec.region : null);
    if (state.track) {
      const pts = [];
      const step = Math.max(1, Math.floor(state.track.x.length / 900));
      for (let i = 0; i < state.track.x.length; i += step) pts.push(`${view.geo.X(state.track.x[i]).toFixed(1)},${view.geo.Y(state.track.y[i]).toFixed(1)}`);
      kids.push(s('polyline', { points: pts.join(' '), class: 'st-path' }));
    }
    if (region) kids.push(shapeNode(region, `st-region${state.draft ? ' st-region--draft' : ''}`));
    if (sites && !state.draft) {
      const shown = sites.test && sites.test.kind !== 'search' ? sites.test : null;
      const good = (sites.controls || []).find((c) => !c.bad) || (sites.controls || [])[0];
      (sites.controls || []).forEach((c) => kids.push(siteNodes(c, 'control', c.bad ? `${c.id}: не подошёл` : `контроль ${c.id}`, shown ? siteMid(shown) : null)));
      if (shown) kids.push(siteNodes(shown, 'test', shown.kind === 'segment' ? 'мерный участок' : 'место опытов', good ? siteMid(good) : null));
      if (sites.sample) {
        const [x, y] = sites.sample;
        kids.push(s('circle', { cx: view.geo.X(x), cy: view.geo.Y(y), r: 14, class: 'st-samplemark' }), tag(x, y, 'образец найден здесь'));
      }
    }
    if (!done) for (const p of state.spec.controls.sites) kids.push(s('rect', { x: view.geo.X(p.x) - 6, y: view.geo.Y(p.y) - 6, width: 12, height: 12, rx: 2, class: 'st-pin' }));
    if (done && state.spec.quantity === 'sensor_law') {
      for (const m of done.measurements || []) {
        if (m.kind === 'listen' && m.used) kids.push(s('circle', { cx: view.geo.X(m.x), cy: view.geo.Y(m.y), r: 5, fill: m.role === 'test' ? '#eb6834' : '#2a78d6', stroke: '#fff', 'stroke-width': 2 }));
      }
    }
    overlay.replaceChildren(...kids.flat().filter(Boolean));
    if (state.tool) mapBox.dataset.tool = state.tool; else delete mapBox.dataset.tool;
    mapHint.textContent = state.tool === 'control' ? 'Щёлкните на карте там, где поставить контрольный участок: на обычном полу, в стороне от области.'
      : state.tool ? `Нажмите на карте и тяните: ${state.tool === 'circle' ? 'от центра круга к его краю' : 'от одного угла прямоугольника к другому'}.`
        : done ? 'Серая линия — путь робота. Наведите на график ниже, чтобы увидеть каждый замер.'
          : 'Оранжевое — где робот будет мерить, синее — с чем сравнивать. Участки он выбирает сам, по карте стен.';
    const keys = [];
    if (region) keys.push(h('span', { class: 'st-key' }, h('i', { class: 'st-key__region' }), 'исследуемая область'));
    if (sites && sites.test && sites.test.kind !== 'search') keys.push(h('span', { class: 'st-key' }, h('i', { class: 'st-key__test' }), sites.test.kind === 'segment' ? 'мерный участок: здесь пойдут пробеги' : 'место опытов'));
    if (sites && (sites.controls || []).length) keys.push(h('span', { class: 'st-key' }, h('i', { class: 'st-key__control' }), 'контрольный участок'));
    if (state.track) keys.push(h('span', { class: 'st-key' }, h('i', { class: 'st-key__path' }), 'путь робота'));
    fill(mapKeys, h('div', { class: 'st-keys' }, keys), state.truth && sc ? arenaLegend(sc) : null);
  }

  // рисование области и контрольных точек на карте
  let drag = null;
  view.svg.addEventListener('pointerdown', (e) => {
    if (!state.tool || state.busy) return;
    const p = toWorld(e);
    if (state.tool === 'control') {
      const sites = state.spec.controls.sites;
      if (sites.length >= Math.max(1, state.spec.controls.max)) sites.shift();
      sites.push({ x: round2(p.x), y: round2(p.y) });
      if (state.spec.controls.max < 1) state.spec.controls.max = 1;
      state.tool = null;
      changed(true);
      return;
    }
    drag = { x: p.x, y: p.y };
    state.draft = { shape: state.tool, x: p.x, y: p.y, r: 0, w: 0, h: 0 };
    try { view.svg.setPointerCapture(e.pointerId); } catch { /* указатель уже отпущен — обойдёмся без захвата */ }
    e.preventDefault();
  });
  view.svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const p = toWorld(e);
    state.draft = state.tool === 'circle'
      ? { shape: 'circle', x: drag.x, y: drag.y, r: Math.hypot(p.x - drag.x, p.y - drag.y), w: 0, h: 0 }
      : { shape: 'rect', x: (drag.x + p.x) / 2, y: (drag.y + p.y) / 2, r: 0, w: Math.abs(p.x - drag.x), h: Math.abs(p.y - drag.y) };
    paintMap();
  });
  const endDrag = () => {
    if (!drag) return;
    const d = state.draft;
    const tool = state.tool;
    drag = null;
    state.draft = null;
    if (!d) return paintMap();
    // Одно нажатие без движения — область обычного размера вокруг этой точки.
    state.spec.region = tool === 'circle'
      ? { shape: 'circle', x: round2(d.x), y: round2(d.y), r: round2(d.r >= 0.15 ? d.r : 0.45) }
      : { shape: 'rect', x: round2(d.x), y: round2(d.y), w: round2(d.w >= 0.2 ? d.w : 0.9), h: round2(d.h >= 0.2 ? d.h : 0.9) };
    state.preset = null;
    changed(true);
  };
  view.svg.addEventListener('pointerup', endDrag);
  view.svg.addEventListener('pointercancel', () => { drag = null; state.draft = null; paintMap(); });

  // ---------- форма ----------
  const formBox = h('div', { class: 'lb-card st-form' });
  const problemsBox = h('div', { class: 'st-problems', 'aria-live': 'polite' });
  const costBox = h('div', { class: 'st-cost' });
  const runBtn = h('button', { class: 'lb-btn lb-btn--primary', type: 'button' }, icon('play'), 'Провести исследование');
  const runNote = h('span', { class: 'lb-muted' });
  const presetBtns = presets.map((p) => {
    const btn = h('button', { class: 'st-preset', type: 'button', title: p.text },
      h('span', { class: 'st-preset__name', text: p.title }), h('span', { class: 'st-preset__text', text: p.text }));
    btn.addEventListener('click', () => {
      state.preset = p.id;
      state.level = p.level;
      state.seed = p.seed;
      state.spec = fullSpec(p.spec);
      state.tool = null;
      history.replaceState(null, '', href('/study', { preset: p.id }));
      paintForm();
      loadScenario();
    });
    return { id: p.id, btn };
  });
  const reportBox = h('section', { class: 'st-report', 'aria-live': 'polite' });

  const numInput = (value, attrs, set) => {
    const el = h('input', { class: `lb-input st-num${attrs.wide ? ' st-num--wide' : ''}`, type: 'number', value, ...attrs, wide: null });
    el.addEventListener('input', () => {
      const v = parseFloat(el.value.replace(',', '.'));
      if (Number.isFinite(v)) { set(v); state.preset = null; changed(false); }
    });
    return el;
  };
  const sec = (n, title, hint, ...kids) => h('div', { class: 'st-sec' },
    h('div', { class: 'st-sec__head' }, h('span', { class: 'st-sec__n', text: String(n) }), h('span', { class: 'st-sec__title', text: title }),
      hint ? h('span', { class: 'st-sec__hint', text: hint }) : null), kids);

  function paintForm() {
    const spec = state.spec;
    const q = spec.quantity;
    const law = q === 'sensor_law';
    presetBtns.forEach((p) => p.btn.setAttribute('aria-pressed', String(p.id === state.preset)));

    // 1. что исследуем
    const qs = (meta.quantities || []).map((x) => {
      const btn = h('button', { class: 'st-q', type: 'button', 'aria-pressed': String(x.id === q) },
        h('span', { class: 'st-q__name', text: Q_NAME[x.id] || x.title }), h('span', { class: 'st-q__text', text: Q_HINT[x.id] || x.label }));
      btn.addEventListener('click', () => {
        if (spec.quantity === x.id) return;
        spec.quantity = x.id;
        spec.hypotheses = [];
        spec.allowed = {};
        const need = x.needs === 'straight' ? ['straight', 'pause'] : x.needs === 'spin' ? ['spin', 'pause'] : x.id === 'sensor_law' ? ['straight', 'pause'] : ['pause'];
        need.forEach((k) => { spec.allowed[k] = { ...ACTION_DEFAULT[k] }; });
        if (x.id === 'sensor_law') spec.allowed.pause.seconds = 2;
        if (x.id === 'idle_cost') { spec.allowed.pause = { seconds: 6, repeats: 3 }; spec.controls.max = 0; }
        if (!NEEDS_REGION[x.id]) spec.region = null;
        state.preset = null;
        state.tool = null;
        paintForm();
        changed(true);
      });
      return btn;
    });

    // 2. где
    const levels = LEVEL_ORDER.map((id) => ({ value: id, label: levelName(id) }));
    const levelSel = select(levels, state.level, (v) => { state.level = v; state.preset = null; loadScenario(); }, { 'aria-label': 'Уровень сценария' });
    const seedIn = h('input', { class: 'lb-input st-num', type: 'number', min: 1, max: MAX_SEED, step: 1, value: state.seed, 'aria-label': 'Номер сценария' });
    seedIn.addEventListener('change', () => { state.seed = Math.min(MAX_SEED, Math.max(1, parseInt(seedIn.value, 10) || 1)); state.preset = null; loadScenario(); });
    const zones = (state.plan && state.plan.zones) || (state.scenario ? state.scenario.soils.map((z) => z.id) : []);
    const toolBtn = (id, text) => {
      const on = state.tool === id || (id === 'zone' && spec.region && spec.region.zone && !state.tool);
      const btn = h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(!!on), text });
      btn.addEventListener('click', () => {
        if (id === 'zone') {
          state.tool = null;
          spec.region = { zone: (spec.region && spec.region.zone) || zones[0] || 'A' };
          state.preset = null;
          paintForm();
          return changed(true);
        }
        state.tool = state.tool === id ? null : id;
        paintForm();
        paintMap();
      });
      return btn;
    };
    const regionRow = NEEDS_REGION[q] ? [
      h('div', { class: 'st-row' },
        h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Как задать область' }, toolBtn('circle', 'Обвести круг'), toolBtn('rect', 'Обвести прямоугольник'), toolBtn('zone', 'Область сценария')),
        spec.region && spec.region.zone ? select(zones.map((z) => ({ value: z, label: `область ${z}` })), spec.region.zone, (v) => { spec.region = { zone: v }; state.preset = null; changed(true); }, { 'aria-label': 'Область сценария' }) : null),
      h('div', { class: 'st-readout' }, h('span', { text: `Область: ${regionWords(spec.region)}. ` }),
        spec.region && NEEDS_REGION[q] !== 'нужна'
          ? h('button', { class: 'st-linkbtn', type: 'button', onclick: () => { spec.region = null; state.preset = null; paintForm(); changed(true); } }, 'Убрать')
          : null),
    ] : [h('div', { class: 'st-readout', text: 'Для этого вопроса область не нужна: робот меряет на обычном полу.' })];

    // 3. опыты
    const actionRow = (kind, fields) => {
      const on = !!spec.allowed[kind];
      const box = h('input', { type: 'checkbox', checked: on });
      box.addEventListener('change', () => {
        if (box.checked) spec.allowed[kind] = { ...ACTION_DEFAULT[kind] }; else delete spec.allowed[kind];
        state.preset = null;
        paintForm();
        changed(true);
      });
      const a = spec.allowed[kind] || ACTION_DEFAULT[kind];
      return h('div', { class: 'st-row' },
        h('label', { class: 'st-check' }, box, ACTION_NAME[kind]),
        h('span', { class: 'st-params', 'data-off': on ? null : '' }, fields.map(([key, before, after, attrs]) => [
          h('span', { text: before }), numInput(a[key], attrs, (v) => { if (spec.allowed[kind]) spec.allowed[kind][key] = v; }), h('span', { text: after })])));
    };

    // 4. контроль
    const sites = spec.controls.sites;
    const controlRows = (q === 'soil_cost' || q === 'load_effect' || q === 'idle_cost') ? [
      h('div', { class: 'st-row' }, 'Не больше', numInput(spec.controls.max, { min: 0, max: 4, step: 1, 'aria-label': 'Сколько контрольных участков' }, (v) => {
        spec.controls.max = Math.round(v);
        spec.controls.sites = spec.controls.sites.slice(0, spec.controls.max);
      }), 'участков.',
      h('button', { class: 'lb-btn', type: 'button', 'aria-pressed': String(state.tool === 'control'), onclick: () => { state.tool = state.tool === 'control' ? null : 'control'; paintForm(); paintMap(); } }, 'Указать на карте')),
      h('div', { class: 'st-readout' }, sites.length
        ? [`Заданы вами: ${sites.map((p) => `(${num(p.x, 1)}; ${num(p.y, 1)})`).join(', ')}. `,
          h('button', { class: 'st-linkbtn', type: 'button', onclick: () => { spec.controls.sites = []; state.preset = null; paintForm(); changed(true); } }, 'Пусть выберет сам')]
        : 'Робот выберет сам: на проверенном полу у дороги к области — по ней он проедет и увидит, что пол там обычный.'),
    ] : [h('div', { class: 'st-readout', text: q === 'turn_cost' ? 'Контроль здесь — паузы на том же месте: они показывают, сколько уходит без поворота.'
      : 'Контроль здесь — опорная точка у самого образца: её повтор в конце показывает, не сбился ли датчик.' })];

    // 5–6. цель и бюджет
    const goal = law
      ? h('div', { class: 'st-row' }, 'Считать объяснение доказанным с вероятности', numInput(Math.round(spec.stop.confidence * 100), { min: 60, max: 99, step: 1, 'aria-label': 'Уверенность, %' }, (v) => { spec.stop.confidence = v / 100; }), '%')
      : h('div', { class: 'st-row' }, 'Погрешность не больше ±', numInput(round2(spec.stop.rel_error * 100), { min: 0.1, max: 100, step: 0.5, 'aria-label': 'Требуемая точность, %' }, (v) => { spec.stop.rel_error = v / 100; }), '% от оценки (95% интервал)');
    const budget = [
      h('div', { class: 'st-row' }, 'Заряд на всё исследование', numInput(spec.budget.energy, { min: 1, max: 60, step: 1, 'aria-label': 'Бюджет заряда' }, (v) => { spec.budget.energy = v; }), 'ед. (в батарее 60)'),
      h('div', { class: 'st-row' }, 'Запас на возврат', numInput(spec.budget.reserve, { min: 0, max: 30, step: 0.5, 'aria-label': 'Запас на возврат' }, (v) => { spec.budget.reserve = v; }), 'ед. сверх дороги домой'),
      h('div', { class: 'st-row' }, 'Не больше', numInput(spec.stop.max_measurements, { min: 1, max: 60, step: 1, 'aria-label': 'Максимум замеров' }, (v) => { spec.stop.max_measurements = Math.round(v); }), 'замеров и',
        numInput(spec.stop.time_s, { min: 20, max: 600, step: 10, 'aria-label': 'Лимит времени, с' }, (v) => { spec.stop.time_s = v; }), 'секунд'),
    ];

    // 7. объяснения
    const hyps = spec.hypotheses;
    const lawOpts = [{ value: 'linear', label: 'линейно' }, { value: 'faster', label: 'быстрее линейного' }, { value: 'slower', label: 'медленнее линейного' }];
    const hypRows = hyps.map((x, i) => {
      const text = h('input', { class: 'lb-input st-hyp__text', type: 'text', value: x.statement, 'aria-label': `Объяснение ${i + 1}` });
      text.addEventListener('input', () => { x.statement = text.value; state.preset = null; changed(false); });
      const del = h('button', { class: 'lb-btn lb-btn--icon', type: 'button', title: 'Убрать объяснение', 'aria-label': 'Убрать объяснение', onclick: () => { hyps.splice(i, 1); state.preset = null; paintForm(); changed(true); } }, icon('cross'));
      return law
        ? [text, select(lawOpts, x.law || 'linear', (v) => { x.law = v; state.preset = null; changed(true); }, { 'aria-label': 'Форма зависимости' }),
          h('span', { class: 'st-params' }, 'до', numInput(x.range_m ?? 2, { min: 0.3, max: 6, step: 0.1, 'aria-label': 'Дальность, м' }, (v) => { x.range_m = v; }), 'м'), del]
        : [text, h('span', { class: 'st-params', text: 'предсказывает' }), numInput(x.value ?? 1, { step: 0.1, 'aria-label': 'Предсказанное значение' }, (v) => { x.value = v; }), del];
    });
    const addHyp = h('button', { class: 'lb-btn', type: 'button' }, 'Добавить объяснение');
    addHyp.addEventListener('click', () => {
      if (law && !hyps.length) hyps.push(...clone(meta.laws || []));
      else hyps.push(law ? { id: `h${Date.now() % 100000}`, statement: 'моё объяснение', law: 'linear', range_m: 2.5, prior: 0.2 }
        : { id: `h${Date.now() % 100000}`, statement: 'моё объяснение', value: 1 });
      state.preset = null;
      paintForm();
      changed(true);
    });
    const hypSec = h('div', { class: 'st-sec' },
      h('div', { class: 'st-sec__head' }, h('span', { class: 'st-sec__n', text: '7' }), h('span', { class: 'st-sec__title', text: 'Объяснения для сравнения' }),
        h('span', { class: 'st-sec__hint', text: law ? 'между ними робот и будет выбирать' : 'по желанию' })),
      hyps.length ? h('div', { class: 'st-hyp' }, hypRows)
        : h('div', { class: 'st-readout', text: law ? 'Не заданы — робот возьмёт пять стандартных: линейно до 2 м, до 1,5 м, до 3 м, быстрее и медленнее линейного.'
          : 'Не заданы — робот просто назовёт число с погрешностью. Добавьте варианты («вдвое дороже», «втрое дороже»), и он скажет, какой подтвердился.' }),
      h('div', { class: 'st-row' }, addHyp));

    const actions = [
      actionRow('straight', [['length_m', 'по', 'м,', { min: 0.25, max: 2, step: 0.05, 'aria-label': 'Длина пробега, м' }], ['repeats', 'не меньше', 'на участок', { min: 1, max: 12, step: 1, 'aria-label': 'Повторов пробега' }]]),
      actionRow('pause', [['seconds', 'по', 'с,', { min: 1.5, max: 30, step: 0.5, 'aria-label': 'Длительность паузы, с' }], ['repeats', 'не меньше', 'раз', { min: 1, max: 12, step: 1, 'aria-label': 'Повторов паузы' }]]),
      actionRow('spin', [['angle_deg', 'на', '°,', { min: 90, max: 1440, step: 90, 'aria-label': 'Угол разворота' }], ['repeats', 'не меньше', 'раз', { min: 1, max: 12, step: 1, 'aria-label': 'Повторов разворота' }]]),
    ];

    fill(formBox,
      sec(1, 'Что исследуем', null, h('div', { class: 'st-qs' }, qs)),
      sec(2, 'Где', NEEDS_REGION[q] ? `область ${NEEDS_REGION[q]}` : null,
        h('div', { class: 'st-row' }, 'Сценарий:', levelSel, 'уровень, №', seedIn), regionRow),
      sec(3, 'Какие опыты разрешены', 'чего нет в списке — того робот не делает', actions),
      sec(4, 'С чем сравнивать', null, controlRows),
      sec(5, law ? 'Когда хватит' : 'Какая точность нужна', null, goal),
      sec(6, 'Сколько можно потратить', null, budget),
      hypSec,
      problemsBox, costBox,
      h('div', { class: 'st-go' }, runBtn, runNote));
    paintProblems();
  }

  function paintProblems() {
    const plan = state.plan;
    const list = plan ? plan.problems.map((p) => (typeof p === 'string' ? { level: 'error', text: p } : p)) : [];
    fill(problemsBox,
      state.planError ? h('div', { class: 'st-problem', data: { level: 'error' } }, icon('warn', 18), h('span', { text: state.planError })) : null,
      list.map((p) => h('div', { class: 'st-problem', data: { level: p.level } }, icon(p.level === 'error' ? 'cross' : 'warn', 18),
        h('span', { text: `${p.level === 'error' ? 'Так не получится: ' : 'Обратите внимание: '}${ruText(p.text)}` }))));
    const c = plan && plan.costs;
    costBox.textContent = c && c.budget != null && plan.ok
      ? `Прикидка до старта: дорога туда и обратно около ${num(c.road, 1)} ед., замеры ${c.measure_min === c.measure_max ? `около ${num(c.measure_min, 1)}` : `от ${num(c.measure_min, 1)} до ${num(c.measure_max, 1)}`} ед., запас ${num(c.reserve, 1)} ед. — при бюджете ${num(c.budget, 0)} ед. Не меньше ${num(c.seconds_min, 0)} с.`
      : '';
    const ok = !!(plan && plan.ok) && !state.planError;
    runBtn.disabled = !ok || state.busy;
    runBtn.classList.toggle('lb-btn--busy', state.busy);
    runNote.textContent = state.busy ? 'Робот работает в быстром симуляторе…' : ok ? 'Прогон в быстром симуляторе занимает около секунды.' : 'Сначала исправьте задание.';
    fill(stepsBox, plan && plan.steps && plan.steps.length && !state.result
      ? [h('div', { class: 'lb-eyebrow', text: 'План робота' }), h('ol', { class: 'st-steps' }, plan.steps.map((t) => h('li', { text: ruText(t) })))] : null);
  }

  async function check() {
    const token = ++state.token;
    try {
      const plan = await api.post('/api/study/plan', { level: state.level, seed: state.seed, spec: wireSpec(state.spec) });
      if (token !== state.token || !ctx.alive()) return;
      state.plan = plan;
      state.planError = '';
    } catch (e) {
      if (token !== state.token || !ctx.alive()) return;
      state.plan = null;
      state.planError = e.status === 400 ? ruText(e.message) : `Сервер не проверил задание: ${e.message}`;
    }
    paintProblems();
    paintMap();
  }

  /** Задание изменилось: прежний отчёт уже не про него. now — проверить сразу, иначе после паузы в наборе. */
  function changed(now) {
    if (state.result) {
      state.result = null;
      state.track = null;
      paintReport();
    }
    clearTimeout(planTimer);
    planTimer = setTimeout(check, now ? 0 : 350);
    if (now) { paintForm(); paintMap(); }
  }

  async function loadScenario() {
    state.result = null;
    state.track = null;
    state.plan = null;
    paintReport();
    try {
      state.scenario = await getScenario(state.level, state.seed);
    } catch {
      state.scenario = null;
    }
    if (!ctx.alive()) return;
    paintForm();
    paintMap();
    await check();
    paintForm();
  }

  function paintReport() {
    if (dropReport) { dropReport(); dropReport = null; }
    if (!state.result) return fill(reportBox);
    dropReport = renderReport(reportBox, { summary: state.result, truth: state.truth });
  }

  runBtn.addEventListener('click', async () => {
    if (state.busy) return;
    clearTimeout(planTimer);
    state.busy = true;
    state.tool = null;
    paintProblems();
    try {
      const summary = await api.post('/api/study', { level: state.level, seed: state.seed, spec: wireSpec(state.spec) });
      if (!ctx.alive()) return;
      state.result = summary;
      state.track = null;
      if (summary.file) getTrace(summary.file).then((tr) => { if (ctx.alive() && state.result === summary) { state.track = tr.track; paintMap(); } }).catch(() => {});
    } catch (e) {
      if (!ctx.alive()) return;
      state.planError = e.status === 400 ? ruText(e.message) : `Исследование не запустилось: ${e.message}`;
    }
    state.busy = false;
    paintProblems();
    paintMap();
    paintReport();
    if (state.result) reportBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  ctx.onLeave(() => { clearTimeout(planTimer); state.token += 1; if (dropReport) dropReport(); });

  fill(root,
    h('header', { class: 'lb-intro lb-intro--tight' },
      h('div', { class: 'lb-eyebrow', text: 'Конструктор исследования' }),
      h('h1', { class: 'lb-h1', text: 'Задайте вопрос — робот поставит опыт' }),
      h('p', { class: 'lb-lead', text: 'Вы решаете, что измерить, какими опытами и сколько заряда на это можно потратить. Робот сам выбирает участки, чередует замеры с контрольными, считает оценку с погрешностью и останавливается, когда точности хватает или бюджет кончился.' })),
    h('div', { class: 'st-presets', role: 'group', 'aria-label': 'Готовые задания' }, presetBtns.map((p) => p.btn)),
    h('div', { class: 'st-grid' },
      formBox,
      h('div', { class: 'lb-card st-mapcard' },
        h('div', { class: 'st-mapcard__head' },
          h('div', { class: 'lb-eyebrow', text: 'Арена' }),
          h('label', { class: 'st-switch' }, truthBox, 'Показать скрытую правду')),
        mapBox, mapHint, mapKeys, stepsBox)),
    reportBox);
  paintForm();
  paintMap();
  await loadScenario();
}
