/* Страница-объяснение: рисунки, проигрыватель и графики. Данные встроены в window.DID. */
(() => {
  'use strict';
  const D = window.DID;
  const A = D.arena;
  const RULES = D.rules;
  const C = {
    ink: '#14181c', ink2: '#4a535c', ink3: '#8a939b', line: '#e3e6e9', surface: '#ffffff', wall: '#4a535c',
    fixed: '#2a78d6', adaptive: '#eb6834', third: '#1baf7a', red: '#d64545', sample: '#f2b705',
    belief: [42, 120, 214],
  };
  const ARM_COLOR = { fixed: C.fixed, adaptive: C.adaptive, fastsim: C.fixed, gazebo: C.third,
    rule: C.fixed, llm: C.adaptive, llm_faulty: C.third, scientist: C.third, scientist_mem: C.adaptive, odom: C.fixed, lidar: C.adaptive };
  const MODE_RU = { start: 'старт', explore: 'разведка', travel: 'в пути', approach: 'подход к образцу',
    collect: 'сбор', return: 'возврат на базу', think: 'думает', escape: 'отъезд назад', done: 'финиш' };
  const EVENT_RU = { sample_collected: 'образец собран', collision: 'столкновение', false_collect: 'ложный сбор',
    hazard_hit: 'въезд в опасную зону', soil_change: 'грунты изменились', new_hazard: 'появилась опасная зона',
    sensor_fault: 'датчик начал шуметь', sensor_recovered: 'датчик пришёл в норму' };
  const KIND_RU = { decision: 'решение', hypothesis: 'гипотеза', verdict: 'вывод', alarm: 'тревога',
    action: 'действие', observe: 'наблюдение', llm: 'модель' };
  const LEVEL_RU = { easy: 'easy', medium: 'medium', hard: 'hard', all: 'все уровни' };

  const num = (v, d = 1) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(d).replace('.', ',');
  const pct = v => v == null ? '—' : Math.round(v * 100) + '%';
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const $ = id => document.getElementById(id);

  /* ------------------------------------------------------------------ арена */

  const VIEW = { x0: -2.95, x1: 2.85, y0: -2.75, y1: 2.8 };
  const ASPECT = (VIEW.y1 - VIEW.y0) / (VIEW.x1 - VIEW.x0);
  const freeGrid = A.free.map(row => Array.from(row, ch => ch === '1'));
  const isFree = (x, y) => {
    const ix = Math.floor((x - A.x0) / A.res), iy = Math.floor((y - A.y0) / A.res);
    return ix >= 0 && iy >= 0 && ix < A.w && iy < A.h && freeGrid[iy][ix];
  };

  // Пол и стены одной картинкой: свободные клетки белые, преграды рядом с ними тёмные.
  const floorImg = (() => {
    const k = 4, cv = document.createElement('canvas');
    cv.width = A.w * k; cv.height = A.h * k;
    const g = cv.getContext('2d');
    for (let iy = 0; iy < A.h; iy++) for (let ix = 0; ix < A.w; ix++) {
      if (freeGrid[iy][ix]) { g.fillStyle = '#ffffff'; g.fillRect(ix * k, (A.h - 1 - iy) * k, k, k); continue; }
      let near = false;
      for (let dy = -3; dy <= 3 && !near; dy++) for (let dx = -3; dx <= 3; dx++) {
        const jy = iy + dy, jx = ix + dx;
        if (jy >= 0 && jx >= 0 && jy < A.h && jx < A.w && freeGrid[jy][jx]) { near = true; break; }
      }
      if (near) { g.fillStyle = C.wall; g.fillRect(ix * k, (A.h - 1 - iy) * k, k, k); }
    }
    return cv;
  })();

  /** Холст под размер контейнера. Возвращает функции перевода мировых координат в пиксели. */
  function stage(cv) {
    const cssW = cv.clientWidth || cv.parentElement.clientWidth || 400;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(cssW * dpr), h = Math.round(cssW * ASPECT * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    cv.style.height = cssW * ASPECT + 'px';
    const s = w / (VIEW.x1 - VIEW.x0);
    const g = cv.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, s, dpr, w, h, X: x => (x - VIEW.x0) * s, Y: y => (VIEW.y1 - y) * s,
      wx: px => VIEW.x0 + px * dpr / s, wy: py => VIEW.y1 - py * dpr / s };
  }
  function drawFloor(T, alpha = 1) {
    const { g } = T;
    g.globalAlpha = alpha;
    g.imageSmoothingEnabled = true;
    g.drawImage(floorImg, T.X(A.x0), T.Y(A.y0 + A.h * A.res), A.w * A.res * T.s, A.h * A.res * T.s);
    g.globalAlpha = 1;
  }
  const soilColor = m => m >= 3.6 ? '#b8481b' : m >= 2.75 ? '#eb6834' : m >= 2.25 ? '#ee824f' : m >= 1.75 ? '#f09d78' : '#f6c3ae';
  function zonePath(T, z, grow = 0) {
    const { g } = T;
    g.beginPath();
    if (z.shape === 'rect') g.rect(T.X(z.x - z.w / 2 - grow), T.Y(z.y + z.h / 2 + grow), (z.w + 2 * grow) * T.s, (z.h + 2 * grow) * T.s);
    else g.arc(T.X(z.x), T.Y(z.y), (z.r + grow) * T.s, 0, 2 * Math.PI);
  }
  function label(T, text, x, y, opts = {}) {
    const { g } = T;
    g.font = `${opts.bold ? 600 : 500} ${Math.round((opts.size || 12) * T.dpr)}px -apple-system, "Segoe UI", Roboto, Arial, sans-serif`;
    g.textAlign = opts.align || 'center'; g.textBaseline = opts.base || 'middle';
    if (opts.halo !== false) { g.lineWidth = 3 * T.dpr; g.strokeStyle = 'rgba(255,255,255,.9)'; g.strokeText(text, x, y); }
    g.fillStyle = opts.color || C.ink; g.fillText(text, x, y);
  }
  function drawSoils(T, soils) {
    const { g } = T;
    for (const z of soils) {
      zonePath(T, z); g.fillStyle = soilColor(z.mult); g.globalAlpha = 0.5; g.fill(); g.globalAlpha = 1;
      g.lineWidth = 1.5 * T.dpr; g.strokeStyle = soilColor(z.mult); g.stroke();
      label(T, '×' + num(z.mult, z.mult % 1 ? 1 : 0), T.X(z.x), T.Y(z.y), { size: 14, bold: true });
    }
  }
  function drawHazards(T, hazards) {
    const { g } = T;
    for (const z of hazards) {
      zonePath(T, z); g.fillStyle = 'rgba(214,69,69,.16)'; g.fill();
      g.lineWidth = 2 * T.dpr; g.strokeStyle = C.red; g.stroke();
      label(T, '!', T.X(z.x), T.Y(z.y), { size: 14, bold: true, color: C.red });
    }
  }
  function drawSamples(T, samples, collected = new Set()) {
    const { g } = T;
    samples.forEach((p, i) => {
      const x = T.X(p[0]), y = T.Y(p[1]), r = 5.5 * T.dpr;
      g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI);
      if (collected.has(i)) { g.fillStyle = '#fff'; g.fill(); g.lineWidth = 1.5 * T.dpr; g.strokeStyle = C.ink3; g.stroke();
        g.beginPath(); g.moveTo(x - r * 0.5, y); g.lineTo(x - r * 0.1, y + r * 0.4); g.lineTo(x + r * 0.55, y - r * 0.4); g.strokeStyle = C.ink3; g.stroke();
      } else { g.fillStyle = C.sample; g.fill(); g.lineWidth = 1.5 * T.dpr; g.strokeStyle = C.ink; g.stroke(); }
    });
  }
  function drawBase(T, base = A.base) {
    const { g } = T, x = T.X(base[0]), y = T.Y(base[1]), r = 7 * T.dpr;
    g.fillStyle = '#fff'; g.strokeStyle = C.ink; g.lineWidth = 2 * T.dpr;
    g.beginPath(); g.rect(x - r, y - r, 2 * r, 2 * r); g.fill(); g.stroke();
    label(T, 'база', x, y + Math.max(r + 9 * T.dpr, 0.105 * T.s + 10 * T.dpr), { size: 12, color: C.ink2 });
  }
  function drawRobot(T, x, y, th, color = C.ink) {
    const { g } = T, r = Math.max(0.105 * T.s, 5 * T.dpr);
    g.beginPath(); g.arc(T.X(x), T.Y(y), r, 0, 2 * Math.PI); g.fillStyle = color; g.fill();
    g.lineWidth = 2 * T.dpr; g.strokeStyle = '#fff'; g.stroke();
    g.beginPath(); g.moveTo(T.X(x), T.Y(y)); g.lineTo(T.X(x + 0.2 * Math.cos(th)), T.Y(y + 0.2 * Math.sin(th)));
    g.lineWidth = 2.5 * T.dpr; g.strokeStyle = color; g.stroke();
  }
  function polyline(T, pts, color, width = 2, dash) {
    if (!pts.length) return;
    const { g } = T;
    g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(T.X(p[0]), T.Y(p[1])) : g.moveTo(T.X(p[0]), T.Y(p[1]))));
    g.lineWidth = width * T.dpr; g.strokeStyle = color; g.lineJoin = 'round'; g.lineCap = 'round';
    g.setLineDash(dash ? dash.map(v => v * T.dpr) : []); g.stroke(); g.setLineDash([]);
  }
  function raycast(x, y, n = 180, rmax = 3.5) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = i * 2 * Math.PI / n, cx = Math.cos(a), cy = Math.sin(a);
      let r = 0.02; while (r < rmax && isFree(x + cx * r, y + cy * r)) r += 0.02;
      out.push([a, r < rmax ? r : null]);
    }
    return out;
  }
  const redrawers = [];
  const onResize = fn => { redrawers.push(fn); fn(); };
  let resizeTimer = null;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => redrawers.forEach(f => f()), 120); });

  /* ------------------------------------------------- состояние прогона во времени */

  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  function lastIdx(arr, t, key) {       // последний элемент с временем <= t
    let lo = 0, hi = arr.length - 1, ans = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; const v = key ? arr[m][key] : arr[m]; if (v <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
    return ans;
  }
  function prep(tr) {
    if (tr._ready) return tr;
    tr._ready = true;
    tr._end = tr.track.t[tr.track.t.length - 1];
    tr._heat = {};
    return tr;
  }
  function truthAt(tr, t) {
    const sc = tr.scenario;
    let soils = sc.soils, hazards = sc.hazards.slice(), fault = false;
    for (const w of tr.world) {
      if (w.t > t) break;
      const ev = sc.events.find(e => e.type === w.type);
      if (w.type === 'soil_change' && ev) soils = ev.soils;
      if (w.type === 'new_hazard' && ev) hazards.push(ev.zone);
      if (w.type === 'sensor_fault') fault = true;
      if (w.type === 'sensor_recovered') fault = false;
    }
    const collected = new Set(tr.events.filter(e => e.type === 'sample_collected' && e.t <= t).map(e => e.sample));
    return { soils, hazards, fault, collected };
  }
  function heatCanvas(tr, kind, i) {
    const key = kind + i;
    if (tr._heat[key]) return tr._heat[key];
    const grid = tr[kind], raw = b64(grid.snaps[i].data);
    const cv = document.createElement('canvas'); cv.width = grid.w; cv.height = grid.h;
    const g = cv.getContext('2d'), img = g.createImageData(grid.w, grid.h);
    for (let iy = 0; iy < grid.h; iy++) for (let ix = 0; ix < grid.w; ix++) {
      const v = raw[iy * grid.w + ix], o = ((grid.h - 1 - iy) * grid.w + ix) * 4;
      if (kind === 'belief') {       // v/255 = корень из вероятности: слабые места тоже видны
        img.data[o] = C.belief[0]; img.data[o + 1] = C.belief[1]; img.data[o + 2] = C.belief[2];
        img.data[o + 3] = Math.min(235, v * 1.25);
      } else {                        // оценка множителя расхода: v/40
        const m = v / 40;
        img.data[o] = 90; img.data[o + 1] = 60; img.data[o + 2] = 40;
        img.data[o + 3] = v === 0 ? 0 : Math.max(0, Math.min(170, (m - 1.15) * 75));
      }
    }
    g.putImageData(img, 0, 0);
    return (tr._heat[key] = cv);
  }
  function drawHeat(T, tr, kind, t, smooth = true) {
    const grid = tr[kind];
    if (!grid || !grid.snaps.length) return;
    const i = Math.max(0, lastIdx(grid.snaps, t, 't'));
    const { g } = T;
    g.imageSmoothingEnabled = smooth;
    g.drawImage(heatCanvas(tr, kind, i), T.X(grid.x0), T.Y(grid.y0 + grid.h * grid.res), grid.w * grid.res * T.s, grid.h * grid.res * T.s);
  }

  /** Кадр прогона в момент t. layers: {truth, belief, soil, path, lidar, ring}. */
  function drawFrame(cv, tr, t, layers, color) {
    const T = stage(cv), { g } = T, tk = tr.track;
    const i = Math.max(0, lastIdx(tk.t, t));
    drawFloor(T);
    const truth = truthAt(tr, t);
    if (layers.soil) drawHeat(T, tr, 'soil', t, false);
    if (layers.belief) drawHeat(T, tr, 'belief', t);
    if (layers.truth) { drawSoils(T, truth.soils); drawHazards(T, truth.hazards); }
    if (layers.belief || layers.soil) {
      for (const h of tr.hazards) if (h.t <= t) {
        g.beginPath(); g.arc(T.X(h.x), T.Y(h.y), h.r * T.s, 0, 2 * Math.PI);
        g.setLineDash([5 * T.dpr, 4 * T.dpr]); g.lineWidth = 2 * T.dpr; g.strokeStyle = C.ink; g.stroke(); g.setLineDash([]);
      }
    }
    drawBase(T, tr.scenario.base);
    if (layers.truth) drawSamples(T, tr.scenario.samples, truth.collected);
    if (layers.path) {
      const p = tr.paths[lastIdx(tr.paths, t, 't')];
      if (p && tr.modes[tk.mode[i]] !== 'done') {
        polyline(T, [[tk.x[i], tk.y[i]], ...p.pts.filter((q, k) => k > 0)], 'rgba(20,24,28,.45)', 1.5, [3, 4]);
        g.beginPath(); g.arc(T.X(p.goal[0]), T.Y(p.goal[1]), 4 * T.dpr, 0, 2 * Math.PI); g.strokeStyle = C.ink; g.lineWidth = 1.5 * T.dpr; g.stroke();
      }
    }
    const trail = []; for (let k = 0; k <= i; k++) trail.push([tk.x[k], tk.y[k]]);
    polyline(T, trail, color, 2.2);
    for (const e of tr.events) if (e.t <= t && e.type !== 'sample_collected') {
      const x = T.X(e.x), y = T.Y(e.y), r = 5 * T.dpr;
      g.strokeStyle = C.red; g.lineWidth = 2.2 * T.dpr; g.beginPath();
      g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.stroke();
    }
    if (layers.lidar && tr.scans.length) {
      const s = tr.scans[Math.max(0, lastIdx(tr.scans, t, 't'))], n = s.r.length;
      g.strokeStyle = 'rgba(27,175,122,.35)'; g.lineWidth = 1 * T.dpr; g.beginPath();
      for (let k = 0; k < n; k++) if (s.r[k] > 0) {
        const a = s.th + k * 2 * Math.PI / n, d = s.r[k] / 100;
        g.moveTo(T.X(s.x), T.Y(s.y)); g.lineTo(T.X(s.x + d * Math.cos(a)), T.Y(s.y + d * Math.sin(a)));
      }
      g.stroke();
    }
    if (layers.ring && tk.sensor[i] > 0.04 && tr.modes[tk.mode[i]] !== 'done') {
      g.beginPath(); g.arc(T.X(tk.x[i]), T.Y(tk.y[i]), (1 - tk.sensor[i]) * RULES.sensor_range_m * T.s, 0, 2 * Math.PI);
      g.setLineDash([2 * T.dpr, 5 * T.dpr]); g.lineWidth = 1.5 * T.dpr; g.strokeStyle = C.ink2; g.stroke(); g.setLineDash([]);
    }
    drawRobot(T, tk.x[i], tk.y[i], tk.th[i], color);
    if (truth.fault && layers.truth) label(T, 'датчик образцов шумит', T.X(VIEW.x1 - 0.1), T.Y(VIEW.y1 - 0.2), { align: 'right', size: 12, bold: true, color: C.red });
    return i;
  }

  /* ---------------------------------------------------------------- проигрыватель */

  function mountPlayer(root, traces, labels, colors, opts = {}) {
    traces.forEach(prep);
    root.innerHTML = '';
    const end = Math.max(...traces.map(t => t._end));
    const state = { t: 0, playing: false, speed: 8, layers: { truth: true, belief: true, soil: false, path: true, lidar: false, ring: false } };
    const panes = el('div', 'pl-panes n' + traces.length), views = [];
    traces.forEach((tr, k) => {
      const r = tr.result, pane = el('div', 'pl-pane');
      pane.appendChild(el('div', 'pl-head', `<span class="lg-i"><i style="background:${colors[k]}"></i><b>${labels[k]}</b></span>
        <span class="pl-res">итог: ${r.samples_collected} из ${r.samples_total}, ${r.returned ? 'вернулся' : (r.reason === 'battery' ? 'заряд кончился в пути' : 'не вернулся')}, ${num(r.score, 1)} очка</span>`));
      const cv = el('canvas', 'pl-cv'); pane.appendChild(cv);
      const st = el('div', 'pl-status'); pane.appendChild(st);
      const jr = el('div', 'pl-journal'); if (!opts.noJournal) pane.appendChild(jr);
      panes.appendChild(pane); views.push({ tr, cv, st, jr, color: colors[k] });
    });
    root.appendChild(panes);

    const ctrl = el('div', 'pl-ctrl');
    const btn = el('button', 'pl-btn', '▶ Пуск');
    const range = el('input'); range.type = 'range'; range.min = 0; range.max = end; range.step = 0.2; range.value = 0;
    const clock = el('span', 'pl-clock');
    const speeds = el('span', 'pl-speeds');
    [2, 8, 24].forEach(v => { const b = el('button', 'pl-sp' + (v === state.speed ? ' on' : ''), '×' + v); b.onclick = () => { state.speed = v; speeds.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); }; speeds.appendChild(b); });
    const track = el('div', 'pl-track');
    track.appendChild(range);
    const marks = el('div', 'pl-marks'); track.appendChild(marks);
    ctrl.append(btn, track, clock, speeds); root.appendChild(ctrl);
    // метки на шкале: что случилось в среде (общее для обоих прогонов) и что получил каждый агент
    for (const w of traces[0].world) marks.appendChild(Object.assign(el('i', 'mk world'), { title: `${num(w.t, 0)} с — среда: ${EVENT_RU[w.type] || w.type}`, style: `left:${w.t / end * 100}%` }));
    traces.forEach((tr, k) => tr.events.forEach(e => marks.appendChild(Object.assign(el('i', 'mk ev' + (e.type === 'sample_collected' ? ' ok' : ' bad')), { title: `${num(e.t, 0)} с — ${labels[k]}: ${EVENT_RU[e.type] || e.type}`, style: `left:${e.t / end * 100}%; top:${6 + k * 7}px; background:${e.type === 'sample_collected' ? colors[k] : C.red}` }))));

    const lay = el('div', 'pl-layers');
    const LAYERS = [['truth', 'Как на самом деле', 'образцы, грунты и опасные зоны — робот их не видит'], ['belief', 'Что думает агент', 'синее — где, по его оценке, лежат образцы; пунктир — запомненные опасные зоны'],
      ['soil', 'Оценка грунта', 'коричневое — где агент считает пол дорогим'], ['path', 'Маршрут', 'куда он едет сейчас'], ['ring', 'Кольцо датчика', 'ближайший образец где-то на этом расстоянии'], ['lidar', 'Лидар', 'лучи до стен']];
    LAYERS.forEach(([key, name, hint]) => {
      const lb = el('label', 'pl-layer'); lb.title = hint;
      const cb = el('input'); cb.type = 'checkbox'; cb.checked = state.layers[key];
      cb.onchange = () => { state.layers[key] = cb.checked; draw(); };
      lb.append(cb, document.createTextNode(' ' + name)); lay.appendChild(lb);
    });
    root.appendChild(lay);

    function draw() {
      for (const v of views) {
        const t = Math.min(state.t, v.tr._end), tk = v.tr.track;
        const i = drawFrame(v.cv, v.tr, t, state.layers, v.color);
        const got = v.tr.events.filter(e => e.type === 'sample_collected' && e.t <= t).length;
        const pen = v.tr.events.filter(e => e.type !== 'sample_collected' && e.t <= t).length;
        v.st.innerHTML = `заряд <b>${num(tk.battery[i], 1)}</b> · собрано <b>${got} из ${v.tr.scenario.samples.length}</b> · штрафов <b>${pen}</b> · ${MODE_RU[v.tr.modes[tk.mode[i]]] || ''}`;
        if (!opts.noJournal) {
          const upto = lastIdx(v.tr.journal, t, 't');
          v.jr.innerHTML = v.tr.journal.slice(Math.max(0, upto - 3), upto + 1).reverse().map((e, n) =>
            `<div class="jr k-${e.kind}${n ? '' : ' new'}"><span class="jt">${num(e.t, 0)} с</span><span class="jk">${KIND_RU[e.kind] || e.kind}</span>${e.text}</div>`).join('') || '<div class="jr">журнал пуст</div>';
        }
      }
      range.value = state.t; clock.textContent = `${num(state.t, 0)} из ${num(end, 0)} с`;
    }
    let last = 0;
    function frame(now) {
      if (!state.playing) return;
      state.t = Math.min(end, state.t + (now - last) / 1000 * state.speed); last = now;
      draw();
      if (state.t >= end) { state.playing = false; btn.innerHTML = '↺ Сначала'; return; }
      requestAnimationFrame(frame);
    }
    btn.onclick = () => {
      if (state.playing) { state.playing = false; btn.innerHTML = '▶ Пуск'; return; }
      if (state.t >= end) state.t = 0;
      state.playing = true; btn.innerHTML = '❚❚ Пауза'; last = performance.now(); requestAnimationFrame(frame);
    };
    range.oninput = () => { state.t = +range.value; if (!state.playing) btn.innerHTML = '▶ Пуск'; draw(); };
    onResize(draw);
    return { seek: t => { state.t = t; draw(); }, stop: () => { state.playing = false; } };
  }

  /* ------------------------------------------------------------------ графики (SVG) */

  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  const tipBox = el('div', 'tip'); document.body.appendChild(tipBox);
  function tip(node, html) {
    node.addEventListener('mousemove', e => { tipBox.innerHTML = html; tipBox.style.display = 'block';
      const w = tipBox.offsetWidth; tipBox.style.left = Math.min(window.innerWidth - w - 12, e.clientX + 14) + 'px'; tipBox.style.top = e.clientY + 14 + 'px'; });
    node.addEventListener('mouseleave', () => { tipBox.style.display = 'none'; });
  }
  function niceTicks(lo, hi, n = 4) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function legend(root, items) {
    const lg = el('div', 'lg');
    items.forEach(it => lg.appendChild(el('span', 'lg-i', `<i style="background:${it.color}"></i>${it.label}`)));
    root.appendChild(lg);
  }
  function tableView(root, head, rows) {
    const d = el('details', 'tbl');
    d.appendChild(el('summary', null, 'Таблицей'));
    d.appendChild(el('table', null, `<thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>`));
    root.appendChild(d);
  }

  /** Точки со средним и 95% интервалом: категории по горизонтали, варианты агента рядом. */
  function dotChart(root, cfg) {
    const many = cfg.cats.length > 5;         // много условий: график во всю ширину, подписи в несколько строк
    const box = el('figure', many ? 'chart wide' : 'chart');
    box.appendChild(el('figcaption', null, `<b>${cfg.title}</b>${cfg.sub ? `<span>${cfg.sub}</span>` : ''}`));
    if (cfg.series.length > 1 && !cfg.noLegend) legend(box, cfg.series);
    const W = many ? 940 : 460, H = many ? 270 : 250, m = { l: 44, r: 14, t: 16, b: many ? 56 : 34 };
    const wrap = text => { const out = ['']; text.split(' ').forEach(w => { if ((out[out.length - 1] + ' ' + w).trim().length > 15 && out[out.length - 1]) out.push(w); else out[out.length - 1] = (out[out.length - 1] + ' ' + w).trim(); }); return out.slice(0, 3); };
    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.title });
    const vals = [];
    cfg.cats.forEach(c => cfg.series.forEach(s => { const v = cfg.get(c, s); if (v) vals.push(v.lo, v.hi, v.mean); }));
    let lo = cfg.ymin != null ? cfg.ymin : Math.min(...vals), hi = cfg.ymax != null ? cfg.ymax : Math.max(...vals);
    if (cfg.ymin == null) lo -= (hi - lo) * 0.12; if (cfg.ymax == null) hi += (hi - lo) * 0.14;
    const y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
    const band = (W - m.l - m.r) / cfg.cats.length;
    niceTicks(lo, hi).forEach(v => {
      S('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: C.line, 'stroke-width': 1 }, svg);
      S('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'ax' }, svg, cfg.fmt(v));
    });
    const rows = [];
    cfg.cats.forEach((c, ci) => {
      const cx = m.l + band * (ci + 0.5);
      if (many) wrap(c.label).forEach((line, li) => S('text', { x: cx, y: H - m.b + 18 + li * 13, 'text-anchor': 'middle', class: 'ax cat' }, svg, line));
      else S('text', { x: cx, y: H - 12, 'text-anchor': 'middle', class: 'ax cat' }, svg, c.label);
      cfg.series.forEach((s, si) => {
        const v = cfg.get(c, s); if (!v) return;
        const x = cx + (si - (cfg.series.length - 1) / 2) * Math.min(34, band / (cfg.series.length + 0.6));
        if (cfg.points) {
          const pts = cfg.points(c, s) || [];
          pts.forEach((p, k) => { if (p >= lo && p <= hi) S('circle', { cx: x + ((k * 7919) % 13 - 6) * 0.9, cy: y(p), r: 1.8, fill: s.color, opacity: 0.22 }, svg); });
        }
        if (v.hi > v.lo) S('line', { x1: x, x2: x, y1: y(v.lo), y2: y(v.hi), stroke: s.color, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);
        const dot = S('circle', { cx: x, cy: y(v.mean), r: 5.5, fill: s.color, stroke: '#fff', 'stroke-width': 2 }, svg);
        S('text', { x: x, y: y(v.hi) - 7, 'text-anchor': 'middle', class: 'val' }, svg, cfg.fmt(v.mean));
        const hit = S('rect', { x: x - 14, y: m.t, width: 28, height: H - m.t - m.b, fill: 'transparent' }, svg);
        const html = `<b>${s.label}</b>, ${c.label}<br>среднее ${cfg.fmt(v.mean)}${cfg.unit ? ' ' + cfg.unit : ''}<br>95% интервал ${cfg.fmt(v.lo)} – ${cfg.fmt(v.hi)}<br>прогонов ${v.n}`;
        tip(hit, html); tip(dot, html);
        rows.push([c.label, s.label, cfg.fmt(v.mean), `${cfg.fmt(v.lo)} – ${cfg.fmt(v.hi)}`, v.n]);
      });
    });
    box.appendChild(svg);
    tableView(box, [cfg.catName || 'Условие', 'Вариант', 'Среднее', '95% интервал', 'Прогонов'], rows);
    root.appendChild(box);
  }

  /** Горизонтальные точки: одна строка — один вариант. */
  function rowChart(root, cfg) {
    const box = el('figure', 'chart wide');
    box.appendChild(el('figcaption', null, `<b>${cfg.title}</b>${cfg.sub ? `<span>${cfg.sub}</span>` : ''}`));
    const rowH = 30, W = 720, m = { l: 250, r: 70, t: 10, b: 30 }, H = m.t + m.b + rowH * cfg.rows.length;
    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.title });
    const all = cfg.rows.flatMap(r => [r.lo, r.hi]).concat(cfg.zero != null ? [cfg.zero] : []);
    let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const x = v => m.l + (W - m.l - m.r) * (v - lo) / (hi - lo);
    niceTicks(lo, hi, 5).forEach(v => {
      S('line', { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: C.line, 'stroke-width': 1 }, svg);
      S('text', { x: x(v), y: H - 10, 'text-anchor': 'middle', class: 'ax' }, svg, cfg.fmt(v));
    });
    if (cfg.ref != null) S('line', { x1: x(cfg.ref), x2: x(cfg.ref), y1: m.t, y2: H - m.b, stroke: C.ink3, 'stroke-width': 1 }, svg);
    if (cfg.zero != null) S('line', { x1: x(cfg.zero), x2: x(cfg.zero), y1: m.t, y2: H - m.b, stroke: C.ink, 'stroke-width': 1 }, svg);
    cfg.rows.forEach((r, i) => {
      const cy = m.t + rowH * (i + 0.5);
      S('text', { x: m.l - 10, y: cy + 4, 'text-anchor': 'end', class: 'ax cat' + (r.strong ? ' strong' : '') }, svg, r.label);
      S('line', { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, stroke: r.color || C.fixed, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);
      const dot = S('circle', { cx: x(r.mean), cy, r: 5.5, fill: r.color || C.fixed, stroke: '#fff', 'stroke-width': 2 }, svg);
      S('text', { x: x(r.hi) + 8, y: cy + 4, class: 'val', 'text-anchor': 'start' }, svg, cfg.fmt(r.mean) + (r.note ? '  ' + r.note : ''));
      tip(dot, `<b>${r.label}</b><br>${cfg.what} ${cfg.fmt(r.mean)}<br>95% интервал ${cfg.fmt(r.lo)} – ${cfg.fmt(r.hi)}${r.n ? '<br>сценариев ' + r.n : ''}${r.extra ? '<br>' + r.extra : ''}`);
    });
    box.appendChild(svg);
    tableView(box, cfg.head || ['Вариант', 'Среднее', '95% интервал'], cfg.rows.map(r => [r.label, cfg.fmt(r.mean), `${cfg.fmt(r.lo)} – ${cfg.fmt(r.hi)}`].concat(r.cells || [])));
    root.appendChild(box);
  }

  /** Линия по времени с курсором. series: [{label, color, t:[], v:[]}], bands: [{t0, t1, label}], marks: [{t, label, color}] */
  function timeChart(root, cfg) {
    const box = el('figure', 'chart wide');
    box.appendChild(el('figcaption', null, `<b>${cfg.title}</b>${cfg.sub ? `<span>${cfg.sub}</span>` : ''}`));
    if (cfg.series.length > 1) legend(box, cfg.series);
    const W = 720, H = 230, m = { l: 44, r: 16, t: 14, b: 30 };
    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.title });
    const tmax = Math.max(...cfg.series.map(s => s.t[s.t.length - 1]));
    const x = t => m.l + (W - m.l - m.r) * t / tmax, y = v => m.t + (H - m.t - m.b) * (1 - (v - cfg.ymin) / (cfg.ymax - cfg.ymin));
    (cfg.bands || []).forEach(b => {
      S('rect', { x: x(b.t0), y: m.t, width: Math.max(1, x(Math.min(tmax, b.t1)) - x(b.t0)), height: H - m.t - m.b, fill: b.color || C.red, opacity: 0.09 }, svg);
      S('text', { x: x(b.t0) + 4, y: H - m.b - 6, class: 'ax band' }, svg, b.label);
    });
    niceTicks(cfg.ymin, cfg.ymax).forEach(v => { S('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: C.line, 'stroke-width': 1 }, svg); S('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'ax' }, svg, cfg.fmt(v)); });
    niceTicks(0, tmax, 6).forEach(v => S('text', { x: x(v), y: H - 10, 'text-anchor': 'middle', class: 'ax' }, svg, v + ' с'));
    cfg.series.forEach(s => {
      S('polyline', { points: s.t.map((t, i) => `${x(t).toFixed(1)},${y(s.v[i]).toFixed(1)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
      S('circle', { cx: x(s.t[s.t.length - 1]), cy: y(s.v[s.v.length - 1]), r: 4.5, fill: s.color, stroke: '#fff', 'stroke-width': 2 }, svg);
    });
    (cfg.marks || []).forEach(k => {
      S('line', { x1: x(k.t), x2: x(k.t), y1: m.t, y2: H - m.b, stroke: k.color || C.ink2, 'stroke-width': 1 }, svg);
      const d = S('circle', { cx: x(k.t), cy: m.t + 4 + (k.row || 0) * 11, r: 4.5, fill: k.color || C.ink2, stroke: '#fff', 'stroke-width': 2 }, svg);
      tip(d, `${num(k.t, 0)} с — ${k.label}`);
    });
    const cross = S('line', { y1: m.t, y2: H - m.b, stroke: C.ink, 'stroke-width': 1, visibility: 'hidden' }, svg);
    const hit = S('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', e => {
      const r = svg.getBoundingClientRect(), t = Math.max(0, Math.min(tmax, ((e.clientX - r.left) / r.width * W - m.l) / (W - m.l - m.r) * tmax));
      cross.setAttribute('x1', x(t)); cross.setAttribute('x2', x(t)); cross.setAttribute('visibility', 'visible');
      tipBox.innerHTML = `<b>${num(t, 0)} с</b><br>` + cfg.series.map(s => { const i = Math.max(0, lastIdx(s.t, t)); return `<i class="k" style="background:${s.color}"></i>${s.label}: ${t > s.t[s.t.length - 1] ? 'прогон окончен' : cfg.fmt(s.v[i])}`; }).join('<br>');
      tipBox.style.display = 'block'; tipBox.style.left = Math.min(window.innerWidth - tipBox.offsetWidth - 12, e.clientX + 14) + 'px'; tipBox.style.top = e.clientY + 14 + 'px';
    });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); tipBox.style.display = 'none'; });
    box.appendChild(svg);
    root.appendChild(box);
  }

  /* -------------------------------------------------------------- данные опытов */

  const E = D.experiments;
  const grp = (exp, arm, cond, level) => (E[exp] ? E[exp].groups : []).find(g => g.arm === arm && g.condition === cond && g.level === level);
  const stat = (exp, arm, cond, level, metric) => { const g = grp(exp, arm, cond, level); const s = g && g.stats[metric]; return s ? { mean: s.mean, lo: s.ci[0], hi: s.ci[1], n: s.n } : null; };
  const runsOf = (exp, arm, cond, level) => (E[exp] ? E[exp].runs : []).filter(r => r.arm === arm && r.condition === cond && (level === 'all' || r.level === level));
  const PALETTE = [C.fixed, C.adaptive, C.third, '#8a5cd6', '#c9a227', C.ink3, C.ink];
  const armsOf = exp => { const free = PALETTE.filter(c => !E[exp].spec.arms.some(a => ARM_COLOR[a.id] === c)); let k = 0; return E[exp].spec.arms.map(a => ({ key: a.id, label: a.label, color: ARM_COLOR[a.id] || free[k++ % free.length] })); };
  const VERDICT = { supported: ['✓', 'подтверждается'], refuted: ['✗', 'опровергается'], inconclusive: ['≈', 'разницу не различить'], no_data: ['·', 'нет данных'], partial: ['◐', 'подтверждается частично'], mixed: ['◐', 'по-разному'], descriptive: ['·', 'описательный'], not_run: ['·', 'не запускался'] };
  const chip = st => `<span class="chip ${st}">${(VERDICT[st] || ['·', st])[0]} ${(VERDICT[st] || ['', st])[1]}</span>`;

  function expHeader(root, exp) {
    const s = E[exp].spec;
    root.appendChild(el('div', 'exp-meta', `
      <div><span>Вопрос</span>${s.question}</div>
      <div><span>Гипотеза</span>${s.hypothesis}</div>
      <div><span>Как проверяем</span>${s.method}</div>
      <div><span>Что опровергнет</span>${s.refute}</div>
      <div><span>Объём</span>${E[exp].runs.length} прогонов: ${s.arms.length} ${s.arms.length < 5 ? 'варианта' : 'вариантов'} агента × ${s.conditions.length > 1 ? s.conditions.length + ' условия × ' : ''}${s.levels.join(', ')} × ${s.seeds} сценариев (номера ${s.seed_start}–${s.seed_start + s.seeds - 1}). Аварийных прогонов: ${E[exp].errors}.</div>`));
  }
  function claimList(root, exp) {
    const ul = el('ul', 'claims');
    E[exp].claims.forEach(c => {
      const cell = c.cells.find(x => x.level === 'all') || c.cells[0], p = cell.pair;
      ul.appendChild(el('li', null, `${chip(c.status)} <b>${c.text}.</b> ${p ? `Разность на одинаковых сценариях ${p.mean > 0 ? '+' : ''}${num(p.mean, Math.abs(p.mean) < 2 ? 2 : 1)} (95% интервал от ${num(p.ci[0], 2)} до ${num(p.ci[1], 2)}); первый лучше в ${c.better === 'lower' ? p.b_higher : p.a_higher} сценариях, хуже в ${c.better === 'lower' ? p.a_higher : p.b_higher}, поровну в ${p.ties}.` : ''}`));
    });
    root.appendChild(ul);
  }
  const METRIC = {
    samples_share: { title: 'Собрано образцов', fmt: pct, ymin: 0, ymax: 1.08 },
    returned: { title: 'Вернулся на базу', fmt: pct, ymin: 0, ymax: 1.08, noPoints: true },
    battery_used: { title: 'Потрачено заряда', fmt: v => num(v, 0), unit: 'ед. из 60' },
    score: { title: 'Счёт', fmt: v => num(v, 0), unit: 'очков' },
    false_collects: { title: 'Ложные сборы за прогон', fmt: v => num(v, 2), ymin: 0 },
    hazard_hits: { title: 'Въезды в опасную зону за прогон', fmt: v => num(v, 2), ymin: 0 },
    distance: { title: 'Путь', fmt: v => num(v, 1), unit: 'м' },
    penalties: { title: 'Штрафы за прогон', fmt: v => num(v, 2), ymin: 0 },
    collisions: { title: 'Столкновения за прогон', fmt: v => num(v, 1), ymin: 0 },
    inq_correct: { title: 'Причина названа верно', sub: 'доля вынесенных выводов', fmt: pct, ymin: 0, ymax: 1.08, noPoints: true },
    faults_found: { title: 'Найдено сбоев', sub: 'доля настоящих сбоев', fmt: pct, ymin: 0, ymax: 1.08, noPoints: true },
    inq_insufficient: { title: 'Исход «недостаточно данных» за прогон', fmt: v => num(v, 2), ymin: 0, noPoints: true },
    inq_energy: { title: 'Заряд на опыты за прогон', fmt: v => num(v, 2), unit: 'ед.', ymin: 0, noPoints: true },
    inq_wrong: { title: 'Ошибочных выводов за прогон', fmt: v => num(v, 2), ymin: 0, noPoints: true },
    study_error_pct: { title: 'Ошибка оценки', sub: '% от настоящего значения', fmt: v => num(v, 1), unit: '%', ymin: 0 },
    study_covered: { title: 'Истина внутри заявленного интервала', sub: 'доля прогонов, должно быть около 95%', fmt: pct, ymin: 0, ymax: 1.08, noPoints: true },
    study_halfwidth_pct: { title: 'Заявленная погрешность', sub: '95%, в % от оценки', fmt: v => num(v, 1), unit: '%', ymin: 0 },
    study_energy: { title: 'Заряд на исследование', fmt: v => num(v, 1), unit: 'ед.', ymin: 0 },
  };
  function expCharts(root, exp, metrics, byCondition) {
    const s = E[exp].spec, grid = el('div', 'chart-grid'), series = armsOf(exp);
    const cats = byCondition ? s.conditions.map(c => ({ key: c.id, label: c.label })) : s.levels.map(l => ({ key: l, label: l }));
    const level = s.levels[0];
    metrics.forEach(mt => {
      const M = METRIC[mt];
      const have = series.filter(sr => cats.some(c => byCondition ? stat(exp, sr.key, c.key, level, mt) : stat(exp, sr.key, 'base', c.key, mt)));
      if (!have.length) return;
      dotChart(grid, { title: M.title, sub: M.sub || M.unit || '', cats, series: have, fmt: M.fmt, unit: M.unit, ymin: M.ymin, ymax: M.ymax, catName: byCondition ? 'Условие' : 'Уровень',
        get: (c, sr) => byCondition ? stat(exp, sr.key, c.key, level, mt) : stat(exp, sr.key, 'base', c.key, mt),
        points: M.noPoints ? null : (c, sr) => (byCondition ? runsOf(exp, sr.key, c.key, level) : runsOf(exp, sr.key, 'base', c.key)).map(r => r[mt]).filter(v => v != null) });
    });
    root.appendChild(grid);
  }

  /* ------------------------------------------------------------- готовые числа */

  const F = {};
  const story = D.stories[0];
  (() => {
    const all = Object.values(E).filter(Boolean);
    F.runs_total = all.reduce((n, e) => n + (e.spec.manual ? 0 : e.runs.length), 0).toLocaleString('ru');
    F.errors_total = all.reduce((n, e) => n + e.errors, 0);
    const walls = E.E1.runs.map(r => r.wall_s).filter(Boolean).sort((a, b) => a - b);
    F.wall = num(walls[walls.length >> 1], 1);
    F.built = D.built;
    F.tests = D.tests || '—';
    for (const lv of ['easy', 'medium', 'hard']) for (const arm of ['fixed', 'adaptive']) for (const mt of ['samples_share', 'returned', 'battery_used', 'score']) {
      const v = stat('E1', arm, 'base', lv, mt);
      F[`e1.${mt}.${lv}.${arm}`] = v ? ((mt === 'samples_share' || mt === 'returned') ? pct(v.mean) : num(v.mean, 1)) : '—';
    }
    const det = k => { const rs = runsOf('E1', 'adaptive', 'base', 'hard').map(r => r.detect && r.detect[k]).filter(v => v !== undefined); const got = rs.filter(v => v != null).sort((a, b) => a - b); return { n: rs.length, got: got.length, med: got.length ? got[got.length >> 1] : null }; };
    for (const k of ['soil_change', 'new_hazard', 'sensor_fault']) { const d = det(k); F[`det.${k}.n`] = d.n; F[`det.${k}.got`] = d.got; F[`det.${k}.med`] = num(d.med, 0); }
    const hyp = runsOf('E1', 'adaptive', 'base', 'hard').map(r => r.hyp).filter(Boolean);
    for (const k of ['total', 'confirmed', 'refuted', 'open']) F['hyp.' + k] = num(hyp.reduce((a, h) => a + h[k], 0) / (hyp.length || 1), 1);
    F['story.seed'] = story.seed;
    for (const arm of ['fixed', 'adaptive']) { const r = story[arm].result; F[`story.${arm}`] = `${r.samples_collected} из ${r.samples_total}`; F[`story.${arm}.score`] = num(r.score, 1); }
    F.route_len = num(D.route.length, 1); F.route_n = D.route.points.length;
    if (D.code) { F.loc_total = (D.code.py + D.code.js).toLocaleString('ru'); F.loc_py = D.code.py.toLocaleString('ru'); F.loc_js = D.code.js.toLocaleString('ru'); }
    if (D.inq_accuracy) for (const k of ['runs', 'total', 'identified', 'wrong']) F['inq_accuracy.' + k] = D.inq_accuracy[k];
    F.gz_n = D.gazebo.length;
    F.gz_ok = D.gazebo.filter(p => p.gazebo.result.returned && p.gazebo.result.samples_collected >= p.fastsim.result.samples_collected - 1 && p.gazebo.result.penalties <= p.fastsim.result.penalties).length;
    if (E.E6) for (const a of ['llm', 'llm_faulty']) { const rs = E.E6.runs.filter(r => r.arm === a); const calls = rs.reduce((n, r) => n + (r.llm_calls || 0), 0), bad = rs.reduce((n, r) => n + (r.llm_failed || 0), 0); const fb = rs.reduce((n, r) => n + ((r.plans && r.plans.fallback) || 0), 0), pl = rs.reduce((n, r) => n + Object.values(r.plans || {}).reduce((x, y) => x + y, 0), 0); F[`e6.${a}.calls`] = num(calls / (rs.length || 1), 0); F[`e6.${a}.bad`] = pct(calls ? bad / calls : 0); F[`e6.${a}.fb`] = pct(pl ? fb / pl : 0); F[`e6.${a}.n`] = rs.length; }
  })();
  document.querySelectorAll('[data-f]').forEach(n => { n.textContent = F[n.dataset.f] != null ? F[n.dataset.f] : '—'; });
  document.querySelectorAll('img[data-shot]').forEach(n => { const src = (D.shots || {})[n.dataset.shot]; if (src) n.src = src; else n.closest('figure').style.display = 'none'; });

  /* ---------------------------------------------- рисунок 1: правда и взгляд робота */

  (() => {
    const cv = $('fig-truth'); if (!cv) return;
    const sc = story.adaptive.scenario; let mode = 'truth';
    const note = $('fig-truth-note');
    function draw() {
      const T = stage(cv), { g } = T;
      if (mode === 'truth') {
        drawFloor(T); drawSoils(T, sc.soils); drawHazards(T, sc.hazards); drawBase(T, sc.base); drawSamples(T, sc.samples);
        drawRobot(T, sc.base[0], sc.base[1], 0);
        note.innerHTML = `Сценарий hard № ${story.seed}: <b>${sc.samples.length} образцов</b> (жёлтые точки), <b>${sc.soils.length} зоны дорогого грунта</b> (оранжевые, число — во сколько раз быстрее садится батарея), <b>${sc.hazards.length} опасная зона</b> (красная). Через ${num(sc.events[0].t, 0)} секунд судья начнёт менять среду.`;
      } else {
        drawFloor(T, 0.55);
        const [bx, by] = sc.base;
        g.strokeStyle = 'rgba(27,175,122,.45)'; g.lineWidth = 1 * T.dpr; g.beginPath();
        for (const [a, r] of raycast(bx, by)) if (r) { g.moveTo(T.X(bx), T.Y(by)); g.lineTo(T.X(bx + r * Math.cos(a)), T.Y(by + r * Math.sin(a))); }
        g.stroke();
        const d = Math.min(...sc.samples.map(p => Math.hypot(p[0] - bx, p[1] - by))), z = Math.max(0, 1 - d / RULES.sensor_range_m);
        g.beginPath(); g.arc(T.X(bx), T.Y(by), d * T.s, 0, 2 * Math.PI); g.setLineDash([3 * T.dpr, 6 * T.dpr]); g.strokeStyle = C.ink2; g.lineWidth = 1.5 * T.dpr; g.stroke(); g.setLineDash([]);
        drawBase(T, sc.base); drawRobot(T, bx, by, 0);
        note.innerHTML = `Зелёные лучи — лидар: стены и столбы робот видит. Больше он не видит ничего. У него есть три числа: <b>заряд ${RULES.battery_start}</b>, <b>датчик образцов ${num(z, 2)}</b> и своё положение. Пунктир — всё, что значит это показание: ближайший образец лежит <i>где-то на этой окружности</i>, а в какой стороне — неизвестно.`;
      }
    }
    document.querySelectorAll('[data-truth]').forEach(b => b.onclick = () => { mode = b.dataset.truth; document.querySelectorAll('[data-truth]').forEach(x => x.classList.toggle('on', x === b)); draw(); });
    onResize(draw);
  })();

  /* ------------------------------------------------------ азбука: лидар */

  (() => {
    const cv = $('fig-lidar'); if (!cv) return;
    const st = { robot: [-0.9, 0.35], drag: false };
    function draw() {
      const cssW = cv.clientWidth || cv.parentElement.clientWidth || 600, dpr = Math.min(2, window.devicePixelRatio || 1);
      const mapW = Math.min(cssW * 0.5, 420), H = mapW * ASPECT;
      cv.width = Math.round(cssW * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + 'px';
      const g = cv.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height);
      const sc = mapW * dpr / (VIEW.x1 - VIEW.x0);
      const T = { g, s: sc, dpr, X: x => (x - VIEW.x0) * sc, Y: y => (VIEW.y1 - y) * sc };
      drawFloor(T);
      const rays = raycast(st.robot[0], st.robot[1], 360, 6);
      g.strokeStyle = 'rgba(27,175,122,.30)'; g.lineWidth = 1 * dpr; g.beginPath();
      rays.forEach(([a, r], k) => { if (k % 3 || r == null) return; g.moveTo(T.X(st.robot[0]), T.Y(st.robot[1])); g.lineTo(T.X(st.robot[0] + r * Math.cos(a)), T.Y(st.robot[1] + r * Math.sin(a))); });
      g.stroke();
      g.fillStyle = C.third; rays.forEach(([a, r]) => { if (r == null) return; g.fillRect(T.X(st.robot[0] + r * Math.cos(a)) - dpr, T.Y(st.robot[1] + r * Math.sin(a)) - dpr, 2.4 * dpr, 2.4 * dpr); });
      drawRobot(T, st.robot[0], st.robot[1], 0, C.adaptive);
      // справа: те же 360 чисел полоской
      const x0 = (mapW + 34) * dpr, w = cv.width - x0 - 8 * dpr, y0 = 34 * dpr, h = cv.height - 90 * dpr, rmax = 5;
      g.fillStyle = C.ink; g.font = `600 ${13 * dpr}px system-ui, sans-serif`; g.textAlign = 'left';
      g.fillText('obs.scan — 360 чисел, метры до стены', x0, 18 * dpr);
      g.strokeStyle = C.line; g.lineWidth = dpr; g.beginPath();
      [0, 1, 2, 3, 4, 5].forEach(v => { const y = y0 + h * (1 - v / rmax); g.moveTo(x0, y); g.lineTo(x0 + w, y); }); g.stroke();
      g.fillStyle = C.third;
      rays.forEach(([a, r], k) => { const v = Math.min(r == null ? rmax : r, rmax); g.fillRect(x0 + w * k / 360, y0 + h * (1 - v / rmax), Math.max(1, w / 360 - 0.4), h * v / rmax); });
      g.font = `${11 * dpr}px system-ui, sans-serif`; g.textAlign = 'right';
      [1, 2, 3, 4, 5].forEach(v => { const y = y0 + h * (1 - v / rmax); g.fillStyle = 'rgba(255,255,255,.82)'; g.fillRect(x0 + w - 30 * dpr, y - 13 * dpr, 30 * dpr, 13 * dpr); g.fillStyle = C.ink2; g.fillText(v + ' м', x0 + w - 3 * dpr, y - 3 * dpr); });
      g.textAlign = 'center';
      [['[0]', 'вперёд', 0], ['[90]', 'слева', 90], ['[180]', 'сзади', 180], ['[270]', 'справа', 270]].forEach(([i, t, k]) => { const x = Math.max(x0 + 20 * dpr, x0 + w * k / 360);
        g.fillStyle = C.ink; g.font = `${11 * dpr}px ui-monospace, Menlo, monospace`; g.fillText(i, x, y0 + h + 15 * dpr);
        g.fillStyle = C.ink2; g.font = `${11 * dpr}px system-ui, sans-serif`; g.fillText(t, x, y0 + h + 29 * dpr); });
      const f = rays[0][1], l = rays[90][1];
      g.textAlign = 'right'; g.fillStyle = C.ink; g.font = `${12 * dpr}px ui-monospace, Menlo, monospace`;
      g.fillText(`scan[0] = ${f == null ? '—' : f.toFixed(2)}   scan[90] = ${l == null ? '—' : l.toFixed(2)}`, x0 + w, y0 + h + 29 * dpr + 14 * dpr);
      st.T = T; st.mapW = mapW;
    }
    const move = e => { const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top; if (px > st.mapW) return;
      const x = VIEW.x0 + px / st.mapW * (VIEW.x1 - VIEW.x0), y = VIEW.y1 - py / st.mapW * (VIEW.x1 - VIEW.x0); if (isFree(x, y)) { st.robot = [x, y]; draw(); } };
    cv.addEventListener('pointerdown', e => { st.drag = true; cv.setPointerCapture(e.pointerId); move(e); });
    cv.addEventListener('pointermove', e => { if (st.drag) move(e); });
    cv.addEventListener('pointerup', () => { st.drag = false; });
    onResize(draw);
  })();

  /* ------------------------------------------- азбука: правило Байеса на коридоре */

  (() => {
    const cv = $('fig-bayes'); if (!cv) return;
    const N = 24, CELL = 0.1, R = 1.2, SIG = 0.07, note = $('fig-bayes-note');
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const st = { robot: 4, sample: 15, p: null, n: 0, show: false, last: null };
    const clean = (i, j) => Math.max(0, 1 - Math.abs(i - j) * CELL / R);
    function hide() { do { st.sample = 2 + Math.floor(rnd() * (N - 4)); } while (Math.abs(st.sample - st.robot) < 5); st.p = Array(N).fill(1 / N); st.n = 0; st.show = false; st.last = null; draw(); }
    function measure() {
      const z = Math.max(0, Math.min(1, clean(st.robot, st.sample) + gauss() * SIG));
      let sum = 0;
      st.p = st.p.map((p, j) => { const v = p * Math.exp(-0.5 * ((z - clean(st.robot, j)) / SIG) ** 2); sum += v; return v; }).map(v => v / (sum || 1));
      st.n += 1; st.last = z; draw();
    }
    function draw() {
      const cssW = cv.clientWidth || cv.parentElement.clientWidth || 600, dpr = Math.min(2, window.devicePixelRatio || 1), H = 230;
      cv.width = Math.round(cssW * dpr); cv.height = H * dpr; cv.style.height = H + 'px';
      const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
      const m = 16 * dpr, w = (cv.width - 2 * m) / N, base = 160 * dpr, top = 18 * dpr, pmax = Math.max(...st.p, 0.2);
      for (let j = 0; j < N; j++) {
        const h = (base - top) * st.p[j] / pmax;
        g.fillStyle = '#eef0f2'; g.fillRect(m + j * w + 1.5 * dpr, top, w - 3 * dpr, base - top);
        g.fillStyle = C.fixed; g.fillRect(m + j * w + 1.5 * dpr, base - h, w - 3 * dpr, h);
        if (st.p[j] >= 0.08) { g.fillStyle = C.ink; g.font = `${10.5 * dpr}px system-ui, sans-serif`; g.textAlign = 'center'; g.fillText(Math.round(st.p[j] * 100) + '%', m + (j + 0.5) * w, base - h - 4 * dpr); }
        g.strokeStyle = C.line; g.lineWidth = dpr; g.strokeRect(m + j * w, base + 8 * dpr, w, 26 * dpr);
      }
      if (st.show) { g.beginPath(); g.arc(m + (st.sample + 0.5) * w, base + 21 * dpr, 7 * dpr, 0, 2 * Math.PI); g.fillStyle = '#f2b705'; g.fill(); g.lineWidth = 1.5 * dpr; g.strokeStyle = C.ink; g.stroke(); }
      g.beginPath(); g.arc(m + (st.robot + 0.5) * w, base + 21 * dpr, 9 * dpr, 0, 2 * Math.PI); g.fillStyle = C.adaptive; g.fill(); g.lineWidth = 2 * dpr; g.strokeStyle = '#fff'; g.stroke();
      g.fillStyle = C.ink2; g.font = `${12 * dpr}px system-ui, sans-serif`; g.textAlign = 'center';
      g.fillText('робот', m + (st.robot + 0.5) * w, base + 52 * dpr);
      if (st.show) g.fillText('образец', m + (st.sample + 0.5) * w, base + 52 * dpr + (Math.abs(st.sample - st.robot) < 3 ? 14 * dpr : 0));
      const best = st.p.indexOf(Math.max(...st.p));
      note.innerHTML = st.n === 0 ? 'Пока измерений нет: все клетки равновероятны, по 4%. Нажмите «Измерить здесь».'
        : `Измерений: <b>${st.n}</b>. Последнее показание <b>${num(st.last, 2)}</b> ${st.last < 0.03 ? '→ ближе 1,2 м образца нет: клетки рядом с роботом обнулились' : `→ до образца около <b>${num((1 - st.last) * R * 100, 0)} см</b>, в какую сторону — неизвестно`}. Самая вероятная клетка сейчас — №${best + 1} (${Math.round(st.p[best] * 100)}%)${st.show ? `, образец на самом деле в клетке №${st.sample + 1}` : ''}.`;
    }
    $('b1-left').onclick = () => { st.robot = Math.max(0, st.robot - 3); draw(); };
    $('b1-right').onclick = () => { st.robot = Math.min(N - 1, st.robot + 3); draw(); };
    $('b1-measure').onclick = measure;
    $('b1-show').onclick = () => { st.show = !st.show; draw(); };
    $('b1-new').onclick = hide;
    hide(); onResize(draw);
  })();

  /* ------------------------------------------ рисунок 2: датчик без направления */

  (() => {
    const cv = $('fig-ring'); if (!cv) return;
    const R = RULES.sensor_range_m, SIGMA = 0.05, cells = [];
    for (let y = -2.6; y < 2.7; y += 0.1) for (let x = -2.8; x < 2.7; x += 0.1) if (isFree(x, y) && isFree(x + 0.12, y) && isFree(x - 0.12, y) && isFree(x, y + 0.12) && isFree(x, y - 0.12)) cells.push([x, y]);
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const st = { robot: [A.base[0], A.base[1]], sample: null, meas: [], p: null, reveal: false, drag: false };
    const out = $('fig-ring-note');
    function hide() { let d; do { st.sample = cells[Math.floor(rnd() * cells.length)]; d = Math.hypot(st.sample[0] - A.base[0], st.sample[1] - A.base[1]); } while (d < 1.3 || d > 1.9); st.meas = []; st.p = cells.map(() => 1 / cells.length); st.robot = [A.base[0], A.base[1]]; st.reveal = false; draw(); }
    const clean = (x, y) => Math.max(0, 1 - Math.hypot(st.sample[0] - x, st.sample[1] - y) / R);
    function measure() {
      const [x, y] = st.robot, z = Math.max(0, Math.min(1, clean(x, y) + gauss() * SIGMA));
      st.meas.push({ x, y, z });
      let sum = 0;
      st.p = st.p.map((p, i) => { const f = Math.max(0, 1 - Math.hypot(cells[i][0] - x, cells[i][1] - y) / R); const v = p * Math.exp(-0.5 * ((z - f) / 0.065) ** 2); sum += v; return v; });
      st.p = st.p.map(v => v / (sum || 1));
      draw();
    }
    function draw() {
      const T = stage(cv), { g } = T;
      drawFloor(T);
      if (st.meas.length) {
        const pmax = Math.max(...st.p);
        st.p.forEach((p, i) => { const a = Math.sqrt(p / pmax); if (a < 0.04) return; g.fillStyle = `rgba(42,120,214,${(a * 0.85).toFixed(3)})`; g.fillRect(T.X(cells[i][0] - 0.05), T.Y(cells[i][1] + 0.05), 0.1 * T.s + 1, 0.1 * T.s + 1); });
      }
      st.meas.forEach((m, k) => {
        if (m.z > 0.02) { g.beginPath(); g.arc(T.X(m.x), T.Y(m.y), (1 - m.z) * R * T.s, 0, 2 * Math.PI); g.lineWidth = 1.5 * T.dpr; g.strokeStyle = 'rgba(20,24,28,.55)'; g.setLineDash([4 * T.dpr, 4 * T.dpr]); g.stroke(); g.setLineDash([]); }
        g.beginPath(); g.arc(T.X(m.x), T.Y(m.y), 3.5 * T.dpr, 0, 2 * Math.PI); g.fillStyle = C.ink; g.fill();
        label(T, String(k + 1), T.X(m.x) + 9 * T.dpr, T.Y(m.y) - 9 * T.dpr, { size: 11, color: C.ink2 });
      });
      drawBase(T);
      if (st.reveal) drawSamples(T, [st.sample]);
      drawRobot(T, st.robot[0], st.robot[1], 0, C.adaptive);
      const z = clean(...st.robot);
      let best = 0; st.p.forEach((p, i) => { if (p > st.p[best]) best = i; });
      const err = Math.hypot(cells[best][0] - st.sample[0], cells[best][1] - st.sample[1]);
      out.innerHTML = `Показание в этой точке: <b>${num(z, 2)}</b> ${z <= 0 ? '(образец дальше 2 м — датчик молчит)' : `→ до образца около <b>${num((1 - z) * R, 1)} м</b>, направление неизвестно`}. Измерений: <b>${st.meas.length}</b>.` +
        (st.meas.length ? ` Самое вероятное место по накопленным измерениям ${st.reveal ? `в <b>${num(err * 100, 0)} см</b> от настоящего.` : 'закрашено гуще всего — нажмите «Показать образец», чтобы сверить.'}` : ' Перетащите робота и нажмите «Измерить».');
    }
    const pos = e => { const r = cv.getBoundingClientRect(), T = stage(cv); draw(); return [T.wx(e.clientX - r.left), T.wy(e.clientY - r.top)]; };
    const move = e => { const [x, y] = pos(e); if (isFree(x, y)) { st.robot = [x, y]; draw(); } };
    cv.addEventListener('pointerdown', e => { st.drag = true; cv.setPointerCapture(e.pointerId); move(e); });
    cv.addEventListener('pointermove', e => { if (st.drag) move(e); });
    cv.addEventListener('pointerup', () => { st.drag = false; });
    $('ring-measure').onclick = measure;
    $('ring-reveal').onclick = () => { st.reveal = !st.reveal; draw(); };
    $('ring-new').onclick = hide;
    hide(); onResize(draw);
  })();

  /* --------------------------------- рисунок 3: как сходится карта вероятностей */

  (() => {
    const root = $('fig-belief'); if (!root) return;
    const tr = prep(story.adaptive);
    const first = tr.events.find(e => e.type === 'sample_collected');
    const t1 = first ? first.t : 10;
    const times = [0.6, t1 * 0.35, t1 * 0.7, Math.max(0, t1 - 0.4)];
    const caps = ['Первое показание: кольцо вокруг базы', 'Робот сдвинулся — кольца пересеклись', 'Осталось одно-два места', 'Перед сбором: одно пятно'];
    const cvs = times.map((t, k) => { const f = el('figure', 'mini'); const cv = el('canvas'); f.append(cv, el('figcaption', null, `<b>${num(t, 0)} с.</b> ${caps[k]}`)); root.appendChild(f); return cv; });
    onResize(() => cvs.forEach((cv, k) => drawFrame(cv, tr, times[k], { truth: false, belief: true, soil: false, path: false, lidar: false, ring: true }, C.adaptive)));
    // настоящие образцы поверх — чтобы было видно, куда сошлось
    onResize(() => cvs.forEach(cv => { const dpr = Math.min(2, window.devicePixelRatio || 1); const s = cv.width / (VIEW.x1 - VIEW.x0); const T = { g: cv.getContext('2d'), s, dpr, X: x => (x - VIEW.x0) * s, Y: y => (VIEW.y1 - y) * s }; drawSamples(T, tr.scenario.samples); }));
  })();

  /* ------------------------------------------------- рисунок 4: дорогой грунт */

  (() => {
    const cv = $('fig-soil'); if (!cv) return;
    const demo = D.soil_demo, slider = $('soil-mult'), out = $('fig-soil-note');
    function draw() {
      const c = demo.cases[+slider.value], T = stage(cv);
      drawFloor(T);
      { const z = { shape: 'rect', x: demo.zone.x, y: demo.zone.y, w: demo.zone.w, h: demo.zone.h }; zonePath(T, z); T.g.fillStyle = soilColor(c.mult); T.g.globalAlpha = 0.5; T.g.fill(); T.g.globalAlpha = 1; T.g.lineWidth = 1.5 * T.dpr; T.g.strokeStyle = soilColor(c.mult); T.g.stroke(); }
      polyline(T, c.blind.pts, C.fixed, 3); polyline(T, c.smart.pts, C.adaptive, 3, c.mult === 1 ? [6, 6] : null);
      label(T, 'грунт ×' + num(c.mult, c.mult % 1 ? 1 : 0), T.X(demo.zone.x), T.Y(demo.zone.y - demo.zone.h / 2 + 0.16), { size: 13, bold: true });
      drawBase(T); const { g } = T;
      g.beginPath(); g.arc(T.X(demo.goal[0]), T.Y(demo.goal[1]), 6 * T.dpr, 0, 2 * Math.PI); g.fillStyle = C.sample; g.fill(); g.lineWidth = 1.5 * T.dpr; g.strokeStyle = C.ink; g.stroke();
      const save = c.blind.energy - c.smart.energy;
      $('soil-mult-val').textContent = '×' + num(c.mult, c.mult % 1 ? 1 : 0);
      out.innerHTML = `<span class="lg-i"><i style="background:${C.fixed}"></i>Короткий путь, не зная о грунте: <b>${num(c.blind.len, 1)} м, ${num(c.blind.energy, 1)} ед. заряда</b></span>
        <span class="lg-i"><i style="background:${C.adaptive}"></i>Путь с учётом цены: <b>${num(c.smart.len, 1)} м, ${num(c.smart.energy, 1)} ед.</b></span>
        <span>${save > 0.05 ? `Объезд длиннее на ${num(c.smart.len - c.blind.len, 1)} м, но экономит <b>${num(save, 1)} ед.</b> — это ${pct(save / c.blind.energy)} расхода на этот переезд.` : 'Пока грунт стоит как обычный пол, оба пути совпадают: объезжать незачем.'}</span>`;
    }
    slider.max = demo.cases.length - 1; slider.value = 3; slider.oninput = draw; onResize(draw);
  })();

  /* ------------------------------------------- рисунок 5: гипотезы об опасной зоне */

  (() => {
    const a = $('fig-haz-a'), b = $('fig-haz-b'); if (!a) return;
    const E0 = [0, 0], head = 0, hyps = [];
    for (let phi = -80; phi <= 80; phi += 20) for (const r of [0.25, 0.33]) hyps.push({ x: E0[0] + r * Math.cos(phi * Math.PI / 180), y: E0[1] + r * Math.sin(phi * Math.PI / 180), r, w: Math.cos(phi * Math.PI / 180) });
    const safe = []; for (let x = -0.75; x <= 0.5; x += 0.05) safe.push([x, -0.36 - 0.02 * Math.sin(x * 4)]);
    const approach = []; for (let x = -0.8; x <= 0; x += 0.05) approach.push([x, 0]);
    function draw(cv, after) {
      const cssW = cv.clientWidth || 300, dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = cssW * dpr; cv.height = cssW * 0.72 * dpr; cv.style.height = cssW * 0.72 + 'px';
      const g = cv.getContext('2d'), s = cv.width / 2.0, T = { g, s, dpr, X: x => (x + 0.95) * s, Y: y => (0.72 - y) * s };
      g.fillStyle = '#fff'; g.fillRect(0, 0, cv.width, cv.height);
      const live = hyps.filter(h => !after || safe.every(p => Math.hypot(p[0] - h.x, p[1] - h.y) >= h.r - 0.02));
      for (const h of hyps) { const ok = live.includes(h); g.beginPath(); g.arc(T.X(h.x), T.Y(h.y), h.r * s, 0, 2 * Math.PI); g.lineWidth = 1.3 * dpr; g.strokeStyle = ok ? `rgba(214,69,69,${0.25 + 0.5 * h.w})` : 'rgba(138,147,155,.18)'; g.setLineDash(ok ? [] : [3 * dpr, 4 * dpr]); g.stroke(); g.setLineDash([]); if (ok) { g.fillStyle = `rgba(214,69,69,${0.035 * h.w})`; g.fill(); } }
      polyline(T, approach, C.adaptive, 2.4);
      if (after) polyline(T, safe, C.third, 2.4);
      g.beginPath(); g.arc(T.X(0), T.Y(0), 5 * dpr, 0, 2 * Math.PI); g.fillStyle = C.red; g.fill(); g.lineWidth = 2 * dpr; g.strokeStyle = '#fff'; g.stroke();
      label(T, 'штраф здесь', T.X(0) - 8 * dpr, T.Y(0) - 14 * dpr, { size: 12, align: 'right', color: C.ink });
      if (after) label(T, 'проехал без штрафа', T.X(-0.1), T.Y(-0.5), { size: 12, color: C.ink });
      label(T, `гипотез осталось: ${live.length} из ${hyps.length}`, 10 * dpr, 16 * dpr, { size: 12, align: 'left', bold: true });
    }
    onResize(() => { draw(a, false); draw(b, true); });
  })();

  /* ------------------------------------- рисунок 6: что случилось и когда заметил */

  (() => {
    const root = $('fig-events'); if (!root) return;
    const tr = story.adaptive, fx = story.fixed;
    const fault = tr.world.find(w => w.type === 'sensor_fault'), rec = tr.world.find(w => w.type === 'sensor_recovered');
    const alarms = tr.journal.filter(e => e.kind === 'alarm' && e.data && e.data.tag);
    const TAG = { model_mismatch: 'агент: «грунты изменились»', hazard: 'агент: «здесь опасная зона»', sensor_degraded: 'агент: «датчик шумит»' };
    const marks = tr.world.map(w => ({ t: w.t, label: 'среда: ' + (EVENT_RU[w.type] || w.type), color: C.ink, row: 0 }))
      .concat(alarms.map(e => ({ t: e.t, label: TAG[e.data.tag] || e.text, color: C.adaptive, row: 1 })));
    timeChart(root, { title: 'Заряд батареи по ходу прогона', sub: `сценарий hard № ${story.seed}; чёрные метки сверху — скрытые изменения среды, оранжевые — когда их заметил адаптивный агент`,
      series: [{ label: 'Фиксированный план', color: C.fixed, t: fx.track.t, v: fx.track.battery }, { label: 'С адаптацией', color: C.adaptive, t: tr.track.t, v: tr.track.battery }],
      ymin: 0, ymax: 60, fmt: v => num(v, 0), marks });
    timeChart(root, { title: 'Показание датчика образцов у адаптивного агента', sub: 'пилообразные подъёмы — подход к образцу; после сбора показание падает, потому что «ближайшим» становится другой образец',
      series: [{ label: 'датчик', color: C.adaptive, t: tr.track.t, v: tr.track.sensor }], ymin: 0, ymax: 1, fmt: v => num(v, 1),
      bands: fault ? [{ t0: fault.t, t1: rec ? rec.t : tr._end || 999, label: 'сбой датчика: шум в 5 раз больше' }] : [],
      marks: tr.events.filter(e => e.type === 'sample_collected').map(e => ({ t: e.t, label: 'образец собран', color: C.third })) });
  })();

  /* ---------------------------------------------- рисунок 7: фиксированный маршрут */

  (() => {
    const cv = $('fig-route'); if (!cv) return;
    onResize(() => {
      const T = stage(cv), { g } = T; drawFloor(T);
      for (const p of D.route.points) { g.beginPath(); g.arc(T.X(p[0]), T.Y(p[1]), 0.8 * T.s, 0, 2 * Math.PI); g.fillStyle = 'rgba(42,120,214,.07)'; g.fill(); }
      polyline(T, D.route.line, C.fixed, 2.5);
      D.route.points.forEach((p, i) => { g.beginPath(); g.arc(T.X(p[0]), T.Y(p[1]), 4.5 * T.dpr, 0, 2 * Math.PI); g.fillStyle = C.fixed; g.fill(); g.lineWidth = 2 * T.dpr; g.strokeStyle = '#fff'; g.stroke(); label(T, String(i + 1), T.X(p[0]) + 10 * T.dpr, T.Y(p[1]) - 9 * T.dpr, { size: 11, color: C.ink2 }); });
      drawBase(T);
    });
  })();

  /* ----------------------------------------------------- проигрыватель историй */

  (() => {
    const root = $('player'); if (!root) return;
    const pick = $('player-pick'); let current = null;
    function show(i) {
      if (current) current.stop();
      const s = D.stories[i];
      current = mountPlayer(root, [s.fixed, s.adaptive], ['Фиксированный план', 'С адаптацией'], [C.fixed, C.adaptive]);
      pick.querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', k === i));
    }
    D.stories.forEach((s, i) => { const b = el('button', 'seg', `${s.title} (hard № ${s.seed})`); b.onclick = () => show(i); pick.appendChild(b); });
    show(0);
  })();

  /* -------------------------------------------------------------- журнал прогона */

  (() => {
    const root = $('journal'); if (!root) return;
    const tr = story.adaptive;
    root.innerHTML = tr.journal.map(e => `<div class="jr k-${e.kind}"><span class="jt">${num(e.t, 0)} с</span><span class="jk">${KIND_RU[e.kind] || e.kind}</span>${e.text}</div>`).join('');
    const h = $('hyp-table'); if (!h) return;
    const ST = { confirmed: 'подтверждена', refuted: 'опровергнута', outdated: 'устарела', open: 'не проверена до конца прогона' };
    h.innerHTML = `<thead><tr><th>№</th><th>Когда</th><th>Гипотеза</th><th>Как проверяется</th><th>Итог</th></tr></thead><tbody>` + tr.hypotheses.map(x => `<tr><td>${x.id}</td><td>${num(x.t_open, 0)} с</td><td>${x.statement}</td><td>${x.test}</td><td><span class="chip ${x.status === 'confirmed' ? 'supported' : x.status === 'refuted' ? 'refuted' : 'inconclusive'}">${ST[x.status]}</span>${x.verdict ? `<br><small>${num(x.t_close, 0)} с: ${x.verdict}</small>` : ''}</td></tr>`).join('') + '</tbody>';
  })();

  /* ------------------------------------------------------------ пример обмена с моделью */

  (() => {
    const a = $('llm-state'), b = $('llm-reply'); if (!a) return;
    if (D.llm.error) { a.textContent = 'Пример недоступен: ' + D.llm.error; return; }
    const st = Object.assign({}, D.llm.state); delete st.mission;
    a.textContent = JSON.stringify(st, null, 1).replace(/\n {3,}/g, ' ').replace(/\n {2}\}/g, ' }');
    try { b.textContent = JSON.stringify(JSON.parse(D.llm.response), null, 1); } catch (e) { b.textContent = D.llm.response; }
  })();

  /* ---------------------------------------------------------------- Gazebo и быстрый */

  (() => {
    const root = $('gazebo'); if (!root) return;
    if (!D.gazebo.length) { root.appendChild(el('p', 'warn', 'Парных прогонов в Gazebo на момент сборки страницы нет.')); return; }
    D.gazebo.forEach(pair => {
      const box = el('div', 'gz');
      const cv = el('canvas'); const side = el('div', 'gz-side');
      const g = pair.gazebo.result, f = pair.fastsim.result;
      const row = (name, a, b, d = 1) => `<tr><td>${name}</td><td>${typeof a === 'string' ? a : num(a, d)}</td><td>${typeof b === 'string' ? b : num(b, d)}</td></tr>`;
      side.innerHTML = `<h4>Уровень ${pair.level}, сценарий № ${pair.gazebo.scenario.seed}</h4>
        <div class="lg"><span class="lg-i"><i style="background:${C.fixed}"></i>быстрый симулятор</span><span class="lg-i"><i style="background:${C.third}"></i>Gazebo (путь по одометрии робота — так его видел сам агент)</span></div>
        <table class="plain"><thead><tr><th></th><th>Быстрый</th><th>Gazebo</th></tr></thead><tbody>
        ${row('Собрано образцов', `${f.samples_collected} из ${f.samples_total}`, `${g.samples_collected} из ${g.samples_total}`)}
        ${row('Вернулся на базу', f.returned ? 'да' : 'нет', g.returned ? 'да' : 'нет')}
        ${row('Потрачено заряда', f.battery_used, g.battery_used)}${row('Путь, м', f.distance, g.distance)}
        ${row('Время, с', f.time, g.time, 0)}${row('Штрафы', f.penalties, g.penalties, 0)}${row('Счёт', f.score, g.score)}</tbody></table>`;
      box.append(cv, side); root.appendChild(box);
      onResize(() => {
        const T = stage(cv); drawFloor(T); const sc = pair.gazebo.scenario;
        drawSoils(T, sc.soils); drawHazards(T, sc.hazards); drawBase(T, sc.base); drawSamples(T, sc.samples);
        for (const [tr, col, dash] of [[pair.fastsim, C.fixed, null], [pair.gazebo, C.third, null]]) polyline(T, tr.track.x.map((x, i) => [x, tr.track.y[i]]), col, 2.4, dash);
      });
    });
  })();

  /* ------------------------------------------------------------------- опыты */

  (() => {
    if ($('exp-E1') && E.E1) {
      const r = $('exp-E1'); expHeader(r, 'E1'); expCharts(r, 'E1', ['samples_share', 'returned', 'battery_used', 'score'], false);
      const c = E.E1.claims.find(x => x.metric === 'samples_share');
      rowChart(r, { title: 'Насколько больше образцов собирает адаптивный агент на том же сценарии', sub: 'разность долей собранных образцов (адаптивный минус фиксированный); вертикальная черта — ноль, то есть «разницы нет»',
        zero: 0, fmt: v => (v > 0 ? '+' : '') + Math.round(v * 100) + ' п.п.', what: 'разность',
        rows: c.cells.map(x => ({ label: LEVEL_RU[x.level], mean: x.pair.mean, lo: x.pair.ci[0], hi: x.pair.ci[1], color: C.adaptive, n: x.pair.n, strong: x.level === 'all',
          extra: `адаптивный лучше в ${x.pair.a_higher}, хуже в ${x.pair.b_higher}, поровну в ${x.pair.ties}`, note: `лучше в ${x.pair.a_higher} из ${x.pair.n}` })) });
      claimList(r, 'E1');
    }
    if ($('exp-E2') && E.E2) {
      const r = $('exp-E2'); expHeader(r, 'E2');
      const full = stat('E2', 'adaptive', 'base', 'hard', 'score');
      rowChart(r, { title: 'Счёт на уровне hard, если выключить один механизм', sub: 'серая черта — полный агент; чем левее строка, тем важнее выключенный механизм',
        ref: full.mean, fmt: v => num(v, 1), what: 'счёт', head: ['Вариант', 'Счёт', '95% интервал', 'Собрано', 'Вернулся'],
        rows: E.E2.spec.arms.map(a => { const v = stat('E2', a.id, 'base', 'hard', 'score'), sm = stat('E2', a.id, 'base', 'hard', 'samples_share'), rt = stat('E2', a.id, 'base', 'hard', 'returned'); return { label: a.label, mean: v.mean, lo: v.lo, hi: v.hi, n: v.n, color: a.id === 'adaptive' ? C.adaptive : C.fixed, strong: a.id === 'adaptive', cells: [pct(sm.mean), pct(rt.mean)], extra: `собрано ${pct(sm.mean)}, вернулся в ${pct(rt.mean)} прогонов` }; }) });
      claimList(r, 'E2');
    }
    if ($('exp-E3') && E.E3) { const r = $('exp-E3'); expHeader(r, 'E3'); expCharts(r, 'E3', ['score', 'samples_share', 'false_collects', 'hazard_hits'], true); claimList(r, 'E3'); }
    if ($('exp-E4') && E.E4) { const r = $('exp-E4'); expHeader(r, 'E4'); expCharts(r, 'E4', ['samples_share', 'returned'], true); claimList(r, 'E4'); }
    if ($('exp-E5') && E.E5) { const r = $('exp-E5'); expHeader(r, 'E5'); expCharts(r, 'E5', ['samples_share', 'false_collects', 'distance'], true); claimList(r, 'E5'); }
    if ($('exp-E6') && E.E6) {
      const r = $('exp-E6'); expHeader(r, 'E6');
      const rows = E.E6.spec.arms.map(a => { const rs = E.E6.runs.filter(x => x.arm === a.id); const calls = rs.reduce((n, x) => n + (x.llm_calls || 0), 0), bad = rs.reduce((n, x) => n + (x.llm_failed || 0), 0), fb = rs.reduce((n, x) => n + ((x.plans && x.plans.fallback) || 0), 0); const sc = lv => stat('E6', a.id, 'base', lv, 'score');
        return `<tr><td>${a.label}</td><td>${num(sc('medium').mean, 1)}</td><td>${num(sc('hard').mean, 1)}</td><td>${num(calls / rs.length, 1)}</td><td>${calls ? pct(bad / calls) : '—'}</td><td>${num(fb / rs.length, 1)}</td><td>0</td></tr>`; }).join('');
      r.appendChild(el('table', 'plain', `<thead><tr><th>Вариант</th><th>Счёт, medium</th><th>Счёт, hard</th><th>Обращений к модели за прогон</th><th>Из них негодных ответов</th><th>Решений по запасному правилу за прогон</th><th>Аварий</th></tr></thead><tbody>${rows}</tbody>`));
      claimList(r, 'E6');
    }
  })();

  (() => {
    const block = (id, metrics, byCond) => { const r = $('exp-' + id); if (!r || !E[id]) return; expHeader(r, id); expCharts(r, id, metrics, byCond); claimList(r, id); };
    block('E8', ['score', 'samples_share', 'returned', 'collisions'], true);
    block('E9', ['samples_share', 'distance', 'battery_used', 'score'], false);
    block('E10', ['returned', 'samples_share', 'score', 'inq_correct', 'faults_found', 'inq_energy'], false);
    block('E11', ['faults_found', 'inq_insufficient', 'returned', 'score'], false);
    block('E12', ['study_error_pct', 'study_covered', 'study_halfwidth_pct', 'study_energy'], false);
    block('E13', ['returned', 'samples_share', 'score', 'battery_used'], false);
    block('E14', ['inq_wrong', 'inq_insufficient', 'faults_found', 'returned', 'score', 'inq_energy'], true);
    const tt = $('traps');
    if (tt && D.traps) tt.innerHTML = '<thead><tr><th style="width:34%">Условие</th><th>Выводов вынесено</th><th>Из них неверных</th><th>Доля неверных</th><th>«Недостаточно данных»</th></tr></thead><tbody>' +
      D.traps.map(r => `<tr><td>${r.label}</td><td>${r.identified}</td><td><b>${r.wrong}</b></td><td>${pct(r.identified ? r.wrong / r.identified : 0)}</td><td>${r.insufficient}</td></tr>`).join('') + '</tbody>';
    else if (tt) tt.remove();
  })();

  /* --------------------------------------------------------- расследования */

  (() => {
    const root = $('inq'), tr = D.science;
    if (!root || !tr || tr.error || !(tr.inquiries || []).length) { if (root) root.textContent = 'Пример расследований не собран.'; return; }
    const TOPIC = { energy: 'расход заряда', sensor: 'датчик образцов', fault: 'проверка после штрафа' };
    const TRUTH = { soil: 'дорогой грунт', leak: 'утечка заряда', noise: 'шум датчика', stuck: 'залипший датчик', bias: 'заниженные показания', none: 'сбоя нет', ok: 'датчик исправен' };
    const VERD = { correct: ['supported', '✓ вывод совпал с правдой'], wrong: ['refuted', '✗ вывод не совпал с правдой'], insufficient: ['inconclusive', '≈ вывода нет'], unverifiable: ['inconclusive', '· проверить нечем'] };
    const COL = [C.fixed, C.adaptive, C.third, '#8a5cd6', '#c9a227', C.ink3];
    const short = t => t.split(',')[0].split(':')[0].replace(/^после штрафа /, '');
    const pick = el('div', 'btns'), box = el('div', 'iq');
    root.append(pick, box);
    function scale(q, test) {
      const ids = q.alternatives.map(a => a.id).filter(id => test.predictions[id]);
      const lo0 = Math.min(...ids.map(id => test.predictions[id].mean - 2.5 * test.predictions[id].sigma), test.measured ? test.measured.value : Infinity);
      const hi0 = Math.max(...ids.map(id => test.predictions[id].mean + 2.5 * test.predictions[id].sigma), test.measured ? test.measured.value : -Infinity);
      const pad = (hi0 - lo0) * 0.08 || 0.1, lo = lo0 - pad, hi = hi0 + pad;
      const W = 640, left = 250, right = 24, rowH = 22, H = 30 + rowH * ids.length + 22;
      const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, class: 'iq-scale', role: 'img', 'aria-label': 'Предсказания и измерение: ' + test.name });
      const x = v => left + (W - left - right) * (v - lo) / (hi - lo);
      niceTicks(lo, hi, 4).forEach(v => { S('line', { x1: x(v), x2: x(v), y1: 22, y2: H - 20, stroke: C.line, 'stroke-width': 1 }, svg); S('text', { x: x(v), y: H - 6, 'text-anchor': 'middle', class: 'ax' }, svg, num(v, Math.abs(hi - lo) < 2 ? 2 : 1)); });
      ids.forEach((id, i) => {
        const a = q.alternatives.find(z => z.id === id), p = test.predictions[id], cy = 30 + rowH * (i + 0.5), col = COL[q.alternatives.indexOf(a) % COL.length];
        S('text', { x: left - 10, y: cy + 4, 'text-anchor': 'end', class: 'ax cat' }, svg, short(a.statement).slice(0, 42));
        S('line', { x1: x(p.mean - p.sigma), x2: x(p.mean + p.sigma), y1: cy, y2: cy, stroke: col, 'stroke-width': 4, 'stroke-linecap': 'round', opacity: 0.45 }, svg);
        const dot = S('circle', { cx: x(p.mean), cy, r: 5, fill: col, stroke: '#fff', 'stroke-width': 2 }, svg);
        tip(dot, `Если верно «${short(a.statement)}», опыт покажет около ${num(p.mean, 2)} ± ${num(p.sigma, 2)} ${test.unit}`);
      });
      if (test.measured) {
        const mx = x(test.measured.value);
        S('line', { x1: mx, x2: mx, y1: 16, y2: H - 20, stroke: C.ink, 'stroke-width': 2 }, svg);
        S('text', { x: Math.min(Math.max(mx, left + 40), W - 60), y: 12, 'text-anchor': 'middle', class: 'val' }, svg, 'измерено ' + num(test.measured.value, 2));
      }
      return svg;
    }
    function show(i) {
      const q = tr.inquiries[i]; box.innerHTML = '';
      pick.querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', k === i));
      const step = (n, title) => { const d = el('div', 'iq-step'); d.appendChild(el('h4', null, `<span>${n}</span>${title}`)); box.appendChild(d); return d; };
      step(1, 'Что заметил').appendChild(el('p', null, `<b>${num(q.t_open, 0)}-я секунда.</b> ${q.anomaly.text}.`));
      const s2 = step(2, 'Какие объяснения возможны: вероятность до опытов и после');
      q.alternatives.forEach((a, k) => s2.appendChild(el('div', 'iq-alt', `<span class="iq-name"><i style="background:${COL[k % COL.length]}"></i>${a.statement}</span>
        <span class="iq-bar"><i class="pr" style="width:${a.prior * 100}%"></i></span><span class="iq-p">${pct(a.prior)}</span>
        <span class="iq-arrow">→</span><span class="iq-bar"><i class="po" style="width:${a.posterior * 100}%;background:${COL[k % COL.length]}"></i></span><span class="iq-p"><b>${pct(a.posterior)}</b></span>`)));
      const s3 = step(3, 'Какие опыты были возможны и что каждое объяснение для них предсказывало');
      q.tests.forEach(t => { const d = el('div', 'iq-test' + (t.chosen ? ' on' : ''));
        d.appendChild(el('div', 'iq-tname', `${t.chosen ? '<b>Проведён:</b>' : 'Не понадобился:'} ${t.name} <small>ожидаемая польза ${num(t.gain_bits, 2)} бит, цена ${num(t.cost, 2)} ед. заряда; измеряется: ${t.unit}</small>`));
        d.appendChild(scale(q, t)); s3.appendChild(d); });
      const c = q.conclusion || {}, v = VERD[q.verdict] || VERD.unverifiable;
      step(4, 'Вывод').appendChild(el('p', null, `${chip(c.status === 'identified' ? 'supported' : 'inconclusive').replace(/>.*</, '>' + (c.status === 'identified' ? '✓ причина названа' : '≈ недостаточно данных') + '<')} ${c.text || ''}.<br>
        <small>Сверка со скрытой правдой сценария (агент её не видит): на самом деле — ${(q.truth || []).map(k => TRUTH[k] || k).join(' и ') || 'явной причины нет'}. <span class="chip ${v[0]}">${v[1]}</span></small>`));
      step(5, 'Что агент сделал дальше').appendChild(el('p', null, q.action || '—'));
    }
    tr.inquiries.forEach((q, i) => { const b = el('button', 'seg', `${q.id} · ${num(q.t_open, 0)} с · ${TOPIC[q.topic] || q.topic}`); b.onclick = () => show(i); pick.appendChild(b); });
    show(Math.max(0, tr.inquiries.findIndex(q => q.conclusion && q.conclusion.best === 'leak')));
    const m = tr.energy_model, law = $('inq-model');
    if (m && law) law.innerHTML = `<thead><tr><th>Что</th><th>Оценка агента к концу прогона</th><th>На самом деле (в правилах)</th></tr></thead><tbody>
      <tr><td>Метр по обычному полу</td><td>${num(m.per_m.value, 2)} ± ${num(m.per_m.sigma, 2)} ед.</td><td>${num(RULES.drain_per_m, 2)}</td></tr>
      <tr><td>Добавка на метр за каждый несомый образец</td><td>${num(m.per_m_load.value, 3)} ± ${num(m.per_m_load.sigma, 3)} ед.</td><td>0,125 (5% от 2,5)</td></tr>
      <tr><td>Поворот на радиан</td><td>${num(m.per_rad.value, 3)} ± ${num(m.per_rad.sigma, 3)} ед.</td><td>0,12</td></tr>
      <tr><td>Секунда простоя</td><td>${num(m.per_s.value, 3)} ± ${num(m.per_s.sigma, 3)} ед.</td><td>${num(RULES.drain_idle_per_s, 2)}</td></tr></tbody>`;
  })();

  (() => {
    const t = $('inq-acc'), a = D.inq_accuracy; if (!t) return;
    if (!a) { t.outerHTML = '<p class="warn">Опыт E10 на момент сборки не посчитан.</p>'; return; }
    const NAME = { energy: 'Почему вырос расход (грунт, утечка, повороты)', fault: 'Батарея после штрафа (утечка или нет)', sensor: 'Датчик образцов (шум, залипание, занижение, исправен)' };
    t.innerHTML = '<thead><tr><th style="width:38%">О чём вывод</th><th>Верно</th><th>Частично</th><th>Неверно</th><th>«Недостаточно данных»</th><th>Проверить нечем</th></tr></thead><tbody>' +
      Object.entries(a.rows).map(([k, r]) => `<tr><td>${NAME[k] || k}</td><td><b>${r.correct}</b></td><td>${r.partial}</td><td>${r.wrong}</td><td>${r.insufficient}</td><td>${r.unverifiable}</td></tr>`).join('') + '</tbody>';
  })();

  /* ---------------------------------------------------------------- память */

  (() => {
    const t = $('kb-table'); if (!t) return;
    if (!D.kb || !(D.kb.rules || []).length) { t.outerHTML = '<p class="warn">Память пуста: опыт E11 ещё не запускался.</p>'; return; }
    const ST = { confirmed: ['supported', '✓ подтверждено'], tentative: ['inconclusive', '≈ предварительно'], retired: ['refuted', '✗ под сомнением'] };
    t.innerHTML = `<thead><tr><th>Знание</th><th>Разброс между прогонами</th><th>Подтверждений</th><th>Противоречий</th><th>Статус</th></tr></thead><tbody>` +
      D.kb.rules.map(r => `<tr><td>${r.statement.replace(/(\d)\.(\d)/g, '$1,$2')}</td><td>${r.sigma ? '± ' + num(r.sigma, 3) + ' ' + r.unit : '—'}</td><td>${r.n}</td><td>${(r.contradictions || []).length}</td><td><span class="chip ${(ST[r.status] || ST.tentative)[0]}">${(ST[r.status] || ST.tentative)[1]}</span></td></tr>`).join('') + '</tbody>';
    const n = $('kb-runs'); if (n) n.textContent = D.kb.runs;
  })();

  /* ------------------------------------------------------- настоящая модель */

  (() => {
    const t = $('llm-real'); if (!t) return;
    const rows = Object.entries(D.llm_real || {});
    if (!rows.length) { t.outerHTML = '<p class="warn">Прогонов с настоящей моделью на момент сборки нет.</p>'; return; }
    const NAME = { 'gpt-6-luna_v1': 'GPT-6 Luna, первый вариант запроса', 'gpt-6-luna': 'GPT-6 Luna, доработанный запрос и схема ответа',
      'jev-router_800': 'Jev Router, лимит ответа 800 токенов', 'jev-router': 'Jev Router, запас на рассуждение и схема ответа',
      'mai-qwen3.8-flash-next': 'МАИ: qwen3.8-flash-next', 'mai-qwen3.8-27b': 'МАИ: qwen3.8-27b', 'mai-qwen3.6-35b-a3b': 'МАИ: qwen3.6-35b-a3b',
      'mai-qwen3.5-122b-a10b': 'МАИ: Qwen3.5-122B-A10B', 'mai-deepseek-v4.1-flash': 'МАИ: deepseek-v4.1-flash', 'mai-deepseek-v4-flash': 'МАИ: DeepSeek-V4-Flash',
      'qwen2.5-3b': 'Qwen 2.5 (3 млрд), локально — больше не используется' };
    const ORDER = ['mai-qwen3.8-flash-next', 'mai-qwen3.6-35b-a3b', 'mai-qwen3.8-27b', 'mai-qwen3.5-122b-a10b', 'mai-deepseek-v4.1-flash', 'mai-deepseek-v4-flash',
      'jev-router', 'jev-router_800', 'gpt-6-luna', 'gpt-6-luna_v1', 'qwen2.5-3b'];
    const rank = k => { const i = ORDER.indexOf(k); return i < 0 ? 99 : i; };
    rows.sort((a, b) => rank(a[0]) - rank(b[0]));
    const same = v => { const d = v.score_diff || []; return d.length ? `${d.filter(x => Math.abs(x) < 0.05).length} из ${d.length}` : '—'; };
    t.innerHTML = `<thead><tr><th>Модель и запрос</th><th>Обращений</th><th>Годный план с первого раза</th><th>После исправления</th><th>Отказ в запасное правило</th><th>Время ответа, медиана</th><th>Счёт: модель / правило</th><th>Сценариев, где счёт совпал с правилом</th></tr></thead><tbody>` +
      rows.map(([k, v]) => `<tr><td>${NAME[k] || k}</td><td>${v.requests}</td><td><b>${pct(v.first_ok_share)}</b></td><td>${pct(v.repaired_share)}</td><td>${pct(v.fallback_share)}</td><td>${num(v.latency_ms.median / 1000, 0)} с</td><td>${num(v.score_mean_complete, 1)} / ${num(v.rule_score_mean_complete, 1)} <small>(${v.complete_runs} прогонов)</small></td><td>${same(v)}</td></tr>`).join('') + '</tbody>';
  })();

  /* ------------------------------------------------- один такт на настоящих данных */

  (() => {
    const root = $('tick'); if (!root) return;
    const tr = prep(story.adaptive), tk = tr.track, n = tk.t.length;
    root.innerHTML = `<div class="tick-grid"><div><canvas id="tick-cv"></canvas>
      <input type="range" id="tick-t" min="1" max="${n - 2}" value="${Math.round(n * 0.3)}" style="width:100%">
      <div class="cap">Сценарий hard № ${story.seed}. Синее — карта вероятностей агента, пунктир — его текущий путь, кольцо — что значит последнее показание датчика.</div></div>
      <div><h4>1. Что пришло на вход</h4><pre id="tick-obs"></pre>
      <h4>2. Что агент об этом думает</h4><div id="tick-mind" class="tick-mind"></div>
      <h4>3. Что ушло роботу</h4><pre id="tick-cmd"></pre></div></div>`;
    const cv = $('tick-cv'), sl = $('tick-t');
    const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
    const f = (v, d = 2) => Number(v).toFixed(d);
    function draw() {
      const i = +sl.value, t = tk.t[i];
      drawFrame(cv, tr, t, { belief: true, path: true, ring: true, lidar: true }, C.adaptive);
      const evs = tr.events.filter(e => e.t > tk.t[i - 1] && e.t <= t);
      const z = tk.sensor[i], deg = Math.round(tk.th[i] * 180 / Math.PI);
      $('tick-obs').innerHTML = `Observation(
    t=${f(t, 1)},              <span class="c"># секунд от старта</span>
    x=${f(tk.x[i])}, y=${f(tk.y[i])},     <span class="c"># где робот, метры</span>
    th=${f(tk.th[i])},             <span class="c"># курс: ${deg}°</span>
    battery=${f(tk.battery[i], 1)},        <span class="c"># заряд из 60</span>
    sensor=${f(z)},          <span class="c"># ${z > 0.04 ? 'до образца ≈ ' + f((1 - z) * RULES.sensor_range_m, 1) + ' м' : 'ближе 2 м образцов нет'}</span>
    scan=[…360 чисел…],
    events=${evs.length ? '[' + evs.map(e => "'" + e.type + "'").join(', ') + ']' : '[]'},${evs.length ? '   <span class="c"># ' + evs.map(e => EVENT_RU[e.type] || e.type).join(', ') + '</span>' : ''}
)`;
      const mode = tr.modes[tk.mode[i]], plan = tr.plans[lastIdx(tr.plans, t, 't')], jr = tr.journal[lastIdx(tr.journal, t, 't')];
      const sg = plan && plan.subgoals && plan.subgoals[0];
      const SG = { investigate: 'проверить место', explore: 'разведать точку', goto: 'доехать до точки', return_base: 'вернуться на базу' };
      $('tick-mind').innerHTML = `<div><span>режим</span><b>${MODE_RU[mode] || mode}</b></div>
        <div><span>подцель</span>${sg ? `${SG[sg.type] || sg.type}${sg.x != null ? ` (${num(sg.x, 1)}; ${num(sg.y, 1)})` : ''}` : '—'}</div>
        <div><span>почему</span>${plan ? plan.reasoning : '—'}</div>
        <div><span>последняя запись журнала</span>${jr ? `${num(jr.t, 0)} с · ${KIND_RU[jr.kind] || jr.kind}: ${jr.text}` : '—'}</div>`;
      const dt = tk.t[i + 1] - t, v = Math.hypot(tk.x[i + 1] - tk.x[i], tk.y[i + 1] - tk.y[i]) / dt, w = wrap(tk.th[i + 1] - tk.th[i]) / dt;
      const got = tr.events.find(e => e.type === 'sample_collected' && Math.abs(e.t - t) <= dt);
      const say = mode === 'done' ? 'прогон закончен' : v < 0.02 && Math.abs(w) < 0.1 ? 'стоит на месте' : v < 0.02 ? 'разворот на месте ' + (w > 0 ? 'влево' : 'вправо') : `вперёд ${Math.round(v * 100)} см/с` + (Math.abs(w) > 0.15 ? ', подруливает ' + (w > 0 ? 'влево' : 'вправо') : ', прямо');
      $('tick-cmd').innerHTML = (got ? `<span class="c"># собрать образец → успех</span>\nio.collect()\n` : '') +
        `<span class="c"># ${say}</span>\nio.command(v=${f(v)}, w=${f(w)})`;
    }
    sl.oninput = draw; onResize(draw);
  })();

  /* ------------------------------------------------------- журнал исследований */

  (() => {
    const root = $('research-log'); if (!root) return;
    const R = D.research;
    if (!R || !(R.studies || []).length) { root.innerHTML = '<p class="warn">План исследований пуст.</p>'; return; }
    const ST = { idea: ['inconclusive', 'в очереди'], assigned: ['partial', 'в работе'], submitted: ['partial', 'сдано, перепроверяется'], returned: ['partial', 'возвращено на доработку'],
      verified: ['supported', 'перепроверено'], published: ['supported', 'готово'], rejected: ['refuted', 'снято'] };
    const esc = t => String(t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const md = t => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\n{2,}/g, '</p><p>').replace(/\n/g, ' ');
    const count = k => R.studies.filter(s => (ST[s.status] || [])[1] === k).length;
    root.appendChild(el('p', null, `Состояние на ${R.built}: исследований в плане <b>${R.studies.length}</b>, готово <b>${R.studies.filter(s => s.status === 'published' || s.status === 'verified').length}</b>, в работе <b>${R.studies.filter(s => ['assigned', 'submitted', 'returned'].includes(s.status)).length}</b>, в очереди <b>${count('в очереди')}</b>.`));
    R.studies.forEach(s => {
      const st = ST[s.status] || ST.idea;
      const card = el('div', 'study', `<div class="study-head"><span class="study-id">${esc(s.id)}</span><b>${esc(s.title)}</b><span class="chip ${st[0]}">${st[1]}</span></div>
        <div class="study-meta">${esc(s.criterion || '')}${s.experiment ? ' · опыт ' + esc(s.experiment) : ''}</div>
        ${s.why ? `<p>${esc(s.why)}</p>` : ''}
        ${s.question ? `<div class="exp-meta"><div><span>Вопрос</span>${esc(s.question)}</div>${s.hypothesis ? `<div><span>Гипотеза</span>${esc(s.hypothesis)}</div>` : ''}${s.refute ? `<div><span>Что опровергнет</span>${esc(s.refute)}</div>` : ''}</div>` : ''}
        ${s.conclusion ? `<div class="read"><b>Вывод</b><p style="margin:0">${md(s.conclusion)}</p>${s.limits ? `<p style="margin:8px 0 0;color:var(--ink-2)"><i>Ограничения.</i> ${md(s.limits)}</p>` : ''}</div>` : ''}
        ${s.note && !s.conclusion ? `<p style="color:var(--ink-2);font-size:14.5px">${esc(s.note)}</p>` : ''}`);
      root.appendChild(card);
    });
  })();

  /* ------------------------------------------------------------- карта кода */

  (() => {
    const t = $('code-map'); if (!t || !D.code) return;
    const ROWS = [
      ['Мир и судья', [['did/config.py', 'Правила: расход, датчик, штрафы, очки; набор «научных» правил'], ['did/arena.py', 'Карта арены из официального пакета: где свободно, сколько до стены, расчёт лучей лидара'],
        ['did/scenario.py', 'Генератор сценариев: уровень и номер → расстановка образцов, грунтов, опасных зон, расписание событий'], ['did/judge.py', 'Судья: заряд, показание датчика, штрафы, сбои, очки'],
        ['did/fastsim.py', 'Быстрый симулятор'], ['did/robot_io.py', '«Разъём» между агентом и миром']]],
      ['Агент', [['did/agent.py', 'Цикл агента и все его решения'], ['did/belief.py', 'Картина мира: образцы, грунт, опасные зоны, здоровье датчика'], ['did/nav.py', 'Самый дешёвый путь по сетке и ведение по нему'],
        ['did/localize.py', 'Поправка позы по лидару и карте стен'], ['did/planner.py', 'Выбор подцелей: правило и обёртка над языковой моделью'], ['did/journal.py', 'Журнал и гипотезы'], ['did/route.py', 'Маршрут фиксированного агента'],
        ['did/explore.py', 'Выбор точки разведки по ожидаемой пользе'], ['did/baselines.py', 'Простые стратегии для сравнения: «пока теплее» и спираль']]],
      ['Исследователь', [['did/energy.py', 'Формула расхода заряда, которую агент уточняет на ходу'], ['did/science.py', 'Расследование как расчёт: объяснения, польза опыта, правило Байеса'], ['did/inquiry.py', 'Сам исследователь: что считать странностью, какие ставить опыты, что делать с выводом'],
        ['did/memory.py', 'Память между прогонами'], ['did/foresight.py', 'Сравнение будущих маршрутов по риску'], ['did/study.py', 'Исследование по заданию человека: план замеров, оценка с погрешностью'], ['did/study_agent.py', 'Агент, который исполняет такое задание'], ['did/study_law.py', 'Проверка гипотез о законе датчика']]],
      ['Языковая модель', [['did/llm.py', 'Обращение к модели, проверка ответа, запасное правило'], ['did/llm_roles.py', 'Роли автора и критика в расследованиях'], ['did/llm_codex.py', 'Доступ к GPT по подписке'], ['did/llm_mock.py', 'Имитатор модели с настраиваемыми ошибками'], ['did/llm_eval.py', 'Сравнение моделей на одних сценариях']]],
      ['Запуск и измерение', [['did/runner.py', 'Один прогон от сценария до файла записи'], ['did/recorder.py', 'Запись прогона'], ['did/metrics.py', 'Метрики, парные разности, сверка выводов исследователя с правдой'], ['did/experiments.py', 'Серии опытов по описаниям из experiments/*.yaml']]],
      ['ROS 2 и показ', [['did/ros_agent.py', 'Тот же агент, подключённый к топикам ROS'], ['ws/src/did_ros/did_ros/judge_node.py', 'Тот же судья как узел ROS'], ['did/pilot.py', 'Пульт: карта по лидару, маршрут кликами, автономная миссия'], ['did/mapping.py', 'Построение карты по лидару'], ['tools/demo.py', 'Запуск показа одной командой']]],
      ['Интерфейс', [['did/lab/server.py', 'Веб-сервер Лаборатории'], ['lab/replay.js', 'Проигрыватель прогонов'], ['lab/views/experiment.js', 'Страница опыта'], ['lab/views/pilot.js', 'Страница «Пульт»'], ['lab/views/study.js', 'Конструктор исследования']]],
    ];
    t.innerHTML = '<thead><tr><th style="width:30%">Файл</th><th>Что в нём</th><th style="width:9%;text-align:right">Строк</th></tr></thead><tbody>' +
      ROWS.map(([title, rows]) => `<tr><td colspan="3" class="group">${title}</td></tr>` + rows.map(([file, what]) => `<tr><td><code>${file}</code></td><td>${what}</td><td style="text-align:right">${D.code.files[file] != null ? D.code.files[file].toLocaleString('ru') : '—'}</td></tr>`).join('')).join('') + '</tbody>';
  })();

  /* ------------------------------------------------------------ оглавление */

  (() => {
    const toc = $('toc'); if (!toc) return;
    const secs = [...document.querySelectorAll('main section[id]')];
    secs.forEach((s, i) => { const h = s.querySelector('h2'); if (!h) return; const n = h.querySelector('.n'); if (n) n.textContent = i + 1;
      if (s.dataset.part) { toc.appendChild(el('span', 'part', s.dataset.part)); s.insertBefore(el('div', 'part-mark', s.dataset.part), s.firstChild); }
      const a = el('a', null, h.dataset.short || h.textContent.replace(/^\d+/, '')); a.href = '#' + s.id; toc.appendChild(a); });
    for (const id of ['E12', 'E13', 'E14']) { const w = $('wrap-' + id); if (w && !E[id]) w.style.display = 'none'; }
    const links = [...toc.querySelectorAll('a')];
    const mark = () => { let cur = secs[0]; for (const sec of secs) if (sec.getBoundingClientRect().top <= 160) cur = sec; links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + cur.id)); };
    window.addEventListener('scroll', mark, { passive: true }); mark();
    // explain.html?at=<id> — сразу показать нужное место (для снимков экрана)
    const at = new URLSearchParams(location.search).get('at');
    if (at && $(at)) { const top = $(at).getBoundingClientRect().top + window.scrollY - 70; document.documentElement.dataset.at = Math.round(top); window.scrollTo({ top, behavior: 'instant' }); }
  })();
})();
