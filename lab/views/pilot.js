// Пульт: оператор ставит точки на карте — робот едет; карта по лидару проявляется на глазах.
// Состояние приходит с сервера четыре раза в секунду (GET /api/pilot/state), команды уходят
// POST /api/pilot/command. Экран один и тот же для Gazebo и для быстрого симулятора.

import { h, fill, api, getArena, num, icon, loading, errorBox, levelName, soilColor, multLabel, runHref, LEVEL_ORDER } from './common.js';

const POLL_MS = 250;
const MARGIN = 0.16;            // поле вокруг арены, м
const FRESH_MS = 1600;          // сколько только что увиденная клетка остаётся подсвеченной
const ROBOT_R = 0.105;          // радиус Burger, м

const COLOR = {
  fog: [208, 215, 223],         // пол, который робот ещё не видел (эталонная карта, бледно)
  seen: [255, 255, 255],        // пол, увиденный лидаром
  fresh: [255, 205, 178],       // только что увиденный пол
  wall: [20, 24, 28],           // преграда по лидару
  ray: '42, 120, 214',
  trail: '#eb6834',
  path: '#2a78d6',
  robot: '#eb6834',
  ink: '#14181c',
  green: '#1baf7a',
  red: '#d64545',
};

const AGENT_MODE = {
  start: 'начинает', explore: 'разведка: едет туда, где ещё не искал', travel: 'едет к точке маршрута',
  approach: 'подъезжает к месту, где датчик показывает образец', collect: 'берёт образец',
  return: 'возвращается на базу', think: 'думает над планом', escape: 'отъезжает от преграды', done: 'закончил',
  probe: 'ставит опыт', experiment: 'ставит опыт', lost: 'потерял положение: стоит и ищет себя по лидару',
};
const KIND = {
  observe: 'наблюдение', hypothesis: 'гипотеза', verdict: 'вывод', decision: 'решение', action: 'действие',
  alarm: 'тревога', llm: 'модель', inquiry: 'расследование',
};
const REASON = { finish: 'миссия завершена', battery: 'села батарея', timeout: 'вышло время' };

function addCss(url) {
  if (document.querySelector(`link[data-lb-css="${url}"]`)) return;
  document.head.append(h('link', { rel: 'stylesheet', href: url, 'data-lb-css': url }));
}

function decode(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const cm = (m) => num(m * 100, 0);

// =================================================================================================
// карта
// =================================================================================================

function createMap(arena, onClick, fps = 30) {
  const canvas = h('canvas', { class: 'pl-map__canvas', 'aria-label': 'Карта арены. Клик добавляет точку маршрута' });
  const el = h('div', { class: 'pl-map' }, canvas);
  const ctx = canvas.getContext('2d');
  const { w: GW, h: GH, res, x0, y0 } = arena;

  // Рамка рисунка — по свободным клеткам арены с небольшим полем.
  let c0 = GW; let c1 = -1; let r0 = GH; let r1 = -1;
  const free = new Uint8Array(GW * GH);
  for (let r = 0; r < GH; r++) {
    for (let c = 0; c < GW; c++) {
      if (arena.free[r].charCodeAt(c) !== 49) continue;
      free[r * GW + c] = 1;
      if (c < c0) c0 = c;
      if (c > c1) c1 = c;
      if (r < r0) r0 = r;
      if (r > r1) r1 = r;
    }
  }
  const bx0 = x0 + c0 * res - MARGIN;
  const bx1 = x0 + (c1 + 1) * res + MARGIN;
  const by0 = y0 + r0 * res - MARGIN;
  const by1 = y0 + (r1 + 1) * res + MARGIN;
  const aspect = (bx1 - bx0) / (by1 - by0);

  // Слои-картинки: одна клетка — одна точка, при показе растягиваются без сглаживания.
  const layer = () => {
    const cv = document.createElement('canvas');
    cv.width = GW;
    cv.height = GH;
    return cv;
  };
  const refCv = layer();
  const mapCv = layer();
  const mapImg = mapCv.getContext('2d').createImageData(GW, GH);
  {
    const img = refCv.getContext('2d').createImageData(GW, GH);
    for (let r = 0; r < GH; r++) {
      for (let c = 0; c < GW; c++) {
        if (!free[r * GW + c]) continue;
        const k = ((GH - 1 - r) * GW + c) * 4;
        img.data[k] = COLOR.fog[0]; img.data[k + 1] = COLOR.fog[1]; img.data[k + 2] = COLOR.fog[2]; img.data[k + 3] = 255;
      }
    }
    refCv.getContext('2d').putImageData(img, 0, 0);
  }
  let heatCv = null;
  let heatKey = '';

  let st = null;                 // последнее состояние с сервера
  let opts = { rays: true, ref: true, truth: false, belief: true, slam: false };
  let grid = null;               // расшифрованная карта занятости
  let gridVersion = -1;
  const seenAt = new Float64Array(GW * GH);
  let freshUntil = 0;
  let mapDirty = true;
  let shown = null;              // поза на экране: плавно догоняет позу из состояния
  let from = null;
  let fromT = 0;
  let span = POLL_MS;
  let lastStateT = 0;
  let hover = null;
  let flashes = [];              // отклонённые клики: [{x, y, until}]
  let ghost = null;              // точка, которую сервер ещё не подтвердил
  let W = 0; let H = 0; let scale = 1;
  let raf = 0;
  let lastDraw = 0;
  let dead = false;

  const X = (x) => (x - bx0) * scale;
  const Y = (y) => (by1 - y) * scale;

  function resize() {
    const grid = el.closest('.pl-grid');
    if (!grid) return;
    // Ширина карты — что осталось от колонок пульта (см. pilot.css), высота — чтобы всё помещалось в экран.
    const full = grid.clientWidth;
    const avail = window.innerWidth >= 1280 ? full - 2 * 330 - 2 * 20 : window.innerWidth >= 900 ? full - 360 - 20 : full;
    const top = el.getBoundingClientRect().top + window.scrollY;
    const maxH = Math.max(360, window.innerHeight - Math.min(top, 260) - 92);
    const width = Math.max(300, Math.min(avail, maxH * aspect));
    if (Math.round(width) === W) return;
    W = Math.round(width);
    H = Math.round(width / aspect);
    el.style.width = `${W}px`;
    el.style.height = `${H}px`;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    scale = W / (bx1 - bx0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    el.dispatchEvent(new CustomEvent('pl-resize', { detail: { height: H } }));
  }

  function paintMap(now) {
    const d = mapImg.data;
    let fresh = false;
    for (let r = 0; r < GH; r++) {
      const src = r * GW;
      let k = (GH - 1 - r) * GW * 4;
      for (let c = 0; c < GW; c++, k += 4) {
        const v = grid ? grid[src + c] : 0;
        if (!v) { d[k + 3] = 0; continue; }
        const p = (v - 1) / 254;
        if (p > 0.5) {
          d[k] = COLOR.wall[0]; d[k + 1] = COLOR.wall[1]; d[k + 2] = COLOR.wall[2];
          d[k + 3] = Math.round(255 * clamp((p - 0.5) * 2.6, 0.45, 1));
          continue;
        }
        const age = now - seenAt[src + c];
        let f = 0;
        if (age < FRESH_MS) { f = 1 - age / FRESH_MS; fresh = true; }
        d[k] = COLOR.seen[0] + (COLOR.fresh[0] - COLOR.seen[0]) * f;
        d[k + 1] = COLOR.seen[1] + (COLOR.fresh[1] - COLOR.seen[1]) * f;
        d[k + 2] = COLOR.seen[2] + (COLOR.fresh[2] - COLOR.seen[2]) * f;
        d[k + 3] = 255;
      }
    }
    mapCv.getContext('2d').putImageData(mapImg, 0, 0);
    mapDirty = false;
    freshUntil = fresh ? now + 50 : 0;
  }

  function heat(b) {
    if (!b || !b.data) return null;
    if (heatCv && b.data === heatKey) return heatCv;        // карта приходит раз в секунду, а кадры рисуются чаще
    heatKey = b.data;
    if (!heatCv) heatCv = document.createElement('canvas');
    heatCv.width = b.w;
    heatCv.height = b.h;
    const img = heatCv.getContext('2d').createImageData(b.w, b.h);
    const src = decode(b.data);
    for (let r = 0; r < b.h; r++) {
      for (let c = 0; c < b.w; c++) {
        const v = src[r * b.w + c] / 255;           // корень из вероятности
        const k = ((b.h - 1 - r) * b.w + c) * 4;
        img.data[k] = 27; img.data[k + 1] = 175; img.data[k + 2] = 122;
        img.data[k + 3] = v < 0.08 ? 0 : Math.round(255 * clamp(v * 2.2, 0, 0.85));
      }
    }
    heatCv.getContext('2d').putImageData(img, 0, 0);
    return heatCv;
  }

  function zonePath(z) {
    ctx.beginPath();
    if (z.shape === 'circle') ctx.arc(X(z.x), Y(z.y), z.r * scale, 0, Math.PI * 2);
    else ctx.rect(X(z.x - z.w / 2), Y(z.y + z.h / 2), z.w * scale, z.h * scale);
  }

  function marker(x, y, label, fill, ring) {
    const r = clamp(scale * 0.085, 11, 17);
    ctx.beginPath();
    ctx.arc(X(x), Y(y), r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = ring || '#fff';
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = `700 ${Math.round(r * 1.05)}px -apple-system, "Segoe UI", Roboto, Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, X(x), Y(y) + 0.5);
  }

  function draw(now) {
    if (!W) return;
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    const gx = X(x0);
    const gy = Y(y0 + GH * res);
    const gw = GW * res * scale;
    const gh = GH * res * scale;
    if (opts.ref) ctx.drawImage(refCv, gx, gy, gw, gh);
    if (mapDirty || (freshUntil && now < freshUntil + 60)) paintMap(now);
    ctx.drawImage(mapCv, gx, gy, gw, gh);
    if (!st || !st.active) return;

    const mission = st.mission;
    if (opts.belief && mission && mission.belief && mission.state !== 'to_base') {
      const b = mission.belief;
      const cv = heat(b);
      if (cv) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(cv, X(b.x0), Y(b.y0 + b.h * b.res), b.w * b.res * scale, b.h * b.res * scale);
        ctx.imageSmoothingEnabled = false;
      }
      for (const [zx, zy, zr] of mission.hazards || []) {
        ctx.beginPath();
        ctx.arc(X(zx), Y(zy), zr * scale, 0, Math.PI * 2);
        ctx.setLineDash([7, 5]);
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = COLOR.red;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    const truth = opts.truth ? st.truth : null;
    if (truth) {
      for (const z of truth.soils || []) {
        zonePath(z);
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = soilColor(z.mult);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = COLOR.ink;
        ctx.font = '650 15px -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(multLabel(z.mult), X(z.x), Y(z.y));
      }
      for (const z of truth.hazards || []) {
        zonePath(z);
        ctx.fillStyle = 'rgba(214, 69, 69, 0.22)';
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = COLOR.red;
        ctx.stroke();
      }
      const diamond = (p, filled) => {
        const s = clamp(scale * 0.07, 8, 13);
        ctx.beginPath();
        ctx.moveTo(X(p[0]), Y(p[1]) - s);
        ctx.lineTo(X(p[0]) + s, Y(p[1]));
        ctx.lineTo(X(p[0]), Y(p[1]) + s);
        ctx.lineTo(X(p[0]) - s, Y(p[1]));
        ctx.closePath();
        ctx.fillStyle = filled ? COLOR.green : '#fff';
        ctx.fill();
        ctx.lineWidth = filled ? 1.5 : 3;
        ctx.strokeStyle = filled ? '#0d7a54' : COLOR.green;
        ctx.stroke();
      };
      for (const p of truth.collected || []) diamond(p, false);
      for (const p of truth.remaining || []) diamond(p, true);
    }

    // База.
    const [bx, by] = st.base || [-2, -0.5];
    ctx.beginPath();
    ctx.arc(X(bx), Y(by), 0.3 * scale, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(20, 24, 28, 0.06)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(74, 83, 92, 0.7)';
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#4a535c';
    ctx.font = '650 14px -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('База', X(bx), Y(by) + 0.3 * scale + 4);

    // Лучи лидара последнего скана.
    const sc = st.scan;
    if (opts.rays && sc && sc.r) {
      const ox = X(sc.x);
      const oy = Y(sc.y);
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(${COLOR.ray}, 0.17)`;
      ctx.beginPath();
      const ends = [];
      for (let i = 0; i < sc.r.length; i++) {
        if (!sc.r[i]) continue;
        const a = sc.th + i * sc.step;
        const ex = X(sc.x + Math.cos(a) * sc.r[i] / 100);
        const ey = Y(sc.y + Math.sin(a) * sc.r[i] / 100);
        ctx.moveTo(ox, oy);
        ctx.lineTo(ex, ey);
        ends.push(ex, ey);
      }
      ctx.stroke();
      ctx.fillStyle = `rgba(${COLOR.ray}, 0.95)`;
      for (let i = 0; i < ends.length; i += 2) ctx.fillRect(ends[i] - 1.5, ends[i + 1] - 1.5, 3, 3);
    }

    // След.
    const tr = st.trail || [];
    if (tr.length > 1) {
      ctx.beginPath();
      ctx.moveTo(X(tr[0][0]), Y(tr[0][1]));
      for (let i = 1; i < tr.length; i++) ctx.lineTo(X(tr[i][0]), Y(tr[i][1]));
      if (shown) ctx.lineTo(X(shown[0]), Y(shown[1]));
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.strokeStyle = COLOR.trail;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Построенный путь и точки маршрута.
    const path = st.path || [];
    if (path.length > 1) {
      ctx.beginPath();
      ctx.moveTo(X(path[0][0]), Y(path[0][1]));
      for (let i = 1; i < path.length; i++) ctx.lineTo(X(path[i][0]), Y(path[i][1]));
      ctx.lineWidth = 3.5;
      ctx.setLineDash([9, 7]);
      ctx.strokeStyle = COLOR.path;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    (st.route || []).forEach((p, i) => {
      if (st.mode === 'home') return;
      marker(p.x, p.y, String(i + 1), p.done ? '#9aa3ab' : COLOR.path);
    });
    if (ghost) marker(ghost.x, ghost.y, '…', 'rgba(42, 120, 214, 0.55)');
    flashes = flashes.filter((f) => f.until > now);
    for (const f of flashes) {
      const a = clamp((f.until - now) / 500, 0, 1);
      ctx.globalAlpha = a;
      ctx.lineWidth = 4;
      ctx.strokeStyle = COLOR.red;
      ctx.beginPath();
      ctx.moveTo(X(f.x) - 10, Y(f.y) - 10); ctx.lineTo(X(f.x) + 10, Y(f.y) + 10);
      ctx.moveTo(X(f.x) + 10, Y(f.y) - 10); ctx.lineTo(X(f.x) - 10, Y(f.y) + 10);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Истинное положение (только быстрый симулятор, вместе со скрытой правдой).
    if (truth && truth.robot) {
      ctx.beginPath();
      ctx.arc(X(truth.robot[0]), Y(truth.robot[1]), Math.max(ROBOT_R * scale, 9) + 4, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = COLOR.ink;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Робот с курсом.
    if (shown) {
      const rr = Math.max(ROBOT_R * scale, 10);
      const px = X(shown[0]);
      const py = Y(shown[1]);
      const th = shown[2];
      ctx.beginPath();
      ctx.arc(px, py, rr, 0, Math.PI * 2);
      ctx.fillStyle = COLOR.robot;
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px + Math.cos(th) * rr * 1.75, py - Math.sin(th) * rr * 1.75);
      ctx.lineTo(px + Math.cos(th + 2.25) * rr * 0.8, py - Math.sin(th + 2.25) * rr * 0.8);
      ctx.lineTo(px + Math.cos(th - 2.25) * rr * 0.8, py - Math.sin(th - 2.25) * rr * 0.8);
      ctx.closePath();
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = COLOR.ink;
      ctx.stroke();
    }

    // Координаты под курсором.
    if (hover) {
      const text = `x ${num(hover.x, 2)}  y ${num(hover.y, 2)}`;
      ctx.font = '600 13px ui-monospace, Menlo, Consolas, monospace';
      const tw = ctx.measureText(text).width + 14;
      let lx = X(hover.x) + 14;
      let ly = Y(hover.y) + 16;
      if (lx + tw > W - 4) lx = X(hover.x) - tw - 10;
      if (ly + 24 > H - 4) ly = Y(hover.y) - 30;
      ctx.fillStyle = 'rgba(20, 24, 28, 0.86)';
      ctx.beginPath();
      ctx.roundRect(lx, ly, tw, 22, 6);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, lx + 7, ly + 11.5);
    }
  }

  function frame(now) {
    if (dead) return;
    raf = requestAnimationFrame(frame);
    if (now - lastDraw < 1000 / fps - 3) return;        // не чаще ~30 кадров в секунду: рядом работает Gazebo
    lastDraw = now;
    if (st && st.active && st.pose && from) {
      const k = clamp((now - fromT) / span, 0, 1);
      shown = [from[0] + (st.pose[0] - from[0]) * k, from[1] + (st.pose[1] - from[1]) * k,
        from[2] + wrapAngle(st.pose[2] - from[2]) * k];
    }
    draw(now);
  }

  function toWorld(ev) {
    const r = canvas.getBoundingClientRect();
    return { x: bx0 + (ev.clientX - r.left) / scale, y: by1 - (ev.clientY - r.top) / scale };
  }
  canvas.addEventListener('click', (ev) => onClick(toWorld(ev)));
  canvas.addEventListener('mousemove', (ev) => { hover = toWorld(ev); });
  canvas.addEventListener('mouseleave', () => { hover = null; });
  const ro = new ResizeObserver(resize);
  window.addEventListener('resize', resize);

  return {
    el,
    mount() { ro.observe(el.closest('.pl-grid')); resize(); raf = requestAnimationFrame(frame); },
    setOpts(o) { opts = { ...opts, ...o }; },
    setGhost(g) { ghost = g; },
    reject(p) { flashes.push({ ...p, until: performance.now() + 1800 }); },
    setState(next) {
      const now = performance.now();
      if (next && next.active && next.pose) {
        const jump = shown ? Math.hypot(next.pose[0] - shown[0], next.pose[1] - shown[1]) > 0.6 : true;
        from = jump ? next.pose : (shown || next.pose);
        if (jump) shown = next.pose.slice();
        span = clamp(now - lastStateT, 120, 600);
        fromT = now;
        lastStateT = now;
      }
      const m = next && (opts.slam && next.slam ? next.slam : next.map);
      const version = m ? `${opts.slam && next.slam ? 's' : 'm'}${m.version}` : -1;
      if (m && m.data && version !== gridVersion) {
        const g = decode(m.data);
        const restart = String(gridVersion)[0] !== version[0] || Number(version.slice(1)) < Number(String(gridVersion).slice(1));
        if (restart) { seenAt.fill(0); grid = null; }               // карту сбросили или переключили
        const stamp = grid ? now : now - FRESH_MS;                  // при открытии страницы готовая карта не вспыхивает
        for (let i = 0; i < g.length; i++) {
          if (g[i] && !seenAt[i]) seenAt[i] = stamp;
          else if (!g[i]) seenAt[i] = 0;
        }
        grid = g;
        gridVersion = version;
        mapDirty = true;
      } else if (!m && grid) {
        grid = null;
        gridVersion = -1;
        seenAt.fill(0);
        mapDirty = true;
      }
      st = next;
    },
    destroy() {
      dead = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', resize);
    },
  };
}

// =================================================================================================
// страница
// =================================================================================================

function meter(share, tone) {
  return h('div', { class: 'pl-meter', data: { tone: tone || 'ink' } }, h('i', { style: { width: `${clamp(share, 0, 1) * 100}%` } }));
}

function tile(label) {
  const value = h('div', { class: 'pl-tile__value', text: '—' });
  const sub = h('div', { class: 'pl-tile__sub' });
  const bar = h('div', { class: 'pl-tile__bar' });
  return { el: h('div', { class: 'pl-tile' }, h('div', { class: 'pl-tile__label', text: label }), value, bar, sub), value, sub, bar };
}

function check(label, checked, onchange) {
  const input = h('input', { type: 'checkbox', class: 'pl-check__box' });
  input.checked = checked;
  input.addEventListener('change', () => onchange(input.checked));
  return h('label', { class: 'pl-check' }, input, h('span', { text: label }));
}

export async function render(root, ctx) {
  addCss('pilot.css');
  root.append(loading('Открываю пульт…'));
  let arena;
  try {
    arena = await getArena();
  } catch (e) {
    root.replaceChildren(errorBox('Не удалось загрузить арену', e.message, ctx.reload));
    return;
  }
  if (!ctx.alive()) return;

  let st = null;                 // последнее состояние
  let flash = null;              // сообщение страницы поверх сообщения пульта: {tone, text, until}
  let timer = null;
  let busy = false;
  let fastForm = { level: 'medium', seed: 3 };
  let formTouched = false;

  // --- команды ---------------------------------------------------------------------------------

  const say = (tone, text, ms = 6000) => { flash = { tone, text, until: Date.now() + ms }; paint(); };

  async function send(cmd, extra, quiet) {
    try {
      const r = await api.post('/api/pilot/command', { cmd, ...(extra || {}) });
      flash = null;
      if (!quiet && r && r.message) say('info', r.message, 2500);
      poll(true);
      return r;
    } catch (e) {
      say('bad', e.message);
      return null;
    }
  }

  const pending = () => ((st && st.route) || []).filter((p) => !p.done).map((p) => [p.x, p.y]);

  async function addPoint(p) {
    if (!st || !st.active) return say('info', 'Пульт не подключён к роботу: запустите показ или быстрый симулятор');
    if (st.mode === 'mission') return say('info', 'Идёт автономная миссия: роботом управляет агент');
    if (st.mode === 'wait') return say('info', 'Стенд Gazebo ещё загружается');
    const pts = st.mode === 'home' ? [] : pending();
    map.setGhost(p);
    const r = await send('route', { points: [...pts, [p.x, p.y]] }, true);
    map.setGhost(null);
    if (!r) map.reject(p);
  }

  async function start(body) {
    if (busy) return;
    busy = true;
    try {
      await api.post('/api/pilot/start', body);
      flash = null;
      await poll(true);
    } catch (e) {
      say('bad', e.message);
    }
    busy = false;
  }

  // --- разметка --------------------------------------------------------------------------------

  // #/pilot?fps=10 — реже перерисовывать карту (для записи видео на занятой машине).
  const map = createMap(arena, addPoint, clamp(Number(ctx.query.fps) || 30, 2, 60));
  const lamp = h('span', { class: 'lb-lamp', 'aria-hidden': 'true' });
  const statusText = h('div', { class: 'pl-status__text' });
  const statusSub = h('div', { class: 'pl-status__sub' });
  const noteBox = h('div', { class: 'pl-note', role: 'status', 'aria-live': 'polite' });
  const lagBox = h('div', { class: 'pl-lag', hidden: true });
  const sourceChip = h('span', { class: 'lb-chip lb-chip--big pl-source' });

  const btn = (label, cls, onclick, title) => h('button', { class: `lb-btn pl-btn ${cls || ''}`, type: 'button', onclick, title }, label);
  const goBtn = btn([icon('play', 18), 'Ехать'], 'lb-btn--primary pl-btn--go', () => send('go'));
  const stopBtn = btn('Стоп', 'pl-btn--stop', () => send('stop'));
  const homeBtn = btn('Домой', '', () => send('home'), 'Вернуться на базу (−2,0; −0,5)');
  const resetBtn = btn('Сбросить', '', () => send('reset'), 'Стереть карту и след, начать прогон заново');
  // Только в режиме карты SLAM (pixi run demo --slam-map): робот сам едет к границам увиденного.
  const exploreBtn = btn('Построить карту', 'pl-btn--explore', () => send('explore'),
    'Робот сам объедет арену: едет туда, где карта SLAM Toolbox ещё обрывается');
  exploreBtn.hidden = true;
  const undoBtn = btn('Убрать точку', 'pl-btn--small',  () => send('route', { points: pending().slice(0, -1) }, true));
  const clearBtn = btn('Очистить', 'pl-btn--small', () => send('route', { points: [] }, true));
  const routeInfo = h('div', { class: 'pl-route__info' });
  const routeDots = h('div', { class: 'pl-route__dots' });

  const tiles = {
    battery: tile('Заряд'), sensor: tile('Датчик образцов'), dist: tile('Пройдено'),
    cover: tile('Карта построена'), agree: tile('Совпадение с эталоном'), pose: tile('Откуда поза'),
  };

  const missionBody = h('div', { class: 'pl-mission__body' });
  const eventsBox = h('ul', { class: 'pl-events' });
  const sourceBody = h('div', { class: 'pl-sourcecard__body' });
  let agentChoice = 'adaptive';

  const colDrive = h('div', { class: 'pl-col' },
    h('section', { class: 'lb-card pl-card pl-status' },
      h('div', { class: 'pl-status__row' }, lamp, statusText),
      statusSub, noteBox, lagBox),
    h('section', { class: 'lb-card pl-card' },
      h('div', { class: 'pl-card__head' }, h('h2', { class: 'pl-h', text: 'Маршрут' }), routeInfo),
      h('div', { class: 'pl-route' }, routeDots, h('div', { class: 'pl-route__edit' }, undoBtn, clearBtn)),
      h('div', { class: 'pl-buttons' }, goBtn, stopBtn, homeBtn, resetBtn, exploreBtn)),
    h('section', { class: 'pl-tiles' }, Object.values(tiles).map((t) => t.el)));
  const colMission = h('div', { class: 'pl-col' },
    h('section', { class: 'lb-card pl-card pl-mission' },
      h('div', { class: 'pl-card__head' }, h('h2', { class: 'pl-h', text: 'Автономная миссия' })),
      missionBody),
    h('section', { class: 'lb-card pl-card' },
      h('div', { class: 'pl-card__head' }, h('h2', { class: 'pl-h', text: 'События' })),
      eventsBox),
    h('section', { class: 'lb-card pl-card pl-sourcecard' },
      h('div', { class: 'pl-card__head' }, h('h2', { class: 'pl-h', text: 'Где едет робот' })),
      sourceBody));

  const legend = h('div', { class: 'pl-legend' },
    h('span', { class: 'pl-key' }, h('i', { class: 'pl-key__seen' }), 'пол: увидел лидар'),
    h('span', { class: 'pl-key' }, h('i', { class: 'pl-key__wall' }), 'преграда'),
    h('span', { class: 'pl-key' }, h('i', { class: 'pl-key__fog' }), 'ещё не видел'),
    h('span', { class: 'pl-key' }, h('i', { class: 'pl-key__trail' }), 'след'),
    h('span', { class: 'pl-key' }, h('i', { class: 'pl-key__path' }), 'путь'));
  const refCheck = check('Эталонная карта', true, (v) => map.setOpts({ ref: v }));
  const toggles = h('div', { class: 'pl-toggles' },
    check('Показать скрытую правду', false, (v) => map.setOpts({ truth: v })),
    check('Лучи лидара', true, (v) => map.setOpts({ rays: v })),
    refCheck,
    check('Что думает агент', true, (v) => map.setOpts({ belief: v })));
  // Переключатель карты появляется, когда рядом работает SLAM Toolbox (pixi run demo --slam).
  let slamOn = false;
  let slamNavSeen = false;       // режим карты SLAM уже замечен: карта SLAM включена, эталонная скрыта
  const slamSeg = h('div', { class: 'lb-seg pl-slam', role: 'group', 'aria-label': 'Чья карта на экране', hidden: true },
    [[false, 'Наша карта по лидару'], [true, 'Карта SLAM Toolbox']].map(([v, label]) => h('button', {
      class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(v === slamOn), data: { slam: v ? '1' : '' },
      onclick: () => {
        slamOn = v;
        map.setOpts({ slam: v });
        slamSeg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(!!b.dataset.slam === slamOn)));
        map.setState(st);
        paint();
      },
    }, label)));
  const stage = h('div', { class: 'pl-stage' }, map.el, h('div', { class: 'pl-under' }, legend, toggles, slamSeg));

  fill(root,
    h('header', { class: 'pl-head' },
      h('div', null,
        h('div', { class: 'lb-eyebrow', text: 'Пульт' }),
        h('h1', { class: 'lb-h1 pl-h1', text: 'Поставьте точку на карте — робот поедет' })),
      sourceChip),
    h('div', { class: 'pl-grid' }, stage, colDrive, colMission));
  map.el.addEventListener('pl-resize', (ev) => {
    for (const col of [colDrive, colMission]) col.style.maxHeight = `${Math.max(ev.detail.height + 66, 420)}px`;
  });
  map.mount();

  // --- отрисовка состояния ---------------------------------------------------------------------

  function statusOf(s) {
    if (!s) return ['off', 'Сервер лаборатории не отвечает', 'Проверьте, что запущен pixi run demo или pixi run lab'];
    if (!s.active) return ['idle', 'Пульт не подключён к роботу', s.error || 'Запустите показ командой pixi run demo или включите быстрый симулятор ниже'];
    if (s.mode === 'wait') return ['idle', 'Жду стенд Gazebo', 'Мир загружается: обычно это 10–20 секунд'];
    if (s.linked === false) return ['off', 'Нет связи со стендом', 'Данные от робота не приходят: стенд остановлен или завис'];
    const m = s.mission;
    const left = s.path_len ? `осталось ${num(s.path_len, 1)} м` : '';
    if (s.mode === 'settle') return ['wait', `Робот оседает на колёса: ещё ${num(s.settle_left, 0)} с`, 'Сразу после появления в мире ехать нельзя — одометрия собьётся. Карта уже строится'];
    if (s.mode === 'drive' && s.nav && s.nav.explore && !s.nav.explore.done) {
      return ['on', 'Строю карту: еду к границе увиденного', `подъезд ${s.nav.explore.visited}, границ осталось ${s.nav.frontiers}${left ? ` · ${left}` : ''}`];
    }
    if (s.mode === 'drive') {
      const done = s.route.filter((p) => p.done).length;
      return ['on', `Еду по маршруту: точка ${Math.min(done + 1, s.route.length)} из ${s.route.length}`, left];
    }
    if (s.mode === 'home') return ['on', 'Возвращаюсь на базу', m && m.state === 'to_base' ? `${left} · затем начнётся миссия` : left];
    if (s.mode === 'mission') return ['on', 'Автономная миссия: робот решает сам', m ? `Сейчас: ${AGENT_MODE[m.agent_mode] || m.agent_mode || '…'}` : ''];
    if (pending().length) return ['idle', 'Маршрут готов', 'Нажмите «Ехать»'];
    return ['idle', 'Жду команду', s.at_base ? 'Робот на базе' : 'Робот стоит'];
  }

  function paintMission(s) {
    const m = s.mission;
    const live = s.mode === 'mission';
    if (!m || m.state === 'to_base') {
      const sel = h('select', { class: 'lb-select pl-select', 'aria-label': 'Агент', onchange: () => { agentChoice = sel.value; } },
        (s.agents || []).map((a) => h('option', { value: a.id, selected: a.id === agentChoice }, a.label)));
      sel.value = agentChoice;
      const key = `idle:${m ? 'to_base' : ''}:${(s.agents || []).length}`;
      if (missionBody.dataset.key === key) return;
      missionBody.dataset.key = key;
      fill(missionBody,
        h('p', { class: 'pl-text', text: m ? 'Робот возвращается на базу — миссия начнётся оттуда.' : 'Робот сам ищет образцы по датчику, собирает их и возвращается на базу. Оператор только смотрит.' }),
        h('div', { class: 'pl-row' }, sel,
          h('button', { class: 'lb-btn lb-btn--primary pl-btn', type: 'button', disabled: !!m, onclick: () => send('mission', { agent: agentChoice }) }, icon('play', 18), 'Запустить миссию')));
      return;
    }
    const r = m.result || {};
    // Кнопки строятся один раз на состояние миссии: иначе перерисовка съедала бы клик.
    const key = `${live ? 'live' : m.state}:${m.agent}:${m.trace_file || ''}`;
    if (missionBody.dataset.key !== key) {
      missionBody.dataset.key = key;
      const actions = live
        ? h('div', { class: 'pl-row' }, h('button', { class: 'lb-btn pl-btn', type: 'button', onclick: () => send('finish') }, 'Завершить миссию'))
        : h('div', { class: 'pl-row' },
          h('button', { class: 'lb-btn lb-btn--primary pl-btn', type: 'button', onclick: () => send('mission', { agent: m.agent }) }, icon('redo', 17), 'Запустить ещё раз'),
          m.trace_file ? h('a', { class: 'lb-link', href: runHref(m.trace_file), text: 'Открыть запись по секундам' }) : null);
      fill(missionBody, h('div', { class: 'pl-mission__now' }), h('div', { class: 'pl-stats' }), h('div', { class: 'pl-plan' }),
        h('ul', { class: 'pl-journal' }), actions);
    }
    const [head, statsBox, planBox, journalBox] = missionBody.children;
    const collected = live ? m.collected : (r.samples_collected ?? m.collected);
    const stat = (label, value) => h('div', { class: 'pl-stat' }, h('div', { class: 'pl-stat__value', text: value }), h('div', { class: 'pl-stat__label', text: label }));
    fill(statsBox, live
      ? [stat('собрано', `${collected} из ${m.total}`), stat('заряд', num(s.battery, 1)), stat('время, с', num(s.t, 0))]
      : [stat('собрано', `${collected} из ${m.total}`), stat('на базе', r.returned ? 'да' : 'нет'), stat('счёт', num(r.score, 1)),
        stat('время, с', num(r.t, 0)), stat('заряд остался', num(r.battery, 1))]);
    planBox.hidden = !(m.plan && live);
    if (m.plan && live) fill(planBox, h('span', { class: 'pl-plan__k', text: 'План: ' }), m.plan.reasoning);
    fill(journalBox, (m.journal || []).slice(-(live ? 5 : 4)).reverse().map((e) => h('li', { class: 'pl-j', data: { kind: e.kind } },
      h('span', { class: 'pl-j__t', text: `${num(e.t, 0)} с` }),
      h('span', { class: 'pl-j__k', text: KIND[e.kind] || e.kind }),
      h('span', { class: 'pl-j__x', text: e.text }))));
    fill(head, live
      ? [h('span', { class: 'lb-chip' }, m.label), h('span', { class: 'pl-mission__mode', text: AGENT_MODE[m.agent_mode] || m.agent_mode || '' })]
      : [h('span', { class: 'lb-badge', data: { tone: m.state === 'finished' && r.returned ? 'ok' : 'warn' } },
        m.state === 'aborted' ? 'Остановлена оператором' : `Итог: ${REASON[r.reason] || 'миссия окончена'}`), h('span', { class: 'lb-chip' }, m.label)]);
  }

  function paintSource(s) {
    const active = s && s.active;
    const fast = active && s.backend === 'fastsim';
    const key = `${active ? s.backend : 'none'}:${fast ? s.speed : ''}:${s && s.gazebo_alive ? 1 : 0}:${active ? `${s.level}-${s.seed}` : ''}`;
    if (sourceBody.dataset.key === key) return;
    sourceBody.dataset.key = key;
    if (active && !formTouched) fastForm = { level: s.level, seed: s.seed };
    const level = h('select', { class: 'lb-select pl-select', 'aria-label': 'Уровень', onchange: () => { fastForm.level = level.value; formTouched = true; } },
      LEVEL_ORDER.map((id) => h('option', { value: id, selected: id === fastForm.level }, levelName(id))));
    level.value = fastForm.level;
    const seed = h('input', { class: 'lb-input lb-input--num', type: 'number', min: 0, max: 999, value: fastForm.seed, 'aria-label': 'Номер сценария',
      oninput: () => { fastForm.seed = Number(seed.value) || 0; formTouched = true; } });
    const run = (label, primary) => h('button', { class: `lb-btn pl-btn pl-btn--small ${primary ? 'lb-btn--primary' : ''}`, type: 'button',
      onclick: () => start({ backend: 'fastsim', level: fastForm.level, seed: fastForm.seed, speed: (st && st.speed) || 1 }) }, label);
    const form = (label, primary) => h('div', { class: 'pl-row pl-row--small' },
      h('label', { class: 'lb-field' }, 'Уровень', level), h('label', { class: 'lb-field' }, 'Сценарий', seed), run(label, primary));
    if (!active) {
      fill(sourceBody,
        h('p', { class: 'pl-text' }, 'Показ с Gazebo запускается одной командой в терминале:'),
        h('pre', { class: 'lb-code', text: 'pixi run demo' }),
        h('p', { class: 'pl-text', text: 'Запасной вариант без Gazebo — быстрый симулятор: та же арена, тот же судья, тот же пульт.' }),
        form('Запустить быстрый симулятор', true));
    } else if (fast) {
      const speeds = h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Скорость времени' },
        (s.speeds || [1]).map((v) => h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(v === s.speed), onclick: () => send('speed', { value: v }, true) }, `×${v}`)));
      fill(sourceBody,
        h('p', { class: 'pl-text', text: 'Быстрый симулятор: та же арена, тот же судья и тот же пульт, что в Gazebo. Колёса тоже врут — позу поправляет лидар.' }),
        h('div', { class: 'pl-row pl-row--small' }, h('span', { class: 'lb-muted', text: 'Скорость времени' }), speeds),
        form('Перезапустить', false),
        h('div', { class: 'pl-row pl-row--small' },
          s.gazebo_alive ? h('button', { class: 'lb-btn pl-btn pl-btn--small', type: 'button', onclick: () => start({ backend: 'gazebo' }) }, 'Вернуться к Gazebo') : null,
          h('button', { class: 'lb-btn pl-btn pl-btn--small', type: 'button', onclick: () => start({ backend: 'off' }) }, 'Выключить симулятор')));
    } else {
      fill(sourceBody,
        h('p', { class: 'pl-text', text: 'Gazebo: настоящая физика, лидар и одометрия TurtleBot3. Пульт слушает /scan и /odom и шлёт /cmd_vel.' }),
        h('p', { class: 'pl-text lb-muted', text: 'Если Gazebo подведёт — тот же пульт на быстром симуляторе:' }),
        form('Перейти на быстрый симулятор', false));
    }
  }

  function paint() {
    const s = st;
    const [mode, text, sub] = statusOf(s);
    lamp.dataset.mode = mode === 'on' ? 'on' : mode === 'off' ? 'off' : 'idle';
    statusText.textContent = text;
    statusSub.textContent = sub || '';
    const active = !!(s && s.active);
    const note = flash && flash.until > Date.now() ? flash : (active ? s.note : null);
    noteBox.hidden = !note;
    if (note) {
      noteBox.dataset.tone = note.tone;
      noteBox.textContent = note.text;
    }
    const lag = active && s.lag && s.lag.late > 0 ? s.lag : null;
    lagBox.hidden = !lag;
    if (lag) lagBox.textContent = `Компьютер перегружен: команды роботу запаздывают до ${num(lag.max, 1)} с${lag.slow ? ' — еду медленнее' : ''}`;
    fill(sourceChip, active
      ? [h('span', { class: 'lb-dot', style: { background: s.backend === 'gazebo' ? COLOR.green : COLOR.path } }),
        s.backend === 'gazebo' ? 'Gazebo' : `Быстрый симулятор ×${s.speed || 1}`,
        h('span', { class: 'pl-source__sub', text: `${levelName(s.level).toLowerCase()} уровень, сценарий ${s.seed}${s.slam_nav ? ' · карта SLAM, готовой нет' : ''}` })]
      : 'робот не подключён');
    paintSource(s);
    const manual = active && !['mission', 'settle', 'wait'].includes(s.mode);
    const todo = active ? pending() : [];
    goBtn.disabled = !(manual && todo.length && s.mode !== 'drive');
    stopBtn.disabled = !(active && ['drive', 'home', 'mission'].includes(s.mode));
    homeBtn.disabled = !(manual && s.mode !== 'home' && !s.at_base);
    resetBtn.disabled = !(active && s.mode !== 'wait');
    exploreBtn.hidden = !(active && s.slam_nav);
    exploreBtn.disabled = !(manual && s.nav && s.nav.ready && !['drive', 'home'].includes(s.mode));
    undoBtn.disabled = clearBtn.disabled = !(manual && todo.length && s.mode !== 'home');
    if (!active || s.mode === 'wait') {
      routeInfo.textContent = '';
      fill(routeDots, h('span', { class: 'lb-muted', text: 'Точек нет' }));
      for (const t of Object.values(tiles)) { t.value.textContent = '—'; t.sub.textContent = ''; fill(t.bar); }
      fill(missionBody, h('p', { class: 'pl-text lb-muted', text: 'Миссию можно запустить, когда пульт подключён к роботу.' }));
      missionBody.dataset.key = '';
      fill(eventsBox);
      return;
    }

    const route = s.mode === 'home' ? [] : s.route;
    routeInfo.textContent = s.path_len && s.mode !== 'mission' ? `путь ${num(s.path_len, 1)} м` : '';
    fill(routeDots, route.length
      ? route.map((p, i) => h('span', { class: `pl-dot${p.done ? ' pl-dot--done' : ''}`, title: `x ${num(p.x, 2)}, y ${num(p.y, 2)}` }, String(i + 1)))
      : h('span', { class: 'lb-muted', text: s.mode === 'mission' ? 'Маршрут выбирает агент' : s.mode === 'home' ? 'Цель — база' : 'Точек пока нет' }));

    const b = s.battery / s.battery_start;
    tiles.battery.value.textContent = num(s.battery, 1);
    tiles.battery.sub.textContent = `из ${num(s.battery_start, 0)}, хватит на ${num(s.battery / 2.5, 0)} м`;
    fill(tiles.battery.bar, meter(b, b < 0.25 ? 'bad' : b < 0.5 ? 'warn' : 'ok'));
    tiles.sensor.value.textContent = num(s.sensor, 2);
    tiles.sensor.sub.textContent = s.sensor > 0.85 ? 'образец совсем рядом' : s.sensor > 0.02 ? 'ближе к образцу — больше' : 'образцов рядом нет';
    fill(tiles.sensor.bar, meter(s.sensor, 'green'));
    tiles.dist.value.textContent = `${num(s.distance, 1)} м`;
    tiles.dist.sub.textContent = `прогон идёт ${num(s.t, 0)} с`;
    slamSeg.hidden = !s.slam;
    if (s.slam_nav && s.slam && !slamNavSeen) {
      // Робот едет по карте SLAM Toolbox: её и показываем, а готовую карту убираем с экрана.
      slamNavSeen = true;
      slamOn = true;
      refCheck.querySelector('input').checked = false;
      map.setOpts({ slam: true, ref: false });
      slamSeg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(!!b.dataset.slam === slamOn)));
      map.setState(s);
    }
    const bySlam = slamOn && s.slam;
    const mp = (bySlam ? s.slam : s.map) || {};
    tiles.cover.value.textContent = `${num((mp.coverage || 0) * 100, 0)} %`;
    tiles.cover.sub.textContent = bySlam ? 'по карте SLAM Toolbox' : 'пола увидел лидар';
    fill(tiles.cover.bar, meter(mp.coverage || 0, 'blue'));
    tiles.agree.value.textContent = mp.coverage ? `${num((mp.agreement || 0) * 100, 1)} %` : '—';
    tiles.agree.sub.textContent = bySlam ? 'клеток SLAM совпало' : 'клеток совпало';
    const f = s.fix || {};
    tiles.pose.value.textContent = f.source === 'slam' ? 'колёса + SLAM' : f.source === 'lidar' ? 'колёса + лидар' : 'только колёса';
    tiles.pose.value.classList.add('pl-tile__value--text');
    tiles.pose.sub.textContent = f.source === 'slam'
      ? `поправка от SLAM Toolbox ${cm(f.shift || 0)} см, ${num(Math.abs(f.dth || 0) * 180 / Math.PI, 1)}°`
      : f.source === 'lidar'
        ? `поправка по лидару ${cm(f.shift || 0)} см, ${num(Math.abs(f.dth || 0) * 180 / Math.PI, 1)}°`
        : 'поправки по лидару нет';

    paintMission(s);
    fill(eventsBox, (s.events || []).slice(-5).reverse().map((e) => h('li', { class: 'pl-ev', data: { kind: e.kind } },
      h('span', { class: 'pl-ev__t', text: `${num(e.t, 0)} с` }), h('span', { text: e.text }))));
  }

  // --- опрос сервера ---------------------------------------------------------------------------

  let polling = false;
  async function poll(now) {
    if (!ctx.alive()) return;
    if (polling) return;
    polling = true;
    if (now) clearTimeout(timer);
    let next = null;
    try { next = await api.get('/api/pilot/state', 0); } catch { next = null; }
    polling = false;
    if (!ctx.alive()) return;
    st = next;
    map.setState(st);
    paint();
    clearTimeout(timer);
    timer = setTimeout(poll, st && st.active ? POLL_MS : 1000);
  }

  ctx.onLeave(() => {
    clearTimeout(timer);
    map.destroy();
  });
  paint();
  poll();
}
