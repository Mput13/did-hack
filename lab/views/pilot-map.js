// Пульт: оператор ставит точки на карте — робот едет; карта по лидару проявляется на глазах.
// Состояние приходит с сервера четыре раза в секунду (GET /api/pilot/state), команды уходят
// POST /api/pilot/command. Экран один и тот же для Gazebo и для быстрого симулятора.

import { h, num, soilColor, multLabel } from './common.js';

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

function decode(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// =================================================================================================
// карта
// =================================================================================================

export function createMap(arena, onClick, fps = 30) {
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
        const k = ((GH - 1 - r) * GW + c) * 4;
        if (!free[r * GW + c]) {
          const near = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => {
            const nc = c + dx, nr = r + dy;
            return nc >= 0 && nr >= 0 && nc < GW && nr < GH && free[nr * GW + nc];
          });
          if (near) { img.data[k] = 20; img.data[k + 1] = 24; img.data[k + 2] = 28; img.data[k + 3] = 255; }
          continue;
        }
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
    if (!grid || !el.offsetParent) return;
    // Ширина карты — что осталось от колонок пульта (см. pilot.css), высота — чтобы всё помещалось в экран.
    const full = grid.clientWidth;
    const avail = window.innerWidth >= 960 ? full - 340 - 24 : full;
    const top = el.getBoundingClientRect().top + window.scrollY;
    const maxH = Math.max(360, window.innerHeight - Math.min(top, 260) - 92);
    const width = Math.max(240, Math.min(avail, maxH * aspect, 760));
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
    // Изменение размера очищает canvas; рисуем сразу, в том числе при снимке всей страницы.
    draw(performance.now());
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
    if (!W || !el.offsetParent) return;
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

    // Ограничения оператора, отдельно от скрытых свойств грунта.
    for (const z of st.zones || []) {
      zonePath({ ...z, shape: 'circle' });
      const danger = z.kind === 'danger';
      ctx.fillStyle = danger ? 'rgba(214,69,69,0.20)' : 'rgba(230,178,40,0.28)';
      ctx.fill();
      ctx.strokeStyle = danger ? '#cc4545' : '#bd901a';
      ctx.lineWidth = 2;
      ctx.stroke();
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
