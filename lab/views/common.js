// Общие вещи для всех страниц лаборатории: запросы к серверу, сборка DOM, подписи, цвета, статусы.

// --- DOM ---------------------------------------------------------------------------------------

const SVG_NS = 'http://www.w3.org/2000/svg';

function setAttrs(el, attrs, svg) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') svg ? el.setAttribute('class', v) : (el.className = v);
    else if (k === 'text') el.textContent = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [p, val] of Object.entries(v)) {
        if (val == null) continue;
        p.startsWith('--') ? el.style.setProperty(p, val) : (el.style[p] = val);
      }
    } else if (k === 'data') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
}

function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false || k === '') continue;
    el.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
}

/** Элемент HTML: h('div', {class: 'x', onclick: fn}, дети…). Строки вставляются как текст. */
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs, false);
  append(el, kids);
  return el;
}

/** Элемент SVG. */
export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs, true);
  append(el, kids);
  return el;
}

/** Заменить содержимое элемента; пустые значения пропускаются. */
export function fill(el, ...kids) {
  el.replaceChildren();
  append(el, kids);
  return el;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- сервер ------------------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, opts, retries) {
  let last;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(path, opts);
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { /* не JSON */ }
      if (res.ok) return data;
      last = new ApiError(res.status, (data && data.error) || `сервер ответил ${res.status}`);
      if (res.status !== 404 && res.status < 500) break;      // 400, 409 — повторять бессмысленно
    } catch {
      last = new ApiError(0, 'сервер лаборатории не отвечает');
    }
    if (i < retries) await sleep(350 * (i + 1));
  }
  throw last;
}

export const api = {
  // Данные иногда пересчитываются, и файл на долю секунды пропадает — поэтому чтение повторяем.
  get: (path, retries = 2) => request(path, { cache: 'no-store' }, retries),
  post: (path, body) => request(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}),
  }, 0),
};

function once(fn) {
  let p = null;
  return () => {
    if (!p) p = fn().catch((e) => { p = null; throw e; });
    return p;
  };
}

export const getArena = once(() => api.get('/api/arena'));
export const getOptions = once(() => api.get('/api/options'));
export const getIndex = () => api.get('/api/index');
export const getExperiment = (id) => api.get(`/api/experiment/${encodeURIComponent(id)}`);
export const getTrace = (file) => api.get(`/api/trace?file=${encodeURIComponent(file)}`);
export const getScenario = (level, seed) =>
  api.get(`/api/scenario?level=${encodeURIComponent(level)}&seed=${encodeURIComponent(seed)}`);
/** Один прогон в быстром симуляторе. rules: 'science' — правила с несколькими причинами расхода и сбоями. */
export const runEpisode = (level, seed, agent, rules) => api.post('/api/run', rules ? { level, seed, agent, rules } : { level, seed, agent });
/** По записи прогона: шёл ли он по усложнённым правилам. */
export const rulesOf = (trace) => (trace && trace.rules && trace.rules.faults ? 'science' : null);

// --- фоновые серии -----------------------------------------------------------------------------

export const jobs = (() => {
  let list = [];
  let timer = null;
  const seen = new Map();
  const subs = new Set();

  async function refresh() {
    clearTimeout(timer);
    try {
      list = (await api.get('/api/jobs', 0)) || [];
    } catch {
      timer = setTimeout(refresh, 2000);
      return list;
    }
    const finished = [];
    for (const j of list) {
      if (seen.get(j.id) === 'running' && j.state !== 'running') finished.push(j);
      seen.set(j.id, j.state);
    }
    for (const fn of [...subs]) {
      try { fn(list, finished); } catch (e) { console.error(e); }
    }
    if (list.some((j) => j.state === 'running')) timer = setTimeout(refresh, 600);
    return list;
  }

  return {
    refresh,
    all: () => list,
    /** Последняя серия по опыту (сервер отдаёт от новых к старым). */
    latest: (expId) => list.find((j) => j.experiment === expId),
    running: (expId) => list.find((j) => j.experiment === expId && j.state === 'running'),
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
    async start(expId, seeds) {
      const body = seeds ? { seeds } : {};
      const job = await api.post(`/api/experiment/${encodeURIComponent(expId)}/run`, body);
      seen.set(job.id, 'running');
      await refresh();
      return job;
    },
  };
})();

// --- адрес страницы ----------------------------------------------------------------------------

export function parseHash(hash = location.hash) {
  const raw = hash.replace(/^#/, '') || '/';
  const [path, qs = ''] = raw.split('?');
  const query = {};
  for (const [k, v] of new URLSearchParams(qs)) query[k] = v;
  return { path: path || '/', query };
}

export function href(path, query) {
  const qs = query ? new URLSearchParams(Object.entries(query).filter(([, v]) => v != null && v !== '')).toString() : '';
  return `#${path}${qs ? `?${qs}` : ''}`;
}

export const runHref = (file) => href('/run', { file });
export const compareHref = (a, b) => href('/compare', { a, b });
export function go(hashUrl) { location.hash = hashUrl; }

// --- числа -------------------------------------------------------------------------------------

/** Число с запятой и настоящим минусом. */
export function num(v, digits = 1) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  let t = Number(v).toFixed(digits);
  if (Number(t) === 0) t = t.replace('-', '');
  return t.replace('.', ',').replace('-', '−');
}

export function signed(v, digits = 1) {
  const t = num(v, digits);
  return Number(v) > 0 && !/^0(,0+)?$/.test(t) ? `+${t}` : t;
}

/**
 * Тексты, собранные в Python (журнал, выводы, правила): десятичная точка → запятая, дефис перед числом → минус,
 * «53%» → «53 %», «после 1 опыт(ов)» → «после 1 опыта».
 */
export function ruText(text) {
  if (text == null) return '';
  return String(text)
    .replace(/(\d)\.(\d)/g, '$1,$2')
    .replace(/(^|[\s(;:=≈×«])-(\d)/g, '$1−$2')
    .replace(/(\d+)\s*опыт\(ов\)/g, (_, n) => `${n} ${plural(Number(n), 'опыта', 'опытов', 'опытов')}`)
    .replace(/(\d)%/g, '$1 %');
}

const LEVEL_WORDS = { easy: ['лёгкий', 'лёгком'], medium: ['средний', 'среднем'], hard: ['трудный', 'трудном'] };
/** Названия уровней в текстах опытов пишутся по-английски (easy, medium, hard) — показываем по-русски. */
export function ruLevels(text) {
  if (text == null) return '';
  const id = '(easy|medium|hard)';
  const cap = (w, like) => (/^[А-ЯЁA-Z]/.test(like) ? w[0].toUpperCase() + w.slice(1) : w);
  return String(text)
    .replace(new RegExp(`([Сс]ценари\\S*\\s+)?([Уу]ровн[яею]|[Уу]ровень)\\s+${id}\\b`, 'g'), (m0, pre, word, lv) => {
      const [nom, loc] = LEVEL_WORDS[lv];
      if (/ень$/.test(word)) return `${pre || ''}${cap(nom, word)} уровень`;
      if (/не$/.test(word)) return `${pre || ''}${cap(loc, word)} уровне`;
      return `${pre || ''}${cap(nom.replace(/ий$/, 'его').replace(/ый$/, 'ого'), word)} уровня`;
    })
    .replace(new RegExp(`(?<![А-Яа-яЁё\\w])([Вв]|[Нн]а)\\s+${id}\\b`, 'g'), (m0, prep, lv) => `${/^[ВН]/.test(prep) ? 'На' : 'на'} ${LEVEL_WORDS[lv][1]} уровне`)
    .replace(new RegExp(`\\b${id}\\b`, 'g'), (m0, lv) => `«${LEVEL_WORDS[lv][0]}»`);
}

/** Склонение: plural(5, 'прогон', 'прогона', 'прогонов'). */
export function plural(n, one, few, many) {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b === 1) return one;
  if (b >= 2 && b <= 4) return few;
  return many;
}

export const count = (n, one, few, many) => `${n} ${plural(n, one, few, many)}`;

export function when(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const months = ['янв.', 'февр.', 'марта', 'апр.', 'мая', 'июня', 'июля', 'авг.', 'сент.', 'окт.', 'нояб.', 'дек.'];
  const pad = (x) => String(x).padStart(2, '0');
  return `${d.getDate()} ${months[d.getMonth()]}, ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// --- метрики -----------------------------------------------------------------------------------

// Как показывать метрику: pct — доля в процентах, count — штуки, num — число с единицей.
const METRIC_VIEW = {
  score: { kind: 'num', digits: 1, unit: 'очки', unitOf: 'очка' },
  samples_share: { kind: 'pct' },
  returned: { kind: 'pct', bool: true },
  battery_used: { kind: 'num', digits: 1, unit: 'ед.', unitOf: 'ед. заряда' },
  battery_left: { kind: 'num', digits: 1, unit: 'ед.', unitOf: 'ед. заряда' },
  energy_per_sample: { kind: 'num', digits: 1, unit: 'ед.', unitOf: 'ед. заряда' },
  distance: { kind: 'num', digits: 1, unit: 'м' },
  time: { kind: 'num', digits: 0, unit: 'с' },
  penalties: { kind: 'count' },
  false_collects: { kind: 'count' },
  hazard_hits: { kind: 'count' },
  collisions: { kind: 'count' },
};

const METRIC_LABELS = {
  score: 'Счёт', samples_share: 'Собрано образцов', returned: 'Вернулся на базу', battery_used: 'Потрачено заряда',
  energy_per_sample: 'Заряд на один образец', distance: 'Путь', time: 'Время прогона', penalties: 'Штрафы',
  false_collects: 'Ложные сборы', hazard_hits: 'Заезды в опасную зону', collisions: 'Столкновения',
};

/** Как показывать метрику, которой нет в списке выше: по единице из ответа сервера и по самим значениям. */
function guessView(meta, hint) {
  const unit = String(meta.unit || '').trim();
  if (hint && hint.bool) return { kind: 'pct', bool: true };
  if (/^доля/i.test(unit) || unit === '%') return { kind: 'pct' };
  if (/^(шт|раз)\.?$/i.test(unit)) return { kind: 'count' };
  const max = hint && Number.isFinite(hint.max) ? Math.abs(hint.max) : null;
  const digits = max == null ? 2 : max >= 100 ? 0 : max >= 10 ? 1 : 2;
  return { kind: 'num', digits, unit };
}

/**
 * Всё, что нужно для показа метрики: подпись, единица, что лучше и форматтеры.
 * metas — словарь metrics из ответа /api/experiment/<id>; hint — {max, bool} по значениям этой метрики в опыте.
 */
export function metricInfo(id, metas, hint) {
  const meta = (metas && metas[id]) || {};
  const view = METRIC_VIEW[id] || guessView(meta, hint);
  const label = meta.label || METRIC_LABELS[id] || String(id).replace(/_/g, ' ');
  const better = meta.better || (['score', 'samples_share', 'returned'].includes(id) ? 'higher' : 'lower');
  const info = { id, label, better, kind: view.kind, bool: !!view.bool };
  if (view.kind === 'pct') {
    info.unit = '%';
    info.diffUnit = 'п. п.';
    info.value = (v) => (v == null ? '—' : `${num(v * 100, 0)} %`);          // один прогон
    info.mean = (v) => (v == null ? '—' : `${num(v * 100, 0)} %`);           // среднее
    info.tick = (v) => num(v * 100, 0);
    info.diff = (v) => (v == null ? '—' : `${signed(v * 100, 0)} п. п.`);
    info.diffTick = (v) => signed(v * 100, 0);
    info.diffBare = (v) => signed(v * 100, 0);
  } else if (view.kind === 'count') {
    info.unit = 'шт.';
    info.diffUnit = 'шт.';
    info.value = (v) => (v == null ? '—' : num(v, 0));
    info.mean = (v) => (v == null ? '—' : num(v, 2));
    info.tick = (v) => num(v, Number.isInteger(v) ? 0 : 1);
    info.diff = (v) => (v == null ? '—' : signed(v, 2));
    info.diffTick = (v) => signed(v, Number.isInteger(v) ? 0 : 2);
    info.diffBare = info.diff;
  } else {
    const d = view.digits ?? 1;
    const tickDigits = (v) => (Number.isInteger(v) ? 0 : Math.abs(v) < 1 && d >= 2 ? 2 : 1);
    info.unit = view.unit || meta.unit || '';
    info.unitOf = view.unitOf || info.unit;        // «на 7,8 очка больше»
    info.diffUnit = info.unit;
    info.value = (v) => (v == null ? '—' : num(v, d));
    info.mean = (v) => (v == null ? '—' : num(v, d));
    info.tick = (v) => num(v, tickDigits(v));
    info.diff = (v) => (v == null ? '—' : signed(v, d || 1));
    info.diffTick = (v) => signed(v, tickDigits(v));
    info.diffBare = info.diff;
  }
  info.title = info.unit ? `${label}, ${info.unit}` : label;
  info.betterText = better === 'higher' ? 'больше — лучше' : 'меньше — лучше';
  return info;
}

/** Значение метрики прогона как число (да/нет → 1/0), либо null. */
export function metricValue(run, id) {
  const v = run.metrics ? run.metrics[id] : undefined;
  if (v == null || typeof v === 'object') return null;      // вложенные сводки (detect, inquiries) — не число
  if (typeof v === 'boolean') return v ? 1 : 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// --- словари -----------------------------------------------------------------------------------

export const LEVELS = {
  easy: { name: 'Лёгкий', lower: 'лёгкий' },
  medium: { name: 'Средний', lower: 'средний' },
  hard: { name: 'Трудный', lower: 'трудный' },
  all: { name: 'Все уровни вместе', lower: 'все уровни' },
};
export const levelName = (id) => (LEVELS[id] ? LEVELS[id].name : id);
export const LEVEL_ORDER = ['easy', 'medium', 'hard'];

export const KINDS = {
  main: 'Главный опыт',
  ablation: 'Отключаем части агента',
  adaptation: 'Адаптивность',
  sensitivity: 'Устойчивость к условиям',
  llm: 'Языковая модель',
  transfer: 'Проверка в Gazebo',
  robustness: 'Устойчивость к сбоям',
  search: 'Стратегии поиска',
  science: 'Расследования агента',
  inquiry: 'Расследования агента',
  knowledge: 'Память между прогонами',
  memory: 'Память между прогонами',
};

export const BACKENDS = {
  fastsim: 'быстрый симулятор',
  gazebo: 'Gazebo',
};
export const backendName = (id) => BACKENDS[id] || id || '';

export const REASONS = {
  finish: 'вернулся на базу и закончил',
  battery: 'села батарея',
  timeout: 'вышло время',
};

export const EVENT_NAMES = {
  soil_change: 'Смена грунтов',
  new_hazard: 'Новая опасная зона',
  sensor_fault: 'Сбой датчика',
};

// Подпись условия опыта. Обычно это spec.conditions[].label; запасной путь — собрать подпись из
// изменённого правила, если подпись в описании опыта не строка или рядом с ней лежат лишние поля
// (так выходит, когда в YAML подпись с запятой записана без кавычек: «label: 1,0 ед/м»).
const COND_KEYS = new Set(['id', 'label', 'rules', 'scenario', 'sim']);
const RULE_LABELS = {
  drain_per_m: (v) => `${num(v, 2).replace(/0$/, '')} ед/м`,
  sensor_sigma: (v) => `шум ${num(v, 2)}`,
};
export function condName(cond) {
  if (!cond) return '';
  const clean = typeof cond.label === 'string' && Object.keys(cond).every((k) => COND_KEYS.has(k));
  if (clean) return cond.label;
  const rules = Object.entries(cond.rules || {});
  if (rules.length) return rules.map(([k, v]) => (RULE_LABELS[k] ? RULE_LABELS[k](v) : `${k} = ${num(v, 2)}`)).join(', ');
  return cond.label != null ? String(cond.label) : String(cond.id);
}

// Запасные подписи: основные приходят с сервера (/api/options) и из описаний опытов.
export const AGENT_LABELS = {
  fixed: 'Фиксированный план',
  adaptive: 'С адаптацией',
  adaptive_ig: 'С адаптацией, разведка по ожидаемой пользе',
  adaptive_llm: 'С адаптацией, планирует языковая модель',
  scientist: 'Исследователь: ведёт расследования',
  scientist_llm: 'Исследователь с языковой моделью',
  scientist_fs: 'Исследователь, сравнивает будущие маршруты',
  adaptive_fs: 'С адаптацией, сравнивает будущие маршруты',
  spiral: 'Спираль',
  gradient: 'Подъём по сигналу',
  no_soil: 'Без обучения грунтам',
  no_change: 'Без обнаружения изменений',
  no_hazard: 'Без памяти об опасных зонах',
  no_sensor_health: 'Без контроля датчика',
  static_reserve: 'Возврат по жёсткому порогу',
  belief_only: 'Только карта образцов',
};

// --- цвета вариантов агента --------------------------------------------------------------------
// Порядок оттенков проверен на различимость (в том числе при цветовой слепоте); дальше восьми — серый.

export const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const OTHER = '#8a939b';

// Пары оттенков, которые нельзя ставить в соседние строки графика: их трудно различить
// (проверено скриптом validate_palette из набора правил для графиков; номера — места в PALETTE).
const CLASH = new Set(['1-3', '1-4', '1-5', '1-7', '4-7']);
const clash = (i, j) => i != null && j != null && CLASH.has(i < j ? `${i}-${j}` : `${j}-${i}`);

/**
 * Цвета вариантов одного опыта: фиксированный план — синий, адаптивный — оранжевый, остальные по порядку палитры,
 * но так, чтобы соседние в списке варианты не получили похожие оттенки. Цвет зависит только от списка вариантов
 * опыта, поэтому не меняется при фильтрах и сортировке.
 */
export function armColors(arms) {
  const list = arms || [];
  const slot = new Array(list.length).fill(null);
  const used = new Set();
  list.forEach((a, i) => {
    if (a.agent === 'fixed' && !used.has(0)) { slot[i] = 0; used.add(0); }
    else if (a.agent === 'adaptive' && !used.has(1)) { slot[i] = 1; used.add(1); }
  });
  list.forEach((a, i) => {
    if (slot[i] != null) return;
    const free = [];
    for (let k = 2; k < PALETTE.length; k++) if (!used.has(k)) free.push(k);
    if (!free.length) return;
    const pick = free.find((k) => !clash(k, slot[i - 1]) && !clash(k, slot[i + 1])) ?? free[0];
    slot[i] = pick;
    used.add(pick);
  });
  const map = new Map();
  list.forEach((a, i) => map.set(a.id, slot[i] == null ? OTHER : PALETTE[slot[i]]));
  return map;
}

export function agentColor(agentId) {
  if (agentId === 'fixed') return PALETTE[0];
  if (agentId === 'adaptive') return PALETTE[1];
  return PALETTE[2];
}

/** Цвет грунта по множителю расхода: один оттенок, чем дороже — тем темнее. */
export function soilColor(mult) {
  const stops = [[1, [252, 228, 216]], [2, [240, 157, 120]], [3, [235, 104, 52]], [4, [184, 72, 27]]];
  const m = Math.max(1, Math.min(4, Number(mult) || 1));
  for (let i = 1; i < stops.length; i++) {
    if (m <= stops[i][0]) {
      const [m0, c0] = stops[i - 1];
      const [m1, c1] = stops[i];
      const t = (m - m0) / (m1 - m0);
      const c = c0.map((x, j) => Math.round(x + (c1[j] - x) * t));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(184,72,27)';
}

export const multLabel = (mult) => `×${num(mult, Number.isInteger(Number(mult)) ? 0 : 1)}`;

// --- статусы -----------------------------------------------------------------------------------

export const STATUS = {
  supported: { word: 'подтверждается', tone: 'ok', icon: 'check' },
  refuted: { word: 'опровергается', tone: 'bad', icon: 'cross' },
  partial: { word: 'подтверждается частично', tone: 'warn', icon: 'half' },
  mixed: { word: 'результаты противоречат друг другу', tone: 'warn', icon: 'swap' },
  inconclusive: { word: 'данных мало', tone: 'none', icon: 'question' },
  no_data: { word: 'нет данных', tone: 'none', icon: 'dash' },
  not_run: { word: 'ещё не запускали', tone: 'none', icon: 'ring' },
  descriptive: { word: 'описательный опыт', tone: 'info', icon: 'info' },
  running: { word: 'считается', tone: 'info', icon: 'spin' },
  failed: { word: 'серия упала', tone: 'bad', icon: 'cross' },
};

/** Статус опыта как фраза о гипотезе. */
export function hypothesisWord(status) {
  return {
    supported: 'Гипотеза подтверждается',
    refuted: 'Гипотеза опровергается',
    partial: 'Гипотеза подтверждается частично',
    mixed: 'Результаты противоречат друг другу',
    inconclusive: 'Данных мало для вывода',
    not_run: 'Серию ещё не запускали',
    descriptive: 'Описательный опыт: без проверки утверждений',
  }[status] || null;
}

const ICONS = {
  check: () => [s('path', { d: 'M3.2 8.6l3.1 3.1 6.5-7.2' })],
  cross: () => [s('path', { d: 'M4 4l8 8M12 4l-8 8' })],
  half: () => [s('circle', { cx: 8, cy: 8, r: 5.5 }), s('path', { d: 'M8 2.5a5.5 5.5 0 0 0 0 11z', fill: 'currentColor', stroke: 'none' })],
  swap: () => [s('path', { d: 'M3 5.5h9.5L10 3M13 10.5H3.5L6 13' })],
  question: () => [s('path', { d: 'M5.6 6a2.4 2.4 0 1 1 3.6 2.1c-.8.5-1.2.9-1.2 1.9' }), s('circle', { cx: 8, cy: 12.6, r: 0.6, fill: 'currentColor' })],
  dash: () => [s('path', { d: 'M4 8h8' })],
  ring: () => [s('circle', { cx: 8, cy: 8, r: 5 })],
  info: () => [s('path', { d: 'M8 7.2v5' }), s('circle', { cx: 8, cy: 4.3, r: 0.6, fill: 'currentColor' })],
  spin: () => [s('path', { d: 'M8 2.5a5.5 5.5 0 1 1-5.5 5.5', class: 'lb-spin' })],
  play: () => [s('path', { d: 'M5 3.5v9l7.5-4.5z', fill: 'currentColor', stroke: 'none' })],
  pair: () => [s('rect', { x: 2, y: 3.5, width: 5, height: 9, rx: 1 }), s('rect', { x: 9, y: 3.5, width: 5, height: 9, rx: 1 })],
  left: () => [s('path', { d: 'M10 3.5L5.5 8l4.5 4.5' })],
  right: () => [s('path', { d: 'M6 3.5L10.5 8 6 12.5' })],
  dice: () => [s('rect', { x: 2.5, y: 2.5, width: 11, height: 11, rx: 2.5 }), s('circle', { cx: 5.7, cy: 5.7, r: 0.7, fill: 'currentColor' }), s('circle', { cx: 10.3, cy: 10.3, r: 0.7, fill: 'currentColor' }), s('circle', { cx: 8, cy: 8, r: 0.7, fill: 'currentColor' })],
  redo: () => [s('path', { d: 'M13 8a5 5 0 1 1-1.6-3.7M13 2.6v2.8h-2.8' })],
  warn: () => [s('path', { d: 'M8 2.5l6 10.5H2z' }), s('path', { d: 'M8 6.5v3' }), s('circle', { cx: 8, cy: 11.2, r: 0.5, fill: 'currentColor' })],
};

export function icon(name, size = 16) {
  const make = ICONS[name] || ICONS.dash;
  return s('svg', {
    class: 'lb-icon', viewBox: '0 0 16 16', width: size, height: size, fill: 'none', stroke: 'currentColor',
    'stroke-width': 1.9, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true',
  }, make());
}

/** Статус словом и значком (цвет — только дополнение). */
export function badge(status, opts = {}) {
  const st = STATUS[status] || { word: status || '—', tone: 'none', icon: 'dash' };
  const word = opts.word || (opts.prefix ? `${opts.prefix} ${st.word}` : st.word);
  return h('span', { class: `lb-badge${opts.big ? ' lb-badge--big' : ''}`, data: { tone: st.tone } },
    icon(st.icon, opts.big ? 18 : 15), h('span', { text: opts.cap === false ? word : word[0].toUpperCase() + word.slice(1) }));
}

/** Только значок статуса с подписью для чтения с экрана. */
export function mark(status) {
  const st = STATUS[status] || { word: status || '', tone: 'none', icon: 'dash' };
  return h('span', { class: 'lb-mark', data: { tone: st.tone }, title: st.word, role: 'img', 'aria-label': st.word }, icon(st.icon, 14));
}

export const dot = (color) => h('span', { class: 'lb-dot', style: { background: color }, 'aria-hidden': 'true' });

// --- подсказка под курсором --------------------------------------------------------------------

let tipEl = null;
export const tip = {
  show(x, y, content) {
    if (!tipEl) {
      tipEl = h('div', { class: 'lb-tip', role: 'tooltip' });
      document.body.append(tipEl);
    }
    tipEl.replaceChildren(...[].concat(content));
    tipEl.hidden = false;
    const r = tipEl.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    let left = x + 14;
    let top = y + 16;
    if (left + r.width > vw - 8) left = x - r.width - 14;
    if (top + r.height > vh - 8) top = y - r.height - 12;
    tipEl.style.left = `${Math.max(8, left)}px`;
    tipEl.style.top = `${Math.max(8, top)}px`;
  },
  hide() { if (tipEl) tipEl.hidden = true; },
};

// --- мелкие блоки ------------------------------------------------------------------------------

export function loading(text = 'Загружаю…') {
  return h('div', { class: 'lb-loading', role: 'status' }, icon('spin', 18), h('span', { text }));
}

export function errorBox(title, detail, retry) {
  return h('div', { class: 'lb-alert', data: { tone: 'bad' }, role: 'alert' },
    h('div', { class: 'lb-alert__icon' }, icon('warn', 20)),
    h('div', { class: 'lb-alert__body' },
      h('div', { class: 'lb-alert__title', text: title }),
      detail ? h('div', { class: 'lb-alert__text', text: detail }) : null,
      retry ? h('div', { class: 'lb-alert__actions' }, h('button', { class: 'lb-btn', type: 'button', onclick: retry }, icon('redo'), 'Повторить')) : null));
}

export function emptyBox(title, text, ...actions) {
  return h('div', { class: 'lb-empty' },
    h('div', { class: 'lb-empty__title', text: title }),
    text ? h('div', { class: 'lb-empty__text', text }) : null,
    actions.length ? h('div', { class: 'lb-empty__actions' }, actions) : null);
}

export function select(options, value, onchange, attrs = {}) {
  const el = h('select', { class: 'lb-select', ...attrs, onchange: () => onchange(el.value) },
    options.map((o) => h('option', { value: o.value, selected: String(o.value) === String(value) }, o.label)));
  el.value = String(value);
  return el;
}

/** Подпись варианта агента: из описания опыта, а для отдельных прогонов — из списка агентов. */
export function agentLabel(agentId, options) {
  const a = options && options.agents ? options.agents.find((x) => x.id === agentId) : null;
  // Сервер, не знающий нового варианта, отдаёт вместо подписи его идентификатор — тогда берём свою.
  if (a && a.label && a.label !== a.id) return a.label;
  return AGENT_LABELS[agentId] || (a && a.label) || agentId || 'агент';
}

/** Порядок сценариев: уровень, затем номер. */
export function byScenario(a, b) {
  const la = LEVEL_ORDER.indexOf(a.level);
  const lb = LEVEL_ORDER.indexOf(b.level);
  return la - lb || a.seed - b.seed;
}

// --- проигрыватель (чужой модуль, может отсутствовать) -----------------------------------------

let replayMod = null;
let replayTries = 0;
let replayWhy = '';

function addCss(url) {
  return new Promise((resolve) => {
    if (document.querySelector(`link[data-lb-css="${url}"]`)) return resolve();
    const link = h('link', { rel: 'stylesheet', href: url, 'data-lb-css': url });
    link.onload = () => resolve();
    link.onerror = () => resolve();
    document.head.append(link);
    setTimeout(resolve, 1500);
  });
}

/** Модуль проигрывателя или null, если его ещё нет или он не загрузился. Причина — в replayProblem(). */
export async function loadReplay() {
  if (replayMod) return replayMod;
  try {
    const probe = await fetch('replay.js', { cache: 'no-store' });
    if (!probe.ok) {
      replayWhy = 'файл lab/replay.js ещё не готов';
      return null;
    }
    const mod = await import(`../replay.js${replayTries ? `?t=${replayTries}` : ''}`);
    if (typeof mod.mountReplay !== 'function') throw new Error('в replay.js нет функции mountReplay');
    await addCss('replay.css');
    replayMod = mod;
    replayWhy = '';
    return mod;
  } catch (e) {
    replayTries += 1;
    replayWhy = `модуль не загрузился: ${e && e.message ? e.message : e}`;
    console.warn('Проигрыватель недоступен:', e);
    return null;
  }
}

export const replayProblem = () => replayWhy;
