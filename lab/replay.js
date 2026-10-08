// Проигрыватель прогона «Лаборатории DID».
//
//   mountReplay(container, {trace, arena, compact = false, follow = false, ...}) -> controller
//   mountCompare(container, {traces: [a, b], labels: [..], arena, ...})          -> controller
//
// controller: { seek(t), play(), pause(), setSpeed(k), onTime(cb), update(trace), destroy(),
//               setView('truth'|'agent'|'both'), setLayers({...}), getTime(), getDuration() }
//
// ES-модуль без сборки и без внешних библиотек. Вся вёрстка — внутри переданного контейнера,
// классы с префиксом rp-. Стили — в replay.css (подключается сам, если оболочка его не подключила).
//
// Необязательные части записи (нет поля — нет блока):
//   inquiries  расследования агента: панель «Расследования», дорожка на шкале времени, метки на арене;
//   pose_fix   поправка положения по лидару: третий график по времени;
//   energy_model, result.inquiries — сводки под панелью расследований и в итоге прогона.

const NS = 'http://www.w3.org/2000/svg';
const TAU = Math.PI * 2;

/* ============================================================================================
 * Мелкие помощники
 * ========================================================================================== */

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, f) => a + (b - a) * f;
const byT = (a, b) => a.t - b.t;

/** Сколько элементов возрастающего массива не больше v. */
function upperBound(arr, v) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function button(cls, text) {
  const b = h('button', cls, text);
  b.type = 'button';
  return b;
}

function s(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

/** Значок из постоянной строки разметки (только свои константы, чужие данные сюда не попадают). */
function icon(markup, cls) {
  const span = h('span', cls || 'rp-ico');
  span.setAttribute('aria-hidden', 'true');
  span.innerHTML = markup;
  return span;
}

/** Число по-русски: запятая и настоящий минус. */
function num(v, d = 1) {
  if (v == null || v === '' || !Number.isFinite(+v)) return '—';
  let out = (+v).toFixed(d);
  if (/^-0(\.0+)?$/.test(out)) out = out.slice(1);
  return out.replace('-', '−').replace('.', ',');
}
const sec = (t) => `${num(t, 1)} с`;
const mult = (m) => `×${num(m, Math.abs(m - Math.round(m)) < 0.05 ? 0 : 1)}`;
const xy = (x, y) => `(${num(x, 1)}; ${num(y, 1)})`;

function plural(n, one, few, many) {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b === 1) return one;
  if (b >= 2 && b <= 4) return few;
  return many;
}

/**
 * Тексты, собранные в Python (журнал, планы, расследования): десятичная точка -> запятая, дефис перед числом -> минус,
 * «53%» -> «53 %», «после 1 опыт(ов)» -> «после 1 опыта».
 */
function ru(text) {
  if (text == null) return '';
  return String(text)
    .replace(/(\d)\.(\d)/g, '$1,$2')
    .replace(/(^|[\s(;:=≈×«])-(\d)/g, '$1−$2')
    .replace(/(\d+)\s*опыт\(ов\)/g, (_, n) => `${n} ${plural(+n, 'опыта', 'опытов', 'опытов')}`)
    .replace(/(\d)%/g, '$1 %');
}
/** Конечное число или null. */
const fin = (v) => (v == null || v === '' || !Number.isFinite(+v) ? null : +v);
const cap = (text) => (text ? text[0].toUpperCase() + text.slice(1) : '');
/** Сколько знаков после запятой нужно, чтобы на отрезке такой длины числа различались. */
const digitsFor = (span) => (span >= 20 ? 0 : span >= 2 ? 1 : span >= 0.2 ? 2 : 3);

let colorProbe = null;
/** Любая запись цвета CSS -> [r, g, b]. */
function rgbOf(color, fallback) {
  try {
    if (!colorProbe) colorProbe = document.createElement('canvas').getContext('2d');
    colorProbe.fillStyle = '#000000';
    colorProbe.fillStyle = color;
    const v = String(colorProbe.fillStyle);
    if (v[0] === '#') return [parseInt(v.slice(1, 3), 16), parseInt(v.slice(3, 5), 16), parseInt(v.slice(5, 7), 16)];
    const m = v.match(/[\d.]+/g);
    if (m && m.length >= 3) return [+m[0], +m[1], +m[2]];
  } catch (e) { /* ниже запасной цвет */ }
  return fallback || [0, 0, 0];
}
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const mix = (a, b, f) => [Math.round(lerp(a[0], b[0], f)), Math.round(lerp(a[1], b[1], f)), Math.round(lerp(a[2], b[2], f))];

let uidCounter = 0;
const uid = (p) => `${p}${++uidCounter}`;

/* ============================================================================================
 * Словари интерфейса
 * ========================================================================================== */

const MODE_RU = {
  start: 'старт', explore: 'разведка', travel: 'в пути', approach: 'подход к образцу', collect: 'сбор',
  return: 'возврат', think: 'думает', escape: 'отъезд назад', done: 'финиш',
  experiment: 'ставит опыт', probe: 'ставит опыт', wait: 'ждёт',
};
const KIND_RU = {
  observe: 'наблюдение', hypothesis: 'гипотеза', verdict: 'вердикт', decision: 'решение',
  action: 'действие', alarm: 'тревога', llm: 'модель', inquiry: 'расследование',
};
const HYP_RU = { open: 'открыта', confirmed: 'подтверждена', refuted: 'опровергнута', outdated: 'устарела' };
const SOURCE_RU = {
  heuristic: 'собственный расчёт', llm: 'языковая модель', fallback: 'запасной вариант',
  fixed: 'маршрут, заданный заранее', rule: 'жёсткое правило', foresight: 'сравнение вариантов',
};
const TRIGGER_RU = {
  start: 'старт', candidate_found: 'появилось место для проверки', candidate_lost: 'место не подтвердилось',
  sample_collected: 'образец собран', sensor_degraded: 'датчик шумит', hazard: 'штраф в опасной зоне',
  return: 'пора возвращаться', model_mismatch: 'расход не сошёлся с прогнозом', subgoal_done: 'подцель выполнена',
  false_collect: 'сбор не удался', queue_empty: 'подцели закончились', inquiry_closed: 'расследование закончено',
  inquiry_opened: 'началось расследование', fault: 'сбой', collision: 'столкновение',
};
const SUBGOAL_RU = {
  explore: 'разведать участок', investigate: 'проверить место', goto: 'ехать в точку',
  return_base: 'вернуться на базу', collect: 'собрать образец',
};
const WORLD_RU = {
  soil_change: 'Грунт изменился', new_hazard: 'Появилась новая опасная зона',
  sensor_fault: 'Датчик начал шуметь', sensor_recovered: 'Датчик снова в норме',
  fault: 'Начался сбой', fault_end: 'Сбой закончился',
};
const WORLD_NOTICE = {
  soil_change: 'Грунт изменился — роботу об этом не сообщили',
  new_hazard: 'Появилась новая опасная зона — роботу об этом не сообщили',
  sensor_recovered: 'Датчик снова в норме',
  fault_end: 'Сбой закончился',
};
// Виды сбоев (правила с несколькими причинами): что именно сломалось.
const FAULT_RU = {
  leak: 'батарея сама теряет заряд', sensor_noise: 'датчик образцов шумит',
  sensor_stuck: 'датчик образцов залип', sensor_bias: 'датчик образцов занижает показания',
};
/** Подпись скрытого события среды с учётом вида сбоя. */
function worldText(w) {
  const kind = w.kind && FAULT_RU[w.kind];
  if (w.type === 'fault') return kind ? `Сбой после штрафа: ${kind}` : WORLD_RU.fault;
  if (w.type === 'sensor_fault' && kind && w.kind !== 'sensor_noise') return `Сбой: ${kind}`;
  if (w.type === 'fault_end') return w.kind === 'leak' ? 'Утечка заряда прекратилась' : WORLD_RU.fault_end;
  return WORLD_RU[w.type] || 'Скрытое изменение среды';
}
// Расследования: темы, исходы, настоящие причины (по ним судья сверяет вывод агента).
const TOPIC_RU = { energy: 'Расход заряда', sensor: 'Датчик образцов', fault: 'Последствия штрафа' };
const CAUSE_RU = {
  leak: 'утечка заряда', soil: 'дорогой грунт', noise: 'шум датчика', stuck: 'залипание датчика',
  bias: 'занижение показаний', none: 'сбоя не было', ok: 'датчик исправен', turn: 'расход на повороты',
  load: 'вес образцов', other: 'причина не из списка',
};
const INQ_SOURCE_RU = { rule: 'объяснения и опыты — по правилам агента', llm: 'объяснения и опыты предложила языковая модель' };
const LETTERS = 'АБВГДЕЖЗИК';
const BACKEND_RU = { fastsim: 'быстрый симулятор', gazebo: 'Gazebo' };
const BACKEND_TITLE = {
  fastsim: 'Прогон посчитан в быстром симуляторе',
  gazebo: 'Прогон шёл в симуляторе Gazebo: робот с физикой, лидаром и навигацией',
};
const LEVEL_RU = { easy: 'лёгкий уровень', medium: 'средний уровень', hard: 'трудный уровень' };
const AGENT_RU = {
  fixed: 'фиксированный план', adaptive: 'адаптивный агент', adaptive_ig: 'адаптивный агент, разведка по ожидаемой пользе',
  adaptive_llm: 'адаптивный агент с языковой моделью', scientist: 'агент-исследователь',
  scientist_llm: 'агент-исследователь с языковой моделью', spiral: 'спираль', gradient: 'подъём по сигналу',
};
// Какая тревога агента отвечает на какое скрытое изменение среды (как в did/metrics.py).
const DETECT_TAG = { soil_change: 'model_mismatch', new_hazard: 'hazard', sensor_fault: 'sensor_degraded' };
const FAULT_TAG = { leak: 'leak', sensor_noise: 'sensor_degraded', sensor_stuck: 'sensor_degraded', sensor_bias: 'sensor_degraded' };
const PENALTY_TYPES = { collision: 1, false_collect: 1, hazard_hit: 1 };

const DEFAULT_LAYERS = {
  truthSamples: true, truthSoil: true, truthHazards: true,
  beliefSamples: true, beliefSoil: true, beliefHazards: true,
  trail: true, path: true, lidar: true, ring: true,
};
const TRUTH_KEYS = ['truthSamples', 'truthSoil', 'truthHazards'];
const BELIEF_KEYS = ['beliefSamples', 'beliefSoil', 'beliefHazards'];
const VIEWS = [
  { id: 'truth', label: 'Истина', title: 'Только то, что есть на самом деле' },
  { id: 'agent', label: 'Глазами агента', title: 'Только то, что знает и предполагает агент' },
  { id: 'both', label: 'Оба', title: 'Истина — контуры и значки, догадки агента — заливка' },
];

function layersForView(view, base) {
  const out = Object.assign({}, base);
  for (const k of TRUTH_KEYS) out[k] = view !== 'agent';
  for (const k of BELIEF_KEYS) out[k] = view !== 'truth';
  return out;
}
function viewOfLayers(l) {
  const tr = TRUTH_KEYS.filter((k) => l[k]).length;
  const be = BELIEF_KEYS.filter((k) => l[k]).length;
  if (tr === 3 && be === 3) return 'both';
  if (tr === 3 && be === 0) return 'truth';
  if (tr === 0 && be === 3) return 'agent';
  return null;
}

const ICO = {
  play: '<svg viewBox="0 0 16 16"><path d="M4.2 2.3v11.4L13.5 8z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M3.6 2.5h3.2v11H3.6zM9.2 2.5h3.2v11H9.2z" fill="currentColor"/></svg>',
  warn: '<svg viewBox="0 0 16 16"><path d="M8 1.6 15 14H1z" fill="currentColor"/><path d="M8 6v4" stroke="#fff" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="12" r="1" fill="#fff"/></svg>',
  down: '<svg viewBox="0 0 16 16"><path d="M8 3v9M4 8.5 8 12.5 12 8.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  // метки шкалы времени
  mWorld: '<svg viewBox="0 0 14 14"><path class="rp-m-world" d="M7 12.6 1.2 2.4h11.6z"/></svg>',
  mSample: '<svg viewBox="0 0 14 14"><path class="rp-m-sample" d="M7 1.2 12.8 7 7 12.8 1.2 7z"/></svg>',
  mPenalty: '<svg viewBox="0 0 14 14"><circle class="rp-m-penalty" cx="7" cy="7" r="5.8"/><path d="M4.7 4.7l4.6 4.6M9.3 4.7 4.7 9.3" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/></svg>',
  mAlarm: '<svg viewBox="0 0 14 14"><path class="rp-m-alarm" d="M7 1.4 12.8 11.6H1.2z"/><path d="M7 5v3.2" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/><circle cx="7" cy="10" r=".85" fill="#fff"/></svg>',
  mPlan: '<svg viewBox="0 0 14 14"><rect class="rp-m-plan" x="6" y="2.5" width="2" height="9" rx="1"/></svg>',
  mInquiry: '<svg viewBox="0 0 14 14"><circle class="rp-m-inquiry" cx="6" cy="6" r="4.3"/><path class="rp-m-inquiry-h" d="M9.2 9.2 12.6 12.6"/></svg>',
  chevron: '<svg viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  check: '<svg viewBox="0 0 16 16"><path d="M3.2 8.6l3.1 3.1 6.5-7.2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross: '<svg viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
  question: '<svg viewBox="0 0 16 16"><path d="M5.6 6a2.4 2.4 0 1 1 3.6 2.1c-.8.5-1.2.9-1.2 1.9" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><circle cx="8" cy="12.6" r="1" fill="currentColor"/></svg>',
  arrow: '<svg viewBox="0 0 16 16"><path d="M3 8h9.5M9 4.5 12.5 8 9 11.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  // образцы легенды
  sSample: '<svg viewBox="0 0 20 14"><path d="M10 1.5 15.5 7 10 12.5 4.5 7z" fill="#fff" stroke="var(--rp-green-ink)" stroke-width="2"/><path d="M10 4.8 12.2 7 10 9.2 7.8 7z" fill="var(--rp-green)"/></svg>',
  sSoil: '<svg viewBox="0 0 20 14"><rect x="2" y="2" width="16" height="10" rx="1.5" fill="none" stroke="var(--rp-soil-3)" stroke-width="2"/><path d="M4 12 12 2M9 12 17 2" stroke="var(--rp-soil-3)" stroke-width="1.1"/></svg>',
  sHazard: '<svg viewBox="0 0 20 14"><circle cx="10" cy="7" r="5.6" fill="none" stroke="var(--rp-red)" stroke-width="2" stroke-dasharray="3.2 2.2"/><path d="M10 4v3.4" stroke="var(--rp-red)" stroke-width="1.6" stroke-linecap="round"/><circle cx="10" cy="9.8" r=".9" fill="var(--rp-red)"/></svg>',
  sHeat: '<svg viewBox="0 0 20 14"><rect x="1" y="1" width="18" height="12" rx="3" fill="var(--rp-green)" opacity=".2"/><ellipse cx="10" cy="7" rx="6.4" ry="4.4" fill="var(--rp-green)" opacity=".45"/><ellipse cx="10" cy="7" rx="3.2" ry="2.4" fill="var(--rp-green-ink)" opacity=".95"/></svg>',
  sSoilCell: '<svg viewBox="0 0 20 14"><rect x="2" y="2" width="5" height="5" fill="var(--rp-soil-2)" opacity=".6"/><rect x="7.5" y="2" width="5" height="5" fill="var(--rp-soil-3)" opacity=".7"/><rect x="13" y="2" width="5" height="5" fill="var(--rp-soil-4)" opacity=".75"/><rect x="7.5" y="7.5" width="5" height="5" fill="var(--rp-soil-2)" opacity=".6"/><rect x="13" y="7.5" width="5" height="5" fill="var(--rp-soil-3)" opacity=".7"/></svg>',
  sHazMem: '<svg viewBox="0 0 20 14"><circle cx="10" cy="7" r="6" fill="var(--rp-red)" opacity=".3"/></svg>',
  sTrail: '<svg viewBox="0 0 20 14"><path d="M2 10C6 2 10 12 18 4" fill="none" stroke="var(--rp-ink-3)" stroke-width="2" stroke-linecap="round"/></svg>',
  sPath: '<svg viewBox="0 0 20 14"><path d="M2 10 12 5" fill="none" stroke="var(--rp-ink)" stroke-width="1.8" stroke-dasharray="3 2.4"/><circle cx="15" cy="4.5" r="3" fill="none" stroke="var(--rp-ink)" stroke-width="1.6"/></svg>',
  sLidar: '<svg viewBox="0 0 20 14"><path d="M10 7 2 3M10 7 4 12M10 7 17 2M10 7 18 10M10 7 11 1" stroke="var(--rp-ink-3)" stroke-width="1"/><circle cx="2" cy="3" r="1.1" fill="var(--rp-ink-2)"/><circle cx="17" cy="2" r="1.1" fill="var(--rp-ink-2)"/><circle cx="18" cy="10" r="1.1" fill="var(--rp-ink-2)"/><circle cx="4" cy="12" r="1.1" fill="var(--rp-ink-2)"/></svg>',
  sRing: '<svg viewBox="0 0 20 14"><circle cx="10" cy="7" r="5.6" fill="none" stroke="var(--rp-blue)" stroke-opacity=".22" stroke-width="3.6"/><circle cx="10" cy="7" r="5.6" fill="none" stroke="var(--rp-blue)" stroke-width="1.6"/><circle cx="10" cy="7" r="1.6" fill="var(--rp-ink)"/></svg>',
};

const LAYER_GROUPS = [
  { id: 'truth', title: 'Как на самом деле', items: [
    { key: 'truthSamples', label: 'образцы', sw: ICO.sSample, title: 'Где лежат образцы. Собранные отмечены галочкой' },
    { key: 'truthSoil', label: 'дорогой грунт', sw: ICO.sSoil, title: 'Участки, где заряд тратится в несколько раз быстрее' },
    { key: 'truthHazards', label: 'опасные зоны', sw: ICO.sHazard, title: 'Въезд — штраф и потеря заряда' },
  ] },
  { id: 'belief', title: 'Что думает агент', items: [
    { key: 'beliefSamples', label: 'где могут быть образцы', sw: ICO.sHeat, title: 'Чем темнее заливка, тем выше вероятность, что образец здесь' },
    { key: 'beliefSoil', label: 'оценка грунта', sw: ICO.sSoilCell, title: 'Клетки, где агент по расходу заряда решил, что грунт дорогой' },
    { key: 'beliefHazards', label: 'запомненные зоны', sw: ICO.sHazMem, title: 'Места, которые агент решил объезжать после штрафа' },
  ] },
  { id: 'robot', title: 'Робот', items: [
    { key: 'trail', label: 'след', sw: ICO.sTrail, title: 'Где робот уже проехал' },
    { key: 'lidar', label: 'лидар', sw: ICO.sLidar, title: 'Лучи дальномера: так робот видит стены и столбы' },
    { key: 'path', label: 'путь и цель', sw: ICO.sPath, title: 'Куда робот едет сейчас и подцели плана' },
    { key: 'ring', label: 'кольцо датчика', sw: ICO.sRing, title: 'Датчик сообщает только расстояние: образец где-то на этом кольце' },
  ] },
];

/* ============================================================================================
 * Стили и оформление
 * ========================================================================================== */

let cssReady = null;
function ensureCss() {
  if (cssReady) return cssReady;
  const done = (link) => new Promise((resolve) => {
    link.addEventListener('load', resolve, { once: true });
    link.addEventListener('error', resolve, { once: true });
    setTimeout(resolve, 3000);
  });
  const existing = document.querySelector('link[rel~="stylesheet"][href*="replay.css"]');
  if (existing) {
    cssReady = existing.sheet ? Promise.resolve() : done(existing);
    return cssReady;
  }
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('./replay.css', import.meta.url).href.replace(/\?.*$/, '');
  cssReady = done(link);
  document.head.appendChild(link);
  return cssReady;
}

const THEME_VARS = {
  bg: ['--rp-bg', '#f5f6f7'], surface: ['--rp-surface', '#ffffff'], ink: ['--rp-ink', '#14181c'],
  ink2: ['--rp-ink-2', '#4a535c'], ink3: ['--rp-ink-3', '#8a939b'], line: ['--rp-line', '#e3e6e9'],
  blue: ['--rp-blue', '#2a78d6'], green: ['--rp-green', '#1baf7a'], greenInk: ['--rp-green-ink', '#0d7a54'],
  red: ['--rp-red', '#d64545'], amber: ['--rp-amber', '#e0a100'],
  soil2: ['--rp-soil-2', '#f09d78'], soil3: ['--rp-soil-3', '#eb6834'], soil4: ['--rp-soil-4', '#b8481b'],
  wall: ['--rp-wall', '#5d6670'], pillar: ['--rp-pillar', '#c5cbd1'], outside: ['--rp-outside', '#eceef1'],
};
function readTheme(root) {
  const cs = getComputedStyle(root);
  const t = { font: cs.fontFamily || 'sans-serif' };
  for (const k in THEME_VARS) {
    const [name, fallback] = THEME_VARS[k];
    const v = cs.getPropertyValue(name).trim() || fallback;
    t[k] = rgbOf(v, rgbOf(fallback));
  }
  return t;
}
function soilRgb(theme, m) {
  if (m <= 2) return theme.soil2;
  if (m >= 4) return theme.soil4;
  return m <= 3 ? mix(theme.soil2, theme.soil3, m - 2) : mix(theme.soil3, theme.soil4, m - 3);
}
function soilCss(m) {
  if (m <= 2.25) return 'var(--rp-soil-2)';
  if (m <= 3.25) return 'var(--rp-soil-3)';
  return 'var(--rp-soil-4)';
}

/* ============================================================================================
 * Геометрия арены: из сетки свободных клеток — гладкий контур стен и круглые столбы
 * ========================================================================================== */

function simplifyOpen(pts, eps) {
  const n = pts.length;
  if (n < 3) return pts.slice();
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const ax = pts[a][0], ay = pts[a][1];
    const dx = pts[b][0] - ax, dy = pts[b][1] - ay;
    const len = Math.hypot(dx, dy) || 1;
    let worst = -1;
    let wi = -1;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i][0] - ax) * dy - (pts[i][1] - ay) * dx) / len;
      if (d > worst) { worst = d; wi = i; }
    }
    if (worst > eps) { keep[wi] = 1; stack.push([a, wi], [wi, b]); }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[i]);
  return out;
}

function simplifyClosed(pts, eps) {
  if (pts.length < 8) return pts.slice();
  const pass = (ring) => {
    let far = 0;
    let fd = -1;
    for (let i = 1; i < ring.length; i++) {
      const d = (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2;
      if (d > fd) { fd = d; far = i; }
    }
    const a = simplifyOpen(ring.slice(0, far + 1), eps);
    const b = simplifyOpen(ring.slice(far).concat([ring[0]]), eps);
    return a.slice(0, -1).concat(b.slice(0, -1));
  };
  // Первый проход начинается со случайной точки стены; второй — с настоящего угла, чтобы не было излома.
  const first = pass(pts);
  if (first.length < 4) return first;
  let best = -1;
  let bestTurn = -1;
  for (let i = 0; i < first.length; i++) {
    const p = first[(i + first.length - 1) % first.length], q = first[i], r = first[(i + 1) % first.length];
    const a1 = Math.atan2(q[1] - p[1], q[0] - p[0]);
    const a2 = Math.atan2(r[1] - q[1], r[0] - q[0]);
    const turn = Math.abs(Math.atan2(Math.sin(a2 - a1), Math.cos(a2 - a1)));
    const span = Math.min(Math.hypot(q[0] - p[0], q[1] - p[1]), Math.hypot(r[0] - q[0], r[1] - q[1]));
    if (span > 3 && turn > bestTurn) { bestTurn = turn; best = i; }
  }
  if (best < 0) return first;
  const start = pts.indexOf(first[best]);
  if (start <= 0) return first;
  return pass(pts.slice(start).concat(pts.slice(0, start)));
}

function buildGeom(arena) {
  const a = arena || {};
  const res = +a.res || 0.05;
  const x0 = a.x0 != null ? +a.x0 : -3;
  const y0 = a.y0 != null ? +a.y0 : -3;
  const rows = Array.isArray(a.free) ? a.free : [];
  const gh = rows.length;
  const gw = gh ? rows[0].length : 0;
  const F = new Uint8Array(gw * gh);
  let minX = gw, maxX = -1, minY = gh, maxY = -1;
  for (let iy = 0; iy < gh; iy++) {
    const row = rows[iy];
    for (let ix = 0; ix < gw; ix++) {
      if (row.charCodeAt(ix) === 49) {
        F[iy * gw + ix] = 1;
        if (ix < minX) minX = ix;
        if (ix > maxX) maxX = ix;
        if (iy < minY) minY = iy;
        if (iy > maxY) maxY = iy;
      }
    }
  }
  const geom = { res, x0, y0, floors: [], obstacles: [], base: a.base || [-2, -0.5] };
  if (maxX < 0) {
    geom.view = { vx0: x0, vy0: y0, vx1: x0 + (gw * res || 6), vy1: y0 + (gh * res || 6) };
    return geom;
  }
  const free = (ix, iy) => ix >= 0 && iy >= 0 && ix < gw && iy < gh && F[iy * gw + ix] === 1;

  // Рёбра между свободной и занятой клеткой, свободная клетка слева по ходу.
  const W1 = gw + 1;
  const out = new Map();
  const add = (ax, ay, bx, by) => {
    const k = ay * W1 + ax;
    let l = out.get(k);
    if (!l) out.set(k, (l = []));
    l.push(by * W1 + bx);
  };
  for (let iy = minY; iy <= maxY; iy++) {
    for (let ix = minX; ix <= maxX; ix++) {
      if (!F[iy * gw + ix]) continue;
      if (!free(ix, iy - 1)) add(ix, iy, ix + 1, iy);
      if (!free(ix + 1, iy)) add(ix + 1, iy, ix + 1, iy + 1);
      if (!free(ix, iy + 1)) add(ix + 1, iy + 1, ix, iy + 1);
      if (!free(ix - 1, iy)) add(ix, iy + 1, ix, iy);
    }
  }
  const toWorld = (p) => [x0 + p[0] * res, y0 + p[1] * res];
  for (const [start, list] of out) {
    while (list.length) {
      const loop = [start];
      let cur = list.pop();
      let guard = 0;
      while (cur !== start && guard++ < 400000) {
        loop.push(cur);
        const l = out.get(cur);
        if (!l || !l.length) break;
        cur = l.pop();
      }
      const pts = loop.map((k) => [k % W1, (k / W1) | 0]);
      let area = 0, cx = 0, cy = 0;
      let bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i], q = pts[(i + 1) % pts.length];
        const cr = p[0] * q[1] - q[0] * p[1];
        area += cr;
        cx += (p[0] + q[0]) * cr;
        cy += (p[1] + q[1]) * cr;
        if (p[0] < bx0) bx0 = p[0];
        if (p[0] > bx1) bx1 = p[0];
        if (p[1] < by0) by0 = p[1];
        if (p[1] > by1) by1 = p[1];
      }
      area /= 2;
      if (Math.abs(area) < 0.5) continue;
      cx /= 6 * area;
      cy /= 6 * area;
      if (area > 0) {
        if (area < 12) continue;                         // свободный «пиксель» в стороне — шум карты
        geom.floors.push(simplifyClosed(pts, 1.05).map(toWorld));
      } else {
        const A = -area;
        const bw = bx1 - bx0, bh = by1 - by0;
        const req = Math.sqrt(A / Math.PI);
        const round = Math.abs(bw - bh) <= 2 && A / (Math.PI * (Math.max(bw, bh) / 2) ** 2) > 0.62 && Math.max(bw, bh) <= 16;
        geom.obstacles.push({
          pts: simplifyClosed(pts, 0.6).map(toWorld),
          circle: round ? { x: x0 + cx * res, y: y0 + cy * res, r: req * res } : null,
        });
      }
    }
  }
  const m = 0.14;
  geom.view = {
    vx0: x0 + minX * res - m, vy0: y0 + minY * res - m,
    vx1: x0 + (maxX + 1) * res + m, vy1: y0 + (maxY + 1) * res + m,
  };
  return geom;
}

/* ============================================================================================
 * Модель прогона: всё, что вычисляется из записи один раз
 * ========================================================================================== */

function decodeGrid(data, size) {
  const outArr = new Uint8Array(size);
  try {
    const bin = atob(data);
    const m = Math.min(size, bin.length);
    for (let i = 0; i < m; i++) outArr[i] = bin.charCodeAt(i);
  } catch (e) { /* битый снимок — пустая карта */ }
  return outArr;
}

/** Снимки карты: декодируются по требованию и один раз. */
function gridStore(g, prev) {
  if (!g || !Array.isArray(g.snaps) || !g.snaps.length || !g.w || !g.h) return null;
  const times = g.snaps.map((x) => +x.t);
  const cache = new Array(g.snaps.length);
  if (prev && prev.w === g.w && prev.h === g.h) {
    for (let k = 0; k < Math.min(times.length, prev.times.length); k++) {
      if (prev.times[k] === times[k] && prev.cache[k] && prev.len[k] === g.snaps[k].data.length) cache[k] = prev.cache[k];
    }
  }
  const len = g.snaps.map((x) => (x.data ? x.data.length : 0));
  return {
    res: +g.res, x0: +g.x0, y0: +g.y0, w: g.w, h: g.h, scale: +g.scale || 40, times, cache, len,
    get(k) { return cache[k] || (cache[k] = decodeGrid(g.snaps[k].data, g.w * g.h)); },
  };
}

function zoneHas(z, x, y) {
  if (z.shape === 'rect') return Math.abs(x - z.x) <= z.w / 2 && Math.abs(y - z.y) <= z.h / 2;
  const dx = x - z.x, dy = y - z.y;
  return dx * dx + dy * dy <= z.r * z.r;
}

function parseSubgoal(text) {
  const m = /^\s*([a-z_]+)\s*([^\s(]+)?\s*(?:\(\s*(-?[\d.]+)\s*;\s*(-?[\d.]+)\s*\))?/i.exec(String(text));
  if (!m) return { type: String(text) };
  const o = { type: m[1] };
  if (m[2]) o.target = m[2];
  if (m[3] != null && m[4] != null) { o.x = +m[3]; o.y = +m[4]; }
  return o;
}

/** Расследования из записи в едином виде: чего нет — null или пустой список, тексты уже с запятыми. */
function readInquiries(list) {
  if (!Array.isArray(list)) return [];
  const out = [];
  for (const q of list) {
    if (!q || typeof q !== 'object') continue;
    const t0 = fin(q.t_open);
    if (t0 == null) continue;
    const an = q.anomaly && typeof q.anomaly === 'object' ? q.anomaly : {};
    const alts = (Array.isArray(q.alternatives) ? q.alternatives : []).filter((a) => a && a.id != null).map((a, i) => ({
      id: String(a.id), letter: LETTERS[i] || String(i + 1), statement: ru(a.statement || CAUSE_RU[a.id] || a.id),
      prior: fin(a.prior), posterior: fin(a.posterior),
    }));
    const t1 = fin(q.t_close);
    const tests = (Array.isArray(q.tests) ? q.tests : []).filter((x) => x && typeof x === 'object').map((x, i) => {
      const ms = x.measured && typeof x.measured === 'object' && fin(x.measured.value) != null
        ? { value: +x.measured.value, sigma: Math.abs(fin(x.measured.sigma) || 0), t: fin(x.measured.t) != null ? +x.measured.t : (t1 != null ? t1 : t0) }
        : null;
      const P = x.predictions && typeof x.predictions === 'object' ? x.predictions : {};
      const preds = [];
      for (const a of alts) {                          // строки шкалы идут в том же порядке, что и объяснения
        const pr = P[a.id];
        const mean = pr == null ? null : fin(Array.isArray(pr) ? pr[0] : pr.mean);
        if (mean == null) continue;
        preds.push({ alt: a, mean, sigma: Math.abs(fin(Array.isArray(pr) ? pr[1] : pr.sigma) || 0) });
      }
      const cost = fin(x.cost);
      const gain = fin(x.gain_bits);
      return {
        id: x.id != null ? String(x.id) : `t${i}`, name: ru(x.name || x.id || 'опыт'), cost, gain, duration: fin(x.duration_s),
        unit: x.unit ? ru(x.unit) : '', preds, measured: ms, order: i,
        // польза на единицу заряда; у почти бесплатного опыта на ноль не делим
        eff: gain == null ? null : gain / Math.max(cost == null ? 0 : cost, 0.05),
        free: cost != null && cost < 0.005,
      };
    });
    // сначала проведённые опыты по времени измерения, затем остальные — по убыванию пользы на единицу заряда
    tests.sort((a, b) => (a.measured ? 0 : 1) - (b.measured ? 0 : 1)
      || (a.measured && b.measured ? a.measured.t - b.measured.t : (b.eff || 0) - (a.eff || 0)) || a.order - b.order);
    const measuredT = tests.filter((x) => x.measured).map((x) => x.measured.t);
    const c = q.conclusion && typeof q.conclusion === 'object' ? q.conclusion : null;
    const best = c && c.best != null ? alts.find((a) => a.id === String(c.best)) || null : null;
    out.push({
      id: q.id != null ? String(q.id) : `Q${out.length + 1}`, topic: q.topic ? String(q.topic) : '', t0, t1,
      anomaly: {
        text: ru(an.text || 'измерения не сошлись с ожиданием'), x: fin(an.x), y: fin(an.y),
        observed: fin(an.observed), expected: fin(an.expected), unit: an.unit ? ru(an.unit) : '',
      },
      alts, tests, measuredT,
      tLast: measuredT.length ? Math.max.apply(null, measuredT) : (t1 != null ? t1 : t0),
      conclusion: c ? { status: String(c.status || ''), best, confidence: fin(c.confidence), text: ru(c.text || '') } : null,
      action: ru(q.action || ''), source: q.source ? String(q.source) : '', note: ru(q.note || ''),
      critique: (Array.isArray(q.critique) ? q.critique : []).map((x) => (typeof x === 'string' ? { issue: ru(x), resolved: null }
        : x && typeof x === 'object' ? { issue: ru(x.issue || x.text || ''), resolved: x.resolved == null ? null : !!x.resolved } : null)).filter((x) => x && x.issue),
      // сверка со скрытой правдой сценария (её ставит судья после прогона)
      truth: Array.isArray(q.truth) ? q.truth.map(String) : null, verdict: q.verdict ? String(q.verdict) : '',
    });
  }
  return out.sort((a, b) => a.t0 - b.t0);
}

/** Поправка положения по лидару: сдвиг в сантиметрах и поворот в градусах по времени. null — показывать нечего. */
function readPoseFix(list) {
  if (!Array.isArray(list) || list.length < 2) return null;
  const T = [];
  const S = [];
  const A = [];
  let max = 0;
  let maxA = 0;
  for (const f of list) {
    const t = f && typeof f === 'object' ? fin(f.t) : null;
    if (t == null) continue;
    const shift = Math.hypot(fin(f.dx) || 0, fin(f.dy) || 0) * 100;
    const ang = ((fin(f.dth) || 0) * 180) / Math.PI;
    T.push(t);
    S.push(shift);
    A.push(ang);
    if (shift > max) max = shift;
    if (Math.abs(ang) > maxA) maxA = Math.abs(ang);
  }
  // В быстром симуляторе одометрия точная и поправка всё время нулевая — такой график ничего не говорит.
  if (T.length < 2 || (max < 0.1 && maxA < 0.05)) return null;
  return { T, S, A, max, maxA };
}

function buildModel(trace, prev) {
  const tr0 = trace || {};
  const tr = tr0.track || {};
  const T = tr.t || [];
  const n = Math.min(T.length, (tr.x || []).length, (tr.y || []).length);
  const rules = Object.assign({
    battery_start: 60, sensor_range_m: 2, sensor_sigma: 0.05, collect_radius_m: 0.3, base_radius_m: 0.3,
  }, tr0.rules || {});
  const sc = tr0.scenario || {};
  const result = tr0.result && typeof tr0.result === 'object' ? tr0.result : null;
  const modes = tr0.modes || [];
  const events = (tr0.events || []).slice().sort(byT);
  const journal = tr0.journal || [];
  const plans = tr0.plans || [];
  const paths = tr0.paths || [];
  const scans = tr0.scans || [];
  const hyps = tr0.hypotheses || [];
  const memHazards = (tr0.hazards || []).slice().sort(byT);
  const llm = tr0.llm || [];
  const worldLog = Array.isArray(tr0.world) ? tr0.world.slice().sort(byT) : null;
  const base = sc.base || null;

  let dur = n ? +T[n - 1] : 0;
  if (result) dur = Math.max(dur, +result.time || 0, +result.t || 0);
  if (journal.length) dur = Math.max(dur, +journal[journal.length - 1].t || 0);
  if (events.length) dur = Math.max(dur, +events[events.length - 1].t || 0);
  // Начало записи. У прогонов из Gazebo часы судьи могут уйти вперёд до первого кадра: тогда шкала
  // начинается не с нуля, а с первого записанного момента (времена в журнале остаются как в записи).
  let t0 = n ? +T[0] || 0 : journal.length ? +journal[0].t || 0 : 0;
  if (journal.length) t0 = Math.min(t0, +journal[0].t || 0);
  if (plans.length) t0 = Math.min(t0, +plans[0].t || 0);
  if (!(t0 > 0.5)) t0 = 0;
  if (dur < t0) dur = t0;
  // У идущего прогона result — текущий счёт судьи (finished: false), а не итог.
  const finished = !!(result && (result.finished || result.reason));
  const tFinish = Math.min(dur, (finished && (+result.time || +result.t)) || dur);

  // --- истина во времени ---
  const seen = {};
  const applied = (ev) => {
    const k = seen[ev.type] = (seen[ev.type] || 0) + 1;
    if (!worldLog) return +ev.t;
    const hits = worldLog.filter((w) => w.type === ev.type);
    if (hits[k - 1]) return +hits[k - 1].t;
    // журнал среды есть, а события в нём нет — прогон до события не дошёл;
    // журнал пуст целиком — он не записан, тогда верим времени из сценария
    return !worldLog.length && +ev.t < dur - 1 ? +ev.t : Infinity;
  };
  const soilStates = [{ t: -Infinity, soils: sc.soils || [] }];
  const trueHazards = (sc.hazards || []).map((z) => ({ t: -Infinity, zone: z }));
  const faults = [];           // сбои датчика образцов: {t0, t1, kind}
  const leaks = [];            // утечки заряда: {t0, t1}
  const planned = {};          // длительность сбоя датчика по расписанию сценария, по видам
  for (const ev of sc.events || []) if (ev.type === 'sensor_fault') planned[ev.kind || 'sensor_noise'] = +ev.duration || 0;
  for (const ev of (sc.events || []).slice().sort(byT)) {
    const ta = applied(ev);
    if (ev.type === 'soil_change') soilStates.push({ t: ta, soils: ev.soils || [] });
    else if (ev.type === 'new_hazard' && ev.zone) trueHazards.push({ t: ta, zone: ev.zone });
    else if (ev.type === 'sensor_fault' && !(worldLog && worldLog.length) && Number.isFinite(ta)) {
      faults.push({ t0: ta, t1: ta + (+ev.duration || 0), kind: ev.kind || 'sensor_noise' });
    }
  }
  if (worldLog && worldLog.length) {          // как в did/metrics.py: когда какой сбой действовал
    const open = {};
    for (const w of worldLog) {
      if (w.type === 'fault') {
        const iv = { t0: +w.t, t1: fin(w.until) != null ? +w.until : dur, kind: w.kind || 'sensor_noise' };
        (w.kind === 'leak' ? leaks : faults).push(iv);
      } else if (w.type === 'sensor_fault') {
        open[w.kind || 'sensor_noise'] = +w.t;
      } else if (w.type === 'sensor_recovered') {
        const k = w.kind && w.kind in open ? w.kind : (!w.kind ? Object.keys(open)[0] : null);
        if (k != null) { faults.push({ t0: open[k], t1: +w.t, kind: k }); delete open[k]; }
      }
    }
    for (const k of Object.keys(open)) faults.push({ t0: open[k], t1: planned[k] ? open[k] + planned[k] : dur, kind: k });
    faults.sort((p, q) => p.t0 - q.t0);
  }
  const samples = (sc.samples || []).map((p) => ({ x: p[0], y: p[1], tc: Infinity, at: null }));
  const collectT = [];
  for (const e of events) {
    if (e.type !== 'sample_collected') continue;
    collectT.push(+e.t);
    if (samples[e.sample]) { samples[e.sample].tc = +e.t; samples[e.sample].at = [e.x, e.y]; }
  }
  const total = (result && result.samples_total) || samples.length || 0;
  const soilsAt = (t) => {
    let cur = soilStates[0].soils;
    for (let i = 1; i < soilStates.length; i++) if (soilStates[i].t <= t) cur = soilStates[i].soils;
    return cur;
  };
  const trueMult = (t, x, y) => {
    let m = 1;
    for (const z of soilsAt(t)) if (zoneHas(z, x, y) && z.mult > m) m = z.mult;
    return m;
  };

  // --- участки пути по дорогому грунту (по истине) ---
  const soilBands = [];
  const trackMult = new Float32Array(n);
  {
    let cur = null;
    for (let i = 0; i < n; i++) {
      const m = trueMult(T[i], tr.x[i], tr.y[i]);
      trackMult[i] = m;
      if (m > 1.01) {
        if (cur && Math.abs(cur.mult - m) < 0.01) cur.t1 = T[Math.min(i + 1, n - 1)];
        else { cur = { t0: T[i], t1: T[Math.min(i + 1, n - 1)], mult: m }; soilBands.push(cur); }
      } else cur = null;
    }
  }

  // --- расследования и поправка положения (в записи их может не быть) ---
  const inquiries = readInquiries(tr0.inquiries);
  const poseFix = readPoseFix(tr0.pose_fix);

  // --- метки шкалы времени ---
  const alarms = journal.filter((j) => j.kind === 'alarm');
  // Агент «заметил» изменение, если после него появилась тревога или запись расследования с нужной меткой.
  const detectors = journal.filter((j) => (j.kind === 'alarm' || j.kind === 'inquiry') && j.data && j.data.tag);
  const worldSrc = worldLog && worldLog.length ? worldLog
    : (sc.events || []).filter((e) => !worldLog || +e.t < dur - 1).map((e) => ({ t: e.t, type: e.type, kind: e.kind })).sort(byT);
  const laneWorld = [];
  const lags = [];
  for (const w of worldSrc) {
    const mark = { t: +w.t, kind: 'world', type: w.type, text: worldText(w) };
    const tag = w.type === 'fault' ? FAULT_TAG[w.kind] : DETECT_TAG[w.type];
    if (tag) {
      const limit = w.type === 'fault' && fin(w.until) != null ? +w.until + 10 : Infinity;
      const hit = detectors.find((al) => al.t >= w.t && al.t <= limit && al.data.tag === tag);
      if (hit) {
        mark.sub = hit.t - w.t < 0.05 ? 'Агент заметил сразу' : `Агент заметил через ${sec(hit.t - w.t)}`;
        lags.push({ t0: +w.t, t1: +hit.t });
      } else if (finished) mark.sub = 'Агент этого не заметил';
    }
    laneWorld.push(mark);
  }
  const penaltyNote = (type) => {
    const pts = { collision: rules.pts_collision, false_collect: rules.pts_false_collect, hazard_hit: rules.pts_hazard_hit }[type];
    const parts = [];
    if (pts) parts.push(`${num(pts, 0)} ${plural(pts, 'балл', 'балла', 'баллов')}`);
    if (type === 'hazard_hit' && rules.hazard_battery_hit) parts.push(`−${num(rules.hazard_battery_hit, 0)} ед. заряда`);
    return parts.length ? `: ${parts.join(' и ')}` : '';
  };
  const EVENT_TEXT = {
    collision: 'Столкновение со стеной', false_collect: 'Сбор впустую', hazard_hit: 'Въезд в опасную зону',
  };
  const laneJudge = [];
  const penalties = [];
  for (const e of events) {
    if (e.type === 'sample_collected') {
      laneJudge.push({ t: +e.t, kind: 'sample', text: `Образец собран${e.collected ? `: ${e.collected}-й из ${total}` : ''}` });
    } else if (PENALTY_TYPES[e.type]) {
      const text = `${EVENT_TEXT[e.type]} — штраф${penaltyNote(e.type)}`;
      laneJudge.push({ t: +e.t, kind: 'penalty', text });
      penalties.push({ t: +e.t, type: e.type, x: e.x, y: e.y, text });
    } else {
      laneJudge.push({ t: +e.t, kind: 'penalty', text: `Сообщение судьи: ${e.type}` });
    }
  }
  const laneAgent = [];
  for (const al of alarms) laneAgent.push({ t: +al.t, kind: 'alarm', text: `Тревога агента: ${ru(al.text)}` });
  for (const p of plans) {
    laneAgent.push({ t: +p.t, kind: 'plan', text: `Новый план${p.trigger ? ` — ${TRIGGER_RU[p.trigger] || p.trigger}` : ''}` });
  }
  laneAgent.sort(byT);
  const laneInquiry = [];
  const inquirySpans = [];
  for (const q of inquiries) {
    const c = q.conclusion;
    const sub = !c ? 'Расследование не закончено'
      : c.status === 'identified' ? `Вывод: причина найдена${c.best ? ` — ${c.best.statement}` : ''}`
        : 'Вывод: недостаточно данных';
    laneInquiry.push({ t: q.t0, kind: 'inquiry', text: `Расследование ${q.id}: ${q.anomaly.text}`, sub });
    inquirySpans.push({ t0: q.t0, t1: q.t1 != null ? q.t1 : dur });
  }
  const evSet = new Set();
  for (const mk of laneWorld) evSet.add(mk.t);
  for (const mk of laneJudge) evSet.add(mk.t);
  for (const al of alarms) evSet.add(+al.t);
  for (const q of inquiries) { evSet.add(q.t0); if (q.t1 != null) evSet.add(q.t1); }
  const eventTimes = Array.from(evSet).sort((p, q) => p - q);

  // --- сообщения поверх арены ---
  const notices = [];
  for (const w of worldSrc) {
    if (w.type === 'sensor_fault') continue;
    if (WORLD_NOTICE[w.type]) notices.push({ t0: +w.t, t1: +w.t + 6, side: 'truth', tone: w.type === 'sensor_recovered' ? 'ok' : 'warn', text: WORLD_NOTICE[w.type] });
  }
  for (const f of faults) {
    const what = f.kind && f.kind !== 'sensor_noise' && FAULT_RU[f.kind] ? `Сбой: ${FAULT_RU[f.kind]}` : 'Сбой датчика: показания шумят';
    notices.push({ t0: f.t0, t1: f.t1, side: 'truth', tone: 'warn', text: `${what}, роботу об этом не сообщили` });
  }
  for (const lk of leaks) notices.push({ t0: lk.t0, t1: lk.t1, side: 'truth', tone: 'warn', text: 'Сбой: батарея сама теряет заряд, роботу об этом не сообщили' });
  for (const al of alarms) notices.push({ t0: +al.t, t1: +al.t + 6, side: 'agent', tone: 'agent', text: `Агент заметил: ${ru(al.text)}` });
  for (const q of inquiries) {
    const end = q.t1 != null ? q.t1 : dur;
    notices.push({ t0: q.t0, t1: Math.max(end, q.t0 + 0.5), side: 'agent', tone: 'agent', text: `Расследование ${q.id}: ${q.anomaly.text}` });
    if (q.t1 != null && q.conclusion) {
      const c = q.conclusion;
      const text = c.status === 'identified' ? `Вывод ${q.id}: ${c.best ? c.best.statement : 'причина найдена'}` : `Вывод ${q.id}: недостаточно данных, причина не названа`;
      notices.push({ t0: q.t1, t1: q.t1 + 7, side: 'agent', tone: c.status === 'identified' ? 'agent' : 'agent', text });
    }
  }
  notices.sort((p, q) => p.t0 - q.t0);

  // --- планы: в старых записях подцели пустые, тогда берём их из журнала ---
  const planList = plans.map((p) => {
    let subgoals = Array.isArray(p.subgoals) ? p.subgoals : [];
    if (!subgoals.length) {
      const j = journal.find((e) => e.kind === 'decision' && Math.abs(e.t - p.t) < 0.051 && e.data && Array.isArray(e.data.subgoals));
      if (j) subgoals = j.data.subgoals.map((x) => (typeof x === 'string' ? parseSubgoal(x) : x));
    }
    return { t: +p.t, source: p.source, trigger: p.trigger, reasoning: p.reasoning, subgoals };
  });

  // --- гипотезы ---
  const hypTimes = [];
  for (const hp of hyps) {
    if (hp.t_open != null) hypTimes.push(+hp.t_open);
    if (hp.t_close != null) hypTimes.push(+hp.t_close);
  }
  hypTimes.sort((p, q) => p - q);

  const penaltyPts = { collision: +rules.pts_collision || 0, false_collect: +rules.pts_false_collect || 0, hazard_hit: +rules.pts_hazard_hit || 0 };
  const canScore = rules.pts_sample != null;

  const model = {
    trace: tr0, n, T, tr, rules, sc, result, modes, events, journal, hyps, llm, paths, scans, samples, total,
    t0, dur, span: Math.max(dur - t0, 0.001), base, finished, faults, trueHazards,
    // показание ниже этого порога не отличить от шума: «в пределах дальности датчика образцов нет»
    quiet: Math.max(0.02, 1.5 * (+rules.sensor_sigma || 0)),
    meta: {
      backend: tr0.backend || null, id: tr0.id || null, arm: tr0.arm || null,
      agent: (tr0.agent && tr0.agent.name) || null, level: sc.level || null, seed: sc.seed != null ? sc.seed : null,
    }, memHazards, penalties, soilBands, trackMult, eventTimes, notices, lags,
    inquiries, inquirySpans, poseFix, leaks,
    energyModel: tr0.energy_model && typeof tr0.energy_model === 'object' ? tr0.energy_model : null,
    lanes: { world: laneWorld, judge: laneJudge, agent: laneAgent, inquiry: laneInquiry },
    plans: planList,
    plansT: planList.map((p) => p.t),
    pathsT: paths.map((p) => +p.t),
    scansT: scans.map((p) => +p.t),
    journalT: journal.map((p) => +p.t),
    llmT: llm.map((p) => +p.t || 0),
    hypTimes,
    belief: gridStore(tr0.belief, prev && prev.belief),
    soil: gridStore(tr0.soil, prev && prev.soil),
    soilsAt,
    collectedAt: (t) => upperBound(collectT, t),
    faultAt: (t) => faults.find((f) => t >= f.t0 && t < f.t1) || null,
    leakAt: (t) => leaks.find((f) => t >= f.t0 && t < f.t1) || null,
    /** Поправка положения к моменту t: {shift, ang} или null, если её ещё не было. */
    poseAt(t) {
      if (!poseFix) return null;
      const i = upperBound(poseFix.T, t) - 1;
      return i < 0 ? null : { shift: poseFix.S[i], ang: poseFix.A[i] };
    },
    stateAt(t) {
      if (!n) {
        const b = base || [0, 0];
        return { i: -1, has: false, x: b[0], y: b[1], th: 0, battery: rules.battery_start, sensor: 0, mode: 'start', mult: 1 };
      }
      let i = upperBound(T, t) - 1;
      if (i < 0) i = 0;
      if (i > n - 1) i = n - 1;
      const j = Math.min(i + 1, n - 1);
      const span = T[j] - T[i];
      const f = j > i && span > 0 ? clamp((t - T[i]) / span, 0, 1) : 0;
      const TH = tr.th || [];
      let dth = (TH[j] || 0) - (TH[i] || 0);
      dth = Math.atan2(Math.sin(dth), Math.cos(dth));
      return {
        i, has: true, f,
        x: lerp(tr.x[i], tr.x[j], f), y: lerp(tr.y[i], tr.y[j], f), th: (TH[i] || 0) + dth * f,
        battery: tr.battery ? lerp(tr.battery[i], tr.battery[j], f) : 0,
        sensor: tr.sensor ? tr.sensor[i] : 0,           // показание берётся как есть, без сглаживания
        mode: (tr.mode && modes[tr.mode[i]]) || 'start', mult: trackMult[i],
      };
    },
    scoreAt(t) {
      if (finished && result && result.score != null && t >= tFinish - 1e-6) return +result.score;
      if (!canScore) return null;
      let pts = rules.pts_sample * upperBound(collectT, t);
      for (const p of penalties) if (p.t <= t) pts += penaltyPts[p.type] || 0;
      return pts;
    },
  };
  return model;
}

function hypStatusAt(hp, t) {
  if (hp.t_open != null && t < hp.t_open) return null;
  if (hp.t_close != null && t >= hp.t_close) return hp.status || 'open';
  return 'open';
}

function outcomeOf(result) {
  if (!result) return null;
  if (result.reason === 'battery') return { tone: 'bad', text: 'заряд кончился, до базы не доехал' };
  if (result.reason === 'timeout') return { tone: 'bad', text: 'время вышло' };
  if (result.returned) return { tone: 'good', text: 'вернулся на базу' };
  if (result.reason === 'finish') return { tone: 'bad', text: 'финиш не на базе' };
  return null;
}

/* ============================================================================================
 * Часы воспроизведения
 * ========================================================================================== */

function createClock() {
  let t = 0, t0 = 0, dur = 0, playing = false, speed = 1, raf = 0, last = 0, held = false, dead = false;
  const timeSubs = new Set();
  const stateSubs = new Set();
  const emitState = () => { for (const cb of stateSubs) cb(); };
  function tick(now) {
    raf = 0;
    if (dead) return;
    if (playing) {
      const dt = clamp((now - last) / 1000, 0, 0.25);
      last = now;
      if (!held) {
        t += dt * speed;
        if (t >= dur) { t = dur; playing = false; emitState(); }
      }
      if (playing) raf = requestAnimationFrame(tick);
    }
    for (const cb of timeSubs) cb(t);
  }
  const request = () => { if (!raf && !dead) raf = requestAnimationFrame(tick); };
  return {
    get t() { return t; },
    get dur() { return dur; },
    get start() { return t0; },
    get playing() { return playing; },
    get speed() { return speed; },
    setRange(a, b) { t0 = Math.max(0, +a || 0); dur = Math.max(t0, +b || 0); t = clamp(t, t0, dur); request(); },
    seek(v) { t = clamp(+v || 0, t0, dur); request(); },
    play() {
      if (playing || dur <= t0) return;
      if (t >= dur - 1e-6) t = t0;
      playing = true;
      last = performance.now();
      emitState();
      request();
    },
    pause() { if (!playing) return; playing = false; emitState(); request(); },
    toggle() { if (playing) this.pause(); else this.play(); },
    setSpeed(k) { speed = Math.max(0.05, +k || 1); emitState(); },
    hold(v) { held = !!v; last = performance.now(); },
    request,
    onTime(cb) { timeSubs.add(cb); return () => timeSubs.delete(cb); },
    onState(cb) { stateSubs.add(cb); return () => stateSubs.delete(cb); },
    destroy() { dead = true; if (raf) cancelAnimationFrame(raf); timeSubs.clear(); stateSubs.clear(); },
  };
}

/* ============================================================================================
 * Клавиши: пробел, стрелки, Shift+стрелки. Отвечает тот проигрыватель, с которым работали последним.
 * ========================================================================================== */

const players = new Set();
let activePlayer = null;
let keysBound = false;
let spaceTaken = false;

const isShown = (el) => el.isConnected && el.getClientRects().length > 0;

function pickPlayer(target) {
  if (target && target.nodeType === 1) for (const p of players) if (p.root.contains(target)) return p;
  if (activePlayer && players.has(activePlayer) && isShown(activePlayer.root)) return activePlayer;
  for (const p of players) if (isShown(p.root)) return p;
  return null;
}

function onKeyDown(e) {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  const space = e.key === ' ' || e.code === 'Space';
  const left = e.key === 'ArrowLeft';
  const right = e.key === 'ArrowRight';
  if (!space && !left && !right) return;
  const tgt = e.target && e.target.nodeType === 1 ? e.target : null;
  const p = pickPlayer(tgt);
  if (!p) return;
  if (tgt) {
    if (tgt.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]')) return;
    if (!p.root.contains(tgt) && tgt.closest('button, a[href], summary, [role="button"], [role="slider"], [role="tab"]')) return;
  }
  e.preventDefault();
  if (space) { spaceTaken = true; if (!e.repeat) p.toggle(); return; }
  if (e.shiftKey) p.jumpEvent(right ? 1 : -1);
  else p.step(right ? 1 : -1);
}
function onKeyUp(e) {
  if (spaceTaken && (e.key === ' ' || e.code === 'Space')) { e.preventDefault(); spaceTaken = false; }
}
function registerPlayer(p) {
  players.add(p);
  if (!activePlayer) activePlayer = p;
  if (!keysBound) {
    keysBound = true;
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
  }
  const touch = () => { activePlayer = p; };
  p.root.addEventListener('pointerdown', touch, true);
  p.root.addEventListener('focusin', touch, true);
  return () => {
    players.delete(p);
    if (activePlayer === p) activePlayer = null;
    p.root.removeEventListener('pointerdown', touch, true);
    p.root.removeEventListener('focusin', touch, true);
  };
}

/* ============================================================================================
 * Арена
 * ========================================================================================== */

function createArena(host, cfg) {
  const geom = cfg.geom;
  let model = cfg.model;
  let layers = cfg.layers;
  let theme = cfg.theme;
  const wrap = h('div', 'rp-arena');
  const canvas = h('canvas', 'rp-arena-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Арена: робот, образцы, грунт и опасные зоны в текущий момент прогона');
  const noticeBox = h('div', 'rp-notices');
  noticeBox.setAttribute('aria-live', 'polite');
  wrap.append(canvas, noticeBox);
  host.appendChild(wrap);
  const ctx = canvas.getContext('2d');
  const staticCv = document.createElement('canvas');
  const heatCv = document.createElement('canvas');
  const soilCv = document.createElement('canvas');
  let heatImg = null, heatKey = '', heatLut = null;
  let soilKey = '';
  let cssW = 0, cssH = 0, dpr = 1, k = 1, ox = 0, oy = 0;
  let floorPath = null;
  let noticeKey = '';
  let lastT = 0;
  const { vx0, vy0, vx1, vy1 } = geom.view;
  const X = (x) => ox + (x - vx0) * k;
  const Y = (y) => oy + (vy1 - y) * k;

  function buildLut() {
    heatLut = new Uint8ClampedArray(256 * 4);
    for (let v = 0; v < 256; v++) {
      const u = v / 255;
      const c = mix(theme.green, theme.greenInk, Math.pow(u, 1.1));
      heatLut[v * 4] = c[0];
      heatLut[v * 4 + 1] = c[1];
      heatLut[v * 4 + 2] = c[2];
      heatLut[v * 4 + 3] = v === 0 ? 0 : Math.round(255 * 0.94 * Math.pow(u, 0.62));
    }
    heatKey = '';
    soilKey = '';
  }

  function layout() {
    const w = wrap.clientWidth;
    if (!w) return false;
    const aspect = (vy1 - vy0) / (vx1 - vx0);
    const maxH = cfg.maxHeight ? cfg.maxHeight() : Infinity;
    const hh = Math.max(160, Math.min(w * aspect, maxH));
    const ratio = Math.min(3, window.devicePixelRatio || 1);
    if (Math.abs(w - cssW) < 0.5 && Math.abs(hh - cssH) < 0.5 && ratio === dpr && floorPath) return true;
    cssW = w;
    cssH = hh;
    dpr = ratio;
    wrap.style.height = `${hh + (wrap.offsetHeight - wrap.clientHeight)}px`;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${hh}px`;
    for (const cv of [canvas, staticCv]) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(hh * dpr);
    }
    k = Math.min(w / (vx1 - vx0), hh / (vy1 - vy0));
    ox = (w - (vx1 - vx0) * k) / 2;
    oy = (hh - (vy1 - vy0) * k) / 2;
    drawStatic();
    return true;
  }

  function drawStatic() {
    const c = staticCv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, cssW, cssH);
    floorPath = new Path2D();
    for (const poly of geom.floors) {
      poly.forEach((p, i) => (i ? floorPath.lineTo(X(p[0]), Y(p[1])) : floorPath.moveTo(X(p[0]), Y(p[1]))));
      floorPath.closePath();
    }
    const pillars = new Path2D();
    for (const o of geom.obstacles) {
      for (const target of [floorPath, pillars]) {
        if (o.circle) {
          target.moveTo(X(o.circle.x) + o.circle.r * k, Y(o.circle.y));
          target.arc(X(o.circle.x), Y(o.circle.y), o.circle.r * k, 0, TAU);
        } else {
          o.pts.forEach((p, i) => (i ? target.lineTo(X(p[0]), Y(p[1])) : target.moveTo(X(p[0]), Y(p[1]))));
        }
        target.closePath();
      }
    }
    c.fillStyle = rgba(theme.surface, 1);
    c.fill(floorPath, 'evenodd');
    // сетка через метр
    c.save();
    c.clip(floorPath, 'evenodd');
    c.strokeStyle = rgba(theme.line, 0.9);
    c.lineWidth = 1;
    c.beginPath();
    for (let gx = Math.ceil(vx0); gx <= vx1; gx++) { const px = Math.round(X(gx)) + 0.5; c.moveTo(px, 0); c.lineTo(px, cssH); }
    for (let gy = Math.ceil(vy0); gy <= vy1; gy++) { const py = Math.round(Y(gy)) + 0.5; c.moveTo(0, py); c.lineTo(cssW, py); }
    c.stroke();
    c.restore();
    c.fillStyle = rgba(theme.pillar, 1);
    c.fill(pillars);
    c.lineJoin = 'round';
    c.strokeStyle = rgba(theme.wall, 1);
    c.lineWidth = 2;
    c.stroke(floorPath);
    // масштаб
    const bx = 12, by = cssH - 12;
    c.strokeStyle = rgba(theme.ink3, 1);
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(bx, by - 4); c.lineTo(bx, by); c.lineTo(bx + k, by); c.lineTo(bx + k, by - 4);
    c.stroke();
    c.fillStyle = rgba(theme.ink3, 1);
    c.font = `12px ${theme.font}`;
    c.textAlign = 'left';
    c.textBaseline = 'alphabetic';
    c.fillText('1 м', bx + 5, by - 5);
  }

  /* ---------- слои убеждений ---------- */

  function heatCanvas(t) {
    const b = model.belief;
    if (!b) return null;
    let a = upperBound(b.times, t) - 1;
    if (a < 0) a = 0;
    const a2 = Math.min(a + 1, b.times.length - 1);
    const span = b.times[a2] - b.times[a];
    const f = a2 > a && span > 0 ? Math.round(clamp((t - b.times[a]) / span, 0, 1) * 8) / 8 : 0;
    const key = `${a}:${f}:${b.times.length}`;
    if (key === heatKey) return heatCv;
    heatKey = key;
    if (heatCv.width !== b.w || heatCv.height !== b.h || !heatImg) {
      heatCv.width = b.w;
      heatCv.height = b.h;
      heatImg = heatCv.getContext('2d').createImageData(b.w, b.h);
    }
    const A = b.get(a);
    const B = f > 0 ? b.get(a2) : A;
    const d = heatImg.data;
    for (let r = 0; r < b.h; r++) {
      const src = r * b.w;
      let dst = (b.h - 1 - r) * b.w * 4;
      for (let cx = 0; cx < b.w; cx++, dst += 4) {
        const v = f > 0 ? Math.round(A[src + cx] + (B[src + cx] - A[src + cx]) * f) : A[src + cx];
        const q = v * 4;
        d[dst] = heatLut[q];
        d[dst + 1] = heatLut[q + 1];
        d[dst + 2] = heatLut[q + 2];
        d[dst + 3] = heatLut[q + 3];
      }
    }
    heatCv.getContext('2d').putImageData(heatImg, 0, 0);
    return heatCv;
  }

  function soilCanvas(t) {
    const g = model.soil;
    if (!g) return null;
    const a = upperBound(g.times, t) - 1;
    if (a < 0) return null;
    const key = `${a}:${g.times.length}`;
    if (key === soilKey) return soilCv;
    soilKey = key;
    if (soilCv.width !== g.w || soilCv.height !== g.h) { soilCv.width = g.w; soilCv.height = g.h; }
    const c = soilCv.getContext('2d');
    const img = c.createImageData(g.w, g.h);
    const A = g.get(a);
    const d = img.data;
    for (let r = 0; r < g.h; r++) {
      let dst = (g.h - 1 - r) * g.w * 4;
      for (let cx = 0; cx < g.w; cx++, dst += 4) {
        const v = A[r * g.w + cx];
        const m = v / g.scale;
        if (!v || m < 1.2) continue;
        const col = soilRgb(theme, m);
        d[dst] = col[0];
        d[dst + 1] = col[1];
        d[dst + 2] = col[2];
        d[dst + 3] = Math.round(255 * clamp(0.3 + 0.17 * (m - 1), 0.34, 0.8));
      }
    }
    c.putImageData(img, 0, 0);
    return soilCv;
  }

  function drawGrid(cv, g, smooth) {
    ctx.imageSmoothingEnabled = smooth;
    if (smooth) ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(cv, X(g.x0), Y(g.y0 + g.h * g.res), g.w * g.res * k, g.h * g.res * k);
    ctx.imageSmoothingEnabled = true;
  }

  /* ---------- примитивы ---------- */

  function zonePath(z) {
    ctx.beginPath();
    if (z.shape === 'rect') ctx.rect(X(z.x - z.w / 2), Y(z.y + z.h / 2), z.w * k, z.h * k);
    else ctx.arc(X(z.x), Y(z.y), z.r * k, 0, TAU);
  }
  function zoneBox(z) {
    const hw = z.shape === 'rect' ? z.w / 2 : z.r;
    const hh = z.shape === 'rect' ? z.h / 2 : z.r;
    return [X(z.x - hw), Y(z.y + hh), hw * 2 * k, hh * 2 * k];
  }
  function label(text, px, py, opts = {}) {
    ctx.font = `${opts.weight || 600} ${opts.size || 12}px ${theme.font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const w = ctx.measureText(text).width + 10;
    const hh = (opts.size || 12) + 7;
    const lx = clamp(px, w / 2 + 3, cssW - w / 2 - 3);
    const ly = clamp(py, hh / 2 + 3, cssH - hh / 2 - 3);
    ctx.fillStyle = rgba(theme.surface, opts.bg != null ? opts.bg : 0.92);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(lx - w / 2, ly - hh / 2, w, hh, 5);
    else ctx.rect(lx - w / 2, ly - hh / 2, w, hh);
    ctx.fill();
    if (opts.border) { ctx.strokeStyle = opts.border; ctx.lineWidth = 1; ctx.stroke(); }
    ctx.fillStyle = rgba(opts.ink || theme.ink, 1);
    ctx.fillText(text, lx, ly + 0.5);
  }
  function diamond(px, py, r) {
    ctx.beginPath();
    ctx.moveTo(px, py - r); ctx.lineTo(px + r, py); ctx.lineTo(px, py + r); ctx.lineTo(px - r, py);
    ctx.closePath();
  }

  /* ---------- кадр ---------- */

  function render(t) {
    lastT = t;
    if (!floorPath && !layout()) return;
    const st = model.stateAt(t);
    const L = layers;
    const th = theme;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.drawImage(staticCv, 0, 0, cssW, cssH);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'butt';

    // 1. Что думает агент — заливка
    ctx.save();
    ctx.clip(floorPath, 'evenodd');
    if (L.beliefSoil) {
      const cv = soilCanvas(t);
      if (cv) drawGrid(cv, model.soil, false);
    }
    if (L.beliefSamples) {
      const cv = heatCanvas(t);
      if (cv) drawGrid(cv, model.belief, true);
    }
    if (L.beliefHazards) {
      for (const z of model.memHazards) {
        if (z.t > t) break;
        ctx.beginPath();
        ctx.arc(X(z.x), Y(z.y), z.r * k, 0, TAU);
        ctx.fillStyle = rgba(th.red, 0.24);
        ctx.fill();
      }
    }
    ctx.restore();

    // 2. Как на самом деле — контуры и значки (зоны обрезаются по стенам)
    ctx.save();
    ctx.clip(floorPath, 'evenodd');
    if (L.truthSoil) {
      for (const z of model.soilsAt(t)) {
        const col = soilRgb(th, z.mult);
        ctx.save();
        zonePath(z);
        ctx.clip();
        const [bx, by, bw, bh] = zoneBox(z);
        ctx.strokeStyle = rgba(col, 0.5);
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        for (let d = -bh; d < bw; d += 9) { ctx.moveTo(bx + d, by + bh); ctx.lineTo(bx + d + bh, by); }
        ctx.stroke();
        ctx.restore();
        zonePath(z);
        ctx.strokeStyle = rgba(col, 1);
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }
    if (L.truthHazards) {
      for (const hz of model.trueHazards) {
        if (hz.t > t) continue;
        const z = hz.zone;
        zonePath(z);
        ctx.fillStyle = rgba(th.red, 0.07);
        ctx.fill();
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = rgba(th.red, 1);
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.setLineDash([]);
        const age = t - hz.t;
        if (age >= 0 && age < 4) {
          const u = (age % 1.3) / 1.3;
          ctx.beginPath();
          ctx.arc(X(z.x), Y(z.y), ((z.shape === 'rect' ? Math.max(z.w, z.h) / 2 : z.r) + 0.3 * u) * k, 0, TAU);
          ctx.strokeStyle = rgba(th.red, 0.7 * (1 - u));
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }
    }
    ctx.restore();
    if (L.truthSoil) {
      for (const z of model.soilsAt(t)) label(mult(z.mult), X(z.x), Y(z.y), { border: rgba(soilRgb(th, z.mult), 1) });
    }
    if (L.truthHazards) {
      for (const hz of model.trueHazards) {
        if (hz.t > t) continue;
        const z = hz.zone;
        // знак «внимание»
        const px = X(z.x), py = Y(z.y);
        ctx.beginPath();
        ctx.moveTo(px, py - 9); ctx.lineTo(px + 9.5, py + 7); ctx.lineTo(px - 9.5, py + 7);
        ctx.closePath();
        ctx.fillStyle = rgba(th.red, 1);
        ctx.fill();
        ctx.strokeStyle = rgba(th.surface, 1);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.8;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(px, py - 3.5); ctx.lineTo(px, py + 1.5);
        ctx.moveTo(px, py + 4.4); ctx.lineTo(px, py + 4.5);
        ctx.stroke();
        ctx.lineCap = 'butt';
      }
    }

    // 3. База
    const base = model.base || geom.base;
    if (base) {
      const px = X(base[0]), py = Y(base[1]);
      const r = model.rules.base_radius_m * k;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, TAU);
      ctx.fillStyle = rgba(th.ink, 0.045);
      ctx.fill();
      ctx.strokeStyle = rgba(th.ink2, 0.9);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.font = `600 12px ${th.font}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.lineWidth = 3;
      ctx.strokeStyle = rgba(th.surface, 0.9);
      ctx.strokeText('база', px, py + r + 3);
      ctx.fillStyle = rgba(th.ink2, 1);
      ctx.fillText('база', px, py + r + 3);
    }

    const rx = X(st.x), ry = Y(st.y);

    // 4. Лидар
    if (L.lidar && st.has && model.scans.length) {
      let a = upperBound(model.scansT, t);
      if (a >= model.scans.length || (a > 0 && t - model.scansT[a - 1] <= model.scansT[a] - t)) a -= 1;
      const scan = model.scans[a];
      if (scan && Math.abs(scan.t - t) < 2.5 && scan.r && scan.r.length) {
        const nr = scan.r.length;
        ctx.beginPath();
        const hits = [];
        for (let i = 0; i < nr; i++) {
          const rr = scan.r[i];
          if (!rr) continue;
          const ang = scan.th + (i * TAU) / nr;
          const hx = X(scan.x + Math.cos(ang) * rr / 100);
          const hy = Y(scan.y + Math.sin(ang) * rr / 100);
          hits.push(hx, hy);
          ctx.moveTo(rx, ry);
          ctx.lineTo(hx, hy);
        }
        ctx.strokeStyle = rgba(th.ink3, 0.13);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = rgba(th.ink2, 0.6);
        for (let i = 0; i < hits.length; i += 2) ctx.fillRect(hits[i] - 1.25, hits[i + 1] - 1.25, 2.5, 2.5);
      }
    }

    // 5. След
    if (L.trail && st.has) {
      ctx.beginPath();
      ctx.moveTo(X(model.tr.x[0]), Y(model.tr.y[0]));
      for (let i = 1; i <= st.i; i++) ctx.lineTo(X(model.tr.x[i]), Y(model.tr.y[i]));
      ctx.lineTo(rx, ry);
      ctx.strokeStyle = rgba(th.ink3, 0.85);
      ctx.lineWidth = 2.25;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.lineCap = 'butt';
    }

    // 6. План и текущий путь
    if (L.path && st.has && st.mode !== 'done') {
      const pi = upperBound(model.plansT, t) - 1;
      const plan = pi >= 0 ? model.plans[pi] : null;
      const goals = plan ? plan.subgoals.filter((g) => g && g.x != null && g.y != null) : [];
      if (goals.length > 1) {
        ctx.beginPath();
        goals.forEach((g, i) => (i ? ctx.lineTo(X(g.x), Y(g.y)) : ctx.moveTo(X(g.x), Y(g.y))));
        ctx.setLineDash([2, 5]);
        ctx.strokeStyle = rgba(th.ink2, 0.55);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const many = goals.length > 8;
      goals.forEach((g, i) => {
        const px = X(g.x), py = Y(g.y);
        if (many) {
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, TAU);
          ctx.fillStyle = rgba(th.surface, 1);
          ctx.fill();
          ctx.strokeStyle = rgba(th.ink2, 0.9);
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else if (goals.length > 1) {
          ctx.beginPath();
          ctx.arc(px, py, 8, 0, TAU);
          ctx.fillStyle = rgba(th.surface, 0.95);
          ctx.fill();
          ctx.strokeStyle = rgba(th.ink2, 0.9);
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.fillStyle = rgba(th.ink, 1);
          ctx.font = `600 11px ${th.font}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(i + 1), px, py + 0.5);
        }
      });
      const qi = upperBound(model.pathsT, t) - 1;
      const path = qi >= 0 ? model.paths[qi] : null;
      if (path && path.pts && path.pts.length) {
        let near = 0, nd = Infinity;
        for (let i = 0; i < path.pts.length; i++) {
          const d = (path.pts[i][0] - st.x) ** 2 + (path.pts[i][1] - st.y) ** 2;
          if (d < nd) { nd = d; near = i; }
        }
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        for (let i = near + 1; i < path.pts.length; i++) ctx.lineTo(X(path.pts[i][0]), Y(path.pts[i][1]));
        const goal = path.goal || path.pts[path.pts.length - 1];
        ctx.lineTo(X(goal[0]), Y(goal[1]));
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = rgba(th.ink, 0.85);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
        const gx = X(goal[0]), gy = Y(goal[1]);
        ctx.beginPath();
        ctx.arc(gx, gy, 7, 0, TAU);
        ctx.strokeStyle = rgba(th.surface, 0.95);
        ctx.lineWidth = 5;
        ctx.stroke();
        ctx.strokeStyle = rgba(th.ink, 1);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(gx - 11, gy); ctx.lineTo(gx - 4, gy); ctx.moveTo(gx + 4, gy); ctx.lineTo(gx + 11, gy);
        ctx.moveTo(gx, gy - 11); ctx.lineTo(gx, gy - 4); ctx.moveTo(gx, gy + 4); ctx.lineTo(gx, gy + 11);
        ctx.stroke();
      }
    }

    // 7. Радиус сбора и кольцо датчика
    let ringNote = null;
    if (st.has) {
      const rc = model.rules.collect_radius_m * k;
      ctx.beginPath();
      ctx.arc(rx, ry, rc, 0, TAU);
      ctx.fillStyle = rgba(th.ink, 0.05);
      ctx.fill();
      ctx.strokeStyle = rgba(th.ink, 0.5);
      ctx.lineWidth = 1.25;
      ctx.stroke();
      if (L.ring && st.mode !== 'done') {
        const range = model.rules.sensor_range_m;
        // берём само показание, без сглаживания: кольцо скачет ровно так, как шумит датчик
        const sv = clamp(st.sensor, 0, 1);
        if (sv <= model.quiet) {
          ctx.beginPath();
          ctx.arc(rx, ry, range * k, 0, TAU);
          ctx.setLineDash([4, 6]);
          ctx.strokeStyle = rgba(th.blue, 0.5);
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.setLineDash([]);
          ringNote = { text: `ближе ${num(range, 0)} м образцов нет`, r: range * k };
        } else {
          const d = (1 - sv) * range;
          const band = Math.max(3, 2 * model.rules.sensor_sigma * range * k);
          ctx.beginPath();
          ctx.arc(rx, ry, Math.max(1, d * k), 0, TAU);
          ctx.strokeStyle = rgba(th.blue, 0.16);
          ctx.lineWidth = band;
          ctx.stroke();
          ctx.strokeStyle = rgba(th.blue, 0.95);
          ctx.lineWidth = 2;
          ctx.stroke();
          ringNote = { text: `образец где-то на кольце: ≈ ${num(d, 1)} м`, r: d * k };
        }
      }
    }

    // 8. Образцы
    if (L.truthSamples) {
      for (const sm of model.samples) {
        const px = X(sm.x), py = Y(sm.y);
        if (sm.tc <= t) {
          diamond(px, py, 7.5);
          ctx.fillStyle = rgba(th.surface, 0.95);
          ctx.fill();
          ctx.strokeStyle = rgba(th.ink3, 1);
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(px - 3, py + 0.2); ctx.lineTo(px - 0.8, py + 2.4); ctx.lineTo(px + 3.2, py - 2.2);
          ctx.strokeStyle = rgba(th.ink2, 1);
          ctx.lineWidth = 1.6;
          ctx.stroke();
          const age = t - sm.tc;
          if (age < 1.6) {
            ctx.beginPath();
            ctx.arc(px, py, (0.12 + 0.45 * (age / 1.6)) * k, 0, TAU);
            ctx.strokeStyle = rgba(th.green, 0.9 * (1 - age / 1.6));
            ctx.lineWidth = 3;
            ctx.stroke();
          }
        } else {
          diamond(px, py, 9.5);
          ctx.fillStyle = rgba(th.surface, 1);
          ctx.fill();
          ctx.strokeStyle = rgba(th.greenInk, 1);
          ctx.lineWidth = 2.5;
          ctx.stroke();
          diamond(px, py, 4);
          ctx.fillStyle = rgba(th.green, 1);
          ctx.fill();
        }
      }
    } else if (L.beliefSamples) {
      // агент знает только то, что сбор удался, и где он в этот момент стоял
      for (const sm of model.samples) {
        if (sm.tc > t || !sm.at) continue;
        const px = X(sm.at[0]), py = Y(sm.at[1]);
        diamond(px, py, 7.5);
        ctx.fillStyle = rgba(th.surface, 0.95);
        ctx.fill();
        ctx.strokeStyle = rgba(th.ink3, 1);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px - 3, py + 0.2); ctx.lineTo(px - 0.8, py + 2.4); ctx.lineTo(px + 3.2, py - 2.2);
        ctx.strokeStyle = rgba(th.ink2, 1);
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }

    // 9. Штрафы: где случились
    for (const p of model.penalties) {
      if (p.t > t || p.x == null) continue;
      const px = X(p.x), py = Y(p.y);
      const age = t - p.t;
      if (age < 2) {
        ctx.beginPath();
        ctx.arc(px, py, 7 + 22 * (age / 2), 0, TAU);
        ctx.strokeStyle = rgba(th.red, 0.9 * (1 - age / 2));
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, TAU);
      ctx.fillStyle = rgba(th.red, 1);
      ctx.fill();
      ctx.strokeStyle = rgba(th.surface, 1);
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 2.6, py - 2.6); ctx.lineTo(px + 2.6, py + 2.6);
      ctx.moveTo(px + 2.6, py - 2.6); ctx.lineTo(px - 2.6, py + 2.6);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.7;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.lineCap = 'butt';
    }

    // 9а. Расследования: где агент заметил странность (это знание агента, а не истина)
    if (model.inquiries.length && (L.beliefSamples || L.beliefSoil || L.beliefHazards)) {
      for (const q of model.inquiries) {
        if (q.t0 > t) break;
        if (q.anomaly.x == null || q.anomaly.y == null) continue;
        const px = X(q.anomaly.x), py = Y(q.anomaly.y);
        const going = q.t1 == null ? !model.finished : t < q.t1;
        if (going) {
          const u = ((t - q.t0) % 1.4) / 1.4;
          ctx.beginPath();
          ctx.arc(px, py, 12 + 16 * u, 0, TAU);
          ctx.strokeStyle = rgba(th.ink, 0.75 * (1 - u));
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        ctx.font = `700 10px ${th.font}`;
        const w2 = Math.max(18, ctx.measureText(q.id).width + 9);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(px - w2 / 2, py - 9, w2, 18, 9);
        else ctx.rect(px - w2 / 2, py - 9, w2, 18);
        ctx.fillStyle = rgba(going ? th.ink : th.surface, 1);
        ctx.fill();
        ctx.strokeStyle = rgba(going ? th.surface : th.ink, 1);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = going ? '#fff' : rgba(th.ink, 1);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(q.id, px, py + 0.5);
      }
    }

    // 10. Робот
    if (st.has) {
      const rr = clamp(0.105 * k, 7.5, 16);
      ctx.save();
      ctx.translate(rx, ry);
      ctx.rotate(-st.th);
      ctx.beginPath();
      ctx.arc(0, 0, rr, 0, TAU);
      ctx.fillStyle = rgba(th.ink, 1);
      ctx.fill();
      ctx.strokeStyle = rgba(th.surface, 1);
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(rr * 0.82, 0); ctx.lineTo(-rr * 0.05, -rr * 0.5); ctx.lineTo(-rr * 0.05, rr * 0.5);
      ctx.closePath();
      ctx.fillStyle = rgba(th.surface, 1);
      ctx.fill();
      ctx.restore();
    }
    if (ringNote && !cfg.compact) {
      const up = ry - ringNote.r - 13 > 14;
      label(ringNote.text, rx, up ? ry - ringNote.r - 13 : ry + ringNote.r + 13, { weight: 500, size: 12, ink: th.ink2 });
    }

    renderNotices(t);
  }

  function renderNotices(t) {
    const truthOn = layers.truthSamples || layers.truthSoil || layers.truthHazards;
    const agentOn = layers.beliefSamples || layers.beliefSoil || layers.beliefHazards;
    const act = [];
    for (let i = 0; i < model.notices.length; i++) {
      const nt = model.notices[i];
      if (nt.t0 > t) break;
      if (t >= nt.t1) continue;
      if (nt.side === 'truth' ? truthOn : agentOn) act.push(i);
    }
    const key = act.join(',');
    if (key === noticeKey) return;
    noticeKey = key;
    noticeBox.textContent = '';
    for (const i of act.slice(-3)) {
      const nt = model.notices[i];
      const chip = h('div', `rp-notice rp-notice-${nt.tone}`);
      if (nt.tone === 'warn') chip.appendChild(icon(ICO.warn));
      chip.appendChild(h('span', null, nt.text));
      noticeBox.appendChild(chip);
    }
  }

  buildLut();
  const ro = new ResizeObserver(() => { if (layout()) render(lastT); });
  ro.observe(wrap);

  return {
    el: wrap,
    render,
    relayout() { floorPath = null; if (layout()) render(lastT); },
    setModel(m) { model = m; heatKey = ''; soilKey = ''; noticeKey = '\u0000'; },
    setLayers(l) { layers = l; noticeKey = '\u0000'; render(lastT); },
    setTheme(th) { theme = th; buildLut(); floorPath = null; if (layout()) render(lastT); },
    destroy() { ro.disconnect(); wrap.remove(); },
  };
}

/* ============================================================================================
 * Переключатель режимов показа и легенда-переключатели слоёв
 * ========================================================================================== */

/** Где шёл прогон (быстрый симулятор или Gazebo) и что это за прогон. */
function backendBadge(meta) {
  if (!meta || !meta.backend) return null;
  const known = BACKEND_RU[meta.backend];
  const b = h('span', `rp-backend rp-backend-${known ? meta.backend : 'other'}`, known || String(meta.backend));
  b.title = BACKEND_TITLE[meta.backend] || `Где шёл прогон: ${meta.backend}`;
  return b;
}

function createRunInfo(host, mode) {
  const root = h('div', `rp-runinfo rp-runinfo-${mode}`);
  host.appendChild(root);
  let key = null;
  return {
    el: root,
    sync(model) {
      const mt = model.meta || {};
      const k = [mt.backend, mt.agent, mt.arm, mt.level, mt.seed].join('|');
      if (k === key) return;
      key = k;
      root.textContent = '';
      const badge = backendBadge(mt);
      if (mode === 'full') {
        if (badge) root.append(h('span', 'rp-runinfo-label', 'Где шёл прогон'), badge);
        const facts = [];
        if (mt.agent) facts.push(AGENT_RU[mt.agent] || `агент ${mt.agent}`);
        const lv = [];
        if (mt.level) lv.push(LEVEL_RU[mt.level] || `уровень ${mt.level}`);
        if (mt.seed != null) lv.push(`сценарий ${mt.seed}`);
        if (lv.length) facts.push(lv.join(', '));
        if (facts.length) {
          const f = h('span', 'rp-runinfo-facts', facts.join(' · '));
          if (mt.id) f.title = `Запись ${mt.id}`;
          root.appendChild(f);
        }
      } else if (badge) root.appendChild(badge);
      root.hidden = !root.childNodes.length;
    },
    destroy() { root.remove(); },
  };
}

function createViewSwitch(host, api, small) {
  const box = h('div', `rp-seg${small ? ' rp-seg-small' : ''}`);
  box.setAttribute('role', 'group');
  box.setAttribute('aria-label', 'Что показывать на арене');
  const btns = VIEWS.map((v) => {
    const b = button('rp-seg-btn', small && v.id === 'agent' ? 'Агент' : v.label);
    b.title = v.title;
    b.addEventListener('click', () => api.setView(v.id));
    box.appendChild(b);
    return b;
  });
  host.appendChild(box);
  return {
    el: box,
    sync(layers) {
      const cur = viewOfLayers(layers);
      VIEWS.forEach((v, i) => btns[i].setAttribute('aria-pressed', String(v.id === cur)));
    },
  };
}

function createLegend(host, api) {
  // Три колонки: истина | догадки агента | робот. Строки первых двух колонок — одна тема
  // (образцы, грунт, опасные зоны), чтобы истина и догадка стояли рядом.
  const root = h('div', 'rp-legend');
  const chips = {};
  const chip = (it) => {
    const b = button('rp-chip');
    b.appendChild(icon(it.sw, 'rp-sw'));
    b.appendChild(h('span', null, it.label));
    b.title = it.title;
    b.addEventListener('click', () => api.toggleLayer(it.key));
    chips[it.key] = b;
    return b;
  };
  LAYER_GROUPS.forEach((g, gi) => {
    const title = h('div', `rp-legend-title${g.id === 'robot' ? ' rp-legend-title-robot' : ''}`, g.title);
    title.style.gridColumn = String(gi + 1);
    root.appendChild(title);
    if (g.id === 'robot') {
      const box = h('div', 'rp-legend-robot');
      for (const it of g.items) box.appendChild(chip(it));
      const note = h('span', 'rp-legend-note');
      note.appendChild(icon('<svg viewBox="0 0 20 14"><circle cx="10" cy="7" r="5.6" fill="rgba(20,24,28,.06)" stroke="var(--rp-ink-2)" stroke-width="1.2"/><circle cx="10" cy="7" r="2.2" fill="var(--rp-ink)"/></svg>', 'rp-sw'));
      note.appendChild(h('span', null, `круг сбора ${num(api.model().rules.collect_radius_m, 1)} м`));
      note.title = 'Образец можно взять, только когда он внутри этого круга';
      box.appendChild(note);
      root.appendChild(box);
      return;
    }
    g.items.forEach((it, ri) => {
      const cell = h('div', 'rp-legend-cell');
      cell.style.gridColumn = String(gi + 1);
      cell.style.gridRow = String(ri + 2);
      const b = chip(it);
      if (it.key === 'truthSoil') {
        const scale = h('span', 'rp-soil-scale');
        for (const m of [2, 3, 4]) {
          const sw = h('i', 'rp-soil-sw');
          sw.style.background = `var(--rp-soil-${m})`;
          scale.appendChild(sw);
        }
        b.appendChild(scale);
        b.title = `${it.title}: цвет — во сколько раз (×2, ×3, ×4)`;
      }
      cell.appendChild(b);
      root.appendChild(cell);
    });
  });
  host.appendChild(root);
  return {
    el: root,
    sync(layers, model) {
      for (const key in chips) chips[key].setAttribute('aria-pressed', String(!!layers[key]));
      const noSoil = !model.soil;
      chips.beliefSoil.classList.toggle('rp-chip-na', noSoil);
      chips.beliefSoil.title = noSoil ? 'У этого агента нет карты грунта: он его не изучает' : LAYER_GROUPS[1].items[1].title;
      const noHaz = !model.memHazards.length;
      chips.beliefHazards.classList.toggle('rp-chip-na', noHaz);
      chips.beliefHazards.title = noHaz ? 'В этом прогоне агент не запомнил ни одной опасной зоны' : LAYER_GROUPS[1].items[2].title;
    },
  };
}

/* ============================================================================================
 * Шкала времени
 * ========================================================================================== */

const MARK_ICON = { world: ICO.mWorld, sample: ICO.mSample, penalty: ICO.mPenalty, alarm: ICO.mAlarm, plan: ICO.mPlan, inquiry: ICO.mInquiry };
const MARK_RANK = { world: 5, penalty: 4, alarm: 3, inquiry: 3, sample: 2, plan: 1 };

function niceStep(span, maxTicks) {
  const steps = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600];
  for (const st of steps) if (span / st <= maxTicks) return st;
  return steps[steps.length - 1];
}

function createTimeline(host, api) {
  // api: clock, lanes() -> [{label, title, marks, lags}], range() -> [начало, конец], seekUser(t), compact, follow
  const clock = api.clock;
  const root = h('div', 'rp-tl');
  const bar = h('div', 'rp-tl-bar');
  const playBtn = button('rp-btn rp-play');
  const playIco = icon(ICO.play);
  const playTxt = h('span', null, 'Пуск');
  playBtn.append(playIco, playTxt);
  playBtn.addEventListener('click', () => api.toggle());
  const speeds = h('div', 'rp-seg rp-seg-small');
  speeds.setAttribute('role', 'group');
  speeds.setAttribute('aria-label', 'Скорость');
  const speedBtns = [1, 4, 16].map((sp) => {
    const b = button('rp-seg-btn', `${sp}×`);
    b.title = `Скорость ${sp}×`;
    b.addEventListener('click', () => clock.setSpeed(sp));
    speeds.appendChild(b);
    return [sp, b];
  });
  const time = h('div', 'rp-tl-time');
  const timeNow = h('b', null, '0,0');
  const timeAll = h('span', null, '');
  time.append(timeNow, timeAll);
  const live = button('rp-live');
  live.hidden = !api.follow;
  live.addEventListener('click', () => api.goLive());
  const hint = h('div', 'rp-tl-hint', 'пробел — пуск · ← → — шаг · Shift — к событию');
  hint.title = 'Клавиши: пробел — пуск и пауза, стрелки — на секунду назад и вперёд, Shift со стрелкой — к предыдущему или следующему событию';
  bar.append(playBtn, speeds, time, live, hint);

  const body = h('div', 'rp-tl-body');
  const overlay = h('div', 'rp-tl-overlay');
  const head = h('div', 'rp-tl-head');
  overlay.appendChild(head);
  const scrubLabel = h('span', 'rp-tl-label', 'время, с');
  const scrub = h('div', 'rp-tl-scrub');
  scrub.tabIndex = 0;
  scrub.setAttribute('role', 'slider');
  scrub.setAttribute('aria-label', 'Момент прогона, секунды');
  scrub.setAttribute('aria-valuemin', '0');
  const rail = h('div', 'rp-tl-rail');
  const fill = h('div', 'rp-tl-fill');
  rail.appendChild(fill);
  const thumb = h('div', 'rp-tl-thumb');
  const ticks = h('div', 'rp-tl-ticks');
  scrub.append(rail, thumb, ticks);
  const tip = h('div', 'rp-tip');
  tip.hidden = true;
  root.append(bar, body, tip);
  host.appendChild(root);

  let t0 = 0, dur = 1, span = 1;
  const pos = (t) => `${(clamp((t - t0) / span, 0, 1) * 100).toFixed(3)}%`;
  let groups = [];        // все метки для подсказок по Shift+стрелке
  let tipTimer = 0;

  function showTip(group, anchor) {
    tip.textContent = '';
    tip.appendChild(h('div', 'rp-tip-time', sec(group.t)));
    for (const it of group.items) {
      const line = h('div', 'rp-tip-line');
      line.appendChild(icon(MARK_ICON[it.kind] || ICO.mPlan, 'rp-tip-ico'));
      const txt = h('div', 'rp-tip-text', it.text);
      if (it.sub) txt.appendChild(h('div', 'rp-tip-sub', it.sub));
      line.appendChild(txt);
      tip.appendChild(line);
    }
    tip.hidden = false;
    const rr = root.getBoundingClientRect();
    const ar = anchor.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const left = clamp(ar.left + ar.width / 2 - rr.left - tw / 2, 0, Math.max(0, rr.width - tw));
    tip.style.left = `${left}px`;
    tip.style.top = `${ar.top - rr.top - tip.offsetHeight - 6}px`;
  }
  function hideTip() { tip.hidden = true; clearTimeout(tipTimer); }

  function build() {
    [t0, dur] = api.range();
    span = Math.max(dur - t0, 0.001);
    const lanes = api.lanes();
    body.textContent = '';
    groups = [];
    body.style.gridTemplateRows = `repeat(${lanes.length + 1}, auto)`;
    const thr = span * 0.009;
    lanes.forEach((lane, li) => {
      const lab = h('span', 'rp-tl-label', lane.label);
      if (lane.title) lab.title = lane.title;
      lab.style.gridRow = String(li + 1);
      const track = h('div', 'rp-tl-lane');
      track.style.gridRow = String(li + 1);
      for (const lg of lane.lags || []) {
        const bar2 = h('i', `rp-tl-lag${lane.lagClass ? ` ${lane.lagClass}` : ''}`);
        bar2.style.left = pos(lg.t0);
        bar2.style.width = `${(clamp((Math.min(lg.t1, dur) - Math.max(lg.t0, t0)) / span, 0, 1) * 100).toFixed(3)}%`;
        track.appendChild(bar2);
      }
      // близкие метки сливаются в одну с общей подсказкой
      const sorted = lane.marks.slice().sort(byT);
      const merged = [];
      for (const mk of sorted) {
        const lastG = merged[merged.length - 1];
        const strong = mk.kind !== 'plan';
        if (lastG && mk.t - lastG.t0 <= thr && (lastG.strong === strong || !strong || !lastG.strong)) {
          lastG.items.push(mk);
          if (MARK_RANK[mk.kind] > MARK_RANK[lastG.kind]) { lastG.kind = mk.kind; lastG.t = mk.t; }
          lastG.strong = lastG.strong || strong;
        } else merged.push({ t0: mk.t, t: mk.t, kind: mk.kind, items: [mk], strong });
      }
      for (const g of merged) {
        const b = button(`rp-mark rp-mark-${g.kind}`);
        b.tabIndex = -1;
        b.style.left = pos(g.t);
        b.setAttribute('aria-label', `${sec(g.t)}: ${g.items.map((it) => it.text).join('; ')}`);
        b.appendChild(icon(MARK_ICON[g.kind] || ICO.mPlan, 'rp-mark-ico'));
        if (g.items.filter((it) => it.kind !== 'plan').length > 1) b.appendChild(h('span', 'rp-mark-n', String(g.items.filter((it) => it.kind !== 'plan').length)));
        b.addEventListener('pointerenter', () => showTip(g, b));
        b.addEventListener('pointerleave', hideTip);
        b.addEventListener('click', (e) => { e.stopPropagation(); api.seekUser(g.t); });
        g.el = b;
        groups.push(g);
        track.appendChild(b);
      }
      body.append(lab, track);
    });
    scrubLabel.style.gridRow = String(lanes.length + 1);
    scrub.style.gridRow = String(lanes.length + 1);
    overlay.style.gridRow = `1 / ${lanes.length + 2}`;
    body.append(scrubLabel, scrub, overlay);
    // деления
    ticks.textContent = '';
    const st = niceStep(span, api.compact ? 5 : 9);
    for (let v = Math.ceil((t0 - 1e-6) / st) * st; v <= dur + 1e-6; v += st) {
      const tk = h('span', 'rp-tl-tick', num(v, 0));
      tk.style.left = pos(v);
      ticks.appendChild(tk);
    }
    scrub.setAttribute('aria-valuemin', t0.toFixed(1));
    scrub.setAttribute('aria-valuemax', dur.toFixed(1));
    timeAll.textContent = ` / ${sec(dur)}`;
    update(clock.t);
  }

  let lastPct = -1;
  let lastTxt = '';
  function update(t) {
    const pct = clamp((t - t0) / span, 0, 1) * 100;
    if (Math.abs(pct - lastPct) > 0.005) {
      lastPct = pct;
      const p = `${pct.toFixed(3)}%`;
      fill.style.width = p;
      thumb.style.left = p;
      head.style.left = p;
    }
    const txt = num(t, 1);
    if (txt !== lastTxt) {
      lastTxt = txt;
      timeNow.textContent = txt;
      scrub.setAttribute('aria-valuenow', t.toFixed(1));
      scrub.setAttribute('aria-valuetext', sec(t));
    }
  }

  function syncState() {
    playBtn.classList.toggle('rp-playing', clock.playing);
    playIco.innerHTML = clock.playing ? ICO.pause : ICO.play;
    playTxt.textContent = clock.playing ? 'Пауза' : 'Пуск';
    playBtn.setAttribute('aria-label', clock.playing ? 'Пауза (пробел)' : 'Пуск (пробел)');
    for (const [sp, b] of speedBtns) b.setAttribute('aria-pressed', String(Math.abs(clock.speed - sp) < 1e-6));
  }

  // перетаскивание
  const timeAt = (clientX) => {
    const r = scrub.getBoundingClientRect();
    return t0 + clamp((clientX - r.left) / Math.max(1, r.width), 0, 1) * span;
  };
  let dragging = false;
  body.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || e.target.closest('.rp-mark') || e.target.closest('.rp-tl-label')) return;
    dragging = true;
    clock.hold(true);
    try { body.setPointerCapture(e.pointerId); } catch (err) { /* не страшно */ }
    root.classList.add('rp-dragging');
    hideTip();
    api.seekUser(timeAt(e.clientX));
    e.preventDefault();
    scrub.focus({ preventScroll: true });
  });
  body.addEventListener('pointermove', (e) => { if (dragging) api.seekUser(timeAt(e.clientX)); });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    clock.hold(false);
    root.classList.remove('rp-dragging');
  };
  body.addEventListener('pointerup', endDrag);
  body.addEventListener('pointercancel', endDrag);

  const offState = clock.onState(syncState);
  syncState();

  return {
    el: root,
    build,
    update,
    setLive(isLive, following) {
      live.hidden = !isLive;
      live.classList.toggle('rp-live-on', !!following);
      live.textContent = following ? 'идёт прогон' : 'к текущему моменту';
      live.title = following ? 'Показывается последний записанный момент' : 'Вернуться к последнему записанному моменту';
    },
    /** Показать подсказку метки, на которую встал курсор (после Shift+стрелки). */
    flashAt(t) {
      const g = groups.filter((x) => x.items.some((it) => Math.abs(it.t - t) < 0.051) && x.kind !== 'plan');
      if (!g.length) return;
      const all = { t, items: [].concat(...g.map((x) => x.items.filter((it) => Math.abs(it.t - t) < 0.051))) };
      showTip(all, g[0].el);
      clearTimeout(tipTimer);
      tipTimer = setTimeout(hideTip, 3200);
    },
    destroy() { offState(); clearTimeout(tipTimer); root.remove(); },
  };
}

/* ============================================================================================
 * Состояние сейчас
 * ========================================================================================== */

function setText(node, text) {
  if (node.__t !== text) { node.__t = text; node.textContent = text; }
}

function batteryTone(frac) {
  if (frac <= 0.15) return 'bad';
  if (frac <= 0.35) return 'warn';
  return 'ok';
}

function createStatus(host, api) {
  const root = h('section', 'rp-status');
  root.setAttribute('aria-label', 'Состояние сейчас');
  const tile = (label, cls) => {
    const el = h('div', `rp-tile ${cls || ''}`);
    el.appendChild(h('div', 'rp-tile-label', label));
    const value = h('div', 'rp-tile-value');
    const sub = h('div', 'rp-tile-sub');
    el.append(value, sub);
    root.appendChild(el);
    return { el, value, sub };
  };
  const meter = () => {
    const m = h('div', 'rp-meter');
    const f = h('i');
    m.appendChild(f);
    return { el: m, fill: f };
  };
  const tTime = tile('Время');
  const tBat = tile('Заряд');
  const mBat = meter();
  const batNote = h('span', 'rp-tile-note');
  tBat.sub.append(mBat.el, batNote);
  const tCol = tile('Собрано');
  const pips = h('div', 'rp-pips');
  tCol.sub.appendChild(pips);
  const tMode = tile('Режим', 'rp-tile-mode');
  const tSen = tile('Датчик');
  const mSen = meter();
  mSen.el.classList.add('rp-meter-blue');
  const senNote = h('span', 'rp-tile-note');
  tSen.sub.append(mSen.el, senNote);
  const tScore = tile('Счёт');
  const outcome = h('div', 'rp-outcome');
  host.append(root, outcome);

  let pipKey = '';
  let outKey = '';
  function update(t, st) {
    const m = api.model();
    setText(tTime.value, sec(t));
    setText(tTime.sub, `из ${sec(m.dur)}`);
    const frac = clamp(st.battery / (m.rules.battery_start || 1), 0, 1);
    setText(tBat.value, num(st.battery, 1));
    mBat.fill.style.width = `${(frac * 100).toFixed(1)}%`;
    const tone = batteryTone(frac);
    if (mBat.el.dataset.tone !== tone) mBat.el.dataset.tone = tone;
    const leak = m.leakAt(t);
    setText(batNote, leak ? 'утечка' : '');
    batNote.classList.toggle('rp-tile-note-warn', !!leak);
    const got = m.collectedAt(t);
    setText(tCol.value, `${got} из ${m.total}`);
    const pk = `${got}/${m.total}`;
    if (pk !== pipKey) {
      pipKey = pk;
      pips.textContent = '';
      for (let i = 0; i < m.total; i++) pips.appendChild(h('i', i < got ? 'rp-pip rp-pip-on' : 'rp-pip'));
    }
    setText(tMode.value, MODE_RU[st.mode] || st.mode);
    const fault = m.faultAt(t);
    setText(tSen.value, num(st.sensor, 2));
    mSen.fill.style.width = `${(clamp(st.sensor, 0, 1) * 100).toFixed(1)}%`;
    const note = fault ? 'сбой' : st.sensor <= m.quiet ? 'рядом пусто' : '';
    setText(senNote, note);
    senNote.classList.toggle('rp-tile-note-warn', !!fault);
    const score = m.scoreAt(t);
    const final = m.finished && m.result.score != null ? +m.result.score : null;
    const liveScore = !m.finished && m.result && m.result.score != null && t >= m.dur - 0.3 ? +m.result.score : null;
    setText(tScore.value, liveScore != null ? num(liveScore, 1) : score != null ? num(score, 1) : final != null ? num(final, 1) : '—');
    setText(tScore.sub, final != null && score != null && Math.abs(score - final) > 0.049 ? `итог ${num(final, 1)}` : final != null ? 'итог прогона' : api.follow ? 'прогон идёт' : 'итога в записи нет');

    const r = m.finished ? m.result : null;
    const ok = outcomeOf(r);
    const key = r ? `${r.score}|${r.reason}|${r.returned}|${r.samples_collected}` : `live:${api.follow}`;
    if (key !== outKey) {
      outKey = key;
      outcome.textContent = '';
      outcome.appendChild(h('span', 'rp-outcome-title', 'Итог прогона'));
      if (!r || !ok) {
        outcome.appendChild(h('span', 'rp-outcome-text', api.follow ? 'прогон ещё идёт' : 'в записи итога нет: прогон не дошёл до конца или судья ещё не подвёл итог'));
      } else {
        outcome.appendChild(h('span', `rp-badge rp-badge-${ok.tone}`, ok.text));
        const facts = [
          `счёт ${num(r.score, 1)}`,
          `собрано ${r.samples_collected} из ${r.samples_total}`,
          `осталось заряда ${num(r.battery_left != null ? r.battery_left : r.battery, 1)}`,
        ];
        const pen = (r.collisions || 0) + (r.false_collects || 0) + (r.hazard_hits || 0);
        facts.push(pen ? `${pen} ${plural(pen, 'штраф', 'штрафа', 'штрафов')}` : 'без штрафов');
        const iq = r.inquiries && typeof r.inquiries === 'object' ? r.inquiries : null;
        if (iq && fin(iq.total)) {
          const bits = [];
          if (fin(iq.identified)) bits.push(`причина найдена в ${iq.identified}`);
          if (fin(iq.insufficient)) bits.push(`данных не хватило в ${iq.insufficient}`);
          facts.push(`${iq.total} ${plural(iq.total, 'расследование', 'расследования', 'расследований')}${bits.length ? ` (${bits.join(', ')})` : ''}`);
        }
        outcome.appendChild(h('span', 'rp-outcome-text', facts.join(' · ')));
      }
    }
  }
  return { update, destroy() { root.remove(); outcome.remove(); } };
}

/** Одна строка состояния: для компактного режима и сравнения. */
function createStatusLine(host, api) {
  const root = h('div', 'rp-statusline');
  const cell = (label) => {
    const c = h('span', 'rp-sl-cell');
    c.appendChild(h('span', 'rp-sl-label', label));
    const v = h('b');
    c.appendChild(v);
    root.appendChild(c);
    return v;
  };
  const vTime = cell('время');
  const vBat = cell('заряд');
  const vCol = cell('собрано');
  const vMode = cell('режим');
  const vSen = cell('датчик');
  const res = h('span', 'rp-sl-result');
  root.appendChild(res);
  host.appendChild(root);
  let outKey = '';
  return {
    el: root,
    update(t, st) {
      const m = api.model();
      setText(vTime, sec(Math.min(t, m.dur)));
      setText(vBat, num(st.battery, 1));
      const tone = batteryTone(clamp(st.battery / (m.rules.battery_start || 1), 0, 1));
      if (vBat.dataset.tone !== tone) vBat.dataset.tone = tone;
      setText(vCol, `${m.collectedAt(t)} из ${m.total}`);
      setText(vMode, MODE_RU[st.mode] || st.mode);
      setText(vSen, num(st.sensor, 2) + (m.faultAt(t) ? ' — сбой' : ''));
      const r = m.finished ? m.result : null;
      const ok = outcomeOf(r);
      const key = r ? `${r.score}|${r.reason}|${r.returned}` : 'live';
      if (key !== outKey) {
        outKey = key;
        res.textContent = '';
        if (r && ok) {
          res.appendChild(h('span', 'rp-sl-label', 'итог'));
          res.appendChild(h('b', null, num(r.score, 1)));
          res.appendChild(h('span', `rp-badge rp-badge-${ok.tone}`, ok.text));
        }
      }
    },
    destroy() { root.remove(); },
  };
}

/* ============================================================================================
 * Графики: заряд и датчик, общий курсор
 * ========================================================================================== */

function createCharts(host, api) {
  const root = h('section', 'rp-charts');
  root.setAttribute('aria-label', 'Графики по времени');
  const tip = h('div', 'rp-tip rp-chart-tip');
  tip.hidden = true;
  const P = { l: 34, r: 12, t: 8, plot: 84, axis: 20 };
  // Общая ось времени у всех графиков. Третий — поправка положения по лидару — есть только там,
  // где одометрия расходилась с картой (прогоны в Gazebo); у каждого графика своя ось значений.
  const charts = [
    { key: 'battery', title: 'Заряд' },
    { key: 'sensor', title: 'Датчик образцов' },
    { key: 'pose', title: 'Поправка положения по лидару, см', hint: 'На сколько сантиметров лидар сдвинул положение робота: так сильно одометрия разошлась с картой' },
  ];
  for (const c of charts) {
    c.fig = h('figure', `rp-chart rp-chart-${c.key}`);
    const head = h('figcaption', 'rp-chart-head');
    const title = h('span', 'rp-chart-title', c.title);
    if (c.hint) title.title = c.hint;
    head.appendChild(title);
    c.keyBox = h('span', 'rp-chart-key');
    c.now = h('span', 'rp-chart-now');
    head.append(c.keyBox, c.now);
    c.svg = s('svg', { class: 'rp-chart-svg', role: 'img' });
    c.fig.append(head, c.svg);
    root.appendChild(c.fig);
  }
  const keyItem = (box, swatchCls, text) => {
    const it = h('span', 'rp-key');
    it.append(h('i', swatchCls), h('span', null, text));
    box.appendChild(it);
  };
  root.appendChild(tip);
  host.appendChild(root);

  let W = 0, t0 = 0, dur = 1, span = 1, plotW = 1;
  let shown = [];
  const xOf = (t) => P.l + ((clamp(t, t0, dur) - t0) / span) * plotW;
  /** Ряд графика: времена и значения. */
  const seriesOf = (c, m) => (c.key === 'pose'
    ? (m.poseFix ? { T: m.poseFix.T, V: m.poseFix.S, n: m.poseFix.T.length } : { T: [], V: [], n: 0 })
    : { T: m.T, V: m.tr[c.key] || [], n: m.n });

  function build() {
    const m = api.model();
    W = Math.floor(root.clientWidth);
    if (!W) return;
    P.plot = W < 470 ? 70 : 84;                 // в узкой колонке графики ниже, чтобы всё помещалось на экран
    t0 = m.t0;
    dur = Math.max(m.dur, t0 + 0.001);
    span = dur - t0;
    plotW = Math.max(10, W - P.l - P.r);
    shown = charts.filter((c) => c.key !== 'pose' || m.poseFix);
    for (const c of charts) c.fig.hidden = !shown.includes(c);
    // пояснения к полосам и отметкам — только то, что в этом прогоне есть
    charts[0].keyBox.textContent = '';
    keyItem(charts[0].keyBox, 'rp-key-soil', 'робот на дорогом грунте');
    if (m.leaks.length) keyItem(charts[0].keyBox, 'rp-key-fault', 'утечка заряда');
    keyItem(charts[0].keyBox, 'rp-key-pen', 'штраф');
    charts[1].keyBox.textContent = '';
    keyItem(charts[1].keyBox, 'rp-key-fault', 'сбой датчика');
    keyItem(charts[1].keyBox, 'rp-key-sample', 'образец собран');
    for (const c of shown) {
      const axis = c === shown[shown.length - 1];
      const plot = c.key === 'pose' ? Math.round(P.plot * 0.66) : P.plot;
      const H = P.t + plot + (axis ? P.axis : 6);
      c.H = H;
      c.plot = plot;
      c.svg.textContent = '';
      c.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      c.svg.setAttribute('width', W);
      c.svg.setAttribute('height', H);
      const ser = seriesOf(c, m);
      const data = ser.V;
      let yMax = 1;
      let yt = [0, 0.5, 1];
      if (c.key === 'battery') {
        yMax = Math.max(m.rules.battery_start || 0, data.length ? Math.max.apply(null, data) : 0, 1);
        const stp = yMax <= 12 ? 5 : yMax <= 30 ? 10 : yMax <= 80 ? 20 : 50;
        yt = [];
        for (let v = 0; v <= yMax + 1e-6; v += stp) yt.push(v);
      } else if (c.key === 'pose') {
        const need = Math.max(m.poseFix.max * 1.08, 1);
        yMax = [2, 5, 10, 20, 50, 100, 200, 500].find((v) => v >= need) || Math.ceil(need / 100) * 100;
        yt = [0, yMax / 2, yMax];
      }
      c.yOf = (v) => P.t + plot - (clamp(v, 0, yMax) / yMax) * plot;
      c.yMax = yMax;

      // фон: полосы
      const bands = s('g', { class: 'rp-ch-bands' }, c.svg);
      if (c.key === 'battery') {
        for (const b of m.soilBands) {
          const x0 = xOf(b.t0);
          const w = Math.max(1.5, xOf(b.t1) - x0);
          s('rect', { x: x0.toFixed(1), y: P.t, width: w.toFixed(1), height: plot, fill: soilCss(b.mult), opacity: 0.34 }, bands);
          if (w >= 24) s('text', { x: (x0 + 3).toFixed(1), y: P.t + 11, class: 'rp-ch-bandlabel' }, bands).textContent = mult(b.mult);
        }
        for (const lk of m.leaks) {
          const x0 = xOf(lk.t0);
          const w = Math.max(1.5, xOf(Math.min(lk.t1, dur)) - x0);
          s('rect', { x: x0.toFixed(1), y: P.t, width: w.toFixed(1), height: plot, class: 'rp-ch-fault' }, bands);
          if (w >= 50) s('text', { x: (x0 + 4).toFixed(1), y: P.t + plot - 5, class: 'rp-ch-bandlabel' }, bands).textContent = 'утечка';
        }
      } else if (c.key === 'sensor') {
        // сбои могут накладываться (по расписанию сценария и после штрафа) — рисуем их одной полосой
        const merged = [];
        for (const f of m.faults.slice().sort((p, q) => p.t0 - q.t0)) {
          const last = merged[merged.length - 1];
          if (last && f.t0 <= last.t1 + 0.05) last.t1 = Math.max(last.t1, f.t1);
          else merged.push({ t0: f.t0, t1: f.t1 });
        }
        for (const f of merged) {
          const x0 = xOf(f.t0);
          const w = Math.max(1.5, xOf(Math.min(f.t1, dur)) - x0);
          s('rect', { x: x0.toFixed(1), y: P.t, width: w.toFixed(1), height: plot, class: 'rp-ch-fault' }, bands);
          if (w >= 78) s('text', { x: (x0 + 4).toFixed(1), y: P.t + 11, class: 'rp-ch-bandlabel' }, bands).textContent = 'сбой датчика';
        }
      }
      // сетка и подписи оси Y
      const grid = s('g', { class: 'rp-ch-grid' }, c.svg);
      for (const v of yt) {
        const y = Math.round(c.yOf(v)) + 0.5;
        s('line', { x1: P.l, x2: W - P.r, y1: y, y2: y, class: v === 0 ? 'rp-ch-base' : 'rp-ch-gridline' }, grid);
        s('text', { x: P.l - 6, y: y + 4, class: 'rp-ch-ytick' }, grid).textContent = c.key === 'sensor' ? num(v, v === 0.5 ? 1 : 0) : num(v, Number.isInteger(v) ? 0 : 1);
      }
      if (axis) {
        const stp = niceStep(span, Math.max(3, Math.floor(plotW / 62)));
        for (let v = Math.ceil((t0 - 1e-6) / stp) * stp; v <= dur + 1e-6; v += stp) {
          const x = xOf(v);
          const txt = s('text', { x: x.toFixed(1), y: P.t + plot + 15, class: 'rp-ch-xtick' }, grid);
          txt.textContent = v + stp > dur + 1e-6 ? `${num(v, 0)} с` : num(v, 0);
        }
      }
      // линия: будущее бледное, прошедшее яркое
      let d = '';
      for (let i = 0; i < ser.n; i++) d += `${i ? 'L' : 'M'}${xOf(ser.T[i]).toFixed(1)} ${c.yOf(data[i] || 0).toFixed(1)}`;
      const clipId = uid('rp-clip');
      const cp = s('clipPath', { id: clipId }, c.svg);
      c.clip = s('rect', { x: 0, y: 0, width: 0, height: H }, cp);
      if (d) {
        s('path', { d, class: 'rp-ch-line rp-ch-future' }, c.svg);
        s('path', { d, class: 'rp-ch-line', 'clip-path': `url(#${clipId})` }, c.svg);
      }
      // отметки
      const marks = s('g', { class: 'rp-ch-marks' }, c.svg);
      if (c.key === 'battery') {
        for (const p of m.penalties) {
          const st = m.stateAt(p.t);
          const g = s('g', { transform: `translate(${xOf(p.t).toFixed(1)} ${c.yOf(st.battery).toFixed(1)})` }, marks);
          s('circle', { r: 7.5, class: 'rp-ch-ring' }, g);
          s('circle', { r: 5.5, class: 'rp-ch-pen' }, g);
          s('path', { d: 'M0-2.8v3M0 2.6v.1', class: 'rp-ch-penmark' }, g);
        }
      } else if (c.key === 'sensor') {
        for (const e of m.events) {
          if (e.type !== 'sample_collected') continue;
          const x = xOf(e.t);
          const y = P.t + 6;
          s('path', { d: `M${x.toFixed(1)} ${y - 5.5}l5.5 5.5-5.5 5.5-5.5-5.5z`, class: 'rp-ch-sample' }, marks);
        }
      }
      c.hover = s('line', { y1: P.t, y2: P.t + plot, class: 'rp-ch-hover', visibility: 'hidden' }, c.svg);
      c.cursor = s('line', { y1: P.t - 2, y2: P.t + plot, class: 'rp-ch-cursor' }, c.svg);
      c.dot = s('circle', { r: 4.5, class: 'rp-ch-dot' }, c.svg);
      const first = data.length ? data[0] : 0;
      const lastV = data.length ? data[data.length - 1] : 0;
      c.svg.setAttribute('aria-label', c.key === 'battery'
        ? `Заряд по времени: в начале ${num(first, 1)}, в конце ${num(lastV, 1)}. Точные значения — в блоке «Состояние сейчас».`
        : c.key === 'sensor' ? 'Показание датчика образцов по времени, от 0 до 1. Точные значения — в блоке «Состояние сейчас».'
          : `Поправка положения по лидару по времени, в сантиметрах: наибольшая ${num(m.poseFix.max, 1)}, в конце ${num(lastV, 1)}.`);
    }
    lastX = -1;
    setCursor(api.clock.t, m.stateAt(api.clock.t));
  }

  let lastX = -1;
  function setCursor(t, st) {
    if (!W) return;
    const m = api.model();
    const x = xOf(t);
    if (Math.abs(x - lastX) > 0.05) {
      lastX = x;
      const xs = x.toFixed(1);
      for (const c of shown) {
        if (!c.cursor) continue;
        c.cursor.setAttribute('x1', xs);
        c.cursor.setAttribute('x2', xs);
        c.dot.setAttribute('cx', xs);
        c.clip.setAttribute('width', xs);
      }
    }
    if (charts[0].dot) {
      charts[0].dot.setAttribute('cy', charts[0].yOf(st.battery).toFixed(1));
      charts[1].dot.setAttribute('cy', charts[1].yOf(st.sensor).toFixed(1));
    }
    setText(charts[0].now, `${num(st.battery, 1)} ед.`);
    setText(charts[1].now, num(st.sensor, 2));
    const pc = charts[2];
    if (shown.includes(pc) && pc.dot) {
      const pf = m.poseAt(t);
      pc.dot.setAttribute('visibility', pf ? 'visible' : 'hidden');
      if (pf) pc.dot.setAttribute('cy', pc.yOf(pf.shift).toFixed(1));
      setText(pc.now, pf ? `${num(pf.shift, 1)} см · поворот ${num(pf.ang, 1)}°` : 'поправок ещё не было');
    }
  }

  // наведение: вертикаль и подсказка со всеми значениями
  function hoverAt(e, c) {
    const m = api.model();
    if (!m.n || !W) return null;
    const r = c.svg.getBoundingClientRect();
    const t = t0 + clamp((e.clientX - r.left - P.l) / plotW, 0, 1) * span;
    let i = upperBound(m.T, t) - 1;
    if (i < 0) i = 0;
    if (i + 1 < m.n && m.T[i + 1] - t < t - m.T[i]) i += 1;
    return { i, t: m.T[i] };
  }
  function showHover(e, c) {
    const hv = hoverAt(e, c);
    if (!hv) return;
    const m = api.model();
    const x = xOf(hv.t).toFixed(1);
    for (const cc of shown) {
      cc.hover.setAttribute('x1', x);
      cc.hover.setAttribute('x2', x);
      cc.hover.setAttribute('visibility', 'visible');
    }
    tip.textContent = '';
    tip.appendChild(h('div', 'rp-tip-time', sec(hv.t)));
    const row = (keyCls, value, name) => {
      const line = h('div', 'rp-tip-row');
      line.append(h('i', `rp-tip-key ${keyCls}`), h('b', null, value), h('span', null, name));
      tip.appendChild(line);
    };
    row('rp-tip-key-battery', num(m.tr.battery[hv.i], 1), 'заряд, ед.');
    row('rp-tip-key-sensor', num(m.tr.sensor[hv.i], 2), 'датчик образцов');
    const pf = m.poseAt(hv.t);
    if (pf) row('rp-tip-key-pose', num(pf.shift, 1), `поправка положения, см (поворот ${num(pf.ang, 1)}°)`);
    const notes = [];
    if (m.trackMult[hv.i] > 1.01) notes.push(`робот на дорогом грунте ${mult(m.trackMult[hv.i])}`);
    if (m.faultAt(hv.t)) notes.push('сбой датчика образцов');
    if (m.leakAt(hv.t)) notes.push('утечка заряда (сбой батареи)');
    const near = span * 0.012;
    for (const p of m.penalties) if (Math.abs(p.t - hv.t) <= near) notes.push(p.text);
    for (const ev of m.events) if (ev.type === 'sample_collected' && Math.abs(ev.t - hv.t) <= near) notes.push('образец собран');
    for (const nt of notes) tip.appendChild(h('div', 'rp-tip-sub', nt));
    tip.hidden = false;
    const rr = root.getBoundingClientRect();
    const cr = c.svg.getBoundingClientRect();
    const px = e.clientX - rr.left;
    const tw = tip.offsetWidth;
    tip.style.left = `${clamp(px + 14 + tw > rr.width ? px - tw - 14 : px + 14, 0, Math.max(0, rr.width - tw))}px`;
    tip.style.top = `${cr.top - rr.top + P.t}px`;
    return hv;
  }
  function hideHover() {
    tip.hidden = true;
    for (const cc of charts) if (cc.hover) cc.hover.setAttribute('visibility', 'hidden');
  }
  for (const c of charts) {
    let drag = false;
    c.svg.addEventListener('pointermove', (e) => {
      const hv = showHover(e, c);
      if (drag && hv) api.seekUser(hv.t);
    });
    c.svg.addEventListener('pointerleave', () => { if (!drag) hideHover(); });
    c.svg.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const hv = hoverAt(e, c);
      if (!hv) return;
      drag = true;
      api.clock.hold(true);
      try { c.svg.setPointerCapture(e.pointerId); } catch (err) { /* не страшно */ }
      api.seekUser(hv.t);
      e.preventDefault();
    });
    const end = () => { if (drag) { drag = false; api.clock.hold(false); } };
    c.svg.addEventListener('pointerup', end);
    c.svg.addEventListener('pointercancel', () => { end(); hideHover(); });
  }

  const ro = new ResizeObserver(() => { if (Math.floor(root.clientWidth) !== W) build(); });
  ro.observe(root);
  return { build, setCursor, destroy() { ro.disconnect(); root.remove(); } };
}

/* ============================================================================================
 * План, журнал, гипотезы, обмены с языковой моделью
 * ========================================================================================== */

function subgoalText(g) {
  const name = SUBGOAL_RU[g.type] || g.type || 'подцель';
  const parts = [name];
  if (g.target && g.type !== 'return_base') parts.push(String(g.target));
  if (g.x != null && g.y != null) parts.push(xy(g.x, g.y));
  return parts.join(' ');
}

function createPlan(host, api) {
  const root = h('section', 'rp-plan rp-card');
  const head = h('div', 'rp-card-head');
  head.appendChild(h('h3', 'rp-card-title', 'Текущий план'));
  const meta = h('span', 'rp-plan-meta');
  head.appendChild(meta);
  const bodyEl = h('div', 'rp-plan-body');
  const why = h('p', 'rp-plan-why');
  const goals = h('ol', 'rp-plan-goals');
  bodyEl.append(why, goals);
  root.append(head, bodyEl);
  host.appendChild(root);
  let key = null;
  return {
    update(t) {
      const m = api.model();
      const i = upperBound(m.plansT, t) - 1;
      const k = `${i}:${m.plans.length}`;
      if (k === key) return;
      key = k;
      meta.textContent = '';
      goals.textContent = '';
      if (i < 0) {
        why.textContent = 'Плана пока нет.';
        return;
      }
      const p = m.plans[i];
      meta.appendChild(h('span', 'rp-tag', SOURCE_RU[p.source] || p.source || 'план'));
      const when = button('rp-link', `с ${sec(p.t)}`);
      when.title = 'Перейти к моменту, когда план принят';
      when.addEventListener('click', () => api.seekUser(p.t));
      meta.appendChild(when);
      if (p.trigger) meta.appendChild(h('span', 'rp-plan-trigger', `причина: ${TRIGGER_RU[p.trigger] || p.trigger}`));
      why.textContent = ru(p.reasoning) || '—';
      const list = p.subgoals || [];
      const shown = list.length > 5 ? list.slice(0, 4) : list;
      for (const g of shown) goals.appendChild(h('li', null, subgoalText(g)));
      if (list.length > shown.length) {
        const rest = list.length - shown.length;
        goals.appendChild(h('li', 'rp-plan-more', `и ещё ${rest} ${plural(rest, 'точка', 'точки', 'точек')}`));
      }
      root.classList.remove('rp-flash');
      void root.offsetWidth;
      root.classList.add('rp-flash');
    },
    destroy() { root.remove(); },
  };
}

function createJournal(host, api) {
  const root = h('section', 'rp-journal rp-card');
  const head = h('div', 'rp-card-head');
  head.appendChild(h('h3', 'rp-card-title', 'Журнал агента'));
  const count = h('span', 'rp-card-count');
  head.appendChild(count);
  const list = h('div', 'rp-j-list');
  list.tabIndex = 0;
  list.setAttribute('aria-label', 'Журнал агента: записи до текущего момента');
  const empty = h('div', 'rp-empty', 'Записей пока нет.');
  const toEnd = button('rp-j-toend');
  toEnd.append(icon(ICO.down), h('span', null, 'к последней записи'));
  toEnd.hidden = true;
  root.append(head, list, toEnd);
  host.appendChild(root);

  let items = [];
  let shown = -1;         // сколько записей показано
  let stick = true;       // держаться за последнюю запись, пока человек сам не пролистал вверх

  const kindOf = (e) => {
    if (e.kind === 'verdict') {
      const stt = e.data && e.data.status;
      return { cls: `verdict${stt ? ` rp-j-${stt}` : ''}`, label: KIND_RU.verdict };
    }
    return { cls: KIND_RU[e.kind] ? e.kind : 'other', label: KIND_RU[e.kind] || e.kind };
  };

  function build() {
    const m = api.model();
    const keepTop = list.scrollTop;
    list.textContent = '';
    items = m.journal.map((e) => {
      const kd = kindOf(e);
      const b = button(`rp-j rp-j-${kd.cls}`);
      b.hidden = true;
      const bodyEl = h('span', 'rp-j-body');
      bodyEl.append(h('span', 'rp-j-kind', kd.label), h('span', 'rp-j-text', ru(e.text)));
      b.append(h('span', 'rp-j-t', num(e.t, 1)), bodyEl);
      b.title = `Перейти к ${sec(e.t)}`;
      b.addEventListener('click', () => api.seekUser(e.t));
      list.appendChild(b);
      return b;
    });
    list.appendChild(empty);
    shown = -1;
    update(api.clock.t);
    if (!stick) list.scrollTop = keepTop;
  }

  function scrollEnd() {
    list.scrollTop = list.scrollHeight;
  }

  function update(t) {
    const m = api.model();
    const k = upperBound(m.journalT, t);
    if (k === shown) return;
    const from = Math.max(0, Math.min(shown < 0 ? 0 : shown, k) - 1);
    const to = Math.max(shown, k);
    for (let i = from; i < Math.min(to + 1, items.length); i++) {
      items[i].hidden = i >= k;
      items[i].classList.toggle('rp-j-new', i === k - 1);
    }
    shown = k;
    empty.hidden = k > 0;
    setText(count, `${k} из ${items.length}`);
    if (stick) scrollEnd();
  }

  // Листает человек — отпускаем; долистал до конца — снова держимся за последнюю запись.
  const atEnd = () => list.scrollHeight - list.scrollTop - list.clientHeight < 28;
  const userScroll = () => { stick = atEnd(); toEnd.hidden = stick; };
  list.addEventListener('wheel', () => requestAnimationFrame(userScroll), { passive: true });
  list.addEventListener('touchmove', () => requestAnimationFrame(userScroll), { passive: true });
  list.addEventListener('keydown', () => requestAnimationFrame(userScroll));
  list.addEventListener('pointerdown', () => { stick = false; });      // тянут полосу прокрутки
  list.addEventListener('pointerup', () => requestAnimationFrame(userScroll));
  list.addEventListener('pointercancel', () => requestAnimationFrame(userScroll));
  list.addEventListener('scroll', () => { if (!stick && atEnd()) { stick = true; toEnd.hidden = true; } });
  toEnd.addEventListener('click', () => { stick = true; toEnd.hidden = true; scrollEnd(); });
  // размер списка меняется (раскладка, стили) — остаёмся у последней записи
  const ro = new ResizeObserver(() => { if (stick) scrollEnd(); });
  ro.observe(list);

  return { build, update, destroy() { ro.disconnect(); root.remove(); } };
}

function createHypotheses(host, api) {
  const root = h('section', 'rp-hyps rp-card');
  const head = h('div', 'rp-card-head');
  head.appendChild(h('h3', 'rp-card-title', 'Гипотезы'));
  const count = h('span', 'rp-card-count');
  head.appendChild(count);
  const list = h('div', 'rp-h-list');
  list.tabIndex = 0;
  list.setAttribute('aria-label', 'Гипотезы агента и их состояние в текущий момент');
  root.append(head, list);
  host.appendChild(root);
  let key = null;
  let prevStatus = {};
  return {
    update(t) {
      const m = api.model();
      const k = `${upperBound(m.hypTimes, t)}:${m.hyps.length}`;
      if (k === key) return;
      key = k;
      const rowsData = [];
      const tally = { open: 0, confirmed: 0, refuted: 0, outdated: 0 };
      for (const hp of m.hyps) {
        const stt = hypStatusAt(hp, t);
        if (!stt) continue;
        tally[stt] = (tally[stt] || 0) + 1;
        rowsData.push({ hp, stt, closed: stt !== 'open' });
      }
      rowsData.sort((a, b) => (a.closed !== b.closed ? (a.closed ? 1 : -1)
        : a.closed ? (b.hp.t_close - a.hp.t_close) : (b.hp.t_open - a.hp.t_open)));
      list.textContent = '';
      if (!rowsData.length) list.appendChild(h('div', 'rp-empty', 'Гипотез пока нет.'));
      const nowStatus = {};
      for (const { hp, stt, closed } of rowsData) {
        nowStatus[hp.id] = stt;
        const b = button(`rp-h rp-h-${stt}`);
        const top = h('div', 'rp-h-top');
        top.append(h('span', 'rp-h-id', hp.id || ''), h('span', `rp-badge rp-badge-${stt}`, HYP_RU[stt] || stt),
          h('span', 'rp-h-when', closed ? `${num(hp.t_open, 1)} → ${num(hp.t_close, 1)} с` : `с ${sec(hp.t_open)}`));
        b.appendChild(top);
        b.appendChild(h('div', 'rp-h-text', ru(hp.statement)));
        if (closed && hp.verdict) b.appendChild(h('div', 'rp-h-sub', `Вердикт: ${ru(hp.verdict)}`));
        else if (hp.test) b.appendChild(h('div', 'rp-h-sub', `Проверка: ${ru(hp.test)}`));
        b.title = `Перейти к ${sec(closed ? hp.t_close : hp.t_open)}`;
        b.addEventListener('click', () => api.seekUser(closed ? hp.t_close : hp.t_open));
        if (prevStatus[hp.id] !== undefined && prevStatus[hp.id] !== stt) b.classList.add('rp-flash');
        else if (prevStatus[hp.id] === undefined && Object.keys(prevStatus).length) b.classList.add('rp-flash');
        list.appendChild(b);
      }
      prevStatus = nowStatus;
      const parts = [];
      if (tally.open) parts.push(`${tally.open} ${plural(tally.open, 'открыта', 'открыты', 'открыто')}`);
      if (tally.confirmed) parts.push(`${tally.confirmed} ${plural(tally.confirmed, 'подтверждена', 'подтверждены', 'подтверждено')}`);
      if (tally.refuted) parts.push(`${tally.refuted} ${plural(tally.refuted, 'опровергнута', 'опровергнуты', 'опровергнуто')}`);
      if (tally.outdated) parts.push(`${tally.outdated} ${plural(tally.outdated, 'устарела', 'устарели', 'устарело')}`);
      setText(count, parts.join(' · '));
    },
    destroy() { root.remove(); },
  };
}

function createLlm(host, api) {
  const root = h('details', 'rp-llm rp-card');
  const sum = h('summary', 'rp-llm-sum');
  const body = h('div', 'rp-llm-body');
  root.append(sum, body);
  host.appendChild(root);
  let key = null;
  const show = (v) => (typeof v === 'string' ? v : JSON.stringify(v, null, 2));
  function fill(t) {
    const m = api.model();
    const k = upperBound(m.llmT, t);
    body.textContent = '';
    if (!k) body.appendChild(h('div', 'rp-empty', 'К этому моменту обменов ещё не было.'));
    for (let i = k - 1; i >= 0; i--) {
      const ex = m.llm[i];
      const d = h('details', 'rp-llm-item');
      const s2 = h('summary');
      s2.append(h('span', 'rp-j-t', num(ex.t, 1)),
        h('span', `rp-badge rp-badge-${ex.ok === false ? 'bad' : 'good'}`, ex.ok === false ? 'ошибка' : 'ответ получен'),
        h('span', 'rp-llm-ms', ex.latency_ms != null ? `${num(ex.latency_ms, 0)} мс` : ''));
      d.appendChild(s2);
      d.addEventListener('toggle', () => {
        if (!d.open || d.dataset.filled) return;
        d.dataset.filled = '1';
        for (const kk of Object.keys(ex)) {
          const kv = h('div', 'rp-kv');
          kv.append(h('div', 'rp-kv-k', kk), h('pre', 'rp-kv-v', show(ex[kk])));
          d.appendChild(kv);
        }
      });
      body.appendChild(d);
    }
  }
  root.addEventListener('toggle', () => { if (root.open) { key = null; fill(api.clock.t); } });
  return {
    update(t) {
      const m = api.model();
      root.hidden = !m.llm.length;
      if (root.hidden) return;
      const k = upperBound(m.llmT, t);
      const kk = `${k}:${m.llm.length}`;
      if (kk === key) return;
      key = kk;
      sum.textContent = `Обмены с языковой моделью: ${k} из ${m.llm.length}`;
      if (root.open) fill(t);
    },
    destroy() { root.remove(); },
  };
}

/* ============================================================================================
 * Расследования: странность -> объяснения -> опыт -> предсказания и измерение -> вывод -> действие
 * ========================================================================================== */

/** Вероятность словами: «53 %», «меньше 1 %», «больше 99 %». */
function pct(p) {
  if (p == null) return '—';
  if (p > 0 && p < 0.005) return 'меньше 1 %';
  if (p >= 0.995 && p < 1) return 'больше 99 %';
  return `${num(p * 100, 0)} %`;
}
/**
 * Число с единицей опыта. В поле unit бывает и настоящая единица («ед/с»), и название величины («разброс»),
 * и множитель («× к обычному полу») — пишем так, чтобы читалось по-русски.
 */
function withUnit(text, unit) {
  if (!unit) return text;
  if (unit[0] === '×') return `×${text}${unit.slice(1)}`;
  if (/\//.test(unit) || /^(ед|м|см|с|%|°|бит)/.test(unit)) return `${text} ${unit}`;
  return `${unit} ${text}`;
}
const bitsWord = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? plural(Math.round(v), 'бит', 'бита', 'бит') : 'бита');

/** Круглые деления шкалы. */
function niceAxis(lo, hi, target) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / Math.max(1, target || 3);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
  const a = Math.floor(lo / step + 1e-9) * step;
  const b = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = a; v <= b + step * 1e-6; v += step) ticks.push(+v.toFixed(10));
  return { lo: a, hi: b, ticks, step };
}

/**
 * Шкала опыта: что предсказывало каждое объяснение (среднее ± разброс) и что получилось на самом деле.
 * show: {measured: показывать ли измерение, winner: объяснение-победитель или null}
 */
function predScale(test, show) {
  if (!test.preds.length) return null;
  const ms = show.measured ? test.measured : null;
  const vals = [];
  for (const p of test.preds) vals.push(p.mean - 1.3 * p.sigma, p.mean + 1.3 * p.sigma);
  if (test.measured) vals.push(test.measured.value - test.measured.sigma, test.measured.value + test.measured.sigma);
  let lo = Math.min.apply(null, vals);
  let hi = Math.max.apply(null, vals);
  // Величины здесь почти всегда неотрицательные (расход, разброс): тогда шкала начинается с нуля.
  const nonNeg = test.preds.every((p) => p.mean >= 0) && (!test.measured || test.measured.value >= 0);
  if (nonNeg && lo < (hi - lo) * 0.6) lo = 0;
  const pad = (hi - lo) * 0.06 || 0.5;
  const ax = niceAxis(lo === 0 && nonNeg ? 0 : lo - pad, hi + pad, 3);
  const span = ax.hi - ax.lo;
  const d = digitsFor(span);
  const pos = (v) => clamp(((v - ax.lo) / span) * 100, 0, 100);

  const root = h('div', 'rp-sc');
  const rows = test.preds.length;
  // сетка делений — под строками
  const grid = h('div', 'rp-sc-grid');
  grid.style.gridRow = `2 / span ${rows}`;
  for (const tk of ax.ticks) {
    const line = h('i', tk === 0 ? 'rp-sc-zero' : null);
    line.style.left = `${pos(tk).toFixed(2)}%`;
    grid.appendChild(line);
  }
  root.appendChild(grid);

  // подпись измерения над шкалой
  const top = h('div', 'rp-sc-top');
  if (ms) {
    const x = pos(ms.value);
    const lab = h('span', 'rp-sc-mlabel');
    lab.append(h('span', null, 'измерено '), h('b', null, withUnit(num(ms.value, d), test.unit)));
    lab.style.left = `${x.toFixed(2)}%`;
    lab.dataset.side = x < 28 ? 'left' : x > 72 ? 'right' : 'mid';
    top.appendChild(lab);
  } else if (test.measured) {
    top.appendChild(h('span', 'rp-sc-wait', 'измерение ещё идёт'));
  } else {
    top.appendChild(h('span', 'rp-sc-wait', 'опыт не проводился: только предсказания'));
  }
  root.appendChild(top);

  const said = [];
  test.preds.forEach((p, i) => {
    const win = show.winner && show.winner === p.alt;
    let fit = null;
    if (ms) {
      const sd = Math.hypot(p.sigma, ms.sigma);
      fit = sd > 0 ? Math.abs(ms.value - p.mean) / sd <= 2 : Math.abs(ms.value - p.mean) < 1e-9;
    }
    const cls = `rp-sc-cell${win ? ' rp-sc-win' : ''}${fit === false ? ' rp-sc-miss' : ''}`;
    const row = String(i + 2);
    const letter = h('span', `rp-q-letter ${cls}`, p.alt.letter);
    letter.title = p.alt.statement;
    letter.style.gridRow = row;
    const track = h('div', `rp-sc-track ${cls}`);
    track.style.gridRow = row;
    const a = pos(p.mean - p.sigma);
    const b = pos(p.mean + p.sigma);
    const band = h('i', 'rp-sc-int');
    band.style.left = `${a.toFixed(2)}%`;
    band.style.width = `${Math.max(b - a, 0).toFixed(2)}%`;
    const dot = h('i', 'rp-sc-dot');
    dot.style.left = `${pos(p.mean).toFixed(2)}%`;
    track.append(band, dot);
    track.title = `${p.alt.letter}. ${p.alt.statement}: предсказание ${withUnit(`${num(p.mean, d)} ± ${num(p.sigma, d)}`, test.unit)}`;
    const val = h('span', `rp-sc-val ${cls}`, `${num(p.mean, d)} ± ${num(p.sigma, d)}`);
    val.style.gridRow = row;
    const mark = h('span', `rp-sc-fit ${cls}`);
    mark.style.gridRow = row;
    if (fit != null) {
      mark.appendChild(icon(fit ? ICO.check : ICO.cross, 'rp-ico rp-sc-fitico'));
      mark.appendChild(h('span', null, fit ? 'сходится' : 'не сходится'));
    }
    root.append(letter, track, val, mark);
    said.push(`${p.alt.letter} — ${num(p.mean, d)} ± ${num(p.sigma, d)}${fit == null ? '' : fit ? ', сходится' : ', не сходится'}`);
  });

  // отметка измерения поверх всех строк
  if (ms) {
    const over = h('div', 'rp-sc-over');
    over.style.gridRow = `1 / span ${rows + 1}`;
    if (ms.sigma > 0) {
      const a = pos(ms.value - ms.sigma);
      const b = pos(ms.value + ms.sigma);
      const band = h('i', 'rp-sc-mband');
      band.style.left = `${a.toFixed(2)}%`;
      band.style.width = `${Math.max(b - a, 0).toFixed(2)}%`;
      over.appendChild(band);
    }
    const line = h('i', 'rp-sc-mline');
    line.style.left = `${pos(ms.value).toFixed(2)}%`;
    over.appendChild(line);
    root.appendChild(over);
  }

  const axis = h('div', 'rp-sc-axis');
  axis.style.gridRow = String(rows + 2);
  ax.ticks.forEach((tk, i) => {
    const lab = h('span', null, num(tk, digitsFor(ax.step * 4)));
    lab.style.left = `${pos(tk).toFixed(2)}%`;
    if (i === 0) lab.dataset.side = 'left';
    else if (i === ax.ticks.length - 1) lab.dataset.side = 'right';
    axis.appendChild(lab);
  });
  const unitNote = h('span', 'rp-sc-unit', test.unit || '');
  unitNote.style.gridRow = String(rows + 2);
  root.append(axis, unitNote);
  root.setAttribute('role', 'img');
  root.setAttribute('aria-label', `Опыт «${test.name}». ${ms ? `Измерено: ${withUnit(num(ms.value, d), test.unit)}. ` : ''}Предсказания объяснений: ${said.join('; ')}.`);
  return root;
}

function createInquiries(host, api) {
  const root = h('section', 'rp-inq');
  root.hidden = true;
  root.setAttribute('aria-label', 'Расследования агента');
  const head = h('div', 'rp-inq-head');
  const title = h('h3', 'rp-inq-title', 'Расследования');
  const sum = h('span', 'rp-inq-sum');
  const lead = h('p', 'rp-inq-lead');
  const chain = ['заметил странность', 'выдвинул объяснения', 'выбрал опыт', 'сверил предсказания с измерением', 'сделал вывод'];
  chain.forEach((txt, i) => {
    if (i) lead.appendChild(icon(ICO.arrow, 'rp-ico rp-inq-arrow'));
    lead.appendChild(h('span', null, txt));
  });
  head.append(title, sum, lead);
  const list = h('div', 'rp-inq-list');
  const extra = h('div', 'rp-inq-extra');
  root.append(head, list, extra);
  host.appendChild(root);

  let cards = [];
  const manual = new Map();          // раскрыл или свернул карточку сам человек

  const seekBtn = (text, t, hint) => {
    const b = button('rp-link rp-q-when', text);
    b.title = hint || `Перейти к ${sec(t)}`;
    b.addEventListener('click', (e) => { e.stopPropagation(); api.seekUser(t); });
    return b;
  };
  const stepBox = (n, name) => {
    const box = h('section', 'rp-q-step');
    const hd = h('div', 'rp-q-stephead');
    hd.append(h('span', 'rp-q-stepn', String(n)), h('span', 'rp-q-stepname', name));
    box.appendChild(hd);
    return box;
  };

  /* --- 1. что заметил --- */
  function stepNoticed(q) {
    const box = stepBox(1, 'Что заметил');
    box.appendChild(h('p', 'rp-q-big', cap(q.anomaly.text)));
    const an = q.anomaly;
    const rate = q.topic === 'energy' || /ед/.test(an.unit);
    if (rate && an.observed != null && an.expected != null && Math.abs(an.observed - an.expected) > 1e-9) {
      const max = Math.max(Math.abs(an.observed), Math.abs(an.expected), 1e-9);
      const d = digitsFor(max);
      const cmp = h('div', 'rp-q-cmp');
      const line = (label, v, cls) => {
        const bar = h('div', `rp-bar ${cls}`);
        const fillEl = h('i');
        fillEl.style.width = `${clamp((Math.abs(v) / max) * 100, 0, 100).toFixed(1)}%`;
        bar.appendChild(fillEl);
        cmp.append(h('span', 'rp-q-cmplab', label), bar, h('b', null, `${num(v, d)}${an.unit ? ` ${an.unit}` : ''}`));
      };
      line('ожидал', an.expected, 'rp-bar-prior');
      line('получил', an.observed, 'rp-bar-post');
      cmp.setAttribute('role', 'img');
      cmp.setAttribute('aria-label', `Ожидал ${num(an.expected, d)}, получил ${num(an.observed, d)} ${an.unit}`);
      box.appendChild(cmp);
    }
    const foot = h('div', 'rp-q-foot');
    foot.appendChild(seekBtn(`на ${sec(q.t0)}`, q.t0, 'Перейти к моменту, когда агент заметил странность'));
    if (an.x != null && an.y != null) foot.appendChild(h('span', null, `в точке ${xy(an.x, an.y)} — на арене отмечена значком ${q.id}`));
    box.appendChild(foot);
    return box;
  }

  /* --- 2. объяснения --- */
  function stepAlts(q, st) {
    const box = stepBox(2, 'Возможные объяснения');
    const key = h('div', 'rp-q-key');
    key.append(h('i', 'rp-q-keysw rp-bar-prior'), h('span', null, 'до опыта'), h('i', 'rp-q-keysw rp-bar-post'), h('span', null, 'после'));
    box.appendChild(key);
    const postShown = st.state === 'closed' || (q.measuredT.length > 0 && st.seen >= q.measuredT.length);
    const c = st.state === 'closed' ? q.conclusion : null;
    const winner = c && c.status === 'identified' ? c.best : null;
    for (const a of q.alts) {
      const post = postShown ? a.posterior : null;
      const out = post != null && post < 0.05;
      const row = h('div', `rp-alt${winner === a ? ' rp-alt-win' : ''}${out ? ' rp-alt-out' : ''}`);
      const main = h('div', 'rp-alt-main');
      const text = h('div', 'rp-alt-text');
      text.appendChild(h('span', null, cap(a.statement)));
      if (winner === a) {
        const tag = h('span', 'rp-alt-tag rp-alt-tag-win');
        tag.append(icon(ICO.check), h('span', null, 'подтвердилось'));
        text.appendChild(tag);
      } else if (out) {
        text.appendChild(h('span', 'rp-alt-tag', 'отпало'));
      } else if (post != null && post >= 0.1 && c && c.status !== 'identified') {
        text.appendChild(h('span', 'rp-alt-tag rp-alt-tag-keep', 'не исключено'));
      }
      const bars = h('div', 'rp-alt-bars');
      const line = (label, v, cls, wait) => {
        const bar = h('div', `rp-bar ${cls}`);
        const fillEl = h('i');
        fillEl.style.width = v == null ? '0%' : `${clamp(v * 100, 0, 100).toFixed(1)}%`;
        bar.appendChild(fillEl);
        bars.append(h('span', 'rp-alt-lab', label), bar, h('b', wait ? 'rp-alt-wait' : null, wait || pct(v)));
      };
      line('до', a.prior, 'rp-bar-prior');
      line('после', post, 'rp-bar-post', postShown ? null : 'ждём опыта');
      main.append(text, bars);
      const letter = h('span', 'rp-q-letter', a.letter);
      row.append(letter, main);
      row.setAttribute('role', 'group');
      row.setAttribute('aria-label', `${a.letter}. ${a.statement}: до опыта ${pct(a.prior)}${postShown ? `, после ${pct(a.posterior)}` : ''}`);
      box.appendChild(row);
    }
    if (!q.alts.length) box.appendChild(h('div', 'rp-empty', 'Объяснения в записи не указаны.'));
    return box;
  }

  /* --- 3. опыты, предсказания и измерение --- */
  function stepTests(q, st) {
    const box = stepBox(3, 'Опыт: что предсказывали объяснения и что вышло');
    const tests = q.tests;
    if (!tests.length) {
      // Бывает, что новый опыт не нужен: объяснения уже различил прежний. Причину агент пишет в примечании.
      const c0 = q.conclusion;
      box.appendChild(h('p', 'rp-q-big rp-q-big-soft', c0 && c0.status === 'identified' ? 'Новый опыт не понадобился' : 'Опыт не ставился'));
      box.appendChild(h('p', 'rp-q-plain', q.note ? cap(q.note) : (c0 && c0.status === 'identified'
        ? 'Объяснения удалось различить по тому, что агент уже знал.' : 'Опытов, которые различили бы объяснения, не нашлось.')));
      return box;
    }
    box.appendChild(h('p', 'rp-q-note', 'Агент берёт опыт, где больше всего пользы на единицу заряда. Польза — на сколько бит опыт уменьшит сомнения: один бит — вдвое меньше.'));
    const maxEff = Math.max.apply(null, tests.map((x) => x.eff || 0).concat([1e-9]));
    const c = st.state === 'closed' ? q.conclusion : null;
    const winner = c && c.status === 'identified' ? c.best : null;
    for (const x of tests) {
      const done = !!x.measured;
      const shown = done && x.measured.t <= st.t + 1e-6;
      const el = h('div', `rp-t${done ? ' rp-t-done' : ' rp-t-skip'}`);
      const hd = h('div', 'rp-t-head');
      hd.appendChild(h('span', 'rp-t-name', cap(x.name)));
      if (done) {
        const tag = h('span', 'rp-t-tag rp-t-tag-done');
        tag.append(icon(ICO.check), h('span', null, shown ? 'проведён' : 'выбран'));
        hd.append(tag, seekBtn(`${sec(x.measured.t)}`, x.measured.t, 'Перейти к моменту измерения'));
      } else {
        hd.appendChild(h('span', 'rp-t-tag', st.state === 'closed' ? 'не понадобился' : 'в запасе'));
      }
      el.appendChild(hd);
      const meta = [];
      if (x.gain != null) meta.push(`польза ${num(x.gain, 2)} ${bitsWord(x.gain)}`);
      if (x.cost != null) meta.push(x.free ? 'заряда почти не стоит' : `цена ${num(x.cost, 2)} ед. заряда`);
      if (x.duration != null) meta.push(`${num(x.duration, x.duration % 1 ? 1 : 0)} с`);
      if (meta.length) el.appendChild(h('div', 'rp-t-meta', meta.join(' · ')));
      if (x.eff != null) {
        const eff = h('div', 'rp-t-eff');
        const bar = h('div', `rp-bar ${done ? 'rp-bar-post' : 'rp-bar-prior'}`);
        const fillEl = h('i');
        fillEl.style.width = `${clamp((x.eff / maxEff) * 100, 0, 100).toFixed(1)}%`;
        bar.appendChild(fillEl);
        const effShown = +(x.eff >= 10 ? x.eff.toFixed(0) : x.eff.toFixed(1));
        eff.append(bar, h('b', null, x.free ? 'почти бесплатно' : `${num(effShown, x.eff >= 10 ? 0 : 1)} ${bitsWord(effShown)} на ед. заряда`));
        eff.title = 'Польза опыта на единицу заряда: чем длиннее полоса, тем выгоднее опыт';
        el.appendChild(eff);
      }
      const scale = predScale(x, { measured: shown, winner });
      if (scale) {
        if (done) el.appendChild(scale);
        else {
          const det = h('details', 'rp-t-more');
          det.appendChild(h('summary', null, 'что предсказывали объяснения'));
          det.appendChild(scale);
          el.appendChild(det);
        }
      }
      box.appendChild(el);
    }
    if (q.alts.some((a) => a.id === 'other')) {
      const other = q.alts.find((a) => a.id === 'other');
      box.appendChild(h('p', 'rp-q-note', `У объяснения ${other.letter} («${other.statement}») предсказаний нет: оно допускает любое значение.`));
    }
    return box;
  }

  /* --- 4. вывод и действие --- */
  function stepConclusion(q, st) {
    const box = stepBox(4, 'Вывод и действие');
    const c = q.conclusion;
    if (st.state !== 'closed' || !c) {
      const wait = h('div', 'rp-q-verdict rp-q-verdict-wait');
      const ended = q.t1 == null && api.model().finished;
      wait.append(h('div', 'rp-q-vtitle', ended ? 'Расследование не закончено' : 'Вывода пока нет'),
        h('div', 'rp-q-vsub', ended ? 'Прогон завершился раньше, чем агент успел сделать вывод.' : 'Агент ставит опыт и ждёт измерения.'));
      box.appendChild(wait);
      if (q.t1 != null) box.appendChild(seekBtn(`перейти к выводу — ${sec(q.t1)}`, q.t1, 'Перейти к моменту, когда агент сделал вывод'));
      return box;
    }
    const ok = c.status === 'identified';
    const v = h('div', `rp-q-verdict ${ok ? 'rp-q-verdict-ok' : 'rp-q-verdict-ins'}`);
    const vt = h('div', 'rp-q-vtitle');
    vt.append(icon(ok ? ICO.check : ICO.question), h('span', null, ok ? 'Причина найдена' : 'Недостаточно данных'));
    v.appendChild(vt);
    const subs = [];
    if (ok && c.confidence != null) subs.push(`уверенность ${pct(c.confidence)}`);
    if (!ok) subs.push('агент не стал называть причину наугад');
    if (subs.length) v.appendChild(h('div', 'rp-q-vsub', subs.join(' · ')));
    const text = cap(c.text.replace(/^недостаточно данных[:.]?\s*/i, ''));
    if (text) v.appendChild(h('p', 'rp-q-vtext', text));
    box.appendChild(v);
    if (q.action) {
      const act = h('div', 'rp-q-action');
      act.append(h('div', 'rp-q-actlab', 'Что сделал дальше'), h('p', null, cap(q.action)));
      box.appendChild(act);
    }
    if (st.truthOn && (q.verdict || (q.truth && q.truth.length))) {
      const names = (q.truth || []).map((id) => CAUSE_RU[id] || (q.alts.find((a) => a.id === id) || {}).statement || id);
      const real = names.length ? `На самом деле: ${names.join(' и ')}.` : '';
      const tone = { correct: 'ok', wrong: 'bad' }[q.verdict] || 'none';
      const word = { correct: 'вывод верный', wrong: 'вывод неверный', unverifiable: 'проверить нечем: явной причины в сценарии нет', insufficient: 'агент причину не назвал' }[q.verdict] || '';
      const tr = h('div', `rp-q-truth rp-q-truth-${tone}`);
      tr.appendChild(h('div', 'rp-q-actlab', 'Сверка судьи со скрытой правдой'));
      const line = h('p');
      if (tone !== 'none') line.appendChild(icon(tone === 'ok' ? ICO.check : ICO.cross));
      line.appendChild(h('span', null, [word ? cap(word) + '.' : '', real].filter(Boolean).join(' ')));
      tr.appendChild(line);
      box.appendChild(tr);
    }
    if (q.critique.length) {
      const cr = h('div', 'rp-q-crit');
      cr.appendChild(h('div', 'rp-q-actlab', 'Возражения к выводу'));
      const ul = h('ul');
      for (const it of q.critique) {
        const li = h('li');
        li.appendChild(h('span', null, cap(it.issue)));
        if (it.resolved != null) li.appendChild(h('span', `rp-t-tag${it.resolved ? ' rp-t-tag-done' : ''}`, it.resolved ? 'снято' : 'осталось'));
        ul.appendChild(li);
      }
      cr.appendChild(ul);
      box.appendChild(cr);
    }
    const notes = [];
    if (INQ_SOURCE_RU[q.source]) notes.push(cap(INQ_SOURCE_RU[q.source]));
    if (q.note && q.tests.length) notes.push(cap(q.note));          // без опытов примечание уже показано в третьем шаге
    if (notes.length) box.appendChild(h('p', 'rp-q-note', notes.join('. ')));
    return box;
  }

  function makeCard(q) {
    const el = h('article', 'rp-q');
    const top = h('div', 'rp-q-top');
    const toggle = button('rp-q-toggle');
    toggle.appendChild(icon(ICO.chevron));
    const headBtn = button('rp-q-head');
    const body = h('div', 'rp-q-body');
    top.append(headBtn, toggle);
    el.append(top, body);
    const card = { q, el, headBtn, toggle, body, key: '', open: false };
    headBtn.addEventListener('click', () => {
      manual.delete(q.id);                   // щелчок по карточке возвращает её к обычному поведению
      api.seekUser(q.t0);
    });
    toggle.addEventListener('click', () => {
      manual.set(q.id, !card.open);
      update(api.clock.t, true);
    });
    return card;
  }

  function paintCard(card, st) {
    const q = card.q;
    const c = q.conclusion;
    card.open = st.open;
    card.el.dataset.state = st.state;
    card.el.dataset.status = st.state === 'closed' && c ? (c.status === 'identified' ? 'identified' : 'insufficient') : '';
    card.el.classList.toggle('rp-q-open', st.open);
    // шапка
    const hb = card.headBtn;
    hb.textContent = '';
    hb.appendChild(h('span', 'rp-q-id', q.id));
    if (TOPIC_RU[q.topic]) hb.appendChild(h('span', 'rp-q-topic', TOPIC_RU[q.topic]));
    if (st.state === 'future') {
      hb.appendChild(h('span', 'rp-q-anomaly rp-q-muted', `впереди — начнётся на ${sec(q.t0)}`));
    } else {
      hb.appendChild(h('span', 'rp-q-anomaly', cap(q.anomaly.text)));
      hb.appendChild(h('span', 'rp-q-time', q.t1 != null && st.state === 'closed' ? `${num(q.t0, 1)} → ${sec(q.t1)}` : `с ${sec(q.t0)}`));
      if (st.state === 'closed' && c) {
        const ok = c.status === 'identified';
        hb.appendChild(h('span', `rp-badge ${ok ? 'rp-badge-good' : 'rp-badge-open'}`, ok ? 'причина найдена' : 'недостаточно данных'));
        if (ok && c.best && !st.open) hb.appendChild(h('span', 'rp-q-short', c.best.statement));
      } else {
        hb.appendChild(h('span', 'rp-badge rp-badge-live', q.t1 == null && api.model().finished ? 'не закончено' : 'идёт'));
      }
    }
    hb.title = st.state === 'future' ? `Перейти к началу расследования: ${sec(q.t0)}` : `Перейти к моменту, когда агент заметил странность: ${sec(q.t0)}`;
    card.toggle.disabled = st.state === 'future';
    card.toggle.setAttribute('aria-expanded', String(st.open));
    card.toggle.setAttribute('aria-label', st.open ? `Свернуть расследование ${q.id}` : `Развернуть расследование ${q.id}`);
    card.toggle.title = st.open ? 'Свернуть' : 'Развернуть';
    // тело
    card.body.textContent = '';
    card.body.hidden = !st.open;
    if (!st.open) return;
    const steps = h('div', 'rp-q-steps');
    steps.append(stepNoticed(q), stepAlts(q, st), stepTests(q, st), stepConclusion(q, st));
    card.body.appendChild(steps);
  }

  function update(t, force) {
    if (!cards.length) return;
    const truthOn = api.truthOn();
    let cur = -1;
    for (let i = 0; i < cards.length; i++) if (cards[i].q.t0 <= t + 1e-6) cur = i;
    cards.forEach((card, i) => {
      const q = card.q;
      const state = t + 1e-6 < q.t0 ? 'future' : (q.t1 != null && t + 1e-6 >= q.t1 ? 'closed' : 'open');
      const seen = state === 'future' ? 0 : q.measuredT.filter((x) => x <= t + 1e-6).length;
      const open = state !== 'future' && (manual.has(q.id) ? manual.get(q.id) : i === cur);
      const key = `${state}:${seen}:${open}:${truthOn}`;
      if (key === card.key && !force) return;
      const flash = card.key && card.key.split(':')[0] !== state && state !== 'future';
      card.key = key;
      paintCard(card, { state, seen, open, truthOn, t });
      if (flash) {
        card.el.classList.remove('rp-flash');
        void card.el.offsetWidth;
        card.el.classList.add('rp-flash');
      }
    });
  }

  function build() {
    const m = api.model();
    const qs = m.inquiries;
    root.hidden = !qs.length;
    list.textContent = '';
    extra.textContent = '';
    cards = qs.map(makeCard);
    for (const card of cards) list.appendChild(card.el);
    if (!qs.length) return;
    const found = qs.filter((q) => q.conclusion && q.conclusion.status === 'identified').length;
    const ins = qs.filter((q) => q.conclusion && q.conclusion.status !== 'identified').length;
    const parts = [`${qs.length} ${plural(qs.length, 'расследование', 'расследования', 'расследований')}`];
    if (found) parts.push(`причина найдена: ${found}`);
    if (ins) parts.push(`недостаточно данных: ${ins}`);
    if (qs.length - found - ins) parts.push(`не закончено: ${qs.length - found - ins}`);
    setText(sum, parts.join(' · '));
    // что агент узнал о расходе заряда за прогон
    const em = m.energyModel;
    const UNIT = { per_m: 'ед/м', per_m_load: 'ед/м', per_rad: 'ед/рад', per_s: 'ед/с' };
    const rows = em ? Object.keys(em).filter((k) => em[k] && fin(em[k].value) != null) : [];
    if (rows.length) {
      extra.appendChild(h('div', 'rp-inq-extratitle', 'Модель расхода заряда, которую агент уточнил за этот прогон'));
      const dl = h('div', 'rp-inq-model');
      for (const k of rows) {
        const sd = Math.abs(fin(em[k].sigma) || 0);
        const d = sd > 0 ? (sd >= 1 ? 1 : sd >= 0.1 ? 2 : 3) : 2;
        const it = h('div', 'rp-inq-modelrow');
        it.append(h('span', null, cap(ru(em[k].label || k))),
          h('b', null, `${num(em[k].value, d)}${sd > 0 ? ` ± ${num(sd, d)}` : ''}${UNIT[k] ? ` ${UNIT[k]}` : ''}`));
        dl.appendChild(it);
      }
      extra.appendChild(dl);
    }
    update(api.clock.t, true);
  }

  return { el: root, build, update, destroy() { root.remove(); } };
}

/* ============================================================================================
 * mountReplay
 * ========================================================================================== */

/** Высота арены: не больше, чем остаётся на экране после остальных частей (reserve), но и не меньше floor. */
function arenaMaxHeight(opts, reserve, floor) {
  if (typeof opts.arenaMaxHeight === 'number') return () => opts.arenaMaxHeight;
  return () => Math.max(floor, (window.innerHeight || 800) - reserve);
}

export function mountReplay(container, opts = {}) {
  if (!container) throw new Error('mountReplay: нужен контейнер');
  const compact = !!opts.compact;
  let follow = !!opts.follow;
  let following = follow;
  const cssPending = ensureCss();
  const geom = buildGeom(opts.arena);
  let model = buildModel(opts.trace, null);
  let layers = Object.assign({}, DEFAULT_LAYERS, opts.view ? layersForView(opts.view, DEFAULT_LAYERS) : null, opts.layers || null);
  const clock = createClock();
  clock.setRange(model.t0, model.dur);
  let dead = false;
  const timeCbs = new Set();

  const root = h('div', `rp-root${compact ? ' rp-compact' : ' rp-full'}`);
  root.tabIndex = -1;
  container.appendChild(root);
  let theme = readTheme(root);

  const api = {
    clock,
    compact,
    follow,
    model: () => model,
    range: () => [model.t0, model.dur],
    lanes: () => [
      { label: 'среда', title: 'Скрытые изменения среды. Полоска после метки — сколько прошло, пока агент заметил', marks: model.lanes.world, lags: model.lags },
      { label: 'судья', title: 'Сообщения судьи: собранные образцы и штрафы', marks: model.lanes.judge },
      { label: 'агент', title: 'Тревоги агента (треугольники) и моменты, когда он менял план (штрихи)', marks: model.lanes.agent },
    ].concat(model.inquiries.length ? [
      { label: 'расследования', title: 'Расследования агента: лупа — момент, когда он заметил странность, полоска — пока шли опыты', marks: model.lanes.inquiry, lags: model.inquirySpans, lagClass: 'rp-tl-lag-inq' },
    ] : []),
    truthOn: () => !!(layers.truthSamples || layers.truthSoil || layers.truthHazards),
    seekUser(t) {
      clock.seek(t);
      if (follow) setFollowing(clock.t >= model.dur - 0.3);
    },
    toggle() {
      clock.toggle();
    },
    goLive() { setFollowing(true); clock.pause(); clock.seek(model.dur); },
    setView(v) { layers = layersForView(v, layers); syncLayers(); },
    toggleLayer(key) { layers = Object.assign({}, layers, { [key]: !layers[key] }); syncLayers(); },
  };

  const main = h('div', 'rp-main');
  const stage = h('div', 'rp-stage');
  main.appendChild(stage);

  // шапка: режимы показа и сведения о прогоне
  let viewSwitch = null;
  let runInfo = null;
  let inqJump = null;
  if (!compact) {
    const head = h('div', 'rp-head');
    const toolbar = h('div', 'rp-toolbar');
    toolbar.appendChild(h('span', 'rp-toolbar-label', 'Показать'));
    viewSwitch = createViewSwitch(toolbar, api, false);
    head.appendChild(toolbar);
    // быстрый переход к панели расследований (она под ареной)
    inqJump = button('rp-inq-jump');
    inqJump.hidden = true;
    inqJump.title = 'Показать панель расследований';
    head.appendChild(inqJump);
    runInfo = createRunInfo(head, 'full');
    root.appendChild(head);
  }
  root.appendChild(main);
  const arena = createArena(stage, {
    geom, model, layers, theme, compact,
    maxHeight: arenaMaxHeight(opts, compact ? 200 : 410, compact ? 280 : 470),
  });
  if (compact) {
    runInfo = createRunInfo(arena.el, 'tag');
    if (opts.viewSwitch !== false) {
      const corner = h('div', 'rp-arena-corner');
      viewSwitch = createViewSwitch(corner, api, true);
      arena.el.appendChild(corner);
    }
  }
  const legend = compact ? null : createLegend(stage, api);
  const timeline = createTimeline(stage, api);
  const statusLine = compact ? createStatusLine(stage, api) : null;

  let status = null, charts = null, plan = null, journal = null, hypotheses = null, llm = null, inquiries = null;
  if (!compact) {
    const sideA = h('div', 'rp-side-a');
    const logs = h('div', 'rp-logs');
    const logcol = h('div', 'rp-logcol');
    logs.appendChild(logcol);
    main.append(sideA, logs);
    const stWrap = h('div', 'rp-card rp-status-card');
    stWrap.appendChild(h('h3', 'rp-card-title', 'Состояние сейчас'));
    sideA.appendChild(stWrap);
    status = createStatus(stWrap, api);
    const chWrap = h('div', 'rp-card rp-charts-card');
    sideA.appendChild(chWrap);
    charts = createCharts(chWrap, api);
    plan = createPlan(sideA, api);
    journal = createJournal(logcol, api);
    llm = createLlm(logcol, api);
    hypotheses = createHypotheses(logs, api);
    inquiries = createInquiries(root, api);
    inqJump.addEventListener('click', () => inquiries.el.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }

  function syncLayers() {
    arena.setLayers(layers);
    if (viewSwitch) viewSwitch.sync(layers);
    if (legend) legend.sync(layers, model);
    if (inquiries) inquiries.update(clock.t);       // сверка с правдой видна только вместе со слоями «как на самом деле»
  }

  function setFollowing(v) {
    following = !!v;
    timeline.setLive(follow, following);
  }

  function render(t) {
    if (dead) return;
    const st = model.stateAt(t);
    arena.render(t);
    timeline.update(t);
    if (statusLine) statusLine.update(t, st);
    if (status) status.update(t, st);
    if (charts) charts.setCursor(t, st);
    if (plan) plan.update(t);
    if (journal) journal.update(t);
    if (hypotheses) hypotheses.update(t);
    if (llm) llm.update(t);
    if (inquiries) inquiries.update(t);
    if (follow && !following && t >= model.dur - 1e-6 && !clock.playing) setFollowing(true);
    for (const cb of timeCbs) {
      try { cb(t, st); } catch (e) { console.error(e); }
    }
  }
  clock.onTime(render);

  function rebuild() {
    timeline.build();
    if (charts) charts.build();
    if (journal) journal.build();
    if (inquiries) inquiries.build();
    if (inqJump) {
      const nq = model.inquiries.length;
      inqJump.hidden = !nq;
      inqJump.textContent = '';
      if (nq) inqJump.append(icon(ICO.mInquiry, 'rp-ico'), h('span', null, `Расследования: ${nq}`), icon(ICO.down, 'rp-ico'));
    }
    if (runInfo) runInfo.sync(model);
    syncLayers();
  }

  const player = {
    root,
    toggle: () => api.toggle(),
    step(dir) { api.seekUser(clock.t + dir); },
    jumpEvent(dir) {
      const ev = model.eventTimes;
      let target = null;
      if (dir > 0) target = ev.find((x) => x > clock.t + 0.051);
      else for (let i = ev.length - 1; i >= 0; i--) if (ev[i] < clock.t - 0.051) { target = ev[i]; break; }
      if (target == null) target = dir > 0 ? model.dur : model.t0;
      api.seekUser(target);
      timeline.flashAt(target);
    },
  };
  const unregister = registerPlayer(player);
  const onWinResize = () => arena.relayout();
  window.addEventListener('resize', onWinResize);

  rebuild();
  timeline.setLive(follow, following);
  if (follow) clock.seek(model.dur);
  else if (typeof opts.startAt === 'number') clock.seek(opts.startAt);
  render(clock.t);
  cssPending.then(() => {
    if (dead) return;
    theme = readTheme(root);
    arena.setTheme(theme);
    if (charts) charts.build();
    render(clock.t);
  });

  return {
    seek(t) { api.seekUser(t); },
    play() { clock.play(); },
    pause() { clock.pause(); },
    setSpeed(kSpeed) { clock.setSpeed(kSpeed); },
    onTime(cb) { timeCbs.add(cb); return () => timeCbs.delete(cb); },
    /** Подменить запись (для живого прогона — на более длинную), не сбрасывая слои, скорость и прокрутку. */
    update(trace, patch) {
      if (dead) return;
      const atLive = follow && following;      // курсор шёл за записью — доводим его и до последнего кадра
      if (patch && typeof patch.follow === 'boolean') { follow = patch.follow; api.follow = follow; following = follow && following; }
      model = buildModel(trace, model);
      arena.setModel(model);
      clock.setRange(model.t0, model.dur);
      rebuild();
      timeline.setLive(follow, following);
      if (atLive) clock.seek(model.dur);
      render(clock.t);
    },
    setView(v) { api.setView(v); },
    setLayers(patch) { layers = Object.assign({}, layers, patch || {}); syncLayers(); },
    getTime: () => clock.t,
    getDuration: () => model.dur,
    isPlaying: () => clock.playing,
    destroy() {
      if (dead) return;
      dead = true;
      unregister();
      window.removeEventListener('resize', onWinResize);
      clock.destroy();
      timeline.destroy();
      arena.destroy();
      for (const part of [status, charts, plan, journal, hypotheses, llm, inquiries, statusLine, runInfo]) if (part) part.destroy();
      timeCbs.clear();
      root.remove();
    },
  };
}

/* ============================================================================================
 * mountCompare
 * ========================================================================================== */

const COMPARE_ROWS = [
  { label: 'Счёт', get: (r) => r.score, fmt: (v) => num(v, 1), better: 'higher' },
  { label: 'Собрано образцов', get: (r) => r.samples_collected, fmt: (v, r) => `${v} из ${r.samples_total}`, better: 'higher' },
  { label: 'Вернулся на базу', get: (r) => (r.returned ? 1 : 0), fmt: (v) => (v ? 'да' : 'нет'), better: 'higher' },
  { label: 'Осталось заряда, ед.', get: (r) => (r.battery_left != null ? r.battery_left : r.battery), fmt: (v) => num(v, 1), better: 'higher' },
  { label: 'Заряд на один образец, ед.', get: (r) => r.energy_per_sample, fmt: (v) => num(v, 1), better: 'lower' },
  { label: 'Заезды в опасную зону', get: (r) => r.hazard_hits, fmt: (v) => num(v, 0), better: 'lower' },
  { label: 'Ложные сборы', get: (r) => r.false_collects, fmt: (v) => num(v, 0), better: 'lower' },
  { label: 'Столкновения', get: (r) => r.collisions, fmt: (v) => num(v, 0), better: 'lower' },
  { label: 'Расследований', get: (r) => (r.inquiries ? fin(r.inquiries.total) : null), fmt: (v) => num(v, 0), better: null },
  { label: 'Причина названа верно', get: (r) => (r.inquiries ? fin(r.inquiries.correct) : null), fmt: (v) => num(v, 0), better: 'higher' },
  { label: 'Путь, м', get: (r) => r.distance, fmt: (v) => num(v, 1), better: null },
  { label: 'Время прогона, с', get: (r) => (r.time != null ? r.time : r.t), fmt: (v) => num(v, 1), better: null },
];
const DETECT_ROWS = {
  soil_change: 'Заметил смену грунта через, с',
  new_hazard: 'Узнал о новой опасной зоне через, с',
  sensor_fault: 'Заметил сбой датчика через, с',
};

export function mountCompare(container, opts = {}) {
  if (!container) throw new Error('mountCompare: нужен контейнер');
  const cssPending = ensureCss();
  const traces = opts.traces || [];
  const labels = opts.labels || ['Первый прогон', 'Второй прогон'];
  const geom = buildGeom(opts.arena);
  let models = [buildModel(traces[0], null), buildModel(traces[1], null)];
  let layers = Object.assign({}, DEFAULT_LAYERS, { lidar: false }, opts.view ? layersForView(opts.view, DEFAULT_LAYERS) : null, opts.layers || null);
  const clock = createClock();
  const totalDur = () => Math.max(models[0].dur, models[1].dur);
  const totalStart = () => Math.min(models[0].t0, models[1].t0);
  clock.setRange(totalStart(), totalDur());
  let dead = false;
  const timeCbs = new Set();

  const root = h('div', 'rp-root rp-compare');
  root.tabIndex = -1;
  container.appendChild(root);
  let theme = readTheme(root);

  const api = {
    clock,
    compact: false,
    follow: false,
    range: () => [totalStart(), totalDur()],
    lanes: () => [
      { label: 'среда', title: 'Скрытые изменения среды — одинаковые для обоих прогонов', marks: (models[0].lanes.world.length ? models[0] : models[1]).lanes.world.map((mk) => ({ t: mk.t, kind: mk.kind, text: mk.text })) },
      { label: labels[0], title: 'Образцы, штрафы, тревоги и расследования первого прогона', marks: models[0].lanes.judge.concat(models[0].lanes.agent.filter((mk) => mk.kind === 'alarm'), models[0].lanes.inquiry) },
      { label: labels[1], title: 'Образцы, штрафы, тревоги и расследования второго прогона', marks: models[1].lanes.judge.concat(models[1].lanes.agent.filter((mk) => mk.kind === 'alarm'), models[1].lanes.inquiry) },
    ],
    seekUser(t) { clock.seek(t); },
    toggle() { clock.toggle(); },
    goLive() {},
    setView(v) { layers = layersForView(v, layers); syncLayers(); },
  };

  const toolbar = h('div', 'rp-toolbar');
  toolbar.appendChild(h('span', 'rp-toolbar-label', 'Показать'));
  const viewSwitch = createViewSwitch(toolbar, api, false);
  root.appendChild(toolbar);

  const grid = h('div', 'rp-cmp-grid');
  root.appendChild(grid);
  const panes = models.map((m, i) => {
    const cell = h('div', 'rp-cmp-cell');
    const head = h('div', 'rp-cmp-head');
    head.appendChild(h('h3', 'rp-cmp-title', labels[i] || `Прогон ${i + 1}`));
    const info = createRunInfo(head, 'inline');
    info.sync(m);
    const ended = h('span', 'rp-cmp-ended');
    ended.hidden = true;
    head.appendChild(ended);
    cell.appendChild(head);
    grid.appendChild(cell);
    const paneApi = { model: () => models[i] };
    const arena = createArena(cell, {
      geom, model: m, layers, theme, compact: true,
      maxHeight: arenaMaxHeight(opts, 400, 320),
    });
    const line = createStatusLine(cell, paneApi);
    return { arena, line, ended, info };
  });

  const timeline = createTimeline(root, api);

  const tableWrap = h('div', 'rp-card rp-cmp-tablewrap');
  root.appendChild(tableWrap);

  function buildTable() {
    tableWrap.textContent = '';
    tableWrap.appendChild(h('h3', 'rp-card-title', 'Итоги'));
    const ra = models[0].finished ? models[0].result : null;
    const rb = models[1].finished ? models[1].result : null;
    if (!ra || !rb) {
      tableWrap.appendChild(h('div', 'rp-empty', 'Итоги появятся, когда оба прогона закончатся.'));
      return;
    }
    const table = h('table', 'rp-cmp-table');
    const thead = h('thead');
    const hr = h('tr');
    hr.append(h('th', null, 'Показатель'), h('th', null, labels[0] || 'Первый'), h('th', null, labels[1] || 'Второй'));
    thead.appendChild(hr);
    const tbody = h('tbody');
    const addRow = (label, va, vb, fa, fb, better) => {
      const trEl = h('tr');
      trEl.appendChild(h('th', null, label));
      let win = -1;
      if (better && va != null && vb != null && va !== vb) win = (better === 'higher' ? va > vb : va < vb) ? 0 : 1;
      else if (better && (va == null) !== (vb == null)) win = va != null ? 0 : 1;
      [fa, fb].forEach((txt, i) => {
        const td = h('td', win === i ? 'rp-best' : null);
        td.appendChild(h('span', null, txt));
        if (win === i) { const mark = h('span', 'rp-best-mark', 'лучше'); td.appendChild(mark); }
        trEl.appendChild(td);
      });
      tbody.appendChild(trEl);
    };
    for (const row of COMPARE_ROWS) {
      const va = row.get(ra), vb = row.get(rb);
      if (va == null && vb == null) continue;
      addRow(row.label, va, vb, va == null ? '—' : row.fmt(va, ra), vb == null ? '—' : row.fmt(vb, rb), row.better);
    }
    const da = ra.detect || {}, db = rb.detect || {};
    for (const key of Object.keys(DETECT_ROWS)) {
      if (!(key in da) && !(key in db)) continue;
      const va = da[key], vb = db[key];
      const f = (v, has) => (v != null ? num(v, 1) : has ? 'не заметил' : '—');
      addRow(DETECT_ROWS[key], va != null ? va : null, vb != null ? vb : null, f(va, key in da), f(vb, key in db), 'lower');
    }
    table.append(thead, tbody);
    tableWrap.appendChild(table);
    tableWrap.appendChild(h('p', 'rp-cmp-note', 'Отметка «лучше» стоит там, где показатель однозначен: больше счёт и образцов, меньше штрафов и расхода. Путь и время — для справки.'));
  }

  function syncLayers() {
    for (const p of panes) p.arena.setLayers(layers);
    viewSwitch.sync(layers);
  }

  function render(t) {
    if (dead) return;
    panes.forEach((p, i) => {
      const m = models[i];
      const tt = clamp(t, m.t0, m.dur);
      p.arena.render(tt);
      p.line.update(tt, m.stateAt(tt));
      const over = t > m.dur + 0.05 && m.dur < totalDur() - 0.05;
      const early = t < m.t0 - 0.05;
      const note = over ? `прогон закончился на ${sec(m.dur)}` : early ? `прогон начнётся на ${sec(m.t0)}` : '';
      if (p.ended.textContent !== note) {
        p.ended.textContent = note;
        p.ended.hidden = !note;
      }
    });
    timeline.update(t);
    for (const cb of timeCbs) {
      try { cb(t); } catch (e) { console.error(e); }
    }
  }
  clock.onTime(render);

  const eventTimes = () => Array.from(new Set(models[0].eventTimes.concat(models[1].eventTimes))).sort((a, b) => a - b);
  const player = {
    root,
    toggle: () => clock.toggle(),
    step(dir) { clock.seek(clock.t + dir); },
    jumpEvent(dir) {
      const ev = eventTimes();
      let target = null;
      if (dir > 0) target = ev.find((x) => x > clock.t + 0.051);
      else for (let i = ev.length - 1; i >= 0; i--) if (ev[i] < clock.t - 0.051) { target = ev[i]; break; }
      if (target == null) target = dir > 0 ? totalDur() : totalStart();
      clock.seek(target);
      timeline.flashAt(target);
    },
  };
  const unregister = registerPlayer(player);
  const onWinResize = () => panes.forEach((p) => p.arena.relayout());
  window.addEventListener('resize', onWinResize);

  timeline.build();
  timeline.setLive(false, false);
  buildTable();
  syncLayers();
  if (typeof opts.startAt === 'number') clock.seek(opts.startAt);
  render(clock.t);
  cssPending.then(() => {
    if (dead) return;
    theme = readTheme(root);
    for (const p of panes) p.arena.setTheme(theme);
    render(clock.t);
  });

  return {
    seek(t) { clock.seek(t); },
    play() { clock.play(); },
    pause() { clock.pause(); },
    setSpeed(kSpeed) { clock.setSpeed(kSpeed); },
    onTime(cb) { timeCbs.add(cb); return () => timeCbs.delete(cb); },
    /** update([a, b]) — подменить обе записи; update(trace, i) — одну из них. */
    update(next, index) {
      if (dead) return;
      const arr = Array.isArray(next) ? next : models.map((m, i) => (i === (index || 0) ? next : m.trace));
      models = [buildModel(arr[0], models[0]), buildModel(arr[1], models[1])];
      panes.forEach((p, i) => { p.arena.setModel(models[i]); p.info.sync(models[i]); });
      clock.setRange(totalStart(), totalDur());
      timeline.build();
      buildTable();
      syncLayers();
      render(clock.t);
    },
    setView(v) { api.setView(v); },
    setLayers(patch) { layers = Object.assign({}, layers, patch || {}); syncLayers(); },
    getTime: () => clock.t,
    getDuration: totalDur,
    isPlaying: () => clock.playing,
    destroy() {
      if (dead) return;
      dead = true;
      unregister();
      window.removeEventListener('resize', onWinResize);
      clock.destroy();
      timeline.destroy();
      for (const p of panes) { p.arena.destroy(); p.line.destroy(); }
      timeCbs.clear();
      root.remove();
    },
  };
}

export default { mountReplay, mountCompare };
