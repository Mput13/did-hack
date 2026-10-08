// Знания: что агент выяснил о среде и переносит из прогона в прогон (GET /api/knowledge).
// Каждое правило — утверждение со значением и разбросом, числом подтверждений и ссылками на прогоны-доказательства.

import {
  h, fill, icon, loading, errorBox, api, href, runHref, num, count, plural, when, levelName, ruText,
} from './common.js';

const STATUS = {
  confirmed: { word: 'Подтверждено', tone: 'ok', icon: 'check', hint: 'совпало в нескольких прогонах' },
  tentative: { word: 'Предварительно', tone: 'warn', icon: 'question', hint: 'наблюдений пока мало' },
  retired: { word: 'Снято', tone: 'bad', icon: 'cross', hint: 'новые прогоны ему противоречат' },
};
const ORDER = ['confirmed', 'tentative', 'retired'];
const SHOWN = 8;        // сколько ссылок на прогоны видно сразу

// Пример для проверки вида страницы, пока настоящих знаний нет: #/knowledge?demo=1
const DEMO = {
  updated: '2026-10-08T03:40:00', runs: 14, demo: true,
  rules: [
    { id: 'energy.per_m', kind: 'расход', statement: 'метр по обычному полу стоит 2.49 ед/м', value: 2.49, sigma: 0.05, unit: 'ед/м', n: 14,
      support: ['E10/scientist/hard-1001', 'E10/scientist/hard-1002', 'E10/scientist/hard-1003', 'E10/scientist/hard-1004', 'E10/scientist/hard-1005', 'E10/scientist/hard-1006', 'E10/scientist/hard-1007', 'E10/scientist/hard-1008', 'E10/scientist/hard-1009', 'E10/scientist/hard-1010', 'E10/scientist/hard-1011', 'E10/scientist/hard-1012'],
      contradictions: [], status: 'confirmed', first_seen: 'E10/scientist/hard-1001', last_seen: 'E10/scientist/hard-1014' },
    { id: 'energy.per_rad', kind: 'расход', statement: 'поворот на радиан стоит 0.13 ед/рад', value: 0.128, sigma: 0.009, unit: 'ед/рад', n: 11,
      support: ['E10/scientist/hard-1001', 'E10/scientist/hard-1003', 'E10/scientist/hard-1004', 'E10/scientist/hard-1006', 'E10/scientist/hard-1007'],
      contradictions: ['E10/scientist/hard-1009'], status: 'confirmed', first_seen: 'E10/scientist/hard-1001', last_seen: 'E10/scientist/hard-1013' },
    { id: 'leak.rate', kind: 'сбой', statement: 'при утечке батарея теряет 0.15 ед/с', value: 0.151, sigma: 0.03, unit: 'ед/с', n: 2,
      support: ['E10/scientist/hard-1005', 'E10/scientist/hard-1011'], contradictions: [], status: 'tentative',
      first_seen: 'E10/scientist/hard-1005', last_seen: 'E10/scientist/hard-1011' },
    { id: 'fault.after_penalty', kind: 'сбой', statement: 'после штрафа в опасной зоне начинается сбой: утечка заряда — 40%, залипание датчика — 35%, без последствий — 25%',
      value: 0.75, sigma: 0, unit: 'доля штрафов со сбоем', n: 20, support: [], contradictions: [], status: 'confirmed', first_seen: null, last_seen: null },
    { id: 'fault.duration', kind: 'сбой', statement: 'сбой после штрафа длится 12.00 с', value: 12.0, sigma: 9.5, unit: 'с', n: 9,
      support: ['E10/scientist/hard-1002', 'E10/scientist/hard-1005', 'E10/scientist/hard-1008'],
      contradictions: ['E10/scientist/hard-1004', 'E10/scientist/hard-1006', 'E10/scientist/hard-1010', 'E10/scientist/hard-1012'], status: 'retired',
      first_seen: 'E10/scientist/hard-1002', last_seen: 'E10/scientist/hard-1012' },
  ],
};

/** «E10/scientist/hard-1001» → {file, text}: ссылка на запись прогона и короткая подпись. */
function runRef(id) {
  const raw = String(id);
  const file = /\.json\.gz$/.test(raw) ? raw : `${raw}.json.gz`;
  const parts = raw.replace(/\.json\.gz$/, '').split('/');
  const m = /^([a-z]+)-(\d+)$/.exec(parts[parts.length - 1] || '');
  const where = m ? `${levelName(m[1]).toLowerCase()} № ${m[2]}` : parts[parts.length - 1];
  const exp = parts.length > 1 ? parts[0] : '';
  return { file, text: exp && exp !== 'adhoc' ? `${exp} · ${where}` : where, title: raw };
}

function runLinks(ids, tone) {
  const list = [...new Set(ids || [])];
  if (!list.length) return null;
  const chip = (id) => {
    const r = runRef(id);
    return h('a', { class: 'lb-kb__run', href: runHref(r.file), title: `Открыть запись прогона ${r.title}`, data: { tone } }, icon('play', 12), r.text);
  };
  const head = list.slice(0, SHOWN).map(chip);
  if (list.length <= SHOWN) return h('div', { class: 'lb-kb__runs' }, head);
  const rest = h('details', { class: 'lb-kb__more' },
    h('summary', { text: `ещё ${count(list.length - SHOWN, 'прогон', 'прогона', 'прогонов')}` }),
    h('div', { class: 'lb-kb__runs' }, list.slice(SHOWN).map(chip)));
  return h('div', { class: 'lb-kb__runswrap' }, h('div', { class: 'lb-kb__runs' }, head), rest);
}

/** Когда правило видели: в данных это либо дата, либо идентификатор прогона. */
function seen(label, v) {
  if (!v) return null;
  const date = /^\d{4}-\d\d-\d\d/.test(String(v)) ? when(v) : '';
  if (date) return h('span', null, `${label} ${date}`);
  const r = runRef(v);
  return h('span', null, `${label} `, h('a', { class: 'lb-link', href: runHref(r.file), title: r.title, text: r.text }));
}

function valueText(rule) {
  if (rule.value == null || !Number.isFinite(Number(rule.value))) return null;
  const v = Number(rule.value);
  const sd = Number(rule.sigma) || 0;
  const share = /^доля/i.test(String(rule.unit || ''));
  // Знаков столько, чтобы разброс был виден: 2,49 ± 0,05, но 12 ± 9.
  const scale = share ? 100 : 1;
  const ref = Math.max(sd * scale, 0);
  const digits = share ? 0 : ref > 0 ? (ref >= 10 ? 0 : ref >= 1 ? 1 : ref >= 0.1 ? 2 : 3) : (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  const unit = share ? '%' : (rule.unit || '');
  return {
    main: `${num(v * scale, digits)}${sd > 0 ? ` ± ${num(sd * scale, digits)}` : ''}`,
    unit: share ? `% — ${ruText(rule.unit)}` : unit,
    note: sd > 0 ? 'значение ± разброс между прогонами' : 'значение',
  };
}

function ruleCard(rule) {
  const st = STATUS[rule.status] || { word: rule.status || 'Без статуса', tone: 'none', icon: 'dash', hint: '' };
  const support = [...new Set(rule.support || [])];
  const against = [...new Set(rule.contradictions || [])];
  const n = Number.isFinite(Number(rule.n)) ? Number(rule.n) : support.length;
  const val = valueText(rule);
  const text = ruText(rule.statement || rule.id || '');
  const total = support.length + against.length;
  return h('article', { class: 'lb-card lb-kb__rule', data: { status: rule.status || '' } },
    h('div', { class: 'lb-kb__top' },
      h('span', { class: 'lb-eyebrow', text: rule.kind ? ruText(rule.kind) : 'правило' }),
      h('span', { class: 'lb-badge', data: { tone: st.tone }, title: st.hint }, icon(st.icon, 15), h('span', { text: st.word }))),
    h('p', { class: 'lb-kb__statement', text: text ? text[0].toUpperCase() + text.slice(1) : '' }),
    val ? h('div', { class: 'lb-kb__value' },
      h('span', { class: 'lb-kb__num', text: val.main }),
      h('span', { class: 'lb-kb__unit', text: val.unit }),
      h('span', { class: 'lb-muted', text: val.note })) : null,
    h('div', { class: 'lb-kb__evidence' },
      h('div', { class: 'lb-kb__counts' },
        h('span', { class: 'lb-kb__count' }, h('b', { text: String(n) }), ` ${plural(n, 'наблюдение', 'наблюдения', 'наблюдений')}`),
        h('span', { class: 'lb-kb__count', data: { tone: 'ok' } }, icon('check', 14), h('b', { text: String(support.length) }), ' подтверждают'),
        h('span', { class: 'lb-kb__count', data: { tone: against.length ? 'bad' : 'none' } }, icon('cross', 14), h('b', { text: String(against.length) }), ' противоречат')),
      total ? h('div', { class: 'lb-kb__meter', role: 'img', 'aria-label': `Подтверждают ${support.length}, противоречат ${against.length}` },
        h('i', { data: { tone: 'ok' }, style: { flexGrow: String(support.length) } }),
        against.length ? h('i', { data: { tone: 'bad' }, style: { flexGrow: String(against.length) } }) : null) : null),
    support.length ? h('div', { class: 'lb-kb__block' },
      h('div', { class: 'lb-kb__label', text: 'Прогоны-доказательства' }), runLinks(support, 'ok')) : null,
    against.length ? h('div', { class: 'lb-kb__block' },
      h('div', { class: 'lb-kb__label', text: 'Прогоны, которые противоречат' }), runLinks(against, 'bad')) : null,
    !support.length && !against.length ? h('div', { class: 'lb-muted', text: 'Ссылок на отдельные прогоны у этого правила нет: оно считается по всем прогонам сразу.' }) : null,
    rule.first_seen || rule.last_seen ? h('div', { class: 'lb-kb__seen lb-muted' }, seen('Впервые:', rule.first_seen), seen('Последний раз:', rule.last_seen)) : null);
}

function tiles(data, rules) {
  const by = (s) => rules.filter((r) => r.status === s).length;
  const tile = (label, value, sub) => h('div', { class: 'lb-tile' },
    h('div', { class: 'lb-tile__label', text: label }),
    h('div', { class: 'lb-tile__value', text: value }),
    sub ? h('div', { class: 'lb-tile__sub', text: sub }) : null);
  return h('div', { class: 'lb-tiles lb-tiles--5' },
    tile('Правил', String(rules.length), 'всего'),
    tile('Подтверждено', String(by('confirmed')), STATUS.confirmed.hint),
    tile('Предварительно', String(by('tentative')), STATUS.tentative.hint),
    tile('Снято', String(by('retired')), STATUS.retired.hint),
    tile('Прогонов учтено', String(data.runs ?? '—'), data.updated ? `обновлено ${when(data.updated)}` : ''));
}

function emptyState() {
  const step = (n, title, text) => h('li', null, h('span', { class: 'lb-kb__stepn', text: String(n) }), h('div', null, h('div', { class: 'lb-kb__steptitle', text: title }), h('div', { class: 'lb-muted', text })));
  return h('div', { class: 'lb-card lb-kb__empty' },
    h('h2', { class: 'lb-h2', text: 'Знаний пока нет' }),
    h('p', { class: 'lb-lead', text: 'Они появятся после первых прогонов агента-исследователя. Страницу обновлять не нужно: она сама проверяет, не появилось ли новое.' }),
    h('div', { class: 'lb-eyebrow', text: 'Что здесь будет' }),
    h('ol', { class: 'lb-kb__steps' },
      step(1, 'Правило', 'Утверждение о среде простыми словами: например, сколько заряда стоит метр пути или как долго длится сбой датчика.'),
      step(2, 'Значение и разброс', 'Число, которое агент измерил, и насколько оно гуляет от прогона к прогону.'),
      step(3, 'Доказательства', 'Сколько прогонов правило подтвердили, сколько ему противоречат — и ссылки на записи этих прогонов.'),
      step(4, 'Статус', 'Подтверждено, предварительно или снято. Знание — это предположение: новый прогон может его опровергнуть.')),
    h('div', { class: 'lb-empty__actions lb-kb__actions' },
      h('a', { class: 'lb-btn lb-btn--primary', href: href('/knowledge', { demo: 1 }) }, 'Показать пример', icon('right')),
      h('a', { class: 'lb-btn', href: href('/scenarios') }, 'Запустить прогон на странице сценариев', icon('right'))));
}

export async function render(root, ctx) {
  const demo = ctx.query.demo != null && ctx.query.demo !== '0';
  root.append(loading('Загружаю знания…'));
  const state = { status: '', json: '' };
  const body = h('div', { class: 'lb-stack' });
  let timer = null;

  const load = async () => {
    if (demo) return DEMO;
    try {
      return (await api.get('/api/knowledge', 1)) || { rules: [], runs: 0 };
    } catch (e) {
      // Сервер старой версии ещё не знает этого адреса — это не ошибка, а «знаний пока нет».
      if (e.status === 404) return { rules: [], runs: 0, missing: true };
      throw e;
    }
  };

  function paint(data) {
    const rules = (Array.isArray(data.rules) ? data.rules : []).slice()
      .sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status) || String(a.kind || '').localeCompare(String(b.kind || ''), 'ru') || String(a.id).localeCompare(String(b.id)));
    if (!rules.length) {
      fill(body, emptyState());
      return;
    }
    const present = ORDER.filter((s) => rules.some((r) => r.status === s));
    if (state.status && !present.includes(state.status)) state.status = '';
    const list = h('div', { class: 'lb-kb__list' });
    const counter = h('span', { class: 'lb-muted' });
    const options = [{ id: '', label: 'Все' }, ...present.map((s) => ({ id: s, label: STATUS[s].word }))];
    const btns = options.map((o) => h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(o.id === state.status), text: o.label }));
    const show = () => {
      const shown = rules.filter((r) => !state.status || r.status === state.status);
      fill(list, shown.map(ruleCard));
      counter.textContent = `Показано ${shown.length} из ${rules.length}`;
      btns.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].id === state.status)));
    };
    btns.forEach((b, i) => b.addEventListener('click', () => { state.status = options[i].id; show(); }));
    show();
    fill(body,
      data.demo ? h('div', { class: 'lb-alert', data: { tone: 'warn' }, role: 'status' },
        h('div', { class: 'lb-alert__icon' }, icon('info', 20)),
        h('div', { class: 'lb-alert__body' },
          h('div', { class: 'lb-alert__title', text: 'Это пример, а не настоящие знания' }),
          h('div', { class: 'lb-alert__text', text: 'Он показывает, как будет выглядеть страница. Ссылки на прогоны в примере никуда не ведут.' }),
          h('div', { class: 'lb-alert__actions' }, h('a', { class: 'lb-btn', href: href('/knowledge') }, 'Показать настоящие данные')))) : null,
      tiles(data, rules),
      h('div', { class: 'lb-toolbar' },
        present.length > 1 ? h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Статус правила' }, btns) : h('span'),
        counter),
      list);
  }

  async function refresh(first) {
    clearTimeout(timer);
    let data;
    try {
      data = await load();
    } catch (e) {
      if (!ctx.alive()) return;
      if (first) fill(body, errorBox('Не удалось получить знания', e.message, () => refresh(true)));
      timer = setTimeout(() => refresh(first), 5000);
      return;
    }
    if (!ctx.alive()) return;
    const json = JSON.stringify(data);
    if (json !== state.json) {
      state.json = json;
      paint(data);
    }
    if (!demo) timer = setTimeout(() => refresh(false), 6000);
  }

  fill(root,
    h('header', { class: 'lb-intro lb-intro--tight' },
      h('div', { class: 'lb-eyebrow', text: 'Память между прогонами' }),
      h('h1', { class: 'lb-h1', text: 'Что робот узнал о среде' }),
      h('p', { class: 'lb-lead', text: 'После каждого прогона агент-исследователь записывает то, что проверил опытом: сколько стоит метр пути, как ведут себя сбои. Следующий прогон начинает с этих знаний как с предположений — и может их опровергнуть.' })),
    body);
  body.append(loading('Загружаю знания…'));
  ctx.onLeave(() => clearTimeout(timer));
  await refresh(true);
}
