// Страница опыта: вопрос, гипотеза, способ проверки → вывод по утверждениям → сравнение вариантов → прогоны.

import {
  h, icon, badge, dot, loading, errorBox, emptyBox, select, getExperiment, jobs, href, runHref, compareHref, go,
  num, count, plural, when, KINDS, REASONS, EVENT_NAMES, LEVEL_ORDER, levelName, metricInfo, metricValue, armColors,
  hypothesisWord, tip, backendName, condName, ruLevels,
} from './common.js';
import { stripChart, diffChart, legend } from '../charts.js';

const BASE_COND = { id: 'base', label: 'Базовые правила' };

function model(data) {
  const spec = data.spec || {};
  const arms = spec.arms || [];
  const conds = (spec.conditions && spec.conditions.length ? spec.conditions : [BASE_COND]).map((c) => ({ ...c, label: condName(c) }));
  const levels = spec.levels || [];
  const byCond = conds.length > 1;
  // Несколько условий и несколько уровней сразу: строки — условия, а уровень выбирается переключателем.
  const pickLevel = byCond && levels.length > 1;
  const view = { level: 'all' };
  const stats = new Map();
  for (const g of data.groups || []) stats.set(`${g.arm}|${g.condition}|${g.level}`, g);
  const runIndex = new Map();
  for (const r of data.runs || []) runIndex.set(`${r.arm}|${r.condition}|${r.level}|${r.seed}`, r);
  const infos = new Map();
  const info = (metric) => {
    if (infos.has(metric)) return infos.get(metric);
    let max = null;
    for (const g of data.groups || []) {
      const st = g.stats && g.stats[metric];
      if (st && Number.isFinite(st.max)) max = Math.max(max ?? 0, Math.abs(st.max), Math.abs(st.min ?? 0));
    }
    const bool = (data.runs || []).some((r) => r.metrics && typeof r.metrics[metric] === 'boolean');
    const made = metricInfo(metric, data.metrics, { max, bool });
    infos.set(metric, made);
    return made;
  };
  return {
    data, spec, arms, conds, levels, byCond, pickLevel, view,
    /** Строки сравнения: уровни (если условие одно) либо условия на выбранном уровне. */
    groups() {
      if (!byCond) return levels.map((l) => ({ id: l, label: levelName(l), cond: conds[0].id, level: l }));
      const level = levels.length > 1 ? view.level : levels[0];
      return conds.map((c) => ({ id: c.id, label: c.label, cond: c.id, level }));
    },
    total: !byCond && levels.length > 1 ? { id: 'all', label: 'Все уровни вместе', cond: conds[0].id, level: 'all', strong: true } : null,
    colors: armColors(arms),
    armLabel: (id) => (arms.find((a) => a.id === id) || {}).label || id,
    condLabel: (id) => (conds.find((c) => c.id === id) || {}).label || id,
    statOf: (arm, g, metric) => {
      const x = stats.get(`${arm}|${g.cond}|${g.level}`);
      return (x && x.stats && x.stats[metric]) || null;
    },
    runsOf: (arm, g) => (data.runs || []).filter((r) => r.arm === arm && r.condition === g.cond && (g.level === 'all' || r.level === g.level)),
    twin: (run, arm) => runIndex.get(`${arm}|${run.condition}|${run.level}|${run.seed}`),
    info,
    describe: (run) => `${levelName(run.level)} уровень, сценарий ${run.seed}${byCond ? ` · ${(conds.find((c) => c.id === run.condition) || {}).label || run.condition}` : ''}`,
  };
}

// --- шапка и пересчёт --------------------------------------------------------------------------

function rerun(m, ctx) {
  const { data, spec } = m;
  const id = spec.id;
  if (spec.manual) {
    // Такие опыты (прогоны в Gazebo) идут по одному из терминала; сводка обновляется сама.
    return h('div', { class: 'lb-rerun lb-manual' },
      h('div', { class: 'lb-manual__title', text: 'Новый прогон запускается командой в терминале' }),
      h('pre', { class: 'lb-code', text: String(spec.manual) }),
      h('div', { class: 'lb-rerun__hint', text: 'Когда прогон закончится, он появится в этом опыте: обновите страницу.' }),
      h('button', { class: 'lb-btn', type: 'button', onclick: ctx.reload }, icon('redo'), 'Обновить'));
  }
  const input = h('input', { class: 'lb-input lb-input--num', type: 'number', min: 1, max: 500, step: 1, value: data.seeds || spec.seeds || 20, id: 'lb-seeds' });
  const btn = h('button', { class: 'lb-btn lb-btn--primary', type: 'button' }, icon('redo'), data.status === 'not_run' ? 'Запустить серию' : 'Пересчитать серию');
  const fill = h('i');
  const bar = h('div', { class: 'lb-progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100 }, fill);
  const text = h('div', { class: 'lb-rerun__text' });
  const box = h('div', { class: 'lb-rerun__run', hidden: true }, bar, text);
  const problem = h('div', { class: 'lb-rerun__problem', role: 'alert', hidden: true });
  const planned = () => m.arms.length * m.conds.length * m.levels.length * Math.max(1, Number(input.value) || 0);
  const hint = h('div', { class: 'lb-rerun__hint' });
  const paintHint = () => { hint.textContent = `Всего будет ${count(planned(), 'прогон', 'прогона', 'прогонов')} в быстром симуляторе`; };
  input.addEventListener('input', paintHint);
  paintHint();

  const paint = () => {
    const job = jobs.running(id);
    btn.disabled = !!job;
    input.disabled = !!job;
    box.hidden = !job;
    if (!job) return;
    const all = job.total || 0;
    const share = all ? Math.min(1, job.done / all) : 0;
    bar.classList.toggle('lb-progress--wait', !all);
    fill.style.width = `${(share * 100).toFixed(1)}%`;
    bar.setAttribute('aria-valuenow', Math.round(share * 100));
    text.textContent = all ? `Посчитано ${job.done} из ${all} прогонов` : 'Серия запускается…';
  };
  paint();
  ctx.onLeave(jobs.subscribe((_, finished) => {
    if (finished.some((j) => j.experiment === id)) ctx.reload();
    else paint();
  }));
  btn.addEventListener('click', async () => {
    problem.hidden = true;
    btn.disabled = true;
    try {
      const seeds = Math.max(1, Math.min(500, Math.round(Number(input.value) || 0))) || undefined;
      await jobs.start(id, seeds);
    } catch (e) {
      problem.hidden = false;
      problem.textContent = `Не удалось запустить: ${e.message}`;
      btn.disabled = false;
    }
    paint();
  });
  return h('div', { class: 'lb-rerun' },
    h('div', { class: 'lb-rerun__row' },
      h('label', { class: 'lb-field', for: 'lb-seeds' }, h('span', { text: 'Сценариев на уровень' }), input),
      btn),
    hint, box, problem);
}

function head(m, ctx) {
  const { data, spec } = m;
  const meta = [];
  if (data.status === 'own' && data.generated) meta.push(`пересчитан ${when(data.generated)}`);
  if (data.runs && data.runs.length) {
    meta.push(count(data.runs.length, 'прогон', 'прогона', 'прогонов'));
    if (data.seeds && !spec.manual) meta.push(`${count(data.seeds, 'сценарий', 'сценария', 'сценариев')} на уровень`);
    if (data.wall_s && !spec.manual) meta.push(`посчитано за ${num(data.wall_s, 1)} с`);
    if (data.generated) meta.push(when(data.generated));
  }
  return h('header', { class: 'lb-exphead' },
    h('div', { class: 'lb-exphead__main' },
      h('a', { class: 'lb-back', href: href('/') }, icon('left'), 'Все опыты'),
      h('div', { class: 'lb-eyebrow', text: `Опыт ${spec.id} · ${KINDS[spec.kind] || 'Опыт'}` }),
      h('h1', { class: 'lb-h1', text: ruLevels(spec.title || spec.id) }),
      h('div', { class: 'lb-exphead__status' },
        badge(data.status, { big: true, word: hypothesisWord(data.status) }),
        meta.length ? h('span', { class: 'lb-muted', text: meta.join(' · ') }) : null)),
    rerun(m, ctx));
}

function alerts(m) {
  const { data, spec } = m;
  const out = [];
  const last = jobs.latest(spec.id);
  if (last && last.state === 'failed' && (!data.generated || last.started * 1000 > Date.parse(data.generated))) {
    out.push(h('div', { class: 'lb-alert', data: { tone: 'bad' }, role: 'alert' },
      h('div', { class: 'lb-alert__icon' }, icon('warn', 20)),
      h('div', { class: 'lb-alert__body' },
        h('div', { class: 'lb-alert__title', text: 'Последний пересчёт серии не дошёл до конца' }),
        h('div', { class: 'lb-alert__text', text: data.runs && data.runs.length ? 'Ниже показаны результаты предыдущего удачного пересчёта. Что напечатала программа серии перед остановкой:' : 'Программа серии остановилась с ошибкой. Что она напечатала перед остановкой:' }),
        h('pre', { class: 'lb-code lb-code--log', text: (last.lines || []).join('\n') || 'Вывода нет.' }))));
  }
  const errors = data.errors || [];
  if (errors.length) {
    const kinds = new Map();
    for (const e of errors) {
      if (!kinds.has(e.error)) kinds.set(e.error, []);
      kinds.get(e.error).push(e);
    }
    const all = errors.length + (data.runs ? data.runs.length : 0);
    out.push(h('div', { class: 'lb-alert', data: { tone: 'bad' }, role: 'alert' },
      h('div', { class: 'lb-alert__icon' }, icon('warn', 20)),
      h('div', { class: 'lb-alert__body' },
        h('div', { class: 'lb-alert__title', text: `${plural(errors.length, 'Упал', 'Упало', 'Упало')} ${count(errors.length, 'прогон', 'прогона', 'прогонов')} из ${all}` }),
        h('div', { class: 'lb-alert__text', text: 'Упавшие прогоны в сравнение не попали. Причины:' }),
        h('ul', { class: 'lb-errors' }, [...kinds.entries()].slice(0, 6).map(([msg, list]) => {
          const where = [...new Set(list.map((e) => m.armLabel(e.arm)))].join(', ');
          return h('li', null,
            h('code', { class: 'lb-code lb-code--inline', text: msg }),
            h('div', { class: 'lb-muted', text: `${count(list.length, 'прогон', 'прогона', 'прогонов')} · варианты: ${where}` }));
        })))));
  }
  return out;
}

// --- научная часть -----------------------------------------------------------------------------

function science(m) {
  const { spec } = m;
  const part = (title, text) => (text ? h('div', { class: 'lb-science__part' }, h('div', { class: 'lb-eyebrow', text: title }), h('p', { text: ruLevels(text) })) : null);
  const seeds = m.data.seeds || spec.seeds;
  const setup = [];
  if (m.arms.length) {
    setup.push(h('div', { class: 'lb-setup__row' }, h('span', { class: 'lb-setup__name', text: 'Варианты агента' }),
      h('span', { class: 'lb-setup__chips' }, m.arms.map((a) => h('span', { class: 'lb-chip' }, dot(m.colors.get(a.id)), a.label)))));
  }
  if (m.byCond) {
    setup.push(h('div', { class: 'lb-setup__row' }, h('span', { class: 'lb-setup__name', text: 'Условия' }),
      h('span', { class: 'lb-setup__chips' }, m.conds.map((c) => h('span', { class: 'lb-chip lb-chip--plain', text: c.label })))));
  }
  setup.push(h('div', { class: 'lb-setup__row' }, h('span', { class: 'lb-setup__name', text: 'Сценарии' }),
    h('span', { text: `${m.levels.length > 1 ? 'уровни' : 'уровень'}: ${m.levels.map((l) => levelName(l).toLowerCase()).join(', ')} · по ${count(seeds, 'сценарию', 'сценария', 'сценариев')} на уровень, одни и те же для всех вариантов` })));
  return h('section', { class: 'lb-card lb-science' },
    h('div', { class: 'lb-science__top' },
      h('div', { class: 'lb-science__q' }, h('div', { class: 'lb-eyebrow', text: 'Вопрос' }), h('p', { class: 'lb-science__big', text: ruLevels(spec.question || '') })),
      h('div', { class: 'lb-science__h' }, h('div', { class: 'lb-eyebrow', text: 'Гипотеза' }), h('p', { class: 'lb-science__big', text: ruLevels(spec.hypothesis || '') }))),
    h('div', { class: 'lb-science__parts' },
      part('Как проверяем', spec.method), part('Чего ждём', spec.expect), part('Что опровергнет', spec.refute)),
    h('div', { class: 'lb-setup' }, setup));
}

// --- утверждения -------------------------------------------------------------------------------

function claimsSection(m) {
  const claims = m.data.claims || [];
  if (!claims.length) return null;
  const groups = m.groups();
  const rowsDef = m.total ? [...groups, m.total] : groups;
  const cards = claims.map((c) => {
    const info = m.info(c.metric);
    const better = c.better || info.better;
    const rows = [];
    for (const g of rowsDef) {
      const cell = (c.cells || []).find((x) => x.condition === g.cond && x.level === g.level);
      if (cell) rows.push({ label: g.label, strong: !!g.strong, pair: cell.pair, verdict: cell.verdict });
    }
    if (!rows.length) {
      for (const cell of c.cells || []) rows.push({ label: `${m.condLabel(cell.condition)} · ${levelName(cell.level)}`, pair: cell.pair, verdict: cell.verdict });
    }
    return h('article', { class: 'lb-card lb-claim' },
      h('div', { class: 'lb-claim__head' },
        h('h3', { class: 'lb-h3', text: ruLevels(c.text || info.label) }),
        badge(c.status, { big: true })),
      h('div', { class: 'lb-claim__sub' },
        h('span', { class: 'lb-chip' }, dot(m.colors.get(c.a)), m.armLabel(c.a)),
        h('span', { class: 'lb-muted', text: 'минус' }),
        h('span', { class: 'lb-chip' }, dot(m.colors.get(c.b)), m.armLabel(c.b)),
        h('span', { class: 'lb-muted', text: `· ${info.label.toLowerCase()}: ${better === 'higher' ? 'больше — лучше' : 'меньше — лучше'}` })),
      diffChart({ info, better, rows, aLabel: m.armLabel(c.a), bLabel: m.armLabel(c.b) }));
  });
  const key = (status, text) => h('li', null, badge(status), h('span', { text }));
  return h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Что показали данные' }),
      h('span', { class: 'lb-muted', text: 'Разность считаем на одинаковых сценариях: тот же уровень и тот же номер у обоих вариантов' })),
    h('ul', { class: 'lb-readme' },
      key('supported', 'весь 95% интервал на стороне утверждения'),
      key('refuted', 'весь интервал на противоположной стороне'),
      key('inconclusive', 'интервал накрывает ноль')),
    cards);
}

// --- метрики -----------------------------------------------------------------------------------

function meansTable(m, metrics) {
  const groups = m.groups();
  const rowsDef = m.total ? [...groups, m.total] : groups;
  const infos = metrics.map((id) => m.info(id));
  const body = [];
  for (const g of rowsDef) {
    m.arms.forEach((a, i) => {
      body.push(h('tr', { class: g.strong ? 'lb-table__strong' : null },
        i === 0 ? h('th', { scope: 'rowgroup', rowspan: m.arms.length, class: 'lb-table__group', text: g.label }) : null,
        h('td', null, h('span', { class: 'lb-cellarm' }, dot(m.colors.get(a.id)), a.label)),
        infos.map((info) => {
          const st = m.statOf(a.id, g, info.id);
          return h('td', { class: 'lb-table__num' }, st
            ? [h('span', { text: info.mean(st.mean) }), h('span', { class: 'lb-table__ci', text: ` от ${info.mean(st.ci[0]).replace(/\s%$/, '')} до ${info.mean(st.ci[1]).replace(/\s%$/, '')}` })]
            : '—');
        })));
    });
  }
  return h('div', { class: 'lb-tablewrap' },
    h('table', { class: 'lb-table' },
      h('caption', { class: 'lb-table__caption', text: 'Среднее и 95% интервал среднего' }),
      h('thead', null, h('tr', null,
        h('th', { scope: 'col', text: m.byCond ? 'Условие' : 'Уровень' }),
        h('th', { scope: 'col', text: 'Вариант' }),
        infos.map((info) => h('th', { scope: 'col', class: 'lb-table__num', text: info.title })))),
      h('tbody', null, body)));
}

function metricsSection(m, ctx) {
  const metrics = [...new Set(m.spec.metrics || [])];
  if (!metrics.length || !(m.data.runs || []).length) return null;
  const rowLabels = m.arms.length > 2;
  const arms = m.arms.map((a) => ({ label: a.label, color: m.colors.get(a.id) }));
  const grid = h('div', { class: 'lb-charts' });
  const charts = [];
  for (const id of metrics) {
    const info = m.info(id);
    const all = m.groups().map((g) => ({
      label: g.label,
      rows: m.arms.map((a) => ({
        arm: a.id,
        label: a.label,
        color: m.colors.get(a.id),
        stat: m.statOf(a.id, g, id),
        points: m.runsOf(a.id, g).map((run) => ({ v: metricValue(run, id), run })).filter((p) => p.v != null),
      })),
    }));
    // Метрика бывает только у части вариантов (расследования ведёт один исследователь) — пустые строки убираем.
    const filled = new Set(all.flatMap((g) => g.rows.filter((r) => r.stat || r.points.length).map((r) => r.arm)));
    if (!filled.size) continue;
    const groups = all.map((g) => ({ label: g.label, rows: g.rows.filter((r) => filled.has(r.arm)) }));
    const partial = filled.size < m.arms.length;
    const host = h('div', { class: 'lb-chart__plot' });
    grid.append(h('figure', { class: 'lb-card lb-chart' },
      h('figcaption', { class: 'lb-chart__head' },
        h('span', { class: 'lb-chart__title', text: info.title }),
        h('span', { class: 'lb-chart__hint', text: info.betterText })),
      host,
      partial ? h('div', { class: 'lb-chart__note', text: `У остальных вариантов этой величины нет: ${m.arms.filter((a) => !filled.has(a.id)).map((a) => `«${a.label}»`).join(', ')}` }) : null));
    charts.push({ host, cfg: { info, groups, rowLabels: rowLabels || partial, describe: m.describe, onPick: (run) => go(runHref(run.file)), aria: `${info.title}: сравнение вариантов агента` } });
  }
  const table = meansTable(m, metrics);
  table.hidden = true;
  const tabs = [
    h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': 'true', text: 'Графики' }),
    h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': 'false', text: 'Таблица' }),
  ];
  tabs.forEach((b, i) => b.addEventListener('click', () => {
    tabs.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
    grid.hidden = i !== 0;
    table.hidden = i !== 1;
  }));
  const section = h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Сравнение вариантов по метрикам' }),
      h('span', { class: 'lb-muted', text: 'Мелкая точка — один прогон (щелчок открывает его). Крупная точка — среднее, черта — его 95% интервал' })),
    h('div', { class: 'lb-toolbar' }, legend(arms), h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Вид' }, tabs)),
    grid, table);
  // Графики измеряют свою ширину, поэтому рисуются после вставки страницы в документ.
  const live = [];
  const mount = () => {
    for (const c of charts) live.push(stripChart(c.host, c.cfg));
  };
  const destroy = () => { for (const ch of live.splice(0)) ch.destroy(); };
  ctx.onLeave(destroy);
  return { section, mount, destroy };
}

// --- обнаружение изменений ---------------------------------------------------------------------

function median(values) {
  const v = values.slice().sort((a, b) => a - b);
  const k = Math.floor(v.length / 2);
  return v.length % 2 ? v[k] : (v[k - 1] + v[k]) / 2;
}

function detectSection(m) {
  const runs = m.data.runs || [];
  const seenTypes = new Set();
  for (const r of runs) for (const t of Object.keys((r.metrics && r.metrics.detect) || {})) seenTypes.add(t);
  // Сначала известные события в привычном порядке, затем новые — под своими именами.
  const types = [...Object.keys(EVENT_NAMES).filter((t) => seenTypes.has(t)), ...[...seenTypes].filter((t) => !(t in EVENT_NAMES))];
  if (!types.length) return null;
  const cell = (arm, type) => {
    const vals = runs.filter((r) => r.arm === arm && r.metrics.detect && type in r.metrics.detect).map((r) => r.metrics.detect[type]);
    if (!vals.length) return h('td', { class: 'lb-muted', text: 'события не было' });
    const seen = vals.filter((v) => v != null);
    const share = seen.length / vals.length;
    return h('td', null,
      h('div', { class: 'lb-detect' },
        h('div', { class: 'lb-detect__meter', role: 'img', 'aria-label': `заметил в ${seen.length} из ${vals.length}` },
          h('i', { style: { width: `${(share * 100).toFixed(1)}%` } })),
        h('div', null,
          h('div', { class: 'lb-detect__main', text: `заметил в ${seen.length} из ${vals.length}` }),
          h('div', { class: 'lb-muted', text: seen.length ? `обычно через ${num(median(seen), 1)} с` : 'ни разу' }))));
  };
  return h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Замечает ли агент изменения среды' }),
      h('span', { class: 'lb-muted', text: 'Считаем прогоны, где событие успело случиться. Время — от события до тревоги агента; «обычно» — в половине случаев быстрее' })),
    h('div', { class: 'lb-tablewrap' },
      h('table', { class: 'lb-table lb-table--roomy' },
        h('thead', null, h('tr', null,
          h('th', { scope: 'col', text: 'Вариант' }),
          types.map((t) => h('th', { scope: 'col', text: EVENT_NAMES[t] || t })))),
        h('tbody', null, m.arms.map((a) => h('tr', null,
          h('th', { scope: 'row' }, h('span', { class: 'lb-cellarm' }, dot(m.colors.get(a.id)), a.label)),
          types.map((t) => cell(a.id, t))))))));
}

// --- расследования агента ----------------------------------------------------------------------

/** Сводка расследований по вариантам: сколько раз агент нашёл причину странности и сколько раз был прав. */
function inquiriesSection(m) {
  const runs = (m.data.runs || []).filter((r) => r.metrics && r.metrics.inquiries && typeof r.metrics.inquiries === 'object');
  if (!runs.length) return null;
  const KEYS = [
    { id: 'correct', label: 'причина названа верно', tone: 'ok' },
    { id: 'wrong', label: 'причина названа неверно', tone: 'bad' },
    { id: 'unverifiable', label: 'причина названа, проверить нечем', tone: 'none' },
    { id: 'insufficient', label: 'честно: данных недостаточно', tone: 'info' },
  ];
  const rows = m.arms.map((a) => {
    const mine = runs.filter((r) => r.arm === a.id);
    if (!mine.length) return null;
    const sum = (k) => mine.reduce((acc, r) => acc + (Number(r.metrics.inquiries[k]) || 0), 0);
    const total = sum('total');
    const parts = KEYS.map((k) => ({ ...k, n: sum(k.id) }));
    // В старых сводках нет разбивки «верно / неверно» — тогда показываем «причина найдена» одним куском.
    const known = parts.reduce((acc, x) => acc + x.n, 0);
    if (known < total) parts.splice(2, 0, { id: 'identified', label: 'причина названа', tone: 'none', n: Math.max(0, sum('identified') - sum('correct') - sum('wrong') - sum('unverifiable')) });
    return { arm: a, runs: mine.length, total, tests: sum('tests'), energy: sum('energy'), parts: parts.filter((x) => x.n > 0) };
  }).filter(Boolean);
  if (!rows.length || !rows.some((r) => r.total)) return null;
  const bar = (row) => h('div', { class: 'lb-inq__bar', role: 'img', 'aria-label': row.parts.map((x) => `${x.label}: ${x.n}`).join('; ') },
    row.parts.map((x) => h('i', { data: { tone: x.tone }, style: { flexGrow: String(x.n) }, title: `${x.label}: ${x.n}` })));
  return h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Расследования агента' }),
      h('span', { class: 'lb-muted', text: 'Агент замечает странность, выдвигает объяснения и ставит опыт. Судья знает настоящую причину и сверяет с ней вывод' })),
    h('ul', { class: 'lb-readme' }, KEYS.map((k) => h('li', null, h('i', { class: 'lb-inq__key', data: { tone: k.tone } }), h('span', { text: k.label })))),
    h('div', { class: 'lb-tablewrap' },
      h('table', { class: 'lb-table lb-table--roomy lb-table--wraphead' },
        h('thead', null, h('tr', null,
          h('th', { scope: 'col', text: 'Вариант' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Расследований' }),
          h('th', { scope: 'col', text: 'Чем закончились' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Верно' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Неверно' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Данных недостаточно' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Опытов на одно расследование' }),
          h('th', { scope: 'col', class: 'lb-table__num', text: 'Заряд на опыты за прогон' }))),
        h('tbody', null, rows.map((r) => {
          const n = (id) => (r.parts.find((x) => x.id === id) || { n: 0 }).n;
          return h('tr', null,
            h('th', { scope: 'row' }, h('span', { class: 'lb-cellarm' }, dot(m.colors.get(r.arm.id)), r.arm.label)),
            h('td', { class: 'lb-table__num' }, String(r.total), h('span', { class: 'lb-table__ci', text: ` в ${count(r.runs, 'прогоне', 'прогонах', 'прогонах')}` })),
            h('td', null, r.total ? bar(r) : h('span', { class: 'lb-muted', text: 'странностей не было' })),
            h('td', { class: 'lb-table__num', text: String(n('correct')) }),
            h('td', { class: 'lb-table__num', text: String(n('wrong')) }),
            h('td', { class: 'lb-table__num', text: String(n('insufficient')) }),
            h('td', { class: 'lb-table__num', text: r.total ? num(r.tests / r.total, 1) : '—' }),
            h('td', { class: 'lb-table__num', text: `${num(r.energy / r.runs, 2)} ед.` }));
        })))));
}

// --- таблица прогонов --------------------------------------------------------------------------

function runsSection(m) {
  const runs = m.data.runs || [];
  if (!runs.length) return null;
  const armPos = new Map(m.arms.map((a, i) => [a.id, i]));
  const condPos = new Map(m.conds.map((c, i) => [c.id, i]));
  const state = { sort: 'scenario', dir: 1, arm: '', level: '', cond: '', back: '', vs: m.arms[0] ? m.arms[0].id : '' };

  const numCol = (id, label, digits) => ({ key: id, label, num: true, value: (r) => metricValue(r, id), cell: (r) => num(metricValue(r, id), digits) });
  const cols = [
    { key: 'arm', label: 'Вариант', value: (r) => armPos.get(r.arm), cell: (r) => h('span', { class: 'lb-cellarm' }, dot(m.colors.get(r.arm)), m.armLabel(r.arm)) },
    m.byCond ? { key: 'cond', label: 'Условие', value: (r) => condPos.get(r.condition), cell: (r) => m.condLabel(r.condition) } : null,
    { key: 'level', label: 'Уровень', value: (r) => LEVEL_ORDER.indexOf(r.level), cell: (r) => levelName(r.level) },
    { key: 'scenario', label: 'Сценарий', num: true, value: (r) => LEVEL_ORDER.indexOf(r.level) * 1e6 + r.seed, cell: (r) => `№ ${r.seed}` },
    new Set(runs.map((r) => r.backend)).size > 1
      ? { key: 'backend', label: 'Где шёл', value: (r) => (r.backend === 'gazebo' ? 1 : 0), cell: (r) => backendName(r.backend) } : null,
    numCol('score', 'Счёт', 1),
    { key: 'samples', label: 'Образцы', num: true, value: (r) => r.metrics.samples_share, cell: (r) => `${r.metrics.samples_collected} из ${r.metrics.samples_total}` },
    {
      key: 'returned', label: 'Вернулся', value: (r) => (r.metrics.returned ? 1 : 0),
      cell: (r) => h('span', { class: `lb-yn lb-yn--${r.metrics.returned ? 'yes' : 'no'}`, title: REASONS[r.metrics.reason] || '' },
        icon(r.metrics.returned ? 'check' : 'cross', 14), r.metrics.returned ? 'да' : `нет: ${REASONS[r.metrics.reason] || 'не дошёл'}`),
    },
    numCol('battery_used', 'Потрачено заряда', 1),
    numCol('penalties', 'Штрафы', 0),
    numCol('time', 'Время, с', 0),
  ].filter(Boolean);
  const have = new Set(['score', 'samples_share', 'returned', 'battery_used', 'penalties', 'time']);
  for (const id of m.spec.metrics || []) {
    if (have.has(id)) continue;
    have.add(id);
    const info = m.info(id);
    cols.push(numCol(id, info.kind === 'num' && info.unit ? `${info.label}, ${info.unit}` : info.label, info.kind === 'count' ? 0 : 1));
  }

  const tbody = h('tbody');
  const counter = h('span', { class: 'lb-muted' });
  const heads = cols.map((c) => {
    const btn = h('button', { class: 'lb-sort', type: 'button' }, h('span', { text: c.label }), h('span', { class: 'lb-sort__arrow', 'aria-hidden': 'true' }));
    const th = h('th', { scope: 'col', class: c.num ? 'lb-table__num' : null }, btn);
    btn.addEventListener('click', () => {
      if (state.sort === c.key) state.dir = -state.dir;
      else { state.sort = c.key; state.dir = c.num && c.key !== 'scenario' ? -1 : 1; }
      paint();
    });
    return { c, th, btn };
  });

  function pair(run) {
    const other = m.arms.length === 2
      ? m.arms.find((a) => a.id !== run.arm)
      : m.arms.find((a) => a.id === state.vs && a.id !== run.arm) || m.arms.find((a) => a.id !== run.arm);
    const twin = other ? m.twin(run, other.id) : null;
    if (!twin) return null;
    const ordered = armPos.get(run.arm) <= armPos.get(twin.arm) ? [run, twin] : [twin, run];
    return { other, url: compareHref(ordered[0].file, ordered[1].file) };
  }

  function paint() {
    const col = cols.find((c) => c.key === state.sort) || cols[0];
    const list = runs.filter((r) => (!state.arm || r.arm === state.arm) && (!state.level || r.level === state.level)
      && (!state.cond || r.condition === state.cond) && (!state.back || String(!!r.metrics.returned) === state.back));
    list.sort((a, b) => {
      const va = col.value(a);
      const vb = col.value(b);
      const d = (va == null ? -Infinity : va) - (vb == null ? -Infinity : vb);
      return d * state.dir || LEVEL_ORDER.indexOf(a.level) - LEVEL_ORDER.indexOf(b.level) || a.seed - b.seed
        || condPos.get(a.condition) - condPos.get(b.condition) || armPos.get(a.arm) - armPos.get(b.arm);
    });
    for (const x of heads) {
      const on = x.c.key === state.sort;
      x.th.setAttribute('aria-sort', on ? (state.dir > 0 ? 'ascending' : 'descending') : 'none');
      x.btn.lastChild.textContent = on ? (state.dir > 0 ? '↑' : '↓') : '';
    }
    tbody.replaceChildren(...list.map((r) => {
      const p = pair(r);
      return h('tr', null,
        cols.map((c) => h('td', { class: c.num ? 'lb-table__num' : null }, c.cell(r))),
        h('td', { class: 'lb-table__actions' },
          h('a', { class: 'lb-link', href: runHref(r.file) }, icon('play', 14), 'Смотреть'),
          p ? h('a', { class: 'lb-link', href: p.url, title: `Рядом с вариантом «${p.other.label}» на этом же сценарии` }, icon('pair', 14), 'Сравнить') : null));
    }));
    counter.textContent = `Показано ${list.length} из ${runs.length}`;
  }

  const filters = [
    h('label', { class: 'lb-field' }, h('span', { text: 'Вариант' }),
      select([{ value: '', label: 'все' }, ...m.arms.map((a) => ({ value: a.id, label: a.label }))], '', (v) => { state.arm = v; paint(); })),
    m.levels.length > 1 ? h('label', { class: 'lb-field' }, h('span', { text: 'Уровень' }),
      select([{ value: '', label: 'все' }, ...m.levels.map((l) => ({ value: l, label: levelName(l) }))], '', (v) => { state.level = v; paint(); })) : null,
    m.byCond ? h('label', { class: 'lb-field' }, h('span', { text: 'Условие' }),
      select([{ value: '', label: 'все' }, ...m.conds.map((c) => ({ value: c.id, label: c.label }))], '', (v) => { state.cond = v; paint(); })) : null,
    h('label', { class: 'lb-field' }, h('span', { text: 'Возврат на базу' }),
      select([{ value: '', label: 'любой' }, { value: 'true', label: 'вернулся' }, { value: 'false', label: 'не вернулся' }], '', (v) => { state.back = v; paint(); })),
    m.arms.length > 2 ? h('label', { class: 'lb-field' }, h('span', { text: '«Сравнить» — с вариантом' }),
      select(m.arms.map((a) => ({ value: a.id, label: a.label })), state.vs, (v) => { state.vs = v; paint(); })) : null,
  ];
  paint();
  return h('section', { class: 'lb-section' },
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Все прогоны' }),
      h('span', { class: 'lb-muted', text: '«Сравнить» ставит рядом два варианта на одном и том же сценарии' })),
    h('div', { class: 'lb-toolbar lb-toolbar--filters' }, filters, counter),
    h('div', { class: 'lb-tablewrap lb-tablewrap--tall' },
      h('table', { class: 'lb-table lb-table--runs' },
        h('thead', null, h('tr', null, heads.map((x) => x.th), h('th', { scope: 'col' }, h('span', { class: 'lb-sr', text: 'Действия' })))),
        tbody)));
}

// --- страница ----------------------------------------------------------------------------------

export async function render(root, ctx) {
  const id = ctx.params[0];
  root.append(loading(`Загружаю опыт ${id}…`));
  let data;
  try {
    [data] = await Promise.all([getExperiment(id), jobs.refresh()]);
  } catch (e) {
    root.replaceChildren(
      h('a', { class: 'lb-back', href: href('/') }, icon('left'), 'Все опыты'),
      errorBox(`Не удалось открыть опыт ${id}`, e.message, ctx.reload));
    return;
  }
  if (!ctx.alive()) return;
  if (!data.spec.id) data.spec.id = id;
  const m = model(data);
  const hasRuns = (data.runs || []).length > 0;
  const dataHost = h('div', { class: 'lb-stack' });
  let metrics = null;
  const paintData = () => {
    if (metrics) metrics.destroy();
    metrics = hasRuns ? metricsSection(m, ctx) : null;
    dataHost.replaceChildren(...[
      hasRuns ? claimsSection(m) : null,
      metrics ? metrics.section : null,
    ].filter(Boolean));
    if (metrics) metrics.mount();
  };
  // Опыт с несколькими условиями и несколькими уровнями: уровень выбирается здесь, строки графиков — условия.
  let levelSwitch = null;
  if (hasRuns && m.pickLevel) {
    const options = [{ id: 'all', label: 'Все уровни вместе' }, ...m.levels.map((l) => ({ id: l, label: levelName(l) }))];
    const btns = options.map((o) => h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(o.id === m.view.level), text: o.label }));
    btns.forEach((b, i) => b.addEventListener('click', () => {
      m.view.level = options[i].id;
      btns.forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
      tip.hide();
      paintData();
    }));
    levelSwitch = h('div', { class: 'lb-toolbar lb-toolbar--filters' },
      h('span', { class: 'lb-switch__name', text: 'Уровень сценариев' }),
      h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Уровень сценариев' }, btns),
      h('span', { class: 'lb-muted', text: 'Строки ниже — условия опыта на выбранном уровне' }));
  }
  root.replaceChildren(...[
    head(m, ctx),
    ...alerts(m),
    science(m),
    hasRuns ? null : data.status === 'own' ? emptyBox('Итоги этого опыта — в отчёте',
      `Опыт считает своя программа, поэтому графиков сравнения здесь нет. Таблицы и выводы: ${data.report || 'research/findings'}; числа: ${data.summary}.`) : emptyBox(
      data.status === 'not_run' ? 'Серию ещё не запускали' : 'В серии нет ни одного удачного прогона',
      data.spec.manual
        ? 'Этот опыт запускается командой из терминала — она показана вверху страницы.'
        : data.status === 'not_run'
          ? 'Нажмите «Запустить серию» вверху страницы: прогоны идут в быстром симуляторе и обычно занимают 5–30 секунд.'
          : 'Посмотрите причины выше, исправьте и пересчитайте серию.'),
    levelSwitch,
    hasRuns ? dataHost : null,
    hasRuns ? inquiriesSection(m) : null,
    hasRuns ? detectSection(m) : null,
    runsSection(m),
  ].filter(Boolean));
  if (hasRuns) paintData();
  ctx.onLeave(() => tip.hide());
}
