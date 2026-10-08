// Два прогона рядом на одном сценарии.

import {
  h, fill, icon, dot, loading, errorBox, emptyBox, getTrace, getArena, getOptions, getIndex, getExperiment,
  href, runHref, compareHref, num, REASONS, levelName, armColors, agentColor, agentLabel, byScenario, PALETTE, condName,
} from './common.js';
import { mountPair, destroyPlayer } from './player.js';

const ROWS = [
  { label: 'Счёт', get: (r) => r.score, show: (v) => num(v, 1), better: 'higher' },
  { label: 'Образцы', get: (r) => r.samples_collected, show: (v, r) => `${v ?? '—'} из ${r.samples_total ?? '—'}`, better: 'higher' },
  { label: 'Вернулся на базу', get: (r) => (r.returned ? 1 : 0), show: (v, r) => (v ? 'да' : `нет: ${REASONS[r.reason] || 'не дошёл'}`), better: 'higher' },
  { label: 'Осталось заряда', get: (r) => r.battery_left, show: (v) => num(v, 1), better: 'higher' },
  { label: 'Штрафы', get: (r) => r.penalties ?? 0, show: (v) => num(v, 0), better: 'lower' },
  { label: 'Время, с', get: (r) => r.time ?? r.t, show: (v) => num(v, 0), better: 'lower' },
];

function summary(results, labels, colors, files) {
  const better = (row, i) => {
    const a = row.get(results[i]);
    const b = row.get(results[1 - i]);
    if (a == null || b == null || a === b) return false;
    return row.better === 'higher' ? a > b : a < b;
  };
  return h('div', { class: 'lb-tablewrap' },
    h('table', { class: 'lb-table lb-table--roomy lb-cmp' },
      h('caption', { class: 'lb-table__caption', text: 'Итог каждого прогона. Галочкой отмечено лучшее из двух значений.' }),
      h('thead', null, h('tr', null,
        h('th', { scope: 'col', text: 'Вариант' }),
        ROWS.map((r) => h('th', { scope: 'col', class: 'lb-table__num', text: r.label })),
        h('th', { scope: 'col' }, h('span', { class: 'lb-sr', text: 'Действия' })))),
      h('tbody', null, results.map((res, i) => h('tr', null,
        h('th', { scope: 'row' }, h('span', { class: 'lb-cellarm' }, dot(colors[i]), labels[i])),
        ROWS.map((row) => {
          const win = better(row, i);
          return h('td', { class: `lb-table__num${win ? ' lb-cmp__win' : ''}`, title: win ? 'лучше, чем у второго варианта' : null },
            win ? h('span', { class: 'lb-cmp__mark' }, icon('check', 13), h('span', { class: 'lb-sr', text: 'лучше: ' })) : null,
            row.show(row.get(res), res));
        }),
        h('td', { class: 'lb-table__actions' }, h('a', { class: 'lb-link', href: runHref(files[i]) }, icon('play', 14), 'Отдельно')))))));
}

export async function render(root, ctx) {
  const { a, b } = ctx.query;
  if (!a || !b) {
    root.append(emptyBox('Нечего сравнивать', 'Выберите пару прогонов кнопкой «Сравнить» в таблице опыта или на странице сценариев.',
      h('a', { class: 'lb-btn', href: href('/') }, 'К опытам'), h('a', { class: 'lb-btn', href: href('/scenarios') }, 'К сценариям')));
    return;
  }
  root.append(loading('Загружаю два прогона…'));
  let ta;
  let tb;
  let arena;
  let options = null;
  let index = null;
  try {
    [ta, tb, arena, options, index] = await Promise.all([
      getTrace(a), getTrace(b), getArena(), getOptions().catch(() => null), getIndex().catch(() => null),
    ]);
  } catch (e) {
    root.replaceChildren(
      h('a', { class: 'lb-back', href: href('/') }, icon('left'), 'Все опыты'),
      errorBox('Не удалось открыть записи прогонов', `${e.message}. Возможно, серию только что пересчитали.`, ctx.reload));
    return;
  }
  let exp = null;
  if (ta.experiment && ta.experiment === tb.experiment && index && (index.experiments || []).some((x) => x.id === ta.experiment)) {
    try { exp = await getExperiment(ta.experiment); } catch { exp = null; }
  }
  if (!ctx.alive()) return;

  const files = [a, b];
  const traces = [ta, tb];
  const scs = traces.map((t) => t.scenario || {});
  const runs = exp ? files.map((f) => (exp.runs || []).find((r) => r.file === f)) : [null, null];
  let labels;
  let colors;
  let back;
  const nav = [];
  if (exp && runs[0] && runs[1]) {
    const arms = exp.spec.arms || [];
    const conds = (exp.spec.conditions || []).map((c) => ({ ...c, label: condName(c) }));
    const palette = armColors(arms);
    const armLabel = (r) => (arms.find((x) => x.id === r.arm) || {}).label || r.arm;
    const condLabel = (r) => (conds.find((x) => x.id === r.condition) || {}).label || r.condition;
    labels = runs.map((r) => (runs[0].arm === runs[1].arm ? `${armLabel(r)} · ${condLabel(r)}` : armLabel(r)));
    colors = runs.map((r) => palette.get(r.arm) || PALETTE[0]);
    back = h('a', { class: 'lb-back', href: href(`/exp/${exp.spec.id}`) }, icon('left'), `Опыт ${exp.spec.id}: ${exp.spec.title || ''}`);
    // Соседние сценарии для той же пары вариантов.
    const find = (r, lvl, seed) => (exp.runs || []).find((x) => x.arm === r.arm && x.condition === r.condition && x.level === lvl && x.seed === seed);
    const same = (exp.runs || []).filter((r) => r.arm === runs[0].arm && r.condition === runs[0].condition).sort(byScenario);
    const pos = same.findIndex((r) => r.file === a);
    const step = (r, text, ic, first) => {
      const twin = r ? find(runs[1], r.level, r.seed) : null;
      return r && twin
        ? h('a', { class: 'lb-btn', href: compareHref(r.file, twin.file) }, first ? icon(ic) : null, text, first ? null : icon(ic))
        : h('span', { class: 'lb-btn lb-btn--off', 'aria-disabled': 'true' }, first ? icon(ic) : null, text, first ? null : icon(ic));
    };
    nav.push(step(same[pos - 1], 'Предыдущий', 'left', true),
      h('span', { class: 'lb-muted lb-runnav__pos', text: `сценарий ${pos + 1} из ${same.length}` }),
      step(same[pos + 1], 'Следующий', 'right', false));
  } else {
    const ids = traces.map((t) => (t.agent && t.agent.name) || t.arm || '');
    labels = ids.map((id) => agentLabel(id, options));
    colors = ids.map((id) => agentColor(id));
    back = h('a', { class: 'lb-back', href: href('/scenarios', { level: scs[0].level, seed: scs[0].seed }) }, icon('left'), 'Сценарии');
  }
  if (labels[0] === labels[1]) labels = labels.map((l, i) => `${l} (${i === 0 ? 'слева' : 'справа'})`);
  if (colors[0] === colors[1]) colors = [PALETTE[0], PALETTE[1]];

  const sameScenario = scs[0].level === scs[1].level && scs[0].seed === scs[1].seed;
  const stage = h('div', { class: 'lb-stage' });
  fill(root,
    h('header', { class: 'lb-runhead' },
      h('div', { class: 'lb-runhead__main' },
        back,
        h('div', { class: 'lb-eyebrow', text: 'Сравнение двух прогонов' }),
        h('h1', { class: 'lb-h1', text: sameScenario ? `${levelName(scs[0].level)} уровень, сценарий ${scs[0].seed}` : 'Два разных сценария' }),
        h('div', { class: 'lb-runhead__arm' },
          h('span', { class: 'lb-chip lb-chip--big' }, dot(colors[0]), labels[0]),
          h('span', { class: 'lb-muted', text: 'и' }),
          h('span', { class: 'lb-chip lb-chip--big' }, dot(colors[1]), labels[1]),
          h('a', { class: 'lb-link', href: compareHref(b, a) }, 'Поменять местами'))),
      nav.length ? h('nav', { class: 'lb-runnav', 'aria-label': 'Соседние сценарии' }, nav) : null),
    sameScenario ? null : h('div', { class: 'lb-alert', data: { tone: 'warn' } },
      h('div', { class: 'lb-alert__icon' }, icon('info', 20)),
      h('div', { class: 'lb-alert__body' },
        h('div', { class: 'lb-alert__title', text: 'Прогоны сняты на разных сценариях' }),
        h('div', { class: 'lb-alert__text', text: `Слева — ${levelName(scs[0].level).toLowerCase()} уровень, сценарий ${scs[0].seed}; справа — ${levelName(scs[1].level).toLowerCase()} уровень, сценарий ${scs[1].seed}. Числа напрямую сравнивать нельзя.` }))),
    summary(traces.map((t) => t.result || {}), labels, colors, files),
    stage);

  const player = await mountPair(stage, { traces, labels, colors, arena });
  ctx.onLeave(() => destroyPlayer(player));
}
