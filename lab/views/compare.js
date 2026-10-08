// Два прогона рядом на одном сценарии.

import {
  h, fill, icon, dot, loading, errorBox, emptyBox, getTrace, getArena, getOptions, getIndex, getExperiment,
  href, runHref, compareHref, levelName, armColors, agentColor, agentLabel, byScenario, PALETTE, condName,
} from './common.js';
import { mountPair, destroyPlayer } from './player.js';

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
          // Итоги обоих прогонов — под аренами, в таблице проигрывателя; здесь только кто с кем сравнивается.
          h('a', { class: 'lb-chip lb-chip--big lb-chip--link', href: runHref(files[0]), title: 'Открыть этот прогон отдельно' }, dot(colors[0]), labels[0], icon('right', 14)),
          h('span', { class: 'lb-muted', text: 'и' }),
          h('a', { class: 'lb-chip lb-chip--big lb-chip--link', href: runHref(files[1]), title: 'Открыть этот прогон отдельно' }, dot(colors[1]), labels[1], icon('right', 14)),
          h('a', { class: 'lb-link', href: compareHref(b, a) }, 'Поменять местами'))),
      nav.length ? h('nav', { class: 'lb-runnav', 'aria-label': 'Соседние сценарии' }, nav) : null),
    sameScenario ? null : h('div', { class: 'lb-alert', data: { tone: 'warn' } },
      h('div', { class: 'lb-alert__icon' }, icon('info', 20)),
      h('div', { class: 'lb-alert__body' },
        h('div', { class: 'lb-alert__title', text: 'Прогоны сняты на разных сценариях' }),
        h('div', { class: 'lb-alert__text', text: `Слева — ${levelName(scs[0].level).toLowerCase()} уровень, сценарий ${scs[0].seed}; справа — ${levelName(scs[1].level).toLowerCase()} уровень, сценарий ${scs[1].seed}. Числа напрямую сравнивать нельзя.` }))),
    stage);

  const player = await mountPair(stage, { traces, labels, colors, arena });
  ctx.onLeave(() => destroyPlayer(player));
}
