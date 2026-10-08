// Один прогон: шапка с итогом, переходы по сценариям и вариантам, проигрыватель.

import {
  h, fill, icon, dot, loading, errorBox, emptyBox, select, getTrace, getArena, getOptions, getIndex, getExperiment, runEpisode,
  href, runHref, compareHref, go, num, REASONS, EVENT_NAMES, levelName, armColors, agentColor, agentLabel, byScenario,
  condName,
} from './common.js';
import { mountPlayer, destroyPlayer } from './player.js';

/** Плитки с итогом прогона. */
export function resultTiles(result, rules) {
  const r = result || {};
  const tile = (label, value, sub) => h('div', { class: 'lb-tile' },
    h('div', { class: 'lb-tile__label', text: label }),
    h('div', { class: 'lb-tile__value' }, value),
    sub ? h('div', { class: 'lb-tile__sub', text: sub }) : null);
  const start = rules && rules.battery_start != null ? rules.battery_start : null;
  const fines = [];
  if (r.collisions) fines.push(`столкновения: ${r.collisions}`);
  if (r.false_collects) fines.push(`ложные сборы: ${r.false_collects}`);
  if (r.hazard_hits) fines.push(`опасные зоны: ${r.hazard_hits}`);
  return h('div', { class: 'lb-tiles' },
    tile('Счёт', num(r.score, 1), 'очки судьи'),
    tile('Образцы', `${r.samples_collected ?? '—'} из ${r.samples_total ?? '—'}`, 'собрано'),
    tile('Возврат на базу', h('span', { class: `lb-yn lb-yn--${r.returned ? 'yes' : 'no'}` }, icon(r.returned ? 'check' : 'cross', 18), r.returned ? 'да' : 'нет'), r.returned ? '' : REASONS[r.reason] || ''),
    tile('Осталось заряда', num(r.battery_left, 1), start != null ? `из ${num(start, 0)}` : ''),
    tile('Время', `${num(r.time ?? r.t, 0)} с`, r.distance != null ? `путь ${num(r.distance, 1)} м` : ''),
    tile('Штрафы', num(r.penalties ?? 0, 0), fines.join(' · ') || 'нет'));
}

/** Заметил ли агент скрытые события среды. */
export function detectChips(result) {
  const d = (result && result.detect) || {};
  const keys = Object.keys(EVENT_NAMES).filter((k) => k in d);
  if (!keys.length) return null;
  return h('div', { class: 'lb-detects' },
    h('span', { class: 'lb-muted', text: 'Скрытые события:' }),
    keys.map((k) => h('span', { class: `lb-chip lb-chip--plain lb-yn--${d[k] != null ? 'yes' : 'no'}` },
      icon(d[k] != null ? 'check' : 'cross', 14),
      `${EVENT_NAMES[k].toLowerCase()} — ${d[k] != null ? `заметил через ${num(d[k], 1)} с` : 'не заметил'}`)));
}

function busy(btn, on) {
  btn.disabled = on;
  btn.classList.toggle('lb-btn--busy', on);
}

export async function render(root, ctx) {
  const file = ctx.query.file;
  if (!file) {
    root.append(emptyBox('Прогон не выбран', 'Откройте прогон из таблицы опыта или запустите агента на странице сценариев.',
      h('a', { class: 'lb-btn', href: href('/') }, 'К опытам'), h('a', { class: 'lb-btn', href: href('/scenarios') }, 'К сценариям')));
    return;
  }
  root.append(loading('Загружаю запись прогона…'));
  let trace;
  let arena;
  let options = null;
  let index = null;
  try {
    [trace, arena, options, index] = await Promise.all([
      getTrace(file), getArena(), getOptions().catch(() => null), getIndex().catch(() => null),
    ]);
  } catch (e) {
    root.replaceChildren(
      h('a', { class: 'lb-back', href: href('/') }, icon('left'), 'Все опыты'),
      errorBox('Не удалось открыть запись прогона', `${e.message}. Возможно, серию только что пересчитали и файл заменён.`, ctx.reload));
    return;
  }
  let exp = null;
  if (trace.experiment && index && (index.experiments || []).some((x) => x.id === trace.experiment)) {
    try { exp = await getExperiment(trace.experiment); } catch { exp = null; }
  }
  if (!ctx.alive()) return;

  const sc = trace.scenario || {};
  const run = exp ? (exp.runs || []).find((r) => r.file === file) : null;
  const agentId = (trace.agent && trace.agent.name) || (run && run.agent) || '';
  const problem = h('div', { class: 'lb-rerun__problem', role: 'alert', hidden: true });
  const fail = (e) => { problem.hidden = false; problem.textContent = `Не получилось: ${e.message}`; };

  let label;
  let color;
  let back;
  let eyebrow;
  const nav = [];
  const switches = [];

  if (exp && run) {
    const arms = exp.spec.arms || [];
    const conds = (exp.spec.conditions || []).map((c) => ({ ...c, label: condName(c) }));
    const colors = armColors(arms);
    const arm = arms.find((a) => a.id === run.arm) || { id: run.arm, label: run.arm };
    const cond = conds.find((c) => c.id === run.condition);
    label = arm.label;
    color = colors.get(arm.id) || agentColor(agentId);
    back = h('a', { class: 'lb-back', href: href(`/exp/${exp.spec.id}`) }, icon('left'), `Опыт ${exp.spec.id}: ${exp.spec.title || ''}`);
    eyebrow = `Прогон из опыта ${exp.spec.id}${conds.length > 1 && cond ? ` · условие «${cond.label}»` : ''}`;
    const find = (a, c, lvl, seed) => (exp.runs || []).find((r) => r.arm === a && r.condition === c && r.level === lvl && r.seed === seed);

    const same = (exp.runs || []).filter((r) => r.arm === run.arm && r.condition === run.condition).sort(byScenario);
    const pos = same.findIndex((r) => r.file === file);
    const step = (r, text, ic, first) => (r
      ? h('a', { class: 'lb-btn', href: runHref(r.file), title: `${levelName(r.level)} уровень, сценарий ${r.seed}` }, first ? icon(ic) : null, text, first ? null : icon(ic))
      : h('span', { class: 'lb-btn lb-btn--off', 'aria-disabled': 'true' }, first ? icon(ic) : null, text, first ? null : icon(ic)));
    nav.push(step(same[pos - 1], 'Предыдущий', 'left', true),
      h('span', { class: 'lb-muted lb-runnav__pos', text: `сценарий ${pos + 1} из ${same.length}` }),
      step(same[pos + 1], 'Следующий', 'right', false));

    if (arms.length > 1) {
      switches.push(h('div', { class: 'lb-switch' },
        h('span', { class: 'lb-switch__name', text: 'Вариант на этом сценарии' }),
        h('div', { class: 'lb-seg', role: 'group' }, arms.map((a) => {
          const r = find(a.id, run.condition, run.level, run.seed);
          const on = a.id === run.arm;
          return r
            ? h('a', { class: 'lb-seg__btn', href: runHref(r.file), 'aria-current': on ? 'true' : null }, dot(colors.get(a.id)), a.label)
            : h('span', { class: 'lb-seg__btn lb-seg__btn--off', title: 'Этого прогона нет' }, dot(colors.get(a.id)), a.label);
        }))));
      const others = arms.filter((a) => a.id !== run.arm).map((a) => ({ a, r: find(a.id, run.condition, run.level, run.seed) })).filter((x) => x.r);
      const pairUrl = (x) => {
        const mine = arms.indexOf(arm) <= arms.indexOf(x.a);
        return mine ? compareHref(file, x.r.file) : compareHref(x.r.file, file);
      };
      if (others.length === 1) {
        switches.push(h('a', { class: 'lb-btn', href: pairUrl(others[0]) }, icon('pair'), `Сравнить с «${others[0].a.label}»`));
      } else if (others.length > 1) {
        const sel = select(others.map((x) => ({ value: x.a.id, label: x.a.label })), others[0].a.id, () => {});
        switches.push(h('div', { class: 'lb-switch' },
          h('span', { class: 'lb-switch__name', text: 'Сравнить с вариантом' }), sel,
          h('button', { class: 'lb-btn', type: 'button', onclick: () => go(pairUrl(others.find((x) => x.a.id === sel.value))) }, icon('pair'), 'Сравнить')));
      }
    }
    if (conds.length > 1) {
      switches.push(h('div', { class: 'lb-switch' },
        h('span', { class: 'lb-switch__name', text: 'Условие' }),
        h('div', { class: 'lb-seg', role: 'group' }, conds.map((c) => {
          const r = find(run.arm, c.id, run.level, run.seed);
          return r
            ? h('a', { class: 'lb-seg__btn', href: runHref(r.file), 'aria-current': c.id === run.condition ? 'true' : null, text: c.label })
            : h('span', { class: 'lb-seg__btn lb-seg__btn--off', text: c.label });
        }))));
    }
  } else {
    // Отдельный прогон: соседние сценарии и другие варианты считаются на лету.
    label = agentLabel(agentId, options);
    color = agentColor(agentId);
    back = h('a', { class: 'lb-back', href: href('/scenarios', { level: sc.level, seed: sc.seed }) }, icon('left'), 'Сценарии');
    eyebrow = 'Отдельный прогон';
    const jump = (seed, text, ic, first) => {
      const btn = h('button', { class: 'lb-btn', type: 'button', disabled: seed < 1 }, first ? icon(ic) : null, text, first ? null : icon(ic));
      btn.addEventListener('click', async () => {
        busy(btn, true);
        try { go(runHref((await runEpisode(sc.level, seed, agentId)).file)); } catch (e) { fail(e); busy(btn, false); }
      });
      return btn;
    };
    if (sc.level && sc.seed != null && agentId) {
      nav.push(jump(sc.seed - 1, 'Предыдущий', 'left', true), h('span', { class: 'lb-muted lb-runnav__pos', text: 'сценарий' }), jump(sc.seed + 1, 'Следующий', 'right', false));
      const agents = options && options.agents ? options.agents : [];
      if (agents.length) {
        const other = agents.find((a) => a.id !== agentId && (a.id === 'fixed' || a.id === 'adaptive')) || agents.find((a) => a.id !== agentId);
        const sel = select(agents.map((a) => ({ value: a.id, label: a.label })), agentId, async (v) => {
          if (v === agentId) return;
          sel.disabled = true;
          try { go(runHref((await runEpisode(sc.level, sc.seed, v)).file)); } catch (e) { fail(e); sel.disabled = false; }
        });
        switches.push(h('label', { class: 'lb-switch' }, h('span', { class: 'lb-switch__name', text: 'Вариант на этом сценарии' }), sel));
        if (other) {
          const vs = select(agents.filter((a) => a.id !== agentId).map((a) => ({ value: a.id, label: a.label })), other.id, () => {});
          const btn = h('button', { class: 'lb-btn', type: 'button' }, icon('pair'), 'Сравнить');
          btn.addEventListener('click', async () => {
            busy(btn, true);
            try { go(compareHref(file, (await runEpisode(sc.level, sc.seed, vs.value)).file)); } catch (e) { fail(e); busy(btn, false); }
          });
          switches.push(h('div', { class: 'lb-switch' }, h('span', { class: 'lb-switch__name', text: 'Сравнить с вариантом' }), vs, btn));
        }
      }
    }
  }

  const result = trace.result || (run && run.metrics) || {};
  const stage = h('div', { class: 'lb-stage' });
  fill(root,
    h('header', { class: 'lb-runhead' },
      h('div', { class: 'lb-runhead__main' },
        back,
        h('div', { class: 'lb-eyebrow', text: eyebrow }),
        h('h1', { class: 'lb-h1' }, `${levelName(sc.level)} уровень, сценарий ${sc.seed ?? '—'}`)),
      nav.length ? h('nav', { class: 'lb-runnav', 'aria-label': 'Соседние сценарии' }, nav) : null),
    h('div', { class: 'lb-switches' },
      // В опыте текущий вариант виден в переключателе; у отдельного прогона — подписью.
      exp && run && (exp.spec.arms || []).length > 1 ? null : h('span', { class: 'lb-chip lb-chip--big' }, dot(color), label),
      switches,
      sc.level && sc.seed != null ? h('a', { class: 'lb-link', href: href('/scenarios', { level: sc.level, seed: sc.seed }) }, 'Что спрятано на этой арене') : null),
    problem,
    resultTiles(result, trace.rules),
    detectChips(result),
    stage);

  const player = await mountPlayer(stage, { trace, arena, color });
  ctx.onLeave(() => destroyPlayer(player));
}
