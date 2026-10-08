// Один прогон: шапка с итогом, переходы по сценариям и вариантам, проигрыватель.

import {
  h, fill, icon, dot, loading, errorBox, emptyBox, select, getTrace, getArena, getOptions, getIndex, getExperiment, runEpisode,
  href, runHref, compareHref, go, num, count, REASONS, EVENT_NAMES, levelName, armColors, agentColor, agentLabel, byScenario,
  condName, ruLevels, rulesOf,
} from './common.js';
import { mountPlayer, destroyPlayer } from './player.js';
import { applyDemo, demoKinds } from './demo.js';

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

/**
 * Запись команды роботов глазами одного из них: его путь, журнал и карты — на месте обычных полей записи,
 * остальные роботы — в partners (проигрыватель рисует их поверх). События судьи берутся у всех сразу:
 * образец, собранный напарником, исчезает с арены и здесь. База — место этого робота (home), поправка
 * положения и сканы — его собственные; в записях, где у роботов их нет, они есть только у первого
 * (верхний уровень записи — это он), у остальных пусто, а не чужое.
 */
export function robotView(trace, k) {
  const r = trace.robots[k];
  const events = trace.robots.flatMap((x) => x.events || []).slice().sort((a, b) => a.t - b.t);
  const own = (key, none) => r[key] ?? (k === 0 ? trace[key] : none);
  const scenario = Array.isArray(r.home) && trace.scenario ? { ...trace.scenario, base: r.home } : trace.scenario;
  return {
    ...trace, scenario, track: r.track, modes: r.modes, events, journal: r.journal, hypotheses: r.hypotheses, plans: r.plans,
    paths: r.paths, belief: r.belief, soil: r.soil, hazards: r.hazards, scans: own('scans', []), pose_fix: own('pose_fix'),
    self: r.name, partners: trace.robots.filter((_, i) => i !== k).map((x) => ({ name: x.name, track: x.track })),
  };
}

/** Плитки с итогом команды и строка по каждому роботу. */
export function teamTiles(trace) {
  const r = trace.result || {};
  const robots = trace.robots || [];
  const tile = (label, value, sub) => h('div', { class: 'lb-tile' },
    h('div', { class: 'lb-tile__label', text: label }),
    h('div', { class: 'lb-tile__value' }, value),
    sub ? h('div', { class: 'lb-tile__sub', text: sub }) : null);
  const back = robots.filter((x) => x.result && x.result.returned).length;
  const said = (trace.team && trace.team.messages) || [];
  const talk = said.filter((m) => m.type !== 'obs').length;
  const fines = [];
  if (r.collisions) fines.push(`столкновения: ${r.collisions}`);
  if (r.false_collects) fines.push(`ложные сборы: ${r.false_collects}`);
  if (r.hazard_hits) fines.push(`опасные зоны: ${r.hazard_hits}`);
  // Показателя в записи может не быть (запись Gazebo без итога судьи на двоих): это «не измерено», а не ноль.
  const measured = (v, unit) => {
    const known = v != null && !Number.isNaN(Number(v));
    const tone = !known ? '' : Number(v) ? ' lb-yn--no' : ' lb-yn--yes';
    return h('span', { class: `lb-chip lb-chip--plain${unit ? '' : tone}`, text: known ? `${num(v, 0)}${unit || ''}` : 'не измерено' });
  };
  // Зазор без min_gap_source — из старых записей, где он считался по несинхронным кадрам: не показываем.
  const gap = r.min_gap_m != null && r.min_gap_source;
  return h('div', null,
    h('div', { class: 'lb-tiles' },
      tile('Счёт команды', num(r.score, 1), `на одного робота ${num(r.score_per_robot, 1)}`),
      tile('Образцы', `${r.samples_collected ?? '—'} из ${r.samples_total ?? '—'}`,
        robots.map((x) => `${x.name}: ${x.result.samples_collected}`).join(' · ')),
      tile('Вернулись на базу', `${back} из ${robots.length}`, robots.map((x) => `${x.name}: заряд ${num(x.result.battery, 1)}`).join(' · ')),
      tile('Последний сбор', r.t_last_collect != null ? `${num(r.t_last_collect, 0)} с` : '—', `весь прогон ${num(r.time, 0)} с`),
      tile('Сообщений', said.length ? num(talk, 0) : 'нет', said.length ? `и ${said.length - talk} с показаниями датчиков` : 'роботы не связаны'),
      tile('Штрафы', num(r.penalties ?? 0, 0), fines.join(' · ') || 'нет')),
    h('div', { class: 'lb-detects' },
      h('span', { class: 'lb-muted', text: 'Цели в планах совпадали:' }),
      measured(r.same_target_s, ' с'),
      h('span', { class: 'lb-muted', text: 'Столкновений друг с другом:' }),
      measured(r.robot_contacts),
      gap ? h('span', { class: 'lb-muted', text: 'Ближе всего между центрами:' }) : null,
      gap ? h('span', { class: 'lb-chip lb-chip--plain', text: `${num(r.min_gap_m, 2)} м` }) : null));
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

/** Расследования агента в этом прогоне: сколько, чем закончились, что сказал судья. */
export function inquiryChips(trace, onShow) {
  const list = Array.isArray(trace.inquiries) ? trace.inquiries : [];
  const sum = trace.result && trace.result.inquiries && typeof trace.result.inquiries === 'object' ? trace.result.inquiries : null;
  const by = (status) => list.filter((q) => q && q.conclusion && q.conclusion.status === status).length;
  const total = sum && Number.isFinite(Number(sum.total)) ? Number(sum.total) : list.length;
  if (!total) return null;
  const identified = sum && sum.identified != null ? Number(sum.identified) : by('identified');
  const insufficient = sum && sum.insufficient != null ? Number(sum.insufficient) : by('insufficient');
  const chip = (text, tone, ic) => h('span', { class: `lb-chip lb-chip--plain${tone ? ` lb-yn--${tone}` : ''}` }, ic ? icon(ic, 14) : null, text);
  const chips = [
    chip(`причина найдена: ${identified}`, identified ? 'yes' : '', identified ? 'check' : null),
    chip(`недостаточно данных: ${insufficient}`, '', insufficient ? 'question' : null),
  ];
  if (sum && (sum.correct != null || sum.wrong != null)) {
    const wrong = Number(sum.wrong) || 0;
    chips.push(chip(`судья: верных выводов ${Number(sum.correct) || 0}, ошибочных ${wrong}`, wrong ? 'no' : 'yes', wrong ? 'cross' : 'check'));
  }
  if (sum && Number.isFinite(Number(sum.energy))) chips.push(chip(`на опыты ушло ${num(sum.energy, 2)} ед. заряда`));
  return h('div', { class: 'lb-detects' },
    h('span', { class: 'lb-muted', text: `${count(total, 'расследование', 'расследования', 'расследований')} агента:` }),
    chips,
    list.length && onShow ? h('button', { class: 'lb-btn lb-btn--small', type: 'button', onclick: onShow }, 'Показать расследования', icon('right', 14)) : null);
}

/** Знания из прошлых прогонов, с которыми агент начал этот прогон. Вид поля ещё не устоялся — читаем терпимо. */
export function knowledgeChips(k) {
  if (k == null || k === false) return null;
  const LABEL = {
    used: 'взято из памяти', applied: 'применено', priors: 'исходных оценок', rules: 'правил', added: 'новых', new: 'новых',
    learned: 'записано новых', updated: 'уточнено', confirmed: 'подтвердилось', contradicted: 'опровергнуто', retired: 'снято',
    tentative: 'предварительных', runs: 'прогонов в памяти',
  };
  const parts = [];
  if (typeof k === 'number') parts.push(`правил: ${k}`);
  else if (Array.isArray(k)) parts.push(`правил: ${k.length}`);
  else if (typeof k === 'object') {
    for (const [key, v] of Object.entries(k)) {
      const n = typeof v === 'number' ? v : Array.isArray(v) ? v.length : v && typeof v === 'object' ? Object.keys(v).length : null;
      if (n == null) continue;
      parts.push(`${LABEL[key] || key.replace(/_/g, ' ')}: ${num(n, Number.isInteger(n) ? 0 : 2)}`);
    }
  } else if (k === true) parts.push('агент начал прогон со знаниями из памяти');
  if (!parts.length) return null;
  return h('div', { class: 'lb-detects' },
    h('span', { class: 'lb-muted', text: 'Знания из прошлых прогонов:' }),
    parts.map((t) => h('span', { class: 'lb-chip lb-chip--plain', text: t })),
    h('a', { class: 'lb-link', href: href('/knowledge') }, 'Открыть знания', icon('right', 14)));
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
  // Примеры новых данных для отладки вида: …&demo=inquiry, …&demo=pose (lab/views/demo.js).
  const demo = demoKinds(ctx.query);
  if (demo.length) trace = applyDemo(trace, demo);
  const rules = rulesOf(trace);
  // Прогон команды: показываем глазами выбранного робота (…&robot=tb2), напарники рисуются поверх.
  const team = Array.isArray(trace.robots) && trace.robots.length > 1 ? trace : null;
  const who = team ? Math.max(0, team.robots.findIndex((r) => r.name === ctx.query.robot)) : 0;
  if (team) trace = robotView(team, who);

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
    back = h('a', { class: 'lb-back', href: href(`/exp/${exp.spec.id}`) }, icon('left'), `Опыт ${exp.spec.id}: ${ruLevels(exp.spec.title || '')}`);
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
    label = team ? ({ team: 'Два робота с координацией', pair: 'Два робота без координации', pair_lidar: 'Два робота без связи, объезд по лидару' }[team.team.mode] || 'Два робота') : agentLabel(agentId, options);
    color = agentColor(agentId);
    back = h('a', { class: 'lb-back', href: href('/scenarios', { level: sc.level, seed: sc.seed }) }, icon('left'), 'Сценарии');
    eyebrow = 'Отдельный прогон';
    const jump = (seed, text, ic, first) => {
      const btn = h('button', { class: 'lb-btn', type: 'button', disabled: seed < 1 }, first ? icon(ic) : null, text, first ? null : icon(ic));
      btn.addEventListener('click', async () => {
        busy(btn, true);
        try { go(runHref((await runEpisode(sc.level, seed, agentId, rules)).file)); } catch (e) { fail(e); busy(btn, false); }
      });
      return btn;
    };
    if (sc.level && sc.seed != null && agentId) {
      nav.push(jump(sc.seed - 1, 'Предыдущий', 'left', true), h('span', { class: 'lb-muted lb-runnav__pos', text: 'сценарий' }), jump(sc.seed + 1, 'Следующий', 'right', false));
      const agents = (options && options.agents ? options.agents : []).map((a) => ({ id: a.id, label: agentLabel(a.id, options) }));
      // Сервер мог быть запущен до появления этого варианта — тогда добавляем его в список сами.
      if (agents.length && !agents.some((a) => a.id === agentId)) agents.unshift({ id: agentId, label: agentLabel(agentId, options) });
      if (agents.length) {
        const other = agents.find((a) => a.id !== agentId && (a.id === 'fixed' || a.id === 'adaptive')) || agents.find((a) => a.id !== agentId);
        const sel = select(agents.map((a) => ({ value: a.id, label: a.label })), agentId, async (v) => {
          if (v === agentId) return;
          sel.disabled = true;
          try { go(runHref((await runEpisode(sc.level, sc.seed, v, rules)).file)); } catch (e) { fail(e); sel.disabled = false; }
        });
        switches.push(h('label', { class: 'lb-switch' }, h('span', { class: 'lb-switch__name', text: 'Вариант на этом сценарии' }), sel));
        if (other) {
          const vs = select(agents.filter((a) => a.id !== agentId).map((a) => ({ value: a.id, label: a.label })), other.id, () => {});
          const btn = h('button', { class: 'lb-btn', type: 'button' }, icon('pair'), 'Сравнить');
          btn.addEventListener('click', async () => {
            busy(btn, true);
            try { go(compareHref(file, (await runEpisode(sc.level, sc.seed, vs.value, rules)).file)); } catch (e) { fail(e); busy(btn, false); }
          });
          switches.push(h('div', { class: 'lb-switch' }, h('span', { class: 'lb-switch__name', text: 'Сравнить с вариантом' }), vs, btn));
        }
      }
    }
  }

  const result = trace.result || (run && run.metrics) || {};
  const stage = h('div', { class: 'lb-stage' });
  // Открыть запись сразу на нужном месте: …&t=44.1 (секунды) или …&q=Q2 (вывод расследования).
  let startAt;
  if (ctx.query.t != null && ctx.query.t !== '' && Number.isFinite(Number(ctx.query.t))) startAt = Number(ctx.query.t);
  else if (ctx.query.q) {
    const q = (trace.inquiries || []).find((x) => x && String(x.id) === String(ctx.query.q));
    if (q) startAt = Number(q.t_close ?? q.t_open);
  }
  const showInquiries = () => {
    const el = stage.querySelector('.rp-inq');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
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
      rules ? h('span', { class: 'lb-chip lb-chip--plain', title: 'Заряд уходит ещё на повороты и вес образцов; штраф в опасной зоне может вызвать утечку заряда или сбой датчика' }, 'Усложнённые правила: сбои и несколько причин расхода') : null,
      sc.level && sc.seed != null ? h('a', { class: 'lb-link', href: href('/scenarios', { level: sc.level, seed: sc.seed, rules }) }, 'Что спрятано на этой арене') : null),
    problem,
    demo.length ? h('div', { class: 'lb-alert', data: { tone: 'warn' }, role: 'status' },
      h('div', { class: 'lb-alert__icon' }, icon('info', 20)),
      h('div', { class: 'lb-alert__body' },
        h('div', { class: 'lb-alert__title', text: 'В запись подмешан пример для проверки вида страницы' }),
        h('div', { class: 'lb-alert__text', text: `${[demo.includes('inquiry') ? 'расследования и сводка по ним' : '', demo.includes('pose') ? 'поправка положения по лидару' : ''].filter(Boolean).join(', ')} — не из этого прогона. Сама запись на диске не меняется.` }),
        h('div', { class: 'lb-alert__actions' }, h('a', { class: 'lb-btn', href: runHref(file) }, 'Показать запись без примера')))) : null,
    team ? teamTiles(team) : resultTiles(result, trace.rules),
    team ? h('div', { class: 'lb-switches' }, h('div', { class: 'lb-switch' },
      h('span', { class: 'lb-switch__name', text: 'Журнал, план и карта — глазами робота' }),
      h('div', { class: 'lb-seg', role: 'group' }, team.robots.map((r, i) => h('a', {
        class: 'lb-seg__btn', href: href('/run', { file, robot: r.name }), 'aria-current': i === who ? 'true' : null, text: r.name,
      }))))) : null,
    detectChips(result),
    inquiryChips(trace, showInquiries),
    knowledgeChips(result.knowledge ?? trace.knowledge),
    stage);

  const player = await mountPlayer(stage, { trace, arena, color, startAt });
  ctx.onLeave(() => destroyPlayer(player));
}
