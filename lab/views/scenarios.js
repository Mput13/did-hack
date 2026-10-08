// Показ генератора сценариев: уровень и номер → что спрятано на арене и что изменится по ходу прогона.

import {
  h, fill, icon, loading, errorBox, select, getArena, getOptions, getScenario, runEpisode, href, runHref, compareHref,
  go, num, count, LEVEL_ORDER, levelName, EVENT_NAMES, multLabel, agentLabel,
} from './common.js';
import { createArena, arenaLegend } from './arena.js';

const MAX_SEED = 9999;

/** Мир после первых k событий (k = 0 — на старте). */
function worldAfter(sc, k) {
  let soils = sc.soils || [];
  let hazards = (sc.hazards || []).slice();
  (sc.events || []).slice(0, k).forEach((ev) => {
    if (ev.type === 'soil_change' && ev.soils) soils = ev.soils;
    if (ev.type === 'new_hazard' && ev.zone) hazards = hazards.concat([ev.zone]);
  });
  return { soils, hazards };
}

function sameShape(a, b) {
  return a.shape === b.shape && Math.hypot(a.x - b.x, a.y - b.y) < 0.02 && Math.abs(a.r - b.r) < 0.01
    && Math.abs(a.w - b.w) < 0.01 && Math.abs(a.h - b.h) < 0.01;
}

/** Что именно меняет событие номер i: слова для ленты и подсветка для арены. */
function eventChange(sc, i, rules) {
  const ev = sc.events[i];
  const before = worldAfter(sc, i);
  const after = worldAfter(sc, i + 1);
  const out = { text: '', ghostSoils: [], freshSoils: [], freshHazards: [], moves: [], banner: '' };
  if (ev.type === 'soil_change') {
    const old = new Map(before.soils.map((z) => [z.id, z]));
    const parts = [];
    for (const z of after.soils) {
      const o = old.get(z.id);
      old.delete(z.id);
      if (!o) {
        parts.push(`появится новая зона ${multLabel(z.mult)}`);
        out.freshSoils.push(z.id);
        continue;
      }
      const moved = !sameShape(o, z);
      const priced = o.mult !== z.mult;
      if (!moved && !priced) continue;
      out.freshSoils.push(z.id);
      if (moved) {
        out.ghostSoils.push(o);
        out.moves.push([o, z]);
      }
      if (moved && priced) parts.push(`зона ${multLabel(o.mult)} переедет и станет ${multLabel(z.mult)}`);
      else if (moved) parts.push(`зона ${multLabel(o.mult)} переедет на новое место`);
      else parts.push(`зона ${multLabel(o.mult)} станет ${multLabel(z.mult)}`);
    }
    for (const o of old.values()) {
      parts.push(`зона ${multLabel(o.mult)} исчезнет`);
      out.ghostSoils.push(o);
    }
    out.text = parts.length ? `${parts.join('; ')}. Робот узнает об этом только по расходу заряда.` : 'Грунты останутся прежними.';
  } else if (ev.type === 'new_hazard') {
    out.freshHazards.push(ev.zone.id);
    out.text = 'На арене появится ещё одна опасная зона. Робот узнает о ней, только когда получит штраф.';
  } else if (ev.type === 'sensor_fault') {
    const usual = rules && rules.sensor_sigma != null ? ` вместо обычных ${num(rules.sensor_sigma, 2)}` : '';
    out.text = `Датчик образцов начнёт врать на ${num(ev.duration, 1)} с: разброс показаний ${num(ev.sigma, 2)}${usual}.`;
    out.banner = `Сбой датчика: с ${num(ev.t, 1)} по ${num(ev.t + ev.duration, 1)} с показаниям нельзя верить`;
  } else {
    out.text = ev.type;
  }
  return out;
}

export async function render(root, ctx) {
  root.append(loading('Загружаю арену…'));
  let arena;
  let options;
  try {
    [arena, options] = await Promise.all([getArena(), getOptions()]);
  } catch (e) {
    root.replaceChildren(errorBox('Не удалось загрузить арену', e.message, ctx.reload));
    return;
  }
  if (!ctx.alive()) return;

  const levels = (options.levels || []).slice().sort((a, b) => LEVEL_ORDER.indexOf(a.id) - LEVEL_ORDER.indexOf(b.id));
  const agents = (options.agents || []).map((a) => ({ id: a.id, label: agentLabel(a.id, options) }));
  const state = {
    level: levels.some((l) => l.id === ctx.query.level) ? ctx.query.level : (levels.some((l) => l.id === 'hard') ? 'hard' : levels[0].id),
    seed: Math.min(MAX_SEED, Math.max(1, parseInt(ctx.query.seed, 10) || 3)),
    rules: ctx.query.rules === 'science' ? 'science' : null,       // усложнённые правила среды для пробных прогонов
    sc: null, pinned: null, token: 0,
  };
  const address = () => href('/scenarios', { level: state.level, seed: state.seed, rules: state.rules });

  // --- выбор уровня и номера ---
  const levelBtns = levels.map((l) => {
    const facts = [count(l.samples, 'образец', 'образца', 'образцов'), count(l.soils, 'зона грунта', 'зоны грунта', 'зон грунта')];
    if (l.hazards) facts.push(count(l.hazards, 'опасная зона', 'опасные зоны', 'опасных зон'));
    if (l.id === 'hard') facts.push('среда меняется');
    const btn = h('button', { class: 'lb-level', type: 'button' },
      h('span', { class: 'lb-level__name', text: levelName(l.id) }), h('span', { class: 'lb-level__facts', text: facts.join(' · ') }));
    btn.addEventListener('click', () => set(l.id, state.seed));
    return { id: l.id, btn };
  });
  const seedInput = h('input', { class: 'lb-input lb-input--num', type: 'number', min: 1, max: MAX_SEED, step: 1, value: state.seed, id: 'lb-seed', 'aria-label': 'Номер сценария' });
  seedInput.addEventListener('change', () => set(state.level, parseInt(seedInput.value, 10) || 1));
  const prevBtn = h('button', { class: 'lb-btn lb-btn--icon', type: 'button', title: 'Предыдущий номер', 'aria-label': 'Предыдущий номер', onclick: () => set(state.level, state.seed - 1) }, icon('left'));
  const nextBtn = h('button', { class: 'lb-btn lb-btn--icon', type: 'button', title: 'Следующий номер', 'aria-label': 'Следующий номер', onclick: () => set(state.level, state.seed + 1) }, icon('right'));
  const randomBtn = h('button', { class: 'lb-btn', type: 'button', onclick: () => set(state.level, 1 + Math.floor(Math.random() * 999)) }, icon('dice'), 'Случайный');

  // --- арена ---
  const view = createArena(arena, { label: 'Арена сценария' });
  const banner = h('div', { class: 'lb-arena__banner', hidden: true });
  const caption = h('div', { class: 'lb-arenacard__caption' });
  const legendHost = h('div');
  const arenaProblem = h('div');
  view.el.append(banner);

  // --- правая колонка ---
  const title = h('h2', { class: 'lb-h2' });
  const facts = h('ul', { class: 'lb-facts' });
  const eventsHost = h('div', { class: 'lb-events' });
  const thumbsHost = h('div', { class: 'lb-thumbs' });

  const has = (id) => agents.some((a) => a.id === id);
  const agentSel = select(agents.map((a) => ({ value: a.id, label: a.label })), state.rules && has('scientist') ? 'scientist' : has('adaptive') ? 'adaptive' : (agents[0] || {}).id, () => {});
  const aSel = select(agents.map((a) => ({ value: a.id, label: a.label })), agents.some((a) => a.id === 'fixed') ? 'fixed' : (agents[0] || {}).id, () => {});
  const bSel = select(agents.map((a) => ({ value: a.id, label: a.label })), state.rules && has('scientist') ? 'scientist' : has('adaptive') ? 'adaptive' : (agents[1] || agents[0] || {}).id, () => {});
  // Правила среды для пробных прогонов: обычные или усложнённые (несколько причин расхода, сбои после штрафа).
  const ruleBtns = [
    { id: null, label: 'Обычные' },
    { id: 'science', label: 'Со сбоями и скрытыми расходами' },
  ].map((r) => {
    const btn = h('button', { class: 'lb-seg__btn', type: 'button', 'aria-pressed': String(state.rules === r.id), text: r.label });
    btn.addEventListener('click', () => {
      state.rules = r.id;
      ruleBtns.forEach((x) => x.btn.setAttribute('aria-pressed', String(x.id === state.rules)));
      ruleHint.hidden = !state.rules;
      // исследователь раскрывается именно на усложнённых правилах — предлагаем его
      if (state.rules && has('scientist') && agentSel.value === 'adaptive') agentSel.value = 'scientist';
      history.replaceState(null, '', address());
    });
    return { id: r.id, btn };
  });
  const ruleHint = h('p', { class: 'lb-muted', hidden: !state.rules, text: 'Заряд уходит ещё на повороты и на вес собранных образцов, показания батареи шумят, а штраф в опасной зоне может вызвать утечку заряда или сбой датчика. На таких правилах агент-исследователь ведёт расследования.' });
  const runBtn = h('button', { class: 'lb-btn lb-btn--primary', type: 'button' }, icon('play'), 'Прогнать агента');
  const cmpBtn = h('button', { class: 'lb-btn lb-btn--primary', type: 'button' }, icon('pair'), 'Сравнить двух агентов');
  const actionProblem = h('div', { class: 'lb-rerun__problem', role: 'alert', hidden: true });
  const working = (btn, on, text) => {
    runBtn.disabled = on;
    cmpBtn.disabled = on;
    btn.classList.toggle('lb-btn--busy', on);
    actionProblem.hidden = true;
    if (text) btn.lastChild.textContent = text;
  };
  runBtn.addEventListener('click', async () => {
    working(runBtn, true);
    try {
      const res = await runEpisode(state.level, state.seed, agentSel.value, state.rules);
      go(runHref(res.file));
    } catch (e) {
      working(runBtn, false);
      actionProblem.hidden = false;
      actionProblem.textContent = `Прогон не удался: ${e.message}`;
    }
  });
  cmpBtn.addEventListener('click', async () => {
    if (aSel.value === bSel.value) {
      actionProblem.hidden = false;
      actionProblem.textContent = 'Выберите два разных варианта агента.';
      return;
    }
    working(cmpBtn, true);
    try {
      const ra = await runEpisode(state.level, state.seed, aSel.value, state.rules);
      const rb = await runEpisode(state.level, state.seed, bSel.value, state.rules);
      go(compareHref(ra.file, rb.file));
    } catch (e) {
      working(cmpBtn, false);
      actionProblem.hidden = false;
      actionProblem.textContent = `Прогон не удался: ${e.message}`;
    }
  });

  // --- показ состояния арены ---
  function show(i) {
    const sc = state.sc;
    if (!sc) return;
    const base = { samples: sc.samples, base: sc.base };
    if (i == null) {
      const w = worldAfter(sc, 0);
      view.render({ ...base, ...w });
      banner.hidden = true;
      caption.textContent = sc.events.length ? 'Арена на старте. Наведите на событие справа — покажу, что изменится.' : 'Арена на старте и до конца прогона.';
      fill(legendHost, arenaLegend(w));
    } else {
      const ch = eventChange(sc, i, options.rules);
      const w = worldAfter(sc, i + 1);
      view.render({ ...base, ...w, ghostSoils: ch.ghostSoils, freshSoils: ch.freshSoils, freshHazards: ch.freshHazards, moves: ch.moves });
      banner.hidden = !ch.banner;
      banner.textContent = ch.banner;
      caption.textContent = `Арена после события ${i + 1} (${num(sc.events[i].t, 1)} с от старта). Жирный контур — то, что изменилось.`;
      fill(legendHost, arenaLegend(w, ch.ghostSoils.length ? [h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__ghost' }), 'где зона была раньше')] : []));
    }
    eventsHost.querySelectorAll('[data-ev]').forEach((el) => {
      el.classList.toggle('lb-on', i != null && Number(el.dataset.ev) === i);
      if (el.matches('.lb-ev')) el.setAttribute('aria-pressed', String(state.pinned === Number(el.dataset.ev)));
    });
  }

  function paintEvents() {
    const sc = state.sc;
    const events = sc.events || [];
    if (!events.length) {
      const hard = levels.find((l) => l.id === 'hard');
      fill(eventsHost,
        h('div', { class: 'lb-eyebrow', text: 'Что изменится во время прогона' }),
        h('p', { class: 'lb-muted', text: 'На этом уровне среда не меняется: всё, что спрятано на старте, остаётся на месте до конца.' }),
        hard && state.level !== 'hard' ? h('button', { class: 'lb-btn', type: 'button', onclick: () => set('hard', state.seed) }, 'Открыть трудный уровень', icon('right')) : null);
      return;
    }
    const end = Math.max(...events.map((e) => e.t + (e.duration || 0)));
    const span = Math.max(100, Math.ceil((end + 8) / 20) * 20);
    const pos = (t) => `${((t / span) * 100).toFixed(2)}%`;
    const hover = (i) => ({
      onpointerenter: () => show(i),
      onpointerleave: () => show(state.pinned),
      onfocus: () => show(i),
      onblur: () => show(state.pinned),
      onclick: () => { state.pinned = state.pinned === i ? null : i; show(state.pinned ?? i); },
    });
    const ticks = [];
    for (let t = 0; t <= span; t += span > 160 ? 40 : 20) ticks.push(t);
    const strip = h('div', { class: 'lb-tl', style: { height: `${events.length * 26 + 30}px` } },
      ticks.map((t) => h('i', { class: 'lb-tl__grid', style: { left: pos(t) } })),
      events.map((ev, i) => [
        ev.duration ? h('i', { class: 'lb-tl__span', data: { ev: i }, style: { left: pos(ev.t), width: pos(ev.duration), top: `${i * 26 + 9}px` } }) : null,
        h('button', { class: 'lb-tl__pt', type: 'button', data: { ev: i }, style: { left: pos(ev.t), top: `${i * 26 + 2}px` }, 'aria-label': `Событие ${i + 1}: ${EVENT_NAMES[ev.type] || ev.type}, ${num(ev.t, 1)} с`, ...hover(i) }, String(i + 1)),
      ]),
      ticks.map((t) => h('span', { class: 'lb-tl__tick', style: { left: pos(t) }, text: t === 0 ? 'старт' : `${t} с` })));
    const list = h('ol', { class: 'lb-evlist' }, events.map((ev, i) => {
      const ch = eventChange(sc, i, options.rules);
      return h('li', null, h('button', { class: 'lb-ev', type: 'button', data: { ev: i }, 'aria-pressed': 'false', ...hover(i) },
        h('span', { class: 'lb-ev__n', text: String(i + 1) }),
        h('span', { class: 'lb-ev__body' },
          h('span', { class: 'lb-ev__head' }, h('span', { class: 'lb-ev__name', text: EVENT_NAMES[ev.type] || ev.type }), h('span', { class: 'lb-ev__t', text: `${num(ev.t, 1)} с от старта` })),
          h('span', { class: 'lb-ev__text', text: ch.text }))));
    }));
    fill(eventsHost,
      h('div', { class: 'lb-eyebrow', text: 'Что изменится во время прогона' }),
      h('p', { class: 'lb-muted', text: 'Роботу об этом не сообщают. Наведите на событие, чтобы увидеть его на арене; щелчок закрепляет показ.' }),
      strip, list);
  }

  function paintFacts() {
    const sc = state.sc;
    const mults = (sc.soils || []).map((z) => z.mult).sort((a, b) => a - b);
    const items = [count(sc.samples.length, 'образец', 'образца', 'образцов')];
    if (mults.length) items.push(`${count(mults.length, 'зона', 'зоны', 'зон')} дорогого грунта (${mults.map(multLabel).join(', ')})`);
    items.push(sc.hazards.length ? count(sc.hazards.length, 'опасная зона', 'опасные зоны', 'опасных зон') : 'опасных зон нет');
    items.push(sc.events.length ? `${count(sc.events.length, 'событие', 'события', 'событий')} по ходу прогона` : 'среда не меняется');
    title.textContent = `${levelName(sc.level)} уровень, сценарий ${sc.seed}`;
    fill(facts, items.map((t) => h('li', { text: t })));
  }

  async function paintThumbs(token) {
    const seeds = [];
    for (let k = 1; k <= 6; k++) seeds.push(((state.seed - 1 + k) % MAX_SEED) + 1);
    const cells = seeds.map((seed) => {
      const btn = h('button', { class: 'lb-thumb', type: 'button', title: `Сценарий ${seed}`, onclick: () => set(state.level, seed) },
        h('span', { class: 'lb-thumb__pic' }), h('span', { class: 'lb-thumb__name', text: `№ ${seed}` }));
      return { seed, btn };
    });
    fill(thumbsHost, cells.map((c) => c.btn));
    for (const c of cells) {
      let sc;
      try { sc = await getScenario(state.level, c.seed); } catch { continue; }
      if (token !== state.token || !ctx.alive()) return;
      const small = createArena(arena, { thumb: true, label: `Сценарий ${c.seed}` });
      small.render({ samples: sc.samples, soils: sc.soils, hazards: sc.hazards, base: sc.base });
      c.btn.firstChild.replaceChildren(small.el);
    }
  }

  async function load() {
    const token = ++state.token;
    state.pinned = null;
    levelBtns.forEach((l) => l.btn.setAttribute('aria-pressed', String(l.id === state.level)));
    seedInput.value = state.seed;
    prevBtn.disabled = state.seed <= 1;
    view.el.classList.add('lb-stale');
    let sc;
    try {
      sc = await getScenario(state.level, state.seed);
    } catch (e) {
      if (token !== state.token || !ctx.alive()) return;
      fill(arenaProblem, errorBox('Генератор не вернул сценарий', e.message, load));
      return;
    }
    if (token !== state.token || !ctx.alive()) return;
    fill(arenaProblem);
    view.el.classList.remove('lb-stale');
    state.sc = { ...sc, soils: sc.soils || [], hazards: sc.hazards || [], events: sc.events || [], samples: sc.samples || [] };
    paintFacts();
    paintEvents();
    show(null);
    paintThumbs(token);
  }

  function set(level, seed) {
    state.level = level;
    state.seed = Math.min(MAX_SEED, Math.max(1, Math.round(seed) || 1));
    history.replaceState(null, '', address());
    load();
  }

  const onKey = (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey || /^(INPUT|SELECT|TEXTAREA)$/.test((e.target && e.target.tagName) || '')) return;
    if (e.key === 'ArrowLeft' && state.seed > 1) set(state.level, state.seed - 1);
    if (e.key === 'ArrowRight') set(state.level, state.seed + 1);
  };
  window.addEventListener('keydown', onKey);
  ctx.onLeave(() => { window.removeEventListener('keydown', onKey); state.token += 1; });

  fill(root,
    h('header', { class: 'lb-intro lb-intro--tight' },
      h('div', { class: 'lb-eyebrow', text: 'Генератор сценариев' }),
      h('h1', { class: 'lb-h1', text: 'Что спрятано на арене' }),
      h('p', { class: 'lb-lead', text: 'Сценарий целиком задаётся уровнем и номером: тот же номер — та же арена. Поэтому разные варианты агента проходят в точности одинаковые сценарии, и их можно честно сравнивать.' })),
    h('div', { class: 'lb-card lb-pick' },
      h('div', { class: 'lb-pick__levels', role: 'group', 'aria-label': 'Уровень' }, levelBtns.map((l) => l.btn)),
      h('div', { class: 'lb-pick__seed' },
        h('label', { class: 'lb-pick__label', for: 'lb-seed', text: 'Номер сценария' }),
        h('div', { class: 'lb-pick__row' }, prevBtn, seedInput, nextBtn, randomBtn),
        h('div', { class: 'lb-muted lb-pick__hint', text: 'Стрелки ← → на клавиатуре листают номера' }))),
    h('div', { class: 'lb-scn' },
      h('div', { class: 'lb-card lb-arenacard' }, arenaProblem, view.el, caption, legendHost),
      h('div', { class: 'lb-scn__side' },
        h('div', { class: 'lb-card' },
          title, facts,
          h('p', { class: 'lb-muted', text: 'Робот ничего этого не видит. У него есть одно число от датчика близости к образцу, расход заряда и штрафы.' })),
        h('div', { class: 'lb-card' }, eventsHost),
        h('div', { class: 'lb-card lb-try' },
          h('div', { class: 'lb-eyebrow', text: 'Проверить на этом сценарии' }),
          h('div', { class: 'lb-switch' },
            h('span', { class: 'lb-switch__name', text: 'Правила среды' }),
            h('div', { class: 'lb-seg', role: 'group', 'aria-label': 'Правила среды' }, ruleBtns.map((x) => x.btn))),
          ruleHint,
          h('div', { class: 'lb-try__row' }, h('label', { class: 'lb-field' }, h('span', { text: 'Вариант агента' }), agentSel), runBtn),
          h('div', { class: 'lb-try__row lb-try__row--two' },
            h('label', { class: 'lb-field' }, h('span', { text: 'Первый вариант' }), aSel),
            h('label', { class: 'lb-field' }, h('span', { text: 'Второй вариант' }), bSel),
            cmpBtn),
          actionProblem,
          h('p', { class: 'lb-muted', text: 'Прогон идёт в быстром симуляторе и занимает меньше секунды.' })))),
    h('section', { class: 'lb-section' },
      h('div', { class: 'lb-section-head' },
        h('h2', { class: 'lb-h2', text: 'Следующие номера того же уровня' }),
        h('span', { class: 'lb-muted', text: 'Генератор каждый раз раскладывает образцы и зоны заново' })),
      thumbsHost));

  if (!ctx.query.level || !ctx.query.seed) history.replaceState(null, '', address());
  await load();
}
