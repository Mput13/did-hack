import { h, fill, api, getTrace, num, loading, levelName } from './common.js';
import { mountPair, mountPlayer, destroyPlayer } from './player.js';
import { createLLMPanel } from './llm-panel.js';

// Сравнение и подробности одного прогона остаются внутри страницы демонстрации.
export async function mountComparison(host, arena, ctx, onBack, current) {
  let player = null;
  let revision = 0;
  let dead = false;
  const destroy = () => { dead = true; revision++; destroyPlayer(player); };
  const close = () => { destroy(); onBack(); };
  ctx.onLeave(destroy);
  const back = h('button', { class: 'lb-btn', onclick: close }, '← К роботу');
  const selector = h('select', { class: 'lb-select', 'aria-label': 'Сценарий сравнения' });
  const provenance = h('p', { class: 'dm-caption' });
  const toolbar = h('div', { class: 'dm-replay-tools' });
  const stage = h('div', { class: 'dm-recording' });
  const results = h('section', { class: 'lb-card dm-cycle dm-comparison-results' });
  const events = h('section', { class: 'lb-card dm-cycle' });
  const llm = createLLMPanel();
  fill(host,
    h('div', { class: 'dm-section-head' }, h('div', null,
      h('h1', { class: 'dm-title' }, 'Один агент — два способа принимать решения'),
      h('p', { class: 'dm-caption' }, 'Без LLM и с LLM. Карта, датчики и численные алгоритмы одинаковые.')), back),
    h('div', { class: 'dm-replay-tools' }, h('label', { class: 'dm-field' }, 'Сохранённый сценарий', selector)),
    provenance, toolbar, stage, llm.el, results, events);
  stage.append(loading('Загружаю записи…'));
  try {
    const catalog = await api.get('/api/demo/comparisons');
    if (dead || !ctx.alive()) return;
    const pairs = catalog.pairs || [];
    if (!pairs.length) throw new Error('Нет пары записей с одинаковыми условиями и настоящими вызовами LLM');
    pairs.forEach((p, i) => selector.append(h('option', { value: i }, `${levelName(p.level)} · №${p.seed}`)));
    const same = pairs.findIndex((p) => p.level === current?.level && p.seed === current?.seed);
    const hard = pairs.findIndex((p) => p.level === 'hard');
    selector.value = String(same >= 0 ? same : Math.max(0, hard));
    let traces = [];
    let selected = null;

    const eventLabels = { alarm: 'Наблюдение', hypothesis: 'Гипотеза', decision: 'Решение', action: 'Действие', verdict: 'Вывод', inquiry: 'Проверка' };
    function paintCycle(t, index = 1) {
      const trace = traces[index];
      if (!trace) return;
      llm.update(trace, t);
      const recent = (trace.journal || []).filter((e) => e.t <= t && eventLabels[e.kind]).slice(-4);
      fill(events, h('h3', { class: 'pl-h' }, index === 1 ? 'Решения системы с LLM' : 'Решения агента без LLM'),
        recent.length ? recent.map((e) => h('div', { class: 'dm-cycle-row' },
          h('span', { class: 'dm-caption' }, `${num(e.t, 1)} с`),
          h('strong', null, e.data?.source === 'llm' ? 'Решение LLM' : eventLabels[e.kind]),
          h('span', null, e.text))) : h('p', { class: 'dm-caption' }, 'Запустите запись, чтобы проследить решения.'));
    }
    async function show(index = null) {
      const my = ++revision;
      destroyPlayer(player); player = null;
      stage.replaceChildren(loading('Открываю прогон…'));
      const pair = selected;
      const buttons = [[null, 'Два пути рядом'], [0, 'Подробнее: без LLM'], [1, 'Подробнее: с LLM']];
      fill(toolbar, buttons.map(([i, label]) => h('button', {
        class: `lb-btn${index === i ? ' lb-btn--primary' : ''}`, disabled: true, 'aria-pressed': String(index === i), onclick: () => show(i).catch(report),
      }, label)));
      const layers = { lidar: false, beliefSamples: false, beliefSoil: false, beliefHazards: false, ring: false, path: false };
      const mounted = index == null
        ? await mountPair(stage, { traces, labels: pair.labels, colors: ['#2a78d6', '#eb6834'], arena, layers })
        : await mountPlayer(stage, { trace: traces[index], arena, compact: true, viewSwitch: false, layers });
      if (dead || !ctx.alive() || my !== revision) { destroyPlayer(mounted); return; }
      player = mounted;
      toolbar.querySelectorAll('button').forEach((b) => { b.disabled = false; });
      selector.disabled = false;
      const idx = index ?? 1;
      paintCycle(0, idx);
      player.controller.onTime?.((t) => paintCycle(t, idx));
    }
    async function load() {
      selector.disabled = true;
      const my = ++revision;
      destroyPlayer(player); player = null;
      selected = pairs[Number(selector.value)];
      provenance.textContent = `Записи быстрого симулятора · ${selected.model} · сценарий №${selected.seed}. Эти записи не используют точки и зоны, добавленные на текущей карте робота.`;
      fill(stage, loading('Загружаю два прогона…')); fill(events); fill(toolbar);
      llm.update(null, 0);
      const loaded = await Promise.all(selected.files.map(getTrace));
      if (dead || !ctx.alive() || my !== revision) return;
      traces = loaded;
      const rows = [
        ['Собрано', (r) => `${r.samples_collected} / ${r.samples_total}`],
        ['На базе', (r) => r.returned ? 'Да' : 'Нет'],
        ['Заряд потрачен', (r) => `${num(r.battery_used ?? (60 - r.battery), 1)} ед.`],
        ['Пройдено', (r) => `${num(r.distance, 1)} м`],
        ['Время в симуляторе', (r) => `${num(r.time ?? r.t, 1)} с`],
      ];
      fill(results, h('h3', { class: 'pl-h' }, 'Итоги записей'),
        h('table', { class: 'dm-results-table' },
          h('thead', null, h('tr', null, h('th', null, 'Результат'), ...selected.labels.map((s) => h('th', null, s)))),
          h('tbody', null, rows.map(([label, fmt]) => h('tr', null, h('th', null, label), ...traces.map((t) => h('td', null, fmt(t.result || {}))))))),
        h('p', { class: 'dm-caption' }, `LLM: ${traces[1].result?.llm_calls ?? traces[1].llm.length} обращений. Ожидание ответов API: ${num((traces[1].result?.llm_stats?.latency_ms?.total ?? traces[1].llm.reduce((n, e) => n + (e.latency_ms || 0), 0)) / 1000, 0)} с; оно не входит во время симулятора.`));
      await show();
    }
    selector.addEventListener('change', () => load().catch(report));
    await load();
  } catch (e) { report(e); }
  function report(e) {
    selector.disabled = false;
    if (!dead && ctx.alive()) fill(stage, h('p', { class: 'lb-alert', role: 'alert' }, e.message));
  }
  return { destroy };
}
