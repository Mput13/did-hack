import { h, fill, api, getArena, getTrace, num, loading, errorBox } from './common.js';
import { createMap } from './pilot-map.js';
import { mountComparison } from './demo-comparison.js';
import { mountPlayer, destroyPlayer } from './player.js';
import { createLLMPanel } from './llm-panel.js';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const busyModes = ['drive', 'home', 'mission', 'settle', 'wait'];

export async function render(root, ctx) {
  if (!document.querySelector('link[data-demo-css]')) document.head.append(h('link', {
    rel: 'stylesheet', href: 'pilot.css', 'data-demo-css': '1',
  }));
  fill(root, loading('Открываю демонстрацию…'));
  let arena;
  try { arena = await getArena(); } catch (e) { fill(root, errorBox('Карта не загрузилась', e.message, ctx.reload)); return; }
  if (!ctx.alive()) return;
  let state = null, timer, polling = false, mutation = false, tool = 'point', task = 'route';
  let comparison = null, recording = null, notice = null, scenarioKey = '', missionKey = '', journalShown = false;
  let slamNavSeen = false, agentsKey = '';     // режим карты SLAM уже замечен; список агентов уже заполнен
  const noticeBox = h('p', { class: 'dm-notice', role: 'status', 'aria-live': 'polite', hidden: true });
  const say = (text, bad = false) => { notice = { text, bad }; paint(); };
  const btn = (text, fn, cls = '') => h('button', { class: `lb-btn ${cls}`, type: 'button', onclick: fn }, text);
  async function send(cmd, extra = {}) {
    if (mutation) return null;
    mutation = true; paint();
    try {
      const r = await api.post('/api/pilot/command', { cmd, ...extra });
      if (!r.ok) throw new Error(r.message || 'Команда отклонена');
      notice = null;
      await poll();
      return r;
    } catch (e) { say(e.message, true); return null; }
    finally { mutation = false; paint(); }
  }
  const pending = () => (state?.route || []).filter((p) => !p.done).map((p) => [p.x, p.y]);
  async function mapClick(p) {
    if (!state?.active) return say('Сначала подключите робота.');
    if (task !== 'route') return say('Для точек и зон выберите задание «По заданному маршруту».');
    if (busyModes.includes(state.mode)) return say('Сначала остановите робота, затем измените маршрут или зоны.');
    if (tool === 'point') {
      map.setGhost(p);
      const r = await send('route', { points: [...pending(), [p.x, p.y]] });
      map.setGhost(null); if (!r) map.reject(p);
    } else {
      const r = await send('zones', { zones: [...(state.zones || []), { kind: tool, x: p.x, y: p.y, r: Number(radius.value) }] });
      if (!r) map.reject(p);
    }
  }
  const map = createMap(arena, mapClick, clamp(Number(ctx.query.fps) || 24, 2, 60));
  map.setOpts({ rays: false, ref: true, belief: false });
  const source = h('span', { class: 'dm-source' });
  const status = h('p', { class: 'dm-status', 'aria-live': 'polite' });
  const steps = h('ol', { class: 'dm-steps', 'aria-label': 'Этапы демонстрации' },
    ['Запуск', 'Карта', 'Задача', 'Движение', 'Результат'].map((s, i) => h('li', null, h('span', null, i + 1), s)));
  const mapType = h('select', { class: 'lb-select', 'aria-label': 'Тип карты' },
    h('option', { value: 'medium' }, 'Эталонная · статичная'), h('option', { value: 'hard' }, 'С изменениями'));
  const seed = h('input', { class: 'lb-input', type: 'number', min: 1, max: 99999, value: 3, 'aria-label': 'Номер сценария' });
  const apply = btn('Применить', async () => {
    mutation = true; paint();
    try {
      const n = Number(seed.value);
      if (!Number.isInteger(n) || n < 1 || n > 99999) throw new Error('Номер сценария: целое число от 1 до 99999');
      await api.post('/api/pilot/start', { backend: 'fastsim', level: mapType.value, seed: n, rules: state?.rules || 'base' });
      notice = null; scenarioKey = ''; await poll();
    } catch (e) { say(e.message, true); } finally { mutation = false; paint(); }
  });
  const mapHelp = h('p', { class: 'dm-caption' });
  const tools = h('div', { class: 'dm-tools', role: 'group', 'aria-label': 'Добавить на карту' });
  const toolButtons = ['point', 'yellow', 'danger'].map((kind, i) => btn(['Точка', 'Жёлтая зона', 'Опасная зона'][i], () => {
    tool = kind; paint();
  }, `dm-tool dm-tool--${kind}`));
  fill(tools, toolButtons);
  const radius = h('input', { type: 'range', min: 0.15, max: 1.2, step: 0.05, value: 0.4, 'aria-label': 'Радиус зоны' });
  const radiusText = h('span', null, '40 см');
  radius.addEventListener('input', () => { radiusText.textContent = `${Math.round(Number(radius.value) * 100)} см`; });
  const size = h('label', { class: 'dm-radius', hidden: true }, 'Радиус', radius, radiusText);
  const hint = h('p', { class: 'dm-caption' });
  const routeText = h('p', { class: 'dm-route' });
  const taskButtons = ['route', 'mission'].map((value, i) => btn(['По заданному маршруту', 'Собрать образцы и вернуться'][i], () => {
    task = value; notice = null; paint();
  }, 'dm-task'));
  const start = btn('Запустить', () => send(task === 'route' ? 'go' : 'mission', task === 'mission' ? { agent: agentSel.value || 'adaptive' } : {}), 'lb-btn--primary dm-start');
  const stop = btn('Стоп', () => send('stop'), 'dm-stop');
  const home = btn('На базу', () => send('home'));
  const reset = btn('Сбросить прогон', () => send('reset'));
  // Режим карты SLAM (pixi run demo --slam-map): готовой карты у робота нет, он достраивает её сам.
  const explore = btn('Построить карту', () => send('explore'));
  explore.hidden = true;
  explore.title = 'Робот сам объедет арену: едет туда, где карта SLAM Toolbox ещё обрывается';
  // Агент автономной миссии: список присылает пульт (did/pilot.py, MISSION_AGENTS), по умолчанию — первый.
  const agentSel = h('select', { class: 'lb-select', 'aria-label': 'Агент миссии' });
  const agentRow = h('label', { class: 'dm-radius', hidden: true }, 'Агент', agentSel);
  const undo = btn('Убрать точку', () => send('route', { points: pending().slice(0, -1) }));
  const clearZones = btn('Убрать зоны', () => send('zones', { zones: [] }));
  const battery = h('strong', null, '—'), distance = h('strong', null, '—'), batterySub = h('span'), distanceSub = h('span');
  const scoreBox = h('strong', null, '—'), scoreSub = h('span');      // счёт судьи: виден весь прогон, не только в итоге
  const result = h('section', { class: 'lb-card dm-result', hidden: true });
  const cycle = h('section', { class: 'lb-card dm-cycle', hidden: true });
  const llm = createLLMPanel();
  llm.update(null, 0, true);
  const detail = h('details', { class: 'dm-map-details' }, h('summary', null, 'Слои карты'));
  const check = (label, key, checked = false) => {
    const box = h('input', { type: 'checkbox', checked, onchange: () => map.setOpts({ [key]: box.checked }) });
    return h('label', { class: 'pl-check' }, box, label);
  };
  const refCheck = check('Известная геометрия', 'ref', true);
  detail.append(h('div', { class: 'pl-toggles' }, check('Лучи лидара', 'rays'), refCheck,
    check('Оценки агента', 'belief'), check('Скрытые объекты для зрителя', 'truth')));
  const slamCheck = check('Карта SLAM Toolbox', 'slam');
  slamCheck.hidden = true;
  detail.querySelector('.pl-toggles').append(slamCheck);
  const live = h('div', { class: 'dm-live' });
  const replayHost = h('div', { class: 'dm-replay', hidden: true });
  const pageHead = h('header', { class: 'dm-head' }, h('div', null,
    h('h1', { class: 'lb-h1 dm-h1' }, 'Робот на карте'),
    h('p', { class: 'dm-caption' }, 'Задайте задачу и проследите путь до результата.')), source);
  const compare = btn('Сравнить с LLM', async () => {
    live.hidden = true; pageHead.hidden = true; replayHost.hidden = false;
    comparison = await mountComparison(replayHost, arena, ctx, () => { comparison = null; replayHost.hidden = true; pageHead.hidden = false; live.hidden = false; window.dispatchEvent(new Event('resize')); }, state);
  }, 'dm-compare');
  const stage = h('div', { class: 'pl-stage' },
    h('div', { class: 'dm-map-toolbar' }, tools, size), map.el,
    h('div', { class: 'dm-map-footer' }, h('span', { class: 'dm-caption' }, 'Оранжевый — пройдено · синий — план'), detail));
  const controls = h('aside', { class: 'dm-controls' },
    h('section', { class: 'lb-card dm-card' },
      h('h2', { class: 'pl-h' }, 'Карта'),
      h('div', { class: 'dm-scenario' }, mapType, seed, apply), mapHelp),
    h('section', { class: 'lb-card dm-card' },
      h('h2', { class: 'pl-h' }, 'Задание'), h('div', { class: 'dm-tasks' }, taskButtons), agentRow, hint, routeText,
      h('div', { class: 'dm-small-actions' }, undo, clearZones),
      h('div', { class: 'dm-actions' }, start, stop),
      h('div', { class: 'dm-small-actions' }, home, reset, explore), status, noticeBox),
    h('section', { class: 'dm-metrics' },
      h('div', null, h('span', null, 'Заряд'), battery, batterySub),
      h('div', null, h('span', null, 'Пройдено'), distance, distanceSub),
      h('div', null, h('span', null, 'Счёт судьи'), scoreBox, scoreSub)), result, cycle, llm.el,
    h('section', { class: 'lb-card dm-card dm-comparison-card' },
      h('h2', { class: 'pl-h' }, 'Что добавляет LLM'),
      h('p', { class: 'dm-caption' }, 'Два пути на одинаковой карте. Можно проследить решения и открыть каждый прогон отдельно.'), compare,
      h('span', { class: 'dm-caption' }, 'Сохранённые прогоны · настоящие ответы модели')));
  fill(live, steps, h('div', { class: 'pl-grid dm-grid' }, stage, controls));
  fill(root, pageHead, live, replayHost);
  map.mount();

  async function openRecording(file) {
    live.hidden = true; pageHead.hidden = true; replayHost.hidden = false;
    const back = btn('← К роботу', () => { destroyPlayer(recording); recording = null; replayHost.hidden = true; pageHead.hidden = false; live.hidden = false; window.dispatchEvent(new Event('resize')); });
    const host = h('div', { class: 'dm-recording' });
    const recordLLM = createLLMPanel();
    fill(replayHost, h('div', { class: 'dm-section-head' }, h('h1', { class: 'dm-title' }, 'Запись текущей миссии'), back), host, recordLLM.el);
    try {
      const trace = await getTrace(file);
      if (!ctx.alive() || replayHost.hidden) return;
      recording = await mountPlayer(host, { trace, arena, compact: true, viewSwitch: false,
        layers: { lidar: false, beliefSamples: false, beliefSoil: false, beliefHazards: false, ring: false, path: false } });
      recordLLM.update(trace, 0);
      recording.controller.onTime?.((t) => recordLLM.update(trace, t));
      if (!ctx.alive() || replayHost.hidden) { destroyPlayer(recording); recording = null; }
    } catch (e) { fill(host, h('p', { role: 'alert' }, e.message)); }
  }
  function paint() {
    const s = state, active = !!s?.active, moving = active && busyModes.includes(s.mode);
    slamCheck.hidden = !s?.slam;
    const key = active ? `${s.backend}:${s.level}:${s.seed}` : '';
    if (key && key !== scenarioKey) { mapType.value = s.level === 'hard' ? 'hard' : 'medium'; seed.value = s.seed; scenarioKey = key; }
    const slamNav = active && !!s.slam_nav;
    if (slamNav && s.slam && !slamNavSeen) {
      // Готовой карты у робота нет: показываем ту, по которой он едет, эталон по умолчанию скрыт.
      slamNavSeen = true;
      map.setOpts({ slam: true, ref: false });
      slamCheck.querySelector('input').checked = true; refCheck.querySelector('input').checked = false;
    }
    const akey = (s?.agents || []).map((a) => a.id).join(',');
    if (akey && akey !== agentsKey) {
      agentsKey = akey; fill(agentSel, s.agents.map((a) => h('option', { value: a.id }, a.label)));
      if (s.agents.some((a) => a.id === ctx.query.agent)) agentSel.value = ctx.query.agent;      // показ: агент из адреса страницы
      else if (ctx.query.agent) notice = { text: `В адресе страницы указан агент «${ctx.query.agent}», такого нет. Выбран «${agentSel.selectedOptions[0]?.textContent}» — проверьте список «Агент». Допустимые: ${s.agents.map((a) => a.id).join(', ')}`, bad: true };
    }
    agentRow.hidden = task !== 'mission' || !akey;
    agentSel.disabled = mutation || moving;
    source.textContent = active ? `${s.backend === 'gazebo' ? 'Gazebo' : 'Быстрый симулятор'} · №${s.seed}${slamNav ? ' · карта SLAM, готовой нет' : ''}` : 'Робот не подключён';
    mapType.disabled = seed.disabled = apply.disabled = mutation || moving || s?.backend === 'gazebo';
    mapHelp.textContent = slamNav ? 'Готовой карты у робота нет: её строит SLAM Toolbox, путь идёт только по увиденному полу.' :
      s?.backend === 'gazebo' ? 'Контур эталонный. Для смены сценария нужен перезапуск стенда Gazebo.' :
      'Меняются грунты, образцы и события. Контур арены остаётся эталонным.';
    toolButtons.forEach((b, i) => { b.setAttribute('aria-pressed', String(['point', 'yellow', 'danger'][i] === tool)); b.disabled = mutation || moving || task !== 'route'; });
    taskButtons.forEach((b, i) => { b.setAttribute('aria-pressed', String(['route', 'mission'][i] === task)); b.disabled = mutation || moving; });
    size.hidden = tool === 'point' || task !== 'route';
    hint.textContent = task === 'mission' ? 'Агент ищет образцы и возвращается на базу. В текущем запуске решения принимает численный алгоритм.' :
      tool === 'yellow' ? 'Кликните по полу: маршрут предпочитает обход жёлтой зоны. Это ограничение оператора, а не изменение расхода батареи.' :
      tool === 'danger' ? 'Кликните по полу: красную зону маршрут не пересекает. Вокруг неё сохраняется запас для корпуса робота.' : 'Кликните по карте, чтобы добавить точку. Несколько точек образуют маршрут.';
    const todo = pending();
    routeText.textContent = task === 'route' ? `Точек: ${todo.length} · зон: ${s?.zones?.length || 0}${s?.path_len ? ` · путь ${num(s.path_len, 1)} м` : ''}` : '';
    start.disabled = mutation || !active || moving || (task === 'route' ? !todo.length : !!s.zones?.length);
    start.textContent = task === 'route' ? 'Запустить маршрут' : 'Запустить миссию';
    stop.disabled = mutation || !active || !['drive', 'home', 'mission'].includes(s.mode);
    home.disabled = mutation || !active || moving || s.at_base;
    reset.disabled = mutation || !active || moving;
    explore.hidden = !slamNav;
    explore.disabled = mutation || moving || !s?.nav?.ready;
    undo.disabled = mutation || !active || moving || !todo.length || task !== 'route';
    clearZones.disabled = mutation || !active || moving || !s.zones?.length;
    compare.disabled = mutation || moving;
    if (task === 'mission' && s?.zones?.length) hint.textContent = 'Уберите зоны оператора перед миссией: автономный агент исследует свойства, заданные сценарием.';
    const m = s?.mission;
    status.textContent = !active ? 'Подключите стенд или примените карту для быстрого симулятора.' : s.mode === 'mission' ? 'Идёт автономная миссия' : s.mode === 'drive' ? 'Робот следует заданному маршруту' :
      s.mode === 'home' ? 'Возвращение на базу' : s.mode === 'wait' || s.mode === 'settle' ? 'Робот готовится к запуску' : 'Готов к заданию';
    const ex = s?.nav?.explore;
    if (active && s.mode === 'drive' && ex && !ex.done) status.textContent = `Строю карту: подъезд ${ex.visited}, границ увиденного осталось ${s.nav.frontiers}`;
    // Сообщение самого пульта (почему робот остановлен, что со SLAM) оператор должен видеть, а не только ошибки команд.
    const told = notice || (active && s.note?.text && (s.note.tone === 'bad' || slamNav) ? { text: s.note.text, bad: s.note.tone === 'bad' } : null);
    noticeBox.hidden = !told;
    if (told) { noticeBox.textContent = told.text; noticeBox.dataset.bad = String(told.bad); }
    battery.textContent = active ? num(s.battery, 1) : '—'; batterySub.textContent = active ? `из ${num(s.battery_start, 0)}` : '';
    distance.textContent = active ? `${num(s.distance, 1)} м` : '—'; distanceSub.textContent = active ? `${num(s.t, 0)} с` : '';
    const sc = active ? s.score : null;
    scoreBox.textContent = sc && sc.score != null ? num(sc.score, 1) : '—';
    scoreSub.textContent = sc && sc.samples_total != null ? `образцов ${sc.samples_collected ?? 0} из ${sc.samples_total}` : '';
    const done = active && !moving && (m?.result || s.route?.length && s.route.every((p) => p.done));
    const step = !active ? 0 : done ? 4 : moving && !['wait', 'settle'].includes(s.mode) ? 3 : todo.length || task === 'mission' ? 2 : 1;
    Array.from(steps.children).forEach((el, i) => { el.dataset.done = String(active && i < step); el.setAttribute('aria-current', i === step ? 'step' : 'false'); });
    result.hidden = !done;
    const rkey = done ? `${m?.trace_file || ''}:${m?.state || 'route'}:${s.route?.length}` : '';
    if (rkey !== missionKey) {
      missionKey = rkey;
      if (done && m?.result) {
        const r = m.result;
        fill(result, h('h2', { class: 'pl-h' }, 'Результат миссии'),
          h('strong', { class: 'dm-result-title' }, `${r.samples_collected} из ${r.samples_total} образцов · ${r.returned ? 'на базе' : 'возврат не выполнен'}`),
          h('p', { class: 'dm-caption' }, `${num(r.t, 0)} с · осталось ${num(r.battery, 1)} ед. заряда`),
          m.trace_file ? btn('Посмотреть этот прогон', () => openRecording(m.trace_file)) : null);
      } else if (done) fill(result, h('h2', { class: 'pl-h' }, 'Результат'), h('strong', { class: 'dm-result-title' }, 'Маршрут выполнен'));
    }
    // Вместе с решениями и гипотезами — расследования агента-исследователя («вопрос — опыт — вывод», kind = inquiry).
    // Карточка узкая, поэтому из расследования опыт и замеры не показываем (остаются странность и вывод), а уточнения одной и той же цели — последним; новое сверху.
    const target = (e) => e?.kind === 'decision' && e.text.startsWith('Кандидат ');
    const entries = (m?.journal || []).filter((e) => ['alarm', 'hypothesis', 'verdict', 'decision'].includes(e.kind) || e.kind === 'inquiry' && !/^Q\d+\. (Опыт|Измерено)/.test(e.text))
      .filter((e, i, all) => !(target(e) && target(all[i + 1]))).slice(-5).reverse();
    cycle.hidden = !entries.length;
    // Карта закреплена, а журнал в правой колонке ниже экрана: с первым решением каждой миссии подводим его к карте (один раз).
    const shown = s?.mode === 'mission' && !!entries.length;
    if (shown && !journalShown) requestAnimationFrame(() => cycle.scrollIntoView({ behavior: 'smooth', block: 'center' }));
    journalShown = shown;
    fill(cycle, h('h2', { class: 'pl-h' }, 'Последние решения агента'), entries.map((e) => h('p', { class: 'dm-caption' }, `${num(e.t, 0)} с · ${e.text.length > 240 ? `${e.text.slice(0, 237)}…` : e.text}`)));
  }
  async function poll() {
    if (!ctx.alive() || polling) return;
    polling = true;
    try { state = await api.get('/api/pilot/state', 0); } catch { state = null; }
    polling = false;
    if (!ctx.alive()) return;
    map.setState(state); paint();
    clearTimeout(timer); timer = setTimeout(poll, state?.active ? 300 : 1000);
  }
  ctx.onLeave(() => { clearTimeout(timer); map.destroy(); comparison?.destroy(); destroyPlayer(recording); });
  paint(); await poll();
}
