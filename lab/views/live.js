// Живой прогон в Gazebo: раз в секунду спрашиваем сервер и дорисовываем запись.

import { h, fill, icon, sleep, loading, errorBox, api, getArena, getTrace, runHref, num, levelName, agentLabel, agentColor, getOptions } from './common.js';
import { mountPlayer, destroyPlayer } from './player.js';

// Текст команд — в одном месте, чтобы его было легко поправить.
const ONE_COMMAND = 'pixi run gazebo-run --level hard --seed 3';
const TWO_TERMINALS = [
  { text: 'Первый терминал — стенд с окном Gazebo:', cmd: 'pixi run stand-gui level:=hard seed:=3' },
  { text: 'Второй терминал — агент:', cmd: 'pixi run agent-ros --level hard --seed 3' },
];

function clock() {
  const d = new Date();
  const pad = (x) => String(x).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export async function render(root, ctx) {
  root.append(loading('Проверяю, идёт ли прогон…'));
  let arena;
  let options = null;
  try {
    [arena, options] = await Promise.all([getArena(), getOptions().catch(() => null)]);
  } catch (e) {
    root.replaceChildren(errorBox('Не удалось загрузить арену', e.message, ctx.reload));
    return;
  }
  if (!ctx.alive()) return;

  const lamp = h('span', { class: 'lb-lamp', 'aria-hidden': 'true' });
  const statusText = h('span', { class: 'lb-live__state' });
  const checked = h('span', { class: 'lb-muted' });
  const idle = h('div', { class: 'lb-card lb-live__idle' },
    h('h2', { class: 'lb-h2', text: 'Сейчас в Gazebo никто не едет' }),
    h('p', { class: 'lb-lead', text: 'Как только робот поедет в симуляторе, здесь появятся его путь, показания датчика и журнал решений — с задержкой около секунды. Страницу обновлять не нужно.' }),
    h('div', { class: 'lb-eyebrow', text: 'Как запустить одной командой' }),
    h('div', null,
      h('pre', { class: 'lb-code', text: ONE_COMMAND }),
      h('p', { class: 'lb-muted' }, 'Чтобы видеть окно Gazebo, добавьте в конец ', h('code', { class: 'lb-code lb-code--inline', text: '--gui' }), '.')),
    h('div', { class: 'lb-eyebrow', text: 'Или в двух терминалах' }),
    h('ol', { class: 'lb-steps' }, TWO_TERMINALS.map((c) => h('li', null, h('div', { text: c.text }), h('pre', { class: 'lb-code', text: c.cmd })))),
    h('p', { class: 'lb-muted', text: 'Нет времени ждать Gazebo? Тот же агент за долю секунды проходит сценарий в быстром симуляторе — на странице «Сценарии».' }));
  const stage = h('div', { class: 'lb-stage', hidden: true });
  const saved = h('div', { class: 'lb-live__saved', hidden: true });
  fill(root,
    h('header', { class: 'lb-intro lb-intro--tight' },
      h('div', { class: 'lb-eyebrow', text: 'Живой прогон' }),
      h('h1', { class: 'lb-h1', text: 'Робот в Gazebo прямо сейчас' }),
      h('div', { class: 'lb-live__bar', role: 'status' }, lamp, statusText, checked),
      saved),
    idle, stage);

  let player = null;
  let mounting = false;
  let currentId = null;
  let lastLen = 0;
  let timer = null;
  let wasActive = false;
  let ended = false;          // прогон, который мы показывали, закончился
  let lastTrace = null;

  // Прогон закончился: убираем отметку «идёт прогон» и, когда судья допишет запись, показываем её целиком с итогом.
  async function finish() {
    const tr = lastTrace;
    if (!tr) return;
    if (player && player.controller && typeof player.controller.update === 'function') {
      try { player.controller.update(tr, { follow: false }); } catch (e) { console.error(e); }
    }
    if (!tr.id) return;
    const file = `${tr.id}.json.gz`;
    let full = null;
    for (let i = 0; i < 5 && !full; i++) {            // запись дописывается через секунду-другую после финиша
      try { full = await getTrace(file); } catch { full = null; }
      if (!ctx.alive() || !ended || lastTrace !== tr) return;
      if (!full) await sleep(1500);
    }
    // В папке мог остаться старый прогон с тем же именем: берём запись, только если она не короче показанной.
    const lenOf = (t) => (t && t.track && t.track.t ? t.track.t.length : 0);
    if (!ctx.alive() || !ended || lastTrace !== tr || !full || lenOf(full) < lenOf(tr)) return;
    if (player && player.controller && typeof player.controller.update === 'function') {
      try { player.controller.update(full, { follow: false }); } catch (e) { console.error(e); }
    }
    saved.hidden = false;
    fill(saved, h('a', { class: 'lb-btn', href: runHref(file) }, icon('play'), 'Открыть запись этого прогона'));
  }

  const setState = (mode, text) => {
    lamp.dataset.mode = mode;
    statusText.textContent = text;
  };
  setState('idle', 'Прогон не идёт');

  async function tick() {
    if (!ctx.alive()) return;
    let st = null;
    try { st = await api.get('/api/live', 0); } catch { st = null; }
    if (!ctx.alive()) return;
    checked.textContent = `проверяю раз в секунду · последняя проверка в ${clock()}`;
    if (!st) {
      setState('off', 'Сервер лаборатории не отвечает');
    } else if (st.active && st.trace) {
      const tr = st.trace;
      const len = tr.track && tr.track.t ? tr.track.t.length : 0;
      const id = tr.id || tr.created || 'live';
      const sc = tr.scenario || {};
      const agent = tr.agent && tr.agent.name ? agentLabel(tr.agent.name, options) : '';
      const t = len ? tr.track.t[len - 1] : 0;
      setState('on', `Идёт прогон: ${sc.level ? `${levelName(sc.level).toLowerCase()} уровень, сценарий ${sc.seed}` : 'сценарий не указан'}${agent ? ` · ${agent}` : ''} · ${num(t, 0)} с от старта`);
      const fresh = !player || id !== currentId || len < lastLen || ended;
      ended = false;
      saved.hidden = true;
      lastTrace = tr;
      if (fresh && !mounting) {
        mounting = true;
        destroyPlayer(player);
        idle.hidden = true;
        stage.hidden = false;
        player = await mountPlayer(stage, { trace: tr, arena, follow: true, color: agentColor(tr.agent && tr.agent.name) });
        mounting = false;
        currentId = id;
        if (!ctx.alive()) { destroyPlayer(player); return; }
      } else if (player && player.controller && typeof player.controller.update === 'function') {
        try { player.controller.update(tr); } catch (e) { console.error(e); }
      }
      lastLen = len;
      wasActive = true;
    } else if (wasActive) {
      if (!ended) {
        ended = true;
        finish();
      }
      setState('idle', 'Прогон закончился — на экране его запись. Жду следующий');
    } else {
      setState('idle', 'Прогон не идёт');
    }
    timer = setTimeout(tick, 1000);
  }

  ctx.onLeave(() => {
    clearTimeout(timer);
    destroyPlayer(player);
  });
  tick();
}
