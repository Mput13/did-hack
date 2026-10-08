// Подключение проигрывателя (lab/replay.js — отдельный модуль). Если его нет или он падает,
// вместо него показывается понятная заглушка и неподвижная схема маршрута.

import { h, icon, loading, loadReplay, replayProblem } from './common.js';
import { createArena, arenaLegend } from './arena.js';

function note(why) {
  return h('div', { class: 'lb-alert', data: { tone: 'warn' }, role: 'status' },
    h('div', { class: 'lb-alert__icon' }, icon('info', 20)),
    h('div', { class: 'lb-alert__body' },
      h('div', { class: 'lb-alert__title', text: 'Проигрыватель пока недоступен' }),
      h('div', { class: 'lb-alert__text', text: `Причина: ${why || 'модуль не загрузился'}. Ниже — неподвижная схема: арена на старте и весь путь робота.` })));
}

function staticMap(trace, arena, color) {
  const sc = trace.scenario || {};
  const collected = new Set((trace.events || []).filter((e) => e.type === 'sample_collected' && e.sample != null).map((e) => e.sample));
  const state = {
    samples: sc.samples || [], soils: sc.soils || [], hazards: sc.hazards || [], base: sc.base,
    track: trace.track ? { x: trace.track.x, y: trace.track.y, color } : null, collected,
  };
  const view = createArena(arena, { label: 'Схема маршрута' });
  view.render(state);
  const extra = [
    h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__track', style: { background: color } }), 'путь робота'),
    h('span', { class: 'lb-key' }, h('i', { class: 'lb-key__sample lb-key__sample--missed' }), 'образец не собран'),
  ];
  return h('div', { class: 'lb-static' }, view.el, arenaLegend(state, extra));
}

/**
 * Один прогон. Возвращает {ok, controller}: controller есть всегда (у заглушки — с update и destroy).
 * opts: {trace, arena, follow, compact, color}
 */
export async function mountPlayer(host, opts) {
  host.replaceChildren(loading('Открываю проигрыватель…'));
  const mod = await loadReplay();
  let why = replayProblem();
  if (mod) {
    const box = h('div', { class: 'lb-player' });
    host.replaceChildren(box);
    try {
      const controller = await mod.mountReplay(box, { trace: opts.trace, arena: opts.arena, compact: !!opts.compact, follow: !!opts.follow });
      return { ok: true, controller: controller || {} };
    } catch (e) {
      console.error('Проигрыватель упал при запуске:', e);
      why = `проигрыватель упал при запуске (${e && e.message ? e.message : e})`;
    }
  }
  const color = opts.color || '#2a78d6';
  const paint = (trace) => host.replaceChildren(note(why), staticMap(trace, opts.arena, color));
  paint(opts.trace);
  return { ok: false, controller: { update: paint, destroy() {} } };
}

/** Два прогона рядом. opts: {traces: [a, b], labels, colors, arena} */
export async function mountPair(host, opts) {
  host.replaceChildren(loading('Открываю проигрыватель…'));
  const mod = await loadReplay();
  let why = replayProblem();
  if (mod && typeof mod.mountCompare !== 'function') why = 'в replay.js нет функции mountCompare';
  if (mod && typeof mod.mountCompare === 'function') {
    const box = h('div', { class: 'lb-player' });
    host.replaceChildren(box);
    try {
      const controller = await mod.mountCompare(box, { traces: opts.traces, labels: opts.labels, arena: opts.arena });
      return { ok: true, controller: controller || {} };
    } catch (e) {
      console.error('Проигрыватель упал при запуске:', e);
      why = `проигрыватель упал при запуске (${e && e.message ? e.message : e})`;
    }
  }
  host.replaceChildren(note(why), h('div', { class: 'lb-static-pair' },
    opts.traces.map((t, i) => h('div', null,
      h('div', { class: 'lb-static__title' }, h('span', { class: 'lb-dot', style: { background: opts.colors[i] } }), opts.labels[i]),
      staticMap(t, opts.arena, opts.colors[i])))));
  return { ok: false, controller: { destroy() {} } };
}

export function destroyPlayer(res) {
  try {
    if (res && res.controller && typeof res.controller.destroy === 'function') res.controller.destroy();
  } catch (e) {
    console.error(e);
  }
}
