import { h, fill, num } from './common.js';

// Only recorded answers, up to the current playback time. Never infer a model call from the journal.
export function llmAt(trace, time) {
  const calls = (trace?.llm || []).filter((e) => Number.isFinite(e.t) && e.t <= time);
  const call = calls.at(-1);
  if (!call) return null;
  let answer = null;
  if (call.ok) {
    try { answer = typeof call.response === 'string' ? JSON.parse(call.response) : call.response; }
    catch { /* Raw answer remains available below. */ }
  }
  return { call, answer, count: calls.length };
}

export function createLLMPanel() {
  const el = h('section', { class: 'lb-card dm-cycle dm-llm' });
  let key = '';
  return {
    el,
    update(trace, time = 0, live = false) {
      const item = llmAt(trace, time);
      const next = JSON.stringify([live, item]);
      if (key === next) return;
      key = next;
      const title = h('h2', { class: 'pl-h' }, 'Что предлагает LLM');
      if (!item) {
        fill(el, title, h('p', { class: 'dm-caption' }, live
          ? 'В текущей миссии работает численный агент. Ответы модели доступны в «Сравнить с LLM».'
          : trace?.llm?.length ? 'До этого момента модель ещё не вызывалась.' : 'В этом прогоне LLM не вызывалась. Решения принимает численный агент.'));
        return;
      }
      const { call, answer, count } = item;
      const text = (v) => typeof v === 'string' ? v : '';
      const goals = Array.isArray(answer?.subgoals) ? answer.subgoals : [];
      const hypotheses = Array.isArray(answer?.hypotheses) ? answer.hypotheses : [];
      fill(el, title,
        h('p', { class: 'dm-caption' }, `${num(call.t, 1)} с · ответ ${count} · ${call.ok ? 'получен' : 'не принят'}${call.latency_ms ? ` · API ${num(call.latency_ms / 1000, 1)} с` : ''}`),
        call.ok ? h('p', { class: 'dm-llm-reason' }, text(answer?.reasoning) || 'Объяснение не указано в ответе.')
          : h('p', { class: 'dm-caption' }, 'Ответ модели не прошёл проверку. Он не показан как выполненный план.'),
        call.ok && hypotheses.length ? h('div', null, h('h3', { class: 'pl-h' }, 'Предложенные гипотезы'),
          hypotheses.slice(0, 3).map((p) => h('p', { class: 'dm-llm-item' }, text(p.statement),
            p.test ? h('span', { class: 'dm-caption' }, `Проверка: ${text(p.test)}`) : null))) : null,
        call.ok && goals.length ? h('p', { class: 'dm-caption' }, `Подцели: ${goals.map((g) => `${({ explore: 'разведка', collect: 'сбор', return: 'на базу', probe: 'измерение' })[g.type] || g.type || 'действие'}${g.target ? ` → ${g.target}` : ''}`).join('; ')}`) : null,
        h('details', { class: 'dm-llm-raw' }, h('summary', null, 'Ответ модели'),
          h('pre', null, typeof call.response === 'string' ? call.response : JSON.stringify(call.response ?? null, null, 2))));
    },
  };
}
