// Обзор: миссия, главная гипотеза со статусом, карточки опытов.

import {
  h, fill, icon, badge, mark, loading, errorBox, getIndex, getExperiment, jobs, href, count, when, KINDS,
  metricInfo, armColors, dot, hypothesisWord, ruLevels,
} from './common.js';
import { diffWords } from '../charts.js';

const MISSION = 'Робот ищет спрятанные образцы на арене, которую не видит, и должен вернуться на базу раньше, чем сядет батарея.';
const MAIN_ID = 'E1';     // опыт главной гипотезы; остальные «главные» опыты — приёмка следующих версий агента

// Номера по порядку (E2 раньше E10); серии, которые ещё не запускали, — в конце списка.
const byId = new Intl.Collator('ru', { numeric: true });
const inOrder = (a, b) => (a.status === 'not_run') - (b.status === 'not_run') || byId.compare(a.id, b.id);

function headline(claim) {
  const cells = claim.cells || [];
  return cells.find((c) => c.level === 'all' && c.pair) || cells.find((c) => c.pair) || null;
}

function hero(item, detail) {
  if (!item) return null;
  const claims = (detail && detail.claims && detail.claims.length ? detail.claims : item.claims) || [];
  const colors = detail ? armColors(detail.spec.arms) : null;
  const rows = claims.map((c) => {
    let sub = null;
    const cell = c.cells ? headline(c) : null;
    if (cell) {
      const info = metricInfo(c.metric, detail.metrics);
      const better = c.better || info.better;
      const wins = better === 'higher' ? cell.pair.a_higher : cell.pair.b_higher;
      sub = `в среднем ${diffWords(info, cell.pair.mean)} · лучше в ${wins} из ${cell.pair.n} сценариев`;
    }
    return h('li', { class: 'lb-hero__claim' },
      h('div', null, h('div', { class: 'lb-hero__claim-text', text: ruLevels(c.text || c.metric) }), sub ? h('div', { class: 'lb-hero__claim-sub', text: sub }) : null),
      h('div', { class: 'lb-hero__claim-status' }, badge(c.status)));
  });
  const arms = detail && detail.spec.arms ? detail.spec.arms : [];
  return h('section', { class: 'lb-card lb-hero' },
    h('div', { class: 'lb-hero__main' },
      h('div', { class: 'lb-eyebrow', text: `Главная гипотеза · проверяет опыт ${item.id}` }),
      h('p', { class: 'lb-hero__text', text: ruLevels(item.hypothesis || '') }),
      h('div', { class: 'lb-hero__status' },
        badge(item.status, { big: true, word: hypothesisWord(item.status) }),
        item.runs ? h('span', { class: 'lb-muted', text: `${count(item.runs, 'прогон', 'прогона', 'прогонов')} на одинаковых сценариях` }) : null),
      arms.length ? h('div', { class: 'lb-hero__arms' }, arms.map((a) => h('span', { class: 'lb-chip' }, dot(colors.get(a.id)), a.label))) : null,
      h('div', { class: 'lb-hero__actions' },
        h('a', { class: 'lb-btn lb-btn--primary', href: href(`/exp/${item.id}`) }, `Открыть опыт ${item.id}`, icon('right')))),
    rows.length
      ? h('div', { class: 'lb-hero__side' },
        h('div', { class: 'lb-eyebrow', text: 'Что из неё следует и что показали данные' }),
        h('ul', { class: 'lb-hero__claims' }, rows))
      : h('div', { class: 'lb-hero__side' },
        h('div', { class: 'lb-eyebrow', text: 'Данных пока нет' }),
        h('p', { class: 'lb-muted', text: 'Серию ещё не запускали. Откройте опыт и нажмите «Запустить серию».' })));
}

function card(item) {
  const status = h('div', { class: 'lb-exp__status' });
  const foot = h('div', { class: 'lb-exp__foot' });
  const paint = () => {
    const job = jobs.running(item.id);
    if (job) {
      status.replaceChildren(badge('running'));
      foot.replaceChildren(job.total ? `Посчитано ${job.done} из ${job.total} прогонов` : 'Серия запускается…');
      return;
    }
    status.replaceChildren(badge(item.status));
    const parts = [];
    if (item.runs) parts.push(h('span', { text: count(item.runs, 'прогон', 'прогона', 'прогонов') }));
    if (item.generated) parts.push(h('span', { text: `пересчитан ${when(item.generated)}` }));
    if (item.errors) parts.push(h('span', { class: 'lb-exp__errors' }, icon('warn', 14), `упало: ${item.errors}`));
    if (!parts.length) parts.push(h('span', { text: 'Можно запустить со страницы опыта' }));
    foot.replaceChildren(...parts);
  };
  paint();
  const claims = item.claims || [];
  const el = h('a', { class: 'lb-exp', href: href(`/exp/${item.id}`) },
    h('div', { class: 'lb-exp__top' },
      h('span', { class: 'lb-exp__id', text: item.id }),
      h('span', { class: 'lb-exp__kind', text: KINDS[item.kind] || 'Опыт' })),
    h('div', { class: 'lb-exp__title', text: ruLevels(item.title || item.id) }),
    status,
    h('div', { class: 'lb-exp__q', text: ruLevels(item.question || '') }),
    claims.length
      ? h('ul', { class: 'lb-exp__claims' }, claims.map((c) => h('li', null, mark(c.status), h('span', { text: ruLevels(c.text || c.metric) }))))
      : null,
    foot);
  return { el, paint };
}

function howWeJudge() {
  const row = (status, text) => h('li', null, badge(status), h('span', { text }));
  return h('section', { class: 'lb-card lb-judge' },
    h('div', { class: 'lb-eyebrow', text: 'Как мы делаем вывод' }),
    h('p', { class: 'lb-judge__lead', text: 'Варианты агента проходят одни и те же сценарии. Для каждого утверждения считаем разность между вариантами на каждом сценарии и её 95% интервал.' }),
    h('ul', { class: 'lb-judge__list' },
      row('supported', 'весь интервал на стороне утверждения'),
      row('refuted', 'весь интервал на противоположной стороне'),
      row('inconclusive', 'интервал накрывает ноль: разницу нельзя отличить от случайной'),
      row('partial', 'на части уровней или условий подтверждается, на части — нет')));
}

export async function render(root, ctx) {
  root.append(loading('Загружаю список опытов…'));
  let index;
  try {
    index = await getIndex();
  } catch (e) {
    root.replaceChildren(errorBox('Не удалось получить список опытов', e.message, ctx.reload));
    return;
  }
  const items = (index.experiments || []).slice().sort(inOrder);
  const mainItem = items.find((x) => x.id === MAIN_ID) || items.find((x) => x.kind === 'main') || items[0];
  let detail = null;
  if (mainItem && mainItem.status !== 'not_run') {
    try { detail = await getExperiment(mainItem.id); } catch { detail = null; }
  }
  if (!ctx.alive()) return;

  const cards = items.map(card);
  fill(root,
    h('header', { class: 'lb-intro' },
      h('div', { class: 'lb-eyebrow', text: 'Автономный исследователь на TurtleBot3' }),
      h('h1', { class: 'lb-h1 lb-intro__mission', text: MISSION }),
      h('p', { class: 'lb-lead', text: 'Образцы, дорогой грунт и опасные зоны скрыты: робот узнаёт о них только по датчику близости, расходу заряда и штрафам. На трудном уровне среда меняется прямо во время прогона.' }),
      h('nav', { class: 'lb-intro__links', 'aria-label': 'Быстрые ссылки' },
        h('a', { class: 'lb-btn', href: href('/pilot') }, 'Демонстрация', icon('right')),
        h('a', { class: 'lb-btn', href: href('/study') }, 'Новое исследование', icon('right')),
        h('a', { class: 'lb-btn', href: href('/scenarios') }, 'Сценарии', icon('right')),
        h('a', { class: 'lb-btn', href: href('/knowledge') }, 'Знания робота', icon('right')))),
    hero(mainItem, detail),
    h('div', { class: 'lb-section-head' },
      h('h2', { class: 'lb-h2', text: 'Опыты' }),
      h('span', { class: 'lb-muted', text: 'Каждый опыт проверяет главную гипотезу или уточняет её' })),
    h('div', { class: 'lb-exps' }, cards.map((c) => c.el)),
    howWeJudge());

  ctx.onLeave(jobs.subscribe((_, finished) => {
    if (finished.length) ctx.reload();
    else cards.forEach((c) => c.paint());
  }));
}
