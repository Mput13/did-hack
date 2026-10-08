// Лаборатория DID: одностраничное приложение на хэш-маршрутах, без сборки и внешних библиотек.
//   #/            обзор            #/exp/E1          опыт
//   #/run?file=   прогон           #/compare?a=&b=   сравнение
//   #/scenarios   сценарии         #/live            живой прогон
//   #/knowledge   знания, накопленные между прогонами

import { h, fill, icon, errorBox, emptyBox, parseHash, href, jobs, tip } from './views/common.js';
import * as overview from './views/overview.js';
import * as experiment from './views/experiment.js';
import * as run from './views/run.js';
import * as compare from './views/compare.js';
import * as scenarios from './views/scenarios.js';
import * as live from './views/live.js';
import * as knowledge from './views/knowledge.js';

const ROUTES = [
  { re: /^\/$/, view: overview, nav: 'home', title: 'Обзор' },
  { re: /^\/exp\/([^/]+)$/, view: experiment, nav: 'home', title: (p) => `Опыт ${p[0]}` },
  { re: /^\/run$/, view: run, nav: 'home', title: 'Прогон', wide: true },
  { re: /^\/compare$/, view: compare, nav: 'home', title: 'Сравнение', wide: true },
  { re: /^\/scenarios$/, view: scenarios, nav: 'scenarios', title: 'Сценарии' },
  { re: /^\/knowledge$/, view: knowledge, nav: 'knowledge', title: 'Знания' },
  { re: /^\/live$/, view: live, nav: 'live', title: 'Живой прогон', wide: true },
  // #/pilot — пульт: модуль подгружается при первом открытии страницы.
  { re: /^\/pilot$/, view: { render: (...a) => import('./views/pilot.js').then((m) => m.render(...a)) }, nav: 'pilot', title: 'Пульт', wide: true },
];

const main = document.getElementById('app');
const jobsBox = document.getElementById('lb-jobs');
let token = 0;
let leave = [];

async function route(opts = {}) {
  const my = ++token;
  const { path, query } = parseHash();
  for (const fn of leave.splice(0)) {
    try { fn(); } catch (e) { console.error(e); }
  }
  tip.hide();

  let match = null;
  let params = [];
  for (const r of ROUTES) {
    const m = path.match(r.re);
    if (m) { match = r; params = m.slice(1).map(decodeURIComponent); break; }
  }
  document.querySelectorAll('[data-nav]').forEach((a) => {
    if (match && a.dataset.nav === match.nav) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });

  // Страницам с проигрывателем нужна вся ширина экрана.
  document.body.toggleAttribute('data-wide', !!(match && match.wide));
  const page = h('div', { class: 'lb-page' });
  const scrollY = window.scrollY;
  // При перерисовке той же страницы держим высоту, чтобы экран не прыгал.
  main.style.minHeight = opts.keepScroll ? `${main.offsetHeight}px` : '';
  main.replaceChildren(page);
  if (!opts.keepScroll) window.scrollTo(0, 0);

  if (!match) {
    document.title = 'Лаборатория DID';
    fill(page, emptyBox('Такой страницы нет', `Адрес «${path}» лаборатории не знаком.`, h('a', { class: 'lb-btn', href: href('/') }, 'На обзор')));
    return;
  }
  const title = typeof match.title === 'function' ? match.title(params) : match.title;
  document.title = `${title} · Лаборатория DID`;
  const ctx = {
    params,
    query,
    alive: () => my === token,
    onLeave: (fn) => { if (my === token) leave.push(fn); else fn(); },
    reload: () => route({ keepScroll: true }),
  };
  try {
    await match.view.render(page, ctx);
  } catch (e) {
    console.error(e);
    if (ctx.alive()) fill(page, errorBox('Страница не открылась', String((e && e.message) || e), ctx.reload));
  }
  if (ctx.alive() && opts.keepScroll) {
    main.style.minHeight = '';
    window.scrollTo(0, scrollY);
  }
}

// Идущие серии видны с любой страницы.
jobs.subscribe((list) => {
  const running = list.filter((j) => j.state === 'running');
  fill(jobsBox, running.map((j) => h('a', { class: 'lb-jobchip', href: href(`/exp/${j.experiment}`) },
    icon('spin', 15), j.total ? `Считается ${j.experiment}: ${j.done} из ${j.total}` : `Считается ${j.experiment}…`)));
});

window.addEventListener('hashchange', () => route());
window.addEventListener('scroll', () => tip.hide(), { passive: true });
jobs.refresh();
route();
