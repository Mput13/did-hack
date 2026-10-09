#!/usr/bin/env python3
"""Собирает explain.html — одну самодостаточную страницу с объяснением проекта и результатами.

    pixi run explain

Текст и рисунки лежат в docs/explainer/ (page.html, style.css, explain.js). Сюда подставляются
живые данные: геометрия арены, записи нескольких прогонов и сводки серий из runs/. Страница
открывается двойным щелчком, сеть и сервер ей не нужны.
"""
import json
import re
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.arena import load_arena                     # noqa: E402
from did.config import BASE, LEVELS, Rules           # noqa: E402
from did.nav import CostGraph, path_length           # noqa: E402
from did.recorder import decode_grid, load_trace     # noqa: E402
from did.route import survey_route                   # noqa: E402
from did.runner import RUNS, run_episode             # noqa: E402

SRC = ROOT / 'docs' / 'explainer'
STORIES = [('typical', 1026, 'Типичный трудный сценарий'), ('failure', 1032, 'Сценарий, где адаптивный агент проиграл')]
KEEP_METRICS = ('score', 'samples_share', 'returned', 'battery_used', 'penalties', 'false_collects',
                'hazard_hits', 'distance', 'time', 'collisions', 'inq_total', 'inq_correct', 'inq_wrong',
                'inq_insufficient', 'inq_energy', 'faults_found', 'study_error_pct', 'study_covered',
                'study_halfwidth_pct', 'study_energy')
SCIENCE_STORY = 1020      # прогон исследователя, в котором есть утечка, залипший датчик и дорогой грунт
# Пример из отчёта P3: после сбора образца № 5 пересчёт карты стирает знание о соседнем образце № 2.
NEIGHBOR = {'exp': 'E26', 'arm': 'adaptive_v2', 'run': 'hard-11005', 'lost': 2, 'taken': 5, 'until': 20.0}


def thin(tr):
    """Запись прогона для встраивания: реже точки пути и снимки, без служебного."""
    out = {k: tr[k] for k in ('id', 'arm', 'backend', 'scenario', 'rules', 'result', 'modes', 'events', 'world',
                              'journal', 'hypotheses', 'plans', 'hazards')}
    out['agent'] = tr['agent']['name']
    out['track'] = {k: v[::2] for k, v in tr['track'].items()}
    out['paths'] = tr['paths']
    for key in ('belief', 'soil'):
        g = tr.get(key)
        out[key] = {**g, 'snaps': g['snaps'][::2]} if g else None
    out['scans'] = [{**s, 'r': s['r'][::2]} for s in tr['scans'][::2]]
    return out


def experiment(exp_id):
    path = RUNS / exp_id / 'summary.json'
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding='utf-8'))
    spec = {k: s['spec'].get(k) for k in ('id', 'title', 'kind', 'question', 'hypothesis', 'method', 'expect',
                                         'refute', 'arms', 'levels', 'conditions', 'metrics', 'seeds',
                                         'seed_start', 'manual')}
    groups = [{'arm': g['arm'], 'condition': g['condition'], 'level': g['level'], 'n': g['n'],
               'stats': {m: (g['stats'].get(m) and {k: g['stats'][m][k] for k in ('mean', 'median', 'ci', 'n')})
                         for m in KEEP_METRICS}} for g in s['groups']]
    runs = [{'arm': r['arm'], 'condition': r['condition'], 'level': r['level'], 'seed': r['seed'],
             'backend': r.get('backend'), **{m: r['metrics'].get(m) for m in KEEP_METRICS},
             'detect': r['metrics'].get('detect'), 'hyp': r['metrics'].get('hypotheses'),
             'llm_calls': r['metrics'].get('llm_calls'), 'llm_failed': r['metrics'].get('llm_failed'),
             'plans': r['metrics'].get('plans'), 'collected': r['metrics'].get('samples_collected'),
             'total': r['metrics'].get('samples_total'), 'reason': r['metrics'].get('reason'),
             'inq': r['metrics'].get('inquiries'), 'order': r.get('order'),
             'wall_s': r.get('wall_s')} for r in s['runs']]
    return {'spec': spec, 'status': s['status'], 'groups': groups, 'claims': s['claims'], 'runs': runs,
            'errors': len(s['errors']), 'generated': s['generated'], 'metrics': s['metrics']}


def soil_demo(arena):
    """Короткий путь через дорогой участок и путь в объезд — для разных множителей расхода."""
    per_m = Rules().drain_per_m
    start, goal = BASE, (1.7, -0.45)
    zone = {'x': -0.55, 'y': -0.5, 'w': 0.9, 'h': 1.0}
    X, Y = arena.cell_centers()
    inside = (np.abs(X - zone['x']) <= zone['w'] / 2) & (np.abs(Y - zone['y']) <= zone['h'] / 2)
    plain = CostGraph(arena)
    straight, _ = plain.plan(start, goal)
    out = {'start': list(start), 'goal': list(goal), 'zone': zone, 'cases': []}
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0):
        grid = np.where(inside, mult, 1.0)
        g = CostGraph(arena)
        g.set_cost(grid)
        smart, smart_cost = g.plan(start, goal)
        dist, pred = g.field(*start)
        # тот же короткий путь, но оплаченный по настоящей цене грунта
        blind = sum(math.dist(a, b) * (mult if inside[arena.w2g(*a)[1], arena.w2g(*a)[0]] else 1.0)
                    for a, b in zip(straight, straight[1:]))
        out['cases'].append({'mult': mult,
                             'blind': {'pts': [[round(x, 2), round(y, 2)] for x, y in straight[::2]],
                                       'len': round(path_length(straight), 2), 'energy': round(blind * per_m, 1)},
                             'smart': {'pts': [[round(x, 2), round(y, 2)] for x, y in smart[::2]],
                                       'len': round(path_length(smart), 2),
                                       'energy': round(smart_cost * per_m, 1)}})
    return out


def fixed_route(arena):
    pts = survey_route(arena)
    g = CostGraph(arena)
    chain, line = [BASE] + pts + [BASE], []
    for a, b in zip(chain, chain[1:]):
        seg, _ = g.plan(a, b)
        line += [[round(x, 2), round(y, 2)] for x, y in seg[::2]]
    return {'points': [list(p) for p in pts], 'line': line, 'length': round(path_length([tuple(p) for p in line]), 1)}


def llm_example():
    """Один настоящий обмен агента с планировщиком-моделью (здесь — с имитатором)."""
    from did.agent import Agent, make_config
    from did.fastsim import FastSim
    from did.llm import LocalClient
    from did.llm_mock import MockResponder
    from did.planner import LLMPlanner
    from did.scenario import generate

    arena = load_arena()
    scenario = generate('hard', 1026, arena)
    world = FastSim(arena, scenario, seed=1026)
    seen = []

    class Capture(LLMPlanner):
        def plan(self, state):
            out = super().plan(state)
            seen.append((state, out))
            return out

    bot = Agent(arena, make_config('adaptive_llm'), n_samples=len(scenario.samples),
                planner=Capture(LocalClient(MockResponder(seed=1026))))
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    # Снимок для рисунка «что помнит модель»: вызов, в котором у агента уже больше всего найденного
    # (зоны, грунты, кандидаты, гипотезы), — чтобы на рисунке были заполнены все поля памяти.
    filled = lambda st: sum(bool(st[k]) for k in ('hazards', 'soil_zones', 'candidates', 'open_hypotheses', 'alarms'))
    memory = max((st for st, _ in seen), key=lambda st: (filled(st), st['time_s']), default=None)
    for state, out in seen:
        if out['source'] == 'llm' and state['candidates'] and len(state['explore_points']) >= 2 and out['exchanges']:
            return {'state': state, 'response': out['exchanges'][-1].get('response'), 'calls': len(seen),
                    'memory_state': memory}
    return {'error': 'подходящий обмен не найден'}


def count_tests():
    """Сколько автоматических проверок в проекте (по сбору pytest, без запуска)."""
    import subprocess
    out = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '--collect-only', '-q'], cwd=ROOT,
                         capture_output=True, text=True).stdout
    m = [line for line in out.splitlines() if 'test' in line and 'collected' in line or ' tests' in line]
    digits = ''.join(ch for ch in (m[-1].split()[0] if m else '') if ch.isdigit())
    return int(digits) if digits else None


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else None


def science_story():
    """Прогон исследователя при правилах с несколькими причинами расхода: для показа расследований."""
    try:
        run_episode('hard', SCIENCE_STORY, 'scientist', experiment='_explainer', rules='science')
        tr = load_trace(RUNS / '_explainer' / 'scientist' / f'hard-{SCIENCE_STORY}.json.gz')
    except Exception as exc:       # noqa: BLE001
        return {'error': str(exc)}
    out = thin(tr)
    out.update(inquiries=tr.get('inquiries', []), energy_model=tr.get('energy_model'),
               fault_durations=tr.get('fault_durations'))
    return out


def inquiry_accuracy(exp='E10', arm='scientist', level='hard'):
    """Сверка выводов исследователя со скрытой правдой по всем прогонам серии: строки по темам."""
    rows = {k: {'correct': 0, 'partial': 0, 'wrong': 0, 'insufficient': 0, 'unverifiable': 0}
            for k in ('energy', 'fault', 'sensor')}
    runs = 0
    for path in sorted((RUNS / exp / arm).glob(f'{level}-*.json.gz')):
        runs += 1
        for q in load_trace(path).get('inquiries', []):
            if q.get('conclusion') and q.get('verdict') in rows.get(q['topic'], {}):
                rows[q['topic']][q['verdict']] += 1
    total = sum(sum(r.values()) for r in rows.values())
    return {'runs': runs, 'rows': rows, 'total': total,
            'identified': total - sum(r['insufficient'] for r in rows.values()),
            'wrong': sum(r['wrong'] for r in rows.values())} if runs else None


def neighbor_story():
    """Настоящая запись (опыт E26): уверенность в месте соседнего образца до и после сбора — для рисунка о потере P3.

    Уверенность считается так же, как у кандидатов агента (SampleBelief.candidates): 1 − exp(−сумма вероятностей
    в круге 0,25 м), лучший круг в 0,3 м от настоящего места образца. Снимки карты в записи идут раз в 2 секунды.
    """
    path = RUNS / NEIGHBOR['exp'] / NEIGHBOR['arm'] / f"{NEIGHBOR['run']}.json.gz"
    if not path.exists():
        return None
    tr = load_trace(path)
    b, samples = tr['belief'], tr['scenario']['samples']
    ys, xs = np.mgrid[0:b['h'], 0:b['w']]
    cx, cy = b['x0'] + (xs + 0.5) * b['res'], b['y0'] + (ys + 0.5) * b['res']

    def confidence(grid, sx, sy):
        near = np.argwhere(np.hypot(cx - sx, cy - sy) <= 0.3)
        best = max(grid[np.hypot(cx - cx[j, i], cy - cy[j, i]) <= 0.25].sum() for j, i in near)
        return round(1.0 - math.exp(-float(best)), 2)

    snaps = [s for s in b['snaps'] if s['t'] <= NEIGHBOR['until']]
    series = []
    for s in snaps:
        grid = (decode_grid(s['data'], b['h'], b['w']) / 255.0) ** 2
        series.append({'t': s['t'], 'lost': confidence(grid, *samples[NEIGHBOR['lost']]),
                       'taken': confidence(grid, *samples[NEIGHBOR['taken']])})
    took = next(e['t'] for e in tr['events'] if e['type'] == 'sample_collected' and e.get('sample') == NEIGHBOR['taken'])
    before = max((s for s in snaps if s['t'] < took), key=lambda s: s['t'])
    after = min((s for s in snaps if s['t'] >= took), key=lambda s: s['t'])
    tk = tr['track']
    n = sum(1 for v in tk['t'] if v <= after['t'])
    return {**NEIGHBOR, 'seed': tr['scenario'].get('seed'), 'grid': {k: b[k] for k in ('res', 'x0', 'y0', 'w', 'h')},
            'before': before, 'after': after, 'took': took, 'series': series, 'samples': samples,
            'track': {'t': tk['t'][:n], 'x': tk['x'][:n], 'y': tk['y'][:n]}, 'base': list(BASE)}


def research():
    """План исследований (research/agenda.yaml) и выводы из сданных отчётов (research/findings/*.md)."""
    import re

    import yaml
    path = ROOT / 'research' / 'agenda.yaml'
    if not path.exists():
        return None
    studies = yaml.safe_load(path.read_text(encoding='utf-8')).get('studies', [])
    for st in studies:
        report = ROOT / 'research' / 'findings' / f"{st['id']}.md"
        if report.exists():
            text = report.read_text(encoding='utf-8')
            m = (re.search(r'^## Коротко\s*\n(.*?)(?=^## |\Z)', text, re.S | re.M)      # короткий текст для страницы
                 or re.search(r'^## Вывод\s*\n(.*?)(?=^## |\Z)', text, re.S | re.M))
            st['conclusion'] = m.group(1).strip() if m else None
            lim = re.search(r'^## Ограничения\s*\n(.*?)(?=^## |\Z)', text, re.S | re.M)
            st['limits'] = lim.group(1).strip() if lim else None
        if not st.get('conclusion') and st.get('result'):      # отчёта нет или в нём нет раздела — итог из плана
            st['conclusion'] = str(st['result']).strip()
        st['computed'] = bool(st.get('experiment')) and (RUNS / str(st['experiment']) / 'summary.json').exists()
    return {'studies': studies, 'built': time.strftime('%d.%m.%Y %H:%M')}


def research_runs():
    """Сколько прогонов и аварий в опытах исследовательского контура: всё, кроме E1–E14 (E15 и дальше, G2, L3a–L3e).

    series — сколько серий описано в experiments/ (без пробных *_pilot), computed — у скольких есть сводка в runs/.
    """
    runs = errors = computed = 0
    specs = [p for p in sorted((ROOT / 'experiments').glob('*.yaml')) if 'pilot' not in p.stem]
    for p in specs:
        path = RUNS / p.stem / 'summary.json'
        if not path.exists():
            continue
        computed += 1
        num = re.fullmatch(r'E(\d+)', p.stem)        # E1–E14 встроены целиком и считаются на странице
        if num and int(num.group(1)) <= 14:
            continue
        s = json.loads(path.read_text(encoding='utf-8'))
        runs += len(s['runs'])
        errors += len(s['errors'])
    return {'runs': runs, 'errors': errors, 'series': len(specs), 'computed': computed}


def shots():
    """Снимки окон (docs/explainer/shots/*.jpg) — внутрь страницы, чтобы она оставалась одним файлом."""
    import base64
    return {p.stem: 'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode('ascii')
            for p in sorted((SRC / 'shots').glob('*.jpg'))}


def code_size():
    """Сколько строк в проекте: по файлам и всего (Python и JavaScript), без чужого и сгенерированного."""
    def lines(path):
        return sum(1 for _ in path.open(encoding='utf-8', errors='ignore'))
    py = [p for pat in ('did/**/*.py', 'tools/*.py', 'tests/*.py', 'ws/src/**/*.py') for p in ROOT.glob(pat)]
    js = [p for pat in ('lab/*.js', 'lab/views/*.js', 'docs/explainer/*.js') for p in ROOT.glob(pat)]
    files = {str(p.relative_to(ROOT)): lines(p) for p in py + js}
    return {'files': files, 'py': sum(lines(p) for p in py), 'js': sum(lines(p) for p in js)}


def trap_table(exp='E14', arm='scientist', level='hard'):
    """Опыт с ловушками: сколько выводов вынесено и сколько из них неверных в каждом условии."""
    spec = experiment(exp)
    if not spec or not spec.get('spec'):
        return None
    rows = []
    for cond in spec['spec']['conditions']:
        folder = arm if cond['id'] == 'base' else f"{arm}@{cond['id']}"
        c = {'correct': 0, 'partial': 0, 'wrong': 0, 'insufficient': 0, 'unverifiable': 0}
        runs = 0
        for path in sorted((RUNS / exp / folder).glob(f'{level}-*.json.gz')):
            runs += 1
            for q in load_trace(path).get('inquiries', []):
                if q.get('conclusion') and q.get('verdict') in c:
                    c[q['verdict']] += 1
        if runs:
            rows.append({'id': cond['id'], 'label': cond['label'], 'runs': runs, **c,
                         'identified': c['correct'] + c['partial'] + c['wrong'] + c['unverifiable']})
    return rows or None


def main():
    arena = load_arena()
    stories = []
    for key, seed, title in STORIES:
        item = {'key': key, 'seed': seed, 'title': title}
        for arm in ('fixed', 'adaptive'):
            item[arm] = thin(load_trace(RUNS / 'E1' / arm / f'hard-{seed}.json.gz'))
        stories.append(item)

    gazebo = []
    for level in LEVELS:
        pair = {}
        for arm in ('gazebo', 'fastsim'):
            path = RUNS / 'E7' / arm / f'{level}-1.json.gz'
            if path.exists():
                pair[arm] = thin(load_trace(path))
        if len(pair) == 2:
            gazebo.append({'level': level, **pair})

    data = {
        'built': time.strftime('%d.%m.%Y %H:%M'),
        'tests': count_tests(),
        'arena': arena.to_dict(),
        'rules': Rules().to_dict(),
        'levels': LEVELS,
        'stories': stories,
        'gazebo': gazebo,
        # Опыты E1–E14 показаны на странице графиками и встраиваются целиком; опыты исследовательского контура
        # (E15 и дальше) представлены выводами в журнале, от них нужен только счёт прогонов.
        'experiments': {p.stem: experiment(p.stem) for p in sorted((ROOT / 'experiments').glob('E*.yaml'))
                        if p.stem[1:].isdigit() and int(p.stem[1:]) <= 14},
        'research_runs': research_runs(),
        'science': science_story(),
        'neighbor': neighbor_story(),
        'inq_accuracy': inquiry_accuracy(),
        'traps': trap_table(),
        'code': code_size(),
        'shots': shots(),
        'research': research(),
        'kb': _json(RUNS / '_knowledge' / 'kb.json'),
        'llm_real': {p.name.removesuffix('.summary.json'): {k: v for k, v in _json(p).items() if k not in ('runs',)}
                     for p in sorted((RUNS / 'llm_real').glob('*.summary.json'))},
        'roles': {p.stem: _json(p) for p in sorted((RUNS / 'llm_real' / 'roles').glob('*.json'))},
        'llm_agreement': _json(RUNS / 'llm_real' / 'agreement.json'),
        'soil_demo': soil_demo(arena),
        'route': fixed_route(arena),
        'llm': llm_example(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=lambda o: o.item())
    page = (SRC / 'page.html').read_text(encoding='utf-8')
    page = page.replace('/*STYLE*/', (SRC / 'style.css').read_text(encoding='utf-8'))
    page = page.replace('/*SCRIPT*/', (SRC / 'explain.js').read_text(encoding='utf-8'))
    page = page.replace('"__DATA__"', payload.replace('</', '<\\/'))
    for target in (ROOT / 'explain.html', ROOT / 'lab' / 'explain.html'):
        target.write_text(page, encoding='utf-8')
    print(f'explain.html: {len(page) / 1e6:.2f} МБ, прогонов встроено {2 * len(stories) + 2 * len(gazebo)}, '
          f'пар Gazebo {len(gazebo)}')


if __name__ == '__main__':
    main()
