"""Разбор потерь: куда уходят очки между потолком сценария и счётом агента.

    ./px python tools/loss_breakdown.py runs/E1/adaptive --level hard      # по готовым записям
    ./px python tools/loss_breakdown.py --run adaptive,scientist --rules science \
        --levels medium,hard --seeds 1-80 --jobs 3                         # прогнать и разобрать

Потолок сценария — счёт оракула: он знает, где лежат образцы, грунты и опасные зоны, едет без
поиска и ошибок по лучшему порядку обхода (точный перебор по подмножествам) и возвращается на базу.
Недобор раскладывается без остатка:

    потолок − счёт = 10 · несобранные + 20 · невозврат + штрафы + 0,1 · (остаток оракула − остаток агента)

Каждому несобранному образцу, невозврату и штрафу приписывается одна причина — по записи прогона
(журнал решений, планы, карта вероятностей, события судьи). Оракул недостижим: поиск образцов по
датчику без направления стоит пути. Поэтому «лишний путь» показан отдельной строкой справки.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.arena import load_arena                      # noqa: E402
from did.config import Rules                          # noqa: E402
from did.metrics import fault_intervals               # noqa: E402
from did.nav import INFLATE, CostGraph                # noqa: E402
from did.recorder import decode_grid, load_trace      # noqa: E402
from did.scenario import Scenario, soil_mult          # noqa: E402

V_CRUISE = 0.2          # м/с: для расхода «за включённость» на пути оракула

# Причины в порядке вывода: (ключ, подпись).
CAUSES = [
    ('sample_unknown', 'образец: агент о нём не узнал'),
    ('sample_no_charge', 'образец: знал, но не поехал — заряда не хватало'),
    ('sample_lost', 'образец: поехал и бросил — кандидат не подтвердился'),
    ('sample_fault', 'образец: бросил при сбое датчика'),
    ('sample_hazard', 'образец: бросил из-за опасной зоны'),
    ('sample_miss', 'образец: бросил после ложного сбора'),
    ('sample_cut', 'образец: подъезд прерван возвратом или концом прогона'),
    ('sample_early', 'образец: вернулся рано — привезённого заряда хватало'),
    ('noreturn_hazard', 'невозврат: зона или сбой по дороге домой'),
    ('noreturn_leak', 'невозврат: утечка заряда'),
    ('noreturn_cost', 'невозврат: недооценена цена пути домой'),
    ('noreturn_late', 'невозврат: заряд кончился до решения о возврате'),
    ('noreturn_time', 'невозврат: не успел по времени'),
    ('noreturn_place', 'невозврат: финиш не на базе'),
    ('hazard_first', 'штраф: первый въезд в зону, бывшую с начала'),
    ('hazard_new', 'штраф: первый въезд в зону, появившуюся в прогоне'),
    ('hazard_repeat', 'штраф: повторный въезд в ту же зону'),
    ('false_collect', 'штраф: ложный сбор'),
    ('collision', 'штраф: столкновение'),
    ('battery', 'остаток заряда меньше, чем у оракула'),
]


# ---------------------------------------------------------------------------------------------
# оракул
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _graph():
    return CostGraph(load_arena())


def final_world(scenario):
    """Грунты и опасные зоны после всех событий сценария: в этом мире идёт большая часть прогона."""
    soils, hazards = list(scenario.soils), list(scenario.hazards)
    for ev in scenario.events:
        if ev['type'] == 'soil_change':
            soils = list(ev['soils'])
        elif ev['type'] == 'new_hazard':
            hazards.append(ev['zone'])
    return soils, hazards


def _segment(arena, soils, hazards, a, b, step=0.02):
    """Прямой отрезок: (проходим ли, длина, длина с множителями грунта)."""
    d = math.dist(a, b)
    n = max(1, int(math.ceil(d / step)))
    e = 0.0
    for i in range(n):
        k = (i + 0.5) / n
        x, y = a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])
        if arena.clearance(x, y) < INFLATE or any(z.contains(x, y) for z in hazards):
            return False, d, math.inf
        e += soil_mult(soils, x, y) * d / n
    return True, d, e


def _pull(arena, soils, hazards, pts):
    """Спрямить путь по клеткам там, где прямая проходима и не дороже. Возвращает (точки, м, м×грунт)."""
    out, i = [pts[0]], 0
    length = energy = 0.0
    while i < len(pts) - 1:
        best = None
        chain_d = chain_e = 0.0
        for j in range(i + 1, min(len(pts), i + 60)):
            ok, d, e = _segment(arena, soils, hazards, pts[j - 1], pts[j])
            chain_d, chain_e = chain_d + d, chain_e + (e if ok else d * soil_mult(soils, *pts[j]))
            ok, d, e = _segment(arena, soils, hazards, pts[i], pts[j])
            if ok and e <= chain_e + 1e-9:
                best = (j, d, e)
            elif j == i + 1:
                best = (j, chain_d, chain_e)          # соседние клетки: принять как есть
        j, d, e = best
        out.append(pts[j])
        length, energy, i = length + d, energy + e, j
    return out, length, energy


def _turning(pts):
    return sum(abs((math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi)
                   % (2 * math.pi) - math.pi) for a, b, c in zip(pts, pts[1:], pts[2:]))


def oracle(scenario, rules, budget=None, start=None, targets=None):
    """Лучший объезд targets (по умолчанию всех образцов) из start (базы) с возвратом на базу.

    Возвращает {'n', 'order', 'energy', 'distance', 'score'}: сколько образцов влезает в заряд budget,
    в каком порядке, сколько заряда и метров на это уходит. Перебор точный (динамика по подмножествам);
    учитываются грунт, расход за включённость, а при «научных» правилах — груз и повороты.
    """
    arena, g = load_arena(), _graph()
    soils, hazards = final_world(scenario)
    X, Y = arena.cell_centers()
    mult = np.ones(arena.free.shape)
    for z in soils:
        mult = np.where(z.mask(X, Y), np.maximum(mult, z.mult), mult)
    forbidden = np.zeros(arena.free.shape, bool)
    for z in hazards:
        forbidden |= (X - z.x) ** 2 + (Y - z.y) ** 2 <= (z.r + 0.05) ** 2
    g.set_cost(mult, forbidden=forbidden)
    base = tuple(scenario.base)
    targets = [tuple(p) for p in (scenario.samples if targets is None else targets)]
    nodes = [tuple(start or base)] + targets + [base]
    n = len(targets)
    budget = rules.battery_start if budget is None else budget
    idle = rules.drain_idle_per_s / V_CRUISE
    D = np.zeros((n + 2, n + 2))
    E = np.zeros((n + 2, n + 2))            # заряд без груза: грунт + включённость
    T = np.zeros((n + 2, n + 2))            # заряд на повороты
    for i in range(n + 1):
        _, pred = g.field(*nodes[i])
        for j in range(1, n + 2):
            if i == j:
                continue
            pts, d, e = _pull(arena, soils, hazards, g.trace(pred, *nodes[j]))
            D[i, j], E[i, j] = d, rules.drain_per_m * e + idle * d
            T[i, j] = rules.drain_per_rad * (_turning(pts) + math.pi / 2)    # плюс доворот на старте отрезка
    g.set_cost(None)

    def leg(i, j, k):                       # из i в j, когда уже несём k образцов
        return (E[i, j] - idle * D[i, j]) * (1.0 + rules.load_drain * k) + idle * D[i, j] + T[i, j]

    best = {(0, 0): (0.0, 0.0, ())}         # (множество, последний узел) → (заряд, метры, порядок)
    for mask in sorted(range(1 << n), key=lambda m: bin(m).count('1')):
        k = bin(mask).count('1')
        for last in ([0] if mask == 0 else [j + 1 for j in range(n) if mask >> j & 1]):
            cur = best.get((mask, last))
            if cur is None:
                continue
            for j in range(n):
                if mask >> j & 1:
                    continue
                cand = (cur[0] + leg(last, j + 1, k), cur[1] + D[last, j + 1], cur[2] + (j,))
                key = (mask | 1 << j, j + 1)
                if key not in best or cand[0] < best[key][0]:
                    best[key] = cand
    top = None
    for (mask, last), (e, d, order) in best.items():
        k = bin(mask).count('1')
        e, d = e + leg(last, n + 1, k), d + D[last, n + 1]
        if e <= budget and (top is None or (k, -e) > (top['n'], -top['energy'])):
            top = {'n': k, 'order': list(order), 'energy': e, 'distance': d}
    if top is None:
        top = {'n': 0, 'order': [], 'energy': 0.0, 'distance': 0.0}
    top['score'] = rules.pts_sample * top['n'] + rules.pts_return + rules.pts_battery_left * (budget - top['energy'])
    return top


# ---------------------------------------------------------------------------------------------
# разбор одной записи
# ---------------------------------------------------------------------------------------------

def _belief_peaks(tr, points, radius=0.35):
    """Для каждой точки — наибольшая за прогон уверенность агента, что рядом с ней лежит образец."""
    b = tr.get('belief')
    out = [0.0] * len(points)
    if not b:
        return out
    ys = b['y0'] + (np.arange(b['h']) + 0.5) * b['res']
    xs = b['x0'] + (np.arange(b['w']) + 0.5) * b['res']
    masks = [np.hypot(xs[None, :] - p[0], ys[:, None] - p[1]) <= radius for p in points]
    for snap in b['snaps']:
        p = (decode_grid(snap['data'], b['h'], b['w']).astype(float) / 255.0) ** 2
        for i, m in enumerate(masks):
            out[i] = max(out[i], 1.0 - math.exp(-float(p[m].sum())))
    return out


def _soils_at(scenario, t):
    soils = scenario.soils
    for ev in scenario.events:
        if ev['type'] == 'soil_change' and ev['t'] <= t:
            soils = ev['soils']
    return soils


def _ledger(tr, scenario, rules):
    """Куда ушёл заряд: путь по обычному полу, наценка грунта, зоны, включённость, повороты, груз, прочее."""
    t, x, y, th = (np.array(tr['track'][k]) for k in ('t', 'x', 'y', 'th'))
    collected = sorted(e['t'] for e in tr['events'] if e['type'] == 'sample_collected')
    soil = load = 0.0
    for i in range(1, len(t)):
        ds = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        m = soil_mult(_soils_at(scenario, t[i]), (x[i] + x[i - 1]) / 2, (y[i] + y[i - 1]) / 2)
        k = sum(1 for c in collected if c <= t[i])
        soil += rules.drain_per_m * (m - 1.0) * ds
        load += rules.drain_per_m * m * rules.load_drain * k * ds
    res = tr['result']
    turn = rules.drain_per_rad * float(np.abs((np.diff(th) + np.pi) % (2 * np.pi) - np.pi).sum())
    out = {'floor': rules.drain_per_m * res['distance'], 'soil': soil,
           'hazard': rules.hazard_battery_hit * res['hazard_hits'], 'idle': rules.drain_idle_per_s * res['t'],
           'turn': turn, 'load': load}
    out['other'] = rules.battery_start - res['battery'] - sum(out.values())     # утечка при сбое, шум оценки
    return out


def _attempts(tr, scenario):
    """Подъезды к кандидатам: [(t начала, t конца, чем кончилось, x, y)] по планам агента."""
    plans = tr['plans']
    out = []
    for i, p in enumerate(plans):
        sg = p['subgoals'][0] if p['subgoals'] else None
        if not sg or sg['type'] != 'investigate':
            continue
        nxt = plans[i + 1] if i + 1 < len(plans) else None
        end = nxt['t'] if nxt else tr['result']['t']
        how = 'end' if nxt is None else 'return' if nxt['subgoals'] and nxt['subgoals'][0]['type'] == 'return_base' \
            and nxt['trigger'] in ('return', 'foresight') else nxt['trigger']
        out.append((p['t'], end, how, sg['x'], sg['y']))
    return out


def analyze(tr):
    """Разбор одной записи прогона: потолок, недобор и его причины."""
    scenario = Scenario.from_dict(tr['scenario'])
    rules = Rules(**tr['rules'])
    res = tr['result']
    top = oracle(scenario, rules)
    samples = [tuple(p) for p in scenario.samples]
    got = {e['sample'] for e in tr['events'] if e['type'] == 'sample_collected'}
    missed = [i for i in range(len(samples)) if i not in got]
    loss = defaultdict(float)
    detail = {'missed': {}}

    # --- несобранные образцы ---------------------------------------------------------------
    peaks = _belief_peaks(tr, [samples[i] for i in missed])
    attempts = _attempts(tr, scenario)
    faults = [(k, a, b) for k, a, b in fault_intervals(tr['world']) if k.startswith('sensor')]
    spare = set()
    if res['returned'] and missed:
        # Что оракул успел бы добрать на привезённый заряд, стартуя с базы: это цена раннего возврата.
        extra = oracle(scenario, rules, budget=res['battery'], targets=[samples[i] for i in missed])
        spare = {missed[j] for j in extra['order']}
    tx, ty = np.array(tr['track']['x']), np.array(tr['track']['y'])
    for i, peak in zip(missed, peaks):
        sx, sy = samples[i]
        # Подъезд относится к образцу, если цель подъезда ближе к нему, чем к любому другому несобранному.
        mine = [a for a in attempts if math.hypot(a[3] - sx, a[4] - sy) <= 0.6]
        near = float(np.hypot(tx - sx, ty - sy).min())
        if mine:
            how = mine[-1][2]
            t_end = mine[-1][1]
            if how == 'hazard':
                cause = 'sample_hazard'
            elif how == 'false_collect':
                cause = 'sample_miss'
            elif how in ('return', 'end'):
                cause = 'sample_cut'
            elif how == 'sensor_degraded' or any(a - 1.0 <= t_end <= b + 8.0 for _, a, b in faults):
                cause = 'sample_fault'
            else:
                cause = 'sample_lost'
        elif i in spare:
            cause = 'sample_early'
        elif peak >= 0.35:
            cause = 'sample_no_charge'
        else:
            cause = 'sample_unknown'
        if cause in ('sample_no_charge', 'sample_unknown', 'sample_cut') and i in spare:
            cause = 'sample_early'
        loss[cause] += rules.pts_sample
        detail['missed'][i] = {'cause': cause, 'belief_peak': round(peak, 2), 'nearest_m': round(near, 2),
                               'attempts': len(mine), 'affordable': i in spare}

    # --- невозврат --------------------------------------------------------------------------
    if not res['returned']:
        decided = next((e['t'] for e in tr['journal'] if e['kind'] == 'decision'
                        and e['text'].startswith('Возвращаюсь на базу')), None)
        if decided is None:
            decided = next((p['t'] for p in tr['plans'] if p['subgoals'] and p['subgoals'][0]['type'] == 'return_base'),
                           None)
        leak = [(a, b) for k, a, b in fault_intervals(tr['world']) if k == 'leak']
        if res['reason'] == 'timeout':
            cause = 'noreturn_time'
        elif res['reason'] == 'finish':
            cause = 'noreturn_place'
        elif decided is None:
            cause = 'noreturn_leak' if any(b >= res['t'] - 30.0 for _, b in leak) else 'noreturn_late'
        elif any(e['type'] == 'hazard_hit' and e['t'] >= decided for e in tr['events']):
            cause = 'noreturn_hazard'
        elif any(b >= decided for _, b in leak):
            cause = 'noreturn_leak'
        else:
            cause = 'noreturn_cost'
        loss[cause] += rules.pts_return
        detail['noreturn'] = {'cause': cause, 'decided_t': decided, 'end_t': res['t']}

    # --- штрафы -----------------------------------------------------------------------------
    start_ids = {z.id for z in scenario.hazards}
    _, zones = final_world(scenario)
    seen = set()
    for e in tr['events']:
        if e['type'] == 'hazard_hit':
            z = min(zones, key=lambda z: abs(math.hypot(e['x'] - z.x, e['y'] - z.y) - z.r), default=None)
            zid = z.id if z else None
            cause = 'hazard_repeat' if zid in seen else 'hazard_first' if zid in start_ids else 'hazard_new'
            seen.add(zid)
            loss[cause] -= rules.pts_hazard_hit
        elif e['type'] == 'false_collect':
            loss['false_collect'] -= rules.pts_false_collect
        elif e['type'] == 'collision':
            loss['collision'] -= rules.pts_collision

    # --- остаток заряда ---------------------------------------------------------------------
    left = res['battery'] if res['returned'] else 0.0
    loss['battery'] += rules.pts_battery_left * (rules.battery_start - top['energy'] - left)
    # Оракул может взять не все образцы (не хватает батареи на полный объезд): тогда потолок ниже.
    cap_gap = rules.pts_sample * (len(samples) - top['n'])
    total = top['score'] - res['score']
    assert abs(sum(loss.values()) - cap_gap - total) < 0.02, (tr['id'], dict(loss), cap_gap, total)
    if cap_gap:                             # эти образцы не взял бы и оракул: из потерь агента они вычитаются
        loss['beyond_oracle'] = -cap_gap

    # Путь оракула по тем же образцам, что собрал агент: во сколько раз агент проехал больше.
    same = oracle(scenario, rules, targets=[samples[i] for i in sorted(got)]) if got else None
    track_mode = np.array(tr['track']['mode'])
    step = np.hypot(np.diff(tx), np.diff(ty))
    by_mode = {m: float(step[track_mode[1:] == k].sum()) for k, m in enumerate(tr['modes'])}
    return {'id': tr['id'], 'level': scenario.level, 'seed': scenario.seed, 'score': res['score'],
            'ceiling': round(top['score'], 2), 'gap': round(total, 2), 'loss': {k: round(v, 3) for k, v in loss.items()},
            'collected': len(got), 'total': len(samples), 'returned': bool(res['returned']),
            'battery_left': res['battery'], 'distance': res['distance'], 'time': res['t'],
            'oracle_distance': round(top['distance'], 2), 'oracle_energy': round(top['energy'], 2),
            'oracle_same_distance': round(same['distance'], 2) if same else None,
            'ledger': {k: round(v, 2) for k, v in _ledger(tr, scenario, rules).items()},
            'distance_by_mode': {k: round(v, 2) for k, v in by_mode.items()}, 'detail': detail}


def _analyze_file(path):
    return analyze(load_trace(path))


# ---------------------------------------------------------------------------------------------
# сводка
# ---------------------------------------------------------------------------------------------

def summarize(rows):
    """Таблица «причина → очков на прогон → в скольких прогонах» и справочные числа."""
    n = len(rows)
    causes = []
    labels = dict(CAUSES)
    labels['beyond_oracle'] = 'поправка: образцы, на которые не хватило бы и оракулу'
    for key in [k for k, _ in CAUSES] + ['beyond_oracle']:
        vals = [r['loss'].get(key, 0.0) for r in rows]
        hit = sum(1 for v in vals if abs(v) > 1e-9) if key != 'battery' else n
        if hit:
            causes.append({'cause': key, 'label': labels[key], 'points': round(sum(vals) / n, 2), 'runs': hit})
    mean = lambda f: round(float(np.mean([f(r) for r in rows])), 2)      # noqa: E731
    ratios = [r['distance'] / r['oracle_same_distance'] for r in rows if r['oracle_same_distance']]
    ledger = {k: mean(lambda r, k=k: r['ledger'][k]) for k in rows[0]['ledger']}
    modes = sorted({m for r in rows for m in r['distance_by_mode']})
    return {'n': n, 'score': mean(lambda r: r['score']), 'ceiling': mean(lambda r: r['ceiling']),
            'gap': mean(lambda r: r['gap']), 'causes': causes,
            'samples_share': mean(lambda r: r['collected'] / r['total']), 'returned': mean(lambda r: r['returned']),
            'distance': mean(lambda r: r['distance']), 'oracle_distance': mean(lambda r: r['oracle_distance']),
            'path_ratio_same_samples': round(float(np.median(ratios)), 2) if ratios else None,
            'battery_left_returned': round(float(np.mean([r['battery_left'] for r in rows if r['returned']] or [0])), 2),
            'time': mean(lambda r: r['time']), 'ledger': ledger,
            'distance_by_mode': {m: mean(lambda r, m=m: r['distance_by_mode'].get(m, 0.0)) for m in modes}}


def render(title, s):
    out = [f'### {title}', '',
           f"Прогонов {s['n']}. Счёт {s['score']}, потолок оракула {s['ceiling']}, недобор {s['gap']}. "
           f"Собрано {s['samples_share']:.0%}, возврат {s['returned']:.0%}.", '',
           '| Причина | Очков на прогон | Прогонов |', '|---|---:|---:|']
    for c in sorted(s['causes'], key=lambda c: -c['points']):
        out.append(f"| {c['label']} | {c['points']:.2f} | {c['runs']} из {s['n']} |")
    out += ['', f"Путь {s['distance']} м при пути оракула {s['oracle_distance']} м на все образцы; на те же образцы, "
            f"что собрал агент, он проехал в {s['path_ratio_same_samples']} раза больше оракула (медиана). "
            f"Остаток заряда у вернувшихся {s['battery_left_returned']} ед., время {s['time']} с.",
            'Заряд, ед.: ' + ', '.join(f'{LEDGER[k]} {v}' for k, v in s['ledger'].items()) + '.',
            'Путь по режимам, м: ' + ', '.join(f'{k} {v}' for k, v in s['distance_by_mode'].items()) + '.', '']
    return '\n'.join(out)


LEDGER = {'floor': 'путь по обычному полу', 'soil': 'наценка грунта', 'hazard': 'потери в зонах',
          'idle': 'включённость', 'turn': 'повороты', 'load': 'груз', 'other': 'прочее (утечка)'}


def breakdown(folder, level=None, jobs=3):
    files = sorted(Path(folder).glob(f"{level or '*'}-*.json.gz"))
    if not files:
        raise SystemExit(f'нет записей: {folder}')
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(_analyze_file, files, chunksize=4))


def _run_job(a):
    from did.runner import run_episode
    agent, level, seed, rules, exp, arm = a
    return run_episode(level, seed, agent, experiment=exp, arm=arm, rules=rules)['id']


def run_series(agents, levels, seeds, rules, jobs, exp='P1'):
    """Прогнать агентов на сценариях и вернуть папки с записями."""
    from did.runner import RUNS
    tag = rules or 'base'
    tasks = [(a, lv, s, rules, exp, f'{a}@{tag}') for a in agents for lv in levels for s in seeds]
    with ProcessPoolExecutor(jobs) as pool:
        list(pool.map(_run_job, tasks, chunksize=2))
    return [RUNS / exp / f'{a}@{tag}' for a in agents]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='*', help='папки с записями прогонов (runs/<опыт>/<вариант>)')
    ap.add_argument('--level', default=None, help='один уровень; по умолчанию medium и hard по отдельности')
    ap.add_argument('--run', default=None, help='сначала прогнать этих агентов (через запятую)')
    ap.add_argument('--rules', default=None, choices=['science'])
    ap.add_argument('--levels', default='medium,hard')
    ap.add_argument('--seeds', default='1-80', help='диапазон сценариев для --run, например 1-80')
    ap.add_argument('--exp', default='P1', help='папка в runs/ для --run')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--json', default=None, help='куда сохранить разбор по каждому прогону')
    args = ap.parse_args()
    folders = [Path(f) for f in args.folders]
    levels = [args.level] if args.level else args.levels.split(',')
    if args.run:
        a, b = (int(v) for v in args.seeds.split('-'))
        folders += run_series(args.run.split(','), levels, range(a, b + 1), args.rules, min(args.jobs, 3), args.exp)
    dump = {}
    for folder in folders:
        for level in levels:
            rows = breakdown(folder, level, min(args.jobs, 3))
            dump[f'{folder.name}/{level}'] = {'summary': summarize(rows), 'runs': rows}
            print(render(f'{folder.parent.name}/{folder.name}, {level}', dump[f'{folder.name}/{level}']['summary']))
    if args.json:
        Path(args.json).write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
