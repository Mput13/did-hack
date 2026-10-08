"""Метрики одного прогона и сравнение серий.

Главные три — те же, что на слайде гипотезы: доля собранных образцов, возврат на базу, потраченный
заряд. Остальное объясняет, почему они получились такими.
"""
import math

import numpy as np

# Какая тревога агента отвечает на какое скрытое событие среды.
DETECTS = {'soil_change': 'model_mismatch', 'new_hazard': 'hazard', 'sensor_fault': 'sensor_degraded'}

METRICS = {
    # id: (подпись, единица, что лучше)
    'score': ('Счёт', 'очки', 'higher'),
    'samples_share': ('Собрано образцов', 'доля', 'higher'),
    'returned': ('Вернулся на базу', 'доля прогонов', 'higher'),
    'battery_used': ('Потрачено заряда', 'ед.', 'lower'),
    'energy_per_sample': ('Заряд на один образец', 'ед.', 'lower'),
    'distance': ('Путь', 'м', 'lower'),
    'time': ('Время прогона', 'с', 'lower'),
    'penalties': ('Штрафы', 'шт.', 'lower'),
    'false_collects': ('Ложные сборы', 'шт.', 'lower'),
    'hazard_hits': ('Заезды в опасную зону', 'шт.', 'lower'),
    'collisions': ('Столкновения', 'шт.', 'lower'),
    'inq_total': ('Расследований за прогон', 'шт.', 'lower'),
    'inq_correct': ('Верно названная причина', 'доля выводов', 'higher'),
    'inq_wrong': ('Ошибочных выводов за прогон', 'шт.', 'lower'),
    'inq_insufficient': ('Исход «недостаточно данных» за прогон', 'шт.', 'lower'),
    'inq_energy': ('Заряд на опыты', 'ед.', 'lower'),
    'faults_found': ('Найдено сбоев', 'доля', 'higher'),
    # исследования по заданию (did/study.py): оценка робота против скрытой правды сценария
    'study_error_pct': ('Ошибка оценки', '% от истины', 'lower'),
    'study_covered': ('Истина в заявленном интервале 95%', 'доля прогонов', 'higher'),
    'study_halfwidth_pct': ('Заявленная погрешность, 95%', '% от оценки', 'lower'),
    'study_reached': ('Требуемая точность достигнута', 'доля прогонов', 'higher'),
    'study_measurements': ('Учтённых замеров', 'шт.', 'lower'),
    'study_energy': ('Заряд на исследование', 'ед.', 'lower'),
    'study_time': ('Время исследования', 'с', 'lower'),
}


def run_metrics(score, rules, journal, world, plans, llm):
    """Плоский словарь метрик прогона. score — итог судьи, world — применённые события среды."""
    used = rules.battery_start - score['battery']
    n = score['samples_collected']
    m = {
        'score': score['score'],
        'samples_collected': n,
        'samples_total': score['samples_total'],
        'samples_share': round(n / max(1, score['samples_total']), 3),
        'returned': bool(score['returned']),
        'reason': score['reason'],
        'battery_left': round(score['battery'], 2),
        'battery_used': round(used, 2),
        'energy_per_sample': round(used / n, 2) if n else None,
        'distance': score['distance'],
        'time': score['t'],
        'collisions': score['collisions'],
        'false_collects': score['false_collects'],
        'hazard_hits': score['hazard_hits'],
        'penalties': score['collisions'] + score['false_collects'] + score['hazard_hits'],
    }
    # Сколько секунд прошло от скрытого изменения среды до тревоги агента.
    alarms = [(e['t'], e.get('data', {}).get('tag')) for e in journal.entries if e['kind'] in ('alarm', 'inquiry')]
    detect = {}
    for ev in world:
        tag = DETECTS.get(ev['type'])
        if tag:
            later = [t for t, g in alarms if g == tag and t >= ev['t']]
            detect[ev['type']] = round(min(later) - ev['t'], 1) if later else None
    m['detect'] = detect
    status = [h['status'] for h in journal.hypotheses]
    m['hypotheses'] = {'total': len(status), **{s: status.count(s) for s in ('confirmed', 'refuted', 'outdated', 'open')}}
    sources = [p['source'] for p in plans]
    m['plans'] = {s: sources.count(s) for s in sorted(set(sources))}
    m['llm_calls'] = len(llm)
    m['llm_failed'] = sum(1 for x in llm if not x.get('ok', True))
    return m


def _values(runs, metric):
    out = []
    for r in runs:
        v = r['metrics'].get(metric)
        if v is not None:
            out.append(float(v))
    return np.array(out)


def summarize(runs, metric, rng=None):
    """Среднее, медиана, квартили и 95% интервал среднего (бутстреп) по списку прогонов."""
    v = _values(runs, metric)
    if len(v) == 0:
        return None
    rng = rng or np.random.default_rng(0)
    boot = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1) if len(v) > 1 else np.array([v[0]] * 2)
    return {'n': int(len(v)), 'mean': round(float(v.mean()), 3), 'median': round(float(np.median(v)), 3),
            'q1': round(float(np.percentile(v, 25)), 3), 'q3': round(float(np.percentile(v, 75)), 3),
            'min': round(float(v.min()), 3), 'max': round(float(v.max()), 3),
            'ci': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}


def paired(runs_a, runs_b, metric, rng=None):
    """Разность a − b на одинаковых сценариях: среднее, 95% интервал, сколько раз кто выиграл."""
    by_b = {(r['level'], r['seed']): r for r in runs_b}
    diffs = []
    for ra in runs_a:
        rb = by_b.get((ra['level'], ra['seed']))
        if rb is None:
            continue
        va, vb = ra['metrics'].get(metric), rb['metrics'].get(metric)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    if not diffs:
        return None
    d = np.array(diffs)
    rng = rng or np.random.default_rng(0)
    boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(axis=1) if len(d) > 1 else np.array([d[0]] * 2)
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {'n': int(len(d)), 'mean': round(float(d.mean()), 3),
            'ci': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            'a_higher': wins, 'b_higher': losses, 'ties': int(len(d) - wins - losses),
            'sign_p': round(_sign_test(wins, losses), 4)}


def _sign_test(wins, losses):
    """Двусторонний знаковый критерий: вероятность такого перекоса при равных шансах."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def verdict(pair, better):
    """Вывод по одному сравнению: интервал разности целиком по нужную сторону от нуля или нет."""
    if pair is None:
        return 'no_data'
    lo, hi = pair['ci']
    if better == 'lower':
        lo, hi = -hi, -lo
    if lo > 0:
        return 'supported'
    if hi < 0:
        return 'refuted'
    return 'inconclusive'


# ---------------------------------------------------------------------------------------------
# качество расследований: сверка выводов агента со скрытой правдой сценария
# ---------------------------------------------------------------------------------------------

def fault_intervals(world):
    """Когда какой сбой действовал: [(вид, начало, конец)] по журналу судьи.

    Повторный штраф в той же зоне продлевает идущий сбой, а не начинает новый: перекрывающиеся
    записи одного вида сливаются в один эпизод.
    """
    raw, open_sensor = [], {}
    for w in world:
        if w['type'] == 'fault':
            raw.append((w['kind'], w['t'], w['until']))
        elif w['type'] == 'sensor_fault':
            open_sensor[w.get('kind', 'sensor_noise')] = w['t']
        elif w['type'] == 'sensor_recovered' and w.get('kind') in open_sensor:
            raw.append((w['kind'], open_sensor.pop(w['kind']), w['t']))
    raw += [(k, t0, 1e9) for k, t0 in open_sensor.items()]
    out = []
    for kind, a, b in sorted(raw):
        if out and out[-1][0] == kind and a <= out[-1][2] + 0.5:
            out[-1] = (kind, out[-1][1], max(out[-1][2], b))
        else:
            out.append((kind, a, b))
    return out


def score_inquiries(inquiries, scenario, world):
    """Сверка выводов исследователя со скрытой правдой сценария.

    Исходы: correct — названа настоящая причина; partial — причин на самом деле было две, названа одна;
    wrong — названа не та; insufficient — агент честно не вынес вывода; unverifiable — проверить нечем.
    Выводы «по исключению» (без нового опыта) считаются наравне с остальными.
    """
    from .scenario import soil_mult
    faults = fault_intervals(world)
    change = next((w['t'] for w in world if w['type'] == 'soil_change'), None)
    after = next((e['soils'] for e in scenario.events if e['type'] == 'soil_change'), None)
    out = {'total': 0, 'identified': 0, 'insufficient': 0, 'correct': 0, 'partial': 0, 'wrong': 0, 'unverifiable': 0,
           'quick': 0, 'tests': 0, 'energy': 0.0, 'by_cause': {}}
    for q in inquiries:
        c = q.get('conclusion')
        if not c:
            continue
        out['total'] += 1
        measured = [x for x in q['tests'] if x.get('measured')]
        out['tests'] += len(measured)
        out['energy'] += sum(x['cost'] for x in measured)       # цена — заряд, реально ушедший на манёвр
        out['quick'] += 0 if measured else 1
        t0, t1 = q['t_open'] - 4.0, q['t_close'] or q['t_open']
        active = {k for k, a, b in faults if a <= t1 and b >= t0}
        if q['topic'] == 'energy':
            soils = after if (change is not None and after is not None and change <= q['t_open']) else scenario.soils
            x, y = q['anomaly']['x'], q['anomaly']['y']
            dear = max(soil_mult(soils, x + dx, y + dy) for dx in (-0.15, 0, 0.15) for dy in (-0.15, 0, 0.15)) >= 1.4
            truth = ({'leak'} if 'leak' in active else set()) | ({'soil'} if dear else set())
        elif q['topic'] == 'fault':        # проверка батареи после штрафа
            truth = {'leak'} if 'leak' in active else {'none'}
        else:
            truth = {k.replace('sensor_', '') for k in active if k.startswith('sensor_')} or {'ok'}
        q['truth'] = sorted(truth)
        if c['status'] != 'identified':
            out['insufficient'] += 1
            verdict = 'insufficient'
        else:
            out['identified'] += 1
            verdict = ('unverifiable' if not truth else 'wrong' if c['best'] not in truth
                       else 'partial' if len(truth) > 1 else 'correct')
            out[verdict] += 1
        q['verdict'] = verdict
        key = '+'.join(sorted(truth)) or 'нет явной причины'
        cell = out['by_cause'].setdefault(key, {'n': 0, 'correct': 0, 'partial': 0, 'wrong': 0, 'insufficient': 0,
                                                'unverifiable': 0})
        cell['n'] += 1
        cell[verdict] += 1
    out['energy'] = round(out['energy'], 2)
    # сколько настоящих эпизодов сбоя агент заметил расследованием
    found = 0
    for k, a, b in faults:
        name = k.replace('sensor_', '')
        if any(q.get('conclusion') and q['conclusion']['status'] == 'identified' and q['conclusion']['best'] == name
               and q['t_open'] >= a - 1.0 and q['t_open'] <= b + 5.0 for q in inquiries):
            found += 1
    out['faults'], out['faults_found'] = len(faults), found
    return out
