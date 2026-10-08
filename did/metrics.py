"""Метрики одного прогона и сравнение серий.

Главные три — те же, что на слайде гипотезы: доля собранных образцов, возврат на базу, потраченный
заряд. Остальное объясняет, почему они получились такими.
"""
import math
import re

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
    # проверка гипотез против скрытой правды сценария (score_hypotheses)
    'hyp_correct_share': ('Доля верных гипотез', 'доля', 'higher'),
    'hyp_confirmed_correct': ('Верно подтверждённых гипотез', 'доля', 'higher'),
    'hyp_refuted_correct': ('Верных среди опровергнутых', 'доля', 'lower'),
    'hyp_soil_error': ('Ошибка множителя грунта', 'доля', 'lower'),
    'hyp_verified_share': ('Охват сверкой: доля гипотез, которые удалось сверить', 'доля', 'higher'),
    # исследования по заданию (did/study.py): оценка робота против скрытой правды сценария
    'study_error_pct': ('Ошибка оценки', '% от истины', 'lower'),
    'study_covered': ('Истина в заявленном интервале 95%', 'доля прогонов', 'higher'),
    'study_halfwidth_pct': ('Заявленная погрешность, 95%', '% от оценки', 'lower'),
    'study_reached': ('Требуемая точность достигнута', 'доля прогонов', 'higher'),
    'study_measurements': ('Учтённых замеров', 'шт.', 'lower'),
    'study_energy': ('Заряд на исследование', 'ед.', 'lower'),
    'study_time': ('Время исследования', 'с', 'lower'),
    'llm_calls': ('Обращений к модели', 'вызовы', 'lower'),
    'llm_failed': ('Ошибок модели', 'вызовы', 'lower'),
    # команда роботов (did/judge_team.py, did/team_runner.py); у обычных прогонов этих метрик нет
    'score_per_robot': ('Счёт на одного робота', 'очки', 'higher'),
    't_last_collect': ('Время до последнего сбора', 'с', 'lower'),
    'mean_sample_time': ('Среднее время до образца (несобранный — 600 с)', 'с', 'lower'),
    'robot_contacts': ('Столкновения роботов друг с другом', 'шт.', 'lower'),
    'same_target_s': ('Оба едут к одной цели', 'с', 'lower'),
    'messages': ('Сообщений между роботами', 'шт.', 'lower'),
}

# Метрики, добавленные после остальных. Их интервалы в сводке опыта считаются на отдельном генераторе:
# иначе лишние выборки сдвинули бы случайные числа и интервалы всех прежних метрик во всех опытах.
SIDE_METRICS = ('llm_calls', 'llm_failed', 'score_per_robot', 't_last_collect', 'mean_sample_time', 'robot_contacts',
                'same_target_s', 'messages')


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
    return _diff_stats(_paired_diffs(runs_a, runs_b, metric).values(), rng)


def paired_did(runs_a, runs_b, base_a, base_b, metric, rng=None):
    """Парная разность разностей: (a − b) в условии с изменением минус (a − b) в условии без него.

    Отвечает на вопрос «помогает ли механизм именно при изменении среды». Берутся только сценарии,
    пройденные всеми четырьмя группами; в ответе a_higher — в скольких сценариях разность больше нуля.
    """
    change, base = _paired_diffs(runs_a, runs_b, metric), _paired_diffs(base_a, base_b, metric)
    return _diff_stats([change[k] - base[k] for k in change if k in base], rng)


def _paired_diffs(runs_a, runs_b, metric):
    by_b = {(r['level'], r['seed']): r for r in runs_b}
    diffs = {}
    for ra in runs_a:
        key = (ra['level'], ra['seed'])
        rb = by_b.get(key)
        if rb is None:
            continue
        va, vb = ra['metrics'].get(metric), rb['metrics'].get(metric)
        if va is None or vb is None:
            continue
        diffs[key] = float(va) - float(vb)
    return diffs


def _diff_stats(diffs, rng=None):
    d = np.array(list(diffs))
    if len(d) == 0:
        return None
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


def verdict(pair, better, kind=None, margin=None):
    """Вывод по одному сравнению.

    По умолчанию — «лучше»: интервал разности целиком по нужную сторону от нуля. С заранее названным
    допуском margin: kind='noninferiority' — «не хуже»: интервал целиком не ниже −margin (в сторону
    «лучше» не ограничен); kind='equivalence' — «не отличается»: интервал целиком внутри ±margin.
    Опровергнуто — когда интервал целиком по другую сторону границы.
    """
    if pair is None:
        return 'no_data'
    lo, hi = pair['ci']
    if better == 'lower':
        lo, hi = -hi, -lo
    if kind == 'noninferiority':
        return 'supported' if lo >= -margin else 'refuted' if hi < -margin else 'inconclusive'
    if kind == 'equivalence':
        return 'supported' if -margin <= lo and hi <= margin else 'refuted' if hi < -margin or lo > margin \
            else 'inconclusive'
    if lo > 0:
        return 'supported'
    if hi < 0:
        return 'refuted'
    return 'inconclusive'


def _outline_points(z, x, y):
    """Точки границы зоны, ближайшие к (x, y): для круга одна, для прямоугольника — по одной на сторону и углы."""
    if z.shape == 'circle':
        d = math.hypot(x - z.x, y - z.y)
        return [(z.x + z.r * (x - z.x) / d, z.y + z.r * (y - z.y) / d) if d else (z.x + z.r, z.y)]
    x0, x1, y0, y1 = z.x - z.w / 2, z.x + z.w / 2, z.y - z.h / 2, z.y + z.h / 2
    cx, cy = min(max(x, x0), x1), min(max(y, y0), y1)
    return [(cx, y0), (cx, y1), (x0, cy), (x1, cy), (x0, y0), (x0, y1), (x1, y0), (x1, y1)]


def _outline_crossings(a, b):
    """Точки пересечения границ двух зон (круги и прямоугольники со сторонами вдоль осей)."""
    def sides(z):
        x0, x1, y0, y1 = z.x - z.w / 2, z.x + z.w / 2, z.y - z.h / 2, z.y + z.h / 2
        return [('h', y0, x0, x1), ('h', y1, x0, x1), ('v', x0, y0, y1), ('v', x1, y0, y1)]

    out = []
    if a.shape == 'circle' and b.shape == 'circle':
        d = math.hypot(b.x - a.x, b.y - a.y)
        if d and abs(a.r - b.r) <= d <= a.r + b.r:
            k = (d * d + a.r * a.r - b.r * b.r) / (2 * d)
            h = math.sqrt(max(a.r * a.r - k * k, 0.0))
            ux, uy = (b.x - a.x) / d, (b.y - a.y) / d
            out = [(a.x + k * ux - s * h * uy, a.y + k * uy + s * h * ux) for s in (-1, 1)]
    elif a.shape == 'circle' or b.shape == 'circle':
        c, r = (a, b) if a.shape == 'circle' else (b, a)
        for kind, at, lo, hi in sides(r):
            off = at - (c.y if kind == 'h' else c.x)
            if abs(off) <= c.r:
                h = math.sqrt(c.r * c.r - off * off)
                for u in ((c.x if kind == 'h' else c.y) - h, (c.x if kind == 'h' else c.y) + h):
                    if lo <= u <= hi:
                        out.append((u, at) if kind == 'h' else (at, u))
    else:
        for ka, at_a, lo_a, hi_a in sides(a):
            for kb, at_b, lo_b, hi_b in sides(b):
                if ka != kb and lo_a <= at_b <= hi_a and lo_b <= at_a <= hi_b:
                    out.append((at_b, at_a) if ka == 'h' else (at_a, at_b))
    return out


def changed_distance(start, now, x, y, eps=1e-6):
    """Расстояние от (x, y) до пола, где множитель расхода сейчас (now) не тот, что был (start).

    Точное, с учётом перекрытия зон: изменившаяся область ограничена дугами и отрезками границ зон, так что
    ближайшая её точка — либо ближайшая точка границы одной из зон, либо пересечение двух границ. Каждая
    такая точка проверяется: изменился ли множитель вплотную к ней. Если изменившегося пола нет — inf.
    """
    from .scenario import soil_mult

    def changed(px, py):
        return soil_mult(now, px, py) != soil_mult(start, px, py)

    if changed(x, y):
        return 0.0
    zones = list({(z.shape, z.x, z.y, z.r, z.w, z.h): z for z in list(start) + list(now)}.values())
    points = [p for z in zones for p in _outline_points(z, x, y)]
    points += [p for i, a in enumerate(zones) for b in zones[i + 1:] for p in _outline_crossings(a, b)]
    around = [(eps * math.cos(k * math.pi / 8), eps * math.sin(k * math.pi / 8)) for k in range(16)]
    best = math.inf
    for px, py in points:
        d = math.hypot(px - x, py - y)
        if d < best and any(changed(px + dx, py + dy) for dx, dy in around):
            best = d
    return best


class SoilProbe:
    """Разбор смены грунта по скрытой правде: что изменение стоило роботу и верны ли его тревоги.

    Считает судья-наблюдатель в быстром симуляторе, агент этих чисел не видит. «Изменившийся пол» —
    место, где множитель расхода сейчас не тот, что в начале прогона.
    """

    NEAR = 0.3        # м: тревога «по делу», если изменившийся пол не дальше этого от места тревоги

    def __init__(self, scenario, rules):
        from .scenario import soil_mult
        self._mult = soil_mult
        self.start = scenario.soils
        self.changes = [(e['t'], e['soils']) for e in scenario.events if e['type'] == 'soil_change']
        self.per_m = rules.drain_per_m
        self.changed_m = self.dearer_m = self.extra = 0.0
        self.first_t = None           # когда робот впервые оказался на изменившемся полу
        self.entries = 0              # сколько раз въезжал на подорожавший пол
        self.entry_known = None       # был ли этот пол знаком агенту при первом въезде
        self._on_dearer = False

    def step(self, a, b, soils, t, known=None):
        """Шаг робота из a в b при действующих грунтах soils.

        known(x, y) — считал ли агент перед этим шагом, что знает цену пола в точке: тревогу
        «модель устарела» он поднимает только на знакомом полу.
        """
        ds = math.dist(a, b)
        if ds <= 0.0 or not self.changes or t < self.changes[0][0]:
            return
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        delta = self._mult(soils, mx, my) - self._mult(self.start, mx, my)
        if delta:
            self.changed_m += ds
            self.extra += self.per_m * ds * delta
            if self.first_t is None:
                self.first_t = t
        if delta > 0:
            self.dearer_m += ds
            if not self._on_dearer:
                self.entries += 1
                if self.entries == 1 and known is not None:
                    self.entry_known = bool(known(mx, my))
        self._on_dearer = delta > 0

    def _changed_near(self, x, y, t):
        soils = self.start
        for t_change, after in self.changes:
            if t_change <= t:
                soils = after
        if soils is self.start:
            return False
        return changed_distance(self.start, soils, x, y) <= self.NEAR + 1e-9      # порог включительно, с запасом на округление

    def metrics(self, journal):
        alarms = [(e['t'], e['data']['x'], e['data']['y']) for e in journal.entries
                  if e.get('data', {}).get('tag') == 'model_mismatch' and 'x' in e['data']]
        true = [t for t, x, y in alarms if self._changed_near(x, y, t)]
        after = [t for t in true if self.first_t is not None and t >= self.first_t]
        return {'soil_changed_m': round(self.changed_m, 3), 'soil_dearer_m': round(self.dearer_m, 3),
                'soil_extra_energy': round(self.extra, 3),
                'soil_dearer_entries': self.entries, 'soil_entry_known': self.entry_known,
                'soil_alarms': len(alarms), 'soil_alarms_true': len(true), 'soil_alarms_false': len(alarms) - len(true),
                'soil_alarm_delay': round(min(after) - self.first_t, 1) if after else None}


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
    changes = _applied_events(scenario, world, 'soil_change')
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
        # странность возникает раньше, чем открыто расследование; сверка у образца смотрит только на последнюю секунду
        lead = 1.0 if q['anomaly'].get('trigger') == 'collect' else 4.0
        t0, t1 = q['t_open'] - lead, q['t_close'] or q['t_open']
        active = {k for k, a, b in faults if a <= t1 and b >= t0}
        if q['topic'] == 'energy':
            soils = scenario.soils
            for change_t, ev in changes:           # действует последняя смена грунта, применённая к этому моменту
                if change_t <= q['t_open']:
                    soils = ev['soils']
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


SOIL_NEAR_M = 0.2       # окрестность названной точки, в которой ищется дорогой грунт


def _zone_distance(z, x, y):
    """Расстояние от точки до зоны (круг или прямоугольник со сторонами вдоль осей); внутри зоны — ноль."""
    get = (lambda k, d=0.0: z.get(k, d)) if isinstance(z, dict) else (lambda k, d=0.0: getattr(z, k, d))
    if (get('shape', 'circle') or 'circle') == 'circle':
        return max(0.0, math.dist((x, y), (get('x'), get('y'))) - get('r'))
    return math.hypot(max(0.0, abs(x - get('x')) - get('w') / 2), max(0.0, abs(y - get('y')) - get('h') / 2))


def _applied_events(scenario, world, kind):
    """События сценария одного вида с временем, когда судья их применил: [(время, событие)] по порядку.

    Судья применяет события в порядке расписания и на каждое пишет запись в свой журнал, поэтому k-е событие
    вида в сценарии — это k-я запись этого вида в журнале; сопоставление по порядку однозначно и тогда, когда
    времена совпали. Если записи нет (журнал не передан или прогон кончился раньше), берётся время из сценария.
    """
    planned = [ev for ev in (getattr(scenario, 'events', []) or []) if ev.get('type') == kind]
    done = [float(w['t']) for w in world if w.get('type') == kind]
    return [(done[k] if k < len(done) else float(ev.get('t', 0.0)), ev) for k, ev in enumerate(planned)]


def score_hypotheses(hypotheses, scenario, world, events=None):
    """Сверка гипотез агента со скрытой правдой сценария.

    Исходы:
      'correct'      — верна
      'wrong'        — неверна
      'unverifiable' — проверить нечем

    Виды (вид берётся из ключа гипотезы; текст — только для старых записей без распознаваемого ключа):
      - sample: верна, если в момент выдвижения в радиусе 0.35 м лежал несобранный образец
        (образцы собираются по ходу прогона — время сбора есть в events записи);
      - soil: верна, если в круге радиуса 0.2 м вокруг названной точки в тот момент действительно был дорогой
        грунт (множитель >= 1.4; действует последняя из смен грунта, применённых к тому моменту);
        дополнительно — относительная ошибка названного множителя;
      - hazard: верна, если названное место пересекается с настоящей зоной, существовавшей к тому моменту
        (каждая новая зона — со своего времени появления);
      - sensor: верна, если в тот момент действительно шёл сбой датчика.

    Доли верных считаются только среди гипотез, которые удалось сверить; 'verified_share' — какую долю
    журнала сверка охватила.
    """
    from .scenario import Scenario
    if hasattr(hypotheses, 'hypotheses'):
        hyp_list = hypotheses.hypotheses
    else:
        hyp_list = list(hypotheses)

    if isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)

    world = world or []
    events = events or []

    def get_kind(h):
        k = h.get('key', '')
        s = h.get('statement', '')
        if k.startswith('law:'):
            return 'law'
        for name in ('sample', 'soil', 'hazard'):
            if k == name or k.startswith(name + ':'):
                return name
        if k.startswith('sensor'):
            return 'sensor'
        # Старые записи без распознаваемого ключа: вид угадывается по тексту.
        if 'образец' in s:
            return 'sample'
        if 'грунт' in s:
            return 'soil'
        if 'опасн' in s:
            return 'hazard'
        if 'датчик' in s:
            return 'sensor'
        return 'unknown'

    def get_point(h):
        d = h.get('data') or {}
        if 'x' in d and 'y' in d and d['x'] is not None and d['y'] is not None:
            return float(d['x']), float(d['y'])
        m = re.search(r'\(([-+]?[0-9]*\.?[0-9]+);\s*([-+]?[0-9]*\.?[0-9]+)\)', h.get('statement', ''))
        if m:
            return float(m.group(1)), float(m.group(2))
        return None

    # Предварительный разбор событий сбора образцов
    collected_samples = []
    for e in events:
        if e.get('type') == 'sample_collected':
            t_col = float(e.get('t', 0.0))
            s_idx = e.get('sample')
            collected_samples.append((t_col, s_idx, (e.get('x'), e.get('y'))))

    faults = fault_intervals(world)
    soil_changes = _applied_events(scenario, world, 'soil_change')
    new_hazards = _applied_events(scenario, world, 'new_hazard')

    for h in hyp_list:
        kind = get_kind(h)
        h['kind'] = kind
        t_open = float(h.get('t_open', 0.0))

        if kind == 'sample':
            pt = get_point(h)
            if pt is None or scenario is None or not hasattr(scenario, 'samples'):
                verdict = 'unverifiable'
            else:
                collected_before = set()
                for t_col, s_idx, (ex, ey) in collected_samples:
                    if t_col < t_open - 1e-4:
                        if s_idx is not None and s_idx < len(scenario.samples):
                            collected_before.add(s_idx)
                        elif ex is not None and ey is not None:
                            for i, sp in enumerate(scenario.samples):
                                if math.dist((ex, ey), sp) <= 0.15:
                                    collected_before.add(i)
                uncollected = [tuple(p) for i, p in enumerate(scenario.samples) if i not in collected_before]
                if uncollected and min(math.dist(pt, sp) for sp in uncollected) <= 0.35:
                    verdict = 'correct'
                else:
                    verdict = 'wrong'

        elif kind == 'soil':
            pt = get_point(h)
            if pt is None or scenario is None or not hasattr(scenario, 'soils'):
                verdict = 'unverifiable'
            else:
                active_soils = list(scenario.soils)
                for ev_t, ev in soil_changes:              # каждая смена задаёт грунты целиком; действует последняя
                    if ev_t <= t_open:
                        active_soils = list(ev.get('soils', []))

                hx, hy = pt
                # Самый дорогой грунт, который задевает круг радиуса SOIL_NEAR_M вокруг названной точки.
                true_mult = max([1.0] + [float(z['mult'] if isinstance(z, dict) else z.mult) for z in active_soils
                                         if _zone_distance(z, hx, hy) <= SOIL_NEAR_M + 1e-9])
                verdict = 'correct' if true_mult >= 1.4 else 'wrong'

                named_mult = (h.get('data') or {}).get('mult')
                if named_mult is None:
                    m_mult = re.search(r'в\s+([\d\.]+)\s+раза', h.get('statement', ''))
                    if m_mult:
                        named_mult = float(m_mult.group(1))
                if named_mult is not None:
                    rel_err = abs(float(named_mult) - true_mult) / true_mult
                    h['rel_error'] = round(float(rel_err), 4)

        elif kind == 'hazard':
            pt = get_point(h)
            if pt is None or scenario is None or not hasattr(scenario, 'hazards'):
                verdict = 'unverifiable'
            else:
                d = h.get('data') or {}
                hr = d.get('r')
                if hr is None:
                    rm = re.search(r'радиусом около\s+([\d\.]+)', h.get('statement', ''))
                    hr = float(rm.group(1)) if rm else 0.3
                else:
                    hr = float(hr)

                active_hazards = list(scenario.hazards)
                active_hazards += [ev['zone'] for ev_t, ev in new_hazards if ev_t <= t_open]

                hx, hy = pt
                intersects = any(_zone_distance(z, hx, hy) <= hr for z in active_hazards)
                verdict = 'correct' if intersects else 'wrong'

        elif kind == 'sensor':
            active = any(k.startswith('sensor') and a - 0.5 <= t_open <= b + 0.5 for k, a, b in faults)
            if not active and scenario is not None:
                sc_events = getattr(scenario, 'events', []) or []
                for ev in sc_events:
                    if ev.get('type') == 'sensor_fault':
                        t0 = float(ev.get('t', 0.0))
                        t1 = t0 + float(ev.get('duration', 0.0))
                        if t0 - 0.5 <= t_open <= t1 + 0.5:
                            active = True
                            break
            verdict = 'correct' if active else 'wrong'

        else:
            verdict = 'unverifiable'

        h['truth_verdict'] = verdict

    total = len(hyp_list)
    correct = sum(1 for h in hyp_list if h.get('truth_verdict') == 'correct')
    wrong = sum(1 for h in hyp_list if h.get('truth_verdict') == 'wrong')
    unverifiable = sum(1 for h in hyp_list if h.get('truth_verdict') == 'unverifiable')

    confirmed = [h for h in hyp_list if h.get('status') == 'confirmed']
    refuted = [h for h in hyp_list if h.get('status') == 'refuted']

    # Доли считаются среди гипотез, которые удалось сверить: «проверить нечем» — это не «неверна».
    def verified(items):
        return sum(1 for h in items if h.get('truth_verdict') in ('correct', 'wrong'))

    confirmed_correct = sum(1 for h in confirmed if h.get('truth_verdict') == 'correct')
    refuted_correct = sum(1 for h in refuted if h.get('truth_verdict') == 'correct')
    all_soil_errors = [h['rel_error'] for h in hyp_list if 'rel_error' in h]

    confusion = {
        st: {v: sum(1 for h in hyp_list if h.get('status') == st and h.get('truth_verdict') == v)
             for v in ('correct', 'wrong', 'unverifiable')}
        for st in ('confirmed', 'refuted', 'outdated', 'open')
    }

    by_kind = {}
    for k in ('sample', 'soil', 'hazard', 'sensor', 'law'):
        kh = [h for h in hyp_list if h.get('kind') == k]
        conf = [h for h in kh if h.get('status') == 'confirmed']
        ref = [h for h in kh if h.get('status') == 'refuted']
        k_errors = [h['rel_error'] for h in kh if 'rel_error' in h]
        by_kind[k] = {
            'total': len(kh),
            'correct': sum(1 for h in kh if h.get('truth_verdict') == 'correct'),
            'wrong': sum(1 for h in kh if h.get('truth_verdict') == 'wrong'),
            'unverifiable': sum(1 for h in kh if h.get('truth_verdict') == 'unverifiable'),
            'verified': verified(kh),
            'confirmed': len(conf),
            'confirmed_correct': sum(1 for h in conf if h.get('truth_verdict') == 'correct'),
            'refuted': len(ref),
            'refuted_correct': sum(1 for h in ref if h.get('truth_verdict') == 'correct'),
            'confusion': {
                st: {v: sum(1 for h in kh if h.get('status') == st and h.get('truth_verdict') == v)
                     for v in ('correct', 'wrong', 'unverifiable')}
                for st in ('confirmed', 'refuted', 'outdated', 'open')
            },
            'rel_error_mean': round(float(np.mean(k_errors)), 4) if k_errors else None,
        }

    return {
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'unverifiable': unverifiable,
        'verified_share': round((correct + wrong) / total, 4) if total > 0 else None,
        'correct_share': round(correct / (correct + wrong), 4) if correct + wrong > 0 else None,
        'confirmed_correct': round(confirmed_correct / verified(confirmed), 4) if verified(confirmed) else None,
        'refuted_correct': round(refuted_correct / verified(refuted), 4) if verified(refuted) else None,
        'soil_error': round(float(np.mean(all_soil_errors)), 4) if all_soil_errors else None,
        'by_kind': by_kind,
        'confusion': confusion,
    }
