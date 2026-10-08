"""Сопоставление потери заряда при штрафе с событием штрафа: свойства на случайных последовательностях.

Порождающий код знает правду — в каком показании батареи сидит какой штраф, пришло ли к нему событие и когда.
Проверяется did/penalty.py вместе с тем, как исследователь придерживает окна расхода (did/inquiry.py, _release):

 (а) штраф, у которого показание и событие разошлись не больше чем на WAIT_S, вычтен ровно один раз
     и ровно в размере из правил агента;
 (б) настоящий расход не пропадает: окно либо дошло до оценки с настоящим расходом, либо явно помечено
     испорченным, и испорченным оно бывает только по названной причине;
 (в) без события ничего не вычитается;
 (г) окно выдаётся не позже WAIT_S после своего последнего отрезка, у которого было что решать
     (необъяснённый скачок или событие рядом), а без таких отрезков — в такте закрытия;
 (д) очередь ограничена константой;
 (е) после конца прогона в очереди ничего не остаётся.
"""
import random
from types import SimpleNamespace

import pytest

from did.penalty import TOL, WAIT_S, PenaltyLedger

DT, PER_M, IDLE, HIT = 0.1, 2.6, 0.01, 3.0
W = round(WAIT_S / DT)             # WAIT_S в тактах
SEEDS = 600


def _most(ds, dth, dt, t):
    return 7.0 * PER_M * ds + IDLE * dt


def _scenario(rng):
    """Случайная поездка. Возвращает такты: путь, настоящий расход, показана ли батарея, и правду о штрафах."""
    n = rng.randint(120, 320)
    step, mult, show, off = [], [], [], []
    while len(step) < n:                                   # участки с разной скоростью и грунтом
        ln = rng.randint(5, 40)
        v = rng.choice([0.0, 0.008, 0.015, 0.015, 0.02])
        m = rng.choice([1.0, 1.0, 1.0, 2.0, 4.0])
        step += [v] * ln
        mult += [m] * ln
    while len(show) < n:                                   # показание батареи отстаёт: расход копится 1–4 такта
        lag = rng.choice([1, 1, 1, 2, 3, 4]) if rng.random() < 0.5 else 1
        for _ in range(rng.randint(3, 30) // lag + 1):
            show += [False] * (lag - 1) + [True]
    show = show[:n]
    show[0] = True
    while len(off) < n:                                    # идёт опыт: окно расхода не копится
        ln = rng.randint(20, 120)
        off += [rng.random() < 0.15] * ln
    off = off[:n]
    shown = [k for k in range(n) if show[k]]
    seg_of = {}                                            # такт → (начало, конец] отрезка между показаниями
    for a, b in zip(shown, shown[1:]):
        for k in range(a + 1, b + 1):
            seg_of[k] = (a, b)
    lo, hi = 2 * W, n - 3 * W
    pens, jumps, events = [], {}, []                       # штраф: {'k', 'seg', 'paired'}; скачок: такт → размер

    def busy(seg):
        return any(p['seg'] == seg for p in pens) or any(seg_of.get(k) == seg for k in jumps)

    def near(seg, e, pad=0):
        return seg[0] - W - pad <= e <= seg[1] + W + pad

    for _ in range(rng.choice([0, 1, 1, 2, 3, 5])):        # штрафы с событием в пределах WAIT_S от показания
        k = rng.randint(lo, hi)
        ks = [k]
        if rng.random() < 0.35:                            # два подряд: в том же такте или в соседнем
            ks.append(k + rng.choice([0, 0, 1, 2]))
        for k in ks:
            seg = seg_of.get(k)
            if seg is None or any(seg_of.get(j) == seg for j in jumps) or any(p['seg'] == seg and not p['paired'] for p in pens):
                continue
            pens.append({'k': k, 'seg': seg, 'paired': True})
            events.append(seg[1] + rng.randint(-W, W))
    for _ in range(rng.choice([0, 0, 1, 2, 4])):           # настоящие скачки без события
        k = rng.randint(lo, hi)
        if k in seg_of and not busy(seg_of[k]):             # в одном показании — один скачок: два дали бы «штраф»
            jumps[k] = rng.uniform(0.3, 1.8)
    orphans = []
    for _ in range(rng.choice([0, 0, 0, 1, 2])):           # штраф без события или с событием слишком далеко
        k = rng.randint(lo, hi)
        seg = seg_of.get(k)
        far = rng.random() < 0.5
        e = None if not far or seg is None else rng.choice([seg[0] - W - rng.randint(2, 6), seg[1] + W + rng.randint(2, 6)])
        if seg is None or busy(seg) or any(near(seg, x, 1) for x in events + orphans):
            continue                                       # такой штраф не отличить от соседнего: правды нет
        if e is not None and (not 1 <= e < n or any(near(p['seg'], e, 1) for p in pens)):
            continue
        pens.append({'k': k, 'seg': seg, 'paired': False})
        if e is not None:
            orphans.append(e)
    noise = rng.choice([0.0, 0.0, 0.005])
    return SimpleNamespace(n=n, step=step, mult=mult, show=show, off=off, seg_of=seg_of, pens=pens, jumps=jumps,
                           events=sorted(events + orphans), orphans=orphans, noise=noise,
                           cut=n if rng.random() < 0.65 else rng.randint(3 * W, n))


def _run(sc, rng, hit=HIT):
    """Прогнать последовательность через журнал штрафов и окна расхода, как это делает исследователь."""
    led = PenaltyLedger(hit, _most)
    battery, shown, x = 60.0, 60.0, 0.0
    win, held, wins, sizes, seen = None, [], [], [0, 0, 0], {}
    at = lambda k: SimpleNamespace(t=k * DT, x=x, y=0.0, battery=shown)      # noqa: E731

    def release(k):
        while held and not led.waits(held[0]):
            held.pop(0)['released'] = k

    for k in range(sc.cut):
        if k:
            x += sc.step[k]
            battery -= PER_M * sc.mult[k] * sc.step[k] + IDLE * DT
            battery -= HIT * sum(1 for p in sc.pens if p['k'] == k) + sc.jumps.get(k, 0.0)
        if sc.show[k]:
            shown = battery + rng.gauss(0.0, sc.noise)
        for _ in range(sc.events.count(k)):
            led.event(k * DT)
        obs = at(k)
        seen[k] = shown
        cur = None if sc.off[k] else win
        led.reading(obs, cur)
        release(k)
        if sc.off[k]:
            win = None
        elif win is None:
            win = {'b0': shown, 'open': k, 'raw': shown}
        elif k - win['open'] >= 8:
            win.update(b1=shown, close=k)
            held.append(win)
            wins.append(win)
            release(k)
            win = {'b0': shown, 'open': k, 'raw': shown}
        sizes = [max(sizes[0], len(led._segs)), max(sizes[1], len(led._events)), max(sizes[2], len(held))]
    led.flush(sc.cut * DT)
    release(sc.cut)
    return led, wins, held, sizes, seen


def _check(sc, led, wins, held, sizes, seen):
    segs = sorted(set(sc.seg_of.values()))
    segs = [s for s in segs if s[1] < sc.cut]
    # Падение заряда по показаниям и расход того же отрезка на самом дорогом грунте: что считать скачком.
    drop = {(a, b): (seen[a] - seen[b], _most(sum(sc.step[k] for k in range(a + 1, b + 1)), 0.0, (b - a) * DT, b * DT))
            for a, b in segs}
    paired = {s: sum(1 for p in sc.pens if p['seg'] == s and p['paired']) for s in segs}
    is_pen = {s: any(p['seg'] == s for p in sc.pens) for s in segs}
    odd = {s: drop[s][0] > drop[s][1] + TOL for s in segs}
    near = lambda s, e: s[0] - W <= e <= s[1] + W          # noqa: E731

    assert not led._segs and not led._events and not held                      # (е)
    assert all('released' in w for w in wins)
    assert sizes[0] <= W + 2 and sizes[1] <= 8 and sizes[2] <= 3, sizes        # (д)
    full = sc.cut == sc.n
    for w in wins:
        mine = [s for s in segs if w['open'] < s[1] <= w['close']]
        took = (w['raw'] - w['b0']) / HIT
        assert abs(took - round(took)) < 1e-9
        settled = full or w['close'] <= sc.cut - 2 * W - 2
        if not settled:
            continue                                       # прогон оборван раньше срока: решено по известному
        assert round(took) == sum(paired[s] for s in mine), (w, mine)          # (а) и (в)
        # (б): испорченным окно бывает, только если рядом с ним событие, которому не досталось потери,
        # или в нём скачок не от парного штрафа, а рядом есть событие, — тогда штраф от расхода не отделить.
        may_be_bad = any(any(near(s, e) for e in sc.orphans) or
                         (odd[s] and not paired[s] and any(near(s, e) for e in sc.events)) for s in mine)
        assert may_be_bad or not w.get('bad'), (w, mine)
        # (г)
        waited = [s[1] for s in mine if odd[s] or any(near(s, e) for e in sc.events)]
        assert w['released'] <= max([w['close']] + [b + W for b in waited]), (w, mine)
        if not waited:
            assert w['released'] == w['close']
        # скачок без события и вдали от событий — настоящая странность: окно дошло до оценки целиком
        if any(odd[s] and not paired[s] for s in mine) and not may_be_bad:
            assert not w.get('bad') and round(took) == sum(paired[s] for s in mine)
    if full:
        assert led.stats['hit'] == sum(paired.values())                       # каждое событие отдано один раз
        assert led.stats['hit'] + led.stats['bad'] + led.stats['lost'] == len(sc.events)
        assert led.early == 0 or max(s[1] for s in segs if odd[s] or any(near(s, e) for e in sc.events)) > sc.n - W - 1
    return sum(paired.values()), sum(1 for w in wins if w.get('bad')), sum(1 for s in segs if odd[s] and not is_pen[s])


def test_random_sequences_keep_the_properties():
    seen = [0, 0, 0, 0]
    for seed in range(SEEDS):
        rng = random.Random(seed)
        sc = _scenario(rng)
        out = _check(sc, *_run(sc, rng))
        seen = [a + b for a, b in zip(seen, (*out, len(sc.orphans)))]
    # Проверка не пустая: в последовательностях много парных штрафов, испорченных окон, настоящих скачков.
    assert seen[0] > 400 and seen[1] > 20 and seen[2] > 300 and seen[3] > 50, seen


def test_agent_that_expects_no_loss_never_subtracts():
    """Правила агента без потери заряда при штрафе: ничего не вычитается, очередь пуста, окна не теряются."""
    for seed in range(100):
        rng = random.Random(seed)
        sc = _scenario(rng)
        led, wins, held, sizes, _ = _run(sc, rng, hit=0.0)
        assert not led._segs and not led._events and not held
        assert all(w['raw'] == w['b0'] and 'released' in w for w in wins)
        assert led.stats['hit'] == 0 and sizes[0] <= W + 2


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_same_sequence_gives_same_result(seed):
    a = _run(_scenario(random.Random(seed)), random.Random(seed + 1000))
    b = _run(_scenario(random.Random(seed)), random.Random(seed + 1000))
    assert a[0].stats == b[0].stats and [w['b0'] for w in a[1]] == [w['b0'] for w in b[1]]
