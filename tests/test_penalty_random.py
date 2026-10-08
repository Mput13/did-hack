"""Сопоставление потери заряда при штрафе с событием штрафа: свойства на случайных последовательностях.

Порождающий код знает правду — в каком показании батареи сидит какой штраф, пришло ли к нему событие и когда.
Последовательность идёт через настоящего агента: наблюдения — через Agent._perceive, последнее — через
Agent.tick(done=True). Окна расхода ведёт сам исследователь (did/inquiry.py), учёт штрафов — did/penalty.py;
проверка только подсматривает, какие отрезки и окна они завели.

 (а) штраф, у которого показание и событие разошлись не больше чем на WAIT_S, вычтен ровно один раз
     и ровно в размере из правил агента;
 (б) настоящий расход не пропадает: окно либо дошло до оценки с настоящим расходом, либо явно помечено
     испорченным, и испорченным оно бывает только по названной причине;
 (в) без события ничего не вычитается;
 (г) окно выдаётся не позже срока своих отрезков: WAIT_S после отрезка, у которого было что решать
     (необъяснённый скачок или событие рядом), а без таких отрезков — в такте закрытия;
 (д) очередь ограничена константой;
 (е) после конца прогона в очереди ничего не остаётся: ни отрезков, ни событий, ни придержанных окон;
     новых расследований и обращений к языковой модели после конца нет.

Показание батареи копится до 15 тактов. За секунду пути расход на самом дорогом грунте больше самой потери,
поэтому длинный отрезок может сходиться с разным числом штрафов. Что тогда считается правдой:

 - отрезок «голодный», если по числам он мог бы взять больше событий, чем у него парных штрафов (скачок без
   события на дорогом грунте; штраф, чьё событие не пришло). Рядом с чужим событием он может его забрать,
   и по показаниям это не отличить от правды. Окна такого отрезка и отрезка, чьё событие рядом с ним,
   проверяются только на безопасность: вычтено целое число потерь, не больше, чем событий рядом;
 - отрезок, который сходится с разным числом штрафов, а рядом чужое событие или на его событие претендует
   другой отрезок, обязан быть испорченным или посчитанным точно;
 - во всех остальных окнах (их подавляющее большинство) счёт точный.
"""
import math
import random
from collections import deque
from types import SimpleNamespace

import pytest

import did.llm_roles
from did.agent import Agent, make_config
from did.arena import load_arena
from did.config import BASE, Rules
from did.penalty import KEEP_S, WAIT_S
from did.robot_io import Observation

DT, HIT = 0.1, 3.0
W = round(WAIT_S / DT)             # WAIT_S в тактах
PILE = 15                          # на сколько тактов может отстать показание батареи
SEEDS = 600
ARENA, WORLD = load_arena(), Rules()
TH = next(a for a in (k * math.pi / 8 for k in range(16))
          if all(ARENA.clearance(BASE[0] + d * math.cos(a), BASE[1] + d * math.sin(a)) >= 0.2
                 for d in (0.3, 0.6, 0.9, 1.2, 1.5)))


class _IO:
    def command(self, v, w):
        pass


def _no_model(*args, **kwargs):
    raise AssertionError('после конца прогона агент обратился к языковой модели')


class _Spy(deque):
    """Очередь отрезков учёта, которая помнит всё, что в неё клали."""

    def __init__(self):
        super().__init__()
        self.all = []

    def append(self, k):
        self.all.append(k)
        super().append(k)


def _scenario(rng):
    """Случайная поездка. Возвращает такты: путь, настоящий расход, показана ли батарея, и правду о штрафах.
    Последний такт — последнее наблюдение прогона (done); события позже него до агента не доходят."""
    n = rng.randint(120, 320)
    if rng.random() < 0.35:                                # прогон обрывается в случайный момент
        n = rng.randint(3 * W, n)
    step, mult, show, off = [], [], [], []
    while len(step) < n:                                   # участки с разной скоростью и грунтом
        ln = rng.randint(5, 40)
        v = rng.choice([0.0, 0.008, 0.015, 0.015, 0.02])
        m = rng.choice([1.0, 1.0, 1.0, 2.0, 4.0])
        step += [v] * ln
        mult += [m] * ln
    while len(show) < n:                                   # показание батареи отстаёт: расход копится 1–15 тактов
        lag = rng.choice([1, 1, 1, 2, 3, 4, 6, 10, PILE]) if rng.random() < 0.5 else 1
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
    lo, hi = 2 * W, n - 1                                  # штрафы и скачки — вплоть до последнего такта
    pens, jumps = [], {}                                   # штраф: {'k', 'seg', 'paired', 'e'}; скачок: такт → размер

    def busy(seg):
        return any(p['seg'] == seg for p in pens) or any(seg_of.get(k) == seg for k in jumps)

    def near(seg, e, pad=0):
        return seg[0] - W - pad <= e <= seg[1] + W + pad

    def spot():
        return hi - rng.randint(0, W) if rng.random() < 0.25 else rng.randint(lo, hi)     # четверть — у самого конца

    for _ in range(rng.choice([0, 1, 1, 2, 3, 5])):        # штрафы с событием в пределах WAIT_S от показания
        k = spot()
        ks = [k]
        if rng.random() < 0.35:                            # два подряд: в том же такте или в соседнем
            ks.append(k + rng.choice([0, 0, 1, 2]))
        for k in ks:
            seg = seg_of.get(k)
            if seg is None or any(seg_of.get(j) == seg for j in jumps) or any(p['seg'] == seg and not p['paired'] for p in pens):
                continue
            # Событие приходит около показания; у длинного отрезка — ещё и около самого штрафа, задолго до показания.
            e = k + rng.randint(-1, 1) if seg[1] - seg[0] > 2 and rng.random() < 0.5 else seg[1] + rng.randint(-W, W)
            pens.append({'k': k, 'seg': seg, 'paired': True, 'e': e})
    long = [s for s in sorted(set(seg_of.values())) if s[1] - s[0] >= 10 and s[0] >= lo]
    for _ in range(rng.choice([0, 0, 1, 2]) if long else 0):       # один или два штрафа в длинном накопленном показании
        seg = rng.choice(long)
        if busy(seg):
            continue
        for _ in range(rng.choice([1, 1, 2])):
            k = rng.randint(seg[0] + 1, seg[1])
            e = k + rng.randint(-1, 1) if rng.random() < 0.5 else seg[1] + rng.randint(-W, W)
            pens.append({'k': k, 'seg': seg, 'paired': True, 'e': e})
    for _ in range(rng.choice([0, 0, 1, 2, 4])):           # настоящие скачки без события
        k = spot()
        if k in seg_of and not busy(seg_of[k]):             # в одном показании — один скачок: два дали бы «штраф»
            jumps[k] = rng.uniform(0.3, 1.8)
    orphans = []
    for _ in range(rng.choice([0, 0, 0, 1, 2])):           # штраф без события или с событием слишком далеко
        k = spot()
        seg = seg_of.get(k)
        far = rng.random() < 0.5
        e = None if not far or seg is None else rng.choice([seg[0] - W - rng.randint(2, 6), seg[1] + W + rng.randint(2, 6)])
        if seg is None or busy(seg) or any(near(seg, x, 1) for x in [p['e'] for p in pens if p['paired']] + orphans):
            continue                                       # такой штраф не отличить от соседнего: правды нет
        if e is not None and (not 1 <= e < n or any(near(p['seg'], e, 1) for p in pens)):
            continue
        pens.append({'k': k, 'seg': seg, 'paired': False, 'e': None})
        if e is not None:
            orphans.append(e)
    for p in pens:                                         # событие позже последнего наблюдения не приходит вовсе
        p['paired'] = p['paired'] and 0 <= p['e'] < n
    events = sorted([p['e'] for p in pens if p['paired']] + orphans)
    return SimpleNamespace(n=n, step=step, mult=mult, show=show, off=off, seg_of=seg_of, pens=pens, jumps=jumps,
                           events=events, orphans=orphans, noise=rng.choice([0.0, 0.0, 0.005]))


def _run(sc, rng, hit=HIT):
    """Прогнать последовательность через настоящего агента; последнее наблюдение — через Agent.tick(done=True)."""
    bot = Agent(ARENA, make_config('scientist'), n_samples=3, rules=Rules(hazard_battery_hit=hit))
    inv, led = bot.inv, bot.inv.penalty
    led._segs = _Spy()
    wins, feed = [], led.reading

    def reading(obs, win, dth=0.0):                        # подсмотреть, какие окна исследователь отдаёт учёту
        if win is not None and all(win is not r['w'] for r in wins[-3:]):
            wins.append({'w': win, 'raw': win['b0']})
        return feed(obs, win, dth)

    led.reading = reading
    battery, shown, pos, way = 60.0, 60.0, 0.0, 1.0
    sizes, seen, opened = [0, 0, 0], {}, 0
    for k in range(sc.n):
        if k:
            if not 0.0 <= pos + way * sc.step[k] <= 1.2:   # робот ездит туда-обратно по свободной прямой
                way = -way
            pos += way * sc.step[k]
            battery -= WORLD.drain_per_m * sc.mult[k] * sc.step[k] + WORLD.drain_idle_per_s * DT
            battery -= HIT * sum(1 for p in sc.pens if p['k'] == k) + sc.jumps.get(k, 0.0)
        if sc.show[k]:
            shown = battery + rng.gauss(0.0, sc.noise)
        seen[k] = shown
        x, y = BASE[0] + pos * math.cos(TH), BASE[1] + pos * math.sin(TH)
        obs = Observation(t=k * DT, x=x, y=y, th=TH, v=sc.step[k] / DT, w=0.0, battery=shown, sensor=None, scan=None,
                          events=[{'type': 'hazard_hit', 't': k * DT, 'x': x, 'y': y}] * sc.events.count(k),
                          done=k == sc.n - 1)
        if not obs.done:
            inv.run = {'test': None} if sc.off[k] else None        # идёт опыт: окно расхода не ведётся
            bot._perceive(obs)
        else:
            # Конец прогона. Расследование, начатое по дороге, считаем законченным, языковая модель подключена:
            # если досчёт откроет новое расследование или спросит модель, это будет видно.
            inv.run = inv.active = None
            inv._quiet_until = -1e9
            inv.roles = object()
            before = len(inv.inquiries)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(did.llm_roles, 'deliberate', _no_model)
                mp.setattr(did.llm_roles, 'explain', _no_model)
                bot.tick(obs, _IO())
                bot.tick(obs, _IO())                       # стенд после конца зовёт tick ещё раз
            opened = len(inv.inquiries) - before
        for r in wins:
            if 't1' in r['w'] and 'released' not in r and all(r['w'] is not h for h in inv._held):
                r['released'] = k
        sizes = [max(sizes[0], len(led._segs)), max(sizes[1], len(led._events)), max(sizes[2], len(inv._held))]
    late = sum(1 for e in bot.journal.entries if (e.get('data') or {}).get('tag') == 'late_anomaly')
    return SimpleNamespace(bot=bot, led=led, segs=led._segs.all, wins=[r for r in wins if 't1' in r['w']],
                           sizes=sizes, seen=seen, opened=opened, late=late)


def _tick(t):
    return round(t / DT)


def _check(sc, run):
    led, inv = run.led, run.bot.inv
    assert not led._segs and not led._events and not inv._held                 # (е)
    assert all('released' in r for r in run.wins)
    assert run.opened == 0                                 # после конца новых расследований нет (и модель молчит)
    assert run.sizes[0] <= 2 * W + 2 and run.sizes[1] <= 12 and run.sizes[2] <= 4, run.sizes       # (д)

    # Отрезки, которые завёл учёт, — те же, что знает порождающий код.
    segs = {(_tick(k['t0']), _tick(k['t1'])): k for k in run.segs}
    assert set(segs) == {s for s in set(sc.seg_of.values()) if seen_changed(run.seen, s)}
    near = lambda s, e: s[0] - W <= e <= s[1] + W          # noqa: E731
    own = {s: [p['e'] for p in sc.pens if p['seg'] == s and p['paired']] for s in segs}
    around = {s: [e for e in sc.events if near(s, e)] for s in segs}
    foreign = {s: len(around[s]) > len(own[s]) for s in segs}
    claims = {s: bool(k['fit']) or k['odd'] for s, k in segs.items()}
    # «Голодный» отрезок по числам мог бы взять больше событий, чем у него парных штрафов.
    hungry = {s: any(m > len(own[s]) for m in k['fit']) for s, k in segs.items()}
    shared = {s: any(near(q, e) for e in own[s] for q in segs if q != s and claims[q]) for s in segs}
    robbed = {s: any(near(q, e) for e in own[s] for q in segs if q != s and hungry[q]) for s in segs}
    murky = {s: (hungry[s] and foreign[s]) or robbed[s] for s in segs}                 # правду не узнать
    unclear = {s: k['open'] and (foreign[s] or shared[s]) for s, k in segs.items()}    # испорчен или точен
    # Из нескольких штрафов одного показания событие пришло не ко всем (остальные — после конца прогона):
    # падение с числом событий не сходится, отрезок обязан быть испорченным и ничего не вычесть.
    partial = {s: bool(own[s]) and len(own[s]) not in k['fit'] for s, k in segs.items()}
    owner = {p['e']: p['seg'] for p in sc.pens if p['paired']}

    # События рядом с такими отрезками могут остаться без потери — и, по цепочке, события отрезков, которые
    # делят их с ними: соседний отрезок берёт чужое событие вместо своего, а своё остаётся свободным.
    tangled = {e for s in segs if murky[s] or unclear[s] or partial[s] for e in around[s]}
    while True:
        more = {e for s in segs if claims[s] and any(near(s, x) for x in tangled) for e in own[s]} - tangled
        if not more:
            break
        tangled |= more

    def stalled(s, e):
        """Событие пришло не позже показания отрезка, а его потеря — позже срока отрезка: показание копилось."""
        return e in owner and e <= s[1] and owner[e][1] > s[1] + W

    def deadline(s):
        k = segs[s]
        if k['open']:
            return max([s[1]] + around[s]) + W
        return s[1] + W if k['odd'] or around[s] else s[1]

    counts = SimpleNamespace(paired=0, bad=0, jumps=0, open_exact=0, two=0, safe_only=0, unclear=0)
    for r in run.wins:
        w = r['w']
        mine = [s for s, k in segs.items() if k['win'] is w]
        took = (r['raw'] - w['b0']) / HIT
        assert abs(took - round(took)) < 1e-9 and 0 <= round(took) <= sum(len(around[s]) for s in mine), (r, mine)
        took, bad = round(took), bool(w.get('bad'))
        want = sum(len(own[s]) for s in mine if not partial[s])
        counts.bad += bad
        # (г): окно держат только его отрезки; впереди них в очереди может стоять отрезок, который решается
        # числом событий и ждёт дольше.
        close = _tick(w['t1'])
        ahead = [deadline(s) for s, k in segs.items() if k['open'] and s[1] <= close]
        limit = min(max([close] + [deadline(s) for s in mine] + ahead), sc.n - 1)
        assert r['released'] <= limit, (r, mine)
        if any(murky[s] for s in mine):
            counts.safe_only += 1
            continue
        if any(unclear[s] for s in mine):
            counts.unclear += 1
            assert bad or took == want, (r, mine)
            continue
        assert took == want, (r, mine)                                         # (а) и (в)
        # (б): испорченным окно бывает, только если рядом с ним событие, которому не досталось потери (или она
        # пришла позже срока), или в нём скачок не от парного штрафа, а рядом есть событие, — тогда штраф
        # от расхода не отделить.
        loose = sc.orphans + sorted(tangled)
        may_be_bad = any(any(near(s, e) for e in loose) or any(stalled(s, e) for e in around[s])
                         or (segs[s]['odd'] and not own[s] and around[s]) for s in mine)
        assert may_be_bad or not bad, (r, mine)
        assert bad or not any(partial[s] for s in mine), (r, mine)
        counts.open_exact += sum(1 for s in mine if segs[s]['open'] and own[s] and not bad)
        counts.two += sum(1 for s in mine if segs[s]['open'] and len(own[s]) == 2 and 1 in segs[s]['fit'] and not bad)
    if not any(murky.values()) and not any(unclear.values()):
        assert led.stats['hit'] == sum(len(e) for s, e in own.items() if not partial[s])   # событие отдано один раз
        assert led.stats['hit'] + led.stats['bad'] + led.stats['lost'] == len(sc.events)
        assert led.early == 0 or max(deadline(s) for s in segs) > sc.n - 1
    is_pen = {p['seg'] for p in sc.pens}
    counts.paired = sum(len(e) for e in own.values())
    counts.jumps = sum(1 for s, k in segs.items() if k['odd'] and s not in is_pen)
    return counts


def seen_changed(seen, s):
    return s[1] in seen and seen[s[0]] != seen[s[1]]


def test_random_sequences_keep_the_properties():
    total, ends = {}, [0, 0, 0]
    for seed in range(SEEDS):
        rng = random.Random(seed)
        sc = _scenario(rng)
        run = _run(sc, rng)
        for name, v in vars(_check(sc, run)).items():
            total[name] = total.get(name, 0) + v
        total['wins'] = total.get('wins', 0) + len(run.wins)
        total['orphans'] = total.get('orphans', 0) + len(sc.orphans)
        ends = [ends[0] + any(p['k'] >= sc.n - 1 - W for p in sc.pens), ends[1] + (sc.n - 1 in sc.events),
                ends[2] + run.late]
    # Проверка не пустая: много парных штрафов, испорченных окон, настоящих скачков, событий без пары; длинные
    # показания решаются числом событий (в том числе «один штраф или два»); окон, где правды не узнать, мало.
    assert total['paired'] > 400 and total['bad'] > 20 and total['jumps'] > 300 and total['orphans'] > 50, total
    assert total['open_exact'] > 25 and total['two'] > 5, total
    assert total['safe_only'] + total['unclear'] < 0.02 * total['wins'], total
    # Конец прогона тоже проверен: штрафы в последних тактах, события в последнем наблюдении, странности,
    # замеченные досчётом.
    assert ends[0] > 100 and ends[1] > 10 and ends[2] > 10, ends


def test_agent_that_expects_no_loss_never_subtracts():
    """Правила агента без потери заряда при штрафе: ничего не вычитается, очередь пуста, окна не теряются."""
    for seed in range(100):
        rng = random.Random(seed)
        sc = _scenario(rng)
        run = _run(sc, rng, hit=0.0)
        assert not run.led._segs and not run.led._events and not run.bot.inv._held and run.opened == 0
        assert all(r['raw'] == r['w']['b0'] and 'released' in r for r in run.wins)
        assert run.led.stats['hit'] == 0 and run.sizes[0] <= W + 2


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_same_sequence_gives_same_result(seed):
    a = _run(_scenario(random.Random(seed)), random.Random(seed + 1000))
    b = _run(_scenario(random.Random(seed)), random.Random(seed + 1000))
    assert a.led.stats == b.led.stats and [r['w']['b0'] for r in a.wins] == [r['w']['b0'] for r in b.wins]


def test_events_do_not_pile_up_when_battery_reading_stalls():
    """Показание батареи замерло на всю последовательность, а события штрафа идут: память событий ограничена
    сроком KEEP_S, после конца прогона пусто."""
    for seed in range(20):
        rng = random.Random(seed)
        sc = _scenario(rng)
        sc.show = [k == 0 for k in range(sc.n)]
        sc.seg_of, sc.pens, sc.jumps, sc.orphans = {}, [], {}, []
        sc.events = sorted(rng.randint(1, sc.n - 1) for _ in range(sc.n // 3))
        run = _run(sc, rng)
        assert run.sizes[1] <= max(sc.events.count(k) for k in set(sc.events)) * (KEEP_S / DT + 1) / 3 + 12
        assert not run.led._events and not run.led._segs and run.opened == 0
        assert run.led.stats['lost'] == len(sc.events) and run.led.stats['hit'] == 0
