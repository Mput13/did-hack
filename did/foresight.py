"""Сравнение будущих маршрутов: прежде чем ехать, прикинуть варианты плана в разных возможных мирах.

Правило «дорога туда + дорога домой × 1,1 + 4 ед.» держит один и тот же запас и на проверенном полу,
и там, где робот ещё не ездил. Здесь запас получается из того, чего агент не знает. Берётся M
возможных состояний среды («миров»), в каждом по-своему задано то, что агенту неизвестно:

  грунт     — где на непроверенном полу лежат зоны дорогого грунта и во сколько раз они дороже;
  опасность — где ещё есть ненайденные опасные зоны и чем кончится въезд (штраф, утечка заряда);
  образец   — есть ли он в кандидате на самом деле;
  расход    — сколько на самом деле стоит метр;
  сбой      — сколько ещё продлится утечка, идущая сейчас;
  перемены  — не устарела ли карта там, где робот уже ездил.

Каждый вариант плана («C1 → домой», «E2 → домой», «C1 → C2 → домой», «сразу домой») проигрывается
во всех мирах: сколько образцов, какой счёт, сколько заряда останется на базе, как часто робот не
вернётся. Годятся варианты, где вероятность не вернуться не выше порога; если таких нет — домой.
Среди годных выбирает либо правило планировщика, либо наибольший ожидаемый счёт (Settings.choice).
Пути считаются один раз (CostGraph), миры — массивами numpy; решение занимает единицы миллисекунд.

Две части: Foresight — сам расчёт на отрезках пути (его проверяют tests/test_foresight.py на
игрушечных входах), Advisor — связка с агентом: строит варианты из его карт и пишет таблицу в журнал.
"""
import math
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

from .planner import HeuristicPlanner

CELL = 0.1            # м: сетка, на которой разложены миры
RETRACE = 1.05        # обратная дорога по своему следу: во сколько раз она длиннее, чем путь туда
DETECT = 5            # через сколько шагов пути (≈0,3 м) агент замечает дорогой грунт под колёсами
HOT_MULT = 4.0        # до какого множителя может подорожать уже известный дорогой грунт, если среда изменилась


@dataclass(frozen=True)
class Settings:
    worlds: int = 48                  # сколько состояний среды в выборке
    risk_limit: float = 0.05          # допустимая вероятность не вернуться
    choice: str = 'planner'           # кто выбирает среди прошедших по риску: planner — правило планировщика
    #                                   («выгода на единицу заряда»), score — наибольший ожидаемый счёт
    abort_margin: float = 1.3         # в пути от цели отказываемся, когда риск выше порога во столько раз
    # --- чего агент не знает о полу
    soil_rate: float = 0.20           # зон дорогого грунта на метр непроверенного пола: исходное предположение
    soil_weight_m: float = 8.0        # его вес в метрах пути: после стольких метров своим наблюдениям верим так же
    soil_radius: tuple = (0.35, 0.60)
    soil_mults: tuple = (2.0, 3.0, 4.0)   # множители по умолчанию, пока своих зон не найдено
    hazard_rate: float = 0.03         # ненайденных опасных зон на метр непроверенного пола
    hazard_weight_m: float = 12.0
    hazard_radius: tuple = (0.25, 0.33)
    stale: float = 0.10               # вероятность, что карта устарела и проверенный пол уже не тот
    stale_exposure: float = 0.6       # насколько в таком мире проверенный пол похож на непроверенный
    # --- расход
    per_m_factor: float = 0.97        # поправка к оценке агента «заряд на метр»: робот срезает углы пути по клеткам
    turn_rate: float = 1.3            # рад на метр на переездах (исследователь знает цену поворота)
    per_m_sd: float = 0.09            # её разброс (доля): повороты, объезды, неточность модели
    speed: float = 0.16               # м/с: средняя скорость, по ней считается расход за время в пути
    p_leak: float = 0.25              # доля въездов в опасную зону, после которых начинается утечка
    leak_s: tuple = (20.0, 30.0)      # сколько она длится
    dwell: float = 0.3                # ед. заряда на подъезд вплотную и сбор
    turn_back_below: float = 2.0      # встретив неожиданность, агент повернёт назад, если иначе на базе осталось бы меньше
    # --- ценность
    explore_value: float = 0.6        # какая доля «неясной массы» вокруг точки разведки превращается в образцы
    pairs: int = 3                    # пары «Ci → Cj → домой» строятся для стольких лучших кандидатов
    seed: int = 0


DEFAULTS = Settings()                 # отладочные серии подменяют это значение целиком


class Leg:
    """Отрезок пути глазами агента: шаги, оценка множителя грунта, доля непроверенного пола."""

    __slots__ = ('ds', 'dm', 'unk', 'open', 'cell', 'risk', 'metres', 'length', 'unknown_m', 'hot')

    def __init__(self, ds, mult, unk, cell, risk=0.0):
        self.ds = np.asarray(ds, dtype=float)             # длина шага, м
        self.unk = np.asarray(unk, dtype=float)           # 0 — пол проверен колёсами, 1 — там не ездили
        self.open = self.unk > 0.5
        self.cell = np.asarray(cell, dtype=np.intp)       # клетка сетки миров под каждым шагом
        self.risk = float(risk)                           # вероятность задеть уже найденную опасную зону
        mult = np.asarray(mult, dtype=float)
        self.length = float(self.ds.sum())
        self.dm = self.ds * mult                          # те же шаги в метрах обычного пола по оценке агента
        self.metres = float(self.dm.sum())
        self.unknown_m = float(self.ds @ self.unk)
        # Сколько метров прибавится, если уже известный дорогой грунт на пути подорожает до верхнего множителя.
        self.hot = self.ds * np.where(mult > 1.4, np.maximum(HOT_MULT - mult, 0.0), 0.0)


@dataclass
class Step:
    id: str               # C1, E2
    kind: str             # candidate | explore
    p: float              # вероятность образца (кандидат) или ожидаемое число находок (разведка)
    leg: Leg              # путь от предыдущей точки
    k: int = 0            # номер цели: по нему берутся одни и те же случайные числа «образец есть / нет»


@dataclass
class Option:
    id: str
    label: str
    steps: list           # [Step]; пусто — «сразу домой»
    homes: list           # [(Leg, цена по оценке агента)]: домой от последней цели, от предпоследней, …, от старта
    subgoals: list = None


class Worlds:
    """M возможных состояний среды на сетке CELL: зоны грунта и опасные зоны на всей арене.

    Случайные числа фиксированы при создании: во всех решениях прогона миры одни и те же, меняется
    только то, что агент успел проверить (проверенный пол миры не трогают). Поэтому варианты
    сравниваются на одинаковых мирах, а оценка риска не дрожит от решения к решению.
    """

    MAX_SOIL = 8
    MAX_HAZARD = 3
    TARGETS = 12

    def __init__(self, bounds, s, centres=None):
        self.s = s
        self.x0, self.y0, x1, y1 = bounds
        self.w, self.h = int(math.ceil((x1 - self.x0) / CELL)), int(math.ceil((y1 - self.y0) / CELL))
        gx = self.x0 + (np.arange(self.w) + 0.5) * CELL
        gy = self.y0 + (np.arange(self.h) + 0.5) * CELL
        X, Y = (g.ravel() for g in np.meshgrid(gx, gy))
        if centres is None:
            centres = np.column_stack([X, Y])
        m = self.n = s.worlds
        rng = np.random.default_rng(s.seed)

        def strata():                     # равномерные числа по одному на слой: ровнее, чем просто случайные
            return (rng.permutation(m) + rng.random(m)) / m

        def zones(k, radius):
            c = centres[rng.integers(len(centres), size=(m, k))]
            r = rng.uniform(radius[0], radius[1], size=(m, k))
            d2 = (X[None, None, :] - c[:, :, :1]) ** 2 + (Y[None, None, :] - c[:, :, 1:]) ** 2
            return d2 <= (r ** 2)[:, :, None]

        self._soil = zones(self.MAX_SOIL, s.soil_radius)          # (M, K, клеток): накрыта ли клетка зоной
        self._hazard = zones(self.MAX_HAZARD, s.hazard_radius)
        self._soil_mult_u = rng.random((m, self.MAX_SOIL))
        self._soil_n_u, self._hazard_n_u = strata(), strata()
        self.z = np.sqrt(2.0) * _erfinv(2.0 * strata() - 1.0)     # отклонение расхода от оценки, в сигмах
        self.u_sample = rng.random((m, self.TARGETS))             # есть ли образец в цели k
        self.u_known = strata()                                   # заденет ли путь уже найденную зону
        self.u_leak, self.u_leak_s = strata(), rng.random(m)      # начнётся ли утечка после штрафа и на сколько
        self.u_now, self.u_pending = rng.random(m), strata()      # идущая утечка: сколько ещё, была ли вообще
        self.stale = strata() < s.stale                           # в этих мирах карта устарела
        self.extra = np.zeros((m, self.w * self.h))               # (множитель − 1) дорогого грунта по клеткам
        self.danger = np.zeros((m, self.w * self.h), dtype=bool)  # ненайденная опасная зона в клетке
        self.rates = None

    def set_rates(self, n_soil, n_hazard, mults):
        """Сколько зон грунта и опасных зон в среднем на арене и какими бывают множители."""
        key = (round(n_soil * 4) / 4, round(n_hazard * 10) / 10, tuple(mults))   # мелкие сдвиги миры не меняют
        if key == self.rates:
            return
        self.rates = key
        pool = np.asarray(mults, dtype=float)
        mult = pool[np.minimum((self._soil_mult_u * len(pool)).astype(int), len(pool) - 1)]
        active = np.arange(self.MAX_SOIL)[None, :] < _poisson(self._soil_n_u, key[0])[:, None]
        self.extra[:] = 0.0
        for k in range(self.MAX_SOIL):
            np.maximum(self.extra, self._soil[:, k] * (active[:, k] * (mult[:, k] - 1.0))[:, None], out=self.extra)
        active = np.arange(self.MAX_HAZARD)[None, :] < _poisson(self._hazard_n_u, key[1])[:, None]
        self.danger = (self._hazard & active[:, :, None]).any(axis=1)

    def cells(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / CELL).astype(np.intp), 0, self.w - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / CELL).astype(np.intp), 0, self.h - 1)
        return iy * self.w + ix

    def metres(self, leg):
        """Длина отрезка в метрах обычного пола в каждом мире: (M,)."""
        if not len(leg.ds):
            return np.zeros(self.n)
        g = self.extra[:, leg.cell]
        out = leg.metres + g @ (leg.ds * leg.unk)
        if self.stale.any():              # карта устарела: зоны лежат и на проверенном полу, известные подорожали
            out[self.stale] += g[self.stale] @ (leg.ds * np.maximum(self.s.stale_exposure - leg.unk, 0.0)) \
                + float(leg.hot.sum())
        return out

    def walk(self, leg):
        """Отрезок по шагам: пройденные метры обычного пола (M, P), где встретится неожиданность (M, P)
        и где из неё — опасная зона (M, P)."""
        unk = np.where(self.stale[:, None], np.maximum(leg.unk, self.s.stale_exposure), leg.unk)
        g = self.extra[:, leg.cell] * unk
        d = self.danger[:, leg.cell] & (leg.open[None, :] | self.stale[:, None])
        return np.cumsum(leg.dm + g * leg.ds + self.stale[:, None] * leg.hot, axis=1), (g > 0.3) | d, d

    def hits(self, leg):
        """Сколько раз на отрезке робот въедет в опасную зону в каждом мире: (M,)."""
        known = (self.u_known < leg.risk).astype(float)
        if not len(leg.ds):
            return known
        d = self.danger[:, leg.cell]
        new = (d & leg.open[None, :]).any(axis=1) | (self.stale & d.any(axis=1))
        return known + new


def _erfinv(x):
    from scipy.special import erfinv
    return erfinv(np.clip(x, -0.999999, 0.999999))


def _poisson(u, mean):
    """Число событий по заранее выбранному равномерному числу u: при том же u оно растёт вместе со средним."""
    if mean <= 0.0:
        return np.zeros(len(u), dtype=int)
    k = np.arange(40)
    logp = -mean + k * math.log(mean) - np.cumsum(np.log(np.maximum(k, 1)))
    return np.searchsorted(np.cumsum(np.exp(logp)), u)


def line_leg(worlds, a, b, unknown=1.0, mult=1.0, risk=0.0, step=0.05):
    """Прямой отрезок из a в b: для проверок и примеров. unknown — какая его доля (с конца b) не проверена."""
    n = max(1, int(round(math.dist(a, b) / step)))
    t = (np.arange(n) + 0.5) / n
    x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    return Leg(np.full(n, math.dist(a, b) / n), np.full(n, mult), (t > 1.0 - unknown).astype(float),
               worlds.cells(x, y), risk)


COLUMNS = [
    # ключ, подпись, единица — для таблицы в интерфейсе
    ('samples', 'Образцов', 'шт.'),
    ('score', 'Счёт', 'очки'),
    ('risk', 'Не вернуться', 'доля'),
    ('battery_p5', 'Заряд на базе, худший случай', 'ед.'),
    ('battery_p50', 'Заряд на базе, обычно', 'ед.'),
    ('battery_p95', 'Заряд на базе, лучший случай', 'ед.'),
    ('cost', 'Расход', 'ед.'),
    ('hits', 'Штрафов в опасных зонах', 'шт.'),
    ('unknown_m', 'Непроверенного пути', 'м'),
]


class Foresight:
    """Расчёт: варианты плана × возможные миры → таблица показателей и выбор."""

    def __init__(self, bounds=(-3.0, -3.0, 3.0, 3.0), settings=None, centres=None, rules=None, area=None):
        from .config import Rules
        s = self.s = settings or DEFAULTS
        self.rules = rules or Rules()
        self.worlds = Worlds(bounds, s, centres)
        # Пока своих наблюдений нет — исходные частоты зон на метр, пересчитанные в число зон на площади.
        area = area or (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        self.worlds.set_rates(s.soil_rate * area / sum(s.soil_radius), s.hazard_rate * area / sum(s.hazard_radius),
                              s.soil_mults)

    def compare(self, options, battery, per_m, load=0.0, leak=0.0, pending=(0.0, 0.0)):
        """Таблица: по строке на вариант.

        battery — заряд сейчас; per_m — заряд на метр обычного пола по оценке агента; load — на какую
        долю каждый новый образец удорожает метр; leak — сколько заряда ещё заберёт идущая утечка;
        pending — (вероятность, ед.): утечка, которая могла начаться после недавнего штрафа, но ещё
        не подтверждена и не исключена.
        """
        w, s, r = self.worlds, self.s, self.rules
        start = battery - leak * (0.5 + w.u_now) - (w.u_pending < pending[0]) * pending[1]
        # Въезд в опасную зону: разовая потеря, а при правилах со сбоями — иногда утечка на 20–30 с.
        hit_loss = np.full(w.n, r.hazard_battery_hit)
        if r.faults:
            hit_loss = hit_loss + (w.u_leak < s.p_leak) * r.leak_per_s * (
                s.leak_s[0] + w.u_leak_s * (s.leak_s[1] - s.leak_s[0]))
        ctx = (start, per_m * s.per_m_factor, load, hit_loss)
        return [self._play(opt, ctx) for opt in options]

    def _play(self, opt, ctx):
        """Один вариант во всех мирах сразу."""
        w, s, r = self.worlds, self.s, self.rules
        start, per_m, load, hit_loss = ctx
        idle = r.drain_idle_per_s / s.speed                 # заряд за время в пути, на метр
        b = start.copy()
        got = np.zeros(w.n)                                 # собрано образцов в этом варианте
        found = np.zeros(w.n)                               # ожидаемые находки разведки
        spent = np.zeros(w.n)
        hits = np.zeros(w.n)
        out = []                                            # метры каждого отрезка «туда» по мирам
        unknown_m = 0.0
        for st in opt.steps:
            m = w.metres(st.leg)
            out.append(m)
            e = m * per_m * (1.0 + load * got) + idle * st.leg.length
            h = w.hits(st.leg)
            b = b - e - h * hit_loss
            spent += e
            hits += h
            unknown_m += st.leg.unknown_m
            if st.kind == 'candidate':
                got += (w.u_sample[:, st.k % w.TARGETS] < st.p) & (b > 0.0)   # не доехал — не собрал
                b = b - s.dwell
            else:
                found += s.explore_value * st.p

        # Домой: по плану от последней цели либо назад по своему следу до точки, откуда дорога уже посчитана.
        # Агент выберет по своей оценке (след к тому времени проверен), а заплатит — по правде мира.
        alts = opt.homes
        est, true_m, true_h = [], [], []
        back = 0.0
        for i, (leg, cost) in enumerate(alts):
            if i:
                back = back + out[-i]
            est.append(back * RETRACE + cost)
            true_m.append(back * RETRACE + w.metres(leg))
            true_h.append(w.hits(leg))
        if len(alts) > 1:
            pick = np.argmin(np.vstack([np.broadcast_to(e, w.n) for e in est]), axis=0)
            m = np.choose(pick, true_m)
            h = np.choose(pick, true_h)
            share = np.bincount(pick, minlength=len(alts)) / w.n
            unknown_m += float(sum(share[i] * alts[i][0].unknown_m for i in range(len(alts))))
        else:
            m, h = true_m[0], true_h[0]
            unknown_m += alts[0][0].unknown_m
        e = m * per_m * (1.0 + load * got) + idle * m
        b = b - e - h * hit_loss
        spent += e
        hits += h

        if opt.steps and len(opt.steps[0].leg.ds):
            # Агент может передумать: встретив по дороге к первой цели неожиданность, с которой заряда
            # на весь план не хватит, он разворачивается и едет домой от того места.
            leg = opt.steps[0].leg
            cum, sur, d = w.walk(leg)
            rows = np.arange(w.n)
            seen = np.minimum(sur.argmax(axis=1) + DETECT, len(leg.ds) - 1)     # заметил через несколько шагов
            home = alts[-1][0]                                                   # дорога домой от старта
            m2 = cum[rows, seen] * (1.0 + RETRACE) + w.metres(home)
            h2 = np.maximum.accumulate(d, axis=1)[rows, seen] + w.hits(home)
            e2 = m2 * per_m + idle * m2
            b2 = start - e2 - h2 * hit_loss
            turn = sur.any(axis=1) & (b < s.turn_back_below) & (b2 > b)
            b, spent, hits, got, found = (np.where(turn, x, y) for x, y in (
                (b2, b), (e2, spent), (h2, hits), (0.0, got), (0.0, found)))

        sd = s.per_m_sd * spent + 0.15                      # неточность самой оценки расхода
        p_back = ndtr(b / sd)                               # вероятность вернуться в этом мире
        left_b = np.maximum(b, 0.0)
        score = (r.pts_sample * (got + found) + r.pts_hazard_hit * hits
                 + p_back * (r.pts_return + r.pts_battery_left * left_b))
        end = np.maximum(b + sd * w.z, 0.0)                 # заряд на базе с учётом разброса расхода
        p5, p50, p95 = np.percentile(end, [5, 50, 95])
        risk = float(1.0 - p_back.mean())
        return {'id': opt.id, 'label': opt.label,
                'samples': round(float(np.mean(got + found)), 2), 'score': round(float(score.mean()), 2),
                'risk': round(risk, 4), 'battery_p5': round(float(p5), 1), 'battery_p50': round(float(p50), 1),
                'battery_p95': round(float(p95), 1), 'cost': round(float(spent.mean()), 1),
                'hits': round(float(hits.mean()), 2), 'unknown_m': round(unknown_m, 1),
                'ok': bool(risk <= s.risk_limit)}

    def choose(self, rows):
        """Наибольший ожидаемый счёт при риске не выше порога; если никто не проходит — домой."""
        ok = [r for r in rows if r['ok']]
        if ok:
            return max(ok, key=lambda r: r['score'])
        return next((r for r in rows if r['id'] == 'home'), None) or min(rows, key=lambda r: r['risk'])


class Advisor:
    """Связка расчёта с агентом: варианты из его карт и путей, выбор, запись таблицы в журнал."""

    def __init__(self, agent, settings=None):
        a = self.a = agent
        self.s = settings or replace(DEFAULTS, risk_limit=getattr(a.cfg, 'risk_limit', DEFAULTS.risk_limit),
                                     choice=getattr(a.cfg, 'foresight_choice', DEFAULTS.choice))
        arena = a.arena
        X, Y = arena.cell_centers()
        free = arena.clear >= 0.10
        bounds = (arena.x0, arena.y0, arena.x0 + arena.w * arena.res, arena.y0 + arena.h * arena.res)
        self.area = float(arena.free.sum()) * arena.res ** 2
        self.core = Foresight(bounds, self.s, np.column_stack([X[free], Y[free]]), a.rules, self.area)
        self.log = []                             # все решения прогона: таблицы «варианты × показатели»
        self.kappa = 1.0
        self.rates = (self.s.soil_rate, self.s.hazard_rate)   # зон грунта и опасных зон на метр непроверенного пола
        self._unk = np.ones(arena.free.shape)
        self._soil_version = None
        self._p = {}                              # цель → вероятность образца на момент выбора

    # --- что агент знает и чего не знает --------------------------------------------------------

    def _sync(self):
        """Обновить то, от чего зависят миры: какой пол проверен и как часто встречаются зоны."""
        a, s = self.a, self.s
        soil = a.soil
        key = (soil.version, len(a.hazard_map.zones))
        if key == self._soil_version:
            return
        self._soil_version = key
        seen = soil.dist > 0.03                                  # клетки грунта, по которым робот проехал сам
        if a.cfg.learn_soil:
            # Проверено — там, где проехал; рядом со следом — отчасти (по сглаженной оценке модели грунта).
            self._unk = np.where(soil._fine(seen.astype(float), 0.0) > 0.5, 0.0, 1.0 - soil.confidence_grid())
        new_m = float(seen.sum()) * soil.res                     # сколько нового пола уже проверено колёсами
        zones = soil.zones() if a.cfg.learn_soil else []
        found = []
        for z in zones:                                          # одна зона, задетая в двух местах, — одна находка
            if all(math.hypot(z['x'] - q['x'], z['y'] - q['y']) > 0.9 for q in found):
                found.append(z)
        # Частота на метр непроверенного пола: исходное предположение плюс то, что встретилось на своём пути.
        lam_s = (s.soil_rate * s.soil_weight_m + len(found)) / (s.soil_weight_m + new_m)
        lam_h = (s.hazard_rate * s.hazard_weight_m + len(a.hazard_map.zones)) / (s.hazard_weight_m + new_m)
        self.rates = (lam_s, lam_h)
        mults = tuple(sorted(round(min(max(z['mult'], 1.5), 5.0) * 2) / 2 for z in zones)) + tuple(s.soil_mults)
        self.core.worlds.set_rates(lam_s * self.area / sum(s.soil_radius), lam_h * self.area / sum(s.hazard_radius),
                                   mults)
        # Обычный пол у агента без модели поворотов и груза выходит дороже номинала: то же ждём от нового пола.
        plain = seen & (soil.drain < 1.45 * soil.dist)
        d = float(soil.dist[plain].sum())
        self.kappa = float(np.clip(soil.drain[plain].sum() / d, 1.0, 1.6)) if d > 1.0 else 1.0

    def _context(self, obs):
        a, s, r = self.a, self.s, self.a.rules
        per_m = a._per_m()
        leak, pending = 0.0, (0.0, 0.0)
        if a.inv:
            # Исследователь считает повороты по всему прогону; на переездах их на метр меньше, чем в поиске.
            path, model = a.inv._path, a.inv.model
            turns = min(path[1] / path[0] if path[0] > 2.0 else 1.5, s.turn_rate)
            per_m = model.per_meter(a.collected) + float(model.mean[2]) * turns
            leak = a.inv.reserve(obs.t)                          # подтверждённая утечка и невыясненный расход
            since = obs.t - a.inv._penalty_t
            unclear = a.inv.leak is None and a.inv._rest_ok_t < a.inv._penalty_t
        else:
            since, unclear = obs.t - a._hazard_t, True
        if r.faults and unclear and since < s.leak_s[1]:         # после штрафа утечку ещё никто не исключил
            pending = (s.p_leak, r.leak_per_s * (s.leak_s[1] - since))
        return {'battery': obs.battery, 'per_m': per_m, 'leak': leak, 'pending': pending,
                'load': float(a.inv.model.mean[1]) / per_m if a.inv else 0.0}

    # --- пути → отрезки ---------------------------------------------------------------------------

    def _leg(self, graph, n, origin):
        """Узлы графа по порядку движения — как отрезок для расчёта."""
        n = np.asarray(n, dtype=np.intp)
        if len(n) < 2:
            return Leg([], [], [], [])
        x, y = graph.xs[n], graph.ys[n]
        iy, ix = graph.iy[n], graph.ix[n]
        u = self._unk[iy, ix]
        unk = 0.5 * (u[1:] + u[:-1])
        mult = 0.5 * (graph.mult[n[1:]] + graph.mult[n[:-1]]) + unk * (self.kappa - 1.0)
        # Риск уже найденных зон; рядом с роботом не считается: он там стоит, и штрафа нет.
        far = np.hypot(x - origin[0], y - origin[1]) > 0.3
        risk = float(self.a._risk[iy, ix][far].max()) if self.a.hazards and far.any() else 0.0
        cell = self.core.worlds.cells(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))
        return Leg(np.hypot(np.diff(x), np.diff(y)), mult, unk, cell, risk)

    @staticmethod
    def _chain(pred, node):
        """Узлы пути по дереву pred: от node к источнику поля."""
        nodes = []
        while node >= 0:
            nodes.append(node)
            node = pred[node]
        return nodes

    def _to(self, pred, x, y, origin):
        """Путь от источника поля pred до (x, y), в порядке движения."""
        return self._leg(self.a.graph, self._chain(pred, self.a.graph.node(x, y))[::-1], origin)

    def _home(self, x, y, origin):
        """Дорога домой из (x, y) тем путём, каким поедет агент: (отрезок, цена пути по оценке агента)."""
        a = self.a
        if a._base_dist is None:
            a._home_cost(x, y)
        node = a.home_graph.node(x, y)
        cost = float(a._base_dist[node])
        if not math.isfinite(cost):
            return None
        return self._leg(a.home_graph, self._chain(a._base_pred, node), origin), cost

    def _options(self, obs, targets):
        """Варианты плана: домой, каждая цель и домой, пары лучших кандидатов."""
        a, s = self.a, self.s
        origin = (obs.x, obs.y)
        dist, pred = a.graph.field(obs.x, obs.y)
        here = self._home(obs.x, obs.y, origin)
        opts = [Option('home', 'сразу домой', [], [here], [{'type': 'return_base'}])]
        steps = {}
        for k, (tg, kind, p) in enumerate(targets):
            node = a.graph.node(tg['x'], tg['y'])
            ix, iy = a.arena.w2g(tg['x'], tg['y'])
            home = self._home(tg['x'], tg['y'], origin)
            if home is None or not math.isfinite(dist[node]) or a._risk[iy, ix] >= 0.5:
                continue
            st = Step(tg['id'], kind, p, self._to(pred, tg['x'], tg['y'], origin), k)
            sg = {'type': 'investigate' if kind == 'candidate' else 'explore', 'target': tg['id']}
            steps[tg['id']] = (st, home, sg, tg)
            opts.append(Option(tg['id'], f"{tg['id']} → домой", [st], [home, here], [sg]))
        top = sorted((i for i, v in steps.items() if v[0].kind == 'candidate'), key=lambda i: -steps[i][0].p)[:s.pairs]
        for i in top if len(top) > 1 else ():
            st_i, home_i, sg_i, tg_i = steps[i]
            _, pred_i = a.graph.field(tg_i['x'], tg_i['y'])
            for j in top:
                if j == i:
                    continue
                st_j, home_j, sg_j, tg_j = steps[j]
                leg = self._to(pred_i, tg_j['x'], tg_j['y'], origin)
                opts.append(Option(f'{i}+{j}', f'{i} → {j} → домой', [st_i, replace(st_j, leg=leg)],
                                   [home_j, home_i, here], [sg_i, sg_j]))
        return opts

    # --- решения ----------------------------------------------------------------------------------

    def plan(self, obs, state):
        """Выбор цели сравнением вариантов. Возвращает план в формате планировщика.

        Заодно помечает в состоянии, какие цели проходят по риску: если план за языковой моделью,
        она выбирает уже только из них — тогда возвращается None.
        """
        t0 = time.perf_counter()
        a = self.a
        left = a.n_samples - a.collected
        self._sync()
        targets = [(c, 'candidate', c['confidence']) for c in state['candidates']]
        targets += [(p, 'explore', min(1.0, p['unseen_share'] * left)) for p in state['explore_points']]
        options = self._options(obs, targets if left > 0 else [])
        rows = self.core.compare(options, **self._context(obs))
        by_id = {r['id']: r for r in rows}
        # Цель годится, если проходит по риску и окупает его прибавку: ожидаемые образцы стоят не меньше,
        # чем то, что теряется при невозврате, умноженное на добавленную вероятность не вернуться.
        home, r = by_id['home'], a.rules
        loss = r.pts_return + r.pts_battery_left * home['battery_p50']
        for row in rows:
            row['worth'] = bool(row is home or r.pts_sample * row['samples'] >= (row['risk'] - home['risk']) * loss)
        for tg in state['candidates'] + state['explore_points']:
            tg['feasible'] = tg['id'] in by_id and by_id[tg['id']]['ok'] and by_id[tg['id']]['worth']
        self._p = {tg['id']: p for tg, _, p in targets}
        best = self.core.choose(rows)
        subgoals, note = next(o for o in options if o.id == best['id']).subgoals, ''
        if self.s.choice == 'planner' or a.cfg.planner == 'llm':
            if a.cfg.planner == 'llm':
                rec = self._record(obs, state['trigger'], rows, best, t0)
                a.journal.add(obs.t, 'observe', self._text(rows, best) + ' Выбор среди прошедших по риску — за моделью.',
                              foresight=rec)
                return None
            inner = HeuristicPlanner().plan(state)             # правило выбирает среди прошедших по риску
            best = by_id[inner['subgoals'][0].get('target', 'home')]
            subgoals, note = inner['subgoals'], ' По правилу планировщика: ' + inner['reasoning']
        for tg in state['candidates'] + state['explore_points']:
            tg['feasible'] = tg['feasible'] or any(sg.get('target') == tg['id'] for sg in subgoals)
        rec = self._record(obs, state['trigger'], rows, best, t0)
        return {'reasoning': self._text(rows, best) + note, 'hypotheses': [], 'subgoals': subgoals,
                'source': 'foresight', 'exchanges': [], 'error': None, 'data': {'foresight': rec}}

    def check(self, obs):
        """Раз в секунду: не пора ли домой. Сравниваются «ехать дальше к текущей цели» и «сразу домой»."""
        a, s = self.a, self.s
        t = obs.t
        if a._returning:
            return
        if a.collected >= a.n_samples:
            return a._go_home(t, 'все образцы собраны')
        if a.rules.time_limit_s - t <= a._home_cost(obs.x, obs.y) / a.rules.drain_per_m / 0.15 + 10.0:
            return a._go_home(t, 'время прогона на исходе')
        sg = a.queue[0] if a.queue else None
        if sg is None or sg['type'] == 'return_base':
            return
        t0 = time.perf_counter()
        self._sync()
        origin = (obs.x, obs.y)
        here = self._home(obs.x, obs.y, origin)
        home = self._home(sg['x'], sg['y'], origin)
        if here is None or home is None:
            return
        pts = a.follower.pts[a.follower.i:]
        if a._path_goal is not None and math.dist(a._path_goal, (sg['x'], sg['y'])) <= 0.08 and len(pts) > 1:
            # Путь к цели уже построен исполнителем: берём его остаток, а не считаем заново.
            xy = np.asarray(pts)
            ix = np.clip(((xy[:, 0] - a.arena.x0) / a.arena.res).astype(int), 0, a.arena.w - 1)
            iy = np.clip(((xy[:, 1] - a.arena.y0) / a.arena.res).astype(int), 0, a.arena.h - 1)
            leg = self._leg(a.graph, a.graph._near[iy, ix], origin)
        else:
            leg = self._to(a.graph.field(obs.x, obs.y)[1], sg['x'], sg['y'], origin)
        name = sg.get('target') or 'цель'
        st = Step(name, 'candidate' if sg['type'] == 'investigate' else 'explore', self._p.get(sg.get('target'), 0.5),
                  leg)
        rows = self.core.compare([Option('home', 'сразу домой', [], [here]),
                                  Option(name, f'{name} → домой', [st], [home, here])], **self._context(obs))
        if rows[1]['risk'] <= s.risk_limit * s.abort_margin:
            return
        rec = self._record(obs, 'check', rows, rows[0], t0)
        a.journal.add(t, 'decision', self._text(rows, rows[0]), foresight=rec)
        a._go_home(t, f"если ехать дальше, риск не вернуться {rows[1]['risk']:.0%} — выше порога {s.risk_limit:.0%}")

    # --- запись -----------------------------------------------------------------------------------

    def _record(self, obs, trigger, rows, best, t0):
        s = self.s
        rec = {'t': round(obs.t, 1), 'trigger': trigger, 'worlds': s.worlds, 'risk_limit': s.risk_limit,
               'battery': round(obs.battery, 1), 'chosen': best['id'],
               'ms': round((time.perf_counter() - t0) * 1e3, 2),
               'unknowns': {'soil_per_m': round(self.rates[0], 3), 'hazard_per_m': round(self.rates[1], 3),
                            'soil_mults': list(self.core.worlds.rates[2]), 'per_m': round(self.a._per_m(), 2),
                            'per_m_sd': s.per_m_sd, 'stale': s.stale},
               'columns': [{'key': k, 'label': label, 'unit': unit} for k, label, unit in COLUMNS],
               'rows': [{**r, 'chosen': r is best} for r in rows]}
        self.log.append(rec)
        if self.a.rec is not None and hasattr(self.a.rec, 'add_foresight'):
            self.a.rec.add_foresight(rec)
        return rec

    def _text(self, rows, best):
        s = self.s
        n = len(rows)
        word = 'вариант' if n % 10 == 1 and n != 11 else 'варианта' if n % 10 in (2, 3, 4) and n not in (12, 13, 14) \
            else 'вариантов'
        text = (f"Сравнил {n} {word} при {s.worlds} возможных состояниях среды: выбрал «{best['label']}» — "
                f"ожидаю {best['samples']:.1f} образца, счёт {best['score']:.0f}, риск не вернуться {best['risk']:.1%}, "
                f"в худшем случае на базе останется {best['battery_p5']:.0f} ед.")
        if not best['ok']:
            text += f' Ни один вариант не проходит порог {s.risk_limit:.0%}, поэтому домой.'
        bad = [r for r in rows if not r['ok'] and r is not best]
        if bad:
            text += ' Отклонены по риску: ' + ', '.join(f"«{r['label']}» {r['risk']:.0%}" for r in bad[:4]) + '.'
        weak = [r for r in rows if r['ok'] and not r.get('worth', True) and r is not best]
        if weak:
            text += ' Не окупают прибавку риска: ' + ', '.join(f"«{r['label']}»" for r in weak[:4]) + '.'
        home = next((r for r in rows if r['id'] == 'home' and r is not best), None)
        if home:
            text += f" Сразу домой дало бы счёт {home['score']:.0f}."
        return text
