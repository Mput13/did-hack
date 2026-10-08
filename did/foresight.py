"""Сравнение будущих маршрутов: прежде чем ехать, прикинуть варианты плана в разных возможных мирах.

Правило «дорога туда + дорога домой × 1,1 + 4 ед.» держит один и тот же запас и на проверенном полу,
и там, где робот ещё не ездил. Здесь запас получается из того, чего агент не знает. Берётся M
возможных состояний среды («миров»), в каждом по-своему задано то, что агенту неизвестно:

  грунт     — где на непроверенном полу лежат зоны дорогого грунта и во сколько раз они дороже;
  опасность — где ещё есть ненайденные опасные зоны и чем кончится въезд (штраф, утечка заряда);
  образец   — есть ли он в кандидате на самом деле;
  расход    — сколько на самом деле стоит метр;
  сбой      — сколько ещё продлится идущая утечка;
  перемены  — не устарела ли карта там, где робот уже ездил.

Каждый вариант плана («C1 → домой», «E2 → домой», «C1 → C2 → домой», «сразу домой») проигрывается
во всех мирах: сколько образцов, какой счёт, сколько заряда останется на базе. Выбирается вариант
с наибольшей ожидаемой ценностью среди тех, где вероятность не вернуться не выше порога; если таких
нет — домой. Пути считаются один раз (CostGraph), миры — массивами numpy.

Две части: Foresight — сам расчёт на отрезках пути (его проверяют tests/test_foresight.py на
игрушечных входах), Advisor — связка с агентом: строит варианты из его карт и пишет таблицу в журнал.
"""
import copy
import math
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

CELL = 0.1            # м: сетка, на которой разложены миры
RETRACE = 1.05        # обратная дорога по своему следу: во сколько раз она длиннее, чем путь туда


@dataclass(frozen=True)
class Settings:
    worlds: int = 48                  # сколько состояний среды в выборке
    risk_limit: float = 0.05          # допустимая вероятность не вернуться
    abort_margin: float = 1.3         # в пути от цели отказываемся, когда риск выше порога во столько раз
    # --- чего агент не знает о полу
    soil_rate: float = 0.15           # зон дорогого грунта на метр непроверенного пола: исходное предположение
    soil_weight_m: float = 8.0        # его вес в метрах пути: после стольких метров своим наблюдениям верим так же
    soil_radius: tuple = (0.30, 0.55)
    soil_mults: tuple = (2.0, 3.0, 4.0)   # множители по умолчанию, пока своих зон не найдено
    hazard_rate: float = 0.03         # ненайденных опасных зон на метр непроверенного пола
    hazard_weight_m: float = 12.0
    hazard_radius: tuple = (0.25, 0.33)
    stale: float = 0.10               # вероятность, что карта устарела и проверенный пол уже не тот
    stale_exposure: float = 0.6       # насколько в таком мире проверенный пол похож на непроверенный
    # --- расход
    per_m_factor: float = 1.0         # поправка к оценке агента «заряд на метр»
    per_m_sd: float = 0.07            # её разброс (доля): повороты, объезды, неточность модели
    speed: float = 0.16               # м/с: средняя скорость, по ней считается расход за время в пути
    p_leak: float = 0.25              # доля въездов в опасную зону, после которых начинается утечка
    leak_s: tuple = (20.0, 30.0)      # сколько она длится
    dwell: float = 0.3                # ед. заряда на подъезд вплотную и сбор
    # --- ценность
    explore_value: float = 0.3        # какая доля «неясной массы» вокруг точки разведки превращается в образцы
    future_per_unit: float = 0.5      # очков за единицу заряда, оставшегося на следующие цели (0 — не учитывать)
    safe_bias: float = 8.0            # маршрут «по проверенному»: во сколько раз дороже непроверенная клетка
    pairs: int = 3                    # пары «Ci → Cj → домой» строятся для стольких лучших кандидатов
    seed: int = 0


DEFAULTS = Settings()                 # отладочные серии подменяют это значение целиком


class Leg:
    """Отрезок пути глазами агента: шаги, оценка множителя грунта, доля непроверенного пола."""

    __slots__ = ('ds', 'unk', 'open', 'cell', 'risk', 'metres', 'length', 'unknown_m', 'est')

    def __init__(self, ds, mult, unk, cell, risk=0.0, est=None):
        self.ds = np.asarray(ds, dtype=float)             # длина шага, м
        self.unk = np.asarray(unk, dtype=float)           # 0 — пол проверен колёсами, 1 — там не ездили
        self.open = self.unk > 0.5
        self.cell = np.asarray(cell, dtype=np.intp)       # клетка сетки миров под каждым шагом
        self.risk = float(risk)                           # вероятность задеть уже найденную опасную зону
        self.length = float(self.ds.sum())
        self.metres = float(self.ds @ np.asarray(mult, dtype=float))   # метры обычного пола по оценке агента
        self.unknown_m = float(self.ds @ self.unk)
        self.est = self.metres if est is None else float(est)          # цена пути, по которой агент выбирает маршрут


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
    homes: dict           # маршрут → [(Leg, оценка)]: домой от последней цели, от предпоследней, …, от старта
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
        self.set_rates(0.0, 0.0, s.soil_mults)

    def set_rates(self, n_soil, n_hazard, mults):
        """Сколько зон грунта и опасных зон в среднем на арене и какими бывают множители."""
        key = (round(n_soil, 1), round(n_hazard, 2), tuple(mults))
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
        if self.stale.any():              # карта устарела: зоны могут лежать и на проверенном полу
            out[self.stale] += g[self.stale] @ (leg.ds * np.maximum(self.s.stale_exposure - leg.unk, 0.0))
        return out

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
    ('value', 'Ценность', 'очки'),
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

    def __init__(self, bounds=(-3.0, -3.0, 3.0, 3.0), settings=None, centres=None, rules=None):
        from .config import Rules
        self.s = settings or DEFAULTS
        self.rules = rules or Rules()
        self.worlds = Worlds(bounds, self.s, centres)

    def compare(self, options, battery, per_m, left=1, load=0.0, leak=0.0, pending=(0.0, 0.0)):
        """Таблица: по строке на вариант (с лучшим для него маршрутом домой).

        battery — заряд сейчас; per_m — заряд на метр обычного пола по оценке агента; left — сколько
        образцов ещё не собрано; load — на какую долю каждый новый образец удорожает метр;
        leak — сколько заряда ещё заберёт идущая утечка; pending — (вероятность, ед.): утечка,
        которая могла начаться после недавнего штрафа, но ещё не подтверждена и не исключена.
        """
        w, s, r = self.worlds, self.s, self.rules
        start = battery - leak * (0.5 + w.u_now) - (w.u_pending < pending[0]) * pending[1]
        # Въезд в опасную зону: разовая потеря, а при правилах со сбоями — иногда утечка на 20–30 с.
        hit_loss = np.full(w.n, r.hazard_battery_hit)
        if r.faults:
            hit_loss = hit_loss + (w.u_leak < s.p_leak) * r.leak_per_s * (
                s.leak_s[0] + w.u_leak_s * (s.leak_s[1] - s.leak_s[0]))
        ctx = (start, per_m * s.per_m_factor, max(0, left), load, hit_loss)
        rows = []
        for opt in options:
            best = None
            for route in opt.homes:
                row = self._play(opt, route, ctx)
                if best is None or _better(row, best, s.risk_limit):
                    best = row
            rows.append(best)
        return rows

    def _play(self, opt, route, ctx):
        """Один вариант с одним маршрутом домой во всех мирах сразу."""
        w, s, r = self.worlds, self.s, self.rules
        start, per_m, left, load, hit_loss = ctx
        idle = r.drain_idle_per_s / s.speed                 # заряд за время в пути, на метр
        b = start.copy()
        got = np.zeros(w.n)                                 # собрано образцов в этом варианте
        found = 0.0                                         # ожидаемые находки разведки
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
        alts = opt.homes[route]
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

        sd = s.per_m_sd * spent + 0.15                      # неточность самой оценки расхода
        p_back = ndtr(b / sd)                               # вероятность вернуться в этом мире
        left_b = np.maximum(b, 0.0)
        score = (r.pts_sample * (got + found) + r.pts_hazard_hit * hits
                 + p_back * (r.pts_return + r.pts_battery_left * left_b))
        # Вариант — не конец прогона: заряд, оставшийся после него, ещё пойдёт на следующие цели.
        future = p_back * np.minimum(r.pts_sample * np.maximum(left - got, 0.0), s.future_per_unit * left_b) \
            if opt.steps else 0.0
        end = np.maximum(b + sd * w.z, 0.0)                 # заряд на базе с учётом разброса расхода
        p5, p50, p95 = np.percentile(end, [5, 50, 95])
        risk = float(1.0 - p_back.mean())
        return {'id': opt.id, 'label': opt.label, 'route': route,
                'samples': round(float(got.mean() + found), 2), 'score': round(float(score.mean()), 2),
                'value': round(float(np.mean(score + future)), 2), 'risk': round(risk, 4),
                'battery_p5': round(float(p5), 1), 'battery_p50': round(float(p50), 1),
                'battery_p95': round(float(p95), 1), 'cost': round(float(spent.mean()), 1),
                'hits': round(float(hits.mean()), 2), 'unknown_m': round(unknown_m, 1),
                'ok': bool(risk <= s.risk_limit)}

    def choose(self, rows):
        """Наибольшая ценность при риске не выше порога; если никто не проходит — домой самым надёжным путём."""
        ok = [r for r in rows if r['ok']]
        if ok:
            return max(ok, key=lambda r: r['value'])
        homes = [r for r in rows if r['id'] == 'home'] or rows
        return min(homes, key=lambda r: r['risk'])


def _better(a, b, limit):
    """Какой из двух маршрутов домой оставить для варианта: проходящий по риску и более ценный."""
    if a['ok'] != b['ok']:
        return a['ok']
    return a['value'] > b['value'] if a['ok'] else a['risk'] < b['risk']


ROUTES = {'fast': 'коротким путём', 'safe': 'по проверенному полу'}


class Advisor:
    """Связка расчёта с агентом: варианты из его карт и путей, выбор, запись таблицы в журнал."""

    def __init__(self, agent, settings=None):
        a = self.a = agent
        self.s = settings or replace(DEFAULTS, risk_limit=getattr(a.cfg, 'risk_limit', DEFAULTS.risk_limit))
        arena = a.arena
        X, Y = arena.cell_centers()
        free = arena.clear >= 0.10
        bounds = (arena.x0, arena.y0, arena.x0 + arena.w * arena.res, arena.y0 + arena.h * arena.res)
        self.core = Foresight(bounds, self.s, np.column_stack([X[free], Y[free]]), a.rules)
        self.area = float(arena.free.sum()) * arena.res ** 2
        self.safe = copy.copy(a.home_graph)       # тот же граф с другими ценами: домой только по проверенному полу
        self.route = 'fast'                       # каким из двух маршрутов сейчас ехать домой
        self.log = []                             # все решения прогона: таблицы «варианты × показатели»
        self.kappa = 1.0
        self.rates = (self.s.soil_rate, self.s.hazard_rate)
        self._unk = np.ones(arena.free.shape)
        self._version = None
        self._safe_field = None
        self._p = {}                              # цель → вероятность образца на момент выбора

    # --- что агент знает и чего не знает --------------------------------------------------------

    def _sync(self):
        """Обновить то, от чего зависят миры: проверенный пол, частоты зон, маршрут по проверенному."""
        a, s = self.a, self.s
        soil = a.soil
        seen = soil.dist > 0.03                                  # клетки грунта, по которым робот проехал сам
        if a.cfg.learn_soil:
            # Проверено — там, где проехал; рядом со следом — отчасти (по сглаженной оценке модели грунта).
            self._unk = np.where(soil._fine(seen.astype(float), 0.0) > 0.5, 0.0, 1.0 - soil.confidence_grid())
        new_m = float(seen.sum()) * soil.res                     # сколько нового пола уже проверено колёсами
        zones = soil.zones() if a.cfg.learn_soil else []
        # Частота на метр непроверенного пола: исходное предположение плюс то, что встретилось на своём пути.
        lam_s = (s.soil_rate * s.soil_weight_m + len(zones)) / (s.soil_weight_m + new_m)
        lam_h = (s.hazard_rate * s.hazard_weight_m + len(a.hazard_map.zones)) / (s.hazard_weight_m + new_m)
        self.rates = (lam_s, lam_h)
        mults = tuple(sorted(round(min(max(z['mult'], 1.5), 5.0) * 2) / 2 for z in zones)) + tuple(s.soil_mults)
        self.core.worlds.set_rates(lam_s * self.area / sum(s.soil_radius), lam_h * self.area / sum(s.hazard_radius),
                                   mults)
        # Обычный пол у агента без модели поворотов и груза выходит дороже номинала: то же ждём от нового пола.
        plain = (soil.dist > 0.03) & (soil.drain < 1.45 * soil.dist)
        d = float(soil.dist[plain].sum())
        self.kappa = float(np.clip(soil.drain[plain].sum() / d, 1.0, 1.6)) if d > 1.0 else 1.0
        if self._version != a._cost_version or self._safe_field is None:
            self._version = a._cost_version
            bias = 1.0 + s.safe_bias * self._unk
            if a.hazards:
                bias = bias * (1.0 + a.cfg.hazard_weight * a._risk)
            self.safe.set_cost(soil.mult_grid() if a.cfg.learn_soil else None, bias=bias)
            self._safe_field = self.safe.field(*a.base)

    def _context(self, obs):
        a, s, r = self.a, self.s, self.a.rules
        per_m = a._per_m()
        leak, pending = 0.0, (0.0, 0.0)
        if a.inv:
            leak = a.inv.reserve(obs.t)                          # подтверждённая утечка и невыясненный расход
            since = obs.t - a.inv._penalty_t
            unclear = a.inv.leak is None and a.inv._rest_ok_t < a.inv._penalty_t
        else:
            since, unclear = obs.t - a._hazard_t, True
        if r.faults and unclear and since < s.leak_s[1]:         # после штрафа утечку ещё никто не исключил
            pending = (s.p_leak, r.leak_per_s * (s.leak_s[1] - since))
        return {'battery': obs.battery, 'per_m': per_m, 'left': a.n_samples - a.collected,
                'load': float(a.inv.model.mean[1]) / per_m if a.inv else 0.0, 'leak': leak, 'pending': pending}

    # --- пути → отрезки ---------------------------------------------------------------------------

    def _leg(self, graph, pred, node, origin, est=None):
        """Путь по дереву pred от node до источника поля — как отрезок для расчёта."""
        nodes = []
        while node >= 0:
            nodes.append(node)
            node = pred[node]
        n = np.asarray(nodes, dtype=np.intp)
        if len(n) < 2:
            return Leg([], [], [], [], 0.0, est)
        x, y = graph.xs[n], graph.ys[n]
        mx, my = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
        u = self._unk[graph.iy[n], graph.ix[n]]
        unk = 0.5 * (u[1:] + u[:-1])
        mult = 0.5 * (graph.mult[n[1:]] + graph.mult[n[:-1]]) + unk * (self.kappa - 1.0)
        # Риск уже найденных зон; рядом с роботом не считается: он там стоит, и штрафа нет.
        far = np.hypot(x - origin[0], y - origin[1]) > 0.3
        risk = float(self.a._risk[graph.iy[n], graph.ix[n]][far].max()) if far.any() else 0.0
        return Leg(np.hypot(np.diff(x), np.diff(y)), mult, unk, self.core.worlds.cells(mx, my), risk, est)

    def _home(self, x, y, origin):
        """Дорога домой из (x, y) двумя маршрутами: {маршрут: (отрезок, цена пути по оценке агента)}."""
        a = self.a
        if a._base_dist is None:
            a._home_cost(x, y)
        out = {}
        for route, graph, (dist, pred) in (('fast', a.home_graph, (a._base_dist, a._base_pred)),
                                           ('safe', self.safe, self._safe_field)):
            node = graph.node(x, y)
            if math.isfinite(dist[node]):
                out[route] = (self._leg(graph, pred, node, origin, float(dist[node])), float(dist[node]))
        return out

    def _options(self, obs, targets):
        """Варианты плана: домой, каждая цель и домой, пары лучших кандидатов."""
        a, s = self.a, self.s
        origin = (obs.x, obs.y)
        dist, pred = a.graph.field(obs.x, obs.y)
        here = self._home(obs.x, obs.y, origin)
        opts = [Option('home', 'сразу домой', [], {r: [h] for r, h in here.items()}, [{'type': 'return_base'}])]
        steps = {}
        for k, (tg, kind, p) in enumerate(targets):
            node = a.graph.node(tg['x'], tg['y'])
            ix, iy = a.arena.w2g(tg['x'], tg['y'])
            if not math.isfinite(dist[node]) or a._risk[iy, ix] >= 0.5:
                continue
            st = Step(tg['id'], kind, p, self._leg(a.graph, pred, node, origin), k)
            homes = self._home(tg['x'], tg['y'], origin)
            sg = {'type': 'investigate' if kind == 'candidate' else 'explore', 'target': tg['id']}
            steps[tg['id']] = (st, homes, sg, tg)
            opts.append(Option(tg['id'], f"{tg['id']} → домой", [st],
                               {r: [homes[r], here[r]] for r in homes if r in here}, [sg]))
        top = sorted((i for i, v in steps.items() if v[0].kind == 'candidate'), key=lambda i: -steps[i][0].p)[:s.pairs]
        for i in top:
            st_i, homes_i, sg_i, tg_i = steps[i]
            _, pred_i = a.graph.field(tg_i['x'], tg_i['y'])
            for j in top:
                if j == i:
                    continue
                st_j, homes_j, sg_j, tg_j = steps[j]
                leg = self._leg(a.graph, pred_i, a.graph.node(tg_j['x'], tg_j['y']), origin)
                opts.append(Option(f'{i}+{j}', f'{i} → {j} → домой', [st_i, replace(st_j, leg=leg)],
                                   {r: [homes_j[r], homes_i[r], here[r]] for r in homes_j if r in homes_i and r in here},
                                   [sg_i, sg_j]))
        return [o for o in opts if o.homes]

    # --- решения ----------------------------------------------------------------------------------

    def plan(self, obs, state):
        """Выбор цели сравнением вариантов. Возвращает план в формате планировщика.

        Заодно помечает в состоянии, какие цели проходят по риску: языковая модель (если план за ней)
        выбирает уже только из них — тогда возвращается None.
        """
        t0 = time.perf_counter()
        a = self.a
        left = a.n_samples - a.collected
        self._sync()
        targets = [(c, 'candidate', c['confidence']) for c in state['candidates']]
        targets += [(p, 'explore', min(1.0, p['unseen_share'] * left)) for p in state['explore_points']]
        options = self._options(obs, targets) if left > 0 else self._options(obs, [])
        rows = self.core.compare(options, **self._context(obs))
        best = self.core.choose(rows)
        opt = next(o for o in options if o.id == best['id'])
        ok = {r['id'] for r in rows if r['ok']} | {sg.get('target') for sg in opt.subgoals}
        for tg in state['candidates'] + state['explore_points']:
            tg['feasible'] = tg['id'] in ok
        self._p = {tg['id']: p for tg, _, p in targets}
        rec = self._record(obs, state['trigger'], rows, best, (time.perf_counter() - t0) * 1e3)
        text = self._text(rows, best)
        if a.cfg.planner == 'llm':
            a.journal.add(obs.t, 'observe', text + ' Выбор среди проходящих по риску — за моделью.', foresight=rec)
            return None
        self.route = best['route']
        return {'reasoning': text, 'hypotheses': [], 'subgoals': opt.subgoals, 'source': 'foresight',
                'exchanges': [], 'error': None, 'data': {'foresight': rec}}

    def check(self, obs):
        """Раз в секунду: не пора ли домой, а на обратном пути — каким маршрутом ехать."""
        a, s = self.a, self.s
        t = obs.t
        if a._returning:
            return self._reroute(obs)
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
        _, pred = a.graph.field(obs.x, obs.y)
        node = a.graph.node(sg['x'], sg['y'])
        homes = self._home(sg['x'], sg['y'], origin)
        routes = [r for r in homes if r in here]
        if not routes:
            return
        kind = 'candidate' if sg['type'] == 'investigate' else 'explore'
        name = sg.get('target') or 'цель'
        st = Step(name, kind, self._p.get(sg.get('target'), 0.5), self._leg(a.graph, pred, node, origin))
        opts = [Option('home', 'сразу домой', [], {r: [here[r]] for r in routes}),
                Option(name, f'{name} → домой', [st], {r: [homes[r], here[r]] for r in routes})]
        home, cont = self.core.compare(opts, **self._context(obs))
        if cont['risk'] <= s.risk_limit * s.abort_margin:
            return
        self.route = home['route']
        rec = self._record(obs, 'check', [home, cont], home, (time.perf_counter() - t0) * 1e3)
        a.journal.add(t, 'decision', self._text([home, cont], home), foresight=rec)
        a._go_home(t, f"если ехать дальше, риск не вернуться {cont['risk']:.0%} — выше порога {s.risk_limit:.0%}")

    def _reroute(self, obs):
        """Обратный путь: короткий маршрут или по проверенному полу — что сейчас надёжнее."""
        a = self.a
        t0 = time.perf_counter()
        self._sync()
        here = self._home(obs.x, obs.y, (obs.x, obs.y))
        if len(here) < 2:
            return
        opts = [Option('home', f'домой {ROUTES[r]}', [], {r: [here[r]]}) for r in here]
        rows = {r['route']: r for r in self.core.compare(opts, **self._context(obs))}
        cur = rows[self.route]
        other = rows['safe' if self.route == 'fast' else 'fast']
        if cur['ok'] and other['ok']:
            switch = other['value'] > cur['value'] + 0.3
        else:
            switch = other['risk'] < cur['risk'] - 0.01
        if not switch:
            return
        self.route = other['route']
        a._path_goal = None
        rec = self._record(obs, 'route', [other, cur], other, (time.perf_counter() - t0) * 1e3)
        a.journal.add(obs.t, 'decision', f"Сравнил 2 маршрута домой при {self.s.worlds} возможных состояниях среды: "
                      f"еду {ROUTES[other['route']]} — риск не вернуться {other['risk']:.0%} против {cur['risk']:.0%}, "
                      f"на базе останется около {other['battery_p50']:.0f} ед. против {cur['battery_p50']:.0f}.",
                      foresight=rec)

    def home_graph(self):
        """Граф, по которому сейчас строится дорога домой."""
        return self.safe if self.route == 'safe' and self._safe_field is not None else self.a.home_graph

    # --- запись -----------------------------------------------------------------------------------

    def _record(self, obs, trigger, rows, best, ms):
        s = self.s
        rec = {'t': round(obs.t, 1), 'trigger': trigger, 'worlds': s.worlds, 'risk_limit': s.risk_limit,
               'battery': round(obs.battery, 1), 'chosen': best['id'], 'route': best['route'], 'ms': round(ms, 2),
               'unknowns': {'soil_per_m': round(self.rates[0], 3), 'hazard_per_m': round(self.rates[1], 3),
                            'soil_mults': list(self.core.worlds.rates[2]), 'per_m': round(self.a._per_m(), 2),
                            'per_m_sd': s.per_m_sd, 'stale': s.stale},
               'columns': [{'key': k, 'label': label, 'unit': unit} for k, label, unit in COLUMNS],
               'rows': [{**r, 'route_label': ROUTES[r['route']], 'chosen': r is best} for r in rows]}
        self.log.append(rec)
        if self.a.rec is not None and hasattr(self.a.rec, 'add_foresight'):
            self.a.rec.add_foresight(rec)
        return rec

    def _text(self, rows, best):
        s = self.s
        head = f'Сравнил {len(rows)} вариантов при {s.worlds} возможных состояниях среды: '
        what = (f"«{best['label']}» ({ROUTES[best['route']]})" if best['id'] == 'home'
                else f"«{best['label']}», обратно {ROUTES[best['route']]}")
        text = (f"{head}выбрал {what} — ожидаю {best['samples']:.1f} образца, счёт {best['score']:.0f}, "
                f"риск не вернуться {best['risk']:.1%}, в худшем случае на базе останется {best['battery_p5']:.0f} ед.")
        if not best['ok']:
            text += f' Ни один вариант не проходит порог {s.risk_limit:.0%}, поэтому домой.'
        bad = [r for r in rows if not r['ok'] and r is not best]
        if bad:
            text += ' Отклонены по риску: ' + ', '.join(f"«{r['label']}» {r['risk']:.0%}" for r in bad[:4]) + '.'
        home = next((r for r in rows if r['id'] == 'home' and r is not best), None)
        if home:
            text += f" Сразу домой дало бы счёт {home['score']:.0f}."
        return text
