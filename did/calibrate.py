"""Самокалибровка: агент проверяет свои допущения о датчике образцов и о расходе заряда (R14).

Формула датчика «показание = 1 − расстояние / 2 м» и расход 2,5 единицы на метр в условии задачи не
заданы — это допущения команды. Здесь инструменты, которые работают только с тем, что робот
измерил сам:

  LawBank    — несколько карт вероятностей с разными законами датчика 1 − (d / R) ** p. Все получают
               одни и те же показания, сборы и промахи; впереди та, чья карта лучше их предсказывала.
               Сетка законов сначала грубая вокруг допущения, потом сужается вокруг лучшего.
  fit_law    — после сбора образец лежал в радиусе сбора от робота, а показания на подъезде
               записаны: по парам «расстояние — показание» напрямую подбираются R и p. Это
               независимая сверка вывода, сделанного по картам.
  DrainMeter — нижняя оценка расхода на метр по пройденным отрезкам и отдельно — вывод, обычный ли
               это пол (пока весь путь мог идти по дорогому грунту, допущение не опровергается).
  Calibrator — всё это в агентском цикле, с записью гипотез и выводов в журнал.

Истинных правил мира калибрующийся агент не получает: то, что он калибрует (CALIBRATED), на старте
берётся из допущений команды — см. assumed_rules и did/runner.py.
"""
import math
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from .belief import SampleBelief
from .config import Rules

CALIBRATED = ('sensor_range_m', 'sensor_law', 'drain_per_m')   # что агент проверяет сам и потому заранее не узнаёт
SHAPES = ((0.5, 'по корню'), (1.0, 'по прямой'), (2.0, 'по квадрату'))


def assumed_rules(rules):
    """Правила, с которыми стартует калибрующийся агент, когда отдельных допущений ему не задано.

    Дальность датчика, форма закона и расход на метр — допущения команды (значения Rules() по
    умолчанию), какими бы они ни были в мире. Остальное (лимит времени, заряд на старте, радиус сбора,
    шум датчика, цена простоя) агент не калибрует и получает как все варианты. Возвращается новый
    объект: общего с судьёй у агента нет.
    """
    default = Rules()
    return replace(rules, **{k: getattr(default, k) for k in CALIBRATED})


def shape_name(power):
    """К какой из трёх названных форм ближе показатель степени (по отношению, а не по разности)."""
    return min(SHAPES, key=lambda s: abs(math.log(power / s[0])))[1]


def law_text(rng, power):
    return f'датчик слышит образец на {rng:.2f} м, показание падает {shape_name(power)}: 1 − (d/R)^{power:.2f}'


def response(d, rng, power):
    return np.maximum(0.0, 1.0 - (np.asarray(d, dtype=float) / rng) ** power)


def _clipped_mean(f, sigma):
    """Среднее показание при чистом значении f: шум добавлен, затем обрезка до [0, 1]."""
    a, b = f / sigma, (1.0 - f) / sigma
    pdf = lambda v: np.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)
    return f * (ndtr(a) - ndtr(-b)) + sigma * (pdf(a) - pdf(b)) + ndtr(-b)


class LawBank:
    """Карты вероятностей с разными законами датчика и счёт того, как каждая предсказывает наблюдения.

    Гипотезы — сетка вокруг center: дальность шагами ×steps[0], показатель степени шагами ×steps[1],
    по half узлов в каждую сторону. ops — уже случившиеся наблюдения: новая сетка проигрывает их
    заново, так что её счёт сравним со счётом прежней. assumed — исходное допущение: оно остаётся
    среди гипотез при любой сетке, чтобы от него отказывались только по явному перевесу.
    """

    def __init__(self, arena, n_samples, center, steps=(1.25, math.sqrt(2.0)), half=2, ops=(), assumed=None):
        self.arena, self.n_samples = arena, n_samples
        self.center, self.steps, self.half = center, steps, half
        self.side = 2 * half + 1
        idx = range(-half, half + 1)
        self.laws = [(center[0] * steps[0] ** i, center[1] * steps[1] ** j) for i in idx for j in idx]
        self.assumed = assumed or center
        if self.assumed not in self.laws:
            self.laws.append(self.assumed)          # вне сетки: в уточнении между узлами не участвует
        self.null = self.laws.index(self.assumed)
        self.maps = [SampleBelief(arena, n_samples, r, power=p) for r, p in self.laws]
        self.score = np.zeros(len(self.laws))       # сумма логарифмов вероятности наблюдений
        self.current = self.null                    # рабочая гипотеза: её карта ведёт агента
        self.ops = []
        self.n_signal = 0                           # показаний с заметным сигналом
        for op in ops:
            self.apply(*op)

    @property
    def belief(self):
        return self.maps[self.current]

    @property
    def law(self):
        return self.laws[self.current]

    def apply(self, kind, *args):
        """Наблюдение идёт во все карты; до обновления каждая «отвечает», насколько она его ждала.

        update(x, y, z, sigma) — показание; unscored(x, y, z, sigma) — показание при сбое датчика:
        карты его учитывают (как у обычного агента), но в сравнение законов оно не идёт;
        collected(x, y, r) — сбор удался; missed(x, y, r) — ложный сбор; clear_disc(x, y, r, factor)
        и relax(alpha) — правки карты без счёта.
        """
        self.ops.append((kind, *args))
        for i, m in enumerate(self.maps):
            if kind == 'update':
                self.score[i] += m.update(*args)
            elif kind == 'unscored':
                m.update(*args)
            elif kind == 'collected':
                self.score[i] += math.log(max(m.prob_within(*args), 1e-3))
                m.collected(*args[:2])
            elif kind == 'missed':
                self.score[i] += math.log(max(1.0 - m.prob_within(*args), 1e-3))
                m.clear_disc(args[0], args[1], 0.30, factor=0.05)
            else:
                getattr(m, kind)(*args)
        if kind == 'update':
            self.n_signal += args[2] > 0.15

    def lead(self):
        """(номер лучшей карты, её перевес над рабочей в логарифмах, её перевес над допущением)."""
        i = int(self.score.argmax())
        return i, float(self.score[i] - self.score[self.current]), float(self.score[i] - self.score[self.null])

    def posterior(self):
        w = np.exp(self.score - self.score.max())
        return w / w.sum()

    def zoom(self, min_steps=(1.05, 1.08)):
        """Новая сетка 3×3 вокруг лучшего узла; шаг по оси вдвое мельче, если лучший узел на ней не с краю.

        Лучший узел остаётся в новой сетке (её центр), так что сужение не может ухудшить оценку.
        Возвращает None, когда мельче некуда и лучший узел уже в центре.
        """
        best = int(self.score.argmax())
        if best >= self.side ** 2:
            return None                                   # лучшая гипотеза — само допущение вне сетки
        edge = [b in (0, self.side - 1) for b in divmod(best, self.side)]
        steps = tuple(s if e else max(math.sqrt(s), m) for s, e, m in zip(self.steps, edge, min_steps))
        if steps == self.steps and self.laws[best] == self.center:
            return None
        return LawBank(self.arena, self.n_samples, self.laws[best], steps, half=1, ops=self.ops, assumed=self.assumed)


def _disc(u, v, radius):
    """Квадрат [−1, 1]² → круг радиуса radius, гладко и взаимно однозначно: смещение не длиннее radius."""
    return radius * u * math.sqrt(1.0 - 0.5 * v * v), radius * v * math.sqrt(1.0 - 0.5 * u * u)


def fit_law(sites, sigma, start=(2.0, 1.0), radius=0.30, bounds=((0.6, 6.0), (0.25, 4.0))):
    """Закон датчика по показаниям вокруг собранных образцов.

    sites — [{'x', 'y', 'pts'}]: где робот стоял при сборе и показания до него, массив (n, 3) из
    x, y, z. Сам образец лежал где-то в круге radius от точки сбора, поэтому вместе с дальностью R
    и показателем p подбирается его положение — именно в круге (см. _disc), а не в квадрате. Часть показаний относится к другому, более близкому
    образцу — они завышены и гасятся устойчивой функцией потерь.
    Возвращает словарь с оценками и их погрешностями или None, если данных мало.
    """
    sites = [s for s in sites if len(s['pts']) >= 8]
    if not sites:
        return None
    pts = [np.asarray(s['pts'], dtype=float) for s in sites]
    k = len(sites)
    z = np.concatenate([a[:, 2] for a in pts])

    def offsets(q):
        return [_disc(q[2 + 2 * i], q[3 + 2 * i], radius) for i in range(k)]

    def dist(q):
        return np.concatenate([np.hypot(a[:, 0] - sites[i]['x'] - ox, a[:, 1] - sites[i]['y'] - oy)
                               for i, (a, (ox, oy)) in enumerate(zip(pts, offsets(q)))])

    def resid(q):
        return (z - _clipped_mean(response(dist(q), q[0], math.exp(q[1])), sigma)) / sigma

    lo = [bounds[0][0], math.log(bounds[1][0])] + [-1.0] * (2 * k)
    hi = [bounds[0][1], math.log(bounds[1][1])] + [1.0] * (2 * k)
    best = None
    for power0 in dict.fromkeys((start[1], 0.5, 1.0, 2.0)):
        for rng0 in dict.fromkeys((start[0], 1.4, 2.8)):
            q0 = np.array([rng0, math.log(power0)] + [0.0] * (2 * k))
            res = least_squares(resid, q0, bounds=(lo, hi), loss='cauchy', f_scale=2.0, max_nfev=80)
            if best is None or res.cost < best.cost:
                best = res
    q = best.x
    rng, power = float(q[0]), float(math.exp(q[1]))
    r = resid(q)
    inlier = np.abs(r) < 4.0
    heard = inlier & (z > 2.0 * sigma)
    if heard.sum() < 3:
        return None
    jac = best.jac[inlier]
    cov = np.linalg.pinv(jac.T @ jac) * max(1.0, float((r[inlier] ** 2).mean()))
    d = dist(q)
    return {'range': rng, 'power': power, 'se_range': math.sqrt(max(cov[0, 0], 0.0)),
            'se_power': power * math.sqrt(max(cov[1, 1], 0.0)), 'n': int(heard.sum()),
            'd_min': float(d[heard].min()), 'd_max': float(d[heard].max()),
            'rms': float(np.sqrt((r[inlier] ** 2).mean())) * sigma, 'sites': k,
            'offsets': [(float(ox), float(oy)) for ox, oy in offsets(q)]}


class DrainMeter:
    """Расход заряда на метр по пройденным отрезкам: нижняя оценка и вывод об обычном поле — порознь.

    Дорогой грунт расход только повышает, поэтому самые дешёвые метры пути — оценка сверху для расхода
    обычного пола: low — расход, дешевле которого пройдено не больше quantile пути и не больше support
    метров (чтобы долгая езда по грунту не подняла оценку). Простой уже вычтен, скачки (опасная зона)
    сюда не попадают.

    Но «самые дешёвые из пройденных» — ещё не «обычный пол»: весь путь мог идти по грунту. Поэтому
    вывод о допущении делается отдельно (verdict) и только когда обычный пол выделен:
      low ниже допущения  — грунт расход не понижает, значит обычный пол не дороже low: допущение завышено;
      low сходится с ним  — допущение подтверждено;
      low выше допущения  — опровергнуто, только если по этой цене пройдено не меньше wide_path метров
                            на участке размахом не меньше wide_span: пятно грунта таким большим не
                            бывает (допущение агента о размере пятен, как у опасных зон).
    До вывода low годится как осторожная оценка для запаса на дорогу домой, но не как расход пола.
    """

    def __init__(self, nominal, min_path=1.2, quantile=0.3, support=1.5, tol=0.05, band=0.08,
                 wide_path=4.0, wide_span=2.5):
        self.nominal = nominal
        self.min_path, self.quantile, self.support = min_path, quantile, support
        self.tol, self.band = tol, band
        self.wide_path, self.wide_span = wide_path, wide_span
        self.rate = []          # заряд на метр на отрезке
        self.ds = []
        self.xy = []            # где отрезок пройден (None — неизвестно)
        self.path = 0.0
        self.low = None         # нижняя оценка; None — данных пока мало
        self.low_se = None      # её погрешность: разброс расхода на «полке» самых дешёвых отрезков
        self.plateau_path = 0.0   # сколько метров пройдено по цене low (±band)
        self.plateau_span = 0.0   # размах этих отрезков на арене, м (0 — положения неизвестны)

    @property
    def value(self):
        return self.low

    def add(self, ds, spent, x=None, y=None):
        """Отрезок длиной ds с концом в (x, y) и заряд на нём без простоя. Возвращает нижнюю оценку или None."""
        if ds <= 0.0:
            return None
        self.rate.append(spent / ds)
        self.ds.append(ds)
        self.xy.append(None if x is None else (x, y))
        self.path += ds
        if self.path < self.min_path:
            return None
        rate, ds_all = np.asarray(self.rate), np.asarray(self.ds)
        order = np.argsort(rate, kind='stable')
        cum = np.cumsum(ds_all[order])
        self.low = float(rate[order][np.searchsorted(cum, min(self.quantile * cum[-1], self.support))])
        near = np.abs(rate - self.low) <= self.band * abs(self.low)
        w = ds_all[near]
        self.plateau_path = float(w.sum())
        mean = float((rate[near] * w).sum() / w.sum())
        spread = math.sqrt(float((w * (rate[near] - mean) ** 2).sum() / w.sum()))
        self.low_se = spread / math.sqrt(int(near.sum()))
        pts = np.array([p for p, n in zip(self.xy, near) if n and p is not None], dtype=float).reshape(-1, 2)
        self.plateau_span = float(np.hypot(*np.ptp(pts, axis=0))) if len(pts) > 1 else 0.0
        return self.low

    def verdict(self):
        """Что можно сказать о допущении nominal: 'confirmed', 'refuted' или None — обычный пол не выделен."""
        if self.low is None or self.plateau_path < self.min_path:
            return None
        if self.low < self.nominal * (1.0 - self.tol):
            return 'refuted'
        if self.low <= self.nominal * (1.0 + self.tol):
            return 'confirmed'
        if self.plateau_path >= self.wide_path and self.plateau_span >= self.wide_span:
            return 'refuted'
        return None

    @property
    def base(self):
        """Расход на метр обычного пола, когда он выделен; иначе None."""
        return self.low if self.verdict() else None


class Calibrator:
    """Самокалибровка в агентском цикле: держит рабочий закон датчика и расход на метр.

    Агент обращается к карте образцов через этот объект, чтобы те же показания, сборы и промахи
    доходили до всех карт-гипотез. Каждая заметная смена закона идёт в журнал как гипотеза с
    проверкой и выводом; мелкие уточнения — как наблюдения.
    """

    LEAVE = 8.0             # перевес (в логарифмах) над допущением, при котором агент от него отказывается
    BACK = 3.0              # перевес меньше этого — агент возвращается к допущению
    SWITCH = 3.0            # перевес, при котором одна гипотеза «не допущение» сменяет другую
    MIN_SIGNAL = 15         # сколько показаний с сигналом нужно до первой смены
    ZOOM_SIGNAL = 40        # и сколько — чтобы сужать сетку
    ZOOM_EVERY = 15.0       # не сужать чаще, с
    NEW = (0.08, 0.20)      # смена дальности или показателя больше этой доли — новая гипотеза, меньше — уточнение
    LOG = 300               # сколько последних показаний до сбора идёт в сверку по парам

    def __init__(self, agent):
        a, r = agent, agent.rules
        self.a = a
        self.bank = LawBank(a.arena, a.n_samples, (r.sensor_range_m, 1.0))
        self.law = self.bank.law
        self.told = self.law                # закон, названный в открытой гипотезе журнала
        self.source = 'assumed'             # assumed — допущение; maps — выбран сравнением карт
        a.belief = self.bank.belief
        self.log = []                       # показания с прошлого сбора: (x, y, z)
        self.sites = []                     # собранные образцы и показания до них
        self.pairs = None                   # последняя сверка по парам
        self._n = 0
        self._key = None
        self._zoom_t = 0.0
        self.meter = DrainMeter(r.drain_per_m)
        self.per_m = r.drain_per_m          # рабочая (осторожная) цена метра: по ней считаются дорога и запас
        self.base = None                    # расход обычного пола, когда он выделен (см. DrainMeter.verdict)
        self._drain_told = False
        self._drain_closed = False

    # --- карта образцов -----------------------------------------------------------------------

    def do(self, method, *args):
        """Правка карты (clear_disc, relax): во всех картах-гипотезах."""
        self.bank.apply(method, *args)

    def reading(self, obs, z, sigma):
        a = self.a
        if self._key is None:
            self._open(obs.t, 'допущение команды, по показаниям ещё не проверено')
            a.journal.open(obs.t, 'cal:drain', f'метр обычного пола стоит {self.per_m:.2f} ед. заряда (допущение команды)',
                           'мерить расход по пройденным отрезкам; вывод — когда обычный пол выделен: самые дешёвые метры '
                           'сходятся с допущением, дешевле его либо тянутся дальше любого пятна грунта')
        if self._faulty():
            # Сбой датчика — не другой закон: карты показание учитывают, но ни в сравнение гипотез,
            # ни в сверку по парам оно не идёт.
            self.bank.apply('unscored', obs.x, obs.y, z, sigma)
            return
        self.log.append((obs.x, obs.y, z))
        self.bank.apply('update', obs.x, obs.y, z, sigma)
        if self.bank.n_signal >= self.MIN_SIGNAL:
            self._choose(obs.t)
            if obs.t - self._zoom_t >= self.ZOOM_EVERY and self.bank.n_signal >= self.ZOOM_SIGNAL \
                    and self.bank.posterior().max() >= 0.95:
                self._zoom(obs.t)

    def _faulty(self):
        """Датчик сейчас неисправен по собственным признакам агента: шум вырос либо сторож пережидает сбой."""
        a = self.a
        return (a.cfg.sensor_health and a.health.degraded) or (a.guard is not None and a.guard.wait is not None)

    def _choose(self, t):
        """Какая гипотеза рабочая. Допущение — нулевая гипотеза: уходим от него только по явному перевесу."""
        b = self.bank
        i, lead, over_null = b.lead()
        if b.current == b.null:
            if over_null < self.LEAVE:
                return
        elif over_null <= self.BACK:
            i, lead = b.null, 0.0
        elif lead < self.SWITCH:
            return
        if i != b.current:
            b.current = i
            self._adopt(t, f'перевес над допущением в логарифмах {over_null:.1f}' if i != b.null
                        else f'перевес других гипотез над допущением упал до {over_null:.1f}')

    def missed(self, obs):
        """Ложный сбор — тоже измерение: карты, которые были уверены в образце здесь, теряют очки."""
        self.bank.apply('missed', obs.x, obs.y, self.a.rules.collect_radius_m)

    def collected(self, obs):
        a = self.a
        self.bank.apply('collected', obs.x, obs.y, a.rules.collect_radius_m)
        self.sites.append({'x': obs.x, 'y': obs.y, 'pts': np.array(self.log[-self.LOG:]).reshape(-1, 3)})
        self.log = []
        if self.bank.n_signal >= self.ZOOM_SIGNAL:
            self._zoom(obs.t)
        self._verify(obs.t)

    def _zoom(self, t):
        """Сузить сетку гипотез вокруг лучшей: уточнение дальности и показателя."""
        self._zoom_t = t
        bank = self.bank.zoom()
        if bank is None:
            return
        if self.bank.current == self.bank.null:
            bank.current = bank.null                # допущение не опровергнуто: сетка сужается «про запас»
            self.bank = bank
            self.a.belief = bank.belief
            return
        best = int(bank.score.argmax())
        bank.current = best if bank.score[best] - bank.score[bank.null] > self.BACK else bank.null
        self.bank = bank
        self._adopt(t, 'сетка гипотез сужена вокруг лучшей, все наблюдения проиграны заново')

    def _adopt(self, t, why):
        """Рабочим стал другой закон: сменить карту агента и записать в журнал."""
        a = self.a
        law = self.bank.law
        a.belief = self.bank.belief
        a._cands_t = -1e9
        if law == self.law:
            return
        self.law, self.source = law, 'maps'
        a.journal.add(t, 'observe', f'Калибровка датчика: {law_text(*law)} — {why}', tag='calibration', what='law',
                      source='maps', range=round(law[0], 3), power=round(law[1], 3),
                      grid=[round(s, 3) for s in self.bank.steps])
        if abs(law[0] / self.told[0] - 1.0) > self.NEW[0] or abs(law[1] / self.told[1] - 1.0) > self.NEW[1]:
            a.journal.close(t, self._key, 'refuted', f'лучше показания объясняет закон «{law_text(*law)}»: {why}')
            self._open(t, f'лучшая из {len(self.bank.laws)} гипотез по {len(self.bank.ops)} наблюдениям')
        a._request_plan('calibration')

    def _open(self, t, basis):
        self._n += 1
        self._key = f'law:cal:{self._n}'
        self.told = self.law
        self.a.journal.open(t, self._key, f'{law_text(*self.law)} ({basis})',
                            'карты с другими дальностями и формами закона получают те же показания: сравниваю, чья '
                            'лучше их предсказывает; после сбора сверяю показания на подъезде с расстоянием до '
                            'найденного образца', range=round(self.law[0], 3), power=round(self.law[1], 3),
                            source=self.source)

    def _verify(self, t):
        """После сбора: сходятся ли пары «расстояние — показание» с рабочим законом."""
        a = self.a
        fit = fit_law(self.sites[-3:], a.health.nominal, start=self.law, radius=a.rules.collect_radius_m)
        if not fit or fit['n'] < 20 or fit['d_max'] - fit['d_min'] < 0.5:
            return
        self.pairs = fit
        tol_r = max(2.5 * fit['se_range'], 0.10 * fit['range'])
        tol_p = max(2.5 * fit['se_power'], 0.20 * fit['power'])
        agree = abs(fit['range'] - self.law[0]) <= tol_r and abs(fit['power'] - self.law[1]) <= tol_p
        text = (f"по {fit['n']} парам «расстояние — показание» у {fit['sites']} собранных образцов "
                f"(от {fit['d_min']:.2f} до {fit['d_max']:.2f} м) дальность {fit['range']:.2f} ± {fit['se_range']:.2f} м, "
                f"показатель степени {fit['power']:.2f} ± {fit['se_power']:.2f}")
        a.journal.add(t, 'observe', f'Сверка датчика по парам: {text}; рабочий закон — R = {self.law[0]:.2f} м, '
                      f'p = {self.law[1]:.2f}: ' + ('сходится' if agree else 'расходится, продолжаю сравнение карт'),
                      tag='calibration', what='pairs', agree=bool(agree), range=round(fit['range'], 3),
                      power=round(fit['power'], 3), se_range=round(fit['se_range'], 3),
                      se_power=round(fit['se_power'], 3), n=fit['n'])
        if agree and a.journal.close(t, self._key, 'confirmed', f'{text} — сходится с рабочим законом'):
            self.source = 'confirmed'
            a.journal.add(t, 'observe', f'Калибровка датчика: {law_text(*self.law)} — подтверждено парами',
                          tag='calibration', what='law', source='pairs', range=round(self.law[0], 3),
                          power=round(self.law[1], 3))
            self._open(t, 'подтверждено парами у собранного образца, уточняется дальше')

    # --- расход -------------------------------------------------------------------------------

    def segment(self, ds, spent, t, x=None, y=None):
        """Отрезок пути без простоя и скачков, с концом в (x, y). Уточняет цену метра.

        Рабочая цена метра (per_m) следует нижней оценке сразу: считать дорогу домой дороже, чем она
        окажется, безопасно. Гипотеза о расходе обычного пола закрывается отдельно — когда пол выделен.
        """
        a = self.a
        m = self.meter
        v = m.add(ds, spent, x, y)
        if v is None or v <= 0.0:
            return
        nominal = m.nominal
        verdict = m.verdict()
        if not self._drain_told or abs(v / self.per_m - 1.0) >= 0.03:
            old, self.per_m = self.per_m, v
            a.soil.drain *= old / v             # множители грунта считались от прежней цены метра
            a.soil.per_m = v
            a.soil.version += 1
            a.soil._cache = None
            a._cost_dirty = True
            a._base_dist = None
            self._drain_told = True
            known = 'обычный пол' if verdict else 'не дороже этого стоит обычный пол; весь путь мог идти по грунту'
            a.journal.add(t, 'observe', f'Калибровка расхода: самые дешёвые метры пути стоят {v:.2f} ± {m.low_se:.2f} ед. '
                          f'(допущение {nominal:.2f}), по {m.path:.1f} м пути — {known}; дорогу и запас на возврат '
                          f'считаю по {v:.2f}', tag='calibration', what='drain', per_m=round(v, 3),
                          se=round(m.low_se, 3), floor=bool(verdict))
        if verdict and not self._drain_closed:
            self._drain_closed = True
            self.base = v
            why = (f'на {m.plateau_path:.1f} м пути метр стоит {v:.2f} ± {m.low_se:.2f} ед.'
                   + ('' if verdict == 'confirmed' else
                      ', а грунт расход только повышает' if v < nominal else
                      f' на участке размахом {m.plateau_span:.1f} м — пятно грунта таким не бывает'))
            a.journal.close(t, 'cal:drain', verdict, why)
            a.journal.add(t, 'observe', f'Калибровка расхода: обычный пол — {v:.2f} ед. на метр ({why})',
                          tag='calibration', what='floor', per_m=round(v, 3), status=verdict)
        elif verdict:
            self.base = v
