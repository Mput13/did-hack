"""Проверка закона датчика образцов: как показание зависит от расстояния. Часть исполнителя исследования.

Образец сначала нужно найти — это делает штатный поиск агента (карта вероятностей, подъезд к месту),
только вместо сбора робот останавливается: образец нужен как источник сигнала. Дальше идёт сравнение
объяснений (did.science.Inquiry).

1. Центрирование. Робот слушает датчик в четырёх точках «креста» вокруг найденного места. Любой
   закон, убывающий с расстоянием, симметричен относительно образца, поэтому перекос показаний
   между противоположными точками показывает, куда и насколько сдвинуть оценку места. Форма закона
   для этого не нужна. Среднее по кресту — заодно первый опыт: показание на малом расстоянии.
2. Опыты. Встать на заданном расстоянии от образца с двух противоположных сторон и послушать:
   среднее двух точек почти не зависит от остаточной ошибки места. Какое расстояние брать
   следующим, решает ожидаемая польза на единицу заряда. Расстояния, на которых к точке, скорее
   всего, ближе другой образец (датчик покажет его), не берутся.
3. Контроль — опорная точка у самого образца: все объяснения ждут там одного и того же показания,
   поэтому она ничего не различает, зато повтор в конце показывает, не сбился ли датчик.
"""
import math

import numpy as np

from .inquiry import FAULT_S
from .science import OTHER, Alternative, Inquiry, TestOption
from .study import DEFAULT_LAWS, LAWS, TRAVEL_V, Hypothesis

LOC_PAIR_M = 0.03       # ошибка расстояния, когда слушаем с двух сторон (ошибка места образца сокращается)
LOC_SINGLE_M = 0.08     # и когда точка одна
MODEL_SLACK = 0.02      # на сколько закон может отличаться от идеальной формулы
REF_GOOD = 0.9          # показание у образца не ниже этого — место найдено точно
DRIFT = 0.08            # на столько изменилось показание в опорной точке — датчик сбоит
RISK_MAX = 0.3          # с такой вероятности «к точке ближе другой образец» опыт на этом расстоянии не ставится
MIN_TESTS = 2           # по одному расстоянию вывод о законе не делается, если заряд позволяет
MAX_CROSSES = 3         # сколько раз можно уточнять место образца
HOME_MARGIN = 1.15


def _clipped(mu, sigma):
    """Среднее показания, обрезанного датчиком в 0…1, если без обрезки оно mu при шуме sigma."""
    def cdf(v):
        return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))

    def pdf(v):
        return math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)

    a, b = -mu / sigma, (1.0 - mu) / sigma
    return (1.0 - cdf(b)) + mu * (cdf(b) - cdf(a)) + sigma * (pdf(a) - pdf(b))


class LawMixin:

    def _law_init(self):
        self._sample = None            # где образец: (x, y)
        self._law = None               # идущий опыт: {'test', 'todo': [точки], 'got': [(расстояние, показание, ±, x, y, запись)]}
        self._law_stop = 'no_tests'
        self._ref = []                 # показания в опорной точке у образца: [(t, показание, ±)]
        self._stations = []            # все прослушанные точки: (расстояние, показание, ±)
        self._crosses = 0              # сколько раз место образца уточнялось крестом
        self._quiet_until = -1e9       # после штрафа датчик может сбоить: показаниям пока не верю
        self._retries = 0
        self._hyps = self.spec.hypotheses or [Hypothesis(**h) for h in DEFAULT_LAWS]
        if self.site:                  # область задана: искать образец начинаю с неё
            self.queue = [{'type': 'goto', 'x': self.site['x'], 'y': self.site['y']}]
            self._trigger = None

    def _law_penalty(self, obs):
        if self.q == 'sensor_law':
            self._quiet_until = obs.t + FAULT_S + 5.0

    # --- образец найден -----------------------------------------------------------------------

    def _try_collect(self, sg, obs, io, confidence):
        """Штатный поиск довёл до образца. Собирать его нельзя: он источник сигнала для опытов."""
        if self.phase != 'search':
            return super()._try_collect(sg, obs, io, confidence)
        self._command(io, 0.0, 0.0)
        if abs(obs.v) > 0.03:
            return
        b = self.belief                # центр тяжести вероятности вокруг пика точнее, чем клетка 10 см
        near = np.hypot(b.cx - sg['x'], b.cy - sg['y']) <= 0.3
        w = b.p[near]
        self._sample = (round(float(w @ b.cx[near] / w.sum()), 3), round(float(w @ b.cy[near] / w.sum()), 3))
        self.phase, self.queue, self._trigger, self._path_goal = 'study', [], None, None
        self.journal.close(obs.t, sg.get('key'), 'confirmed', 'образец найден, оставлен на месте как источник сигнала')
        self.journal.add(obs.t, 'decision', f'Образец найден около ({self._sample[0]:.2f}; {self._sample[1]:.2f}), уверенность '
                         f'{confidence:.0%}. Не собираю его: он нужен как источник сигнала для проверки датчика')
        self._open_law(obs)

    def _law_pred(self, d, n):
        """Что каждое объяснение предсказывает для среднего показания на расстоянии d (n — сколько точек)."""
        loc = LOC_PAIR_M if n >= 2 else LOC_SINGLE_M
        sigma = self.rules.sensor_sigma
        out = {}
        for h in self._hyps:
            def f(x, h=h):
                return _clipped(LAWS[h.law](x, h.range_m), sigma)
            slope = abs(f(d + 0.05) - f(max(d - 0.05, 0.0))) / 0.1
            out[h.id] = (f(d), math.hypot(MODEL_SLACK, slope * loc))
        return out

    def _listen_sigma(self, n=1):
        k = max(2.0, self.spec.allowed.pause.seconds * self.rules.sensor_hz)
        return self.rules.sensor_sigma / math.sqrt(k * n)

    def _open_law(self, obs):
        spec, hyps = self.spec, self._hyps
        step = spec.allowed.straight.length_m / 2      # расстояния через полпробега: ближние точки самые надёжные
        alts = [Alternative(h.id, h.statement, h.prior) for h in hyps]
        alts.append(Alternative(OTHER, 'зависимость не из этого списка', 0.1 * sum(h.prior for h in hyps)))
        dist, _ = self.graph.field(obs.x, obs.y)
        tests = []
        for k in range(2, 12):                         # первое расстояние (k = 1) проверяет крест центрирования
            d = k * step
            if d > 2.6:
                break
            pts, risk = self._ring(d, dist)
            if pts:
                tests.append(TestOption(f'd{k}', f'слушать датчик в {d:.2f} м от образца' + (' с двух сторон' if len(pts) == 2 else ''),
                                        cost=1.0, duration_s=10.0, unit='показание', predictions=self._law_pred(d, len(pts)),
                                        action={'kind': 'listen', 'd': d, 'stations': pts, 'risk': risk},
                                        sigma=self._listen_sigma(len(pts))))
        self.inq = Inquiry('S1', obs.t, 'study',
                           {'text': 'задание: проверить, как показание датчика образцов зависит от расстояния',
                            'x': self._sample[0], 'y': self._sample[1], 'observed': 0.0, 'expected': 0.0, 'unit': 'показание'},
                           alts, tests, accept=spec.stop.confidence, max_tests=max(2, (spec.stop.max_measurements - 6) // 2 + 1),
                           energy_budget=1e9, min_gain=0.03, source='study')
        text = '; '.join(f'«{a.statement}» {a.prior:.0%}' for a in self.inq.alternatives)
        far = [t.action['d'] for t in tests if t.action['risk'] > RISK_MAX]
        self.journal.add(obs.t, 'decision', f'Сравниваю объяснения: {text}. Опыты — слушать датчик на разных расстояниях от образца'
                         + (f'. Дальше {min(far) - step:.2f} м не отъезжаю: там к точке, скорее всего, ближе другой образец, '
                            'и датчик покажет его' if far else ''))

    def _free(self, p, dist):
        a = self.arena
        ix, iy = a.w2g(*p)
        return (a.inside(ix, iy) and self.graph.ok[iy, ix] and self._risk[iy, ix] < 0.05
                and math.isfinite(self.graph.cost_at(dist, *p)))

    def _other_risk(self, p, d):
        """Вероятность, что к точке p ближе, чем наш образец, лежит другой: тогда датчик покажет его."""
        b = self.belief
        near = ((np.hypot(b.cx - p[0], b.cy - p[1]) < d + 0.05)
                & (np.hypot(b.cx - self._sample[0], b.cy - self._sample[1]) > 0.35))
        return float(min(1.0, b.p[near].sum()))

    def _ring(self, d, dist):
        """Точки на расстоянии d от образца, куда можно доехать, и риск, что там слышен другой образец.

        Лучше пара с противоположных сторон; из пар — та, где другой образец наименее вероятен.
        """
        sx, sy = self._sample
        pairs, singles = [], []
        for k in range(16):
            u = (math.cos(k * math.pi / 8), math.sin(k * math.pi / 8))
            p, q = (sx + d * u[0], sy + d * u[1]), (sx - d * u[0], sy - d * u[1])
            if not self._free(p, dist):
                continue
            cost, risk = self.graph.cost_at(dist, *p), self._other_risk(p, d)
            singles.append((round(risk, 1), cost, risk, [p]))
            if self._free(q, dist):
                both = max(risk, self._other_risk(q, d))
                pairs.append((round(both, 1), cost + 1.25 * 2 * d, both, [p, q]))
        best = min(pairs or singles, default=None, key=lambda v: v[:2])
        return ([tuple(round(v, 3) for v in p) for p in best[3]], round(best[2], 3)) if best else ([], 1.0)

    # --- центрирование ------------------------------------------------------------------------

    def _start_cross(self, obs):
        """Крест из четырёх точек вокруг оценки места образца. False — поставить его негде."""
        rho = self.spec.allowed.straight.length_m / 2
        sx, sy = self._sample
        dist, _ = self.graph.field(obs.x, obs.y)

        def ring(r, angles):
            return [(round(sx + r * math.cos(a), 3), round(sy + r * math.sin(a), 3)) for a in angles]

        pts = r = None
        for r in (rho, 0.8 * rho):                     # сначала ровный крест: при каком повороте свободны все четыре точки
            for k in range(4):
                cand = ring(r, [k * math.pi / 8 + j * math.pi / 2 for j in range(4)])
                if all(self._free(p, dist) for p in cand):
                    pts = cand
                    break
            if pts:
                break
        if not pts:                                    # образец у стены: беру свободные точки кольца, лишь бы не на одной прямой
            for r in (rho, 0.8 * rho):
                free = [k for k in range(8) if self._free(ring(r, [k * math.pi / 4])[0], dist)]
                if len(free) >= 3 and max(min((a - b) % 8, (b - a) % 8) for a in free for b in free) >= 2:
                    pts = ring(r, [k * math.pi / 4 for k in free[:5]])
                    break
        if not pts:
            return False
        self._crosses += 1
        test = TestOption(f'c{self._crosses}', f'{len(pts)} точки вокруг образца в {r:.2f} м от него', cost=len(pts) * r * self._per_m(),
                          duration_s=30.0, unit='показание', predictions=self._law_pred(r, 2),
                          action={'kind': 'listen', 'd': r, 'stations': pts, 'risk': 0.0}, sigma=self._listen_sigma(len(pts)))
        test.gain_bits = round(self.inq.gain(test), 3)
        self.inq.tests.insert(self._crosses - 1, test)
        self._law = {'test': test, 'todo': list(pts), 'got': [], 'cross': True}
        self.journal.add(obs.t, 'decision', f'Уточняю место образца: слушаю датчик в {len(pts)} точках вокруг него, в {r:.2f} м. '
                         'Перекос показаний между точками покажет, куда сдвинуть оценку; среднее по ним — первый опыт '
                         'для сравнения объяснений', inquiry=self.inq.id)
        return True

    def _center(self, run, obs):
        """Сдвинуть оценку места образца по перекосу показаний в кресте. Возвращает (точки с новыми расстояниями, сдвиг)."""
        got = run['got']
        if len(got) < 3:
            return got, 0.0
        sx, sy = self._sample
        A = np.array([[g[3] - sx, g[4] - sy, 1.0] for g in got])
        (gx, gy, ring), *_ = np.linalg.lstsq(A, np.array([g[1] for g in got]), rcond=None)
        rho = float(np.mean([g[0] for g in got]))
        grad = math.hypot(gx, gy)
        # Конус 1 − s·|p − c|: наклон плоскости по кресту равен s·δ/ρ, центр выше кольца на s·(ρ − δ). Отсюда δ без s.
        gain = grad * rho / max(self._ref[-1][1] - ring, 0.02)
        shift = rho * gain / (1.0 + gain)
        if grad > 1e-9 and shift >= 0.02:
            self._sample = (round(sx + shift * gx / grad, 3), round(sy + shift * gy / grad, 3))
            self.journal.add(obs.t, 'observe', f'Показания в кресте перекошены: сдвигаю оценку места образца на {shift * 100:.0f} см, '
                             f'в ({self._sample[0]:.2f}; {self._sample[1]:.2f})')
            self._ref = []                             # опорную точку нужно прослушать заново, уже в новом месте
        return [(math.dist((g[3], g[4]), self._sample), *g[1:]) for g in got], shift / rho

    # --- ход опытов ---------------------------------------------------------------------------

    def _listen_step(self, obs, point, role, site):
        seconds = self.spec.allowed.pause.seconds
        return self._priced({'kind': 'pause', 'role': role, 'site': site, 'start': tuple(point), 'win': seconds / 3,
                             'hold': seconds / 3, 'listen': True}, obs)

    def _next_law(self, obs):
        if self.inq is None:
            return self._conclude(obs, 'not_found')
        if obs.t < self._quiet_until:              # после штрафа или сбоя датчика: стою и жду
            return self._pause_step(obs, None, 'check', 3.0)
        run = self._law
        if run and run['todo']:
            step = self._listen_step(obs, run['todo'].pop(0), 'test', run['test'].id)
            return step if step is not None else self._next_law(obs)     # до точки не доехать — обхожусь остальными
        if run and run['got']:
            self._law_record(obs)
        elif run:                                  # ни одной точки этого опыта прослушать не удалось
            run['test'].action['dead'] = True
            self._law = None
        if not self._ref:
            return self._listen_step(obs, self._sample, 'control', 'R') or self._conclude(obs, 'no_site')
        if len(self._ref) == 1 and (not self._crosses or self._ref[0][1] < REF_GOOD) and self._crosses < MAX_CROSSES:
            if self._start_cross(obs):
                return self._next_law(obs)
            self._crosses = MAX_CROSSES
            self._warn('nocross', 'уточнить место образца не удалось: вокруг него нет трёх свободных точек для замеров')
        if self._ref[0][1] < REF_GOOD:
            self.failures.append(f'в найденном месте датчик показывает только {self._ref[0][1]:.2f}, а у самого образца должно '
                                 'быть около единицы: место найдено неточно, отмерять расстояния не от чего')
            return self._conclude(obs, 'not_found')
        test = self._choose_law(obs)
        if test is not None:
            self._law = {'test': test, 'todo': list(test.action['stations']), 'got': []}
            exp = '; '.join(f"«{a.statement}» — {test.predictions[a.id][0]:.2f}" for a in self.inq.alternatives
                            if a.id in test.predictions)
            text = (f'Опыт: {test.name} (польза {test.gain_bits:.2f} бит, цена около {test.cost:.1f} ед.). '
                    f'Ожидаю показание: {exp}')
            self.journal.add(obs.t, 'decision', text, inquiry=self.inq.id)
            if self.rec:
                self.rec.add_plan(obs.t, 'rule', 'subgoal_done', text,
                                  [{'type': 'goto', 'x': p[0], 'y': p[1]} for p in test.action['stations']])
            return self._next_law(obs)
        if len(self._ref) < 2 and self.inq.order:  # в конце — повтор опорной точки: не сбился ли датчик
            step = self._listen_step(obs, self._sample, 'control', 'R')
            if step is not None and self._fits(obs, step) is None:
                return step
            self.failures.append('на повтор опорного замера у образца не хватило заряда или времени: '
                                 'не проверено, не сбился ли датчик за время опытов')
        return self._conclude(obs, self._law_stop)

    def _choose_law(self, obs):
        """Следующее расстояние: наибольшая ожидаемая польза на единицу заряда среди тех, что по карману."""
        q, b = self.inq, self.spec.budget
        accept = q.accept
        if q.settled:
            if len(q.order) >= MIN_TESTS:
                self._law_stop = 'settled'
                return None
            q.accept = 2.0                         # вывод по одному расстоянию — не вывод: нужен ещё хотя бы один опыт
        dist, _ = self.graph.field(obs.x, obs.y)
        left = b.energy - self.spent(obs)
        limit = min(self.spec.stop.time_s, self.rules.time_limit_s - 8.0)
        blocked, hidden = set(), []
        for t in q.tests:
            if t.measured is not None:
                continue
            pts, risk = ([], 1.0) if t.action.get('dead') else self._ring(t.action['d'], dist)
            t.action['risk'] = risk
            if not pts or risk > RISK_MAX:
                if pts:
                    blocked.add('other')
                hidden.append(t)
                continue
            road = self.graph.cost_at(dist, *pts[0]) + (1.25 * math.dist(pts[0], pts[1]) if len(pts) == 2 else 0.0)
            t.action['stations'] = pts
            t.cost = road * self._per_m() + 0.05
            t.duration_s = round(road / TRAVEL_V + len(pts) * (self.spec.allowed.pause.seconds + 2.0), 1)
            t.sigma = self._listen_sigma(len(pts))
            t.predictions = self._law_pred(t.action['d'], len(pts))
            if t.cost + self._home_cost(*pts[-1]) * HOME_MARGIN + b.reserve > left:
                blocked.add('budget')
                hidden.append(t)
            elif obs.t + t.duration_s + self._home_time(*pts[-1]) > limit:
                blocked.add('time')
                hidden.append(t)
        for t in hidden:                           # недоступные опыты из выбора убираются
            t.measured = {}
        test = q.choose()
        q.accept = accept
        for t in hidden:
            t.measured = None
        if test is None:
            self._law_stop = ('settled' if q.settled else 'max_measurements' if len(q.order) >= q.max_tests else
                              'budget' if 'budget' in blocked else 'time' if 'time' in blocked else 'no_tests')
            if 'other' in blocked and not q.settled:
                self._warn('far', 'на дальних расстояниях к точке замера, скорее всего, ближе другой образец, и датчик показал бы '
                                  'его: проверить закон там не удалось')
        return test

    def _account_listen(self, rec, m, obs):
        zs = np.array(m['zs'], dtype=float)
        nominal = self.rules.sensor_sigma
        rec.update(unit='показание')
        sd = float(zs.std(ddof=1)) if len(zs) > 1 else 0.0
        stuck = len(zs) >= 4 and float(np.ptp(zs)) < 1e-9 and 0.0 < zs[0] < 1.0
        fault = ('датчик залип: показания не меняются' if stuck else 'датчик шумит втрое сильнее обычного' if sd > 3.0 * nominal
                 else 'мало показаний датчика' if len(zs) < 4 else None)
        if fault:
            rec.update(used=False, note=fault)
            self._retries += 1
            self.journal.add(obs.t, 'alarm', f'Замер у датчика не годится — {fault}. Жду 8 с и повторяю')
            if self._retries > 6:
                self.failures.append('датчик образцов сбоил дольше минуты: часть опытов провести не удалось')
                self._law = None
                return self._conclude(obs, 'no_tests')
            self._quiet_until = obs.t + 8.0
            if rec['role'] == 'test' and self._law:
                self._law['todo'].insert(0, (m['start'][0], m['start'][1]))
            return
        z, se = float(zs.mean()), max(sd, 0.6 * nominal) / math.sqrt(len(zs))
        rec.update(value=z, sigma=se, chart=True)
        if rec['role'] == 'control':
            self._ref.append((obs.t, z, se))
            rec['d'] = 0.0
            if len(self._ref) == 1:
                self.journal.add(obs.t, 'observe', f'Опорная точка у образца: показание {z:.2f} ± {se:.2f}. Все объяснения ждут здесь '
                                 'почти единицу, поэтому она нужна не для выбора, а для контроля датчика и места')
            return
        self._law['got'].append((math.dist((obs.x, obs.y), self._sample), z, se, obs.x, obs.y, rec))

    def _law_record(self, obs):
        run, self._law = self._law, None
        test = run['test']
        got, shift = self._center(run, obs) if run.get('cross') else (run['got'], 0.0)
        for g in got:                                  # расстояния — от уточнённого места образца
            g[5]['d'] = round(g[0], 3)
        if shift > 0.5:
            # Место было найдено грубо: расстояния этих точек ненадёжны, в сравнение объяснений они не идут.
            for g in got:
                g[5].update(used=False, note='только для уточнения места образца')
            test.action['dead'] = True
            return
        self._stations += [g[:3] for g in got]
        if len(got) == 2 and abs(got[0][1] - got[1][1]) > 0.12 + 3.0 * math.hypot(got[0][2], got[1][2]):
            # Показания с двух сторон разошлись: одну точку «слышит» другой образец, он ближе. Верю меньшему.
            d, z, se = min(got, key=lambda g: g[1])[:3]
            n = 1
            self._warn('other', 'на части расстояний одна из двух точек оказалась ближе к другому образцу: там взято '
                                'меньшее из двух показаний')
        else:
            d = float(np.mean([g[0] for g in got]))
            z = float(np.mean([g[1] for g in got]))
            se = math.sqrt(sum(g[2] ** 2 for g in got)) / len(got)
            n = len(got)
        test.predictions = self._law_pred(d, n)
        self.inq.record(test.id, z, se, obs.t)
        post = '; '.join(f"«{a.statement}» {self.inq.posterior[a.id]:.0%}" for a in self.inq.alternatives)
        self.journal.add(obs.t, 'observe', f'В {d:.2f} м от образца датчик показывает {z:.3f} ± {se:.3f}'
                         + (f' (среднее {n} точек)' if n > 1 else '') + f'. Теперь: {post}', inquiry=self.inq.id)

    # --- итог ---------------------------------------------------------------------------------

    def _law_estimate(self):
        """Дальность датчика в предположении линейного закона: подгонка 1 − d/R по прослушанным точкам."""
        pts = [(d, z, math.hypot(se, 0.5 * LOC_PAIR_M)) for d, z, se in getattr(self, '_stations', []) if z > 0.1 and d > 0.15]
        if len(pts) < 2:
            return None
        d, z, s = (np.array(v) for v in zip(*pts))
        w = 1.0 / s ** 2
        slope = float((w * d * (1.0 - z)).sum() / (w * d * d).sum())
        if slope <= 1e-3:
            return None
        r, sr = 1.0 / slope, math.sqrt(1.0 / float((w * d * d).sum())) / slope ** 2
        return {'value': r, 'sigma': sr, 'ci95': [r - 1.96 * sr, r + 1.96 * sr], 'rel': 1.96 * sr / r, 'spread': 1.0,
                'unit': 'м', 'label': 'Дальность датчика, если закон линейный'}

    def _law_close(self, obs, reason):
        q = self.inq
        if q is None or not q.order:
            self.status = 'failed'
            self.conclusion = ('Проверить датчик не удалось: место образца определить достаточно точно не получилось.'
                               if q is not None and reason == 'not_found' else
                               'Проверить датчик не удалось: образец, от которого можно отмерять расстояния, не найден '
                               'в пределах бюджета и времени.')
            self.inq = None
            return
        c = q.close(obs.t, 'закон датчика внесён в отчёт')
        if reason == 'not_found':                      # место образца так и не найдено точно: расстояниям верить нельзя
            c.update(status='insufficient', text='место образца определить достаточно точно не получилось, поэтому '
                                                 'расстояния до него ненадёжны и вывод о законе сделать нельзя')
            self.status, self.conclusion = 'failed', c['text'][0].upper() + c['text'][1:] + '.'
            return
        if len(self._ref) == 2 and abs(self._ref[1][1] - self._ref[0][1]) > DRIFT:
            a, b = self._ref[0][1], self._ref[1][1]
            c.update(status='insufficient', text=f'показание в опорной точке у образца за время опытов изменилось с {a:.2f} до '
                                                 f'{b:.2f} — датчик сбоит, и выводу верить нельзя. Без этого было бы: {c["text"]}')
            self.failures.append('датчик образцов изменил показания в опорной точке: вывод о законе ненадёжен')
        self.status = 'done' if c['status'] == 'identified' else 'not_reached'
        est = self._law_estimate()
        self.conclusion = c['text'][0].upper() + c['text'][1:] + '.'
        if est:
            self.conclusion += (f" Если закон линейный, показание падает до нуля на {est['value']:.2f} ± "
                                f"{1.96 * est['sigma']:.2f} м.")

    def _law_control(self):
        if not getattr(self, '_ref', None):
            return None
        a = self._ref[0]
        out = {'unit': 'показание', 'control': {'value': round(a[1], 4), 'sigma': round(a[2], 4)}}
        if len(self._ref) == 2:
            b = self._ref[1]
            same = abs(b[1] - a[1]) <= DRIFT
            out['repeat'] = {'value': round(b[1], 4), 'sigma': round(b[2], 4)}
            out['text'] = (f'В опорной точке у образца датчик показал {a[1]:.2f} в начале опытов и {b[1]:.2f} в конце: '
                           + ('за это время он не сбился.' if same else 'показания уехали, датчику верить нельзя.'))
        else:
            out['text'] = (f'В опорной точке у образца датчик показал {a[1]:.2f}. Повторить этот замер в конце не удалось, '
                           'поэтому не проверено, не сбился ли датчик за время опытов.')
        return out
