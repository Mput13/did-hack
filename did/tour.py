"""План на весь остаток прогона: в каком порядке объехать цели, чтобы собрать больше и вернуться.

Правило HeuristicPlanner жадное: «лучшая цель сейчас» по отношению уверенность / цена пути, разведка —
только когда кандидатов нет, возврат — когда заряд опустился до запаса. Здесь решается задача целиком
(задача ориентирования): из всех текущих кандидатов и точек разведки выбирается подмножество и порядок
обхода с наибольшим ожидаемым счётом при условии, что заряда хватает на весь объезд и дорогу домой с
запасом. Целей не больше десяти, поэтому перебор точный — динамическое программирование по подмножествам.
План пересчитывается после каждого нового сведения (сбор, промах, новый кандидат, штраф, смена грунта),
исполняется всегда только его начало. Возврат на базу — это конец плана, а не отдельный порог: пока в
плане есть цель по заряду, робот к ней едет, даже если «по дороге домой». Объезд сравнивается с немедленным
возвратом: если ни один не прибавляет ожидаемых очков, план — база. Раз в секунду проверяется, что текущая
цель всё ещё по заряду вместе с остановкой, грузом и дорогой домой от неё.

Вторая схема в этом же классе — «сначала разведка, потом сбор» (survey_cover > 0): короткий обзорный
объезд по заранее посчитанным точкам (did/route.py) с попутным сбором того, что оказалось рядом, затем
план на остаток прогона.

Правды сценария схема не видит: на входе та же сводка, что получает планировщик, и карты самого агента.
Времена — в секундах прогона, от такта симулятора ничего не зависит.
"""
import math
from dataclasses import dataclass

import numpy as np

from .route import survey_route


@dataclass(frozen=True)
class Settings:
    max_nodes: int = 10               # целей в переборе; лишние отбрасываются начиная с самых слабых участков разведки
    targets: str = 'tour'             # tour — цели выбирает перебор; rule — правило планировщика, а от схемы только возврат
    back: str = 'plan'                # plan — возврат решает план (попутный сбор по дороге домой); rule — правило агента
    haste: float = 0.0                # ед. заряда: > 0 — образец, до которого ещё столько заряда пути, ценится в e раз меньше
    #                                   («сначала ближнее»); 0 — порядок только по длине объезда
    haste_nodes: int = 7              # при haste > 0 перебираются порядки, а не подмножества: целей не больше стольких
    explore_value: float = 0.5        # какая доля неясной вероятности вокруг точки разведки станет собранными образцами
    explore: str = 'regions'          # разведка: regions — участки арены с неясной вероятностью входят в объезд наравне
    #                                   с кандидатами; rule — только когда кандидатов по заряду нет, точку выбирает правило
    region_m: float = 1.0             # м: сторона участка, по которому суммируется неясная вероятность
    regions: int = 5                  # столько самых «тяжёлых» участков идёт в перебор
    unknown_mult: float = 0.0         # на сколько (доля) дороже считать метр по полу, где робот ещё не ездил
    switch_gain: float = 2.0          # очки: начатую цель план бросает, только если другой порядок лучше на столько
    explore_min: float = 0.03         # участки с меньшей ценностью (в образцах) в план не идут
    dwell: float = 0.3                # ед. заряда на подъезд вплотную, остановку и сбор
    commit_slack: float = 1.0         # ед. заряда: к уже выбранной цели робот едет, пока на неё (с остановкой, грузом,
    #                                   дорогой домой и запасом) не хватает не больше чем столько; проверка — раз в секунду
    true_rates: bool = False          # только оракул: расход на груз и повороты берётся из правил, а не из оценки агента
    turn_rate: float = 1.3            # рад на метр пути (для true_rates)
    # --- «сначала разведка, потом сбор»
    survey_cover: float = 0.0         # м: радиус покрытия обзорного объезда; 0 — без объезда
    survey_budget: float = 0.0        # ед. заряда на объезд, считая от старта; вышел в пути — остаток объезда снимается
    survey_detour: float = 3.0        # ед. заряда: на столько можно отклониться от объезда за найденным образцом
    survey_pick: float = 0.5          # с какой уверенности кандидат подбирается по пути
    survey_found: float = 1.0         # объезд кончается раньше, если уверенных кандидатов уже столько (доля оставшихся образцов)


class TourScheme:

    def __init__(self, agent):
        self.a = agent
        self.s = Settings(**dict(agent.cfg.scheme_opts))
        self.log = []                                   # решения прогона: для проверок и разбора
        self._plan_t = -1e9
        self._mine = self.s.targets == 'tour'           # текущую цель выбрал перебор (а не правило и не объезд)
        self._wps = list(survey_route(agent.arena, self.s.survey_cover)) if self.s.survey_cover > 0 else []
        self._survey = bool(self._wps)

    # --- расход -----------------------------------------------------------------------------------

    def _rates(self):
        """(заряд на метр при нынешнем грузе, добавка на метр за каждый следующий образец)."""
        a, r = self.a, self.a.rules
        if self.s.true_rates:
            base = r.drain_per_m * (1.0 + r.load_drain * a.collected) + r.drain_per_rad * self.s.turn_rate
            return base, r.drain_per_m * r.load_drain
        return a._per_m(), 0.0

    def _reserve(self, obs):
        """(множитель к дороге домой, постоянный запас) — те же, что у правила агента."""
        a = self.a
        extra = a.inv.reserve(obs.t) if a.inv else 0.0
        return a.cfg.reserve_margin, a.cfg.reserve_abs + extra

    # --- цели --------------------------------------------------------------------------------------

    def _nodes(self, obs, state):
        """Цели перебора с дорогой до каждой и от каждой домой, в метрах обычного пола."""
        a, s = self.a, self.s
        nodes = [{'id': c['id'], 'kind': 'investigate', 'x': c['x'], 'y': c['y'], 'value': c['confidence'],
                  'carry': c['confidence']} for c in state['candidates'] if c['feasible']]
        nodes.sort(key=lambda c: -c['value'])
        if s.explore == 'regions':
            nodes += self._regions(state)
        nodes = nodes[:s.max_nodes]
        w = self._weights()
        g = a.graph
        dist, pred = g.field(obs.x, obs.y)
        a._home_cost(obs.x, obs.y)                       # поле «домой» посчитано и лежит в a._base_pred
        out = []
        for nd in nodes:
            k = g.node(nd['x'], nd['y'])
            if math.isfinite(dist[k]) and math.isfinite(a._base_dist[k]):
                nd['to_m'], nd['back_m'] = _walk(g, pred, k, w), _walk(g, a._base_pred, k, w)
                out.append(nd)
        return out, w

    def _regions(self, state):
        """Участки арены, где осталась вероятность образца, ещё не названная кандидатом: куда ехать искать."""
        a, s = self.a, self.s
        b = a.belief
        if not hasattr(b, 'p'):
            return []
        p = b.p.copy()
        for c in state['candidates']:                    # кандидаты входят в план сами
            p[np.hypot(b.cx - c['x'], b.cy - c['y']) <= 0.3] = 0.0
        nx = int(math.ceil(b.w * b.res / s.region_m))
        block = ((b.cy - b.y0) // s.region_m).astype(int) * nx + ((b.cx - b.x0) // s.region_m).astype(int)
        mass = np.bincount(block, weights=p)
        out = []
        for k in np.argsort(-mass)[:s.regions]:
            value = s.explore_value * float(mass[k])
            if value < s.explore_min:
                break
            sel = block == k
            x, y = float(p[sel] @ b.cx[sel] / mass[k]), float(p[sel] @ b.cy[sel] / mass[k])
            ix, iy = a.arena.w2g(x, y)
            if a._risk[iy, ix] >= 0.5:
                continue
            node = a.graph.node(x, y)                    # ближайшая клетка, куда робот может встать
            out.append({'id': f'R{len(out) + 1}', 'kind': 'goto', 'x': round(float(a.graph.xs[node]), 2),
                        'y': round(float(a.graph.ys[node]), 2), 'value': value, 'carry': 0.0})
        return out

    def _weights(self):
        """Множитель расхода по клеткам графа: оценка агента, а на непроверенном полу — с надбавкой."""
        a, s = self.a, self.s
        w = a.graph.mult
        if s.unknown_mult > 0.0 and a.cfg.learn_soil and a._truth is None:
            unknown = 1.0 - a.soil.confidence_grid()[a.graph.iy, a.graph.ix]
            w = w * (1.0 + s.unknown_mult * unknown)
        return w

    def _metres(self, nodes, w):
        """Путь между целями в метрах обычного пола по карте стоимостей агента (симметричная матрица)."""
        g, n = self.a.graph, len(nodes)
        m = np.zeros((n, n))
        for i in range(n - 1):
            _, pred = g.field(nodes[i]['x'], nodes[i]['y'])
            for j in range(i + 1, n):
                m[i, j] = m[j, i] = _walk(g, pred, g.node(nodes[j]['x'], nodes[j]['y']), w)
        return m

    # --- перебор -----------------------------------------------------------------------------------

    def solve(self, nodes, metres, battery, rates, reserve, pts_sample, pts_left, first=None, home=0.0):
        """Лучший объезд: (порядок номеров целей, сводка). first — номер цели, с которой объезд обязан начаться.
        home — заряд на дорогу домой из нынешней точки: с ней сравнивается любой объезд.

        Заряд на объезд: дорога до первой цели, переезды, остановки, дорога домой от последней; груз
        удорожает метр после каждого кандидата. Условие: (объезд + дорога домой) × запас + постоянный запас
        не больше заряда. Среди годных — наибольший ожидаемый счёт: очки за образцы плюс очки за остаток.
        Счёт считается относительно немедленного возвращения: у пустого порядка он равен нулю, и объезд
        выбирается, только если его счёт выше. Пустой порядок: why = 'charge' — ни одна цель не по заряду,
        'gain' — по заряду есть, но ни один объезд не прибавляет очков, 'none' — целей нет.
        """
        n = len(nodes)
        stay = {'value': 0.0, 'cost': home, 'score': 0.0, 'gain': 0.0, 'why': 'none'}
        if n == 0:
            return [], stay
        if self.s.haste > 0.0:
            return self._solve_haste(nodes, metres, battery, rates, reserve, pts_sample, pts_left, first, home)
        per_m, per_load = rates
        margin, absolute = reserve
        full = 1 << n
        bits = (np.arange(full)[:, None] >> np.arange(n)) & 1
        value = bits @ np.array([nd['value'] for nd in nodes])
        pm = per_m + per_load * (bits @ np.array([nd['carry'] for nd in nodes]))     # цена метра после объезда набора
        to_m = np.array([nd['to_m'] for nd in nodes])
        back_m = np.array([nd['back_m'] for nd in nodes])
        dwell = self.s.dwell
        dp = np.full((full, n), np.inf)
        par = np.full((full, n), -1, dtype=np.int16)
        for i in range(n) if first is None else [first]:
            dp[1 << i, i] = to_m[i] * per_m + dwell
        limit = (battery - absolute) / margin
        for mask in range(1, full):
            row = dp[mask]
            if row.min() > limit:
                continue
            ext = row[:, None] + metres * pm[mask] + dwell
            last = ext.argmin(axis=0)
            for nxt in range(n):
                if mask >> nxt & 1:
                    continue
                c = ext[last[nxt], nxt]
                if c < dp[mask | 1 << nxt, nxt]:
                    dp[mask | 1 << nxt, nxt] = c
                    par[mask | 1 << nxt, nxt] = last[nxt]
        back = back_m[None, :] * pm[:, None]
        ok = (dp + back) * margin + absolute <= battery
        if not ok.any():
            return [], {**stay, 'why': 'charge'}
        score = np.where(ok, pts_sample * value[:, None] - pts_left * (dp + back - home), -np.inf)
        mask, last = np.unravel_index(int(score.argmax()), score.shape)
        if score[mask, last] <= 0.0:
            return [], {**stay, 'why': 'gain'}           # немедленный возврат не хуже любого объезда
        info = {'value': float(value[mask]), 'cost': float(dp[mask, last] + back[mask, last]),
                'score': float(score[mask, last]), 'gain': float(score[mask, last]), 'why': ''}
        order = []
        while last >= 0:
            order.append(int(last))
            mask, last = mask & ~(1 << last), int(par[mask, last])
        return order[::-1], info

    def _solve_haste(self, nodes, metres, battery, rates, reserve, pts_sample, pts_left, first, home=0.0):
        """То же с убыванием ценности по ходу объезда: перебор порядков в глубину с отсечением по заряду.

        Датчик слышит только ближайший образец, поэтому новые кандидаты появляются по мере сбора ближних;
        чем позже цель стоит в плане, тем вероятнее, что план до неё успеет измениться. «Спешка» выбирает
        порядок, а не оценивает выгоду: с немедленным возвратом объезд сравнивается по ожидаемому счёту без
        убывания (gain), и среди объездов, которые его прибавляют, берётся лучший по счёту с убыванием.
        """
        n = min(len(nodes), self.s.haste_nodes)
        per_m, per_load = rates
        margin, absolute = reserve
        dwell, tau = self.s.dwell, self.s.haste
        value = [nd['value'] for nd in nodes]
        carry = [nd['carry'] for nd in nodes]
        to_m = [nd['to_m'] for nd in nodes]
        back_m = [nd['back_m'] for nd in nodes]
        m = metres.tolist()
        best = {'score': -math.inf, 'order': [], 'value': 0.0, 'cost': home, 'gain': 0.0}
        order = []
        fits = [False]

        def visit(last, used, cost, gain, total, load):
            pm = per_m + per_load * load
            for nxt in range(n):
                if used >> nxt & 1:
                    continue
                c = cost + (to_m[nxt] if last < 0 else m[last][nxt]) * pm + dwell
                pm2 = pm + per_load * carry[nxt]
                back = back_m[nxt] * pm2
                if (c + back) * margin + absolute > battery:
                    continue
                fits[0] = True
                g = gain + pts_sample * value[nxt] * math.exp(-c / tau)
                order.append(nxt)
                spent = pts_left * (c + back - home)
                score, real = g - spent, pts_sample * (total + value[nxt]) - spent
                if real > 0.0 and score > best['score']:
                    best.update(score=score, order=list(order), value=total + value[nxt], cost=c + back, gain=real)
                visit(nxt, used | 1 << nxt, c, g, total + value[nxt], load + carry[nxt])
                order.pop()

        if first is None:
            visit(-1, 0, 0.0, 0.0, 0.0, 0.0)
        elif first < n:
            c = to_m[first] * per_m + dwell
            back = back_m[first] * (per_m + per_load * carry[first])
            if (c + back) * margin + absolute <= battery:
                fits[0] = True
                g = pts_sample * value[first] * math.exp(-c / tau)
                order.append(first)
                spent = pts_left * (c + back - home)
                if pts_sample * value[first] - spent > 0.0:
                    best.update(score=g - spent, order=[first], value=value[first], cost=c + back,
                                gain=pts_sample * value[first] - spent)
                visit(first, 1 << first, c, g, value[first], carry[first])
        if not best['order']:
            return [], {'value': 0.0, 'cost': home, 'score': 0.0, 'gain': 0.0, 'why': 'gain' if fits[0] else 'charge'}
        return best['order'], {'value': best['value'], 'cost': best['cost'], 'score': best['score'],
                               'gain': best['gain'], 'why': ''}

    # --- связка с агентом --------------------------------------------------------------------------

    def plan(self, obs, state):
        """План в формате планировщика: вся очередь целей до базы. None — пусть решает правило."""
        a = self.a
        left = state['samples']['total'] - state['samples']['collected']
        if left <= 0 or (self.s.targets == 'rule' and not self._survey):
            return None
        self._plan_t = obs.t
        self._visit(obs)
        self._mine = False
        if self._survey:
            plan = self._survey_plan(obs, state)
            if plan is not None or self.s.targets == 'rule':
                return plan
        nodes, w = self._nodes(obs, state)
        if self.s.explore == 'rule' and not nodes:
            return None                                  # кандидатов по заряду нет: разведка или база — по правилу
        self._mine = True
        r = a.rules
        rates = self._rates()
        home = _walk(a.graph, a._base_pred, a.graph.node(obs.x, obs.y), w) * rates[0]
        args = (nodes, self._metres(nodes, w), obs.battery, rates, self._reserve(obs), r.pts_sample,
                r.pts_battery_left)
        order, info = self.solve(*args, home=home)
        # Начатую цель не бросаем из-за мелкой разницы: иначе робот мечется между равноценными порядками.
        sg = a.queue[0] if a.queue else None
        now = next((k for k, nd in enumerate(nodes) if sg is not None and nd['kind'] == sg['type']
                    and math.hypot(nd['x'] - sg['x'], nd['y'] - sg['y']) <= 0.45), None)
        if now is not None and order and order[0] != now:
            kept, kept_info = self.solve(*args, first=now, home=home)
            if kept and info['score'] - kept_info['score'] < self.s.switch_gain:
                order, info = kept, kept_info
        rec = {'t': round(obs.t, 1), 'trigger': state['trigger'], 'battery': round(obs.battery, 1),
               'nodes': [nd['id'] for nd in nodes], 'order': [nodes[i]['id'] for i in order],
               'expected_samples': round(info['value'], 2), 'cost': round(info['cost'], 1)}
        self.log.append(rec)
        if not order:
            why = ('Ни одна цель не окупает заряда на дорогу к ней: ожидаемые очки за образцы меньше очков за '
                   'сбережённый заряд' if info['why'] == 'gain' else
                   'Ни одна цель не укладывается в заряд с дорогой домой')
            return self._result(why + ' — возвращаюсь на базу.', [{'type': 'return_base'}], rec)
        chosen = [nodes[i] for i in order]
        skipped = [nd['id'] for k, nd in enumerate(nodes) if k not in order]
        text = (f"План на остаток прогона: {' → '.join(nd['id'] for nd in chosen)} → база. Ожидаю собрать "
                f"{info['value']:.1f} образца, на объезд и дорогу домой уйдёт около {info['cost']:.0f} ед. "
                f"из {obs.battery:.0f}.")
        if skipped:
            text += f" Не вошли в план: {', '.join(skipped)}."
        subgoals = [{'type': 'goto', 'x': nd['x'], 'y': nd['y']} if nd['kind'] == 'goto'
                    else {'type': nd['kind'], 'target': nd['id']} for nd in chosen]
        return self._result(text, subgoals, rec)

    @staticmethod
    def _result(reasoning, subgoals, rec):
        return {'reasoning': reasoning, 'hypotheses': [], 'subgoals': subgoals, 'source': 'tour',
                'exchanges': [], 'error': None, 'data': {'tour': rec}}

    def check(self, obs):
        """Раз в секунду вместо правила «заряд ниже запаса — домой».

        Пока текущая цель по заряду (с остановкой, грузом, дорогой домой от неё и запасом), робот едет к ней.
        Если нет — план пересчитывается: в нём останутся только цели, которые ещё по заряду, либо одна база.
        Проверка идёт каждую секунду, а не с того момента, когда заряд упал до цены возврата из нынешней
        точки: рядом с базой эта цена мала, и недоступная цель оставалась бы в очереди.
        """
        a, s = self.a, self.s
        t = obs.t
        if self.s.back == 'rule':
            a.tour = None
            try:
                a._check_return(obs)                     # правило агента как есть
            finally:
                a.tour = self
            if a._returning:
                return
        if a._returning:
            return
        self._visit(obs)
        home = a._home_cost(obs.x, obs.y)
        if a.collected >= a.n_samples:
            return a._go_home(t, 'все образцы собраны')
        if a.rules.time_limit_s - t <= home / a.rules.drain_per_m / 0.15 + 10.0:
            return a._go_home(t, 'время прогона на исходе')
        if a._trigger:
            return                                       # план и так будет пересчитан в этот же такт
        spent = a.rules.battery_start - obs.battery
        if self._survey and spent >= s.survey_budget:
            # Заряд на объезд вышел по дороге: оставшиеся обзорные точки снимаются, дальше — сбор.
            self._survey_end(obs, f'на него потрачено {spent:.0f} ед. заряда')
            return self._replan('survey')
        sg = a.queue[0] if a.queue else None
        if sg is None or 'x' not in sg:
            return
        if s.back == 'plan' and not self._fits(obs, sg):
            return self._replan('battery')
        if self._survey and t - self._plan_t >= 2.0 and any(
                c['mass'] >= s.survey_pick for c in a._candidates(t)):
            return a._request_plan('survey')             # по пути появился уверенный кандидат

    def _replan(self, trigger):
        a = self.a
        a.queue = []
        a._path_goal = None
        a._request_plan(trigger)

    def _trip(self, obs, sg):
        """Заряд на цель sg по счёту перебора: дорога до неё, остановка на сбор и дорога домой с грузом."""
        a = self.a
        g, w = a.graph, self._weights()
        per_m, per_load = self._rates()
        a._home_cost(obs.x, obs.y)                       # поле «домой» посчитано и лежит в a._base_pred
        node = g.node(sg['x'], sg['y'])
        take = sg['type'] == 'investigate'
        return (_walk(g, g.field(obs.x, obs.y)[1], node, w) * per_m + (self.s.dwell if take else 0.0)
                + _walk(g, a._base_pred, node, w) * (per_m + (per_load if take else 0.0)))

    def _fits(self, obs, sg):
        """Текущая цель ещё по заряду? Формула — того, кто цель выбрал; commit_slack — допуск, чтобы робот не
        бросал цель из-за колебаний оценки."""
        a, s = self.a, self.s
        if self._mine:                                   # перебор: запас — множителем ко всему объезду
            margin, absolute = self._reserve(obs)
            return obs.battery + s.commit_slack >= self._trip(obs, sg) * margin + absolute
        # Правило планировщика или обзорный объезд: запас — множителем к дороге домой (Agent._affordable).
        dist, pred = a.graph.field(obs.x, obs.y)
        to = a.graph.energy(dist, pred, sg['x'], sg['y']) * a._per_m()
        stop = s.dwell if sg['type'] == 'investigate' else 0.0
        return a._affordable(obs.battery + s.commit_slack - stop, to, a._home_cost(sg['x'], sg['y']))

    # --- обзорный объезд ---------------------------------------------------------------------------

    def _visit(self, obs):
        if self._survey:
            self._wps = [p for p in self._wps if math.hypot(p[0] - obs.x, p[1] - obs.y) > 0.35]

    def _survey_end(self, obs, why):
        self._survey = False
        self.a.journal.add(obs.t, 'decision', f'Обзорный объезд закончен: {why}. Дальше — сбор по плану.')

    def _survey_plan(self, obs, state):
        a, s = self.a, self.s
        left = state['samples']['total'] - state['samples']['collected']
        spent = a.rules.battery_start - obs.battery
        sure = [c for c in state['candidates'] if c['confidence'] >= s.survey_pick]
        if not self._wps:
            return self._survey_end(obs, 'все точки пройдены')
        if spent >= s.survey_budget:
            return self._survey_end(obs, f'на него потрачено {spent:.0f} ед. заряда')
        # Бюджет ограничивает и исполнение: check() раз в секунду снимает обзорную очередь, когда он вышел.
        if len(sure) >= s.survey_found * left:
            return self._survey_end(obs, f'уверенных кандидатов уже {len(sure)} на {left} оставшихся образцов')
        wx, wy = self._wps[0]
        dist, pred = a.graph.field(wx, wy)
        to = a.graph.energy(dist, pred, obs.x, obs.y) * a._per_m()
        if not a._affordable(obs.battery, to, a._home_cost(wx, wy)):
            return self._survey_end(obs, 'следующая точка не по заряду')
        subgoals, note = [], ''
        near = [(c['cost_to'] + a.graph.energy(dist, pred, c['x'], c['y']) * a._per_m() - to, c)
                for c in sure if c['feasible']]
        near = [(d, c) for d, c in near if d <= s.survey_detour]
        if near:
            detour, c = min(near, key=lambda dc: dc[0])
            subgoals.append({'type': 'investigate', 'target': c['id']})
            note = f" По пути забираю {c['id']} (уверенность {c['confidence']:.0%}, крюк {max(detour, 0.0):.1f} ед.)."
        subgoals += [{'type': 'goto', 'x': x, 'y': y} for x, y in self._wps]
        rec = {'t': round(obs.t, 1), 'trigger': state['trigger'], 'battery': round(obs.battery, 1),
               'survey_left': len(self._wps)}
        self.log.append(rec)
        return self._result(f'Обзорный объезд: осталось точек {len(self._wps)}, потрачено {spent:.0f} ед. из '
                            f'{s.survey_budget:.0f} отведённых.' + note, subgoals, rec)


def _walk(graph, pred, node, weight):
    """Длина пути по дереву pred от его корня до клетки node, метры × множитель weight по клеткам."""
    total = 0.0
    xs, ys = graph.xs, graph.ys
    while pred[node] >= 0:
        prev = pred[node]
        total += math.hypot(xs[node] - xs[prev], ys[node] - ys[prev]) * 0.5 * (weight[node] + weight[prev])
        node = prev
    return total
