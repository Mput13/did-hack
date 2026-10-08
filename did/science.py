"""Расследование: несколько объяснений одной странности и опыт, который их различает.

Агент замечает, что измерения не сходятся с его моделью. Вместо одного объяснения он держит
несколько, для каждого возможного опыта считает, что предсказывает каждое объяснение, выбирает
опыт с наибольшей ожидаемой пользой на единицу заряда, проводит его и пересчитывает вероятности
по правилу Байеса. Если объяснения различить не удалось — честно пишет «недостаточно данных».

Здесь только расчёт; какие бывают объяснения и опыты, решает агент (did/agent.py), а предлагать и
критиковать их может языковая модель (did/llm_roles.py). Вероятности модель не назначает.
"""
import math
from dataclasses import dataclass, field

import numpy as np

OTHER = 'other'      # объяснение «причина не из списка»: предсказаний у него нет

# Как выбирается следующий опыт среди допустимых (абляция R12, research/findings/R12.md). Допустимый — ещё не
# проведён, укладывается в бюджет заряда и обещает не меньше min_gain бит. Всё остальное в расследовании
# (объяснения, пересчёт вероятностей, остановка, бюджет) от способа выбора не зависит.
CHOICES = {
    'gain': 'наибольшая ожидаемая польза на единицу заряда (как всегда было)',
    'bits': 'наибольшая ожидаемая польза, цена не учитывается',
    'random': 'случайный из допустимых',
    'cheapest': 'самый дешёвый из допустимых',
    'fixed': 'первый допустимый в порядке объявления: постоять, проехать прямо, развернуться',
    'worst': 'наименее различающий из допустимых (нижняя граница)',
    'blind': 'случайный из непроведённых и посильных по заряду, польза не считается вовсе',
}


@dataclass
class Alternative:
    id: str
    statement: str
    prior: float


@dataclass
class TestOption:
    id: str
    name: str
    cost: float                 # ожидаемый расход заряда на опыт
    duration_s: float
    unit: str
    predictions: dict           # id объяснения -> (среднее, разброс) того, что покажет измерение
    action: dict = field(default_factory=dict)     # как агенту провести опыт
    sigma: float = 0.0          # точность самого измерения
    measured: dict = None       # {'value', 'sigma', 't'} после проведения (последний замер)
    gain_bits: float = 0.0
    repeatable: bool = False    # опыт можно повторить: новый замер — новые данные
    repeats: int = 0


def entropy_bits(p):
    p = np.asarray([v for v in p if v > 1e-12], dtype=float)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


class Inquiry:

    def __init__(self, qid, t, topic, anomaly, alternatives, tests, *, accept=0.85, max_tests=3,
                 energy_budget=2.5, min_gain=0.05, source='rule'):
        total = sum(a.prior for a in alternatives)
        self.id, self.topic, self.anomaly, self.source = qid, topic, anomaly, source
        self.t_open, self.t_close = round(float(t), 1), None
        self.alternatives = [Alternative(a.id, a.statement, a.prior / total) for a in alternatives]
        self.posterior = {a.id: a.prior for a in self.alternatives}
        self.tests = list(tests)
        self.accept, self.max_tests, self.budget, self.min_gain = accept, max_tests, energy_budget, min_gain
        self.spent = 0.0
        self.order = []               # какие замеры сделаны, по порядку
        self.maneuvers = 0            # сколько манёвров на них ушло (одна пауза может дать несколько замеров)
        self.conclusion = None
        self.action = ''
        self.critique = []
        self.note = ''
        self.follow = None            # L3: опыты в порядке плана автора (список id) вместо выбора по пользе
        self.llm = None               # L3: что предложили автор и критик (для сверки с выбором расчёта)
        self.policy, self.rng = 'gain', None      # способ выбора опыта (CHOICES) и генератор для случайных способов
        self.stop = None              # почему опыты кончились: settled | max_tests | budget | no_gain | no_tests | ...
        self.steps = []               # каждый выбор: что было допустимо, что выбрано (в запись прогона, агент не читает)
        for test in self.tests:
            test.gain_bits = round(self.gain(test), 3)

    # --- расчёт ------------------------------------------------------------------------------

    def _span(self, test):
        """Диапазон значений, которые опыт вообще может показать, с учётом шума самого измерения."""
        lo = min(m - 4 * math.hypot(s, test.sigma) for m, s in test.predictions.values())
        hi = max(m + 4 * math.hypot(s, test.sigma) for m, s in test.predictions.values())
        pad = 0.5 * (hi - lo) + 1e-6
        return lo - pad, hi + pad

    def _likelihood(self, test, y, sigma_meas):
        """Плотность измерения y при каждом объяснении.

        У «причины не из списка» предсказания нет, поэтому ей отвечает широкое распределение с
        тяжёлыми хвостами (Коши) по всему диапазону опыта. Оно нормировано на всей прямой: любое
        значение для неё возможно, но ни одно не «ожидается» — совпадение с узким предсказанием
        настоящего объяснения всегда весит больше, а промах мимо всех предсказаний достаётся ей.
        """
        lo, hi = self._span(test)
        centre, scale = 0.5 * (lo + hi), 0.25 * (hi - lo)
        out = {}
        for a in self.alternatives:
            pred = test.predictions.get(a.id)
            if pred is None:
                out[a.id] = 1.0 / (math.pi * scale * (1.0 + ((np.asarray(y, dtype=float) - centre) / scale) ** 2))
            else:
                s = math.hypot(pred[1], sigma_meas)
                out[a.id] = np.exp(-0.5 * ((np.asarray(y) - pred[0]) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        return out

    def gain(self, test):
        """Ожидаемое сокращение неопределённости от опыта, в битах (взаимная информация)."""
        lo, hi = self._span(test)
        y = np.linspace(lo, hi, 1200)
        like = self._likelihood(test, y, test.sigma)
        joint = np.array([self.posterior[a.id] * like[a.id] for a in self.alternatives])   # объяснение × y
        py = joint.sum(axis=0)
        post = joint / np.maximum(py, 1e-300)
        h_after = -(post * np.log2(np.maximum(post, 1e-300))).sum(axis=0)
        dy = y[1] - y[0]
        mass = float((py * dy).sum())
        expected = float((py * h_after * dy).sum()) / max(mass, 1e-9)
        return max(0.0, entropy_bits(self.posterior.values()) - expected)

    @property
    def best(self):
        return max(self.posterior.items(), key=lambda kv: kv[1])

    @property
    def settled(self):
        """Одно объяснение набрало достаточно, и это не «причина не из списка»."""
        best, p = self.best
        return p >= self.accept and best != OTHER

    def peek(self):
        """Что выбрал бы choose() сейчас. Генератор случайного выбора, запись шагов и причина остановки не меняются
        (оценки пользы опытов обновляются, как и при choose): это справка для записи, а не решение."""
        stop, n = self.stop, len(self.steps)
        state = self.rng.bit_generator.state if self.rng is not None else None
        try:
            return self.choose()
        finally:
            self.stop = stop
            del self.steps[n:]
            if state is not None:
                self.rng.bit_generator.state = state

    def choose(self):
        """Следующий опыт: наибольшая польза на единицу заряда. None — опыты больше не нужны или невозможны.

        Другие способы выбора (self.policy, см. CHOICES) нужны только для сравнения с этим.
        """
        if self.conclusion:
            return None
        if self.settled or self.maneuvers >= self.max_tests:
            self.stop = 'settled' if self.settled else 'max_tests'
            return None
        if self.follow is not None:
            return self._next_planned()
        blind = self.policy == 'blind'
        options, left, costly = [], 0, []
        for test in self.tests:
            if test.measured is not None and (not test.repeatable or test.repeats >= 2):
                continue
            left += 1
            if self.spent + test.cost > self.budget:
                costly.append(test)
                continue
            if blind:
                options.append(test)
                continue
            test.gain_bits = round(self.gain(test), 3)
            if test.gain_bits < self.min_gain:
                continue
            options.append(test)
        best = self._pick(options)
        # Не по карману: записанная польза такого опыта не обновляется (как и раньше), считается только для учёта.
        over = [{'id': x.id, 'gain_bits': round(self.gain(x), 3), 'cost': round(x.cost, 3)} for x in costly]
        if best is None:
            useful = [x for x in over if blind or x['gain_bits'] >= self.min_gain]
            self.stop = 'no_tests' if not left else 'budget' if useful else 'no_gain'
        self.steps.append({'options': [{'id': x.id, 'gain_bits': x.gain_bits, 'cost': round(x.cost, 3),
                                        'kind': x.action.get('kind')} for x in options],
                           'over_budget': over, 'chosen': best.id if best else None,
                           'posterior': {k: round(v, 3) for k, v in self.posterior.items()}})
        return best

    def _next_planned(self):
        """Следующий опыт из плана автора: первый ещё не проведённый, который проходит по бюджету.

        Польза не проверяется: что и в каком порядке мерить, решил автор. План исчерпан — опытов больше нет.
        """
        by_id = {x.id: x for x in self.tests}
        for tid in self.follow:
            test = by_id.get(tid)
            if test is None or test.measured is not None or self.spent + test.cost > self.budget:
                continue
            test.gain_bits = round(self.gain(test), 3)
            return test
        return None

    def _pick(self, options):
        if not options:
            return None
        if self.policy == 'gain':
            best, score = None, 0.0
            for test in options:
                value = test.gain_bits / (test.cost + 0.05)
                if value > score:
                    best, score = test, value
            return best
        if self.policy in ('random', 'blind'):
            return options[int(self.rng.integers(len(options)))]
        if self.policy == 'bits':
            return max(options, key=lambda x: x.gain_bits)
        if self.policy == 'cheapest':
            return min(options, key=lambda x: x.cost)
        if self.policy == 'worst':
            return min(options, key=lambda x: x.gain_bits)
        if self.policy == 'fixed':
            return options[0]
        raise ValueError(f'Unknown inquiry choice: {self.policy}')

    def record(self, test_id, value, sigma, t, cost=None, maneuver=True):
        """Результат опыта: пересчитать вероятности объяснений. cost — сколько заряда ушло на самом деле."""
        test = next(x for x in self.tests if x.id == test_id)
        like = self._likelihood(test, float(value), sigma)
        weights = {a.id: self.posterior[a.id] * float(like[a.id]) for a in self.alternatives}
        total = sum(weights.values())
        if total > 0:
            self.posterior = {k: v / total for k, v in weights.items()}
        test.measured = {'value': round(float(value), 4), 'sigma': round(float(sigma), 4), 't': round(float(t), 1)}
        test.repeats += 1
        if cost is not None:
            test.cost = max(float(cost), 0.0)
        self.spent += test.cost
        self.order.append(test_id)
        if maneuver:
            self.maneuvers += 1

    def close(self, t, action='', veto=None):
        """Зафиксировать вывод. veto — причина, по которой вывод нельзя считать установленным, даже если
        вероятность высока (например, объяснение принято только по исключению, без прямого опыта)."""
        best, p = self.best
        by_id = {a.id: a for a in self.alternatives}
        rivals = sorted(((v, k) for k, v in self.posterior.items() if k != best), reverse=True)
        if self.settled and not veto:
            status = 'identified'
            text = (f'{by_id[best].statement} — вероятность {p:.0%}; замеров: {len(self.order)}' if self.order
                    else f'{by_id[best].statement} — по исключению, без нового опыта')
        else:
            status = 'insufficient'
            rival = by_id[rivals[0][1]].statement if rivals else '—'
            reason = (veto if veto else 'ни одно объяснение из списка не подошло' if best == OTHER else
                      'опытов, которые их различили бы, не осталось' if self.maneuvers < self.max_tests
                      else 'лимит опытов исчерпан')
            text = (f'недостаточно данных: вероятнее всего «{by_id[best].statement}» ({p:.0%}), но не исключено '
                    f'«{rival}» ({rivals[0][0]:.0%}); {reason}') if rivals else 'недостаточно данных'
        self.conclusion = {'status': status, 'best': best, 'confidence': round(p, 3), 'text': text}
        self.action = action
        self.t_close = round(float(t), 1)
        return self.conclusion

    # --- для журнала и интерфейса ------------------------------------------------------------

    def to_dict(self):
        return {
            'id': self.id, 't_open': self.t_open, 't_close': self.t_close, 'topic': self.topic,
            'anomaly': self.anomaly,
            'alternatives': [{'id': a.id, 'statement': a.statement, 'prior': round(a.prior, 3),
                              'posterior': round(self.posterior[a.id], 3)} for a in self.alternatives],
            'tests': [{'id': x.id, 'name': x.name, 'cost': round(x.cost, 2), 'duration_s': x.duration_s,
                       'unit': x.unit, 'gain_bits': x.gain_bits,
                       'predictions': {k: {'mean': round(m, 3), 'sigma': round(s, 3)}
                                       for k, (m, s) in x.predictions.items()},
                       'chosen': x.measured is not None, 'measured': x.measured} for x in self.tests],
            'conclusion': self.conclusion, 'action': self.action, 'source': self.source,
            'critique': self.critique, 'note': self.note,
            **({'llm': self.llm} if self.llm else {}),
        }

    def context(self, state):
        """То же в виде, который получает языковая модель (автор и критик)."""
        return {
            'topic': self.topic, 'anomaly': self.anomaly, 'state': state,
            'alternatives': [{'id': a.id, 'statement': a.statement, 'prior': round(a.prior, 2)}
                             for a in self.alternatives],
            'tests': [{'id': x.id, 'name': x.name, 'cost': round(x.cost, 2), 'duration_s': x.duration_s,
                       'unit': x.unit, 'gain_bits': x.gain_bits,
                       'predictions': {k: {'mean': round(m, 3), 'sigma': round(s, 3)}
                                       for k, (m, s) in x.predictions.items()}} for x in self.tests],
            'budget': {'energy': self.budget, 'max_tests': self.max_tests},
        }

    def restrict(self, consider=None, order=None):
        """Сузить расследование по предложению автора: какие объяснения рассматривать и в каком порядке опыты."""
        if consider:
            keep = [a for a in self.alternatives if a.id in consider or a.id == OTHER]
            if len(keep) >= 2:
                total = sum(a.prior for a in keep)
                self.alternatives = [Alternative(a.id, a.statement, a.prior / total) for a in keep]
                self.posterior = {a.id: a.prior for a in self.alternatives}
        if order:
            rank = {tid: i for i, tid in enumerate(order)}
            self.tests.sort(key=lambda x: rank.get(x.id, len(rank)))
        for test in self.tests:
            test.gain_bits = round(self.gain(test), 3)
