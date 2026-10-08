"""Оракулы: агенты-мерила, которым быстрый симулятор сообщает часть скрытой правды сценария.

Нужны, чтобы разложить недобор очков обычного агента: сколько стоит поиск образцов, сколько —
незнание среды, и каков потолок при данном запасе заряда. В PRESETS оракулов нет: в Gazebo и на
роботе такой правды взять неоткуда. Правда приходит отдельным каналом — объектом Truth, который
создаёт did/runner.py; обычным вариантам агента он не передаётся.

  samples — места несобранных образцов (вместо карты вероятностей — KnownSamples);
  env     — множители грунта и опасные зоны, включая будущие: изменение из расписания сценария
            оракул учитывает за LEAD_S секунд до того, как оно случится. Значение 'touch' — урезанная
            правда: зона грунта известна целиком с момента, когда робот в неё въехал, и только она.

Сбой датчика образцов и расход на повороты и груз («научные» правила) в правду не входят.
"""
import math

import numpy as np

from .belief import HazardMap

LEAD_S = 15.0     # с: за сколько до изменения среды оракул начинает его учитывать

# имя → (вариант-основа из PRESETS, поправки настроек, какая правда известна)
ORACLES = {
    # Знает только места образцов и выбирает цель прежним правилом: разность с adaptive — цена поиска.
    'oracle_samples': ('adaptive', {}, {'samples': True}),
    # То же, но порядок обхода — план на весь остаток прогона (did/tour.py).
    'oracle_samples_tour': ('adaptive', {'scheme': 'tour'}, {'samples': True}),
    # Знает грунты и опасные зоны, образцы ищет сам: разность с adaptive — цена незнания среды.
    'oracle_env': ('no_change', {'avoid_hazards': False}, {'env': True}),
    'oracle_env_tour': ('no_change', {'avoid_hazards': False, 'scheme': 'tour'}, {'env': True}),
    # Зона грунта становится известна целиком в тот миг, когда робот в неё въехал; опасные зоны — как у adaptive.
    # Это потолок для любого способа «достроить» зону по первым сантиметрам дорогого пола.
    'oracle_touch': ('no_change', {}, {'env': 'touch'}),
    # Знает всё и обходит образцы по лучшему порядку: потолок для данного запаса заряда. Запас на возврат
    # меньше обычного: неожиданностей на пути у него нет (на отладочных сценариях возвращался всегда).
    'oracle_all': ('no_change', {'avoid_hazards': False, 'scheme': 'tour', 'reserve_margin': 1.0,
                                 'reserve_abs': 1.5, 'scheme_opts': {'true_rates': True}},
                   {'samples': True, 'env': True}),
}


class KnownSamples:
    """Замена карты вероятностей (интерфейс SampleBelief) для оракула: места образцов известны точно."""

    def __init__(self, arena, remaining, sub=2):
        self.arena = arena
        self._remaining = remaining            # функция: список (x, y) ещё не собранных образцов
        self.res = arena.res * sub
        self.x0, self.y0 = arena.x0, arena.y0
        self.h, self.w = arena.h // sub, arena.w // sub
        self.updates = 0

    @property
    def left(self):
        return len(self._remaining())

    def update(self, x, y, z, sigma):
        self.updates += 1

    def collected(self, x, y):
        pass

    def clear_disc(self, x, y, r, factor=0.0):
        pass

    def relax(self, alpha):
        pass

    def candidates(self, min_mass=0.3, radius=0.25, limit=None):
        return [{'x': float(x), 'y': float(y), 'mass': 0.99} for x, y in self._remaining()]

    def explore_points(self, radius=0.8, step=0.5, limit=6):
        return []

    def prob_within(self, x, y, r):
        return 1.0 if any(math.hypot(x - sx, y - sy) <= r for sx, sy in self._remaining()) else 0.0

    def total(self):
        return float(self.left)

    def grid(self):
        g = np.zeros((self.h, self.w))
        for sx, sy in self._remaining():
            i = min(max(int((sx - self.x0) / self.res), 0), self.w - 1)
            j = min(max(int((sy - self.y0) / self.res), 0), self.h - 1)
            g[j, i] = 1.0
        return g


class Truth:
    """Канал правды от быстрого симулятора к оракулу. samples, env — что именно известно."""

    def __init__(self, judge, scenario, arena, samples=False, env=False, lead_s=LEAD_S):
        self.judge, self.arena = judge, arena
        self.samples, self.env, self.lead = samples, env, lead_s
        self._X, self._Y = arena.cell_centers()
        # Расписание грунтов: [(с какого момента, зоны)]; опасных зон: [(с какого момента, зона)].
        self._soils = [(-math.inf, list(scenario.soils))]
        self._hazards = [(-math.inf, z) for z in scenario.hazards]
        for ev in sorted(scenario.events, key=lambda e: e['t']):
            if ev['type'] == 'soil_change':
                self._soils.append((ev['t'], list(ev['soils'])))
            elif ev['type'] == 'new_hazard':
                self._hazards.append((ev['t'], ev['zone']))
        self._soil_cache = {}
        self._n_hazards = -1
        self._touched = {}                     # зоны, в которые робот въезжал: id(зоны) → зона

    def attach(self, agent):
        if self.samples:
            agent.belief = KnownSamples(self.arena, lambda: list(self.judge.remaining.values()))
        if self.env:
            agent._soil_truth = self.touched_grid if self.env == 'touch' else self.soil_grid

    def soil_grid(self):
        """Множитель расхода по клеткам: наибольший из действующего сейчас и наступающих в ближайшие lead секунд.

        Пока набор действующих карт тот же, возвращается тот же массив: агент узнаёт о смене по подмене объекта.
        """
        t = self.judge.t
        now = max(i for i, (t0, _) in enumerate(self._soils) if t0 <= t)
        key = tuple(i for i, (t0, _) in enumerate(self._soils) if i >= now and t0 <= t + self.lead)
        if key not in self._soil_cache:
            grid = np.ones(self._X.shape)
            for i in key:
                for z in self._soils[i][1]:
                    grid = np.where(z.mask(self._X, self._Y), np.maximum(grid, z.mult), grid)
            self._soil_cache[key] = grid
        return self._soil_cache[key]

    def touched_grid(self):
        """Множитель расхода только по тем действующим зонам грунта, в которых робот уже побывал."""
        pose = self.judge._pose
        for z in self.judge.soils:
            if pose is not None and id(z) not in self._touched and z.contains(*pose):
                self._touched[id(z)] = z
        key = tuple(id(z) for z in self.judge.soils if id(z) in self._touched)
        if key not in self._soil_cache:
            grid = np.ones(self._X.shape)
            for z in self.judge.soils:
                if id(z) in self._touched:
                    grid = np.where(z.mask(self._X, self._Y), np.maximum(grid, z.mult), grid)
            self._soil_cache[key] = grid
        return self._soil_cache[key]

    def refresh(self, agent, obs):
        """Раз в такт: опасные зоны, действующие сейчас и появляющиеся в ближайшие lead секунд."""
        if self.env is not True:
            return
        zones = [z for t0, z in self._hazards if t0 <= obs.t + self.lead]
        if len(zones) == self._n_hazards:
            return
        first = self._n_hazards < 0
        self._n_hazards = len(zones)
        risk = np.zeros(self._X.shape)
        for z in zones:
            risk = np.maximum(risk, np.hypot(self._X - z.x, self._Y - z.y) <= z.r + HazardMap.MARGIN)
        agent._risk = agent._risk_full = risk
        agent.hazards = [(z.x, z.y, z.r) for z in zones]
        agent._cost_dirty = True
        agent._cost_t = -1e9
        if not first:
            agent._path_goal = None
            agent._request_plan('hazard_truth')
