"""Быстрый симулятор с несколькими роботами на одной арене (исследование M1).

Каждый робот — обычный FastSim со своим судьёй: своя батарея, свой датчик образцов, свои штрафы.
Общее у них — мир: один набор образцов (собранный одним исчезает для всех), одни грунты, опасные
зоны и события сценария. Роботы — препятствия друг для друга: проехать сквозь нельзя, лидар
напарника видит, сближение ближе суммы радиусов — столкновение.

Сам FastSim и судья не меняются. Робот получает арену-обёртку, в которой к стенам добавлены
остальные роботы, а судьи делят один словарь несобранных образцов. С одним роботом мир совпадает
с обычным FastSim шаг в шаг (tests/test_team.py).
"""
import math
from dataclasses import replace

import numpy as np

from .config import LIDAR_MAX, LIDAR_RAYS, ROBOT_RADIUS, Rules
from .fastsim import LIDAR_OFFSET, FastSim
from .judge_team import CONTACT_M, CONTACT_MARGIN_M, CONTACT_RELEASE_M, ROBOT_NAMES, SLOTS, team_score

# Лидар Burger стоит на верхней площадке: у напарника он видит не корпус, а такую же башенку лидара
# (цилиндр радиусом около 5 см в модели turtlebot3_burger). Поэтому вплотную по лидару напарник заметен
# хуже стены: когда корпуса уже соприкасаются, до башенки ещё около 0,19 м.
LIDAR_SEEN_RADIUS = 0.05


class _SharedArena:
    """Арена глазами одного робота: стены карты и остальные роботы."""

    def __init__(self, arena, world, me):
        self._arena, self._world, self._me = arena, world, me
        self.blocked_by = None               # номер робота, в которого упёрся последний запрос зазора

    def __getattr__(self, name):
        return getattr(self._arena, name)

    def clearance(self, x, y):
        c = self._arena.clearance(x, y)
        for j, other in enumerate(self._world.robots):
            if j == self._me:
                continue
            gap = math.hypot(x - other.x, y - other.y) - ROBOT_RADIUS     # до корпуса напарника
            if gap < ROBOT_RADIUS:
                self.blocked_by = j
            c = min(c, gap)
        return c

    def raycast(self, x, y, theta, n=LIDAR_RAYS, rmax=LIDAR_MAX):
        r = self._arena.raycast(x, y, theta, n, rmax)
        ang = theta + np.arange(n) * (2 * np.pi / n)
        dx, dy = np.cos(ang), np.sin(ang)
        for j, other in enumerate(self._world.robots):
            if j == self._me:
                continue
            ox = other.x + LIDAR_OFFSET * math.cos(other.th) - x
            oy = other.y + LIDAR_OFFSET * math.sin(other.th) - y
            along = ox * dx + oy * dy
            off2 = ox * ox + oy * oy - along * along
            hit = (along > 0.0) & (off2 < LIDAR_SEEN_RADIUS ** 2)
            d = along - np.sqrt(np.maximum(LIDAR_SEEN_RADIUS ** 2 - off2, 0.0))
            r = np.where(hit & (d > 0.0) & (d < r) & (d <= rmax), d, r)
        return r


class TeamSim:
    """Несколько роботов в одном мире. robots[i] — FastSim: он же RobotIO для агента i."""

    def __init__(self, arena, scenario, rules=None, seed=0, n_robots=2, slots=SLOTS, **sim):
        self.arena = arena
        self.scenario = scenario
        self.rules = rules or Rules()
        self.robots = []
        self._views = []
        self.contacts = 0                    # сколько раз роботы упёрлись друг в друга
        self._touching = set()               # пары, которые сейчас в контакте
        self.t = 0.0
        for i in range(n_robots):
            view = _SharedArena(arena, self, i)
            bot = FastSim(view, replace(scenario, base=tuple(slots[i])), self.rules, seed=seed, **sim)
            bot.judge.arena = arena          # штраф за стену судья считает по карте; контакт роботов — ниже
            if i:
                # Образцы одни на всех; шум датчиков, батареи и лидара у каждого робота свой.
                bot.judge.remaining = self.robots[0].judge.remaining
                bot.judge.rng = np.random.default_rng([int(seed), 7, i])
                bot.judge.rng_battery = np.random.default_rng([int(seed), 8, i])
                bot.rng = np.random.default_rng([int(seed), 11, i])
            self.robots.append(bot)
            self._views.append(view)
        self.dt = self.robots[0].dt

    @property
    def done(self):
        return all(r.judge.done for r in self.robots)

    @property
    def judges(self):
        return [r.judge for r in self.robots]

    def advance(self):
        """Один шаг dt для всех роботов по очереди; закончивший прогон робот стоит на месте препятствием."""
        bumped = []
        for i, (bot, view) in enumerate(zip(self.robots, self._views)):
            view.blocked_by = None
            if bot.judge.done:
                continue
            bot.advance()
            if bot._blocked and view.blocked_by is not None:
                bumped.append((i, view.blocked_by))
        self.t = round(self.t + self.dt, 6)
        for i, j in bumped:
            pair = (min(i, j), max(i, j))
            if pair in self._touching:
                continue
            self._touching.add(pair)
            self.contacts += 1
            # Въехавший получил штраф от своего судьи (упор). Второму участнику — тот же штраф, один раз за контакт.
            other = self.robots[j]
            if not other.judge.done and not any(a == j for a, _ in bumped):
                other.judge._penalty('collision', other.x, other.y)
        for pair in list(self._touching):
            a, b = self.robots[pair[0]], self.robots[pair[1]]
            if math.hypot(a.x - b.x, a.y - b.y) > CONTACT_M + CONTACT_MARGIN_M + CONTACT_RELEASE_M:
                self._touching.discard(pair)
                continue
            # Пока роботы не разъехались, это один и тот же контакт: упёршийся робот дёргается у самого
            # напарника (шаг вперёд — упор — шаг вперёд), и без этого судья считал бы каждый рывок заново.
            a.judge._colliding = b.judge._colliding = True

    def collected(self):
        """Все сборы команды: [(номер образца, t, номер робота)] по времени."""
        out = [(idx, t, i) for i, r in enumerate(self.robots) for idx, t in r.judge.collected]
        return sorted(out, key=lambda c: c[1])

    def score(self):
        scores = [r.judge.score() for r in self.robots]
        total = team_score(scores, [t for _, t, _ in self.collected()], len(self.scenario.samples),
                           self.rules.time_limit_s)
        total['robot_contacts'] = self.contacts
        total['per_robot'] = [{'name': ROBOT_NAMES[i] if i < len(ROBOT_NAMES) else f'tb{i + 1}', **s}
                              for i, s in enumerate(scores)]
        return total
