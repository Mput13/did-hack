"""Быстрый симулятор: та же арена, тот же судья и тот же интерфейс агента, но без Gazebo.

Робот — точка с курсом (кинематика дифференциального привода с пределами Burger), стены — клетки
карты, лидар — трассировка лучей по карте. Прогон в сотни раз быстрее реального времени, поэтому
серии экспериментов считаются здесь, а Gazebo подтверждает, что поведение переносится.
"""
import math

import numpy as np

from .config import ACC_MAX, LIDAR_MAX, LIDAR_MIN, LIDAR_SIGMA, ROBOT_RADIUS, V_MAX, W_MAX, Rules
from .judge import Judge
from .robot_io import Observation

LIDAR_OFFSET = -0.032     # лидар Burger стоит чуть позади центра
W_ACC_MAX = 8.0           # рад/с²: грубая оценка инерции корпуса


class FastSim:

    def __init__(self, arena, scenario, rules=None, seed=0, dt=0.1, lidar_hz=2.5, odom_drift=0.0,
                 odom_turn_slip=0.0, odom_turn_scale=0.0, odom_path_scale=0.0):
        self.arena = arena
        self.rules = rules or Rules()
        self.judge = Judge(scenario, arena, self.rules, seed=seed)
        self.rng = np.random.default_rng([int(seed), 11])
        self.dt = dt
        self.t = 0.0
        self.x, self.y = scenario.base
        self.th = 0.0
        self.v = self.w = 0.0
        self._cmd = (0.0, 0.0)
        self._blocked = False
        self._sensor_period = 1.0 / self.rules.sensor_hz
        self._lidar_period = 1.0 / lidar_hz if lidar_hz else None
        self._next_sensor = 0.0
        self._next_lidar = 0.0
        self._sensor = None
        self._scan = None
        # Уход одометрии: агент видит позу, накопленную «по колёсам», судья считает по истинной.
        # Модель по наблюдениям в Gazebo: колёса слушаются команды сразу, корпус отстаёт, и при резкой
        # смене угловой скорости одометрия засчитывает поворот, которого не было.
        self._drift = odom_drift              # случайное блуждание: разброс шага одометрии в долях шага пути
        self._turn_slip = odom_turn_slip      # доля отставания корпуса, попадающая в одометрию (1 — колёса скользят)
        self._turn_scale = odom_turn_scale    # постоянная ошибка масштаба поворота (0.02 — на 2% больше)
        self._path_scale = odom_path_scale    # постоянная ошибка масштаба пути
        drifting = odom_drift or odom_turn_slip or odom_turn_scale or odom_path_scale
        self._odom = [self.x, self.y, self.th] if drifting else None     # None — одометрия точная
        self.judge.step(0.0, self.x, self.y)
        self._sense()

    # --- интерфейс агента --------------------------------------------------------------------

    def observe(self):
        ox, oy, oth = self._odom or (self.x, self.y, self.th)
        obs = Observation(
            t=self.t, x=ox, y=oy, th=oth,
            v=self.v, w=self.w, battery=self.judge.read_battery(), sensor=self._sensor, scan=self._scan,
            events=self.judge.pop_events(), done=self.judge.done)
        self._sensor = None
        self._scan = None
        return obs

    def command(self, v, w):
        self._cmd = (float(np.clip(v, -V_MAX, V_MAX)), float(np.clip(w, -W_MAX, W_MAX)))

    def collect(self):
        return self.judge.collect(self.x, self.y)

    def finish(self):
        return self.judge.finish(self.x, self.y)

    # --- ход симуляции -----------------------------------------------------------------------

    @property
    def done(self):
        return self.judge.done

    def advance(self):
        """Один шаг dt: движение, расход, датчики."""
        if self.judge.done:
            return
        dt = self.dt
        v_cmd, w_cmd = self._cmd
        self.v += float(np.clip(v_cmd - self.v, -ACC_MAX * dt, ACC_MAX * dt))
        self.w += float(np.clip(w_cmd - self.w, -W_ACC_MAX * dt, W_ACC_MAX * dt))
        th_mid = self.th + 0.5 * self.w * dt
        nx = self.x + self.v * math.cos(th_mid) * dt
        ny = self.y + self.v * math.sin(th_mid) * dt
        self._blocked = self.arena.clearance(nx, ny) < ROBOT_RADIUS
        if self._blocked:
            self.v = 0.0                      # упёрся в преграду: стоит на месте, крутиться может
        else:
            self.x, self.y = nx, ny
        if self._odom is not None:
            self._step_odom(self.v * dt, self.w * dt, w_cmd)
        self.th = _wrap(self.th + self.w * dt)
        self.t = round(self.t + dt, 6)
        self.judge.step(self.t, self.x, self.y, blocked=self._blocked, th=self.th)
        self._sense()

    def _step_odom(self, ds, dth, w_cmd):
        """Шаг одометрии: тот же путь и поворот, но с ошибками колёс."""
        o = self._odom
        # Колёса уже крутятся с заданной скоростью, корпус ещё нет: разницу одометрия считает поворотом.
        dth_o = dth * (1.0 + self._turn_scale) + self._turn_slip * (w_cmd - self.w) * self.dt
        ds_o = ds * (1.0 + self._path_scale)
        if self._drift:
            o[0] += self.rng.normal(0.0, self._drift) * abs(ds)
            o[1] += self.rng.normal(0.0, self._drift) * abs(ds)
            dth_o += self.rng.normal(0.0, self._drift) * abs(ds)
        mid = o[2] + 0.5 * dth_o
        o[0] += ds_o * math.cos(mid)
        o[1] += ds_o * math.sin(mid)
        o[2] = _wrap(o[2] + dth_o)

    def _sense(self):
        if self.t + 1e-9 >= self._next_sensor:
            self._sensor = self.judge.read_sensor(self.x, self.y)
            self._next_sensor += self._sensor_period
        if self._lidar_period and self.t + 1e-9 >= self._next_lidar:
            lx = self.x + LIDAR_OFFSET * math.cos(self.th)
            ly = self.y + LIDAR_OFFSET * math.sin(self.th)
            r = self.arena.raycast(lx, ly, self.th)
            r = r + self.rng.normal(0.0, LIDAR_SIGMA, r.shape)
            r[(r < LIDAR_MIN) | (r > LIDAR_MAX)] = np.inf
            self._scan = r
            self._next_lidar += self._lidar_period


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
