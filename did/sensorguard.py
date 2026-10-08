"""Пережидание сбоя датчика образцов: стоять дешевле, чем ездить за ложными кандидатами.

Пока датчик врёт (шумит, залип, занижает), карта образцов рисует кандидатов там, где их нет, и агент
тратит на них заряд. Стоять же почти ничего не стоит: расход «за включённость» из правил в сотни раз
меньше расхода на путь, а времени прогона хватает с запасом. Поэтому при признаках сбоя агент
останавливается и ждёт, пока датчик не придёт в норму.

Признаки — только из собственных измерений агента:
  шум      — оценка разброса показаний выросла (did.belief.SensorHealth);
  залип    — показания одинаковы до последнего знака, хотя робот едет;
  подряд   — несколько подъездов к кандидатам подряд кончились ничем.
Пока робот стоит, исправный датчик обязан показывать одно и то же, поэтому конец сбоя виден по самим
показаниям: разброс вернулся к норме, значение сдвинулось или снова начало меняться.

Ожидание ограничено по времени и прерывается, если стоя робот теряет заряд быстрее положенного
(утечка): тогда стоять уже не дёшево. Все пороги — в секундах и метрах, а не в тактах симулятора.
"""
import math
from collections import deque

import numpy as np


class SensorGuard:

    MAX_WAIT = {'noise': 60.0, 'stuck': 60.0, 'streak': 25.0}   # с: дольше этого не ждём
    MIN_WAIT = 3.0             # с: столько стоим, прежде чем судить о датчике
    LOST_WINDOW = 40.0         # с: за какое время они считаются «подряд»
    COOLDOWN = 6.0             # с после ожидания: дать карте образцов набрать новые показания
    TIME_SLACK = 150.0         # с до конца прогона: меньше — не ждём
    SHIFT = 0.10               # сдвиг среднего показания стоя, который означает смену состояния датчика

    def __init__(self, agent):
        self.a = agent
        self.z = deque(maxlen=40)          # (t, x, y, показание): последние 8 секунд
        self.lost_limit = agent.cfg.fault_wait_lost     # подъездов без сбора подряд
        self.lost = deque(maxlen=max(1, self.lost_limit))
        self.wait = None                   # {'why', 't0', 'b0', 'ref', 'value'}
        self.quiet_until = -1e9
        self._noise_done = False           # этот эпизод шума уже переждали (или ждать его оказалось без толку)
        self._stuck_done = None            # залипшее значение, которое ждать больше не будем
        self.waited_s = 0.0
        self.n_waits = 0

    # --- что агент сообщает ------------------------------------------------------------------

    def reading(self, z, obs):
        self.z.append((obs.t, obs.x, obs.y, z))

    def subgoal_ended(self, trigger, t):
        if trigger == 'candidate_lost':
            self.lost.append(t)
        elif trigger == 'sample_collected':
            self.lost.clear()

    # --- решение -----------------------------------------------------------------------------

    def _sigma(self, since):
        """Оценка шума по разностям соседних показаний (как в SensorHealth, но и при сигнале у нуля)."""
        v = np.array([z for t, _, _, z in self.z if t >= since])
        if len(v) < 12:
            return None
        d = np.diff(v)
        return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))

    def _symptom(self, obs):
        a = self.a
        if self._noisy() and not self._noise_done:
            return 'noise'
        if a.inv is None and len(self.z) >= 10:
            last = list(self.z)[-10:]
            moved = math.hypot(last[-1][1] - last[0][1], last[-1][2] - last[0][2])
            same = max(v[3] for v in last) - min(v[3] for v in last) < 1e-9 and 0.0 < last[-1][3] < 1.0
            if same and moved >= 0.12 and last[-1][3] != self._stuck_done:
                return 'stuck'
        if self.lost_limit and len(self.lost) >= self.lost_limit and obs.t - self.lost[0] <= self.LOST_WINDOW:
            return 'streak'
        return None

    def _noisy(self):
        return self.a.cfg.sensor_health and self.a.health.degraded

    def hold(self, obs):
        """True — этот такт робот стоит и ждёт датчик."""
        a = self.a
        t = obs.t
        if not self._noisy():
            self._noise_done = False       # оценка шума в норме: следующий шум — уже новый эпизод
        if self.wait is None:
            if t < self.quiet_until or a._returning or a.rules.time_limit_s - t < self.TIME_SLACK:
                return False
            if a.inv is not None and (a.inv.leak or a.inv.active is not None or a.inv.sensor['mode'] in ('stuck', 'bias')):
                return False               # исследователь с этим сбоем уже разбирается сам либо стоять сейчас дорого
            if t - a._hazard_t < 5.0 or a._grace[1] is not None or self._in_risk(obs):
                return False               # сначала выехать из опасной зоны: стоя в ней, робот копит штрафы
            why = self._symptom(obs)
            if why is None:
                return False
            self.wait = {'why': why, 't0': t, 'b0': obs.battery, 'value': self.z[-1][3] if self.z else 0.0}
            self.n_waits += 1
            a.journal.add(t, 'decision', {
                'noise': 'Датчик образцов шумит: стою и жду, пока шум спадёт, — ехать за кандидатами по таким '
                         'показаниям дороже, чем постоять',
                'stuck': 'Датчик образцов повторяет одно и то же значение, хотя я еду: стою и жду, пока он оживёт',
                'streak': f'Подъезды к кандидатам {self.lost_limit} раза подряд ничего не дали: похоже, датчик врёт. '
                          'Стою и смотрю, изменятся ли показания',
            }[why], tag='sensor_wait', why=why)
            return True
        w = self.wait
        dt = t - w['t0']
        end = None
        if a._returning:
            end = 'пора возвращаться на базу'
        elif a._hazard_t > w['t0']:
            end = 'получен штраф опасной зоны, здесь стоять нельзя'
        elif dt >= self.MIN_WAIT and (w['b0'] - obs.battery) / dt > 5.0 * a.rules.drain_idle_per_s + 0.03:
            end = 'стоя теряю заряд быстрее обычного, ждать дорого'
        elif dt >= self.MAX_WAIT[w['why']]:
            end = 'ждать дольше нет смысла'
        elif w['why'] == 'noise':
            sigma = self._sigma(max(w['t0'], t - 4.0))
            if dt >= self.MIN_WAIT and sigma is not None and sigma < 1.5 * a.health.nominal:
                end = f'шум вернулся к {sigma:.2f}'
        elif w['why'] == 'stuck':
            if self.z and abs(self.z[-1][3] - w['value']) > 1e-9:
                end = 'показания снова меняются'
        else:
            stand = [z for zt, _, _, z in self.z if zt >= w['t0'] + 0.5]
            if self._noisy() and not self._noise_done:
                w['why'] = 'noise'         # пока стоял, проявился шум: дальше ждём по его правилу
            elif len(stand) >= 25:
                ref = w.setdefault('ref', float(np.mean(stand[:15])))
                if abs(float(np.mean(stand[-10:])) - ref) > self.SHIFT:
                    end = 'показания стоя сдвинулись: состояние датчика изменилось'
        if end is None:
            return True
        self.waited_s += dt
        self.quiet_until = t + self.COOLDOWN
        self.lost.clear()
        self.wait = None
        # Один эпизод сбоя пережидается один раз: иначе при шуме без конца робот стоял бы снова и снова.
        if w['why'] == 'noise' or self._noisy():
            self._noise_done = True
        if w['why'] == 'stuck':
            self._stuck_done = w['value'] if self.z and abs(self.z[-1][3] - w['value']) < 1e-9 else None
        # Показаниям, набранным за время сбоя, доверия меньше: карта частично возвращается к незнанию.
        a.belief.relax(0.3 if w['why'] != 'noise' else 0.15)
        a.journal.add(t, 'decision', f'Ожидание датчика закончено через {dt:.0f} с: {end}', tag='sensor_wait_end')
        a._request_plan('sensor_wait')
        return False

    def _in_risk(self, obs):
        a = self.a
        if not a.hazards:
            return False
        ix, iy = a.arena.w2g(obs.x, obs.y)
        return a._risk_full[iy, ix] > 0.02
