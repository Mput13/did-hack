"""Пережидание сбоя датчика образцов: стоять, пока это дешевле, чем ездить за ложными кандидатами.

Пока датчик врёт (шумит, залип), карта образцов рисует кандидатов там, где их нет, и агент тратит на
них заряд. Стоять в это время выгодно, только если простой дёшев, — а сколько он стоит, решают правила
судьи, которых агент заранее не знает. Поэтому цена простоя не берётся из правил, а измеряется: по
падению батареи за секунды, когда робот стоит. Так же измеряется цена секунды езды.

Признаки сбоя — только из собственных измерений агента:
  шум      — оценка разброса показаний выросла (did.belief.SensorHealth);
  залип    — показания одинаковы до последнего знака, хотя робот едет;
  подряд   — несколько подъездов к кандидатам подряд кончились ничем. Это не повод ждать, а повод
             коротко проверить датчик: постоять пару секунд и посмотреть, держится ли показание.
             В пресетах v2 выключено (fault_wait_lost=0): выигрыша проверка не показала.

Что ограничивает ожидание:
  выгода   — секунда простоя должна стоить меньше, чем секунда езды вслепую тратит впустую;
  бюджет   — на весь прогон: не больше fault_wait_s секунд и не больше fault_wait_charge единиц заряда
             (настройки агента), сколько бы эпизодов ни было; заряд считается по самой батарее;
  возврат  — ожидание не начинается, если после него заряда не хватит на дорогу домой, и кончается,
             как только агент решил возвращаться;
  повтор   — тот же сбой пережидается снова только после устойчивого восстановления датчика
             (RECOVER_S секунд исправных показаний), а не после одного переключения оценки.

Конец сбоя виден стоя: исправный датчик неподвижного робота показывает одно и то же. Ожидание кончается,
когда оценка здоровья датчика вернулась к норме, когда разброс показаний стоя стал обычным или по
времени. Все пороги — в секундах, метрах и единицах заряда; числа показаний в секунду в них нет, так
что сторож одинаково работает при 2, 5 и 10 показаниях в секунду и при показаниях с задержкой.
"""
import math
from collections import deque

import numpy as np


class SensorGuard:

    MIN_WAIT = 3.0             # с: столько стоим, прежде чем судить о датчике и о цене простоя
    PROBE_S = 2.5              # с: короткая проверка датчика после подъездов впустую
    MAX_PROBES = 2             # таких проверок за прогон
    WINDOW = 4.0               # с: по показаниям за это время судим о датчике стоя…
    WINDOW_N = 16              # …но не меньше чем по стольким показаниям (редкий датчик — окно длиннее)
    WINDOW_MAX = 10.0          # с: и не длиннее этого
    RECOVER_S = 10.0           # с исправных показаний подряд: после этого сбой считается новым
    LOST_WINDOW = 40.0         # с: за какое время подъезды считаются «подряд»
    COOLDOWN = 6.0             # с после ожидания: дать карте образцов набрать новые показания
    TIME_SLACK = 150.0         # с до конца прогона: меньше — не ждём
    SILENT = 5.0               # с без показаний: судить не по чему, ждать бессмысленно
    WASTE = 0.5                # доля расхода на езду, которая при сбое датчика уходит впустую (P1.md, п. 3)
    TYPICAL = 35.0             # с: сколько обычно длится сбой — хватит ли заряда и на ожидание, и на дорогу домой
    TRIP_SPEED = 0.15          # м/с: средняя скорость на маршруте, для расхода «за включённость» в дороге
    IDLE_COVERED = 0.01        # ед./с: простой такой цены уже покрыт запасом на возврат (reserve_abs подобран при нём)
    CHUNK = 2.0                # с простоя на один замер цены простоя
    MOVE_PRIOR = 0.45          # ед./с: расход на ходу, пока своих замеров мало (2,5 ед./м × 0,18 м/с)

    def __init__(self, agent):
        self.a = agent
        self.time_budget = agent.cfg.fault_wait_s          # с на прогон: дольше в сумме не стоим
        self.charge_budget = agent.cfg.fault_wait_charge   # ед. заряда на прогон: столько можно потратить стоя
        self.z = deque(maxlen=400)         # (t измерения, x, y, показание): последние ~12 секунд
        self.lost_limit = agent.cfg.fault_wait_lost     # подъездов без сбора подряд
        self.lost = deque(maxlen=max(1, self.lost_limit))
        self.wait = None                   # {'why', 't0', 'b0', 'value', 'probe', 'seen'}
        self.quiet_until = -1e9
        self._noise_block = False          # шум уже пережидали: снова — только после устойчивого восстановления
        self._clean_s = 0.0                # сколько секунд подряд оценка шума в норме
        self._stuck_done = None            # залипшее значение, которое ждать больше не будем
        self._last_z_t = None
        self.waited_s = 0.0                # всего простояли за прогон, с
        self.spent = 0.0                   # и потратили на это заряда, ед.
        self.n_waits = 0
        self.n_probes = 0
        # замеры цены простоя и езды
        self._prev = None                  # (t, battery) прошлого такта
        self._still_t = 0.0                # сколько секунд подряд робот стоит
        self._chunk = [0.0, 0.0]           # (секунд, ед.) в недобранном замере простоя
        self._idle = [agent.rules.drain_idle_per_s] * 2     # замеры, ед./с; первые два — из правил, пока своих нет
        self._move = [20.0, 20.0 * self.MOVE_PRIOR]         # (секунд, ед.) на ходу

    # --- что агент сообщает ------------------------------------------------------------------

    def observe(self, obs):
        """Каждый такт: сколько заряда ушло за такт стоя и сколько — на ходу."""
        prev, self._prev = self._prev, (obs.t, obs.battery)
        still = abs(obs.v) < 0.01 and abs(obs.w) < 0.05
        self._still_t = self._still_t + (obs.t - prev[0]) if still and prev else 0.0
        if prev is None or any(e.get('type') == 'hazard_hit' for e in obs.events):
            return                         # разовая потеря в опасной зоне — не цена простоя и не цена езды
        dt, drop = obs.t - prev[0], prev[1] - obs.battery
        if not 0.0 < dt <= 1.0:
            return
        if not still:
            self._move[0] += dt
            self._move[1] += drop
        elif self._still_t > 0.3:          # первые доли секунды после остановки батарея ещё «догоняет» езду
            self._chunk[0] += dt
            self._chunk[1] += drop
            if self._chunk[0] >= self.CHUNK:
                self._idle.append(self._chunk[1] / self._chunk[0])
                self._chunk = [0.0, 0.0]

    def idle_rate(self):
        """Измеренная цена простоя, ед./с: медиана замеров за прогон (разовая утечка её не сдвигает)."""
        return max(0.0, float(np.median(self._idle)))

    def waste_rate(self):
        """Сколько заряда в секунду уходит впустую, если при неисправном датчике ехать, ед./с."""
        return self.WASTE * max(0.0, self._move[1] / self._move[0])

    def trip_idle(self, meters):
        """Сколько прибавить к цене пути длиной meters за расход «за включённость» в дороге.

        Простой ценой до IDLE_COVERED уже покрыт запасом на возврат, с которым агент отлажен, поэтому
        прибавляется только расход сверх него — и только если замеры показывают, что простой заметно дороже.
        При обычной цене простоя поправка равна нулю, и маршруты агента v2 совпадают с прежними.
        """
        idle = self.idle_rate()
        if idle <= 2.0 * self.IDLE_COVERED:
            return 0.0
        return (idle - self.IDLE_COVERED) * meters / self.TRIP_SPEED

    def reading(self, z, obs):
        t = obs.t - (getattr(obs, 'sensor_age', 0.0) or 0.0)      # время измерения, а не доставки
        while self.z and self.z[0][0] < t - 12.0:
            self.z.popleft()
        self.z.append((t, obs.x, obs.y, z))
        dt = 0.0 if self._last_z_t is None else min(max(t - self._last_z_t, 0.0), 2.0)
        self._last_z_t = t
        if self._noisy():
            self._clean_s = 0.0
        elif self._health_informative():
            self._clean_s += dt
            if self._clean_s >= self.RECOVER_S:
                self._noise_block = False  # датчик устойчиво исправен: следующий шум — новый сбой

    def subgoal_ended(self, trigger, t):
        if trigger == 'candidate_lost':
            self.lost.append(t)
        elif trigger == 'sample_collected':
            self.lost.clear()

    # --- оценки ------------------------------------------------------------------------------

    def _noisy(self):
        return self.a.cfg.sensor_health and self.a.health.degraded

    def _health_informative(self):
        """Оценка шума сейчас обновляется: у нуля сигнал обрезан, и молчание оценки — не признак здоровья."""
        buf = getattr(self.a.health, 'buf', None)
        return buf is None or (len(buf) >= 12 and float(np.mean(buf)) >= 0.2)

    def _window(self, t, t0):
        """С какого момента брать показания, чтобы судить о датчике стоя: WINDOW секунд, но не меньше
        WINDOW_N показаний и не раньше, чем робот остановился."""
        times = [v[0] for v in self.z]
        since = min(t - self.WINDOW, times[-self.WINDOW_N] if len(times) >= self.WINDOW_N else -math.inf)
        return max(t0 + 0.5, since, t - self.WINDOW_MAX)

    def _spread(self, since, few=12):
        """Разброс показаний стоящего робота с момента since; None — показаний мало.

        Два способа, берётся больший: полуразмах между 16-й и 84-й долями и расстояние от середины до
        95-й доли. Второй нужен у нуля, где половина шумных показаний обрезана до 0. Нужно few показаний
        за две секунды и больше; если датчик настолько редкий, что их не набирается, — пять за восемь секунд.
        """
        v = [(t, z) for t, _, _, z in self.z if t >= since]
        span = v[-1][0] - v[0][0] if v else 0.0
        if not ((len(v) >= few and span >= 2.0) or (len(v) >= 5 and span >= 8.0)):
            return None
        q16, q50, q84, q95 = np.percentile([z for _, z in v], [16, 50, 84, 95])
        return float(max((q84 - q16) / 2.0, (q95 - q50) / 1.645))

    def _stuck_now(self, t):
        """Показания за последние секунды одинаковы до последнего знака, хотя робот проехал заметный путь."""
        last = [v for v in self.z if v[0] >= t - 2.5]
        if len(last) < 4 or last[-1][0] - last[0][0] < 1.5:
            return False
        moved = math.hypot(last[-1][1] - last[0][1], last[-1][2] - last[0][2])
        same = max(v[3] for v in last) - min(v[3] for v in last) < 1e-9 and 0.0 < last[-1][3] < 1.0
        return same and moved >= 0.12 and last[-1][3] != self._stuck_done

    def _symptom(self, obs):
        if self._noisy() and not self._noise_block:
            return 'noise'
        if self.a.inv is None and self._stuck_now(obs.t):
            return 'stuck'
        if self.lost_limit and len(self.lost) >= self.lost_limit and obs.t - self.lost[0] <= self.LOST_WINDOW \
                and self.n_probes < self.MAX_PROBES:
            return 'streak'
        return None

    def _worth(self, obs, why):
        """Стоит ли пережидать: None — да, иначе причина отказа."""
        idle, waste = self.idle_rate(), self.waste_rate()
        left = min(self.time_budget - self.waited_s,
                   (self.charge_budget - self.spent) / idle if idle > 1e-6 else math.inf)
        if left < self.MIN_WAIT:
            return 'бюджет ожидания на прогон исчерпан'
        # Простой за оставшееся время сбоя стоит idle × время, езда вслепую за то же время теряет waste × время:
        # время сокращается, сравниваются цены секунды.
        if idle >= waste:
            return (f'стоять стоит {idle:.2f} ед./с, а езда по таким показаниям теряет впустую около '
                    f'{waste:.2f} ед./с: ждать невыгодно')
        expect = min(self.TYPICAL, left)           # сколько, скорее всего, придётся простоять
        if obs.battery - idle * expect <= self.a._return_need(obs)[1] + 0.5:
            return 'после ожидания заряда не хватит на дорогу домой'
        return None

    def _decline(self, obs, why, reason):
        """Сбой есть, но ждать его не будем: запомнить, чтобы не решать то же каждый такт."""
        if why == 'noise':
            self._noise_block, self._clean_s = True, 0.0
        else:
            self._stuck_done = self.z[-1][3] if self.z else None
        self.a.journal.add(obs.t, 'decision', f'Датчик образцов неисправен, но ждать не буду: {reason}',
                           tag='sensor_wait_skip', why=why)

    # --- решение -----------------------------------------------------------------------------

    def hold(self, obs):
        """True — этот такт робот стоит и ждёт датчик."""
        a = self.a
        t = obs.t
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
            probe = why == 'streak'
            if probe:
                self.n_probes += 1
                self.lost.clear()
                if self.time_budget - self.waited_s < self.PROBE_S:
                    return False
            else:
                no = self._worth(obs, why)
                if no:
                    self._decline(obs, why, no)
                    return False
                self.n_waits += 1
            self.wait = {'why': why, 't0': t, 'b0': obs.battery, 'probe': probe, 'seen': self._noisy(),
                         'value': self.z[-1][3] if self.z else 0.0}
            cost = f'стоять стоит {self.idle_rate():.2f} ед./с, езда по таким показаниям теряет впустую около ' \
                   f'{self.waste_rate():.2f} ед./с'
            a.journal.add(t, 'decision', {
                'noise': f'Датчик образцов шумит: стою и жду, пока шум спадёт ({cost})',
                'stuck': f'Датчик образцов повторяет одно и то же значение, хотя я еду: стою и жду, пока он оживёт ({cost})',
                'streak': f'Подъезды к кандидатам {self.lost_limit} раза подряд ничего не дали: останавливаюсь на '
                          f'{self.PROBE_S:.1f} с и смотрю, держится ли показание',
            }[why], tag='sensor_probe' if probe else 'sensor_wait', why=why)
            return True

        w = self.wait
        dt = t - w['t0']
        drop = w['b0'] - obs.battery
        w['seen'] = w['seen'] or self._noisy()
        end = None
        if a._returning:
            end = 'пора возвращаться на базу'
        elif a._hazard_t > w['t0']:
            end = 'получен штраф опасной зоны, здесь стоять нельзя'
        elif a.rules.time_limit_s - t < self.TIME_SLACK:
            end = 'до конца прогона мало времени'
        elif self.waited_s + dt >= self.time_budget:
            end = f'за прогон простоял уже {self.waited_s + dt:.0f} с — больше не жду'
        elif self.spent + drop >= self.charge_budget:
            end = f'на ожидание за прогон ушло уже {self.spent + drop:.1f} ед. заряда — больше не жду'
        elif dt >= self.MIN_WAIT and drop / dt > min(self.waste_rate(), 3.0 * self.idle_rate() + 0.05):
            end = f'стоя теряю {drop / dt:.2f} ед./с — ждать дорого'
        elif self.z and t - self.z[-1][0] > self.SILENT:
            end = 'датчик молчит, судить не по чему'
        elif w['probe']:
            end = self._probe(obs, w, dt)
        elif w['why'] == 'noise':
            # Стоя разброс виден напрямую, и этой оценке веры больше, чем оценке на ходу: та у нуля сигнала
            # принимает обрезанный шум за тишину. Оценка на ходу решает, только когда показаний слишком мало.
            spread = self._spread(self._window(t, w['t0']))
            if dt >= self.MIN_WAIT and spread is not None and spread < 1.6 * a.health.nominal:
                end = f'разброс показаний стоя вернулся к {spread:.2f}'
            elif dt >= self.MIN_WAIT and spread is None and w['seen'] and not self._noisy() \
                    and self._health_informative():
                end = 'оценка шума датчика вернулась к норме'
        elif w['why'] == 'stuck':
            if self.z and abs(self.z[-1][3] - w['value']) > 1e-9:
                end = 'показания снова меняются'
        if end is None:
            return True
        self.waited_s += dt
        self.spent += max(0.0, drop)
        self.quiet_until = t + self.COOLDOWN
        self.lost.clear()
        self.wait = None
        if w['why'] == 'noise':
            self._noise_block, self._clean_s = True, 0.0
        if w['why'] == 'stuck':
            self._stuck_done = w['value'] if self.z and abs(self.z[-1][3] - w['value']) < 1e-9 else None
        if not w['probe']:
            # Показаниям, набранным за время сбоя, доверия меньше: карта частично возвращается к незнанию.
            a.belief.relax(0.3 if w['why'] != 'noise' else 0.15)
        a.journal.add(t, 'decision', f"{'Проверка' if w['probe'] else 'Ожидание'} датчика: {dt:.0f} с, "
                      f'{max(0.0, drop):.2f} ед. заряда. {end[0].upper()}{end[1:]}',
                      tag='sensor_probe_end' if w['probe'] else 'sensor_wait_end')
        a._request_plan('sensor_wait')
        return False

    def _probe(self, obs, w, dt):
        """Короткая проверка стоя: либо виден сбой (тогда решаем, ждать ли его), либо едем дальше."""
        a = self.a
        why = None
        spread = self._spread(w['t0'] + 0.5, few=5)
        stand = [z for zt, _, _, z in self.z if zt >= w['t0'] + 0.5]
        if (self._noisy() or (spread is not None and spread > 3.0 * a.health.nominal)) and not self._noise_block:
            why = 'noise'
        elif a.inv is None and spread is not None and max(stand) - min(stand) < 1e-9 and 0.0 < stand[-1] < 1.0 \
                and stand[-1] != self._stuck_done:
            why = 'stuck'                  # у исправного датчика шум есть всегда: совпадение до знака — залип
        if why is None:
            return 'показания стоя устойчивы, еду дальше' if dt >= self.PROBE_S else None
        no = self._worth(obs, why)
        if no:
            self._decline(obs, why, no)
            return no
        w.update(why=why, probe=False, value=stand[-1] if stand else w['value'])
        self.n_waits += 1
        a.journal.add(obs.t, 'decision', 'Проверка стоя показала сбой датчика образцов ('
                      + ('шум' if why == 'noise' else 'показание застыло') + '): жду, пока он пройдёт',
                      tag='sensor_wait', why=why)
        return None

    def _in_risk(self, obs):
        a = self.a
        if not a.hazards:
            return False
        ix, iy = a.arena.w2g(obs.x, obs.y)
        return a._risk_full[iy, ix] > 0.02
