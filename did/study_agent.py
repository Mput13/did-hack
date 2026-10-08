"""Исполнитель исследования по заданию (did/study.py). Интерфейс тот же, что у остальных агентов: tick(obs, io).

Ход работы: постоять у базы и измерить шум показаний батареи → доехать до места → ставить замеры →
после каждого пересчитать оценку и погрешность → выбрать следующий замер → остановиться по условиям
задания → вернуться на базу и вызвать finish.

Замер устроен как опыты исследователя (did/inquiry.py: встать, навестись, измерить), но с окнами
усреднения: робот стоит и усредняет показания батареи, выполняет воздействие (прямой пробег,
разворот или пауза), снова стоит и усредняет. Разность двух средних — расход на воздействие; её шум
известен из разброса показаний на месте.

Оценка. Замеры на обычном полу (контрольные пробеги, паузы, развороты) уточняют модель расхода
did.energy.EnergyModel: заряд = per_m·путь + per_rad·поворот + per_s·время. Искомая величина — либо
коэффициент модели (цена поворота, расход на месте, влияние груза), либо отношение расхода на
пробегах в области к per_m (цена пола). Погрешность — перенос ошибок: шум замеров в области плюс
неуверенность модели (ковариация её коэффициентов). Если повторы расходятся сильнее, чем позволяет
шум прибора, погрешность увеличивается во столько же раз.

Выбор следующего замера. Модель линейная, поэтому погрешность после любого замера известна заранее,
до его результата. Для каждого разрешённого замера считается, на сколько он сузит погрешность и
сколько заряда уйдёт на него вместе с дорогой до его места; берётся лучший по отношению одного к
другому. Чередование области и контроля получается само: дорожает то, что измерено хуже.
"""
import json
import math

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .agent import Agent, AgentConfig
from .energy import EnergyModel
from .inquiry import BATTERY_NOISE, _wrap
from .nav import path_length
from .scenario import Zone
from .science import OTHER, Alternative, Inquiry, TestOption
from .study import (EDGE_M, MEASURE_V, QUANTITIES, RUN_CLEAR_M, SOIL_GUESS, SPIN_W, TRAVEL_FACTOR, TRAVEL_V, _grown,
                    _mask, _mid, control_segment, describe, digits_for, find_segment, plural, prepare, to_markdown)
from .study_law import LawMixin

WIN_S = 0.8             # окно усреднения показаний батареи до и после воздействия, с
CALIB_S = 6.0           # пауза у базы: шум показаний и расход на месте (он входит в поправку каждого замера)
CHECK_S = 3.0           # пауза-проверка после штрафа или странного замера: не идёт ли утечка
NEAR_M = 0.09           # ближе этого к началу участка подъезжать заново не нужно
AIM_TOL = 0.02          # рад: точность наведения перед пробегом (за 0,5 м увод не больше сантиметра)
SURPRISE_Z = 5.0        # замер, который модель не ждала: в подбор не идёт, сначала проверка
CONTROL_LIMIT = 0.12    # контрольный участок дороже номинала на столько — это не обычный пол
CONTROL_DOUBT = 0.04    # дороже на столько — подозрительно: если можно, беру другой участок
CHANGE_REL = 0.03       # пробеги в области разошлись на столько — пол изменился по ходу исследования
MAX_CHANGES = 2         # столько раз можно начать набор заново; дальше расхождение идёт в погрешность
HOME_MARGIN = 1.15      # запас к оценке дороги домой
TIME_PRICE = 0.02       # ед. заряда за секунду: чтобы дешёвые, но долгие паузы не вытесняли остальное
MIN_GAIN = 0.01         # замер, сужающий погрешность меньше чем на 1 %, не стоит заряда
MODE = {'test': 'опыт', 'control': 'контроль', 'calibration': 'калибровка', 'check': 'проверка'}

STUDY_CONFIG = AgentConfig(name='study', detect_change=False, sensor_health=False, localize=False)

STOP_TEXT = {
    'precision': 'требуемая точность достигнута',
    'settled': 'одно объяснение набрало нужную уверенность',
    'budget': 'бюджет заряда исчерпан: на следующий замер и дорогу домой с запасом уже не хватает',
    'max_measurements': 'набран максимум замеров из задания',
    'time': 'истёк лимит времени',
    'no_gain': 'новые замеры почти не сужают погрешность: её ограничивает то, что измерить не разрешено',
    'no_tests': 'опытов, которые различили бы объяснения, не осталось',
    'not_found': 'образец, по которому можно проверить датчик, не найден',
    'no_site': 'для замеров не осталось безопасного места',
    'refused': 'задание не принято',
    'battery': 'батарея села до возврата на базу',
    'timeout': 'вышло время прогона',
}


class StudyAgent(LawMixin, Agent):

    def __init__(self, arena, config, n_samples, rules=None, planner=None, recorder=None, study=None, carried=0):
        super().__init__(arena, config, n_samples, rules=rules, planner=planner, recorder=recorder)
        if not (isinstance(study, dict) and 'plan' in study):       # задание без проверки: проверяем сами
            study = prepare(study, arena, self.rules, carried=carried)
        self.spec, self.plan, self.raw = study['spec'], study['plan'], study.get('raw')
        self.collected = int(study.get('carried') or 0)
        self.refused = [p['text'] for p in self.plan['problems'] if p['level'] == 'error']
        self.failures = []                                          # чего не удалось — словами, для отчёта
        q = self.q = self.spec.quantity if self.spec else None
        self.phase = 'refused' if self.refused else 'search' if q == 'sensor_law' else 'study'
        r = self.rules
        self.model = EnergyModel(r.drain_per_m, r.drain_idle_per_s, noise=BATTERY_NOISE)
        widen = {'turn_cost': ('per_rad', 0.5), 'idle_cost': ('per_s', 0.05), 'load_effect': ('per_m_load', 0.5)}.get(q)
        if widen:
            self.model.widen(*widen)                                # искомый коэффициент — только из замеров
        sites = self.plan['sites']
        self.site = dict(sites['test']) if sites['test'] else None
        self.controls = [dict(c) for c in sites['controls']]
        self.region = Zone('R', **self.plan['region']) if self.plan['region'] else None
        self.measurements = []          # все замеры по порядку
        self.tests = []                 # пробеги в исследуемой области (для цены пола)
        self.learned = []               # что ушло в модель расхода: (признаки, расход, шум)
        self.inq = None                 # сравнение объяснений (did.science.Inquiry), если оно задано
        self.stop = None                # почему остановились
        self.status = 'refused' if self.refused else 'running'
        self.conclusion = ''
        self._step = None
        self._calibrated = q == 'sensor_law'
        self._check_at = None           # пора проверить паузой, не идёт ли утечка
        self._leak = 0                  # сколько проверок подряд показали утечку
        self._doubt = None              # контрольный участок, на котором расход вдруг вырос
        self._ctrl_doubt = 0.0          # на сколько принятый контроль дороже номинала, если заменить его нечем
        self._changes = 0               # сколько раз замечена смена пола в области
        self._noise = [2 * BATTERY_NOISE ** 2 / 2, 2]     # сумма квадратов и число: шум одного показания
        self._b0 = self._t_prev = self._last = None
        self._dt = 0.1
        self._spent = {'travel': 0.0, 'test': 0.0, 'control': 0.0, 'home': 0.0}
        self._bucket = ['travel', None]
        self._win = None
        self._trail_xy = None
        self._travel = [0.0, 0.0]       # заряд и метры в дороге по обычному полу: настоящая цена метра пути
        self._spins = 0
        self._told = set()
        if self.phase == 'search':
            self._law_init()
        elif self.phase == 'study':
            self._trigger = None

    # ======================================================================================
    # такт, восприятие, учёт заряда
    # ======================================================================================

    def tick(self, obs, io):
        if self._b0 is None:
            self._b0 = obs.battery
        if self._t_prev is not None and obs.t > self._t_prev:
            self._dt = obs.t - self._t_prev
        self._t_prev = obs.t
        self._last = obs
        super().tick(obs, io)
        self._charge(obs)

    def spent(self, obs):
        return self._b0 - obs.battery

    def _charge(self, obs):
        """Разнести расход по статьям: дорога, замеры в области, контроль и проверки, возврат."""
        s = self._step
        now = ('home' if self._returning else
               ('test' if s['role'] == 'test' else 'control') if s and s.get('phase') not in (None, 'travel') else 'travel')
        name, since = self._bucket
        if since is None:
            self._bucket = [now, obs.battery]
        elif now != name or obs.done or self.finished:
            self._spent[name] += since - obs.battery
            self._bucket = [now, obs.battery]

    def _perceive(self, obs):
        if self.phase == 'search':
            return super()._perceive(obs)          # штатный поиск образца: карта вероятностей, грунт, опасные зоны
        if obs.scan is not None:
            self._front = float(min(obs.scan[:20].min(), obs.scan[-20:].min()))
        for ev in obs.events:
            self._on_event(ev, obs)
        self._floor(obs)
        if obs.sensor is not None and self.q == 'sensor_law':
            self._on_reading(obs.sensor, obs)      # карта образцов нужна и дальше: где датчик услышит другой образец

    def _on_event(self, ev, obs):
        super()._on_event(ev, obs)
        kind = ev.get('type')
        if kind == 'hazard_hit':
            self._law_penalty(obs)
        if self.phase != 'study' or kind not in ('hazard_hit', 'collision'):
            return
        self._spoil(obs, 'штраф в опасной зоне' if kind == 'hazard_hit' else 'столкновение')
        if kind == 'hazard_hit':
            # Опыт нельзя ставить в зоне: робот сразу пятится из неё, зона с этой секунды объезжается.
            self._grace = (-1e9, None)
            self._sync_hazards(obs.t)
            self._escape = {'until': obs.t + 2.0, 'v': 0.15 if self._last_cmd[0] < 0 else -0.15}
            self._check_at = obs.t + 2.0

    def _floor(self, obs):
        """В дороге: расход на отрезках по 0,3 м показывает, где пол обычный, и сколько на деле стоит метр пути."""
        if self._trail_xy is None or math.hypot(obs.x - self._trail_xy[0], obs.y - self._trail_xy[1]) >= 0.06:
            self._trail_xy = (obs.x, obs.y)
            self._trail_point(obs.x, obs.y, obs.t)
        s = self._step
        if (s and s.get('phase') not in (None, 'travel')) or self._leak or self._escape:
            self._win = None                       # замер считается отдельно, утечка — не свойство пола
            return
        w = self._win
        if w is None:
            self._win = {'x0': obs.x, 'y0': obs.y, 'x': obs.x, 'y': obs.y, 'th': obs.th, 't0': obs.t, 'b0': obs.battery,
                         'ds': 0.0, 'dth': 0.0}
            return
        w['ds'] += math.hypot(obs.x - w['x'], obs.y - w['y'])
        w['dth'] += abs(_wrap(obs.th - w['th']))
        w['x'], w['y'], w['th'] = obs.x, obs.y, obs.th
        if w['ds'] < 0.3:
            return
        spent = w['b0'] - obs.battery
        ratio = float(np.clip(self.model.soil_ratio(w['ds'], w['dth'], obs.t - w['t0'], self.collected, spent), 0.2, 7.0))
        if w['dth'] / w['ds'] <= 2.0:              # на крутых поворотах расход говорит о повороте, а не о поле
            self.soil.add((w['x0'] + obs.x) / 2, (w['y0'] + obs.y) / 2, w['ds'], ratio)
            self._cost_dirty = True
        if ratio < 1.6:
            self._travel[0] += spent
            self._travel[1] += w['ds']
        self._win = None

    def _per_m(self):
        """Заряд на метр дороги: модель плюс то, что на деле уходит на повороты и простой."""
        base = self.model.per_meter(self.collected)
        if self._travel[1] >= 1.0:
            return max(base, self._travel[0] / self._travel[1])
        return base * TRAVEL_FACTOR

    def _home_time(self, x, y):
        self._home_cost(x, y)                      # строит поле расстояний до базы, если его ещё нет
        return self.home_graph.energy(self._base_dist, self._base_pred, x, y) / TRAVEL_V + 3.0

    # ======================================================================================
    # решения верхнего уровня
    # ======================================================================================

    def _deliberate(self, obs):
        if self.phase == 'search':
            return super()._deliberate(obs)
        self._trigger = None                       # в исследовании план ведёт не планировщик, а _next

    def _check_return(self, obs):
        """Запас на возврат: дорога домой с запасом и неприкосновенный остаток из задания."""
        if self._returning or self.phase == 'refused':
            return
        home = self._home_cost(obs.x, obs.y)
        left = self.spec.budget.energy - self.spent(obs)
        if left < home * 1.05 + 0.5 * self.spec.budget.reserve or obs.battery < home * 1.2 + 1.5:
            self._spoil(obs, 'заряд на исходе')
            return self._conclude(obs, 'budget')
        limit = min(self.spec.stop.time_s, self.rules.time_limit_s - 8.0)
        if obs.t + self._home_time(obs.x, obs.y) > limit + 15.0:     # замер, начатый вовремя, дать закончить
            self._spoil(obs, 'время на исходе')
            self._conclude(obs, 'time')

    def _act(self, obs, io):
        if self.phase == 'refused':
            if not self.finished:
                self.journal.add(obs.t, 'decision', 'Задание не принято: ' + '; '.join(self.refused))
                io.command(0.0, 0.0)
                io.finish()
                self.finished, self.mode, self.stop = True, 'done', 'refused'
            return
        if self._returning or self.phase == 'search':
            if self.phase == 'search' and self._returning and not self.stop:
                self._conclude(obs, 'not_found', go_home=False)     # штатный поиск сдался: целей по заряду нет
            return super()._act(obs, io)
        if self._escape:
            if obs.t < self._escape['until']:
                self.mode = 'escape'
                return self._command(io, self._escape['v'], 0.0)
            self._escape, self._path_goal, self._stuck = None, None, None
        if self._step is None:
            self._step = self._next(obs)
            if self._step is None:                 # исследование закончено, дальше дорога домой
                return self._command(io, 0.0, 0.0)
        self._run_measure(self._step, obs, io)

    def _next(self, obs):
        if not self._calibrated:
            return self._pause_step(obs, None, 'calibration', CALIB_S)
        if self._check_at is not None:
            if self.q != 'sensor_law' and not self._relocate(obs):
                return self._conclude(obs, 'no_site')
            return self._pause_step(obs, None, 'check', CHECK_S)
        if self.q == 'sensor_law':
            return self._next_law(obs)
        return self._decide(obs)

    # --- что уже есть и чего не хватает ------------------------------------------------------

    def _count(self, role, kind=None):
        return sum(1 for m in self.measurements if m['used'] and m['role'] == role and (kind is None or m['kind'] == kind))

    def _control(self):
        return next((c for c in self.controls if not c.get('bad')), None)

    def _required(self):
        """Чего не хватает до минимального набора замеров из задания: (роль, вид) или None."""
        a = self.spec.allowed
        if self.q == 'soil_cost':
            if self._control() and self._count('control', 'straight') < a.straight.repeats:
                return 'control', 'straight'
            if sum(1 for t in self.tests if t['used']) < a.straight.repeats:
                return 'test', 'straight'
        elif self.q == 'load_effect' and self._count('test', 'straight') < a.straight.repeats:
            return 'test', 'straight'
        elif self.q == 'turn_cost' and self._count('test', 'spin') < a.spin.repeats:
            return 'test', 'spin'
        elif self.q == 'idle_cost' and self._count('test', 'pause') < a.pause.repeats:
            return 'test', 'pause'
        return None

    def _options(self, obs):
        """Все замеры, разрешённые заданием, с ценой и ожидаемой пользой."""
        a, out = self.spec.allowed, []
        if self.q == 'soil_cost':
            if self.site:
                out.append(self._leg_step(obs, self.site, 'test'))
            if self._control():
                out.append(self._leg_step(obs, self._control(), 'control'))
            if a.pause:
                out.append(self._pause_step(obs, None, 'control', a.pause.seconds))
        elif self.q == 'load_effect':
            if self._control():
                out.append(self._leg_step(obs, self._control(), 'test'))
            if a.pause:
                out.append(self._pause_step(obs, None, 'control', a.pause.seconds))
        elif self.q == 'turn_cost':
            out.append(self._spin_step(obs, self.site))
            if a.pause:
                out.append(self._pause_step(obs, self.site, 'control', a.pause.seconds))
        elif self.q == 'idle_cost':
            out.append(self._pause_step(obs, self.site, 'test', a.pause.seconds))
            if self._control():
                out.append(self._pause_step(obs, self._control(), 'control', a.pause.seconds))
        return [c for c in out if c is not None]

    def _fits(self, obs, c, worst=True):
        """Хватит ли заряда и времени на замер, дорогу до него и возврат с запасом. None — хватит."""
        b = self.spec.budget
        need = c['e_travel'] + (c['e_worst'] if worst else c['e_meas']) + self._home_cost(*c['after']) * HOME_MARGIN + b.reserve
        if need > b.energy - self.spent(obs) or need > obs.battery - 1.0:
            return 'budget'
        limit = min(self.spec.stop.time_s, self.rules.time_limit_s - 8.0)
        if obs.t + c['t_travel'] + c['t_meas'] + self._home_time(*c['after']) > limit:
            return 'time'
        return None

    def _afford_control(self, obs):
        """Хватит ли бюджета на контрольные пробеги и после них — на пробеги в области и дорогу домой.

        Если нет, контроль пропускается: оценка против номинала полезнее, чем контроль без оценки.
        """
        site, n = self._control(), self.spec.allowed.straight.repeats
        ctrl, test = self._leg_step(obs, site, 'control'), self._leg_step(obs, self.site, 'test')
        if ctrl is None or test is None:
            return True
        hop, e = self.graph.plan(ctrl['after'], test['start'])
        if hop is None:
            return True
        need = (ctrl['e_travel'] + n * ctrl['e_meas'] + e * self._per_m() + n * test['e_meas']
                + self._home_cost(*test['after']) * HOME_MARGIN + self.spec.budget.reserve)
        if need <= self.spec.budget.energy - self.spent(obs):
            return True
        for c in self.controls:
            c.setdefault('bad', 'на него не хватило заряда')
        left = self.spec.budget.energy - self.spent(obs)
        if self._count('control', 'straight'):
            self._warn('skipctrl', 'на повтор контрольного пробега заряда не хватило: контроль измерен без повтора')
            self.journal.add(obs.t, 'decision', f'На повтор контрольного пробега и замеры в области бюджета не хватит (нужно около '
                             f'{need:.0f} ед., осталось {left:.0f}): еду в область, контроль остаётся без повтора')
        else:
            self._warn('skipctrl', 'на контрольный участок заряда не хватило: робот поехал сразу в область, сравнение идёт '
                                   'с номиналом из условия, а он известен хуже')
            self.journal.add(obs.t, 'decision', f'Бюджета не хватит и на контрольные пробеги, и на замеры в области (нужно около '
                             f'{need:.0f} ед., осталось {left:.0f}). Контроль пропускаю: оценка против номинала из условия '
                             'полезнее, чем контроль без оценки')
        return False

    def _decide(self, obs):
        """Остановиться или выбрать следующий замер: наибольшее сужение погрешности на единицу заряда."""
        st, est, need = self.spec.stop, self.estimate(), self._required()
        if est and not need and est['rel'] <= st.rel_error:
            return self._conclude(obs, 'precision')
        if self._count('test') + self._count('control') >= st.max_measurements:
            return self._conclude(obs, 'max_measurements')
        if need == ('control', 'straight') and not self.tests and not self._afford_control(obs):
            need = self._required()                # на контроль заряда не хватит: сразу в область, сравнение с номиналом
        cands = self._options(obs)
        if need:
            cands = [c for c in cands if (c['role'], c['kind']) == need] or cands
        best, blocked = None, set()
        for c in cands:
            why = self._fits(obs, c)
            if why:
                blocked.add(why)
                continue
            c['gain'] = self._gain(c)
            c['score'] = c['gain'] / (c['e_travel'] + c['e_meas'] + 0.05 + TIME_PRICE * (c['t_travel'] + c['t_meas']))
            useful = need or not est or c['gain'] >= MIN_GAIN * est['sigma']
            if useful and (best is None or c['score'] > best['score']):
                best = c
        if best is None:                           # полезного и доступного замера нет: что именно мешает
            return self._conclude(obs, 'no_site' if not cands else 'budget' if 'budget' in blocked else
                                  'time' if 'time' in blocked else 'no_gain')
        what = self._name(best)
        if need or not est:
            why = 'это часть минимального набора замеров'
        else:
            d = digits_for(est['sigma'])
            why = (f"погрешность сейчас ±{1.96 * est['sigma']:.{d}f}, после него будет около "
                   f"±{1.96 * (est['sigma'] - best['gain']):.{d}f}; цена {best['e_travel'] + best['e_meas']:.1f} ед.")
            rest = [c for c in cands if c is not best and 'score' in c]
            if rest:
                other = max(rest, key=lambda c: c['score'])
                why += (f" Другой вариант — {self._name(other)} — сузил бы до ±{1.96 * (est['sigma'] - other['gain']):.{d}f} "
                        f"за {other['e_travel'] + other['e_meas']:.1f} ед.: на единицу заряда это хуже")
        text = f'Следующий замер: {what}. Почему: {why}.'
        self.journal.add(obs.t, 'decision', text)
        if self.rec:
            goal = best.get('start') or (obs.x, obs.y)
            self.rec.add_plan(obs.t, 'rule', 'subgoal_done', text, [{'type': 'goto', 'x': goal[0], 'y': goal[1]}])
        return best

    def _name(self, c):
        where = {'test': 'в исследуемой области', 'control': f"на контрольном участке {c['site']}" if c['site'] else 'на месте'}
        if c['kind'] == 'straight':
            return f"пробег {c['length']:.2f} м {where.get(c['role'], '')}".strip()
        if c['kind'] == 'spin':
            return f"разворот на {math.degrees(c['angle']):.0f}° {'против часовой' if c['dir'] > 0 else 'по часовой'}"
        return f"пауза {c['win'] * 3:.0f} с" + (' для сравнения' if c['role'] == 'control' else '')

    # --- заготовки замеров -------------------------------------------------------------------

    def _sigma_read(self):
        return max(math.sqrt(self._noise[0] / self._noise[1]), 0.002)

    def _sigma_spent(self, win):
        return self._sigma_read() * math.sqrt(2.0 / max(1, round(win / self._dt)))

    def _pause_step(self, obs, site, role, seconds, listen=False):
        start = (site['x'], site['y']) if site else None
        return self._priced({'kind': 'pause', 'role': role, 'site': site['id'] if site else None, 'start': start,
                             'win': seconds / 3, 'hold': seconds / 3, 'listen': listen}, obs)

    def _spin_step(self, obs, site):
        angle = math.radians(self.spec.allowed.spin.angle_deg)
        return self._priced({'kind': 'spin', 'role': 'test', 'site': site['id'], 'start': (site['x'], site['y']),
                             'win': WIN_S, 'angle': angle, 'dir': 1 if self._spins % 2 == 0 else -1}, obs)

    def _leg_step(self, obs, site, role):
        a, b = site['a'], site['b']
        if math.dist((obs.x, obs.y), b) < math.dist((obs.x, obs.y), a):
            a, b = b, a                            # обратный пробег начинается там, где закончился прямой
        return self._priced({'kind': 'straight', 'role': role, 'site': site['id'], 'start': tuple(a), 'end': tuple(b),
                             'win': WIN_S, 'length': site['length']}, obs)

    def _priced(self, s, obs):
        """Дописать к замеру ожидаемые путь, поворот и время, шум, расход заряда и дорогу до его места."""
        here = (obs.x, obs.y)
        start = s['start'] or here
        if math.dist(here, start) <= NEAR_M:
            s['e_travel'] = s['t_travel'] = 0.0
        else:
            pts, e = self.graph.plan(here, start)
            if pts is None:
                return None
            s['e_travel'], s['t_travel'] = e * self._per_m(), path_length(pts) / TRAVEL_V + 1.5
        k = k_hi = 1.0
        aim = 0.0
        if s['kind'] == 'straight':
            feat = (s['length'], 0.0, s['length'] / MEASURE_V + 0.45 + s['win'])
            aim = 2.0 * max(float(self.model.mean[2]), 0.1)          # наведение перед пробегом — не часть замера
            if self.q == 'soil_cost' and s['role'] == 'test':
                est = self.estimate()
                k, k_hi = (est['value'], est['value'] + 2 * est['sigma']) if est else (sum(SOIL_GUESS) / 2, SOIL_GUESS[1])
        elif s['kind'] == 'spin':
            feat = (0.0, s['angle'], s['angle'] / SPIN_W + 0.45 + s['win'])
        else:
            feat = (0.0, 0.0, s['hold'] + s['win'])
        s['do'], s['feat'], s['sigma'] = 'measure', feat, self._sigma_spent(s['win'])
        s['e_meas'] = self.model.predict(*feat, self.collected, mult=k)[0] + aim
        s['e_worst'] = self.model.predict(*feat, self.collected, mult=k_hi)[0] + aim
        s['t_meas'] = feat[2] + s['win'] + (3.0 if s['kind'] == 'straight' else 0.5)
        s['after'] = s.get('end') or start
        return s

    # ======================================================================================
    # оценка и её погрешность
    # ======================================================================================

    def _parts(self, cov=None, extra=0.0):
        """(оценка, дисперсия от шума пробегов в области, дисперсия от неуверенности модели) при ковариации cov.

        extra — добавка информации от ещё одного пробега в области: (длина / шум)².
        """
        m = self.model
        C = m.cov if cov is None else cov
        mean, n = m.mean, self.collected
        if self.q == 'turn_cost':
            return (float(mean[2]), 0.0, max(C[2, 2], 0.0)) if self._count('test', 'spin') else None
        if self.q == 'idle_cost':
            return (float(mean[3]), 0.0, max(C[3, 3], 0.0)) if self._count('test', 'pause') else None
        if self.q == 'load_effect':
            if not self._count('test', 'straight') or n <= 0:
                return None
            g = np.array([-mean[1] / mean[0] ** 2, 1.0 / mean[0], 0.0, 0.0])
            return float(mean[1] / mean[0]), 0.0, max(float(g @ C @ g), 0.0)
        legs = [t for t in self.tests if t['used']]
        if not legs:
            return None
        w = np.array([1.0 / t['noise'] ** 2 for t in legs])
        ds, dth, dt = (np.array([t[k] for t in legs]) for k in ('ds', 'dth', 'dt'))
        y = np.array([t['spent'] for t in legs]) - mean[2] * dth - mean[3] * dt
        per_m = m.per_meter(n)
        S = float((w * ds ** 2).sum())
        k = float((w * ds * y).sum()) / (per_m * S)
        g = np.array([-k / per_m, -k * n / per_m, -float((w * ds * dth).sum()) / (per_m * S),
                      -float((w * ds * dt).sum()) / (per_m * S)])
        return k, 1.0 / (per_m ** 2 * (S + extra)), max(float(g @ C @ g), 0.0)

    def _target(self, cov=None, extra=0.0):
        """(оценка, погрешность 1σ по шуму прибора) — без поправки на разброс повторов: так считается польза замера."""
        got = self._parts(cov, extra)
        return None if got is None else (got[0], math.sqrt(got[1] + got[2]))

    @staticmethod
    def _excess(chi, nu):
        """Во сколько раз разброс больше шума прибора; 1 — согласуется с шумом (проверка хи-квадрат на уровне 1 %)."""
        return 1.0 if nu < 1 or chi <= chi2_dist.ppf(0.99, nu) else math.sqrt(chi / nu)

    def _birge(self):
        """Разброс повторов против шума прибора: (для пробегов в области, для замеров, уточнявших модель)."""
        m, model, site = self.model, 1.0, 1.0
        if len(self.learned) > 1:
            X = np.array([x for x, _, _ in self.learned])
            y, s = (np.array([v[i] for v in self.learned]) for i in (1, 2))
            fitted = int(((np.abs(X) > np.array([0.05, 0.05, 0.5, 1e-9])).any(axis=0)).sum())
            model = self._excess(float((((y - X @ m.mean) / s) ** 2).sum()), len(self.learned) - fitted)
        legs = [t for t in self.tests if t['used']]
        if len(legs) > 1 and self.q == 'soil_cost':
            k = self._parts()[0] * m.per_meter(self.collected)
            site = self._excess(sum(((t['spent'] - m.mean[2] * t['dth'] - m.mean[3] * t['dt'] - k * t['ds']) / t['noise']) ** 2
                                    for t in legs), len(legs) - 1)
        return site, model

    def estimate(self):
        """Текущая оценка: value, sigma (1σ), ci95, rel (полуширина 95% интервала в долях оценки), spread."""
        got = self._parts()
        if got is None:
            return None
        value, var_site, var_model = got
        b_site, b_model = self._birge()            # каждая часть погрешности растёт от разброса своих повторов
        spread = max(b_site, b_model)
        sigma = math.sqrt(b_site ** 2 * var_site + b_model ** 2 * var_model
                          + (value * self._ctrl_doubt) ** 2)             # сомнительный контроль — тоже погрешность
        half = 1.96 * sigma
        return {'value': value, 'sigma': sigma, 'ci95': [value - half, value + half],
                'rel': half / abs(value) if abs(value) > 1e-9 else math.inf, 'spread': spread}

    def _gain(self, c):
        """На сколько замер сузит погрешность (1σ). Для линейной модели это известно до его результата."""
        now = self._target()
        if now is None:
            return math.inf
        if self.q == 'soil_cost' and c['role'] == 'test':
            after = self._target(extra=(c['feat'][0] / c['sigma']) ** 2)
        else:
            x = EnergyModel.features(*c['feat'], self.collected)
            cx = self.model.cov @ x
            after = self._target(cov=self.model.cov - np.outer(cx, cx) / (float(x @ cx) + c['sigma'] ** 2))
        return max(0.0, now[1] - after[1])

    # ======================================================================================
    # один замер: подъехать → навестись → встать и усреднить → воздействие → встать и усреднить
    # ======================================================================================

    def _run_measure(self, m, obs, io):
        ph = m.setdefault('phase', 'travel')
        if ph == 'travel':
            self.mode = 'travel'
            start = m['start']
            if 'here' not in m:                    # уже на месте (обратный пробег, замер на месте) — ехать не нужно
                m['here'] = start is None or math.dist((obs.x, obs.y), start) <= NEAR_M
            if m['here'] or self._drive_to(obs, io, start, tol=0.04):
                self._path_goal = None
                self._command(io, 0.0, 0.0)
                m.update(phase='aim' if m['kind'] == 'straight' else 'settle', t_ph=obs.t)
                if m['kind'] == 'straight':        # курс — на дальний конец участка, длина — полная
                    m['heading'] = math.atan2(m['end'][1] - obs.y, m['end'][0] - obs.x)
            return
        self.mode = MODE[m['role']]
        still = abs(obs.v) < 0.01 and abs(obs.w) < 0.02
        if ph == 'aim':
            err = _wrap(m['heading'] - obs.th)
            if abs(err) <= AIM_TOL:
                m.update(phase='settle', t_ph=obs.t)
                return self._command(io, 0.0, 0.0)
            return self._command(io, 0.0, float(np.clip(2.5 * err, -1.5, 1.5)))
        if ph == 'settle':
            self._command(io, 0.0, 0.0)
            if still and obs.t - m['t_ph'] >= 0.2:
                m.update(phase='pre', t_ph=obs.t, x0=obs.x, y0=obs.y, x=obs.x, y=obs.y, th=obs.th, path=0.0, dth=0.0,
                         pre=[], post=[], trace=[], zs=[], n=self.collected)
            return
        m['path'] += math.hypot(obs.x - m['x'], obs.y - m['y'])
        m['dth'] += abs(_wrap(obs.th - m['th']))
        m['x'], m['y'], m['th'] = obs.x, obs.y, obs.th
        if m.get('listen') and obs.sensor is not None:
            m['zs'].append(obs.sensor)
        dt = obs.t - m['t_ph']
        if ph == 'pre':
            self._command(io, 0.0, 0.0)
            m['pre'].append((obs.t, obs.battery))
            if dt >= m['win'] - 1e-6:
                m.update(phase='go', t_ph=obs.t)
        elif ph == 'go':
            m['trace'].append((m['path'], obs.battery))
            if m['kind'] == 'pause':
                self._command(io, 0.0, 0.0)
                done = dt >= m['hold'] - 1e-6
            elif m['kind'] == 'straight':
                blocked = self._front < 0.16 or dt > m['length'] / MEASURE_V + 4.0
                # Курс на пробеге не подправляется: поворот тоже стоит заряда и попал бы в цену метра.
                self._command(io, 0.0 if blocked else MEASURE_V, 0.0)
                done = m['path'] >= m['length'] - 0.012 or blocked
            else:
                self._command(io, 0.0, m['dir'] * SPIN_W)
                done = m['dth'] >= m['angle'] - 0.08 or dt > m['angle'] / SPIN_W + 3.0
            if done:
                m.update(phase='stop', t_ph=obs.t)
                self._command(io, 0.0, 0.0)
        elif ph == 'stop':
            self._command(io, 0.0, 0.0)
            if still and dt >= 0.2:
                m.update(phase='post', t_ph=obs.t)
        else:
            self._command(io, 0.0, 0.0)
            m['post'].append((obs.t, obs.battery))
            if dt >= m['win'] - 1e-6:
                self._step = None
                self._win = None
                self._finish_measure(m, obs)

    def _spoil(self, obs, why):
        """Замер сорван (штраф, столкновение, пора домой): в расчёт он не идёт, но в отчёте остаётся."""
        m, self._step = self._step, None
        if not m or m.get('phase') in (None, 'travel', 'aim', 'settle'):
            return
        rec = self._record_of(m, obs, obs.t, obs.t, 0.0, self._sigma_spent(m['win']))
        rec.update(used=False, note=f'сорван: {why}')
        self.measurements.append(rec)

    def _record_of(self, m, obs, t0, t1, spent, noise):
        chord = math.hypot(obs.x - m['x0'], obs.y - m['y0'])
        return {'n': len(self.measurements) + 1, 't': round(t1, 1), 't0': round(t0, 2), 't1': round(t1, 2),
                'kind': 'listen' if m.get('listen') else m['kind'], 'role': m['role'], 'site': m['site'],
                'x': round((m['x0'] + obs.x) / 2, 3), 'y': round((m['y0'] + obs.y) / 2, 3),
                'x0': round(m['x0'], 3), 'y0': round(m['y0'], 3), 'x1': round(obs.x, 3), 'y1': round(obs.y, 3),
                # прямой пробег меряется хордой: она не зависит от дрожания оценки позы
                'ds': chord if m['kind'] == 'straight' else m['path'], 'dth': m['dth'], 'dt': t1 - t0,
                'spent': spent, 'noise': noise, 'carried': m['n'], 'used': True, 'note': '',
                'value': None, 'sigma': None, 'unit': '', 'chart': False}

    def _finish_measure(self, m, obs):
        (t0, b0), (t1, b1) = (np.mean(m[k], axis=0) for k in ('pre', 'post'))
        for k in ('pre', 'post'):                  # шум одного показания — по соседним разностям в окнах на месте
            d = np.diff([b for _, b in m[k]])
            self._noise[0] += float((d ** 2).sum()) / 2
            self._noise[1] += len(d)
        noise = self._sigma_read() * math.sqrt(1.0 / len(m['pre']) + 1.0 / len(m['post']))
        rec = self._record_of(m, obs, float(t0), float(t1), float(b0 - b1), noise)
        self.measurements.append(rec)
        if m['kind'] == 'spin':
            self._spins += 1
        if m['kind'] == 'straight' and rec['ds'] < 0.8 * m['length']:
            rec.update(used=False, note='пробег прерван: впереди преграда')
        elif m.get('listen'):
            self._account_listen(rec, m, obs)
        else:
            if m['kind'] == 'straight':
                rec['halves'] = self._halves(m['trace'])
            self._account(rec, obs)
        est = self.estimate()
        if est:
            rec['estimate'] = {'value': round(est['value'], 5), 'sigma': round(est['sigma'], 5)}

    def _halves(self, trace):
        """Расход на метр в первой и второй половине пробега: одинаков ли пол вдоль него. None — данных мало."""
        if len(trace) < 12:
            return None
        s, b = (np.array(v) for v in zip(*trace))
        out = []
        for part in (s <= s[-1] / 2, s > s[-1] / 2):
            if part.sum() < 5 or np.ptp(s[part]) < 0.05:
                return None
            slope = np.polyfit(s[part], b[part], 1)[0]
            sd = self._sigma_read() / (math.sqrt(part.sum()) * float(np.std(s[part])))
            out.append((-float(slope), sd))
        (c1, s1), (c2, s2) = out
        uneven = abs(c1 - c2) > max(0.15 * (c1 + c2) / 2, 5.0 * math.hypot(s1, s2))
        return {'first': round(c1, 3), 'second': round(c2, 3), 'uneven': bool(uneven)}

    # --- что замер значит ---------------------------------------------------------------------

    def _learn(self, rec):
        x = EnergyModel.features(rec['ds'], rec['dth'], rec['dt'], rec['carried'])
        self.model.noise = rec['noise']
        self.model.learn(rec['ds'], rec['dth'], rec['dt'], rec['carried'], rec['spent'])
        self.learned.append((x, rec['spent'], rec['noise']))

    def _seen(self, kind):
        """Были ли уже учтённые замеры этого вида: без них прогноз модели — только исходное предположение."""
        col = {'straight': 0, 'spin': 2, 'pause': 3}[kind]
        return any(x[col] > (0.5 if col == 2 else 1e-9) and (col != 3 or (x[0] < 0.05 and x[2] < 0.5)) for x, _, _ in self.learned)

    def _account(self, rec, obs):
        m, t = self.model, obs.t
        role, kind = rec['role'], rec['kind']
        m.noise = rec['noise']
        pred, sd = m.predict(rec['ds'], rec['dth'], rec['dt'], rec['carried'])
        over = rec['spent'] - pred
        if role == 'calibration':
            self._calibrated = True
            self._learn(rec)
            self.journal.add(t, 'observe', f"Калибровка: показания батареи шумят на ±{self._sigma_read():.3f} ед.; усредняя их "
                             f"{WIN_S:.1f} с до и после воздействия, расход на замер знаю с точностью ±{self._sigma_spent(WIN_S):.3f} ед. "
                             f"На месте уходит {rec['spent'] / rec['dt']:.3f} ед/с")
            return
        if role == 'check':
            if over > max(0.04 * rec['dt'], 4.0 * sd):
                self._leak += 1
                rec.update(used=False, note='идёт утечка заряда')
                if self._leak == 1:
                    self.journal.add(t, 'alarm', f"На месте уходит {rec['spent'] / rec['dt']:.2f} ед/с вместо {pred / rec['dt']:.2f}: "
                                     'батарея теряет заряд сама по себе. Жду, пока утечка кончится, — иначе она попадёт в замеры')
                if self._leak >= 20:
                    self.failures.append('утечка заряда не прекратилась за минуту: замеры после неё могут быть завышены')
                    self._leak, self._check_at = 0, None
                self._doubt = None                 # странный замер объясняется утечкой, участок ни при чём
                return
            if self._leak:
                self.journal.add(t, 'observe', f'Утечка прекратилась (ждал {self._leak * CHECK_S:.0f} с): продолжаю замеры')
            elif self._doubt is not None and not self._doubt.get('bad'):
                self._drop_control(self._doubt, 'расход на нём вырос, а утечки нет — пол там изменился', obs)
            self._leak, self._check_at, self._doubt = 0, None, None
            return self._learn(rec)
        test_leg = role == 'test' and kind == 'straight' and self.q == 'soil_cost'
        if not test_leg and self._seen(kind) and over > SURPRISE_Z * sd and over > 0.08 * max(pred, 0.1):
            # Замер, которого модель не ждала, в подбор не идёт: сначала проверка, не утечка ли это.
            rec.update(used=False, note='расход не сошёлся с прежними замерами')
            self._check_at = t
            self.journal.add(t, 'alarm', f"{self._name_rec(rec)}: ушло {rec['spent']:.2f} ед. при прогнозе {pred:.2f}. "
                             'Замер не учитываю и проверяю паузой, не идёт ли утечка')
            if role == 'control' and kind == 'straight':
                self._doubt = self._control()
            return
        if kind == 'straight':
            return self._account_leg(rec, obs, test_leg)
        self._learn(rec)
        if role == 'test':
            self._inq_record(rec, obs)
        self._say(rec, obs)

    def _account_leg(self, rec, obs, test_leg):
        m, t = self.model, obs.t
        per_m = (rec['spent'] - m.mean[2] * rec['dth'] - m.mean[3] * rec['dt']) / rec['ds']
        rec['per_m'] = round(float(per_m), 4)
        uneven = bool(rec.get('halves') and rec['halves']['uneven'])
        if not test_leg:
            site = self._control()
            dev = per_m / self.rules.drain_per_m - 1.0
            can_swap = len(self.controls) < self.spec.controls.max
            n = rec['carried']
            limit = CONTROL_LIMIT + 0.08 * n
            if self.q == 'soil_cost' and (dev > limit or uneven or (n == 0 and dev > CONTROL_DOUBT and can_swap)):
                why = ('расход вдоль него неодинаков' if uneven and dev <= limit else
                       f'метр пути на нём стоит {per_m:.2f} ед. — на {dev:.0%} дороже номинала {self.rules.drain_per_m:g}')
                rec.update(used=False, note='участок не годится как контроль')
                return self._drop_control(site, why, obs)
            if self.q == 'soil_cost' and dev > CONTROL_DOUBT:
                self._ctrl_doubt = max(self._ctrl_doubt, dev)
                self._warn('ctrl', f'контрольный участок на {dev:.0%} дороже номинала, а другого задание не разрешает: '
                                   'оценка может быть занижена, погрешность увеличена на эту величину')
            self._learn(rec)
            if rec['role'] == 'test':
                self._inq_record(rec, obs)
            return self._say(rec, obs)
        # Пробег в исследуемой области: сначала — не изменился ли пол с прошлых пробегов.
        rec['noise_m'] = rec['noise'] / rec['ds']
        old = [x for x in self.tests if x['used']]
        if old:
            prev = float(np.mean([x['per_m'] for x in old]))
            if self._changes < MAX_CHANGES and abs(per_m - prev) > max(CHANGE_REL * prev, 6.0 * rec['noise_m']):
                self._changes += 1
                for x in old:
                    x.update(used=False, note='до изменения пола')
                self._warn('change', f'на {t:.0f}-й секунде расход в области изменился: было {prev:.2f} ед/м, стало {per_m:.2f}. '
                                     'Оценка — по замерам после изменения')
                self.journal.add(t, 'alarm', f'Расход в области изменился: {prev:.2f} → {per_m:.2f} ед/м. Прежние пробеги '
                                 'не учитываю, набираю новые', tag='model_mismatch')
                self.inq = None
        if uneven:
            self._warn('uneven', 'расход вдоль пробега в области неодинаков: оценка — среднее по мерному участку')
        self.tests.append(rec)
        self.soil.add(rec['x'], rec['y'], rec['ds'], float(np.clip(per_m / m.per_meter(rec['carried']), 0.2, 7.0)))
        self._cost_dirty = True
        self._inq_record(rec, obs)
        self._say(rec, obs)

    def _warn(self, key, text):
        if key not in self._told:
            self._told.add(key)
            self.failures.append(text)

    def _drop_control(self, site, why, obs):
        """Контрольный участок оказался не обычным полом: взять другой, если задание разрешает."""
        site['bad'] = why
        text = f"Контрольный участок {site['id']} не годится: {why}."
        if len(self.controls) < self.spec.controls.max and site['kind'] == 'segment':
            near = _mid(self.site) if self.site else (obs.x, obs.y)
            seg = control_segment(self.arena, self.region, self.spec.allowed.straight.length_m, near,
                                  taken=[_mid(c) for c in self.controls], bad=self._bad_floor())
            if seg:
                self.controls.append({'id': f'C{len(self.controls) + 1}', 'kind': 'segment', 'by': 'auto', **seg})
                text += f" Беру другой — около ({_mid(seg)[0]:.1f}; {_mid(seg)[1]:.1f})."
        if not self._control():
            text += ' Другого контрольного участка нет: сравниваю с номиналом из условия.'
            self._warn('noctrl', f"контрольный участок {site['id']} оказался не обычным полом ({why}), а другого нет: "
                                 'сравнение идёт с номиналом из условия, он известен хуже')
        self.journal.add(obs.t, 'alarm', text)

    def _bad_floor(self):
        """Клетки, где опыты ставить нельзя: опасная зона или пол, который колёса уже показали дорогим."""
        return (self._risk > 0.05) | ((self.soil.mult_grid() > 1.3) & (self.soil.confidence_grid() > 0.3))

    def _relocate(self, obs):
        """После штрафа: если место опытов попало в опасную зону — перенести его. False — переносить некуда."""
        risky = self._risk > 0.05
        if not risky.any():
            return True

        def hit(site):
            pts = [site['a'], site['b'], _mid(site)] if site['kind'] == 'segment' else [(site['x'], site['y'])]
            return any(risky[self.arena.w2g(*p)[::-1]] for p in pts)

        for c in self.controls:
            if not c.get('bad') and hit(c):
                self._drop_control(c, 'он попал в опасную зону', obs)
        s = self.site
        if s is None or not hit(s):
            return True
        if s['kind'] == 'segment':
            ok = _mask(self.arena, _grown(self.region, -EDGE_M)) & (self.arena.clear >= RUN_CLEAR_M) & ~risky
            seg = find_segment(self.arena, ok, self.spec.allowed.straight.length_m, (self.region.x, self.region.y))
            if seg is None:
                self.failures.append('исследуемая область накрыта опасной зоной: безопасного места для пробега в ней нет')
                return False
            self.site.update(seg)
            self._warn('moved', 'мерный участок пришлось перенести: прежний попал в опасную зону')
        else:
            X, Y = self.arena.cell_centers()
            safe = np.argwhere(self.graph.ok & (self.arena.clear >= 0.25) & ~risky
                               & (self.region is None or _mask(self.arena, self.region)))
            if not len(safe):
                self.failures.append('место опытов попало в опасную зону, а другого в заданной области нет')
                return False
            iy, ix = min(safe, key=lambda c: math.hypot(X[c[0], c[1]] - s['x'], Y[c[0], c[1]] - s['y']))
            s.update(x=float(X[iy, ix]), y=float(Y[iy, ix]))
        self.journal.add(obs.t, 'decision', 'Место опытов попало в опасную зону: переношу его')
        return True

    # --- слова и сравнение объяснений ---------------------------------------------------------

    def _name_rec(self, rec):
        what = {'straight': f"Пробег {rec['ds']:.2f} м", 'spin': f"Разворот на {math.degrees(rec['dth']):.0f}°",
                'pause': f"Пауза {rec['dt']:.1f} с"}[rec['kind']]
        where = {'test': ' в области' if rec['kind'] == 'straight' else '',
                 'control': f" на контрольном участке {rec['site']}" if rec['kind'] == 'straight' else ' для сравнения'}
        return what + where.get(rec['role'], '')

    def _say(self, rec, obs):
        est = self.estimate()
        text = f"{self._name_rec(rec)}: ушло {rec['spent']:.3f} ± {rec['noise']:.3f} ед. за {rec['dt']:.1f} с"
        if 'per_m' in rec:
            text += f", это {rec['per_m']:.2f} ед/м"
        if est:
            d = digits_for(est['sigma'])
            text += (f". Оценка: {est['value']:.{d}f} ± {1.96 * est['sigma']:.{d}f} {self.spec.unit} "
                     f"(±{est['rel']:.1%} при требуемых ±{self.spec.stop.rel_error:.1%})")
            if est['spread'] > 1.0:
                text += f"; повторы расходятся в {est['spread']:.1f} раза сильнее шума прибора — погрешность увеличена"
        self.journal.add(obs.t, 'observe', text)

    def _single(self, rec):
        """Значение искомой величины по одному замеру и его погрешность — для сравнения объяснений."""
        m = self.model
        sd = np.sqrt(np.maximum(np.diag(m.cov), 0.0))
        if rec['kind'] == 'spin':
            return (rec['spent'] - m.mean[3] * rec['dt']) / rec['dth'], math.hypot(rec['noise'], sd[3] * rec['dt']) / rec['dth']
        if rec['kind'] == 'pause':
            return rec['spent'] / rec['dt'], rec['noise'] / rec['dt']
        per_m, n = m.per_meter(rec['carried']), rec['carried']
        if self.q == 'load_effect':
            return (rec['per_m'] / m.mean[0] - 1.0) / n, math.hypot(rec['noise'] / rec['ds'] / m.mean[0], sd[0] / m.mean[0]) / n
        k = rec['per_m'] / per_m
        return k, math.hypot(rec['noise'] / rec['ds'] / per_m, k * sd[0] / per_m)

    def _inq_record(self, rec, obs):
        hyps = self.spec.hypotheses
        if not hyps:
            return
        if self.inq is None:
            alts = [Alternative(h.id, h.statement, h.prior) for h in hyps]
            alts.append(Alternative(OTHER, 'значение не из этого списка', 0.1 * sum(h.prior for h in hyps)))
            self.inq = Inquiry('S1', obs.t, 'study',
                               {'text': f'задание: выяснить, {self.spec.label}', 'x': rec['x'], 'y': rec['y'],
                                'observed': 0.0, 'expected': 0.0, 'unit': self.spec.unit},
                               alts, [], accept=self.spec.stop.confidence, max_tests=99, energy_budget=1e9, source='study')
        value, sigma = self._single(rec)
        preds = {h.id: (h.value, h.sigma or 0.05 * abs(h.value) + 1e-3) for h in hyps}
        test = TestOption(f"m{rec['n']}", f"замер {rec['n']}: {self._name_rec(rec).lower()}", cost=rec['spent'],
                          duration_s=round(rec['dt'], 1), unit=self.spec.unit, predictions=preds,
                          action={'kind': rec['kind']}, sigma=sigma)
        test.gain_bits = round(self.inq.gain(test), 3)
        self.inq.tests.append(test)
        self.inq.record(test.id, value, sigma, obs.t)

    # ======================================================================================
    # итог
    # ======================================================================================

    def _conclude(self, obs, reason, go_home=True):
        """Закончить замеры: зафиксировать причину и вывод, ехать на базу. Всегда возвращает None."""
        if self.stop:
            return None
        self.stop = reason
        est = self.estimate() if self.q != 'sensor_law' else None
        spec = self.spec
        if self.q == 'sensor_law':
            self._law_close(obs, reason)
        elif est is None:
            self.status = 'failed'
            self.conclusion = f'Оценку получить не удалось: {STOP_TEXT[reason]}.'
        else:
            d = digits_for(est['sigma'])
            head = (f"{QUANTITIES[self.q][3]}: {est['value']:.{d}f} ± {1.96 * est['sigma']:.{d}f} {spec.unit} "
                    f"(95% интервал {est['ci95'][0]:.{d}f}…{est['ci95'][1]:.{d}f})")
            n = self._count('test') + self._count('control')
            if reason == 'precision':
                self.status = 'done'
                self.conclusion = (f"{head}. Требуемая точность ±{spec.stop.rel_error:.1%} достигнута: получено "
                                   f"±{est['rel']:.1%} за {n} {plural(n, 'замер', 'замера', 'замеров')}.")
            elif est['rel'] <= spec.stop.rel_error:
                self.status = 'incomplete'         # по шуму прибора точность есть, но повторов меньше, чем просили
                need = self._required()
                what = ('контрольных пробегов' if need and need[0] == 'control' else 'замеров в исследуемом месте')
                self.conclusion = (f"{head}. Погрешность по шуму прибора уже ±{est['rel']:.1%} при требуемых "
                                   f"±{spec.stop.rel_error:.1%}, но набор замеров из задания неполный: не хватило {what} — "
                                   f"{STOP_TEXT[reason]}. Без повторов результат не проверен на воспроизводимость.")
            else:
                self.status = 'not_reached'
                self.conclusion = (f"Точность не достигнута: получено ±{est['rel']:.1%} при требуемых ±{spec.stop.rel_error:.1%} — "
                                   f"{STOP_TEXT[reason]}. {head}.")
            if self.q == 'soil_cost':
                z = (est['value'] - 1.0) / est['sigma']
                self.conclusion += (' Пол в области дороже обычного, отличие значимо.' if z > 3 else
                                    ' Пол в области дешевле обычного, отличие значимо.' if z < -3 else
                                    ' В пределах погрешности пол в области не отличается от обычного.')
            if self.inq is not None:
                c = self.inq.close(obs.t, 'оценка внесена в отчёт')
                self.conclusion += f" Из объяснений: {c['text']}."
        self.journal.add(obs.t, 'verdict' if self.status == 'done' else 'decision', 'Вывод исследования. ' + self.conclusion)
        if go_home:
            self._go_home(obs.t, STOP_TEXT[reason])
        return None

    def _site_out(self, s):
        if s is None:
            return None
        out = {k: v for k, v in s.items() if k in ('id', 'kind', 'x', 'y', 'length', 'by', 'bad')}
        if s['kind'] == 'segment':
            out.update(a=[float(v) for v in s['a']], b=[float(v) for v in s['b']])
        return out

    def study_report(self, reason=None):
        """Отчёт для записи прогона: задание, план, замеры, оценка, сравнение с контролем, вывод, цена."""
        spec, obs = self.spec, self._last
        if spec is None or self.refused:
            report = {'version': 1, 'spec': spec.to_dict() if spec else (self.raw or {}), 'status': 'refused',
                      'task': describe(spec) if spec else '', 'plan': self._plan_out(), 'measurements': [], 'estimate': None,
                      'control': None, 'inquiry': None, 'conclusion': 'Задание не принято, робот остался на базе.',
                      'stop': {'reason': 'refused', 'text': STOP_TEXT['refused']}, 'failures': list(self.refused),
                      'energy': {'budget': spec.budget.energy if spec else None, 'spent': 0.0}, 'time_s': 0.0}
            report['markdown'] = to_markdown(report)
            return report
        if not self.finished and reason in ('battery', 'timeout'):      # судья закончил прогон раньше робота
            if not self.stop:
                self._conclude(obs, reason, go_home=False)
            self.status, self.stop = 'failed', reason
            self.failures.append(f'робот не вернулся на базу: {STOP_TEXT[reason]}')
        est = self._law_estimate() if self.q == 'sensor_law' else self.estimate()
        self._finalize_values()
        spent = self.spent(obs)
        if spent > spec.budget.energy + 0.05:
            self.failures.append(f'бюджет превышен на {spent - spec.budget.energy:.1f} ед.: дорога домой вышла дороже оценки')
        skipped = [m for m in self.measurements if not m['used']]
        if skipped:
            self.failures.append(f"{len(skipped)} из {len(self.measurements)} замеров не учтены: "
                                 + '; '.join(sorted({m['note'] for m in skipped})))
        spec_out = spec.to_dict()
        if self.q == 'sensor_law' and not spec_out.get('hypotheses'):      # объяснения не заданы: взяты стандартные
            spec_out['hypotheses'] = [x.model_dump(exclude_none=True) for x in self._hyps]
        report = {
            'version': 1, 'spec': spec_out, 'status': self.status, 'task': describe(spec),
            'plan': self._plan_out(),
            'measurements': [self._meas_out(m) for m in self.measurements],
            'estimate': None if est is None else {
                'value': round(est['value'], 6), 'sigma': round(est['sigma'], 6),
                'ci95': [round(v, 6) for v in est['ci95']], 'rel_error': round(est['rel'], 5), 'unit': est.get('unit', spec.unit),
                'label': est.get('label', QUANTITIES[self.q][3]), 'spread': round(est.get('spread', 1.0), 2),
                'target': spec.stop.rel_error},
            'control': self._control_out(),
            'inquiry': self.inq.to_dict() if self.inq is not None else None,
            'conclusion': self.conclusion,
            'stop': {'reason': self.stop, 'text': STOP_TEXT.get(self.stop, self.stop)},
            'failures': list(dict.fromkeys(self.failures + [p['text'] for p in self.plan['problems']
                                                             if p['level'] == 'warning' and p['field'] != 'budget'])),
            'energy': {'budget': spec.budget.energy, 'reserve': spec.budget.reserve, 'spent': round(spent, 2),
                       **{k: round(v, 2) for k, v in self._spent.items()}, 'battery_left': round(obs.battery, 2)},
            'time_s': round(obs.t, 1),
            'energy_model': self.model.summary(),
        }
        report = json.loads(json.dumps(report, ensure_ascii=False, default=float))      # без типов numpy
        report['markdown'] = to_markdown(report)
        return report

    def _plan_out(self):
        return {'steps': self.plan['steps'], 'region': self.plan['region'], 'route': self.plan['route'],
                'costs': self.plan['costs'], 'problems': self.plan['problems'],
                'sites': {'test': self._site_out(self.site), 'controls': [self._site_out(c) for c in self.controls],
                          'sample': getattr(self, '_sample', None)}}

    def _finalize_values(self):
        """Значение каждого замера в единицах итоговой величины — для таблицы и графика «область против контроля»."""
        m = self.model
        per_m = m.per_meter(self.collected)
        for r in self.measurements:
            if r['kind'] == 'listen' or r['dt'] <= 0:
                continue
            if r['kind'] == 'straight' and self.q == 'soil_cost' and r['ds'] > 0.05:
                c = (r['spent'] - m.mean[2] * r['dth'] - m.mean[3] * r['dt']) / r['ds']
                r.update(per_m=round(float(c), 4), value=float(c / per_m), sigma=r['noise'] / r['ds'] / per_m,
                         unit='× к обычному полу', chart=True)
            elif r['kind'] == 'straight' and r['ds'] > 0.05:
                c = (r['spent'] - m.mean[2] * r['dth'] - m.mean[3] * r['dt']) / r['ds']
                r.update(per_m=round(float(c), 4), value=float(c), sigma=r['noise'] / r['ds'], unit='ед/м',
                         chart=self.q == 'load_effect')
            elif r['kind'] == 'spin' and r['dth'] > 0.3:
                r.update(value=float((r['spent'] - m.mean[3] * r['dt']) / r['dth']), sigma=r['noise'] / r['dth'], unit='ед/рад',
                         chart=self.q == 'turn_cost')
            elif r['kind'] == 'pause':
                r.update(value=r['spent'] / r['dt'], sigma=r['noise'] / r['dt'], unit='ед/с', chart=self.q == 'idle_cost')

    def _meas_out(self, m):
        out = {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()
               if k not in ('noise_m',)}
        return out

    def _control_out(self):
        """Сравнение с контролем словами и числами."""
        m, s = self.model, self.model.summary()
        if self.q == 'soil_cost':
            est = self.estimate()
            if est is None:
                return None
            c, sd = s['per_m']['value'], s['per_m']['sigma']
            nominal = self.rules.drain_per_m
            n_c = self._count('control', 'straight')
            text = (f"В области метр пути стоит {est['value'] * c:.2f} ед., "
                    + (f"на контрольном участке — {c:.2f} ± {1.96 * sd:.2f} ед. по {n_c} {plural(n_c, 'пробегу', 'пробегам', 'пробегам')} "
                       f"(номинал из условия {nominal:g}). "
                       if n_c else f"контрольных пробегов нет, за обычный пол взят номинал из условия {nominal:g} ± {1.96 * sd:.2f} ед. ")
                    + f"Отношение — {est['value']:.2f}.")
            checks = [r for r in self.measurements if r['role'] in ('check', 'calibration') or r['kind'] == 'pause']
            if any(r['note'] == 'идёт утечка заряда' for r in checks):
                text += ' По ходу была утечка заряда: замеры ставились после того, как паузы показали, что она кончилась.'
            elif n_c:
                text += ' Оба участка мерились одним способом и вперемежку, поэтому разница — свойство места, а не утечка или дрейф.'
            return {'unit': 'ед/м', 'test': {'value': round(est['value'] * c, 4)}, 'control': {'value': c, 'sigma': sd},
                    'nominal': nominal, 'vs_nominal': round(c / nominal, 4), 'n_control': n_c, 'text': text}
        if self.q == 'turn_cost':
            spins = [r for r in self.measurements if r['used'] and r['kind'] == 'spin']
            if not spins:
                return None
            idle = s['per_s']
            spin, dt = float(np.mean([r['spent'] for r in spins])), float(np.mean([r['dt'] for r in spins]))
            n_p = self._count('control', 'pause') + 1
            text = (f"Разворот на {math.degrees(np.mean([r['dth'] for r in spins])):.0f}° стоит {spin:.2f} ед. за {dt:.1f} с; "
                    f"простоять столько же стоит {idle['value'] * dt:.2f} ед. (на месте уходит {idle['value']:.3f} ± "
                    f"{1.96 * idle['sigma']:.3f} ед/с по {n_p} {plural(n_p, 'паузе', 'паузам', 'паузам')}). Разница — цена самого поворота.")
            return {'unit': 'ед. за замер', 'test': {'value': round(spin, 4)}, 'control': {'value': round(idle['value'] * dt, 4)},
                    'idle': idle, 'text': text}
        if self.q == 'idle_cost':
            groups = {}
            for r in self.measurements:
                if r['used'] and r['kind'] == 'pause':
                    groups.setdefault('control' if r['role'] == 'control' else 'test', []).append(r)
            if not groups.get('control') or not groups.get('test'):
                return None
            rate = {k: (sum(r['spent'] for r in v) / sum(r['dt'] for r in v),
                        math.sqrt(sum(r['noise'] ** 2 for r in v)) / sum(r['dt'] for r in v)) for k, v in groups.items()}
            z = (rate['test'][0] - rate['control'][0]) / math.hypot(rate['test'][1], rate['control'][1])
            text = (f"На основном месте уходит {rate['test'][0]:.4f} ± {1.96 * rate['test'][1]:.4f} ед/с, на контрольном — "
                    f"{rate['control'][0]:.4f} ± {1.96 * rate['control'][1]:.4f} ед/с: "
                    + ('различие в пределах погрешности, от места расход не зависит.' if abs(z) < 2.5
                       else 'различие значимо — расход на месте зависит от места или менялся со временем.'))
            return {'unit': 'ед/с', 'test': {'value': round(rate['test'][0], 5), 'sigma': round(rate['test'][1], 5)},
                    'control': {'value': round(rate['control'][0], 5), 'sigma': round(rate['control'][1], 5)}, 'text': text}
        if self.q == 'load_effect':
            return {'unit': 'ед/м', 'nominal': self.rules.drain_per_m,
                    'text': f'Сравнение — с номинальным расходом без груза из условия ({self.rules.drain_per_m:g} ед/м): '
                            'снять груз и проехать пустым робот не может.'}
        return self._law_control()

    def export(self):
        """То, что кладётся в запись прогона рядом с отчётом: расследование в формате, знакомом интерфейсу."""
        return {'inquiries': [self.inq.to_dict()] if self.inq is not None else [], 'energy_model': self.model.summary()}
