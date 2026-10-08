"""Исследователь внутри агента: следит, сходятся ли измерения с моделью, и ведёт расследования.

Работает при правилах с несколькими причинами одного симптома (did.config.SCIENCE): расход растёт
от дорогого грунта, от поворотов, от груза и от утечки после сбоя; датчик образцов может зашуметь,
залипнуть или начать занижать показания. Заметив странность, исследователь не выбирает одно
объяснение, а открывает расследование (did/science.py): несколько объяснений, опыт, который их
различит, измерение, вывод. Опыты — короткие манёвры: постоять, проехать прямо, развернуться.
"""
import math
from collections import deque
from types import SimpleNamespace

import numpy as np

from .energy import EnergyModel
from .penalty import PenaltyLedger
from .science import OTHER, Alternative, Inquiry, TestOption

WINDOW_M = 0.12        # окно для оценки расхода в движении, м
STILL_S = 1.0          # окно на месте, с
PAUSE_S = 2.0          # опыт «постоять»
STRAIGHT_M = 0.30      # опыт «проехать прямо»
SPIN_RAD = 1.5         # опыт «развернуться на месте»
BATTERY_NOISE = 0.03   # шум разности двух показаний батареи (уточняется на старте)
FAULT_S = 25.0         # сколько, по умолчанию, длится сбой (уточняется памятью между прогонами)
ONSET_S = 5.0          # сколько секунд после штрафа сравниваются две версии карты образцов
SOIL_MAX = 7.0         # самый дорогой грунт, какой агент допускает (то же ограничение, что у оценок для карты)
BIAS = 0.2             # предполагаемое занижение показаний при сбое
# Перевес версии «занижает» над версией «исправен» (логарифм отношения правдоподобий) за ONSET_S секунд:
# чего ждать при каждом состоянии датчика. Измерено на отладочных сценариях 1–40 (tools/onset_calibration.py).
ONSET = {'ok': (-13.3, 3.5), 'bias': (6.1, 6.1), 'noise': (-8.3, 9.1), 'stuck': (-2.6, 7.9)}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Investigator:

    def __init__(self, agent, knowledge=None, roles=None):
        self.a = agent
        self.roles = roles             # языковая модель в ролях автора и критика (did/llm_roles.py); None — без неё
        self.know = knowledge or {}
        r = agent.rules
        self.model = EnergyModel(r.drain_per_m, r.drain_idle_per_s, noise=BATTERY_NOISE,
                                 prior=self.know.get('energy'))
        self.inquiries = []            # все расследования прогона
        self.active = None             # текущее
        self.run = None                # идущий опыт
        self.leak = None               # подтверждённая утечка: {'rate', 't0', 'until', 'low'}
        self.sensor = {'mode': 'ok'}   # ok | noise | stuck | bias
        self.sites = []                # места, где дорогой грунт уже подтверждён опытом
        self.reserve_until = (0.0, -1e9)
        self.durations = []            # длительности закончившихся сбоев: [(вид, секунды)]
        self._win = None
        self._held = []                # закрытые окна, которые ждут решения по своим отрезкам (did/penalty.py)
        self._closed = False           # прогон закончен: странности только записываются, расследований больше нет
        # Разовая потеря заряда при штрафе — не расход на путь (did/penalty.py). Её размер берётся из правил,
        # в которые верит агент; расход на самом дорогом грунте — из модели расхода.
        self.penalty = PenaltyLedger(r.hazard_battery_hit, lambda ds, dth, dt, t: self.model.predict(
            ds, dth, dt, self.a.collected, mult=SOIL_MAX)[0] + self._leak_rate(t) * dt)
        self._th = None
        self._susp = []                # подряд идущие окна, которых модель не ждала
        self._cusum = 0.0
        self._checkup_at = None        # когда провести проверку после штрафа
        self.companion = None          # расследование датчика, которое делит паузу с проверкой батареи
        self._noise_pending = False    # тревога о шуме пришла во время другого расследования
        self._recheck_until = -1e9     # датчик не удалось проверить (сигнал у нуля): проверить, когда появится
        self._rest_ok_t = -1e9         # когда опыт «постоять» последний раз показал, что утечки нет
        self._path = [0.0, 0.0]        # пройдено метров и накручено радиан: сколько поворотов приходится на метр
        self._wait_from = None         # с какого времени робот стоит и ждёт, пока оживёт залипший датчик
        self._pre_penalty = 0.0        # среднее показание датчика перед штрафом
        self._onset = None             # сравнение двух версий карты образцов после штрафа (см. _track_onset)
        self._doubt_until = -1e9       # до какого времени к датчику остаётся вопрос, который закроет сверка у образца
        self._quiet_until = 0.0
        self._penalty_t = -1e9
        self._collect_t = -1e9
        self._z = deque(maxlen=14)     # (t, x, y, показание)
        self._trail_xy = None
        self._told = set()
        self._fault_seen_after_penalty = False

    # ======================================================================================
    # расход: окна и странности
    # ======================================================================================

    def observe(self, obs):
        """Каждый такт: копить окно пути и, когда оно закрылось, сверять расход с моделью."""
        a = self.a
        if self._trail_xy is None or math.hypot(obs.x - self._trail_xy[0], obs.y - self._trail_xy[1]) >= 0.06:
            self._trail_xy = (obs.x, obs.y)
            a._trail_point(obs.x, obs.y, obs.t)
        # Потеря заряда при штрафе сопоставляется с событием штрафа; окно ждёт решения только по своим отрезкам.
        self.penalty.reading(obs, self._win if self.run is None else None,
                             dth=abs(_wrap(obs.th - self._th)) if self._th is not None else 0.0)
        self._th = obs.th
        self._release(obs)
        if self._win is None or self.run is not None:
            self._open(obs)            # во время опыта окно не копится: у опыта свой замер
            return
        w = self._win
        w['ds'] += math.hypot(obs.x - w['x'], obs.y - w['y'])
        w['dth'] += abs(_wrap(obs.th - w['th']))
        w['x'], w['y'], w['th'] = obs.x, obs.y, obs.th
        dt = obs.t - w['t0']
        if w['ds'] >= WINDOW_M or (dt >= STILL_S and w['ds'] < 0.03) or dt >= 2.0:
            w.update(t1=obs.t, b1=obs.battery)
            self._held.append(w)
            self._release(obs)
            self._open(obs)

    def _release(self, obs):
        """Разобрать закрытые окна по порядку. Обычно окно разбирается в том же такте, в котором закрылось.

        Задержка бывает только у окна, в котором есть необъяснённый скачок заряда или рядом с которым событие
        штрафа ещё не нашло своей потери: такое окно ждёт не дольше, чем показание и событие могут разойтись.
        Более поздние скачки его не держат. Загрязнённое окно (потеря заряда не сошлась с правилами,
        см. did/penalty.py) в тревогу, карту грунта и модель расхода не идёт.
        """
        while self._held and not self.penalty.waits(self._held[0]):
            w = self._held.pop(0)
            if not w.get('bad'):
                self._window(w, obs)

    def finish(self, obs):
        """Прогон закончен: всё, что ждало решения о потере заряда, досчитывается по тому, что известно.

        obs — последнее наблюдение, обычным путём оно не разбиралось: его события штрафа и показание батареи
        сначала идут в учёт штрафов. Новых расследований после конца нет (см. _window): ни опытов, ни обращений
        к языковой модели. Повторные вызовы ничего не делают.
        """
        if self._closed:
            return
        self._closed = True
        for ev in obs.events:
            if ev.get('type') == 'hazard_hit':
                self.penalty.event(obs.t)
        self.penalty.reading(obs, None, dth=abs(_wrap(obs.th - self._th)) if self._th is not None else 0.0)
        self.penalty.flush(obs.t)
        self._release(obs)

    def _open(self, obs):
        self._win = {'x0': obs.x, 'y0': obs.y, 'x': obs.x, 'y': obs.y, 'th': obs.th, 't0': obs.t, 'b0': obs.battery,
                     'ds': 0.0, 'dth': 0.0, 'n': self.a.collected}

    def _near_site(self, x, y):
        return any(math.hypot(x - sx, y - sy) <= 0.9 for sx, sy in self.sites)

    def _leak_rate(self, t):
        return self.leak['rate'] if self.leak and t <= self.leak['until'] + 6.0 else 0.0

    def _window(self, w, obs):
        """Разбор закрытого окна. Замер — из самого окна; obs — текущий такт (окно могло быть придержано)."""
        a, m = self.a, self.model
        ds, dth, n, dt = w['ds'], w['dth'], w['n'], w['t1'] - w['t0']
        self._path[0] += ds
        self._path[1] += dth
        mx, my = (w['x0'] + w['x']) / 2, (w['y0'] + w['y']) / 2
        spent = w['b0'] - w['b1'] - self._leak_rate(w['t1']) * dt
        moving = ds >= 0.05
        mult, conf = a.soil.at(mx, my) if (a.cfg.learn_soil and moving) else (1.0, 1.0)
        known = conf >= 0.4 or self._near_site(mx, my)
        mean, sd = m.predict(ds, dth, dt, n, mult=mult if known else 1.0)
        if known and moving:
            sd = math.hypot(sd, 0.3 * mult * m.per_meter(n) * ds)        # оценка грунта сама неточна
        z = (spent - mean) / sd
        if self.leak:
            self._track_leak(w['b0'] - w['b1'], m.predict(ds, dth, dt, n, mult=mult if known else 1.0)[0], dt, obs.t)
        if self.active is not None or obs.t < self._quiet_until:
            return
        # Странность — это либо одно-два резких расхождения, либо слабое, но устойчивое (накопленная сумма).
        self._cusum = max(0.0, self._cusum + z - 1.0)
        if z > 2.5 or (self._cusum > 0.0 and z > 1.0):
            self._susp.append({'ds': ds, 'dth': dth, 'dt': dt, 'spent': spent, 'mean': mean, 'x': mx, 'y': my, 'n': n,
                               't': w['t0'], 'known': known, 'mult': mult, 'moving': moving})
            strong = sum(1 for q in self._susp[-2:] if (q['spent'] - q['mean']) > 0) >= 2 and z > 2.5
            if (strong and len(self._susp) >= 2) or z > 8.0 or self._cusum >= 5.0:
                self._cusum = 0.0
                if self._closed:
                    text = self._anomaly(self._susp)['text']
                    self._susp = []
                    a.journal.add(obs.t, 'inquiry', f'Странность: {text}. Замечено в конце прогона, проверить '
                                  'не успел: расследование не открываю', tag='late_anomaly')
                else:
                    self._open_energy(obs)
                return
        elif self._cusum == 0.0:
            self._susp.clear()
        # В обучение идут все окна в пределах ±2,5 разброса — и «дороже», и «дешевле» прогноза. Если брать
        # только те, что не дороже, оценка расхода систематически занижается.
        if abs(z) > 2.5:
            if z < -3.0 and known and mult > 1.4 and moving and a.cfg.learn_soil:
                a._on_model_mismatch('дешевле', mx, my, m.soil_ratio(ds, dth, dt, n, spent), mult, obs.t)
            return
        if moving and a.cfg.learn_soil:
            ratio = float(np.clip(m.soil_ratio(ds, dth, dt, n, spent), 0.2, 7.0))
            a.soil.add(mx, my, ds, ratio)
            a._cost_dirty = True
        if not moving or not known or mult < 1.25:
            m.learn(ds, dth, dt, n, spent)                 # обычный пол: уточняем коэффициенты модели
            self._discoveries(obs.t)

    def _discoveries(self, t):
        """Коэффициент, о котором не было известно, стал значимо больше нуля — это открытие."""
        s = self.model.summary()
        for name, text, unit in (('per_rad', 'повороты тоже стоят заряда', 'ед. на радиан'),
                                 ('per_m_load', 'каждый несомый образец удорожает метр пути', 'ед. на метр за образец')):
            v = s[name]
            if name not in self._told and v['value'] > 4 * v['sigma'] and v['value'] > 0.02:
                self._told.add(name)
                self.a.journal.open(t, f'law:{name}', f"{text}: около {v['value']:.2f} {unit}",
                                    'коэффициент должен подтверждаться на следующих отрезках', law=name)
                self.a.journal.close(t, f'law:{name}', 'confirmed',
                                     f"оценка {v['value']:.3f} ± {v['sigma']:.3f} по {self.model.n} отрезкам; "
                                     'учтено в модели расхода')

    # ======================================================================================
    # расследование расхода
    # ======================================================================================

    def _open_energy(self, obs):
        a, m, S = self.a, self.model, self._susp
        tot = lambda k: sum(s[k] for s in S)      # noqa: E731
        ds, dth, dt = tot('ds'), tot('dth'), tot('dt')
        excess = tot('spent') - tot('mean')
        n, last = S[-1]['n'], S[-1]
        per_m, per_s, per_rad = m.per_meter(n), float(m.mean[3]), float(m.mean[2])
        moving = ds > 0.03
        rate = max(excess / dt, 0.02)
        base = last['mult'] if last['known'] else 1.0       # что модель уже считала множителем этого места
        ratio = base + excess / (per_m * ds) if moving else None
        turn = per_rad + excess / dth if dth > 0.6 and dth / max(ds, 0.05) > 2.5 else None
        known_normal = last['known'] and last['mult'] < 1.25
        penalty = obs.t - self._penalty_t < 45.0
        if moving and turn is None and obs.t - self._rest_ok_t < 60.0 and self._penalty_t < self._rest_ok_t:
            # Утечку недавно исключил опыт, штрафов с тех пор не было: вывод «грунт» делается по исключению,
            # без нового опыта. Он записывается как расследование и проверяется наравне с остальными.
            what = 'грунт здесь изменился и стал дороже' if known_normal else 'здесь дорогой грунт'
            q = Inquiry(self._qid(), obs.t, 'energy',
                        {'text': f'расход {tot("spent") / ds:.1f} ед/м при прогнозе {tot("mean") / ds:.1f}',
                         'x': round(last['x'], 2), 'y': round(last['y'], 2), 'observed': round(tot('spent') / ds, 2),
                         'expected': round(tot('mean') / ds, 2), 'unit': 'ед/м'},
                        [Alternative('soil', f'{what}, примерно ×{ratio:.1f}', 0.9),
                         Alternative(OTHER, 'причина не из этого списка', 0.1)], [])
            q.note = (f'Без нового опыта: утечку исключила пауза {obs.t - self._rest_ok_t:.0f} с назад, '
                      'штрафов с тех пор не было.')
            self._start(q, obs, held=list(S), est={'rate': rate, 'ratio': ratio, 'turn': None,
                                                   'known_normal': known_normal})
            self._conclude(obs)
            return
        p_leak = self.know.get('p_leak_after_penalty', 0.5) if penalty else 0.12

        alts = [Alternative('leak', f'батарея теряет около {rate:.2f} ед/с сама по себе (сбой)', p_leak)]
        if moving:
            what = 'грунт здесь изменился и стал дороже' if known_normal else 'здесь дорогой грунт'
            alts.append(Alternative('soil', f'{what}, примерно ×{ratio:.1f}', 0.25 if known_normal else 0.6))
        if turn is not None:
            alts.append(Alternative('turn', f'повороты стоят около {turn:.2f} ед/рад, а не {per_rad:.2f}',
                                    0.3 if dth / max(ds, 0.05) > 4.0 else 0.06))
        alts.append(Alternative(OTHER, 'причина не из этого списка', 0.1))
        ids = {x.id for x in alts}

        def preds(**kw):
            return {k: v for k, v in kw.items() if k in ids}

        tests = [TestOption('rest', f'постоять {PAUSE_S:.0f} секунды и измерить расход на месте',
                            cost=per_s * PAUSE_S + 0.05, duration_s=PAUSE_S, unit='ед/с',
                            predictions=preds(leak=(per_s + rate, 0.35 * rate + 0.02), soil=(per_s, 0.012),
                                              turn=(per_s, 0.012)),
                            action={'kind': 'pause'}, sigma=BATTERY_NOISE / PAUSE_S)]
        heading = self._free_heading(obs)
        if moving and heading is not None:
            tests.append(TestOption('straight', f'проехать прямо {STRAIGHT_M:.1f} м без поворотов',
                                    cost=per_m * STRAIGHT_M * (0.6 * ratio + 0.4), duration_s=STRAIGHT_M / 0.15,
                                    unit='× к обычному полу',
                                    predictions=preds(soil=(ratio, 0.3 * ratio), leak=(1.0 + rate / (per_m * 0.15), 0.35),
                                                      turn=(1.0, 0.15)),
                                    action={'kind': 'straight', 'heading': heading},
                                    sigma=BATTERY_NOISE / (per_m * STRAIGHT_M)))
        if turn is not None:
            tests.append(TestOption('spin', 'развернуться на месте на 90° и измерить расход на поворот',
                                    cost=max(per_rad, 0.1) * SPIN_RAD + 0.05, duration_s=SPIN_RAD, unit='ед/рад',
                                    predictions=preds(turn=(turn, 0.3 * turn + 0.03), soil=(per_rad, 0.05),
                                                      leak=(per_rad + rate, 0.35 * rate + 0.04)),
                                    action={'kind': 'spin'}, sigma=BATTERY_NOISE / SPIN_RAD))
        self._start(Inquiry(self._qid(), obs.t, 'energy', self._anomaly(S), alts, tests,
                            energy_budget=self._budget(obs)),
                    obs, held=list(S), est={'rate': rate, 'ratio': ratio, 'turn': turn, 'known_normal': known_normal})

    @staticmethod
    def _anomaly(S):
        """Странность словами и числами: сколько ушло на подозрительных окнах и сколько ждала модель."""
        tot = lambda k: sum(s[k] for s in S)      # noqa: E731
        ds, dt, last = tot('ds'), tot('dt'), S[-1]
        if ds > 0.03:
            text = f'расход {tot("spent") / ds:.1f} ед/м при прогнозе {tot("mean") / ds:.1f}'
            anomaly = {'observed': round(tot('spent') / ds, 2), 'expected': round(tot('mean') / ds, 2), 'unit': 'ед/м'}
        else:
            text = f'на месте уходит {tot("spent") / dt:.2f} ед/с при прогнозе {tot("mean") / dt:.2f}'
            anomaly = {'observed': round(tot('spent') / dt, 3), 'expected': round(tot('mean') / dt, 3), 'unit': 'ед/с'}
        anomaly.update(text=text, x=round(last['x'], 2), y=round(last['y'], 2))
        return anomaly

    def _budget(self, obs):
        a = self.a
        home = a._home_cost(obs.x, obs.y)
        spare = obs.battery - home * a.cfg.reserve_margin - a.cfg.reserve_abs
        return float(np.clip(spare - 1.0, 0.3, 2.5))     # на стоячие опыты заряд есть всегда

    def _free_heading(self, obs):
        """Направление, в котором есть 45 см свободного пола: сначала текущий курс, потом соседние."""
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
            th = obs.th + off
            if all(self.a.arena.clearance(obs.x + d * math.cos(th), obs.y + d * math.sin(th)) >= 0.17
                   for d in (0.15, 0.3, 0.45)):
                return th
        return None

    def _in_danger(self, obs):
        """Можно ли здесь стоять: риск считается по всем зонам, включая ту, из которой робот сейчас выезжает."""
        ix, iy = self.a.arena.w2g(obs.x, obs.y)
        return bool(self.a._risk_full[iy, ix] > 0.05) or obs.t - self._penalty_t < 4.5

    def _qid(self):
        return f'Q{len(self.inquiries) + 1}'

    def _start(self, inquiry, obs, held=None, est=None, activate=True):
        if activate:
            self.active = inquiry
        self.inquiries.append(inquiry)
        inquiry.held, inquiry.est = held or [], est or {}
        self._susp = []
        if self.roles is not None and hasattr(obs, 'battery'):     # мгновенная сверка у образца идёт без обсуждения
            self._deliberate(inquiry, obs)
        alts = '; '.join(f'«{x.statement}» {x.prior:.0%}' for x in inquiry.alternatives)
        self.a.journal.add(obs.t, 'inquiry', f"{inquiry.id}. Странность: {inquiry.anomaly['text']}. "
                           f'Возможные объяснения: {alts}', inquiry=inquiry.id)

    def _deliberate(self, q, obs):
        """Автор предлагает, какие объяснения рассматривать и в каком порядке ставить опыты; критик ищет дыры.

        Вероятности и вывод по-прежнему считает расчёт: модель только выбирает из допустимого и формулирует.
        """
        from .llm_roles import deliberate
        a = self.a
        state = {'battery': round(obs.battery, 1), 'return_cost': round(a._home_cost(obs.x, obs.y), 1),
                 'carried': a.collected, 'time_s': round(obs.t, 1),
                 'last_penalty_s_ago': round(obs.t - self._penalty_t, 1) if self._penalty_t > 0 else None}
        d = deliberate(self.roles, q.context(state))
        q.restrict(consider=d.proposal.consider, order=[step.test for step in d.proposal.plan])
        q.source = d.proposal.source
        q.note = d.proposal.rationale
        q.critique = [{'issue': i.text, 'kind': i.kind, 'fix': i.fix, 'resolved': i not in (d.open_issues or [])}
                      for i in (d.critique.issues if d.critique else [])]
        if a.rec:
            for ex in d.exchanges:
                a.rec.add_llm(obs.t, ex)
        a.journal.add(obs.t, 'llm', f'{q.id}. Автор: {d.proposal.rationale}'
                      + (f' Критик: замечаний {len(q.critique)}.' if q.critique else ''), inquiry=q.id)

    # ======================================================================================
    # проведение опытов
    # ======================================================================================

    def act(self, obs, io):
        """Если идёт расследование — занять робота опытом. Возвращает True, когда такт использован."""
        q, a = self.active, self.a
        if q is None and self._checkup_at is not None and obs.t >= self._checkup_at:
            # Останавливаться можно только выехав из опасной зоны: внутри неё штраф повторяется.
            if not self._in_danger(obs):
                self._checkup_at = None
                self._open_checkup(obs)
                q = self.active
            elif obs.t - self._checkup_at > 12.0:
                self._checkup_at = None
        if q is None and self._noise_pending:
            self._noise_pending = False
            if a.health.degraded and self.sensor['mode'] == 'ok':
                self.on_noise(obs)
                q = self.active
        if q is None:
            return self._wait_for_sensor(obs, io)
        if self.run is None and self._in_danger(obs):
            return False                   # сначала выехать из опасного места, опыт — потом
        if self.run is None:
            test = q.choose()
            if test is None or a._returning and test.action['kind'] != 'pause':
                self._conclude(obs)
                return False
            self.run = {'test': test, 'phase': 'settle', 't': obs.t}
            exp = '; '.join(f"при «{x.statement.split(',')[0]}» — {test.predictions[x.id][0]:.2f}"
                            for x in q.alternatives if x.id in test.predictions)
            a.journal.add(obs.t, 'inquiry', f'{q.id}. Опыт: {test.name} (польза {test.gain_bits:.2f} бит, '
                          f'цена {test.cost:.1f} ед.). Ожидаю, {test.unit}: {exp}', inquiry=q.id)
        r = self.run
        kind = r['test'].action['kind']
        a.mode = 'experiment'
        if r['phase'] == 'settle':
            a._command(io, 0.0, 0.0)
            if abs(obs.v) < 0.02 and abs(obs.w) < 0.05 and obs.t - r['t'] >= 0.3:
                err = _wrap(r['test'].action['heading'] - obs.th) if kind == 'straight' else 0.0
                if abs(err) > 0.12:
                    r['phase'] = 'aim'
                else:
                    r.update(phase='measure', t0=obs.t, b0=obs.battery, x=obs.x, y=obs.y, th=obs.th, ds=0.0, dth=0.0,
                             zs=[], n=a.collected)
            return True
        if r['phase'] == 'aim':
            err = _wrap(r['test'].action['heading'] - obs.th)
            if abs(err) <= 0.12:
                r.update(phase='settle', t=obs.t)
                a._command(io, 0.0, 0.0)
            else:
                a._command(io, 0.0, float(np.clip(2.0 * err, -1.2, 1.2)))
            return True
        r['ds'] += math.hypot(obs.x - r['x'], obs.y - r['y'])
        r['dth'] += abs(_wrap(obs.th - r['th']))
        r['x'], r['y'], r['th'] = obs.x, obs.y, obs.th
        if obs.sensor is not None:
            r['zs'].append(obs.sensor)
        dt = obs.t - r['t0']
        if kind == 'pause':
            a._command(io, 0.0, 0.0)
            done = dt >= PAUSE_S
        elif kind == 'straight':
            blocked = a._front < 0.2 or dt > 5.0
            a._command(io, 0.0 if blocked else 0.15, 0.0)
            done = r['ds'] >= STRAIGHT_M or blocked
        else:
            a._command(io, 0.0, 1.0)
            done = r['dth'] >= SPIN_RAD or dt > 3.0
        if done:
            self._measure(r, dt, obs)
            self.run = None
            self._win = None
        return True

    def _wait_for_sensor(self, obs, io):
        """Датчик залип: стоять почти бесплатно, а ехать и собирать вслепую — дорого. Ждём, но не дольше 35 с."""
        a = self.a
        if self.sensor['mode'] != 'stuck' or a._returning or self._in_danger(obs):
            self._wait_from = None
            return False
        if self._wait_from is None:
            self._wait_from = obs.t
            a.journal.add(obs.t, 'decision', 'Датчик образцов залип: стою и жду, пока он оживёт — это дешевле, чем искать вслепую')
        if obs.t - self._wait_from > 35.0:
            self._recovered('stuck', obs.t, 'ждать дольше нет смысла, продолжаю без уверенности в датчике')
            a.belief.relax(0.2)
            return False
        a.mode = 'wait'
        a._command(io, 0.0, 0.0)
        return True

    def _measure(self, r, dt, obs):
        q, m, a = self.active, self.model, self.a
        spent = r['b0'] - obs.battery
        per_m, per_s, per_rad = m.per_meter(r['n']), float(m.mean[3]), float(m.mean[2])
        kind = r['test'].action['kind']
        zs = np.array(r['zs'])
        heard = len(zs) >= 5                    # меньше пяти показаний — о датчике судить нельзя
        frozen = heard and float(zs.std()) < 1e-9
        if frozen and not 0.0 < float(zs.mean()) < 1.0:
            heard = False                       # показание упёрлось в край шкалы: одинаковые нули ни о чём не говорят
        values = {'rest': (spent / dt, BATTERY_NOISE / dt)}
        if kind == 'straight' and r['ds'] >= 0.15:
            values['straight'] = ((spent - per_rad * r['dth'] - per_s * dt) / (per_m * r['ds']),
                                  BATTERY_NOISE / (per_m * r['ds']))
        if kind == 'spin' and r['dth'] >= 0.6:
            values['spin'] = ((spent - per_s * dt) / r['dth'], BATTERY_NOISE / r['dth'])
        if heard:
            # одинаковые до последнего знака показания посреди шкалы у исправного датчика невозможны
            values['listen_std'] = (float(zs.std()), 0.001 if frozen else 0.012)
            values['listen_shift'] = (float(zs.mean()), 0.02)
        if kind == 'pause' and values['rest'][0] < per_s + 0.05:
            self._rest_ok_t = obs.t
        primary = r['test']
        if primary.id not in values:
            q.tests.remove(primary)             # опыт не удался (помеха, мало показаний): в расчёт не идёт
            a.journal.add(obs.t, 'inquiry', f'{q.id}. Опыт «{primary.name}» не удался, результат не учитываю', inquiry=q.id)
            return
        for inquiry in (q, self.companion):     # одна пауза даёт замеры и батарее, и датчику
            if inquiry is None:
                continue
            inquiry.last_z = float(zs[-1]) if heard else None
            for test in list(inquiry.tests):
                mine = test is primary
                if not (mine or (kind == 'pause' and test.action['kind'] == 'pause' and test.measured is None)):
                    continue
                if test.id not in values:
                    continue
                value, sigma = values[test.id]
                if test.id == 'listen_shift':
                    value -= getattr(inquiry, 'ref', 0.0)
                inquiry.record(test.id, value, sigma, obs.t, cost=max(spent, 0.0) if mine else 0.0, maneuver=mine)
                a.journal.add(obs.t, 'inquiry', f'{inquiry.id}. Измерено: {value:.2f} {test.unit}. Теперь: '
                              + '; '.join(f"«{x.statement.split(',')[0]}» {inquiry.posterior[x.id]:.0%}"
                                          for x in inquiry.alternatives), inquiry=inquiry.id)
        if kind == 'straight':
            q.held.append({'ds': r['ds'], 'dth': r['dth'], 'dt': dt, 'spent': spent, 'x': obs.x, 'y': obs.y,
                           'n': r['n'], 'moving': True, 't': r['t0'], 'mean': 0.0, 'known': False, 'mult': 1.0})

    # ======================================================================================
    # выводы и их последствия
    # ======================================================================================

    def _conclude(self, obs):
        q, a = self.active, self.a
        best = q.best[0]
        veto = None
        if best == 'turn' and not any(x.id == 'spin' and x.measured for x in q.tests):
            veto = 'цену поворота нельзя менять без опыта с поворотом'
        status = 'identified' if q.settled and not veto else 'insufficient'
        t = obs.t
        if q.topic in ('energy', 'fault'):
            action = self._apply_energy(q, best, status, obs)
        else:
            action = self._apply_sensor(q, best, status, obs)
        c = q.close(t, action, veto=veto)
        if self.roles is not None:
            from .llm_roles import explain
            text = explain(self.roles, q.to_dict())
            if str(text).strip():
                q.note = (q.note + ' ' if q.note else '') + 'Объяснение модели: ' + str(text)
                a.journal.add(t, 'llm', f'{q.id}. {text}', inquiry=q.id)
        a.journal.add(t, 'inquiry', f"{q.id}. Вывод: {c['text']}. Действие: {action}", inquiry=q.id,
                      status=c['status'], best=c['best'], tag=getattr(q, 'tag', None))
        self.active = None
        if self.companion is not None:          # расследование датчика, делившее паузу с проверкой батареи
            nxt, self.companion = self.companion, None
            if nxt.order:
                self.active = nxt
                return self._conclude(obs)
            self.inquiries.remove(nxt)
            self._recheck_until = t + 25.0
        self._quiet_until = t + 0.6
        self._win = None
        a._path_goal = None
        a._request_plan('inquiry_closed')

    def _apply_energy(self, q, best, status, obs):
        a, m, est = self.a, self.model, q.est
        rest = next((x.measured for x in q.tests if x.id == 'rest' and x.measured), None)
        if status != 'identified':
            self.reserve_until = (2.0, obs.t + 30.0)
            return 'причина не установлена: на 30 секунд держу 2 ед. заряда в запасе сверх обычного'
        if best == 'none':
            return 'батарея в порядке, еду дальше как обычно'
        if best == 'leak':
            rate = max((rest['value'] - float(m.mean[3])) if rest else est['rate'], 0.02)
            t0 = q.held[0]['t'] if q.held else (self._penalty_t if q.topic == 'fault' else obs.t)
            self.leak = {'rate': rate, 't0': t0, 'until': t0 + self.know.get('fault_s', FAULT_S), 'low': 0}
            q.tag = 'leak'
            left = max(0.0, self.leak['until'] - obs.t)
            # Причины могут совпасть: утечка могла идти на дорогом грунте. Окна, вызвавшие тревогу,
            # разбираются заново уже без утечки — что осталось сверх обычного пола, уходит в карту грунта.
            for s in q.held:
                if s['moving'] and a.cfg.learn_soil:
                    a.soil.add(s['x'], s['y'], s['ds'], float(np.clip(
                        m.soil_ratio(s['ds'], s['dth'], s['dt'], s['n'], s['spent'] - rate * s['dt']), 0.2, 7.0)))
            return (f'расход {rate:.2f} ед/с не приписываю грунту; закладываю в запас на возврат '
                    f'{rate * left:.1f} ед. на оставшиеся ~{left:.0f} с сбоя')
        if best == 'soil':
            q.tag = 'model_mismatch' if est.get('known_normal') else 'soil_found'
            x, y = q.anomaly['x'], q.anomaly['y']
            if est.get('known_normal'):
                a.soil.forget(x, y, radius=0.7, keep=0.1)
            for s in q.held:
                if s['moving']:
                    a.soil.add(s['x'], s['y'], s['ds'],
                               float(np.clip(m.soil_ratio(s['ds'], s['dth'], s['dt'], s['n'], s['spent']), 0.2, 7.0)))
            self.sites.append((x, y))
            a._cost_dirty, a._cost_t = True, -1e9
            return 'участок внесён в карту стоимостей, маршрут строится в объезд'
        if best == 'turn':
            spin = next((x.measured for x in q.tests if x.id == 'spin' and x.measured), None)
            m.widen('per_rad', 0.1)
            if spin:
                m.mean[2] = max(spin['value'], 0.0)
            return f'цена поворота в модели расхода исправлена на {m.mean[2]:.2f} ед/рад'
        return 'модель не меняю'

    def _track_leak(self, spent_raw, mean, dt, t):
        """Следить, идёт ли ещё утечка: когда расход вернулся к прогнозу — сбой кончился."""
        lk = self.leak
        extra = (spent_raw - mean) / dt
        lk['low'] = lk['low'] + 1 if extra < 0.35 * lk['rate'] else 0
        if t > lk['until'] and lk['low'] == 0:
            lk['until'] = t + 4.0                              # сбой длится дольше ожидаемого
        if lk['low'] >= 4 and t - lk['t0'] > 6.0:
            self.durations.append(('leak', round(t - lk['t0'], 1)))
            self.a.journal.add(t, 'inquiry', f'Утечка заряда прекратилась: длилась около {t - lk["t0"]:.0f} с, '
                               'запас на возврат возвращён к обычному', tag='leak_end')
            self.leak = None

    def per_meter(self):
        """Сколько заряда на самом деле стоит метр пути сейчас: с грузом и с обычной долей поворотов."""
        turns = self._path[1] / self._path[0] if self._path[0] > 2.0 else 1.5      # радиан на метр пути
        return self.model.per_meter(self.a.collected) + float(self.model.mean[2]) * min(turns, 4.0)

    def reserve(self, t):
        """Сколько заряда держать сверх обычного запаса на возврат."""
        extra = self.reserve_until[0] if t <= self.reserve_until[1] else 0.0
        if self.leak:
            extra += self.leak['rate'] * max(0.0, self.leak['until'] - t)
        return extra

    # ======================================================================================
    # датчик образцов
    # ======================================================================================

    def on_penalty(self, obs):
        """Штраф в опасной зоне: идущий опыт прерывается, а после выезда из зоны стоит проверить, не начался ли сбой."""
        self._penalty_t = obs.t
        self.penalty.event(obs.t)           # потерю заряда при штрафе найдёт и вычтет observe
        self._checkup_at = obs.t + 4.5
        if self.run is not None:
            self.run = None
            self.a.journal.add(obs.t, 'inquiry', 'Опыт прерван штрафом: измерение не учитываю, сначала выезжаю из зоны')
        zs = [v for _, _, _, v in self._z][-5:]
        self._pre_penalty = float(np.mean(zs)) if zs else 0.0
        b = self.a.belief
        if self._onset is None or obs.t - self._onset['t0'] > 12.0 or self._onset['col'] != self.a.collected:
            # С этого момента показания могут быть испорчены сбоем. Две версии — «датчик исправен» и «показания
            # занижены» — ведут каждая свою карту образцов от общей исходной; чья карта лучше предсказывает
            # показания, та версия и правдоподобнее.
            self._onset = {'t0': obs.t, 'ok': b.p.copy(), 'bias': b.p.copy(), 'lbf': 0.0, 'n': 0,
                           'col': self.a.collected, 'log': len(b._log), 'done': False}

    def _open_checkup(self, obs):
        """Проверка после штрафа. Батарея и датчик проверяются одной паузой, но как два отдельных вопроса:
        сбой батареи и сбой датчика друг друга не исключают."""
        a, m = self.a, self.model
        per_s = float(m.mean[3])
        leak = self.know.get('leak_rate', {'value': 0.15, 'sigma': 0.1})
        p_leak = self.know.get('p_leak_after_penalty', 0.3)
        q = Inquiry(self._qid(), obs.t, 'fault',
                    {'text': 'после штрафа в опасной зоне проверяю, не теряет ли батарея заряд', 'x': round(obs.x, 2),
                     'y': round(obs.y, 2), 'observed': 0.0, 'expected': 0.0, 'unit': ''},
                    [Alternative('leak', 'после штрафа батарея теряет заряд сама по себе', p_leak),
                     Alternative('none', 'батарея в порядке', 0.9 - p_leak),
                     Alternative(OTHER, 'причина не из этого списка', 0.1)],
                    [TestOption('rest', f'постоять {PAUSE_S:.0f} секунды и измерить расход на месте',
                                cost=per_s * PAUSE_S + 0.05, duration_s=PAUSE_S, unit='ед/с',
                                predictions={'leak': (per_s + leak['value'], max(leak['sigma'], 0.03)),
                                             'none': (per_s, 0.012)},
                                action={'kind': 'pause'}, sigma=BATTERY_NOISE / PAUSE_S, repeatable=True)],
                    energy_budget=1.0, max_tests=2)
        self._start(q, obs)
        if self.sensor['mode'] != 'ok':
            return
        # Что датчик должен показывать здесь, если он исправен: прогноз по карте образцов, какой она была до штрафа.
        # Прежнее показание для сравнения не годится: робот успел отъехать, и оно изменилось бы само.
        on = self._onset
        self._doubt_until = obs.t + 40.0
        if on and on['col'] == a.collected and on['n'] >= 5:
            on['done'] = True
            c = self.companion = self._open_sensor(obs, 'checkup', ref=self._pre_penalty, jump=BIAS, activate=False)
            c.record('onset', on['lbf'], 1.0, obs.t, cost=0.0, maneuver=False)
            c.onset = on
            a.journal.add(obs.t, 'inquiry', f"{c.id}. Измерено: перевес версии «занижает» {on['lbf']:+.1f} по "
                          f"{on['n']} показаниям после штрафа. Теперь: "
                          + '; '.join(f"«{x.statement.split(',')[0]}» {c.posterior[x.id]:.0%}" for x in c.alternatives),
                          inquiry=c.id)
        else:
            # Сигнал у нуля: шум, залипание и занижение сейчас неотличимы от нормы. Проверим, когда появится сигнал.
            self._recheck_until = obs.t + 25.0
            a.journal.add(obs.t, 'inquiry', 'Датчик образцов сейчас проверить нельзя: рядом нет образца, показание у нуля. '
                          'Проверю, как только появится сигнал')

    def on_miss(self, obs):
        """Ложный сбор при высокой уверенности: значит, показаниям сейчас верить нельзя."""
        if self.sensor['mode'] == 'bias':
            self.a.journal.add(obs.t, 'inquiry', 'Поправка к показаниям датчика привела к промаху: отменяю её, '
                               'сбой датчика, видимо, уже кончился', tag='sensor_recovered')
            self.durations.append(('sensor_bias', round(obs.t - self.sensor.get('t0', obs.t), 1)))
            self.sensor = {'mode': 'ok'}
        self.a.belief.relax(0.15)
        self._z.clear()

    def on_collect(self, t):
        self._calibrate(t)
        self._collect_t = t
        self._z.clear()

    def _calibrate(self, t):
        """Сверка датчика с известной точкой. Образец только что взят — значит, он лежал ближе радиуса сбора,
        и исправный датчик обязан был показывать почти единицу. Это единственное место, где расстояние до
        образца известно без самого датчика, поэтому здесь занижение показаний отличается от нормы наверняка."""
        a, s, r = self.a, self.sensor, self.a.rules
        seen = [(x, y, v) for tt, x, y, v in self._z if t - tt <= 1.0]
        if len(seen) < 4 or self.active is not None or s['mode'] == 'stuck':
            return
        if s['mode'] == 'noise' and not s.get('assumed'):
            return                              # шум уже установлен опытом; когда он кончится, покажет разброс
        vals = np.array([v for _, _, v in seen])
        zbar, zstd, n = float(vals.mean()), float(vals.std()), len(vals)
        if zstd < 1e-9 and zbar < 1.0:
            return                              # одинаковые показания: похоже на залипание, сверять нечего
        nominal = a.health.nominal
        quiet = a.health.sigma < 1.8 * nominal  # разброс за последние секунды в норме
        near = 1.0 - 0.5 * r.collect_radius_m / r.sensor_range_m       # среднее показание в радиусе сбора
        width = math.hypot(0.3 * r.collect_radius_m / r.sensor_range_m, 0.02)
        asked = t <= max(self._doubt_until, self._recheck_until) or s['mode'] != 'ok'
        normal = zbar >= near - 2.5 * width and zstd <= 2.2 * nominal
        if not asked and normal:
            return                              # показания такие, какими и должны быть; вопросов к датчику нет
        bias = s.get('bias', 0.2)
        # У самого образца показание упирается в единицу, и по пяти отсчётам шум от нормы не отличить.
        # Поэтому про шум здесь спрашивается у оценки разброса за последние пять секунд.
        prior = {'ok': 0.6, 'bias': 0.3} if normal else {'ok': 0.3, 'bias': 0.6}
        prior = {k: v * (0.95 if quiet else 0.5) for k, v in prior.items()}
        prior['noise'] = 0.05 if quiet else 0.5
        text = (f'образец взят, значит он был ближе {r.collect_radius_m:.1f} м; датчик при этом показывал {zbar:.2f}'
                + ('' if zbar >= near - 2.5 * width else f' вместо ожидаемых {near:.2f}')
                + ('' if zstd <= 2.2 * nominal else f', разброс {zstd:.2f} при норме {nominal:.2f}'))
        obs = SimpleNamespace(t=t, x=seen[-1][0], y=seen[-1][1])
        calm, loud = (1.1 * nominal, 0.45 * nominal), (3.2 * nominal, 1.4 * nominal)   # у единицы шум срезается
        q = Inquiry(self._qid(), t, 'sensor',
                    {'text': text, 'x': round(obs.x, 2), 'y': round(obs.y, 2), 'observed': round(zbar, 3),
                     'expected': round(near, 3), 'unit': 'показание', 'trigger': 'collect'},
                    [Alternative('ok', 'датчик исправен: у образца он показывает столько, сколько должен', prior['ok']),
                     Alternative('bias', f'показания занижены примерно на {bias:.2f} (сбой)', prior['bias']),
                     Alternative('noise', 'датчик образцов стал шуметь сильнее (сбой)', prior['noise']),
                     Alternative(OTHER, 'причина не из этого списка', 0.1)],
                    [TestOption('at_sample', 'сверить показание с известной точкой: образец взят, расстояние до него известно',
                                cost=0.0, duration_s=0.0, unit='показание',
                                predictions={'ok': (near, width), 'bias': (near - bias, math.hypot(width, 0.04)),
                                             'noise': (near - 0.04, 5.0 * nominal / math.sqrt(n))},
                                action={'kind': 'none'}, sigma=nominal / math.sqrt(n)),
                     TestOption('at_sample_std', 'там же измерить разброс показаний',
                                cost=0.0, duration_s=0.0, unit='разброс',
                                predictions={'ok': calm, 'bias': calm, 'noise': loud},
                                action={'kind': 'none'}, sigma=0.01)],
                    energy_budget=1.0)
        q.ref, q.jump, q.trigger, q.last_z = near, float(np.clip(near - zbar, 0.1, 0.35)), 'collect', seen[-1][2]
        self._start(q, obs)
        q.record('at_sample_std', zstd, 0.01, t, cost=0.0, maneuver=False)
        q.record('at_sample', zbar, nominal / math.sqrt(n), t, cost=0.0, maneuver=False)
        a.journal.add(t, 'inquiry', f'{q.id}. Измерено: показание {zbar:.2f}, разброс {zstd:.2f}. Теперь: '
                      + '; '.join(f"«{x.statement.split(',')[0]}» {q.posterior[x.id]:.0%}" for x in q.alternatives),
                      inquiry=q.id)
        if q.settled:
            self._doubt_until = self._recheck_until = -1e9
            if q.best[0] == 'ok' and s['mode'] in ('bias', 'noise'):
                self._recovered(s['mode'], t, 'у взятого образца датчик показал норму')
                a.health.degraded = False
        self._conclude(obs)

    def _track_onset(self, z, obs):
        """После штрафа: какая версия лучше предсказывает очередное показание — «исправен» или «занижает»."""
        on, b = self._onset, self.a.belief
        if on is None or on['done'] or on['col'] != self.a.collected:
            return
        if obs.t - on['t0'] > ONSET_S:
            on['done'] = True
            return
        if not 0.03 < z < 1.0 - BIAS - 0.03:
            return                              # у края шкалы показание обрезано и версии не различает
        nominal = self.a.health.nominal
        on['lbf'] += b.loglik(obs.x, obs.y, z + BIAS, nominal, on['bias']) - b.loglik(obs.x, obs.y, z, nominal, on['ok'])
        on['n'] += 1
        on['ok'] = b.trial(obs.x, obs.y, z, nominal, p=on['ok'])[0]
        on['bias'] = b.trial(obs.x, obs.y, z + BIAS, nominal, p=on['bias'])[0]

    def reading(self, z, obs, sigma):
        """Показание датчика → (показание для карты образцов или None, шум). Здесь же ищутся сбои датчика."""
        a, s, t = self.a, self.sensor, obs.t
        self._z.append((t, obs.x, obs.y, z))
        self._track_onset(z, obs)
        if s['mode'] == 'stuck':
            if abs(z - s['value']) > 1e-9:
                self._recovered('stuck', t, 'датчик снова меняет показания')
                self._z.clear()                 # прежние одинаковые показания со скачком сравнивать нельзя
                self._z.append((t, obs.x, obs.y, z))
                return z, sigma
            else:
                return None, sigma
        elif s['mode'] == 'bias':
            prev = [v for _, _, _, v in list(self._z)[-8:-3]]
            last = [v for _, _, _, v in list(self._z)[-3:]]
            if (len(prev) >= 4 and np.mean(last) - np.mean(prev) > 0.6 * s['bias']) or t > s['until'] + 20.0:
                self._recovered('bias', t, 'показания вернулись на прежний уровень')
                self._z.clear()
            else:
                return min(1.0, z + s['bias']), sigma
        elif s['mode'] == 'noise' and not a.health.degraded:
            self._recovered('noise', t, f'шум вернулся к {a.health.sigma:.2f}')
        if s['mode'] != 'ok' or self.active is not None or t < self._quiet_until or t - self._collect_t < 2.5:
            return z, sigma
        if t <= self._recheck_until and len(self._z) >= 5 and np.mean([v for _, _, _, v in list(self._z)[-5:]]) > 0.3:
            self._recheck_until = -1e9          # сигнал появился: теперь датчик можно проверить после штрафа
            self._open_sensor(obs, 'noise', ref=float(np.mean([v for _, _, _, v in self._z])))
            return z, sigma
        zs = list(self._z)
        if len(zs) >= 10:
            vals = [v for _, _, _, v in zs[-10:]]
            moved = math.hypot(zs[-1][1] - zs[-10][1], zs[-1][2] - zs[-10][2])
            if max(vals) - min(vals) < 1e-9 and 0.0 < z < 1.0 and moved >= 0.12:
                self._open_sensor(obs, 'stuck', ref=z)
                return None, sigma
            prev, last = np.mean(vals[:5]), np.mean(vals[-3:])
            step = math.hypot(zs[-1][1] - zs[-8][1], zs[-1][2] - zs[-8][2])
            if max(vals[:5]) - min(vals[:5]) < 1e-9 and abs(prev - last) > 0.1:
                # До скачка показания были одинаковыми до последнего знака: датчик был залипшим и ожил.
                a.journal.add(t, 'inquiry', 'Показания датчика были одинаковыми и вдруг изменились: похоже, датчик '
                              'был залипшим и ожил; скачок сбоем не считаю', tag='sensor_recovered')
                self._z.clear()
            elif prev - last > max(0.13, step / a.rules.sensor_range_m + 0.1) and prev > 0.2:
                self._open_sensor(obs, 'shift', ref=float(prev), jump=float(prev - last))
        return z, sigma

    def on_noise(self, obs):
        """Оценка шума датчика выросла (did.belief.SensorHealth): это сбой или робот просто быстро едет к образцу?"""
        if self.active is not None:
            self._noise_pending = True          # не терять тревогу: разберём после текущего расследования
        elif self.sensor['mode'] == 'ok':
            self._open_sensor(obs, 'noise', ref=float(np.mean([v for _, _, _, v in self._z])) if self._z else 0.0)

    def _open_sensor(self, obs, trigger, ref=0.0, jump=0.0, activate=True):
        a = self.a
        nominal, est = a.health.nominal, max(a.health.sigma, 4.0 * a.health.nominal)
        penalty = obs.t - self._penalty_t < 45.0
        base = {'stuck': {'stuck': 0.75, 'noise': 0.03, 'bias': 0.02, 'ok': 0.1},
                'shift': {'bias': 0.4, 'noise': 0.2, 'stuck': 0.02, 'ok': 0.28},
                'noise': {'noise': 0.55, 'bias': 0.05, 'stuck': 0.02, 'ok': 0.28},
                'checkup': {'noise': 0.2, 'bias': 0.2, 'stuck': 0.2, 'ok': 0.3}}[trigger]
        base.update({k: v for k, v in self.know.get('sensor_priors', {}).items() if trigger == 'checkup'})
        if penalty:
            base['ok'] *= 0.4
        text = {'stuck': f'датчик образцов десять раз подряд выдал одно и то же значение {ref:.3f}, хотя робот едет',
                'shift': f'показание датчика упало на {jump:.2f} почти без движения',
                'noise': f'разброс показаний датчика вырос до {a.health.sigma:.2f} при норме {nominal:.2f}',
                'checkup': 'после штрафа в опасной зоне проверяю датчик образцов'}[trigger]
        alts = [Alternative('noise', 'датчик образцов стал шуметь сильнее (сбой)', base['noise']),
                Alternative('stuck', 'датчик залип и повторяет одно значение (сбой)', base['stuck']),
                Alternative('bias', f'показания занижены примерно на {jump or 0.2:.2f} (сбой)', base['bias']),
                Alternative('ok', 'датчик исправен: показание изменилось по естественной причине', base['ok']),
                Alternative(OTHER, 'причина не из этого списка', 0.1)]
        tests = [TestOption('listen_std', f'постоять {PAUSE_S:.0f} секунды и измерить разброс показаний',
                            cost=float(self.model.mean[3]) * PAUSE_S, duration_s=PAUSE_S, unit='разброс',
                            predictions={'noise': (est, 0.4 * est), 'stuck': (0.0, 0.004), 'bias': (nominal, 0.4 * nominal),
                                         'ok': (nominal, 0.4 * nominal)},
                            action={'kind': 'pause'}, sigma=0.012)]
        if trigger == 'shift':
            tests.append(TestOption('listen_shift', 'там же сравнить среднее показание с прежним',
                                    cost=0.0, duration_s=PAUSE_S, unit='сдвиг',
                                    # при шуме «прежнее» значение завышено самим отбором: тревогу поднял выброс
                                    predictions={'bias': (-jump, 0.05), 'ok': (0.0, 0.04), 'noise': (-0.5 * jump, 0.15),
                                                 'stuck': (-jump, 0.2)},
                                    action={'kind': 'pause'}, sigma=0.02))
        if trigger == 'checkup':
            # Робот успел отъехать, прежнее показание для сравнения не годится. Сравниваются две версии карты.
            tests.append(TestOption('onset', 'по показаниям с момента штрафа сравнить две версии карты образцов: '
                                    '«датчик исправен» и «показания занижены»',
                                    cost=0.0, duration_s=ONSET_S, unit='перевес версии «занижает»',
                                    predictions=dict(ONSET), action={'kind': 'none'}, sigma=1.0))
        q = Inquiry(self._qid(), obs.t, 'sensor', {'text': text, 'x': round(obs.x, 2), 'y': round(obs.y, 2),
                                                  'observed': round(jump or a.health.sigma, 3),
                                                  'expected': round(nominal, 3), 'unit': 'показание',
                                                  'trigger': trigger},
                    alts, tests, energy_budget=1.0)
        q.ref, q.jump, q.trigger = ref, jump, trigger
        self._start(q, obs, activate=activate)
        return q

    def _apply_sensor(self, q, best, status, obs):
        a, t = self.a, obs.t
        until = t + self.know.get('fault_s', FAULT_S)
        if status != 'identified':
            self._doubt_until = t + 40.0      # вопрос остался: его закроет сверка у ближайшего взятого образца
            a.health.degraded = True          # на всякий случай верим датчику меньше
            a.health.sigma = max(a.health.sigma, 2.0 * a.health.nominal)
            self.sensor = {'mode': 'noise', 't0': q.t_open, 'until': until, 'assumed': True}
            return 'причина не установлена: временно снижаю вес показаний датчика вдвое'
        if best == 'ok':
            return 'датчику верю как раньше'
        q.tag = 'sensor_degraded'
        a.belief.relax(0.2)
        if best == 'noise':
            a.health.degraded = True
            self.sensor = {'mode': 'noise', 't0': q.t_open, 'until': until}
            return f'снижаю вес показаний: считаю шум равным {a.health.effective_sigma():.2f}'
        if best == 'stuck':
            value = q.last_z if getattr(q, 'last_z', None) is not None else q.ref     # то, что датчик показывал в паузе
            self.sensor = {'mode': 'stuck', 'value': value, 't0': q.t_open, 'until': until}
            return 'показания датчика не учитываю, пока они не начнут меняться; образцы не собираю вслепую'
        shift = next((x.measured['value'] for x in q.tests if x.id == 'listen_shift' and x.measured), -q.jump)
        on = getattr(q, 'onset', None)
        if on and on['col'] == a.collected and len(a.belief._log) >= on['log']:
            # Показания с момента штрафа уже попали в карту заниженными: поправить их задним числом.
            log = a.belief._log
            log[on['log']:] = [(x, y, min(1.0, z + abs(shift)), sg) for x, y, z, sg in log[on['log']:]]
            a.belief.p = a.belief._epoch.copy()
            for x, y, z, sg in log:
                a.belief._apply(x, y, z, sg)
            a.belief._normalize()
        self.sensor = {'mode': 'bias', 'bias': abs(shift), 't0': q.t_open, 'until': until}
        return f'прибавляю к показаниям {abs(shift):.2f}, пока они не вернутся на прежний уровень'

    def _recovered(self, kind, t, why):
        t0 = self.sensor.get('t0', t)
        if self.sensor.get('assumed'):          # сбой не был установлен, осторожность была на всякий случай
            self.a.journal.add(t, 'inquiry', f'Датчику снова верю как обычно: {why}', tag='sensor_recovered')
            self.sensor = {'mode': 'ok'}
            return
        self.durations.append((f'sensor_{kind}', round(t - t0, 1)))
        self.a.journal.add(t, 'inquiry', f'Сбой датчика закончился: {why} (длился около {t - t0:.0f} с)',
                           tag='sensor_recovered')
        self.sensor = {'mode': 'ok'}

    # ======================================================================================
    # итог для записи прогона и памяти
    # ======================================================================================

    def export(self):
        return {'inquiries': [q.to_dict() for q in self.inquiries], 'energy_model': self.model.summary(),
                'fault_durations': self.durations}
