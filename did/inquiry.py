"""Исследователь внутри агента: следит, сходятся ли измерения с моделью, и ведёт расследования.

Работает при правилах с несколькими причинами одного симптома (did.config.SCIENCE): расход растёт
от дорогого грунта, от поворотов, от груза и от утечки после сбоя; датчик образцов может зашуметь,
залипнуть или начать занижать показания. Заметив странность, исследователь не выбирает одно
объяснение, а открывает расследование (did/science.py): несколько объяснений, опыт, который их
различит, измерение, вывод. Опыты — короткие манёвры: постоять, проехать прямо, развернуться.
"""
import math
from collections import deque

import numpy as np

from .energy import EnergyModel
from .science import OTHER, Alternative, Inquiry, TestOption

WINDOW_M = 0.12        # окно для оценки расхода в движении, м
STILL_S = 1.0          # окно на месте, с
PAUSE_S = 2.0          # опыт «постоять»
STRAIGHT_M = 0.30      # опыт «проехать прямо»
SPIN_RAD = 1.5         # опыт «развернуться на месте»
BATTERY_NOISE = 0.03   # шум разности двух показаний батареи (уточняется на старте)
FAULT_S = 25.0         # сколько, по умолчанию, длится сбой (уточняется памятью между прогонами)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Investigator:

    def __init__(self, agent, knowledge=None):
        self.a = agent
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
        self._susp = []                # подряд идущие окна, которых модель не ждала
        self._cusum = 0.0
        self._checkup_at = None        # когда провести проверку после штрафа
        self._rest_ok_t = -1e9         # когда опыт «постоять» последний раз показал, что утечки нет
        self._pre_penalty = 0.0        # среднее показание датчика перед штрафом
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
        if self._win is None or self.run is not None:
            self._open(obs)            # во время опыта окно не копится: у опыта свой замер
            return
        w = self._win
        w['ds'] += math.hypot(obs.x - w['x'], obs.y - w['y'])
        w['dth'] += abs(_wrap(obs.th - w['th']))
        w['x'], w['y'], w['th'] = obs.x, obs.y, obs.th
        dt = obs.t - w['t0']
        if w['ds'] >= WINDOW_M or (dt >= STILL_S and w['ds'] < 0.03) or dt >= 2.0:
            self._window(w, dt, obs)
            self._open(obs)

    def _open(self, obs):
        self._win = {'x0': obs.x, 'y0': obs.y, 'x': obs.x, 'y': obs.y, 'th': obs.th, 't0': obs.t, 'b0': obs.battery,
                     'ds': 0.0, 'dth': 0.0, 'n': self.a.collected}

    def _near_site(self, x, y):
        return any(math.hypot(x - sx, y - sy) <= 0.9 for sx, sy in self.sites)

    def _leak_rate(self, t):
        return self.leak['rate'] if self.leak and t <= self.leak['until'] + 6.0 else 0.0

    def _window(self, w, dt, obs):
        a, m = self.a, self.model
        ds, dth, n = w['ds'], w['dth'], w['n']
        mx, my = (w['x0'] + obs.x) / 2, (w['y0'] + obs.y) / 2
        spent = w['b0'] - obs.battery - self._leak_rate(obs.t) * dt
        moving = ds >= 0.05
        mult, conf = a.soil.at(mx, my) if (a.cfg.learn_soil and moving) else (1.0, 1.0)
        known = conf >= 0.4 or self._near_site(mx, my)
        mean, sd = m.predict(ds, dth, dt, n, mult=mult if known else 1.0)
        if known and moving:
            sd = math.hypot(sd, 0.3 * mult * m.per_meter(n) * ds)        # оценка грунта сама неточна
        z = (spent - mean) / sd
        if self.leak:
            self._track_leak(w['b0'] - obs.battery, m.predict(ds, dth, dt, n, mult=mult if known else 1.0)[0], dt, obs.t)
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
                self._open_energy(obs)
            return
        if self._cusum == 0.0:
            self._susp.clear()
        if moving and a.cfg.learn_soil:
            ratio = float(np.clip(m.soil_ratio(ds, dth, dt, n, spent), 0.2, 7.0))
            a.soil.add(mx, my, ds, ratio)
            a._cost_dirty = True
            if z < -3.0 and known and mult > 1.4:
                a._on_model_mismatch('дешевле', mx, my, ratio, mult, obs.t)
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
        moving = ds > 0.08
        rate = max(excess / dt, 0.02)
        ratio = 1.0 + excess / (per_m * ds) if moving else None
        turn = per_rad + excess / dth if dth > 0.6 and dth / max(ds, 0.05) > 2.5 else None
        known_normal = last['known'] and last['mult'] < 1.25
        penalty = obs.t - self._penalty_t < 45.0
        if moving and turn is None and obs.t - self._rest_ok_t < 60.0 and self._penalty_t < self._rest_ok_t:
            # Очевидный случай: утечку недавно исключил опыт, штрафов с тех пор не было — остаётся грунт.
            if known_normal:
                a._on_model_mismatch('дороже', last['x'], last['y'], ratio, last['mult'], obs.t)
            for q in S:
                if q['moving']:
                    a.soil.add(q['x'], q['y'], q['ds'],
                               float(np.clip(m.soil_ratio(q['ds'], q['dth'], q['dt'], q['n'], q['spent']), 0.2, 7.0)))
            self.sites.append((last['x'], last['y']))
            a._cost_dirty, a._cost_t = True, -1e9
            a.journal.add(obs.t, 'observe', f'Расход ×{ratio:.1f} около ({last["x"]:.1f}; {last["y"]:.1f}). '
                          f'Утечку исключил опыт {obs.t - self._rest_ok_t:.0f} с назад, штрафов с тех пор не было — значит, грунт',
                          tag='soil_found')
            self._susp = []
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
                            cost=(per_s + 0.4 * rate) * PAUSE_S, duration_s=PAUSE_S, unit='ед/с',
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
                                    cost=max(turn, per_rad) * SPIN_RAD, duration_s=SPIN_RAD, unit='ед/рад',
                                    predictions=preds(turn=(turn, 0.3 * turn + 0.03), soil=(per_rad, 0.05),
                                                      leak=(per_rad + rate, 0.35 * rate + 0.04)),
                                    action={'kind': 'spin'}, sigma=BATTERY_NOISE / SPIN_RAD))
        if moving:
            text = f'расход {tot("spent") / ds:.1f} ед/м при прогнозе {tot("mean") / ds:.1f}'
            anomaly = {'observed': round(tot('spent') / ds, 2), 'expected': round(tot('mean') / ds, 2), 'unit': 'ед/м'}
        else:
            text = f'на месте уходит {tot("spent") / dt:.2f} ед/с при прогнозе {tot("mean") / dt:.2f}'
            anomaly = {'observed': round(tot('spent') / dt, 3), 'expected': round(tot('mean') / dt, 3), 'unit': 'ед/с'}
        anomaly.update(text=text, x=round(last['x'], 2), y=round(last['y'], 2))
        self._start(Inquiry(self._qid(), obs.t, 'energy', anomaly, alts, tests, energy_budget=self._budget(obs)),
                    obs, held=list(S), est={'rate': rate, 'ratio': ratio, 'turn': turn, 'known_normal': known_normal})

    def _budget(self, obs):
        a = self.a
        home = a._home_cost(obs.x, obs.y)
        spare = obs.battery - home * a.cfg.reserve_margin - a.cfg.reserve_abs
        return float(np.clip(spare - 1.0, 0.0, 2.5))

    def _free_heading(self, obs):
        """Направление, в котором есть 45 см свободного пола: сначала текущий курс, потом соседние."""
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
            th = obs.th + off
            if all(self.a.arena.clearance(obs.x + d * math.cos(th), obs.y + d * math.sin(th)) >= 0.17
                   for d in (0.15, 0.3, 0.45)):
                return th
        return None

    def _qid(self):
        return f'Q{len(self.inquiries) + 1}'

    def _start(self, inquiry, obs, held=None, est=None):
        self.active = inquiry
        self.inquiries.append(inquiry)
        inquiry.held, inquiry.est = held or [], est or {}
        self._susp = []
        alts = '; '.join(f'«{x.statement}» {x.prior:.0%}' for x in inquiry.alternatives)
        self.a.journal.add(obs.t, 'inquiry', f"{inquiry.id}. Странность: {inquiry.anomaly['text']}. "
                           f'Возможные объяснения: {alts}', inquiry=inquiry.id)

    # ======================================================================================
    # проведение опытов
    # ======================================================================================

    def act(self, obs, io):
        """Если идёт расследование — занять робота опытом. Возвращает True, когда такт использован."""
        q, a = self.active, self.a
        if q is None and self._checkup_at is not None and obs.t >= self._checkup_at:
            self._checkup_at = None
            if self.sensor['mode'] == 'ok' and self.leak is None:
                self._open_checkup(obs)
                q = self.active
        if q is None:
            return False
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

    def _measure(self, r, dt, obs):
        q, m = self.active, self.model
        spent = r['b0'] - obs.battery
        per_m, per_s, per_rad = m.per_meter(r['n']), float(m.mean[3]), float(m.mean[2])
        zs = np.array(r['zs']) if r['zs'] else np.array([0.0])
        values = {
            'rest': (spent / dt, BATTERY_NOISE / dt),
            'straight': ((spent - per_rad * r['dth'] - per_s * dt) / max(per_m * r['ds'], 1e-6),
                         BATTERY_NOISE / max(per_m * r['ds'], 0.05)),
            'spin': ((spent - per_s * dt) / max(r['dth'], 0.2), BATTERY_NOISE / max(r['dth'], 0.2)),
            'listen_std': (float(zs.std()), 0.012),
            'listen_shift': (float(zs.mean()) - getattr(q, 'ref', 0.0), 0.02),
        }
        kind = r['test'].action['kind']
        if kind == 'pause' and values['rest'][0] < per_s + 0.05:
            self._rest_ok_t = obs.t
        for test in q.tests:              # одна пауза даёт сразу все «стоячие» измерения
            if test.measured is None and (test is r['test'] or (kind == 'pause' and test.action['kind'] == 'pause')):
                value, sigma = values[test.id]
                q.record(test.id, value, sigma, obs.t)
                self.a.journal.add(obs.t, 'inquiry', f'{q.id}. Измерено: {value:.2f} {test.unit}. Теперь: '
                                   + '; '.join(f"«{x.statement.split(',')[0]}» {q.posterior[x.id]:.0%}"
                                               for x in q.alternatives), inquiry=q.id)
        if kind == 'straight' and r['ds'] > 0.1:
            q.held.append({'ds': r['ds'], 'dth': r['dth'], 'dt': dt, 'spent': spent, 'x': obs.x, 'y': obs.y,
                           'n': r['n'], 'moving': True, 't': r['t0'], 'mean': 0.0, 'known': False, 'mult': 1.0})

    # ======================================================================================
    # выводы и их последствия
    # ======================================================================================

    def _conclude(self, obs):
        q, a, m = self.active, self.a, self.model
        best = q.best[0]
        status = 'identified' if q.settled else 'insufficient'
        t = obs.t
        if q.topic == 'energy' or (q.topic == 'fault' and best == 'leak'):
            action = self._apply_energy(q, best, status, obs)
        elif q.topic == 'fault' and best == 'none' and status == 'identified':
            action = 'еду дальше как обычно'
        else:
            action = self._apply_sensor(q, best, status, obs)
        c = q.close(t, action)
        a.journal.add(t, 'inquiry', f"{q.id}. Вывод: {c['text']}. Действие: {action}", inquiry=q.id,
                      status=c['status'], best=c['best'], tag=getattr(q, 'tag', None))
        self.active = None
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
        if best == 'leak':
            rate = max((rest['value'] - float(m.mean[3])) if rest else est['rate'], 0.02)
            t0 = q.held[0]['t'] if q.held else (self._penalty_t if q.topic == 'fault' else obs.t)
            self.leak = {'rate': rate, 't0': t0, 'until': t0 + self.know.get('fault_s', FAULT_S), 'low': 0}
            q.tag = 'leak'
            left = max(0.0, self.leak['until'] - obs.t)
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
        """Штраф в опасной зоне: после выезда из неё стоит проверить, не начался ли сбой."""
        self._penalty_t = obs.t
        self._checkup_at = obs.t + 3.5
        zs = [v for _, _, _, v in self._z][-5:]
        self._pre_penalty = float(np.mean(zs)) if zs else 0.0

    def _open_checkup(self, obs):
        a, m = self.a, self.model
        per_s, nominal = float(m.mean[3]), a.health.nominal
        k = self.know
        leak = k.get('leak_rate', {'value': 0.15, 'sigma': 0.1})
        p = k.get('fault_priors', {'leak': 0.2, 'noise': 0.15, 'stuck': 0.15, 'bias': 0.15, 'none': 0.25})
        alts = [Alternative('leak', 'после штрафа батарея начала терять заряд сама по себе', p['leak']),
                Alternative('noise', 'после штрафа датчик образцов стал шуметь', p['noise']),
                Alternative('stuck', 'после штрафа датчик образцов залип', p['stuck']),
                Alternative('bias', 'после штрафа показания датчика занижены', p['bias']),
                Alternative('none', 'штраф прошёл без последствий для батареи и датчика', p['none']),
                Alternative(OTHER, 'причина не из этого списка', 0.1)]
        same = (nominal, 0.4 * nominal)
        tests = [TestOption('rest', f'постоять {PAUSE_S:.0f} секунды и измерить расход на месте',
                            cost=(per_s + 0.05) * PAUSE_S, duration_s=PAUSE_S, unit='ед/с',
                            predictions={'leak': (per_s + leak['value'], max(leak['sigma'], 0.03)), 'noise': (per_s, 0.012),
                                         'stuck': (per_s, 0.012), 'bias': (per_s, 0.012), 'none': (per_s, 0.012)},
                            action={'kind': 'pause'}, sigma=BATTERY_NOISE / PAUSE_S),
                 TestOption('listen_std', 'за ту же паузу измерить разброс показаний датчика',
                            cost=0.0, duration_s=PAUSE_S, unit='разброс',
                            predictions={'noise': (0.22, 0.08), 'stuck': (0.0, 0.004), 'leak': same, 'bias': same,
                                         'none': same},
                            action={'kind': 'pause'}, sigma=0.012),
                 TestOption('listen_shift', 'и сравнить среднее показание с тем, что было до штрафа',
                            cost=0.0, duration_s=PAUSE_S, unit='сдвиг',
                            predictions={'bias': (-0.2, 0.1), 'none': (0.0, 0.1), 'leak': (0.0, 0.1), 'noise': (0.0, 0.14),
                                         'stuck': (0.0, 0.25)},
                            action={'kind': 'pause'}, sigma=0.02)]
        q = Inquiry(self._qid(), obs.t, 'fault',
                    {'text': 'после штрафа в опасной зоне проверяю, не начался ли сбой', 'x': round(obs.x, 2),
                     'y': round(obs.y, 2), 'observed': 0.0, 'expected': 0.0, 'unit': ''},
                    alts, tests, energy_budget=1.0, max_tests=3)
        q.ref, q.jump, q.trigger = self._pre_penalty, 0.2, 'checkup'
        self._start(q, obs)

    def on_collect(self, t):
        self._collect_t = t
        self._z.clear()

    def reading(self, z, obs, sigma):
        """Показание датчика → (показание для карты образцов или None, шум). Здесь же ищутся сбои датчика."""
        a, s, t = self.a, self.sensor, obs.t
        self._z.append((t, obs.x, obs.y, z))
        if s['mode'] == 'stuck':
            if abs(z - s['value']) > 1e-9:
                self._recovered('stuck', t, 'датчик снова меняет показания')
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
        zs = list(self._z)
        if len(zs) >= 10:
            vals = [v for _, _, _, v in zs[-10:]]
            moved = math.hypot(zs[-1][1] - zs[-10][1], zs[-1][2] - zs[-10][2])
            if max(vals) - min(vals) < 1e-9 and 0.0 < z < 1.0 and moved >= 0.12:
                self._open_sensor(obs, 'stuck', ref=z)
                return None, sigma
            prev, last = np.mean(vals[:5]), np.mean(vals[-3:])
            step = math.hypot(zs[-1][1] - zs[-8][1], zs[-1][2] - zs[-8][2])
            if prev - last > max(0.13, step / a.rules.sensor_range_m + 0.1) and prev > 0.2:
                self._open_sensor(obs, 'shift', ref=float(prev), jump=float(prev - last))
        return z, sigma

    def on_noise(self, obs):
        """Оценка шума датчика выросла (did.belief.SensorHealth): это сбой или робот просто быстро едет к образцу?"""
        if self.sensor['mode'] == 'ok' and self.active is None:
            self._open_sensor(obs, 'noise', ref=float(np.mean([v for _, _, _, v in self._z])) if self._z else 0.0)

    def _open_sensor(self, obs, trigger, ref=0.0, jump=0.0):
        a = self.a
        nominal, est = a.health.nominal, max(a.health.sigma, 2.5 * a.health.nominal)
        penalty = obs.t - self._penalty_t < 45.0
        base = {'stuck': {'stuck': 0.75, 'noise': 0.03, 'bias': 0.02, 'ok': 0.1},
                'shift': {'bias': 0.4, 'noise': 0.2, 'stuck': 0.02, 'ok': 0.28},
                'noise': {'noise': 0.55, 'bias': 0.05, 'stuck': 0.02, 'ok': 0.28}}[trigger]
        if penalty:
            base['ok'] *= 0.4
        text = {'stuck': f'датчик образцов десять раз подряд выдал одно и то же значение {ref:.3f}, хотя робот едет',
                'shift': f'показание датчика упало на {jump:.2f} почти без движения',
                'noise': f'разброс показаний датчика вырос до {a.health.sigma:.2f} при норме {nominal:.2f}'}[trigger]
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
                                    predictions={'bias': (-jump, 0.05), 'ok': (0.0, 0.04), 'noise': (0.0, 0.12),
                                                 'stuck': (-jump, 0.2)},
                                    action={'kind': 'pause'}, sigma=0.02))
        q = Inquiry(self._qid(), obs.t, 'sensor', {'text': text, 'x': round(obs.x, 2), 'y': round(obs.y, 2),
                                                  'observed': round(jump or a.health.sigma, 3),
                                                  'expected': round(nominal, 3), 'unit': 'показание'},
                    alts, tests, energy_budget=1.0)
        q.ref, q.jump, q.trigger = ref, jump, trigger
        self._start(q, obs)

    def _apply_sensor(self, q, best, status, obs):
        a, t = self.a, obs.t
        until = t + self.know.get('fault_s', FAULT_S)
        if status != 'identified':
            a.health.degraded = True          # на всякий случай верим датчику меньше
            a.health.sigma = max(a.health.sigma, 2.0 * a.health.nominal)
            self.sensor = {'mode': 'noise', 't0': q.t_open, 'until': until}
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
            self.sensor = {'mode': 'stuck', 'value': q.ref, 't0': q.t_open, 'until': until}
            return 'показания датчика не учитываю, пока они не начнут меняться; образцы не собираю вслепую'
        shift = next((x.measured['value'] for x in q.tests if x.id == 'listen_shift' and x.measured), -q.jump)
        self.sensor = {'mode': 'bias', 'bias': abs(shift), 't0': q.t_open, 'until': until}
        return f'прибавляю к показаниям {abs(shift):.2f}, пока они не вернутся на прежний уровень'

    def _recovered(self, kind, t, why):
        t0 = self.sensor.get('t0', t)
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
