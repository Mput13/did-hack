"""Разовая потеря заряда при штрафе в опасной зоне: как отделить её от расхода на путь.

Судья при штрафе отнимает заряд одним скачком и сообщает о штрафе событием. Показание батареи и событие
приходят разными путями: в быстром симуляторе — одним наблюдением, в ROS — двумя темами, и батарея может
оказаться как раньше события, так и позже. Поэтому здесь ведётся короткая память: отрезки между соседними
показаниями батареи и недавние события штрафа. Событию сопоставляется ровно один отрезок.

- Агент знает из своих правил, сколько заряда стоит штраф. Если падение заряда на отрезке сходится с этим
  числом (остаток — правдоподобный расход на путь), из окна расхода вычитается именно оно.
- Если рядом с событием есть скачок, которого не объяснить даже самым дорогим грунтом, и он с этим числом
  не сходится, судья штрафует иначе. Из одного падения нельзя отделить неизвестный штраф от неизвестного
  расхода на грунт, поэтому окно с таким отрезком помечается загрязнённым: в оценки оно не идёт.
- Скачок без события остаётся странностью, но разбирается на WAIT_S позже: событие могло отстать.

Пока вопрос не решён (hold), закрытые окна расхода нужно придержать и не разбирать.
"""
from collections import deque

WAIT_S = 0.6     # на сколько показание батареи и событие штрафа могут разойтись во времени
STALE_S = 3.0    # дольше этого решения не ждём, даже если показания батареи перестали приходить
TOL = 0.15       # допуск на шум показаний батареи: пять разбросов разности двух показаний


class PenaltyLedger:

    def __init__(self, hit, most):
        self.hit = float(hit)          # сколько заряда отнимает штраф по правилам, в которые верит агент
        self.most = most               # most(ds, dth, dt, t): расход отрезка на самом дорогом грунте
        self.stats = {'hit': 0, 'bad': 0, 'lost': 0}    # штраф вычтен; не сошёлся; ожидался, но не найден
        self._tick = None              # последнее показание батареи и путь после него
        self._ticks = deque()          # недавние отрезки между показаниями
        self._events = []              # события штрафа, которым отрезок ещё не сопоставлен

    def event(self, t):
        self._events.append({'t': t})

    @property
    def hold(self):
        """Есть нерешённый вопрос: необъяснённый скачок ждёт события или событие ждёт своей потери заряда."""
        return any(k['state'] == 'odd' for k in self._ticks) or (self.hit > 0 and bool(self._events))

    def reading(self, obs, win, dth=0.0):
        """Каждый такт. win — окно расхода (словарь с ключом 'b0'), в которое попадёт это показание, или None."""
        k = self._tick
        if k is None:
            self._tick = {'t': obs.t, 'x': obs.x, 'y': obs.y, 'b': obs.battery, 'ds': 0.0, 'dth': 0.0}
            return
        k['ds'] += ((obs.x - k['x']) ** 2 + (obs.y - k['y']) ** 2) ** 0.5
        k['dth'] += dth
        k['x'], k['y'] = obs.x, obs.y
        if obs.battery != k['b']:      # показание не обновилось — путь копится до следующего показания
            drop = k['b'] - obs.battery
            most = self.most(k['ds'], k['dth'], obs.t - k['t'], obs.t)
            self._ticks.append({'t0': k['t'], 't1': obs.t, 'drop': drop, 'most': most, 'win': win,
                                'state': 'odd' if drop > most + TOL else 'plain'})
            k.update(t=obs.t, b=obs.battery, ds=0.0, dth=0.0)
        self._settle(obs.t)

    def _near(self, k, e):
        return k['t1'] >= e['t'] - WAIT_S and k['t0'] <= e['t'] + WAIT_S

    def _settle(self, now):
        for e in list(self._events):
            if not any(e is x for x in self._events):
                continue               # событие уже учтено вместе с соседним: два штрафа в одном отрезке
            free = sorted((k for k in self._ticks if k['state'] in ('plain', 'odd') and self._near(k, e)),
                          key=lambda k: (abs(k['t1'] - e['t']), -k['t1']))
            for k in free:
                m = self._fits(k)
                if m:
                    self._mark(k, 'hit', m * self.hit, sorted((x for x in self._events if self._near(k, x)),
                                                              key=lambda x: abs(x['t'] - e['t']))[:m])
                    break
            else:
                odd = next((k for k in free if k['state'] == 'odd'), None)
                if odd is not None:
                    self._mark(odd, 'bad', 0.0, [e])
                elif self._tick['t'] >= e['t'] + WAIT_S or now >= e['t'] + STALE_S:
                    # Ждать больше нечего. Если потеря ожидалась и не нашлась, она спрятана где-то рядом
                    # и неизвестна: все отрезки вокруг события загрязнены.
                    if self.hit > 0:
                        self.stats['lost'] += 1
                        for k in free:
                            self._mark(k, 'bad', 0.0, [])
                    self._drop(e)
        for k in self._ticks:
            if k['state'] == 'odd' and now >= k['t1'] + WAIT_S:
                k['state'] = 'plain'   # событие так и не пришло: это настоящая странность, а не штраф
        while self._ticks and self._ticks[0]['t1'] < now - 2 * WAIT_S - STALE_S:
            self._ticks.popleft()

    def _fits(self, k):
        """Сколько штрафов сразу объясняют падение заряда на отрезке (0 — ни один)."""
        if self.hit <= 0:
            return 0
        n = sum(1 for e in self._events if self._near(k, e))
        return next((m for m in range(1, n + 1) if -TOL <= k['drop'] - m * self.hit <= k['most'] + TOL), 0)

    def _mark(self, k, state, loss, events):
        k['state'] = state
        if state in self.stats:
            self.stats[state] += len(events)
        if k['win'] is not None:
            if state == 'hit':
                k['win']['b0'] -= loss
            else:
                k['win']['bad'] = True
        for e in events:
            self._drop(e)

    def _drop(self, e):
        self._events = [x for x in self._events if x is not e]
