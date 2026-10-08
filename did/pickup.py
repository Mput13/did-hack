"""Правка P3: попутный сбор (research/findings/P3.md). За флажком настройки `pickup`, по умолчанию выключена.

Прежде робот пробовал сбор только на подъезде к назначенному кандидату. На разведке, в пути и по дороге на
базу он проезжал образец, даже когда его же карта говорила «образец в радиусе сбора» с той же уверенностью,
с какой он собирает на подъезде: подъезд к этому месту был не назначен (другая цель выгоднее, заряда на
«туда и обратно» по расчёту не хватает) или только что отменён решением о возврате.

Здесь робот в таком месте останавливается и собирает. Условия те же, что на подъезде (did/agent.py,
_do_investigate): уверенность `collect_confidence` в радиусе `collect_reach` вокруг робота, нет паузы после
двух промахов, не место недавнего промаха. Пути это не добавляет: робот уже стоит где нужно; цена — секунда
простоя. Сам сбор и учёт его исхода — прежний `_try_collect`.

По дороге на базу (правка по ревью P3) попутный сбор уступает возврату: у базы, где уже доступен финиш, робот
финиширует, а не собирает; в пути собирает только при запасе заряда и времени сверх дороги домой. Запас — тот
же, по которому агент решает, можно ли ещё стоять на месте (did/agent.py, _cannot_wait).

Скрытой правды модуль не читает: только карта образцов агента и его поза.
"""
import math


class Pickup:

    RETRY_S = 2.0          # с: не сошлось после остановки — столько не пробуем снова, едем дальше
    FINISH_TOL = 0.08      # м: ближе к базе доступен финиш (допуск _do_return) — он важнее сбора
    SPARE_BATTERY = 0.5    # ед. сверх дороги домой с reserve_margin: меньше — по дороге домой не останавливаюсь
    SPARE_TIME_S = 20.0    # с сверх дороги домой на осторожной скорости ALERT_V

    def __init__(self, agent):
        self.a = agent
        self.cfg = agent.cfg
        self._sg = None            # идущий попутный сбор: временная подцель
        self._skip_until = -1e9
        self.count = 0             # сколько попутных сборов начато за прогон

    def here(self, obs, io, homeward=False):
        """Вызывается из разведки, пути и возврата (homeward). True — такт занят попутным сбором."""
        a, t = self.a, obs.t
        if homeward and math.dist((obs.x, obs.y), a.base) <= self.FINISH_TOL:
            return self._drop()                    # робот на базе: финиш важнее
        if self.cfg.search != 'belief' or t < a._no_collect_until or t < self._skip_until:
            return self._drop()
        here = a.belief.prob_within(obs.x, obs.y, self.cfg.collect_reach)
        if here < self.cfg.collect_confidence:
            if self._sg is not None:               # пока тормозил, уверенность упала: не собираю
                self._skip_until = t + self.RETRY_S
            return self._drop()
        if any(t - mt < 30.0 and math.hypot(obs.x - mx, obs.y - my) < 0.35 for mt, mx, my in a._misses):
            return self._drop()                    # здесь только что был промах
        if homeward and not self._spare(obs):
            return self._drop()                    # запаса на остановку нет: домой
        if self._sg is None:
            self._sg = {'type': 'investigate', 'x': obs.x, 'y': obs.y, 'pickup': True}
            self.count += 1
            a.journal.add(t, 'decision', f'Образец в радиусе сбора у ({obs.x:.2f}; {obs.y:.2f}), уверенность '
                          f'{here:.0%}: собираю попутно, цель прежняя')
        # Сбор и учёт исхода — как на подъезде. Подцель временная: _try_collect снимает её сам, когда сбор
        # состоялся (удачно или нет); пока робот тормозит, её снимаю я, и следующий такт начнётся здесь же.
        sg = self._sg
        a.queue.insert(0, sg)
        a._try_collect(sg, obs, io, here)
        if a.queue and a.queue[0] is sg:
            a.queue.pop(0)
        else:
            self._sg = None
        return True

    def _spare(self, obs):
        """Есть ли по дороге домой запас на остановку — мерка та же, что в Agent._cannot_wait."""
        a = self.a
        home = a._home_cost(obs.x, obs.y)
        if obs.battery <= home * self.cfg.reserve_margin + self.SPARE_BATTERY:
            return False
        return a.rules.time_limit_s - obs.t > home / a.rules.drain_per_m / a.ALERT_V + self.SPARE_TIME_S

    def _drop(self):
        self._sg = None
        return False
