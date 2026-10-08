"""Правка P3: попутный сбор (research/findings/P3.md). За флажком настройки `pickup`, по умолчанию выключена.

Прежде робот пробовал сбор только на подъезде к назначенному кандидату. На разведке, в пути и по дороге на
базу он проезжал образец, даже когда его же карта говорила «образец в радиусе сбора» с той же уверенностью,
с какой он собирает на подъезде: подъезд к этому месту был не назначен (другая цель выгоднее, заряда на
«туда и обратно» по расчёту не хватает) или только что отменён решением о возврате.

Здесь робот в таком месте останавливается и собирает. Условия те же, что на подъезде (did/agent.py,
_do_investigate): уверенность `collect_confidence` в радиусе `collect_reach` вокруг робота, нет паузы после
двух промахов, не место недавнего промаха. Пути это не добавляет: робот уже стоит где нужно; цена — секунда
простоя. Сам сбор и учёт его исхода — прежний `_try_collect`.

Скрытой правды модуль не читает: только карта образцов агента и его поза.
"""
import math


class Pickup:

    RETRY_S = 2.0          # с: не сошлось после остановки — столько не пробуем снова, едем дальше

    def __init__(self, agent):
        self.a = agent
        self.cfg = agent.cfg
        self._sg = None            # идущий попутный сбор: временная подцель
        self._skip_until = -1e9
        self.count = 0             # сколько попутных сборов начато за прогон

    def here(self, obs, io):
        """Вызывается из разведки, пути и возврата. True — такт занят попутным сбором."""
        a, t = self.a, obs.t
        if self.cfg.search != 'belief' or t < a._no_collect_until or t < self._skip_until:
            return self._drop()
        here = a.belief.prob_within(obs.x, obs.y, self.cfg.collect_reach)
        if here < self.cfg.collect_confidence:
            if self._sg is not None:               # пока тормозил, уверенность упала: не собираю
                self._skip_until = t + self.RETRY_S
            return self._drop()
        if any(t - mt < 30.0 and math.hypot(obs.x - mx, obs.y - my) < 0.35 for mt, mx, my in a._misses):
            return self._drop()                    # здесь только что был промах
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

    def _drop(self):
        self._sg = None
        return False
