"""Правки P2: бережный расход заряда там, где робот не знает среду (research/findings/P2.md).

Три независимые части, каждая за своим флажком настройки агента (по умолчанию все выключены):

  known_ground   — дорога к цели предпочитает пол, по которому робот уже ездил. На нехоженом полу его ждут
                   и невидимая опасная зона, и дорогой грунт; на хоженом — только то, что появилось позже.
                   Прежде так выбиралась только дорога домой (unknown_risk).
  own_drain      — робот сверяет расход каждого отрезка пути с тем, что обещала его карта стоимостей до
                   проезда, и, если расход устойчиво выше, поднимает цену всех будущих дорог: и до цели, и
                   домой. Причину превышения (груз, повороты, чужие правила судьи) он не знает и не угадывает.
  sensor_offset  — робот замечает, что датчик образцов занижает (или завышает) показания на постоянную
                   величину, и поправляет их (см. класс OffsetWatch).

Ни одна часть не читает правил судьи сверх того, что агенту дано и раньше (номинальный расход на метр,
дальность и шум датчика образцов), и ничего не знает о сценарии.
"""
import math
from collections import deque

import numpy as np


class Frugal:

    GAUGE_M = 3.0          # м: по скольким последним метрам пути судим о превышении расхода
    GAUGE_MIN_M = 1.0      # м: меньше — замеров мало, поправки нет
    GAUGE_FREE = 1.15      # превышение, уже покрытое запасом на возврат, с которым агент отлажен
    GAUGE_MAX = 1.6        # предел поправки: дальше — не свойство дороги, а разовая потеря

    @staticmethod
    def wanted(cfg):
        return bool(cfg.known_ground or cfg.own_drain or cfg.sensor_offset)

    def __init__(self, agent):
        self.a = agent
        self.cfg = agent.cfg
        self._gauge = deque()              # последние отрезки пути: (длина, расход по факту, расход по карте)
        self._gauge_m = 0.0
        self.offset = OffsetWatch(agent) if agent.cfg.sensor_offset else None

    # --- дорога к цели по хоженому полу ----------------------------------------------------------

    def outbound_bias(self, danger):
        """Во сколько раз клетка нежелательна для дороги к цели: риск опасных зон и нехоженый пол."""
        a = self.a
        if not (self.cfg.known_ground and a.cfg.learn_soil) or a._truth is not None:
            return danger
        unknown = 1.0 + self.cfg.known_ground * (1.0 - a.soil.confidence_grid())
        return unknown if danger is None else danger * unknown

    # --- свой замер расхода ----------------------------------------------------------------------

    def segment(self, ds, ratio, predicted):
        """Отрезок пути ds: расход оказался в ratio раз выше обычного пола, карта до проезда обещала predicted."""
        if not self.cfg.own_drain or ds <= 0.0:
            return
        self._gauge.append((ds, ratio * ds, predicted * ds))
        self._gauge_m += ds
        while self._gauge_m - self._gauge[0][0] >= self.GAUGE_M:
            self._gauge_m -= self._gauge.popleft()[0]

    def excess(self):
        """Во сколько раз расход на последних метрах пути выше того, что обещала карта (None — замеров мало)."""
        if self._gauge_m < self.GAUGE_MIN_M:
            return None
        spent = sum(g[1] for g in self._gauge)
        promised = sum(g[2] for g in self._gauge)
        return spent / promised if promised > 0.0 else None

    def drain_factor(self):
        """Множитель к цене будущих дорог: 1, пока превышение не больше обычного."""
        k = self.excess() if self.cfg.own_drain else None
        if k is None:
            return 1.0
        return min(self.GAUGE_MAX, max(1.0, k / self.GAUGE_FREE))


class OffsetWatch:
    """Постоянный сдвиг показаний датчика образцов: заметить, оценить и поправить.

    Исправный датчик показывает 1 − d / дальность, где d — расстояние до ближайшего образца. Если показания
    занижены на постоянную величину, карта образцов рисует образец дальше, чем он есть: робот проезжает
    прямо по образцу и не пробует его собрать. Уровень показаний сдвиг не выдаёт — «0,8» бывает и в 40 см
    от образца, и вплотную при заниженных показаниях. Его выдаёт форма: как показание меняется вдоль пути
    робота. Рядом с образцом оно меняется круто и с изломом, в стороне от него — плавно.

    Поэтому к последним показаниям (окно в несколько секунд, пока ближайший образец один и тот же)
    подбираются сразу место образца и сдвиг — наименьшими квадратами по клеткам карты. Сдвиг считается
    найденным, если с ним показания объясняются намного лучше, чем без него, остаток подбора не больше
    обычного шума датчика, а оценка держится несколько показаний подряд. Тогда показания поправляются до
    того, как попасть в карту, а уже учтённые показания окна пересчитываются.

    Поправка не вечна: она снимается, если её давно не подтверждал новый подбор, если поправленное показание
    стало невозможным (заметно больше 1), если показания стали лучше объясняться без сдвига или если сбор
    при поправке оказался ложным. Успешный сбор — известное расстояние (не дальше радиуса сбора): по нему
    сдвиг сверяется ещё раз.
    """

    WINDOW_N = 30          # показаний в окне подбора, не больше
    WINDOW_MIN = 30        # и не меньше: на коротком окне сдвиг не отличить от образца в стороне от пути
    PATH_MIN = 0.30        # м пути за окно: стоя на месте форму не увидеть
    GAIN = 12.0            # во сколько квадратов шума сдвиг должен улучшить подбор (хи-квадрат с одной степенью)
    FIT = 1.4              # остаток подбора на показание — не больше стольких шумов датчика
    MIN_SHIFT = 0.10       # меньший сдвиг не отличить от неточности карты, и вреда от него мало
    MAX_SHIFT = 0.30       # большим оценкам не верим: на отладочных сценариях они оказывались ошибкой подбора
    LOW = 0.15             # показания ниже этого в подбор не идут: рядом с нулём шкала обрезает шум
    CONFIRM = 3            # столько подборов подряд должны дать близкую оценку
    AGREE = 0.06           # насколько близкую
    BAND = 4.0             # подборы хуже лучшего не больше чем на столько квадратов шума считаются равноправными
    HOLD_S = 20.0          # с: столько поправка живёт без нового подтверждения
    NEAR = 0.85            # показание, ниже которого сбор не засчитывают: 1 − радиус сбора / дальность

    def __init__(self, agent):
        self.a = agent
        self.shift = 0.0                   # на сколько сейчас поправляются показания
        self.until = -1e9                  # до какого времени поправка действует без подтверждения
        self.found = 0                     # сколько раз за прогон сдвиг найден
        self._buf = deque(maxlen=self.WINDOW_N)     # (x, y, показание как есть, номер в журнале карты)
        self._seen = deque(maxlen=self.CONFIRM)     # последние оценки сдвига подряд
        self._last = None                  # последнее показание как есть
        self._est = []                     # подтверждённые оценки сдвига в нынешнем эпизоде

    def _sigma(self):
        return math.hypot(self.a.rules.sensor_sigma, 0.02)

    def reading(self, z, obs):
        """Показание z → показание для карты образцов (None — в карту не вносить)."""
        a, t = self.a, obs.t
        self._last = z
        if a.cfg.sensor_health and a.health.degraded:
            self._buf.clear()              # шумящий датчик: форму показаний не разобрать
            self._seen.clear()
            return self._fixed(z)
        if self.LOW <= z < 1.0 - 1e-6:     # показания у края шкалы обрезаны и о сдвиге ничего не говорят
            self._buf.append((obs.x, obs.y, z, len(a.belief._log)))
        if self.shift and (t > self.until or z + self.shift > 1.0 + 2.5 * self._sigma()):
            self._drop(t, 'давно не подтверждался' if t > self.until else 'поправленное показание больше единицы')
        fit = self._fit()
        if fit is not None:
            shift, gain, ok, back = fit
            if ok and gain >= self.GAIN and self.MIN_SHIFT <= shift <= self.MAX_SHIFT:
                self._seen.append(shift)
                if len(self._seen) == self.CONFIRM and max(self._seen) - min(self._seen) <= self.AGREE:
                    self._set(t, float(np.median(self._seen)), back)
            else:
                self._seen.clear()
                if self.shift and ok and back is not None and back >= self.GAIN:
                    self._drop(t, 'показания снова объясняются без сдвига')
        return self._fixed(z)

    def _fixed(self, z):
        if not self.shift:
            return z
        if z <= 1e-6 and self.shift > 0.0:
            return None                    # «ноль» при заниженных показаниях — не «образцов рядом нет»
        return min(1.0, max(0.0, z + self.shift))

    def _fit(self):
        """Подбор места образца и сдвига к окну: (сдвиг, выигрыш подбора со сдвигом, годен ли подбор,
        выигрыш подбора без сдвига перед подбором с нынешней поправкой — или None, если поправки нет)."""
        if len(self._buf) < self.WINDOW_MIN:
            return None
        x, y, z, _ = (np.array(v) for v in zip(*self._buf))
        if float(np.hypot(np.diff(x), np.diff(y)).sum()) < self.PATH_MIN or len(set(z.tolist())) < len(z) // 2:
            return None                    # робот стоял — или датчик залип: одинаковые показания подряд
        b = self.a.belief
        near = np.hypot(b.cx - x[-1], b.cy - y[-1]) <= b.range      # образец, который слышен, не дальше дальности датчика
        if not near.any():
            return None
        f = 1.0 - np.hypot(b.cx[near][None, :] - x[:, None], b.cy[near][None, :] - y[:, None]) / b.range
        r = z[:, None] - f                                  # остаток без сдвига: показание минус ожидаемое
        n = len(z)
        sse0 = (r ** 2).sum(axis=0)
        mean = r.mean(axis=0)
        free = sse0 - n * mean ** 2                         # остаток с наилучшим сдвигом для каждой клетки
        j = int(free.argmin())
        s2 = self._sigma() ** 2
        ok = free[j] / max(n - 3, 1) <= self.FIT ** 2 * s2
        back = None
        if self.shift:
            held = ((r + self.shift) ** 2).sum(axis=0)       # остаток при нынешней поправке
            back = float(held.min() - sse0.min()) / s2
        # Сдвиг и место образца отчасти заменяют друг друга, поэтому в поправку идёт не лучшая оценка, а
        # наименьший сдвиг среди подборов, почти не уступающих лучшему: недобрать поправку безопасно
        # (робот просто подъедет ближе), перебрать — ложный сбор.
        low = float((-mean[free <= free[j] + self.BAND * s2]).min())
        return low, float(sse0.min() - free[j]) / s2, bool(ok), back

    def _set(self, t, shift, back):
        a = self.a
        new = not self.shift
        self.until = t + self.HOLD_S
        self._est.append(shift)            # каждая новая оценка уточняет поправку: берётся их медиана
        shift = float(np.median(self._est))
        self.shift = shift
        if new:
            self.found += 1
            first = self._buf[0][3]        # показания окна уже в карте — пересчитать их с поправкой
            a.belief.amend(len(a.belief._log) - first, shift)
            word = 'занижает' if shift > 0 else 'завышает'
            a.journal.add(t, 'alarm', f'Датчик образцов {word} показания примерно на {abs(shift):.2f}: вдоль пути они '
                          'меняются так, как бывает только рядом с образцом. Поправляю показания', tag='sensor_offset',
                          shift=round(shift, 3))
            a.journal.open(t, f'offset:{self.found}', f'датчик образцов {word} показания на {abs(shift):.2f}',
                           'подъехать к образцу по поправленным показаниям и собрать его')
            a._request_plan('sensor_offset')

    def _drop(self, t, why):
        a = self.a
        a.journal.add(t, 'action', f'Поправку показаний датчика образцов ({self.shift:+.2f}) снимаю: {why}',
                      tag='sensor_offset_end')
        a.journal.close(t, f'offset:{self.found}', 'outdated', why)
        self.shift = 0.0
        self._est = []
        self._seen.clear()

    def collected(self, t):
        """Образец взят: робот не дальше радиуса сбора, значит исправный датчик показывал бы не меньше NEAR."""
        z, s = self._last, 2.5 * self._sigma()
        self._buf.clear()                  # ближайший образец теперь другой
        self._seen.clear()
        if z is not None and self.shift:
            self.a.journal.close(t, f'offset:{self.found}', 'confirmed', 'образец собран по поправленным показаниям')
            if z + self.shift < self.NEAR - s or z + self.shift > 1.0 + s:
                self._drop(t, 'сбор показал, что поправка неверна')
            else:
                self.until = max(self.until, t + self.HOLD_S / 2)

    def missed(self, t):
        if self.shift:
            self._drop(t, 'сбор по поправленным показаниям оказался ложным')
