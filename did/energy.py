"""Модель расхода заряда, которую агент уточняет сам: метры, груз, повороты, время.

Заранее агенту известны только номинальный расход на метр и за простой. Сколько стоят повороты и
как влияет груз, в условии не сказано — эти коэффициенты подбираются по ходу прогона рекурсивным
методом наименьших квадратов. Измерение, которого модель не ждала, в подбор не идёт: это повод
для расследования (did/inquiry.py), а не для подгонки.
"""
import math

import numpy as np

NAMES = ('per_m', 'per_m_load', 'per_rad', 'per_s')
LABELS = {'per_m': 'заряд на метр обычного пола', 'per_m_load': 'добавка на метр за каждый несомый образец',
          'per_rad': 'заряд на радиан поворота', 'per_s': 'заряд за секунду простоя'}
PRIOR_SD = (0.25, 0.15, 0.15, 0.01)


class EnergyModel:

    def __init__(self, per_m, per_s, noise=0.03, prior=None):
        self.mean = np.array([per_m, 0.0, 0.0, per_s], dtype=float)
        self.cov = np.diag(np.square(PRIOR_SD))
        if prior:                                     # знания из прошлых прогонов (did/memory.py)
            for i, name in enumerate(NAMES):
                if name in prior:
                    self.mean[i] = prior[name]['value']
                    self.cov[i, i] = max(prior[name]['sigma'], 0.005) ** 2
        self.noise = noise                            # шум разности двух показаний батареи
        self.n = 0

    @staticmethod
    def features(ds, dth, dt, carried):
        return np.array([ds, ds * carried, dth, dt], dtype=float)

    def predict(self, ds, dth, dt, carried, mult=1.0):
        """Ожидаемый расход на окно и разброс прогноза (неуверенность в коэффициентах плюс шум)."""
        x = self.features(ds, dth, dt, carried)
        x[:2] *= mult
        return float(x @ self.mean), math.sqrt(float(x @ self.cov @ x) + self.noise ** 2)

    def learn(self, ds, dth, dt, carried, spent):
        """Уточнить коэффициенты по окну, пройденному по обычному полу."""
        x = self.features(ds, dth, dt, carried)
        gain = self.cov @ x / (float(x @ self.cov @ x) + self.noise ** 2)
        self.mean = self.mean + gain * (spent - float(x @ self.mean))
        self.cov = self.cov - np.outer(gain, x) @ self.cov
        self.mean[1:] = np.maximum(self.mean[1:], 0.0)         # отрицательных цен не бывает
        self.n += 1

    def per_meter(self, carried):
        return float(self.mean[0] + self.mean[1] * carried)

    def soil_ratio(self, ds, dth, dt, carried, spent):
        """Во сколько раз этот отрезок дороже обычного пола, если вычесть повороты и простой."""
        return (spent - self.mean[2] * dth - self.mean[3] * dt) / max(self.per_meter(carried) * ds, 1e-6)

    def widen(self, name, sd):
        """Перестать верить оценке одного коэффициента: пусть переучится."""
        i = NAMES.index(name)
        self.cov[i, :] = self.cov[:, i] = 0.0
        self.cov[i, i] = sd ** 2

    def summary(self):
        return {name: {'value': round(float(self.mean[i]), 4), 'sigma': round(math.sqrt(max(self.cov[i, i], 0)), 4),
                       'label': LABELS[name]} for i, name in enumerate(NAMES)}
