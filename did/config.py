"""Правила стенда и параметры робота.

Из условия задачи взяты: старт (−2.0; −0.5), батарея 60, датчик образцов 0..1 с шумом без
направления, сбор ближе 0,30 м, число образцов и зон по уровням, виды событий.
Всё остальное (формула расхода, дальность и шум датчика, штрафы, счёт) в условии не задано —
это допущения стенда, они собраны здесь, чтобы их можно было заменить на правила судьи
организаторов.
"""
from dataclasses import asdict, dataclass

BASE = (-2.0, -0.5)          # старт и база, мировые координаты
ROBOT_RADIUS = 0.105         # TurtleBot3 Burger, м
V_MAX = 0.22                 # м/с
W_MAX = 2.84                 # рад/с
ACC_MAX = 1.0                # м/с², из модели Burger (max_linear_acceleration)
LIDAR_RAYS = 360
LIDAR_MAX = 3.5
LIDAR_MIN = 0.12
LIDAR_SIGMA = 0.01

LEVELS = {
    #          образцов, зон грунта, опасных зон на старте
    'easy':   {'samples': 3, 'soils': 1, 'hazards': 0},
    'medium': {'samples': 5, 'soils': 3, 'hazards': 0},
    'hard':   {'samples': 7, 'soils': 4, 'hazards': 1},
}


@dataclass(frozen=True)
class Rules:
    battery_start: float = 60.0
    drain_per_m: float = 2.5          # ед. заряда на метр по обычному полу: полной батареи хватает на 24 м
    drain_idle_per_s: float = 0.01    # ед. в секунду просто за то, что робот включён
    sensor_range_m: float = 2.0       # датчик образцов: 1 вплотную, 0 дальше этой дистанции
    sensor_sigma: float = 0.05
    sensor_hz: float = 5.0
    collect_radius_m: float = 0.30
    base_radius_m: float = 0.30
    time_limit_s: float = 600.0
    hazard_battery_hit: float = 3.0   # разовая потеря заряда при входе в опасную зону
    hazard_repeat_s: float = 5.0      # повтор штрафа, пока робот стоит в зоне
    # --- «научные» правила: у роста расхода и у странностей датчика несколько настоящих причин.
    # По умолчанию выключены (нули), чтобы прежние серии воспроизводились; набор SCIENCE ниже их включает.
    drain_per_rad: float = 0.0        # заряд за радиан поворота
    load_drain: float = 0.0           # на какую долю каждый несомый образец удорожает метр
    battery_sigma: float = 0.0        # шум показаний /did/battery
    faults: bool = False              # опасная зона вызывает сбой (утечка заряда или отказ датчика), а не разовую потерю
    leak_per_s: float = 0.15          # утечка заряда в секунду при сбое батареи
    sensor_fault_sigma: float = 0.25  # шум датчика при сбое «шум»
    sensor_bias: float = 0.2          # на сколько занижены показания при сбое «смещение»
    collision_clearance_m: float = 0.11
    # счёт
    pts_sample: float = 10.0
    pts_return: float = 20.0
    pts_collision: float = -2.0
    pts_false_collect: float = -3.0
    pts_hazard_hit: float = -5.0
    pts_battery_left: float = 0.1     # за единицу остатка, только если робот вернулся

    def to_dict(self):
        return asdict(self)


# Набор правил с несколькими причинами одного симптома: на нём идут расследования агента.
SCIENCE = {'drain_per_rad': 0.12, 'load_drain': 0.05, 'battery_sigma': 0.02, 'faults': True,
           'hazard_battery_hit': 0.0}
FAULT_KINDS = ('leak', 'sensor_noise', 'sensor_stuck', 'sensor_bias')
