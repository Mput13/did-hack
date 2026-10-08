"""Что агент видит и чем управляет. Одинаково для быстрого симулятора и для ROS 2 + Gazebo."""
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


@dataclass
class Observation:
    t: float                       # секунды от начала прогона (время симуляции)
    x: float                       # мировая поза: точка старта + одометрия
    y: float
    th: float
    v: float
    w: float
    battery: float                 # /did/battery
    sensor: Optional[float]        # свежее показание /did/sample_sensor или None, если нового нет
    scan: Optional[np.ndarray]     # свежий скан лидара (360 дальностей от курса против часовой) или None
    events: list = field(default_factory=list)   # новые сообщения /did/events
    done: bool = False             # судья завершил прогон
    scan_pose: Optional[tuple] = None   # поза по одометрии в момент скана (x, y, th); None — текущая
    scan_step: Optional[float] = None   # угол между лучами скана, рад; None — 2π / число лучей


class RobotIO(Protocol):
    def observe(self) -> Observation: ...
    def command(self, v: float, w: float) -> None: ...
    def collect(self) -> tuple: ...     # (успех, сообщение) — /did/collect
    def finish(self) -> tuple: ...      # (успех, сообщение) — /did/finish
