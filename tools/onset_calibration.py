#!/usr/bin/env python3
"""Калибровка сравнения двух версий карты образцов после штрафа (did/inquiry.py, константа ONSET).

После въезда в опасную зону исследователь ведёт две карты образцов от общей исходной: «датчик
исправен» и «показания занижены». Этот скрипт на отладочных сценариях измеряет, какой перевес
версии «занижает» набирается за ONSET_S секунд при каждом настоящем состоянии датчика.

    pixi run python tools/onset_calibration.py            # сценарии 1–40
    pixi run python tools/onset_calibration.py 1001 40    # проверка на отложенных сценариях
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.agent import Agent                      # noqa: E402
from did.arena import load_arena                 # noqa: E402
from did.config import SCIENCE, Rules            # noqa: E402
from did.fastsim import FastSim                  # noqa: E402
from did.runner import make_config               # noqa: E402
from did.scenario import generate                # noqa: E402

KINDS = {'ok': 'leak', 'bias': 'sensor_bias', 'noise': 'sensor_noise', 'stuck': 'sensor_stuck'}


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    arena = load_arena()
    print(f'сценарии hard {start}–{start + count - 1}; перевес версии «занижает» к концу окна')
    for name, kind in KINDS.items():
        values = []
        for seed in range(start, start + count):
            rules = dataclasses.replace(Rules(), **SCIENCE)
            # без сбоя датчика по расписанию: он наложился бы на проверяемое состояние
            scenario = generate('hard', seed, arena, fault_kinds=[kind], n_hazards=3,
                                events=['soil_change', 'new_hazard'])
            world = FastSim(arena, scenario, rules=rules, seed=seed)
            bot = Agent(arena, make_config('scientist'), n_samples=len(scenario.samples), rules=rules)
            seen = None
            while not world.done:
                bot.tick(world.observe(), world)
                world.advance()
                on = bot.inv._onset
                if on is not None and on is not seen and on['done'] and on['n'] >= 5:
                    seen = on
                    values.append(on['lbf'])
        v = np.array(values)
        print(f'  {name:5s} эпизодов {len(v):3d}: среднее {v.mean():+6.1f}, разброс {v.std():5.1f}, '
              f'10% {np.percentile(v, 10):+6.1f}, медиана {np.median(v):+6.1f}, 90% {np.percentile(v, 90):+6.1f}')


if __name__ == '__main__':
    main()
