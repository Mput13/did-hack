"""Короткий прогон пульта в обычном режиме (готовая карта): одни и те же команды → позы робота.

Эталон tests/data/pilot_main_poses.json снят этим же файлом с кода ветки main (коммит указан в эталоне),
проверка tests/test_slam_map.py::test_normal_mode_matches_main сравнивает с ним нынешний код. Переснять:

    mkdir /tmp/main && git archive <коммит main> did maps | tar -x -C /tmp/main
    cd /tmp/main && python <дерево>/tests/data/pilot_main_poses.py <коммит main>
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).with_name('pilot_main_poses.json')
# (такт, команда): маршрут из двух точек, остановка на ходу, снова ехать, домой, короткий отрезок миссии.
SCRIPT = [(5, {'cmd': 'route', 'points': [[-1.2, -0.6], [-0.5, 0.6]]}), (6, {'cmd': 'go'}), (120, {'cmd': 'stop'}),
          (140, {'cmd': 'go'}), (420, {'cmd': 'home'}), (700, {'cmd': 'mission', 'agent': 'adaptive'})]
TICKS = 1100


def run():
    from did.arena import load_arena
    from did.pilot import FastWorld, Pilot
    arena = load_arena()
    world = FastWorld(arena, 'medium', 3)
    pilot = Pilot(arena, world, 'medium', 3)
    script = dict(SCRIPT)
    rows = []
    for n in range(TICKS):
        if n in script:
            res = pilot.command(script[n])
            rows.append({'tick': n, 'cmd': script[n]['cmd'], 'ok': res['ok'], 'message': res['message']})
        pilot.tick(world.observe())
        world.advance()
        if n % 10 == 0:
            rows.append({'tick': n, 'mode': pilot.mode, 'pose': [round(v, 6) for v in pilot.pose],
                         'true': [round(v, 6) for v in (world.sim.x, world.sim.y, world.sim.th)],
                         'cmd_vw': [round(v, 6) for v in pilot._last_cmd], 'arrivals': list(pilot.arrivals)})
    return rows


if __name__ == '__main__':
    sys.path.insert(0, str(Path.cwd()))
    OUT.write_text(json.dumps({'main_commit': sys.argv[1], 'rows': run()}, ensure_ascii=False, indent=0))
    print(OUT)
