#!/usr/bin/env python3
"""Сверка карты SLAM Toolbox с готовой картой и с миром: одно число для жюри, которое можно повторить.

Запускать при работающем показе в режиме карты SLAM (в другом терминале `pixi run demo --slam-map`):

    pixi run python tools/slam_check.py --explore     # робот сам объедет арену, затем сверка
    pixi run python tools/slam_check.py               # сверка карты, какая есть сейчас
    pixi run python tools/slam_check.py --goal 1.5 1.2    # ещё и доехать до точки по карте SLAM
    pixi run python tools/slam_check.py --from runs/F2/slam_map.npz   # пересчитать по сохранённой сетке

Что считается (did/slam_map.py):
- какую долю пола SLAM увидел и в какой доле увиденных клеток он согласен с готовой картой (свободно / занято);
- где на карте SLAM стоят девять столбов мира и на сколько сантиметров каждый отличается от своего места
  (±1,1; ±1,1), (±1,1; 0), (0; ±1,1), (0; 0);
- сдвиг и поворот начала координат карты относительно мирового — по тем же столбам.

Сетка берётся из темы /slam/map (нужен тот же ROS_DOMAIN_ID, что у показа) и сохраняется в runs/F2/, числа —
в runs/F2/slam_check.json. Готовая карта используется только здесь, для сверки: роботу она не передаётся.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.arena import load_arena                                    # noqa: E402
from did.slam_map import PILLARS, origin_offset, outline, pillars, report, resample                  # noqa: E402

OUT = ROOT / 'runs' / 'F2'


def call(url, body=None, timeout=20.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {'ok': False, 'error': json.loads(e.read() or b'{}').get('error', str(e))}


def read_slam_map(topic='/slam/map', wait_s=15.0):
    """Последняя сетка из темы SLAM Toolbox: (сетка, клетка, x0, y0, кадр)."""
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile

    rclpy.init()
    node = rclpy.create_node('did_slam_check')
    got = []
    node.create_subscription(OccupancyGrid, topic, got.append,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    deadline = time.monotonic() + wait_s
    try:
        while not got and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if not got:
        raise SystemExit(f'за {wait_s:.0f} с в теме {topic} ничего нет: запущен ли показ с ключом --slam-map '
                         'и совпадает ли ROS_DOMAIN_ID?')
    i = got[-1].info
    grid = np.asarray(got[-1].data, dtype=np.int8).reshape(i.height, i.width)
    return grid, float(i.resolution), float(i.origin.position.x), float(i.origin.position.y), got[-1].header.frame_id


def drive(base, explore, goal, limit):
    """Объезд и поездка к точке через пульт. Возвращает то, что получилось, для отчёта."""
    out = {}

    def state():
        return call(base + '/state', timeout=5.0)

    def wait(done, limit_s):
        t0 = time.time()
        while time.time() - t0 < limit_s:
            s = state()
            if done(s):
                return s, time.time() - t0
            time.sleep(0.5)
        return None, limit_s

    def healthy(s):
        return ((s.get('nav') or {}).get('slam') or {}).get('ok')

    s, _ = wait(lambda s: s.get('active') and s.get('linked') and s.get('mode') == 'idle'
                and (s.get('nav') or {}).get('ready') and healthy(s), 180)
    if s is None:
        raise SystemExit('пульт не готов: нет связи со стендом, карта SLAM не пришла или SLAM молчит')
    if not s.get('slam_nav'):
        raise SystemExit('показ запущен без --slam-map: робот едет по готовой карте, проверять нечего')
    if explore:
        print('Объезд: робот едет к границам увиденного, пока они не кончатся')
        res = call(base + '/command', {'cmd': 'explore'})
        if not res.get('ok'):
            raise SystemExit(f"команда explore отклонена: {res.get('error')}")
        time.sleep(1.0)                                 # состояние на странице должно успеть обновиться
        # Пульт сам прекращает объезд, если SLAM замолчал или карта стала непригодной: тогда explore пропадает.
        s, dt = wait(lambda s: s['mode'] == 'idle' and (((s.get('nav') or {}).get('explore') or {}).get('done')
                                                      or (s.get('nav') or {}).get('explore') is None), limit)
        if s is None:
            raise SystemExit(f'объезд не закончился за {limit:.0f} с')
        if not (s['nav'].get('explore') or {}).get('done'):
            raise SystemExit(f"объезд прерван пультом: {s['note']['text']}")
        ex = s['nav']['explore']
        out['explore'] = {'seconds_sim': ex.get('seconds'), 'legs': ex['visited'], 'frontiers_left': ex.get('left'),
                          'distance': s['distance'], 'collisions': s['score'].get('collisions'),
                          'lag_max': s['lag']['max'], 'lag_late': s['lag']['late']}
        print(f"   {dt:.0f} с по часам; подъездов {ex['visited']}, пройдено {s['distance']:.1f} м, "
              f"столкновений {s['score'].get('collisions')}")
        time.sleep(2.5)                                 # дать SLAM опубликовать последнюю сетку
    if goal:
        print(f'Точка назначения ({goal[0]}; {goal[1]}) — путь по карте SLAM')
        res = call(base + '/command', {'cmd': 'route', 'points': [goal]})
        if not res.get('ok'):
            raise SystemExit(f"маршрут отклонён: {res.get('error')}")
        n = len(state().get('arrivals', []))
        call(base + '/command', {'cmd': 'go'})
        time.sleep(1.0)
        s, dt = wait(lambda s: s['mode'] == 'idle', 180)        # приехал либо пульт остановил езду с сообщением
        if s is None:
            raise SystemExit('робот не доехал до точки за 180 с')
        if len(s.get('arrivals', [])) <= n:
            raise SystemExit(f"робот до точки не доехал, пульт остановил езду: {s['note']['text']}")
        truth = (s.get('truth') or {}).get('robot')
        out['goal'] = {'goal': goal, 'pose_slam': s['pose'][:2], 'err_slam': s['arrivals'][-1],
                       'collisions': s['score'].get('collisions'), 'seconds': round(dt, 1),
                       'slam': (s.get('nav') or {}).get('slam')}
        if truth:
            out['goal']['err_true'] = round(math.dist(truth[:2], goal), 3)
        print(f"   {dt:.0f} с; до точки {s['arrivals'][-1] * 100:.0f} см по позе SLAM, "
              f"столкновений {s['score'].get('collisions')}")
        time.sleep(2.5)
    return out


def accepted(rep):
    """Порог приёмки карты: все девять столбов подтверждены, каждый не дальше 10 см, совпадение не ниже 90 %."""
    return bool(rep['pillars_found'] == len(PILLARS) and rep['pillar_err_max'] is not None
                and rep['pillar_err_max'] <= 0.10 and rep['agreement'] >= 0.9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765, help='порт сервера интерфейса (для --explore и --goal)')
    ap.add_argument('--explore', action='store_true', help='сначала объехать арену: команда пульта explore')
    ap.add_argument('--goal', type=float, nargs=2, metavar=('X', 'Y'), help='затем доехать до точки по карте SLAM')
    ap.add_argument('--limit', type=float, default=600.0, help='сколько секунд ждать конца объезда')
    ap.add_argument('--from', dest='source', help='не читать тему, а взять сохранённую сетку (.npz)')
    ap.add_argument('--tag', default='slam', help='имя файлов в runs/F2/')
    args = ap.parse_args()

    driven = {}
    if args.source:
        z = np.load(args.source)
        grid, res, ox, oy, frame = z['grid'], float(z['res']), float(z['ox']), float(z['oy']), str(z['frame'])
    else:
        if args.explore or args.goal:
            driven = drive(f'http://127.0.0.1:{args.port}/api/pilot', args.explore, args.goal, args.limit)
        grid, res, ox, oy, frame = read_slam_map()
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f'{args.tag}_map.npz', grid=grid, res=res, ox=ox, oy=oy, frame=frame)

    ref = load_arena()
    values = resample(grid, res, ox, oy, (ref.res, ref.x0, ref.y0, ref.w, ref.h))
    rep = report(values, ref)
    # Та же мерка для готовой карты: насколько она сама стоит в мире.
    ref_found = pillars(outline(ref), (ref.res, ref.x0, ref.y0, ref.w, ref.h))
    ref_errs = [p['err'] for p in ref_found if p['err'] is not None]
    rep['ready_map'] = {'pillars_found': len(ref_errs), 'pillar_err_mean': float(np.mean(ref_errs)),
                        'pillar_err_max': float(np.max(ref_errs)), 'origin': origin_offset(ref_found)}
    rep.update(frame=frame, grid={'w': int(grid.shape[1]), 'h': int(grid.shape[0]), 'res': res, 'origin': [ox, oy]}, **driven)

    cm = lambda v: '—' if v is None else f'{v * 100:.1f}'      # noqa: E731
    print(f'\nКарта SLAM Toolbox: кадр {frame}, сетка {grid.shape[1]}×{grid.shape[0]} по {res * 100:.0f} см, '
          f'угол сетки в мире ({ox:.2f}; {oy:.2f})')
    print(f"Пола увидено:            {rep['coverage']:.1%}")
    print(f"Совпало с готовой картой: {rep['agreement']:.1%} увиденных клеток "
          f"(свободные — {rep['free_ok']:.1%}, занятые — {rep['occ_ok']:.1%} у преграды готовой карты)")
    b = rep['best_shift']
    print(f"Лучший сдвиг карты целыми клетками: ({b['dx'] * 100:+.0f}; {b['dy'] * 100:+.0f}) см, "
          f"совпадение при нём {b['agreement']:.1%}")
    print('\nСтолбы: место в мире → центр на карте SLAM, ошибка')
    for p in rep['pillars']:
        where = '      не найден' if p['found'] is None else f"({p['found'][0]:+.3f}; {p['found'][1]:+.3f})"
        shape = f", остаток {cm(p['rms'])} см, секторов {p['arc']} из 12" if 'rms' in p else ''
        print(f"  ({p['x']:+.1f}; {p['y']:+.1f}) → {where}   {cm(p['err']):>5} см   клеток {p['cells']}{shape}"
              + (f" — не подтверждён: {p['why']}" if p.get('why') else ''))
    print(f"Найдено столбов {rep['pillars_found']} из {len(PILLARS)}; ошибка средняя {cm(rep['pillar_err_mean'])} см, "
          f"наибольшая {cm(rep['pillar_err_max'])} см")
    o = rep['origin']
    if o:
        print(f"Начало координат карты относительно мирового: сдвиг {o['shift'] * 100:.1f} см "
              f"({o['dx'] * 100:+.1f}; {o['dy'] * 100:+.1f}), поворот {o['rot_deg']:+.2f}°; "
              f"после их вычитания столбы расходятся в среднем на {o['residual'] * 100:.1f} см")
    r = rep['ready_map']
    print(f"Для сравнения, готовая карта той же меркой: столбов подтверждено {r['pillars_found']} из {len(PILLARS)}, "
          f"ошибка столбов средняя {cm(r['pillar_err_mean'])} см, "
          f"наибольшая {cm(r['pillar_err_max'])} см; сдвиг начала координат {r['origin']['shift'] * 100:.1f} см "
          f"({r['origin']['dx'] * 100:+.1f}; {r['origin']['dy'] * 100:+.1f})")
    if not args.source:
        (OUT / f'{args.tag}_check.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"\nСетка: {OUT / f'{args.tag}_map.npz'}; числа: {OUT / f'{args.tag}_check.json'}")
    ok = accepted(rep)
    print('\nИтог: ' + ('карта SLAM совмещена с миром (все девять столбов подтверждены кольцом нужного радиуса и стоят '
                        'на месте с точностью 10 см, совпадение не ниже 90%)'
                        if ok else 'НЕ ПРОШЛО: подтверждены не все столбы, ошибка больше 10 см или совпадение ниже 90%'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
