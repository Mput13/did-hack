#!/usr/bin/env python3
"""Запись объезда в режиме карты SLAM для анимации на слайде: что видит робот, кадр за кадром.

    ROS_DOMAIN_ID=37 GZ_PARTITION=did_gif pixi run demo --slam-map --no-gui --port 8766      # в другом терминале
    /usr/local/bin/python3 presentation/team/slam_record.py --port 8766                      # → runs/SLAMgif/record.json.gz

Опрашивает состояние пульта (/api/pilot/state) раз в 0,4 с: сетка SLAM Toolbox, поза, скан лидара, путь, след,
состояние объезда. Командует «Построить карту», затем едет к точке (1,5; 1,2) по построенной карте.
"""
import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'runs' / 'SLAMgif' / 'record.json.gz'


def call(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--goal', type=float, nargs=2, default=[1.5, 1.2])
    a = ap.parse_args()
    base = f'http://127.0.0.1:{a.port}/api/pilot'
    frames, last_ver, phase = [], None, 'wait'

    def snap(s, phase):
        nonlocal last_ver
        slam = s.get('slam') or {}
        f = {'t': s['t'], 'phase': phase, 'mode': s['mode'], 'pose': s['pose'], 'scan': s.get('scan'), 'path': s.get('path'),
             'trail': s.get('trail'), 'route': s.get('route'), 'note': (s.get('note') or {}).get('text'),
             'nav': {k: v for k, v in (s.get('nav') or {}).items() if k in ('frontiers', 'free_m2', 'explore')},
             'coverage': slam.get('coverage'), 'agreement': slam.get('agreement'), 'distance': s.get('distance')}
        if slam.get('version') != last_ver:
            last_ver = slam.get('version')
            f['slam'] = {k: slam[k] for k in ('res', 'x0', 'y0', 'w', 'h', 'data')}
        frames.append(f)

    def run(phase, done, limit):
        t0 = time.time()
        while time.time() - t0 < limit:
            s = call(base + '/state')
            snap(s, phase)
            if done(s):
                return s
            time.sleep(0.4)
        raise SystemExit(f'{phase}: не дождался за {limit} с')

    ready = lambda s: (s.get('linked') and s['mode'] == 'idle' and (s.get('nav') or {}).get('ready')
                       and ((s.get('nav') or {}).get('slam') or {}).get('ok'))
    run('wait', ready, 240)
    frames[:] = frames[-3:]                                   # до готовности робот стоит: хватит трёх кадров
    print('объезд…', call(base + '/command', {'cmd': 'explore'}), flush=True)
    time.sleep(1.0)
    s = run('explore', lambda s: s['mode'] == 'idle' and ((s['nav'].get('explore') or {}).get('done') or s['nav'].get('explore') is None), 400)
    print('объезд:', s['nav'].get('explore'), 'пройдено', s.get('distance'), flush=True)
    for _ in range(6):
        snap(call(base + '/state'), 'built'); time.sleep(0.4)
    print('точка…', call(base + '/command', {'cmd': 'route', 'points': [a.goal]}), call(base + '/command', {'cmd': 'go'}), flush=True)
    time.sleep(1.0)
    s = run('goal', lambda s: s['mode'] == 'idle', 200)
    for _ in range(4):
        snap(call(base + '/state'), 'arrived'); time.sleep(0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, 'wt', encoding='utf-8') as f:
        json.dump({'goal': a.goal, 'arrivals': s.get('arrivals'), 'score': s.get('score'), 'frames': frames}, f)
    print(OUT, 'кадров', len(frames), 'столкновений', s['score'].get('collisions'), 'до точки', s.get('arrivals'))


if __name__ == '__main__':
    main()
