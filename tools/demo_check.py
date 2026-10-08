#!/usr/bin/env python3
"""Репетиция показа без человека: тот же сценарий, что в docs/demo_script.md, но команды шлёт скрипт.

Запускать при работающем показе (в другом терминале `pixi run demo` или `pixi run demo --fast`):

    pixi run python tools/demo_check.py            # объезд → цель → домой → автономная миссия
    pixi run python tools/demo_check.py --agent scientist

Печатает, что получилось на каждом шаге, и возвращает ненулевой код, если шаг не удался.
"""
import argparse
import json
import sys
import time
import urllib.request

TOUR = [[-1.6, 0.6], [-0.55, 1.6], [0.55, 0.55], [1.6, -0.55], [0.55, -1.6], [-0.55, -0.55]]
GOAL = [1.5, 1.2]


def call(url, body=None, timeout=20.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--skip-mission', action='store_true')
    args = ap.parse_args()
    base = f'http://127.0.0.1:{args.port}/api/pilot'
    failed = []

    def state():
        return call(base + '/state', timeout=5.0)

    def command(**cmd):
        res = call(base + '/command', cmd)
        if not res.get('ok', True):
            print(f"   команда {cmd['cmd']} отклонена: {res.get('error')}")
        return res

    def wait(what, done, limit):
        t0 = time.time()
        s = state()
        while time.time() - t0 < limit:
            s = state()
            if done(s):
                return s, time.time() - t0
            time.sleep(0.5)
        failed.append(what)
        print(f'   НЕ ДОЖДАЛСЯ: {what} за {limit:.0f} с (режим {s.get("mode")}, {s.get("note", {}).get("text")})')
        return s, limit

    def step(title):
        print(f'\n{title}')

    step('0. Жду готовности пульта')
    s, dt = wait('пульт готов', lambda s: s.get('active') and s.get('linked') and s.get('mode') == 'idle'
                 and s.get('settle_left', 0) <= 0, 120)
    print(f"   готов через {dt:.0f} с: {s['backend']}, уровень {s['level']}, сценарий {s['seed']}; "
          f"карта из одной точки {s['map']['coverage']:.0%}")

    step('1. Объезд по шести точкам')
    command(cmd='route', points=TOUR)
    print(f"   путь {state()['path_len']:.1f} м")
    command(cmd='go')
    s, dt = wait('объезд закончен', lambda s: s['mode'] == 'idle' and len(s.get('arrivals', [])) >= 1, 240)
    print(f"   {dt:.0f} с; карта построена на {s['map']['coverage']:.0%}, совпадение с эталоном "
          f"{s['map']['agreement']:.0%}; поправка позы по лидару {s['fix']['shift'] * 100:.0f} см; "
          f"столкновений {s['score']['collisions']}")
    if s['map']['coverage'] < 0.9:
        failed.append('карта построена меньше чем на 90%')

    step('2. Одна точка назначения')
    n = len(s.get('arrivals', []))
    command(cmd='route', points=[GOAL])
    command(cmd='go')
    s, dt = wait('цель достигнута', lambda s: s['mode'] == 'idle' and len(s.get('arrivals', [])) > n, 120)
    px, py = s['pose'][0], s['pose'][1]
    miss = ((px - GOAL[0]) ** 2 + (py - GOAL[1]) ** 2) ** 0.5
    print(f'   {dt:.0f} с; до заданной точки {miss * 100:.0f} см')
    if miss > 0.15:
        failed.append('робот остановился дальше 15 см от цели')

    step('3. Домой')
    command(cmd='home')
    s, dt = wait('робот на базе', lambda s: s['mode'] == 'idle' and s.get('at_base'), 120)
    print(f"   {dt:.0f} с; на базе: {s.get('at_base')}")

    if not args.skip_mission:
        step(f'4. Автономная миссия, агент «{args.agent}»')
        command(cmd='mission', agent=args.agent)
        s, dt = wait('миссия закончена', lambda s: (s.get('mission') or {}).get('done') or
                     (s['score'].get('finished') and s['mode'] != 'mission'), 400)
        sc = s['score']
        print(f"   {dt:.0f} с; собрано {sc['samples_collected']} из {sc['samples_total']}, вернулся: {sc['returned']}, "
              f"счёт {sc['score']}, столкновений {sc['collisions']}, ложных сборов {sc['false_collects']}")
        if not sc['returned']:
            failed.append('в миссии робот не вернулся на базу')

    lag = state().get('lag', {})
    print(f"\nЗапаздывание команд: наибольшее {lag.get('max')} с, компьютер перегружен: {lag.get('slow')}")
    print('\nИтог: ' + ('репетиция прошла' if not failed else 'НЕ ПРОШЛО — ' + '; '.join(failed)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
