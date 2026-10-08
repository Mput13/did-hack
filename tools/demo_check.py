#!/usr/bin/env python3
"""Репетиция показа без человека: тот же сценарий, что в docs/demo_script.md, но команды шлёт скрипт.

Запускать при работающем показе (в другом терминале `pixi run demo` или `pixi run demo --fast`):

    pixi run python tools/demo_check.py            # объезд → цель → домой → автономная миссия
    pixi run python tools/demo_check.py --agent scientist
    pixi run python tools/demo_check.py --agent scientist_v2 --tour short --no-goal --expect hard 9 --save runs/F1demo

Печатает, что получилось на каждом шаге, и возвращает ненулевой код, если шаг не удался.
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

TOUR = [[-1.6, 0.6], [-0.55, 1.6], [0.55, 0.55], [1.6, -0.55], [0.55, -1.6], [-0.55, -0.55]]
TOUR_SHORT = [[-1.6, 0.6], [-0.55, 1.6], [0.55, 0.55], [-0.55, -0.55]]      # короткий объезд: левая половина и центр
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
    ap.add_argument('--tour', choices=['full', 'short', 'none'], default='full',
                    help='объезд перед миссией: шесть точек, четыре точки или без объезда')
    ap.add_argument('--no-goal', action='store_true', help='пропустить шаги «одна точка назначения» и «домой»')
    ap.add_argument('--expect', nargs=2, metavar=('УРОВЕНЬ', 'СЦЕНАРИЙ'),
                    help='проверить, что показ запущен именно на этом уровне и сценарии')
    ap.add_argument('--save', metavar='ПАПКА', help='скопировать запись миссии в эту папку (внутри runs/)')
    ap.add_argument('--tag', default='', help='добавка к имени сохранённой записи')
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

    if args.expect and (s['level'], str(s['seed'])) != (args.expect[0], args.expect[1]):
        print(f"   НЕ ТОТ СЦЕНАРИЙ: ждали {args.expect[0]}-{args.expect[1]}")
        failed.append('показ запущен на другом уровне или сценарии')
    known = [a['id'] for a in s.get('agents', [])]
    if not args.skip_mission and args.agent not in known:
        print(f"   агента «{args.agent}» нет в списке пульта: {', '.join(known)}")
        return 1

    tour = {'full': TOUR, 'short': TOUR_SHORT, 'none': None}[args.tour]
    if tour:
        step(f'1. Объезд по {len(tour)} точкам')
        command(cmd='route', points=tour)
        print(f"   путь {state()['path_len']:.1f} м")
        command(cmd='go')
        s, dt = wait('объезд закончен', lambda s: s['mode'] == 'idle' and len(s.get('arrivals', [])) >= 1, 240)
        print(f"   {dt:.0f} с; карта построена на {s['map']['coverage']:.0%}, совпадение с эталоном "
              f"{s['map']['agreement']:.0%}; поправка позы по лидару {s['fix']['shift'] * 100:.0f} см; "
              f"столкновений {s['score']['collisions']}")
        if s['map']['coverage'] < (0.9 if args.tour == 'full' else 0.6):
            failed.append('карта построена меньше, чем должна после такого объезда')
        if s['score']['collisions']:
            failed.append('столкновение на объезде')

    if not args.no_goal:
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
        t_cmd = time.time()
        command(cmd='mission', agent=args.agent)
        s, _ = wait('миссия началась', lambda s: s['mode'] == 'mission' and (s.get('mission') or {}).get('state') == 'running', 120)
        t_start = time.time()
        if t_start - t_cmd > 2.0:
            print(f'   до базы перед миссией {t_start - t_cmd:.0f} с')
        s, dt = wait('миссия закончена', lambda s: s['mode'] != 'mission' and (s.get('mission') or {}).get('state') == 'finished', 400)
        m = s.get('mission') or {}
        sc = m.get('result') or s['score']
        print(f"   {dt:.0f} с по часам ({sc.get('time', sc.get('t'))} с симуляции); собрано {sc['samples_collected']} из "
              f"{sc['samples_total']}, вернулся: {sc['returned']}, счёт {sc['score']}, столкновений {sc['collisions']}, "
              f"заездов в опасные зоны {sc['hazard_hits']}, ложных сборов {sc['false_collects']}")
        if not sc['returned']:
            failed.append('в миссии робот не вернулся на базу')
        if sc['collisions']:
            failed.append('столкновение в миссии')
        trace = ROOT / 'runs' / m['trace_file'] if m.get('trace_file') else None
        if trace is not None and trace.exists():
            from demo_pick import line, load, summarize
            row = summarize(load(trace))
            print('   ' + line(row).replace('\n', '\n   '))
            if row['lag'] > 0.03:
                failed.append(f"опоздавших тактов {row['lag']:.1%}: компьютер был перегружен, прогон не засчитывается")
            if args.save:
                out = ROOT / args.save / f"{args.agent}-{s['level']}-{s['seed']}{args.tag}.json.gz"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(trace, out)
                print(f'   запись миссии: {out.relative_to(ROOT)}')
        else:
            print('   запись миссии не найдена (сервер интерфейса запущен из другого каталога?)')

    lag = state().get('lag', {})
    print(f"\nЗапаздывание команд: наибольшее {lag.get('max')} с, компьютер перегружен: {lag.get('slow')}")
    print('\nИтог: ' + ('репетиция прошла' if not failed else 'НЕ ПРОШЛО — ' + '; '.join(failed)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
