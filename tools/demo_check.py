#!/usr/bin/env python3
"""Репетиция показа без человека: тот же сценарий, что в docs/demo_script.md, но команды шлёт скрипт.

Запускать при работающем показе (в другом терминале `pixi run demo` или `pixi run demo --fast`):

    pixi run python tools/demo_check.py            # объезд → цель → домой → автономная миссия
    pixi run python tools/demo_check.py --agent scientist
    pixi run python tools/demo_check.py --agent scientist_v2 --tour short --no-goal --expect hard 2 --min-samples 7 --clean

Печатает, что получилось на каждом шаге, и возвращает ненулевой код, если показ не прошёл: объезд не пройден
целиком, миссия не стартовала на новом прогоне судьи, робот не вернулся, было столкновение, нет записи именно
этого прогона (нельзя проверить нагрузку машины) или не выполнены ожидания --min-samples / --min-score / --clean.
Решение «прошло / не прошло» — функция verdict(), она проверяется в tests/test_demo_check.py.
"""
import argparse
import json
import math
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

TOUR = [[-1.6, 0.6], [-0.55, 1.6], [0.55, 0.55], [1.6, -0.55], [0.55, -1.6], [-0.55, -0.55]]
TOUR_SHORT = [[-1.6, 0.6], [-0.55, 1.6], [0.55, 0.55], [-0.55, -0.55]]      # короткий объезд: левая половина и центр
GOAL = [1.5, 1.2]


GOAL_MISS_M = 0.15             # робот остановился дальше от цели — шаг не удался
LAG_MAX = 0.03                 # доля опоздавших тактов в записи миссии: больше — компьютер был перегружен
COVERAGE_MIN = {'full': 0.9, 'short': 0.6}


def call(url, body=None, timeout=20.0):
    """Ответ сервера интерфейса. Отказ (код 4xx/5xx) — тоже ответ: {'ok': False, 'error': причина}."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            res = json.loads(e.read().decode('utf-8') or 'null')
        except ValueError:
            res = None
        return {'ok': False, 'error': (res or {}).get('error') or f'код {e.code}'}


def tour_done(state, n_points):
    """Объезд пройден целиком: робот стоит, в маршруте пульта все точки объезда, и каждая отмечена пройденной.

    Список arrivals для этого не годится: в нём остаются приезды прежних маршрутов, а остановка посреди
    маршрута («пути к точке нет», срок подъезда, потеря карты) тоже переводит пульт в режим ожидания.
    """
    route = state.get('route') or []
    return state.get('mode') == 'idle' and len(route) == n_points and all(p.get('done') for p in route)


def trace_foreign(row, result, mtime, t_mission):
    """Почему запись — не от этого прогона (None — от этого). Имя файла записи у прогонов одного агента на одном
    сценарии общее, поэтому существование файла ничего не доказывает: сверяем с итогом, который отдал пульт.

    row — строка demo_pick.summarize() по записи, result — итог миссии от пульта, mtime — когда файл записан,
    t_mission — когда скрипт отправил команду миссии (часы одной и той же машины).
    """
    if mtime < t_mission:
        return f'файл записан за {t_mission - mtime:.0f} с до старта миссии'
    for key, name, in_row in (('score', 'счёт', 'score'), ('samples_collected', 'образцов', 'collected')):
        if row.get(in_row) != result.get(key):
            return f'{name} в записи {row.get(in_row)}, а в итоге пульта {result.get(key)}'
    return None


def verdict(facts, min_samples=None, min_score=None, clean=False):
    """Почему репетиция не прошла: список причин (пустой — прошла). Только по собранным фактам, без сети.

    facts — что main() увидел на шагах: 'ready', 'expect' и 'scenario' (уровень, сценарий), 'tour', 'goal',
    'home', 'mission'; шаг, который не выполнялся, — None или отсутствует. Ожидания результата миссии
    (min_samples, min_score, clean) без ключей не проверяются.
    """
    if not facts.get('ready'):
        return ['пульт не готов: показ не запущен или робот не осел на колёса']
    failed = []
    expect, got = facts.get('expect'), facts.get('scenario')
    if expect and (str(got[0]), str(got[1])) != (str(expect[0]), str(expect[1])):
        failed.append(f'показ запущен на другом уровне или сценарии: {got[0]}-{got[1]} вместо {expect[0]}-{expect[1]}')

    tour = facts.get('tour')
    if tour:
        state, n = tour.get('state') or {}, tour['points']
        if tour.get('refused'):
            failed.append(f"объезд не начат: {tour['refused']}")
        elif not tour_done(state, n):
            passed = sum(1 for p in state.get('route') or [] if p.get('done'))
            note = (state.get('note') or {}).get('text')
            failed.append(f"объезд не пройден целиком: пройдено точек {min(passed, n)} из {n}, режим пульта "
                          f"{state.get('mode')}" + (f' («{note}»)' if note else ''))
        else:
            if (state.get('map') or {}).get('coverage', 0.0) < tour['coverage_min']:
                failed.append('карта построена меньше, чем должна после такого объезда')
        if (state.get('score') or {}).get('collisions'):
            failed.append('столкновение на объезде')

    goal = facts.get('goal')
    if goal:
        if not goal.get('arrived'):
            failed.append('робот не доехал до точки назначения')
        elif goal['miss'] > GOAL_MISS_M:
            failed.append('робот остановился дальше 15 см от цели')
    home = facts.get('home')
    if home and not home.get('at_base'):
        failed.append('робот не вернулся на базу по команде «домой»')

    mission = facts.get('mission')
    if mission:
        failed += _mission_verdict(mission, min_samples, min_score, clean)
    return failed


def _mission_verdict(mission, min_samples, min_score, clean):
    state, why = mission.get('state'), mission.get('reason')
    if state == 'refused':
        return [f'миссия не стартовала: судья не начал новый прогон ({why or "причина не сообщена"})']
    if state != 'finished':
        return [f'миссия не прошла до конца: состояние «{state}»' + (f' ({why})' if why else '')]
    failed = []
    if mission.get('fresh_run') is not True:
        # Без нового прогона у агента остаток заряда и времени после объезда: результат не тот, что на показе.
        failed.append('миссия шла не на новом прогоне судьи (пульт не подтвердил сброс: fresh_run ложно)')
    r = mission.get('result') or {}
    if not r:
        return failed + ['у миссии нет итога судьи']
    if not r.get('returned'):
        failed.append('в миссии робот не вернулся на базу')
    if r.get('collisions'):
        failed.append('столкновение в миссии')
    got, total = r.get('samples_collected', 0), r.get('samples_total', '?')
    if min_samples is not None and got < min_samples:
        failed.append(f'собрано {got} из {total}, а нужно не меньше {min_samples}')
    if min_score is not None and r.get('score', -math.inf) < min_score:
        failed.append(f"счёт {r.get('score')} меньше ожидаемого {min_score:g}")
    if clean and (r.get('hazard_hits') or r.get('false_collects')):
        failed.append(f"прогон не чистый: заездов в опасные зоны {r.get('hazard_hits', 0)}, "
                      f"ложных сборов {r.get('false_collects', 0)}")
    trace = mission.get('trace')
    if trace == 'foreign':
        failed.append(f"запись не от этого прогона ({mission.get('trace_why')}): нагрузку машины проверить нельзя "
                      '(сервер интерфейса запущен из другого каталога?)')
    elif trace != 'ok':
        failed.append('запись прогона не найдена: нагрузку машины проверить нельзя (сервер интерфейса запущен '
                      'из другого каталога?)' if trace in (None, 'missing')
                      else f'запись прогона не читается: нагрузку машины проверить нельзя ({trace})')
    elif mission.get('lag', 0.0) > LAG_MAX:
        failed.append(f"опоздавших тактов {mission['lag']:.1%}: компьютер был перегружен, прогон не засчитывается")
    return failed


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
    ap.add_argument('--min-samples', type=int, metavar='N', help='в миссии собрано не меньше N образцов')
    ap.add_argument('--min-score', type=float, metavar='X', help='счёт миссии не меньше X')
    ap.add_argument('--clean', action='store_true',
                    help='в миссии ни одного заезда в опасную зону и ни одного ложного сбора')
    ap.add_argument('--save', metavar='ПАПКА', help='скопировать запись миссии в эту папку (внутри runs/)')
    ap.add_argument('--tag', default='', help='добавка к имени сохранённой записи')
    args = ap.parse_args()
    base = f'http://127.0.0.1:{args.port}/api/pilot'
    facts = {'ready': False, 'expect': args.expect}

    def state():
        try:
            return call(base + '/state', timeout=5.0)
        except OSError:                     # сервер интерфейса ещё поднимается
            return {}

    def command(**cmd):
        """None — команда принята; иначе причина отказа."""
        try:
            res = call(base + '/command', cmd)
        except OSError as e:
            res = {'ok': False, 'error': f'сервер интерфейса не ответил ({e})'}
        if res.get('ok', True):
            return None
        print(f"   команда {cmd['cmd']} отклонена: {res.get('error')}")
        return res.get('error') or 'команда отклонена'

    def wait(what, done, limit, gave_up=None):
        """Ждать, пока done(состояние). gave_up(состояние) — причина, по которой ждать дальше незачем."""
        t0 = time.time()
        s = state()
        while time.time() - t0 < limit:
            s = state()
            if done(s):
                return s, time.time() - t0
            why = gave_up(s) if gave_up else None
            if why:
                print(f'   НЕ ВЫШЛО: {what} — {why}')
                return s, time.time() - t0
            time.sleep(0.5)
        print(f'   НЕ ДОЖДАЛСЯ: {what} за {limit:.0f} с (режим {s.get("mode")}, {(s.get("note") or {}).get("text")})')
        return s, limit

    def mission_of(s):
        return s.get('mission') or {}

    def step(title):
        print(f'\n{title}')

    def finish():
        failed = verdict(facts, args.min_samples, args.min_score, args.clean)
        print('\nИтог: ' + ('репетиция прошла' if not failed else 'НЕ ПРОШЛО — ' + '; '.join(failed)))
        return 1 if failed else 0

    step('0. Жду готовности пульта')
    s, dt = wait('пульт готов', lambda s: s.get('active') and s.get('linked') and s.get('mode') == 'idle'
                 and s.get('settle_left', 0) <= 0, 120)
    if not (s.get('active') and s.get('mode') == 'idle'):
        return finish()
    facts.update(ready=True, scenario=(s['level'], s['seed']))
    print(f"   готов через {dt:.0f} с: {s['backend']}, уровень {s['level']}, сценарий {s['seed']}; "
          f"карта из одной точки {s['map']['coverage']:.0%}")

    if args.expect and (s['level'], str(s['seed'])) != (args.expect[0], args.expect[1]):
        print(f"   НЕ ТОТ СЦЕНАРИЙ: ждали {args.expect[0]}-{args.expect[1]}")
    known = [a['id'] for a in s.get('agents', [])]
    if not args.skip_mission and args.agent not in known:
        print(f"   агента «{args.agent}» нет в списке пульта: {', '.join(known)}")
        return 1

    tour = {'full': TOUR, 'short': TOUR_SHORT, 'none': None}[args.tour]
    if tour:
        step(f'1. Объезд по {len(tour)} точкам')
        facts['tour'] = {'points': len(tour), 'coverage_min': COVERAGE_MIN[args.tour]}
        refused = command(cmd='route', points=tour) or command(cmd='go')
        if refused:
            facts['tour'].update(refused=refused, state=state())
        else:
            length = 0.0                    # путь пульт показывает только с началом езды
            for _ in range(10):
                length = state().get('path_len', 0.0)
                if length:
                    break
                time.sleep(0.2)
            print(f"   путь {length:.1f} м")
            idle = [0]                      # сколько опросов подряд пульт стоит, а маршрут не пройден

            def stalled(s):
                idle[0] = idle[0] + 1 if s.get('mode') == 'idle' else 0
                return 'робот остановился, не пройдя маршрут' if idle[0] >= 6 else None

            s, dt = wait('объезд закончен', lambda s: tour_done(s, len(tour)), 240, stalled)
            facts['tour']['state'] = s
            if s.get('map') and s.get('fix') and s.get('score'):
                print(f"   {dt:.0f} с; пройдено точек {sum(1 for p in s.get('route', []) if p.get('done'))} из {len(tour)}; "
                      f"карта построена на {s['map']['coverage']:.0%}, совпадение с эталоном "
                      f"{s['map']['agreement']:.0%}; поправка позы по лидару {s['fix']['shift'] * 100:.0f} см; "
                      f"столкновений {s['score'].get('collisions')}")

    if not args.no_goal:
        step('2. Одна точка назначения')
        n = len(s.get('arrivals', []))
        refused = command(cmd='route', points=[GOAL]) or command(cmd='go')
        facts['goal'] = {'arrived': False, 'miss': None}
        if not refused:
            # В списке arrivals пульт отдаёт последние десять приездов: кроме его длины смотрим, что цель пройдена.
            s, dt = wait('цель достигнута', lambda s: tour_done(s, 1) and (
                len(s.get('arrivals', [])) > n or n >= 10), 120)
            if tour_done(s, 1):
                px, py = s['pose'][0], s['pose'][1]
                miss = ((px - GOAL[0]) ** 2 + (py - GOAL[1]) ** 2) ** 0.5
                facts['goal'] = {'arrived': True, 'miss': miss}
                print(f'   {dt:.0f} с; до заданной точки {miss * 100:.0f} см')

        step('3. Домой')
        command(cmd='home')
        s, dt = wait('робот на базе', lambda s: s.get('mode') == 'idle' and s.get('at_base'), 120)
        facts['home'] = {'at_base': bool(s.get('mode') == 'idle' and s.get('at_base'))}
        print(f"   {dt:.0f} с; на базе: {s.get('at_base')}")

    if not args.skip_mission:
        step(f'4. Автономная миссия, агент «{args.agent}»')
        t_cmd = time.time()
        refused = command(cmd='mission', agent=args.agent)
        m = mission_of(state())
        if refused:
            # Отказ сразу: либо судья не начал новый прогон (пульт пишет это в mission), либо команда не принята.
            facts['mission'] = {'state': 'refused', 'reason': m.get('reason') or refused} if m.get('state') == 'refused' \
                else {'state': 'не начата', 'reason': refused}
            return finish()
        s, _ = wait('миссия началась', lambda s: s.get('mode') == 'mission' and mission_of(s).get('state') == 'running',
                    150, lambda s: mission_of(s).get('reason') if mission_of(s).get('state') == 'refused' else None)
        m = mission_of(s)
        if m.get('state') != 'running':
            facts['mission'] = {'state': m.get('state') or 'не начата', 'reason': m.get('reason')}
            return finish()
        t_start = time.time()
        if t_start - t_cmd > 2.0:
            print(f'   до базы и нового прогона судьи перед миссией {t_start - t_cmd:.0f} с')
        s, dt = wait('миссия закончена', lambda s: s.get('mode') != 'mission' and mission_of(s).get('state') == 'finished',
                     400, lambda s: 'миссия прервана' if s.get('mode') not in (None, 'mission')
                     and mission_of(s).get('state') != 'finished' else None)
        m = mission_of(s)
        fact = facts['mission'] = {'state': m.get('state'), 'fresh_run': m.get('fresh_run'), 'reason': m.get('reason'),
                                   'result': m.get('result') or (s.get('score') if m.get('state') == 'finished' else None),
                                   'trace': 'missing', 'lag': None}
        sc = fact['result']
        if sc:
            print(f"   {dt:.0f} с по часам ({sc.get('time', sc.get('t'))} с симуляции); собрано {sc.get('samples_collected')} из "
                  f"{sc.get('samples_total')}, вернулся: {sc.get('returned')}, счёт {sc.get('score')}, столкновений "
                  f"{sc.get('collisions')}, заездов в опасные зоны {sc.get('hazard_hits')}, ложных сборов "
                  f"{sc.get('false_collects')}; новый прогон судьи: {'да' if m.get('fresh_run') is True else 'НЕТ'}")
        trace = ROOT / 'runs' / m['trace_file'] if m.get('trace_file') else None
        if trace is not None and trace.exists():
            from demo_pick import line, load, summarize
            try:
                row = summarize(load(trace))
            except Exception as e:              # noqa: BLE001 — нечитаемая запись — причина отказа, а не падение
                fact['trace'] = f'{type(e).__name__}: {e}'
                print(f"   запись миссии не читается: {fact['trace']}")
            else:
                foreign = trace_foreign(row, sc or {}, trace.stat().st_mtime, t_cmd)
                if foreign:
                    fact.update(trace='foreign', trace_why=foreign)
                    print(f'   ЗАПИСЬ НЕ ОТ ЭТОГО ПРОГОНА: {foreign} ({trace})')
                else:
                    fact.update(trace='ok', lag=row['lag'])
                    print('   ' + line(row).replace('\n', '\n   '))
                    if args.save:
                        out = ROOT / args.save / f"{args.agent}-{s['level']}-{s['seed']}{args.tag}.json.gz"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(trace, out)
                        print(f'   запись миссии: {out.relative_to(ROOT)}')
        else:
            print('   ЗАПИСЬ МИССИИ НЕ НАЙДЕНА (сервер интерфейса запущен из другого каталога?)')

    lag = state().get('lag', {})
    print(f"\nЗапаздывание команд: наибольшее {lag.get('max')} с, компьютер перегружен: {lag.get('slow')}")
    return finish()


if __name__ == '__main__':
    sys.exit(main())
