#!/usr/bin/env python3
"""Что увидит зритель в прогоне: разбор записи миссии и подбор сценария для показа адаптации.

    ./px python tools/demo_pick.py runs/F1/scientist_v2/hard-1.json.gz …      # разобрать готовые записи (Gazebo или быстрый симулятор)
    ./px python tools/demo_pick.py --scan 1-60 --agents scientist_v2 adaptive_v2   # подбор в быстром симуляторе
    ./px python tools/demo_pick.py --story runs/F1/scientist_v2/hard-1.json.gz    # журнал реакции на изменения по секундам

По каждой записи: итог судьи, наибольший простой, отъезды защиты, доля опоздавших тактов и — главное — что стало с
каждым из трёх скрытых изменений среды (смена грунта, новая опасная зона, сбой датчика): через сколько секунд агент
его заметил и что сделал. Подбор ничего не меняет в сценариях: это обычные сценарии генератора с номером.
"""
import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LAG_GAP_S = 0.45            # такт записи пришёл позже — компьютер был перегружен (та же мерка, что в gazebo_batch.py)
STILL_M = 0.02              # робот «стоит», пока не отъехал от точки дальше


def load(path):
    with gzip.open(path) as f:
        return json.load(f)


def longest_stop(track):
    """Наибольшее время, которое робот простоял на одном месте: (секунды, момент начала)."""
    t, x, y = track['t'], track['x'], track['y']
    best, start, i = 0.0, 0.0, 0
    for j in range(len(t)):
        while math.hypot(x[j] - x[i], y[j] - y[i]) > STILL_M:
            i += 1
        if t[j] - t[i] > best:
            best, start = t[j] - t[i], t[i]
    return round(best, 1), round(start, 1)


def lag_share(track):
    gaps = [b - a for a, b in zip(track['t'], track['t'][1:])]
    return sum(1 for g in gaps if g > LAG_GAP_S) / max(1, len(gaps))


def inside(zone, x, y, pad=0.0):
    if zone.get('shape') == 'rect':
        return abs(x - zone['x']) <= zone['w'] / 2 + pad and abs(y - zone['y']) <= zone['h'] / 2 + pad
    return math.hypot(x - zone['x'], y - zone['y']) <= zone['r'] + pad


def mult(soils, x, y):
    return max([1.0] + [z['mult'] for z in soils if inside(z, x, y)])


def changed_near(before, after, x, y, radius=0.3):
    """Изменился ли множитель грунта в точке или не дальше radius от неё."""
    pts = [(x, y)] + [(x + radius * k / 3 * math.cos(a), y + radius * k / 3 * math.sin(a))
                      for k in (1, 2, 3) for a in [i * math.pi / 6 for i in range(12)]] if radius else [(x, y)]
    return any(abs(mult(before, px, py) - mult(after, px, py)) > 1e-6 for px, py in pts)


def pose_at(track, t):
    i = min(range(len(track['t'])), key=lambda k: abs(track['t'][k] - t))
    return track['x'][i], track['y'][i]


def summarize(tr):
    """Одна строка таблицы и разбор трёх изменений среды."""
    r, journal, track = tr['result'], tr['journal'], tr['track']
    events = {e['type']: e for e in tr['scenario'].get('events', [])}
    texts = [(j['t'], j['kind'], j['text'], (j.get('data') or {}).get('tag')) for j in journal]
    stop_s, stop_t = longest_stop(track)
    row = {
        'agent': tr['agent']['name'], 'level': tr['scenario'].get('level', r.get('level')), 'seed': tr['scenario'].get('seed'),
        'backend': tr.get('backend'), 'score': r.get('score'), 'collected': r.get('samples_collected'),
        'total': r.get('samples_total'), 'returned': bool(r.get('returned')), 'collisions': r.get('collisions'),
        'hazard_hits': r.get('hazard_hits'), 'false_collects': r.get('false_collects'), 'time': r.get('time'),
        'reason': r.get('reason'), 'battery_left': r.get('battery_left'),
        'stop_s': stop_s, 'stop_t': stop_t, 'lag': round(lag_share(track), 4),
        'stuck': sum(1 for _, _, s, _ in texts if s.startswith('Робот не движется')),
        'blocked': sum(1 for _, _, s, _ in texts if s.startswith('Преграда вплотную')),
        'inquiries': len({(j.get('data') or {}).get('inquiry') for j in journal if j['kind'] == 'inquiry'} - {None}),
        'hyp': (r.get('hypotheses') or {}).get('total'), 'hyp_confirmed': (r.get('hypotheses') or {}).get('confirmed'),
        'hyp_refuted': (r.get('hypotheses') or {}).get('refuted'),
    }
    # --- сбой датчика: заметил ли, ждал ли, сколько
    f = events.get('sensor_fault')
    fault = None
    if f:
        seen = [t for t, k, s, g in texts if t >= f['t'] and g in ('sensor_degraded', 'sensor_fault')]
        wait0 = [t for t, k, s, g in texts if t >= f['t'] and g == 'sensor_wait']
        wait1 = [(t, s) for t, k, s, g in texts if g == 'sensor_wait_end']
        m = re.search(r'Ожидание датчика: (\d+) с', wait1[0][1]) if wait1 else None
        fault = {'t': f['t'], 'kind': f.get('kind'), 'duration': f.get('duration'),
                 'noticed_after': round(seen[0] - f['t'], 1) if seen else None,
                 'wait_from': wait0[0] if wait0 else None, 'wait_s': int(m.group(1)) if m else None,
                 'in_run': f['t'] < (r.get('time') or 0)}
    # --- новая опасная зона: въехал ли, выдвинул ли гипотезу о зоне
    h = events.get('new_hazard')
    hazard = None
    if h:
        z = h['zone']
        hits = [e for e in tr.get('events', []) if e['type'] == 'hazard_hit' and e['t'] >= h['t'] and inside(z, e['x'], e['y'], 0.12)]
        hyp = [t for t, k, s, g in texts if hits and t >= hits[0]['t'] and k == 'hypothesis' and 'опасная зона' in s]
        near = [tt for tt, xx, yy in zip(track['t'], track['x'], track['y']) if tt >= h['t'] and inside(z, xx, yy, 0.35)]
        hazard = {'t': h['t'], 'x': z['x'], 'y': z['y'], 'r': z['r'], 'hits': len(hits),
                  'hit_t': hits[0]['t'] if hits else None, 'hypothesis_t': hyp[0] if hyp else None,
                  'passed_near_t': near[0] if near else None, 'in_run': h['t'] < (r.get('time') or 0)}
    # --- смена грунта: тревога о расходе после события и именно там, где множитель изменился
    s_ev = events.get('soil_change')
    soil = None
    if s_ev:
        before, after = tr['scenario'].get('soils', []), s_ev.get('soils', [])
        # Тревога о расходе считается с момента, когда вопрос возник: вывод по вопросу, начатому до смены
        # грунта, — это найденный дорогой участок, а не замеченное изменение.
        asked = {}
        for j in journal:
            q = (j.get('data') or {}).get('inquiry')
            if q:
                asked.setdefault(q, j['t'])
        alarms = sorted((asked.get((j.get('data') or {}).get('inquiry'), j['t']), j['text']) for j in journal
                        if (j.get('data') or {}).get('tag') == 'model_mismatch')
        alarms = [(t, s) for t, s in alarms if t >= s_ev['t']]
        on_changed = [(t, s) for t, s in alarms if changed_near(before, after, *pose_at(track, t))]
        on_path = [tt for tt, xx, yy in zip(track['t'], track['x'], track['y'])
                   if tt >= s_ev['t'] and changed_near(before, after, xx, yy, 0.0)]
        soil = {'t': s_ev['t'], 'noticed_after': round(on_changed[0][0] - s_ev['t'], 1) if on_changed else None,
                'entered_t': on_path[0] if on_path else None,
                'reaction_after_entry': round(on_changed[0][0] - on_path[0], 1) if on_changed and on_path else None,
                'any_alarm_after': round(alarms[0][0] - s_ev['t'], 1) if alarms else None,
                'in_run': s_ev['t'] < (r.get('time') or 0)}
    row.update(fault=fault, hazard=hazard, soil=soil)
    # Все штрафы за опасные зоны (и старые, и новую) и что агент после них записал.
    row['hazard_log'] = [(t, s) for t, k, s, g in texts if g == 'hazard' or (k == 'hypothesis' and 'опасная зона' in s)]
    return row


def line(row):
    f, h, s = row['fault'] or {}, row['hazard'] or {}, row['soil'] or {}
    fault = '—' if not f else ('после конца' if not f['in_run'] else
                               f"t={f['t']:.0f}: " + ('не заметил' if f['noticed_after'] is None else
                                                      f"заметил +{f['noticed_after']:.0f} с"
                                                      + (f", ждал {f['wait_s']} с" if f['wait_s'] else ', не ждал')))
    hazard = '—' if not h else ('после конца' if not h['in_run'] else
                                f"t={h['t']:.0f}: " + (f"въезд t={h['hit_t']:.0f}" + (', гипотеза' if h['hypothesis_t'] else '')
                                                       if h['hits'] else ('рядом t=%.0f' % h['passed_near_t'] if h['passed_near_t'] else 'не встретил')))
    soil = '—' if not s else ('после конца' if not s['in_run'] else
                              f"t={s['t']:.0f}: " + ('на изменившийся пол не въезжал' if s['entered_t'] is None else
                                                     f"въехал t={s['entered_t']:.0f}, " + ('тревоги нет' if s['noticed_after'] is None
                                                                                          else f"тревога через {s['reaction_after_entry']:.0f} с")))
    return (f"{row['agent']:<13} {row['level']}-{row['seed']:<3} счёт {row['score']:5.1f}  {row['collected']}/{row['total']}  "
            f"{'вернулся' if row['returned'] else 'НЕ ВЕРНУЛСЯ'}  столкн. {row['collisions']}  зон {row['hazard_hits']}  "
            f"{row['time']:5.1f} с  простой {row['stop_s']:.0f} с (с {row['stop_t']:.0f})  отъезды {row['stuck']}+{row['blocked']}  "
            f"опозд. {row['lag']:.1%}\n      датчик: {fault} | зона: {hazard} | грунт: {soil} | вопросов {row['inquiries']}, "
            f"гипотез {row['hyp']} (подтв. {row['hyp_confirmed']}, опроверг. {row['hyp_refuted']})")


def showy(row):
    """Годится ли прогон для показа адаптации: вернулся, ничего не задел, и на экране есть реакция на изменение."""
    f, h, s = row['fault'] or {}, row['hazard'] or {}, row['soil'] or {}
    clean = row['returned'] and row['collisions'] == 0 and row['hazard_hits'] == 0 and row['false_collects'] == 0 and row['stuck'] + row['blocked'] == 0
    reacted = sum([bool(f.get('wait_s')), bool(h.get('hypothesis_t')), s.get('noticed_after') is not None])
    return clean, reacted


def story(tr, t0=0.0, t1=1e9):
    world = {round(e['t'], 1): e for e in tr.get('world', [])}
    out = [(e['t'], 'СРЕДА', {'soil_change': 'скрыто: грунты сменились', 'new_hazard': 'скрыто: появилась опасная зона',
                              'sensor_fault': f"скрыто: сбой датчика ({e.get('kind')})",
                              'sensor_recovered': 'скрыто: датчик восстановился'}.get(e['type'], e['type'])) for e in world.values()]
    out += [(e['t'], 'СУДЬЯ', {'hazard_hit': 'штраф: опасная зона', 'collision': 'штраф: столкновение',
                               'false_collect': 'штраф: ложный сбор'}[e['type']] + f" в ({e['x']:.2f}; {e['y']:.2f})")
            for e in tr.get('events', []) if e['type'] in ('hazard_hit', 'collision', 'false_collect')]
    out += [(j['t'], j['kind'], j['text']) for j in tr['journal']]
    for t, k, s in sorted(out, key=lambda o: o[0]):
        if t0 <= t <= t1:
            print(f'{t:6.1f}  {k:<10} {s}')


def seeds(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += range(int(a), int(b or a) + 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='записи прогонов (.json.gz)')
    ap.add_argument('--scan', metavar='НОМЕРА', help='прогнать в быстром симуляторе сценарии, например 1-60 или 1,3,5')
    ap.add_argument('--level', default='hard')
    ap.add_argument('--agents', nargs='+', default=['scientist_v2'])
    ap.add_argument('--story', action='store_true', help='напечатать журнал записи по секундам вместе со скрытыми событиями')
    ap.add_argument('--between', type=float, nargs=2, default=(0.0, 1e9), metavar=('С', 'ПО'))
    ap.add_argument('--json', help='куда сохранить строки таблицы')
    ap.add_argument('--only-showy', action='store_true', help='печатать только прогоны, годные для показа адаптации')
    args = ap.parse_args()

    rows = []
    for path in args.files:
        tr = load(path)
        if args.story:
            print(f'== {path}')
            story(tr, *args.between)
            continue
        rows.append(summarize(tr))
        rows[-1]['file'] = str(path)
    if args.scan:
        from did.runner import RUNS, run_episode
        for agent in args.agents:
            for seed in seeds(args.scan):
                s = run_episode(args.level, seed, agent, experiment='F1scan')
                rows.append(summarize(load(RUNS / s['file'])))
                rows[-1]['file'] = 'runs/' + s['file']
    for row in rows:
        clean, reacted = showy(row)
        row['clean'], row['reacted'] = clean, reacted
        if not args.only_showy or (clean and reacted >= 2):
            print(line(row) + f"\n      для показа: {'чисто' if clean else 'не чисто'}, реакций на изменения {reacted} из 3")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
