#!/usr/bin/env python3
"""Журнал поз в Gazebo: что думает одометрия и где робот на самом деле.

    pixi run gazebo-run --level medium --seed 1 --exp gz_loc --pose-log          # прогон сразу с журналом
    pixi run python tools/gz_pose_log.py --out runs/gz_loc/pose-medium-1.csv     # или отдельно, пока идёт прогон
    pixi run python tools/gz_pose_log.py --report runs/gz_loc/pose-medium-1.csv  # разбор готового журнала
    ... --report pose.csv --trace runs/gz_loc/adaptive/medium-1.json.gz           # и ошибка позы агента

Узел только слушает: /odom, истинную позу (/did/gz/dynamic_pose, служебный топик судьи), /cmd_vel,
/did/events. Запускается до агента или вместе с ним, сам завершается, когда судья закончил прогон.
Строка журнала — раз в 0,1 с времени симуляции; истинная поза приводится ко времени одометрии.
С ключом --scans рядом пишется .npz со сканами лидара и позами в момент скана (для отладки
локализации без Gazebo).
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COLUMNS = ('t', 'tj', 'ox', 'oy', 'oth', 'ov', 'ow', 'gx', 'gy', 'gth', 'gpitch', 'cv', 'cw', 'tgap', 'events')
# tgap — промежуток между соседними замерами истинной позы вокруг строки, с: большой — строке не верить
GZ_POSES = '/did/gz/dynamic_pose'


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# =============================================================================================
# запись
# =============================================================================================

def record(out, scans=None, max_s=900.0, period=0.1):
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    from did.config import BASE

    class PoseLog(Node):

        def __init__(self):
            super().__init__('did_pose_log')
            self.clock = None
            self.clock_wall = time.monotonic()
            self.offset = None              # время судьи минус время симуляции
            self.finished = False
            self.cmd = (0.0, 0.0)
            self.events = []
            self.truth = deque(maxlen=400)  # (время симуляции, x, y, курс, тангаж)
            self.odom = deque(maxlen=400)   # (время, x, y, курс)
            self.pending = deque()          # строки, ждущие истинную позу на своё время
            self.scan_wait = deque()
            self.scan_rows = []
            self.scan_step = None
            self.next_row = None
            self.rows = 0
            self._idx = self._n = None
            self.file = open(out, 'w', newline='')
            self.csv = csv.writer(self.file)
            self.csv.writerow(COLUMNS)
            fast = qos_profile_sensor_data
            # Очередь в одно сообщение: если узел не успевает, старая истинная поза не должна получить
            # свежее время (метка времени у неё — по последнему /clock).
            last = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Clock, '/clock', self.on_clock, last)
            self.create_subscription(Odometry, '/odom', self.on_odom, fast)
            self.create_subscription(TFMessage, GZ_POSES, self.on_truth, last)
            self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, 10)
            self.create_subscription(String, '/did/events', self.on_event, 50)
            self.create_subscription(String, '/did/score', self.on_score, 10)
            if scans:
                self.create_subscription(LaserScan, '/scan', self.on_scan, fast)

        def on_clock(self, msg):
            self.clock = msg.clock.sec + msg.clock.nanosec * 1e-9
            self.clock_wall = time.monotonic()

        def on_cmd(self, msg):
            self.cmd = (msg.twist.linear.x, msg.twist.angular.z)

        def on_event(self, msg):
            self.events.append(json.loads(msg.data).get('type', '?'))

        def on_score(self, msg):
            score = json.loads(msg.data)
            if score.get('finished'):
                self.finished = True
            elif self.clock is not None:
                self.offset = score['t'] - self.clock

        def on_truth(self, msg):
            # В сообщении моста нет ни имён, ни времени. Робот — запись с наибольшим |x|+|y| (звенья
            # заданы относительно модели, около нуля); время — последнее пришедшее /clock.
            n = len(msg.transforms)
            if n == 0 or self.clock is None:
                return
            if self._idx is None or n != self._n:
                size = [abs(tr.transform.translation.x) + abs(tr.transform.translation.y) for tr in msg.transforms]
                self._idx, self._n = int(np.argmax(size)), n
            tr = msg.transforms[self._idx].transform
            q = tr.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
            self.truth.append((self.clock, tr.translation.x, tr.translation.y, yaw, pitch))
            self.flush()

        def on_odom(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            x, y = BASE[0] + p.x, BASE[1] + p.y
            self.odom.append((t, x, y, yaw))
            if self.next_row is None or t + 1e-6 >= self.next_row:
                self.next_row = (t if self.next_row is None or t - self.next_row > period else self.next_row) + period
                events, self.events = self.events, []
                tj = t + self.offset if self.offset is not None else float('nan')
                self.pending.append([t, tj, x, y, yaw, tw.linear.x, tw.angular.z, *self.cmd, '+'.join(events)])

        def on_scan(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.scan_step = msg.angle_increment
            self.scan_wait.append((t, np.asarray(msg.ranges, dtype=np.float32)))

        def flush(self):
            last = self.truth[-1][0]
            while self.pending and self.pending[0][0] <= last:
                t, tj, x, y, yaw, v, w, cv, cw, events = self.pending.popleft()
                g = _at(self.truth, t)
                self.csv.writerow([f'{t:.3f}', f'{tj:.3f}', f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}', f'{v:.3f}', f'{w:.3f}',
                                   f'{g[0]:.4f}', f'{g[1]:.4f}', f'{g[2]:.4f}', f'{g[3]:.4f}', f'{cv:.3f}', f'{cw:.3f}',
                                   f'{_gap(self.truth, t):.3f}', events])
                self.rows += 1
                if self.rows % 50 == 0:
                    self.file.flush()
            while self.scan_wait and self.scan_wait[0][0] <= min(last, self.odom[-1][0] if self.odom else -1.0):
                t, r = self.scan_wait.popleft()
                self.scan_rows.append((t, _at(self.odom, t)[:3], _at(self.truth, t)[:3], r))

        def close(self):
            self.file.close()
            if scans and self.scan_rows:
                np.savez_compressed(
                    scans, t=np.array([s[0] for s in self.scan_rows]),
                    odom=np.array([s[1] for s in self.scan_rows]), truth=np.array([s[2] for s in self.scan_rows]),
                    ranges=np.stack([s[3] for s in self.scan_rows]), step=self.scan_step)

    rclpy.init()
    node = PoseLog()
    wall0 = time.monotonic()
    done_at = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if node.finished and done_at is None:
                done_at = now
            if done_at is not None and now - done_at > 1.0:
                break                                   # судья закончил прогон
            if node.rows and now - node.clock_wall > 10.0:
                break                                   # симуляция остановлена
            if now - wall0 > max_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        print(f'журнал поз: {node.rows} строк в {out}' + (f', сканов {len(node.scan_rows)}' if scans else ''))
        node.destroy_node()
        rclpy.try_shutdown()


def _gap(buf, t):
    """Промежуток между соседними записями буфера вокруг момента t."""
    newer = None
    for item in reversed(buf):
        if item[0] <= t:
            return (newer[0] - item[0]) if newer else t - item[0]
        newer = item
    return 0.0


def _at(buf, t):
    """Значение из буфера [(время, x, y, курс, ...)] на момент t: линейно между соседними записями."""
    prev = None
    for item in reversed(buf):
        if item[0] <= t:
            if prev is None or prev[0] - item[0] < 1e-9:
                return item[1:]
            k = (t - item[0]) / (prev[0] - item[0])
            out = [a + k * (b - a) for a, b in zip(item[1:], prev[1:])]
            out[2] = _wrap(item[3] + k * _wrap(prev[3] - item[3]))
            return tuple(out)
        prev = item
    return (prev or buf[-1])[1:]


# =============================================================================================
# разбор
# =============================================================================================

def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    data = {k: np.array([float(r.get(k) or 0.0) for r in rows]) for k in COLUMNS if k != 'events'}
    data['events'] = [r['events'] for r in rows]
    known = np.isfinite(data['tj'])          # первые строки пишутся раньше, чем узнаём время судьи
    if known.any():
        data['tj'] = data['t'] + np.median((data['tj'] - data['t'])[known])
    return data


def increments(d):
    """Приращения за шаг в системе робота: по одометрии и по истине (вперёд, вбок, курс)."""
    out = {}
    for key, (x, y, th) in {'o': (d['ox'], d['oy'], d['oth']), 'g': (d['gx'], d['gy'], d['gth'])}.items():
        dx, dy = np.diff(x), np.diff(y)
        c, s = np.cos(th[:-1]), np.sin(th[:-1])
        out[key] = (c * dx + s * dy, -s * dx + c * dy, _wrap(np.diff(th)))
    return out


def agent_pose(d, trace):
    """Поза агента на времена журнала: поправка из записи прогона (pose_fix), применённая к одометрии.

    Поправка меняется медленно, поэтому её можно брать по времени без точной привязки часов; сама
    одометрия в журнале уже сведена с истиной по времени симуляции.
    """
    fix = trace.get('pose_fix') or []
    if not fix:
        return d['ox'], d['oy'], d['oth']
    tr = trace['track']
    at = {t: i for i, t in enumerate(tr['t'])}
    ts, tf = [], []
    for f in fix:
        i = at.get(f['t'])
        if i is None:
            continue
        # Преобразование «одометрия → карта»: поворот на dth вокруг начала координат и сдвиг.
        xo, yo = tr['x'][i] - f['dx'], tr['y'][i] - f['dy']
        c, s = math.cos(f['dth']), math.sin(f['dth'])
        ts.append(f['t'])
        tf.append((tr['x'][i] - (c * xo - s * yo), tr['y'][i] - (s * xo + c * yo), f['dth']))
    tf = np.array(tf)
    tx, ty, dth = (np.interp(d['tj'], ts, tf[:, k]) for k in range(3))
    c, s = np.cos(dth), np.sin(dth)
    return c * d['ox'] - s * d['oy'] + tx, s * d['ox'] + c * d['oy'] + ty, _wrap(d['oth'] + dth)


def report(path, trace=None, out=sys.stdout):
    d = load(path)
    t = d['tj'] if np.isfinite(d['tj']).all() else d['t'] - d['t'][0]
    run = d['cv'] * 0.0 == 0.0
    if trace:                               # ошибку позы агента считаем, пока он ведёт робота
        from did.recorder import load_trace
        tr = load_trace(trace)
        run = (t >= tr['track']['t'][0]) & (t <= tr['track']['t'][-1])
    n = len(t)
    p = lambda *a: print(*a, file=out)   # noqa: E731
    err = np.hypot(d['ox'] - d['gx'], d['oy'] - d['gy'])
    eth = _wrap(d['oth'] - d['gth'])
    p(f'журнал {path}: {n} строк, {t[0]:.1f}–{t[-1]:.1f} с')
    # Если машина занята и узел не успевал, истинная поза получала чужое время: строки рядом с
    # пропуском в записи для максимумов не годятся.
    gap = (np.diff(d['t'], prepend=d['t'][0]) > 0.25) | (d['tgap'] > 0.08)
    shaky = np.convolve(gap, np.ones(21), 'same') > 0
    if shaky.any():
        p(f'запись прерывалась {int(gap.sum())} раз: {int(shaky.sum())} строк рядом с пропусками в итоги по позе не идут')
        run = run & ~shaky
    p(f'одометрия против истины: положение — медиана {np.median(err) * 100:.1f} см, максимум {err.max() * 100:.1f} см; '
      f'курс — медиана {np.median(np.abs(eth)):.3f} рад, максимум {np.abs(eth).max():.3f} рад')
    p('\nпо времени (ошибка положения, см / ошибка курса, рад):')
    for t0 in np.arange(0.0, t[-1], 20.0):
        m = (t >= t0) & (t < t0 + 20.0)
        if m.any():
            ev = sorted({e for i in np.nonzero(m)[0] for e in d['events'][i].split('+') if e})
            p(f'  {t0:5.0f}–{t0 + 20:<4.0f} {np.median(err[m]) * 100:6.1f}  {np.median(eth[m]):+.3f}   {" ".join(ev)}')

    # Где ошибка рождается: приращения за шаг в системе робота не зависят от уже накопленного ухода.
    inc = increments(d)
    df = inc['o'][0] - inc['g'][0]          # одометрия проехала вперёд больше, чем робот
    dl = inc['o'][1] - inc['g'][1]          # робота снесло вбок
    da = inc['o'][2] - inc['g'][2]          # одометрия повернула больше, чем робот
    cv, cw = d['cv'][:-1], d['cw'][:-1]
    gf, ga = inc['g'][0], inc['g'][2]
    hit = np.zeros(n - 1, bool)
    for i, e in enumerate(d['events'][:-1]):
        if 'collision' in e:
            hit[max(0, i - 10):i + 20] = True          # секунда до события и две после
    kinds = {
        'стоит': (np.abs(cv) < 0.01) & (np.abs(cw) < 0.05),
        'разворот на месте': (np.abs(cv) < 0.01) & (np.abs(cw) >= 0.05),
        'вперёд прямо': (cv >= 0.01) & (np.abs(cw) < 0.3),
        'вперёд по дуге': (cv >= 0.01) & (np.abs(cw) >= 0.3),
        'задом': cv <= -0.01,
    }
    p('\nгде рождается ошибка (сумма по шагам; «вперёд» — лишний путь по одометрии, «вбок» — снос, «курс» — лишний поворот):')
    p(f'  {"манёвр":<22}{"время, с":>9}{"путь, м":>9}{"поворот, рад":>14}{"вперёд, см":>12}{"вбок, см":>10}{"курс, рад":>11}')

    def line(name, m):
        if m.any():
            p(f'  {name:<22}{m.sum() * 0.1:9.1f}{np.abs(gf[m]).sum():9.2f}{np.abs(ga[m]).sum():14.2f}'
              f'{df[m].sum() * 100:12.1f}{dl[m].sum() * 100:10.1f}{da[m].sum():11.3f}')

    for name, m in kinds.items():
        line(name, m & ~hit)
    line('около столкновений', hit)
    line('всего', np.ones(n - 1, bool))
    p('\nуход курса при повороте по скорости разворота (вне столкновений):')
    for lo, hi in ((0.05, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0)):
        m = (np.abs(cw) >= lo) & (np.abs(cw) < hi) & ~hit
        turned = np.abs(ga[m]).sum()
        if turned > 0.2:
            p(f'  |w| {lo:.2f}–{hi:.2f}: повёрнуто {turned:6.2f} рад, лишнего по одометрии '
              f'{(np.sign(inc["o"][2][m]) * da[m]).sum():+.3f} рад ({(np.sign(inc["o"][2][m]) * da[m]).sum() / turned:+.2%})')
    moving = (np.abs(gf) > 0.002) & ~hit
    if moving.any():
        p(f'проскальзывание по пути (вне столкновений): одометрия {np.abs(inc["o"][0][moving]).sum():.2f} м, '
          f'истина {np.abs(gf[moving]).sum():.2f} м')
    stalled = (np.abs(inc['o'][0]) > 0.004) & (np.abs(gf) < 0.3 * np.abs(inc['o'][0]))
    p(f'колёса крутятся, робот почти стоит: {stalled.sum() * 0.1:.1f} с, лишний путь {np.abs(df[stalled]).sum() * 100:.1f} см, '
      f'лишний поворот {np.abs(da[stalled]).sum():.3f} рад')
    p(f'наклон корпуса (тангаж): максимум {np.abs(d["gpitch"]).max():.3f} рад')

    out_stats = {'odom': _stats(err[run]), 'odom_th': _stats(np.abs(eth[run]))}
    if trace:
        ax, ay, ath = agent_pose(d, tr)
        aerr = np.hypot(ax - d['gx'], ay - d['gy'])[run]
        aeth = np.abs(_wrap(ath - d['gth']))[run]
        res = tr['result']
        o, oth = out_stats['odom'], out_stats['odom_th']
        p(f'\nпока агент ведёт робота ({t[run][0]:.0f}–{t[run][-1]:.0f} с; {trace}, поправок в записи: '
          f'{len(tr.get("pose_fix") or [])}):')
        p(f'  одометрия:   положение — медиана {o["median"] * 100:.1f} см, 95% {o["p95"] * 100:.1f} см, максимум '
          f'{o["max"] * 100:.1f} см; курс — медиана {oth["median"]:.3f}, максимум {oth["max"]:.3f} рад')
        p(f'  поза агента: положение — медиана {np.median(aerr) * 100:.1f} см, 95% {np.percentile(aerr, 95) * 100:.1f} см, '
          f'максимум {aerr.max() * 100:.1f} см (на {t[run][int(aerr.argmax())]:.0f}-й с); курс — медиана '
          f'{np.median(aeth):.3f}, 95% {np.percentile(aeth, 95):.3f}, максимум {aeth.max():.3f} рад')
        p(f'  итог: образцов {res.get("samples_collected")}/{res.get("samples_total")}, вернулся: {res.get("returned")}, '
          f'столкновений {res.get("collisions")}, очки {res.get("score")}, время {res.get("t")} с, '
          f'заряд {res.get("battery_used")}')
        out_stats.update(agent=_stats(aerr), agent_th=_stats(aeth), result=res)
    return out_stats


def _stats(v):
    return {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help='куда писать журнал (CSV)')
    ap.add_argument('--scans', default=None, help='куда писать сканы лидара с позами (.npz)')
    ap.add_argument('--max-s', type=float, default=900.0, help='предел записи по часам машины')
    ap.add_argument('--report', default=None, help='разобрать готовый журнал')
    ap.add_argument('--trace', default=None, help='запись прогона агента: для ошибки его позы')
    args = ap.parse_args()
    if args.report:
        report(args.report, args.trace)
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        record(args.out, args.scans, args.max_s)
    else:
        ap.error('нужен --out или --report')


if __name__ == '__main__':
    main()
