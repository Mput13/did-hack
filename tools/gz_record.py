#!/usr/bin/env python3
"""Видео из окна Gazebo: серия снимков сцены через сервис /gui/screenshot, затем сборка в mp4.

    pixi run python tools/gz_record.py --out demo/gazebo.mp4                  # до Ctrl+C
    pixi run python tools/gz_record.py --out demo/gazebo.mp4 --seconds 300 --view top
    pixi run python tools/gz_record.py --shot demo/shots/gazebo.png           # один снимок

У окна Gazebo Harmonic нет сервиса записи видео (/gui/record_video), есть только снимок сцены,
поэтому видео собирается из снимков (по умолчанию 2 кадра в секунду) программой ffmpeg. Окно
Gazebo должно быть открыто (pixi run demo); если оно свёрнуто, кадры не обновляются.
"""
import argparse
import math
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VIEWS = {
    # откуда смотрит камера: (x, y, z) и точка, на которую она направлена
    'iso': ((0.0, -3.9, 4.7), (0.0, 0.1, 0.0)),
    'top': ((0.0, 0.0, 6.0), (0.0, 0.0, 0.0)),
}


def look_at(eye, target):
    """Кватернион камеры (x вперёд, z вверх), смотрящей из eye в target."""
    dx, dy, dz = (t - e for e, t in zip(eye, target))
    yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else math.pi / 2
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return cp * cy, -sp * sy, sp * cy, cp * sy          # w, x, y, z


class Gui:

    def __init__(self):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.gui_camera_pb2 import GUICamera
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.transport13 import Node
        self._m = (Boolean, GUICamera, StringMsg)
        self.node = Node()

    def ready(self):
        return '/gui/screenshot' in self.node.service_list()

    def view(self, name):
        Boolean, GUICamera, _ = self._m
        eye, target = VIEWS[name]
        req = GUICamera()
        req.pose.position.x, req.pose.position.y, req.pose.position.z = eye
        q = req.pose.orientation
        q.w, q.x, q.y, q.z = look_at(eye, target)
        ok, _ = self.node.request('/gui/move_to/pose', req, GUICamera, Boolean, 2000)
        return ok

    def shot(self, folder):
        """Попросить окно сохранить снимок сцены в папку (имя файла — время)."""
        Boolean, _, StringMsg = self._m
        req = StringMsg()
        req.data = str(folder)
        ok, rep = self.node.request('/gui/screenshot', req, StringMsg, Boolean, 1500)
        return bool(ok and rep.data)


def assemble(frames, out, fps, speed):
    """Кадры (по времени создания) → mp4. Длительность каждого кадра — по времени между снимками."""
    files = sorted(frames.glob('*.png'), key=lambda f: f.stat().st_mtime)
    if len(files) < 2:
        return 0
    stamps = [f.stat().st_mtime for f in files]
    listing = frames / 'frames.txt'
    with open(listing, 'w') as f:
        for a, b, t in zip(files, files[1:] + [files[-1]], [*(y - x for x, y in zip(stamps, stamps[1:])), 1.0 / fps]):
            f.write(f"file '{a}'\nduration {max(t, 0.02) / speed:.4f}\n")
        f.write(f"file '{files[-1]}'\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', '23', '-movflags', '+faststart', str(out)]
    subprocess.run(cmd, check=True)
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='demo/gazebo.mp4', help='куда сохранить видео')
    ap.add_argument('--shot', default=None, help='сделать один снимок в этот файл и выйти')
    ap.add_argument('--fps', type=float, default=2.0, help='снимков в секунду')
    ap.add_argument('--seconds', type=float, default=0.0, help='длительность записи; 0 — до Ctrl+C')
    ap.add_argument('--speed', type=float, default=1.0, help='ускорение готового видео')
    ap.add_argument('--view', default=None, choices=list(VIEWS), help='поставить камеру: iso — сбоку сверху, top — сверху')
    ap.add_argument('--frames', default=None, help='папка для кадров (по умолчанию временная, удаляется)')
    args = ap.parse_args()

    gui = Gui()
    deadline = time.monotonic() + 10.0
    while not gui.ready():
        if time.monotonic() > deadline:
            sys.exit('окно Gazebo не найдено: нет сервиса /gui/screenshot (запущен ли показ с окном?)')
        time.sleep(0.3)
    if args.view:
        print('камера:', args.view, 'поставлена' if gui.view(args.view) else 'не ответила', flush=True)
        time.sleep(1.5)

    frames = Path(args.frames or tempfile.mkdtemp(prefix='gz_frames_')).resolve()
    frames.mkdir(parents=True, exist_ok=True)
    if args.shot:
        gui.shot(frames)
        time.sleep(1.2)
        files = sorted(frames.glob('*.png'), key=lambda f: f.stat().st_mtime)
        if not files:
            sys.exit('снимок не получен: окно Gazebo свёрнуто или не отвечает')
        out = Path(args.shot).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(files[-1], out)
        print('снимок:', out)
        if not args.frames:
            shutil.rmtree(frames, ignore_errors=True)
        return

    halt = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.append(1))
    print(f'запись сцены Gazebo: {args.fps:g} кадра в секунду, Ctrl+C — остановить', flush=True)
    t0 = time.monotonic()
    n = 0
    while not halt and (not args.seconds or time.monotonic() - t0 < args.seconds):
        tick = time.monotonic()
        n += gui.shot(frames)
        time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))
    time.sleep(0.8)                                 # последний кадр ещё пишется
    out = Path(args.out).resolve()
    saved = assemble(frames, out, args.fps, args.speed)
    print(f'запрошено кадров {n}, сохранено {saved}, видео: {out}' if saved else 'кадров нет: окно Gazebo свёрнуто?')
    if not args.frames:
        shutil.rmtree(frames, ignore_errors=True)


if __name__ == '__main__':
    main()
