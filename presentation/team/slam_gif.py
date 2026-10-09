#!/usr/bin/env python3
"""Анимация для слайда «SLAM вместо готовой карты»: fig/slam_build.gif из записи runs/SLAMgif/record.json.gz.

    /usr/local/bin/python3 presentation/team/slam_gif.py

На кадре — то, что есть у робота: сетка SLAM Toolbox (серое — не видел, белое — пол, тёмное — преграда), лучи
лидара, пройденный след, путь к краю увиденного. Первый кадр — готовая карта: в PDF, где анимации нет, виден итог.
"""
import base64
import gzip
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REC = HERE.parents[1] / 'runs' / 'SLAMgif' / 'record.json.gz'
OUT = HERE / 'fig' / 'slam_build.gif'
S = 2                      # рисуем вдвое крупнее и уменьшаем: линии выходят гладкими
CELL = 4 * S               # пикселей на клетку 5 см
X0, X1, Y0, Y1 = -2.75, 2.75, -2.7, 2.7    # окно мира, м
HEAD = 46                  # высота строки с подписью над картой, пикселей итогового кадра
BG, UNKNOWN, FREE, WALL = (248, 249, 252), (223, 229, 239), (255, 255, 255), (24, 43, 69)
BLUE, VIOLET, GREEN, RAY, INK2 = (61, 99, 221), (232, 89, 12), (46, 158, 115), (196, 211, 247), (64, 83, 107)   # VIOLET — путь: оранжевый
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'


def main():
    with gzip.open(REC, 'rt', encoding='utf-8') as f:
        rec = json.load(f)
    frames = rec['frames']
    grid = None
    for fr in frames:                      # сетка приходит только при смене версии: протягиваем последнюю
        grid = fr.get('slam') or grid
        fr['grid'] = grid
    g0 = frames[0]['grid']
    res, gx0, gy0, gw, gh = g0['res'], g0['x0'], g0['y0'], g0['w'], g0['h']
    W, H = round((X1 - X0) / res) * CELL, round((Y1 - Y0) / res) * CELL
    font, small = ImageFont.truetype(FONT, 15 * S), ImageFont.truetype(FONT, 12 * S)

    def P(x, y):
        return (x - X0) / res * CELL, (Y1 - y) / res * CELL

    def draw(fr, label, sub):
        raw = base64.b64decode(fr['grid']['data'])
        cells = Image.frombytes('L', (gw, gh), raw).transpose(Image.FLIP_TOP_BOTTOM)
        cells = cells.point(lambda v: 0 if v == 0 else (2 if v > 1 else 1))
        cells.putpalette(list(UNKNOWN) + list(FREE) + list(WALL) + [0] * (253 * 3))
        cells = cells.convert('P')
        cells.putpalette(list(UNKNOWN) + list(FREE) + list(WALL) + [0] * (253 * 3))
        big = cells.convert('RGB').resize((gw * CELL, gh * CELL), Image.NEAREST)
        left, top = round((X0 - gx0) / res) * CELL, round((gy0 + gh * res - Y1) / res) * CELL
        im = big.crop((left, top, left + W, top + H))
        d = ImageDraw.Draw(im)
        sc = fr.get('scan')
        if sc and sc.get('r'):
            unit = 0.01 if max(sc['r']) > 20 else 1.0
            ox, oy = P(sc['x'], sc['y'])
            for i, r in enumerate(sc['r']):
                if r and i % 2 == 0:
                    a = sc['th'] + i * sc['step']
                    d.line([ox, oy, *P(sc['x'] + r * unit * math.cos(a), sc['y'] + r * unit * math.sin(a))], fill=RAY, width=S)
        trail = fr.get('trail') or []
        if len(trail) > 1:
            d.line([P(x, y) for x, y in trail], fill=BLUE, width=3 * S, joint='curve')
        path = fr.get('path') or []
        if len(path) > 1 and fr['mode'] != 'idle':
            pts = [P(p[0], p[1]) for p in path]
            d.line(pts, fill=VIOLET, width=2 * S, joint='curve')
            ex, ey = pts[-1]
            color = GREEN if fr['phase'] == 'goal' else VIOLET
            d.ellipse([ex - 6 * S, ey - 6 * S, ex + 6 * S, ey + 6 * S], outline=color, width=2 * S)
        bx, by = P(-2.0, -0.5)
        d.rectangle([bx - 6 * S, by - 6 * S, bx + 6 * S, by + 6 * S], outline=WALL, width=2 * S)
        x, y, th = fr['pose']
        px, py = P(x, y)
        d.ellipse([px - 7 * S, py - 7 * S, px + 7 * S, py + 7 * S], fill=BLUE, outline=(255, 255, 255), width=2 * S)
        d.line([px, py, px + 11 * S * math.cos(th), py - 11 * S * math.sin(th)], fill=(255, 255, 255), width=2 * S)
        out = Image.new('RGB', (W, H + HEAD * S), BG)      # подпись — над картой, карту не закрывает
        out.paste(im, (0, HEAD * S))
        d = ImageDraw.Draw(out)
        d.text((4 * S, 5 * S), label, font=font, fill=WALL)
        d.text((4 * S, 25 * S), sub, font=small, fill=INK2)
        return out.resize((W // S, (H + HEAD * S) // S), Image.LANCZOS)

    explore = [f for f in frames if f['phase'] == 'explore']
    goal = [f for f in frames if f['phase'] == 'goal']
    t0 = explore[0]['t']
    shots = []
    for fr in explore[::3]:
        shots.append((draw(fr, 'Робот строит карту сам', f"{fr['t'] - t0:.0f} с · увидено {round(100 * (fr['coverage'] or 0))}% пола"), 130))
    done = explore[-1]
    built = draw(done, 'Карта построена', f"{done['t'] - t0:.0f} с · увидено {round(100 * (done['coverage'] or 0))}% пола")
    shots.append((built, 1400))
    for fr in goal[::3]:
        shots.append((draw(fr, 'Едет к заданной точке по своей карте', 'путь проложен только по увиденному полу'), 130))
    last = draw(frames[-1], 'Приехал: карта, поза и путь — свои', f"увидено {round(100 * (frames[-1]['coverage'] or 0))}% пола, столкновений нет")
    shots.append((last, 2600))
    shots.insert(0, (last, 900))            # первый кадр — итог: его покажет PDF

    # Палитра задана явно: цвета рисунка и их смеси с фоном (сглаженные края). Иначе путь и цель сливаются с синим.
    keys, grounds, white = [WALL, BLUE, VIOLET, GREEN, RAY, INK2], [FREE, UNKNOWN, BG], (255, 255, 255)
    colors = [BG, UNKNOWN, FREE, white] + keys
    for c in keys:
        for g in grounds:
            colors += [tuple(round(c[i] * k + g[i] * (1 - k)) for i in range(3)) for k in (0.33, 0.66)]
    colors += [tuple(round(BLUE[i] * k + white[i] * (1 - k)) for i in range(3)) for k in (0.33, 0.66)]
    ref = Image.new('P', (1, 1))
    ref.putpalette([v for c in colors for v in c] + [0] * (3 * (256 - len(colors))))
    pal = [im.quantize(palette=ref, dither=Image.NONE) for im, _ in shots]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=[ms for _, ms in shots], loop=0, optimize=False, disposal=1)
    print(OUT, f'{OUT.stat().st_size / 1024:.0f} КБ', 'кадров', len(shots), 'размер', last.size,
          '| объезд', f"{done['t'] - t0:.0f} с, увидено {done['coverage']}, до точки {rec.get('arrivals')}, столкновений {rec['score'].get('collisions')}")


if __name__ == '__main__':
    main()
