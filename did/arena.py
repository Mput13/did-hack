"""Арена turtlebot3_world по карте из turtlebot3_navigation2: клетки, зазоры до стен, лидар."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from . import ROOT
from .config import BASE, LIDAR_MAX, LIDAR_RAYS

CROP = (-3.0, -3.0, 3.0, 3.0)   # мировые границы рабочей области: шестиугольник с запасом


def _read_pgm(path):
    raw = Path(path).read_bytes()
    pos = 0
    tokens = []
    while len(tokens) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b'#':
            pos = raw.index(b'\n', pos)
            continue
        end = pos
        while not raw[end:end + 1].isspace():
            end += 1
        tokens.append(raw[pos:end])
        pos = end
    assert tokens[0] == b'P5', 'ожидается бинарный PGM'
    width, height = int(tokens[1]), int(tokens[2])
    data = np.frombuffer(raw[pos + 1:pos + 1 + width * height], dtype=np.uint8)
    return data.reshape(height, width)[::-1]   # строка 0 — нижняя, как в мировых координатах


class Arena:
    """Сетка 0,05 м. Индексация [iy, ix], клетка (ix, iy) покрывает [x0 + ix*res, x0 + (ix+1)*res)."""

    def __init__(self, map_yaml=None):
        map_yaml = Path(map_yaml or ROOT / 'maps' / 'map.yaml')
        meta = yaml.safe_load(map_yaml.read_text())
        self.res = float(meta['resolution'])
        img = _read_pgm(map_yaml.parent / meta['image'])
        ox, oy = meta['origin'][0], meta['origin'][1]
        ix0, iy0 = round((CROP[0] - ox) / self.res), round((CROP[1] - oy) / self.res)
        ix1, iy1 = round((CROP[2] - ox) / self.res), round((CROP[3] - oy) / self.res)
        img = img[iy0:iy1, ix0:ix1]
        self.x0, self.y0 = CROP[0], CROP[1]
        self.h, self.w = img.shape

        # Свободно только то, что связано с базой: внутри столбов и за стенами на карте «неизвестно».
        labels, _ = ndimage.label(img > 250)
        bx, by = self.w2g(*BASE)
        self.free = labels == labels[by, bx]
        self.solid = ~self.free
        self.clear = ndimage.distance_transform_edt(self.free) * self.res   # м до ближайшей преграды
        self._ray_r = np.arange(0.02, LIDAR_MAX + 0.02, 0.02)

    def w2g(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def g2w(self, ix, iy):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def inside(self, ix, iy):
        return 0 <= ix < self.w and 0 <= iy < self.h

    def clearance(self, x, y):
        ix, iy = self.w2g(x, y)
        if not self.inside(ix, iy):
            return 0.0
        return float(self.clear[iy, ix])

    def is_free(self, x, y, margin=0.0):
        return self.clearance(x, y) > margin

    def cell_centers(self):
        xs = self.x0 + (np.arange(self.w) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.h) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def raycast(self, x, y, theta, n=LIDAR_RAYS, rmax=LIDAR_MAX):
        """Дальности n лучей от 0 до 2π против часовой, первый луч смотрит по курсу. inf — нет преграды."""
        ang = theta + np.arange(n) * (2 * np.pi / n)
        r = self._ray_r[self._ray_r <= rmax + 1e-9]
        ix = ((x + np.cos(ang)[:, None] * r[None, :] - self.x0) / self.res).astype(np.int32)
        iy = ((y + np.sin(ang)[:, None] * r[None, :] - self.y0) / self.res).astype(np.int32)
        np.clip(ix, 0, self.w - 1, out=ix)
        np.clip(iy, 0, self.h - 1, out=iy)
        hit = self.solid[iy, ix]
        first = hit.argmax(axis=1)
        return np.where(hit.any(axis=1), r[first], np.inf)

    def to_dict(self):
        """Геометрия для интерфейса: свободные клетки построчно, строка 0 — нижняя."""
        return {
            'res': self.res, 'x0': self.x0, 'y0': self.y0, 'w': self.w, 'h': self.h,
            'free': [''.join('1' if v else '0' for v in row) for row in self.free],
            'base': list(BASE),
        }


@lru_cache(maxsize=2)
def load_arena(map_yaml=None):
    return Arena(map_yaml)
