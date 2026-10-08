"""Конструктор исследования: пользователь задаёт, что роботу измерить, какими опытами и с каким бюджетом.

    pixi run python -m did.study --spec файл.json --level medium --seed 3
    pixi run python -m did.study --preset soil_b --level medium --seed 3

Задание (StudySpec) читается из JSON или YAML. Пример: «Исследовать расход в области B. Сравнивать
прямолинейное движение. Разрешить максимум два контрольных участка. Обязательно сохранить запас на
возврат»:

    {"quantity": "soil_cost", "region": {"zone": "B"}, "allowed": {"straight": {"length_m": 0.5}},
     "controls": {"max": 2}, "budget": {"energy": 40, "reserve": 3}}

Здесь схема задания, смысловая проверка с понятными сообщениями, выбор участков для опытов, скрытая
правда для сверки, отчёт словами и запуск. Сам исполнитель — did/study_agent.py.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import BASE, SCIENCE, Rules
from .nav import CostGraph, path_length
from .scenario import Zone, soil_mult

MIN_RUN_M = 0.25         # короче пробег не даёт различимого расхода
EDGE_M = 0.08            # отступ пробега от границы области: робот не должен выехать за неё
CONTROL_GAP_M = 0.25     # контрольный участок не ближе этого к области
RUN_CLEAR_M = 0.20       # зазор до стен на прямом пробеге
SPOT_CLEAR_M = 0.25      # и на месте разворотов и пауз
TRAVEL_FACTOR = 1.12     # дорога дороже, чем «метры × номинал»: повороты и простой
TRAVEL_V = 0.17          # средняя скорость в пути, м/с
MEASURE_V = 0.15         # скорость на мерном пробеге, м/с
LEG_OVERHEAD_S = 6.0     # разворот, остановка и два окна усреднения на один пробег
SPIN_W = 1.0             # рад/с на мерном развороте
SOIL_GUESS = (1.0, 4.0)  # до опыта цена пола в области неизвестна: считаем бюджет для обоих краёв

QUANTITIES = {
    # id: (что измеряем, единица, нужное воздействие, коротко для заголовка)
    'soil_cost': ('во сколько раз пол в области дороже обычного', '× к обычному полу', 'straight', 'Цена пола в области'),
    'turn_cost': ('сколько заряда стоит поворот', 'ед/рад', 'spin', 'Цена поворота'),
    'idle_cost': ('сколько заряда уходит на месте', 'ед/с', 'pause', 'Расход на месте'),
    'load_effect': ('на какую долю каждый несомый образец удорожает метр пути', 'доля на образец', 'straight',
                    'Влияние груза'),
    'sensor_law': ('как показание датчика образцов зависит от расстояния', 'м дальности', 'pause', 'Закон датчика'),
}
ACTIONS = {'straight': 'прямые пробеги', 'pause': 'паузы на месте', 'spin': 'развороты на месте'}
LAWS = {
    # форма зависимости показания от расстояния d при дальности R
    'linear': lambda d, r: max(0.0, 1.0 - d / r),
    'faster': lambda d, r: max(0.0, 1.0 - d / r) ** 2,           # вблизи падает резко
    'slower': lambda d, r: max(0.0, 1.0 - (d / r) ** 2),         # вблизи почти не меняется
}
DEFAULT_LAWS = [
    {'id': 'linear_2m', 'statement': 'показание линейно убывает до нуля на 2 м', 'law': 'linear', 'range_m': 2.0, 'prior': 0.4},
    {'id': 'linear_1p5m', 'statement': 'линейно, но дальность меньше: 1,5 м', 'law': 'linear', 'range_m': 1.5, 'prior': 0.15},
    {'id': 'linear_3m', 'statement': 'линейно, но дальность больше: 3 м', 'law': 'linear', 'range_m': 3.0, 'prior': 0.15},
    {'id': 'faster', 'statement': 'убывает быстрее линейного: вблизи падает резко', 'law': 'faster', 'range_m': 2.0, 'prior': 0.1},
    {'id': 'slower', 'statement': 'убывает медленнее линейного: вблизи почти не меняется', 'law': 'slower', 'range_m': 2.0,
     'prior': 0.1},
]


# =================================================================================================
# схема задания
# =================================================================================================

class _Part(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Region(_Part):
    """Область на карте: круг, прямоугольник со сторонами вдоль осей или имя области сценария («B»)."""
    zone: Optional[str] = None
    shape: Optional[Literal['circle', 'rect']] = None
    x: float = 0.0
    y: float = 0.0
    r: float = Field(0.0, ge=0.0, le=3.0)
    w: float = Field(0.0, ge=0.0, le=6.0)
    h: float = Field(0.0, ge=0.0, le=6.0)

    @model_validator(mode='after')
    def _given(self):
        if not self.zone and not self.shape:
            raise ValueError('нужно либо имя области (zone), либо фигура (shape: circle или rect)')
        if self.shape == 'circle' and self.r <= 0:
            raise ValueError('у круга нужен радиус r больше нуля')
        if self.shape == 'rect' and (self.w <= 0 or self.h <= 0):
            raise ValueError('у прямоугольника нужны ширина w и высота h больше нуля')
        return self


class Point(_Part):
    x: float
    y: float


class Straight(_Part):
    length_m: float = Field(0.5, ge=MIN_RUN_M, le=2.0)     # длина одного пробега
    repeats: int = Field(2, ge=1, le=12)                   # не меньше стольких пробегов на участок (2 — туда и обратно)


class Pause(_Part):
    seconds: float = Field(3.0, ge=1.5, le=30.0)
    repeats: int = Field(1, ge=1, le=12)


class Spin(_Part):
    angle_deg: float = Field(360.0, ge=90.0, le=1440.0)
    repeats: int = Field(2, ge=1, le=12)                   # 2 — по часовой и против


class Allowed(_Part):
    """Какие воздействия разрешены. Чего нет в задании — того робот не делает."""
    straight: Optional[Straight] = None
    pause: Optional[Pause] = None
    spin: Optional[Spin] = None

    @model_validator(mode='before')
    @classmethod
    def _names(cls, v):
        if isinstance(v, (list, tuple)):          # ['straight', 'pause'] — с параметрами по умолчанию
            return {str(k): {} for k in v}
        if isinstance(v, dict):
            return {k: ({} if val is True else val) for k, val in v.items() if val is not False}
        return v

    def kinds(self):
        return [k for k in ACTIONS if getattr(self, k) is not None]


class Controls(_Part):
    max: int = Field(1, ge=0, le=4)                        # сколько контрольных участков можно занять
    sites: list[Point] = Field(default_factory=list)       # пусто — «выбери сам на проверенном полу»


class Stop(_Part):
    rel_error: float = Field(0.05, gt=0.0, le=1.0)         # требуемая точность: полуширина 95% интервала в долях оценки
    max_measurements: int = Field(16, ge=1, le=60)
    time_s: float = Field(300.0, ge=20.0, le=600.0)
    confidence: float = Field(0.9, ge=0.6, le=0.999)       # для сравнения объяснений: с какой вероятности считать доказанным


class Budget(_Part):
    energy: float = Field(30.0, gt=0.0, le=1000.0)         # заряд на всё исследование вместе с дорогой туда и обратно
    reserve: float = Field(3.0, ge=0.0, le=100.0)          # неприкосновенный запас сверх оценки дороги домой


class Hypothesis(_Part):
    id: str = Field(min_length=1, max_length=24)
    statement: str = Field(min_length=1, max_length=200)
    value: Optional[float] = None                          # что объяснение предсказывает для измеряемой величины
    sigma: Optional[float] = Field(None, gt=0.0)
    law: Optional[Literal['linear', 'faster', 'slower']] = None     # для датчика: форма зависимости
    range_m: Optional[float] = Field(None, gt=0.2, le=6.0)          # и дальность
    prior: float = Field(1.0, gt=0.0)


class StudySpec(_Part):
    title: str = Field('', max_length=160)
    quantity: Literal['soil_cost', 'turn_cost', 'idle_cost', 'load_effect', 'sensor_law']
    region: Optional[Region] = None
    allowed: Allowed = Field(default_factory=lambda: Allowed(straight=Straight(), pause=Pause(), spin=Spin()))
    controls: Controls = Field(default_factory=Controls)
    stop: Stop = Field(default_factory=Stop)
    budget: Budget = Field(default_factory=Budget)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=8)

    @field_validator('hypotheses')
    @classmethod
    def _ids(cls, v):
        ids = [x.id for x in v]
        if len(set(ids)) != len(ids):
            raise ValueError('у объяснений должны быть разные id')
        if 'other' in ids:
            raise ValueError('id «other» занят: так называется объяснение «причина не из списка»')
        return v

    @property
    def unit(self):
        return QUANTITIES[self.quantity][1]

    @property
    def label(self):
        return QUANTITIES[self.quantity][0]

    def to_dict(self):
        return self.model_dump(exclude_none=True)


class StudyError(ValueError):
    """Задание не прошло проверку. problems — список сообщений для человека."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('; '.join(self.problems))


_FIELD_RU = {'quantity': 'что исследуем', 'region': 'область', 'allowed': 'разрешённые опыты',
             'controls': 'контрольные участки', 'stop': 'условия остановки', 'budget': 'бюджет',
             'hypotheses': 'объяснения', 'length_m': 'длина пробега', 'repeats': 'число повторов',
             'seconds': 'длительность паузы', 'angle_deg': 'угол разворота', 'rel_error': 'точность',
             'max_measurements': 'максимум замеров', 'time_s': 'лимит времени', 'energy': 'заряд',
             'reserve': 'запас на возврат', 'max': 'сколько участков', 'sites': 'места', 'confidence': 'уверенность'}
_TYPE_RU = {'missing': 'не задано', 'extra_forbidden': 'такого поля в задании нет',
            'greater_than': 'нужно больше {gt}', 'greater_than_equal': 'нужно не меньше {ge}',
            'less_than': 'нужно меньше {lt}', 'less_than_equal': 'нужно не больше {le}',
            'float_parsing': 'нужно число', 'float_type': 'нужно число', 'int_parsing': 'нужно целое число',
            'int_type': 'нужно целое число', 'int_from_float': 'нужно целое число', 'string_type': 'нужна строка',
            'literal_error': 'нужно одно из: {expected}', 'model_type': 'нужен объект с полями',
            'dict_type': 'нужен объект с полями', 'list_type': 'нужен список', 'too_long': 'слишком длинный список',
            'string_too_long': 'слишком длинная строка', 'string_too_short': 'пустая строка'}


def _schema_problems(exc):
    out = []
    for e in exc.errors(include_url=False):
        where = ' → '.join(f'№{p + 1}' if isinstance(p, int) else _FIELD_RU.get(p, str(p)) for p in e['loc'])
        try:
            what = _TYPE_RU[e['type']].format(**(e.get('ctx') or {}))
        except (KeyError, IndexError):
            what = str(e['msg']).removeprefix('Value error, ')
        out.append(f'{where}: {what}' if where else what)
    return out


def load_spec(source):
    """StudySpec из словаря, строки JSON/YAML или пути к файлу .json/.yaml. Ошибки — StudyError по-русски."""
    if isinstance(source, StudySpec):
        return source
    if isinstance(source, (str, Path)):
        text = str(source)
        if '\n' not in text and Path(text).suffix.lower() in ('.json', '.yaml', '.yml'):
            path = Path(text)
            if not path.is_file():
                raise StudyError([f'файл задания не найден: {path}'])
            text = path.read_text(encoding='utf-8')
        try:
            source = yaml.safe_load(text)            # JSON — подмножество YAML
        except yaml.YAMLError as e:
            raise StudyError([f'задание не читается как JSON или YAML: {e}']) from e
    if not isinstance(source, dict):
        raise StudyError(['задание должно быть объектом с полями (quantity, region, allowed, …)'])
    try:
        return StudySpec.model_validate(source)
    except ValidationError as e:
        raise StudyError(_schema_problems(e)) from e


# =================================================================================================
# геометрия: область, прямые участки, место для опытов на месте
# =================================================================================================

def region_zone(region, scenario=None):
    """Область задания как фигура на карте. Имя заменяется контуром области сценария — без её цены."""
    if region is None:
        return None, None
    if region.zone:
        soils = list(getattr(scenario, 'soils', None) or [])
        z = next((s for s in soils if str(s.id).lower() == region.zone.strip().lower()), None)
        if z is None:
            have = ', '.join(str(s.id) for s in soils) or 'ни одной'
            return None, (f'в этом сценарии нет области «{region.zone}»: есть {have}. '
                          'Выберите другую или обведите область на карте')
        return Zone('R', z.shape, float(z.x), float(z.y), r=float(z.r), w=float(z.w), h=float(z.h)), None
    return Zone('R', region.shape, region.x, region.y, r=region.r, w=region.w, h=region.h), None


def _grown(zone, by):
    """Та же фигура, расширенная (by > 0) или суженная (by < 0) на by метров."""
    return Zone(zone.id, zone.shape, zone.x, zone.y, r=max(zone.r + by, 0.0), w=max(zone.w + 2 * by, 0.0),
                h=max(zone.h + 2 * by, 0.0))


def _mask(arena, zone):
    X, Y = arena.cell_centers()
    return zone.mask(X, Y)


def find_segment(arena, ok, length, near, min_len=MIN_RUN_M):
    """Прямой отрезок по разрешённым клеткам: как можно ближе к длине length, серединой ближе к точке near.

    Возвращает {'a': (x, y), 'b': (x, y), 'length': м} или None, если не помещается и отрезок в min_len.
    """
    res = arena.res
    n = max(1, int(round(length / res)))
    X, Y = arena.cell_centers()
    found = []                                 # (угол, сколько шагов по 5 см свободно от каждой клетки)
    for k in range(8):
        th = k * math.pi / 8
        acc = ok.copy()
        count = np.zeros(ok.shape, dtype=np.int32)
        for i in range(1, n + 1):
            ix = np.floor((X + i * res * math.cos(th) - arena.x0) / res).astype(np.int32)
            iy = np.floor((Y + i * res * math.sin(th) - arena.y0) / res).astype(np.int32)
            inside = (ix >= 0) & (ix < arena.w) & (iy >= 0) & (iy < arena.h)
            acc &= inside & ok[np.clip(iy, 0, arena.h - 1), np.clip(ix, 0, arena.w - 1)]
            count += acc
        found.append((th, count))
    steps = min(n, max(int(c.max()) for _, c in found))
    if steps * res < min_len - 1e-9:
        return None
    best = None
    for th, count in found:
        iy, ix = np.nonzero(count >= steps)
        if not len(ix):
            continue
        ax, ay = X[iy, ix], Y[iy, ix]
        half = 0.5 * steps * res
        d = np.hypot(ax + half * math.cos(th) - near[0], ay + half * math.sin(th) - near[1])
        j = int(np.argmin(d))
        if best is None or d[j] < best[0] - 1e-9:
            best = (float(d[j]), float(ax[j]), float(ay[j]), th)
    _, ax, ay, th = best
    L = steps * res
    return {'a': (round(ax, 3), round(ay, 3)), 'b': (round(ax + L * math.cos(th), 3), round(ay + L * math.sin(th), 3)),
            'length': round(L, 3)}


def find_spot(arena, graph, near, inside=None, clear=SPOT_CLEAR_M):
    """Свободная точка для опытов на месте, ближайшая к near (и внутри фигуры inside, если она задана)."""
    ok = graph.ok & (arena.clear >= clear)
    if inside is not None:
        ok &= _mask(arena, inside)
    iy, ix = np.nonzero(ok)
    if not len(ix):
        return None
    X, Y = arena.cell_centers()
    j = int(np.argmin(np.hypot(X[iy, ix] - near[0], Y[iy, ix] - near[1])))
    return (round(float(X[iy[j], ix[j]]), 3), round(float(Y[iy[j], ix[j]]), 3))


def control_segment(arena, region, length, near, taken=(), bad=None):
    """Контрольный участок: прямой отрезок вне области (с отступом), не на забракованном полу, ближе к near."""
    ok = arena.clear >= RUN_CLEAR_M
    if region is not None:
        ok &= ~_mask(arena, _grown(region, CONTROL_GAP_M))
    if bad is not None:
        ok &= ~bad
    X, Y = arena.cell_centers()
    for x, y in taken:                         # второй участок — не вплотную к первому
        ok &= np.hypot(X - x, Y - y) > max(0.6, length)
    return find_segment(arena, ok, length, near)


def _mid(seg):
    return ((seg['a'][0] + seg['b'][0]) / 2, (seg['a'][1] + seg['b'][1]) / 2)


def _approach_anchor(arena, graph, region, target, length):
    """Точка на дороге от базы к области, где ещё помещается контрольный участок: по ней робот проедет сам."""
    pts, _ = graph.plan(BASE, target)
    if not pts:
        return target
    outer = _grown(region, CONTROL_GAP_M + length / 2) if region is not None else None
    for p in reversed(pts):
        if outer is None or not outer.contains(*p):
            return p
    return pts[0]


# =================================================================================================
# смысловая проверка и план
# =================================================================================================

def check_study(spec, arena, rules=None, scenario=None, carried=0, graph=None):
    """Смысловая проверка задания и черновой план: где ставить опыты и во что это обойдётся.

    Возвращает словарь: ok, problems [{level: error|warning, field, text}], region (фигура), sites
    {test, controls}, route, costs, steps (план словами). Скрытая цена пола сюда не попадает: из
    сценария берётся только контур названной области.
    """
    rules = rules or Rules()
    graph = graph or CostGraph(arena)
    spec = load_spec(spec)
    problems = []

    def bad(field, text, level='error'):
        problems.append({'level': level, 'field': field, 'text': text})

    q = spec.quantity
    need = QUANTITIES[q][2]
    allowed = spec.allowed
    if getattr(allowed, need) is None:
        bad('allowed', f'для этой величины нужны {ACTIONS[need]}, а в задании они не разрешены')
    if q == 'sensor_law' and allowed.straight is None:
        bad('allowed', 'чтобы сравнить показания на разных расстояниях, нужно разрешить прямые пробеги: робот отъезжает от образца')
    if q == 'load_effect' and carried <= 0:
        bad('quantity', 'робот сейчас не несёт образцов: влияние груза измерить нечем. Это исследование доступно, '
                        'только когда груз уже есть')
    per_m = rules.drain_per_m * (1.0 + rules.load_drain * carried)

    # --- область -----------------------------------------------------------------------------
    region, why = region_zone(spec.region, scenario)
    if why:
        bad('region', why)
    if q == 'soil_cost' and spec.region is None:
        bad('region', 'не задана область: обведите на карте место, где нужно измерить расход')
    sites = {'test': None, 'controls': []}
    usable = None
    if region is not None:
        inside = _mask(arena, region)
        if not inside.any():
            bad('region', 'область целиком вне арены')
        elif not (inside & arena.free).any():
            bad('region', 'область целиком в стене или за пределами пола')
        else:
            usable = inside & graph.ok
            if not usable.any():
                bad('region', 'в области нет места, куда робот может заехать: она слишком близко к стенам')
            elif (inside & ~arena.free).any():
                bad('region', 'часть области занята стеной или столбом: опыты пройдут на свободной части', 'warning')
            if usable.any():
                iy, ix = np.nonzero(usable)
                if not math.isfinite(graph.plan(BASE, arena.g2w(ix[0], iy[0]))[1]):
                    bad('region', 'до области не доехать: она отрезана от базы стенами')
                    usable = None

    # --- места опытов ------------------------------------------------------------------------
    straight = allowed.straight or Straight()
    length = straight.length_m
    if q == 'soil_cost' and usable is not None and usable.any() and allowed.straight is not None:
        ok = _mask(arena, _grown(region, -EDGE_M)) & (arena.clear >= RUN_CLEAR_M)
        seg = find_segment(arena, ok, length, (region.x, region.y))
        if seg is None:
            bad('allowed', f'прямой пробег не помещается: в области нет свободного отрезка длиннее {MIN_RUN_M:.2f} м '
                           f'(нужен отступ {EDGE_M * 100:.0f} см от края области и {RUN_CLEAR_M * 100:.0f} см от стен)')
        else:
            if seg['length'] < length - 1e-6:
                bad('allowed', f'пробег {length:.2f} м в область не помещается: робот проедет {seg["length"]:.2f} м', 'warning')
            sites['test'] = {'id': 'T', 'kind': 'segment', **seg}
    elif q in ('turn_cost', 'idle_cost'):
        spot = find_spot(arena, graph, (region.x, region.y) if region is not None else BASE,
                         inside=region if usable is not None and usable.any() else None)
        if spot is None and region is not None:
            bad('region', f'в области нет точки с зазором {SPOT_CLEAR_M * 100:.0f} см до стен: развернуться и постоять негде')
        elif spot is not None:
            sites['test'] = {'id': 'T', 'kind': 'spot', 'x': spot[0], 'y': spot[1]}
    elif q == 'sensor_law' and region is not None and usable is not None and usable.any():
        spot = find_spot(arena, graph, (region.x, region.y), inside=region, clear=RUN_CLEAR_M)
        if spot is not None:
            sites['test'] = {'id': 'T', 'kind': 'search', 'x': spot[0], 'y': spot[1]}

    # --- контрольные участки -----------------------------------------------------------------
    want_controls = q in ('soil_cost', 'load_effect') or (q == 'idle_cost' and spec.controls.max > 0)
    if len(spec.controls.sites) > spec.controls.max:
        bad('controls', f'задано мест: {len(spec.controls.sites)}, а разрешено участков: {spec.controls.max}')
    if q in ('soil_cost', 'load_effect') and spec.controls.max == 0:
        bad('controls', 'без контрольного участка сравнивать придётся с номиналом из условия (он известен с точностью '
                        'около 10%) — и утечку или дрейф от свойства места отделить будет нечем', 'warning')
    if want_controls and spec.controls.max > 0 and (q != 'soil_cost' or sites['test']):
        target = sites['test']['a'] if (sites['test'] and sites['test']['kind'] == 'segment') else \
            ((sites['test']['x'], sites['test']['y']) if sites['test'] else BASE)
        if q == 'idle_cost':
            for i, p in enumerate(spec.controls.sites[:spec.controls.max] or [None]):
                near = (p.x, p.y) if p else BASE
                if p is not None and not arena.is_free(p.x, p.y):
                    bad('controls', f'контрольная точка {i + 1} в стене или вне пола')
                    continue
                spot = find_spot(arena, graph, near)
                if spot and sites['test'] and math.dist(spot, (sites['test']['x'], sites['test']['y'])) < 0.5:
                    spot = None                    # та же точка, что и основная: сравнивать не с чем
                if spot:
                    sites['controls'].append({'id': f'C{len(sites["controls"]) + 1}', 'kind': 'spot', 'x': spot[0],
                                              'y': spot[1], 'by': 'user' if p else 'auto'})
        else:
            taken = []
            given = list(spec.controls.sites[:spec.controls.max])
            for i, p in enumerate(given):
                if not arena.is_free(p.x, p.y):
                    bad('controls', f'контрольный участок {i + 1} в стене или вне пола')
                    continue
                if region is not None and _grown(region, CONTROL_GAP_M).contains(p.x, p.y):
                    bad('controls', f'контрольный участок {i + 1} внутри исследуемой области или вплотную к ней: '
                                    f'отодвиньте его хотя бы на {CONTROL_GAP_M * 100:.0f} см от края')
                    continue
                seg = control_segment(arena, region, length, (p.x, p.y), taken)
                if seg is None or math.dist(_mid(seg), (p.x, p.y)) > 0.6:
                    bad('controls', f'у контрольного участка {i + 1} нет места для прямого пробега {length:.2f} м')
                    continue
                taken.append(_mid(seg))
                sites['controls'].append({'id': f'C{len(taken)}', 'kind': 'segment', 'by': 'user', **seg})
            if not given:                          # «выбери сам»: у дороги к области, по ней робот проедет и проверит пол
                anchor = _approach_anchor(arena, graph, region, target, length)
                seg = control_segment(arena, region, length, anchor, taken)
                if seg is None:
                    bad('controls', 'не нашлось свободного места для контрольного участка: сравнение пойдёт с номиналом',
                        'warning')
                else:
                    sites['controls'].append({'id': 'C1', 'kind': 'segment', 'by': 'auto', **seg})

    # --- объяснения --------------------------------------------------------------------------
    hyps = spec.hypotheses
    if q == 'sensor_law':
        for x in hyps:
            if x.law is None or x.range_m is None:
                bad('hypotheses', f'объяснению «{x.id}» нужны форма зависимости (law) и дальность (range_m)')
    else:
        for x in hyps:
            if x.value is None:
                bad('hypotheses', f'объяснению «{x.id}» нужно значение (value), которое оно предсказывает')
    if len(hyps) == 1:
        bad('hypotheses', 'сравнивать нечего: объяснение одно. Добавьте хотя бы второе или уберите список')

    # --- дорога, заряд, время ----------------------------------------------------------------
    stops = []                                     # точки маршрута: контроль, потом основное место
    for c in sites['controls'][:1]:
        stops.append(c['a'] if c['kind'] == 'segment' else (c['x'], c['y']))
    t = sites['test']
    if t:
        stops.append(t['a'] if t['kind'] == 'segment' else (t['x'], t['y']))
    road_m, road_e, prev, reachable = 0.0, 0.0, BASE, True
    for p in stops + [BASE]:
        pts, e = graph.plan(prev, p)
        if pts is None:
            reachable = False
            break
        road_m += path_length(pts)
        road_e += e
        prev = p
    road_e *= per_m * TRAVEL_FACTOR
    idle = rules.drain_idle_per_s
    n_test = n_ctrl = 0
    lo = hi = meas_s = 0.0
    if q == 'soil_cost' and t:
        n_test = straight.repeats
        n_ctrl = straight.repeats if sites['controls'] else 0
        leg_s = t['length'] / MEASURE_V + LEG_OVERHEAD_S
        fixed = (n_test + n_ctrl) * (0.4 + idle * leg_s) + n_ctrl * length * per_m
        lo, hi = (fixed + n_test * t['length'] * per_m * k for k in SOIL_GUESS)
        meas_s = (n_test + n_ctrl) * leg_s
    elif q == 'load_effect':
        n_ctrl = straight.repeats if sites['controls'] else 0
        lo = hi = n_ctrl * (length * per_m + 0.4)
        meas_s = n_ctrl * (length / MEASURE_V + LEG_OVERHEAD_S)
    elif q == 'turn_cost' and allowed.spin:
        n_test = allowed.spin.repeats
        ang = math.radians(allowed.spin.angle_deg)
        spin_s = ang / SPIN_W + 3.0
        lo = hi = n_test * (0.2 * ang + idle * spin_s)           # цена поворота неизвестна: берём с запасом
        meas_s = n_test * spin_s + (allowed.pause.seconds if allowed.pause else 0.0)
    elif q == 'idle_cost' and allowed.pause:
        n_test = allowed.pause.repeats
        meas_s = n_test * allowed.pause.seconds * (2 if sites['controls'] else 1)
        lo = hi = idle * meas_s * 2
    elif q == 'sensor_law':
        step = straight.length_m
        lo, hi = 4 * step * per_m * TRAVEL_FACTOR, 12 * step * per_m * TRAVEL_FACTOR
        meas_s = 60.0
    road_s = road_m / TRAVEL_V + 2.5
    budget, reserve = spec.budget.energy, spec.budget.reserve
    start = rules.battery_start
    if budget > start - 1.0:
        bad('budget', f'бюджет {budget:.0f} ед. больше, чем есть в батарее: полный заряд — {start:.0f} ед.')
    if not reachable:
        bad('region', 'до места опытов не доехать: оно отрезано от базы стенами')
    elif stops:
        if budget < road_e + reserve:
            bad('budget', f'бюджет {budget:.1f} ед. меньше дороги туда и обратно: она стоит около {road_e:.1f} ед. '
                          f'и ещё {reserve:.1f} ед. запаса на возврат. Нужно не меньше {road_e + reserve + lo:.0f} ед.')
        elif budget < road_e + reserve + lo:
            bad('budget', f'бюджета {budget:.1f} ед. хватит на дорогу (около {road_e:.1f} ед.) и запас ({reserve:.1f} ед.), '
                          f'но не на замеры: на них нужно ещё хотя бы {lo:.1f} ед.')
        elif budget < road_e + reserve + hi:
            bad('budget', f'если пол в области окажется в {SOIL_GUESS[1]:.0f} раза дороже обычного, бюджета может не хватить: '
                          f'дорога около {road_e:.1f} ед., замеры от {lo:.1f} до {hi:.1f} ед., запас {reserve:.1f} ед.', 'warning')
        if spec.stop.time_s < road_s + 5.0:
            bad('stop', f'за {spec.stop.time_s:.0f} с робот не успеет даже доехать: дорога туда и обратно займёт '
                        f'около {road_s:.0f} с')
        elif spec.stop.time_s < road_s + meas_s:
            bad('stop', f'лимит времени {spec.stop.time_s:.0f} с впритык: дорога около {road_s:.0f} с, минимальный набор '
                        f'замеров — около {meas_s:.0f} с', 'warning')
    elif q == 'sensor_law' and budget < 12.0:
        bad('budget', f'бюджет {budget:.1f} ед. мал: образец сначала нужно найти, на это обычно уходит 10–25 ед.', 'warning')
    if spec.stop.max_measurements < n_test + n_ctrl:
        bad('stop', f'максимум замеров ({spec.stop.max_measurements}) меньше минимального набора: '
                    f'{n_test} в области и {n_ctrl} контрольных', 'warning')

    plan = {
        'ok': not any(p['level'] == 'error' for p in problems),
        'problems': problems,
        'region': None if region is None else {'shape': region.shape, 'x': round(region.x, 3), 'y': round(region.y, 3),
                                              'r': round(region.r, 3), 'w': round(region.w, 3), 'h': round(region.h, 3)},
        'sites': sites,
        'route': {'meters': round(road_m, 2), 'energy': round(road_e, 1), 'seconds': round(road_s)},
        'costs': {'road': round(road_e, 1), 'measure_min': round(lo, 1), 'measure_max': round(hi, 1),
                  'reserve': reserve, 'budget': budget, 'seconds_min': round(road_s + meas_s)},
    }
    plan['steps'] = plan_steps(spec, plan)
    return plan


def plan_steps(spec, plan):
    """План исследования словами — то, что робот собирается делать, до первого замера."""
    q, a, s = spec.quantity, spec.allowed, plan['sites']
    out = ['Постоять у базы и измерить шум показаний батареи: от него зависит погрешность каждого замера.']
    t = s['test']
    ctrl = s['controls']
    if q == 'soil_cost' and t:
        if ctrl:
            c = ctrl[0]
            out.append(f"Контрольный участок {c['id']} — {c['length']:.2f} м обычного пола около "
                       f"({_mid(c)[0]:.1f}; {_mid(c)[1]:.1f}), "
                       + ('выбран мной у дороги к области: по ней я проеду и проверю пол колёсами.' if c.get('by') == 'auto'
                          else 'указан в задании.')
                       + f" Не меньше {a.straight.repeats} пробегов.")
        out.append(f"Мерный участок в области — {t['length']:.2f} м около ({_mid(t)[0]:.1f}; {_mid(t)[1]:.1f}), "
                   f"не меньше {a.straight.repeats} пробегов туда и обратно. Перед пробегом и после него стою и "
                   'усредняю показания батареи.')
        out.append('Дальше чередую область и контроль: каждый раз беру тот замер, который сильнее всего сужает '
                   'погрешность на единицу заряда.' if ctrl else
                   'Контрольного участка нет: сравниваю с номинальным расходом из условия.')
    elif q == 'turn_cost' and t:
        out.append(f"Развороты на месте в ({t['x']:.1f}; {t['y']:.1f}) на {a.spin.angle_deg:.0f}° попеременно в обе стороны, "
                   f"не меньше {a.spin.repeats}."
                   + (' Между ними паузы той же длительности: так расход на месте не попадёт в цену поворота.'
                      if a.pause else ' Пауз в задании нет: расход на месте беру из условия.'))
    elif q == 'idle_cost' and t:
        out.append(f"Паузы по {a.pause.seconds:.0f} с в ({t['x']:.1f}; {t['y']:.1f}), не меньше {a.pause.repeats}."
                   + (f" Для сравнения — такие же паузы в ({ctrl[0]['x']:.1f}; {ctrl[0]['y']:.1f})." if ctrl else ''))
    elif q == 'load_effect':
        out.append('Прямые пробеги по обычному полу с грузом; расход без груза известен только по номиналу из условия.')
    elif q == 'sensor_law':
        out.append('Найти образец по росту показаний датчика и встать рядом с ним, не собирая: он нужен как источник сигнала.')
        out.append(f"Отъезжать от образца на {a.straight.length_m:.1f} м, {2 * a.straight.length_m:.1f} м и дальше — по две точки "
                   'с разных сторон — и слушать датчик на месте. Каждый раз выбираю расстояние, которое лучше всего '
                   'различает объяснения на единицу заряда.')
        out.append('В конце вернуться к образцу и повторить первый замер: так видно, не сбился ли датчик за это время.')
    out.append(f"Остановиться, когда погрешность станет не больше ±{spec.stop.rel_error:.0%}"
               if q != 'sensor_law' else
               f"Остановиться, когда одно объяснение наберёт {spec.stop.confidence:.0%}")
    out[-1] += (f", либо кончится бюджет {spec.budget.energy:.0f} ед., либо наберётся {spec.stop.max_measurements} замеров, "
                f"либо пройдёт {spec.stop.time_s:.0f} с. Запас на возврат — дорога домой и ещё {spec.budget.reserve:.1f} ед.")
    return out


def describe(spec):
    """Задание простыми словами, одной-двумя фразами."""
    a = spec.allowed
    parts = []
    if a.straight:
        parts.append(f'прямые пробеги по {a.straight.length_m:.2f} м')
    if a.spin:
        parts.append(f'развороты на {a.spin.angle_deg:.0f}°')
    if a.pause:
        parts.append(f'паузы по {a.pause.seconds:.0f} с')
    r = spec.region
    where = ''
    if r is not None:
        where = (f' в области {r.zone}' if r.zone else
                 f' в круге радиусом {r.r:.2f} м с центром ({r.x:.1f}; {r.y:.1f})' if r.shape == 'circle' else
                 f' в прямоугольнике {r.w:.2f} × {r.h:.2f} м с центром ({r.x:.1f}; {r.y:.1f})')
    goal = (f'сравнить {len(spec.hypotheses) or len(DEFAULT_LAWS)} объяснений с уверенностью {spec.stop.confidence:.0%}'
            if spec.quantity == 'sensor_law' else f'точность ±{spec.stop.rel_error:.1%}')
    return (f'Выяснить, {spec.label}{where}. Разрешено: {", ".join(parts) or "ничего"}; контрольных участков — '
            f'не больше {spec.controls.max}. Цель: {goal}. Бюджет {spec.budget.energy:.0f} ед. заряда, запас на возврат '
            f'{spec.budget.reserve:.1f} ед., не больше {spec.stop.max_measurements} замеров и {spec.stop.time_s:.0f} с.')


def prepare(spec, arena, rules=None, scenario=None, carried=0):
    """Задание, готовое к исполнению: схема, проверка, план. Сценарий нужен только чтобы найти область по имени."""
    try:
        spec = load_spec(spec)
    except StudyError as e:
        return {'spec': None, 'raw': spec if isinstance(spec, dict) else None, 'carried': carried,
                'plan': {'ok': False, 'problems': [{'level': 'error', 'field': 'spec', 'text': t} for t in e.problems],
                         'region': None, 'sites': {'test': None, 'controls': []}, 'route': {}, 'costs': {}, 'steps': []}}
    return {'spec': spec, 'plan': check_study(spec, arena, rules, scenario, carried), 'carried': carried}


# =================================================================================================
# готовые задания
# =================================================================================================

PRESETS = [
    {'id': 'soil_b', 'title': 'Расход в области B', 'level': 'medium', 'seed': 3,
     'text': 'Исследовать расход в области B. Сравнивать прямолинейное движение. Разрешить максимум два '
             'контрольных участка. Обязательно сохранить запас на возврат.',
     'spec': {'title': 'Расход в области B', 'quantity': 'soil_cost', 'region': {'zone': 'B'},
              'allowed': {'straight': {'length_m': 0.5, 'repeats': 2}}, 'controls': {'max': 2},
              'stop': {'rel_error': 0.05, 'max_measurements': 12, 'time_s': 300},
              'budget': {'energy': 40, 'reserve': 3}}},
    {'id': 'soil_which', 'title': 'Какой это грунт?', 'level': 'medium', 'seed': 1001,
     'text': 'В области A пол дороже обычного в 2, в 3 или в 4 раза? Проверить прямыми пробегами и паузами, '
             'сравнить объяснения и назвать множитель с точностью ±2 %.',
     'spec': {'title': 'Какой грунт в области A', 'quantity': 'soil_cost', 'region': {'zone': 'A'},
              'allowed': {'straight': {'length_m': 0.5, 'repeats': 2}, 'pause': {'seconds': 3}}, 'controls': {'max': 1},
              'stop': {'rel_error': 0.02, 'max_measurements': 14, 'time_s': 300},
              'budget': {'energy': 35, 'reserve': 3},
              'hypotheses': [{'id': 'x1', 'statement': 'пол в области обычный', 'value': 1.0, 'sigma': 0.1},
                             {'id': 'x2', 'statement': 'пол вдвое дороже обычного', 'value': 2.0, 'sigma': 0.15},
                             {'id': 'x3', 'statement': 'пол втрое дороже обычного', 'value': 3.0, 'sigma': 0.2},
                             {'id': 'x4', 'statement': 'пол вчетверо дороже обычного', 'value': 4.0, 'sigma': 0.25}]}},
    {'id': 'turn', 'title': 'Сколько стоит поворот', 'level': 'medium', 'seed': 3,
     'text': 'Сколько заряда уходит на поворот? Полные обороты на месте в обе стороны, между ними паузы для '
             'сравнения. Точность ±5 %, далеко от базы не уезжать.',
     'spec': {'title': 'Цена поворота', 'quantity': 'turn_cost',
              'allowed': {'spin': {'angle_deg': 360, 'repeats': 2}, 'pause': {'seconds': 3}},
              'stop': {'rel_error': 0.05, 'max_measurements': 12, 'time_s': 200},
              'budget': {'energy': 10, 'reserve': 1}}},
    {'id': 'idle', 'title': 'Расход на месте, жёсткий лимит', 'level': 'medium', 'seed': 3,
     'text': 'Сколько заряда уходит, когда робот просто стоит? Только паузы по 6 с, точность ±5 %, не дольше '
             'минуты. (Здесь точности не хватит — и робот честно об этом скажет.)',
     'spec': {'title': 'Расход на месте', 'quantity': 'idle_cost', 'allowed': {'pause': {'seconds': 6, 'repeats': 3}},
              'controls': {'max': 0}, 'stop': {'rel_error': 0.05, 'max_measurements': 12, 'time_s': 60},
              'budget': {'energy': 5, 'reserve': 1}}},
    {'id': 'sensor', 'title': 'Закон датчика образцов', 'level': 'easy', 'seed': 3,
     'text': 'Проверить гипотезу: показание датчика образцов линейно убывает до нуля на 2 м. Альтернативы: '
             'дальность 1,5 м или 3 м, убывание быстрее или медленнее линейного.',
     'spec': {'title': 'Закон датчика образцов', 'quantity': 'sensor_law',
              'allowed': {'straight': {'length_m': 0.5}, 'pause': {'seconds': 2}},
              'stop': {'confidence': 0.9, 'max_measurements': 14, 'time_s': 400},
              'budget': {'energy': 45, 'reserve': 3}, 'hypotheses': DEFAULT_LAWS}},
]


def preset(pid):
    item = next((p for p in PRESETS if p['id'] == pid), None)
    if item is None:
        raise StudyError([f'нет готового задания «{pid}»: есть {", ".join(p["id"] for p in PRESETS)}'])
    return item


# =================================================================================================
# скрытая правда: сверка оценки робота со сценарием (роботу недоступна)
# =================================================================================================

def _soils_at(scenario, world, t):
    change = next((w['t'] for w in world if w['type'] == 'soil_change'), None)
    after = next((e['soils'] for e in scenario.events if e['type'] == 'soil_change'), None)
    return after if (change is not None and after is not None and t >= change) else scenario.soils


def _path_mult(soils, m, n=24):
    xs = np.linspace(m['x0'], m['x1'], n)
    ys = np.linspace(m['y0'], m['y1'], n)
    return float(np.mean([soil_mult(soils, x, y) for x, y in zip(xs, ys)]))


def add_truth(report, scenario, rules, world):
    """Дописать в отчёт настоящее значение из сценария и то, попало ли оно в заявленный интервал."""
    spec = report.get('spec') or {}
    q = spec.get('quantity')
    est = report.get('estimate')
    truth = None
    if q == 'soil_cost':
        legs = [m for m in report['measurements'] if m['role'] == 'test' and m['kind'] == 'straight' and m.get('used')]
        if legs:
            soils = _soils_at(scenario, world, legs[-1]['t'])          # как было при последнем учтённом замере
            w = [m['ds'] for m in legs]
            value = float(np.average([_path_mult(soils, m) for m in legs], weights=w))
            ctrl = [m for m in report['measurements'] if m['role'] == 'control' and m['kind'] == 'straight' and m.get('used')]
            base = float(np.average([_path_mult(_soils_at(scenario, world, m['t']), m) for m in ctrl],
                                    weights=[m['ds'] for m in ctrl])) if ctrl else 1.0
            # Истина — отношение к обычному полу, а не к тому, что робот принял за контроль: ошибка с контролем — его ошибка.
            truth = {'value': round(value, 4), 'unit': '× к обычному полу',
                     'text': f'по сценарию пол на мерном участке дороже обычного в {value:.2f} раза'
                             + (f'; контрольный участок на самом деле не обычный пол (×{base:.2f})' if abs(base - 1) > 0.01 else '')}
            changed = next((w0['t'] for w0 in world if w0['type'] == 'soil_change'), None)
            if changed is not None and legs[0]['t'] < changed < legs[-1]['t']:
                truth['text'] += f'; на {changed:.0f}-й секунде грунты в сценарии поменялись'
    elif q == 'turn_cost':
        truth = {'value': rules.drain_per_rad, 'unit': 'ед/рад', 'text': f'по правилам стенда поворот стоит {rules.drain_per_rad:g} ед/рад'}
    elif q == 'idle_cost':
        truth = {'value': rules.drain_idle_per_s, 'unit': 'ед/с', 'text': f'по правилам стенда на месте уходит {rules.drain_idle_per_s:g} ед/с'}
    elif q == 'load_effect':
        truth = {'value': rules.load_drain, 'unit': 'доля на образец',
                 'text': f'по правилам стенда каждый образец удорожает метр на {rules.load_drain:.0%}'}
    elif q == 'sensor_law':
        hyps = spec.get('hypotheses') or DEFAULT_LAWS
        right = next((x['id'] for x in hyps if x.get('law') == 'linear'
                      and abs((x.get('range_m') or 0) - rules.sensor_range_m) < 0.01), None)
        truth = {'value': rules.sensor_range_m, 'unit': 'м дальности', 'best': right,
                 'text': f'по правилам стенда показание линейно убывает до нуля на {rules.sensor_range_m:g} м'}
        c = (report.get('inquiry') or {}).get('conclusion')
        if c:
            truth['verdict'] = ('insufficient' if c['status'] != 'identified' else
                                'correct' if c['best'] == right else 'wrong')
    if truth is None:
        return report
    if est and est.get('value') is not None and truth['value']:
        truth['error_pct'] = round(abs(est['value'] - truth['value']) / abs(truth['value']) * 100, 2)
        truth['covered'] = bool(est['ci95'][0] <= truth['value'] <= est['ci95'][1])
    report['truth'] = truth
    return report


def study_metrics(report):
    """Метрики исследования для сводки опыта (did/metrics.py: METRICS)."""
    est, truth = report.get('estimate') or {}, report.get('truth') or {}
    used = [m for m in report.get('measurements', []) if m.get('used') and m['role'] in ('test', 'control')]
    return {
        'study_status': report.get('status'),
        'study_error_pct': truth.get('error_pct'),
        'study_covered': truth.get('covered'),
        'study_halfwidth_pct': round(est['rel_error'] * 100, 2) if est.get('rel_error') is not None else None,
        'study_reached': report.get('status') == 'done',
        'study_measurements': len(used),
        'study_energy': (report.get('energy') or {}).get('spent'),
        'study_time': report.get('time_s'),
    }


# =================================================================================================
# отчёт словами
# =================================================================================================

def _n(v, digits=2):
    return '—' if v is None else f'{v:.{digits}f}'.replace('.', ',')


def digits_for(sigma):
    """Сколько знаков после запятой показывать при такой погрешности: две значащие цифры погрешности."""
    if not sigma or sigma <= 0:
        return 2
    return int(min(5, max(0, 1 - math.floor(math.log10(sigma)))))


ROLE_RU = {'test': 'исследуемое', 'control': 'контроль', 'calibration': 'калибровка', 'check': 'проверка'}
KIND_RU = {'straight': 'пробег', 'pause': 'пауза', 'spin': 'разворот', 'listen': 'слушаю датчик'}


def to_markdown(report):
    """Отчёт в читаемом виде — для вставки в документ или на слайд."""
    spec = report.get('spec') or {}
    q = spec.get('quantity')
    title = spec.get('title') or (QUANTITIES[q][3] if q in QUANTITIES else 'Исследование')
    out = [f'# Отчёт об исследовании: {title}', '']
    if report.get('task'):
        out += [f"**Задание.** {report['task']}", '']
    est = report.get('estimate')
    if est and est.get('value') is not None:
        d = digits_for(est['sigma'])
        out += [f"**Итог.** {est['label']}: **{_n(est['value'], d)} ± {_n(1.96 * est['sigma'], d)} {est['unit']}** "
                f"(95% интервал от {_n(est['ci95'][0], d)} до {_n(est['ci95'][1], d)}; "
                f"погрешность ±{_n(est['rel_error'] * 100, 1)} %).", '']
    inq = report.get('inquiry')
    if inq and inq.get('conclusion'):
        out += [f"**Сравнение объяснений.** {inq['conclusion']['text']}.", '']
    if report.get('conclusion'):
        out += [f"**Вывод.** {report['conclusion']}", '']
    c = report.get('control')
    if c and c.get('text'):
        out += [f"**Сравнение с контролем.** {c['text']}", '']
    if report.get('stop'):
        out += [f"**Почему остановились.** {report['stop']['text']}", '']
    e = report.get('energy')
    if e:
        parts = ', '.join(f'{name} {_n(e[key], 1)}' for key, name in
                          (('travel', 'дорога'), ('test', 'замеры'), ('control', 'контроль и проверки'), ('home', 'возврат'))
                          if e.get(key))
        out += [f"**Цена.** Потрачено {_n(e['spent'], 1)} ед. заряда из бюджета {_n(e['budget'], 0)}"
                f"{' (' + parts + ')' if parts else ''}; время {_n(report.get('time_s'), 0)} с.", '']
    truth = report.get('truth')
    if truth:
        line = f"**Сверка со сценарием (роботу недоступна).** {truth['text'][0].upper() + truth['text'][1:]}"
        if truth.get('error_pct') is not None:
            line += (f"; ошибка оценки {_n(truth['error_pct'], 1)} %, настоящее значение "
                     f"{'попало' if truth['covered'] else 'не попало'} в заявленный интервал")
        out += [line + '.', '']
    if report.get('plan', {}).get('steps'):
        out += ['## План', ''] + [f'{i}. {s}' for i, s in enumerate(report['plan']['steps'], 1)] + ['']
    ms = report.get('measurements') or []
    if ms:
        out += ['## Замеры', '', '| № | t, с | что | где | путь, м | поворот, рад | время, с | расход, ед. | значение | ± (1σ) | учтён |',
                '|---|---|---|---|---|---|---|---|---|---|---|']
        for m in ms:
            d = digits_for(m.get('sigma'))
            value = '—' if m.get('value') is None else f"{_n(m['value'], d)} {m.get('unit', '')}"
            out.append(f"| {m['n']} | {_n(m['t'], 0)} | {KIND_RU.get(m['kind'], m['kind'])}, {ROLE_RU.get(m['role'], m['role'])} "
                       f"| {m.get('site') or '—'} ({_n(m['x'], 1)}; {_n(m['y'], 1)}) | {_n(m['ds'], 2)} | {_n(m['dth'], 2)} "
                       f"| {_n(m['dt'], 1)} | {_n(m['spent'], 3)} | {value} | {_n(m.get('sigma'), d)} "
                       f"| {'да' if m.get('used') else 'нет: ' + (m.get('note') or '—')} |")
        out.append('')
    if inq:
        out += ['## Цепочка рассуждений', '', 'Объяснения и их вероятности до и после опытов:', '']
        out += [f"- {a['statement']}: {a['prior']:.0%} → {a['posterior']:.0%}" for a in inq['alternatives']]
        done = [x for x in inq['tests'] if x.get('measured')]
        if done:
            out += ['', 'Опыты:', '']
            for x in done:
                pred = '; '.join(f"{k}: {_n(v['mean'], 2)}" for k, v in x['predictions'].items())
                out.append(f"- {x['name']} — ожидалось ({pred}), измерено {_n(x['measured']['value'], 3)} "
                           f"± {_n(x['measured']['sigma'], 3)} {x['unit']}")
        out.append('')
    if report.get('failures'):
        out += ['## Чего не удалось', ''] + [f'- {f}' for f in report['failures']] + ['']
    return '\n'.join(out).rstrip() + '\n'


# =================================================================================================
# запуск
# =================================================================================================

def spec_key(spec):
    """Короткий отпечаток задания: разные задания на одном сценарии пишутся в разные файлы."""
    text = json.dumps(spec if isinstance(spec, dict) else load_spec(spec).to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]


def run_study(spec, level='medium', seed=3, rules='science', experiment='study', save=True, **kw):
    """Провести исследование в быстром симуляторе. Возвращает сводку прогона: метрики, отчёт (study), файл записи."""
    from .runner import run_episode
    raw = spec if isinstance(spec, dict) else load_spec(spec).to_dict()
    arm = kw.pop('arm', None) or f"{raw.get('quantity', 'spec')}-{spec_key(raw)}"
    return run_episode(level, seed, 'study', experiment=experiment, arm=arm, rules=rules, study=raw, save=save, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', default=None, help='файл задания: JSON или YAML')
    ap.add_argument('--preset', default=None, help='готовое задание: ' + ', '.join(p['id'] for p in PRESETS))
    ap.add_argument('--level', default=None, choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rules', default='science', choices=['science', 'base'], help='base — правила без поворотов и шума')
    ap.add_argument('--check', action='store_true', help='только проверить задание и показать план')
    ap.add_argument('--json', action='store_true', help='вывести отчёт как JSON, а не текстом')
    args = ap.parse_args()
    try:
        if args.preset:
            item = preset(args.preset)
            spec, level, seed = item['spec'], args.level or item['level'], args.seed or item['seed']
        elif args.spec:
            spec, level, seed = load_spec(args.spec).to_dict(), args.level or 'medium', args.seed or 3
        else:
            ap.error('нужно --spec файл или --preset имя')
        if args.check:
            from .arena import load_arena
            from .scenario import generate
            arena = load_arena()
            rules = Rules(**(SCIENCE if args.rules == 'science' else {}))
            plan = check_study(spec, arena, rules, generate(level, seed, arena))
            print(describe(load_spec(spec)))
            for p in plan['problems']:
                print(('ОШИБКА: ' if p['level'] == 'error' else 'Замечание: ') + p['text'])
            print('\n'.join(f'{i}. {s}' for i, s in enumerate(plan['steps'], 1)))
            raise SystemExit(0 if plan['ok'] else 2)
    except StudyError as e:
        raise SystemExit('Задание не принято:\n- ' + '\n- '.join(e.problems)) from e
    summary = run_study(spec, level, seed, rules='science' if args.rules == 'science' else None)
    report = summary['study']
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str) if args.json else report['markdown'])
    if summary.get('file'):
        print(f"Запись прогона: runs/{summary['file']}  (в интерфейсе: #/run?file={summary['file']})")


if __name__ == '__main__':
    main()
