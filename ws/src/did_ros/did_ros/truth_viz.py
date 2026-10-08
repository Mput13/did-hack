"""Скрытая правда как набор плоских фигур: одни и те же фигуры идут в RViz и в окно Gazebo.

Фигуры лежат на полу слоями и не поднимаются выше 2 см: лидар Burger смотрит в плоскости
0,17 м и их не видит, коллизий у них нет.
"""
import hashlib

from visualization_msgs.msg import Marker, MarkerArray

PREFIX = 'did_truth_'            # имена моделей-подсказок в мире Gazebo
THICK = 0.004                    # толщина слоя, м
LAYERS = ('soil', 'base', 'hazard', 'sample')   # снизу вверх; верх последнего слоя — 0,018 м
SAMPLE_R = 0.06

BASE_RGB = (0.10, 0.60, 0.95)
HAZARD_RGB = (0.90, 0.08, 0.08)
SAMPLE_RGB = (1.00, 0.85, 0.00)
SOIL_LIGHT = (0.96, 0.80, 0.45)  # множитель 1,5 и меньше
SOIL_DARK = (0.40, 0.18, 0.06)   # множитель 4 и больше


def soil_color(mult):
    k = min(1.0, max(0.0, (mult - 1.5) / 2.5))
    return tuple(a + (b - a) * k for a, b in zip(SOIL_LIGHT, SOIL_DARK))


def _shape(kind, ident, form, x, y, rgb, alpha, r=0.0, w=0.0, h=0.0, label=''):
    s = {'kind': kind, 'form': form, 'x': float(x), 'y': float(y),
         'z': THICK * (LAYERS.index(kind) + 1), 'r': float(r), 'w': float(w), 'h': float(h),
         'rgb': tuple(round(c, 3) for c in rgb), 'alpha': alpha, 'label': label}
    # Имя зависит от геометрии и цвета: изменившаяся зона — это другая модель, старая удаляется.
    digest = hashlib.md5(repr(sorted(s.items())).encode()).hexdigest()[:6]
    s['name'] = f'{PREFIX}{kind}_{ident}_{digest}'
    return s


def _zone(kind, z, rgb, alpha, label):
    return _shape(kind, z['id'], z['shape'], z['x'], z['y'], rgb, alpha,
                  r=z['r'], w=z['w'], h=z['h'], label=label)


def shapes(truth):
    """Фигуры по словарю правды — тому же, что уходит в /did/truth."""
    out = [_zone('soil', z, soil_color(z['mult']), 0.6, f'x{z["mult"]:g}') for z in truth['soils']]
    bx, by = truth['scenario']['base']
    out.append(_shape('base', '0', 'circle', bx, by, BASE_RGB, 0.5,
                      r=truth['rules']['base_radius_m']))
    out += [_zone('hazard', z, HAZARD_RGB, 0.75, '') for z in truth['hazards']]
    out += [_shape('sample', s['id'], 'circle', s['x'], s['y'], SAMPLE_RGB, 1.0, r=SAMPLE_R,
                   label=str(s['id'])) for s in truth['remaining']]
    return out


def markers(items, frame='map'):
    """MarkerArray для RViz. Первым идёт DELETEALL: собранное и переехавшее исчезает."""
    wipe = Marker()
    wipe.header.frame_id = frame
    wipe.action = Marker.DELETEALL
    arr = MarkerArray(markers=[wipe])
    for i, s in enumerate(items):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id = s['kind'], i
        m.type = Marker.CYLINDER if s['form'] == 'circle' else Marker.CUBE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = s['x'], s['y'], s['z']
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y = (2 * s['r'], 2 * s['r']) if s['form'] == 'circle' else (s['w'], s['h'])
        m.scale.z = THICK
        m.color.r, m.color.g, m.color.b = s['rgb']
        m.color.a = s['alpha']
        arr.markers.append(m)
        if s['label']:
            t = Marker()
            t.header.frame_id = frame
            t.ns, t.id = s['kind'] + '_label', i
            t.type = Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = s['x'], s['y'], 0.10
            t.pose.orientation.w = 1.0
            t.scale.z = 0.14
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = s['label']
            arr.markers.append(t)
    return arr


def sdf(s):
    """Статическая модель Gazebo из одного visual: без коллизий, без теней."""
    if s['form'] == 'circle':
        geom = f'<cylinder><radius>{s["r"]:.3f}</radius><length>{THICK}</length></cylinder>'
    else:
        geom = f'<box><size>{s["w"]:.3f} {s["h"]:.3f} {THICK}</size></box>'
    rgb = ' '.join(f'{c:.3f}' for c in s['rgb'])
    glow = ' '.join(f'{c * 0.5:.3f}' for c in s['rgb'])   # видно и в тени стен
    return (
        f'<sdf version="1.8"><model name="{s["name"]}"><static>true</static><link name="link">'
        f'<visual name="visual"><cast_shadows>false</cast_shadows>'
        f'<transparency>{1.0 - s["alpha"]:.2f}</transparency><geometry>{geom}</geometry>'
        f'<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse>'
        f'<specular>0 0 0 1</specular><emissive>{glow} 1</emissive></material>'
        f'</visual></link></model></sdf>')
