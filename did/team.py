"""Координация двух роботов (исследование M1): канал сообщений и агент, который им пользуется.

Роботы не делят память. Всё, что один знает о другом, пришло сообщением по общему каналу
(в быстром симуляторе — TeamChannel, в ROS — топик /did/team, одна строка JSON на сообщение).
Канал записывается в прогон целиком: видно, что именно роботы сказали друг другу.

Виды сообщений (поля сверх общих t, from, type):
  hello      {x, y, home}                 я в сети, вот моё место на базе
  obs        {x, y, th, battery, state, claim, readings, soil, safe} — раз в секунду:
               readings [[x, y, z, sigma, t], ...]  показания моего датчика образцов: напарник вносит их в свою
                                                 карту вероятностей так же, как свои;
               soil     [[x, y, ds, mult], ...]  отрезки пути и во сколько раз на них расход выше обычного;
               safe     [[x, y], ...]            где я проехал без штрафа (сужает гипотезы об опасной зоне);
               claim    {kind, x, y, cost}       моя текущая цель и сколько заряда до неё осталось
  claim      {kind, x, y, cost}           «эту цель взял я» — сразу, как только выбрал
  collected  {x, y, left}                 образец собран, на арене осталось left
  hazard     {x, y, heading, trail}       я получил штраф опасной зоны: точка, курс, подход
  soil       {x, y, mult}                 гипотеза: здесь дорогой грунт
  soil_changed {x, y}                     расход перестал сходиться с оценкой: грунты изменились
  sector     {line: {c, n}, owners} | {all}, reason — деление арены на участки или «вся арена моя»
  status     {state, battery}             returning — еду на базу, finished — на базе, stopped — встал

Что координация меняет в поведении:
  1. общая карта вероятностей: каждое показание одного сужает карту обоих;
  2. цель берёт тот, кому до неё дешевле; занятую цель и разведку рядом с ней второй не выбирает;
  3. сообщение о сборе — напарник пересчитывает карту и план, вдвоём знают, сколько образцов осталось;
  4. штраф одного — опасная зона на карте обоих; дорогой грунт и его смена — тоже;
  5. арена поделена на участки: разведку вне своего участка робот берёт неохотно; участки
     пересматриваются, когда напарник уехал на базу или встал и когда свой участок проверен;
  6. роботы знают положение друг друга: путь строится в обход напарника, младший уступает дорогу.
Каждое из шести можно выключить по отдельности (TeamConfig) — так меряется вклад.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

from .agent import Agent
from .nav import LIDAR_OFFSET, CostGraph


@dataclass(frozen=True)
class TeamConfig:
    share_readings: bool = True       # 1: обмен показаниями датчика образцов
    claims: bool = True               # 2: «эту взял я» и уступка тому, кому дешевле
    share_collected: bool = True      # 3: сообщение о собранном образце
    share_hazards: bool = True        # 4: опасные зоны
    share_soil: bool = True           # 4: дорогой грунт и его смена
    sectors: bool = True              # 5: деление арены на участки
    avoid: bool = True                # 6: объезд напарника и уступка дороги по его сообщениям о положении
    lidar_avoid: bool = True          # то же по лидару, без связи: помеха, которой нет на карте, — другой робот
    out_of_sector: float = 0.25       # вес разведки на чужом участке (1 — участки не действуют)
    explore_r: float = 1.0            # м: разведка ближе этого к цели напарника — та же работа
    steal: float = 1.5                # ед. заряда: перехватить занятую цель можно, только если до неё настолько дешевле
    standby: bool = True              # все цели заняты напарником — ждать, а не уезжать на базу
    # Проверено на сценариях 1–80 и выигрыша не дало, поэтому выключено: последний образец проверяет
    # напарник — встать сбоку от цели и мерить с другой стороны.
    assist: bool = False
    assist_m: float = 0.8             # на каком расстоянии от цели напарника встаёт помощник
    obs_period_s: float = 1.0

    def to_dict(self):
        return asdict(self)


class TeamChannel:
    """Общий канал: каждый пишет, каждый читает чужое. Сообщение — словарь, сериализуемый в JSON."""

    def __init__(self):
        self.log = []
        self._cursor = {}

    def post(self, sender, t, kind, /, **body):
        msg = {'t': round(float(t), 2), 'from': sender, 'type': kind, **body}
        self.log.append(msg)
        return msg

    def read(self, reader):
        """Сообщения остальных, пришедшие после прошлого чтения."""
        start = self._cursor.get(reader, 0)
        self._cursor[reader] = len(self.log)
        return [m for m in self.log[start:] if m['from'] != reader]


class _TeamGraph(CostGraph):
    """Граф путей, в котором клетки рядом с напарником дороги: путь его объезжает."""

    def __init__(self, arena):
        self.extra = None
        self._args = (None, None, None)
        super().__init__(arena)

    def set_cost(self, mult_grid=None, forbidden=None, bias=None):
        self._args = (mult_grid, forbidden, bias)
        if self.extra is not None:
            bias = self.extra if bias is None else np.asarray(bias) * self.extra
        super().set_cost(mult_grid, forbidden, bias)

    def refresh(self):
        self.set_cost(*self._args)


class TeamAgent(Agent):

    CLAIM_R = 0.6          # м: кандидат ближе этого к цели напарника — та же цель (образцы не ближе 0,9 м друг к другу)
    SILENT_S = 5.0         # с: напарник молчит дольше — считаем, что его нет
    SECTOR_MIN_S = 20.0    # с: участки не пересматриваются чаще
    NEAR_M = 0.55          # м: ближе — младший по имени уступает дорогу
    AVOID_R = (0.25, 0.45)  # м: ближе первого путь мимо напарника почти запрещён, дальше второго — как обычно
    AVOID_W = 40.0
    RETHINK_S = 3.0        # с: не чаще пересматривать план из-за смены цели напарника
    YIELD_MAX_S = 6.0      # с: дольше не ждём — едем медленно в объезд
    SPOT_CLEAR = 0.15      # м: отражение лидара дальше этого от стен карты — не стена
    SPOT_KEEP_S = 3.0      # с: столько помним робота, замеченного лидаром

    def __init__(self, arena, config, n_samples, *, name, home, channel, team=None, **kw):
        super().__init__(arena, config, n_samples, **kw)
        self.name = name
        self.base = tuple(home)
        self.channel = channel
        self.team = team or TeamConfig()
        self.partners = {}                 # имя -> что известно из его сообщений
        self.sector = None                 # действующее деление арены (тело сообщения sector)
        self.left = n_samples              # сколько образцов осталось на арене по сведениям команды
        self.graph, self.home_graph = _TeamGraph(arena), _TeamGraph(arena)
        self._pose = None
        self._battery = self.rules.battery_start
        self._hello = False
        self._bye = False
        self._said_returning = False
        self._claim = None                 # моя объявленная цель
        self._obs_t = -1e9
        self._out = {'readings': [], 'soil': [], 'safe': []}
        self._sector_t = -1e9
        self._heard_soil = []
        self._avoid_at = None
        self._yield_t = None
        self._seen = None                  # (x, y, t): робот, замеченный лидаром
        self._collect_t = -1e9             # время последнего известного мне сбора (моего или напарника)
        self._standby = False              # жду, пока напарник проверит последнюю цель
        self._rethink_t = -1e9
        self._explore_r = self.team.explore_r   # на каком расстоянии от цели напарника я сейчас не разведываю
        self._t_now = 0.0
        soil_observe = self.soil.observe

        def observe(x0, y0, x1, y1, spent, dt):
            out = soil_observe(x0, y0, x1, y1, spent, dt)
            self._out['soil'].append([round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2),
                                      round(math.hypot(x1 - x0, y1 - y0), 3), round(out[0], 2)])
            return out
        self.soil.observe = observe
        explore_points = self.belief.explore_points

        def explore(*a, **k):
            pts = explore_points(*a, **k)
            if self.team.claims:               # рядом с целью напарника разведывать нечего: он там и так всё измерит
                taken = [p['claim'] for p in self._active_partners() if p['claim']]
                for radius in (self.team.explore_r, self.CLAIM_R):      # всё занято — подойти ближе, но не к самой цели
                    free = [q for q in pts if all(math.hypot(q['x'] - c['x'], q['y'] - c['y']) >= radius
                                                  for c in taken)]
                    if free:
                        break
                pts, self._explore_r = free, radius
            if self.team.sectors and self.sector:
                for q in pts:
                    if not self._mine(q['x'], q['y']):
                        q['mass'] *= self.team.out_of_sector
            return pts
        self.belief.explore_points = explore

    # ======================================================================================
    # приём
    # ======================================================================================

    def _perceive(self, obs):
        self._pose = (obs.x, obs.y, obs.th)
        self._battery = obs.battery
        if not self._hello:
            self._hello = True
            self._say(obs.t, 'hello', x=round(obs.x, 2), y=round(obs.y, 2), home=list(self.base))
        self._receive(obs)
        super()._perceive(obs)
        if self.team.lidar_avoid and obs.scan is not None:
            self._spot(obs.t)

    def _receive(self, obs):
        for m in self.channel.read(self.name) if self.channel is not None else ():
            p = self.partners.setdefault(m['from'], {'name': m['from'], 'state': 'active', 'claim': None})
            p['t'] = obs.t
            handler = getattr(self, '_rx_' + m['type'], None)
            if handler:
                handler(m, p, obs)
        for p in self.partners.values():
            if p['state'] in ('active', 'returning') and obs.t - p['t'] > self.SILENT_S:
                p['state'] = 'silent'
                p['claim'] = None
                self.journal.add(obs.t, 'alarm', f"Напарник {p['name']} молчит {self.SILENT_S:.0f} с: считаю, что "
                                 'его нет, и беру всю арену', tag='partner_silent')
                self._take_all(obs.t, f"{p['name']} не отвечает")

    def _rx_hello(self, m, p, obs):
        p.update(x=m['x'], y=m['y'], home=tuple(m['home']))
        self.journal.add(obs.t, 'observe', f"На связи напарник {p['name']}: стоит в ({m['x']:.1f}; {m['y']:.1f})")
        if self.team.sectors and self.sector is None and self.name < p['name']:
            self._divide(obs.t, 'начало работы')

    def _rx_obs(self, m, p, obs):
        moved = 'x' not in p or math.hypot(m['x'] - p['x'], m['y'] - p['y']) > 0.02
        p.update(x=m['x'], y=m['y'], th=m['th'], battery=m['battery'], moving=moved)
        self._set_state(p, m['state'], obs.t)
        if self.team.share_readings:
            for x, y, z, sigma, t in m['readings']:
                # Показание снято до сбора, о котором я уже знаю: оно ещё «слышит» собранный образец.
                if t > self._collect_t:
                    self.belief.update(x, y, z, sigma)
        if self.team.share_soil and self.cfg.learn_soil and m['soil']:
            for x, y, ds, mult in m['soil']:
                self.soil.add(x, y, ds, mult)
            self._cost_dirty = True
        if self.team.share_hazards and self.hazard_map.zones:
            for x, y in m['safe']:
                self.hazard_map.safe(x, y)
            if self.hazard_map.version != self._risk_version:
                self._sync_hazards(obs.t)
        self._rx_claim(m['claim'] or {'kind': None}, p, obs)
        self._avoid_update(obs.t)

    def _rx_claim(self, m, p, obs):
        sg = self.queue[0] if self.queue else None
        helping = sg.get('assist') if sg else None
        if not m.get('kind'):
            p['claim'] = None
            if helping:
                self._rethink(obs.t)                         # помогать больше некому
            return
        fresh = p['claim'] is None or math.hypot(m['x'] - p['claim']['x'], m['y'] - p['claim']['y']) > 0.5
        p['claim'] = {'kind': m['kind'], 'x': m['x'], 'y': m['y'], 'cost': m['cost']}
        if helping and math.hypot(m['x'] - helping[0], m['y'] - helping[1]) > 0.5:
            self._rethink(obs.t)                             # цель, которой я помогал, сменилась
        elif fresh and m['kind'] == 'investigate' and self.team.claims and not (sg and sg['type'] == 'investigate'):
            self._rethink(obs.t)                             # новая цель напарника: вдруг мне до неё дешевле
        mine = self._target()
        if not (self.team.claims and mine) or self._returning:
            return
        d = math.hypot(mine['x'] - m['x'], mine['y'] - m['y'])
        both = mine['kind'] == 'investigate' and m['kind'] == 'investigate'
        if not (d < self.CLAIM_R if both else d < self._explore_r):
            return
        if mine['kind'] == 'investigate' and not both:
            return                              # я еду за образцом, он — на разведку рядом: уступает он
        cost = self._claim['cost'] if self._claim else self._cost_left(mine)
        if m['kind'] == 'investigate' and not both or (m['cost'], p['name']) < (cost, self.name):
            if fresh or self._trigger != 'partner_claim':
                self.journal.add(obs.t, 'decision', f"Цель ({mine['x']:.1f}; {mine['y']:.1f}) взял {p['name']}: ему до неё "
                                 f"{m['cost']:.1f} ед., мне {cost:.1f}. Уступаю и выбираю другую", tag='yield')
            self.journal.close(obs.t, self.queue[0].get('key'), 'outdated', f"цель уступлена напарнику {p['name']}")
            self.queue = []
            self._path_goal = None
            self._request_plan('partner_claim')

    def _rethink(self, t):
        """Пересмотреть план из-за смены цели напарника — не чаще раза в RETHINK_S: его цель уточняется
        с каждым показанием, и без этого предела оба робота только и делали бы, что перепланировали."""
        if not self._returning and t - self._rethink_t >= self.RETHINK_S:
            self._rethink_t = t
            self._request_plan('partner_claim')

    def _rx_collected(self, m, p, obs):
        if not self.team.share_collected:
            return
        self.belief.collected(m['x'], m['y'])
        self._collect_t = obs.t
        p['claim'] = None                       # свою цель он только что закрыл; новую объявит сам
        self.n_samples -= 1                     # на мою долю осталось меньше: «все собраны» — это про команду
        self.left = m['left']
        self.journal.add(obs.t, 'observe', f"{p['name']} сообщил о сборе образца в ({m['x']:.2f}; {m['y']:.2f}): "
                         f"на арене осталось {m['left']}. Пересчитываю карту и план", tag='partner_collected')
        sg = self.queue[0] if self.queue else None
        if sg and sg['type'] == 'investigate' and math.hypot(sg['x'] - m['x'], sg['y'] - m['y']) <= self.CLAIM_R:
            self.journal.close(obs.t, sg.get('key'), 'outdated', f"образец забрал {p['name']}")
            self.queue.pop(0)
            self._path_goal = None
        self._request_plan('partner_collected')

    def _rx_hazard(self, m, p, obs):
        if not (self.team.share_hazards and self.cfg.avoid_hazards):
            return
        known = len(self.hazard_map.zones)
        self.hazard_map.hit(m['x'], m['y'], m['heading'], [tuple(q) for q in m['trail']])
        self.journal.add(obs.t, 'alarm', f"{p['name']} получил штраф опасной зоны в ({m['x']:.2f}; {m['y']:.2f}): "
                         'вношу зону в свою карту и объезжаю', tag='partner_hazard')
        self._sync_hazards(obs.t, new=len(self.hazard_map.zones) > known)
        self._path_goal = None
        self._request_plan('partner_hazard')

    def _rx_soil(self, m, p, obs):
        self._heard_soil.append((m['x'], m['y']))
        self.journal.add(obs.t, 'observe', f"{p['name']} сообщил: около ({m['x']:.1f}; {m['y']:.1f}) грунт дороже "
                         f"обычного примерно в {m['mult']:.1f} раза", tag='partner_soil')

    def _rx_soil_changed(self, m, p, obs):
        if not (self.team.share_soil and self.cfg.learn_soil):
            return
        self.soil.forget(m['x'], m['y'], radius=0.7, keep=0.1, keep_elsewhere=0.6)
        self._cost_dirty = True
        self._cost_t = -1e9
        self.journal.add(obs.t, 'alarm', f"{p['name']} сообщил: около ({m['x']:.1f}; {m['y']:.1f}) расход перестал "
                         'сходиться с оценкой. Старым данным о грунте там больше не верю', tag='partner_soil_changed')
        self._request_plan('partner_soil_changed')

    def _rx_sector(self, m, p, obs):
        self.sector = {k: m[k] for k in ('line', 'owners', 'all', 'reason') if k in m}
        self._sector_t = obs.t
        self.journal.add(obs.t, 'decision', f"{p['name']} предложил деление арены ({m['reason']}): "
                         + self._sector_text(), tag='sector')
        if not self._returning:
            self._request_plan('sector')

    def _rx_status(self, m, p, obs):
        p['battery'] = m['battery']
        self._set_state(p, m['state'], obs.t)

    def _set_state(self, p, state, t):
        if p['state'] == state:
            return
        p['state'] = state
        if state == 'active':
            return
        p['claim'] = None
        word = {'returning': 'едет на базу', 'finished': 'закончил на базе', 'stopped': 'встал'}.get(state, state)
        self.journal.add(t, 'observe', f"Напарник {p['name']} {word}", tag='partner_' + state)
        self._take_all(t, f"{p['name']} {word}")

    # ======================================================================================
    # участки
    # ======================================================================================

    def _active_partners(self):
        return [p for p in self.partners.values() if p['state'] == 'active']

    def _mine(self, x, y):
        s = self.sector
        if not s:
            return True
        if 'all' in s:
            return s['all'] == self.name
        (cx, cy), (nx, ny) = s['line']['c'], s['line']['n']
        return s['owners']['pos' if (x - cx) * nx + (y - cy) * ny >= 0.0 else 'neg'] == self.name

    def _sector_text(self):
        s = self.sector
        if 'all' in s:
            return f"вся арена — {s['all']}"
        (cx, cy), (nx, ny) = s['line']['c'], s['line']['n']
        return (f"граница через ({cx:.1f}; {cy:.1f}), {s['owners']['neg']} — сторона "
                f"{_side_name(-nx, -ny)}, {s['owners']['pos']} — {_side_name(nx, ny)}")

    def _divide(self, t, reason):
        """Поделить арену пополам по оставшейся вероятности образцов: граница — поперёк линии между роботами."""
        others = [p for p in self._active_partners() if 'x' in p]
        if not (self.team.sectors and others and self._pose):
            return
        p = others[0]
        ax, ay = p['x'] - self._pose[0], p['y'] - self._pose[1]
        if math.hypot(ax, ay) < 0.5 and 'home' in p:        # стоят рядом: берём линию между местами на базе
            ax, ay = p['home'][0] - self.base[0], p['home'][1] - self.base[1]
        norm = math.hypot(ax, ay) or 1.0
        ax, ay = ax / norm, ay / norm
        s = self.belief.cx * ax + self.belief.cy * ay
        order = np.argsort(s)
        half = np.searchsorted(np.cumsum(self.belief.p[order]), 0.5 * self.belief.p.sum())
        cut = float(s[order[min(half, len(order) - 1)]])
        self.sector = {'line': {'c': [round(cut * ax, 2), round(cut * ay, 2)], 'n': [round(ax, 3), round(ay, 3)]},
                       'owners': {'neg': self.name, 'pos': p['name']}, 'reason': reason}
        self._sector_t = t
        self._say(t, 'sector', **self.sector)
        self.journal.add(t, 'decision', f'Делю арену ({reason}): ' + self._sector_text(), tag='sector')
        self._request_plan('sector')

    def _take_all(self, t, reason):
        if not self.team.sectors or self._returning or self.finished or self._active_partners():
            return
        if self.sector and self.sector.get('all') == self.name:
            return
        self.sector = {'all': self.name, 'reason': reason}
        self._sector_t = t
        self._say(t, 'sector', **self.sector)
        self.journal.add(t, 'decision', f'Пересматриваю участки ({reason}): вся арена теперь моя', tag='sector')
        self._request_plan('sector')

    def _review_sector(self, t):
        """Свой участок проверен, а на чужом ещё много неясного — поделить то, что осталось, заново."""
        s = self.sector
        if not (self.team.sectors and s and 'line' in s) or t - self._sector_t < self.SECTOR_MIN_S:
            return
        (cx, cy), (nx, ny) = s['line']['c'], s['line']['n']
        pos = (self.belief.cx - cx) * nx + (self.belief.cy - cy) * ny >= 0.0
        mine = pos if s['owners']['pos'] == self.name else ~pos
        m_mine, m_other = float(self.belief.p[mine].sum()), float(self.belief.p[~mine].sum())
        if m_mine < 0.25 and m_other > 3.0 * max(m_mine, 0.1):
            self._divide(t, f'мой участок проверен, на участке напарника осталось {m_other:.1f} образца по карте')

    # ======================================================================================
    # выбор целей
    # ======================================================================================

    def _state(self, obs, trigger):
        self._review_sector(obs.t)
        st = super()._state(obs, trigger)
        if not self.team.claims:
            return st
        for p in self._active_partners():
            cl = p['claim']
            if not cl:
                continue
            for c in st['candidates']:
                near = math.hypot(c['x'] - cl['x'], c['y'] - cl['y']) < self.CLAIM_R
                if near and cl['kind'] == 'investigate' and not c['cost_to'] < cl['cost'] - self.team.steal:
                    c['feasible'] = False
                    c['claimed_by'] = p['name']
        self._assist(obs, st)
        return st

    def _assist(self, obs, st):
        """Искать больше нечего — все оставшиеся образцы уже проверяет напарник. Тогда помочь ему: датчик
        даёт только расстояние, и показания с одной стороны оставляют на карте кольцо. Второй робот
        встаёт сбоку от цели, и два кольца пересекаются в точке."""
        claims = [(p, p['claim']) for p in self._active_partners()
                  if p['claim'] and p['claim']['kind'] == 'investigate' and 'x' in p]
        if not (self.team.assist and claims and self.left <= len(claims)) or any(c['feasible'] for c in st['candidates']):
            return
        p, cl = claims[0]
        ux, uy = cl['x'] - p['x'], cl['y'] - p['y']
        if math.hypot(ux, uy) < 0.3:                       # напарник уже на месте: меряем поперёк линии «я — цель»
            ux, uy = cl['x'] - obs.x, cl['y'] - obs.y
        norm = math.hypot(ux, uy) or 1.0
        ux, uy = ux / norm, uy / norm
        r = self.team.assist_m
        spots = [(cl['x'] - uy * r, cl['y'] + ux * r), (cl['x'] + uy * r, cl['y'] - ux * r)]
        spots = [q for q in spots if self.arena.clearance(*q) >= 0.2]
        if not spots:
            return
        x, y = min(spots, key=lambda q: math.hypot(q[0] - obs.x, q[1] - obs.y))
        if math.hypot(x - obs.x, y - obs.y) < 0.25:
            st['explore_points'] = []                      # уже на месте: стоять и мерить (см. _apply_plan)
            st['assist'] = {'at': True, 'partner': p['name'], 'x': cl['x'], 'y': cl['y']}
            return
        dist, pred = self.graph.field(obs.x, obs.y)
        to = self.graph.energy(dist, pred, x, y) * self._per_m()
        back = self._home_cost(x, y)
        ix, iy = self.arena.w2g(x, y)
        ok = bool(self._risk[iy, ix] < 0.5 and math.isfinite(to) and self._affordable(obs.battery, to, back))
        st['explore_points'] = [{'id': 'E1', 'x': round(x, 2), 'y': round(y, 2), 'unseen_share': 1.0,
                                 'cost_to': round(to, 1), 'cost_back': round(back, 1), 'feasible': ok}]
        st['assist'] = {'at': False, 'partner': p['name'], 'x': cl['x'], 'y': cl['y']}

    def _apply_plan(self, obs, plan, state, trigger):
        """Образцы ещё есть, но все цели заняты напарником: не уезжать на базу, а ждать на месте.

        Возврат у агента необратим. Уехавший раньше времени робот оставляет напарника одного искать
        последние образцы, поэтому «делать нечего» из-за занятых целей — это пауза, а не конец работы.
        """
        first = (plan.get('subgoals') or [{}])[0].get('type')
        taken = [c for c in state['candidates'] if c.get('claimed_by')]
        helping = state.get('assist')
        if helping and first == 'explore' and plan.get('source') != 'llm':
            plan = {**plan, 'reasoning': f"Искать больше нечего: оставшийся образец около ({helping['x']:.1f}; "
                    f"{helping['y']:.1f}) проверяет {helping['partner']}. Еду мерить с другой стороны — датчик даёт "
                    'только расстояние, с двух сторон место определяется точнее'}
            super()._apply_plan(obs, plan, state, trigger)
            if self.queue:
                self.queue[0]['assist'] = (helping['x'], helping['y'])
            self._standby = False
            return
        wait = self.team.standby and first == 'return_base' and taken and self.left > 0
        if (wait or helping and helping['at'] and first == 'return_base') and plan.get('source') != 'llm':
            self._trigger = None
            self.queue = []
            self._wait_until = obs.t + 2.0
            if not self._standby:
                self._standby = True
                who = helping or {'x': taken[0]['x'], 'y': taken[0]['y'], 'partner': taken[0]['claimed_by']}
                text = (f"Свободных целей нет: кандидата в ({who['x']:.1f}; {who['y']:.1f}) проверяет "
                        f"{who['partner']}. Образцов осталось {self.left} — стою и меряю, на базу не еду")
                self.journal.add(obs.t, 'decision', text, tag='standby')
                if self.rec:
                    self.rec.add_plan(obs.t, 'rule', trigger, text, [])
            return
        self._standby = False
        super()._apply_plan(obs, plan, state, trigger)

    def _target(self):
        sg = self.queue[0] if self.queue and not self._returning else None
        if not sg or sg['type'] not in ('investigate', 'explore', 'goto') or sg.get('assist'):
            return None                         # помощник цель не занимает: он едет к чужой
        return {'kind': 'investigate' if sg['type'] == 'investigate' else 'explore', 'x': sg['x'], 'y': sg['y']}

    def _cost_left(self, target):
        """Сколько заряда осталось до цели: по построенному пути, а пока его нет — по прямой."""
        x, y, _ = self._pose
        goal = self._path_goal
        if goal is not None and math.dist(goal, (target['x'], target['y'])) < 0.1 and self.follower.active:
            return self.follower.remaining(x, y) * self.rules.drain_per_m
        return math.hypot(target['x'] - x, target['y'] - y) * self.rules.drain_per_m

    # ======================================================================================
    # передача
    # ======================================================================================

    def _say(self, t, kind, /, **body):
        if self.channel is not None:           # без канала (режим без связи) робот молчит
            self.channel.post(self.name, t, kind, **body)

    def _on_reading(self, z, obs):
        super()._on_reading(z, obs)
        sigma = self.health.effective_sigma() if self.cfg.sensor_health else self.rules.sensor_sigma
        self._out['readings'].append([round(obs.x, 3), round(obs.y, 3), round(float(z), 3), round(float(sigma), 3),
                                      round(obs.t, 2)])

    def _trail_point(self, x, y, t):
        super()._trail_point(x, y, t)
        if self.cfg.avoid_hazards and t > self._trail_pause:
            self._out['safe'].append([round(x, 2), round(y, 2)])

    def _on_event(self, ev, obs):
        trail = [[round(x, 2), round(y, 2)] for x, y in self._trail]
        heading = obs.th + (math.pi if self._last_cmd[0] < 0 else 0.0)
        before = self.collected
        super()._on_event(ev, obs)
        if ev.get('type') == 'hazard_hit' and self.cfg.avoid_hazards and self.team.share_hazards:
            self._say(obs.t, 'hazard', x=round(obs.x, 2), y=round(obs.y, 2), heading=round(heading, 3), trail=trail)
        if self.collected > before:
            self._said_collected(obs)

    def _try_collect(self, sg, obs, io, confidence):
        before = self.collected
        super()._try_collect(sg, obs, io, confidence)
        if self.collected > before:
            self._said_collected(obs)

    def _said_collected(self, obs):
        self.left = max(0, self.left - 1)
        self._collect_t = obs.t
        if self.team.share_collected:
            self._flush(obs.t)                  # мои показания до сбора — раньше сообщения о сборе
            self._say(obs.t, 'collected', x=round(obs.x, 2), y=round(obs.y, 2), left=self.left)

    def _soil_hypotheses(self, t):
        known = len(self._soil_h)
        super()._soil_hypotheses(t)
        if not self.team.share_soil:
            return
        for h in self._soil_h[known:]:
            if any(math.hypot(h['x'] - x, h['y'] - y) <= 0.6 for x, y in self._heard_soil):
                continue                        # об этом участке напарник уже сказал
            data = (self.journal.get(h['key']) or {}).get('data', {})
            self._say(t, 'soil', x=h['x'], y=h['y'], mult=data.get('mult', 0.0))

    def _on_model_mismatch(self, verdict, x, y, ratio, predicted, t):
        if self.team.share_soil:
            self._flush(t)                      # сначала отрезки, по которым поднята тревога, потом сама тревога
        super()._on_model_mismatch(verdict, x, y, ratio, predicted, t)
        if self.team.share_soil:
            self._say(t, 'soil_changed', x=round(x, 2), y=round(y, 2))

    def _go_home(self, t, reason):
        super()._go_home(t, reason)
        self._say_returning(t)

    def _say_returning(self, t):
        if not self._said_returning:
            self._said_returning = True
            self._claim = None
            self._say(t, 'status', state='returning', battery=round(self._battery, 1))

    def _record(self, obs):
        super()._record(obs)
        t = obs.t
        if obs.done or self.finished:
            if not self._bye:
                self._bye = True
                self._flush(t)
                self._say(t, 'status', state='finished' if self.finished else 'stopped', battery=round(obs.battery, 1))
            return
        if self._pose is None:
            return
        if self._returning:
            self._say_returning(t)
        target = self._target()
        if target and self.team.claims:
            old = self._claim
            cost = round(self._cost_left(target), 1)
            new = old is None or old['kind'] != target['kind'] or math.hypot(old['x'] - target['x'],
                                                                              old['y'] - target['y']) > 0.3
            self._claim = {'kind': target['kind'], 'x': round(target['x'], 2), 'y': round(target['y'], 2), 'cost': cost}
            if new:
                self._say(t, 'claim', **self._claim)
        elif not target:
            self._claim = None
        if t - self._obs_t >= self.team.obs_period_s:
            self._flush(t)

    def _flush(self, t):
        if self._pose is None:
            return
        self._obs_t = t
        out, self._out = self._out, {'readings': [], 'soil': [], 'safe': []}
        x, y, th = self._pose
        self._say(t, 'obs', x=round(x, 3), y=round(y, 3), th=round(th, 3), battery=round(self._battery, 1),
                  state='returning' if self._returning else 'active', claim=self._claim,
                  readings=out['readings'] if self.team.share_readings else [],
                  soil=out['soil'] if self.team.share_soil else [],
                  safe=out['safe'] if self.team.share_hazards else [])

    # ======================================================================================
    # объезд напарника
    # ======================================================================================

    def _spot(self, t):
        """Найти в скане то, чего нет на карте: отражения посреди свободного места — башенка лидара другого робота."""
        scan, step, (x, y, th) = self._scan
        step = step or 2 * math.pi / len(scan)
        ang = th + np.arange(len(scan)) * step
        ok = np.isfinite(scan) & (scan < 3.0)
        px = x + LIDAR_OFFSET * math.cos(th) + np.where(ok, scan, 0.0) * np.cos(ang)
        py = y + LIDAR_OFFSET * math.sin(th) + np.where(ok, scan, 0.0) * np.sin(ang)
        ix = np.clip(((px - self.arena.x0) / self.arena.res).astype(int), 0, self.arena.w - 1)
        iy = np.clip(((py - self.arena.y0) / self.arena.res).astype(int), 0, self.arena.h - 1)
        ok &= self.arena.clear[iy, ix] >= self.SPOT_CLEAR
        if ok.sum() >= 2 or (ok.any() and scan[ok].min() < 1.0):
            k = np.nonzero(ok)[0][scan[ok].argmin()]            # ближайшее такое отражение и его соседи
            near = ok & (np.hypot(px - px[k], py - py[k]) < 0.2)
            r = float(scan[near].mean()) + 0.04                 # до оси башенки, а не до её поверхности
            a = float(np.arctan2(np.sin(ang[near]).mean(), np.cos(ang[near]).mean()))
            self._seen = (x + LIDAR_OFFSET * math.cos(th) + r * math.cos(a),
                          y + LIDAR_OFFSET * math.sin(th) + r * math.sin(a), t)
        self._avoid_update(t)

    def _obstacles(self, t):
        """Другие роботы, о которых я знаю: [(x, y, едет ли, имя или None)] — из сообщений и по лидару."""
        out = []
        if self.team.avoid:
            out = [[p['x'], p['y'], bool(p.get('moving', True)) and p['state'] in ('active', 'returning'), p['name']]
                   for p in self.partners.values() if 'x' in p]
        if self.team.lidar_avoid and self._seen and t - self._seen[2] < self.SPOT_KEEP_S:
            sx, sy, _ = self._seen
            same = [o for o in out if math.hypot(o[0] - sx, o[1] - sy) < 0.4]
            if same:
                same[0][0], same[0][1] = sx, sy         # лидар свежее сообщения
            else:
                out.append([sx, sy, True, None])
        return out

    def _avoid_update(self, t):
        """Клетки рядом с другим роботом дорожают для маршрута; обновляется, когда он сдвинулся."""
        if not (self.team.avoid or self.team.lidar_avoid) or self._pose is None:
            return
        near = [(o[0], o[1]) for o in self._obstacles(t)
                if math.hypot(o[0] - self._pose[0], o[1] - self._pose[1]) < 2.0]
        key = tuple((round(x, 1), round(y, 1)) for x, y in near) or None
        if key == self._avoid_at:
            return
        self._avoid_at = key
        extra = None
        if near:
            r0, r1 = self.AVOID_R
            extra = np.ones(self._X.shape)
            for x, y in near:
                d = np.hypot(self._X - x, self._Y - y)
                extra = np.maximum(extra, 1.0 + self.AVOID_W * np.clip((r1 - d) / (r1 - r0), 0.0, 1.0))
        for g in (self.graph, self.home_graph):
            g.extra = extra
            g.refresh()
        self._cost_version += 1
        self._base_dist = None

    def _command(self, io, v, w):
        if (self.team.avoid or self.team.lidar_avoid) and v > 0.0 and self._pose is not None:
            v = self._give_way(v)
        super()._command(io, v, w)

    def _give_way(self, v):
        x, y, th = self._pose
        t = self._t_now
        if self._seen and t - self._seen[2] >= self.SPOT_KEEP_S:
            self._avoid_update(t)                       # замеченный лидаром робот пропал из виду: путь снова свободен
        for ox, oy, moving, name in self._obstacles(t):
            d = math.hypot(ox - x, oy - y)
            bearing = math.remainder(math.atan2(oy - y, ox - x) - th, 2 * math.pi)
            if d >= self.NEAR_M or abs(bearing) >= 1.2:
                continue
            # Кто ждёт: со связью — младший по имени; без связи — тот, у кого другой робот справа («помеха
            # справа»). Стоящего робота объезжают (см. _avoid_update), ждать его незачем.
            wait = moving and (self.name > name if name else bearing < 0.0)
            if wait:
                self._yield_t = t if self._yield_t is None else self._yield_t
                if t - self._yield_t < self.YIELD_MAX_S:
                    return 0.0
            return min(v, 0.10)
        self._yield_t = None
        return v

    def tick(self, obs, io):
        self._t_now = obs.t
        super().tick(obs, io)


def _side_name(nx, ny):
    """Сторона света, в которую смотрит вектор (для подписи участков)."""
    if abs(nx) > abs(ny):
        return 'восток' if nx > 0 else 'запад'
    return 'север' if ny > 0 else 'юг'
