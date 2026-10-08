"""Действовать, пока модель думает (исследование R16).

Модели МАИ отвечают около 15 секунд. Прежде робот всё это время стоял (режим think), и стоянка в опасной
зоне стоила штрафа каждые 5 секунд. Здесь правило — быстрое решение, модель — медленное уточнение:

  повод        — сразу берётся план правила на том же состоянии и исполняется; вопрос модели уходит
                 параллельно (в ROS — в поток агента, в быстром симуляторе ответ «приходит» через время
                 ответа по часам прогона, а мир и робот всё это время движутся);
  ответ пришёл — совпал с тем, что робот делал в момент вопроса, или с тем, что делает сейчас: продолжать;
                 иначе перейти на план модели, если он выполним в НЫНЕШНЕМ состоянии
                 (цель на месте, заряда хватает, дорога есть), а если нет — отбросить и записать в журнал;
  новый повод, пока ответа нет — решает правило, вопрос не отменяется и второй не задаётся: в работе всегда
                 не больше одного запроса. Когда ответ придёт, модель спросят заново уже про нынешнее
                 состояние. Отменять и спрашивать заново хуже по двум причинам: запрос по сети, который уже
                 ушёл, отменить нельзя (поток занят, пока сервер не ответит), а поводы у едущего робота идут
                 чаще, чем отвечает модель, и при отмене по каждому поводу ответа не было бы вовсе;
  ответ устарел — модель спрашивают заново про нынешнее состояние: с планом правила она не согласилась;
  возврат на базу — окончательное решение, как и раньше: ответ, пришедший позже, его не отменяет.

Ответ, который не совпал с правилом, а план с тех пор уже сменился (правило решало заново по более свежим
данным, чем видела модель), — настройка llm_wait_superseded: apply — перейти на план модели, если он выполним
сейчас; drop — отбросить. Возврат на базу принимается в обоих случаях.

Что разрешено до ответа — настройка llm_act_while_waiting:
  rule — весь план правила;
  safe — только обратимое: не собирать образцы и не отъезжать от места вопроса дальше llm_wait_leash_m, но
         из опасной зоны выезжать без ожидания. Нужен миссиям, заданным словами: их понимает только модель,
         и правило способно нарушить запрет раньше, чем она ответит.

Врезки в did/agent.py — вызовы decide, poll и hold; вся логика здесь.
"""
import math
from collections import Counter

from .planner import HeuristicPlanner, resolve_subgoals

MODES = ('off', 'rule', 'safe')
DANGER_RISK = 0.2        # с такой вероятности зоны под роботом он «в опасной зоне» и стоять не должен
NEAR_TARGET_M = 0.45     # safe: ближе к кандидату до ответа не подъезжать (сбор — с 0,30 м)
SAME_PLACE_M = 0.5       # цели ближе этого — одна и та же цель (как у «нового кандидата» в агенте)
# Исходы ответа модели. Поздние — ответ пришёл, когда вопрос уже не стоял или цель пропала.
OUTCOMES = ('agree', 'agree_late', 'same', 'switched', 'stale', 'returning', 'failed')
LATE = ('agree_late', 'stale', 'returning')


def answer_delay(cfg, plan):
    """Сколько секунд прогона модель «думала» над этим планом в быстром симуляторе."""
    if cfg.llm_wait_measured:      # настоящее время ответа; у ответа из кэша — время исходного вызова
        return sum(ex.get('latency_ms') or 0 for ex in plan.get('exchanges') or []) / 1000.0
    return cfg.llm_wait_s


class _Delayed:
    """Быстрый симулятор: ответ посчитан сразу, но агент узнаёт его в ready_t по часам прогона."""

    def __init__(self, plan, ready_t):
        self.plan, self.ready_t = plan, ready_t

    def ready(self, t):
        return t >= self.ready_t

    def result(self):
        return self.plan


class _Threaded:
    """ROS: ответ считает поток агента, готовность — по самому потоку."""

    def __init__(self, future):
        self.future = future

    def ready(self, t):
        return self.future.done()

    def result(self):
        return self.future.result()


def _same(a, b, tol=0.05):
    """Одна ли это подцель: тот же вид и то же место."""
    if not a or not b or a['type'] != b['type']:
        return False
    return a['type'] == 'return_base' or math.hypot(a['x'] - b['x'], a['y'] - b['y']) <= tol


class ActWhileWaiting:

    def __init__(self, agent):
        if agent.cfg.llm_act_while_waiting not in MODES[1:]:
            raise ValueError(f"llm_act_while_waiting: «{agent.cfg.llm_act_while_waiting}»; есть: {', '.join(MODES)}")
        if agent.cfg.llm_wait_superseded not in ('apply', 'drop'):
            raise ValueError(f'llm_wait_superseded: «{agent.cfg.llm_wait_superseded}»; есть: apply, drop')
        self.a = agent
        self.mode = agent.cfg.llm_act_while_waiting
        self.pending = None          # вопрос, на который ждём ответ: всегда не больше одного
        self.again = None            # повод, случившийся в ожидании: после ответа спросить заново
        self.stats = Counter()
        self.waited_s = 0.0

    # --- повод для нового плана --------------------------------------------------------------

    def decide(self, obs, state, trigger, use_llm):
        """Повод для плана у агента с моделью. True — решение принято здесь, False — прежним кодом."""
        a = self.a
        if self.pending is None and not use_llm:
            return False                     # модель спрашивать рано (llm_min_interval_s): решает правило, как раньше
        rule = HeuristicPlanner().plan(state)
        if self.pending is None:
            note = 'Пока модель думает, действую по правилу. '
        else:
            # Прежний вопрос не отменяется и новый не задаётся: запросы не копятся.
            self.pending['superseded'] += 1
            self.again = trigger
            self.stats['rule_while_pending'] += 1
            note = 'Модель ещё отвечает на прежний вопрос, новый повод решает правило. '
        a._apply_plan(obs, {**rule, 'reasoning': note + rule['reasoning']}, state, trigger)
        if self.pending is None and not a._returning:      # возврат окончателен, спрашивать не о чем
            self._ask(obs, state, trigger, rule)
            if a.rec:
                self.pending['plan_i'] = len(a.rec.plans) - 1
        return True

    def _ask(self, obs, state, trigger, rule):
        a = self.a
        a._last_llm_t = obs.t
        if a._pool is not None:
            answer = _Threaded(a._pool.submit(a.planner.plan, state))
        else:
            plan = a.planner.plan(state)
            answer = _Delayed(plan, obs.t + answer_delay(a.cfg, plan))
        first = resolve_subgoals(rule['subgoals'], state)
        # base — что робот делает в момент вопроса: после повода это план правила, при повторном вопросе —
        # то, что осталось от прежних решений. С ним ответ и сверяется.
        self.pending = {'answer': answer, 'state': state, 'trigger': trigger, 't': obs.t, 'at': (obs.x, obs.y),
                        'rule_first': first[0] if first else None, 'base': dict(a.queue[0]) if a.queue else None,
                        'superseded': 0, 'plan_i': None}
        self.stats['asked'] += 1

    # --- ответ модели ------------------------------------------------------------------------

    def poll(self, obs):
        """Раз в такт: пришёл ли ответ, и не пора ли спросить заново."""
        a = self.a
        if a._trigger:
            return                           # повод ещё не разобран (пауза после сбора): сначала он
        p = self.pending
        if p is not None:
            if not p['answer'].ready(obs.t):
                return
            self.pending = None
            from .llm import CacheMiss       # здесь, а не в начале файла: агент без модели клиент модели не грузит
            try:
                plan = p['answer'].result()
            except CacheMiss:
                raise
            except Exception as e:           # noqa: BLE001 — сбой в потоке модели не должен ронять такт робота
                plan = {'reasoning': '', 'subgoals': [], 'source': 'fallback', 'exchanges': [],
                        'error': f'{type(e).__name__}: {e}'}
            self._receive(obs, p, plan)
        if self.again is not None and self.pending is None:
            self._ask_again(obs)

    def _ask_again(self, obs):
        """В ожидании были новые поводы, их решило правило: теперь спросить модель про нынешнее состояние."""
        a = self.a
        if a._returning:
            self.again = None
            return
        if obs.t - a._last_llm_t < a.cfg.llm_min_interval_s:
            return
        trigger, self.again = self.again, None
        state = a._state(obs, trigger)
        self._ask(obs, state, trigger, HeuristicPlanner().plan(state))    # робот продолжает то, что делает
        self.stats['asked_again'] += 1

    def _receive(self, obs, p, plan):
        a = self.a
        waited = obs.t - p['t']
        self.waited_s += waited
        self.stats['answered'] += 1
        exchanges = plan.get('exchanges') or []
        if a.rec:
            for ex in exchanges:
                a.rec.add_llm(p['t'], ex)
        subgoals = resolve_subgoals(plan['subgoals'], p['state'])
        first = subgoals[0] if subgoals else None
        why = ''
        if plan['source'] == 'fallback' or first is None:
            outcome = 'failed'
            why = plan.get('error') or 'ни одна подцель ответа не исполнима'
        elif a._returning:
            outcome = 'agree' if first['type'] == 'return_base' else 'returning'
        elif self._alike(first, p['base']):
            outcome = 'agree_late' if p['superseded'] else 'agree'
        elif self._doing(first):
            outcome = 'same'
        elif p['superseded'] and a.cfg.llm_wait_superseded == 'drop' and first['type'] != 'return_base':
            # Правило с тех пор решало заново по более свежим данным, чем видела модель. Возврат на базу — не
            # выбор среди целей, а запрет ехать дальше: он от свежести списка целей не зависит.
            outcome, why = 'stale', 'план с тех пор сменился по новым данным'
        else:
            subgoals, why = self._still_valid(obs, subgoals)
            outcome = 'switched' if subgoals else 'stale'
        self.stats[outcome] += 1
        if outcome == 'stale' and self.again is None:
            # Модель с планом правила не согласилась, а её собственный план уже не годится: вопрос остаётся
            # открытым, его надо задать заново про нынешнее состояние (в safe робот до ответа так и ограничен).
            self.again = p['trigger']
        if outcome == 'switched':
            self._switch(obs, p, plan, subgoals, waited)
        else:
            text = {'agree': 'модель подтвердила то, что робот делает, продолжаю',
                    'agree_late': 'модель подтвердила прежний план, но с тех пор он уже сменился',
                    'same': 'модель выбрала то, что робот уже делает, продолжаю',
                    'stale': f'ответ устарел и отброшен: {why}',
                    'returning': 'робот уже возвращается на базу, ответ отброшен',
                    'failed': f'пригодного ответа нет ({why}), остаётся план правила'}[outcome]
            a.journal.add(obs.t, 'llm', f'Ответ модели через {waited:.0f} с: {text}.', tag='wait_answer',
                          outcome=outcome, asked_t=round(p['t'], 1), trigger=p['trigger'],
                          answer=_label(first), rule=_label(p['rule_first']))
            if a.rec and p['plan_i'] is not None and exchanges and outcome != 'failed':
                # Запись плана правила, на который был задан вопрос: что ответила модель (как в прежнем режиме).
                a.rec.plans[p['plan_i']].update(rule_first=p['rule_first'], rule_match=first == p['rule_first'],
                                                latency_ms=sum(ex.get('latency_ms') or 0 for ex in exchanges))

    @staticmethod
    def _alike(first, doing):
        """Та же ли это цель: кандидат уточняется по ходу, поэтому для него допуск шире."""
        return _same(first, doing, SAME_PLACE_M if first['type'] == 'investigate' else 0.1)

    def _doing(self, first):
        """Делает ли робот уже то, что выбрала модель."""
        return bool(self.a.queue) and self._alike(first, self.a.queue[0])

    def _still_valid(self, obs, subgoals):
        """Подцели ответа, выполнимые сейчас. Первая не выполнима — ответ устарел: ([], причина)."""
        a = self.a
        dist, pred = a.graph.field(obs.x, obs.y)
        cands = a.belief.candidates(min_mass=a.cfg.candidate_mass)
        out = []
        for sg in subgoals:
            if sg['type'] == 'return_base':
                out.append(sg)
                break
            x, y, why = sg['x'], sg['y'], None
            if sg['type'] == 'investigate':
                near = [c for c in cands if math.hypot(c['x'] - x, c['y'] - y) <= SAME_PLACE_M]
                if near:
                    best = max(near, key=lambda c: c['mass'])
                    x, y = round(best['x'], 2), round(best['y'], 2)
                else:
                    why = 'кандидата на этом месте больше нет'
            elif sg['type'] == 'explore' and (math.hypot(x - obs.x, y - obs.y) < 0.3 or any(
                    obs.t - vt < 40.0 and math.hypot(x - vx, y - vy) < 0.4 for vt, vx, vy in a._visited)):
                why = 'точка разведки уже осмотрена'
            why = why or self._infeasible(obs, dist, pred, x, y)
            if why is None:
                out.append({**sg, 'x': x, 'y': y})
            elif not out:
                return [], why
        return out, None

    def _infeasible(self, obs, dist, pred, x, y):
        """Почему цель сейчас не годится (те же условия, что у признака feasible в сводке состояния)."""
        a = self.a
        to = a.graph.energy(dist, pred, x, y) * a._per_m()
        if not math.isfinite(to):
            return 'до цели нет дороги'
        ix, iy = a.arena.w2g(x, y)
        if a._risk[iy, ix] >= 0.5:
            return 'цель оказалась в опасной зоне'
        if not a._affordable(obs.battery, to, a._home_cost(x, y)):
            return 'заряда на цель и возврат уже не хватает'
        return None

    def _switch(self, obs, p, plan, subgoals, waited):
        """Перейти на план модели: подцели уже проверены на нынешнем состоянии."""
        a = self.a
        targets = [sg for sg in subgoals if 'target' in sg]
        now = {'candidates': [{'id': sg['target'], 'x': sg['x'], 'y': sg['y']} for sg in targets
                              if sg['type'] == 'investigate'],
               'explore_points': [{'id': sg['target'], 'x': sg['x'], 'y': sg['y']} for sg in targets
                                  if sg['type'] == 'explore']}
        a._apply_plan(obs, {**plan, 'subgoals': subgoals, 'exchanges': [],
                            'reasoning': f'Ответ модели пришёл через {waited:.0f} с и расходится с тем, что делает '
                                         f'робот: перехожу на план модели. ' + plan['reasoning']}, now, p['trigger'])
        # Кандидаты, найденные за время ожидания, роботу уже известны: новым поводом они не считаются.
        a._known_cands = [(c['x'], c['y']) for c in a.belief.candidates(min_mass=a.cfg.candidate_mass)]
        if a.rec:
            a.rec.plans[-1].update(rule_first=p['rule_first'], rule_match=False, asked_t=round(p['t'], 1),
                                   latency_ms=sum(ex.get('latency_ms') or 0 for ex in plan.get('exchanges') or []))

    # --- что можно до ответа -----------------------------------------------------------------

    def hold(self, obs):
        """safe: True — до ответа модели стоять. Из опасной зоны робот выезжает без ожидания."""
        a = self.a
        if self.mode != 'safe' or self.pending is None or a._returning or not a.queue:
            return False
        a._no_collect_until = max(a._no_collect_until, obs.t + 0.3)       # сбор необратим: до ответа не собираю
        if self.in_danger(obs):
            return False
        px, py = self.pending['at']
        if math.hypot(obs.x - px, obs.y - py) >= a.cfg.llm_wait_leash_m:
            return True
        sg = a.queue[0]
        return sg['type'] == 'investigate' and math.hypot(obs.x - sg['x'], obs.y - sg['y']) <= NEAR_TARGET_M

    def in_danger(self, obs):
        a = self.a
        if a._grace[1] is not None:          # только что получен штраф: робот выезжает из зоны
            return True
        if not a.hazards:
            return False
        ix, iy = a.arena.w2g(obs.x, obs.y)
        return bool(a._risk_full[iy, ix] >= DANGER_RISK)

    def summary(self):
        """Счётчики за прогон: вопросы, исходы ответов (OUTCOMES), ожидание в секундах."""
        s = {k: int(self.stats.get(k, 0)) for k in ('asked', 'asked_again', 'answered', 'rule_while_pending',
                                                     *OUTCOMES)}
        s['unanswered'] = int(self.pending is not None)
        s['waited_s'] = round(self.waited_s, 1)
        return s


def _label(sg):
    if not sg:
        return None
    if sg['type'] == 'return_base':
        return 'return_base'
    return f"{sg['type']} {sg.get('target', '')} ({sg['x']:g}; {sg['y']:g})".replace('  ', ' ')


def wait_metrics(rec, bot):
    """Метрики ожидания модели по записи прогона.

    idle_s — сколько секунд робот стоял в режиме think (ждал ответа модели; сюда же входит секунда после
    каждого сбора, она одинакова у всех вариантов). В режиме «действовать, пока модель думает» ещё llm_wait
    (счётчики ActWhileWaiting.summary), llm_late — ответы, пришедшие поздно, llm_switches — переходы на план
    модели.
    """
    think = rec.modes.index('think') if 'think' in rec.modes else None
    m = {'idle_s': round(sum(1 for i in rec.track['mode'] if i == think) * rec.track_dt, 1)}
    aw = getattr(bot, 'aw', None)
    if aw is not None:
        s = aw.summary()
        m.update(llm_wait=s, llm_late=sum(s[k] for k in LATE), llm_switches=s['switched'])
    return m
