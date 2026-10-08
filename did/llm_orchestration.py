"""Способы обращения к одной и той же модели за планом (исследование R3).

  single      — один запрос (ветка LLMPlanner, здесь её нет: она не менялась);
  critic      — автор → критик → исправление, если критик потребовал;
  vote        — несколько независимых запросов, большинство по первой подцели;
  scored      — модель предлагает до K планов, код считает последствия каждого по данным состояния,
                модель выбирает по этой таблице;
  scored_calc — контроль: те же предложения модели, но выбирает сам расчёт, модель не спрашивают.

Первый запрос critic и vote дословно тот же, что у single: при кэше ответов это тот же ответ, и
разница между способами — только в добавленных шагах. Первый запрос scored и scored_calc одинаков
между собой. Расчёт участвует в решении только у scored_calc: если выбор модели в scored не разобран,
остаётся первый её вариант (тот, что она назвала своим), а не выбор расчёта.

Каждый способ возвращает то же, что LLMPlanner.plan, и ещё orchestration — итог каждого шага.
Тот же итог записан в последнем обмене шага (поле outcome), так что он виден в записи прогона.
Плана нет ни от одного шага — решает запасное правило (source='fallback').
"""
import json
from collections import Counter

from pydantic import ValidationError

from .llm import (LLMError, Plan, _jsonable, _obj, _schema_errors, _trim_targets, build_messages, extract_json,
                  load_system_prompt, parse_plan, plan_schema, request_json, validate_plan)

STRATEGIES = ('single', 'critic', 'vote', 'scored', 'scored_calc')
VOTERS = 3              # сколько запросов в голосовании
MAX_PROPOSALS = 3       # K: сколько планов модель предлагает на выбор

# По этим словам и полю request имитатор (did/llm_mock.py) узнаёт, какой шаг у него спрашивают.
PROPOSE_MARK = 'планов на выбор'

CRITIC_PROMPT = (
    'Ты — критик планов автономного робота-исследователя. Планировщик получил инструкцию (она ниже) и '
    'состояние и предложил план. Проверь первую подцель плана по состоянию и по правилам выбора из инструкции.\n'
    'Ищи конкретную ошибку: цель с feasible: false или не из списков; выбран кандидат, хотя есть достижимый '
    'кандидат заметно выгоднее по confidence / (cost_to + 0.5); разведка или возврат на базу при достижимом '
    'кандидате; преждевременный return_base (достижимые цели есть, собраны не все образцы, время есть); '
    'возврата нет, хотя достижимых целей не осталось.\n'
    'Конкретной ошибки нет — verdict "accept": исправление стоит роботу времени, правка ради правки вредна.\n'
    'Ответ — только один JSON-объект, без Markdown и без текста вне JSON:\n'
    '{"verdict": "accept" или "revise", "issues": [строка, ...], "advice": строка}\n'
    'issues — найденные ошибки с числами из состояния (при accept — пустой список); advice — какую первую '
    'подцель взять вместо выбранной (при accept — пустая строка). Отвечай коротко.\n\n'
    '# Инструкция, по которой работал планировщик\n')

REVIEW_SCHEMA = _obj(verdict={'type': 'string', 'enum': ['accept', 'revise']},
                     issues={'type': 'array', 'items': {'type': 'string'}}, advice={'type': 'string'})

PROPOSE_TEXT = (
    'Сейчас нужен не один план, а до {k} разных {mark}: у каждого своя первая подцель. Первым поставь план, '
    'который выбрал бы сам; дальше — лучшие из остальных разумных вариантов (другая цель, разведка, возврат на '
    'базу). Разумных вариантов меньше {k} — верни меньше. Каждый план — в обычном формате; reasoning — до 200 '
    'символов; hypotheses — только у первого плана, у остальных пустой список.\n'
    'Ответ — один JSON-объект: {{"plans": [план, ...]}}.')

CHOICE_TEXT = (
    'Для каждого твоего варианта код робота посчитал последствия первой подцели по данным состояния — это '
    'расчёт, а не мнение:\n{table}\n'
    'Поля: sample_probability — вероятность образца у цели; unseen_share — доля непроверенной области вокруг '
    'точки; cost_to и cost_back — заряд на путь до цели и оттуда до базы; battery_after_return — сколько заряда '
    'останется, если после цели сразу вернуться на базу; enough_battery — хватает ли заряда с запасом; '
    'sample_per_charge = sample_probability / (cost_to + 0.5); explore_per_charge = unseen_share / (cost_to + 1); '
    'ends_run — вариант завершает прогон, несобранные образцы пропадут; null — оценки нет.\n'
    'Выбери один вариант по правилам из инструкции. Ответ — один JSON-объект: '
    '{{"choice": номер варианта, "reasoning": почему, до 200 символов}}.')

_KIND_ORDER = {'investigate': 3, 'explore': 2, 'goto': 1, 'return_base': 0}


def first_key(plan):
    """Первая подцель плана одним значением: по ней сравнивают планы и считают голоса."""
    sg = plan.subgoals[0]
    if sg.type == 'goto':
        return ('goto', round(sg.x, 2), round(sg.y, 2))
    return (sg.type, getattr(sg, 'target', None))


def label(plan):
    """Первая подцель словами: «investigate C1», «goto (0.5; -1)», «return_base»."""
    key = first_key(plan)
    if key[0] == 'goto':
        return f'goto ({key[1]:g}; {key[2]:g})'
    return ' '.join(str(k) for k in key if k is not None)


def _dump(plan):
    return plan.model_dump(exclude_none=True)


def _text(obj):
    return json.dumps(obj, ensure_ascii=False, default=_jsonable)


class _Trace:
    """Обмены, время и итоги шагов одного решения."""

    def __init__(self, strategy):
        self.exchanges = []
        self.latency_ms = 0
        self.info = {'strategy': strategy, 'steps': []}

    def ask(self, client, messages, parse, schema, role, max_repairs, noun='план'):
        obj, error, exchanges, ms = request_json(client, messages, parse, schema, max_repairs, role=role, noun=noun)
        self.exchanges += exchanges
        self.latency_ms += ms or 0
        return obj, error

    def step(self, role, outcome):
        self.info['steps'].append({'role': role, 'outcome': outcome})
        for ex in reversed(self.exchanges):
            if ex.get('role') == role:
                ex['outcome'] = outcome
                break

    def result(self, plan, note=''):
        data = _dump(plan)
        return {'reasoning': note + data['reasoning'], 'hypotheses': data.get('hypotheses', []),
                'subgoals': data['subgoals'], 'source': 'llm', 'exchanges': self.exchanges, 'error': None,
                'latency_ms': self.latency_ms, 'orchestration': self.info}

    def fallback(self, planner, state, error):
        out = planner.plan(state)
        out.update(source='fallback', exchanges=self.exchanges, error=error, latency_ms=self.latency_ms,
                   orchestration=self.info)
        out['reasoning'] = f'Модель не дала пригодного плана ({error}). Решение по правилу: ' + out['reasoning']
        return out


# --- автор, критик, исправление ----------------------------------------------------------------

def parse_review(text):
    """Ответ критика -> ({verdict, issues, advice} или None, ошибки)."""
    try:
        data = extract_json(text)
    except LLMError as e:
        return None, [str(e)]
    verdict = str(data.get('verdict') or '').strip().lower()
    if verdict not in ('accept', 'revise'):
        return None, ['verdict: нужно "accept" или "revise"']
    issues = data.get('issues') or []
    issues = [issues] if isinstance(issues, str) else issues if isinstance(issues, list) else []
    return {'verdict': verdict, 'issues': [str(i) for i in issues if str(i).strip()],
            'advice': str(data.get('advice') or '').strip()}, []


def critic_messages(state, plan, system_prompt=None):
    """Запрос критику: отдельный разговор, в нём инструкция планировщика, состояние и план автора."""
    plan = {k: v for k, v in _dump(plan).items() if k != 'hypotheses'}
    payload = {'request': 'review', 'state': state, 'plan': plan}
    return [{'role': 'system', 'content': CRITIC_PROMPT + (system_prompt or load_system_prompt())},
            {'role': 'user', 'content': f'Проверь план (поле plan) по состоянию (поле state):\n{_text(payload)}\n'
                                        'Верни отзыв: один JSON-объект.'}]


def revision_message(review):
    issues = '\n'.join(f'- {i}' for i in review['issues']) or '- (критик не перечислил ошибки)'
    advice = f"\nСовет критика: {review['advice']}" if review['advice'] else ''
    return ('Критик проверил твой план и требует исправить. Замечания:\n' + issues + advice +
            '\nЗамечание верно — верни исправленный план; критик ошибся — верни прежний план. '
            'Один JSON-объект по схеме, без пояснений и без Markdown.')


def plan_critic(client, state, fallback, system_prompt=None, max_repairs=1):
    tr = _Trace('critic')
    parse = lambda text: parse_plan(text, state)                                   # noqa: E731
    messages = build_messages(state, system_prompt)
    plan, error = tr.ask(client, messages, parse, plan_schema(state), 'author', max_repairs)
    if plan is None:
        tr.step('author', f'плана нет: {error}')
        return tr.fallback(fallback, state, error)
    tr.step('author', label(plan))
    tr.info.update(author=label(plan), verdict=None, revised=False, changed=False)

    review, error = tr.ask(client, critic_messages(state, plan, system_prompt), parse_review, REVIEW_SCHEMA,
                           'critic', max_repairs, noun='отзыв')
    if review is None:
        tr.step('critic', f'отзыва нет, остаётся план автора: {error}')
        return tr.result(plan)
    tr.info['verdict'] = review['verdict']
    if review['verdict'] == 'accept':
        tr.step('critic', 'accept')
        return tr.result(plan)
    tr.step('critic', 'revise: ' + ('; '.join(review['issues']) or review['advice'] or 'без пояснений')[:300])

    revised, error = tr.ask(client, messages + [{'role': 'assistant', 'content': _text(_dump(plan))},
                                                {'role': 'user', 'content': revision_message(review)}],
                            parse, plan_schema(state), 'revision', max_repairs)
    if revised is None:
        tr.step('revision', f'исправления нет, остаётся план автора: {error}')
        return tr.result(plan)
    changed = first_key(revised) != first_key(plan)
    tr.info.update(revised=True, changed=changed)
    tr.step('revision', label(revised) + (' — первая подцель изменена' if changed else ' — первая подцель прежняя'))
    return tr.result(revised)


# --- голосование -------------------------------------------------------------------------------

def plan_vote(client, state, fallback, system_prompt=None, max_repairs=1, n=VOTERS):
    """n запросов, побеждает первая подцель с большинством голосов; при равенстве — более ранний запрос.

    Первый запрос — тот же, что у single. К остальным дописан номер: одинаковые запросы кэш (и наш, и
    серверный) вернул бы одним и тем же ответом, и голосовать было бы не из чего.
    """
    tr = _Trace('vote')
    plans, error = [], None
    for i in range(n):
        messages = build_messages(state, system_prompt)
        if i:
            messages[-1]['content'] += f'\n(Запрос № {i + 1}.)'
        role = f'voter_{i + 1}'
        plan, err = tr.ask(client, messages, lambda text: parse_plan(text, state), plan_schema(state), role,
                           max_repairs)
        tr.step(role, label(plan) if plan is not None else f'плана нет: {err}')
        if plan is None:
            error = err
        else:
            plans.append(plan)
    if not plans:
        return tr.fallback(fallback, state, error)
    counts = Counter(first_key(p) for p in plans)
    top = max(counts.values())
    chosen = next(p for p in plans if counts[first_key(p)] == top)
    tr.info.update(votes={label(p): counts[first_key(p)] for p in plans}, chosen=label(chosen),
                   unanimous=len(counts) == 1 and len(plans) == n)
    return tr.result(chosen)


# --- предложения, расчёт последствий, выбор ----------------------------------------------------

def proposals_messages(state, k=MAX_PROPOSALS, system_prompt=None):
    messages = build_messages(state, system_prompt)
    head = messages[-1]['content'].rsplit('\n', 1)[0]            # «Состояние: {...}» без просьбы об одном плане
    messages[-1]['content'] = head + '\n' + PROPOSE_TEXT.format(k=k, mark=PROPOSE_MARK)
    return messages


def proposals_schema(state):
    return _obj(plans={'type': 'array', 'items': plan_schema(state)})


def parse_proposals(text, state, k=MAX_PROPOSALS):
    """Ответ с вариантами -> (список Plan с разными первыми подцелями, не длиннее k, или None; ошибки).

    Негодный вариант отбрасывается, годные остаются: один плохой план не губит весь ответ.
    """
    try:
        data = extract_json(text)
    except LLMError as e:
        return None, [str(e)]
    items = data.get('plans')
    if items is None and 'subgoals' in data:                     # модель вернула один план без обёртки
        items = [data]
    if not isinstance(items, list) or not items:
        return None, ['plans: нужен непустой список планов']
    plans, errors, seen = [], [], set()
    for i, item in enumerate(items):
        if isinstance(item, dict):
            _trim_targets(item, state)
        try:
            plan = Plan.model_validate(item)
        except ValidationError as e:
            errors += [f'plans[{i}].{msg}' for msg in _schema_errors(e)]
            continue
        bad = validate_plan(plan, state)
        if bad:
            errors += [f'plans[{i}].{msg}' for msg in bad]
        elif first_key(plan) not in seen and len(plans) < k:
            seen.add(first_key(plan))
            plans.append(plan)
    return (plans, []) if plans else (None, errors)


def consequences(plan, state):
    """Строка таблицы: что даёт и чего стоит первая подцель плана — только числа из состояния.

    Дальше первой подцели таблица не смотрит: остальные агент пересчитает после ближайшего события.
    Для goto оценок пути в состоянии нет — в строке null.
    """
    sg = plan.subgoals[0]
    battery = float(state.get('battery') or 0)
    row = {'first_subgoal': label(plan), 'kind': sg.type, 'sample_probability': 0.0, 'unseen_share': None,
           'cost_to': None, 'cost_back': None, 'battery_after_return': None, 'enough_battery': None,
           'sample_per_charge': None, 'explore_per_charge': None, 'ends_run': sg.type == 'return_base'}
    if sg.type == 'return_base':
        back = float(state.get('return_cost') or 0)
        row.update(cost_to=0.0, cost_back=back, battery_after_return=round(battery - back, 1),
                   enough_battery=battery >= back)
        return row
    if sg.type == 'goto':
        return row
    name = 'candidates' if sg.type == 'investigate' else 'explore_points'
    item = next((it for it in state.get(name) or [] if it.get('id') == sg.target), None)
    if item is None:
        return row
    to, back = float(item.get('cost_to') or 0), float(item.get('cost_back') or 0)
    row.update(cost_to=to, cost_back=back, battery_after_return=round(battery - to - back, 1),
               enough_battery=bool(item.get('feasible', True)))
    if sg.type == 'investigate':
        p = float(item.get('confidence') or 0)
        row.update(sample_probability=p, sample_per_charge=round(p / (to + 0.5), 3))
    else:
        share = float(item.get('unseen_share') or 0)
        row.update(unseen_share=share, explore_per_charge=round(share / (to + 1.0), 3))
        if 'gain_bits' in item:                                  # агент посчитал пользу измерений — как у правила
            row.update(gain_bits=item['gain_bits'], gain_per_charge=round(float(item['gain_bits']) / (to + 1.0), 3))
    return row


def calc_choice(rows, state):
    """Номер строки (с нуля), которую выбирает расчёт. Порядок записан заранее и от результатов не зависит.

    Это критерий запасного правила (HeuristicPlanner), применённый к предложениям модели: хватает заряда;
    кандидат раньше разведки, разведка раньше goto и возврата; внутри вида — больше пользы на единицу
    заряда; при равенстве — вариант, который модель назвала раньше. Все образцы собраны — возврат.
    """
    samples = state.get('samples') or {}
    done = bool(samples.get('total')) and samples.get('collected', 0) >= samples['total']

    def rank(i):
        row = rows[i]
        value = (row['sample_per_charge'] if row['kind'] == 'investigate'
                 else row.get('gain_per_charge', row['explore_per_charge']))
        return (done and row['kind'] == 'return_base', row['enough_battery'] is not False,
                _KIND_ORDER[row['kind']], value or 0.0, -i)

    return max(range(len(rows)), key=rank)


def parse_choice(text, n):
    """Ответ с выбором -> ({choice, reasoning} или None, ошибки); choice — номер варианта от 1 до n."""
    try:
        data = extract_json(text)
    except LLMError as e:
        return None, [str(e)]
    choice = data.get('choice')
    if isinstance(choice, str) and choice.strip().isdigit():
        choice = int(choice)
    if isinstance(choice, float) and choice.is_integer():
        choice = int(choice)
    if type(choice) is not int or not 1 <= choice <= n:
        return None, [f'choice: нужен номер варианта от 1 до {n}']
    return {'choice': choice, 'reasoning': str(data.get('reasoning') or '').strip()}, []


def choice_message(rows):
    table = _text({'request': 'choice', 'options': [{'option': i + 1, **row} for i, row in enumerate(rows)]})
    return CHOICE_TEXT.format(table=table)


def plan_scored(client, state, fallback, system_prompt=None, max_repairs=1, k=MAX_PROPOSALS, calculate=False):
    """Модель предлагает до k планов; выбирает по таблице последствий модель (или расчёт при calculate)."""
    tr = _Trace('scored_calc' if calculate else 'scored')
    messages = proposals_messages(state, k, system_prompt)
    plans, error = tr.ask(client, messages, lambda text: parse_proposals(text, state, k), proposals_schema(state),
                          'proposals', max_repairs, noun='список планов')
    if plans is None:
        tr.step('proposals', f'планов нет: {error}')
        return tr.fallback(fallback, state, error)
    rows = [consequences(p, state) for p in plans]
    by_calc = calc_choice(rows, state)
    tr.step('proposals', ', '.join(label(p) for p in plans))
    tr.info.update(proposals=[label(p) for p in plans], table=rows, calc_choice=by_calc + 1, model_choice=None,
                   chosen_by='calc' if calculate else 'model')
    if calculate:
        tr.info['steps'].append({'role': 'calc', 'outcome': f'вариант {by_calc + 1}: {label(plans[by_calc])}'})
        return tr.result(plans[by_calc], f'[Из {len(plans)} вариантов модели расчёт выбрал {by_calc + 1}.] ')
    if len(plans) == 1:
        tr.info.update(model_choice=1, chosen_by='only')
        tr.info['steps'].append({'role': 'choice', 'outcome': 'вариант один, выбирать не из чего'})
        return tr.result(plans[0])
    picked, error = tr.ask(client, messages + [{'role': 'assistant', 'content': _text({'plans': [_dump(p) for p in plans]})},
                                               {'role': 'user', 'content': choice_message(rows)}],
                           lambda text: parse_choice(text, len(plans)),
                           _obj(choice={'type': 'integer', 'enum': list(range(1, len(plans) + 1))},
                                reasoning={'type': 'string'}), 'choice', max_repairs, noun='выбор')
    if picked is None:                       # расчёт за модель не выбирает: остаётся вариант, который она назвала своим
        tr.info['chosen_by'] = 'first'
        tr.step('choice', f'выбора нет, остаётся первый вариант: {error}')
        return tr.result(plans[0])
    tr.info['model_choice'] = picked['choice']
    tr.step('choice', f"вариант {picked['choice']}: {label(plans[picked['choice'] - 1])}")
    note = f"[Выбор из {len(plans)} вариантов по таблице: {picked['reasoning']}] " if picked['reasoning'] else ''
    return tr.result(plans[picked['choice'] - 1], note)


def plan_orchestrated(strategy, client, state, fallback, system_prompt=None, max_repairs=1):
    """План способом strategy (не single). state — уже с полем mission, как в запросе single."""
    if strategy == 'critic':
        return plan_critic(client, state, fallback, system_prompt, max_repairs)
    if strategy == 'vote':
        return plan_vote(client, state, fallback, system_prompt, max_repairs)
    if strategy in ('scored', 'scored_calc'):
        return plan_scored(client, state, fallback, system_prompt, max_repairs, calculate=strategy == 'scored_calc')
    raise ValueError(f"неизвестный способ «{strategy}»; есть: {', '.join(STRATEGIES)}")
