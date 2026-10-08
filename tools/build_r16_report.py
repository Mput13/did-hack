"""R16: таблицы отчёта по сводкам опытов. Чисел по памяти нет — всё из runs/*/summary.json.

    ./px python tools/build_r16_report.py      # research/findings/R16-tables.md и снимок R16-results.json

Берёт то, что посчитано: E24 (нетронутые сценарии), E24_deadline (срок ожидания), E24_r3 (20 сценариев R3),
E24_pilot (отладка), R16_missions (миссии словами), R16_cached (миссии на записанных ответах модели), R16_real
(настоящая модель). Чего нет — пропускает и пишет об этом. Таблица «было / стало» сравнивает E24 с числами
первого круга (research/findings/R16-results-round1.json, код до G2).
Интервалы — 95%, бутстреп по сценариям (did.metrics.paired и summarize с отдельным генератором на ячейку).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.metrics import paired, summarize                    # noqa: E402
from did.runner import RUNS                                  # noqa: E402
from did.waiting import LATE, OUTCOMES                       # noqa: E402

OUT = ROOT / 'research' / 'findings'
OLD = 'первый круг, код до G2, не пересчитывалось'
EXPERIMENTS = {'E24': 'нетронутые сценарии 7001–7040, пересчёт на нынешнем коде',
               'E24_deadline': 'срок ожидания: модель молчит или отвечает через 60 с (hard, 7001–7020)',
               'E24_r3': f'те же 20 сценариев, что в E16 (1001–1020); {OLD}',
               'E24_pilot': f'отладочные сценарии 1–40; {OLD}',
               'E24_pilot2': f'отладочные сценарии 1–40, ответ после смены плана отбрасывается; {OLD}'}
TIMELY = ('agree', 'same', 'tail', 'switched')      # ответ пришёл, пока вопрос стоял: подтвердил или изменил план
SHOW = [('score', 'Счёт', 1), ('hazard_hits', 'Штрафы за зоны', 2), ('returned', 'Возврат', 2),
        ('samples_share', 'Доля собранных', 2), ('idle_s', 'Простой, с', 0), ('time', 'Время, с', 0),
        ('llm_calls', 'Запросов', 1)]
DIFFS = [('score', 'Счёт', 1), ('hazard_hits', 'Штрафы за зоны', 2), ('returned', 'Возврат', 2),
         ('samples_share', 'Доля собранных', 2), ('idle_s', 'Простой, с', 0)]
PAIRS = [('act15', 'stand15'), ('safe15', 'stand15'), ('act29', 'stand29'), ('safe29', 'stand29'),
         ('safe15_far', 'stand15'), ('safe15_still', 'stand15'), ('act15_drop', 'stand15'), ('safe15_drop', 'stand15'),
         ('act15_drop', 'nowait'), ('safe15_drop', 'nowait'), ('stand15', 'nowait'), ('act15', 'nowait'), ('safe15', 'nowait'), ('stand29', 'nowait'), ('act29', 'nowait'),
         ('safe29', 'nowait'), ('act15', 'safe15'), ('nowait', 'rule'),
         ('act60', 'stand60'), ('leash60', 'stand60'), ('act_never', 'stand60'), ('leash_never', 'stand60'),
         ('act60', 'rule'), ('leash60', 'rule'), ('act_never', 'rule'), ('leash_never', 'rule')]
RECOVERY = [('act15', 'stand15'), ('safe15', 'stand15'), ('act29', 'stand29'), ('safe29', 'stand29'),
            ('safe15_far', 'stand15'), ('safe15_still', 'stand15'), ('act15_drop', 'stand15'),
            ('safe15_drop', 'stand15')]


def num(v, digits=1):
    return '—' if v is None else f'{v:.{digits}f}'.replace('.', ',').replace('-', '−')


def signed(v, digits=1):
    return '—' if v is None else f'{v:+.{digits}f}'.replace('.', ',').replace('-', '−')


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def load(exp):
    path = RUNS / exp / 'summary.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def pick(runs, arm, cond, level):
    return [r for r in runs if r['arm'] == arm and r['condition'] == cond and r['level'] == level]


def recovery(runs, new, stand, cond, level):
    """Какую долю потери от ожидания (nowait − stand) возвращает вариант new: отношение средних и интервал."""
    def by_seed(arm):
        return {r['seed']: r['metrics']['score'] for r in pick(runs, arm, cond, level)}
    a, b, top = by_seed(new), by_seed(stand), by_seed('nowait')
    seeds = sorted(set(a) & set(b) & set(top))
    if not seeds:
        return None
    gain = np.array([a[s] - b[s] for s in seeds])
    loss = np.array([top[s] - b[s] for s in seeds])
    out = {'n': len(seeds), 'loss': round(float(loss.mean()), 2), 'gain': round(float(gain.mean()), 2)}
    idx = np.random.default_rng(0).integers(len(seeds), size=(4000, len(seeds)))
    losses, gains = loss[idx].mean(axis=1), gain[idx].mean(axis=1)
    out['loss_ci'] = [round(float(np.percentile(losses, q)), 2) for q in (2.5, 97.5)]
    if out['loss_ci'][0] > 0:                # потеря установлена — долю есть от чего считать
        share = gains / losses
        out['share'] = round(float(gain.mean() / loss.mean()), 3)
        out['share_ci'] = [round(float(np.percentile(share, q)), 3) for q in (2.5, 97.5)]
    return out


def waits(runs, arm, cond, level=None):
    sel = [r for r in runs if r['arm'] == arm and r['condition'] == cond and (level is None or r['level'] == level)
           and r['metrics'].get('llm_wait')]
    if not sel:
        return None
    total = {k: sum(r['metrics']['llm_wait'].get(k, 0) for r in sel)
             for k in ('asked', 'asked_again', 'answered', 'rule_while_pending', 'rule_while_busy', 'after_replan',
                       'expired', 'unanswered', 'gave_up', *OUTCOMES)}
    total['runs'] = len(sel)
    total['late'] = sum(total[k] for k in LATE)
    total['timely'] = sum(total[k] for k in TIMELY)
    total['longest_hold_s'] = max(r['metrics']['llm_wait'].get('longest_hold_s', 0.0) for r in sel)
    total['differs'] = total['switched'] + total['stale'] + total['same'] + total['returning']
    return total


def experiment_section(exp, note, s, snapshot):
    runs = s['runs']
    arms = [a['id'] for a in s['spec']['arms']]
    conds = [c['id'] for c in s['spec']['conditions']]
    labels = {a['id']: a['label'] for a in s['spec']['arms']}
    cond_labels = {c['id']: c['label'] for c in s['spec']['conditions']}
    levels = s['spec']['levels']
    text = [f"## {exp}: {note}", '',
            f"Прогонов {len(runs)}, ошибок {len(s['errors'])}, сценариев на уровень {s['seeds']}, посчитано "
            f"{s['generated']}.", '', 'Варианты: ' + '; '.join(f'`{a}` — {labels[a]}' for a in arms) + '.', '']
    snap = snapshot.setdefault(exp, {'generated': s['generated'], 'runs': len(runs), 'errors': len(s['errors']),
                                     'means': {}, 'diffs': {}, 'recovery': {}, 'answers': {}})
    for cond in conds:
        text += [f'### Условие «{cond_labels[cond]}»', '']
        for level in levels:
            rows = []
            for arm in arms:
                sel = pick(runs, arm, cond, level)
                if not sel:
                    continue
                cells = []
                for i, (m, _, d) in enumerate(SHOW):
                    st = summarize(sel, m, np.random.default_rng(i))
                    snap['means'][f'{cond}/{level}/{arm}/{m}'] = st and [st['mean'], *st['ci']]
                    cells.append('—' if st is None else f"{num(st['mean'], d)} [{num(st['ci'][0], d)}; {num(st['ci'][1], d)}]"
                                 if m == 'score' else num(st['mean'], d))
                rows.append([f'`{arm}`', len(sel), *cells])
            text += [f'**{level}** — средние (у счёта в скобках 95% интервал):', '',
                     table(['Вариант', 'n', *[h for _, h, _ in SHOW]], rows), '']
            rows = []
            for a, b in PAIRS:
                if not pick(runs, a, cond, level) or not pick(runs, b, cond, level):
                    continue
                cells = []
                for i, (m, _, d) in enumerate(DIFFS):
                    p = paired(pick(runs, a, cond, level), pick(runs, b, cond, level), m, np.random.default_rng(100 + i))
                    snap['diffs'][f'{cond}/{level}/{a}-{b}/{m}'] = p and [p['mean'], *p['ci'], p['a_higher'], p['b_higher']]
                    cells.append('—' if p is None else
                                 f"{signed(p['mean'], d)} [{signed(p['ci'][0], d)}; {signed(p['ci'][1], d)}]")
                rows.append([f'`{a}` − `{b}`', *cells])
            text += [f'**{level}** — парные разности на одних сценариях, 95% интервал:', '',
                     table(['Разность', *[h for _, h, _ in DIFFS]], rows), '']
        rows = []
        for new, stand in RECOVERY:
            for level in levels:
                r = recovery(runs, new, stand, cond, level)
                if r is None:
                    continue
                snap['recovery'][f'{cond}/{level}/{new}'] = r
                share = (f"{num(100 * r['share'], 0)}% [{num(100 * r['share_ci'][0], 0)}; {num(100 * r['share_ci'][1], 0)}]"
                         if 'share' in r else 'потеря не установлена (ноль в интервале)')
                rows.append([f'`{new}`', level, f"{num(r['loss'], 1)} [{num(r['loss_ci'][0], 1)}; {num(r['loss_ci'][1], 1)}]",
                             num(r['gain'], 1), share])
        text += ['Возвращённая доля потери от ожидания: потеря — счёт `nowait` минус счёт «стоять и ждать» с тем же '
                 'временем ответа, возвращено — счёт варианта минус счёт «стоять и ждать»:', '',
                 table(['Вариант', 'Уровень', 'Потеря от ожидания', 'Возвращено', 'Доля потери'], rows), '']
        rows = []
        for arm in arms:
            w = waits(runs, arm, cond)
            if w is None:
                continue
            snap['answers'][f'{cond}/{arm}'] = w
            n = max(1, w['answered'] + w['timeout'])
            rows.append([f'`{arm}`', w['runs'], w['asked'], w['answered'], w['rule_while_pending'],
                         w['agree'], w['same'], w['tail'], w['switched'], w['agree_late'], w['stale'], w['returning'],
                         w['timeout'], w['failed'], f"{num(100 * w['timely'] / n, 0)}%", f"{num(100 * w['late'] / n, 0)}%",
                         f"{num(100 * w['after_replan'] / max(1, w['answered']), 0)}%", w['expired'], w['gave_up'],
                         num(w['longest_hold_s'], 1)])
        text += ['Ответы модели в новом режиме, все уровни вместе (штук за все прогоны). «В срок» — ответ пришёл, '
                 'пока вопрос стоял: подтвердил то, что робот делает, добавил остаток плана или сменил план. '
                 '«Поздних» — вопрос уже не стоял: план успел смениться, цель пропала, робот уже едет домой или '
                 'истёк срок ожидания. Доли — от вопросов с исходом (ответ или истёкший срок). «После смены плана» — '
                 'доля ответов, к приходу которых правило уже решало заново, при любом исходе:', '',
                 table(['Вариант', 'Прогонов', 'Вопросов', 'Ответов', 'Поводов решило правило, пока ждали',
                        'Подтвердила', 'То, что робот уже делает', 'Остаток плана в очередь', 'Переход на план модели',
                        'Подтвердила сменившийся план', 'Устарел, отброшен', 'Робот уже едет домой', 'Срок истёк',
                        'Нет ответа', 'В срок', 'Поздних', 'После смены плана', 'Просроченных ответов',
                        'Прогонов с отказом от модели', 'Самая долгая стоянка по привязи, с'], rows), '']
    return text


def was_now_section(snapshot):
    """E24: числа первого круга (код до G2) рядом с пересчётом на нынешнем коде."""
    path = OUT / 'R16-results-round1.json'
    if not path.exists() or 'E24' not in snapshot:
        return []
    old, new = json.loads(path.read_text(encoding='utf-8'))['E24'], snapshot['E24']
    text = ['## E24: было (первый круг, код до G2) и стало (нынешний код)', '',
            f"Те же 40 сценариев 7001–7040 и те же условия; это пересчёт после изменения кода, а не новое измерение. "
            f"Было посчитано {old['generated']}, стало — {new['generated']}. Условие — имитатор отвечает как правило.", '']

    def cell(src, key, d, sign=False):
        v = src.get(key)
        if not v:
            return '—'
        f = signed if sign else num
        return f"{f(v[0], d)} [{f(v[1], d)}; {f(v[2], d)}]"
    arms = [('rule', 'правило без ожидания'), ('nowait', 'имитатор без ожидания'), ('stand15', 'стоять и ждать, 15 с'),
            ('act15', 'ехать по правилу, 15 с'), ('safe15', 'привязь, 15 с'), ('stand29', 'стоять и ждать, 29 с'),
            ('act29', 'ехать по правилу, 29 с'), ('safe29', 'привязь, 29 с')]
    for level in ('hard', 'medium'):
        rows = []
        for arm, label in arms:
            row = [label]
            for m, d in (('score', 1), ('hazard_hits', 2), ('returned', 2), ('idle_s', 0)):
                k = f'base/{level}/{arm}/{m}'
                o, n = old['means'].get(k), new['means'].get(k)
                row.append(cell(old['means'], k, d) + ' → ' + cell(new['means'], k, d) if m == 'score'
                           else f"{num(o[0], d) if o else '—'} → {num(n[0], d) if n else '—'}")
            rows.append(row)
        text += [f'**{level}** — средние, «было → стало» (у счёта 95% интервал):', '',
                 table(['Вариант', 'Счёт', 'Штрафы за зоны', 'Возврат', 'Простой, с'], rows), '']
        rows = []
        for a, b in (('act15', 'stand15'), ('safe15', 'stand15'), ('act29', 'stand29'), ('safe29', 'stand29'),
                     ('stand15', 'nowait'), ('act15', 'nowait'), ('safe15', 'nowait'), ('act15', 'safe15')):
            row = [f'`{a}` − `{b}`']
            for m, d in (('score', 1), ('hazard_hits', 2), ('idle_s', 0)):
                k = f'base/{level}/{a}-{b}/{m}'
                row += [cell(old['diffs'], k, d, True), cell(new['diffs'], k, d, True)]
            rows.append(row)
        text += [f'**{level}** — парные разности, 95% интервал:', '',
                 table(['Разность', 'Счёт, было', 'Счёт, стало', 'Штрафы, было', 'Штрафы, стало', 'Простой, было',
                        'Простой, стало'], rows), '']
    rows = []
    for arm in ('act15', 'safe15', 'act29', 'safe29'):
        for level in ('hard', 'medium'):
            o, n = old['recovery'].get(f'base/{level}/{arm}'), new['recovery'].get(f'base/{level}/{arm}')

            def share(r):
                return (f"{num(100 * r['share'], 0)}% [{num(100 * r['share_ci'][0], 0)}; {num(100 * r['share_ci'][1], 0)}]"
                        if r and 'share' in r else 'потеря не установлена')
            rows.append([f'`{arm}`', level, f"{num(o['loss'], 1) if o else '—'} → {num(n['loss'], 1) if n else '—'}",
                         share(o), share(n)])
    text += ['Доля потери от ожидания, возвращённая вариантом:', '',
             table(['Вариант', 'Уровень', 'Потеря от ожидания, было → стало', 'Возвращено, было', 'Возвращено, стало'],
                   rows), '']
    rows = []
    for arm in ('act15', 'safe15', 'act29', 'safe29'):
        o, n = old['answers'].get(f'base/{arm}'), new['answers'].get(f'base/{arm}')
        if o and n:
            rows.append([f'`{arm}`', f"{o['answered']} → {n['answered']}",
                         f"{num(100 * o['late'] / max(1, o['answered']), 0)}% → "
                         f"{num(100 * n['late'] / max(1, n['answered'] + n['timeout']), 0)}%",
                         f"{o['switched']} → {n['switched']}", n['tail'], n['timeout']])
    text += ['Ответы модели (оба уровня, 80 прогонов). В первом круге ответ, подтвердивший прежний план после его '
             'смены, считался поздним, даже если робот снова делал то же самое; теперь он сверяется с тем, что '
             'робот делает сейчас, поэтому доли «поздних» двух кругов сравнимы только приблизительно:', '',
             table(['Вариант', 'Ответов', 'Поздних', 'Переходов на план модели', 'Остаток плана в очередь (стало)',
                    'Срок истёк (стало)'], rows), '']
    return text


def missions_section(snapshot):
    path = RUNS / 'R16_missions' / 'summary.json'
    if not path.exists():
        return ['## Миссии словами', '', 'Не посчитано (`./px python -m did.wait_eval missions`).', '']
    s = json.loads(path.read_text(encoding='utf-8'))
    snapshot['missions'] = {'created': s['created'], 'table': s['table']}
    text = ['## Миссии словами (имитатор, который понимает миссию; ответ через 15 с)', '',
            f"Сценарии: {len(s['scenarios'])} ({s['scenarios'][0]} … {s['scenarios'][-1]}), посчитано {s['created']}.", '']
    heads = {'M1': ['Ровно два и вернулся', 'Собрал больше двух'], 'M2': ['Остаток заряда в среднем'],
             'M3': ['Заезжал в правую половину', 'Там в среднем, с', 'Наибольший x, м'],
             'M4': ['Прогонов со штрафом', 'Из них: после штрафа домой и без сбора', 'Собрано после штрафа, всего']}
    for m_id in heads:
        rows = []
        for e in s['table']:
            if e['mission'] != m_id:
                continue
            extra = {'M1': lambda: [f"{e['exactly_two']} из {e['n']}", e['more_than_two']],
                     'M2': lambda: [num(e['battery_mean'], 1)],
                     'M3': lambda: [f"{e['entered_right']} из {e['n']}", num(e['time_right_s_mean'], 1), num(e['max_x'], 2)],
                     'M4': lambda: [e['with_penalty'], f"{e['home_after_penalty']} из {e['with_penalty']}",
                                    e['collected_after']]}[m_id]()
            rows.append([f"`{e['arm']}`", f"{e['success']} из {e['n']}", *extra, num(e['score_mean'], 1),
                         num(e['collected_mean'], 2), f"{e['returned']} из {e['n']}", num(e['idle_s_mean'], 0),
                         ', '.join(f'{k}: {v}' for k, v in sorted(e['outcomes'].items()))])
        head = ['Вариант', 'Выполнено по критерию R13', *heads[m_id]]
        text += [f"**{m_id}** — «{s['missions'][m_id]}»", '',
                 table([*head, 'Счёт', 'Собрано', 'Вернулся', 'Простой, с', 'Исходы'], rows), '']
    return text


def cached_section(snapshot):
    path = RUNS / 'R16_cached' / 'summary.json'
    if not path.exists():
        return ['## Миссии на записанных ответах настоящей модели', '',
                'Не посчитано (`./px python -m did.wait_eval cached`).', '']
    s = json.loads(path.read_text(encoding='utf-8'))
    snapshot['cached'] = {k: s[k] for k in ('created', 'requests', 'hits', 'misses', 'table')}
    rows = [[e['mission'], f"`{e['arm']}`", e['n'], e['hits'] + e['misses'], e['hits'], e['misses'],
             e['covered_before_first_miss'], e['fully_covered']] for e in s['table']]
    return [f"## Миссии на записанных ответах {s['model']}: только кэш, сеть не трогалась", '',
            f"Сценарии: {', '.join(s['scenarios'])}; посчитано {s['created']}. Всего запросов {s['requests']}: в кэше "
            f"нашлось {s['hits']}, не покрыто {s['misses']}. На непокрытый запрос отвечает правило (как при отказе "
            f"модели), поэтому эти прогоны — не измерение модели, а счёт покрытия.", '',
            table(['Миссия', 'Вариант', 'Прогонов', 'Запросов', 'Из кэша', 'Не покрыто', 'Из кэша до первого промаха',
                   'Прогонов целиком из кэша'], rows), '']


def real_section(snapshot):
    path = RUNS / 'R16_real' / 'summary.json'
    if not path.exists():
        return ['## Настоящая модель', '', 'Не посчитано (`./px python -m did.wait_eval real`).', '']
    s = json.loads(path.read_text(encoding='utf-8'))
    runs = s['runs']
    arms = [a for a in ('stand', 'act', 'safe') if any(r['arm'] == a for r in runs)]
    text = [f"## Настоящая модель {s['model']}: шесть сценариев R3, время ответа настоящее ({OLD})", '',
            'Настоящих обращений к сети по запускам: '
            + '; '.join(f"{'+'.join(c['arms'])} — {c['calls']}" for c in s['network_calls']) + '.', '']
    rows = []
    for r in sorted(runs, key=lambda r: (r['level'], r['seed'], arms.index(r['arm']))):
        w = r.get('llm_wait') or {}
        rows.append([f"{r['level']}-{r['seed']}", f"`{r['arm']}`", num(r['score'], 1), r['hazard_hits'],
                     'да' if r['returned'] else 'нет', f"{r['collected']} из {r['total']}", num(r['idle_s'], 0),
                     num(r['time_s'], 0), r['requests'], r['failed'], num(r['latency_s']['median'], 1),
                     num(r['latency_s']['max'], 1),
                     '—' if not w else f"{w['agree']}/{w['agree_late']}/{w['same']}/{w['switched']}/{w['stale']}/"
                                       f"{w['returning']}/{w['failed']}"])
    text += [table(['Сценарий', 'Вариант', 'Счёт', 'Штрафы за зоны', 'Вернулся', 'Собрано', 'Простой, с', 'Время, с',
                    'Запросов', 'Отказов', 'Медиана ответа, с', 'Самый долгий, с',
                    'Ответы: подтв./подтв. поздно/уже делает/переход/устарел/домой/нет'], rows), '']
    snapshot['real'] = {'created': s['created'], 'network_calls': s['network_calls'], 'means': {}, 'diffs': {}}
    rows = []
    for arm in arms:
        sel = [r for r in runs if r['arm'] == arm]
        n = len(sel)
        mean = {k: sum(r[k] or 0 for r in sel) / n for k in ('score', 'hazard_hits', 'idle_s', 'time_s', 'requests')}
        snapshot['real']['means'][arm] = {k: round(v, 2) for k, v in mean.items()} | {'n': n}
        rows.append([f'`{arm}`', n, num(mean['score'], 1), num(mean['hazard_hits'], 2),
                     f"{sum(r['returned'] for r in sel)} из {n}", num(mean['idle_s'], 0), num(mean['time_s'], 0),
                     num(mean['requests'], 1)])
    text += ['Средние по сценариям:', '',
             table(['Вариант', 'n', 'Счёт', 'Штрафы за зоны', 'Вернулся', 'Простой, с', 'Время, с', 'Запросов'], rows), '']
    rows = []
    for a in arms:
        if a == 'stand':
            continue
        cells = []
        for i, m in enumerate(('score', 'hazard_hits', 'idle_s')):
            wrap = lambda arm: [{'level': r['level'], 'seed': r['seed'], 'metrics': r} for r in runs if r['arm'] == arm]  # noqa: E731
            p = paired(wrap(a), wrap('stand'), m, np.random.default_rng(200 + i))
            snapshot['real']['diffs'][f'{a}-stand/{m}'] = p and [p['mean'], *p['ci'], p['a_higher'], p['b_higher']]
            cells.append('—' if p is None else f"{signed(p['mean'], 1)} [{signed(p['ci'][0], 1)}; {signed(p['ci'][1], 1)}], "
                                               f"выше в {p['a_higher']}, ниже в {p['b_higher']} из {p['n']}")
        rows.append([f'`{a}` − `stand`', *cells])
    text += ['Парные разности (шесть пар — интервал описательный):', '',
             table(['Разность', 'Счёт', 'Штрафы за зоны', 'Простой, с'], rows), '']
    return text


def main():
    snapshot = {}
    text = ['# R16. Таблицы', '', 'Собрано `tools/build_r16_report.py` из `runs/*/summary.json`; руками не правится.', '']
    for exp, note in EXPERIMENTS.items():
        s = load(exp)
        if s is None:
            text += [f'## {exp}: {note}', '', f'Не посчитано (`./px python -m did.experiments {exp} --jobs 3`).', '']
            continue
        text += experiment_section(exp, note, s, snapshot)
        if exp == 'E24':
            text += was_now_section(snapshot)
    text += missions_section(snapshot)
    text += cached_section(snapshot)
    text += real_section(snapshot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'R16-tables.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    (OUT / 'R16-results.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"таблицы: {OUT / 'R16-tables.md'}; снимок чисел: {OUT / 'R16-results.json'}")


if __name__ == '__main__':
    main()
