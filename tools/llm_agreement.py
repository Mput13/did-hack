#!/usr/bin/env python3
"""Насколько выбор модели совпадает с выбором простого правила.

Берёт сохранённые ответы моделей (runs/_llm_cache/<модель>/*.json: полный запрос и ответ), для каждого запроса
плана восстанавливает состояние, спрашивает правило HeuristicPlanner и сравнивает первую подцель.
Пишет runs/llm_real/agreement.json — его читает страница-объяснение.

    pixi run python tools/llm_agreement.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import extract_json                 # noqa: E402
from did.planner import HeuristicPlanner         # noqa: E402

CACHE = ROOT / 'runs' / '_llm_cache'
# папка кэша -> имя строки в таблице моделей
ARMS = {'gpt-6-luna': 'gpt-6-luna', 'typesafe-jev-router': 'jev-router', 'qwen3.8-flash-next': 'mai-qwen3.8-flash-next',
        'qwen3.6-35b-a3b': 'mai-qwen3.6-35b-a3b', 'Qwen3.5-122B-A10B': 'mai-qwen3.5-122b-a10b', 'qwen3.8-27b': 'mai-qwen3.8-27b',
        'deepseek-v4.1-flash': 'mai-deepseek-v4.1-flash', 'DeepSeek-V4-Flash': 'mai-deepseek-v4-flash'}


def first(subgoals):
    if not subgoals:
        return None
    sg = subgoals[0]
    return (sg.get('type'), sg.get('target')) if sg.get('type') in ('investigate', 'explore') else (sg.get('type'),)


def main():
    rule = HeuristicPlanner()
    out = {}
    for folder, arm in ARMS.items():
        same = total = 0
        diff = collections.Counter()
        for path in sorted((CACHE / folder).glob('*.json')):
            rec = json.loads(path.read_text(encoding='utf-8'))
            users = [m['content'] for m in rec.get('messages', []) if m.get('role') == 'user']
            if len(users) != 1 or 'Состояние:' not in users[0]:
                continue                      # повторные запросы на исправление и запросы других ролей не считаем
            try:
                state, _ = json.JSONDecoder().raw_decode(users[0][users[0].index('{'):])   # после состояния идёт текст
                found = extract_json(rec['text'])
                plan = found if isinstance(found, dict) else json.loads(found)
                mine, base = first(plan.get('subgoals')), first(rule.plan(state)['subgoals'])
            except Exception:                 # noqa: BLE001 — негодный ответ: его разбирает обвязка, здесь он не нужен
                continue
            total += 1
            same += mine == base
            if mine != base:
                diff[f"{mine[0] if mine else '—'} вместо {base[0] if base else '—'}"] += 1
        if total:
            out[arm] = {'decisions': total, 'same': same, 'share': round(same / total, 3), 'differences': dict(diff)}
            print(f'{arm:26s} решений {total:3d}, совпало с правилом {same:3d} ({same / total:.0%}); иначе: {dict(diff)}')
    (ROOT / 'runs' / 'llm_real' / 'agreement.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
