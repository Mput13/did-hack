"""L4b, дополнение: видит ли планировщик на языковой модели память о лаборатории и меняет ли она его решения.

    ./px python tools/l4b_llm.py --labs 4          # 4 лаборатории × 2 прогона на настоящей модели → runs/L4b_llm

Память готовит обычный агент adaptive_v2_lab (первый прогон, раскладка 1, без модели). Затем на раскладке 2 идут
два прогона с планировщиком на модели: adaptive_llm (без памяти) и adaptive_llm_lab (в снимке состояния — поле
lab_memory). Лаборатории — отладочные (сценарии 1–80), только те, где в памяти после первого прогона есть опасная
зона. Запросы к модели идут по одному. Адрес и ключ — из .env основного каталога проекта; ключ не печатается.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.labmemory import LabMemory                    # noqa: E402
from did.llm import load_env                           # noqa: E402
from did.recorder import load_trace                    # noqa: E402
from did.runner import RUNS, run_episode               # noqa: E402

MODEL = 'qwen3.8-flash-next'
EXP = 'L4b_llm'
WORDS = re.compile(r'памят|прошл\w* прогон|lab_memory|из прошлых|предыдущ\w* прогон|\bM\d\b', re.IGNORECASE)


def digest(summary):
    tr = load_trace(RUNS / summary['file'])
    plans = [p for p in tr['plans'] if p.get('source') == 'llm']
    cited = [p for p in plans if WORDS.search(p.get('reasoning') or '')]
    judged = [p for p in plans if 'rule_match' in p]
    m = summary['metrics']
    return {'score': m['score'], 'samples_share': m['samples_share'], 'returned': m['returned'],
            'hazard_hits': m['hazard_hits'], 'llm_plans': len(plans), 'plans_citing_memory': len(cited),
            'rule_match': round(sum(p['rule_match'] for p in judged) / len(judged), 3) if judged else None,
            'llm_stats': m.get('llm_stats'), 'quotes': [p['reasoning'][:400] for p in cited[:3]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labs', type=int, default=4)
    ap.add_argument('--level', default='hard')
    args = ap.parse_args()
    main_dir = Path(__file__).resolve().parent.parent
    for line in (main_dir / '.git').read_text().splitlines() if (main_dir / '.git').is_file() else []:
        if line.startswith('gitdir:'):          # рабочее дерево исследования: .env лежит в основном каталоге
            main_dir = Path(line.split(':', 1)[1].strip()).parent.parent.parent
    load_env(main_dir / '.env')
    llm = {'kind': 'http', 'model': MODEL, 'min_tokens': 3000}
    rows, seed = [], 0
    while len(rows) < args.labs and seed < 80:
        seed += 1
        first = run_episode(args.level, seed, 'adaptive_v2_lab', save=False, scenario_args={'sample_seed': 1})
        mem = LabMemory()
        mem.learn(first['lab'], first['id'])
        prior = mem.priors()
        if not prior['hazards']:
            continue
        row = {'seed': seed, 'memory': {'hazards': len(prior['hazards']), 'soil_zones': len(prior['soil_zones'])}}
        for agent in ('adaptive_llm', 'adaptive_llm_lab'):
            s = run_episode(args.level, seed, agent, experiment=EXP, llm=dict(llm), scenario_args={'sample_seed': 2},
                            lab=prior if agent.endswith('_lab') else None)
            row[agent] = digest(s)
            print(f"{args.level}-{seed} {agent}: счёт {row[agent]['score']:.1f}, решений модели {row[agent]['llm_plans']}, "
                  f"из них со ссылкой на память {row[agent]['plans_citing_memory']}, совпало с правилом "
                  f"{row[agent]['rule_match']}", flush=True)
        rows.append(row)
        (RUNS / EXP).mkdir(parents=True, exist_ok=True)
        (RUNS / EXP / 'summary.json').write_text(json.dumps({'model': MODEL, 'level': args.level, 'rows': rows},
                                                            ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
