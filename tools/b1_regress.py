"""B1: прежние варианты агента считают как раньше — итоги прогона и сама карта образцов.

    ./px python tools/b1_regress.py                      # сверить нынешний код с tests/data/b1_reference.json
    ./px python tools/b1_regress.py --write <каталог с прежним кодом>   # записать эталон прежним кодом

Эталон записан кодом основной ветки до правки B1 (коммит 8a97164): 34 прогона прежних пресетов при базовых и
научных правилах. Сверяются итоговые метрики и отпечаток карты образцов в конце прогона и после каждого сбора —
то есть не только итог, но и то, что пересчёт карты после сбора дал те же числа.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / 'tests' / 'data' / 'b1_reference.json'
KEYS = ('score', 'samples_collected', 'returned', 'battery_left', 'distance', 'time', 'hazard_hits', 'false_collects')
CHECKS = [('adaptive_v2', None, 'hard', range(8001, 8009)),
          ('adaptive_v2', 'science', 'hard', range(8001, 8009)),
          ('adaptive_v2', None, 'medium', range(8001, 8004)),
          ('scientist_v2', 'science', 'hard', range(8001, 8005)),
          ('scientist_v2', None, 'hard', range(8001, 8003)),
          ('adaptive', None, 'hard', range(1001, 1004)),
          ('adaptive_v4', 'science', 'hard', range(8001, 8003)),
          ('adaptive_v3', None, 'hard', range(8001, 8003)),
          ('adaptive_cal_v2', None, 'hard', range(8001, 8003))]


def one(agent, rules, level, seed, config=None):
    """Метрики прогона и отпечатки карты образцов: после каждого сбора и в конце."""
    from did import agent as agent_mod
    from did.belief import SampleBelief
    from did.runner import run_episode
    marks, bots = [], []
    tick, collected = agent_mod.Agent.tick, SampleBelief.collected

    def tick_w(self, obs, io):
        if not bots:
            bots.append(self)
        return tick(self, obs, io)

    def collected_w(self, x, y):
        out = collected(self, x, y)
        marks.append(hashlib.sha1(self.p.tobytes()).hexdigest()[:12])
        return out

    agent_mod.Agent.tick, SampleBelief.collected = tick_w, collected_w
    try:
        m = run_episode(level, seed, agent, rules=rules, config=config, save=False)['metrics']
    finally:
        agent_mod.Agent.tick, SampleBelief.collected = tick, collected
    return {**{k: m[k] for k in KEYS}, 'collects': marks,
            'belief': hashlib.sha1(bots[0].belief.p.tobytes()).hexdigest()[:12]}


def key(agent, rules, level, seed):
    return f"{agent}@{rules or 'base'}/{level}-{seed}"


def compare(checks=CHECKS):
    """Список расхождений с эталоном (пустой — всё совпало) и число сверенных прогонов."""
    ref = json.loads(REF.read_text(encoding='utf-8'))
    bad, n = [], 0
    for agent, rules, level, seeds in checks:
        for seed in seeds:
            k = key(agent, rules, level, seed)
            got = json.loads(json.dumps(one(agent, rules, level, seed)))
            n += 1
            if got != ref[k]:
                bad.append((k, {f: (ref[k][f], got[f]) for f in got if got[f] != ref[k][f]}))
    return bad, n


def main():
    if '--write' in sys.argv:
        sys.path.insert(0, str(Path(sys.argv[sys.argv.index('--write') + 1]).resolve()))
        import did
        print('код:', Path(did.__file__).parent)
        out = {key(a, r, lv, s): one(a, r, lv, s) for a, r, lv, seeds in CHECKS for s in seeds}
        REF.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding='utf-8')
        print(f'записано прогонов: {len(out)}')
        return 0
    sys.path.insert(0, str(ROOT))
    bad, n = compare()
    for k, diff in bad:
        print('расхождение', k, diff)
    print(f'сверено прогонов: {n}, расхождений: {len(bad)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
