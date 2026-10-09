"""B1: отладочные сравнения на сценариях 1–80 — tools/p2_ablation.py с условиями, которые делят «научные» правила на части.

    ./px python tools/b1_dev.py 'v2=adaptive_v2:science:{}' 'rep=adaptive_v2:science:{"exact_replay":true}' --cond=sci_nofaults

Научные правила меняют сразу несколько вещей (расход на повороты и груз, шум оценки заряда, сбои от опасных зон,
три вида сбоя датчика). Условия ниже включают их по частям, чтобы увидеть, с чем связан вред пересчёта карты.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p2_ablation      # noqa: E402

p2_ablation.CONDITIONS.update({
    # научные правила без сбоев: остаются расход на повороты и груз и шум оценки заряда
    'sci_nofaults': {'rules': {'faults': False}},
    # то же, и событий среды по расписанию нет
    'sci_nofaults_noev': {'rules': {'faults': False}, 'scenario': {'events': []}},
    # научные правила, но сбои — только утечка заряда (датчик образцов исправен), сбоя датчика по расписанию нет
    'sci_leak_only': {'scenario': {'fault_kinds': ['leak'], 'events': ['soil_change', 'new_hazard']}},
    # научные правила, сбои датчика есть, утечки нет
    'sci_sensor_only': {'scenario': {'fault_kinds': ['sensor_noise', 'sensor_stuck', 'sensor_bias']}},
    # по одному виду сбоя датчика от зон (сбой по расписанию остаётся любым)
    'sci_noise': {'scenario': {'fault_kinds': ['sensor_noise']}},
    'sci_stuck': {'scenario': {'fault_kinds': ['sensor_stuck']}},
    'sci_bias': {'scenario': {'fault_kinds': ['sensor_bias']}},
    # научные правила без опасных зон: сбой датчика по расписанию (шум, залипание или сдвиг) остаётся
    'sci_nozones': {'scenario': {'n_hazards': 0, 'events': ['soil_change', 'sensor_fault']}},
    # базовые правила плюс одна из частей научных (правила ставить пустыми)
    'b_rad': {'rules': {'drain_per_rad': 0.12}},
    'b_load': {'rules': {'load_drain': 0.05}},
    'b_bsig': {'rules': {'battery_sigma': 0.02}},
    'b_hit0': {'rules': {'hazard_battery_hit': 0.0}},
    # базовые правила плюс только сбои (с базовыми ставить правила пустыми: 'v2=adaptive_v2::{}')
    'faults_only': {'rules': {'faults': True, 'hazard_battery_hit': 0.0}},
})

if __name__ == '__main__':
    p2_ablation.main()
