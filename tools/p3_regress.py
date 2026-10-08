"""Прежние варианты агента дают прежние числа: сверка прогонов ветки P3 со сводками опытов основного каталога.

    ./px python tools/p3_regress.py            # 20 прогонов E1, 20 прогонов adaptive_v2 и 5 scientist_v2 из E22b

Проверка та же, что tools/p2_regress.py (счёт, образцы, возврат, остаток заряда, путь, время и штрафы — до
последней цифры сводки), набор прогонов — по заданию P3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p2_regress      # noqa: E402

p2_regress.CHECKS = [('E1', 'adaptive', 'base', 'hard', range(1001, 1011)),
                     ('E1', 'adaptive', 'base', 'medium', range(1001, 1006)),
                     ('E1', 'fixed', 'base', 'hard', range(1001, 1006)),
                     ('E22b', 'adaptive_v2', 'base', 'hard', range(8001, 8011)),
                     ('E22b', 'adaptive_v2', 'base', 'medium', range(8001, 8006)),
                     ('E22b', 'adaptive_v2', 'science', 'hard', range(8001, 8006)),
                     ('E22b', 'scientist_v2', 'science', 'hard', range(8001, 8006))]

if __name__ == '__main__':
    sys.exit(p2_regress.main())
