"""Общее для опытов L3 (языковые модели МАИ): клиент с кэшем, запуск ячеек, доли с интервалами.

Модели — только сервер МАИ (адрес и ключ в .env основного каталога). Все ответы кладутся в runs/_llm_cache,
поэтому повтор с --cache-only идёт без сети и даёт те же числа.
"""
import json
import math
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import CacheMiss, load_env, make_client          # noqa: E402
from did.runner import RUNS                                    # noqa: E402

ENV_FILE = '/Users/a/MAI/DID/.env'
MAIN = 'qwen3.8-flash-next'
MODELS = (MAIN, 'DeepSeek-V4-Flash', 'qwen3.6-35b-a3b')
# Запас на рассуждение 3000 токенов и срок 90 с — как в L2, R13 и R3.
LLM = {'kind': 'http', 'use_schema': True, 'min_tokens': 3000, 'timeout_s': 90}
_lock = threading.Lock()


def llm_opts(model, cache_only=False):
    return {**LLM, 'model': model, 'cache': 'only' if cache_only else True}


def client(model, cache_only=False):
    if not cache_only:
        load_env(ENV_FILE)
    return make_client(**llm_opts(model, cache_only))


def slug(model):
    return model.replace('/', '-').replace(':', '-')


# --- запуск ячеек --------------------------------------------------------------------------------

def run_cells(cells, fn, jobs, label=str):
    """fn(cell) по всем ячейкам в jobs потоков (прогон в основном ждёт сеть). Возвращает {cell: исход или None}.

    Ошибка ячейки не роняет серию: она печатается и ячейка остаётся без результата.
    """
    done = {}

    def one(cell):
        t0 = time.time()
        try:
            out = fn(cell)
        except CacheMiss as e:
            print(f'[{label(cell)}] нет в кэше: {e}', file=sys.stderr, flush=True)
            return cell, None
        except Exception as e:                      # noqa: BLE001
            traceback.print_exc()
            print(f'[{label(cell)}] ОШИБКА: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            return cell, None
        print(f'[{label(cell)}] готово за {time.time() - t0:.0f} с', flush=True)
        return cell, out

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for cell, out in pool.map(one, cells):
            done[cell] = out
    return done


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


# --- статистика ----------------------------------------------------------------------------------

def share(k, n):
    """Доля k из n с 95% интервалом Уилсона."""
    if not n:
        return {'k': 0, 'n': 0, 'share': None, 'ci': None}
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {'k': int(k), 'n': int(n), 'share': round(p, 3), 'ci': [round(max(0.0, centre - half), 3),
                                                                   round(min(1.0, centre + half), 3)]}


def share_diff(k1, n1, k2, n2):
    """Разность двух независимых долей (первая − вторая) с 95% интервалом (Ньюкомб, по интервалам Уилсона)."""
    a, b = share(k1, n1), share(k2, n2)
    if a['share'] is None or b['share'] is None:
        return None
    d = a['share'] - b['share']
    lo = d - math.hypot(a['share'] - a['ci'][0], b['ci'][1] - b['share'])
    hi = d + math.hypot(a['ci'][1] - a['share'], b['share'] - b['ci'][0])
    return {'diff': round(d, 3), 'ci': [round(lo, 3), round(hi, 3)]}


def mean_ci(values, seed=0):
    """Среднее и 95% интервал (бутстреп)."""
    v = np.array([float(x) for x in values if x is not None])
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(4000, len(v)), replace=True).mean(axis=1) if len(v) > 1 else np.array([v[0]] * 2)
    return {'n': int(len(v)), 'mean': round(float(v.mean()), 3),
            'ci': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}


def paired_diff(a, b, seed=0):
    """Парная разность a − b по общим ключам словарей {ключ: число}: среднее, 95% интервал, кто чаще выше."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if not keys:
        return None
    d = [float(a[k]) - float(b[k]) for k in keys]
    out = mean_ci(d, seed)
    out.update(a_higher=sum(x > 1e-9 for x in d), b_higher=sum(x < -1e-9 for x in d),
               ties=sum(abs(x) <= 1e-9 for x in d))
    return out


def paired_binary(a, b):
    """Парные исходы да/нет на одних объектах: доли, разность a − b с 95% интервалом (бутстреп по объектам),
    и сколько раз верен только один из двух (точный знаковый критерий)."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if not keys:
        return None
    x, y = np.array([bool(a[k]) for k in keys], float), np.array([bool(b[k]) for k in keys], float)
    d = x - y
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(4000, len(d)), replace=True).mean(axis=1)
    only_a, only_b = int((d > 0).sum()), int((d < 0).sum())
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {'n': len(keys), 'a': share(int(x.sum()), len(keys)), 'b': share(int(y.sum()), len(keys)),
            'diff': round(float(d.mean()), 3),
            'ci': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            'only_a': only_a, 'only_b': only_b, 'sign_p': round(p, 4)}


def exchange_stats(exchanges):
    """Сводка по обменам с моделью: запросы, годные сразу и после исправления, отказы, время ответа.

    Время — по ответам, пришедшим за одну попытку: от 1,5 с (быстрее отвечает только кэш самого сервера) до срока
    ответа 90 с. Ответ с временем больше срока — это таймаут и повтор транспорта; такие считаются отдельно
    (retried), а в медиану не идут. no_answer — запросы, на которые сервер не ответил вовсе (не ошибка модели).
    """
    from did.llm import error_kind, llm_stats
    st = llm_stats(exchanges)
    n = max(1, st['requests'])
    got = [int(ex.get('latency_ms') or 0) for ex in exchanges if ex.get('response')]
    ms = sorted(x for x in got if 1500 <= x <= 90000)
    lost = sum(1 for ex in exchanges if ex.get('response') is None
               and any(error_kind(str(e)) == 'нет ответа модели' for e in ex.get('errors') or []))
    return {'requests': st['requests'], 'exchanges': st['exchanges'], 'first_ok': st['first_ok'],
            'repaired': st['repaired'], 'failed': st['failed'], 'no_answer': lost,
            'first_ok_share': round(st['first_ok'] / n, 3), 'failed_share': round(st['failed'] / n, 3),
            'errors': st['errors'], 'tokens': st['tokens'], 'retried': sum(x > 90000 for x in got),
            'latency_s': {'median': round(ms[len(ms) // 2] / 1000, 1) if ms else None,
                          'mean': round(sum(ms) / len(ms) / 1000, 1) if ms else None,
                          'max': round(ms[-1] / 1000, 1) if ms else None, 'n': len(ms)}}


def fmt_share(s, digits=0):
    if not s or s['share'] is None:
        return '—'
    return f"{s['share'] * 100:.{digits}f}% ({s['k']} из {s['n']}; {s['ci'][0] * 100:.0f}–{s['ci'][1] * 100:.0f}%)"


def gazebo_running():
    import subprocess
    return subprocess.run(['pgrep', '-f', 'gz sim'], capture_output=True).returncode == 0


def wait_for_idle():
    """Перед серией: пока идёт Gazebo или стоит пауза — ждать (другой инженер гоняет сверку)."""
    pause = Path('/Users/a/MAI/DID/research/PAUSE')
    while gazebo_running() or pause.exists():
        print('Gazebo идёт или стоит пауза — жду 60 с', flush=True)
        time.sleep(60)
