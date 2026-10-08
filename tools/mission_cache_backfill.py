"""R13: дописать в кэш отказы модели из уже сделанных записей прогонов. Сеть не используется.

Раньше кэш хранил только непустые ответы: пустой ответ и обрыв связи в него не попадали, и строгий повтор
(«только кэш») такого прогона останавливался на первом же отказе. Отказ при этом записан в самом прогоне
(поле llm: response пуст или None, errors). Скрипт повторяет прогон строго из кэша и, дойдя до запроса без
пары, сверяется с записью: если там на этом месте отказ — кладёт его в кэш (<хэш>.fail.json) и идёт дальше;
если там обычный ответ — это настоящий промах, ячейка остаётся невоспроизводимой.

    ./px python tools/mission_cache_backfill.py            # все ячейки плана, где в записи есть отказ без ответа
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from did.llm import CacheMiss, ChatReply, LLMError                       # noqa: E402
from did.llm_cache import CachedClient, OfflineClient                    # noqa: E402
from did.mission_criteria import MISSIONS                                # noqa: E402
from did.recorder import load_trace                                      # noqa: E402
from did.runner import run_episode                                       # noqa: E402
from mission_eval import EXPECTED, EXPERIMENT, MODEL, VARIANTS, cell_id, cell_path    # noqa: E402

CUT = 'ответ обрезан по лимиту токенов'


class Backfill(CachedClient):
    """Строгий кэш, который на промахе берёт отказ из записи прогона и сохраняет его в кэш."""

    def __init__(self, recorded):
        super().__init__(OfflineClient(MODEL, use_schema=True), strict=True)
        self.recorded = recorded           # обмены исходного прогона по порядку вызовов
        self.n = 0
        self.added = []

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        i, self.n = self.n, self.n + 1
        was = self.recorded[i] if i < len(self.recorded) else None
        if was is None:
            raise CacheMiss(f'повтор сделал вызов № {i + 1}, а в записи их {len(self.recorded)}: прогон пошёл иначе')
        try:
            reply = super().chat(messages, temperature=temperature, max_tokens=max_tokens, schema=schema)
        except CacheMiss:
            key = self.cache.key(self.model, messages, temperature, max_tokens, schema if self.use_schema else None)
            if was.get('ok') or was.get('response'):
                raise                       # в записи здесь обычный ответ, а в кэше его нет: настоящий промах
            if was.get('response') is None:
                self._keep_failure(key, messages, error=was['errors'][0], attempts=was.get('http_attempts') or 0,
                                   source='запись прогона')
            else:
                cut = any(CUT in e for e in was.get('errors') or [])
                self._keep_failure(key, messages, text='', latency_ms=was.get('latency_ms') or 0,
                                   usage=was.get('usage') or {}, finish_reason='length' if cut else None,
                                   attempts=was.get('http_attempts') or 0, source='запись прогона')
            self.added.append((i, was.get('t'), was['errors']))
            return super().chat(messages, temperature=temperature, max_tokens=max_tokens, schema=schema)
        except LLMError:
            if was.get('response') is not None:
                raise CacheMiss(f'вызов № {i + 1}: в кэше отказ, а в записи ответ') from None
            raise
        if (reply.text or '') != (was.get('response') or ''):
            raise CacheMiss(f'вызов № {i + 1}: ответ в кэше не тот, что в записи прогона')
        return ChatReply(reply.text, reply.latency_ms, reply.usage, reply.attempts, reply.finish_reason, cached=True)


def main():
    total = 0
    for task in EXPECTED:
        m_id, variant, level, seed = task
        agent, llm, cfg = VARIANTS[variant]
        path = cell_path(EXPERIMENT, task)
        if not llm or not path.is_file():
            continue
        recorded = load_trace(path).get('llm') or []
        if not any(not ex.get('ok') and not ex.get('response') for ex in recorded):
            continue
        client = Backfill(recorded)
        try:
            run_episode(level, seed, agent, experiment='_backfill', arm=f'{m_id}_{variant}',
                        config={'mission': MISSIONS[m_id]['text'], **cfg}, llm={'client': client}, save=False)
        except CacheMiss as e:
            print(f'{cell_id(task)}: не воспроизводится из кэша — {e}')
            continue
        total += len(client.added)
        for i, t, errors in client.added:
            print(f'{cell_id(task)}: вызов № {i + 1} ({t} с) — отказ записан в кэш: {errors[0][:90]}')
        if not client.added:
            print(f'{cell_id(task)}: отказы уже в кэше')
    print(f'Отказов дописано в кэш: {total}; обращений к сети: 0')


if __name__ == '__main__':
    main()
