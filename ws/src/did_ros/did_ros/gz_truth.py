"""Показ скрытой правды в окне Gazebo: плоские модели-подсказки через сервисы мира.

Работает напрямую через gz-transport (питоновские привязки из того же окружения): запрос
create_multiple/remove занимает миллисекунды и не требует запуска `gz service` на каждую модель.
Отдельный поток сверяет желаемый набор моделей с тем, что реально есть в мире, поэтому
потерянный запрос или перезапуск судьи исправляются на следующем круге.

Привязки gz на время запроса не отпускают GIL. Живой мир отвечает за миллисекунду, а к миру,
который ещё не поднялся, поток не обращается; зависший мир задержит судью не больше чем на
TIMEOUT_MS за круг.
"""
import threading

from .truth_viz import PREFIX, sdf

TIMEOUT_MS = 300
PERIOD_S = 0.5
MAX_ROUNDS = 10      # столько кругов подряд пробуем привести мир к набору, потом сдаёмся


class GzTruth:

    def __init__(self, world='default', log=None):
        # Импорт здесь: без привязок gz судья работает, просто без подсказок в окне.
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.entity_factory_v_pb2 import EntityFactory_V
        from gz.msgs10.entity_pb2 import Entity
        from gz.msgs10.scene_pb2 import Scene
        from gz.transport13 import Node

        self._m = (Boolean, Empty, EntityFactory_V, Entity, Scene)
        self._node = Node()
        self._world = f'/world/{world}'
        self._log = log or (lambda text: None)
        self._want = {}                 # {имя модели: фигура}
        self._dirty = False
        self._rounds = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name='gz_truth', daemon=True)
        self._thread.start()

    def show(self, items):
        """Желаемый набор фигур; мир приводится к нему в фоне."""
        with self._lock:
            self._want = {s['name']: s for s in items}
            self._dirty = True
            self._rounds = 0
        self._wake.set()

    def close(self):
        """Убрать подсказки, если мир ещё жив: остановленный судья не оставляет чужую правду."""
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)
        if self._ready():
            for name in self._present() or ():
                if not self._remove(name, timeout_ms=200):
                    break

    # --- поток сверки ------------------------------------------------------------------------

    def _run(self):
        while not self._stop:
            self._wake.wait(PERIOD_S)
            self._wake.clear()
            if self._stop or not self._dirty or not self._ready():
                continue
            try:
                self._reconcile()
            except Exception as e:   # показ не должен ронять судью
                self._log(f'подсказки в Gazebo: {e!r}')

    def _reconcile(self):
        with self._lock:
            want = dict(self._want)
            self._dirty = False
        have = self._present()
        if have is None:
            self._dirty = True
            return
        stale = [n for n in have if n not in want]
        fresh = [s for n, s in want.items() if n not in have]
        if not stale and not fresh:
            return
        self._rounds += 1
        if self._rounds > MAX_ROUNDS:
            self._log(f'подсказки в Gazebo: мир не приводится к набору, не создано {len(fresh)}, '
                      f'не удалено {len(stale)}')
            return
        # Запросы выполняются на следующем шаге симуляции, поэтому результат проверяется
        # следующим кругом: набор считается приведённым, когда расхождений не осталось.
        self._dirty = True
        for name in stale:
            self._remove(name)
        if fresh:
            self._create(fresh)
        if self._rounds == 1:
            self._log(f'подсказки в Gazebo: +{len(fresh)} −{len(stale)}, в мире будет {len(want)}')

    # --- запросы к миру ----------------------------------------------------------------------

    def _ready(self):
        return f'{self._world}/create_multiple' in self._node.service_list()

    def _present(self):
        """Имена наших моделей в мире или None, если мир не ответил."""
        _, Empty, _, _, Scene = self._m
        ok, scene = self._node.request(f'{self._world}/scene/info', Empty(), Empty, Scene, TIMEOUT_MS)
        if not ok:
            return None
        return {m.name for m in scene.model if m.name.startswith(PREFIX)}

    def _create(self, items):
        Boolean, _, EntityFactory_V, _, _ = self._m
        req = EntityFactory_V()
        for s in items:
            f = req.data.add()
            f.sdf = sdf(s)
            f.name = s['name']
            f.pose.position.x, f.pose.position.y, f.pose.position.z = s['x'], s['y'], s['z']
            f.pose.orientation.w = 1.0
        ok, rep = self._node.request(f'{self._world}/create_multiple', req, EntityFactory_V, Boolean,
                                     TIMEOUT_MS)
        return bool(ok and rep.data)

    def _remove(self, name, timeout_ms=TIMEOUT_MS):
        Boolean, _, _, Entity, _ = self._m
        req = Entity()
        req.name = name
        req.type = Entity.MODEL
        ok, rep = self._node.request(f'{self._world}/remove', req, Entity, Boolean, timeout_ms)
        return bool(ok and rep.data)
