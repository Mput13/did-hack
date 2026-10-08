# DID Hack — автономный ИИ-исследователь на TurtleBot3

Агент управляет роботом в Gazebo: ищет скрытые образцы по датчику близости, оценивает стоимость
грунта по расходу батареи, расследует странности опытами, запоминает выясненное и возвращается на
базу. Подробное объяснение с рисунками и результатами — `explain.html` (открывается двойным щелчком).

## Запуск

Окружение (ROS 2 Jazzy, Gazebo Sim Harmonic, TurtleBot3) ставится из RoboStack через
[pixi](https://pixi.sh); версии закреплены в `pixi.lock`. Root не нужен. Платформы: macOS arm64, Linux x86-64.

```bash
curl -fsSL https://pixi.sh/install.sh | sh    # один раз
pixi run demo                                 # показ: окно Gazebo, судья, интерфейс и «пульт» (#/pilot)
pixi run demo --fast                          # то же без Gazebo, на быстром симуляторе
pixi run lab                                  # только интерфейс: http://127.0.0.1:8765
```

| Команда | Что делает |
|---|---|
| `pixi run exp E1` · `pixi run exp all` | пересчитать серию опытов или все |
| `pixi run run --level hard --seed 3 --agent adaptive` | один прогон в быстром симуляторе |
| `pixi run run --level hard --seed 3 --agent scientist --rules science` | исследователь на «научных» правилах |
| `pixi run gazebo-run --level hard --seed 1 [--gui] [--rules science --agent scientist]` | один прогон в Gazebo целиком |
| `pixi run stand-gui level:=hard seed:=3` + `pixi run agent-ros --level hard --seed 3` | стенд и агент в двух терминалах |
| `pixi run run --level medium --seed 1001 --agent adaptive_llm --llm codex` | планировщик на настоящей модели (`ollama` — локальная Qwen, `mock` — имитатор) |
| `pixi run check` | все автоматические проверки |
| `pixi run explain` | пересобрать `explain.html` со свежими данными |
| `pixi run smoke`, `pixi run smoke-judge`, `pixi run teleop` | проверки уровня 0 и ручное управление |

## Что где лежит

- `did/` — библиотека стенда (без ROS):
  `config.py` правила (базовые и набор `SCIENCE`), `scenario.py` генератор сценариев, `judge.py` судья,
  `fastsim.py` быстрый симулятор, `arena.py` карта, `nav.py` путь и ведение, `localize.py` поправка положения
  по лидару, `mapping.py` построение карты, `belief.py` картина мира агента, `energy.py` + `science.py` +
  `inquiry.py` модель расхода и расследования, `memory.py` память между прогонами, `planner.py` + `llm*.py`
  планировщик и языковая модель, `agent.py` агентский цикл, `baselines.py` базовые стратегии,
  `recorder.py` запись прогона, `metrics.py` + `experiments.py` серии опытов, `pilot.py` пульт для показа,
  `ros_agent.py` агент в ROS 2, `lab/server.py` сервер интерфейса.
- `ws/src/did_bringup`, `ws/src/did_ros` — launch-файлы и узел судьи для ROS 2 и Gazebo.
- `lab/` — веб-интерфейс «Лаборатория»: опыты, графики, проигрыватель, пульт.
- `experiments/*.yaml` — описания опытов; `runs/` — результаты и записи прогонов (в git не входят).
- `docs/explainer/` — исходники страницы-объяснения; `docs/demo_script.md` — сценарий показа.
- `presentation/` — слайды чекпоинтов.
- `tests/` — автоматические проверки.

## Координаты и правила

Робот стартует в (−2.0; −0.5) с нулевым курсом, `/odom` на старте равен (0; 0); положение уточняется
по лидару и карте. Интерфейс агента — топики и сервисы из условия задачи (`/cmd_vel`, `/scan`, `/odom`,
`/did/battery`, `/did/sample_sensor`, `/did/collect`, `/did/finish`, `/did/score`, `/did/events`).
Формула расхода, датчик, штрафы и счёт в условии не заданы — это допущения стенда, они собраны в
`did/config.py` и разобраны в `explain.html`.
