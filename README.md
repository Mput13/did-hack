# DID Hack — автономный ИИ-исследователь на TurtleBot3

Агент ставит эксперимент на роботе в Gazebo: ищет скрытые образцы по датчику близости,
оценивает «грунты» по расходу батареи, планирует маршрут и возвращается на базу.

## Окружение

ROS 2 Jazzy, Gazebo Sim (Harmonic) и TurtleBot3 ставятся из RoboStack через [pixi](https://pixi.sh),
версии закреплены в `pixi.lock`. Root не нужен. Платформы: linux-64, osx-arm64.

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # один раз
pixi run sim        # мир turtlebot3_world и Burger без окна Gazebo; первый запуск скачает окружение (~1 ГБ) и соберёт ws/
pixi run smoke      # во втором терминале: проверка уровня 0
pixi run sim-gui    # то же с окном Gazebo
pixi run teleop     # ручное управление с клавиатуры
```

На Linux без дисплея `pixi run sim` сам включает рендер лидара через EGL, Xvfb не нужен.

## Структура

- `ws/src/did_bringup` — launch-файлы
- `maps/` — карта арены из `turtlebot3_navigation2`: 384×384, 0.05 м/клетка, начало (−10; −10)
- `tools/smoke_sim.py` — проверка уровня 0: данные идут, робот слушается `/cmd_vel`
- `env/activate.sh` — переменные окружения (модель робота, сеть только внутри машины)
- `presentation/` — слайды чекпоинта 1. Пересборка: `python3 presentation/figures/make_figures.py`,
  затем `node presentation/build_deck.js` (нужен `npm install` в `presentation/`)

## Координаты

Робот стартует в (−2.0; −0.5) с нулевым курсом, `/odom` на старте равен (0; 0):
`x_world = −2.0 + odom.x`, `y_world = −0.5 + odom.y`.
