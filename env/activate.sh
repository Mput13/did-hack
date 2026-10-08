# Окружение проекта поверх ROS 2 из pixi: модель робота, сеть, собранный воркспейс.
export TURTLEBOT3_MODEL=burger

# ROS 2 и Gazebo общаются только внутри машины: соседние команды в общей сети не видят наши
# топики. Сам ROS_AUTOMATIC_DISCOVERY_RANGE оставляет сокеты DDS на 0.0.0.0, поэтому профиль
# Fast DDS дополнительно привязывает их к 127.0.0.1 (VM разработки стоит на публичном IP).
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export FASTRTPS_DEFAULT_PROFILES_FILE="$PIXI_PROJECT_ROOT/env/fastdds_loopback.xml"
export GZ_IP=127.0.0.1

# Библиотека стенда did/ лежит в корне проекта и нужна и скриптам, и узлам ROS 2.
export PYTHONPATH="$PIXI_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$PIXI_PROJECT_ROOT/ws/install/setup.bash" ]; then
    . "$PIXI_PROJECT_ROOT/ws/install/setup.bash"
fi
