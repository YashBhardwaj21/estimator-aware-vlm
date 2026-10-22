#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/ros2_ws/src/scene_graph_ros:$SCRIPT_DIR/ros2_ws/src/d455_bridge:$PYTHONPATH"

exec ros2 run d455_bridge d455_receiver
