#!/usr/bin/env bash
# Production live runner for Intel RealSense D455 with RTAB-Map SLAM and SceneGraph
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source ROS 2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Source workspace install
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
fi

# Detect Python virtualenv and dependencies
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"
EXTRA_SITE_PACKAGES=""
if [ -n "$VIRTUAL_ENV" ] && [ -d "$VIRTUAL_ENV/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$VIRTUAL_ENV/lib/python$PY_VER/site-packages"
elif [ -d "$HOME/myenv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$HOME/myenv/lib/python$PY_VER/site-packages"
elif [ -d "$SCRIPT_DIR/.venv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$SCRIPT_DIR/.venv/lib/python$PY_VER/site-packages"
elif [ -d "$SCRIPT_DIR/myenv/lib/python$PY_VER/site-packages" ]; then
    EXTRA_SITE_PACKAGES=":$SCRIPT_DIR/myenv/lib/python$PY_VER/site-packages"
fi

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/ros2_ws/src/scene_graph_ros:$SCRIPT_DIR/ros2_ws/src/d455_bridge${EXTRA_SITE_PACKAGES}:$PYTHONPATH"

# Set display for WSLg GUI window rendering
export DISPLAY="${DISPLAY:-:0}"
if [ -z "$WAYLAND_DISPLAY" ] && [ -e "/mnt/wslg/runtime-dir/wayland-0" ]; then
    export WAYLAND_DISPLAY="wayland-0"
fi

cd "$SCRIPT_DIR"

REPLAY_PATH=""
EXPORT_DIR=""
REPLAY_RATE="1.0"
REPLAY_EXTRA_ARGS=()
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --replay)
            REPLAY_PATH="$2"
            shift 2
            ;;
        --export-dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --rate)
            REPLAY_RATE="$2"
            shift 2
            ;;
        --start-index)
            REPLAY_EXTRA_ARGS+=("--start-index" "$2")
            shift 2
            ;;
        --end-index)
            REPLAY_EXTRA_ARGS+=("--end-index" "$2")
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$REPLAY_PATH" ]; then
    echo "Starting Intel RealSense D455 Live Runtime"
    echo "Pipeline: RealSense D455 -> RTAB-Map SLAM -> SceneGraph"
    echo "Time Domain: System wall clock (use_sim_time=false)"
    echo "Freshness: High-accuracy live SLAM (pose_max_age=80ms)"
    echo "Visualization: RViz2 (Window 3) + 2D Detections & Tracks (Windows 1 & 2)"

    exec ros2 launch scene_graph_ros live_scene_graph.launch.py \
        use_bridge:=false \
        use_rtabmap:=true \
        use_scenegraph:=true \
        use_rviz:=true \
        use_viewer:=true \
        config_path:=configs/runtime/live_d455.yaml \
        localization_mode:=slam \
        "${PASSTHROUGH_ARGS[@]}"
else
    RUN_DIR="${EXPORT_DIR:-outputs/run_d455_replay}"
    echo "================================================================="
    echo "Starting Reproducible End-to-End Replay Pipeline"
    echo "Dataset: $REPLAY_PATH"
    echo "Output:  $RUN_DIR"
    echo "Rate:    ${REPLAY_RATE}x"
    echo "================================================================="

    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/slam" "$RUN_DIR/visualizations"

    # Start live scene graph + RTAB-Map SLAM pipeline
    ros2 launch scene_graph_ros live_scene_graph.launch.py \
        use_bridge:=false \
        use_rtabmap:=true \
        use_scenegraph:=true \
        use_rviz:=false \
        use_viewer:=false \
        config_path:=configs/runtime/live_d455.yaml \
        localization_mode:=slam \
        database_path:="$RUN_DIR/slam/rtabmap.db" \
        export_dir:="$RUN_DIR" \
        telemetry_log_path:="$RUN_DIR/slam/telemetry.csv" \
        "${PASSTHROUGH_ARGS[@]}" > "$RUN_DIR/logs/launch.log" 2>&1 &
    LAUNCH_PID=$!

    cleanup() {
        echo "Finalizing pipeline and shutting down nodes (PID $LAUNCH_PID)..."
        kill -INT "$LAUNCH_PID" 2>/dev/null || true
        wait "$LAUNCH_PID" 2>/dev/null || true
    }
    trap cleanup EXIT

    echo "Waiting for SLAM and SceneGraph nodes to initialize..."
    for i in {1..120}; do
        if grep -q "SceneGraphROSNode initialized with worker thread" "$RUN_DIR/logs/launch.log" 2>/dev/null; then
            echo "SceneGraphROSNode is initialized and ready!"
            break
        fi
        sleep 1
    done
    sleep 3

    # Run reproducible replay
    echo "Streaming RGB-D sequence..."
    python3 "$SCRIPT_DIR/recording/replay_rgbd_ros.py" \
        --root "$REPLAY_PATH" \
        --rate "$REPLAY_RATE" \
        "${REPLAY_EXTRA_ARGS[@]}"

    echo "Replay stream completed. Allowing queue to flush..."
    sleep 8

    # Capture topic diagnostics
    echo "Capturing topic diagnostics..."
    ros2 topic list > "$RUN_DIR/logs/topic_rates.txt" 2>&1 || true

    # Capture TF tree
    echo "Capturing TF frames..."
    timeout 5 ros2 run tf2_tools view_frames --output-file "$RUN_DIR/logs/tf_frames" 2>/dev/null || true

    # Stop launch process gracefully so destroy_node() exports artifacts
    echo "Stopping ROS nodes gracefully..."
    trap - EXIT
    pkill -TERM -P "$LAUNCH_PID" 2>/dev/null || true
    kill -TERM "$LAUNCH_PID" 2>/dev/null || true
    sleep 4
    kill -KILL "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true

    # Export map from RTAB-Map SQLite DB if created
    if [ -f "$RUN_DIR/slam/rtabmap.db" ] && [ -x "/opt/ros/humble/bin/rtabmap-export" ]; then
        echo "Exporting RTAB-Map point cloud and poses..."
        /opt/ros/humble/bin/rtabmap-export \
            --output "$RUN_DIR/slam" \
            --poses \
            --poses_camera \
            --poses_format 10 \
            --voxel 0.02 \
            "$RUN_DIR/slam/rtabmap.db" > "$RUN_DIR/logs/rtabmap_export.log" 2>&1 || true
    fi

    # Save run manifest
    cat <<EOF > "$RUN_DIR/manifest.json"
{
  "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "dataset": "$REPLAY_PATH",
  "run_dir": "$RUN_DIR",
  "rate": "$REPLAY_RATE",
  "localization_mode": "slam",
  "pipeline": "D455 RGB-D -> RTAB-Map SLAM -> YOLOE -> 3D Geometry -> CausalTracker -> TemporalSceneGraph"
}
EOF

    echo "================================================================="
    echo "End-to-End Replay Run Completed Successfully!"
    echo "Artifacts saved in: $RUN_DIR"
    echo "================================================================="
fi
