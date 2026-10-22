# VLM Subsystem: Integrated Perception & Navigation Modules

This directory contains the self-contained Vision-Language Model (VLM) subsystem, together with all essential upstream and downstream perception modules copied from the main workspace.

> **Note on Datasets and Models**: As per architecture requirements, large raw datasets (`data/`), model weights (`models/`), and generated outputs (`outputs/`) remain preserved at their canonical repository locations and are referenced directly without duplicate copying.

---

## 1. Directory Structure

```
vlm/
├── recording/              # 1. Live D455 capture, diagnostics, and bag extraction
├── slam/                   # 2. Visual SLAM trajectories, evaluation, and configs
├── mapping/                # 3. 2D/3D semantic mapping and multi-session fusion
├── navigation/             # 4. Autonomous A* path planning & web visualizer
├── configs/                # 5. Sensor, runtime, estimation, and navigation configs
├── tools/                  # 6. Evaluation benchmarks, diagnostics, and reports
├── src/
│   ├── scene_graph/        # 7. Core perception engine (YOLOE, geometry, tracking)
│   ├── stage5_adapter.py   # Multi-session evidence integration
│   ├── selective_router.py # Estimator-aware utility calculation U(q)
│   └── ...                 # Additional VLM policy modules
├── docs/                   # System documentation & evaluation protocols
├── benchmarks/             # Versioned manifests and ground-truth evaluations
└── tests/                  # Automated regression suite
```

---

## 2. Module Inventory & Key Files

### 1. Live Data Recording (`vlm/recording/`)
- `record_d455.py`: High-speed synchronized RealSense D455 RGB-D + IMU recording pipeline.
- `check_live_d455.py` & `check_d455_hardware.py`: USB 3.0 throughput and stream integrity verification.
- `check_d455_imu.py`: BMI085 IMU gyroscope drift and polarity test script.
- `convert_ros2_bag_to_rgbd.py`: Converts ROS 2 `.db3` bag recordings into synchronized PNG image sequences.
- `replay_rgbd_ros.py`: Replays recorded RGB-D sequences back to ROS 2 topics.
- `validate_recording.py`: Validates frame timestamp monotonicity and drops.
- `record_video.py`: Headless MP4 video stream recorder.
- `run_live_d455.sh` & `start_d455_receiver.sh`: Linux deployment scripts for Raspberry Pi 5.

### 2. SLAM & Trajectory Estimation (`vlm/slam/`)
- `trajectory_manager.py`: Parses TUM format trajectories (`CameraTrajectory.txt`), validates pose alignment, and performs nearest-neighbor timestamp matching ($\le 35\text{ ms}$).
- `rtabmap.py`: Ingests RTAB-Map graph nodes, odometry constraints, and landmark graph vertices.
- `evaluate_trajectory.py`: Computes Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) against ground truth.
- `d455_orbslam3.yaml` & `d455_desk_run_161552_orbslam3.yaml`: Camera calibration and feature tracking parameters for ORB-SLAM3.

### 3. Semantic Mapping & Multi-Session Fusion (`vlm/mapping/`)
- `run_offline_semantic_mapping.py`: **Primary pipeline generator**. Ingests RGB-D sequences + SLAM trajectory, runs YOLOE object detection, tracks 2D boxes, unprojects depth into 3D point clouds, fits ground floor plane, and produces `object_observations.jsonl`.
- `fuse_multi_runs.py`: Solves inter-session $\text{SE}(2)$ registration via consensus RANSAC, detects physical object displacements, and generates the fused 3D scene graph and metric occupancy grid.
- `run_mapping_pipeline.py`: Automated multi-step runner across raw recordings.
- `generate_clean_floor_plan.py`: Renders publication-grade 2D architectural floor plans.
- `visualize_complete_room_context.py`: Multi-run camera viewpoint visualizer.
- `clean_recording_02.py`: Preprocessing utility for Session 2 recordings.

### 4. Autonomous Navigation & Chassis Control (`vlm/navigation/`)
- `nav.py`: Lightweight autonomous navigation stack using 8-connected grid A* path planning on the metric occupancy map, RealSense D455 200 Hz hardware gyro closed-loop heading, and forward depth obstacle avoidance ($< 0.40\text{ m}$).
- `live_nav_server.py`: Real-time web server (port 8080) with live camera feed, interactive map visualizer, telemetry HUD, and manual/autonomous dispatch buttons.
- `generate_desk18_navigation_map.py`: Generates landmark navigation submaps.
- `generate_sequential_runs_visual.py`: Multi-run visual path and trajectory comparison tool.

### 5. Perception Configuration Files (`vlm/configs/`)
- `default.yaml`: Top-level perception and scene graph parameters.
- `sensors/realsense_d455.yaml`: D455 stream resolutions (640x480), frame rates (15/30 fps), and IMU settings.
- `runtime/default.yaml` & `runtime/live_d455.yaml`: Pipeline execution rates, queue depths, and buffer thresholds.
- `navigation/motor_interface_config.yaml`: L298N 4WD pinout, PWM duty cycles, and calibrated linear/angular velocities.
- `estimation/`: ORB-SLAM3 feature detector, vocabulary, and calibration profiles.

### 6. Tools & Utilities (`vlm/tools/`)
- `evaluation/`: `evaluate_semantic_perception.py` (YOLO precision/recall metrics) and `compare_ros_regression.py`.
- `diagnostics/`: `sync_quality_report.py` and `view_graph_logs.py`.
- `maintenance/`: `class_distribution_report.py`.

### 7. Core Spatial Engine (`vlm/src/scene_graph/`)
Directly accessible from `vlm/src/`:
- `data/`: Dataset discovery, frame packet synchronization, and sensor frame representations.
- `geometry/`: Camera pinhole back-projection, RANSAC floor leveling (`floor_alignment.py`), and 3D point cloud downsampling.
- `perception/`: `YOLOEDetector` open-vocabulary 23-class object detection and mask generation.
- `tracking/`: Multi-frame 3D object association and lifespan tracking (`causal_tracker.py`).
- `mapping/`: `GlobalObjectAssociator` and `OccupancyGrid` generation.
- `relations/`: Spatial predicates (support, containment, proximity, occlusion).
- `graph/`: Dynamic temporal 3D scene graph data structures (`temporal_graph.py`).
