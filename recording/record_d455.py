#!/usr/bin/env python3
# ruff: noqa: BLE001, S110, S112, SIM115, DTZ005
"""Record synchronized RGB-D video sequences from Intel RealSense D455.

Saves sequences directly in TUM RGB-D format compatible with the Scene Graph pipeline:
  data/raw/<sequence_name>/
    |-- rgb/                  # Color images (PNG, BGR8)
    |-- depth/                # Metric depth images (16-bit PNG, millimeters, scale=1000)
    |-- rgb.txt               # TUM index: timestamp rgb/timestamp.png
    |-- depth.txt             # TUM index: timestamp depth/timestamp.png
    |-- groundtruth.txt       # Placeholder / odometry poses (timestamp tx ty tz qx qy qz qw)
    |-- accelerometer.txt     # IMU acceleration data (timestamp ax ay az)
    |-- camera_intrinsics.json # Factory calibration intrinsics
    +-- config.yaml           # Ready-to-run SceneGraph replay config

Usage:
  python tools/record_d455.py
  python tools/record_d455.py --name my_desk_01 --duration 20
  python tools/record_d455.py --name room_scan --no-preview
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


WIDTH = 640
HEIGHT = 480
FPS = 30


class AsyncFrameWriter:
    """Multi-threaded asynchronous disk writer to prevent frame drops during recording."""

    def __init__(self, max_queue_size: int = 150):
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.written_frames = 0
        self.worker_thread.start()

    def submit(self, rgb_path: str, rgb_img: np.ndarray, depth_path: str, depth_img: np.ndarray):
        """Submit a pair of frames for writing."""
        try:
            self.queue.put_nowait((rgb_path, rgb_img, depth_path, depth_img))
        except queue.Full:
            print("[Warning] Disk write queue full! Dropping frame from disk cache.")

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                rgb_path, rgb_img, depth_path, depth_img = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                cv2.imwrite(rgb_path, rgb_img)
                cv2.imwrite(depth_path, depth_img)
                self.written_frames += 1
            except Exception as e:
                print(f"[Error] Writing frame to disk failed: {e}")
            finally:
                self.queue.task_done()

    def stop_and_flush(self):
        """Signal stop and block until all queued frames are written."""
        self.stop_event.set()
        self.queue.join()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)


def generate_scenegraph_config(
    sequence_dir: Path,
    intrinsics: dict[str, Any],
    depth_scale: float = 1000.0,
) -> str:
    """Generate a ready-to-run SceneGraph YAML config for the recorded sequence."""
    rel_path = sequence_dir.as_posix()
    # Normalize for relative path from repo root
    if "data/" in rel_path:
        rel_root = "data/" + rel_path.split("data/")[-1]
    else:
        rel_root = str(sequence_dir)

    return f"""# Auto-generated SceneGraph config for recorded sequence: {sequence_dir.name}
# Recorded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

dataset:
  type: tum
  root: {rel_root}

camera:
  fx: {intrinsics['fx']:.4f}
  fy: {intrinsics['fy']:.4f}
  cx: {intrinsics['cx']:.4f}
  cy: {intrinsics['cy']:.4f}
  width: {intrinsics['width']}
  height: {intrinsics['height']}

depth:
  scale: {depth_scale:.1f}

reference_frame:
  type: camera
  up_axis: [0.0, -1.0, 0.0]
  heading_axis: [0.0, 0.0, 1.0]

tracking:
  confirmation:
    min_hits: 3
  occlusion:
    max_missing_seconds: 5.0
    max_missing_seconds_out_of_view: 10.0
  process:
    acceleration_std_mps2: 1.0
  measurement:
    position_std_m: 0.03
  gating:
    chi2_probability: 0.999
  association:
    max_distance_m: 0.45
    size_weight: 0.25
  history:
    observation_buffer_size: 10
  initialization:
    position_variance_m2: 0.02
    velocity_variance_m2s2: 1.0

temporal:
  relation:
    confirmation_threshold: 0.70
    contradiction_threshold: -0.70
    decay_per_second: 0.1
    unknown_after_seconds: 3.0
    lost_after_seconds: 6.0
    min_confirmation_hits: 1
    min_evidence_confidence: 0.05

perception:
  type: yoloe
  mode: prompt_free
  model: models/yoloe-26s-seg-pf.pt
  confidence_threshold: 0.25
  image_size: 480
  device: cpu

geometry:
  min_valid_points: 20
  min_depth_m: 0.20
  max_depth_m: 6.0
  measurement_noise_std_m: 0.03
  spatial_outlier_sigma: 3.0
  robust_depth:
    k: 2.5
  downsampling:
    voxel_size_m: 0.02

sync:
  rgb_depth_max_dt: 0.05
  rgb_pose_max_dt: 0.05
"""


def main():
    parser = argparse.ArgumentParser(description="Record synchronized RGB-D sequence with RealSense D455")
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Sequence directory name (default: d455_seq_<timestamp>)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/raw",
        help="Base directory to save recordings (default: data/raw)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 = record until Ctrl+C or 'q' is pressed)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV real-time visual preview window",
    )
    parser.add_argument(
        "--enable-imu",
        action="store_true",
        help="Enable IMU (accelerometer/gyroscope) recording (disabled by default to prevent USB stall issues)",
    )
    parser.add_argument(
        "--record-bag",
        action="store_true",
        help="Also record native RealSense ROS .bag file alongside TUM dataset",
    )
    parser.add_argument(
        "--preview-file",
        type=str,
        default=None,
        help="Optional path to write live preview image for browser dashboards",
    )
    args = parser.parse_args()

    if rs is None:
        print("[Error] pyrealsense2 is not installed! Run: pip install pyrealsense2")
        sys.exit(1)

    # 1. Setup sequence output directories
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_name = args.name if args.name else f"d455_seq_{timestamp_str}"
    seq_dir = Path(args.output_dir) / seq_name
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" Intel RealSense D455 RGB-D Sequence Recorder")
    print(f" Target Directory: {seq_dir.resolve()}")
    print(f" Resolution: {WIDTH}x{HEIGHT} @ {FPS} FPS | Color: BGR8 | Depth: Z16 (mm)")
    print("=" * 65)

    # 2. Configure RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    imu_enabled = False
    if args.enable_imu:
        try:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            imu_enabled = True
            print("[D455] IMU streams enabled (accel + gyro @ 200Hz)")
        except Exception as ex:
            print(f"[D455] Motion streams not available: {ex}")
    else:
        print("[D455] IMU streams disabled (pass --enable-imu to record motion data)")

    if args.record_bag:
        bag_path = str(seq_dir / "realsense_raw.bag")
        config.enable_record_to_file(bag_path)
        print(f"[D455] Recording native RealSense bag to: {bag_path}")

    # Start camera
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"[Error] Failed to start RealSense camera: {e}")
        print("Make sure no other program (like d455_sender or RealSense Viewer) is using the camera.")
        sys.exit(1)

    # Device info & USB connection
    device = profile.get_device()
    usb_type = device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown"
    print(f"[D455] Device: {device.get_info(rs.camera_info.name)} | USB Port: {usb_type}")
    if str(usb_type).startswith("2."):
        print("[Warning] Camera is running on USB 2.x! High-bandwidth operations may be limited. Connect to a USB 3.x port for best performance.")

    # Depth sensor scale
    depth_sensor = device.first_depth_sensor()
    depth_unit_m = depth_sensor.get_depth_scale()
    depth_scale = 1.0 / depth_unit_m  # Typically 1000.0 (mm)
    print(f"[D455] Depth scale: {depth_scale:.1f} integer units per meter ({depth_unit_m} m/unit)")

    # Color stream intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    raw_intrinsics = color_stream.get_intrinsics()
    intrinsics = {
        "width": raw_intrinsics.width,
        "height": raw_intrinsics.height,
        "fx": float(raw_intrinsics.fx),
        "fy": float(raw_intrinsics.fy),
        "cx": float(raw_intrinsics.ppx),
        "cy": float(raw_intrinsics.ppy),
        "distortion_model": str(raw_intrinsics.model),
        "coeffs": [float(c) for c in raw_intrinsics.coeffs],
        "depth_scale": float(depth_scale),
        "depth_unit_meters": float(depth_unit_m),
    }

    # Save camera intrinsics JSON
    with open(seq_dir / "camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"[D455] Saved camera intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")

    # Save SceneGraph config YAML
    config_yaml_content = generate_scenegraph_config(seq_dir, intrinsics, depth_scale)
    with open(seq_dir / "config.yaml", "w") as f:
        f.write(config_yaml_content)
    print(f"[D455] Generated ready-to-run SceneGraph config: {seq_dir / 'config.yaml'}")

    # Align depth to color stream
    align = rs.align(rs.stream.color)

    # Prepare index files
    rgb_txt = open(seq_dir / "rgb.txt", "w")
    depth_txt = open(seq_dir / "depth.txt", "w")
    assoc_txt = open(seq_dir / "associations.txt", "w")
    groundtruth_txt = open(seq_dir / "groundtruth.txt", "w")
    accel_txt = open(seq_dir / "accelerometer.txt", "w") if imu_enabled else None
    gyro_txt = open(seq_dir / "gyroscope.txt", "w") if imu_enabled else None

    rgb_txt.write("# RGB images\n# timestamp filename\n")
    depth_txt.write("# Depth images\n# timestamp filename\n")
    groundtruth_txt.write("# Camera trajectory (SYNTHETIC IDENTITY PLACEHOLDER - NO GROUND TRUTH AVAILABLE)\n# timestamp tx ty tz qx qy qz qw\n")
    if accel_txt:
        accel_txt.write("# Accelerometer samples\n# timestamp ax ay az\n")
    if gyro_txt:
        gyro_txt.write("# Gyroscope samples\n# timestamp wx wy wz\n")

    # Start async writer
    writer = AsyncFrameWriter(max_queue_size=200)

    print("\n>>> RECORDING STARTED!")
    if args.duration > 0:
        print(f">>> Will record for {args.duration:.1f} seconds. Press Ctrl+C or 'q' to stop early.")
    else:
        print(">>> Press Ctrl+C in terminal or 'q' / 'ESC' in preview window to STOP recording.")
    print("-" * 65)

    recorded_frames = 0
    t_start = time.time()
    last_log_time = t_start
    last_color_ts = -1.0
    last_depth_ts = -1.0

    try:
        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[D455] Target duration ({args.duration}s) reached.")
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as ex:
                if t_now - last_log_time >= 2.0:
                    print(f"\n[Warning] Frame wait timed out or failed: {ex}")
                continue

            raw_color = frames.get_color_frame()
            raw_depth = frames.get_depth_frame()
            if not raw_color or not raw_depth:
                continue

            try:
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            except Exception as ex:
                if t_now - last_log_time >= 2.0:
                    print(f"\n[Warning] Stream alignment failed: {ex}")
                continue

            if not color_frame or not depth_frame:
                continue

            # Preserved distinct hardware timestamps in seconds
            color_ts_sec = color_frame.get_timestamp() * 1e-3
            depth_ts_sec = depth_frame.get_timestamp() * 1e-3

            # Strictly enforce timestamp monotonicity (filter camera buffer repeat / duplicate frames)
            if color_ts_sec <= last_color_ts or depth_ts_sec <= last_depth_ts:
                continue
            last_color_ts = color_ts_sec
            last_depth_ts = depth_ts_sec

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())

            # File names matching exact sensor timestamps
            color_ts_str = f"{color_ts_sec:.6f}"
            depth_ts_str = f"{depth_ts_sec:.6f}"
            rgb_rel = f"rgb/{color_ts_str}.png"
            depth_rel = f"depth/{depth_ts_str}.png"
            rgb_abs = str(seq_dir / rgb_rel)
            depth_abs = str(seq_dir / depth_rel)

            # Submit for asynchronous background write
            writer.submit(rgb_abs, color_np, depth_abs, depth_np)

            # Write indices
            rgb_txt.write(f"{color_ts_str} {rgb_rel}\n")
            depth_txt.write(f"{depth_ts_str} {depth_rel}\n")
            assoc_txt.write(f"{color_ts_str} {rgb_rel} {depth_ts_str} {depth_rel}\n")
            groundtruth_txt.write(f"{color_ts_str} 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000\n")

            # Record 6-Axis IMU (accel + gyro) with individual hardware timestamps
            if imu_enabled:
                accel = frames.first_or_default(rs.stream.accel)
                gyro = frames.first_or_default(rs.stream.gyro)
                if accel and accel_txt:
                    accel_data = accel.as_motion_frame().get_motion_data()
                    accel_ts = accel.get_timestamp() * 1e-3
                    accel_txt.write(f"{accel_ts:.6f} {accel_data.x:.6f} {accel_data.y:.6f} {accel_data.z:.6f}\n")
                if gyro and gyro_txt:
                    gyro_data = gyro.as_motion_frame().get_motion_data()
                    gyro_ts = gyro.get_timestamp() * 1e-3
                    gyro_txt.write(f"{gyro_ts:.6f} {gyro_data.x:.6f} {gyro_data.y:.6f} {gyro_data.z:.6f}\n")


            recorded_frames += 1

            # Export live preview frame for dashboard if requested
            if args.preview_file and (recorded_frames % 2 == 0):
                try:
                    p_rgb = cv2.resize(color_np, (320, 240))
                    p_depth = cv2.resize(depth_np, (320, 240))
                    p_depth_vis = cv2.normalize(p_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    p_depth_cmap = cv2.applyColorMap(p_depth_vis, cv2.COLORMAP_JET)
                    p_comb = np.hstack((p_rgb, p_depth_cmap))
                    p_tmp = f"{args.preview_file}.tmp.jpg"
                    cv2.imwrite(p_tmp, p_comb)
                    import os
                    os.replace(p_tmp, args.preview_file)
                except Exception:
                    pass

            # Log progress every second
            if t_now - last_log_time >= 1.0:
                cur_fps = recorded_frames / elapsed if elapsed > 0 else 0.0
                q_size = writer.queue.qsize()
                mins, secs = divmod(int(elapsed), 60)
                sys.stdout.write(f"\r[REC] Time: {mins:02d}:{secs:02d} | Frames: {recorded_frames:05d} ({cur_fps:.1f} FPS) | Queue: {q_size:02d} | Disk: {writer.written_frames:05d}")
                sys.stdout.flush()
                last_log_time = t_now

            # Live OpenCV Preview
            if not args.no_preview:
                # Downsample for fast display
                preview_rgb = cv2.resize(color_np, (320, 240))
                preview_depth = cv2.resize(depth_np, (320, 240))

                # Normalize depth for colormap visualization
                depth_vis = cv2.normalize(preview_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Overlay status text
                hud_text = f"REC {int(elapsed)}s | {recorded_frames} frames | Press 'q' to stop"
                cv2.putText(preview_rgb, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                combined = np.hstack((preview_rgb, depth_colormap))
                cv2.imshow("RealSense D455 Recorder (Press 'q' or ESC to stop)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    print("\n[D455] Stop key pressed by user.")
                    break

    except KeyboardInterrupt:
        print("\n[D455] Recording interrupted by Ctrl+C.")
    finally:
        print("\n\n" + "=" * 65)
        print(" Finalizing recording and flushing writes to disk...")
        try:
            pipeline.stop()
        except Exception:
            pass

        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            rgb_txt.close()
            depth_txt.close()
            assoc_txt.close()
            groundtruth_txt.close()
            if accel_txt:
                accel_txt.close()
            if gyro_txt:
                gyro_txt.close()
        except Exception:
            pass


        # Wait for all images to write
        try:
            writer.stop_and_flush()
        except Exception as e:
            print(f"[Warning] Writer flush error: {e}")

        elapsed_total = time.time() - t_start
        actual_fps = recorded_frames / elapsed_total if elapsed_total > 0 else 0.0

        # Save metadata summary
        try:
            meta = {
                "sequence_name": seq_name,
                "recorded_at": timestamp_str,
                "duration_seconds": round(elapsed_total, 2),
                "total_frames": recorded_frames,
                "average_fps": round(actual_fps, 2),
                "resolution": [WIDTH, HEIGHT],
                "intrinsics": intrinsics,
            }
            with open(seq_dir / "dataset_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        print("=" * 65)
        print(" RECORDING COMPLETE!")
        print(f"  Sequence:       {seq_name}")
        print(f"  Location:       {seq_dir.resolve()}")
        print(f"  Total Frames:   {recorded_frames} frames in {elapsed_total:.1f}s ({actual_fps:.1f} FPS)")
        print("  Files Created:  rgb.txt, depth.txt, camera_intrinsics.json, config.yaml")
        print("=" * 65)
        print("\n>>> HOW TO RUN THIS RECORDED SEQUENCE IN SCENE GRAPH:")
        print("  In WSL2:")
        print(f"    bash run_demo.sh config_path:={seq_dir.as_posix()}/config.yaml")
        print("=" * 65)


if __name__ == "__main__":
    main()
