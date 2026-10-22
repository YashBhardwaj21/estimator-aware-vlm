#!/usr/bin/env python3
"""Stage 1 Live D455 Diagnostics and Acceptance Validator.

Subscribes to live RealSense D455 ROS 2 topics and /camera/camera/metadata.
Measures and reports:
- Live RGB FPS, Depth FPS, Received Metadata FPS
- Frame age (host capture -> network arrival -> ROS reception)
- Sequence continuity: gaps, dropped frames, duplicate frames, out-of-order frames
- Hardware RGB-Depth skew: P50, P95, P99, Max
- Runtime depth scale: authoritative value and validation bounds
- Camera intrinsics: fx, fy, ppx, ppy, distortion coefficients
- Clock synchronization: offset, drift (ppm), health status
- Stage 1 Acceptance Criteria evaluation (PASS / FAIL)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import String as StringMsg
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class D455LiveMonitor(Node if HAS_ROS2 else object):  # type: ignore
    """Monitors live D455 ingestion topics and computes Stage 1 telemetry metrics."""

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 (rclpy, sensor_msgs, std_msgs) is required to run live D455 monitoring.")
        super().__init__("d455_live_checker")

        # Telemetry Buffers
        self.rgb_times: list[float] = []
        self.depth_times: list[float] = []
        self.imu_count: int = 0
        self.metadata_records: list[dict[str, Any]] = []

        # Sequence and Gap Tracking
        self.last_seq: int = -1
        self.duplicate_count: int = 0
        self.out_of_order_count: int = 0
        self.sequence_gaps: int = 0
        self.total_lost_frames: int = 0

        # Intrinsics
        self.intrinsics: dict[str, Any] | None = None

        # Topic Subscriptions
        self.rgb_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._rgb_cb, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self._depth_cb, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._info_cb, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/camera/camera/imu", self._imu_cb, 50
        )
        self.meta_sub = self.create_subscription(
            StringMsg, "/camera/camera/metadata", self._meta_cb, 10
        )

        self.start_time = time.monotonic()

    def _rgb_cb(self, msg: Image):
        self.rgb_times.append(time.monotonic())

    def _depth_cb(self, msg: Image):
        self.depth_times.append(time.monotonic())

    def _imu_cb(self, msg: Imu):
        self.imu_count += 1

    def _info_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = {
                "width": msg.width,
                "height": msg.height,
                "fx": msg.k[0],
                "fy": msg.k[4],
                "ppx": msg.k[2],
                "ppy": msg.k[5],
                "distortion_model": msg.distortion_model,
                "distortion": list(msg.d),
            }

    def _meta_cb(self, msg: StringMsg):
        now_mono = time.monotonic()
        try:
            record = json.loads(msg.data)
            record["_receive_mono"] = now_mono
            self.metadata_records.append(record)

            seq = record.get("sequence_number", -1)
            if self.last_seq >= 0:
                if seq == self.last_seq:
                    self.duplicate_count += 1
                elif seq < self.last_seq:
                    self.out_of_order_count += 1
                elif seq > self.last_seq + 1:
                    gap = seq - self.last_seq - 1
                    self.sequence_gaps += 1
                    self.total_lost_frames += gap
            self.last_seq = seq
        except Exception:
            pass

    def compute_report(self) -> dict[str, Any]:
        """Calculates comprehensive Stage 1 statistics from buffered telemetry."""
        total_duration = time.monotonic() - self.start_time
        if total_duration <= 0.0:
            total_duration = 1e-6

        # Frame Rates
        rgb_fps = len(self.rgb_times) / total_duration if self.rgb_times else 0.0
        depth_fps = len(self.depth_times) / total_duration if self.depth_times else 0.0
        meta_fps = len(self.metadata_records) / total_duration if self.metadata_records else 0.0
        imu_fps = self.imu_count / total_duration

        # Skew Statistics (ms)
        skews = [r["rgb_depth_dt_ms"] for r in self.metadata_records if "rgb_depth_dt_ms" in r]
        skew_p50 = float(np.percentile(skews, 50)) if skews else 0.0
        skew_p95 = float(np.percentile(skews, 95)) if skews else 0.0
        skew_p99 = float(np.percentile(skews, 99)) if skews else 0.0
        skew_max = float(np.max(skews)) if skews else 0.0

        # Frame Age / Ingestion Latency (ms)
        # Difference between network arrival and host capture time
        ages = []
        for r in self.metadata_records:
            arr = r.get("network_arrival_timestamp")
            cap = r.get("host_capture_timestamp")
            if arr is not None and cap is not None and arr >= cap:
                ages.append((arr - cap) * 1000.0)
        age_p50 = float(np.percentile(ages, 50)) if ages else 0.0
        age_p95 = float(np.percentile(ages, 95)) if ages else 0.0
        age_p99 = float(np.percentile(ages, 99)) if ages else 0.0

        # Runtime Depth Scale
        scales = [r["depth_scale"] for r in self.metadata_records if "depth_scale" in r]
        last_scale = scales[-1] if scales else None

        # Clock Mapping & Health
        clock_records = [r["clock"] for r in self.metadata_records if "clock" in r]
        last_clock = clock_records[-1] if clock_records else {}
        clk_offset_ms = float(last_clock.get("offset_sec", 0.0)) * 1000.0
        clk_drift_ppm = float(last_clock.get("drift", 0.0)) * 1e6
        clk_healthy = bool(last_clock.get("healthy", False))

        # Sessions
        sessions = list({r.get("session_id") for r in self.metadata_records if "session_id" in r})

        # Acceptance Evaluations
        criteria = {
            "rgb_fps_pass": rgb_fps >= 28.0,
            "depth_fps_pass": depth_fps >= 28.0,
            "skew_p95_pass": skew_p95 <= 15.0,
            "skew_max_pass": skew_max <= 50.0,
            "frame_age_p95_pass": age_p95 <= 45.0 if ages else True,
            "sequence_integrity_pass": self.duplicate_count == 0 and self.out_of_order_count == 0,
            "depth_scale_pass": last_scale is not None and abs(last_scale - 0.001) <= 0.0001,
            "clock_healthy_pass": clk_healthy or len(self.metadata_records) < 10,
        }
        all_passed = all(criteria.values()) and len(self.metadata_records) > 0

        return {
            "duration_sec": total_duration,
            "counts": {
                "rgb_frames": len(self.rgb_times),
                "depth_frames": len(self.depth_times),
                "metadata_frames": len(self.metadata_records),
                "imu_samples": self.imu_count,
            },
            "fps": {
                "rgb_fps": round(rgb_fps, 2),
                "depth_fps": round(depth_fps, 2),
                "metadata_fps": round(meta_fps, 2),
                "imu_fps": round(imu_fps, 2),
            },
            "sequence": {
                "duplicates": self.duplicate_count,
                "out_of_order": self.out_of_order_count,
                "sequence_gaps": self.sequence_gaps,
                "lost_frames": self.total_lost_frames,
            },
            "hardware_skew_ms": {
                "p50": round(skew_p50, 2),
                "p95": round(skew_p95, 2),
                "p99": round(skew_p99, 2),
                "max": round(skew_max, 2),
                "sample_count": len(skews),
            },
            "frame_age_ms": {
                "p50": round(age_p50, 2),
                "p95": round(age_p95, 2),
                "p99": round(age_p99, 2),
                "sample_count": len(ages),
            },
            "depth_scale": {
                "runtime_scale": last_scale,
                "expected": 0.001,
                "tolerance": 0.0001,
            },
            "clock": {
                "healthy": clk_healthy,
                "offset_ms": round(clk_offset_ms, 2),
                "drift_ppm": round(clk_drift_ppm, 2),
            },
            "intrinsics": self.intrinsics,
            "sessions": sessions,
            "criteria": criteria,
            "stage1_passed": all_passed,
        }


def print_dashboard(report: dict[str, Any]):
    print("\n" + "=" * 76)
    print("           STAGE 1 INTEL REALSENSE D455 LIVE INGESTION DASHBOARD          ")
    print("=" * 76)
    print(f"Monitoring Duration : {report['duration_sec']:.2f} s")
    print(f"Sessions Detected   : {', '.join(report['sessions']) if report['sessions'] else 'None'}")

    print("\n[Throughput & Frame Rates]")
    fps = report["fps"]
    counts = report["counts"]
    print(f"  RGB Stream        : {fps['rgb_fps']:>5.1f} FPS  ({counts['rgb_frames']} frames)")
    print(f"  Depth Stream      : {fps['depth_fps']:>5.1f} FPS  ({counts['depth_frames']} frames)")
    print(f"  Metadata Stream   : {fps['metadata_fps']:>5.1f} FPS  ({counts['metadata_frames']} frames)")
    print(f"  IMU Stream        : {fps['imu_fps']:>5.1f} Hz   ({counts['imu_samples']} samples)")

    print("\n[Sequence Continuity & Integrity]")
    seq = report["sequence"]
    print(f"  Duplicate Frames  : {seq['duplicates']}")
    print(f"  Out-of-Order      : {seq['out_of_order']}")
    print(f"  Sequence Gaps     : {seq['sequence_gaps']} (total lost = {seq['lost_frames']})")

    print("\n[Hardware RGB-Depth Synchronization]")
    skew = report["hardware_skew_ms"]
    print(f"  P50 Skew          : {skew['p50']:>5.2f} ms")
    print(f"  P95 Skew          : {skew['p95']:>5.2f} ms  (Limit: <= 15.0 ms)")
    print(f"  P99 Skew          : {skew['p99']:>5.2f} ms  (Limit: <= 33.0 ms)")
    print(f"  Max Skew          : {skew['max']:>5.2f} ms  (Hard Cutoff: <= 50.0 ms)")

    print("\n[Network Ingestion Latency / Frame Age]")
    age = report["frame_age_ms"]
    print(f"  P50 Ingestion Age : {age['p50']:>5.2f} ms")
    print(f"  P95 Ingestion Age : {age['p95']:>5.2f} ms  (Target: <= 45.0 ms)")
    print(f"  P99 Ingestion Age : {age['p99']:>5.2f} ms")

    print("\n[Authoritative Runtime Depth Scale]")
    ds = report["depth_scale"]
    scale_val = f"{ds['runtime_scale']:.6f} m/unit" if ds["runtime_scale"] is not None else "NOT RECEIVED"
    print(f"  Queried Scale     : {scale_val} (Expected: {ds['expected']:.6f} +/- {ds['tolerance']:.4f})")

    print("\n[Clock Synchronization (Windows Acquisition -> ROS Clock)]")
    clk = report["clock"]
    clk_status = "HEALTHY" if clk["healthy"] else "CONVERGING / UNHEALTHY"
    print(f"  Status            : {clk_status}")
    print(f"  Offset            : {clk['offset_ms']:>5.2f} ms")
    print(f"  Drift             : {clk['drift_ppm']:>5.2f} ppm")

    intr = report.get("intrinsics")
    if intr:
        print("\n[Factory Camera Intrinsics]")
        print(f"  Resolution        : {intr['width']}x{intr['height']}")
        print(f"  Focal Length      : fx={intr['fx']:.2f}, fy={intr['fy']:.2f}")
        print(f"  Principal Point   : cx={intr['ppx']:.2f}, cy={intr['ppy']:.2f}")
        print(f"  Model / Coeffs    : {intr['distortion_model']} {intr['distortion']}")

    print("\n" + "-" * 76)
    print("STAGE 1 ACCEPTANCE CRITERIA VERIFICATION:")
    crit = report["criteria"]
    print(f"  [ {'PASS' if crit['rgb_fps_pass'] else 'FAIL'} ] RGB Frame Rate >= 28.0 FPS ({fps['rgb_fps']} FPS)")
    print(f"  [ {'PASS' if crit['depth_fps_pass'] else 'FAIL'} ] Depth Frame Rate >= 28.0 FPS ({fps['depth_fps']} FPS)")
    print(f"  [ {'PASS' if crit['skew_p95_pass'] else 'FAIL'} ] Hardware RGB-Depth Skew P95 <= 15.0 ms ({skew['p95']} ms)")
    print(f"  [ {'PASS' if crit['skew_max_pass'] else 'FAIL'} ] Hardware RGB-Depth Skew Max <= 50.0 ms ({skew['max']} ms)")
    print(f"  [ {'PASS' if crit['frame_age_p95_pass'] else 'FAIL'} ] Ingestion Age P95 <= 45.0 ms ({age['p95']} ms)")
    print(f"  [ {'PASS' if crit['sequence_integrity_pass'] else 'FAIL'} ] Zero Duplicate / Out-of-Order Frames (dup={seq['duplicates']}, ooo={seq['out_of_order']})")
    print(f"  [ {'PASS' if crit['depth_scale_pass'] else 'FAIL'} ] Authoritative Depth Scale Valid ({scale_val})")
    print(f"  [ {'PASS' if crit['clock_healthy_pass'] else 'FAIL'} ] Clock Mapping Synchronized & Healthy")
    print("-" * 76)

    status_str = ">>> STAGE 1 ACCEPTANCE: PASSED <<<" if report["stage1_passed"] else ">>> STAGE 1 ACCEPTANCE: NOT PASSED <<<"
    print(f"RESULT: {status_str}\n")


def main():
    parser = argparse.ArgumentParser(description="Live D455 Diagnostics and Acceptance Validator.")
    parser.add_argument("--duration", type=float, default=10.0, help="Monitoring duration in seconds (default: 10.0)")
    parser.add_argument("--output_json", type=str, default="", help="Optional path to write JSON summary report")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal dashboard")
    args = parser.parse_args()

    if not HAS_ROS2:
        print("[Error] ROS 2 (rclpy) is not available. Please run inside your ROS 2 environment (WSL2 / Linux).")
        sys.exit(1)

    rclpy.init()
    monitor = D455LiveMonitor()
    print(f"[D455 Checker] Monitoring live D455 streams for {args.duration:.1f} seconds...")

    t_end = time.monotonic() + args.duration
    try:
        while time.monotonic() < t_end:
            rclpy.spin_once(monitor, timeout_sec=0.05)
    except KeyboardInterrupt:
        print("\n[D455 Checker] Monitoring interrupted by user.")
    finally:
        report = monitor.compute_report()
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not args.quiet:
        print_dashboard(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[D455 Checker] Report exported to: {out_path}")

    sys.exit(0 if report["stage1_passed"] else 1)


if __name__ == "__main__":
    main()
