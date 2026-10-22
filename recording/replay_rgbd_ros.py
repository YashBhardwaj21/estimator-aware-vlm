#!/usr/bin/env python3
"""Replay a recorded RGB-D sequence as ROS 2 sensor topics."""

from __future__ import annotations

import sys

for _p in [
    "/home/saturn/myenv/lib/python3.10/site-packages",
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
]:
    if _p not in sys.path:
        sys.path.append(_p)

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def read_associations(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{line_number}: expected 4 columns, got {len(fields)}"
            )

        rgb_ts, rgb_rel, depth_ts, depth_rel = fields
        rgb_path = path.parent / rgb_rel
        depth_path = path.parent / depth_rel
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Missing frame on line {line_number}: "
                f"{rgb_path} or {depth_path}"
            )
        rows.append((float(rgb_ts), rgb_path, float(depth_ts), depth_path))

    if not rows:
        raise ValueError(f"No frames found in {path}")
    return rows


def read_calibration(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        intr = data.get("color_intrinsics", data)
        coeffs = intr.get("distortion_coefficients", [0.0] * 5)
        return {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "width": int(intr["width"]),
            "height": int(intr["height"]),
            "k1": float(coeffs[0]) if len(coeffs) > 0 else 0.0,
            "k2": float(coeffs[1]) if len(coeffs) > 1 else 0.0,
            "p1": float(coeffs[2]) if len(coeffs) > 2 else 0.0,
            "p2": float(coeffs[3]) if len(coeffs) > 3 else 0.0,
            "k3": float(coeffs[4]) if len(coeffs) > 4 else 0.0,
        }

    text = path.read_text()

    def value(key: str, default=None) -> float:
        match = re.search(
            rf"^\s*{re.escape(key)}:\s*"
            rf"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            text,
            re.MULTILINE,
        )
        if match:
            return float(match.group(1))
        if default is not None:
            return float(default)
        raise ValueError(f"Missing calibration field {key!r} in {path}")

    return {
        "fx": value("Camera1.fx"),
        "fy": value("Camera1.fy"),
        "cx": value("Camera1.cx"),
        "cy": value("Camera1.cy"),
        "width": int(value("Camera.width")),
        "height": int(value("Camera.height")),
        "k1": value("Camera1.k1", 0.0),
        "k2": value("Camera1.k2", 0.0),
        "p1": value("Camera1.p1", 0.0),
        "p2": value("Camera1.p2", 0.0),
        "k3": value("Camera1.k3", 0.0),
    }


def set_stamp(header, seconds: float) -> None:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    header.stamp.sec = sec
    header.stamp.nanosec = nanosec


def make_image(data, encoding: str, frame_id: str, stamp: float) -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    set_stamp(msg.header, stamp)
    msg.height, msg.width = int(data.shape[0]), int(data.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(data.strides[0])
    msg.data = data.tobytes(order="C")
    return msg


class RGBDReplay(Node):
    def __init__(self, calibration, args):
        super().__init__("recorded_rgbd_replay")
        self.cal = calibration
        self.args = args
        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, 5)
        self.depth_pub = self.create_publisher(Image, args.depth_topic, 5)
        self.info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, 5)

    def camera_info(self, stamp: float) -> CameraInfo:
        c = self.cal
        msg = CameraInfo()
        msg.header.frame_id = self.args.optical_frame
        set_stamp(msg.header, stamp)
        msg.width = c["width"]
        msg.height = c["height"]
        msg.distortion_model = "plumb_bob"
        msg.d = [c["k1"], c["k2"], c["p1"], c["p2"], c["k3"]]
        msg.k = [
            c["fx"], 0.0, c["cx"],
            0.0, c["fy"], c["cy"],
            0.0, 0.0, 1.0,
        ]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            c["fx"], 0.0, c["cx"], 0.0,
            0.0, c["fy"], c["cy"], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def publish_frame(self, rgb_path, depth_path, rgb_stamp, depth_stamp):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if rgb is None:
            raise RuntimeError(f"Could not read RGB image: {rgb_path}")
        if depth is None:
            raise RuntimeError(f"Could not read depth image: {depth_path}")
        if rgb.dtype.name != "uint8" or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Unexpected RGB format: {rgb.shape}, {rgb.dtype}")
        if depth.dtype.name != "uint16" or depth.ndim != 2:
            raise ValueError(
                f"Expected uint16 depth, got {depth.shape}, {depth.dtype}"
            )
        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"RGB/depth dimensions differ: {rgb.shape} vs {depth.shape}"
            )

        expected = (self.cal["height"], self.cal["width"])
        if rgb.shape[:2] != expected:
            raise ValueError(
                f"Image size {rgb.shape[1]}x{rgb.shape[0]} does not match "
                f"calibration {expected[1]}x{expected[0]}"
            )

        frame = self.args.optical_frame
        self.rgb_pub.publish(make_image(rgb, "bgr8", frame, rgb_stamp))
        self.depth_pub.publish(make_image(depth, "16UC1", frame, depth_stamp))
        self.info_pub.publish(self.camera_info(rgb_stamp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "datasets/d455_orb_aligned_sequence",
    )
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None,
                        help="Exclusive frame index in the original sequence")
    parser.add_argument("--optical-frame", default="camera_color_optical_frame")
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/camera/camera/color/camera_info",
    )
    parser.add_argument(
        "--calibration",
        default="",
        help="Optional explicit path to calibration file (.yaml or .json)",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be greater than zero")

    root = args.root.expanduser().resolve()
    rows = read_associations(root / "associations.txt")

    if args.calibration:
        cal_path = Path(args.calibration).expanduser().resolve()
    elif (root / "metadata.json").is_file():
        cal_path = root / "metadata.json"
    elif (root / "camera_intrinsics.json").is_file():
        cal_path = root / "camera_intrinsics.json"
    else:
        cal_path = root / "D455_MyDesk_rectified.yaml"

    calibration = read_calibration(cal_path)

    start = args.start_index
    end = len(rows) if args.end_index is None else min(args.end_index, len(rows))
    if start < 0 or start >= len(rows) or end <= start:
        raise ValueError(
            f"Invalid frame range [{start}, {end}) for {len(rows)} frames"
        )
    rows = rows[start:end]

    rclpy.init()
    node = RGBDReplay(calibration, args)

    node.get_logger().info(
        f"Replaying {len(rows)} frames from {root} at {args.rate:.2f}x"
    )

    node.get_logger().info("Waiting for downstream subscriber matching...")
    for _ in range(60):
        if node.rgb_pub.get_subscription_count() > 0:
            node.get_logger().info(f"Matched {node.rgb_pub.get_subscription_count()} RGB subscriber(s).")
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.0)

    first_rgb = rows[0][0]
    wall_start = time.time() + 0.5
    monotonic_start = time.monotonic() + 0.5

    try:
        for i, (rgb_ts, rgb_path, depth_ts, depth_path) in enumerate(rows):
            elapsed = (rgb_ts - first_rgb) / args.rate
            deadline = monotonic_start + elapsed

            while rclpy.ok():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))

            rgb_ros_ts = wall_start + elapsed
            depth_ros_ts = wall_start + (depth_ts - first_rgb) / args.rate
            node.publish_frame(
                rgb_path, depth_path, rgb_ros_ts, depth_ros_ts
            )
            rclpy.spin_once(node, timeout_sec=0.0)

            if i % 50 == 0 or i == len(rows) - 1:
                node.get_logger().info(f"Published {i + 1}/{len(rows)} frames")

            if not rclpy.ok():
                break

        node.get_logger().info(f"Replay loop finished: {len(rows)} frames published. Flushing DDS...")
        for _ in range(20):
            if not rclpy.ok():
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning("Replay interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
