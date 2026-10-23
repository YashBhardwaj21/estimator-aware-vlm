#!/usr/bin/env python3
"""Convert ROS 2 (.db3) RealSense RGB-D recording bag into standard dataset format.

Produces:
  - rgb/       (PNG color images, BGR encoded)
  - depth/     (16-bit PNG depth images in device units/mm)
  - rgb.txt    (timestamp rgb/<stamp>.png)
  - depth.txt  (timestamp depth/<stamp>.png)
  - associations.txt (synchronized 1-to-1 pairs)
  - metadata.json / camera_intrinsics.json (calibration parameters)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Detect virtualenv and ROS 2 site packages dynamically
_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
_extra_paths = [
    os.getenv("VIRTUAL_ENV_SITE_PACKAGES", ""),
    os.getenv("EXTRA_SITE_PACKAGES", ""),
    f"/opt/ros/{os.getenv('ROS_DISTRO', 'humble')}/local/lib/python{_py_ver}/dist-packages",
    f"/opt/ros/{os.getenv('ROS_DISTRO', 'humble')}/lib/python{_py_ver}/site-packages",
]
for _p in _extra_paths:
    if _p and os.path.exists(_p) and _p not in sys.path:
        sys.path.append(_p)

import cv2
import numpy as np

# Optional/guarded ROS 2 message deserialization
try:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
except ImportError:
    deserialize_message = None
    Image = None

from estimator_aware_vlm.common.paths import (
    get_repo_root,
    resolve_path,
    get_raw_data_dir,
    get_processed_data_dir,
)
from scene_graph.data.synchronization import associate


def convert_bag_to_rgbd(bag_dir: Path, manifest_path: Path, output_dir: Path):
    if deserialize_message is None or Image is None:
        raise ImportError(
            "ROS 2 Python modules ('rclpy', 'sensor_msgs') could not be imported. "
            "Please source your ROS 2 environment (e.g. source /opt/ros/humble/setup.bash) before running bag conversion."
        )

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_out_dir = output_dir / "rgb"
    depth_out_dir = output_dir / "depth"
    rgb_out_dir.mkdir(exist_ok=True)
    depth_out_dir.mkdir(exist_ok=True)

    db3_file = next(bag_dir.glob("*.db3"), None)
    if not db3_file:
        raise FileNotFoundError(f"No .db3 database file found in {bag_dir}")

    print("=" * 70)
    print("CONVERTING ROS 2 BAG TO STANDARD RGB-D DATASET")
    print(f"Source Bag:      {db3_file}")
    print(f"Manifest:        {manifest_path}")
    print(f"Destination:     {output_dir}")
    print("=" * 70)

    # 1. Parse Manifest
    manifest_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    # Calibration parameters from manifest
    color_info = manifest_data.get("color", {})
    depth_info = manifest_data.get("depth", {})
    depth_scale = manifest_data.get("depth_scale_m_per_unit", 0.001)

    metadata_json = {
        "sequence_name": output_dir.name,
        "recording_source": str(db3_file.name),
        "device": {
            "name": manifest_data.get("camera", "RealSense D455"),
            "serial": manifest_data.get("serial_number", "239622300767"),
            "firmware": manifest_data.get("firmware_version", "5.13.0.55"),
        },
        "depth_scale_m_per_unit": float(depth_scale),
        "depth_units_per_meter": 1.0 / float(depth_scale),
        "color_intrinsics": {
            "width": int(color_info.get("width", 640)),
            "height": int(color_info.get("height", 480)),
            "fx": float(color_info.get("fx", 383.5022)),
            "fy": float(color_info.get("fy", 383.0602)),
            "cx": float(color_info.get("ppx", 320.2671)),
            "cy": float(color_info.get("ppy", 244.5699)),
            "distortion_model": color_info.get("distortion_model", "distortion.inverse_brown_conrady"),
            "distortion_coefficients": color_info.get("distortion_coefficients", [-0.057, 0.071, 0.0, 0.0, -0.023]),
        },
        "depth_intrinsics": {
            "width": int(depth_info.get("width", 640)),
            "height": int(depth_info.get("height", 480)),
            "fx": float(depth_info.get("fx", 388.8132)),
            "fy": float(depth_info.get("fy", 388.8132)),
            "cx": float(depth_info.get("ppx", 321.6112)),
            "cy": float(depth_info.get("ppy", 240.3065)),
        },
        "notes": [
            "Extracted directly from ROS 2 db3 bag.",
            "Color saved in 8-bit BGR PNG.",
            "Depth saved in 16-bit uint16 mm PNG.",
            "Timestamps represent synchronized hardware seconds.",
        ],
    }

    camera_intrinsics_json = {
        "width": int(color_info.get("width", 640)),
        "height": int(color_info.get("height", 480)),
        "fx": float(color_info.get("fx", 383.5022)),
        "fy": float(color_info.get("fy", 383.0602)),
        "cx": float(color_info.get("ppx", 320.2671)),
        "cy": float(color_info.get("ppy", 244.5699)),
        "distortion_model": color_info.get("distortion_model", "distortion.inverse_brown_conrady"),
        "coeffs": color_info.get("distortion_coefficients", [-0.057, 0.071, 0.0, 0.0, -0.023]),
        "depth_scale": 1.0 / float(depth_scale),
        "depth_unit_meters": float(depth_scale),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    with open(output_dir / "camera_intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(camera_intrinsics_json, f, indent=2)

    # 2. Query SQLite Database for Messages
    print("\nReading messages from database...")
    conn = sqlite3.connect(str(db3_file))
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM topics WHERE name LIKE '%image/data%'")
    topic_map = dict(cur.fetchall())
    color_topic_id = None
    depth_topic_id = None

    for tid, tname in topic_map.items():
        if "Color_0" in tname:
            color_topic_id = tid
        elif "Depth_0" in tname:
            depth_topic_id = tid

    if color_topic_id is None or depth_topic_id is None:
        raise ValueError(f"Could not identify both Color and Depth topic IDs in {topic_map}")

    print(f"Color topic ID: {color_topic_id} ({topic_map[color_topic_id]})")
    print(f"Depth topic ID: {depth_topic_id} ({topic_map[depth_topic_id]})")

    # Fetch all image messages ordered by id (which is strictly chronological)
    cur.execute(
        "SELECT id, topic_id, timestamp, data FROM messages "
        "WHERE topic_id IN (?, ?) ORDER BY id ASC",
        (color_topic_id, depth_topic_id)
    )

    rgb_records: List[Tuple[float, str]] = []
    depth_records: List[Tuple[float, str]] = []

    print("\nExtracting and saving PNG frames...")
    msg_count = 0
    while True:
        rows = cur.fetchmany(100)
        if not rows:
            break
        for mid, tid, t_ns, raw_data in rows:
            msg_count += 1
            t_sec = float(t_ns) * 1e-9

            if tid == color_topic_id:
                fname = f"rgb/{t_sec:.6f}.png"
                full_path = output_dir / fname
                if not full_path.exists():
                    msg = deserialize_message(raw_data, Image)
                    h, w = msg.height, msg.width
                    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                    bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(full_path), bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                rgb_records.append((t_sec, fname))

            elif tid == depth_topic_id:
                fname = f"depth/{t_sec:.6f}.png"
                full_path = output_dir / fname
                if not full_path.exists():
                    msg = deserialize_message(raw_data, Image)
                    h, w = msg.height, msg.width
                    depth_data = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
                    cv2.imwrite(str(full_path), depth_data, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                depth_records.append((t_sec, fname))

        if msg_count % 300 == 0 or msg_count == 2089:
            print(f"  Processed {msg_count} messages (RGB: {len(rgb_records)}, Depth: {len(depth_records)})...")

    conn.close()


    print(f"\nExtracted {len(rgb_records)} RGB frames and {len(depth_records)} Depth frames.")

    # 3. Write rgb.txt and depth.txt
    print("Writing rgb.txt and depth.txt...")
    rgb_records.sort(key=lambda x: x[0])
    depth_records.sort(key=lambda x: x[0])

    with open(output_dir / "rgb.txt", "w", encoding="utf-8") as f:
        f.write("# color images\n# timestamp filename\n")
        for ts, p in rgb_records:
            f.write(f"{ts:.6f} {p}\n")

    with open(output_dir / "depth.txt", "w", encoding="utf-8") as f:
        f.write("# depth images\n# timestamp filename\n")
        for ts, p in depth_records:
            f.write(f"{ts:.6f} {p}\n")

    # 4. Generate Optimal Monotonic Timestamp Associations
    print("Generating optimal monotonic RGB-D associations...")
    rgb_ts = [t for t, _ in rgb_records]
    depth_ts = [t for t, _ in depth_records]
    matches = associate(rgb_ts, depth_ts, max_dt=0.05)

    with open(output_dir / "associations.txt", "w", encoding="utf-8") as f:
        for i_rgb, i_depth in matches:
            t_r, p_r = rgb_records[i_rgb]
            t_d, p_d = depth_records[i_depth]
            f.write(f"{t_r:.6f} {p_r} {t_d:.6f} {p_d}\n")

    print(f"Associated {len(matches)} 1-to-1 synchronized pairs (out of {len(rgb_records)} RGB frames).")
    print(f"Conversion completed in {time.time() - start_time:.1f} s!")
    print(f"Converted dataset ready at: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Convert RealSense ROS 2 db3 bag to RGB-D dataset")
    parser.add_argument("--bag-dir", type=str,
                        default=str(get_raw_data_dir() / "D455_Recordings" / "desk_run_20261009_161552"),
                        help="Path to bag folder containing .db3 file")
    parser.add_argument("--manifest", type=str,
                        default=str(get_raw_data_dir() / "D455_Recordings" / "desk_run_20261009_161552_manifest.json"),
                        help="Path to recording manifest JSON")
    parser.add_argument("--output-dir", type=str,
                        default=str(get_processed_data_dir() / "desk_run_20261009_161552"),
                        help="Output directory for extracted dataset")
    args = parser.parse_args()

    convert_bag_to_rgbd(
        resolve_path(args.bag_dir),
        resolve_path(args.manifest),
        resolve_path(args.output_dir),
    )


if __name__ == "__main__":
    main()
