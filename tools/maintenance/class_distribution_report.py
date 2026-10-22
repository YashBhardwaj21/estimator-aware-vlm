#!/usr/bin/env python3
"""Generate a class distribution report from an observation stream."""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate Class Distribution Report.")
    parser.add_argument("--observations_dir", type=str, default="data/processed/tum_fr1_desk/observations_controlled")
    args = parser.parse_args()

    obs_dir = Path(args.observations_dir)
    if not obs_dir.exists():
        print(f"Directory not found: {obs_dir}")
        return

    # Metrics
    total_frames = 0
    frames_with_detections = 0
    frames_without_detections = 0
    total_detections = 0

    # Class-specific metrics
    # class_name -> {"instances": int, "frames": set(frame_index)}
    class_stats = defaultdict(lambda: {"instances": 0, "frames": set()})

    chunk_files = sorted(obs_dir.glob("chunk_*.json"))
    if not chunk_files:
        print(f"No chunks found in {obs_dir}")
        return

    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for frame_data in data.get("frames", []):
            total_frames += 1
            frame_idx = frame_data["frame_index"]
            observations = frame_data.get("observations", [])

            if len(observations) > 0:
                frames_with_detections += 1
            else:
                frames_without_detections += 1

            for obs in observations:
                total_detections += 1
                cls_name = obs["class_name"]
                class_stats[cls_name]["instances"] += 1
                class_stats[cls_name]["frames"].add(frame_idx)

    # Compute derived metrics
    unique_classes = len(class_stats)
    detections_per_frame = total_detections / total_frames if total_frames > 0 else 0

    print("=" * 60)
    print("Class Distribution Report")
    print("=" * 60)
    print(f"Stream directory:          {obs_dir}")
    print(f"Total Frames:              {total_frames}")
    print(f"Total Detections:          {total_detections}")
    print(f"Unique Classes:            {unique_classes}")
    print(f"Detections / Frame:        {detections_per_frame:.2f}")
    print(f"Frames w/ >= 1 Detection:  {frames_with_detections}")
    print(f"Frames w/ 0 Detections:    {frames_without_detections}")
    print("-" * 60)
    print(f"{'Class':<22} | {'Instances':<10} | {'Frames':<10}")
    print("-" * 60)

    # Sort by instances descending
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]["instances"], reverse=True)
    for cls_name, stats in sorted_classes:
        instances = stats["instances"]
        frames = len(stats["frames"])
        print(f"{cls_name:<22} | {instances:<10} | {frames:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
