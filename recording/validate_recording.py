#!/usr/bin/env python3
"""CLI tool to validate a RealSense D455 RGB-D recording session directory.

Usage:
  python tools/recording/validate_recording.py data/raw/d455_rgbd_20261010_123443
  python tools/recording/validate_recording.py data/raw/desk_run_20261009_161552
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
# sys.path.insert(0, os.path.abspath("src"))

from scene_graph.data.dataset_discovery import validate_recording_session


def main():
    parser = argparse.ArgumentParser(description="Validate D455 RGB-D recording session")
    parser.add_argument("recording_path", type=str, help="Path to recording session directory")
    parser.add_argument("--sample-stride", type=int, default=15, help="Sampling stride for image decode checks")
    args = parser.parse_args()

    rec_dir = Path(args.recording_path)
    if not rec_dir.exists():
        print(f"[ERROR] Directory does not exist: {rec_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"VALIDATING D455 RECORDING SESSION: {rec_dir}")
    print("=" * 70)

    report = validate_recording_session(rec_dir, sample_stride=args.sample_stride)

    print(f"Sequence:        {report.get('sequence_name', 'unknown')}")
    print(f"Status:          {report['status']}")
    counts = report.get("counts", {})
    print(f"RGB Frames:      {counts.get('rgb_frames', 0)}")
    print(f"Depth Frames:    {counts.get('depth_frames', 0)}")
    print(f"Associated:      {counts.get('associated_pairs', 0)}")
    timing = report.get("timing", {})
    print(f"Duration:        {timing.get('duration_s', 0)} s (~{timing.get('fps_estimate', 0)} FPS)")
    print(f"Skew Median/P95: {timing.get('skew_ms_median', 0)} ms / {timing.get('skew_ms_p95', 0)} ms (Max: {timing.get('skew_ms_max', 0)} ms)")
    quality = report.get("quality", {})
    print(f"Valid Depth:     {quality.get('mean_valid_depth_fraction', 0)*100:.1f}%")
    print(f"Calibration:     {quality.get('calibration_source', 'none')}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    if report.get("errors"):
        print("\nErrors:")
        for e in report["errors"]:
            print(f"  [FAIL] {e}")

    report_path = rec_dir / "recording_validation.json"
    print("\n" + "=" * 70)
    print(f"Validation report saved to: {report_path}")
    print("=" * 70)

    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
