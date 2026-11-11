#!/usr/bin/env python3
"""Run Temporal Tracking and Physical Motion Estimation.

Performs multi-cue temporal association and decoupled physical motion estimation
across benchmark sequences:
1. Multi-cue matching: semantic class, appearance (HSV histogram), 2D IoU,
   3D centroid distance with constant-velocity prediction, 3D dimensions, depth continuity.
2. Kinematic calculation: velocity, acceleration, directional consistency, path efficiency.
3. Ego-motion decoupling: isolates physical object displacement from camera ego-motion.
4. Generates deliverables:
   - results/tracking/tracks_3d.jsonl
   - results/tracking/tracking_observations.jsonl
   - results/tracking/tracking_metrics.json
   - results/tracking/motion_analysis_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from estimator_aware_vlm.perception.sequence_tracker import build_tracking_deliverables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal tracking and physical motion estimation."
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=REPO_ROOT,
        help="Workspace root containing results and configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tracking",
        help="Directory where tracking deliverables are written.",
    )
    parser.add_argument(
        "--detections-file",
        type=Path,
        default=None,
        help="Path to detections JSONL file (defaults to results/perception/detections.jsonl).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Subset of sequence keys to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_dir = args.workspace_dir.resolve()
    output_dir = args.output_dir.resolve()
    detections_path = args.detections_file.resolve() if args.detections_file else None

    print(f"Starting temporal tracking and physical motion estimation")
    print(f"Workspace: {workspace_dir}")
    print(f"Output directory: {output_dir}")
    if detections_path:
        print(f"Detections file: {detections_path}")

    t0 = time.time()
    results = build_tracking_deliverables(
        repo_root=workspace_dir,
        output_dir=output_dir,
        input_obs_path=detections_path,
        sequences=args.sequences,
    )
    elapsed = time.time() - t0

    global_summary = results.get("global", {})
    seq_list = results.get("sequences", [])
    print(f"\nTracking run completed in {elapsed:.2f}s")
    print(f"Total sequences processed: {len(seq_list)}")
    print(f"Total tracks formed: {global_summary.get('total_tracks', 0)}")
    print(f"Total observations tracked: {global_summary.get('total_tracked_observations', 0)}")
    print(f"Moving tracks: {global_summary.get('moving_tracks', 0)}")
    print(f"Static tracks: {global_summary.get('static_tracks', 0)}")
    print(f"Displaced tracks: {global_summary.get('displaced_tracks', 0)}")
    print(f"Deliverables written to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
