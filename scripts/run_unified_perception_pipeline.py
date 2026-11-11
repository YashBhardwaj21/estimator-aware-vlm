#!/usr/bin/env python3
"""Run unified perception pipeline across all benchmark RGB-D sequences."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Direct Ultralytics config and cache into workspace directory
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "scratch" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))

from estimator_aware_vlm.perception.unified_pipeline import (
    ALL_SEQUENCES,
    UnifiedPerceptionPipeline,
    SequenceSpec,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Perception Pipeline")
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Process only this sequence_id. Omit to process all 8.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        dest="max_frames",
        help="Limit frame count per sequence (for validation/testing).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Override default output directory.",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.40,
        dest="min_conf",
        help="YOLO detection confidence threshold (default: 0.40).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override per-sequence stride for all sequences.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Determine output directory
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "results" / "perception_harvesting" / f"run_{run_tag}"

    print(f"Unified Perception Pipeline")
    print(f"Repo root:  {REPO_ROOT}")
    print(f"Output dir: {output_dir}")
    print(f"Run tag:    {run_tag}")

    # Select sequences to run
    sequences = ALL_SEQUENCES
    if args.sequence:
        selected_ids = [s.strip() for s in args.sequence.split(",") if s.strip()]
        sequences = [s for s in ALL_SEQUENCES if s.sequence_id in selected_ids]
        if not sequences:
            valid = [s.sequence_id for s in ALL_SEQUENCES]
            print(f"ERROR: Unknown sequence '{args.sequence}'. Valid: {valid}")
            sys.exit(1)

    # Apply CLI overrides
    for spec in sequences:
        if args.max_frames is not None:
            spec.max_frames = args.max_frames
        if args.stride is not None:
            spec.stride = args.stride

    # Build pipeline
    pipeline = UnifiedPerceptionPipeline(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        min_conf=args.min_conf,
    )

    print(f"\nYOLO model: {pipeline.yolo_model_path}")
    model_hash = UnifiedPerceptionPipeline.hash_file(pipeline.yolo_model_path)
    print(f"Model SHA-256: {model_hash[:16]}...")

    # Process sequences
    results = []
    for spec in sequences:
        print(f"\n{'─'*60}")
        print(f"Processing: {spec.sequence_id}")
        try:
            result = pipeline.process_sequence(spec)
            results.append(result)
        except Exception as exc:
            import traceback
            print(f"[ERROR] {spec.sequence_id} failed: {exc}")
            traceback.print_exc()
            print("Continuing with remaining sequences...")

    if not results:
        print("No sequences completed successfully.")
        sys.exit(1)

    # Write all output files
    pipeline.write_outputs(results, model_hash)

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

