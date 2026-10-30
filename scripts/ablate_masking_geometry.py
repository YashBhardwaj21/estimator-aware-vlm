#!/usr/bin/env python3
"""Geometric Masking Ablation: Polygon Contours vs Coarse Bounding Boxes.

Empirically compares pixel suppression and background preservation between:
1. Polygon Instance Segmentation Masking: tight boundary following object contour.
2. Bounding Box Masking: coarse axis-aligned rectangular suppression.

Quantifies the excess background context suppressed by coarse bounding boxes,
illustrating why polygon masking is essential for SLAM in cluttered environments.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import load_dataset_config_from_yaml
from scripts.run_pass_b_slam import generate_pass_b_masked_associations


def main() -> None:
    root = get_repo_root()
    cfg_path = root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    cfg = load_dataset_config_from_yaml(cfg_path)
    obs_path = root / "outputs" / "perception_tracks" / "per_frame_instance_observations.jsonl"
    dec_path = root / "outputs" / "vlm_inference" / "routing_decisions.json"

    with open(dec_path, "r", encoding="utf-8") as f:
        dec_data = json.load(f)
    masked_track_ids = {
        d["track_id"] for d in dec_data.get("decisions", [])
        if d.get("action") == "MASK_FROM_TRACKING"
    }

    print(f"Targeting {len(masked_track_ids)} dynamic tracks: {sorted(list(masked_track_ids))}")

    # 1. Polygon Masking
    poly_dir = root / "outputs" / "experiments" / "masking_ablation_polygon"
    print("\n--- Generating Tight Polygon Masks ---")
    generate_pass_b_masked_associations(
        raw_seq_dir=cfg.sequence_dir,
        output_dir=poly_dir,
        masked_track_ids=masked_track_ids,
        observations_path=obs_path,
        routing_decisions_path=dec_path,
        causality_report_path=poly_dir / "mask_causality_report.json",
        dataset_cfg=cfg,
        mask_type="polygon",
    )

    # 2. Bounding Box Masking
    bbox_dir = root / "outputs" / "experiments" / "masking_ablation_bbox"
    print("\n--- Generating Coarse Bounding Box Masks ---")
    generate_pass_b_masked_associations(
        raw_seq_dir=cfg.sequence_dir,
        output_dir=bbox_dir,
        masked_track_ids=masked_track_ids,
        observations_path=obs_path,
        routing_decisions_path=dec_path,
        causality_report_path=bbox_dir / "mask_causality_report.json",
        dataset_cfg=cfg,
        mask_type="bbox",
    )

    with open(poly_dir / "mask_causality_report.json", "r", encoding="utf-8") as f:
        p_rep = json.load(f)
    with open(bbox_dir / "mask_causality_report.json", "r", encoding="utf-8") as f:
        b_rep = json.load(f)

    p_px = p_rep["total_pixels_masked"]
    b_px = b_rep["total_pixels_masked"]
    diff_px = b_px - p_px
    excess_pct = ((b_px - p_px) / max(1, p_px)) * 100.0

    print("\n" + "=" * 80)
    print("           GEOMETRIC MASKING ABLATION: POLYGON VS BOUNDING BOX            ")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Polygon (Tight)':<18} | {'Bounding Box (Coarse)'}")
    print("-" * 80)
    print(f"{'Total Masked Frames':<35} | {p_rep['total_masked_frames']:<18} | {b_rep['total_masked_frames']}")
    print(f"{'Total Pixels Suppressed':<35} | {p_px:<18,} | {b_px:,}")
    print(f"{'Mean Pixels / Masked Frame':<35} | {p_px // max(1, p_rep['total_masked_frames']):<18,} | {b_px // max(1, b_rep['total_masked_frames']):,}")
    print(f"{'Excess Background Suppressed':<35} | Baseline (0.0%)    | +{excess_pct:.1f}% (+{diff_px:,} px)")
    print("=" * 80)

    # Save comparison summary
    comparison_summary = {
        "sequence_name": cfg.sequence_name,
        "masked_tracks": sorted(list(masked_track_ids)),
        "polygon_total_pixels": p_px,
        "bbox_total_pixels": b_px,
        "excess_pixels_suppressed_by_bbox": diff_px,
        "excess_background_percentage": round(excess_pct, 2),
        "polygon_causality_report": str(poly_dir / "mask_causality_report.json"),
        "bbox_causality_report": str(bbox_dir / "mask_causality_report.json"),
    }
    summary_path = root / "outputs" / "experiments" / "masking_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(comparison_summary, sf, indent=2)
    print(f"Ablation summary exported to: {summary_path}\n")


if __name__ == "__main__":
    main()
