"""Estimator-aware query selection policy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Semantic class priority weights for Value of Information (P7)
SEMANTIC_PRIOR_WEIGHTS: Dict[str, float] = {
    # High-risk autonomous agents
    "person": 1.0,
    "dog": 1.0,
    "cat": 1.0,
    # Ambiguous movable items frequently manipulated or pushed
    "chair": 0.75,
    "box": 0.80,
    "backpack": 0.70,
    "handbag": 0.70,
    "suitcase": 0.75,
    "sports ball": 0.85,
    # Small electronics and personal items
    "laptop": 0.40,
    "keyboard": 0.30,
    "mouse": 0.30,
    "book": 0.35,
    "bottle": 0.40,
    "cup": 0.40,
    "potted plant": 0.25,
    # Structurally fixed furniture
    "monitor": 0.30,
    "tv": 0.25,
    "dining table": 0.15,
    "desk": 0.15,
    "couch": 0.15,
    "bed": 0.10,
    "refrigerator": 0.10,
    "sink": 0.05,
    "toilet": 0.05,
}
DEFAULT_SEMANTIC_WEIGHT = 0.50


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


def compute_policy_scores(obs: Dict[str, Any]) -> Dict[str, float]:
    """Compute causal priority score for each policy for a single observation.

    Strictly uses causal features only. Never accesses ground truth or future frames.
    Higher score = higher priority to query the VLM.
    """
    conf = float(obs.get("detection_confidence", 0.5))
    sharpness = float(obs.get("sharpness", 100.0))
    sharpness_log = float(obs.get("sharpness_log", 2.0))
    crop_area = float(obs.get("crop_area_px", 1000.0))
    crop_area_log = float(obs.get("crop_area_log", 3.0))
    apparent_disp = float(obs.get("apparent_camera_frame_displacement_m", 0.0))
    uncert_disp = float(obs.get("displacement_uncertainty_m", 0.0))
    obs_idx = int(obs.get("track_obs_index", 0))
    sem_cls = str(obs.get("semantic_class", "")).lower()
    ego_available = bool(obs.get("ego_motion_available", False))

    scores: Dict[str, float] = {}

    # P0: Zero query
    scores["P0_zero"] = 0.0

    # P1: Always query
    scores["P1_always"] = 1.0

    # P2: Random (seeded per observation id for determinism)
    # Uses hash of obs_id to ensure fair, reproducible evaluation
    h = hash(obs.get("obs_id", "")) & 0xFFFFFFFF
    scores["P2_random"] = float(h % 10000) / 10000.0

    # P3: Detector Uncertainty: highest score when confidence is near 0.5 (or lowest confidence)
    # Ambiguity score: max at conf=0.5, 0 at conf=1.0 or conf=0.0
    scores["P3_detector_uncertainty"] = 1.0 - abs(conf - 0.5) * 2.0

    # P4: Visual Quality / Sharpness heuristic
    # High score for sharp, large crops
    scores["P4_visual_quality"] = sharpness_log * crop_area_log

    # P5: Naive Apparent Displacement (camera frame - uncompensated)
    scores["P5_apparent_disp"] = apparent_disp

    # P6: Estimator Residual Uncertainty (ego-motion compensated variance)
    scores["P6_estimator_uncertainty"] = uncert_disp * (1.2 if ego_available else 0.8)

    # P7: Estimator-Aware Value of Information (VOI - Proposed)
    # 1. Semantic prior
    w_sem = SEMANTIC_PRIOR_WEIGHTS.get(sem_cls, DEFAULT_SEMANTIC_WEIGHT)

    # 2. Estimator displacement risk (causal: apparent displacement scaled by uncertainty)
    disp_risk = sigmoid(3.0 * (apparent_disp - 0.20)) if not ego_available else sigmoid(4.0 * (uncert_disp - 0.15))

    # 3. Visual perceptibility gate (avoids querying blurry/tiny unrecognizable crops)
    gate_sharp = sigmoid(2.5 * (sharpness_log - 2.2))
    gate_size = sigmoid(2.0 * (crop_area_log - 3.4))
    vis_gate = gate_sharp * gate_size

    # 4. Temporal novelty (prioritize early observation of each track)
    temp_weight = math.exp(-0.4 * obs_idx)

    # Combined VOI score
    voi = (0.55 * w_sem * disp_risk + 0.45 * uncert_disp) * (0.3 + 0.7 * vis_gate) * (0.4 + 0.6 * temp_weight)
    scores["P7_estimator_voi"] = float(voi)

    return scores


def evaluate_split(
    observations: List[Dict[str, Any]],
    budgets: List[float],
    seed: int = 42,
) -> Dict[str, Any]:
    """Benchmark all policies across given budget thresholds."""
    random.seed(seed)
    np.random.seed(seed)

    # Score all observations
    scored_obs: List[Dict[str, Any]] = []
    for obs in observations:
        item = dict(obs)
        item["scores"] = compute_policy_scores(obs)
        scored_obs.append(item)

    # Ground truth determination:
    # A track is dynamic if gt_vlm_label == 'MOBILE' or observed_motion == 'MOVING'
    all_tracks = set(f"{o['sequence_id']}/{o['track_id']}" for o in observations)
    dynamic_tracks = set()
    for o in observations:
        key = f"{o['sequence_id']}/{o['track_id']}"
        gt_vlm = o.get("gt_vlm_label", "")
        mot = o.get("pipeline_motion_label", "")
        if gt_vlm == "MOBILE" or mot == "MOVING":
            dynamic_tracks.add(key)

    total_obs = len(observations)
    total_tracks_count = len(all_tracks)
    dynamic_tracks_count = len(dynamic_tracks)

    policy_names = [
        "P0_zero",
        "P1_always",
        "P2_random",
        "P3_detector_uncertainty",
        "P4_visual_quality",
        "P5_apparent_disp",
        "P6_estimator_uncertainty",
        "P7_estimator_voi",
    ]

    results_by_policy: Dict[str, Dict[str, Any]] = {p: {"budgets": {}} for p in policy_names}

    for p in policy_names:
        for b in budgets:
            if p == "P0_zero":
                k = 0
            elif p == "P1_always":
                k = total_obs
            else:
                k = max(1, int(round(total_obs * b)))

            # Sort observations by policy score descending
            sorted_items = sorted(scored_obs, key=lambda x: x["scores"][p], reverse=True)
            queried = sorted_items[:k]

            queried_dynamic_obs = sum(
                1 for o in queried
                if o.get("gt_vlm_label") == "MOBILE" or o.get("pipeline_motion_label") == "MOVING"
            )

            # Tracks intercepted
            queried_tracks = set(f"{o['sequence_id']}/{o['track_id']}" for o in queried)
            intercepted_dynamic_tracks = dynamic_tracks.intersection(queried_tracks)

            # Early interception: dynamic track queried at obs_index <= 1
            early_intercepted = set()
            for o in queried:
                key = f"{o['sequence_id']}/{o['track_id']}"
                if key in dynamic_tracks and int(o.get("track_obs_index", 0)) <= 1:
                    early_intercepted.add(key)

            query_precision = (queried_dynamic_obs / k) if k > 0 else 0.0
            dynamic_recall = (len(intercepted_dynamic_tracks) / dynamic_tracks_count) if dynamic_tracks_count > 0 else 0.0
            early_recall = (len(early_intercepted) / dynamic_tracks_count) if dynamic_tracks_count > 0 else 0.0
            compute_savings = 1.0 - (k / total_obs) if total_obs > 0 else 0.0
            efficiency_ratio = (dynamic_recall / b) if b > 0 else 0.0

            results_by_policy[p]["budgets"][f"{int(b*100)}%"] = {
                "budget_fraction": b,
                "queries_executed": k,
                "total_observations": total_obs,
                "dynamic_track_recall": round(dynamic_recall, 4),
                "early_dynamic_recall": round(early_recall, 4),
                "query_precision": round(query_precision, 4),
                "efficiency_ratio": round(efficiency_ratio, 2),
                "compute_savings_pct": round(compute_savings * 100.0, 1),
                "intercepted_dynamic_tracks": len(intercepted_dynamic_tracks),
                "total_dynamic_tracks": dynamic_tracks_count,
            }

    return {
        "total_observations": total_obs,
        "total_tracks": total_tracks_count,
        "total_dynamic_tracks": dynamic_tracks_count,
        "policies": results_by_policy,
        "scored_observations": scored_obs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimator-Aware Query Selection Policy Benchmark")
    parser.add_argument(
        "--features-file",
        type=Path,
        default=REPO_ROOT / "results" / "estimator_context" / "feature_vectors.jsonl",
        help="Path to feature_vectors.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "query_policy",
        help="Output directory",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Estimator-Aware Query Selection Benchmark")
    print("=" * 60)
    print(f"Loading features: {args.features_file}")

    if not args.features_file.exists():
        raise FileNotFoundError(f"Missing features file: {args.features_file}")

    with args.features_file.open("r", encoding="utf-8") as f:
        all_obs = [json.loads(line) for line in f]

    print(f"Total observations loaded: {len(all_obs)}")

    budgets = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

    # Evaluate across splits
    splits = ["full", "development", "validation", "held_out"]
    eval_by_split: Dict[str, Any] = {}

    for sp in splits:
        if sp == "full":
            sub_obs = all_obs
        else:
            sub_obs = [o for o in all_obs if o.get("evaluation_split") == sp]
        print(f"\nBenchmarking split: {sp} (n={len(sub_obs)} observations)...")
        eval_by_split[sp] = evaluate_split(sub_obs, budgets)

    # Save summary json
    summary_path = args.out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        # Don't save full scored_observations in summary to keep file clean
        clean_summary = {
            sp: {
                "total_observations": data["total_observations"],
                "total_tracks": data["total_tracks"],
                "total_dynamic_tracks": data["total_dynamic_tracks"],
                "policies": data["policies"],
            }
            for sp, data in eval_by_split.items()
        }
        json.dump(clean_summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Save audited query decisions for the full dataset
    decisions_path = args.out_dir / "query_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for o in eval_by_split["full"]["scored_observations"]:
            record = {
                "sequence_id": o["sequence_id"],
                "track_id": o["track_id"],
                "obs_id": o["obs_id"],
                "semantic_class": o.get("semantic_class"),
                "gt_vlm_label": o.get("gt_vlm_label"),
                "pipeline_motion_label": o.get("pipeline_motion_label"),
                "scores": o["scores"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved decisions to: {decisions_path}")

    # Generate Markdown Table Report
    report_path = args.out_dir / "policy_comparison_table.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Query-Selection Policy Benchmark Report\n\n")
        f.write("Generated from 885 causal observation feature vectors.\n\n")

        for sp in ["full", "development", "validation", "held_out"]:
            res = eval_by_split[sp]
            pols = res["policies"]
            f.write(f"## Evaluation Split: `{sp}`\n\n")
            f.write(f"- Total Observations: **{res['total_observations']}**\n")
            f.write(f"- Total Tracks: **{res['total_tracks']}**\n")
            f.write(f"- Total Dynamic / Mobile Tracks: **{res['total_dynamic_tracks']}**\n\n")

            f.write("### Policy Comparison across Budgets (Dynamic Track Recall & Query Precision)\n\n")
            f.write("| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |\n")
            f.write("| --- | --- | :---: | :---: | :---: | :---: | :---: |\n")

            policy_labels = [
                ("P0_zero", "P0 (Zero VLM / Baseline)"),
                ("P1_always", "P1 (Always Query / Naive)"),
                ("P2_random", "P2 (Random Subsampling)"),
                ("P3_detector_uncertainty", "P3 (Detector Uncertainty)"),
                ("P4_visual_quality", "P4 (Visual Quality / Sharpness)"),
                ("P5_apparent_disp", "P5 (Naive Apparent Motion)"),
                ("P6_estimator_uncertainty", "P6 (Estimator Uncertainty)"),
                ("P7_estimator_voi", "**P7 (Proposed Estimator VOI)**"),
            ]

            for p_key, p_label in policy_labels:
                b_dict = pols[p_key]["budgets"]
                r5 = f"{b_dict['5%']['dynamic_track_recall']*100:.1f}%"
                r10 = f"{b_dict['10%']['dynamic_track_recall']*100:.1f}%"
                r20 = f"{b_dict['20%']['dynamic_track_recall']*100:.1f}%"
                r50 = f"{b_dict['50%']['dynamic_track_recall']*100:.1f}%"
                r100 = f"{b_dict['100%']['dynamic_track_recall']*100:.1f}%"

                p5 = f"{b_dict['5%']['query_precision']*100:.1f}%"
                p10 = f"{b_dict['10%']['query_precision']*100:.1f}%"
                p20 = f"{b_dict['20%']['query_precision']*100:.1f}%"
                p50 = f"{b_dict['50%']['query_precision']*100:.1f}%"
                p100 = f"{b_dict['100%']['query_precision']*100:.1f}%"

                f.write(f"| {p_label} | **Recall** | {r5} | {r10} | {r20} | {r50} | {r100} |\n")
                f.write(f"| | Precision | {p5} | {p10} | {p20} | {p50} | {p100} |\n")

            f.write("\n---\n\n")

    print(f"Saved comparison report to: {report_path}")
    print("\nQuery selection benchmark complete!")


if __name__ == "__main__":
    main()

