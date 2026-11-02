"""Estimator-aware query selection policy benchmark.

Features:
- Observation-level selection with exact image SHA-256 and causal estimator features.
- 1,000 random permutation draws for P2 routing distribution with empirical randomization intervals.
- Distinct targets: Observed Motion (Target 1) vs Self-Mobile Affordance (Target 2).
- Formal policy naming: "Estimator-Context Query-Priority Heuristic (P7)".
- P7 temporal decay ablation: "P7 (Ablation: No Temporal Decay)".
- Exports selected observation manifests at 10% budget for live VLM execution (Run B).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def annotation_to_vlm_class(row: Dict[str, str]) -> Optional[str]:
    """Map annotation labels to the 3-class VLM evaluation schema."""
    mob = row.get("physical_mobility", "")
    cls = row.get("object_class", "")
    if mob == "STRUCTURALLY_IMMOVABLE":
        return "IMMOVABLE"
    if cls == "person":
        return "MOBILE"
    if mob == "MOVABLE":
        return "MOVABLE"
    return None


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
    sharpness_log = float(obs.get("sharpness_log", 2.0))
    crop_area_log = float(obs.get("crop_area_log", 3.0))
    apparent_disp = float(obs.get("apparent_camera_frame_displacement_m", 0.0) or 0.0)

    # Causal estimator features (prefix computation)
    causal_disp = float(obs.get("causal_world_displacement_m") or 0.0)
    causal_dispersion = float(obs.get("causal_centroid_dispersion_m") or 0.0)
    elapsed_sec = float(obs.get("causal_elapsed_sec", 0.0) or 0.0)

    sem_cls = str(obs.get("semantic_class", "")).lower()
    world_valid = bool(obs.get("world_position_valid", False))

    scores: Dict[str, float] = {}

    # P0: Zero query
    scores["P0_zero"] = 0.0

    # P1: Always query
    scores["P1_always"] = 1.0

    # P2: Random baseline (hash-based fallback for deterministic individual scoring)
    h = int(hashlib.md5(str(obs.get("obs_id", "")).encode("utf-8")).hexdigest()[:8], 16)
    scores["P2_random"] = float(h % 10000) / 10000.0

    # P3: Detector Uncertainty: highest score when confidence is near 0.5
    scores["P3_detector_uncertainty"] = 1.0 - abs(conf - 0.5) * 2.0

    # P4: Visual Quality / Sharpness heuristic
    scores["P4_visual_quality"] = sharpness_log * crop_area_log

    # P5: Naive Apparent Displacement (camera frame - uncompensated)
    scores["P5_apparent_disp"] = apparent_disp

    # P6: Causal Centroid Dispersion
    scores["P6_estimator_uncertainty"] = causal_dispersion * (1.2 if world_valid else 0.8)

    # P7: Estimator-Context Query-Priority Heuristic (Proposed)
    # 1. Semantic prior
    w_sem = SEMANTIC_PRIOR_WEIGHTS.get(sem_cls, DEFAULT_SEMANTIC_WEIGHT)

    # 2. Causal displacement risk
    disp_risk = sigmoid(3.0 * (apparent_disp - 0.20)) if not world_valid else sigmoid(4.0 * (causal_disp - 0.15))

    # 3. Visual perceptibility gate
    gate_sharp = sigmoid(2.5 * (sharpness_log - 2.2))
    gate_size = sigmoid(2.0 * (crop_area_log - 3.4))
    vis_gate = gate_sharp * gate_size

    # 4. Time-based early-intervention weight (tau = 5.0 seconds)
    tau = 5.0
    temp_weight = math.exp(-elapsed_sec / tau)

    # Combined priority score
    p7_score = (0.55 * w_sem * disp_risk + 0.45 * causal_dispersion) * (0.3 + 0.7 * vis_gate) * (0.4 + 0.6 * temp_weight)
    scores["P7_estimator_voi"] = float(p7_score)

    # P7 ablation: Without temporal decay (to evaluate temporal term contribution)
    p7_no_temp = (0.55 * w_sem * disp_risk + 0.45 * causal_dispersion) * (0.3 + 0.7 * vis_gate)
    scores["P7_no_temporal"] = float(p7_no_temp)

    return scores


def evaluate_random_routing_distribution(
    observations: List[Dict[str, Any]],
    budgets: List[float],
    moving_tracks: Set[str],
    self_mobile_tracks: Set[str],
    eligible_moving_tracks: Set[str],
    eligible_self_mobile_tracks: Set[str],
    n_draws: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate 1,000 random permutation draws for P2 routing distribution."""
    rng = np.random.default_rng(seed)
    total_obs = len(observations)
    moving_tracks_count = len(moving_tracks)
    self_mobile_tracks_count = len(self_mobile_tracks)

    obs_track_keys = [f"{o['sequence_id']}/{o['track_id']}" for o in observations]

    draw_results_by_budget: Dict[str, Dict[str, List[float]]] = {}
    for b in budgets:
        b_key = f"{int(round(b * 100))}%"
        draw_results_by_budget[b_key] = {
            "moving_track_recall": [],
            "moving_track_precision": [],
            "moving_track_f1": [],
            "self_mobile_track_recall": [],
            "self_mobile_track_precision": [],
            "self_mobile_track_f1": [],
            "queried_unique_tracks": [],
            "eligible_moving_track_recall": [],
            "eligible_self_mobile_track_recall": [],
        }

    for _ in range(n_draws):
        perm = rng.permutation(total_obs)
        for b in budgets:
            b_key = f"{int(round(b * 100))}%"
            k = max(1, int(round(total_obs * b)))
            selected_indices = perm[:k]
            queried_tracks = {obs_track_keys[idx] for idx in selected_indices}

            # Interceptions
            intercepted_moving = moving_tracks.intersection(queried_tracks)
            intercepted_mobile = self_mobile_tracks.intersection(queried_tracks)
            intercepted_eligible_moving = eligible_moving_tracks.intersection(queried_tracks)
            intercepted_eligible_mobile = eligible_self_mobile_tracks.intersection(queried_tracks)

            # Target 1: Moving tracks
            m_rec = len(intercepted_moving) / moving_tracks_count if moving_tracks_count > 0 else 0.0
            m_prec = len(intercepted_moving) / len(queried_tracks) if len(queried_tracks) > 0 else 0.0
            m_f1 = (2 * m_prec * m_rec / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0

            # Target 2: Self-mobile tracks
            sm_rec = len(intercepted_mobile) / self_mobile_tracks_count if self_mobile_tracks_count > 0 else 0.0
            sm_prec = len(intercepted_mobile) / len(queried_tracks) if len(queried_tracks) > 0 else 0.0
            sm_f1 = (2 * sm_prec * sm_rec / (sm_prec + sm_rec)) if (sm_prec + sm_rec) > 0 else 0.0

            em_rec = len(intercepted_eligible_moving) / len(eligible_moving_tracks) if len(eligible_moving_tracks) > 0 else 0.0
            esm_rec = len(intercepted_eligible_mobile) / len(eligible_self_mobile_tracks) if len(eligible_self_mobile_tracks) > 0 else 0.0

            d = draw_results_by_budget[b_key]
            d["moving_track_recall"].append(m_rec)
            d["moving_track_precision"].append(m_prec)
            d["moving_track_f1"].append(m_f1)
            d["self_mobile_track_recall"].append(sm_rec)
            d["self_mobile_track_precision"].append(sm_prec)
            d["self_mobile_track_f1"].append(sm_f1)
            d["queried_unique_tracks"].append(float(len(queried_tracks)))
            d["eligible_moving_track_recall"].append(em_rec)
            d["eligible_self_mobile_track_recall"].append(esm_rec)

    # Compute statistics across the 1,000 draws
    summary_by_budget: Dict[str, Dict[str, Any]] = {}
    for b in budgets:
        b_key = f"{int(round(b * 100))}%"
        k = max(1, int(round(total_obs * b)))
        raw = draw_results_by_budget[b_key]

        stats_dict: Dict[str, Any] = {
            "budget_fraction": b,
            "queries_executed": k,
            "total_observations": total_obs,
            "n_draws": n_draws,
        }

        for metric_name, values in raw.items():
            arr = np.array(values)
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr))
            p2_5 = float(np.percentile(arr, 2.5))
            p97_5 = float(np.percentile(arr, 97.5))
            median_v = float(np.median(arr))

            stats_dict[metric_name] = round(mean_v, 4)
            stats_dict[f"{metric_name}_mean"] = round(mean_v, 4)
            stats_dict[f"{metric_name}_std"] = round(std_v, 4)
            stats_dict[f"{metric_name}_p2_5"] = round(p2_5, 4)
            stats_dict[f"{metric_name}_p97_5"] = round(p97_5, 4)
            stats_dict[f"{metric_name}_median"] = round(median_v, 4)

        # Backward compatibility aliases
        stats_dict["dynamic_track_recall"] = stats_dict["moving_track_recall"]
        stats_dict["query_precision"] = stats_dict["moving_track_precision"]
        summary_by_budget[b_key] = stats_dict

    return summary_by_budget


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

    # Load independent human reference annotations
    ann_map = {}
    ann_path = REPO_ROOT / "results" / "track_annotation" / "track_annotations.csv"
    if ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ann_map[f"{row['sequence_id']}/{row['track_id']}"] = row

    all_tracks = set(f"{o['sequence_id']}/{o['track_id']}" for o in observations)

    # Target 1: Actually observed moving tracks (human annotation reference)
    moving_tracks = {
        key for key in all_tracks
        if ann_map.get(key, {}).get("observed_motion") == "MOVING"
    }
    # Target 2: Self-mobile tracks (pedestrians/autonomous agents capable of locomotion)
    self_mobile_tracks = {
        key for key in all_tracks
        if annotation_to_vlm_class(ann_map.get(key, {})) == "MOBILE"
    }

    # High-confidence eligible cohort tracks
    eligible_moving_tracks = {
        key for key in moving_tracks
        if ann_map.get(key, {}).get("track_identity_validity") == "VALID"
        and ann_map.get(key, {}).get("evidence_sufficiency") == "SUFFICIENT"
    }
    eligible_self_mobile_tracks = {
        key for key in self_mobile_tracks
        if ann_map.get(key, {}).get("track_identity_validity") == "VALID"
    }

    total_obs = len(observations)
    total_tracks_count = len(all_tracks)
    moving_tracks_count = len(moving_tracks)
    self_mobile_tracks_count = len(self_mobile_tracks)

    # Deterministic policies
    deterministic_policies = [
        "P0_zero",
        "P1_always",
        "P3_detector_uncertainty",
        "P4_visual_quality",
        "P5_apparent_disp",
        "P6_estimator_uncertainty",
        "P7_estimator_voi",
        "P7_no_temporal",
    ]

    results_by_policy: Dict[str, Dict[str, Any]] = {p: {"budgets": {}} for p in deterministic_policies}

    for p in deterministic_policies:
        for b in budgets:
            b_key = f"{int(round(b * 100))}%"
            if p == "P0_zero":
                k = 0
            elif p == "P1_always":
                k = total_obs
            else:
                k = max(1, int(round(total_obs * b)))

            sorted_items = sorted(scored_obs, key=lambda x: x["scores"][p], reverse=True)
            queried = sorted_items[:k]

            queried_tracks = set(f"{o['sequence_id']}/{o['track_id']}" for o in queried)
            intercepted_moving = moving_tracks.intersection(queried_tracks)
            intercepted_mobile = self_mobile_tracks.intersection(queried_tracks)
            intercepted_eligible_moving = eligible_moving_tracks.intersection(queried_tracks)
            intercepted_eligible_mobile = eligible_self_mobile_tracks.intersection(queried_tracks)

            moving_track_rec = (len(intercepted_moving) / moving_tracks_count) if moving_tracks_count > 0 else 0.0
            moving_track_prec = (len(intercepted_moving) / len(queried_tracks)) if len(queried_tracks) > 0 else 0.0
            moving_track_f1 = (2 * moving_track_prec * moving_track_rec / (moving_track_prec + moving_track_rec)) if (moving_track_prec + moving_track_rec) > 0 else 0.0

            self_mobile_rec = (len(intercepted_mobile) / self_mobile_tracks_count) if self_mobile_tracks_count > 0 else 0.0
            self_mobile_prec = (len(intercepted_mobile) / len(queried_tracks)) if len(queried_tracks) > 0 else 0.0
            self_mobile_f1 = (2 * self_mobile_prec * self_mobile_rec / (self_mobile_prec + self_mobile_rec)) if (self_mobile_prec + self_mobile_rec) > 0 else 0.0

            eligible_moving_rec = (len(intercepted_eligible_moving) / len(eligible_moving_tracks)) if len(eligible_moving_tracks) > 0 else 0.0
            eligible_mobile_rec = (len(intercepted_eligible_mobile) / len(eligible_self_mobile_tracks)) if len(eligible_self_mobile_tracks) > 0 else 0.0

            compute_savings = 1.0 - (k / total_obs) if total_obs > 0 else 0.0
            efficiency_ratio = (moving_track_rec / b) if b > 0 else 0.0

            results_by_policy[p]["budgets"][b_key] = {
                "budget_fraction": b,
                "queries_executed": k,
                "total_observations": total_obs,
                "queried_unique_tracks": len(queried_tracks),
                "compute_savings_pct": round(compute_savings * 100.0, 1),
                "efficiency_ratio": round(efficiency_ratio, 2),
                # Target 1: Observed motion
                "moving_track_recall": round(moving_track_rec, 4),
                "moving_track_precision": round(moving_track_prec, 4),
                "moving_track_f1": round(moving_track_f1, 4),
                "intercepted_moving_tracks": len(intercepted_moving),
                "total_moving_tracks": moving_tracks_count,
                "eligible_moving_track_recall": round(eligible_moving_rec, 4),
                # Target 2: Self-mobile affordance
                "self_mobile_track_recall": round(self_mobile_rec, 4),
                "self_mobile_track_precision": round(self_mobile_prec, 4),
                "self_mobile_track_f1": round(self_mobile_f1, 4),
                "intercepted_self_mobile_tracks": len(intercepted_mobile),
                "total_self_mobile_tracks": self_mobile_tracks_count,
                "eligible_self_mobile_track_recall": round(eligible_mobile_rec, 4),
                # Backward-compatibility aliases
                "dynamic_track_recall": round(moving_track_rec, 4),
                "query_precision": round(moving_track_prec, 4),
                "intercepted_dynamic_tracks": len(intercepted_moving),
                "total_dynamic_tracks": moving_tracks_count,
            }

    # Evaluate 1,000 random permutation draws for P2
    p2_dist = evaluate_random_routing_distribution(
        scored_obs,
        budgets,
        moving_tracks,
        self_mobile_tracks,
        eligible_moving_tracks,
        eligible_self_mobile_tracks,
        n_draws=1000,
        seed=seed,
    )
    results_by_policy["P2_random"] = {"budgets": p2_dist}

    return {
        "total_observations": total_obs,
        "total_tracks": total_tracks_count,
        "total_moving_tracks": moving_tracks_count,
        "total_self_mobile_tracks": self_mobile_tracks_count,
        "total_dynamic_tracks": moving_tracks_count,
        "policies": results_by_policy,
        "scored_observations": scored_obs,
    }


def compute_sha256(file_path: Optional[str]) -> str:
    """Compute sha256 checksum of an image file."""
    if not file_path:
        return ""
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            return ""
    return ""


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

    print("Estimator-Aware Query Selection Benchmark")
    print(f"Loading features: {args.features_file}")

    if not args.features_file.exists():
        raise FileNotFoundError(f"Missing features file: {args.features_file}")

    with args.features_file.open("r", encoding="utf-8") as f:
        all_obs = [json.loads(line) for line in f]

    print(f"Total observations loaded: {len(all_obs)}")

    # Precompute SHA-256 for all observations
    print("Computing SHA-256 for observation crop images...")
    sha_cache: Dict[str, str] = {}
    for obs in all_obs:
        cp = obs.get("crop_path")
        if cp not in sha_cache:
            sha_cache[cp] = compute_sha256(cp)
        obs["image_sha256"] = sha_cache[cp]
    print(f"Computed SHA-256 for {len(sha_cache)} unique image files.")

    budgets = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

    # Evaluate across splits
    splits = ["full", "development", "validation", "held_out"]
    eval_by_split: Dict[str, Any] = {}

    for sp in splits:
        if sp == "full":
            sub_obs = all_obs
        else:
            sub_obs = [o for o in all_obs if o.get("evaluation_split") == sp]
        print(f"\nBenchmarking split: {sp} (n={len(sub_obs)} observations, 1000 draws for P2)...")
        eval_by_split[sp] = evaluate_split(sub_obs, budgets)

    # Save summary json
    summary_path = args.out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        clean_summary = {
            sp: {
                "total_observations": data["total_observations"],
                "total_tracks": data["total_tracks"],
                "total_moving_tracks": data["total_moving_tracks"],
                "total_self_mobile_tracks": data["total_self_mobile_tracks"],
                "total_dynamic_tracks": data["total_dynamic_tracks"],
                "policies": data["policies"],
            }
            for sp, data in eval_by_split.items()
        }
        json.dump(clean_summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Save audited query decisions for the full dataset with image_sha256
    decisions_path = args.out_dir / "query_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for o in eval_by_split["full"]["scored_observations"]:
            record = {
                "sequence_id": o["sequence_id"],
                "track_id": o["track_id"],
                "obs_id": o["obs_id"],
                "frame_index": o.get("frame_index"),
                "timestamp": o.get("timestamp"),
                "semantic_class": o.get("semantic_class"),
                "image_path": o.get("crop_path"),
                "image_sha256": o.get("image_sha256"),
                "scores": o["scores"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved decisions to: {decisions_path}")

    # Generate dedicated 10% budget execution manifests for P7 and P2 (seed 42)
    full_scored = eval_by_split["full"]["scored_observations"]
    k_10 = max(1, int(round(len(full_scored) * 0.10)))

    # P7 top 10%
    p7_sorted = sorted(full_scored, key=lambda x: x["scores"]["P7_estimator_voi"], reverse=True)
    p7_selected = p7_sorted[:k_10]
    p7_manifest = [
        {
            "obs_id": o["obs_id"],
            "sequence_id": o["sequence_id"],
            "track_id": o["track_id"],
            "frame_index": o.get("frame_index"),
            "timestamp": o.get("timestamp"),
            "image_path": o.get("crop_path"),
            "image_sha256": o.get("image_sha256"),
            "semantic_class": o.get("semantic_class"),
            "p7_score": round(o["scores"]["P7_estimator_voi"], 5),
        }
        for o in p7_selected
    ]
    p7_manifest_path = args.out_dir / "p7_selected_queries_10pct.json"
    with p7_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(p7_manifest, f, indent=2)

    # P2 top 10% (from canonical seed 42 draw)
    rng = np.random.default_rng(42)
    p2_perm = rng.permutation(len(full_scored))
    p2_selected = [full_scored[idx] for idx in p2_perm[:k_10]]
    p2_manifest = [
        {
            "obs_id": o["obs_id"],
            "sequence_id": o["sequence_id"],
            "track_id": o["track_id"],
            "frame_index": o.get("frame_index"),
            "timestamp": o.get("timestamp"),
            "image_path": o.get("crop_path"),
            "image_sha256": o.get("image_sha256"),
            "semantic_class": o.get("semantic_class"),
            "p2_draw_seed": 42,
        }
        for o in p2_selected
    ]
    p2_manifest_path = args.out_dir / "p2_selected_queries_10pct.json"
    with p2_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(p2_manifest, f, indent=2)

    # Union of P7 and P2 queries for Live Run B
    union_dict = {o["obs_id"]: o for o in p7_manifest}
    for o in p2_manifest:
        if o["obs_id"] not in union_dict:
            union_dict[o["obs_id"]] = o
    union_manifest = list(union_dict.values())
    union_manifest_path = args.out_dir / "union_selected_queries_10pct.json"
    with union_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(union_manifest, f, indent=2)

    print(f"Exported P7 10% manifest ({len(p7_manifest)} queries): {p7_manifest_path}")
    print(f"Exported P2 10% manifest ({len(p2_manifest)} queries): {p2_manifest_path}")
    print(f"Exported union manifest ({len(union_manifest)} unique queries for Run B): {union_manifest_path}")

    # Generate Markdown Table Report
    report_path = args.out_dir / "policy_comparison_table.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Query-Selection Policy Benchmark Report\n\n")
        f.write("Generated from 885 causal observation feature vectors across 177 tracks.\n\n")
        f.write("> **Methodological Note on Random Routing Baseline (P2)**: "
                "The intervals reported for P2 are empirical **randomization intervals across 1,000 random permutation draws** "
                "of routing decisions over dataset observations. They quantify the variance of stochastic query selection, "
                "not an environment-generalization confidence interval.\n\n")

        for sp in ["full", "development", "validation", "held_out"]:
            res = eval_by_split[sp]
            pols = res["policies"]
            f.write(f"## Evaluation Split: `{sp}`\n\n")
            f.write(f"- Total Observations: **{res['total_observations']}**\n")
            f.write(f"- Total Tracks: **{res['total_tracks']}**\n")
            f.write(f"- Truly Moving Tracks (Target 1 Reference): **{res['total_moving_tracks']}**\n")
            f.write(f"- Self-Mobile Tracks (Target 2 Reference): **{res['total_self_mobile_tracks']}**\n\n")

            f.write("### Policy Comparison: Moving-Object Interception (Target 1: Actual Motion)\n\n")
            f.write("| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |\n")
            f.write("| --- | --- | :---: | :---: | :---: | :---: | :---: |\n")

            policy_labels = [
                ("P0_zero", "P0 (Zero VLM / Baseline)"),
                ("P1_always", "P1 (Always Query / Naive)"),
                ("P2_random", "P2 (Random Subsampling - 1,000 draws)"),
                ("P3_detector_uncertainty", "P3 (Detector Uncertainty)"),
                ("P4_visual_quality", "P4 (Visual Quality / Sharpness)"),
                ("P5_apparent_disp", "P5 (Naive Apparent Motion)"),
                ("P6_estimator_uncertainty", "P6 (Estimator Uncertainty)"),
                ("P7_estimator_voi", "**Estimator-Context Query-Priority Heuristic (P7)**"),
                ("P7_no_temporal", "P7 (Ablation: No Temporal Decay)"),
            ]

            for p_key, p_label in policy_labels:
                b_dict = pols[p_key]["budgets"]
                if p_key == "P2_random":
                    # Format as mean [p2.5, p97.5]
                    r5 = f"{b_dict['5%']['moving_track_recall_mean']*100:.1f}% [{b_dict['5%']['moving_track_recall_p2_5']*100:.1f}-{b_dict['5%']['moving_track_recall_p97_5']*100:.1f}]"
                    r10 = f"{b_dict['10%']['moving_track_recall_mean']*100:.1f}% [{b_dict['10%']['moving_track_recall_p2_5']*100:.1f}-{b_dict['10%']['moving_track_recall_p97_5']*100:.1f}]"
                    r20 = f"{b_dict['20%']['moving_track_recall_mean']*100:.1f}% [{b_dict['20%']['moving_track_recall_p2_5']*100:.1f}-{b_dict['20%']['moving_track_recall_p97_5']*100:.1f}]"
                    r50 = f"{b_dict['50%']['moving_track_recall_mean']*100:.1f}% [{b_dict['50%']['moving_track_recall_p2_5']*100:.1f}-{b_dict['50%']['moving_track_recall_p97_5']*100:.1f}]"
                    r100 = f"{b_dict['100%']['moving_track_recall_mean']*100:.1f}% [{b_dict['100%']['moving_track_recall_p2_5']*100:.1f}-{b_dict['100%']['moving_track_recall_p97_5']*100:.1f}]"

                    p5 = f"{b_dict['5%']['moving_track_precision_mean']*100:.1f}% [{b_dict['5%']['moving_track_precision_p2_5']*100:.1f}-{b_dict['5%']['moving_track_precision_p97_5']*100:.1f}]"
                    p10 = f"{b_dict['10%']['moving_track_precision_mean']*100:.1f}% [{b_dict['10%']['moving_track_precision_p2_5']*100:.1f}-{b_dict['10%']['moving_track_precision_p97_5']*100:.1f}]"
                    p20 = f"{b_dict['20%']['moving_track_precision_mean']*100:.1f}% [{b_dict['20%']['moving_track_precision_p2_5']*100:.1f}-{b_dict['20%']['moving_track_precision_p97_5']*100:.1f}]"
                    p50 = f"{b_dict['50%']['moving_track_precision_mean']*100:.1f}% [{b_dict['50%']['moving_track_precision_p2_5']*100:.1f}-{b_dict['50%']['moving_track_precision_p97_5']*100:.1f}]"
                    p100 = f"{b_dict['100%']['moving_track_precision_mean']*100:.1f}% [{b_dict['100%']['moving_track_precision_p2_5']*100:.1f}-{b_dict['100%']['moving_track_precision_p97_5']*100:.1f}]"
                else:
                    r5 = f"{b_dict['5%']['moving_track_recall']*100:.1f}%"
                    r10 = f"{b_dict['10%']['moving_track_recall']*100:.1f}%"
                    r20 = f"{b_dict['20%']['moving_track_recall']*100:.1f}%"
                    r50 = f"{b_dict['50%']['moving_track_recall']*100:.1f}%"
                    r100 = f"{b_dict['100%']['moving_track_recall']*100:.1f}%"

                    p5 = f"{b_dict['5%']['moving_track_precision']*100:.1f}%"
                    p10 = f"{b_dict['10%']['moving_track_precision']*100:.1f}%"
                    p20 = f"{b_dict['20%']['moving_track_precision']*100:.1f}%"
                    p50 = f"{b_dict['50%']['moving_track_precision']*100:.1f}%"
                    p100 = f"{b_dict['100%']['moving_track_precision']*100:.1f}%"

                f.write(f"| {p_label} | **Recall** | {r5} | {r10} | {r20} | {r50} | {r100} |\n")
                f.write(f"| | Precision | {p5} | {p10} | {p20} | {p50} | {p100} |\n")

            f.write("\n### Policy Comparison: Self-Mobile Affordance Interception (Target 2: Mobile Affordance)\n\n")
            f.write("| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |\n")
            f.write("| --- | --- | :---: | :---: | :---: | :---: | :---: |\n")

            for p_key, p_label in policy_labels:
                b_dict = pols[p_key]["budgets"]
                if p_key == "P2_random":
                    r5 = f"{b_dict['5%']['self_mobile_track_recall_mean']*100:.1f}%"
                    r10 = f"{b_dict['10%']['self_mobile_track_recall_mean']*100:.1f}%"
                    r20 = f"{b_dict['20%']['self_mobile_track_recall_mean']*100:.1f}%"
                    r50 = f"{b_dict['50%']['self_mobile_track_recall_mean']*100:.1f}%"
                    r100 = f"{b_dict['100%']['self_mobile_track_recall_mean']*100:.1f}%"

                    p5 = f"{b_dict['5%']['self_mobile_track_precision_mean']*100:.1f}%"
                    p10 = f"{b_dict['10%']['self_mobile_track_precision_mean']*100:.1f}%"
                    p20 = f"{b_dict['20%']['self_mobile_track_precision_mean']*100:.1f}%"
                    p50 = f"{b_dict['50%']['self_mobile_track_precision_mean']*100:.1f}%"
                    p100 = f"{b_dict['100%']['self_mobile_track_precision_mean']*100:.1f}%"
                else:
                    r5 = f"{b_dict['5%']['self_mobile_track_recall']*100:.1f}%"
                    r10 = f"{b_dict['10%']['self_mobile_track_recall']*100:.1f}%"
                    r20 = f"{b_dict['20%']['self_mobile_track_recall']*100:.1f}%"
                    r50 = f"{b_dict['50%']['self_mobile_track_recall']*100:.1f}%"
                    r100 = f"{b_dict['100%']['self_mobile_track_recall']*100:.1f}%"

                    p5 = f"{b_dict['5%']['self_mobile_track_precision']*100:.1f}%"
                    p10 = f"{b_dict['10%']['self_mobile_track_precision']*100:.1f}%"
                    p20 = f"{b_dict['20%']['self_mobile_track_precision']*100:.1f}%"
                    p50 = f"{b_dict['50%']['self_mobile_track_precision']*100:.1f}%"
                    p100 = f"{b_dict['100%']['self_mobile_track_precision']*100:.1f}%"

                f.write(f"| {p_label} | **Recall** | {r5} | {r10} | {r20} | {r50} | {r100} |\n")
                f.write(f"| | Precision | {p5} | {p10} | {p20} | {p50} | {p100} |\n")

            f.write("\n---\n\n")

    print(f"Saved comparison report to: {report_path}")
    print("\nQuery selection benchmark complete!")


if __name__ == "__main__":
    main()
