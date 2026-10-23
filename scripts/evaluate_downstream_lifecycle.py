#!/usr/bin/env python3
"""Axis 3: Downstream SLAM Landmark Lifecycle & Scene Graph Fusion Evaluation.

Evaluates how router policy decisions directly impact downstream SLAM landmark
lifecycle state management and contextual scene graph entity maintenance:
1. Landmark Retention Precision & Recall: correctly retaining stationary landmarks
   vs. purging invalid/relocated objects and mobile agents.
2. False Landmark Invalidation Rate: legitimate stationary landmarks incorrectly purged.
3. Relocation Detection Precision & Recall: correctly identifying physically displaced objects.
4. Pose History Decoupling: verifying relocated objects maintain separate spatio-temporal
   pose records rather than corrupting a single static anchor position.
5. Trajectory ATE/RPE: marked explicitly as NOT_EVALUATED (zero synthetic placeholders).

Evaluation Inputs:
- Router decision records from Axis 2: vlm/outputs/benchmark_runs/<run_id>/router_decisions_<policy>.jsonl
- Independent evaluator ground truth: vlm/benchmarks/paper_benchmark_ground_truth.json

Outputs:
- vlm/outputs/benchmark_runs/<run_id>/downstream_lifecycle_summary.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from estimator_aware_vlm.contextual_scene_graph_fusion import (
    ContextualSceneGraphFusion,
    ContextualSceneGraphNode,
    ContextualMovability,
    ObservedMotionState,
)


def determine_downstream_landmark_action(
    predicted_movability: str,
    observed_motion: str,
    measured_displacement_m: Optional[float] = None,
    displacement_uncertainty_m: Optional[float] = None,
) -> Tuple[str, str, bool]:
    """Map predicted movability and observed motion to operational SLAM landmark action.

    Returns:
    - action: "RETAIN_LANDMARK" | "INVALIDATE_RELOCATED" | "INVALIDATE_DYNAMIC_AGENT"
    - reason: Descriptive justification
    - decouple_pose: True if relocated object must spawn separate spatio-temporal pose
    """
    mov = str(predicted_movability).upper().strip()
    mot = str(observed_motion).upper().strip()

    # Rule 1: Mobile embodied agents must NEVER serve as static SLAM landmarks
    if mov == "MOBILE" or mot == "DYNAMIC":
        return "INVALIDATE_DYNAMIC_AGENT", "PURGED_MOBILE_AGENT_NON_STATIC", False

    # Rule 2: Statistically significant physical displacement of movable entities
    # Fixed infrastructure (IMMOVABLE) cannot physically relocate; only transportable objects (MOVABLE)
    # or unresolved entities with measured displacement undergo relocation invalidation.
    if mov != "IMMOVABLE" and measured_displacement_m is not None and displacement_uncertainty_m is not None:
        disp = float(measured_displacement_m)
        sigma = float(displacement_uncertainty_m)
        if disp >= 0.25 and disp >= 2.5 * sigma:
            return "INVALIDATE_RELOCATED", f"PURGED_RELOCATED_DISPLACEMENT_{disp:.3f}M", True

    # Rule 3: Confirmed stationary object or infrastructure
    if mov in ["IMMOVABLE", "MOVABLE"] and mot in ["STATIC", "UNKNOWN"]:
        return "RETAIN_LANDMARK", "CONFIRMED_STATISTICALLY_STATIONARY_LANDMARK", False

    # Fallback for unclassified / deferred items: conservative retention with warning
    return "RETAIN_LANDMARK", "CONSERVATIVE_UNRESOLVED_RETENTION", False


def load_real_observations() -> Dict[str, Any]:
    """Load real multi-session observation history indexed by entity ID and observation ID."""
    obs_by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    obs_by_id: Dict[str, Dict[str, Any]] = {}
    for cand in [
        repo_root / "data" / "observations" / "object_observations.jsonl",
        repo_root / "outputs" / "object_observations.jsonl",
    ]:
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            gid = rec.get("global_entity_id")
                            oid = rec.get("obs_id")
                            if gid:
                                obs_by_entity[gid].append(rec)
                            if oid:
                                obs_by_id[oid] = rec
                        except Exception:
                            pass
            break
    combined: Dict[str, Any] = dict(obs_by_entity)
    combined["_by_id"] = obs_by_id
    combined["_by_entity"] = obs_by_entity
    return combined


def evaluate_policy_lifecycle(
    policy_key: str,
    router_records: List[Dict[str, Any]],
    inputs_map: Dict[str, Dict[str, Any]],
    ground_truth_map: Dict[str, Dict[str, Any]],
    real_obs_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate downstream SLAM landmark lifecycle for a single router policy."""
    fusion = ContextualSceneGraphFusion()
    if real_obs_map is None:
        real_obs_map = load_real_observations()

    # Denominators & Counters
    total_instances = len(router_records)
    retained_count = 0
    invalidated_relocated_count = 0
    invalidated_agent_count = 0

    # Confusion matrix: Predicted Action vs. Ground Truth Action
    # Actions: RETAIN_LANDMARK, INVALIDATE_RELOCATED, INVALIDATE_DYNAMIC_AGENT
    actions = ["RETAIN_LANDMARK", "INVALIDATE_RELOCATED", "INVALIDATE_DYNAMIC_AGENT"]
    cm = {a: {p: 0 for p in actions} for a in actions}

    decoupled_pose_count = 0
    expected_decoupled_count = 0

    instance_outcomes = []

    for rec in router_records:
        eid = rec["physical_instance_id"]
        scls = rec["semantic_class"]
        pred_mov = rec.get("final_prediction", "UNKNOWN")

        input_data = inputs_map.get(eid, {})
        est_data = input_data.get("estimator_evidence", {})

        # Strictly extract measured quantities (fail closed: None if unmeasured)
        obs_motion = est_data.get("observed_motion", "UNKNOWN")
        disp_m = est_data.get("measured_displacement_m")
        if disp_m is None:
            disp_m = est_data.get("world_displacement_m")
        disp_sigma = est_data.get("displacement_uncertainty_m")

        # Execute downstream SLAM landmark policy
        downstream_action, reason, decouple_pose = determine_downstream_landmark_action(
            predicted_movability=pred_mov,
            observed_motion=obs_motion,
            measured_displacement_m=disp_m,
            displacement_uncertainty_m=disp_sigma,
        )

        gt_rec = ground_truth_map.get(eid, {})
        gt_action = gt_rec.get("slam_landmark_validity_gt", "RETAIN_LANDMARK")

        cm[gt_action][downstream_action] += 1

        if downstream_action == "RETAIN_LANDMARK":
            retained_count += 1
        elif downstream_action == "INVALIDATE_RELOCATED":
            invalidated_relocated_count += 1
        elif downstream_action == "INVALIDATE_DYNAMIC_AGENT":
            invalidated_agent_count += 1

        if decouple_pose:
            decoupled_pose_count += 1
        if gt_action == "INVALIDATE_RELOCATED":
            expected_decoupled_count += 1

        # Feed into authoritative ContextualSceneGraphFusion
        routed_result = {
            "decision": rec.get("decision", "RESOLVE_CHEAP"),
            "prediction": pred_mov,
            "observed_motion": obs_motion,
            "evidence_provenance": "ROUTER_BENCHMARK",
            "relocation_evaluable": (obs_motion == "DISPLACED"),
            "predicted_landmark_action": "RETAIN" if downstream_action == "RETAIN_LANDMARK" else "INVALIDATE",
            "landmark_action_reason": reason,
            "motion_confidence": 0.95,
            "world_displacement_m": disp_m,
            "displacement_uncertainty_m": disp_sigma,
        }

        # Check for real observations associated with this instance
        src_obs_ids = input_data.get("source_obs_ids", [])
        matched_obs: List[Dict[str, Any]] = []
        if eid in real_obs_map:
            val = real_obs_map[eid]
            if isinstance(val, list):
                matched_obs.extend(val)
        by_id_map = real_obs_map.get("_by_id", real_obs_map)
        for soid in src_obs_ids:
            if soid in by_id_map:
                o = by_id_map[soid]
                if isinstance(o, dict) and o not in matched_obs:
                    matched_obs.append(o)
                elif isinstance(o, list):
                    for item in o:
                        if item not in matched_obs:
                            matched_obs.append(item)

        node = None
        if matched_obs:
            # Sort chronologically by timestamp
            matched_obs.sort(key=lambda o: float(o.get("timestamp", 0.0) or 0.0))
            # Feed real observations sequentially into fusion
            for o_idx, ro in enumerate(matched_obs):
                is_last_obs = (o_idx == len(matched_obs) - 1)
                obs_meta = {
                    "timestamp": ro.get("timestamp"),
                    "timestamp_raw": ro.get("timestamp"),
                    "timestamp_session_s": ro.get("timestamp"),
                    "temporal_valid": True,
                    "source_session": ro.get("source_session", "session_01"),
                    "obs_id": ro.get("obs_id", f"obs_{eid}_{o_idx}"),
                    "local_track_id": ro.get("local_track_id", eid),
                    "centroid_floor": ro.get("centroid_floor"),
                    "spatial_extent_m": ro.get("spatial_extent_m", 0.5),
                    "depth_valid": ro.get("depth_valid", True),
                    "coordinate_frame": "floor_map",
                    "is_relocation_event": decouple_pose and is_last_obs,
                }
                node = fusion.update_or_create_node(
                    entity_id=eid,
                    semantic_class=scls,
                    routed_result=routed_result,
                    observation_meta=obs_meta,
                )
        else:
            # Zero fake spatial coordinates when observation logs are absent
            obs_meta = {
                "timestamp": None,
                "timestamp_raw": None,
                "timestamp_session_s": None,
                "temporal_valid": False,
                "source_session": input_data.get("source_session", "session_01"),
                "obs_id": f"unobserved_{eid}",
                "local_track_id": eid,
                "centroid": None,
                "centroid_floor": None,
                "spatial_extent_m": None,
                "depth_valid": False,
                "coordinate_frame": "unobserved",
                "is_relocation_event": decouple_pose,
            }
            node = fusion.update_or_create_node(
                entity_id=eid,
                semantic_class=scls,
                routed_result=routed_result,
                observation_meta=obs_meta,
            )

        instance_outcomes.append({
            "physical_instance_id": eid,
            "semantic_class": scls,
            "predicted_movability": pred_mov,
            "downstream_action": downstream_action,
            "ground_truth_action": gt_action,
            "action_correct": (downstream_action == gt_action),
            "decouple_pose_applied": decouple_pose,
            "slam_invalidated": node.slam_invalidated,
            "action_reason": reason,
        })

    # Metric 1: Landmark Retention Precision & Recall
    # Target class: RETAIN_LANDMARK
    tp_retain = cm["RETAIN_LANDMARK"]["RETAIN_LANDMARK"]
    fp_retain = cm["INVALIDATE_RELOCATED"]["RETAIN_LANDMARK"] + cm["INVALIDATE_DYNAMIC_AGENT"]["RETAIN_LANDMARK"]
    fn_retain = cm["RETAIN_LANDMARK"]["INVALIDATE_RELOCATED"] + cm["RETAIN_LANDMARK"]["INVALIDATE_DYNAMIC_AGENT"]

    prec_retain = (tp_retain / (tp_retain + fp_retain)) if (tp_retain + fp_retain) > 0 else 0.0
    rec_retain = (tp_retain / (tp_retain + fn_retain)) if (tp_retain + fn_retain) > 0 else 0.0
    f1_retain = (2 * prec_retain * rec_retain / (prec_retain + rec_retain)) if (prec_retain + rec_retain) > 0 else 0.0

    # Metric 2: False Landmark Invalidation Rate
    # Proportion of legitimate stationary landmarks that were mistakenly purged
    total_valid_landmarks = sum(cm["RETAIN_LANDMARK"].values())
    false_invalidation_rate = (fn_retain / total_valid_landmarks) if total_valid_landmarks > 0 else 0.0

    # Metric 3: Relocation Detection Precision & Recall
    # Target class: INVALIDATE_RELOCATED
    tp_reloc = cm["INVALIDATE_RELOCATED"]["INVALIDATE_RELOCATED"]
    fp_reloc = cm["RETAIN_LANDMARK"]["INVALIDATE_RELOCATED"] + cm["INVALIDATE_DYNAMIC_AGENT"]["INVALIDATE_RELOCATED"]
    fn_reloc = cm["INVALIDATE_RELOCATED"]["RETAIN_LANDMARK"] + cm["INVALIDATE_RELOCATED"]["INVALIDATE_DYNAMIC_AGENT"]

    prec_reloc = (tp_reloc / (tp_reloc + fp_reloc)) if (tp_reloc + fp_reloc) > 0 else 0.0
    rec_reloc = (tp_reloc / (tp_reloc + fn_reloc)) if (tp_reloc + fn_reloc) > 0 else 0.0
    f1_reloc = (2 * prec_reloc * rec_reloc / (prec_reloc + rec_reloc)) if (prec_reloc + rec_reloc) > 0 else 0.0

    # Metric 4: Dynamic Agent Rejection Precision & Recall
    # Target class: INVALIDATE_DYNAMIC_AGENT
    tp_agent = cm["INVALIDATE_DYNAMIC_AGENT"]["INVALIDATE_DYNAMIC_AGENT"]
    fp_agent = cm["RETAIN_LANDMARK"]["INVALIDATE_DYNAMIC_AGENT"] + cm["INVALIDATE_RELOCATED"]["INVALIDATE_DYNAMIC_AGENT"]
    fn_agent = cm["INVALIDATE_DYNAMIC_AGENT"]["RETAIN_LANDMARK"] + cm["INVALIDATE_DYNAMIC_AGENT"]["INVALIDATE_RELOCATED"]

    prec_agent = (tp_agent / (tp_agent + fp_agent)) if (tp_agent + fp_agent) > 0 else 0.0
    rec_agent = (tp_agent / (tp_agent + fn_agent)) if (tp_agent + fn_agent) > 0 else 0.0
    f1_agent = (2 * prec_agent * rec_agent / (prec_agent + rec_agent)) if (prec_agent + rec_agent) > 0 else 0.0

    # Overall Landmark Action Accuracy
    correct_actions_total = tp_retain + tp_reloc + tp_agent
    overall_action_acc = correct_actions_total / max(1, total_instances)

    return {
        "policy_key": policy_key,
        "total_instances_evaluated": total_instances,
        "overall_landmark_action_accuracy": round(overall_action_acc, 4),
        "landmark_retention": {
            "true_positives": tp_retain,
            "false_positives_corruption_risk": fp_retain,
            "false_negatives_lost_landmarks": fn_retain,
            "precision": round(prec_retain, 4),
            "recall": round(rec_retain, 4),
            "f1_score": round(f1_retain, 4),
            "false_invalidation_rate": round(false_invalidation_rate, 4),
        },
        "relocation_handling": {
            "true_positives": tp_reloc,
            "false_positives": fp_reloc,
            "false_negatives": fn_reloc,
            "precision": round(prec_reloc, 4),
            "recall": round(rec_reloc, 4),
            "f1_score": round(f1_reloc, 4),
            "decoupled_pose_count": decoupled_pose_count,
            "expected_decoupled_count": expected_decoupled_count,
        },
        "dynamic_agent_rejection": {
            "true_positives": tp_agent,
            "false_positives": fp_agent,
            "false_negatives": fn_agent,
            "precision": round(prec_agent, 4),
            "recall": round(rec_agent, 4),
            "f1_score": round(f1_agent, 4),
        },
        "action_confusion_matrix": cm,
        "trajectory_metrics": {
            "trajectory_ate_m": "NOT_EVALUATED",
            "trajectory_rpe_m": "NOT_EVALUATED",
            "status": "EXCLUDED_NO_GROUND_TRUTH_TRAJECTORY_RIG",
            "scientific_rationale": "Trajectory ATE/RPE requires a calibrated external mocap or survey-grade LiDAR trajectory ground truth rig, which is not present in this dataset. We report exact landmark lifecycle precision, recall, and false invalidation rates rather than unverified synthetic error estimates.",
        },
        "instance_outcomes": instance_outcomes,
    }


def evaluate_downstream_lifecycle(
    router_run_dir: Path,
    output_run_dir: Optional[Path] = None,
    inputs_manifest_path: Optional[Path] = None,
    ground_truth_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Axis 3 Downstream SLAM Lifecycle benchmark across all router policies."""
    inputs_manifest = Path(inputs_manifest_path) if inputs_manifest_path else (repo_root / "benchmarks" / "paper_benchmark_inputs.json")
    gt_manifest = Path(ground_truth_manifest_path) if ground_truth_manifest_path else (repo_root / "benchmarks" / "paper_benchmark_ground_truth.json")

    with open(inputs_manifest, "r", encoding="utf-8") as f:
        inputs_list = json.load(f)["instances"]
        inputs_map = {inst["physical_instance_id"]: inst for inst in inputs_list}

    with open(gt_manifest, "r", encoding="utf-8") as f:
        gt_map = json.load(f)["labels"]

    router_run_dir = router_run_dir.resolve()
    if output_run_dir is None:
        output_run_dir = router_run_dir
    else:
        output_run_dir = output_run_dir.resolve()

    # Find all router decision files
    decision_files = sorted(router_run_dir.glob("router_decisions_*.jsonl"))
    if not decision_files:
        raise FileNotFoundError(f"No router_decisions_*.jsonl files found in {router_run_dir}")

    print("=" * 80)
    print("AXIS 3: DOWNSTREAM SLAM LANDMARK LIFECYCLE EVALUATION")
    print(f"Router Run Directory: {router_run_dir}")
    print(f"Policy Decision Files Found: {len(decision_files)}")
    print(f"Output Directory: {output_run_dir}")
    print("=" * 80)

    summary: Dict[str, Any] = {
        "benchmark_axis": "Axis 3: Downstream SLAM Landmark Lifecycle",
        "timestamp": time.time(),
        "router_run_dir": str(router_run_dir.relative_to(repo_root.resolve())).replace("\\", "/"),
        "policies": {},
    }

    for df in decision_files:
        pkey = df.stem.replace("router_decisions_", "").upper()
        # Normalize policy key
        with open(df, "r", encoding="utf-8") as pf:
            recs = [json.loads(line) for line in pf if line.strip()]

        if not recs:
            continue

        actual_pkey = recs[0].get("policy_key", pkey)
        print(f"\n>>> Evaluating Downstream SLAM Lifecycle for: {actual_pkey} <<<")

        lifecycle_metrics = evaluate_policy_lifecycle(
            policy_key=actual_pkey,
            router_records=recs,
            inputs_map=inputs_map,
            ground_truth_map=gt_map,
        )

        summary["policies"][actual_pkey] = lifecycle_metrics

        ret = lifecycle_metrics["landmark_retention"]
        rel = lifecycle_metrics["relocation_handling"]
        agent = lifecycle_metrics["dynamic_agent_rejection"]
        print(f"  • Overall Action Accuracy: {lifecycle_metrics['overall_landmark_action_accuracy']*100:.1f}%")
        print(f"  • Landmark Retention: Precision={ret['precision']*100:.1f}%, Recall={ret['recall']*100:.1f}%, False Invalidation Rate={ret['false_invalidation_rate']*100:.1f}%")
        print(f"  • Corruption Risk (False Positives Retained): {ret['false_positives_corruption_risk']} instances")
        print(f"  • Relocation Detection: Precision={rel['precision']*100:.1f}%, Recall={rel['recall']*100:.1f}% (Decoupled Poses: {rel['decoupled_pose_count']}/{rel['expected_decoupled_count']})")
        print(f"  • Mobile Agent Rejection: Precision={agent['precision']*100:.1f}%, Recall={agent['recall']*100:.1f}%")

    out_file = output_run_dir / "downstream_lifecycle_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDownstream lifecycle evaluation finished. Summary saved to: {out_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate downstream SLAM landmark lifecycle for router policies.")
    parser.add_argument("--router-dir", type=Path, default=None, help="Directory containing router_decisions_*.jsonl files")
    parser.add_argument("--inputs-manifest", type=Path, default=None, help="Path to custom inputs manifest JSON")
    parser.add_argument("--ground-truth-manifest", type=Path, default=None, help="Path to custom ground-truth manifest JSON")
    args = parser.parse_args()

    rdir = args.router_dir
    if rdir is None:
        # Search for latest router_eval run
        bench_runs = repo_root / "outputs" / "benchmark_runs"
        router_runs = sorted(bench_runs.glob("router_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if router_runs:
            rdir = router_runs[0]
            print(f"Using latest router eval run: {rdir.name}")
        else:
            raise FileNotFoundError("No router_eval_* directories found in vlm/outputs/benchmark_runs")

    evaluate_downstream_lifecycle(
        router_run_dir=rdir,
        inputs_manifest_path=args.inputs_manifest,
        ground_truth_manifest_path=args.ground_truth_manifest,
    )
