#!/usr/bin/env python3
"""Axis 2: Estimator-Aware Selective Router Evaluation & Policy Suite.

Empirically evaluates whether the proposed estimator-aware selective router (B6)
makes better decisions than standard baselines:
- B0: Majority Class Baseline (predicts dominant class MOVABLE, 0% VLM queries)
- B1: Semantic Prior Only (R0 policy, static lookup, 0% VLM queries)
- B2: Geometry & Observed Motion Only (R1 policy, observed displacement, 0% VLM queries)
- B3: Zero-Shot Visual Classifier - Crop Only (100% VLM queries)
- B4: Zero-Shot Visual Classifier - Crop + Context (100% VLM queries)
- B5: Always-Query-VLM (R4 reference, queries all image-eligible instances)
- B6: Estimator-Aware Selective Router (R3 proposed policy: delta_L_SLAM vs lambda * C_VLM)
- B7: Oracle Upper Bound (queries VLM if and only if needed to achieve correct GT)

Also evaluates single-component ablations of B6:
- A1: No-Uncertainty Router (disables displacement sigma gate)
- A2: No-Displacement Router (disables world displacement measurement)
- A3: No-Prior Router (neutralizes semantic prior)
- A4: Crop-Only Router (uses CROP_ONLY when VLM is triggered)

Evaluation Modes:
1. 'replay': Uses recorded VLM predictions from a visual classification run directory.
   Enables deterministic, fast re-evaluation of routing policies and ablations.
2. 'live': Executes strict-live inference against the VLM server when ESCALATE is chosen.
   Records honest wire telemetry, latencies, and cache status (0 cache hits).

Ground Truth Isolation:
- Inputs read strictly from benchmarks/paper_benchmark_inputs.json.
- Evaluator joins predictions against benchmarks/paper_benchmark_ground_truth.json strictly post-inference.

Outputs:
- outputs/benchmark_runs/<run_id>/router_policies_summary.json
- outputs/benchmark_runs/<run_id>/router_decisions_<policy>.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
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

from estimator_aware_vlm.router_policy import (
    evaluate_policy_r0,
    evaluate_policy_r1,
    evaluate_policy_r2,
    evaluate_policy_r3,
    evaluate_policy_r4,
    RouterDecision,
    DEFAULT_VLM_COST,
    CANONICAL_IMMOVABLE,
    CANONICAL_MOVABLE,
    CANONICAL_MOBILE,
    AMBIGUOUS_AFFORDANCE_CLASSES,
)
from estimator_aware_vlm.multimodal_inference_engine import (
    MultimodalInferenceEngine,
    WireInferenceRecord,
    CLASSES,
)
from scripts.evaluate_visual_classification import build_canonical_prompt


POLICY_KEYS = [
    "B0_MAJORITY",
    "B1_SEMANTIC_PRIOR",
    "B2_GEOMETRY_MOTION",
    "B3_ZERO_SHOT_CROP_ONLY",
    "B4_ZERO_SHOT_CROP_PLUS_CONTEXT",
    "B5_ALWAYS_VLM",
    "B6_ESTIMATOR_AWARE_ROUTER",
    "B7_ORACLE",
    "A1_NO_UNCERTAINTY",
    "A2_NO_DISPLACEMENT",
    "A3_NO_PRIOR",
    "A4_CROP_ONLY",
]


def extract_evidence_from_input_record(input_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Format inference-visible instance metadata into evidence dictionary for router policies.
    
    Strictly fail-closed: If estimator evidence or displacement is missing, world_displacement_m
    and displacement_uncertainty_m remain None, and observed_motion defaults to 'UNKNOWN'.
    Never infers motion or displacement from entity ID or semantic class.
    """
    eid = input_rec["physical_instance_id"]
    scls = input_rec["semantic_class"]
    vis = input_rec.get("visual_inputs", {})
    usable_views = vis.get("usable_views", 0)
    has_images = vis.get("has_images", False) and (usable_views > 0)

    # Read verified estimator evidence block
    est = input_rec.get("estimator_evidence", {})

    # Extract strictly measured quantities (fail closed: None if unavailable)
    obs_motion = est.get("observed_motion", "UNKNOWN")
    world_disp = est.get("measured_displacement_m")
    disp_sigma = est.get("displacement_uncertainty_m")
    has_valid_pose = est.get("has_valid_pose", False)
    depth_valid = est.get("depth_valid", False)
    total_obs = est.get("observation_count", 0)
    measurement_status = est.get("measurement_status", "MEASUREMENT_UNAVAILABLE")

    # Semantic class prior mobility (derived strictly from ontology categories, never entity ID)
    prior_mobility = "MOVABLE"
    if scls in CANONICAL_IMMOVABLE:
        prior_mobility = "IMMOVABLE"
    elif scls in CANONICAL_MOBILE:
        prior_mobility = "MOBILE"

    return {
        "entity_id": eid,
        "semantic_class": scls,
        "class_prior_mobility": prior_mobility,
        "observed_motion": obs_motion,
        "measurement_status": measurement_status,
        "world_displacement_m": world_disp,
        "displacement_uncertainty_m": disp_sigma,
        "has_valid_pose": has_valid_pose,
        "depth_valid": depth_valid,
        "total_observations": total_obs,
        "views_count": usable_views,
        "has_images": has_images,
    }


def execute_policy_decision(
    policy_key: str,
    evidence: Dict[str, Any],
    lambda_cost: float = DEFAULT_VLM_COST,
    oracle_gt_class: Optional[str] = None,
    oracle_vlm_prediction: Optional[str] = None,
) -> Tuple[RouterDecision, Literal["CROP_ONLY", "CROP_PLUS_CONTEXT"]]:
    """Execute routing policy decision given instance evidence."""
    eid = evidence["entity_id"]
    scls = evidence["semantic_class"]
    has_images = evidence.get("has_images", False)
    default_view_mode: Literal["CROP_ONLY", "CROP_PLUS_CONTEXT"] = "CROP_PLUS_CONTEXT"

    # B0: Majority Class Baseline
    if policy_key == "B0_MAJORITY":
        return RouterDecision(
            entity_id=eid,
            decision="RESOLVE_CHEAP",
            reason_codes=["B0_MAJORITY_CLASS_MOVABLE"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.60,
        ), default_view_mode

    # B1: Semantic Prior Only
    elif policy_key == "B1_SEMANTIC_PRIOR":
        dec = evaluate_policy_r0(evidence)
        return dec, default_view_mode

    # B2: Geometry & Observed Motion Only
    elif policy_key == "B2_GEOMETRY_MOTION":
        dec = evaluate_policy_r1(evidence)
        return dec, default_view_mode

    # B3: Zero-Shot Crop Only (Always query VLM with CROP_ONLY)
    elif policy_key == "B3_ZERO_SHOT_CROP_ONLY":
        if not has_images:
            return RouterDecision(
                entity_id=eid,
                decision="DEFER",
                reason_codes=["NO_IMAGES_FOR_VLM"],
                vlm_required=False,
                cheap_label="UNKNOWN",
                confidence=0.0,
            ), "CROP_ONLY"
        return RouterDecision(
            entity_id=eid,
            decision="ESCALATE",
            reason_codes=["B3_ALWAYS_QUERY_CROP_ONLY"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        ), "CROP_ONLY"

    # B4: Zero-Shot Crop + Context (Always query VLM with CROP_PLUS_CONTEXT)
    elif policy_key == "B4_ZERO_SHOT_CROP_PLUS_CONTEXT":
        if not has_images:
            return RouterDecision(
                entity_id=eid,
                decision="DEFER",
                reason_codes=["NO_IMAGES_FOR_VLM"],
                vlm_required=False,
                cheap_label="UNKNOWN",
                confidence=0.0,
            ), "CROP_PLUS_CONTEXT"
        return RouterDecision(
            entity_id=eid,
            decision="ESCALATE",
            reason_codes=["B4_ALWAYS_QUERY_CROP_PLUS_CONTEXT"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        ), "CROP_PLUS_CONTEXT"

    # B5: Always-Query-VLM (R4 policy)
    elif policy_key == "B5_ALWAYS_VLM":
        dec = evaluate_policy_r4(evidence)
        return dec, "CROP_PLUS_CONTEXT"

    # B6: Proposed Estimator-Aware Selective Router (R3 policy)
    elif policy_key == "B6_ESTIMATOR_AWARE_ROUTER":
        dec = evaluate_policy_r3(evidence, lambda_cost=lambda_cost)
        return dec, "CROP_PLUS_CONTEXT"

    # B7: Oracle Upper Bound
    elif policy_key == "B7_ORACLE":
        cheap_r0 = evaluate_policy_r0(evidence).cheap_label
        # If cheap prior matches GT, resolve cheaply
        if cheap_r0 == oracle_gt_class:
            return RouterDecision(
                entity_id=eid,
                decision="RESOLVE_CHEAP",
                reason_codes=["ORACLE_CHEAP_SUFFICIENT"],
                vlm_required=False,
                cheap_label=cheap_r0,
                confidence=1.0,
            ), "CROP_PLUS_CONTEXT"
        # Otherwise if VLM is available, query VLM
        if has_images:
            return RouterDecision(
                entity_id=eid,
                decision="ESCALATE",
                reason_codes=["ORACLE_VLM_REQUIRED"],
                vlm_required=True,
                cheap_label=None,
                confidence=1.0,
            ), "CROP_PLUS_CONTEXT"
        return RouterDecision(
            entity_id=eid,
            decision="RESOLVE_CHEAP",
            reason_codes=["ORACLE_NO_IMAGES_FALLBACK"],
            vlm_required=False,
            cheap_label=cheap_r0 or "UNKNOWN",
            confidence=0.5,
        ), "CROP_PLUS_CONTEXT"

    # A1: No-Uncertainty Router (Uncertainty gate disabled, sigma = 0)
    elif policy_key == "A1_NO_UNCERTAINTY":
        ev_ablated = copy.deepcopy(evidence)
        ev_ablated["displacement_uncertainty_m"] = 0.001
        dec = evaluate_policy_r3(ev_ablated, lambda_cost=lambda_cost)
        return dec, "CROP_PLUS_CONTEXT"

    # A2: No-Displacement Router (Geometry displacement ignored)
    elif policy_key == "A2_NO_DISPLACEMENT":
        ev_ablated = copy.deepcopy(evidence)
        ev_ablated["world_displacement_m"] = 0.0
        ev_ablated["observed_motion"] = "STATIC"
        dec = evaluate_policy_r3(ev_ablated, lambda_cost=lambda_cost)
        return dec, "CROP_PLUS_CONTEXT"

    # A3: No-Prior Router (Semantic class prior neutralized)
    elif policy_key == "A3_NO_PRIOR":
        ev_ablated = copy.deepcopy(evidence)
        ev_ablated["class_prior_mobility"] = "UNKNOWN"
        ev_ablated["semantic_class"] = "unclassified_object"
        dec = evaluate_policy_r3(ev_ablated, lambda_cost=lambda_cost)
        return dec, "CROP_PLUS_CONTEXT"

    # A4: Crop-Only Router (R3 routing, but VLM queried with CROP_ONLY)
    elif policy_key == "A4_CROP_ONLY":
        dec = evaluate_policy_r3(evidence, lambda_cost=lambda_cost)
        return dec, "CROP_ONLY"

    else:
        raise ValueError(f"Unknown policy_key: {policy_key}")


def compute_policy_evaluation_metrics(
    eval_records: List[Dict[str, Any]],
    ground_truth_map: Dict[str, Dict[str, Any]],
    vlm_cost: float = DEFAULT_VLM_COST,
) -> Dict[str, Any]:
    """Compute comprehensive router policy evaluation metrics."""
    eval_classes = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    pred_columns = ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    cm = {c: {p: 0 for p in pred_columns} for c in eval_classes}

    total_instances = len(eval_records)
    vlm_queries_count = 0
    cheap_resolutions_count = 0
    defer_count = 0
    abstain_count = 0

    correct_total = 0
    correct_vlm = 0
    correct_cheap = 0
    correct_by_cls = defaultdict(int)
    total_by_gt = defaultdict(int)
    total_by_pred = defaultdict(int)

    latencies_ms: List[float] = []

    decision_dist = defaultdict(int)

    for rec in eval_records:
        eid = rec["physical_instance_id"]
        gt_cls = ground_truth_map.get(eid, {}).get("contextual_movability_gt", "UNKNOWN")
        final_pred = rec.get("final_prediction", "UNKNOWN")
        if final_pred not in pred_columns:
            final_pred = "UNKNOWN"

        dec_type = rec["decision"]
        decision_dist[dec_type] += 1

        vlm_invoked = rec["vlm_invoked"]
        if vlm_invoked:
            vlm_queries_count += 1
        elif dec_type == "RESOLVE_CHEAP":
            cheap_resolutions_count += 1
        elif dec_type == "DEFER":
            defer_count += 1
        elif dec_type == "ABSTAIN":
            abstain_count += 1

        lat = rec.get("total_latency_ms", 0.0)
        latencies_ms.append(float(lat))

        if gt_cls in eval_classes:
            cm[gt_cls][final_pred] += 1
            total_by_gt[gt_cls] += 1
            total_by_pred[final_pred] += 1

            if final_pred == gt_cls:
                correct_total += 1
                correct_by_cls[gt_cls] += 1
                if vlm_invoked:
                    correct_vlm += 1
                else:
                    correct_cheap += 1

    overall_acc = correct_total / max(1, total_instances)
    query_rate = vlm_queries_count / max(1, total_instances)
    selective_acc = (correct_vlm / vlm_queries_count) if vlm_queries_count > 0 else 0.0
    cheap_acc = (correct_cheap / cheap_resolutions_count) if cheap_resolutions_count > 0 else 0.0

    recalls = {c: round(correct_by_cls[c] / total_by_gt[c], 4) if total_by_gt[c] > 0 else 0.0 for c in eval_classes}
    precisions = {c: round(correct_by_cls[c] / total_by_pred[c], 4) if total_by_pred[c] > 0 else 0.0 for c in eval_classes}
    f1s = {}
    for c in eval_classes:
        p, r = precisions[c], recalls[c]
        f1s[c] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    balanced_acc = float(np.mean([recalls[c] for c in eval_classes if total_by_gt[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in eval_classes if total_by_gt[c] > 0]))

    normalized_cost = (vlm_queries_count * vlm_cost) / max(1, total_instances)

    return {
        "total_instances": total_instances,
        "vlm_queries_count": vlm_queries_count,
        "vlm_query_rate": round(query_rate, 4),
        "overall_accuracy": round(overall_acc, 4),
        "end_to_end_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "selective_accuracy": round(selective_acc, 4),
        "selective_accuracy_on_vlm": round(selective_acc, 4),
        "cheap_accuracy": round(cheap_acc, 4),
        "normalized_compute_cost": round(normalized_cost, 4),
        "latency_per_object_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "p50": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "p95": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        },
        "decision_distribution": dict(decision_dist),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": cm,
    }


def evaluate_router_policies(
    policy_keys: List[str] = POLICY_KEYS,
    model_key: Literal["qwen2_5_vl_3b"] = "qwen2_5_vl_3b",
    execution_mode: Literal["replay", "live"] = "replay",
    task_mode: Literal["class-conditioned", "unconditioned"] = "class-conditioned",
    replay_run_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    lambda_cost: float = DEFAULT_VLM_COST,
    inputs_manifest_path: Optional[Path] = None,
    ground_truth_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute evaluation suite across all router policies and ablations."""
    inputs_manifest = Path(inputs_manifest_path) if inputs_manifest_path else (repo_root / "benchmarks" / "paper_benchmark_inputs.json")
    gt_manifest = Path(ground_truth_manifest_path) if ground_truth_manifest_path else (repo_root / "benchmarks" / "paper_benchmark_ground_truth.json")

    with open(inputs_manifest, "r", encoding="utf-8") as f:
        inputs_data = json.load(f)["instances"]
    with open(gt_manifest, "r", encoding="utf-8") as f:
        gt_map = json.load(f)["labels"]

    if run_dir is None:
        run_id = f"router_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = repo_root / "outputs" / "benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # In replay mode, load recorded VLM predictions
    replay_cache: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)  # [view_mode][eid] -> record
    if execution_mode == "replay":
        if replay_run_dir is None:
            # Search for latest visual classification run
            bench_runs = repo_root / "outputs" / "benchmark_runs"
            vis_runs = sorted(bench_runs.glob("visual_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if vis_runs:
                replay_run_dir = vis_runs[0]
                print(f"[REPLAY MODE] Using latest visual classification run: {replay_run_dir.name}")
            else:
                raise FileNotFoundError("Replay mode requested but no prior visual classification runs found.")

        # Load predictions for both view modes
        for vmode in ["crop_only", "crop_plus_context"]:
            pfile = replay_run_dir / f"visual_classification_predictions_{model_key}_{vmode}.jsonl"
            if not pfile.exists():
                pfile = replay_run_dir / f"visual_classification_predictions_{model_key}_{vmode}_class_conditioned.jsonl"
            if pfile.exists():
                with open(pfile, "r", encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            r = json.loads(line)
                            replay_cache[vmode.upper()][r["physical_instance_id"]] = r

    # In live mode, initialize inference engine
    engine = None
    if execution_mode == "live":
        engine = MultimodalInferenceEngine(model_key=model_key)
        health = engine.check_health()
        if health.get("status") != "ONLINE":
            raise ConnectionError(f"Strict Live Failure: {model_key} server is OFFLINE at {engine.server_url}: {health}")
        print(f"[LIVE MODE] Probed {model_key} server ONLINE. Running preflight smoke test...")
        engine.run_preflight_smoke_test()

    print("=" * 80)
    print("AXIS 2: ESTIMATOR-AWARE SELECTIVE ROUTER POLICY BENCHMARK")
    print(f"Total Physical Instances: {len(inputs_data)}")
    print(f"Target Policies: {len(policy_keys)} policies")
    print(f"Execution Mode: {execution_mode.upper()} (Model: {model_key})")
    print(f"Output Run Directory: {run_dir}")
    print("=" * 80)

    summary: Dict[str, Any] = {
        "benchmark_axis": "Axis 2: Router Policy Suite",
        "timestamp": time.time(),
        "execution_mode": execution_mode,
        "task_mode": task_mode,
        "model_key": model_key,
        "total_physical_instances": len(inputs_data),
        "lambda_cost": lambda_cost,
        "policy_summaries": {},
    }

    for pkey in policy_keys:
        print(f"\n>>> Evaluating Policy: {pkey} <<<")
        policy_records: List[Dict[str, Any]] = []
        pfile_out = run_dir / f"router_decisions_{pkey.lower()}.jsonl"

        with open(pfile_out, "w", encoding="utf-8") as pf:
            for inst in inputs_data:
                eid = inst["physical_instance_id"]
                scls = inst["semantic_class"]
                vis = inst.get("visual_inputs", {})
                evidence = extract_evidence_from_input_record(inst)

                # Ground truth for Oracle policy only
                oracle_gt = gt_map.get(eid, {}).get("contextual_movability_gt")
                oracle_vlm_pred = replay_cache.get("CROP_PLUS_CONTEXT", {}).get(eid, {}).get("parsed_prediction")

                # Router makes decision based STRICTLY on evidence (ZERO GT leakage)
                decision, view_mode = execute_policy_decision(
                    policy_key=pkey,
                    evidence=evidence,
                    lambda_cost=lambda_cost,
                    oracle_gt_class=oracle_gt,
                    oracle_vlm_prediction=oracle_vlm_pred,
                )

                final_pred: str = "UNKNOWN"
                vlm_invoked = False
                total_latency_ms = 0.0
                vlm_record = None

                if decision.decision == "RESOLVE_CHEAP":
                    final_pred = decision.cheap_label or "UNKNOWN"
                    total_latency_ms = 0.05  # sub-millisecond local rule lookup

                elif decision.decision == "ESCALATE":
                    vlm_invoked = True
                    if execution_mode == "replay":
                        cached_rec = replay_cache.get(view_mode, {}).get(eid)
                        if cached_rec:
                            final_pred = cached_rec.get("parsed_prediction", "UNKNOWN")
                            total_latency_ms = cached_rec.get("client_latency_ms", 1500.0)
                            vlm_record = cached_rec
                        else:
                            final_pred = "UNKNOWN"
                            total_latency_ms = 0.0
                    else:
                        # Strict live execution
                        crop_p = repo_root / vis["crop_view_paths"][0]
                        scene_p = repo_root / vis["scene_view_paths"][0] if vis.get("scene_view_paths") else None
                        prompt = build_canonical_prompt(eid, scls, task_mode=task_mode)
                        expected_class = scls if task_mode == "class-conditioned" else None
                        wire_rec = engine.execute_live_query(
                            physical_instance_id=eid,
                            semantic_class=expected_class,
                            prompt=prompt,
                            crop_image_path=crop_p,
                            scene_image_path=scene_p,
                            view_mode=view_mode,
                            image_hashes=vis.get("image_sha256", []),
                        )
                        final_pred = wire_rec.parsed_prediction
                        total_latency_ms = wire_rec.client_latency_ms
                        vlm_record = wire_rec.to_dict()

                elif decision.decision in ["DEFER", "ABSTAIN"]:
                    final_pred = decision.cheap_label or "UNKNOWN"
                    total_latency_ms = 0.05

                rec_out = {
                    "physical_instance_id": eid,
                    "semantic_class": scls,
                    "policy_key": pkey,
                    "task_mode": task_mode,
                    "decision": decision.decision,
                    "reason_codes": decision.reason_codes,
                    "vlm_invoked": vlm_invoked,
                    "view_mode_requested": view_mode,
                    "cheap_label": decision.cheap_label,
                    "final_prediction": final_pred,
                    "total_latency_ms": round(total_latency_ms, 2),
                    "query_utility": decision.query_utility,
                    "estimator_influence": decision.estimator_influence,
                    "vlm_record": vlm_record,
                }
                pf.write(json.dumps(rec_out) + "\n")
                policy_records.append(rec_out)

        # Compute metrics using independent ground truth
        metrics = compute_policy_evaluation_metrics(policy_records, gt_map, vlm_cost=lambda_cost)
        is_deployable = (pkey != "B7_ORACLE")
        policy_type = "ORACLE_DIAGNOSTIC_ONLY" if pkey == "B7_ORACLE" else ("ABLATION" if pkey.startswith("A") else "DEPLOYABLE_BASELINE")
        summary["policy_summaries"][pkey] = {
            "policy_type": policy_type,
            "is_deployable": is_deployable,
            "metrics": metrics,
            "decisions_file": str(pfile_out.relative_to(repo_root)).replace("\\", "/"),
        }

        print(f"  -> Acc: {metrics['overall_accuracy']*100:.1f}% | Macro-F1: {metrics['macro_f1']*100:.1f}% | VLM Query Rate: {metrics['vlm_query_rate']*100:.1f}% ({metrics['vlm_queries_count']}/{metrics['total_instances']}) | Cost: {metrics['normalized_compute_cost']:.3f}")

    out_file = run_dir / "router_policies_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nRouter policy evaluation finished. Summary saved to: {out_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate estimator-aware router policies and baselines.")
    parser.add_argument("--policies", nargs="+", default=POLICY_KEYS, choices=POLICY_KEYS)
    parser.add_argument("--model", default="qwen2_5_vl_3b", choices=["qwen2_5_vl_3b"])
    parser.add_argument("--mode", default="replay", choices=["replay", "live"], help="Replay from visual eval run or execute live")
    parser.add_argument("--replay-dir", type=Path, default=None, help="Directory of visual eval run for replay mode")
    parser.add_argument("--lambda-cost", type=float, default=DEFAULT_VLM_COST)
    parser.add_argument("--inputs-manifest", type=Path, default=None, help="Path to custom inputs manifest JSON")
    parser.add_argument("--ground-truth-manifest", type=Path, default=None, help="Path to custom ground-truth manifest JSON")
    args = parser.parse_args()

    evaluate_router_policies(
        policy_keys=args.policies,
        model_key=args.model,
        execution_mode=args.mode,
        replay_run_dir=args.replay_dir,
        lambda_cost=args.lambda_cost,
        inputs_manifest_path=args.inputs_manifest,
        ground_truth_manifest_path=args.ground_truth_manifest,
    )
