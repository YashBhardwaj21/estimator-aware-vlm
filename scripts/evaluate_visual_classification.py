#!/usr/bin/env python3
"""Axis 1: Visual Movability Classification & Paired Model Comparison.

Evaluates how well neural vision models classify physical movability from images alone:
- Evaluates Qwen2.5-VL-3B-Instruct (llama.cpp)
- Evaluates two visual input modes: CROP_ONLY vs. CROP_PLUS_CONTEXT
- Strict two-file isolation: reads purely inference-visible metadata from paper_benchmark_inputs.json
- Evaluator joins predictions against paper_benchmark_ground_truth.json strictly post-inference
- Records complete wire records (latency, tokens, SHA-256 hashes, status booleans)

Outputs:
- vlm/outputs/benchmark_runs/<run_id>/visual_classification_predictions_<model>.jsonl
- vlm/outputs/benchmark_runs/<run_id>/visual_classification_summary.json
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

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "vlm"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vlm.src.multimodal_inference_engine import (
    MultimodalInferenceEngine,
    WireInferenceRecord,
    CLASSES,
)

CANONICAL_VISUAL_PROMPT = """You are an expert robotics perception system evaluating physical affordance and contextual movability for indoor scene understanding.
Target Instance ID: {entity_id}
Expected Semantic Class: {semantic_class}

ONTOLOGY DEFINITIONS (Canonical Embodied Affordance Schema):
- IMMOVABLE: Fixed architectural infrastructure (e.g., walls, structural doors, ceiling fans, built-in cabinets). Cannot be relocated without deconstruction.
- MOVABLE: Passively transportable objects lacking intrinsic locomotion (e.g., chairs, desks, boxes, backpacks, laptops, pedestal fans). An object currently being carried by an agent remains strictly MOVABLE.
- MOBILE: Active embodied agents whose defining category is self-propelled locomotion (e.g., humans, mobile robots, drones), even if temporarily stationary.

EVALUATION INSTRUCTIONS:
1. Inspect the visual view(s) of the target instance.
2. Identify physical_support: 'mounted_to_structure', 'freestanding_on_surface', 'wheeled_base', 'carried_by_agent', or 'unknown'.
3. Identify transport_context: 'active_carrier', 'stationary_environment', or 'unknown'.
4. Assign movability strictly as IMMOVABLE, MOVABLE, or MOBILE based on physical affordance, not temporary motion.
5. If visual evidence is severely occluded, corrupted, or missing, set abstain to true.
6. Provide a concise short_explanation citing visual cues.

OUTPUT SCHEMA (Respond ONLY with a valid JSON object matching these fields):
{{
  "object_id": "<must match Target Instance ID: {entity_id}>",
  "object_class": "<must match Expected Semantic Class: {semantic_class}>",
  "movability": "<IMMOVABLE | MOVABLE | MOBILE | UNKNOWN>",
  "physical_support": "<mounted_to_structure | freestanding_on_surface | wheeled_base | carried_by_agent | unknown>",
  "transport_context": "<stationary_environment | active_carrier | unknown>",
  "confidence": <float between 0.0 and 1.0>,
  "abstain": <true | false>,
  "short_explanation": "<concise justification>"
}}
"""

CANONICAL_UNCONDITIONED_VISUAL_PROMPT = """You are an expert robotics perception system evaluating physical affordance and contextual movability for indoor scene understanding.
Target Instance ID: {entity_id}

ONTOLOGY DEFINITIONS (Canonical Embodied Affordance Schema):
- IMMOVABLE: Fixed architectural infrastructure (e.g., walls, structural doors, ceiling fans, built-in cabinets). Cannot be relocated without deconstruction.
- MOVABLE: Passively transportable objects lacking intrinsic locomotion (e.g., chairs, desks, boxes, backpacks, laptops, pedestal fans). An object currently being carried by an agent remains strictly MOVABLE.
- MOBILE: Active embodied agents whose defining category is self-propelled locomotion (e.g., humans, mobile robots, drones), even if temporarily stationary.

EVALUATION INSTRUCTIONS:
1. Inspect the visual view(s) of the target instance and infer what object it is.
2. Identify physical_support: 'mounted_to_structure', 'freestanding_on_surface', 'wheeled_base', 'carried_by_agent', or 'unknown'.
3. Identify transport_context: 'active_carrier', 'stationary_environment', or 'unknown'.
4. Assign movability strictly as IMMOVABLE, MOVABLE, or MOBILE based on physical affordance, not temporary motion.
5. If visual evidence is severely occluded, corrupted, or missing, set abstain to true.
6. Provide a concise short_explanation citing visual cues.

OUTPUT SCHEMA (Respond ONLY with a valid JSON object matching these fields):
{{
  "object_id": "<must match Target Instance ID: {entity_id}>",
  "object_class": "<inferred semantic class of the target object>",
  "movability": "<IMMOVABLE | MOVABLE | MOBILE | UNKNOWN>",
  "physical_support": "<mounted_to_structure | freestanding_on_surface | wheeled_base | carried_by_agent | unknown>",
  "transport_context": "<stationary_environment | active_carrier | unknown>",
  "confidence": <float between 0.0 and 1.0>,
  "abstain": <true | false>,
  "short_explanation": "<concise justification>"
}}
"""


def build_canonical_prompt(
    entity_id: str,
    semantic_class: Optional[str] = None,
    task_mode: Literal["class-conditioned", "unconditioned"] = "class-conditioned",
) -> str:
    if task_mode == "unconditioned":
        return CANONICAL_UNCONDITIONED_VISUAL_PROMPT.format(entity_id=entity_id)
    return CANONICAL_VISUAL_PROMPT.format(
        entity_id=entity_id,
        semantic_class=semantic_class or "UNKNOWN",
    )


def compute_classification_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute exact classification metrics with confusion matrix and denominators."""
    eval_classes = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    pred_columns = ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    cm = {c: {p: 0 for p in pred_columns} for c in eval_classes}

    correct_by_cls = defaultdict(int)
    total_by_gt = defaultdict(int)
    total_by_pred = defaultdict(int)
    latencies_ms = []

    attempted_count = len(predictions)
    successful_responses = 0
    valid_schema_count = 0
    accepted_predictions = 0

    for rec in predictions:
        eid = rec["physical_instance_id"]
        if eid not in ground_truth_map:
            continue
        gt = ground_truth_map[eid]["contextual_movability_gt"]
        pred = rec.get("parsed_prediction", "UNKNOWN")
        if pred not in eval_classes:
            pred = "UNKNOWN"

        if rec.get("response_received"):
            successful_responses += 1
        if rec.get("schema_valid"):
            valid_schema_count += 1
        if rec.get("prediction_accepted"):
            accepted_predictions += 1

        lat = rec.get("client_latency_ms")
        if lat is not None and isinstance(lat, (int, float)):
            latencies_ms.append(float(lat))

        cm[gt][pred] += 1
        total_by_gt[gt] += 1
        total_by_pred[pred] += 1
        if pred == gt:
            correct_by_cls[gt] += 1

    recalls = {c: round(correct_by_cls[c] / total_by_gt[c], 4) if total_by_gt[c] > 0 else 0.0 for c in eval_classes}
    precisions = {c: round(correct_by_cls[c] / total_by_pred[c], 4) if total_by_pred[c] > 0 else 0.0 for c in eval_classes}
    f1s = {}
    for c in eval_classes:
        p, r = precisions[c], recalls[c]
        f1s[c] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    balanced_acc = float(np.mean([recalls[c] for c in eval_classes if total_by_gt[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in eval_classes if total_by_gt[c] > 0]))
    overall_acc = sum(correct_by_cls.values()) / max(1, attempted_count)

    # Selective accuracy among accepted predictions only
    accepted_correct = sum(
        1 for r in predictions
        if r.get("prediction_accepted") and r.get("parsed_prediction") == ground_truth_map.get(r["physical_instance_id"], {}).get("contextual_movability_gt")
    )
    selective_acc = (accepted_correct / accepted_predictions) if accepted_predictions > 0 else 0.0

    return {
        "total_instances_evaluated": attempted_count,
        "class_support": {c: total_by_gt[c] for c in eval_classes},
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": cm,
        "wire_quality": {
            "requests_attempted": attempted_count,
            "successful_responses": successful_responses,
            "valid_schema_count": valid_schema_count,
            "accepted_predictions": accepted_predictions,
            "coverage_pct": round((accepted_predictions / max(1, attempted_count)) * 100.0, 2),
            "selective_accuracy_on_accepted": round(selective_acc, 4),
        },
        "client_latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "p50": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "p95": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        },
    }


def evaluate_visual_classification(
    model_keys: List[Literal["qwen2_5_vl_3b"]],
    view_modes: List[Literal["CROP_ONLY", "CROP_PLUS_CONTEXT", "MULTI_VIEW"]] = ["CROP_ONLY", "CROP_PLUS_CONTEXT"],
    task_mode: Literal["class-conditioned", "unconditioned"] = "class-conditioned",
    run_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    skip_offline_models: bool = False,
    inputs_manifest_path: Optional[Path] = None,
    ground_truth_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute visual classification evaluation across models, view modes, and task modes."""
    inputs_manifest = Path(inputs_manifest_path) if inputs_manifest_path else (repo_root / "vlm" / "benchmarks" / "paper_benchmark_inputs.json")
    gt_manifest = Path(ground_truth_manifest_path) if ground_truth_manifest_path else (repo_root / "vlm" / "benchmarks" / "paper_benchmark_ground_truth.json")

    with open(inputs_manifest, "r", encoding="utf-8") as f:
        inputs_data = json.load(f)["instances"]
    with open(gt_manifest, "r", encoding="utf-8") as f:
        gt_map = json.load(f)["labels"]

    # Filter to image-eligible instances with usable views
    image_instances = [
        inst for inst in inputs_data
        if inst.get("visual_inputs", {}).get("has_images") and inst["visual_inputs"].get("usable_views", 0) > 0
    ]

    if max_instances is not None:
        image_instances = image_instances[:max_instances]

    if run_dir is None:
        run_id = f"visual_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = repo_root / "vlm" / "outputs" / "benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AXIS 1: VISUAL MOVABILITY CLASSIFICATION (DUAL-MODEL STUDY)")
    print(f"Total Image-Eligible Physical Instances: {len(image_instances)}")
    print(f"Target Models: {model_keys}")
    print(f"Target View Modes: {view_modes}")
    print(f"Task Mode: {task_mode}")
    print(f"Output Run Directory: {run_dir}")
    print("=" * 80)

    summary_results: Dict[str, Any] = {
        "benchmark_axis": "Axis 1: Visual Classification",
        "task_mode": task_mode,
        "timestamp": time.time(),
        "total_physical_instances": len(image_instances),
        "evaluations": {},
    }

    for mkey in model_keys:
        engine = MultimodalInferenceEngine(model_key=mkey)
        health = engine.check_health()
        print(f"\n[Model: {mkey}] Probing health at {engine.server_url}... -> {health.get('status')}")

        if health.get("status") != "ONLINE":
            if skip_offline_models:
                print(f"  [SKIPPING] {mkey} server is offline ({health.get('error')}) and skip_offline_models=True.")
                summary_results["evaluations"][mkey] = {"status": "SKIPPED_OFFLINE", "health": health}
                continue
            else:
                raise ConnectionError(f"Strict Live Failure: {mkey} server is OFFLINE at {engine.server_url}: {health}")

        # Run preflight smoke test
        print(f"  -> Running preflight multimodal smoke test for {mkey}...")
        smoke = engine.run_preflight_smoke_test()
        print(f"  -> Smoke test PASSED ({smoke['latency_ms']:.1f}ms).")

        summary_results["evaluations"][mkey] = {}

        for vmode in view_modes:
            print(f"\n  --- Evaluating {mkey} under {vmode} [{task_mode}] ({len(image_instances)} instances) ---")
            predictions: List[Dict[str, Any]] = []

            mode_tag = f"{vmode.lower()}_{task_mode.replace('-', '_')}"
            pred_file = run_dir / f"visual_classification_predictions_{mkey}_{mode_tag}.jsonl"

            with open(pred_file, "w", encoding="utf-8") as pf:
                for idx, inst in enumerate(image_instances, start=1):
                    eid = inst["physical_instance_id"]
                    scls = inst["semantic_class"]
                    vis = inst["visual_inputs"]
                    crop_p = repo_root / vis["crop_view_paths"][0]
                    scene_p = repo_root / vis["scene_view_paths"][0] if vis.get("scene_view_paths") else None
                    add_crops = [repo_root / p for p in vis.get("crop_view_paths", [])[1:]] if len(vis.get("crop_view_paths", [])) > 1 else None

                    prompt = build_canonical_prompt(eid, scls, task_mode=task_mode)
                    expected_class = scls if task_mode == "class-conditioned" else None

                    print(f"    [{idx}/{len(image_instances)}] Querying {mkey} ({vmode}|{task_mode}) for {eid} ({scls})...", end="", flush=True)

                    rec: WireInferenceRecord = engine.execute_live_query(
                        physical_instance_id=eid,
                        semantic_class=expected_class,
                        prompt=prompt,
                        crop_image_path=crop_p,
                        scene_image_path=scene_p,
                        additional_crop_paths=add_crops,
                        view_mode=vmode,
                        image_hashes=vis.get("image_sha256", []),
                    )

                    rec_dict = rec.to_dict()
                    rec_dict["task_mode"] = task_mode
                    pf.write(json.dumps(rec_dict) + "\n")
                    predictions.append(rec_dict)

                    gt_cls = gt_map.get(eid, {}).get("contextual_movability_gt", "UNKNOWN")
                    correct = (rec.parsed_prediction == gt_cls)
                    print(f" -> Pred: {rec.parsed_prediction} (GT: {gt_cls}) | Correct: {correct} | Latency: {rec.client_latency_ms:.1f}ms")

            # Score predictions using independent evaluator ground truth
            metrics = compute_classification_metrics(predictions, gt_map)
            summary_results["evaluations"][mkey][vmode] = {
                "metrics": metrics,
                "predictions_file": str(pred_file.relative_to(repo_root)).replace("\\", "/"),
            }

            print(f"  • {mkey} ({vmode}) Overall Acc: {metrics['overall_accuracy']*100:.1f}% | Balanced Acc: {metrics['balanced_accuracy']*100:.1f}% | Macro-F1: {metrics['macro_f1']*100:.1f}%")

    # Also check if existing predictions exist in run_dir
    for other_m in ["qwen2_5_vl_3b"]:
        if other_m not in summary_results["evaluations"]:
            for vmode in view_modes:
                mode_tag = f"{vmode.lower()}_{task_mode.replace('-', '_')}"
                existing_pred = run_dir / f"visual_classification_predictions_{other_m}_{mode_tag}.jsonl"
                if existing_pred.exists():
                    p_list = []
                    with open(existing_pred, "r", encoding="utf-8") as pf:
                        for line in pf:
                            if line.strip():
                                p_list.append(json.loads(line))
                    if p_list:
                        if other_m not in summary_results["evaluations"]:
                            summary_results["evaluations"][other_m] = {}
                        summary_results["evaluations"][other_m][vmode] = {
                            "metrics": compute_classification_metrics(p_list, gt_map),
                            "predictions_file": str(existing_pred.relative_to(repo_root)).replace("\\", "/"),
                        }

    out_summary_file = run_dir / "visual_classification_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, default=str)

    print(f"\nVisual classification evaluation completed. Summary saved to: {out_summary_file}")
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual physical movability classification across models.")
    parser.add_argument("--models", nargs="+", default=["qwen2_5_vl_3b"], choices=["qwen2_5_vl_3b"])
    parser.add_argument(
        "--view-modes",
        nargs="+",
        default=["CROP_ONLY", "CROP_PLUS_CONTEXT"],
        choices=["CROP_ONLY", "CROP_PLUS_CONTEXT", "MULTI_VIEW"],
    )
    parser.add_argument(
        "--task-mode",
        default="class-conditioned",
        choices=["class-conditioned", "unconditioned"],
        help="class-conditioned (provides expected class) vs unconditioned (withholds class from prompt)",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--skip-offline", action="store_true", help="Skip models whose servers are offline instead of hard error")
    parser.add_argument("--inputs-manifest", type=Path, default=None, help="Path to custom inputs manifest JSON")
    parser.add_argument("--ground-truth-manifest", type=Path, default=None, help="Path to custom ground-truth manifest JSON")
    args = parser.parse_args()

    evaluate_visual_classification(
        model_keys=args.models,
        view_modes=args.view_modes,
        task_mode=args.task_mode,
        max_instances=args.max_instances,
        skip_offline_models=args.skip_offline,
        inputs_manifest_path=args.inputs_manifest,
        ground_truth_manifest_path=args.ground_truth_manifest,
    )
