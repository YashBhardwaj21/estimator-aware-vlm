#!/usr/bin/env python3
"""Zero-Shot Live Evaluation of Qwen2.5-VL in WSL.

Evaluates pure zero-shot affordance classification directly on physical object image crops
via the live running llama-server in WSL, without any SLAM odometry, class priors, or motion filters.

Metrics recorded:
- Accuracy, Balanced Accuracy, Per-Class Precision/Recall/F1
- Measured live wire latency per instance (ms)
- Token usage (prompt tokens, completion tokens)
- Output validity & JSON adherence
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from estimator_aware_vlm.selective_router import (
    DEFAULT_SERVER_URL,
    check_vlm_server_health,
    encode_image,
    parse_vlm_output,
    query_llama_server,
    validate_image_inputs,
)

CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]

ZERO_SHOT_PROMPT_TEMPLATE = """You are an embodied robotics vision-language expert analyzing a 3D scene.
Examine the provided image(s) showing a physical object in an indoor environment.

Determine the physical contextual movability of the focused object into exactly one of three categories:
1. IMMOVABLE: Fixed architectural infrastructure permanently affixed to the building structure (walls, structural doors, ceiling fans, built-in cabinets).
2. MOVABLE: Passively transportable objects lacking intrinsic locomotion (chairs, desks, laptops, backpacks, pedestal fans, storage boxes). An object being carried by an agent remains strictly MOVABLE.
3. MOBILE: Active embodied agents whose defining category is self-propelled locomotion (humans, mobile robots, drones), even if temporarily stationary.

Respond with STRICT JSON ONLY conforming to:
{{
  "object_id": "{entity_id}",
  "object_class": "{semantic_class}",
  "movability": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "confidence": 0.0 to 1.0,
  "physical_support": "<how is the object supported, e.g. floor, desk, mounted>",
  "short_explanation": "<one sentence physical justification>"
}}
"""


def build_zero_shot_prompt(entity_id: str, semantic_class: str) -> str:
    return ZERO_SHOT_PROMPT_TEMPLATE.format(
        entity_id=entity_id,
        semantic_class=semantic_class,
    )


def run_zero_shot_evaluation(
    server_url: str = DEFAULT_SERVER_URL,
    manifest_path: Optional[Path] = None,
    max_instances: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute live zero-shot evaluation across benchmark instances with images."""
    if manifest_path is None:
        manifest_path = repo_root / "benchmarks" / "router_evaluation_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter to image-eligible instances
    instances = [
        inst for inst in manifest["evaluation_instances"]
        if inst.get("has_images") and inst.get("views_count", 0) > 0 and inst.get("crops_dir")
    ]

    if max_instances is not None:
        instances = instances[:max_instances]

    health = check_vlm_server_health(server_url)
    if health.get("status") != "ONLINE":
        raise ConnectionError(f"Live VLM server offline at {server_url}: {health}")

    print("=" * 80)
    print("LIVE ZERO-SHOT EVALUATION: QWEN2.5-VL-3B-INSTRUCT (WSL)")
    print(f"Target Server:   {server_url} (Status: ONLINE)")
    print(f"Total Instances: {len(instances)} image-eligible physical objects")
    print("=" * 80)

    results = []
    latencies_ms = []

    for i, inst in enumerate(instances, start=1):
        eid = inst["entity_id"]
        scls = inst["semantic_class"]
        gt = inst["contextual_movability_gt"]
        crops_dir = repo_root / inst["crops_dir"]

        print(f"\n[{i}/{len(instances)}] Querying Zero-Shot Qwen: {eid} ({scls}) [GT: {gt}]...")

        # Validate images
        img_tel = validate_image_inputs(crops_dir, configured_views=1)
        pairs = img_tel.get("crop_scene_pairs", [])

        if not pairs:
            print(f"  -> Skipping {eid}: no valid image pairs found.")
            results.append({
                "entity_id": eid,
                "semantic_class": scls,
                "ground_truth": gt,
                "prediction": "UNKNOWN",
                "abstained": True,
                "valid": False,
                "latency_ms": 0.0,
            })
            continue

        c_p, s_p = pairs[0]
        c_b64 = encode_image(c_p, max_dim=200)
        s_b64 = encode_image(s_p, max_dim=280)

        prompt = build_zero_shot_prompt(eid, scls)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{c_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s_b64}"}},
            ],
        }]

        t0 = time.perf_counter()
        raw_text, lat_s, usage = query_llama_server(messages, server_url=server_url, max_tokens=75)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        parsed, is_json, is_schema = parse_vlm_output(raw_text, eid, scls)
        pred = parsed.get("movability", "UNKNOWN")
        correct = (pred == gt)

        print(f"  -> Response: {pred} (Confidence: {parsed.get('confidence')}) | Correct: {correct} | Latency: {lat_ms:.1f}ms")

        results.append({
            "entity_id": eid,
            "semantic_class": scls,
            "ground_truth": gt,
            "prediction": pred,
            "correct": correct,
            "confidence": parsed.get("confidence"),
            "physical_support": parsed.get("physical_support"),
            "short_explanation": parsed.get("short_explanation"),
            "is_json_valid": is_json,
            "is_schema_valid": is_schema,
            "raw_output": raw_text.strip(),
            "latency_ms": round(lat_ms, 2),
            "token_usage": usage,
        })

    # Compute Zero-Shot Metrics
    eval_classes = CLASSES
    cm = {c: {p: 0 for p in ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]} for c in eval_classes}
    correct_by_cls = defaultdict(int)
    total_by_gt = defaultdict(int)
    total_by_pred = defaultdict(int)

    for r in results:
        gt = r["ground_truth"]
        p = r["prediction"] if r["prediction"] in eval_classes else "UNKNOWN"
        cm[gt][p] += 1
        total_by_gt[gt] += 1
        total_by_pred[p] += 1
        if p == gt:
            correct_by_cls[gt] += 1

    recalls = {c: round(correct_by_cls[c] / total_by_gt[c], 4) if total_by_gt[c] > 0 else 0.0 for c in eval_classes}
    precisions = {c: round(correct_by_cls[c] / total_by_pred[c], 4) if total_by_pred[c] > 0 else 0.0 for c in eval_classes}
    f1s = {}
    for c in eval_classes:
        p, r = precisions[c], recalls[c]
        f1s[c] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    balanced_acc = float(np.mean([recalls[c] for c in eval_classes if total_by_gt[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in eval_classes if total_by_gt[c] > 0]))
    raw_acc = sum(correct_by_cls.values()) / max(1, len(results))

    summary = {
        "benchmark_title": "Zero-Shot Qwen2.5-VL-3B-Instruct Live Evaluation",
        "model_name": "Qwen2.5-VL-3B-Instruct-Q4_K_M (llama.cpp WSL)",
        "server_url": server_url,
        "live_execution": True,
        "total_instances_evaluated": len(results),
        "overall_accuracy": round(raw_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
        "confusion_matrix": cm,
        "latency_stats_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "p50": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "p95": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
            "total_inference_time_s": round(sum(latencies_ms) / 1000.0, 2),
        },
        "records": results,
    }

    if out_path is None:
        out_path = repo_root / "outputs" / "router_ablation" / "zero_shot_qwen_evaluation.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION COMPLETED")
    print(f"Overall Accuracy:  {raw_acc * 100:.1f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.1f}%")
    print(f"Macro-F1:          {macro_f1 * 100:.1f}%")
    print(f"Mean Latency:      {summary['latency_stats_ms']['mean']:.1f} ms")
    print(f"Saved to:          {out_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on live Qwen2.5-VL")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="llama-server URL")
    parser.add_argument("--max-instances", type=int, default=None, help="Limit number of instances")
    args = parser.parse_args()

    run_zero_shot_evaluation(
        server_url=args.server_url,
        max_instances=args.max_instances,
    )
