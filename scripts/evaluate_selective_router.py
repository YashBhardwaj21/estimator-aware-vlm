#!/usr/bin/env python3
"""Offline and Live Evaluation of Estimator-Aware Selective Router."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

from estimator_aware_vlm.selective_router import (
    SelectiveRouter,
    check_vlm_server_health,
    DEFAULT_SERVER_URL,
)
from estimator_aware_vlm.evidence_adapter import EvidenceAdapter


CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Quality, Efficiency, and Reliability metrics."""
    total = len(records)
    if total == 0:
        return {}

    eval_classes = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    pred_columns = ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    cm = {c: {p: 0 for p in pred_columns} for c in eval_classes}

    correct_by_class = defaultdict(int)
    total_by_gt = defaultdict(int)
    total_by_pred = defaultdict(int)

    abstain_count = 0
    vlm_query_count = 0
    vlm_query_requested_count = 0
    vlm_inference_executed_count = 0
    vlm_cache_hits_count = 0
    vlm_outputs_valid_count = 0
    vlm_predictions_accepted_count = 0
    decision_latencies_ms = []
    end_to_end_latencies_s = []

    for r in records:
        gt = r["ground_truth"]
        pred = r["prediction"]
        abstained = r.get("abstained", False) or pred in ["UNKNOWN", None]
        d_lat = r.get("decision_latency_ms")
        if d_lat is not None and isinstance(d_lat, (int, float)):
            decision_latencies_ms.append(float(d_lat))

        # End to end latency: decision time + inference time if queried
        mode = r.get("execution_mode", "CACHED_HISTORICAL")
        if mode == "LIVE_INFERENCE":
            inf_ms = r.get("inference_latency_ms", 0.0)
        else:
            inf_ms = r.get("cached_latency_ms") if r.get("cached_latency_ms") is not None else r.get("inference_latency_ms", 0.0)

        if d_lat is not None and isinstance(d_lat, (int, float)):
            if r.get("vlm_required", False):
                if inf_ms is not None and isinstance(inf_ms, (int, float)):
                    end_to_end_latencies_s.append((float(d_lat) + float(inf_ms)) / 1000.0)
            else:
                end_to_end_latencies_s.append(float(d_lat) / 1000.0)

        # Detailed query accounting
        if r.get("vlm_required", False):
            vlm_query_requested_count += 1
        if r.get("vlm_inference_executed", False):
            vlm_inference_executed_count += 1
        if r.get("vlm_cache_hit", False):
            vlm_cache_hits_count += 1
        if r.get("vlm_output_valid", False) or (r.get("vlm_cache_hit", False) and not abstained):
            vlm_outputs_valid_count += 1
        if r.get("vlm_prediction_accepted", False) or (r.get("vlm_cache_hit", False) and pred in eval_classes):
            vlm_predictions_accepted_count += 1

        total_by_gt[gt] += 1
        if abstained:
            abstain_count += 1
            cm[gt]["UNKNOWN"] += 1
        else:
            p_val = pred if pred in eval_classes else "UNKNOWN"
            cm[gt][p_val] += 1
            total_by_pred[p_val] += 1
            if p_val == gt:
                correct_by_class[gt] += 1

    # Quality metrics
    recalls = {}
    precisions = {}
    f1s = {}
    for c in eval_classes:
        rec = (correct_by_class[c] / total_by_gt[c]) if total_by_gt[c] > 0 else 0.0
        prec = (correct_by_class[c] / total_by_pred[c]) if total_by_pred[c] > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        recalls[c] = round(rec, 4)
        precisions[c] = round(prec, 4)
        f1s[c] = round(f1, 4)

    balanced_acc = float(np.mean([recalls[c] for c in eval_classes if total_by_gt[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in eval_classes if total_by_gt[c] > 0]))

    # Queries invoked = actual VLM results used (live executed + cache hits)
    # When VLM is disabled, queries invoked is strictly 0.
    vlm_queries_invoked = vlm_inference_executed_count + vlm_cache_hits_count
    query_rate_pct = (vlm_queries_invoked / max(1, total)) * 100.0
    abstention_rate_pct = (abstain_count / max(1, total)) * 100.0

    queried_latencies_s = [
        end_to_end_latencies_s[i] for i, r in enumerate(records)
        if r.get("vlm_inference_executed") or r.get("vlm_cache_hit")
    ]

    return {
        "quality": {
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_recall": recalls,
            "per_class_precision": precisions,
            "per_class_f1": f1s,
            "confusion_matrix": cm,
            "abstention_rate_pct": round(abstention_rate_pct, 2),
        },
        "efficiency": {
            "total_instances": total,
            "vlm_queries_requested": vlm_query_requested_count,
            "vlm_inference_executed": vlm_inference_executed_count,
            "vlm_cache_hits": vlm_cache_hits_count,
            "vlm_queries_invoked": vlm_queries_invoked,
            "vlm_queries_avoided": total - vlm_queries_invoked,
            "vlm_query_rate_pct": round(query_rate_pct, 2),
            "vlm_outputs_valid": vlm_outputs_valid_count,
            "vlm_predictions_accepted": vlm_predictions_accepted_count,
            "latency_accounting": "MEASURED_LIVE_INFERENCE" if any(r.get("execution_mode") == "LIVE_INFERENCE" for r in records) else "HISTORICAL_REPLAY_PROJECTION",
            "decision_latency_ms": {
                "population": "all_instances",
                "mean": round(float(np.mean(decision_latencies_ms)), 3),
                "p50": round(float(np.percentile(decision_latencies_ms, 50)), 3),
                "p95": round(float(np.percentile(decision_latencies_ms, 95)), 3),
            },
            "end_to_end_latency_s": {
                "population": "all_instances_unqueried_include_0ms",
                "mean": round(float(np.mean(end_to_end_latencies_s)), 3),
                "p50": round(float(np.percentile(end_to_end_latencies_s, 50)), 3),
                "p95": round(float(np.percentile(end_to_end_latencies_s, 95)), 3),
            },
            "queried_instances_latency_s": {
                "population": "queried_instances_only",
                "mean": round(float(np.mean(queried_latencies_s)), 3) if queried_latencies_s else 0.0,
                "p50": round(float(np.percentile(queried_latencies_s, 50)), 3) if queried_latencies_s else 0.0,
                "p95": round(float(np.percentile(queried_latencies_s, 95)), 3) if queried_latencies_s else 0.0,
            },
        },
        "reliability": {
            "false_cheap_resolutions": sum(1 for r in records if r["decision"] == "RESOLVE_CHEAP" and r["prediction"] != r["ground_truth"]),
            "invalid_evidence_escalations": sum(1 for r in records if r["decision"] == "ESCALATE" and not r["evidence_valid"]),
            "vlm_abstentions": sum(1 for r in records if r.get("vlm_required") and r.get("abstained")),
        }
    }


def run_selective_router_evaluation(
    enable_vlm: bool = True,
    cache_path: Optional[Path] = None,
    mode: str = "LIVE_INFERENCE",
    server_url: str = DEFAULT_SERVER_URL,
) -> Dict[str, Any]:
    manifest_path = repo_root / "benchmarks" / "router_evaluation_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if cache_path is None:
        cache_path = repo_root / "outputs" / "router_ablation" / "router_vlm_cache.json"

    if enable_vlm and mode == "LIVE_INFERENCE":
        health = check_vlm_server_health(server_url)
        if health.get("status") != "ONLINE":
            print(f"[WARNING] VLM Server is OFFLINE at {server_url}: {health}. Falling back to CACHED_HISTORICAL.")
            mode = "CACHED_HISTORICAL"
        else:
            print(f"[LIVE MODE] VLM Server is ONLINE at {server_url}. Metrics will be derived strictly from live run.")

    router = SelectiveRouter(
        cache_path=cache_path,
        server_url=server_url,
        enable_vlm=enable_vlm,
        mode=mode,
    )

    adapter = EvidenceAdapter(repo_root=repo_root)
    instances = manifest["evaluation_instances"]
    print(f"\nEvaluating Selective Router on {len(instances)} physical instances (VLM Enabled: {enable_vlm}, Mode: {mode})...")

    results = []
    for inst in instances:
        # Obtain evidence strictly through evidence adapter (zero ground-truth leakage)
        evidence = adapter.build_instance_evidence(inst)
        routed = router.route_entity(evidence, repo_root=repo_root)

        routed["ground_truth"] = inst["contextual_movability_gt"]
        routed["semantic_class"] = inst["semantic_class"]
        results.append(routed)

    metrics = compute_metrics(results)
    out_dir = repo_root / "outputs" / "router_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "vlm_enabled" if enable_vlm else "vlm_disabled"
    out_json = out_dir / f"router_evaluation_{suffix}.json"

    summary = {
        "evaluation_name": f"Estimator-Aware Selective Router ({suffix})",
        "execution_mode": mode if enable_vlm else "DISABLED",
        "timestamp": time.time(),
        "metrics": metrics,
        "records": results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to: {out_json}")
    print(f"  • Balanced Accuracy: {metrics['quality']['balanced_accuracy']*100:.1f}%")
    print(f"  • Macro-F1:          {metrics['quality']['macro_f1']*100:.1f}%")
    print(f"  • VLM Query Rate:    {metrics['efficiency']['vlm_query_rate_pct']}% ({metrics['efficiency']['vlm_queries_invoked']}/{metrics['efficiency']['total_instances']})")
    print(f"  • Mean Latency:      {metrics['efficiency']['end_to_end_latency_s']['mean']:.2f} s")
    print(f"  • Abstention Rate:   {metrics['quality']['abstention_rate_pct']}%")

    return summary


if __name__ == "__main__":
    enable = "--no-vlm" not in sys.argv
    eval_mode = "CACHED_HISTORICAL" if "--cached" in sys.argv else "LIVE_INFERENCE"
    run_selective_router_evaluation(enable_vlm=enable, mode=eval_mode)
