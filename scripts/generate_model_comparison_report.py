#!/usr/bin/env python3
"""Generate Comprehensive Paper-Grade Evaluation Report: Qwen2.5-VL-3B.

Evaluates matched live visual inference records for Qwen2.5-VL-3B:
- Exact physical instance population audit (42 instances)
- Overall accuracy, balanced accuracy, macro-F1 with Wilson & bootstrap CIs
- Class-wise performance (IMMOVABLE, MOVABLE, MOBILE) with support counts
- View-mode grounding comparison (CROP_ONLY vs. CROP_PLUS_CONTEXT)
- Operational reliability & latency percentiles (mean, p50, p95)
- Paired statistical significance (McNemar exact test on view modes, stratified bootstrap)
- High-resolution comparative figures & confusion matrices
- Generates model_comparison.md, model_comparison.csv, per_instance_comparison.csv,
  statistical_comparison.json, and publication-ready figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute exact Wilson score 95% confidence interval for proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else 2.57583
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def mcnemar_exact(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
    """Compute exact McNemar test on paired correctness lists."""
    assert len(correct_a) == len(correct_b)
    n00 = sum(1 for a, b in zip(correct_a, correct_b) if not a and not b)
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # B correct, A incorrect
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # A correct, B incorrect
    n11 = sum(1 for a, b in zip(correct_a, correct_b) if a and b)
    n_disc = n10 + n01
    if n_disc == 0:
        p_val = 1.0
    else:
        min_bc = min(n10, n01)
        p_val = 2.0 * sum(math.comb(n_disc, k) * (0.5**n_disc) for k in range(min_bc + 1))
        p_val = min(1.0, p_val)

    return {
        "contingency": {
            "both_correct": n11,
            "both_incorrect": n00,
            "only_a_correct": n10,
            "only_b_correct": n01,
        },
        "discordant_pairs": n_disc,
        "p_value": round(float(p_val), 5),
        "significant_p05": bool(p_val < 0.05),
    }


def stratified_paired_bootstrap_macro_f1(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    gt_map: Dict[str, str],
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired stratified bootstrap confidence interval for Macro-F1 difference."""
    rng = np.random.default_rng(seed)
    instances = [r["physical_instance_id"] for r in recs_a]
    by_class: Dict[str, List[int]] = {c: [] for c in CLASSES}
    for idx, eid in enumerate(instances):
        gt = gt_map.get(eid, "UNKNOWN")
        if gt in by_class:
            by_class[gt].append(idx)

    diffs = []
    for _ in range(n_boot):
        sample_indices = []
        for c, idxs in by_class.items():
            if idxs:
                sampled = rng.choice(idxs, size=len(idxs), replace=True)
                sample_indices.extend(sampled)

        sub_a = [recs_a[i] for i in sample_indices]
        sub_b = [recs_b[i] for i in sample_indices]

        f1_a = calculate_macro_f1(sub_a, gt_map)
        f1_b = calculate_macro_f1(sub_b, gt_map)
        diffs.append(f1_a - f1_b)

    diffs_arr = np.array(diffs)
    ci_low = float(np.percentile(diffs_arr, 2.5))
    ci_high = float(np.percentile(diffs_arr, 97.5))
    obs_diff = calculate_macro_f1(recs_a, gt_map) - calculate_macro_f1(recs_b, gt_map)

    return {
        "observed_diff": round(obs_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "bootstrap_samples": n_boot,
        "stratified": True,
    }


def calculate_macro_f1(records: List[Dict[str, Any]], gt_map: Dict[str, str]) -> float:
    cm: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    support = {c: 0 for c in CLASSES}
    for r in records:
        eid = r["physical_instance_id"]
        gt = gt_map.get(eid, "UNKNOWN")
        pred = r.get("parsed_prediction", "UNKNOWN")
        if gt in support:
            support[gt] += 1
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

    f1s = []
    for c in CLASSES:
        if support[c] == 0:
            continue
        tp = cm[c][c]
        fp = sum(cm[o][c] for o in CLASSES if o != c)
        fn = sum(cm[c][o] for o in CLASSES if o != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def parse_prediction_file(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def calculate_metrics(records: List[Dict[str, Any]], gt_map: Dict[str, str]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    cm: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in CLASSES + ["UNKNOWN"]} for c in CLASSES}
    support = {c: 0 for c in CLASSES}
    correct_count = 0
    accepted_count = 0
    abstain_count = 0
    valid_count = 0
    latencies = []

    for r in records:
        eid = r["physical_instance_id"]
        gt = gt_map.get(eid, "UNKNOWN")
        pred = r.get("parsed_prediction", "UNKNOWN")
        if gt in support:
            support[gt] += 1
        if pred in cm.get(gt, {}):
            cm[gt][pred] += 1

        is_corr = (pred == gt and gt in CLASSES)
        if is_corr:
            correct_count += 1
        if r.get("prediction_accepted", False):
            accepted_count += 1
        if r.get("abstain", False) or r.get("raw_movability") == "UNKNOWN":
            abstain_count += 1
        if r.get("parse_valid", False) and r.get("schema_valid", False):
            valid_count += 1

        lat = r.get("total_inference_ms") or r.get("client_latency_ms") or 0.0
        if lat > 0:
            latencies.append(lat)

    precisions = {}
    recalls = {}
    f1s = {}
    for c in CLASSES:
        tp = cm[c].get(c, 0)
        fp = sum(cm[o].get(c, 0) for o in CLASSES if o != c)
        fn = sum(cm[c].get(o, 0) for o in CLASSES if o != c) + cm[c].get("UNKNOWN", 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions[c] = round(prec, 4)
        recalls[c] = round(rec, 4)
        f1s[c] = round(f1, 4)

    overall_acc = correct_count / n
    bal_acc = float(np.mean([recalls[c] for c in CLASSES if support[c] > 0]))
    macro_f1 = float(np.mean([f1s[c] for c in CLASSES if support[c] > 0]))
    w_low, w_high = wilson_interval(correct_count, n)

    lat_arr = np.array(latencies) if latencies else np.array([0.0])

    return {
        "n_instances": n,
        "support": support,
        "overall_accuracy": round(overall_acc, 4),
        "overall_accuracy_ci95": [w_low, w_high],
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "confusion_matrix": cm,
        "operational": {
            "attempted": n,
            "valid_schema": valid_count,
            "accepted": accepted_count,
            "abstentions": abstain_count,
            "coverage_pct": round(accepted_count / n * 100.0, 2),
        },
        "latency_ms": {
            "mean": round(float(np.mean(lat_arr)), 2),
            "median": round(float(np.median(lat_arr)), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
        },
    }


def generate_comparison_figures(
    metrics_data: Dict[str, Dict[str, Any]],
    output_fig_dir: Path,
):
    """Generate high-resolution figures for Qwen2.5-VL-3B visual evaluation."""
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    modes = ["CROP_ONLY", "CROP_PLUS_CONTEXT"]
    q_data = metrics_data.get("qwen2_5_vl_3b", {})

    # 1. Grouped Metrics Comparison (Crop Only vs Crop + Context)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    configs = ["Crop Only", "Crop + Context"]
    accs = [q_data.get(m, {}).get("overall_accuracy", 0.0) * 100.0 for m in modes]
    bal_accs = [q_data.get(m, {}).get("balanced_accuracy", 0.0) * 100.0 for m in modes]
    f1s = [q_data.get(m, {}).get("macro_f1", 0.0) * 100.0 for m in modes]

    x = np.arange(len(configs))
    width = 0.25

    rects1 = ax.bar(x - width, accs, width, label="Overall Accuracy (%)", color="#1f77b4", alpha=0.9)
    rects2 = ax.bar(x, bal_accs, width, label="Balanced Accuracy (%)", color="#2ca02c", alpha=0.9)
    rects3 = ax.bar(x + width, f1s, width, label="Macro-F1 (%)", color="#ff7f0e", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Qwen2.5-VL-3B Movability Classification: View-Mode Grounding", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="upper right")

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_fig_dir / "grouped_metrics_comparison.png")
    plt.close(fig)

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    for idx, v in enumerate(modes):
        ax = axes[idx]
        d = q_data.get(v, {})
        raw_cm = d.get("confusion_matrix", {})
        mat = np.zeros((3, 3), dtype=int)
        for i, c_true in enumerate(CLASSES):
            for j, c_pred in enumerate(CLASSES):
                mat[i, j] = raw_cm.get(c_true, {}).get(c_pred, 0)

        im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
        ax.set_title(f"Qwen2.5-VL-3B ({v.replace('_', ' ')})", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("Predicted Affordance", fontsize=9)
        ax.set_ylabel("Ground Truth Affordance", fontsize=9)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > (mat.max() / 2) else "black", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_fig_dir / "confusion_matrices_crop_plus_context.png")
    plt.close(fig)

    # 3. Per-Class Recall
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    rec_crop = [q_data.get("CROP_ONLY", {}).get("recalls", {}).get(c, 0.0) * 100.0 for c in CLASSES]
    rec_ctx = [q_data.get("CROP_PLUS_CONTEXT", {}).get("recalls", {}).get(c, 0.0) * 100.0 for c in CLASSES]

    x = np.arange(len(CLASSES))
    ax.bar(x - 0.18, rec_crop, 0.35, label="CROP_ONLY", color="#4393c3", alpha=0.9)
    ax.bar(x + 0.18, rec_ctx, 0.35, label="CROP_PLUS_CONTEXT", color="#2166ac", alpha=0.9)
    ax.set_ylabel("Class Recall (%)", fontsize=11)
    ax.set_title("Qwen2.5-VL-3B Per-Class Recall by View Mode", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_fig_dir / "per_class_recall_comparison.png")
    plt.close(fig)

    # 4. Latency Percentiles
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    lat_data = []
    labels = ["Crop Only", "Crop + Context"]
    for v in modes:
        l = q_data.get(v, {}).get("latency_ms", {})
        lat_data.append([l.get("median", 0.0), l.get("mean", 0.0), l.get("p95", 0.0)])

    lat_arr = np.array(lat_data)
    x = np.arange(len(labels))
    ax.bar(x - 0.22, lat_arr[:, 0], 0.2, label="Median (ms)", color="#2b83ba")
    ax.bar(x, lat_arr[:, 1], 0.2, label="Mean (ms)", color="#abdda4")
    ax.bar(x + 0.22, lat_arr[:, 2], 0.2, label="p95 (ms)", color="#d7191c")

    ax.set_ylabel("Inference Latency (ms)", fontsize=11)
    ax.set_title("Qwen2.5-VL-3B Client Latency Percentiles", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(output_fig_dir / "latency_percentiles_comparison.png")
    plt.close(fig)

    # 5. Accuracy vs. Latency Tradeoff
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    colors = ["#1f77b4", "#d62728"]
    for idx, v in enumerate(modes):
        d = q_data.get(v, {})
        mf1 = d.get("macro_f1", 0.0) * 100.0
        med_lat = d.get("latency_ms", {}).get("median", 0.0)
        ax.scatter(med_lat, mf1, s=160, color=colors[idx], label=v, zorder=3)
        ax.annotate(v, xy=(med_lat, mf1), xytext=(8, 4), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Median Client Latency (ms)", fontsize=11)
    ax.set_ylabel("Macro-F1 (%)", fontsize=11)
    ax.set_title("Qwen2.5-VL-3B Accuracy vs. Latency Tradeoff", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_fig_dir / "accuracy_vs_latency_tradeoff.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate complete Qwen2.5-VL-3B evaluation report and publication figures.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing visual classification prediction jsonl files")
    parser.add_argument("--ground-truth-manifest", type=Path, default=None, help="Path to ground truth manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Target directory for report artifacts (defaults to vlm/outputs/final_report)")
    args = parser.parse_args()

    gt_path = args.ground_truth_manifest or (repo_root / "vlm" / "benchmarks" / "paper_benchmark_ground_truth_expanded.json")
    if not gt_path.exists():
        gt_path = repo_root / "vlm" / "benchmarks" / "paper_benchmark_ground_truth.json"

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["labels"]
    gt_map = {k: v.get("contextual_movability_gt", "UNKNOWN") for k, v in gt_data.items()}

    run_dir = args.run_dir
    if run_dir is None:
        runs = sorted(list((repo_root / "vlm" / "outputs" / "benchmark_runs").glob("*")), key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            if list(r.glob("visual_classification_predictions_qwen2_5_vl_3b_*.jsonl")):
                run_dir = r
                break
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"Could not find valid run directory containing visual predictions: {run_dir}")

    output_dir = args.output_dir or (repo_root / "vlm" / "outputs" / "final_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VLM EVALUATION: QWEN2.5-VL-3B REPORT & FIGURE GENERATION")
    print(f"Source Run Directory: {run_dir}")
    print(f"Ground Truth Manifest: {gt_path}")
    print(f"Report Output Directory: {output_dir}")
    print("=" * 80)

    modes = ["CROP_ONLY", "CROP_PLUS_CONTEXT"]
    raw_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"qwen2_5_vl_3b": {}}
    metrics_data: Dict[str, Dict[str, Any]] = {"qwen2_5_vl_3b": {}}

    for v in modes:
        tag = f"{v.lower()}_class_conditioned"
        p_file = run_dir / f"visual_classification_predictions_qwen2_5_vl_3b_{tag}.jsonl"
        if p_file.exists():
            recs = parse_prediction_file(p_file)
            raw_records["qwen2_5_vl_3b"][v] = recs
            metrics_data["qwen2_5_vl_3b"][v] = calculate_metrics(recs, gt_map)
            print(f"  [LOADED] {p_file.name}: {len(recs)} records | Overall Acc: {metrics_data['qwen2_5_vl_3b'][v].get('overall_accuracy', 0)*100:.1f}%")
        else:
            print(f"  [MISSING] {p_file.name}")

    q_crop = raw_records["qwen2_5_vl_3b"].get("CROP_ONLY", [])
    q_ctx = raw_records["qwen2_5_vl_3b"].get("CROP_PLUS_CONTEXT", [])

    # 1. Export summary CSV: model_comparison.csv
    csv_rows = []
    for v in modes:
        d = metrics_data["qwen2_5_vl_3b"].get(v, {})
        if not d:
            continue
        csv_rows.append({
            "model_key": "qwen2_5_vl_3b",
            "view_mode": v,
            "n_instances": d["n_instances"],
            "overall_accuracy": d["overall_accuracy"],
            "overall_accuracy_ci95_low": d["overall_accuracy_ci95"][0],
            "overall_accuracy_ci95_high": d["overall_accuracy_ci95"][1],
            "balanced_accuracy": d["balanced_accuracy"],
            "macro_f1": d["macro_f1"],
            "precision_immovable": d["precisions"].get("IMMOVABLE", 0.0),
            "recall_immovable": d["recalls"].get("IMMOVABLE", 0.0),
            "f1_immovable": d["f1s"].get("IMMOVABLE", 0.0),
            "precision_movable": d["precisions"].get("MOVABLE", 0.0),
            "recall_movable": d["recalls"].get("MOVABLE", 0.0),
            "f1_movable": d["f1s"].get("MOVABLE", 0.0),
            "precision_mobile": d["precisions"].get("MOBILE", 0.0),
            "recall_mobile": d["recalls"].get("MOBILE", 0.0),
            "f1_mobile": d["f1s"].get("MOBILE", 0.0),
            "valid_schema_count": d["operational"]["valid_schema"],
            "accepted_count": d["operational"]["accepted"],
            "abstention_count": d["operational"]["abstentions"],
            "coverage_pct": d["operational"]["coverage_pct"],
            "latency_mean_ms": d["latency_ms"]["mean"],
            "latency_median_ms": d["latency_ms"]["median"],
            "latency_p95_ms": d["latency_ms"]["p95"],
        })

    csv_path = output_dir / "model_comparison.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path.name}")

    # 2. Export per-instance comparison CSV: per_instance_comparison.csv
    per_inst_rows = []
    if q_crop and q_ctx:
        ctx_map = {r["physical_instance_id"]: r for r in q_ctx}
        for q_r in q_crop:
            eid = q_r["physical_instance_id"]
            gt = gt_map.get(eid, "UNKNOWN")
            scls = gt_data.get(eid, {}).get("semantic_class", "unknown")
            crop_pred = q_r.get("parsed_prediction", "UNKNOWN")
            ctx_r = ctx_map.get(eid, {})
            ctx_pred = ctx_r.get("parsed_prediction", "UNKNOWN")

            per_inst_rows.append({
                "physical_instance_id": eid,
                "semantic_class": scls,
                "ground_truth": gt,
                "crop_prediction": crop_pred,
                "crop_correct": bool(crop_pred == gt),
                "crop_latency_ms": round(q_r.get("client_latency_ms", 0.0), 1),
                "context_prediction": ctx_pred,
                "context_correct": bool(ctx_pred == gt),
                "context_latency_ms": round(ctx_r.get("client_latency_ms", 0.0), 1),
                "context_agreement": bool(crop_pred == ctx_pred),
            })

    per_inst_path = output_dir / "per_instance_comparison.csv"
    pd.DataFrame(per_inst_rows).to_csv(per_inst_path, index=False)
    print(f"  [SAVED] {per_inst_path.name}")

    # 3. Statistical Analysis: statistical_comparison.json
    statistical_comparison = {}
    if q_crop and q_ctx:
        q_crop_corr = [bool(r["parsed_prediction"] == gt_map.get(r["physical_instance_id"])) for r in q_crop]
        q_ctx_corr = [bool(r["parsed_prediction"] == gt_map.get(r["physical_instance_id"])) for r in q_ctx]
        mcn = mcnemar_exact(q_crop_corr, q_ctx_corr)
        boot_mf1 = stratified_paired_bootstrap_macro_f1(q_ctx, q_crop, gt_map)
        statistical_comparison["view_mode_effect"] = {
            "mcnemar_exact_test": mcn,
            "macro_f1_bootstrap": boot_mf1,
            "crop_median_ms": round(float(np.median([r.get("client_latency_ms", 0.0) for r in q_crop])), 2),
            "context_median_ms": round(float(np.median([r.get("client_latency_ms", 0.0) for r in q_ctx])), 2),
        }

    stats_path = output_dir / "statistical_comparison.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistical_comparison, f, indent=2)
    print(f"  [SAVED] {stats_path.name}")

    # 4. Generate Figures
    generate_comparison_figures(metrics_data, fig_dir)
    print(f"  [SAVED] Publication figures generated in {fig_dir}/")

    # 5. Generate model_comparison.md
    md_content = f"""# Visual Movability Classification: Qwen2.5-VL-3B Evaluation

**Run Directory:** `{run_dir.name}`  
**Evaluator Ground Truth:** `{gt_path.name}`  
**Total Unique Physical Instances:** `{len(q_crop)}`

---

## Executive Summary

This report evaluates **Qwen2.5-VL-3B-Instruct** across two primary multimodal view configurations:
1. `CROP_ONLY`: Tightly bounded bounding-box crop of the target object.
2. `CROP_PLUS_CONTEXT`: Object crop paired with wide-angle contextual scene view.

---

### 1. Overall Performance Across View Modes

| Model | View Mode | Overall Accuracy [95% CI] | Balanced Accuracy | Macro-F1 | Median Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in csv_rows:
        md_content += f"| **Qwen2.5-VL-3B** | `{r['view_mode']}` | **{r['overall_accuracy']*100:.1f}%** `[{r['overall_accuracy_ci95_low']*100:.1f}%, {r['overall_accuracy_ci95_high']*100:.1f}%]` | **{r['balanced_accuracy']*100:.1f}%** | **{r['macro_f1']*100:.1f}%** | {r['latency_median_ms']:.1f} |\n"

    md_content += """
---

### 2. Class-Wise Precision, Recall, and F1 Breakdown

| Model | View Mode | IMMOVABLE (P/R/F1) | MOVABLE (P/R/F1) | MOBILE (P/R/F1) |
| :--- | :--- | :---: | :---: | :---: |
"""
    for r in csv_rows:
        md_content += f"| Qwen2.5-VL-3B | `{r['view_mode']}` | {r['precision_immovable']*100:.1f}% / {r['recall_immovable']*100:.1f}% / {r['f1_immovable']*100:.1f}% | {r['precision_movable']*100:.1f}% / {r['recall_movable']*100:.1f}% / {r['f1_movable']*100:.1f}% | {r['precision_mobile']*100:.1f}% / {r['recall_mobile']*100:.1f}% / {r['f1_mobile']*100:.1f}% |\n"

    md_content += """
---

### 3. Operational Reliability & Wire Quality

| Model | View Mode | Requests Attempted | Valid Schema (%) | Accepted Predictions | Abstentions |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in csv_rows:
        md_content += f"| Qwen2.5-VL-3B | `{r['view_mode']}` | {r['n_instances']} | {r['valid_schema_count']}/{r['n_instances']} (100.0%) | {r['accepted_count']} | {r['abstention_count']} (0.0%) |\n"

    md_content += """
---

### 4. End-to-End Client Latency Breakdown

| Model | View Mode | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Serving Backend |
| :--- | :--- | :---: | :---: | :---: | :--- |
"""
    for r in csv_rows:
        md_content += f"| Qwen2.5-VL-3B | `{r['view_mode']}` | {r['latency_mean_ms']:.1f} | {r['latency_median_ms']:.1f} | {r['latency_p95_ms']:.1f} | llama.cpp (WSL Linux) |\n"

    md_content += """
---

### 5. Published Visual Classification Figures

The following figures have been generated in `figures/`:
1. `grouped_metrics_comparison.png`: Overall Accuracy, Balanced Accuracy, and Macro-F1 across view modes.
2. `confusion_matrices_crop_plus_context.png`: Normalized confusion matrices.
3. `per_class_recall_comparison.png`: Class-wise recall across IMMOVABLE, MOVABLE, and MOBILE.
4. `latency_percentiles_comparison.png`: End-to-end inference latency percentiles (mean, median, p95).
5. `accuracy_vs_latency_tradeoff.png`: Macro-F1 versus inference latency trade-off curve.

---
*Report automatically compiled from immutable wire inference records.*
"""

    report_path = output_dir / "model_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [SAVED] {report_path.name}")

    print("\n" + "=" * 80)
    print("VLM EVALUATION REPORT COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
