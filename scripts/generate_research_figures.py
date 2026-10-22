#!/usr/bin/env python3
"""Generate Minimal, Professional Scientific Figures for Research Report.

Generates:
- High-resolution (300 DPI) publication-quality PNGs
- Clean vector SVGs
Saved to:
- results/figures/
- vlm/outputs/final_report/figures/
- vlm/outputs/research_package/figures/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Minimalist professional aesthetic (clean neutral slate/blue styling, clear typography)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "medium",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})

PRIMARY_COLOR = "#1f4e79"
SECONDARY_COLOR = "#2e75b6"
ACCENT_COLOR = "#c55a11"
SUCCESS_COLOR = "#385723"
NEUTRAL_COLOR = "#595959"
LIGHT_BG = "#f8f9fa"


def save_fig(fig: plt.Figure, fig_id: str, out_dirs: List[Path]):
    """Save figure in both vector SVG and 300 DPI PNG formats."""
    fig.tight_layout()
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = out_dir / f"{fig_id}.svg"
        png_path = out_dir / f"{fig_id}.png"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_system_architecture_svg(out_dirs: List[Path]):
    """Figure 3.1: Actual Overall System Architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    boxes = [
        ("Sensor Observation Stream\n(RGB-D + YOLO Bounding Boxes)", 0.05, 0.65, 0.25, 0.22, "#d9e1f2"),
        ("Stage 5 Pre-Router Extraction\n(Geometry, Temporal, Prior)", 0.36, 0.65, 0.26, 0.22, "#bdd7ee"),
        ("Estimator-Aware Router (B6)\n(Utility = \u0394L_SLAM - \u03bb C_VLM)", 0.68, 0.65, 0.27, 0.22, "#f8cbad"),
        ("Decoupled Scene Graph\n(Pose History & Metric Nodes)", 0.68, 0.15, 0.27, 0.22, "#c6e0b4"),
        ("VLM Multi-View Inference\n(Qwen2.5-VL-3B Prompting)", 0.36, 0.15, 0.26, 0.22, "#ffe699"),
        ("SE(2) Anchor Registration\n(Consensus Residual = 0.1198m)", 0.05, 0.15, 0.25, 0.22, "#e2efda"),
    ]

    for title, x, y, w, h, col in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="#333333", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", transform=ax.transAxes, fontsize=9.5, weight="semibold")

    arrow_props = dict(facecolor="#262626", edgecolor="#262626", arrowstyle="->", lw=1.5)
    ax.annotate("", xy=(0.36, 0.76), xytext=(0.30, 0.76), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.68, 0.76), xytext=(0.62, 0.76), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.49, 0.37), xytext=(0.49, 0.65), xycoords="axes fraction", arrowprops=arrow_props)
    ax.text(0.50, 0.51, "Escalate (Query Utility \u2265 0)", ha="left", va="center", transform=ax.transAxes, fontsize=8, color="#833c0c")
    ax.annotate("", xy=(0.68, 0.26), xytext=(0.62, 0.26), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.815, 0.37), xytext=(0.815, 0.65), xycoords="axes fraction", arrowprops=arrow_props)
    ax.text(0.825, 0.51, "Cheap Prior (Avoids VLM)", ha="left", va="center", transform=ax.transAxes, fontsize=8, color="#375623")
    ax.annotate("", xy=(0.30, 0.26), xytext=(0.36, 0.26), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("Figure 3.1: Dynamic 3D Scene Graph Architecture and Evidence Flow", pad=12)
    save_fig(fig, "fig_3_1_system_architecture", out_dirs)


def generate_pipeline_data_flow_svg(out_dirs: List[Path]):
    """Figure 3.2: Pipeline Data Flow."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    stages = [
        ("Input Observations\n(N=7,956 records)", 0.03, 0.40, 0.16, 0.25, "#d9e1f2"),
        ("Tracklet Consolidation\n(1970 tracks -> clusters)", 0.23, 0.40, 0.16, 0.25, "#bdd7ee"),
        ("SE(2) Registration\n(RANSAC, 7 inliers)", 0.43, 0.40, 0.16, 0.25, "#bdd7ee"),
        ("Selective Router\n(B0 - B7, A1 - A4)", 0.63, 0.40, 0.16, 0.25, "#f8cbad"),
        ("Contextual Scene Graph\n(53 physical entities)", 0.83, 0.40, 0.16, 0.25, "#c6e0b4"),
    ]
    for title, x, y, w, h, col in stages:
        rect = plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="#333333", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", transform=ax.transAxes, fontsize=9)

    arrow_props = dict(facecolor="#262626", edgecolor="#262626", arrowstyle="->", lw=1.5)
    for i in range(len(stages) - 1):
        x_from = stages[i][1] + stages[i][3]
        x_to = stages[i+1][1]
        ax.annotate("", xy=(x_to, 0.525), xytext=(x_from, 0.525), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("Figure 3.2: End-to-End Pipeline Execution and Data Processing Lineage", pad=12)
    save_fig(fig, "fig_3_2_pipeline_data_flow", out_dirs)


def generate_router_decision_flow_svg(out_dirs: List[Path]):
    """Figure 4.1: Cheap Path vs Selective VLM Decision Flow."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")

    steps = [
        ("Input Physical Instance", 0.05, 0.70, 0.22, 0.20, "#e7e6e6"),
        ("Estimator Pose Validity\n(has_valid_pose ?)", 0.35, 0.70, 0.26, 0.20, "#fff2cc"),
        ("Affordance / Displacement\nAmbiguity Check", 0.68, 0.70, 0.26, 0.20, "#fff2cc"),
        ("Resolve Cheap\n(Zero VLM Query, ~0ms)", 0.35, 0.20, 0.26, 0.20, "#c6e0b4"),
        ("Escalate to VLM\n(Multi-View Query, ~21s)", 0.68, 0.20, 0.26, 0.20, "#f8cbad"),
    ]
    for title, x, y, w, h, col in steps:
        rect = plt.Rectangle((x, y), w, h, facecolor=col, edgecolor="#333333", linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", transform=ax.transAxes, fontsize=9)

    arrow_props = dict(facecolor="#262626", edgecolor="#262626", arrowstyle="->", lw=1.5)
    ax.annotate("", xy=(0.35, 0.80), xytext=(0.27, 0.80), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.68, 0.80), xytext=(0.61, 0.80), xycoords="axes fraction", arrowprops=arrow_props)
    ax.text(0.645, 0.83, "Valid Pose", ha="center", va="bottom", transform=ax.transAxes, fontsize=8)

    ax.annotate("", xy=(0.48, 0.40), xytext=(0.48, 0.70), xycoords="axes fraction", arrowprops=arrow_props)
    ax.text(0.50, 0.55, "Unambiguous / Mobile", ha="left", va="center", transform=ax.transAxes, fontsize=8)

    ax.annotate("", xy=(0.81, 0.40), xytext=(0.81, 0.70), xycoords="axes fraction", arrowprops=arrow_props)
    ax.text(0.83, 0.55, "Utility \u2265 0", ha="left", va="center", transform=ax.transAxes, fontsize=8)

    ax.set_title("Figure 4.1: Estimator-Aware Selective Routing Decision Logic", pad=12)
    save_fig(fig, "fig_4_1_router_decision_flow", out_dirs)


def find_latest_benchmark_run_dir() -> Path:
    bench_dir = repo_root / "vlm" / "outputs" / "benchmark_runs"
    runs = sorted([d for d in bench_dir.iterdir() if d.is_dir() and d.name.startswith("paper_benchmark_")])
    if runs:
        return runs[-1]
    return repo_root / "vlm" / "outputs" / "benchmark_runs" / "paper_benchmark_20261019_190419"


def generate_pareto_efficiency_frontier(out_dirs: List[Path]):
    """Figure 6.1: Pareto Efficiency Frontier (Normalized Cost vs. Macro-F1)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    run_dir = find_latest_benchmark_run_dir()
    summary_path = run_dir / "router_policies_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        policies = summary.get("policy_summaries", {})
    else:
        policies = {}

    policy_order = [
        ("B0_MAJORITY", "B0: Majority", "#888888", "o"),
        ("B1_SEMANTIC_PRIOR", "B1: Semantic Prior", "#2e75b6", "s"),
        ("B2_GEOMETRY_MOTION", "B2: Geometry/Motion", "#5b9bd5", "^"),
        ("B3_ZERO_SHOT_CROP_ONLY", "B3: Crop-Only", "#ed7d31", "d"),
        ("B4_ZERO_SHOT_CROP_PLUS_CONTEXT", "B4: Crop+Context", "#f1a340", "d"),
        ("B5_ALWAYS_VLM", "B5: Always-VLM", "#c55a11", "v"),
        ("B6_ESTIMATOR_AWARE_ROUTER", "B6: Estimator-Aware (Proposed)", "#1f4e79", "*"),
        ("B7_ORACLE", "B7: Oracle Bound", "#385723", "p"),
        ("A1_NO_UNCERTAINTY", "A1: No Uncertainty", "#70ad47", "x"),
        ("A2_NO_DISPLACEMENT", "A2: No Displacement", "#a9d18e", "+"),
        ("A3_NO_PRIOR", "A3: No Prior", "#e06666", "1"),
        ("A4_CROP_ONLY", "A4: Router (Crop Only)", "#674ea7", "2"),
    ]

    costs = []
    f1s = []
    labels = []

    for key, disp_name, col, marker in policy_order:
        p_data = policies.get(key, {}).get("metrics", {})
        cost = p_data.get("normalized_compute_cost", 0.0)
        f1 = p_data.get("macro_f1", 0.0) * 100.0
        costs.append(cost)
        f1s.append(f1)
        labels.append((key, disp_name, cost, f1, col, marker))

    # Plot scatter
    for key, disp_name, cost, f1, col, marker in labels:
        size = 180 if "B6" in key or "B7" in key or "B5" in key else 100
        ax.scatter(cost, f1, color=col, marker=marker, s=size, zorder=5, label=disp_name)

        offset_x = 0.008
        offset_y = 1.0
        if key == "B6_ESTIMATOR_AWARE_ROUTER":
            offset_y = -3.2
            offset_x = -0.015
        elif key == "B1_SEMANTIC_PRIOR":
            offset_y = -3.2
        elif key == "B5_ALWAYS_VLM":
            offset_y = 1.5
            offset_x = -0.05
        elif key in ["B3_ZERO_SHOT_CROP_ONLY", "B4_ZERO_SHOT_CROP_PLUS_CONTEXT", "A3_NO_PRIOR"]:
            continue  # Avoid overlapping clutter for identical zero-shot points

        ax.annotate(
            disp_name,
            (cost, f1),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8.5,
            weight="bold" if "B6" in key else "normal",
            color=col,
        )

    # Draw Pareto Frontier Line connecting B0 -> B1 -> B6 -> B7
    frontier_x = [0.0, 0.0, 0.085, 0.0094]  # Oracle is theoretical bound
    frontier_y = [30.61, 95.93, 96.60, 100.0]
    ax.plot([0.0, 0.0, 0.085], [30.61, 95.93, 96.60], linestyle="--", color="#1f4e79", alpha=0.7, lw=1.5, label="Empirical Pareto Frontier")

    # Annotate efficiency gain of B6 over B5
    ax.annotate(
        "B6 vs B5:\n+20.8% Macro-F1\n-78.6% Compute Cost",
        xy=(0.085, 96.60),
        xytext=(0.18, 86.0),
        arrowprops=dict(facecolor="#1f4e79", edgecolor="#1f4e79", arrowstyle="->", lw=1.3),
        fontsize=9,
        weight="semibold",
        color="#1f4e79",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef4f9", ec="#1f4e79", lw=1.0),
    )

    ax.set_xlabel("Normalized VLM Inference Compute Cost (0.0 = Cheap, 1.0 = All VLM)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_xlim(-0.02, 0.45)
    ax.set_ylim(20, 105)
    ax.set_title("Figure 6.1: Pareto Efficiency Frontier: Quality vs. Compute Cost (N=53 Physical Entities)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    save_fig(fig, "fig_6_1_pareto_efficiency_frontier", out_dirs)


def generate_dual_confusion_matrices(out_dirs: List[Path]):
    """Figure 6.2: Dual Confusion Matrices (Visual Classification & B6 Router)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    labels = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    short_labels = ["IMM", "MOV", "MOB"]

    # Panel A: Visual Classification (Qwen2.5-VL-3B, N=42 image-eligible instances)
    # Ground truth: 3 IMM, 37 MOV, 2 MOB
    # Qwen predictions: IMM: 3 TP, 0 FN (2 false alarms from MOV: fan, shelf); MOV: 35 TP, 2 FN; MOB: 2 TP, 0 FN
    cm_vis = np.array([
        [3, 0, 0],
        [2, 35, 0],
        [0, 0, 2],
    ])

    im1 = ax1.imshow(cm_vis, cmap="Blues", interpolation="nearest")
    ax1.set_title("Panel A: Qwen2.5-VL-3B Visual Model (N=42)\nOverall Acc: 95.2% | Macro-F1: 90.7%", fontsize=10)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(short_labels)
    ax1.set_yticklabels(short_labels)
    ax1.set_xlabel("Predicted Mobility")
    ax1.set_ylabel("Ground Truth Mobility")

    for i in range(3):
        for j in range(3):
            val = cm_vis[i, j]
            color = "white" if val > 15 else "black"
            ax1.text(j, i, str(val), ha="center", va="center", color=color, weight="bold", fontsize=11)

    # Panel B: Estimator-Aware Router B6 (N=53 physical entities)
    # Ground truth: 5 IMM, 45 MOV, 3 MOB
    # B6 predictions: IMM: 5 TP (1 FP from MOV); MOV: 44 TP (1 FN); MOB: 3 TP (0 FN)
    cm_router = np.array([
        [5, 0, 0],
        [1, 44, 0],
        [0, 0, 3],
    ])

    im2 = ax2.imshow(cm_router, cmap="Blues", interpolation="nearest")
    ax2.set_title("Panel B: Estimator-Aware Router B6 (N=53)\nOverall Acc: 98.1% | Macro-F1: 96.6%", fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(short_labels)
    ax2.set_yticklabels(short_labels)
    ax2.set_xlabel("Predicted Mobility")
    ax2.set_ylabel("Ground Truth Mobility")

    for i in range(3):
        for j in range(3):
            val = cm_router[i, j]
            color = "white" if val > 20 else "black"
            ax2.text(j, i, str(val), ha="center", va="center", color=color, weight="bold", fontsize=11)

    fig.suptitle("Figure 6.2: Dual Confusion Matrices: Visual Efficacy vs. Estimator-Aware Fusion", fontsize=11, weight="bold")
    save_fig(fig, "fig_6_2_dual_confusion_matrices", out_dirs)


def generate_downstream_lifecycle_actions(out_dirs: List[Path]):
    """Figure 6.3: Downstream SLAM Landmark Lifecycle & Corruption Risk Breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    policies = ["B0: Majority", "B1: Prior", "B2: Motion", "B5: Always VLM", "B6: Router (Ours)", "B7: Oracle"]
    x = np.arange(len(policies))
    width = 0.35

    # Panel A: Action Accuracy and Landmark Retention Recall
    accs = [86.8, 94.3, 92.5, 92.5, 94.3, 94.3]
    retention_recall = [92.0, 94.0, 92.0, 94.0, 94.0, 94.0]

    b1 = ax1.bar(x - width/2, accs, width, label="Landmark Action Accuracy (%)", color=PRIMARY_COLOR, alpha=0.9)
    b2 = ax1.bar(x + width/2, retention_recall, width, label="Landmark Retention Recall (%)", color=SECONDARY_COLOR, alpha=0.85)

    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylim(70, 105)
    ax1.axhline(94.0, color="#2e75b6", linestyle="--", alpha=0.5, label="Ceiling Retention (94.0%)")
    ax1.set_title("Panel A: SLAM Landmark Decision Accuracy & Retention", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8)

    # Panel B: Corruption Risk FPs & Mobile Agent Rejection Rate
    fps = [2, 0, 0, 1, 0, 0]  # False positives retained (corruption risk)
    mobile_rejection = [0.0, 100.0, 100.0, 66.7, 100.0, 100.0]

    ax2_twin = ax2.twinx()
    b3 = ax2.bar(x - width/2, fps, width, label="Corruption Risk (FPs Retained)", color=ACCENT_COLOR, alpha=0.9)
    b4 = ax2_twin.bar(x + width/2, mobile_rejection, width, label="Mobile Agent Rejection (%)", color=SUCCESS_COLOR, alpha=0.85)

    ax2.set_ylabel("Corrupting Landmarks Retained (Count)", color=ACCENT_COLOR)
    ax2_twin.set_ylabel("Mobile Agent Purge Rate (%)", color=SUCCESS_COLOR)
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 3)
    ax2_twin.set_ylim(0, 115)
    ax2_twin.spines["top"].set_visible(False)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    ax2.set_title("Panel B: Map Protection: Corruption Risk vs Agent Purge", fontsize=10)

    fig.suptitle("Figure 6.3: Downstream SLAM Landmark Lifecycle and Map Integrity Maintenance", fontsize=11, weight="bold")
    save_fig(fig, "fig_6_3_landmark_lifecycle_actions", out_dirs)


def generate_controlled_relocation_figure(out_dirs: List[Path]):
    """Figure 6.4: Controlled Object Relocation Ground Truth vs Estimated Shift."""
    fig, ax = plt.subplots(figsize=(7, 4))
    events = ["Chair 002\n(Caster)", "Chair 003\n(Office)", "Storage Box 007\n(Container)"]
    gt_vals = [0.5630, 0.3860, 0.3960]
    est_vals = [0.4889, 0.3512, 0.5729]
    errors_pct = [13.2, 9.0, 44.7]
    x = np.arange(len(events))
    width = 0.35

    ax.bar(x - width/2, gt_vals, width, label="Physical Tape Ground Truth (m)", color=NEUTRAL_COLOR, alpha=0.85)
    ax.bar(x + width/2, est_vals, width, label="SLAM Estimated 3D Shift (m)", color=PRIMARY_COLOR, alpha=0.9)
    for i, p in enumerate(errors_pct):
        ax.text(x[i] + width/2, est_vals[i] + 0.02, f"\u0394 {p:.1f}%", ha="center", fontsize=8.5, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Displacement Magnitude (m)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Figure 6.4: Controlled Object Relocation: Ground Truth vs Estimated 3D Displacement")
    ax.legend(loc="upper right")
    save_fig(fig, "fig_6_4_controlled_relocation", out_dirs)


if __name__ == "__main__":
    out_dirs = [
        repo_root / "results" / "figures",
        repo_root / "vlm" / "outputs" / "final_report" / "figures",
        repo_root / "vlm" / "outputs" / "research_package" / "figures",
    ]
    print("Generating publication-grade scientific figures...")
    generate_system_architecture_svg(out_dirs)
    generate_pipeline_data_flow_svg(out_dirs)
    generate_router_decision_flow_svg(out_dirs)
    generate_pareto_efficiency_frontier(out_dirs)
    generate_dual_confusion_matrices(out_dirs)
    generate_downstream_lifecycle_actions(out_dirs)
    generate_controlled_relocation_figure(out_dirs)
    print("All scientific figures successfully generated in PNG (300 DPI) and vector SVG!")
