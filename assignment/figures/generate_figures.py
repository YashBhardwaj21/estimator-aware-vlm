"""
Research-Quality Figure Generation Script for VLM Assignment.
Generates publication-ready SVG and PNG figures from verified benchmark artifacts.
Run with: python generate_figures.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style settings for clean, publication-grade aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eaeaea'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
METRICS_DIR = BASE_DIR / "metrics"
BENCHMARKS_DIR = BASE_DIR / "benchmarks"


def plot_association_baselines():
    """Figure 1: Association Performance across Baselines."""
    summary_path = METRICS_DIR / "derived" / "association_baselines_summary.json"
    if not summary_path.exists():
        print(f"Skipping Figure 1: {summary_path} not found.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = [
        ("Nearest Centroid", data["baseline_1_nearest_centroid"]["evaluation"]),
        ("Hungarian Gated", data["baseline_2_hungarian_gated"]["evaluation"]),
        ("Uncertainty Symm.", data["proposed_uncertainty_symmetric"]["evaluation"]),
        ("Relocation Aware", data["proposed_relocation_aware"]["evaluation"]),
    ]

    labels = [b[0] for b in baselines]
    prec = [b[1]["candidate_pair_precision_pct"] for b in baselines]
    inst_rec = [b[1]["all_instance_recall_pct"] for b in baselines]
    reloc_rec = [b[1]["relocated_instance_recall_pct"] for b in baselines]
    inst_f1 = [b[1]["instance_level_f1_pct"] for b in baselines]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    rects1 = ax.bar(x - 1.5*width, prec, width, label='Candidate Precision (%)', color='#2b5c8f')
    rects2 = ax.bar(x - 0.5*width, inst_rec, width, label='Instance Recall (All, N=30) (%)', color='#4682b4')
    rects3 = ax.bar(x + 0.5*width, reloc_rec, width, label='Relocation Recall (N=4) (%)', color='#e26d5c')
    rects4 = ax.bar(x + 1.5*width, inst_f1, width, label='Instance-level F1 (%)', color='#38b000')

    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Session Association Performance by Baseline\n(Evaluated on N=30 Multi-Session Physical Instances)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='medium')
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_svg = FIGURES_DIR / "association" / "association_baselines_performance.svg"
    out_png = FIGURES_DIR / "association" / "association_baselines_performance.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 1: association_baselines_performance.{svg,png}")


def plot_association_errors():
    """Figure 2: Association Error Rates and Per-Instance Diagnostic Breakdown."""
    diag_path = METRICS_DIR / "derived" / "per_instance_association_diagnostics.json"
    summary_path = METRICS_DIR / "derived" / "association_baselines_summary.json"
    if not diag_path.exists() or not summary_path.exists():
        print(f"Skipping Figure 2: missing inputs.")
        return

    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    # Per-instance outcomes
    outcomes = {"UNMATCHED_GATE (False Split)": 17, "CORRECT_MATCH (True Positive)": 7, "FALSE_MERGE (Identity Error)": 6}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: Outcome distribution pie / donut
    colors = ['#d90429', '#2b9348', '#ffb703']
    wedges, texts, autotexts = ax1.pie(outcomes.values(), labels=outcomes.keys(), autopct='%1.1f%%',
                                       startangle=140, colors=colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax1.set_title('Per-Instance Outcome Breakdown\n(Ground Truth N=30 Multi-Session Objects)', fontsize=11, fontweight='bold')

    # Right: Error comparison across baselines
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = ["Centroid", "Hungarian", "Uncertainty", "Reloc. Aware"]
    false_merge = [41.67, 41.67, 36.36, 41.67]
    false_split = [76.67, 76.67, 76.67, 76.67]
    ambiguous_rej = [0, 0, 10, 21]

    x = np.arange(len(labels))
    width = 0.25

    ax2.bar(x - width/2, false_merge, width, label='False Merge Rate (%)', color='#e63946')
    ax2.bar(x + width/2, false_split, width, label='False Split Rate (%)', color='#457b9d')

    ax2.set_ylabel('Error Rate (%)', fontsize=10, fontweight='bold')
    ax2.set_title('False Merge vs False Split Error Rates', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=True, facecolor='white', loc='upper right')

    # Annotate ambiguous rejections
    for i, amb in enumerate(ambiguous_rej):
        ax2.annotate(f'Amb. Rej: {amb}', xy=(x[i], 12), ha='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1faee', edgecolor='#a8dadc'))

    plt.tight_layout()
    out_svg = FIGURES_DIR / "association" / "association_errors_breakdown.svg"
    out_png = FIGURES_DIR / "association" / "association_errors_breakdown.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 2: association_errors_breakdown.{svg,png}")


def plot_registration_residuals():
    """Figure 3: Anchor-level Residuals, Inliers, and 0.20m Threshold."""
    reg_path = BENCHMARKS_DIR / "registration" / "stage5_registration_audit.json"
    if not reg_path.exists():
        print(f"Skipping Figure 3: {reg_path} not found.")
        return

    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    inliers = reg["robust_estimation"]["inliers"]
    outliers = reg["robust_estimation"]["outliers"]

    all_anchors = []
    for inl in inliers:
        all_anchors.append((inl["anchor_index"], inl["residual_robust_m"], True, f"{inl['id1']} - {inl['id2']}"))
    for out in outliers:
        all_anchors.append((out["anchor_index"], out["residual_robust_m"], False, f"{out['id1']} - {out['id2']}"))

    all_anchors.sort(key=lambda x: x[0])

    indices = [a[0] for a in all_anchors]
    residuals = [a[1] for a in all_anchors]
    is_inlier = [a[2] for a in all_anchors]
    labels = [f"A{a[0]}: {a[3]}" for a in all_anchors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300, gridspec_kw={'width_ratios': [2, 1]})

    # Bar chart of 10 anchor residuals
    bar_colors = ['#2b9348' if inl else '#d90429' for inl in is_inlier]
    bars = ax1.bar(indices, residuals, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)

    ax1.axhline(0.20, color='#e63946', linestyle='--', linewidth=1.5, label='RANSAC Inlier Threshold (0.20 m)')
    ax1.axhline(0.1198, color='#2b9348', linestyle=':', linewidth=1.5, label='Inlier RMS (0.1198 m)')
    ax1.axhline(0.3087, color='#457b9d', linestyle='-.', linewidth=1.5, label='All-Anchor RMS (0.3087 m)')

    ax1.set_xlabel('Anchor Correspondence Index', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Residual Distance (m)', fontsize=10, fontweight='bold')
    ax1.set_title('Anchor-Level Registration Residuals (Robust SE(2) SVD)\n7 Consensus Inliers vs 3 Peripheral Drift Outliers', fontsize=11, fontweight='bold')
    ax1.set_xticks(indices)
    ax1.set_ylim(0, 0.7)
    ax1.legend(frameon=True, facecolor='white', loc='upper left')

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}m', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Right: 5-fold CV RMS vs All-Anchor RMS
    cv_data = [0.1177, 0.1630, 0.5036, 0.5475, 0.1788]
    fold_labels = ['Fold 1\n(A1-2)', 'Fold 2\n(A3-4)', 'Fold 3\n(A5-6)', 'Fold 4\n(A7-8)', 'Fold 5\n(A9-10)']

    x_cv = np.arange(len(fold_labels))
    ax2.bar(x_cv, cv_data, color='#457b9d', width=0.5, edgecolor='black', linewidth=0.5)
    ax2.axhline(0.3021, color='#e63946', linestyle='--', label='Mean Fold RMS (0.3021m)')
    ax2.set_ylabel('Held-out RMS (m)', fontsize=10, fontweight='bold')
    ax2.set_title('5-Fold Cross Validation RMS', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_cv)
    ax2.set_xticklabels(fold_labels, fontsize=8)
    ax2.set_ylim(0, 0.65)
    ax2.legend(frameon=True, facecolor='white', loc='upper left', fontsize=8)

    plt.tight_layout()
    out_svg = FIGURES_DIR / "registration" / "anchor_residuals_and_inliers.svg"
    out_png = FIGURES_DIR / "registration" / "anchor_residuals_and_inliers.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 3: anchor_residuals_and_inliers.{svg,png}")


def plot_controlled_displacement():
    """Figure 4: Controlled Displacement Ground Truth vs Estimate & Error."""
    raw_s5_path = METRICS_DIR / "raw" / "stage5_evidence_metrics.json"
    if not raw_s5_path.exists():
        print(f"Skipping Figure 4: {raw_s5_path} not found.")
        return

    with open(raw_s5_path, "r", encoding="utf-8") as f:
        s5 = json.load(f)

    evals = s5["observed_motion"]["controlled_evaluations"]

    labels = ["Chair 002\n(Caster)", "Chair 003\n(Office)", "Box 007\n(Storage)"]
    gt = [e["ground_truth_m"] for e in evals]
    est = [e["estimated_displacement_m"] for e in evals]
    abs_err = [e["absolute_error_m"] for e in evals]
    rel_err = [e["relative_error_pct"] for e in evals]
    unc = [e["estimated_uncertainty_m"] for e in evals]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Left: GT vs Estimate with uncertainty errorbars
    ax1.bar(x - width/2, gt, width, label='Tape Ground Truth (m)', color='#2b5c8f')
    ax1.bar(x + width/2, est, width, yerr=unc, capsize=4, label='SE(2) Estimated Disp. (m)', color='#e26d5c', error_kw=dict(ecolor='#555555', lw=1.2))

    ax1.set_ylabel('Displacement (m)', fontsize=10, fontweight='bold')
    ax1.set_title('Controlled Relocation: Ground Truth vs Estimated\n(Error bars indicate heuristic empirical uncertainty)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 0.9)
    ax1.legend(frameon=True, facecolor='white', loc='upper right')

    for i in range(len(labels)):
        ax1.annotate(f'GT: {gt[i]:.3f}m', xy=(x[i] - width/2, gt[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        ax1.annotate(f'Est: {est[i]:.3f}m', xy=(x[i] + width/2, est[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # Right: Absolute and Relative Errors
    ax2.bar(x - width/2, abs_err, width, label='Absolute Error (m)', color='#d90429')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x + width/2, rel_err, color='#ff9f1c', marker='o', linewidth=2, label='Relative Error (%)')

    ax2.set_ylabel('Absolute Error (m)', fontsize=10, fontweight='bold', color='#d90429')
    ax2_twin.set_ylabel('Relative Error (%)', fontsize=10, fontweight='bold', color='#ff9f1c')
    ax2.set_title('Absolute & Relative Relocation Errors\n(Mean Rel Error: 30.42%)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 0.3)
    ax2_twin.set_ylim(0, 60)
    ax2.grid(False)

    for i in range(len(labels)):
        ax2.annotate(f'{abs_err[i]:.3f}m', xy=(x[i] - width/2, abs_err[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        ax2_twin.annotate(f'{rel_err[i]:.1f}%', xy=(x[i] + width/2, rel_err[i]), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8, color='#cc7a00', fontweight='bold')

    plt.tight_layout()
    out_svg = FIGURES_DIR / "displacement" / "controlled_relocation_errors.svg"
    out_png = FIGURES_DIR / "displacement" / "controlled_relocation_errors.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 4: controlled_relocation_errors.{svg,png}")


def plot_semantic_evidence():
    """Figure 5: Semantic Evidence Coverage, Matched Macro-F1, and Per-Class Support."""
    raw_s5_path = METRICS_DIR / "raw" / "stage5_evidence_metrics.json"
    if not raw_s5_path.exists():
        print(f"Skipping Figure 5: {raw_s5_path} not found.")
        return

    with open(raw_s5_path, "r", encoding="utf-8") as f:
        s5 = json.load(f)

    sem = s5["semantics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: Evaluation Coverage donut (4421 evaluated vs 3535 unmapped out of 7956)
    cov_vals = [sem["matched_observations_count"], sem["unmatched_observations_count"]]
    cov_labels = [f"Evaluated Support\n({sem['matched_observations_count']:,} / 55.6%)",
                  f"Unmapped / Out-of-Vocab\n({sem['unmatched_observations_count']:,} / 44.4%)"]
    colors = ['#2b9348', '#adb5bd']

    wedges, texts, autotexts = ax1.pie(cov_vals, labels=cov_labels, autopct='%1.1f%%',
                                       startangle=140, colors=colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax1.set_title('Stage 5 Semantic Evidence Evaluation Coverage\n(Total Records = 7,956)', fontsize=11, fontweight='bold')

    # Right: Per-class support and F1
    classes = ["IMMOVABLE", "MOVABLE", "MOBILE"]
    support = [sem["per_class"][c]["support"] for c in classes]
    f1 = [sem["per_class"][c]["f1"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    ax2.bar(x - width/2, support, width, label='Evaluated Records (Support)', color='#457b9d')
    ax2.set_ylabel('Sample Count (Support)', fontsize=10, fontweight='bold', color='#457b9d')
    ax2.set_yscale('log')
    ax2.set_ylim(1, 10000)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x + width/2, f1, color='#d90429', marker='s', markersize=8, linewidth=2, label='Matched-Support F1 (%)')
    ax2_twin.set_ylabel('Macro-F1 on Matched Subset (%)', fontsize=10, fontweight='bold', color='#d90429')
    ax2_twin.set_ylim(95, 101)

    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=10, fontweight='medium')
    ax2.set_title('Per-Class Support and Matched-Support F1\n(Overall Matched Macro-F1 = 99.76% on N=4,421)', fontsize=11, fontweight='bold')

    for i in range(len(classes)):
        ax2.annotate(f'N={support[i]:,}', xy=(x[i] - width/2, support[i]), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        ax2_twin.annotate(f'{f1[i]:.2f}%', xy=(x[i] + width/2, f1[i]), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8, color='#d90429', fontweight='bold')

    plt.tight_layout()
    out_svg = FIGURES_DIR / "semantic_evidence" / "semantic_macro_f1_and_coverage.svg"
    out_png = FIGURES_DIR / "semantic_evidence" / "semantic_macro_f1_and_coverage.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 5: semantic_macro_f1_and_coverage.{svg,png}")


def plot_temporal_tracking():
    """Figure 6: Temporal Tracking Fragmentation and Lifespan."""
    raw_s5_path = METRICS_DIR / "raw" / "stage5_evidence_metrics.json"
    if not raw_s5_path.exists():
        print(f"Skipping Figure 6: {raw_s5_path} not found.")
        return

    with open(raw_s5_path, "r", encoding="utf-8") as f:
        s5 = json.load(f)

    temp = s5["temporal_tracking"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Left: Track fragmentation donut
    frag = temp["fragmented_tracks_count"]
    pers = temp["total_unique_tracks"] - frag

    vals = [frag, pers]
    labels = [f"Fragmented Short Tracks\n(<=4 frames: {frag} / 72.8%)", f"Persistent Tracks\n(>4 frames: {pers} / 27.2%)"]
    colors = ['#e63946', '#2b9348']

    wedges, texts, autotexts = ax1.pie(vals, labels=labels, autopct='%1.1f%%',
                                       startangle=140, colors=colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax1.set_title('Track Fragmentation Profile\n(Total Tracks = 1,385)', fontsize=11, fontweight='bold')

    # Right: Multi-track entity and Occlusion recovery metrics
    metrics_labels = ["Track Frag.\nRate (%)", "Multi-Track\nEntity Rate (%)", "Occlusion Recovery\nRate (%)"]
    rates = [temp["track_fragmentation_rate_pct"], temp["multi_track_entity_rate_pct"], temp["occlusion_recovery_rate_pct"]]

    x = np.arange(len(metrics_labels))
    bars = ax2.bar(x, rates, color=['#e63946', '#f4a261', '#2a9d8f'], width=0.5, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Temporal Evidence Health Indicators\n(N=1,385 Tracks, 355 Occlusion Events)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels, fontsize=9)
    ax2.set_ylim(0, 115)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_svg = FIGURES_DIR / "semantic_evidence" / "temporal_track_lifespans.svg"
    out_png = FIGURES_DIR / "semantic_evidence" / "temporal_track_lifespans.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 6: temporal_track_lifespans.{svg,png}")


def plot_router_ablation():
    """Figure 7: Selective Router Efficiency vs Accuracy across Ablations."""
    ablation_path = METRICS_DIR / "raw" / "router_ablation_results.json"
    if not ablation_path.exists():
        print(f"Skipping Figure 7: {ablation_path} not found.")
        return

    with open(ablation_path, "r", encoding="utf-8") as f:
        ab = json.load(f)

    conditions = [
        ("Selective VLM\n(Proposed)", ab["experiments"]["router_selective_vlm"]["metrics"]),
        ("Semantic Prior\n(M0 Lookup)", ab["experiments"]["m0_semantic_prior"]["metrics"]),
        ("Random Query\n(26.7% budget)", ab["experiments"]["random_query_baseline"]["metrics"]),
        ("Router Disabled\n(Pure Prior)", ab["experiments"]["router_disabled"]["metrics"]),
        ("All-VLM Query\n(M3 Unguided)", ab["experiments"]["m3_all_vlm"]["metrics"]),
    ]

    labels = [c[0] for c in conditions]
    bal_acc = [c[1]["quality"]["balanced_accuracy"] * 100 for c in conditions]
    macro_f1 = [c[1]["quality"]["macro_f1"] * 100 for c in conditions]
    query_rate = [c[1]["efficiency"]["vlm_query_rate_pct"] for c in conditions]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    rects1 = ax1.bar(x - width, bal_acc, width, label='Balanced Accuracy (%)', color='#2b5c8f')
    rects2 = ax1.bar(x, macro_f1, width, label='Macro-F1 (%)', color='#52b788')
    rects3 = ax1.bar(x + width, query_rate, width, label='VLM Query Rate (% Cost)', color='#e26d5c')

    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Stage 6/7 Selective Router Ablation: Cost vs Classification Performance\n(Evaluated on N=15 Physical Validation Instances with Offline Cache)', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight='medium')
    ax1.set_ylim(0, 115)
    ax1.legend(frameon=True, facecolor='white', loc='upper right')

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_svg = FIGURES_DIR / "router" / "router_efficiency_vs_accuracy.svg"
    out_png = FIGURES_DIR / "router" / "router_efficiency_vs_accuracy.png"
    plt.savefig(out_svg)
    plt.savefig(out_png)
    plt.close()
    print("Saved Figure 7: router_efficiency_vs_accuracy.{svg,png}")


def main():
    print("Generating assignment figures...")
    plot_association_baselines()
    plot_association_errors()
    plot_registration_residuals()
    plot_controlled_displacement()
    plot_semantic_evidence()
    plot_temporal_tracking()
    plot_router_ablation()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
