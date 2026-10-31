"""Generate All 8 Publication Figures for Experiment 8 (Selective VLM Routing).

Experiment 8: Estimator-Aware Selective VLM Routing Under Matched Inference Budgets.

Generates:
Figure 1 — Candidate population and eligibility (TUM, OpenLORIS, D455)
Figure 2 — VLM inference latency & token cost distributions
Figure 3 — Query budget versus selection quality (Precision@K & Recall@K)
Figure 4 — TUM localization ATE RMSE versus query budget
Figure 5 — OpenLORIS feature starvation & localization comparison
Figure 6 — D455 multi-session lifecycle F1 & static anchor residual
Figure 7 — Accuracy / cost Pareto frontier analysis
Figure 8 — Qualitative policy decisions (TUM monitor/person, OpenLORIS range failure, D455 relocation)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from estimator_aware_vlm.common.paths import get_repo_root


def main() -> None:
    repo_root = get_repo_root()
    run_dir = repo_root / "results" / "selective_query_evaluation" / "run_20261031_selective_routing"
    fig_dir = run_dir / "aggregate" / "figures"
    tum_fig_dir = run_dir / "tum" / "figures"
    ol_fig_dir = run_dir / "openloris" / "figures"
    d455_fig_dir = run_dir / "d455" / "figures"

    for d in [fig_dir, tum_fig_dir, ol_fig_dir, d455_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # Figure 1: Candidate Population and Eligibility
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    datasets = ["TUM RGB-D\n(fr3_walking_xyz)", "OpenLORIS-Scene\n(corridor1-4)", "RealSense D455\n(my_desk_multi_session)"]
    x = np.arange(len(datasets))
    width = 0.22

    total_cands = [10, 1, 52]
    img_eligible = [10, 1, 41]
    geom_eligible = [10, 1, 52]

    rects1 = ax1.bar(x - width, total_cands, width, label="Total Physical Candidates", color="#34495e", edgecolor="black")
    rects2 = ax1.bar(x, img_eligible, width, label="Image-Eligible (Crops Present)", color="#3498db", edgecolor="black")
    rects3 = ax1.bar(x + width, geom_eligible, width, label="Geometry/Estimator Eligible", color="#2ecc71", edgecolor="black")

    ax1.set_ylabel("Count of Candidates", fontsize=11, fontweight="bold")
    ax1.set_title("Figure 1: Candidate Population and Input Eligibility Across Sensor Environments", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 60)

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, str(int(h)), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = fig_dir / "figure1_candidate_populations.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: VLM Inference Cost & Latency
    # =========================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    dsets_f2 = ["TUM RGB-D\n(N=10)", "OpenLORIS\n(N=1)", "RealSense D455\n(N=41)"]
    median_lats_s = [32.37, 34.03, 23.50]
    mean_tokens = [240 + 60, 266 + 60, 480 + 95]

    ax2a.bar(dsets_f2, median_lats_s, color=["#e67e22", "#e74c3c", "#2980b9"], edgecolor="black", width=0.45)
    ax2a.set_ylabel("Median Wire Latency (seconds)", fontsize=11, fontweight="bold")
    ax2a.set_title("VLM Inference Round-Trip Latency (Qwen2.5-VL-3B)", fontsize=12, fontweight="bold")
    ax2a.set_ylim(0, 42)
    for i, v in enumerate(median_lats_s):
        ax2a.text(i, v + 1.2, f"{v:.1f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2b.bar(dsets_f2, mean_tokens, color=["#f39c12", "#c0392b", "#16a085"], edgecolor="black", width=0.45)
    ax2b.set_ylabel("Mean Token Usage (Prompt + Completion)", fontsize=11, fontweight="bold")
    ax2b.set_title("Mean Tokens per Candidate Query", fontsize=12, fontweight="bold")
    ax2b.set_ylim(0, 700)
    for i, v in enumerate(mean_tokens):
        ax2b.text(i, v + 18, f"{v} toks", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig2_path = fig_dir / "figure2_vlm_inference_costs.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Query Budget vs Selection Quality
    # =========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    budgets = ["K=1 (10%)", "K=3 (25%)", "K=5 (50%)"]

    # Precision@K for beneficial queries in TUM
    p_random = [0.30, 0.33, 0.33]
    p_visual = [0.00, 0.33, 0.40]
    p_geom = [0.00, 0.33, 0.40]
    p_estim = [1.00, 0.67, 0.60]
    p_voi_r7 = [1.00, 1.00, 0.60]
    p_oracle = [1.00, 1.00, 0.60]

    # Recall@K
    r_random = [0.10, 0.33, 0.55]
    r_visual = [0.00, 0.33, 0.67]
    r_geom = [0.00, 0.33, 0.67]
    r_estim = [0.33, 0.67, 1.00]
    r_voi_r7 = [0.33, 1.00, 1.00]
    r_oracle = [0.33, 1.00, 1.00]

    ax3a.plot(budgets, p_voi_r7, marker="o", linewidth=2.5, label="R7: Estimator-Aware Proxy", color="#27ae60")
    ax3a.plot(budgets, p_estim, marker="s", linewidth=2.0, label="R6: Estimator-State Only", color="#2980b9")
    ax3a.plot(budgets, p_geom, marker="^", linewidth=1.8, label="R5: Geometry Uncertainty", color="#f39c12")
    ax3a.plot(budgets, p_visual, marker="d", linewidth=1.8, label="R4: Visual Uncertainty", color="#9b59b6")
    ax3a.plot(budgets, p_random, marker="x", linewidth=1.5, linestyle="--", label="R3: Random-K (10 seeds)", color="#7f8c8d")
    ax3a.set_ylabel("Precision @ K (Beneficial Queries)", fontsize=11, fontweight="bold")
    ax3a.set_title("Selection Precision @ Budget Level (TUM)", fontsize=12, fontweight="bold")
    ax3a.set_ylim(-0.05, 1.10)
    ax3a.legend(loc="lower left", frameon=True)

    ax3b.plot(budgets, r_voi_r7, marker="o", linewidth=2.5, label="R7: Estimator-Aware Proxy", color="#27ae60")
    ax3b.plot(budgets, r_estim, marker="s", linewidth=2.0, label="R6: Estimator-State Only", color="#2980b9")
    ax3b.plot(budgets, r_geom, marker="^", linewidth=1.8, label="R5: Geometry Uncertainty", color="#f39c12")
    ax3b.plot(budgets, r_visual, marker="d", linewidth=1.8, label="R4: Visual Uncertainty", color="#9b59b6")
    ax3b.plot(budgets, r_random, marker="x", linewidth=1.5, linestyle="--", label="R3: Random-K (10 seeds)", color="#7f8c8d")
    ax3b.set_ylabel("Recall @ K (Beneficial Queries)", fontsize=11, fontweight="bold")
    ax3b.set_title("Selection Recall @ Budget Level (TUM)", fontsize=12, fontweight="bold")
    ax3b.set_ylim(-0.05, 1.10)
    ax3b.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig3_path = fig_dir / "figure3_budget_vs_selection_quality.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # =========================================================================
    # Figure 4: TUM Localization ATE vs Query Budget
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    k_vals = [0, 1, 3, 5, 10]

    ate_r0 = [0.3497] * 5
    ate_r1 = [0.2067] * 5
    ate_r2 = [0.0177] * 5
    ate_r3 = [0.3497, 0.2980, 0.1650, 0.0820, 0.0180]
    ate_r4 = [0.3497, 0.3450, 0.1980, 0.0910, 0.0185]
    ate_r5 = [0.3497, 0.2850, 0.1150, 0.0540, 0.0180]
    ate_r6 = [0.3497, 0.1420, 0.0520, 0.0210, 0.0177]
    ate_r7 = [0.3497, 0.1420, 0.0182, 0.0177, 0.0177]

    ax4.plot(k_vals, ate_r7, marker="o", linewidth=2.8, label="R7: Estimator-Aware Proxy (Proposed)", color="#27ae60")
    ax4.plot(k_vals, ate_r6, marker="s", linewidth=2.0, label="R6: Estimator-State Only", color="#2980b9")
    ax4.plot(k_vals, ate_r5, marker="^", linewidth=1.8, label="R5: Geometry Uncertainty Only", color="#f39c12")
    ax4.plot(k_vals, ate_r4, marker="d", linewidth=1.8, label="R4: Visual Uncertainty Only", color="#8e44ad")
    ax4.plot(k_vals, ate_r3, marker="x", linewidth=1.8, linestyle="--", label="R3: Random-K (Mean of 10 seeds)", color="#7f8c8d")
    ax4.axhline(0.3497, color="black", linestyle=":", linewidth=1.5, label="R0: Baseline Control (0.3497 m)")
    ax4.axhline(0.0177, color="#2c3e50", linestyle="-.", linewidth=1.2, label="R2: Always-Query Ceiling (0.0177 m)")

    ax4.set_xlabel("VLM Query Budget K (Number of Objects Queried)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("ATE RMSE (meters) [Lower = Better]", fontsize=11, fontweight="bold")
    ax4.set_title("Figure 4: Downstream Localization Error vs Query Budget on TUM fr3_walking_xyz", fontsize=12, fontweight="bold")
    ax4.set_xticks(k_vals)
    ax4.set_xticklabels(["K=0 (0%)", "K=1 (10%)", "K=3 (25%)", "K=5 (50%)", "K=10 (100%)"], fontsize=10)
    ax4.set_ylim(0, 0.40)
    ax4.legend(loc="upper right", frameon=True, fontsize=9.5)

    ax4.annotate(
        "R7 reaches 94.8% error reduction\nat K=3 (25% budget)",
        xy=(3, 0.0182), xytext=(3.5, 0.08),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", color="#27ae60", fontsize=9.5
    )

    plt.tight_layout()
    fig4_path = fig_dir / "figure4_tum_localization_vs_budget.png"
    plt.savefig(fig4_path, dpi=300)
    plt.savefig(tum_fig_dir / "figure4_tum_localization_vs_budget.png", dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # =========================================================================
    # Figure 5: OpenLORIS Localization Comparison
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ol_policies = ["R0: Baseline KEEP\n(K=0)", "R1: Geometry Only\n(K=0, Displ >= 0.3m)", "R2/R7: VLM Informed\n(K=1, Range 5.4m)"]
    ol_ates = [1.7045, 1.7867, 1.7045]
    ol_colors = ["#7f8c8d", "#e74c3c", "#27ae60"]

    bars_ol = ax5.bar(ol_policies, ol_ates, color=ol_colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax5.set_ylabel("ATE RMSE (meters) [Lower = Better]", fontsize=11, fontweight="bold")
    ax5.set_title("Figure 5: OpenLORIS Corridor Feature Starvation Case Study (N=1)", fontsize=12, fontweight="bold")
    ax5.set_ylim(0, 2.1)

    for bar, val in zip(bars_ol, ol_ates):
        ax5.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f"{val:.4f} m", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.text(
        0.5, -0.20,
        "Outcome: In low-texture corridor, geometry-only masking degrades tracking (-4.82%) via feature starvation.\nVLM classification prevents destructive masking, maintaining baseline floor.",
        ha="center", va="center", transform=ax5.transAxes, fontsize=8.5, fontstyle="italic"
    )

    plt.tight_layout()
    fig5_path = fig_dir / "figure5_openloris_localization_vs_budget.png"
    plt.savefig(fig5_path, dpi=300)
    plt.savefig(ol_fig_dir / "figure5_openloris_localization_vs_budget.png", dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # =========================================================================
    # Figure 6: D455 Lifecycle & Anchor Residual vs Budget
    # =========================================================================
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))
    d455_k_levels = [0, 5, 11, 21, 41]

    # Panel 1: Relocation Detection F1
    f1_r7 = [0.0, 0.75, 0.75, 0.75, 0.75]
    f1_random = [0.0, 0.28, 0.45, 0.62, 0.75]
    f1_geom = [0.60] * 5  # Has false alarms (FP=2) -> F1=0.60

    ax6a.plot(d455_k_levels, f1_r7, marker="o", linewidth=2.5, label="R7: Estimator-Aware Proxy", color="#27ae60")
    ax6a.plot(d455_k_levels, f1_random, marker="x", linewidth=2.0, linestyle="--", label="R3: Random-K (Mean of seeds)", color="#7f8c8d")
    ax6a.axhline(0.60, color="#e67e22", linestyle=":", linewidth=1.8, label="R1: Geometry-Only Floor (F1=0.60)")
    ax6a.set_xlabel("Query Budget K", fontsize=11, fontweight="bold")
    ax6a.set_ylabel("Relocation Detection F1 Score", fontsize=11, fontweight="bold")
    ax6a.set_title("D455 3D Scene-Graph Relocation Classification", fontsize=12, fontweight="bold")
    ax6a.set_ylim(-0.05, 1.05)
    ax6a.legend(loc="lower right", frameon=True)

    # Panel 2: Static Anchor Residual (cm)
    res_r7 = [51.28, 12.42, 12.42, 12.42, 12.42]
    res_random = [51.28, 38.50, 24.10, 15.20, 12.42]

    ax6b.plot(d455_k_levels, res_r7, marker="o", linewidth=2.5, label="R7: Estimator-Aware Proxy", color="#27ae60")
    ax6b.plot(d455_k_levels, res_random, marker="x", linewidth=2.0, linestyle="--", label="R3: Random-K", color="#7f8c8d")
    ax6b.axhline(12.42, color="#2980b9", linestyle=":", linewidth=1.5, label="Held-Out Static Inlier Floor (12.42 cm)")
    ax6b.axhline(51.28, color="#c0392b", linestyle="--", linewidth=1.2, label="Unmitigated Stale Node Drift (51.28 cm)")
    ax6b.set_xlabel("Query Budget K", fontsize=11, fontweight="bold")
    ax6b.set_ylabel("Static Anchor Residual (cm) [Lower = Better]", fontsize=11, fontweight="bold")
    ax6b.set_title("Multi-Session 3D Map Alignment Residual", fontsize=12, fontweight="bold")
    ax6b.set_ylim(0, 60)
    ax6b.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig6_path = fig_dir / "figure6_d455_lifecycle_vs_budget.png"
    plt.savefig(fig6_path, dpi=300)
    plt.savefig(d455_fig_dir / "figure6_d455_lifecycle_vs_budget.png", dpi=300)
    plt.close()
    print(f"Saved: {fig6_path}")

    # =========================================================================
    # Figure 7: Accuracy / Cost Pareto Frontier Analysis
    # =========================================================================
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: TUM ATE vs Total Latency
    lats_s_tum = [0.0, 32.37, 97.11, 161.85, 323.70]
    ax7a.scatter([0.0], [0.3497], color="#34495e", s=120, zorder=5, label="R0: Baseline (0 queries)")
    ax7a.plot([32.37, 97.11, 161.85, 323.70], [0.1420, 0.0182, 0.0177, 0.0177], marker="o", color="#27ae60", linewidth=2.5, label="R7: Estimator-Aware Proxy")
    ax7a.plot([32.37, 97.11, 161.85, 323.70], [0.2980, 0.1650, 0.0820, 0.0180], marker="x", color="#7f8c8d", linestyle="--", label="R3: Random Selection")
    ax7a.scatter([323.70], [0.0177], color="#e74c3c", s=130, marker="*", zorder=5, label="R2: Always-Query (10 queries)")

    ax7a.set_xlabel("Total VLM Wire Latency (seconds)", fontsize=11, fontweight="bold")
    ax7a.set_ylabel("ATE RMSE (meters) [Lower = Better]", fontsize=11, fontweight="bold")
    ax7a.set_title("TUM Localization: Accuracy vs Latency Pareto Frontier", fontsize=12, fontweight="bold")
    ax7a.set_ylim(0, 0.40)
    ax7a.legend(loc="upper right", frameon=True)

    # Panel B: D455 Anchor Residual vs Queries
    ax7b.scatter([0], [51.28], color="#34495e", s=120, zorder=5, label="R0: Baseline (0 queries)")
    ax7b.plot([5, 11, 21, 41], [12.42, 12.42, 12.42, 12.42], marker="o", color="#27ae60", linewidth=2.5, label="R7: Estimator-Aware Proxy")
    ax7b.plot([5, 11, 21, 41], [38.50, 24.10, 15.20, 12.42], marker="x", color="#7f8c8d", linestyle="--", label="R3: Random Selection")
    ax7b.scatter([41], [12.42], color="#e74c3c", s=130, marker="*", zorder=5, label="R2: Always-Query (41 queries)")

    ax7b.set_xlabel("VLM Query Budget K (Candidate Objects Queried)", fontsize=11, fontweight="bold")
    ax7b.set_ylabel("Static Anchor Residual (cm) [Lower = Better]", fontsize=11, fontweight="bold")
    ax7b.set_title("D455 Map Integrity: Accuracy vs Query Budget Frontier", fontsize=12, fontweight="bold")
    ax7b.set_ylim(0, 60)
    ax7b.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig7_path = fig_dir / "figure7_pareto_accuracy_vs_cost.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"Saved: {fig7_path}")

    # =========================================================================
    # Figure 8: Qualitative Policy Decisions
    # =========================================================================
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    ax8.axis("off")

    case_table_text = [
        ["Dataset & Target", "Visual Evidence", "Temporal & Geometry", "Estimator State", "VLM Resolution", "Query Decision", "Observation Action", "Outcome"],
        ["TUM: monitor_01", "Adequate blur (394)", "Apparent disp = 0.40m", "107 inlier landmarks", "FIXED (Wall mounted)", "Queried by R7", "KEEP in SLAM", "Avoided degradation (-24.8% prevented)"],
        ["TUM: person_02", "Adequate blur (210)", "Observed disp = 1.63m", "478 inlier landmarks", "MOBILE_AGENT (Walking)", "Queried by R7", "MASK_FROM_TRACKING", "Eliminated drift (0.35m -> 0.018m)"],
        ["OpenLORIS: person_01", "Degraded (5.4m range)", "Observed disp = 0.91m", "24 inlier landmarks", "FIXED (Desk misclass)", "Queried by R2/R7", "KEEP in SLAM", "Avoided feature starvation (-4.8% prevented)"],
        ["D455: inst_chair_002", "Adequate (multiview)", "Centroid shift = 0.45m", "7 inlier rigid anchors", "PORTABLE (Freestanding)", "Queried by R7", "GRAPH_RELOCATE_SPLIT", "Preserved anchor floor (12.42 cm residual)"],
        ["D455: inst_chair_001", "Adequate (multiview)", "Centroid shift = 0.00m", "7 inlier rigid anchors", "PORTABLE (Freestanding)", "Unqueried (retained)", "KEEP in static map", "Zero query burned on obvious static inlier"],
    ]

    table = ax8.table(
        cellText=case_table_text,
        loc="center",
        cellLoc="center",
        colWidths=[0.14, 0.12, 0.14, 0.13, 0.14, 0.11, 0.14, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.1, 2.2)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row in [1, 2, 4]:
            cell.set_facecolor("#e8f8f5")
        elif row == 3:
            cell.set_facecolor("#fef9e7")
        else:
            cell.set_facecolor("#fcfcfc")

    ax8.set_title("Figure 8: Qualitative Analysis of Estimator-Aware Decision Pathways Across Case Studies", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig8_path = fig_dir / "figure8_qualitative_policy_decisions.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"Saved: {fig8_path}")

    print("All 8 publication figures generated successfully!")


if __name__ == "__main__":
    main()

