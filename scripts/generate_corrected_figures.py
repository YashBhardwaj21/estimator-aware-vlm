"""Generate publication figures for campaign audits."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "results", "publication_campaign", "run_20261101_execution_audit", "figures"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#cccccc"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5

# FIGURE 1: Multi-Sequence Localization (Verified Baselines vs Interventions)
def generate_fig1():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    sequences = [
        "TUM XYZ\n(9.12m)",
        "TUM Halfsphere\n(5.38m)",
        "TUM Static\n(1.22m)",
        "OpenLORIS\nCorridor (33.72m)",
        "Bonn Crowd\n(18.23m)",
        "Bonn Box\n(12.44m)"
    ]

    # Verified Baselines
    baseline_ates = [0.4597, 0.2166, 0.0166, 1.7045, 0.6368, 0.5108]

    # Verified Interventions (Single candidate verified where available; Joint marked with hatch)
    # On Walking XYZ, single person intervention (T1) is 0.3729; Joint (T3) is 0.1242
    # On Halfsphere, joint (S2) is 0.1216
    # On Static, no moving object masking physically performed (ATE remains 0.0166 m, negative -0.0254 expunged)
    # On OpenLORIS, single person (C1) is 1.6395
    # On Bonn Crowd, no isolated trajectory on disk -> baseline plotted, annotated "No Isolated Run"
    # On Bonn Box, no isolated trajectory on disk -> baseline plotted, annotated "No Isolated Run"

    x = np.arange(len(sequences))
    width = 0.35

    rects1 = ax.bar(x - width/2, baseline_ates, width, label="Unmitigated Baseline (Verified)", color="#2b5c8f", edgecolor="#1a365d")

    # Second bar: Interventions
    interv_values = [0.3729, 0.1216, 0.0166, 1.6395, 0.6368, 0.5108]
    colors = ["#2e7d32", "#43a047", "#78909c", "#2e7d32", "#bdbdbd", "#bdbdbd"]
    hatches = ["", "///", "xx", "", "\\\\\\", "\\\\\\"]

    rects2 = ax.bar(x + width/2, interv_values, width, label="Post-Intervention (Verified)", color=colors, edgecolor="#263238", hatch=hatches)

    ax.set_ylabel("Absolute Trajectory Error RMSE (m)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1: Verified Multi-Sequence Localization Accuracy (ATE RMSE)\nAudited Physical SLAM Trajectories vs Controls", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sequences, fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(0, 1.95)

    # Value labels
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.4f}m", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    for i, r in enumerate(rects2):
        h = r.get_height()
        if i == 2:
            lbl = f"{h:.4f}m\n(Static)"
        elif i in [4, 5]:
            lbl = f"{h:.4f}m\n(Unchanged)"
        elif i == 1:
            lbl = f"{h:.4f}m\n(Joint S2)"
        else:
            lbl = f"{h:.4f}m"
        ax.annotate(lbl, xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Annotations
    ax.text(2, 0.35, "Static sequence:\nNegative ATE (-0.0254m)\nexpunged as formulaic fault.\nTrue baseline: 0.0166m",
            ha="center", va="bottom", fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="#fff9c4", ec="#fbc02d", lw=1))

    ax.text(4.5, 1.25, "Bonn Sequences:\nIndividual candidate trajectories\nnot executed on disk.\nBaselines verified.",
            ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#9e9e9e", lw=1))

    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "corrected_fig1.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Generated {p}")

# FIGURE 2: Policy Cost vs Utility Tradeoff (Global K=9 vs Per-Sequence Ceiling)
def generate_fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Left: Tokens Consumed vs Exclusion Precision
    policies = ["R0 (Passive)", "R1 (Geom Only)", "R2 (Exhaustive)", "R3 (Random)", "R4 (Vis Unc)", "R5 (Geom Unc)", "R6 (Disp)", "R7 (EDRR Proxy)", "R8 (Oracle)"]
    tokens_global = [0, 0, 9522, 2382, 2382, 2382, 2382, 2382, 2382]
    precision_global = [0.0, 66.7, 100.0, 44.4, 77.8, 100.0, 100.0, 100.0, 100.0]

    colors = ["#78909c", "#8d6e63", "#d32f2f", "#ffa000", "#1976d2", "#0288d1", "#7b1fa2", "#2e7d32", "#388e3c"]

    for i in range(len(policies)):
        ax1.scatter(tokens_global[i], precision_global[i], color=colors[i], s=120, zorder=4, edgecolor="black", linewidth=1)
        offset_y = 3 if i != 6 else -6
        offset_x = 100 if i not in [2, 0] else -150
        ax1.annotate(policies[i], (tokens_global[i], precision_global[i]),
                     xytext=(offset_x, offset_y), textcoords="offset points", fontsize=8, fontweight="bold")

    ax1.axvline(2382, color="#2e7d32", linestyle=":", alpha=0.7, label="Global 25% Budget ($K=9$, 75% savings)")
    ax1.set_xlabel("VLM Tokens Consumed", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Dynamic Object Exclusion Precision (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A: Token Cost vs Detection Precision (Global $K=9$)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 115)
    ax1.set_xlim(-500, 10500)
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Right: Jaccard Similarity Heatmap under Global Budget K=9
    labels = ["R4 (Vis)", "R5 (Geom)", "R6 (Disp)", "R7 (EDRR)"]
    jaccard_matrix = np.array([
        [1.000, 0.455, 0.455, 0.385],
        [0.455, 1.000, 0.800, 0.500],
        [0.455, 0.800, 1.000, 0.636],
        [0.385, 0.500, 0.636, 1.000]
    ])

    cax = ax2.imshow(jaccard_matrix, cmap="YlGnBu", vmin=0.3, vmax=1.0)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax2.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax2.set_title("B: Candidate Selection Overlap (Jaccard Index, $K=9$)", fontsize=11, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = jaccard_matrix[i, j]
            color = "white" if val > 0.75 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)

    fig.colorbar(cax, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle("Figure 2: Selective Routing Efficiency & Global Policy Discrimination\n(Offline Response-Bank Replay against Qwen2.5-VL)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "corrected_fig2.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {p}")

# FIGURE 3: Authentic Native Factorial SLAM Integration (I0, I1, I2, I3)
def generate_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

    modes = [
        "I0: Baseline\nControl",
        "I1: Image Mask\nOnly",
        "I2: Landmark\nWeight Only",
        "I3: Dual Level\nIntegration"
    ]

    # Authentic physical metrics from step10_factorial_comparison.json
    ate_rmse = [0.3637, 0.3729, 0.4002, 0.3729]
    rpe_rot = [0.5311, 0.4793, 0.5315, 0.4793]

    x = np.arange(len(modes))
    width = 0.5

    # ATE plot
    colors = ["#1976d2", "#388e3c", "#d32f2f", "#00796b"]
    bars1 = ax1.bar(x, ate_rmse, width, color=colors, edgecolor="#212121", linewidth=1.2)
    ax1.set_ylabel("ATE RMSE (m)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A: Absolute Trajectory Error (Authentic Measured)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=9)
    ax1.set_ylim(0, 0.48)
    ax1.grid(axis="y")

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}m", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Add degradation annotation on I2
    ax1.annotate("+10.01% Error\n(Degradation)", xy=(2, 0.4002), xytext=(2, 0.435),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                 ha="center", fontsize=8, fontweight="bold", color="#b71c1c")

    # RPE Rotation plot
    bars2 = ax2.bar(x, rpe_rot, width, color=colors, edgecolor="#212121", linewidth=1.2)
    ax2.set_ylabel("RPE Rotation RMSE (deg)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B: Relative Rotational Drift (RPE)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes, fontsize=9)
    ax2.set_ylim(0, 0.65)
    ax2.grid(axis="y")

    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}°", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Annotate optical benefit on I1 and I3
    ax2.annotate("-9.75% Drift\nReduction", xy=(1, 0.4793), xytext=(1, 0.54),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
                 ha="center", fontsize=8, fontweight="bold", color="#1b5e20")

    plt.suptitle("Figure 3: Factorial Integration of Frontend Image Masking vs Backend Optimizer Weighting\nRevealing the Critical Negative Finding of Landmark Downweighting Alone", fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "corrected_fig3.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {p}")

# FIGURE 4: D455 Multi-Session Map Registration & Anchor Provenance
def generate_fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)

    # Left: Quarantined Analytical Simulation vs Actual Map Residuals
    categories = [
        "Quarantined D0\n(Hardcoded Sim)",
        "Quarantined D2\n(Hardcoded Sim)",
        "Verified ICP\nUnfiltered Desk",
        "Verified ICP\nStatic Structures"
    ]
    residuals = [0.5128, 0.3812, 0.0514, 0.0382]
    colors = ["#e57373", "#ffb74d", "#64b5f6", "#81c784"]
    hatches = ["//", "//", "", ""]

    bars = ax1.bar(categories, residuals, width=0.5, color=colors, edgecolor="#37474f", hatch=hatches, linewidth=1.1)
    ax1.set_ylabel("Map Alignment Residual RMSE (m)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A: Multi-Session Desk Registration Residuals", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.62)
    ax1.grid(axis="y")

    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}m", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.axvspan(-0.5, 1.5, color="#ffebee", alpha=0.5, zorder=0)
    ax1.text(0.5, 0.56, "QUARANTINED SIMULATION\n(Lookup in run_d455.py)", ha="center", va="center",
             fontsize=8, fontweight="bold", color="#c62828")

    ax1.axvspan(1.5, 3.5, color="#e8f5e9", alpha=0.5, zorder=0)
    ax1.text(2.5, 0.12, "VERIFIED EMPIRICAL ICP\n(Point Cloud Registration)", ha="center", va="center",
             fontsize=8, fontweight="bold", color="#2e7d32")

    # Right: Camera Trajectory MoCap Status Across Sequences
    datasets = ["TUM RGB-D", "OpenLORIS", "Bonn Dynamic", "RealSense D455"]
    mocap_avail = [1.0, 1.0, 1.0, 0.0]
    bar_cols = ["#2b5c8f", "#2b5c8f", "#2b5c8f", "#d32f2f"]

    ax2.bar(datasets, mocap_avail, width=0.45, color=bar_cols, edgecolor="#212121", linewidth=1.1)
    ax2.set_ylabel("MoCap Ground Truth Availability", fontsize=10.5, fontweight="bold")
    ax2.set_title("B: Ground Truth MoCap Protocol Governance", fontsize=11, fontweight="bold")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["null\n(Unconstrained)", "1.0\n(Verified MoCap)"], fontsize=9, fontweight="bold")
    ax2.grid(axis="y")

    ax2.text(3, 0.2, "Camera ATE = null\nStrictly Enforced\n(No Fake MoCap)", ha="center", va="center",
             fontsize=8.5, fontweight="bold", color="#b71c1c", bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#ef5350"))

    plt.suptitle("Figure 4: RealSense D455 Anchor Provenance & Research Governance Status", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "corrected_fig4.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {p}")

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    print("All 4 corrected figures generated successfully.")

