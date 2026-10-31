"""Generate Publication-Quality Visualizations for Counterfactual SLAM Benchmark."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from estimator_aware_vlm.common.paths import get_repo_root


def main() -> None:
    repo_root = get_repo_root()
    base_dir = repo_root / "results" / "counterfactual_interventions" / "run_20261031_counterfactual"
    fig_dir = base_dir / "aggregate" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Trajectory ATE Error Reduction across Counterfactual Arms
    labels = [
        "TUM: Polygon Mask\n(monitor_01)",
        "TUM: BBox Mask\n(monitor_01)",
        "OpenLORIS: Poly Mask\n(person_01)",
        "OpenLORIS: BBox Mask\n(person_01)",
    ]
    pct_changes = [0.16, 52.73, -5.20, -5.11]
    p_values = ["p=0.44 (ns)", "p=2.76e-115 (***)", "p=0.00 (***)", "p=4.55e-209 (***)"]
    colors = ["#95a5a6", "#2ecc71", "#e74c3c", "#e67e22"]

    bars = ax1.bar(labels, pct_changes, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("ATE RMSE Improvement (%) [Positive = Better]", fontsize=12, fontweight="bold")
    ax1.set_title("Downstream Localization Impact of Image-Space Interventions", fontsize=13, fontweight="bold")
    ax1.set_ylim(-15, 65)

    for bar, val, pval in zip(bars, pct_changes, p_values):
        ypos = val + 2.5 if val >= 0 else val - 4.5
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.1f}%\n{pval}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 2: D455 3D Scene-Graph Anchor Integrity
    d455_labels = [
        "Verified Inlier Floor\n(Noise Baseline)",
        "C0: KEEP\n(Stale Node Retention)",
        "C3: MARK_SUSPECT\n(10x Covariance Inflation)",
        "C4: TEMP_EXCLUDE\n(Mask from Active Map)",
        "RELOCATE_SPLIT\n(Bifurcate Node + d)",
    ]
    anchor_rms_cm = [11.98, 51.28, 11.98, 11.98, 11.98]
    d455_colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#27ae60"]

    bars2 = ax2.bar(d455_labels, anchor_rms_cm, color=d455_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Static Anchor RMS Residual (cm) [Lower = Better]", fontsize=12, fontweight="bold")
    ax2.set_title("D455 Multi-Session Map Integrity across Lifecycle Actions", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 60)

    for bar, val in zip(bars2, anchor_rms_cm):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.0,
            f"{val:.1f} cm",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.axhline(11.98, color="#2980b9", linestyle=":", linewidth=1.5, label="Verified Inlier Floor (11.98 cm)")
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = fig_dir / "counterfactual_benchmark_synthesis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated synthesis plot at: {out_path}")


if __name__ == "__main__":
    main()

