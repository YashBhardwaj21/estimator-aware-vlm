"""Evaluate Scene-Graph Lifecycle Interventions on Intel RealSense D455 Multi-Session Dataset.

Evaluates counterfactual scene-graph actions across 52 verified physical entities:
- C0: KEEP (Stale landmark retention without split)
- C3: GRAPH_MARK_SUSPECT (Uncertainty inflation; exclude from rigid anchor alignment)
- C4: GRAPH_TEMPORARY_EXCLUDE (Removal from active reference map during exploration)
- RELOCATE_SPLIT: Bifurcate entity into (prior, current) with displacement edge
- REACTIVATE: Re-anchor at confirmed current pose with refreshed spatial covariance

Quantifies:
1. Object displacement error and map distortion penalty (meters).
2. Held-out static anchor registration residual (meters).
3. Relocation detection accuracy (Precision, Recall, F1).
4. Physical identity continuity vs. erroneous merge rate.
"""

import argparse
from pathlib import Path
import sys

# Inject src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.interventions.lifecycle import evaluate_lifecycle_interventions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D455 Scene-Graph Lifecycle Interventions.")
    parser.add_argument("--run-id", type=str, default="run_20261031_counterfactual", help="Run identifier.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_root / "results" / "counterfactual_interventions" / args.run_id / "d455"
    )

    gt_path = repo_root / "benchmarks" / "movability_ground_truth.json"
    reg_audit_path = repo_root / "outputs" / "reports" / "registration_audit.json"

    evaluate_lifecycle_interventions(
        ground_truth_path=gt_path,
        registration_audit_path=reg_audit_path,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
