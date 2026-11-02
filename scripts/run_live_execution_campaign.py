"""Live execution pipeline runner.

Coordinates end-to-end evaluation:
- Policy-selected visual-language inference on union queries (168 observations).
- Multi-view visual-language evaluation across development tracks (30 tracks x 4 configurations).
- Decision and evidence fusion under policy isolation.
- Paired statistical hypothesis testing and component ablation analysis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline_stage(stage_description: str, cmd: list[str]) -> None:
    print(f"\n[RUNNING] {stage_description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[FAILED] {stage_description} exited with code {res.returncode} after {elapsed:.1f}s\n", flush=True)
        sys.exit(res.returncode)
    print(f"\n[COMPLETED] {stage_description} in {elapsed:.1f}s\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Visual-Language and SLAM Evaluation Pipeline")
    parser.add_argument("--skip-policy-inference", action="store_true", help="Skip policy-selected inference")
    parser.add_argument("--skip-controlled-eval", action="store_true", help="Skip controlled evaluation")
    parser.add_argument("--limit-queries", type=int, default=0, help="Limit number of policy queries (0=all)")
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. Policy-selected visual-language inference on union queries
    if not args.skip_policy_inference:
        cmd_inference = [python_exe, "scripts/run_policy_selected_vlm_inference.py"]
        if args.limit_queries > 0:
            cmd_inference.extend(["--limit", str(args.limit_queries)])
        run_pipeline_stage("Policy-Selected Visual-Language Inference (168 observations)", cmd_inference)

    # 2. Controlled multi-view evaluation across development tracks
    if not args.skip_controlled_eval:
        cmd_controlled = [
            python_exe, "scripts/run_controlled_vlm_evaluation.py",
            "--split", "development",
            "--configs", "V1", "V2", "V3", "V4",
        ]
        run_pipeline_stage("Controlled Multi-View Visual-Language Evaluation (30 tracks x 4 configurations)", cmd_controlled)

    # 3. Decision and evidence fusion consuming live records
    cmd_fusion = [
        python_exe, "scripts/run_decision_evidence_fusion.py",
        "--vlm-records-file", "results/policy_vlm_inference/live_policy_vlm_records.jsonl",
        "--prompt-version", "V4",
    ]
    run_pipeline_stage("Decision and Evidence Fusion", cmd_fusion)

    # 4. Statistical significance testing and component ablations
    cmd_stats = [python_exe, "scripts/run_statistical_significance_and_ablation.py"]
    run_pipeline_stage("Statistical Hypothesis Testing and Component Ablation", cmd_stats)

    print("\nLive execution pipeline completed successfully.\n")


if __name__ == "__main__":
    main()

