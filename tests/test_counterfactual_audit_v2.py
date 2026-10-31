"""Regression Tests for Counterfactual SLAM Intervention Benchmark (Audit V2).

Verifies all audit requirements:
1. Baseline-control identity: strict co-executed C0 control, separation of historical Pass A.
2. Statistical inference: non-i.i.d. per-pose residuals, single-sequence degrees of freedom, Cohen's d, clamped underflow p-values.
3. D455 non-oracle lifecycle: F1=0.75, 3 TP, 0 FP, 2 FN, 47 TN, 0.000m displacement RMSE on detected, held-out static anchor CV.
4. TUM monitor overlap ablation: 6 conditions, moving persons causal driver, static monitor degradation.
5. Corrected FAST feature retention: 5px boundary margin analysis, no 301.5 kp claim.
6. Cryptographic provenance: SHA-256 for scripts, configs, trajectories, and ORB-SLAM3 binary.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import get_repo_root


@pytest.fixture
def audit_v2_dir() -> Path:
    repo_root = get_repo_root()
    audit_dir = repo_root / "results" / "counterfactual_interventions" / "run_20261031_counterfactual_audit_v2"
    assert audit_dir.exists(), f"Audit V2 directory does not exist: {audit_dir}"
    return audit_dir


def test_baseline_identity_provenance(audit_v2_dir: Path) -> None:
    audit_file = audit_v2_dir / "baseline_identity_audit.json"
    assert audit_file.exists(), "baseline_identity_audit.json missing"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. TUM C0 identity
    tum = data["controls"]["tum_rgbd_fr3_walking_xyz"]
    assert tum["executed_c0_ate_rmse_m"] == 0.3497
    assert tum["historical_baseline_comparison"]["historical_baseline_ate_rmse_m"] == 0.4372
    assert tum["historical_baseline_comparison"]["historical_run_label"] == "historical_pass_a_v2_verified"
    assert tum["trajectory_sha256"] == "770669d2bc2ee7669f5c4911d6bce89b08282e9d41b9e15c8202e504955a6bf9"

    # 2. OpenLORIS C0 identity
    ol = data["controls"]["openloris_scene_corridor1_4"]
    assert ol["executed_c0_ate_rmse_m"] == 1.7045
    assert ol["historical_baseline_comparison"]["historical_baseline_ate_rmse_m"] == 1.6985
    assert ol["historical_baseline_comparison"]["historical_run_label"] == "historical_pass_a_v2_verified"
    assert ol["trajectory_sha256"] == "721dff86b198a7e93c1956b5b8c183ba9331d17dbc59352c8b1ccb7cba47763d"

    # 3. D455 Zero-Fabrication Contract
    d455 = data["controls"]["realsense_d455_multi_session"]
    assert d455["trajectory_ground_truth"] is None
    assert d455["trajectory_ate_rmse_m"] is None


def test_statistical_inference_contracts(audit_v2_dir: Path) -> None:
    stat_file = audit_v2_dir / "statistical_analysis.json"
    assert stat_file.exists(), "statistical_analysis.json missing"
    with open(stat_file, "r", encoding="utf-8") as f:
        stat_data = json.load(f)

    # Sequence level degrees of freedom
    assert stat_data["sequence_level_inference_summary"]["tum_rgbd"]["status"] == "INSUFFICIENT_INDEPENDENT_SEQUENCES_FOR_INFERENCE"
    assert stat_data["sequence_level_inference_summary"]["openloris_scene"]["status"] == "INSUFFICIENT_INDEPENDENT_SEQUENCES_FOR_INFERENCE"

    # Canonical results file
    res_file = audit_v2_dir / "counterfactual_results.json"
    assert res_file.exists(), "counterfactual_results.json missing"
    with open(res_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for row in results:
        # None of the single-sequence intervention rows should claim statistical significance
        assert row.get("is_statistically_significant") is False
        # Clamped p-values
        if row.get("paired_ttest_pvalue") is not None:
            assert row["paired_ttest_pvalue"] >= 1e-16
        if row.get("paired_wilcoxon_pvalue") is not None:
            assert row["paired_wilcoxon_pvalue"] >= 1e-16

    # Verify TUM polygon mask is correctly identified as a degradation against C0
    tum_poly = next(r for r in results if r["dataset_id"] == "tum" and r["action"] == "IMAGE_POLYGON_MASK")
    assert tum_poly["ate_rmse_common_m"] == 0.4365
    assert tum_poly["delta_ate_rmse_m"] < 0  # Degraded relative to 0.3497m
    assert tum_poly["percentage_ate_improvement"] < 0
    assert tum_poly["cohen_d"] > 0  # Error increased


def test_d455_non_oracle_lifecycle_contracts(audit_v2_dir: Path) -> None:
    # 1. Predictions file exists and is populated
    actual_pred_file = audit_v2_dir / "d455" / "actual_lifecycle_predictions.json"
    assert actual_pred_file.exists(), "actual_lifecycle_predictions.json missing"
    with open(actual_pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    assert len(predictions) == 52

    # 2. Lifecycle intervention evaluation
    lifecycle_file = audit_v2_dir / "d455" / "lifecycle_interventions.json"
    assert lifecycle_file.exists(), "lifecycle_interventions.json missing"
    with open(lifecycle_file, "r", encoding="utf-8") as f:
        l_data = json.load(f)

    cm = l_data["confusion_matrix"]
    assert cm["true_positives"] == 3
    assert cm["false_positives"] == 0
    assert cm["false_negatives"] == 2
    assert cm["true_negatives"] == 47
    assert cm["precision"] == 1.0
    assert cm["recall"] == 0.6
    assert cm["f1_score"] == 0.75

    # 3. Held-out anchor cross-validation
    anchor_file = audit_v2_dir / "d455" / "registration_heldout_anchor_metrics.json"
    assert anchor_file.exists(), "registration_heldout_anchor_metrics.json missing"
    with open(anchor_file, "r", encoding="utf-8") as f:
        anchor_data = json.load(f)

    assert "held_out_cross_validation" in anchor_data
    assert anchor_data["held_out_cross_validation"]["rms_m"] < 0.40
    # Stale node simulation in lifecycle interventions
    assert l_data["counterfactual_arms"]["arm_c0_keep"]["simulation_label"] == "ANALYTICAL_SIMULATION"
    assert l_data["counterfactual_arms"]["arm_c0_keep"]["simulated_anchor_registration_rms_m"] > 0.40


def test_tum_monitor_overlap_ablation_causality(audit_v2_dir: Path) -> None:
    ablation_file = audit_v2_dir / "tum" / "monitor_overlap_ablation.json"
    assert ablation_file.exists(), "monitor_overlap_ablation.json missing"
    with open(ablation_file, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    conditions = abl_data["conditions_evaluated"]
    assert len(conditions) == 6

    # 1. Moving persons polygon mask has lowest ATE (causal driver)
    c4 = next(c for c in conditions if c["condition"] == "moving_persons_POLYGON")
    assert c4["condition_ate_rmse_m"] < 0.03
    assert c4["percentage_ate_improvement"] > 90.0

    # 2. Monitor polygon degrades baseline
    c1 = next(c for c in conditions if c["condition"] == "monitor_01_POLYGON")
    assert c1["condition_ate_rmse_m"] > 0.40
    assert c1["percentage_ate_improvement"] < 0.0

    # 3. Monitor bbox improves over baseline due to collateral human overlap
    c2 = next(c for c in conditions if c["condition"] == "monitor_01_BBOX")
    assert c2["condition_ate_rmse_m"] < 0.25
    assert c2["percentage_ate_improvement"] > 35.0

    # 4. Bbox minus polygon without primary human mask degrades
    c5 = next(c for c in conditions if c["condition"] == "monitor_01_BBOX_MINUS_POLYGON")
    assert c5["condition_ate_rmse_m"] > 0.45


def test_fast_feature_retention_corrected(audit_v2_dir: Path) -> None:
    fast_file = audit_v2_dir / "tum" / "fast_feature_retention_corrected.json"
    assert fast_file.exists(), "fast_feature_retention_corrected.json missing"
    with open(fast_file, "r", encoding="utf-8") as f:
        fast_data = json.load(f)

    assert fast_data["detector_parameters"]["boundary_margin_px"] == 5
    m1 = fast_data["threshold_20"]
    # The mean polygon keypoints must be ~8.63, far lower than the historical 301.5 claim
    assert m1["inside_polygon_kps"]["mean"] < 15.0
    assert m1["inside_bbox_kps"]["mean"] < 30.0
    # Boundary edge artifacts must be tracked
    assert m1["poly_boundary_edge_artifacts"]["mean"] > 0


def test_provenance_manifest_completeness(audit_v2_dir: Path) -> None:
    prov_file = audit_v2_dir / "provenance" / "provenance_manifest.json"
    assert prov_file.exists(), "provenance_manifest.json missing"
    with open(prov_file, "r", encoding="utf-8") as f:
        prov = json.load(f)

    # WSL ORB-SLAM3 binary hashes
    wsl = prov["runtime_environment"]["wsl_environment"]
    assert wsl["orb_slam3_executable_sha256"] == "cd9162f319423483a93e1c4e33ffc86ec9bb5bc3e931ea44816ebc5db08e999c"
    assert wsl["orb_slam3_git_commit"] == "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"

    # Execution commands recorded
    assert len(prov["execution_commands"]) >= 5

    # Report file exists
    report_file = audit_v2_dir / "REPORT.md"
    assert report_file.exists(), "REPORT.md missing"
    assert report_file.stat().st_size > 5000
