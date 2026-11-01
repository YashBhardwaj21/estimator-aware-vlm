"""Unit tests for EDRR formulation, loss normalization, and sequence-level grouped validation."""

import numpy as np
import pytest

from estimator_aware_vlm.edrr import (
    CandidateQueryProfile,
    DownstreamSLAMLoss,
    EDRRDecision,
    InterventionDataset,
    SingleCandidateInterventionRecord,
    SLAMLossWeights,
    check_data_sufficiency,
    compute_candidate_edrr,
    compute_normalized_slam_loss,
    run_sequence_level_grouped_validation,
    select_top_k_edrr,
)


def test_slam_loss_weights_normalization():
    # Valid weights summing to 1.0
    w = SLAMLossWeights(alpha_pose=0.4, alpha_tracking=0.3, alpha_map=0.2, alpha_feature_starvation=0.1)
    assert w.alpha_pose == 0.4

    # Invalid weights not summing to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SLAMLossWeights(alpha_pose=0.5, alpha_tracking=0.5, alpha_map=0.2, alpha_feature_starvation=0.1)

    # Negative weight
    with pytest.raises(ValueError, match="must be non-negative"):
        SLAMLossWeights(alpha_pose=1.2, alpha_tracking=-0.2, alpha_map=0.0, alpha_feature_starvation=0.0)


def test_compute_normalized_slam_loss():
    loss = compute_normalized_slam_loss(
        ate_rmse_m=0.20,
        tracking_lost_ratio=0.05,
        map_anchor_residual_m=0.10,
        min_feature_count=120,
        nominal_feature_count=150,
        max_tolerable_ate_m=2.0,
        max_tolerable_residual_m=1.0,
    )
    assert 0.0 <= loss.l_pose <= 1.0
    assert 0.0 <= loss.l_tracking <= 1.0
    assert 0.0 <= loss.l_map <= 1.0
    assert 0.0 <= loss.l_feature_starvation <= 1.0
    assert 0.0 <= loss.total_loss <= 1.0
    assert loss.l_pose == pytest.approx(0.10)  # 0.20 / 2.0


def test_edrr_formulation_ranking_and_budget():
    p1 = CandidateQueryProfile(
        candidate_id="c1",
        sequence_id="s1",
        semantic_class="person",
        measured_displacement_m=1.2,
        displacement_uncertainty_m=0.2,
        observation_count=200,
        camera_ego_motion_displacement_m=1.0,
        lifespan_sec=8.0,
        local_keypoint_count=160,
    )
    p2 = CandidateQueryProfile(
        candidate_id="c2",
        sequence_id="s1",
        semantic_class="chair",
        measured_displacement_m=0.1,
        displacement_uncertainty_m=0.05,
        observation_count=100,
        camera_ego_motion_displacement_m=1.0,
        lifespan_sec=5.0,
        local_keypoint_count=140,
    )
    p3 = CandidateQueryProfile(
        candidate_id="c3",
        sequence_id="s1",
        semantic_class="monitor",
        measured_displacement_m=0.01,
        displacement_uncertainty_m=0.01,
        observation_count=150,
        camera_ego_motion_displacement_m=1.0,
        lifespan_sec=7.0,
        local_keypoint_count=150,
    )

    candidates = [
        (p1, 0.45),  # High benefit
        (p2, 0.15),  # Modest benefit
        (p3, -0.05), # Negative / harmful
    ]

    # Budget K=2 with lambda_cost=0.20 (query cost = 0.50 -> threshold cost = 0.10)
    decisions = select_top_k_edrr(candidates, budget_k=2, lambda_cost=0.20)
    assert len(decisions) == 3

    # Rank 1: c1 (net score = 0.45 - 0.10 = 0.35 > 0)
    assert decisions[0].candidate_id == "c1"
    assert decisions[0].should_query is True
    assert decisions[0].rank == 1

    # Rank 2: c2 (net score = 0.15 - 0.10 = 0.05 > 0)
    assert decisions[1].candidate_id == "c2"
    assert decisions[1].should_query is True
    assert decisions[1].rank == 2

    # Rank 3: c3 (net score = -0.05 - 0.10 = -0.15 <= 0)
    assert decisions[2].candidate_id == "c3"
    assert decisions[2].should_query is False


def test_intervention_dataset_attribution_and_safeguards():
    ds = InterventionDataset()

    rec = SingleCandidateInterventionRecord(
        intervention_id="tum_fr3_walking_xyz_person_01_IMAGE_POLYGON_MASK",
        sequence_id="tum_fr3_walking_xyz",
        target_id="person_01",
        semantic_class="person",
        action_type="IMAGE_POLYGON_MASK",
        measured_displacement_m=1.2,
        displacement_uncertainty_m=0.2,
        observation_count=200,
        lifespan_sec=8.0,
        camera_ego_motion_displacement_m=1.0,
        local_keypoint_count=150,
        baseline_ate_rmse_m=0.3952,
        intervention_ate_rmse_m=0.0385,
        empirical_delta_ate_m=0.3567,
        common_support_poses=700,
    )
    ds.add_record(rec)
    assert len(ds) == 1

    # Rejection of joint interventions
    joint_rec = SingleCandidateInterventionRecord(
        intervention_id="tum_fr3_walking_xyz_joint_01",
        sequence_id="tum_fr3_walking_xyz",
        target_id="person_01_and_person_02",
        semantic_class="person",
        action_type="JOINT_MASK",
        measured_displacement_m=1.2,
        displacement_uncertainty_m=0.2,
        observation_count=200,
        lifespan_sec=8.0,
        camera_ego_motion_displacement_m=1.0,
        local_keypoint_count=150,
        baseline_ate_rmse_m=0.3952,
        intervention_ate_rmse_m=0.0385,
        empirical_delta_ate_m=0.3567,
        common_support_poses=700,
        is_joint_intervention=True,
    )
    with pytest.raises(ValueError, match="cannot be added as single-candidate"):
        ds.add_record(joint_rec)

    # Rejection of D455 with non-null trajectory ATE
    d455_invalid = SingleCandidateInterventionRecord(
        intervention_id="d455_recording_01_chair_002",
        sequence_id="realsense_d455_recording_01",
        target_id="chair_002",
        semantic_class="chair",
        action_type="OBJECT_RELOCATED",
        measured_displacement_m=0.56,
        displacement_uncertainty_m=0.02,
        observation_count=100,
        lifespan_sec=10.0,
        camera_ego_motion_displacement_m=0.8,
        local_keypoint_count=120,
        baseline_ate_rmse_m=0.15,  # Prohibited!
        intervention_ate_rmse_m=0.10,
        empirical_delta_ate_m=0.05,
        common_support_poses=500,
    )
    with pytest.raises(ValueError, match="D455 recordings lack external MoCap ground-truth"):
        ds.add_record(d455_invalid)


def test_data_sufficiency_gate_insufficient():
    ds = InterventionDataset()
    # Add 4 records across 2 sequences (below N=30, sequences=5)
    for i in range(4):
        seq = "seq_a" if i < 2 else "seq_b"
        ds.add_record(
            SingleCandidateInterventionRecord(
                intervention_id=f"{seq}_target_{i}",
                sequence_id=seq,
                target_id=f"target_{i}",
                semantic_class="person",
                action_type="IMAGE_POLYGON_MASK",
                measured_displacement_m=0.8 + 0.1 * i,
                displacement_uncertainty_m=0.1,
                observation_count=100,
                lifespan_sec=5.0,
                camera_ego_motion_displacement_m=1.0,
                local_keypoint_count=150,
                baseline_ate_rmse_m=0.40,
                intervention_ate_rmse_m=0.10,
                empirical_delta_ate_m=0.30,
                common_support_poses=500,
            )
        )

    is_suff, status = check_data_sufficiency(ds)
    assert is_suff is False
    assert "NOT_TRAINED_INSUFFICIENT_DATA (N=4 < 30" in status

    # Run grouped validation with gate enforced
    res = run_sequence_level_grouped_validation(ds, enforce_sufficiency_gate=True)
    assert res.status.startswith("NOT_TRAINED_INSUFFICIENT_DATA")
    assert res.oof_mae is None
    assert "strictly prohibited" in res.rejection_reason


def test_grouped_validation_fold_isolated_normalization_synthetic_sufficient():
    # Construct synthetic sufficient dataset: 35 samples across 5 sequences
    ds = InterventionDataset()
    for s_idx in range(5):
        seq_name = f"sequence_{s_idx:02d}"
        for c_idx in range(7):
            t_id = f"target_{s_idx}_{c_idx}"
            disp = 0.5 + 0.1 * c_idx
            delta_ate = 0.25 * disp
            ds.add_record(
                SingleCandidateInterventionRecord(
                    intervention_id=f"{seq_name}_{t_id}",
                    sequence_id=seq_name,
                    target_id=t_id,
                    semantic_class="person",
                    action_type="IMAGE_POLYGON_MASK",
                    measured_displacement_m=disp,
                    displacement_uncertainty_m=0.05,
                    observation_count=120,
                    lifespan_sec=6.0,
                    camera_ego_motion_displacement_m=1.0,
                    local_keypoint_count=150,
                    baseline_ate_rmse_m=0.50,
                    intervention_ate_rmse_m=0.50 - delta_ate,
                    empirical_delta_ate_m=delta_ate,
                    common_support_poses=600,
                )
            )

    is_suff, status = check_data_sufficiency(ds)
    assert is_suff is True
    assert status == "DATA_SUFFICIENT"

    res = run_sequence_level_grouped_validation(ds, enforce_sufficiency_gate=True)
    assert res.status == "VALIDATED_GROUPED_CV"
    assert res.n_samples == 35
    assert res.n_sequences == 5
    assert res.oof_mae is not None
    assert res.oof_mae < 0.05
    assert len(res.fold_metrics) == 5

