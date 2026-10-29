"""Unit tests for Counterfactual Utility Calibration (Stage 5)."""

import numpy as np
import pytest

from estimator_aware_vlm.calibration.counterfactual import (
    CalibratedUtilityModel,
    CounterfactualRecord,
    extract_track_features,
    fit_calibrated_utility_model,
)


@pytest.fixture
def sample_records():
    base_ate = 0.40
    return [
        CounterfactualRecord("p1", "person", 200, 8.0, 0.90, 0.25, 1.8, 1.2, base_ate, 0.08, 0.32),
        CounterfactualRecord("p2", "person", 180, 7.0, 0.80, 0.22, 1.6, 1.1, base_ate, 0.10, 0.30),
        CounterfactualRecord("p3", "person", 140, 5.0, 0.60, 0.18, 1.4, 1.0, base_ate, 0.20, 0.20),
        CounterfactualRecord("c1", "chair", 90, 3.0, 0.15, 0.06, 0.4, 0.8, base_ate, 0.38, 0.02),
        CounterfactualRecord("m1", "monitor", 210, 7.5, 0.02, 0.01, 0.15, 1.2, base_ate, 0.42, -0.02),
        CounterfactualRecord("m2", "monitor", 190, 6.8, 0.02, 0.01, 0.16, 1.1, base_ate, 0.41, -0.01),
    ]


def test_extract_track_features():
    track_data = {
        "measured_displacement_m": 0.85,
        "displacement_uncertainty_m": 0.20,
        "observation_count": 150,
        "camera_ego_motion_displacement_m": 1.10,
        "lifespan_sec": 6.0,
    }
    feats = extract_track_features(track_data)
    assert len(feats) == 6
    assert feats[0] == 0.85
    assert feats[1] == 0.20
    assert feats[2] == 1.0  # capped at 1.0
    assert feats[3] == 1.10
    assert feats[4] == 0.60  # 6.0 / 10.0


def test_fit_calibrated_utility_model_loocv(sample_records):
    model = fit_calibrated_utility_model(sample_records, lambda_cost=0.45, ridge_alpha=0.1)
    assert len(model.weights) == 6
    assert model.loocv_mae < 0.08
    assert model.loocv_r2 > 0.70


def test_calibrated_utility_prediction(sample_records):
    model = fit_calibrated_utility_model(sample_records, lambda_cost=0.45, ridge_alpha=0.1)

    # Dynamic walking human features
    human_feats = extract_track_features({
        "measured_displacement_m": 0.95,
        "displacement_uncertainty_m": 0.28,
        "observation_count": 220,
        "camera_ego_motion_displacement_m": 1.25,
        "lifespan_sec": 8.0,
    })
    pred_human = model.predict_delta_L(human_feats)
    assert pred_human > 0.20

    # Static desk monitor features
    monitor_feats = extract_track_features({
        "measured_displacement_m": 0.01,
        "displacement_uncertainty_m": 0.01,
        "observation_count": 200,
        "camera_ego_motion_displacement_m": 1.20,
        "lifespan_sec": 7.5,
    })
    pred_monitor = model.predict_delta_L(monitor_feats)
    assert pred_monitor < 0.05


def test_should_query_vlm_decision_rule(sample_records):
    model = fit_calibrated_utility_model(sample_records, lambda_cost=0.40, c_vlm=0.30)

    # High benefit dynamic target: delta_L ~ 0.30 > lambda * C_vlm (0.12) -> Query VLM
    human_feats = extract_track_features({
        "measured_displacement_m": 0.90,
        "displacement_uncertainty_m": 0.25,
        "observation_count": 200,
        "camera_ego_motion_displacement_m": 1.20,
        "lifespan_sec": 7.0,
    })
    assert model.should_query_vlm(human_feats, utility_threshold=0.0) is True

    # Static target: delta_L ~ 0.0 < lambda * C_vlm (0.12) -> Do not query VLM
    monitor_feats = extract_track_features({
        "measured_displacement_m": 0.01,
        "displacement_uncertainty_m": 0.01,
        "observation_count": 200,
        "camera_ego_motion_displacement_m": 1.20,
        "lifespan_sec": 7.0,
    })
    assert model.should_query_vlm(monitor_feats, utility_threshold=0.0) is False


def test_model_serialization_and_deserialization(sample_records, tmp_path):
    model = fit_calibrated_utility_model(sample_records)
    save_path = tmp_path / "model.json"
    model.save_json(save_path)

    loaded = CalibratedUtilityModel.load_json(save_path)
    assert loaded.feature_names == model.feature_names
    assert np.allclose(loaded.weights, model.weights, atol=1e-5)
    assert abs(loaded.bias - model.bias) < 1e-5
    assert abs(loaded.loocv_mae - model.loocv_mae) < 1e-5


def test_fit_fails_on_insufficient_records():
    with pytest.raises(ValueError, match="At least 3 counterfactual records required"):
        fit_calibrated_utility_model([])

