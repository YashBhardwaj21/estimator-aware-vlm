"""Tests for strict feature causality, prefix computation, and ground-truth isolation."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _compute_prefix_features(obs_list: List[Dict[str, Any]], target_idx: int) -> Dict[str, Any]:
    """Helper mirroring the prefix feature computation in build_estimator_context_features."""
    target_obs = obs_list[target_idx]
    prefix_obs = obs_list[: target_idx + 1]
    first_ts = obs_list[0]["timestamp"]

    causal_elapsed = round(float(target_obs["timestamp"] - first_ts), 4)
    causal_count = target_idx + 1

    depth = target_obs.get("median_depth_m", 0.0)
    wld_c = target_obs.get("centroid_world") or [0.0, 0.0, 0.0]
    ego_available = bool(target_obs.get("ego_motion_available", False))
    world_valid = bool(
        ego_available
        and wld_c
        and len(wld_c) == 3
        and any(abs(v) > 1e-6 for v in wld_c)
        and (0.0 < float(depth) < 10.0)
    )

    valid_prefix_wld = [
        o.get("centroid_world")
        for o in prefix_obs
        if o.get("ego_motion_available", False)
        and o.get("centroid_world")
        and len(o.get("centroid_world")) == 3
        and any(abs(v) > 1e-6 for v in o.get("centroid_world"))
        and (0.0 < float(o.get("median_depth_m", 0.0)) < 10.0)
    ]

    if world_valid and valid_prefix_wld:
        wld_0 = valid_prefix_wld[0]
        causal_world_disp = round(float(math.sqrt(sum((a - b)**2 for a, b in zip(wld_c, wld_0)))), 4)

        path_len = 0.0
        for k in range(1, len(valid_prefix_wld)):
            step = math.sqrt(sum((a - b)**2 for a, b in zip(valid_prefix_wld[k], valid_prefix_wld[k-1])))
            path_len += step
        causal_path_len = round(float(path_len), 4)

        n_pts = len(valid_prefix_wld)
        mean_c = [sum(c[d] for c in valid_prefix_wld) / n_pts for d in range(3)]
        variance = sum(sum((c[d] - mean_c[d])**2 for d in range(3)) for c in valid_prefix_wld) / n_pts
        causal_dispersion = round(float(math.sqrt(max(0.0, variance))), 4)
    else:
        causal_world_disp = None
        causal_path_len = None
        causal_dispersion = None

    return {
        "causal_elapsed_sec": causal_elapsed,
        "causal_observation_count": causal_count,
        "causal_world_displacement_m": causal_world_disp,
        "causal_path_length_m": causal_path_len,
        "causal_centroid_dispersion_m": causal_dispersion,
        "world_position_valid": world_valid,
    }


def test_future_truncation_invariance():
    """Features at timestamp t_i must be identical whether future observations exist or are truncated."""
    track_obs = [
        {"timestamp": 10.0, "median_depth_m": 2.0, "ego_motion_available": True, "centroid_world": [1.0, 0.0, 2.0]},
        {"timestamp": 12.0, "median_depth_m": 2.1, "ego_motion_available": True, "centroid_world": [1.2, 0.1, 2.0]},
        {"timestamp": 15.0, "median_depth_m": 2.2, "ego_motion_available": True, "centroid_world": [1.5, 0.2, 2.1]},
        {"timestamp": 20.0, "median_depth_m": 2.5, "ego_motion_available": True, "centroid_world": [2.5, 1.0, 2.5]},
    ]
    # Evaluate at index 1 with full track
    feats_full = _compute_prefix_features(track_obs, 1)

    # Evaluate at index 1 with future truncated
    feats_truncated = _compute_prefix_features(track_obs[:2], 1)

    assert feats_full == feats_truncated, "Feature vector at t_i must not depend on future observations"


def test_future_mutation_invariance():
    """Mutating or appending future observations must have zero effect on features at t_i."""
    track_obs_1 = [
        {"timestamp": 10.0, "median_depth_m": 2.0, "ego_motion_available": True, "centroid_world": [1.0, 0.0, 2.0]},
        {"timestamp": 12.0, "median_depth_m": 2.1, "ego_motion_available": True, "centroid_world": [1.2, 0.1, 2.0]},
        {"timestamp": 15.0, "median_depth_m": 2.2, "ego_motion_available": True, "centroid_world": [1.5, 0.2, 2.1]},
    ]
    track_obs_2 = [
        {"timestamp": 10.0, "median_depth_m": 2.0, "ego_motion_available": True, "centroid_world": [1.0, 0.0, 2.0]},
        {"timestamp": 12.0, "median_depth_m": 2.1, "ego_motion_available": True, "centroid_world": [1.2, 0.1, 2.0]},
        # Drastic future displacement added
        {"timestamp": 50.0, "median_depth_m": 5.0, "ego_motion_available": True, "centroid_world": [50.0, 50.0, 50.0]},
        {"timestamp": 60.0, "median_depth_m": 6.0, "ego_motion_available": True, "centroid_world": [100.0, 100.0, 100.0]},
    ]
    feats_1 = _compute_prefix_features(track_obs_1, 1)
    feats_2 = _compute_prefix_features(track_obs_2, 1)

    assert feats_1 == feats_2, "Features at index 1 must be invariant to future trajectory changes"


def test_prefix_count_and_elapsed_time():
    """Causal observation count must equal prefix length and elapsed time must measure from first detection."""
    track_obs = [
        {"timestamp": 100.0, "median_depth_m": 1.5, "ego_motion_available": True, "centroid_world": [0.0, 0.0, 1.5]},
        {"timestamp": 103.5, "median_depth_m": 1.5, "ego_motion_available": True, "centroid_world": [0.1, 0.0, 1.5]},
        {"timestamp": 107.0, "median_depth_m": 1.5, "ego_motion_available": True, "centroid_world": [0.2, 0.0, 1.5]},
    ]
    f0 = _compute_prefix_features(track_obs, 0)
    assert f0["causal_observation_count"] == 1
    assert f0["causal_elapsed_sec"] == 0.0

    f1 = _compute_prefix_features(track_obs, 1)
    assert f1["causal_observation_count"] == 2
    assert f1["causal_elapsed_sec"] == 3.5

    f2 = _compute_prefix_features(track_obs, 2)
    assert f2["causal_observation_count"] == 3
    assert f2["causal_elapsed_sec"] == 7.0


def test_missing_pose_produces_null_not_zero():
    """When ego-motion or 3D world pose is missing, displacement must be null, not 0.0."""
    track_obs = [
        # Missing ego motion (e.g. handheld sequence without MoCap/odometry)
        {"timestamp": 10.0, "median_depth_m": 2.0, "ego_motion_available": False, "centroid_world": [0.0, 0.0, 0.0]},
        {"timestamp": 12.0, "median_depth_m": 2.0, "ego_motion_available": False, "centroid_world": [0.0, 0.0, 0.0]},
    ]
    feats = _compute_prefix_features(track_obs, 1)
    assert feats["world_position_valid"] is False
    assert feats["causal_world_displacement_m"] is None, "Missing world position must NOT be substituted with 0.0"
    assert feats["causal_path_length_m"] is None
    assert feats["causal_centroid_dispersion_m"] is None


def test_no_retrospective_inputs_in_feature_vectors():
    """Assert forbidden retrospective and ground-truth keys are strictly absent from feature_vectors.jsonl."""
    repo_root = Path(__file__).resolve().parent.parent
    fv_path = repo_root / "results" / "estimator_context" / "feature_vectors.jsonl"
    if not fv_path.exists():
        pytest.skip("feature_vectors.jsonl not present")

    forbidden_keys = {
        "world_displacement_m",          # Full-sequence total
        "lifespan_sec",                  # Full-sequence total
        "total_detections",              # Full-sequence total
        "pipeline_motion_label",         # Full-sequence estimate
        "gt_vlm_label",                  # Ground-truth label
        "observed_motion_reference",     # Human ground truth
        "physical_mobility_reference",   # Human ground truth
        "annotation_confidence",         # Ground-truth metadata
    }

    with open(fv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            leaked = forbidden_keys.intersection(rec.keys())
            assert not leaked, f"Record {idx} ({rec.get('obs_id')}) leaks retrospective/GT fields: {leaked}"
