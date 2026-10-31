"""Unit tests for Phase 3 TUM monitor 3D displacement diagnostic."""

import json
from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from scripts.evaluate_tum_monitor_displacement import (
    load_tum_depth_index,
    find_nearest_depth_file,
    run_monitor_diagnostic,
)


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


def test_load_depth_index(repo_root: Path):
    """Verify depth index loads valid non-empty timestamp mapping."""
    depth_txt = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "depth.txt"
    mapping = load_tum_depth_index(depth_txt)
    assert len(mapping) > 800
    sample_ts = list(mapping.keys())[0]
    assert isinstance(sample_ts, float)
    assert sample_ts > 1.3e9


def test_nearest_depth_matching(repo_root: Path):
    """Verify nearest depth matching selects closest timestamp within max_dt."""
    depth_txt = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "depth.txt"
    mapping = load_tum_depth_index(depth_txt)
    seq_dir = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz"

    exact_ts = list(mapping.keys())[10]
    matched = find_nearest_depth_file(exact_ts + 0.005, mapping, seq_dir, max_dt_sec=0.033)
    assert matched is not None
    assert matched.exists()


def test_3d_dispersion_invariance():
    """Verify spatial dispersion (std norm) is invariant under global translation."""
    points = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.05, 3.02],
        [0.95, 1.98, 2.99],
    ])
    std_orig = np.linalg.norm(np.std(points, axis=0))

    # Shift points by arbitrary translation vector
    shifted = points + np.array([100.0, -50.0, 20.0])
    std_shifted = np.linalg.norm(np.std(shifted, axis=0))

    assert std_orig == pytest.approx(std_shifted, abs=1e-8)


def test_monitor_diagnostic_output_schema(repo_root: Path):
    """Verify monitor diagnostic generates valid JSON report with causal verdict."""
    diag_json = repo_root / "results" / "counterfactual_interventions" / "run_20261031_counterfactual" / "tum" / "diagnostics" / "monitor_reconstruction_diagnostic.json"
    assert diag_json.exists(), f"Diagnostic JSON missing at {diag_json}"

    with open(diag_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["target_track_id"] == "monitor_01"
    assert data["evaluated_frames_count"] > 100
    metrics = data["dispersion_metrics"]
    assert "vicon_ground_truth_poses" in metrics
    assert "estimated_poses" in metrics
    assert data["causal_conclusion"]["is_fixed_in_reality"] is True
    assert "ESTIMATOR_INDUCED_APPARENT_DISPLACEMENT" in data["causal_conclusion"]["verdict"]

