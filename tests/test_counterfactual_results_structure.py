"""Unit and integration tests for Phase 1 results infrastructure and provenance."""

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import get_repo_root


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture
def run_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "counterfactual_interventions" / "run_20261031_counterfactual"


def test_results_readme_exists(repo_root: Path):
    """Verify permanent results README exists and defines governance policies."""
    readme_path = repo_root / "results" / "README.md"
    assert readme_path.exists(), "results/README.md is missing"
    content = readme_path.read_text(encoding="utf-8")
    assert "Permanent, reproducible experimental results" in content or "Estimator-Aware VLM" in content
    assert "Zero Fabrication" in content


def test_experiment_manifest_schema(run_dir: Path):
    """Verify experiment_manifest.json conforms to intervention specification."""
    manifest_path = run_dir / "experiment_manifest.json"
    assert manifest_path.exists(), f"Experiment manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_run_id"] == "run_20261031_counterfactual"
    assert "source_commit" in data
    assert "branch" in data
    assert len(data["research_scope"]) == 3

    # Check 8 intervention arms (C0 - C7)
    arms = {arm["id"]: arm for arm in data["intervention_arms"]}
    assert len(arms) == 8
    expected_ids = [f"C{i}" for i in range(8)]
    for eid in expected_ids:
        assert eid in arms, f"Missing intervention arm {eid}"

    # Verify initial executable vs native landmark statuses
    assert arms["C0"]["status"] == "IMPLEMENTED_AND_TESTED"
    assert arms["C1"]["status"] == "IMPLEMENTED_AND_TESTED"
    assert arms["C2"]["status"] == "IMPLEMENTED_AND_TESTED"
    assert arms["C3"]["status"] == "IMPLEMENTED_AND_TESTED"
    assert arms["C4"]["status"] == "IMPLEMENTED_AND_TESTED"
    assert arms["C5"]["status"] == "NOT_IMPLEMENTED"
    assert arms["C6"]["status"] == "NOT_IMPLEMENTED"
    assert arms["C7"]["status"] == "NOT_IMPLEMENTED"

    # Strict zero-fabrication contract
    assert data["zero_fabrication_contract"]["d455_camera_trajectory_ground_truth"] is None


def test_dataset_inventory_completeness(run_dir: Path):
    """Verify dataset_inventory.json captures all 3 experimental datasets."""
    inv_path = run_dir / "dataset_inventory.json"
    assert inv_path.exists(), f"Dataset inventory missing at {inv_path}"

    with open(inv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = data.get("datasets", {})
    assert "tum_rgbd" in datasets
    assert "openloris" in datasets
    assert "realsense_d455" in datasets

    # Verify TUM artifacts
    tum = datasets["tum_rgbd"]["artifacts"]
    assert "config" in tum and tum["config"]["sha256"]
    assert "groundtruth" in tum and tum["groundtruth"]["sha256"]
    assert "baseline_trajectory" in tum and tum["baseline_trajectory"]["sha256"]

    # Verify OpenLORIS artifacts
    ol = datasets["openloris"]["artifacts"]
    assert "config" in ol and ol["config"]["sha256"]
    assert "groundtruth" in ol and ol["groundtruth"]["sha256"]

    # Verify D455 artifacts
    d455 = datasets["realsense_d455"]["artifacts"]
    assert "config_s1" in d455 and d455["config_s1"]["sha256"]
    assert "config_s2" in d455 and d455["config_s2"]["sha256"]
    assert "movability_ground_truth" in d455 and d455["movability_ground_truth"]["sha256"]


def test_input_hashes_provenance(run_dir: Path):
    """Verify input_hashes.json contains valid 64-char hex SHA-256 hashes."""
    hashes_path = run_dir / "provenance" / "input_hashes.json"
    assert hashes_path.exists(), f"input_hashes.json missing at {hashes_path}"

    with open(hashes_path, "r", encoding="utf-8") as f:
        hashes = json.load(f)

    assert len(hashes) >= 12
    for rel_path, sha in hashes.items():
        assert len(sha) == 64, f"Invalid SHA-256 length for {rel_path}: {sha}"
        assert all(c in "0123456789abcdefABCDEF" for c in sha)

