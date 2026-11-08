"""Gate 1 acceptance tests for native factorial experiment configuration and trajectory mapping.

Verifies:
- Valid I0 and I1 manifests pass.
- Old I2 -> T1 mapping fails.
- Old I3 -> T3 mapping fails.
- Missing I2 and I3 trajectories are handled as NOT_RUN.
- Incorrect source configuration fails.
- Incorrect trajectory hash fails.
- Duplicate source-run or trajectory assignment fails.
- Partial evaluation is labelled INCOMPLETE_FACTORIAL_EXPERIMENT.
- Complete evaluation with missing modes fails closed.
"""

import copy
import json
from pathlib import Path
import pytest
import yaml

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.slam.factorial_manifest import (
    FactorialExperimentConfig,
    FactorialManifestValidator,
    FactorialModeConfig,
    FactorialValidationError,
    IncompleteFactorialError,
    STATUS_AVAILABLE,
    STATUS_NOT_RUN,
    STATUS_INCOMPLETE,
    STATUS_COMPLETE,
)


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture
def canonical_config_path(repo_root: Path) -> Path:
    p = repo_root / "configs" / "experiments" / "native_factorial_tum_xyz.yaml"
    assert p.is_file(), f"Canonical config not found: {p}"
    return p


@pytest.fixture
def canonical_config(canonical_config_path: Path) -> FactorialExperimentConfig:
    return FactorialExperimentConfig.load_yaml(canonical_config_path)


@pytest.fixture
def validator(repo_root: Path) -> FactorialManifestValidator:
    return FactorialManifestValidator(repo_root=repo_root)


@pytest.fixture
def gt_path(repo_root: Path) -> Path:
    p = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "groundtruth.txt"
    assert p.is_file(), f"Ground truth not found: {p}"
    return p


def test_canonical_yaml_config_valid(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """The canonical experiment YAML config must validate without errors."""
    report = validator.validate_manifest(canonical_config)
    assert report["validation_status"] == "PASSED"
    assert report["modes"]["I0"]["status"] == STATUS_AVAILABLE
    assert report["modes"]["I1"]["status"] == STATUS_AVAILABLE
    assert report["modes"]["I2"]["status"] == STATUS_NOT_RUN
    assert report["modes"]["I3"]["status"] == STATUS_NOT_RUN


def test_valid_i0_manifest(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """I0 baseline arm must resolve to the valid T0 baseline trajectory."""
    report = validator.validate_manifest(canonical_config)
    i0_rec = report["modes"]["I0"]
    assert i0_rec["status"] == STATUS_AVAILABLE
    assert i0_rec["valid_poses"] > 800
    assert i0_rec["duration_sec"] > 25.0
    assert canonical_config.modes["I0"].image_mask is False
    assert canonical_config.modes["I0"].native_landmark_weighting is False


def test_valid_i1_manifest(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """I1 image-mask arm must resolve to the valid joint polygon trajectory."""
    report = validator.validate_manifest(canonical_config)
    i1_rec = report["modes"]["I1"]
    assert i1_rec["status"] == STATUS_AVAILABLE
    assert i1_rec["valid_poses"] > 800
    assert canonical_config.modes["I1"].image_mask is True
    assert canonical_config.modes["I1"].native_landmark_weighting is False


def test_old_i2_t1_mapping_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator, repo_root: Path):
    """Old mapping of I2 to T1_person_02_POLYGON_ONLY must fail validation.

    T1 is an image polygon mask, not a native weighting run. It violates both
    image_mask=False and native_landmark_weighting=True.
    """
    bad_cfg = copy.deepcopy(canonical_config)
    t1_traj = (
        "results/independent_intervention_expansion/run_20261101_expansion/tum/"
        "individual_person_interventions/T1_person_02_POLYGON_ONLY/CameraTrajectory.txt"
    )
    bad_cfg.modes["I2"] = FactorialModeConfig(
        mode_id="I2",
        type="native_weighting_only",
        image_mask=False,
        native_landmark_weighting=True,
        status="available",
        source="T1_person_02_POLYGON_ONLY",
        source_run_id="run_20261101_expansion",
        source_manifest="results/independent_intervention_expansion/run_20261101_expansion/tum/individual_person_interventions/intervention_manifest.json",
        trajectory_path=t1_traj,
        trajectory_sha256="1a346fb9044fad879423df801d57b0a33d1dae61a0f18a2e2cb75329633c0036",
    )

    with pytest.raises(FactorialValidationError) as exc_info:
        validator.validate_manifest(bad_cfg)
    err_msg = str(exc_info.value)
    # Must fail because T1 has mask_mode="polygon" (violating image_mask=False) or lacks native weighting
    assert "image_mask=False" in err_msg or "native_landmark_weighting" in err_msg


def test_old_i3_t3_mapping_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """Old mapping of I3 to T3_both_persons_joint_POLYGON must fail validation.

    Reusing T3 duplicates I1's trajectory and SHA-256 digest, and lacks native weighting.
    """
    bad_cfg = copy.deepcopy(canonical_config)
    t3_traj = canonical_config.modes["I1"].trajectory_path
    t3_sha = canonical_config.modes["I1"].trajectory_sha256

    bad_cfg.modes["I3"] = FactorialModeConfig(
        mode_id="I3",
        type="combined",
        image_mask=True,
        native_landmark_weighting=True,
        status="available",
        source="T3_both_persons_joint_POLYGON",
        source_run_id="run_20261101_expansion",
        source_manifest=canonical_config.modes["I1"].source_manifest,
        trajectory_path=t3_traj,
        trajectory_sha256=t3_sha,
    )

    with pytest.raises(FactorialValidationError) as exc_info:
        validator.validate_manifest(bad_cfg)
    err_msg = str(exc_info.value)
    # Must fail due to duplicate trajectory / hash or duplicate source
    assert "Duplicate trajectory" in err_msg or "Duplicate source arm" in err_msg or "native_landmark_weighting" in err_msg


def test_missing_i2_trajectory_handled_as_not_run(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """Missing I2 trajectory must be safely represented as status='not_run' with null trajectory."""
    assert canonical_config.modes["I2"].status == STATUS_NOT_RUN
    assert canonical_config.modes["I2"].trajectory_path is None
    assert canonical_config.modes["I2"].trajectory_sha256 is None

    report = validator.validate_manifest(canonical_config)
    assert report["modes"]["I2"]["status"] == STATUS_NOT_RUN
    assert report["modes"]["I2"]["validated"] is True


def test_missing_i3_trajectory_handled_as_not_run(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """Missing I3 trajectory must be safely represented as status='not_run' with null trajectory."""
    assert canonical_config.modes["I3"].status == STATUS_NOT_RUN
    assert canonical_config.modes["I3"].trajectory_path is None
    assert canonical_config.modes["I3"].trajectory_sha256 is None

    report = validator.validate_manifest(canonical_config)
    assert report["modes"]["I3"]["status"] == STATUS_NOT_RUN
    assert report["modes"]["I3"]["validated"] is True


def test_not_run_with_trajectory_path_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """An arm declared as NOT_RUN must not define a trajectory path."""
    bad_cfg = copy.deepcopy(canonical_config)
    bad_cfg.modes["I2"].trajectory_path = "some/fictional/path.txt"

    with pytest.raises(FactorialValidationError, match="Must be null"):
        validator.validate_manifest(bad_cfg)


def test_incorrect_source_configuration_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """If mode type flags contradict the canonical specification, validation must fail."""
    bad_cfg = copy.deepcopy(canonical_config)
    bad_cfg.modes["I0"].image_mask = True  # Contradicts I0 baseline spec

    with pytest.raises(FactorialValidationError, match="violates canonical requirement image_mask=False"):
        validator.validate_manifest(bad_cfg)


def test_incorrect_trajectory_hash_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """If the trajectory file hash does not match the declared hash, fail closed."""
    bad_cfg = copy.deepcopy(canonical_config)
    bad_cfg.modes["I0"].trajectory_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(FactorialValidationError, match="trajectory SHA-256 mismatch"):
        validator.validate_manifest(bad_cfg)


def test_duplicate_source_run_assignment_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator):
    """Distinct arms cannot reference the same source execution arm."""
    bad_cfg = copy.deepcopy(canonical_config)
    bad_cfg.modes["I1"].source = bad_cfg.modes["I0"].source  # Both set to T0_baseline_KEEP

    with pytest.raises(FactorialValidationError, match="Duplicate source arm assignment|requires image_mask=True"):
        validator.validate_manifest(bad_cfg)


def test_partial_evaluation_labeled_correctly(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator, gt_path: Path):
    """Partial evaluation of I0 and I1 must return INCOMPLETE_FACTORIAL_EXPERIMENT status."""
    res = validator.evaluate(canonical_config, ground_truth_path=gt_path, allow_partial=True)
    assert res["status"] == STATUS_INCOMPLETE
    assert res["is_complete_factorial"] is False
    assert res["available_modes"] == ["I0", "I1"]
    assert res["pending_modes"] == ["I2", "I3"]

    evaluated = res["evaluated_modes"]
    assert len(evaluated) == 4

    i0_row = next(r for r in evaluated if r["mode_id"] == "I0")
    i1_row = next(r for r in evaluated if r["mode_id"] == "I1")
    i2_row = next(r for r in evaluated if r["mode_id"] == "I2")
    i3_row = next(r for r in evaluated if r["mode_id"] == "I3")

    assert i0_row["status"] == "available"
    assert i0_row["ate_rmse_m"] == pytest.approx(0.3637, abs=1e-4)

    assert i1_row["status"] == "available"
    assert i1_row["ate_rmse_m"] == pytest.approx(0.3729, abs=1e-4)
    assert i1_row["delta_ate_vs_i0_m"] == pytest.approx(-0.0092, abs=1e-4)

    assert i2_row["status"] == "NOT_RUN"
    assert i2_row["ate_rmse_m"] is None
    assert "PENDING_NATIVE_EXECUTION" in i2_row["note"]

    assert i3_row["status"] == "NOT_RUN"
    assert i3_row["ate_rmse_m"] is None
    assert "PENDING_NATIVE_EXECUTION" in i3_row["note"]


def test_complete_evaluation_with_missing_modes_fails(canonical_config: FactorialExperimentConfig, validator: FactorialManifestValidator, gt_path: Path):
    """Requesting complete factorial evaluation when arms are missing must fail closed."""
    with pytest.raises(IncompleteFactorialError, match="Missing available arms: \\['I2', 'I3'\\]"):
        validator.evaluate(canonical_config, ground_truth_path=gt_path, allow_partial=False)

