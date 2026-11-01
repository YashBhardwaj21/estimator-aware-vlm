"""
Tests for the AntiFabricationValidator.
Ensures the system fails closed on invalid metrics, duplicate trajectories,
negative RMSE, and prompt telemetry leakage.
"""

import os
import pytest
import tempfile
from estimator_aware_vlm.audit.anti_fabrication import (
    AntiFabricationValidator,
    IntegrityViolationError
)

def test_negative_ate_rejected():
    validator = AntiFabricationValidator()
    with pytest.raises(IntegrityViolationError, match="cannot be negative"):
        validator.validate_metric_bounds(ate_rmse=-0.0254)

def test_non_finite_metric_rejected():
    validator = AntiFabricationValidator()
    with pytest.raises(IntegrityViolationError, match="non-finite"):
        validator.validate_metric_bounds(ate_rmse=float("nan"))

def test_prompt_leakage_rejected():
    validator = AntiFabricationValidator(strict_leakage_check=True)
    leaked_prompt = "Candidate: person. Measured 3D displacement: 1.240 m."
    with pytest.raises(IntegrityViolationError, match="leakage detected in VLM prompt"):
        validator.validate_vlm_prompt_cleanliness(leaked_prompt)

def test_clean_prompt_accepted():
    validator = AntiFabricationValidator(strict_leakage_check=True)
    clean_prompt = "Candidate: chair. Bounding box: [100, 150, 200, 250]. Classify physical affordance."
    # Should not raise
    validator.validate_vlm_prompt_cleanliness(clean_prompt)

def test_identical_treatment_to_baseline_rejected():
    validator = AntiFabricationValidator()
    with pytest.raises(IntegrityViolationError, match="identical to baseline"):
        validator.register_intervention("cand_01", traj_hash="abcd1234ef", baseline_hash="abcd1234ef")

def test_duplicate_treatment_trajectory_rejected():
    validator = AntiFabricationValidator()
    validator.register_intervention("cand_01", traj_hash="hash_alpha", baseline_hash="hash_base")
    with pytest.raises(IntegrityViolationError, match="Duplicate treatment trajectory hash detected"):
        validator.register_intervention("cand_02", traj_hash="hash_alpha", baseline_hash="hash_base")

def test_valid_trajectory_parsing():
    validator = AntiFabricationValidator()
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        # Write 15 valid poses
        for i in range(15):
            f.write(f"{1.0 + i*0.033:.4f} {i*0.01:.4f} 0.0000 0.5000 0.0000 0.0000 0.0000 1.0000\n")
        tmp_name = f.name
        
    try:
        res = validator.validate_trajectory(tmp_name, min_poses=10)
        assert res["valid_poses"] == 15
        assert len(res["sha256"]) == 64
    finally:
        os.remove(tmp_name)

