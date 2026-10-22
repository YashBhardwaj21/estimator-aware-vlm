"""Regression tests for model comparison metrics, confusion matrices, and population equality."""

import json
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent
from vlm.scripts.generate_model_comparison_report import (
    calculate_metrics,
    wilson_interval,
    mcnemar_exact,
    stratified_paired_bootstrap_macro_f1,
    CLASSES,
)


def test_metric_calculation_mathematical_consistency():
    gt_map = {
        "inst_1": "IMMOVABLE",
        "inst_2": "MOVABLE",
        "inst_3": "MOVABLE",
        "inst_4": "MOBILE",
    }
    records = [
        {"physical_instance_id": "inst_1", "parsed_prediction": "IMMOVABLE", "prediction_accepted": True, "parse_valid": True, "schema_valid": True, "total_inference_ms": 100.0},
        {"physical_instance_id": "inst_2", "parsed_prediction": "MOVABLE", "prediction_accepted": True, "parse_valid": True, "schema_valid": True, "total_inference_ms": 110.0},
        {"physical_instance_id": "inst_3", "parsed_prediction": "IMMOVABLE", "prediction_accepted": True, "parse_valid": True, "schema_valid": True, "total_inference_ms": 120.0},  # incorrect
        {"physical_instance_id": "inst_4", "parsed_prediction": "MOBILE", "prediction_accepted": True, "parse_valid": True, "schema_valid": True, "total_inference_ms": 130.0},
    ]

    metrics = calculate_metrics(records, gt_map)
    # 3 correct out of 4 = 75.0%
    assert metrics["overall_accuracy"] == 0.75
    assert metrics["support"]["IMMOVABLE"] == 1
    assert metrics["support"]["MOVABLE"] == 2
    assert metrics["support"]["MOBILE"] == 1

    # Recalls: IMMOVABLE = 1/1 = 1.0; MOVABLE = 1/2 = 0.5; MOBILE = 1/1 = 1.0
    assert metrics["recalls"]["IMMOVABLE"] == 1.0
    assert metrics["recalls"]["MOVABLE"] == 0.5
    assert metrics["recalls"]["MOBILE"] == 1.0

    # Balanced accuracy = (1.0 + 0.5 + 1.0) / 3 = 2.5 / 3 = 0.8333
    assert abs(metrics["balanced_accuracy"] - (2.5 / 3.0)) < 1e-3

    # Confusion matrix row sums must match class support
    cm = metrics["confusion_matrix"]
    for c in CLASSES:
        assert sum(cm[c].values()) == metrics["support"][c]


def test_mcnemar_exact_properties():
    # Both identical = p-value 1.0
    res_ident = mcnemar_exact([True, False, True], [True, False, True])
    assert res_ident["p_value"] == 1.0
    assert res_ident["discordant_pairs"] == 0

    # Perfectly discordant (one model gets all correct, other fails)
    res_disc = mcnemar_exact([True] * 10, [False] * 10)
    assert res_disc["discordant_pairs"] == 10
    # Binomial p-value for 10 discordant pairs all favoring one model: 2 * (0.5^10) = 0.00195
    assert res_disc["p_value"] < 0.01
    assert res_disc["significant_p05"] is True


def test_wilson_interval_bounds():
    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert 0.0 < low < 1.0

    low_z, high_z = wilson_interval(0, 10)
    assert low_z == 0.0
    assert 0.0 < high_z < 1.0


def test_stratified_bootstrap_reproducibility():
    gt_map = {f"inst_{i}": "MOVABLE" if i % 2 == 0 else "IMMOVABLE" for i in range(20)}
    preds_a = [{"physical_instance_id": f"inst_{i}", "parsed_prediction": "MOVABLE"} for i in range(20)]
    preds_b = [{"physical_instance_id": f"inst_{i}", "parsed_prediction": "IMMOVABLE"} for i in range(20)]

    res1 = stratified_paired_bootstrap_macro_f1(preds_a, preds_b, gt_map, n_boot=500, seed=123)
    res2 = stratified_paired_bootstrap_macro_f1(preds_a, preds_b, gt_map, n_boot=500, seed=123)
    assert res1["observed_diff"] == res2["observed_diff"]
    assert res1["ci_95"] == res2["ci_95"]
