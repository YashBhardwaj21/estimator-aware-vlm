"""Unit tests for Pass B Causal Instance Segmentation Masking & Attribution."""

import json
from pathlib import Path
import numpy as np
import pytest
import cv2

from estimator_aware_vlm.common.paths import get_repo_root


def test_selective_instance_masking_same_frame(tmp_path: Path):
    """Verify that in a frame with two objects, ONLY the router-selected track_id is masked."""
    h, w = 480, 640
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    depth = np.full((h, w), 2500, dtype=np.uint16)

    # Object 1: person_01 (dynamic, polygon in top-left)
    poly_person = [[50, 50], [150, 50], [150, 150], [50, 150]]
    # Object 2: monitor_01 (static, polygon in bottom-right)
    poly_monitor = [[400, 300], [550, 300], [550, 450], [400, 450]]

    # Router decision: only mask person_01
    masked_track_ids = {"person_01"}

    observations = [
        {"track_id": "person_01", "mask_contour": poly_person, "bbox_xyxy": [50, 50, 150, 150]},
        {"track_id": "monitor_01", "mask_contour": poly_monitor, "bbox_xyxy": [400, 300, 550, 450]},
    ]

    # Apply causal masking logic
    for obs in observations:
        if obs["track_id"] in masked_track_ids:
            pts = np.array(obs["mask_contour"], dtype=np.int32)
            cv2.fillPoly(rgb, [pts], (0, 0, 0))
            cv2.fillPoly(depth, [pts], 0)

    # Assert: person_01 region is masked (0)
    assert np.all(rgb[60:140, 60:140] == 0)
    assert np.all(depth[60:140, 60:140] == 0)

    # Assert: monitor_01 region is completely UNMASKED (200 / 2500)
    assert np.all(rgb[320:430, 420:530] == 200)
    assert np.all(depth[320:430, 420:530] == 2500)


def test_same_class_different_instance_masking(tmp_path: Path):
    """Verify that two instances of the SAME class (chair_01 and chair_02) can be masked independently."""
    h, w = 480, 640
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    depth = np.full((h, w), 1500, dtype=np.uint16)

    poly_chair1 = [[30, 30], [100, 30], [100, 100], [30, 100]]
    poly_chair2 = [[200, 200], [280, 200], [280, 280], [200, 280]]

    # Only chair_01 is displaced/masked; chair_02 is stationary and retained
    masked_track_ids = {"chair_01"}

    obs_chair1 = {"track_id": "chair_01", "mask_contour": poly_chair1}
    obs_chair2 = {"track_id": "chair_02", "mask_contour": poly_chair2}

    for obs in [obs_chair1, obs_chair2]:
        if obs["track_id"] in masked_track_ids:
            pts = np.array(obs["mask_contour"], dtype=np.int32)
            cv2.fillPoly(rgb, [pts], (0, 0, 0))
            cv2.fillPoly(depth, [pts], 0)

    # chair_01 is masked
    assert np.all(rgb[40:90, 40:90] == 0)
    assert np.all(depth[40:90, 40:90] == 0)

    # chair_02 is retained untouched
    assert np.all(rgb[210:270, 210:270] == 180)
    assert np.all(depth[210:270, 210:270] == 1500)


def test_causality_report_consistency():
    """Verify that the generated mask causality report matches the actual routing decisions."""
    repo_root = get_repo_root()
    report_path = repo_root / "outputs" / "pass_b_slam" / "mask_causality_report.json"
    decisions_path = repo_root / "outputs" / "vlm_inference" / "routing_decisions.json"

    if not report_path.exists() or not decisions_path.exists():
        pytest.skip("Pass B outputs not yet generated on disk")

    with open(report_path, "r", encoding="utf-8") as rf:
        report = json.load(rf)
    with open(decisions_path, "r", encoding="utf-8") as df:
        decisions = json.load(df)

    expected_masked = sorted([
        d["track_id"] for d in decisions.get("decisions", [])
        if d.get("action") == "MASK_FROM_TRACKING"
    ])

    assert report["masked_track_ids"] == expected_masked
    assert report["mask_type"] == "polygon_instance_segmentation"
    assert report["total_masked_frames"] > 0
    assert report["total_pixels_masked"] > 0
