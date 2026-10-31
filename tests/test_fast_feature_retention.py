"""Unit tests for OpenCV FAST feature retention diagnostic."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.evaluate_fast_feature_retention import (
    evaluate_frame_fast,
    compute_distribution_stats,
    run_fast_diagnostic,
)


@pytest.fixture
def synthetic_textured_image():
    """Generate a 480x640 synthetic textured image with guaranteed FAST corners."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    return img


@pytest.fixture
def fast_detectors():
    return {
        "fast_threshold_20": cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True),
        "fast_threshold_7": cv2.FastFeatureDetector_create(threshold=7, nonmaxSuppression=True),
    }


def test_empty_mask(synthetic_textured_image, fast_detectors):
    """Verify empty/zero-area mask removes zero features."""
    bbox = [100.0, 100.0, 100.0, 100.0]  # Zero area
    contour = []
    res = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    assert res["mask_area_pixels"] == 0
    t20 = res["metrics"]["fast_threshold_20"]
    assert t20["original_inside_polygon"] == 0
    assert t20["original_inside_bbox"] == 0


def test_full_frame_mask(synthetic_textured_image, fast_detectors):
    """Verify full-frame mask captures 100% of features."""
    h, w = synthetic_textured_image.shape
    bbox = [0.0, 0.0, float(w), float(h)]
    contour = [[0, 0], [w, 0], [w, h], [0, h]]
    res = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    t20 = res["metrics"]["fast_threshold_20"]
    assert t20["original_total"] > 1000
    assert t20["original_inside_bbox"] == t20["original_total"]
    assert t20["pct_inside_bbox"] == pytest.approx(100.0, abs=1e-2)


def test_polygon_inside_bounding_box(synthetic_textured_image, fast_detectors):
    """Verify polygon features are strictly a subset of bounding box features."""
    bbox = [100.0, 100.0, 300.0, 300.0]
    # Triangle strictly inside the 200x200 bbox
    contour = [[150, 150], [250, 150], [200, 250]]
    res = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    t20 = res["metrics"]["fast_threshold_20"]
    assert t20["original_inside_polygon"] <= t20["original_inside_bbox"]
    assert t20["excess_inside_bbox_vs_polygon"] >= 0


def test_threshold_20_vs_threshold_7(synthetic_textured_image, fast_detectors):
    """Verify threshold 7 detects more or equal features than threshold 20."""
    bbox = [100.0, 100.0, 300.0, 300.0]
    contour = [[100, 100], [300, 100], [300, 300], [100, 300]]
    res = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    t20 = res["metrics"]["fast_threshold_20"]
    t7 = res["metrics"]["fast_threshold_7"]
    assert t7["original_total"] >= t20["original_total"]
    assert t7["original_inside_bbox"] >= t20["original_inside_bbox"]


def test_out_of_bounds_coordinates(synthetic_textured_image, fast_detectors):
    """Verify coordinates exceeding image bounds are safely clipped."""
    bbox = [-50.0, -50.0, 1000.0, 1000.0]
    contour = [[-50, -50], [1000, -50], [1000, 1000], [-50, 1000]]
    res = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    assert res["mask_area_pixels"] > 0
    t20 = res["metrics"]["fast_threshold_20"]
    assert t20["original_inside_bbox"] == t20["original_total"]


def test_deterministic_output(synthetic_textured_image, fast_detectors):
    """Verify evaluating same frame twice produces bitwise identical results."""
    bbox = [50.0, 50.0, 200.0, 200.0]
    contour = [[50, 50], [200, 50], [200, 200], [50, 200]]
    res1 = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    res2 = evaluate_frame_fast(synthetic_textured_image, bbox, contour, fast_detectors)
    assert res1 == res2


def test_empty_target_population(tmp_path: Path):
    """Verify runner handles empty observations file safely."""
    empty_jsonl = tmp_path / "empty_obs.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    res = run_fast_diagnostic(
        obs_path=empty_jsonl,
        target_track_id="nonexistent_track",
        output_dir=tmp_path / "out",
    )
    assert res["summary"]["successfully_measured_frames"] == 0
    assert res["summary"]["threshold_20"]["inside_polygon_kps"]["mean"] == 0.0
