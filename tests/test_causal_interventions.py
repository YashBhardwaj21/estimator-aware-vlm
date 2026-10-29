"""Unit and contract tests for Stage 7: Valid & Causally Interpretable Interventions.

Verifies:
1. Polygon vs bounding-box geometric ablation (pixel suppression precision).
2. Mask causality report integrity (track attribution, frame counts, pixel totals).
3. Immutability guarantee (raw directories remain untouched).
4. Cryptographic cache invalidation upon mask geometry or policy alteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set
import cv2
import numpy as np
import pytest

from scripts.run_pass_b_slam import generate_pass_b_masked_associations


@pytest.fixture
def mock_intervention_workspace(tmp_path: Path) -> Dict[str, Any]:
    """Create isolated mock sequence directory, observations, and routing decisions."""
    raw_dir = tmp_path / "raw_sequence"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = raw_dir / "rgb"
    depth_dir = raw_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 synthetic RGB-D frames (640x480) with textured background
    frame_filenames = ["1000.000.png", "1000.033.png", "1000.066.png"]
    timestamps = [1000.000, 1000.033, 1000.066]

    for fname in frame_filenames:
        rgb_img = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Add background grid
        cv2.line(rgb_img, (0, 240), (640, 240), (100, 100, 100), 2)
        cv2.line(rgb_img, (320, 0), (320, 480), (100, 100, 100), 2)
        cv2.imwrite(str(rgb_dir / fname), rgb_img)

        depth_img = np.full((480, 640), 1500, dtype=np.uint16)  # 1.5m depth
        cv2.imwrite(str(depth_dir / fname), depth_img)

    # Create raw associations file
    raw_assoc_file = tmp_path / "associations_raw.txt"
    with open(raw_assoc_file, "w", encoding="utf-8") as f:
        for t, fname in zip(timestamps, frame_filenames):
            f.write(f"{t:.6f} rgb/{fname} {t:.6f} depth/{fname}\n")

    # Observations: Frame 1 & 2 contain dynamic entity with a diamond/polygon contour
    # Bounding box is [100, 100, 300, 300] (area = 200 * 200 = 40,000 px)
    # Polygon is a diamond inside bbox: (200, 100), (300, 200), (200, 300), (100, 200)
    # Area of diamond = (200 * 200) / 2 = 20,000 px (exactly half!)
    diamond_contour = [[200, 100], [300, 200], [200, 300], [100, 200]]
    bbox = [100, 100, 300, 300]

    obs_file = tmp_path / "observations.jsonl"
    with open(obs_file, "w", encoding="utf-8") as f:
        # Frame 0: dynamic entity present
        f.write(json.dumps({
            "rgb_path": f"rgb/{frame_filenames[0]}",
            "track_id": "person_01",
            "semantic_class": "person",
            "mask_contour": diamond_contour,
            "bbox_xyxy": bbox,
        }) + "\n")
        # Frame 1: dynamic entity present
        f.write(json.dumps({
            "rgb_path": f"rgb/{frame_filenames[1]}",
            "track_id": "person_01",
            "semantic_class": "person",
            "mask_contour": diamond_contour,
            "bbox_xyxy": bbox,
        }) + "\n")
        # Frame 2: static entity present
        f.write(json.dumps({
            "rgb_path": f"rgb/{frame_filenames[2]}",
            "track_id": "monitor_01",
            "semantic_class": "monitor",
            "mask_contour": [[50, 50], [90, 50], [90, 90], [50, 90]],
            "bbox_xyxy": [50, 50, 90, 90],
        }) + "\n")

    # Routing decisions: person_01 flagged for MASK_FROM_TRACKING, monitor_01 RETAIN
    routing_file = tmp_path / "routing_decisions.json"
    with open(routing_file, "w", encoding="utf-8") as f:
        json.dump({
            "policy_key": "B6_Estimator_Aware",
            "decisions": [
                {"track_id": "person_01", "action": "MASK_FROM_TRACKING"},
                {"track_id": "monitor_01", "action": "RETAIN"},
            ],
        }, f, indent=2)

    return {
        "raw_dir": raw_dir,
        "raw_assoc_file": raw_assoc_file,
        "obs_file": obs_file,
        "routing_file": routing_file,
        "masked_track_ids": {"person_01"},
        "tmp_path": tmp_path,
    }


def test_polygon_vs_bbox_masking_geometry(mock_intervention_workspace: Dict[str, Any]) -> None:
    """Verify that polygon masking suppresses significantly fewer static background pixels than bounding box."""
    ws = mock_intervention_workspace
    tmp = ws["tmp_path"]

    out_poly = tmp / "exp_poly"
    out_bbox = tmp / "exp_bbox"

    # 1. Run Polygon Masking
    assoc_poly = generate_pass_b_masked_associations(
        raw_seq_dir=ws["raw_dir"],
        output_dir=out_poly,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=out_poly / "mask_causality_report.json",
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="polygon",
    )

    # 2. Run Bounding Box Masking
    assoc_bbox = generate_pass_b_masked_associations(
        raw_seq_dir=ws["raw_dir"],
        output_dir=out_bbox,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=out_bbox / "mask_causality_report.json",
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="bbox",
    )

    assert assoc_poly.exists()
    assert assoc_bbox.exists()

    # Ingest causality reports
    with open(out_poly / "mask_causality_report.json", "r", encoding="utf-8") as f:
        rep_poly = json.load(f)
    with open(out_bbox / "mask_causality_report.json", "r", encoding="utf-8") as f:
        rep_bbox = json.load(f)

    # Verify frame counts match
    assert rep_poly["total_masked_frames"] == 2
    assert rep_bbox["total_masked_frames"] == 2

    # Verify bounding box suppresses roughly twice as many pixels as the inscribed diamond polygon
    # Theoretical: Box = 40,000 px/frame, Diamond = 20,000 px/frame
    poly_px = rep_poly["total_pixels_masked"]
    bbox_px = rep_bbox["total_pixels_masked"]

    assert bbox_px > poly_px
    ratio = bbox_px / max(1, poly_px)
    assert 1.8 <= ratio <= 2.2, f"Expected box/polygon pixel ratio ~2.0, got {ratio:.2f}"

    # Verify pixel-level image suppression
    masked_poly_img = cv2.imread(str(out_poly / "masked_pass_b" / "rgb" / "1000.000.png"))
    masked_bbox_img = cv2.imread(str(out_bbox / "masked_pass_b" / "rgb" / "1000.000.png"))

    # Pixel at (110, 110) is inside bounding box [100, 100, 300, 300], but OUTSIDE diamond polygon
    # Therefore, it should be BLACK (0, 0, 0) in bbox masking, but UNTOUCHED (200, 200, 200) in polygon masking!
    assert np.all(masked_bbox_img[110, 110] == 0), "BBox mask failed to suppress inside bounding box corner"
    assert np.all(masked_poly_img[110, 110] == 200), "Polygon mask erroneously suppressed background outside contour"


def test_causality_report_contents(mock_intervention_workspace: Dict[str, Any]) -> None:
    """Verify causality report contains complete audit trail."""
    ws = mock_intervention_workspace
    out_dir = ws["tmp_path"] / "exp_causality"
    rep_path = out_dir / "mask_causality_report.json"

    generate_pass_b_masked_associations(
        raw_seq_dir=ws["raw_dir"],
        output_dir=out_dir,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=rep_path,
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="polygon",
    )

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["total_sequence_frames"] == 3
    assert rep["total_masked_frames"] == 2
    assert rep["masked_track_ids"] == ["person_01"]
    assert "person_01" in rep["per_track_frame_count"]
    assert rep["per_track_frame_count"]["person_01"] == 2
    assert rep["mask_type"] == "polygon_segmentation"
    assert rep["masking_percentage"] == round((2 / 3) * 100.0, 2)


def test_immutability_during_intervention(mock_intervention_workspace: Dict[str, Any]) -> None:
    """Verify that raw sequence directory has zero files created or modified."""
    ws = mock_intervention_workspace
    raw_dir = ws["raw_dir"]

    # Snapshot raw dir
    initial_files = {p: p.stat().st_mtime for p in raw_dir.rglob("*")}

    out_dir = ws["tmp_path"] / "exp_immutability"
    generate_pass_b_masked_associations(
        raw_seq_dir=raw_dir,
        output_dir=out_dir,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=out_dir / "mask_causality_report.json",
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="polygon",
    )

    # Post-check raw dir
    post_files = {p: p.stat().st_mtime for p in raw_dir.rglob("*")}
    assert initial_files == post_files, "Raw sequence directory was illegally modified during intervention!"


def test_cache_invalidation_on_mask_type_change(mock_intervention_workspace: Dict[str, Any]) -> None:
    """Verify that switching mask_type invalidates cryptographic cache and re-generates masks."""
    ws = mock_intervention_workspace
    out_dir = ws["tmp_path"] / "exp_cache_test"
    rep_path = out_dir / "mask_causality_report.json"

    # 1. Run polygon
    generate_pass_b_masked_associations(
        raw_seq_dir=ws["raw_dir"],
        output_dir=out_dir,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=rep_path,
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="polygon",
    )
    with open(rep_path, "r", encoding="utf-8") as f:
        rep1 = json.load(f)
    assert rep1["mask_type"] == "polygon_segmentation"

    # 2. Run bbox in same output dir -> must invalidate cache and re-generate
    generate_pass_b_masked_associations(
        raw_seq_dir=ws["raw_dir"],
        output_dir=out_dir,
        masked_track_ids=ws["masked_track_ids"],
        observations_path=ws["obs_file"],
        routing_decisions_path=ws["routing_file"],
        causality_report_path=rep_path,
        raw_assoc_path=ws["raw_assoc_file"],
        mask_type="bbox",
    )
    with open(rep_path, "r", encoding="utf-8") as f:
        rep2 = json.load(f)
    assert rep2["mask_type"] == "bounding_box_suppression"
    assert rep2["total_pixels_masked"] > rep1["total_pixels_masked"]

