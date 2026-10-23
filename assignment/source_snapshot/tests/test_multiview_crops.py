"""Unit tests for Multi-View Crop Harvester."""

import json
from pathlib import Path
import cv2
import pytest


def test_multiview_dataset_manifest_and_crops():
    """Verify that harvested multi-view dataset satisfies all observation assembly quality gates."""
    manifest_path = Path("vlm/outputs/multiview_dataset.json")
    assert manifest_path.exists(), "multiview_dataset.json does not exist"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_physical_objects"] == 45
    assert manifest["total_crops_harvested"] == 135

    crops_dir = Path("vlm/outputs/crops")
    assert crops_dir.exists()

    all_objects = []
    for seq_name, objs in manifest["sequences"].items():
        assert len(objs) in (20, 25)
        for obj in objs:
            all_objects.append(obj)
            views = obj["views"]
            assert len(views) == 3, f"Object {obj['object_id']} does not have 3 views"

            # Check each view
            frames = []
            for v in views:
                frames.append(v["frame_index"])
                assert v["detector_confidence"] > 0.0
                assert v["sharpness_score"] > 0.0
                assert len(v["camera_pose"]) == 7
                assert len(v["bbox_xyxy"]) == 4

                # Check crop image file on disk
                img_path = Path("vlm/outputs") / v["crop_rel_path"]
                assert img_path.exists(), f"Image {img_path} not found"
                img = cv2.imread(str(img_path))
                assert img is not None, f"Failed to read image {img_path}"
                h, w = img.shape[:2]
                assert h >= 16 and w >= 16, f"Crop {img_path} too small ({w}x{h})"

            # Pass condition: views are not redundant copies of identical frames
            assert len(set(frames)) == len(frames), (
                f"Object {obj['object_id']} has duplicate frame views: {frames}"
            )
            # Ensure temporal separation between views is at least 5 frames
            for i in range(len(frames) - 1):
                diff = abs(frames[i + 1] - frames[i])
                assert diff >= 5, f"Views too close in time ({frames}) for {obj['object_id']}"


def test_stage3_persistent_identity_and_multiview_samples():
    """Verify observation assembly artifacts: object_tracks, object_observations, multiview_samples, and metrics."""
    out_dir = Path("vlm/outputs")

    tracks_file = out_dir / "object_tracks.jsonl"
    obs_file = out_dir / "object_observations.jsonl"
    samples_dir = out_dir / "multiview_samples"
    metrics_file = out_dir / "multiview_observation_metrics.json"
    vis_file = out_dir / "association_visualizations" / "multiview_association_montage.jpg"

    assert tracks_file.exists() and tracks_file.stat().st_size > 0
    assert obs_file.exists() and obs_file.stat().st_size > 0
    assert samples_dir.exists()
    assert metrics_file.exists()
    assert vis_file.exists()

    # Verify metrics structure
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["dataset_summary"]["total_verified_physical_objects"] >= 40
    assert metrics["provenance_and_data_integrity"]["pose_match_rate_pct"] >= 99.0
    assert metrics["provenance_and_data_integrity"]["bbox_match_rate_pct"] >= 99.0
    assert metrics["provenance_and_data_integrity"]["silent_fallback_substitutions"] == 0
    assert metrics["viewpoint_diversity"]["angular_spread_min_deg"] >= 12.0
    assert metrics["track_continuity_and_provenance"]["coverage_pct"] >= 70.0

    # Verify per-object samples have both crops and highlighted full scenes
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    assert len(sample_dirs) >= 30

    for s_dir in sample_dirs[:10]:
        meta_file = s_dir / "sample_meta.json"
        assert meta_file.exists()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "one_view" in meta["conditions"]
        assert "two_views" in meta["conditions"]
        assert "three_views" in meta["conditions"]

        views = meta["views"]
        assert len(views) >= 1
        for v in views:
            crop_path = s_dir / Path(v["crop_path"]).name
            scene_path = s_dir / Path(v["scene_path"]).name
            assert crop_path.exists(), f"Crop image missing: {crop_path}"
            assert scene_path.exists(), f"Scene image missing: {scene_path}"
            assert v["sharpness_score"] >= 0.0
            assert len(v["camera_pose"]) == 7

