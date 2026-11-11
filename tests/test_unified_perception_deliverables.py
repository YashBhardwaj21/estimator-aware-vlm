"""Contract tests for unified perception deliverables across all eight benchmark sequences."""

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import get_repo_root


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture
def perception_dir(repo_root: Path) -> Path:
    p = repo_root / "results" / "perception"
    if not p.exists() or not (p / "run_manifest.json").exists():
        pytest.skip("results/perception deliverables not yet generated")
    return p


EXPECTED_SEQUENCE_IDS = [
    "tum_xyz",
    "tum_halfsphere",
    "tum_static",
    "openloris",
    "bonn_crowd",
    "bonn_box",
    "d455_01",
    "d455_02",
]


def test_deliverable_files_exist(perception_dir: Path):
    """Verify all 11 required deliverable files and crop directory exist."""
    required_files = [
        "run_manifest.json",
        "dataset_manifest.json",
        "tracks.jsonl",
        "detections.jsonl",
        "candidate_observations.jsonl",
        "rejected_observations.jsonl",
        "mask_manifest.jsonl",
        "vlm_sampling_manifest.jsonl",
        "perception_metrics.json",
        "perception_quality_audit.json",
        "visual_quality_report.md",
    ]
    for filename in required_files:
        p = perception_dir / filename
        assert p.exists(), f"Missing deliverable: {p}"
        assert p.stat().st_size > 0, f"Empty deliverable: {p}"

    crops_dir = perception_dir / "crops"
    assert crops_dir.exists() and crops_dir.is_dir()


def test_all_eight_sequences_in_manifests(perception_dir: Path):
    """Verify all 8 canonical sequences are processed in run_manifest and perception_metrics."""
    with open(perception_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    manifest_seqs = [s["sequence_id"] for s in run_manifest.get("sequences", [])]
    assert manifest_seqs == EXPECTED_SEQUENCE_IDS

    with open(perception_dir / "perception_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics.get("total_sequences") == 8
    metrics_seqs = [s["sequence_id"] for s in metrics.get("per_sequence", [])]
    assert metrics_seqs == EXPECTED_SEQUENCE_IDS


def test_model_provenance(perception_dir: Path, repo_root: Path):
    """Verify model provenance links strictly to yolov8s-seg.pt with valid SHA-256."""
    from estimator_aware_vlm.perception.unified_pipeline import UnifiedPerceptionPipeline

    with open(perception_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    model_path = run_manifest.get("yolo_model", "")
    assert "yolov8s-seg.pt" in model_path

    model_sha = run_manifest.get("yolo_model_sha256", "")
    actual_sha = UnifiedPerceptionPipeline.hash_file(model_path)
    assert model_sha == actual_sha
    assert len(model_sha) == 64


def test_observation_fields_and_integrity(perception_dir: Path, repo_root: Path):
    """Verify observations contain authentic fields, positive areas, and real image paths."""
    candidate_path = perception_dir / "candidate_observations.jsonl"
    count = 0

    with open(candidate_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obs = json.loads(line)
            count += 1

            assert obs["sequence_id"] in EXPECTED_SEQUENCE_IDS
            assert obs["track_id"] != ""
            assert obs["obs_id"] != ""
            assert obs["frame_index"] >= 0
            assert obs["timestamp"] > 0.0
            assert obs["semantic_class"] != ""
            assert 0.0 <= obs["detection_confidence"] <= 1.0
            assert "class_probabilities" in obs
            assert obs["semantic_class"] in obs["class_probabilities"]

            bbox = obs["bbox_xyxy"]
            assert len(bbox) == 4
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]

            assert obs["area_px"] > 0
            assert obs["median_depth_m"] > 0.0
            assert obs["sharpness"] >= 0.0

            if obs.get("has_mask"):
                assert obs["mask_area_px"] > 0
                assert 0.0 < obs["mask_to_box_ratio"] <= 1.25
                assert obs["mask_contour"] is not None
                assert len(obs["mask_contour"]) >= 3

            assert obs["rejection_reason"] is None
            if count >= 100:
                break

    assert count > 0


def test_quality_audit_metrics(perception_dir: Path):
    """Verify quality audit metrics are strictly populated for all 8 sequences."""
    with open(perception_dir / "perception_quality_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert "aggregate" in audit
    agg = audit["aggregate"]
    assert agg["total_detections"] > 0
    assert 0.0 <= agg["duplicate_rate"] <= 1.0
    assert 0.0 <= agg["small_objects_rate"] <= 1.0
    assert 0.0 <= agg["mask_fragmentation_rate"] <= 1.0
    assert 0.0 <= agg["transient_spurious_rate"] <= 1.0
    assert 0.0 <= agg["track_dropout_rate"] <= 1.0

    seq_audits = audit.get("per_sequence", [])
    assert len(seq_audits) == 8
    audit_ids = [sa["sequence_id"] for sa in seq_audits]
    assert audit_ids == EXPECTED_SEQUENCE_IDS

