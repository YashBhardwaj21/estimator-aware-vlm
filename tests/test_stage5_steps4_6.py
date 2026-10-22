"""Regression and validation tests for Stage 5 Steps 4, 5, and 6.

Step 4: Geometric unaliased reprojection metrics and point-to-box consistency.
Step 5: Temporal multi-scale displacement, move-and-return detection, gap tracking, and session isolation.
Step 6: Semantic honest missing values (no 0.50/bbox fabrications), detection vs prior validity separation,
        empirical Shannon entropy, and multi-frame voting.
"""

import math
import pytest
from vlm.src.geometric_evidence import (
    GeometricEvidenceExtractor,
    GeometricEvidenceRecord,
)
from vlm.src.temporal_evidence import (
    TemporalEvidenceExtractor,
    TemporalEvidenceRecord,
    CUMULATIVE_PATH_THRESHOLD_M,
    STEP_MOTION_THRESHOLD_M,
)
from vlm.src.semantic_evidence import (
    SemanticEvidenceExtractor,
    SemanticEvidenceRecord,
)
from vlm.src.stage5_pipeline import run_stage5_pipeline


# ==============================================================================
# Step 4 Tests: Geometric Evidence & Honest Metric Unaliasing
# ==============================================================================

def test_step4_reprojection_residual_strictly_none_and_unaliased():
    """Verify that mean_reprojection_residual_px is never conflated with bbox distance."""
    rec = GeometricEvidenceRecord(
        obs_id="obs_test_geo",
        frame_index=1,
        timestamp=0.1,
        entity_id="chair_1",
        track_id="trk_1",
        source_session="session_1",
        depth_valid=True,
        measured_depth_m=1.85,
        depth_uncertainty_sigma_m=0.012,
        depth_provenance="MEASURED_SENSOR",
        point_to_bbox_valid=True,
        point_to_bbox_provenance="MEASURED_MAPPOINTS",
        point_to_bbox_distance_px=14.2,
        point_to_bbox_inlier_ratio=0.88,
        mappoints_candidate_count=25,
        mappoints_inlier_count=22,
        observation_points_count=150,
        feature_reprojection_residual_valid=False,
        feature_reprojection_residual_px=None,
        feature_reprojection_provenance="UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES",
        landmark_support_count=25,
        landmark_spatial_extent_m=0.45,
        geometric_consistency_score=0.85,
        provenance="MEASURED_FULL",
        has_valid_evidence=True,
    )
    # The property and dict entry MUST be None
    assert rec.mean_reprojection_residual_px is None
    assert rec.median_reprojection_residual_px is None
    assert rec.p95_reprojection_residual_px is None
    assert rec.reprojection_residual_valid is False
    assert rec.feature_reprojection_residual_valid is False
    assert rec.feature_reprojection_residual_px is None

    # Check to_dict serialization
    d = rec.to_dict()
    assert d["mean_reprojection_residual_px"] is None
    assert d["point_to_bbox_distance_px"] == 14.2
    assert d["point_to_bbox_inlier_ratio"] == 0.88


# ==============================================================================
# Step 5 Tests: Temporal Multi-Scale Motion, Gap Detection, & Session Isolation
# ==============================================================================

def test_step5_move_and_return_trajectory_classified_as_displaced():
    """Verify that move-and-return trajectory (low net displacement, high cumulative path) is DISPLACED."""
    extractor = TemporalEvidenceExtractor()

    # Step 1: Initial position at (0.0, 0.0, 0.0)
    rec1 = extractor.extract_observation_temporal({
        "obs_id": "m1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec1.current_motion == "TRANSIENT"
    assert rec1.cumulative_path_length_m == 0.0

    # Step 2: Second observation at (0.0, 0.0, 0.0) (n_obs=2 is still TRANSIENT until 3 samples seen)
    rec2 = extractor.extract_observation_temporal({
        "obs_id": "m2", "frame_index": 2, "timestamp": 0.1,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec2.current_motion == "TRANSIENT"

    # Step 3: Moves 0.35m away to (0.35, 0.0, 0.0) (n_obs=3, step_disp > 0.05 -> MOVING)
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "m3", "frame_index": 5, "timestamp": 0.25,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.35, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec3.current_motion == "MOVING"

    # Step 4: Moves back to near-origin (0.02, 0.0, 0.0)
    rec4 = extractor.extract_observation_temporal({
        "obs_id": "m4", "frame_index": 8, "timestamp": 0.40,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.02, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Step 5: Settles at returned position (0.02, 0.0, 0.0) with step_disp=0.0 <= 0.05
    # Net displacement is 0.02m (< 0.25m threshold), but cumulative path is 0.68m (> 0.50m)
    rec5 = extractor.extract_observation_temporal({
        "obs_id": "m5", "frame_index": 10, "timestamp": 0.50,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.02, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec5.net_displacement_m == pytest.approx(0.02, abs=1e-3)
    assert rec5.cumulative_path_length_m == pytest.approx(0.68, abs=1e-2)
    assert rec5.cumulative_path_length_m > CUMULATIVE_PATH_THRESHOLD_M
    assert rec5.current_motion == "DISPLACED"


def test_step5_sampling_gap_detection_and_reappearance():
    """Verify that gaps exceeding thresholds increment observation_gap_events_count."""
    extractor = TemporalEvidenceExtractor()

    # Observation 1 at frame 0
    extractor.extract_observation_temporal({
        "obs_id": "g1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Observation 2 after normal interval (frame 5, +0.15s) -> no gap
    rec2 = extractor.extract_observation_temporal({
        "obs_id": "g2", "frame_index": 5, "timestamp": 0.15,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec2.observation_gap_events_count == 0
    assert rec2.is_reappearance_after_gap is False

    # Observation 3 after 25 frames and 1.2s -> gap event triggered!
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "g3", "frame_index": 30, "timestamp": 1.35,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec3.observation_gap_events_count == 1
    assert rec3.is_reappearance_after_gap is True
    assert rec3.frames_since_last_seen == 25
    # Backward compatibility properties
    assert rec3.occlusion_events_count == 1
    assert rec3.is_reappearance_after_occlusion is True


def test_step5_session_isolation_in_temporal_tracks():
    """Verify that same local_track_id in different sessions does NOT cross-contaminate history."""
    extractor = TemporalEvidenceExtractor()

    # Session 1 track "trk_1"
    extractor.extract_observation_temporal({
        "obs_id": "s1_obs1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "ent_1", "local_track_id": "trk_1", "source_session": "session_1",
        "centroid_floor": [1.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    extractor.extract_observation_temporal({
        "obs_id": "s1_obs2", "frame_index": 5, "timestamp": 0.2,
        "global_entity_id": "ent_1", "local_track_id": "trk_1", "source_session": "session_1",
        "centroid_floor": [1.5, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Session 2 track with the exact same "trk_1" should be treated as brand new track
    rec_s2 = extractor.extract_observation_temporal({
        "obs_id": "s2_obs1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "ent_2", "local_track_id": "trk_1", "source_session": "session_2",
        "centroid_floor": [5.0, 5.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    # Should be observation_count = 1, current_motion = TRANSIENT, displacement = 0.0
    assert rec_s2.observation_count == 1
    assert rec_s2.current_motion == "TRANSIENT"
    assert rec_s2.net_displacement_m == 0.0


# ==============================================================================
# Step 6 Tests: Semantic Honest Missing Values & Prior Separation
# ==============================================================================

def test_step6_no_synthetic_defaults_for_missing_fields():
    """Verify missing detector confidence and bbox return None rather than 0.50 or fake coordinates."""
    extractor = SemanticEvidenceExtractor()

    # Missing confidence and bbox
    rec = extractor.extract_observation_semantics({
        "obs_id": "sem_empty",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "chair_001",
        "class_name": "chair",
        # confidence omitted
        # bbox_xyxy omitted
    })
    assert rec.detector_confidence is None
    assert rec.bbox_area_px is None
    assert rec.detection_valid is False
    assert any("Missing or non-finite detector confidence" in diag for diag in rec.semantic_diagnostics)
    assert any("Missing or malformed bbox_xyxy" in diag for diag in rec.semantic_diagnostics)


def test_step6_separation_of_detection_valid_and_class_known():
    """Verify distinction between valid detection and known vocabulary prior."""
    extractor = SemanticEvidenceExtractor()

    # Valid detection of unknown object class "hoverboard"
    rec = extractor.extract_observation_semantics({
        "obs_id": "sem_unknown_class",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "ent_hover",
        "class_name": "hoverboard",
        "confidence": 0.88,
        "bbox_xyxy": [10.0, 20.0, 110.0, 120.0],
    })
    assert rec.detection_valid is True
    assert rec.class_known_to_prior is False
    assert rec.is_unknown_class is True
    assert rec.class_prior_mobility == "UNKNOWN"
    assert rec.class_prior_confidence == 0.0
    assert "Unmodeled object class 'hoverboard'" in rec.cheap_prior_description


def test_step6_session_isolation_in_semantic_voting():
    """Verify that multi-frame class voting history is strictly isolated across sessions."""
    extractor = SemanticEvidenceExtractor()

    # In Session 1, trk_shared voted "chair"
    extractor.extract_observation_semantics({
        "obs_id": "s1_1", "frame_index": 0, "timestamp": 0.0,
        "local_track_id": "trk_shared", "source_session": "session_1",
        "class_name": "chair", "confidence": 0.90,
    })

    # In Session 2, trk_shared is a "table"
    rec_s2 = extractor.extract_observation_semantics({
        "obs_id": "s2_1", "frame_index": 0, "timestamp": 0.0,
        "local_track_id": "trk_shared", "source_session": "session_2",
        "class_name": "table", "confidence": 0.85,
    })

    # In Session 2, there should only be 1 vote (for table), purity = 1.0, entropy = 0.0
    assert rec_s2.vote_count_total == 1
    assert rec_s2.semantic_purity == 1.0
    assert rec_s2.semantic_entropy == 0.0
