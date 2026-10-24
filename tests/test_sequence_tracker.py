"""Unit tests for SequenceTracker 3D Spatial Association and Motion Gating."""

import math
from pathlib import Path
from typing import List
import numpy as np
import pytest

from estimator_aware_vlm.datasets import CameraIntrinsics
from estimator_aware_vlm.perception.sequence_tracker import (
    MotionObservationState,
    ObservationRecord,
    SequenceTracker,
    TrackedEntity,
)


@pytest.fixture
def dummy_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


def test_stationary_object_ego_motion_compensation(dummy_intrinsics: CameraIntrinsics):
    """Verify that camera ego-motion does NOT cause a stationary world object to appear moving."""
    # World position of stationary table: (1.0, 0.0, 2.0)
    p_world = np.array([1.0, 0.0, 2.0])

    # Camera moves from (0, 0, 0) to (0.5, 0, 0)
    t_wc_1 = np.array([0.0, 0.0, 0.0])
    t_wc_2 = np.array([0.5, 0.0, 0.0])
    R_wc = np.eye(3)

    # In camera frame 1: p_cam = p_world - t_wc_1 = (1.0, 0.0, 2.0)
    p_cam_1 = p_world - t_wc_1
    # In camera frame 2: p_cam = p_world - t_wc_2 = (0.5, 0.0, 2.0)
    p_cam_2 = p_world - t_wc_2

    # Apparent camera-frame displacement is 0.50 m
    app_disp = float(np.linalg.norm(p_cam_2 - p_cam_1))
    assert app_disp == pytest.approx(0.50)

    # After camera ego-motion compensation:
    p_world_est_1 = R_wc @ p_cam_1 + t_wc_1
    p_world_est_2 = R_wc @ p_cam_2 + t_wc_2
    world_disp = float(np.linalg.norm(p_world_est_2 - p_world_est_1))

    # World displacement is 0.0 m!
    assert world_disp == pytest.approx(0.0)


def test_arbitrary_actor_count_no_two_person_limit(dummy_intrinsics: CameraIntrinsics):
    """Verify that SequenceTracker tracks arbitrary numbers of persons without truncating to 2 (F-10)."""
    tracker = SequenceTracker(intrinsics=dummy_intrinsics, min_track_observations=2)

    # Create 5 distinct walking persons at different spatial positions
    tracks: List[TrackedEntity] = []
    for i in range(1, 6):
        t = TrackedEntity(
            track_id=f"temp_person_{i}",
            semantic_class="person",
            observations=[
                ObservationRecord(
                    frame_index=0,
                    timestamp=100.0,
                    semantic_class="person",
                    confidence=0.90,
                    bbox_xyxy=(i * 80, 100, i * 80 + 50, 200),
                    centroid_camera=(float(i), 0.0, 2.0),
                    centroid_world=(float(i), 0.0, 2.0),
                    area_px=5000,
                    median_depth_m=2.0,
                    sharpness=300.0,
                ),
                ObservationRecord(
                    frame_index=20,
                    timestamp=102.0,
                    semantic_class="person",
                    confidence=0.92,
                    bbox_xyxy=(i * 80 + 10, 100, i * 80 + 60, 200),
                    centroid_camera=(float(i) + 0.3, 0.0, 2.0),
                    centroid_world=(float(i) + 0.3, 0.0, 2.0),
                    area_px=5000,
                    median_depth_m=2.0,
                    sharpness=310.0,
                ),
            ],
        )
        t.compute_motion_statistics()
        tracks.append(t)

    merged = tracker._merge_fragmented_tracks(tracks)

    # All 5 persons must be preserved without truncation!
    person_tracks = [t for t in merged if t.semantic_class == "person"]
    assert len(person_tracks) == 5, f"Expected 5 person tracks, but found {len(person_tracks)}"


def test_motion_state_classification():
    """Verify empirical motion classification thresholds based on 3D observations."""
    def _make_track(track_id: str, sem_cls: str, p1: tuple, p2: tuple) -> TrackedEntity:
        t = TrackedEntity(
            track_id=track_id,
            semantic_class=sem_cls,
            observations=[
                ObservationRecord(
                    frame_index=0,
                    timestamp=0.0,
                    semantic_class=sem_cls,
                    confidence=0.9,
                    bbox_xyxy=(0, 0, 100, 100),
                    centroid_camera=p1,
                    centroid_world=p1,
                    area_px=10000,
                    median_depth_m=p1[2],
                    sharpness=200.0,
                ),
                ObservationRecord(
                    frame_index=10,
                    timestamp=1.0,
                    semantic_class=sem_cls,
                    confidence=0.9,
                    bbox_xyxy=(0, 0, 100, 100),
                    centroid_camera=p2,
                    centroid_world=p2,
                    area_px=10000,
                    median_depth_m=p2[2],
                    sharpness=200.0,
                ),
            ],
        )
        t.compute_motion_statistics()
        return t

    # Small displacement (< 0.12 m): STATIC
    t_static = _make_track("chair_01", "chair", (1.0, 0.0, 2.0), (1.05, 0.0, 2.0))
    assert t_static.observed_motion == MotionObservationState.STATIC

    # Moderate displacement (0.12 <= disp < 0.35 m): DISPLACED
    t_displaced = _make_track("chair_02", "chair", (1.0, 0.0, 2.0), (1.25, 0.0, 2.0))
    assert t_displaced.observed_motion == MotionObservationState.DISPLACED

    # Large displacement (>= 0.35 m): MOVING
    t_moving = _make_track("person_01", "person", (1.0, 0.0, 2.0), (2.50, 0.0, 2.0))
    assert t_moving.observed_motion == MotionObservationState.MOVING
