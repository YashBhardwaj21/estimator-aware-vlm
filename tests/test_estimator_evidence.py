"""Unit tests for estimator evidence extraction, 3D tracking, and crop harvesting."""

from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.datasets import (
    CameraIntrinsics,
    TUM_FREIBURG3_INTRINSICS,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.perception import (
    MotionObservationState,
    SequenceTracker,
    TrackedEntity,
)
from estimator_aware_vlm.perception.sequence_tracker import ObservationRecord


def test_motion_observation_classification():
    """Verify 3-tier motion classification: STATIC, DISPLACED, MOVING."""
    # 1. Static object (< 0.10 m displacement)
    obs_static = [
        ObservationRecord(
            frame_index=0,
            timestamp=0.0,
            semantic_class="person",
            confidence=0.90,
            bbox_xyxy=(10, 10, 50, 50),
            centroid_camera=(0.0, 0.0, 1.0),
            centroid_world=(1.00, 2.00, 0.50),
            area_px=1600,
            median_depth_m=1.0,
            sharpness=100.0,
            rgb_path=Path("dummy.png"),
        ),
        ObservationRecord(
            frame_index=1,
            timestamp=0.03,
            semantic_class="person",
            confidence=0.90,
            bbox_xyxy=(10, 10, 50, 50),
            centroid_camera=(0.0, 0.0, 1.0),
            centroid_world=(1.02, 2.01, 0.50),  # ~0.022 m displacement
            area_px=1600,
            median_depth_m=1.0,
            sharpness=100.0,
            rgb_path=Path("dummy.png"),
        ),
    ]
    track_s = TrackedEntity(track_id="t_static", observations=obs_static)
    track_s.compute_motion_statistics()
    assert track_s.observed_motion == MotionObservationState.STATIC
    assert track_s.measured_displacement_m < 0.10

    # 2. Displaced object (0.10 m <= disp < 0.30 m)
    obs_displaced = [
        obs_static[0],
        ObservationRecord(
            frame_index=5,
            timestamp=0.15,
            semantic_class="person",
            confidence=0.90,
            bbox_xyxy=(10, 10, 50, 50),
            centroid_camera=(0.0, 0.0, 1.0),
            centroid_world=(1.15, 2.05, 0.50),  # ~0.158 m displacement
            area_px=1600,
            median_depth_m=1.0,
            sharpness=100.0,
            rgb_path=Path("dummy.png"),
        ),
    ]
    track_d = TrackedEntity(track_id="t_displaced", observations=obs_displaced)
    track_d.compute_motion_statistics()
    assert track_d.observed_motion == MotionObservationState.DISPLACED
    assert 0.10 <= track_d.measured_displacement_m < 0.30

    # 3. Actively moving object (disp >= 0.30 m)
    obs_moving = [
        obs_static[0],
        ObservationRecord(
            frame_index=10,
            timestamp=0.30,
            semantic_class="person",
            confidence=0.90,
            bbox_xyxy=(10, 10, 50, 50),
            centroid_camera=(0.0, 0.0, 1.0),
            centroid_world=(1.80, 2.50, 0.50),  # ~0.943 m displacement
            area_px=1600,
            median_depth_m=1.0,
            sharpness=100.0,
            rgb_path=Path("dummy.png"),
        ),
    ]
    track_m = TrackedEntity(track_id="t_moving", observations=obs_moving)
    track_m.compute_motion_statistics()
    assert track_m.observed_motion == MotionObservationState.MOVING
    assert track_m.measured_displacement_m >= 0.30


def test_tracker_role_enforcement():
    """Verify SequenceTracker rejects ground truth poses passed as estimated input."""
    tracker = SequenceTracker(intrinsics=TUM_FREIBURG3_INTRINSICS)
    gt_pose = TrajectoryPose(
        timestamp=0.0,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        role=TrajectoryRole.GROUND_TRUTH,
    )
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.uint16)

    with pytest.raises(PermissionError, match="TrajectoryPose role violation"):
        tracker.detect_frame_candidates(
            rgb_img=dummy_rgb,
            depth_img=dummy_depth,
            depth_scale=0.0002,
            camera_pose=gt_pose,
            frame_idx=0,
            timestamp=0.0,
            rgb_path=Path("dummy.png"),
        )


def test_camera_ego_motion_decoupling():
    """Verify camera movement does not cause a stationary object to be classified as MOVING."""
    # Stationary object at (2.0, 1.0, 0.0) in the room
    fixed_world_pos = (2.0, 1.0, 0.0)

    # Observation 1: Camera at (0.0, 0.0, 0.0) -> object is at (2.0, 1.0, 0.0) relative to camera
    obs1 = ObservationRecord(
        frame_index=0,
        timestamp=0.0,
        semantic_class="chair",
        confidence=0.88,
        bbox_xyxy=(100, 100, 200, 200),
        centroid_camera=(2.0, 1.0, 0.0),
        centroid_world=fixed_world_pos,
        area_px=10000,
        median_depth_m=2.23,
        sharpness=80.0,
        rgb_path=Path("dummy1.png"),
    )

    # Observation 2: Camera translates by 1.5 meters to (1.5, 0.0, 0.0)
    # Relative to camera, object shifted to (0.5, 1.0, 0.0) -> apparent displacement is 1.5 m!
    # BUT in world frame, object position is still (2.0, 1.0, 0.0)!
    obs2 = ObservationRecord(
        frame_index=30,
        timestamp=1.0,
        semantic_class="chair",
        confidence=0.91,
        bbox_xyxy=(300, 100, 400, 200),
        centroid_camera=(0.5, 1.0, 0.0),
        centroid_world=fixed_world_pos,
        area_px=10000,
        median_depth_m=1.11,
        sharpness=85.0,
        rgb_path=Path("dummy2.png"),
    )

    track = TrackedEntity(track_id="fixed_chair", semantic_class="chair", observations=[obs1, obs2])
    track.compute_motion_statistics()

    # The apparent camera-frame displacement is 1.5 m due to camera ego-motion
    assert track.apparent_camera_frame_displacement_m == 1.5

    # BUT the world-frame displacement is 0.0 m, so the object remains correctly STATIC!
    assert track.measured_displacement_m == 0.0
    assert track.observed_motion == MotionObservationState.STATIC
