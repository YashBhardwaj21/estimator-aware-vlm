"""Unit tests for temporal association, kinematics estimation, and motion decoupling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from estimator_aware_vlm.datasets.base import CameraIntrinsics, TrajectoryPose
from estimator_aware_vlm.perception.sequence_tracker import (
    MotionObservationState,
    ObservationRecord,
    SequenceTracker,
    TrackedEntity,
    bhattacharyya_distance,
    compute_box_iou,
    compute_color_histogram,
)


@pytest.fixture
def mock_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )


def test_compute_box_iou():
    box_a = (100.0, 100.0, 200.0, 200.0)
    box_b = (100.0, 100.0, 200.0, 200.0)
    assert pytest.approx(compute_box_iou(box_a, box_b), abs=1e-5) == 1.0

    box_c = (300.0, 300.0, 400.0, 400.0)
    assert pytest.approx(compute_box_iou(box_a, box_c), abs=1e-5) == 0.0

    box_d = (150.0, 100.0, 250.0, 200.0)
    assert pytest.approx(compute_box_iou(box_a, box_d), abs=1e-3) == 1.0 / 3.0


def test_color_histogram_and_bhattacharyya_distance():
    red_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    red_crop[:, :, 2] = 255

    blue_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    blue_crop[:, :, 0] = 255

    hist_red1 = compute_color_histogram(red_crop, (0, 0, 50, 50))
    hist_red2 = compute_color_histogram(red_crop, (0, 0, 50, 50))
    hist_blue = compute_color_histogram(blue_crop, (0, 0, 50, 50))

    assert pytest.approx(bhattacharyya_distance(hist_red1, hist_red2), abs=1e-4) == 0.0
    assert bhattacharyya_distance(hist_red1, hist_blue) > 0.5


def test_apparent_vs_physical_motion_decoupling():
    """Verify that camera ego-motion does not cause stationary objects to be classified as MOVING."""
    track = TrackedEntity(
        track_id="monitor_01",
        semantic_class="monitor",
        observations=[],
    )

    timestamps = [10.0 + i * 0.1 for i in range(25)]
    cam_x = [0.05 * i for i in range(25)]

    for i, t in enumerate(timestamps):
        c_x = cam_x[i]
        cam_pos = (c_x, 0.0, 0.0)
        p_world = (1.50 + 0.005 * math.sin(i), 0.20, 0.80)
        p_cam = (p_world[0] - c_x, p_world[1], p_world[2])

        obs = ObservationRecord(
            frame_index=i,
            timestamp=t,
            semantic_class="monitor",
            confidence=0.85,
            bbox_xyxy=(200, 150, 350, 300),
            centroid_camera=p_cam,
            centroid_world=p_world,
            area_px=22500,
            median_depth_m=float(p_cam[2]),
            sharpness=100.0,
            camera_position_world=cam_pos,
            dimensions_3d_m=(0.60, 0.40, 0.10),
            track_id="monitor_01",
        )
        track.observations.append(obs)

    track.compute_motion_statistics()

    assert track.observed_motion == MotionObservationState.STATIC
    assert track.net_world_displacement_m < 0.20
    assert track.path_length_camera_m > 1.0
    assert track.path_length_camera_m != track.path_length_world_m


def test_moving_object_kinematic_classification():
    """Verify that genuinely moving people with directional persistence are classified as MOVING."""
    track = TrackedEntity(
        track_id="person_01",
        semantic_class="person",
        observations=[],
    )

    timestamps = [10.0 + i * 0.1 for i in range(30)]

    for i, t in enumerate(timestamps):
        p_world = (0.05 * i, 1.0, 0.0)
        p_cam = p_world

        obs = ObservationRecord(
            frame_index=i,
            timestamp=t,
            semantic_class="person",
            confidence=0.90,
            bbox_xyxy=(100, 100, 200, 350),
            centroid_camera=p_cam,
            centroid_world=p_world,
            area_px=25000,
            median_depth_m=1.0,
            sharpness=120.0,
            camera_position_world=(0.0, 0.0, 0.0),
            dimensions_3d_m=(0.50, 0.40, 1.70),
            track_id="person_01",
        )
        track.observations.append(obs)

    track.compute_motion_statistics()

    assert track.observed_motion == MotionObservationState.MOVING
    assert track.net_world_displacement_m > 1.20
    assert track.mean_velocity_mps > 0.40
    assert track.motion_direction_consistency > 0.90
    assert track.path_efficiency > 0.85


def test_displaced_object_classification():
    """Verify that objects that move and then remain stationary are classified as DISPLACED."""
    track = TrackedEntity(
        track_id="chair_01",
        semantic_class="chair",
        observations=[],
    )

    timestamps = [10.0 + i * 0.1 for i in range(40)]

    for i, t in enumerate(timestamps):
        if i < 15:
            x_w = 0.0
        elif i < 20:
            x_w = 0.15 * (i - 15)
        else:
            x_w = 0.75

        p_world = (x_w, 2.0, 0.0)
        p_cam = p_world

        obs = ObservationRecord(
            frame_index=i,
            timestamp=t,
            semantic_class="chair",
            confidence=0.88,
            bbox_xyxy=(150, 150, 250, 280),
            centroid_camera=p_cam,
            centroid_world=p_world,
            area_px=13000,
            median_depth_m=2.0,
            sharpness=95.0,
            camera_position_world=(0.0, 0.0, 0.0),
            dimensions_3d_m=(0.60, 0.60, 0.80),
            track_id="chair_01",
        )
        track.observations.append(obs)

    track.compute_motion_statistics()

    assert track.observed_motion == MotionObservationState.DISPLACED
    assert pytest.approx(track.net_world_displacement_m, abs=0.05) == 0.75


def test_multi_cue_association_cost(mock_intrinsics: CameraIntrinsics):
    """Verify association cost calculation behavior."""
    tracker = SequenceTracker(
        intrinsics=mock_intrinsics,
        min_conf_threshold=0.40,
        association_dist_threshold_m=0.85,
    )

    track = TrackedEntity(track_id="track_01", semantic_class="chair", observations=[])
    obs1 = ObservationRecord(
        frame_index=1,
        timestamp=1.0,
        semantic_class="chair",
        confidence=0.85,
        bbox_xyxy=(100, 100, 200, 200),
        centroid_camera=(0.0, 0.0, 2.0),
        centroid_world=(0.0, 0.0, 2.0),
        area_px=10000,
        median_depth_m=2.0,
        sharpness=110.0,
        camera_position_world=(0.0, 0.0, 0.0),
        dimensions_3d_m=(0.5, 0.5, 0.8),
        track_id="track_01",
    )
    track.observations.append(obs1)

    matching_cand = ObservationRecord(
        frame_index=2,
        timestamp=1.05,
        semantic_class="chair",
        confidence=0.88,
        bbox_xyxy=(105, 102, 205, 202),
        centroid_camera=(0.02, 0.01, 2.01),
        centroid_world=(0.02, 0.01, 2.01),
        area_px=10000,
        median_depth_m=2.01,
        sharpness=110.0,
        camera_position_world=(0.0, 0.0, 0.0),
        dimensions_3d_m=(0.51, 0.49, 0.81),
        track_id=None,
    )

    cost_match = tracker._calculate_association_cost(
        cand=matching_cand,
        track=track,
    )
    assert cost_match < 0.25

    mismatch_cand = ObservationRecord(
        frame_index=2,
        timestamp=1.05,
        semantic_class="person",
        confidence=0.88,
        bbox_xyxy=(105, 102, 205, 202),
        centroid_camera=(0.02, 0.01, 2.01),
        centroid_world=(0.02, 0.01, 2.01),
        area_px=10000,
        median_depth_m=2.01,
        sharpness=110.0,
        camera_position_world=(0.0, 0.0, 0.0),
        dimensions_3d_m=(0.51, 0.49, 0.81),
        track_id=None,
    )

    cost_mismatch = tracker._calculate_association_cost(
        cand=mismatch_cand,
        track=track,
    )
    assert cost_mismatch == float("inf")


def test_deliverables_schema_and_exit_condition():
    """Verify that deliverables exist, adhere to schema, and satisfy the exit condition."""
    output_dir = Path("results/tracking")
    tracks_file = output_dir / "tracks_3d.jsonl"
    obs_file = output_dir / "tracking_observations.jsonl"
    metrics_file = output_dir / "tracking_metrics.json"
    report_file = output_dir / "motion_analysis_report.md"

    if not tracks_file.exists():
        pytest.skip("Deliverables not yet generated")

    assert tracks_file.stat().st_size > 0
    assert obs_file.stat().st_size > 0
    assert metrics_file.stat().st_size > 0
    assert report_file.stat().st_size > 0

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "sequences" in metrics
    assert "global" in metrics
    assert len(metrics["sequences"]) == 8

    # Exit condition: Stationary objects observed from moving camera in tum_xyz
    # are not systematically classified as moving.
    xyz_tracks = []
    with open(tracks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            seq = t.get("sequence_id") or t["track_id"].split("__")[0]
            if seq == "tum_xyz":
                xyz_tracks.append(t)

    assert len(xyz_tracks) > 0
    stationary_labels = {"monitor", "chair", "keyboard", "mouse", "laptop", "cup", "book"}
    moving_furniture = [
        t for t in xyz_tracks
        if t["semantic_class"] in stationary_labels and t.get("observed_motion") == "MOVING"
    ]
    assert len(moving_furniture) == 0, f"Found misclassified stationary objects in tum_xyz: {moving_furniture}"
