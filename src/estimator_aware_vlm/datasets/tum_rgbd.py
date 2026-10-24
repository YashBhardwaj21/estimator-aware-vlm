"""TUM RGB-D Dataset Adapter.

Implements ingestion for the TUM RGB-D benchmark format (Sturm et al., IROS 2012),
such as rgbd_dataset_freiburg3_walking_xyz.
Features:
- Precise timestamp association between RGB and depth streams.
- Strict TrajectoryRole.GROUND_TRUTH isolation for groundtruth.txt poses.
- Correct depth scale factor (0.0002 m/unit = 5000 units/m).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)


# Standard TUM Freiburg3 calibrated camera intrinsics
TUM_FREIBURG3_INTRINSICS = CameraIntrinsics(
    fx=535.4,
    fy=539.2,
    cx=320.1,
    cy=247.6,
    width=640,
    height=480,
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    distortion_model="plumb_bob",
)


def parse_tum_association_file(file_path: Path) -> List[Tuple[float, str]]:
    """Parse a two-column TUM file (timestamp filename)."""
    entries: List[Tuple[float, str]] = []
    if not file_path.exists():
        raise FileNotFoundError(f"TUM association file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ts = float(parts[0])
                rel_path = parts[1]
                entries.append((ts, rel_path))

    # Sort strictly by timestamp
    entries.sort(key=lambda x: x[0])
    return entries


def parse_tum_ground_truth(file_path: Path) -> Trajectory:
    """Parse TUM groundtruth.txt file into a Trajectory with TrajectoryRole.GROUND_TRUTH."""
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    poses: List[TrajectoryPose] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                pose = TrajectoryPose(
                    timestamp=ts,
                    position=(tx, ty, tz),
                    orientation=(qx, qy, qz, qw),
                    role=TrajectoryRole.GROUND_TRUTH,
                )
                poses.append(pose)

    poses.sort(key=lambda p: p.timestamp)
    return Trajectory(poses=poses, role=TrajectoryRole.GROUND_TRUTH, name="tum_ground_truth")


def associate_streams(
    first_list: List[Tuple[float, str]],
    second_list: List[Tuple[float, str]],
    max_difference_sec: float = 0.02,
) -> List[Tuple[float, str, float, str]]:
    """Associate two timestamped streams using TUM benchmark nearest-neighbor criteria.
    
    Returns list of tuples: (ts1, path1, ts2, path2).
    """
    first_keys = [x[0] for x in first_list]
    second_keys = [x[0] for x in second_list]
    second_arr = np.array(second_keys, dtype=np.float64)

    matches: List[Tuple[float, str, float, str]] = []
    if len(second_arr) == 0:
        return matches

    for ts1, path1 in first_list:
        diffs = np.abs(second_arr - ts1)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_difference_sec:
            ts2, path2 = second_list[min_idx]
            matches.append((ts1, path1, ts2, path2))

    return matches


class TUMRGBDAdapter(DatasetAdapter):
    """Adapter for TUM RGB-D format datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sequence_dir = Path(config.sequence_dir)
        self.sequence_name = config.sequence_name

        self.rgb_file = config.rgb_file or (self.sequence_dir / "rgb.txt")
        self.depth_file = config.depth_file or (self.sequence_dir / "depth.txt")
        self.groundtruth_file = config.groundtruth_file or (self.sequence_dir / "groundtruth.txt")

        self.depth_scale = config.depth_scale
        self.intrinsics = config.intrinsics or TUM_FREIBURG3_INTRINSICS

        # Load raw streams
        self.rgb_entries = parse_tum_association_file(self.rgb_file)
        self.depth_entries = parse_tum_association_file(self.depth_file)
        
        # Load ground truth trajectory if file exists
        self.ground_truth_trajectory: Optional[Trajectory] = None
        if self.groundtruth_file.exists():
            self.ground_truth_trajectory = parse_tum_ground_truth(self.groundtruth_file)

        # Compute synchronized associations
        self.associated_pairs = associate_streams(
            self.rgb_entries,
            self.depth_entries,
            max_difference_sec=config.association_tolerance_sec,
        )

        # Prebuild synchronized FrameRecords
        self._frames: List[FrameRecord] = []
        for idx, (rgb_ts, rgb_rel, depth_ts, depth_rel) in enumerate(self.associated_pairs):
            rgb_path = self.sequence_dir / rgb_rel
            depth_path = self.sequence_dir / depth_rel

            gt_pose = None
            if self.ground_truth_trajectory is not None:
                gt_pose = self.ground_truth_trajectory.get_pose_at(
                    rgb_ts, max_dt_sec=config.association_tolerance_sec
                )

            self._frames.append(
                FrameRecord(
                    frame_index=idx,
                    timestamp=rgb_ts,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    depth_scale=self.depth_scale,
                    intrinsics=self.intrinsics,
                    estimated_pose=None,
                    ground_truth_pose=gt_pose,
                )
            )

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        return self.intrinsics

    def get_frames(self) -> Iterator[FrameRecord]:
        return iter(self._frames)

    def get_frame(self, index: int) -> FrameRecord:
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Frame index {index} out of range (0..{len(self._frames)-1})")
        return self._frames[index]

    def get_ground_truth_trajectory(self) -> Trajectory:
        if self.ground_truth_trajectory is None:
            raise FileNotFoundError(f"Ground truth trajectory not available for {self.sequence_name}")
        return self.ground_truth_trajectory

    def __len__(self) -> int:
        return len(self._frames)

    def export_associations(self, output_path: Union[str, Path]) -> Path:
        """Export 4-column associations file for ORB-SLAM3: rgb_ts rgb_file depth_ts depth_file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rgb_ts, rgb_rel, depth_ts, depth_rel in self.associated_pairs:
                f.write(f"{rgb_ts:.6f} {rgb_rel} {depth_ts:.6f} {depth_rel}\n")
        return out
