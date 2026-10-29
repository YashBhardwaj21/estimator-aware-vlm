"""OpenLORIS-Scene Dataset Adapter.

Implements ingestion for the OpenLORIS-Scene benchmark format (Shi et al., ICRA 2020),
such as OpenLORIS-Scene_corridor1-4.
Features:
- Precise 1-to-1 timestamp association between RealSense D400 color and aligned depth streams.
- Strict TrajectoryRole.GROUND_TRUTH isolation for groundtruth.txt poses (LiDAR/MoCap).
- Accurate RealSense D400 depth scale factor (0.001 m/unit = 1000 units/m).
- Canonical FrameRecord interface matching TUMRGBDAdapter for full dataset independence.
- Enforced raw data immutability: prohibits exporting generated artifacts into raw data folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    associate_streams,
)


# Calibrated camera intrinsics for RealSense D400 color frame (848x480 @ 30fps)
OPENLORIS_D400_INTRINSICS = CameraIntrinsics(
    fx=611.45098876953125,
    fy=611.48571777343750,
    cx=433.20397949218750,
    cy=249.47302246093750,
    width=848,
    height=480,
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    distortion_model="plumb_bob",
)


def parse_openloris_association_file(file_path: Path) -> List[Tuple[float, str]]:
    """Parse an OpenLORIS two-column stream file (timestamp relative_path)."""
    entries: List[Tuple[float, str]] = []
    if not file_path.exists():
        raise FileNotFoundError(f"OpenLORIS association file not found: {file_path}")

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

    entries.sort(key=lambda x: x[0])
    return entries


def parse_openloris_ground_truth(file_path: Path) -> Trajectory:
    """Parse OpenLORIS groundtruth.txt (#Time px py pz qx qy qz qw) with TrajectoryRole.GROUND_TRUTH."""
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
    return Trajectory(poses=poses, role=TrajectoryRole.GROUND_TRUTH, name="openloris_ground_truth")


class OpenLORISAdapter(DatasetAdapter):
    """Adapter for OpenLORIS-Scene format datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sequence_dir = Path(config.sequence_dir)
        self.sequence_name = config.sequence_name

        self.color_file = config.rgb_file or (self.sequence_dir / "color.txt")
        self.depth_file = config.depth_file or (self.sequence_dir / "aligned_depth.txt")
        self.groundtruth_file = config.groundtruth_file or (self.sequence_dir / "groundtruth.txt")

        # Depth scale: 0.001 m/unit (1000 units/m) for RealSense D400 aligned depth
        self.depth_scale = config.depth_scale if config.depth_scale != 0.0002 else 0.001
        self.intrinsics = config.intrinsics or OPENLORIS_D400_INTRINSICS

        # Load raw streams
        self.color_entries = parse_openloris_association_file(self.color_file)
        self.depth_entries = parse_openloris_association_file(self.depth_file)

        # Load ground truth trajectory if file exists
        self.ground_truth_trajectory: Optional[Trajectory] = None
        if self.groundtruth_file.exists():
            self.ground_truth_trajectory = parse_openloris_ground_truth(self.groundtruth_file)

        # Compute strict 1-to-1 synchronized associations
        raw_pairs = associate_streams(
            self.color_entries,
            self.depth_entries,
            max_difference_sec=config.association_tolerance_sec,
            one_to_one=True,
        )

        # Filter strictly by disk existence (guards against rosbag truncations)
        self.associated_pairs = [
            pair for pair in raw_pairs
            if (self.sequence_dir / pair[1]).exists() and (self.sequence_dir / pair[3]).exists()
        ]

        # Prebuild synchronized FrameRecords
        self._frames: List[FrameRecord] = []
        for idx, (color_ts, color_rel, depth_ts, depth_rel) in enumerate(self.associated_pairs):
            color_path = self.sequence_dir / color_rel
            depth_path = self.sequence_dir / depth_rel

            gt_pose = None
            if self.ground_truth_trajectory is not None:
                gt_pose = self.ground_truth_trajectory.get_pose_at(
                    color_ts, max_dt_sec=config.association_tolerance_sec
                )

            self._frames.append(
                FrameRecord(
                    frame_index=idx,
                    timestamp=color_ts,
                    rgb_path=color_path,
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
        """Export 4-column associations file for ORB-SLAM3: rgb_ts rgb_file depth_ts depth_file.
        
        Strictly prohibits writing into raw sequence data directory to guarantee immutability.
        """
        out = Path(output_path).resolve()
        seq_res = self.sequence_dir.resolve()
        try:
            if out.is_relative_to(seq_res):
                raise PermissionError(
                    f"Immutability violation: Cannot write generated associations into "
                    f"raw sequence directory '{self.sequence_dir}'. Export to outputs/ instead."
                )
        except AttributeError:
            # Python < 3.9 fallback
            if str(out).startswith(str(seq_res)):
                raise PermissionError(
                    f"Immutability violation: Cannot write generated associations into raw sequence directory."
                )

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for color_ts, color_rel, depth_ts, depth_rel in self.associated_pairs:
                f.write(f"{color_ts:.6f} {color_rel} {depth_ts:.6f} {depth_rel}\n")
        return out
