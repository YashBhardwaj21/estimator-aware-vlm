"""Intel RealSense D455 Dataset Adapter.

Implements unified ingestion for RealSense D455 multi-session capture sequences
(e.g., my_desk_recording_01, my_desk_recording_02, and multi-session desk recordings).
Features:
- Precise timestamp association between RealSense D455 RGB and 16-bit depth streams.
- Calibrated D455 factory intrinsics (640x480 resolution, depth scale 0.001 m/unit).
- Support for estimated SLAM trajectories (TrajectoryRole.ESTIMATED).
- Strict ground-truth isolation: when external MoCap is absent, ground_truth_trajectory
  is explicitly None rather than fabricated.
- Canonical FrameRecord interface matching TUMRGBDAdapter and OpenLORISAdapter.
- Enforced raw data immutability: prohibits exporting generated artifacts into raw data folders.
"""

from __future__ import annotations

import json
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
    associate_streams,
    parse_tum_trajectory,
)


# Factory calibrated intrinsics for Intel RealSense D455 (640x480 @ 30fps)
REALSENSE_D455_INTRINSICS = CameraIntrinsics(
    fx=383.9205627441406,
    fy=383.4780578613281,
    cx=320.2670593261719,
    cy=244.56988525390625,
    width=640,
    height=480,
    distortion=(-0.05704856, 0.07136440, 2.77367e-05, 0.00079690, -0.02323733),
    distortion_model="distortion.inverse_brown_conrady",
)


def parse_d455_association_file(file_path: Path) -> List[Tuple[float, str]]:
    """Parse a two-column association file (timestamp relative_path)."""
    entries: List[Tuple[float, str]] = []
    if not file_path.exists():
        raise FileNotFoundError(f"D455 stream file not found: {file_path}")

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


class RealSenseD455Adapter(DatasetAdapter):
    """Adapter for Intel RealSense D455 multi-session capture format."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sequence_dir = Path(config.sequence_dir)
        self.sequence_name = config.sequence_name

        self.rgb_file = config.rgb_file or (self.sequence_dir / "rgb.txt")
        self.depth_file = config.depth_file or (self.sequence_dir / "depth.txt")
        self.associations_file = self.sequence_dir / "associations.txt"

        # Depth scale: RealSense D455 depth is 1000 units/meter (0.001 m/unit)
        self.depth_scale = config.depth_scale if config.depth_scale != 0.0002 else 0.001
        self.intrinsics = config.intrinsics or REALSENSE_D455_INTRINSICS

        # Check for camera_intrinsics.json override if present in sequence directory
        intr_json = self.sequence_dir / "camera_intrinsics.json"
        if intr_json.exists() and config.intrinsics is None:
            try:
                with open(intr_json, "r", encoding="utf-8") as f:
                    ij = json.load(f)
                    self.intrinsics = CameraIntrinsics(
                        fx=float(ij.get("fx", 383.92)),
                        fy=float(ij.get("fy", 383.48)),
                        cx=float(ij.get("cx", 320.27)),
                        cy=float(ij.get("cy", 244.57)),
                        width=int(ij.get("width", 640)),
                        height=int(ij.get("height", 480)),
                        distortion=tuple(float(c) for c in ij.get("coeffs", [-0.057, 0.071, 0.0, 0.0, -0.023])),
                        distortion_model=str(ij.get("distortion_model", "distortion.inverse_brown_conrady")),
                    )
            except Exception:
                pass

        # Load raw stream index entries
        if self.rgb_file.exists():
            self.rgb_entries = parse_d455_association_file(self.rgb_file)
        else:
            self.rgb_entries = []

        if self.depth_file.exists():
            self.depth_entries = parse_d455_association_file(self.depth_file)
        else:
            self.depth_entries = []

        # Load pre-computed associations or compute online
        self.associated_pairs: List[Tuple[float, str, float, str]] = []
        if self.associations_file.exists():
            with open(self.associations_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        r_ts, r_path, d_ts, d_path = float(parts[0]), parts[1], float(parts[2]), parts[3]
                        self.associated_pairs.append((r_ts, r_path, d_ts, d_path))
        elif self.rgb_entries and self.depth_entries:
            raw_pairs = associate_streams(
                self.rgb_entries,
                self.depth_entries,
                max_difference_sec=config.association_tolerance_sec,
                one_to_one=True,
            )
            self.associated_pairs = raw_pairs

        # Filter to pairs that exist on disk
        self.associated_pairs = [
            pair for pair in self.associated_pairs
            if (self.sequence_dir / pair[1]).exists() and (self.sequence_dir / pair[3]).exists()
        ]

        # Check for genuine ground truth (strictly avoid identity placeholders)
        self.ground_truth_trajectory: Optional[Trajectory] = None
        gt_file = config.groundtruth_file or (self.sequence_dir / "groundtruth.txt")
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                header_text = f.read(500)
            if "SYNTHETIC" not in header_text and "PLACEHOLDER" not in header_text:
                try:
                    self.ground_truth_trajectory = parse_tum_trajectory(
                        gt_file, role=TrajectoryRole.GROUND_TRUTH, name=f"{self.sequence_name}_gt"
                    )
                except Exception:
                    self.ground_truth_trajectory = None

        # Check for estimated trajectory in semantic mapping run directories
        self.estimated_trajectory: Optional[Trajectory] = None
        est_candidates = [
            self.sequence_dir / "CameraTrajectory.txt",
            self.sequence_dir.parent.parent / f"semantic_mapping_{self.sequence_name}" / "trajectory" / "CameraTrajectory.txt",
        ]
        for ec in est_candidates:
            if ec.exists():
                try:
                    self.estimated_trajectory = parse_tum_trajectory(
                        ec, role=TrajectoryRole.ESTIMATED, name=f"{self.sequence_name}_estimated"
                    )
                    break
                except Exception:
                    pass

        # Build FrameRecord list
        self._frames: List[FrameRecord] = []
        for idx, (rgb_ts, rgb_rel, depth_ts, depth_rel) in enumerate(self.associated_pairs):
            rgb_path = self.sequence_dir / rgb_rel
            depth_path = self.sequence_dir / depth_rel

            gt_pose = None
            if self.ground_truth_trajectory is not None:
                gt_pose = self.ground_truth_trajectory.get_pose_at(
                    rgb_ts, max_dt_sec=config.association_tolerance_sec
                )

            est_pose = None
            if self.estimated_trajectory is not None:
                est_pose = self.estimated_trajectory.get_pose_at(
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
                    estimated_pose=est_pose,
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
            raise FileNotFoundError(
                f"Ground-truth camera trajectory is not available for {self.sequence_name}. "
                "This sequence was captured with RealSense D455 without external motion capture reference."
            )
        return self.ground_truth_trajectory

    def get_estimated_trajectory(self) -> Optional[Trajectory]:
        return self.estimated_trajectory

    def __len__(self) -> int:
        return len(self._frames)

    def export_associations(self, output_path: Union[str, Path]) -> Path:
        """Export associations file for SLAM processing."""
        out = Path(output_path).resolve()
        seq_res = self.sequence_dir.resolve()
        try:
            if out.is_relative_to(seq_res):
                raise PermissionError(
                    f"Immutability violation: Cannot write generated associations into "
                    f"raw sequence directory '{self.sequence_dir}'."
                )
        except AttributeError:
            if str(out).startswith(str(seq_res)):
                raise PermissionError("Immutability violation: Cannot write generated associations into raw sequence directory.")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rgb_ts, rgb_rel, depth_ts, depth_rel in self.associated_pairs:
                f.write(f"{rgb_ts:.6f} {rgb_rel} {depth_ts:.6f} {depth_rel}\n")
        return out

