"""Dataset abstractions and protocols for Estimator-Aware VLM."""

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    parse_tum_trajectory,
)
from estimator_aware_vlm.datasets.tum_rgbd import (
    TUMRGBDAdapter,
    TUM_FREIBURG3_INTRINSICS,
    associate_streams,
    parse_tum_association_file,
    parse_tum_ground_truth,
)

__all__ = [
    "CameraIntrinsics",
    "DatasetAdapter",
    "DatasetConfig",
    "FrameRecord",
    "Trajectory",
    "TrajectoryPose",
    "TrajectoryRole",
    "parse_tum_trajectory",
    "TUMRGBDAdapter",
    "TUM_FREIBURG3_INTRINSICS",
    "associate_streams",
    "parse_tum_association_file",
    "parse_tum_ground_truth",
]
