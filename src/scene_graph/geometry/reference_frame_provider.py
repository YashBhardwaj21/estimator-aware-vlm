"""Reference frame provider: constructs the fixed global relation frame.

The relation reference frame defines the semantic meaning of LEFT/RIGHT,
ABOVE/BELOW, and IN_FRONT_OF/BEHIND. It MUST be established from a validated
global convention (e.g., dataset gravity axis, IMU data, or support plane
estimation), NOT from an arbitrary first camera pose.

Each dataset type provides its own convention. The provider is configured
once per sequence and produces a frozen RelationReferenceFrame.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from scene_graph.geometry.reference_frame import RelationReferenceFrame


# Dataset-specific gravity and heading conventions
# TUM RGB-D: Camera looks along +Z, gravity is approximately -Y in world.
# The groundtruth poses are in a world frame where Z is roughly up for
# the freiburg sequences recorded on a desk.
DATASET_CONVENTIONS = {
    "tum_rgbd": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "tum": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "d455": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "realsense": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "world": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "map": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
    "default": {
        "up_axis": np.array([0.0, 0.0, 1.0]),
        "heading_axis": np.array([1.0, 0.0, 0.0]),
    },
}


@dataclass
class ReferenceFrameProvider:
    """Provides the fixed global relation frame for a sequence.
    
    The frame is determined by the dataset type's gravity convention,
    NOT by the first camera pose. This ensures that LEFT/RIGHT, ABOVE/BELOW
    have consistent physical meaning regardless of initial camera orientation.
    """
    
    dataset_type: str
    _frame: Optional[RelationReferenceFrame] = None
    
    def get_frame(self, origin_world: np.ndarray = None) -> RelationReferenceFrame:
        """Return the frozen global relation frame.
        
        Args:
            origin_world: Optional origin point. Defaults to world origin.
                          Only used on first call; subsequent calls return
                          the cached frame.
        """
        if self._frame is not None:
            return self._frame
            
        if origin_world is None:
            origin_world = np.zeros(3)
        
        convention = DATASET_CONVENTIONS.get(self.dataset_type)
        if convention is None:
            raise ValueError(
                f"Unknown dataset type '{self.dataset_type}'. "
                f"Register its gravity/heading convention in DATASET_CONVENTIONS. "
                f"Known types: {list(DATASET_CONVENTIONS.keys())}"
            )
        
        self._frame = RelationReferenceFrame.from_gravity_and_heading(
            origin_world=origin_world,
            up_axis_world=convention["up_axis"].copy(),
            heading_world=convention["heading_axis"].copy(),
        )
        
        return self._frame
