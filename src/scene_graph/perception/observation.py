"""Observation dataclass and mask serialization (Substage 2.1 & 2.2)."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from scene_graph.geometry.point_cloud import ObjectGeometry


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encode a 2D boolean or uint8 mask to Run-Length Encoding (RLE).
    
    Format matches standard COCO RLE representation for portability.
    
    Args:
        mask: 2D numpy array (H, W) where non-zero is foreground.
        
    Returns:
        dict with keys: 'counts' (list of ints), 'size' (list [H, W]).
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        
    # Fortran order (column-major) is standard for COCO RLE
    flat_mask = np.asfortranarray(mask > 0).flatten(order='F')
    
    # Find indices where value changes
    diffs = np.diff(flat_mask)
    change_indices = np.where(diffs)[0] + 1
    
    # Append start and end
    indices = np.concatenate(([0], change_indices, [len(flat_mask)]))
    
    # Compute run lengths
    counts = np.diff(indices).tolist()
    
    # RLE always starts with the count of background pixels.
    # If the very first pixel is foreground, the first background count is 0.
    if flat_mask[0]:
        counts = [0] + counts
        
    return {
        "counts": counts,
        "size": [mask.shape[0], mask.shape[1]]
    }


def decode_mask_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode RLE dict back to 2D boolean mask.
    
    Args:
        rle: dict with 'counts' and 'size'
        
    Returns:
        2D boolean numpy array (H, W).
    """
    counts = rle["counts"]
    h, w = rle["size"]
    
    # Ensure total counts match image size
    if sum(counts) != h * w:
        raise ValueError(f"RLE counts sum {sum(counts)} != size {h * w}")
        
    # Alternate background (False) and foreground (True)
    val = False
    flat_mask = np.zeros(h * w, dtype=bool)
    
    idx = 0
    for count in counts:
        if count > 0:
            flat_mask[idx:idx + count] = val
            idx += count
        val = not val
        
    # Reshape back to (H, W) using Fortran order
    return flat_mask.reshape((h, w), order='F')


@dataclass
class Observation:
    """A single object detected in an image.
    
    Separates the detection (what the model saw) from the tracking state
    (what the graph believes over time).
    """
    obs_id: str
    frame_index: int
    timestamp: float
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray          # (4,) [x_min, y_min, x_max, y_max]
    
    # Mask is stored as RLE for ultralytics-independence and efficient JSON serialization
    mask_rle: Optional[Dict[str, Any]]
    
    # Optional path to heavy point cloud data if serialized
    point_cloud_ref: Optional[str] = None
    
    # Populated downstream by geometry subsystem
    object_geometry: Optional[ObjectGeometry] = None
    
    def get_mask(self) -> Optional[np.ndarray]:
        """Decode and return the boolean mask if available."""
        if self.mask_rle is None:
            return None
        return decode_mask_rle(self.mask_rle)
