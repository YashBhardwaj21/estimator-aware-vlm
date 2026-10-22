"""Stage 5: Geometric Evidence Extraction for Object Observations.

Extracts:
1. Physical RealSense D455 stereo depth uncertainty:
   sigma_z = (z^2 / (f_x * B)) * sigma_d
2. MapPoint-to-2D-bounding-box geometric consistency:
   point_to_bbox_distance_px and point_to_bbox_inlier_ratio.
   (Clearly distinguished from feature-matching reprojection residuals, which
    require 2D-3D keypoint correspondences that are not output by 2D box detectors).
3. Explicit evidence provenance: MEASURED_FULL, DERIVED_DEPTH_ONLY, MISSING_FLOOR_TRANSFORM,
   MISSING_SESSION_MAP, or MISSING.
4. Multi-session MapPoints and SE(3) floor plane transform management.
5. Calibrated geometric consistency score bounded in [0.0, 1.0].
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Intel RealSense D455 Factory Intrinsics (640x480 resolution mode)
FX = 383.924896
FY = 383.924896
CX = 320.123016
CY = 244.569885
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
STEREO_BASELINE_M = 0.095  # 95 mm physical baseline between IR stereo sensors
DISPARITY_NOISE_PX = 0.08  # Subpixel disparity matching noise (sigma_d)
MIN_VALID_DEPTH_M = 0.20
MAX_VALID_DEPTH_M = 4.80


@dataclass
class GeometricEvidenceRecord:
    """Timestamp-aligned geometric evidence record for a single object observation."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    source_session: str
    depth_valid: bool
    measured_depth_m: Optional[float]
    depth_uncertainty_sigma_m: Optional[float]
    depth_provenance: str  # "MEASURED_SENSOR" or "MISSING"

    # Point-to-Bounding-Box Consistency (honestly named, not feature reprojection)
    point_to_bbox_valid: bool
    point_to_bbox_provenance: str  # "MEASURED_MAPPOINTS", "MISSING_MAPPOINTS", "MISSING_FLOOR_TRANSFORM"
    point_to_bbox_distance_px: Optional[float]  # Distance outside box in pixels (0 for inliers)
    point_to_bbox_inlier_ratio: Optional[float]  # Fraction of candidate MapPoints projecting inside box
    mappoints_candidate_count: int
    mappoints_inlier_count: int

    # Feature-Level Reprojection Residuals (Requires 2D keypoint-to-3D MapPoint matches)
    feature_reprojection_residual_valid: bool
    feature_reprojection_residual_px: Optional[float]
    feature_reprojection_provenance: str  # "UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES"

    # Spatial support & calibration
    landmark_support_count: int
    landmark_spatial_extent_m: float
    geometric_consistency_score: float
    provenance: str  # "MEASURED_FULL", "DERIVED_DEPTH_ONLY", "MISSING_FLOOR_TRANSFORM", "MISSING"
    has_valid_evidence: bool

    # Backwards-compatibility properties for existing downstream consumers
    @property
    def inlier_ratio(self) -> Optional[float]:
        return self.point_to_bbox_inlier_ratio

    @property
    def mean_reprojection_residual_px(self) -> Optional[float]:
        return self.point_to_bbox_distance_px

    @property
    def median_reprojection_residual_px(self) -> Optional[float]:
        return self.point_to_bbox_distance_px

    @property
    def p95_reprojection_residual_px(self) -> Optional[float]:
        return self.point_to_bbox_distance_px

    @property
    def reprojection_residual_valid(self) -> bool:
        return self.point_to_bbox_valid

    @property
    def inlier_count(self) -> int:
        return self.mappoints_inlier_count

    @property
    def mappoints_count(self) -> int:
        return self.mappoints_candidate_count

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep backwards-compatible keys in dictionary representation
        d["inlier_ratio"] = self.point_to_bbox_inlier_ratio
        d["mean_reprojection_residual_px"] = self.point_to_bbox_distance_px
        d["median_reprojection_residual_px"] = self.point_to_bbox_distance_px
        d["p95_reprojection_residual_px"] = self.point_to_bbox_distance_px
        d["reprojection_residual_valid"] = self.point_to_bbox_valid
        d["inlier_count"] = self.mappoints_inlier_count
        d["mappoints_count"] = self.mappoints_candidate_count
        return d


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion to 3x3 orthonormal rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]], dtype=np.float64)


def invert_se3_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert SE(3) transformation matrix: [R | t] -> [R^T | -R^T * t]."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def project_points_to_camera(
    points_world: np.ndarray,
    cam_pose: List[float],
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D world points into camera pixel coordinates.

    Args:
        points_world: (N, 3) array of world coordinates.
        cam_pose: 7-element list [tx, ty, tz, qx, qy, qz, qw] representing T_world_camera.

    Returns:
        pixels: (N, 2) pixel coordinates (u, v).
        valid_z: (N,) boolean mask indicating points in front of camera (Z > 0.15m).
        points_cam: (N, 3) points in camera reference frame.
    """
    if len(points_world) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=bool), np.empty((0, 3))

    t_wc = np.array(cam_pose[:3], dtype=np.float64)
    R_wc = quaternion_to_rotation_matrix(cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6])
    R_cw, t_cw = invert_se3_transform(R_wc, t_wc)

    points_cam = (R_cw @ points_world.T).T + t_cw
    valid_z = points_cam[:, 2] > 0.15

    pixels = np.zeros((len(points_world), 2), dtype=np.float64)
    z_safe = np.where(valid_z, points_cam[:, 2], 1.0)
    pixels[:, 0] = fx * (points_cam[:, 0] / z_safe) + cx
    pixels[:, 1] = fy * (points_cam[:, 1] / z_safe) + cy

    return pixels, valid_z, points_cam


def compute_point_to_bbox_distances(
    pixels: np.ndarray,
    bbox_xyxy: List[float],
    valid_z: np.ndarray,
    margin_px: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances outside 2D bounding box and inlier mask.

    Points inside [bx1, by1, bx2, by2] have distance 0.0 px.
    Points outside have Euclidean distance to the nearest box edge.
    """
    if len(pixels) == 0 or not np.any(valid_z):
        return np.empty(0), np.empty(0, dtype=bool)

    bx1, by1, bx2, by2 = bbox_xyxy
    u = pixels[valid_z, 0]
    v = pixels[valid_z, 1]

    du = np.maximum(0.0, np.maximum(bx1 - u, u - bx2))
    dv = np.maximum(0.0, np.maximum(by1 - v, v - by2))
    distances = np.sqrt(du**2 + dv**2)

    inlier_mask = (u >= bx1 - margin_px) & (u <= bx2 + margin_px) & (v >= by1 - margin_px) & (v <= by2 + margin_px)
    return distances, inlier_mask


def compute_depth_uncertainty(depth_z: float) -> float:
    """Compute physical depth uncertainty (1-sigma) for stereo triangulation.

    Formula: sigma_z = (z^2 / (f_x * B)) * sigma_d
    """
    if depth_z <= 0.05:
        return 0.10
    z = max(0.15, min(depth_z, 6.0))
    sigma_z = (z * z / (FX * STEREO_BASELINE_M)) * DISPARITY_NOISE_PX
    return float(max(0.001, sigma_z))


class GeometricEvidenceExtractor:
    """Extracts geometric consistency evidence for object observations."""

    def __init__(
        self,
        mappoints_path: Optional[Path] = None,
        floor_plane_path: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ):
        self.session_data: Dict[str, Dict[str, Any]] = {}

        # Default single map support
        self.mappoints = np.empty((0, 3), dtype=np.float64)
        self.T_world_floor = np.eye(4, dtype=np.float64)
        self.has_floor_transform = False

        if repo_root is None:
            # Guess repo_root from current file
            repo_root = Path(__file__).resolve().parent.parent.parent

        # Initialize multi-session map storage if available
        self._discover_and_load_sessions(repo_root)

        # Fallback manual paths
        if mappoints_path and Path(mappoints_path).exists():
            self._load_single_mappoints_ply(Path(mappoints_path))
        if floor_plane_path and Path(floor_plane_path).exists():
            self._load_single_floor_plane(Path(floor_plane_path))

    def _discover_and_load_sessions(self, repo_root: Path):
        """Discover and load session-specific MapPoints and floor planes."""
        cand_sessions = [
            "semantic_mapping_my_desk_recording_01",
            "semantic_mapping_my_desk_recording_02",
        ]
        for sname in cand_sessions:
            s_dir = repo_root / "outputs" / sname
            if not s_dir.exists():
                continue

            pts = np.empty((0, 3), dtype=np.float64)
            ply_path = s_dir / "trajectory" / "MapPoints.ply"
            if ply_path.exists():
                pts = self._read_ply_points(ply_path)

            T_wf = np.eye(4, dtype=np.float64)
            has_fp = False
            fp_path = s_dir / "floor_plane.json"
            if fp_path.exists():
                try:
                    with open(fp_path, "r", encoding="utf-8") as f:
                        fp_data = json.load(f)
                        if "T_world_floor" in fp_data:
                            T_wf = np.array(fp_data["T_world_floor"], dtype=np.float64)
                            has_fp = True
                except Exception:
                    pass

            self.session_data[sname] = {
                "mappoints": pts,
                "T_world_floor": T_wf,
                "has_floor_transform": has_fp,
            }

            # Set default if recording_01
            if "recording_01" in sname:
                self.mappoints = pts
                self.T_world_floor = T_wf
                self.has_floor_transform = has_fp

    def _read_ply_points(self, path: Path) -> np.ndarray:
        points = []
        in_header = True
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if in_header:
                    if line == "end_header":
                        in_header = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
        return np.array(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)

    def _load_single_mappoints_ply(self, path: Path):
        self.mappoints = self._read_ply_points(path)

    def _load_single_floor_plane(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                fp_data = json.load(f)
                if "T_world_floor" in fp_data:
                    self.T_world_floor = np.array(fp_data["T_world_floor"], dtype=np.float64)
                    self.has_floor_transform = True
        except Exception:
            pass

    def extract_observation_geometry(
        self,
        obs: Dict[str, Any],
    ) -> GeometricEvidenceRecord:
        """Process a single observation record and compute real geometric metrics."""
        obs_id = obs.get("obs_id", "")
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = obs.get("global_entity_id", "")
        track_id = obs.get("local_track_id", "")
        points_count = int(obs.get("points_count", 0))
        bbox_xyxy = obs.get("bbox_xyxy")
        source_sess = obs.get("source_session", "")

        # 1. Depth extraction & physical uncertainty
        centroid_cam = obs.get("centroid_camera")
        if centroid_cam is not None and len(centroid_cam) >= 3 and centroid_cam[2] is not None:
            z_depth = float(centroid_cam[2])
            depth_valid = (MIN_VALID_DEPTH_M <= z_depth <= MAX_VALID_DEPTH_M)
            sigma_z = compute_depth_uncertainty(z_depth)
            depth_prov = "MEASURED_SENSOR"
        else:
            z_depth = None
            depth_valid = False
            sigma_z = None
            depth_prov = "MISSING"

        # 2. 3D spatial extent from floor bounding box
        b_min_raw = obs.get("bbox_min_floor")
        b_max_raw = obs.get("bbox_max_floor")
        if b_min_raw is not None and b_max_raw is not None and len(b_min_raw) == 3 and len(b_max_raw) == 3:
            b_min = np.array(b_min_raw, dtype=np.float64)
            b_max = np.array(b_max_raw, dtype=np.float64)
            spatial_extent = float(np.linalg.norm(b_max - b_min))
        else:
            b_min = None
            b_max = None
            spatial_extent = 0.0

        # 3. Camera pose
        cam_pose = obs.get("camera_pose")
        has_pose = (cam_pose is not None and len(cam_pose) >= 7 and not all(x == 0 for x in cam_pose[:7]))

        # 4. Resolve session-specific map and floor transform
        if source_sess in self.session_data:
            s_info = self.session_data[source_sess]
            sess_mappoints = s_info["mappoints"]
            T_wf = s_info["T_world_floor"]
            has_fp = s_info["has_floor_transform"]
        else:
            sess_mappoints = self.mappoints
            T_wf = self.T_world_floor
            has_fp = self.has_floor_transform

        # 5. Point-to-box consistency evaluation
        candidate_pts = np.empty((0, 3), dtype=np.float64)
        point_to_bbox_prov = "MISSING_MAPPOINTS"
        point_to_bbox_valid = False
        pt_dist_px = None
        inlier_ratio = None
        inliers_count = 0

        if not has_fp:
            # If floor transform is unverified or missing, DO NOT assume floor coords == world coords
            point_to_bbox_prov = "MISSING_FLOOR_TRANSFORM"
        elif len(sess_mappoints) == 0:
            point_to_bbox_prov = "MISSING_MAPPOINTS"
        elif not has_pose:
            point_to_bbox_prov = "MISSING_CAMERA_POSE"
        elif b_min is not None and b_max is not None:
            # Transform 8 corners from floor_map to slam_world
            corners_floor = np.array([
                [b_min[0], b_min[1], b_min[2], 1.0], [b_min[0], b_min[1], b_max[2], 1.0],
                [b_min[0], b_max[1], b_min[2], 1.0], [b_min[0], b_max[1], b_max[2], 1.0],
                [b_max[0], b_min[1], b_min[2], 1.0], [b_max[0], b_min[1], b_max[2], 1.0],
                [b_max[0], b_max[1], b_min[2], 1.0], [b_max[0], b_max[1], b_max[2], 1.0],
            ], dtype=np.float64)
            corners_world = (T_wf @ corners_floor.T).T[:, :3]
            w_min = np.min(corners_world, axis=0) - 0.05
            w_max = np.max(corners_world, axis=0) + 0.05

            mask = np.all((sess_mappoints >= w_min) & (sess_mappoints <= w_max), axis=1)
            candidate_pts = sess_mappoints[mask]

            if len(candidate_pts) >= 3 and bbox_xyxy is not None and len(bbox_xyxy) == 4:
                pixels, valid_z, _ = project_points_to_camera(candidate_pts, cam_pose)
                if np.sum(valid_z) >= 3:
                    dists, inlier_mask = compute_point_to_bbox_distances(pixels, bbox_xyxy, valid_z)
                    if len(dists) > 0:
                        pt_dist_px = float(np.mean(dists))
                        inliers_count = int(np.sum(inlier_mask))
                        inlier_ratio = float(inliers_count / max(1, len(dists)))
                        point_to_bbox_valid = True
                        point_to_bbox_prov = "MEASURED_MAPPOINTS"

        n_candidates = len(candidate_pts)
        support_count = max(points_count, n_candidates)

        # 6. Overall Evidence Provenance
        if point_to_bbox_valid and depth_valid:
            provenance = "MEASURED_FULL"
        elif depth_valid:
            provenance = "DERIVED_DEPTH_ONLY"
        elif not has_fp:
            provenance = "MISSING_FLOOR_TRANSFORM"
        else:
            provenance = "MISSING"

        # 7. Geometric Consistency Score
        if provenance == "MEASURED_FULL" and pt_dist_px is not None and inlier_ratio is not None and sigma_z is not None:
            inlier_score = min(1.0, max(0.0, inlier_ratio))
            residual_score = math.exp(-pt_dist_px / 35.0)
            depth_score = math.exp(-sigma_z / 0.04)
            consistency_score = 0.40 * inlier_score + 0.30 * residual_score + 0.30 * depth_score
            has_valid_evidence = (depth_valid and inlier_score > 0.10)
        elif depth_valid and sigma_z is not None:
            depth_score = math.exp(-sigma_z / 0.04)
            consistency_score = 0.60 * depth_score + 0.40 * min(1.0, points_count / 2000.0)
            has_valid_evidence = depth_valid
        else:
            consistency_score = 0.0
            has_valid_evidence = False

        return GeometricEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            source_session=source_sess,
            depth_valid=depth_valid,
            measured_depth_m=round(z_depth, 4) if z_depth is not None else None,
            depth_uncertainty_sigma_m=round(sigma_z, 4) if sigma_z is not None else None,
            depth_provenance=depth_prov,
            point_to_bbox_valid=point_to_bbox_valid,
            point_to_bbox_provenance=point_to_bbox_prov,
            point_to_bbox_distance_px=round(pt_dist_px, 2) if pt_dist_px is not None else None,
            point_to_bbox_inlier_ratio=round(inlier_ratio, 4) if inlier_ratio is not None else None,
            mappoints_candidate_count=n_candidates,
            mappoints_inlier_count=inliers_count,
            feature_reprojection_residual_valid=False,
            feature_reprojection_residual_px=None,
            feature_reprojection_provenance="UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES",
            landmark_support_count=support_count,
            landmark_spatial_extent_m=round(spatial_extent, 4),
            geometric_consistency_score=round(consistency_score, 4),
            provenance=provenance,
            has_valid_evidence=has_valid_evidence,
        )
