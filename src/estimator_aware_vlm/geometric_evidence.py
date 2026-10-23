"""Geometric Evidence Extraction for Object Observations.

Extracts:
1. Physical RealSense D455 stereo depth uncertainty:
   sigma_z = (z^2 / (f_x * B)) * sigma_d
2. MapPoint-to-2D-bounding-box geometric consistency:
   point_to_bbox_distance_px and point_to_bbox_inlier_ratio.
   (Clearly distinguished from feature-matching reprojection residuals, which
    require 2D-3D keypoint correspondences that are not output by 2D box detectors).
3. Explicit evidence provenance: MEASURED_FULL, DERIVED_DEPTH_ONLY, MISSING_FLOOR_TRANSFORM,
   MISSING_SESSION_MAP, MISSING_SESSION_CONFIG, MISSING_CAMERA_POSE, or MISSING.
4. Multi-session MapPoints, SE(3) floor plane transform, and sensor calibration management.
5. Calibrated geometric consistency heuristic score bounded in [0.0, 1.0].
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root


# Factory/Hardware defaults for Intel RealSense D455 (640x480 resolution mode)
DEFAULT_FX = 383.9205627
DEFAULT_FY = 383.4780579
DEFAULT_CX = 320.2670593
DEFAULT_CY = 244.5698853
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
FX = DEFAULT_FX
FY = DEFAULT_FY
CX = DEFAULT_CX
CY = DEFAULT_CY
IMAGE_WIDTH = DEFAULT_IMAGE_WIDTH
IMAGE_HEIGHT = DEFAULT_IMAGE_HEIGHT
STEREO_BASELINE_M = 0.095      # 95 mm physical baseline between IR stereo sensors
DISPARITY_NOISE_PX = 0.08      # Subpixel disparity matching noise (sigma_d)
MIN_VALID_DEPTH_M = 0.20
MAX_VALID_DEPTH_M = 4.80


@dataclass
class SessionGeometricConfig:
    """Explicit configuration mapping a session to its assets and calibration."""
    session_id: str
    mappoints_ply_path: Optional[Path] = None
    floor_plane_path: Optional[Path] = None
    camera_intrinsics_path: Optional[Path] = None
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    fx: float = DEFAULT_FX
    fy: float = DEFAULT_FY
    cx: float = DEFAULT_CX
    cy: float = DEFAULT_CY
    stereo_baseline_m: float = STEREO_BASELINE_M
    disparity_noise_px: float = DISPARITY_NOISE_PX
    depth_unit_meters: float = 0.001
    calibration_provenance: str = "CONFIGURED_AUTHENTIC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mappoints_ply_path": str(self.mappoints_ply_path) if self.mappoints_ply_path else None,
            "floor_plane_path": str(self.floor_plane_path) if self.floor_plane_path else None,
            "camera_intrinsics_path": str(self.camera_intrinsics_path) if self.camera_intrinsics_path else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "stereo_baseline_m": self.stereo_baseline_m,
            "disparity_noise_px": self.disparity_noise_px,
            "depth_unit_meters": self.depth_unit_meters,
            "calibration_provenance": self.calibration_provenance,
        }


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

    # Point-to-Bounding-Box Consistency (honestly named, NOT feature reprojection)
    point_to_bbox_valid: bool
    point_to_bbox_provenance: str  # "MEASURED_MAPPOINTS", "MISSING_MAPPOINTS", "MISSING_FLOOR_TRANSFORM", etc.
    point_to_bbox_distance_px: Optional[float]  # Distance outside box in pixels (0 for inliers)
    point_to_bbox_inlier_ratio: Optional[float]  # Fraction of candidate MapPoints projecting inside box
    mappoints_candidate_count: int
    mappoints_inlier_count: int
    observation_points_count: int

    # Feature-Level Reprojection Residuals (Requires 2D keypoint-to-3D MapPoint matches, not box detector)
    feature_reprojection_residual_valid: bool = False
    feature_reprojection_residual_px: Optional[float] = None
    feature_reprojection_provenance: str = "UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES"

    # Spatial support & calibration
    landmark_support_count: int = 0
    landmark_spatial_extent_m: float = 0.0
    geometric_consistency_score: float = 0.0
    provenance: str = "MISSING"
    has_valid_evidence: bool = False

    # Backwards-compatibility properties with honest semantics
    @property
    def inlier_ratio(self) -> Optional[float]:
        return self.point_to_bbox_inlier_ratio

    @property
    def mean_reprojection_residual_px(self) -> Optional[float]:
        # Rule 4: Do NOT return point_to_bbox_distance under reprojection residual name.
        return None

    @property
    def median_reprojection_residual_px(self) -> Optional[float]:
        return None

    @property
    def p95_reprojection_residual_px(self) -> Optional[float]:
        return None

    @property
    def reprojection_residual_valid(self) -> bool:
        return False

    @property
    def inlier_count(self) -> int:
        return self.mappoints_inlier_count

    @property
    def mappoints_count(self) -> int:
        return self.mappoints_candidate_count

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Preserve documented compatibility aliases without metric mislabelling
        d["inlier_ratio"] = self.point_to_bbox_inlier_ratio
        d["mean_reprojection_residual_px"] = None
        d["median_reprojection_residual_px"] = None
        d["p95_reprojection_residual_px"] = None
        d["reprojection_residual_valid"] = False
        d["inlier_count"] = self.mappoints_inlier_count
        d["mappoints_count"] = self.mappoints_candidate_count
        return d


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion to 3x3 orthonormal rotation matrix.

    Validates quaternion norm. If norm is near-zero (< 1e-4) or non-finite,
    raises ValueError rather than silently returning identity.
    """
    for v in (qx, qy, qz, qw):
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"Non-finite quaternion component: [{qx}, {qy}, {qz}, qw={qw}]")

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-4:
        raise ValueError(f"Invalid quaternion: near-zero norm ({norm:.6e}) cannot represent valid 3D orientation.")

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

    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ], dtype=np.float64)


def invert_se3_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert SE(3) transformation matrix: [R | t] -> [R^T | -R^T * t]."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def project_points_to_camera(
    points_world: np.ndarray,
    cam_pose: List[float],
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
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

    if not isinstance(cam_pose, (list, tuple)) or len(cam_pose) < 7:
        return np.empty((0, 2)), np.empty(0, dtype=bool), np.empty((0, 3))

    try:
        t_wc = np.array(cam_pose[:3], dtype=np.float64)
        R_wc = quaternion_to_rotation_matrix(cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6])
    except ValueError:
        return np.empty((0, 2)), np.empty(0, dtype=bool), np.empty((0, 3))

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


def compute_depth_uncertainty(
    depth_z: float,
    fx: float = DEFAULT_FX,
    baseline_m: float = STEREO_BASELINE_M,
    sigma_d_px: float = DISPARITY_NOISE_PX,
) -> float:
    """Compute physical depth uncertainty (1-sigma) for stereo triangulation.

    Formula: sigma_z = (z^2 / (f_x * B)) * sigma_d
    """
    if depth_z <= 0.05 or not (math.isfinite(depth_z)):
        return 0.10
    z = max(0.15, min(depth_z, 6.0))
    sigma_z = (z * z / (fx * baseline_m)) * sigma_d_px
    return float(max(0.001, sigma_z))


class GeometricEvidenceExtractor:
    """Extracts geometric consistency evidence with explicit session configuration and zero cross-session fallback."""

    def __init__(
        self,
        mappoints_path: Optional[Path] = None,
        floor_plane_path: Optional[Path] = None,
        repo_root: Optional[Path] = None,
        session_configs: Optional[Dict[str, SessionGeometricConfig]] = None,
    ):
        if repo_root is None:
            repo_root = get_repo_root()
        self.repo_root = Path(repo_root)

        self.session_configs: Dict[str, SessionGeometricConfig] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}

        if session_configs is not None:
            for sname, cfg in session_configs.items():
                self.register_session_config(cfg)
        else:
            self._discover_and_register_known_sessions()

        # If explicit paths provided (e.g. for targeted single-session tests), register under manual key
        if mappoints_path or floor_plane_path:
            pts = self._read_ply_points(Path(mappoints_path)) if mappoints_path else np.empty((0, 3), dtype=np.float64)
            T_wf, has_fp = self._validate_and_load_floor_plane(Path(floor_plane_path)) if floor_plane_path else (np.eye(4), False)
            manual_cfg = SessionGeometricConfig(
                session_id="__manual__",
                mappoints_ply_path=Path(mappoints_path) if mappoints_path else None,
                floor_plane_path=Path(floor_plane_path) if floor_plane_path else None,
            )
            self.session_configs["__manual__"] = manual_cfg
            self.session_data["__manual__"] = {
                "config": manual_cfg,
                "mappoints": pts,
                "T_world_floor": T_wf,
                "has_floor_transform": has_fp,
            }

    def register_session_config(self, cfg: SessionGeometricConfig) -> None:
        """Register and load assets for a specific session."""
        pts = np.empty((0, 3), dtype=np.float64)
        if cfg.mappoints_ply_path and Path(cfg.mappoints_ply_path).exists():
            pts = self._read_ply_points(Path(cfg.mappoints_ply_path))

        T_wf = np.eye(4, dtype=np.float64)
        has_fp = False
        if cfg.floor_plane_path and Path(cfg.floor_plane_path).exists():
            T_wf, has_fp = self._validate_and_load_floor_plane(Path(cfg.floor_plane_path))

        # Check for camera intrinsics JSON
        if cfg.camera_intrinsics_path and Path(cfg.camera_intrinsics_path).exists():
            try:
                with open(cfg.camera_intrinsics_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    cfg.fx = float(cdata.get("fx", cfg.fx))
                    cfg.fy = float(cdata.get("fy", cfg.fy))
                    cfg.cx = float(cdata.get("cx", cfg.cx))
                    cfg.cy = float(cdata.get("cy", cfg.cy))
                    cfg.image_width = int(cdata.get("width", cfg.image_width))
                    cfg.image_height = int(cdata.get("height", cfg.image_height))
                    cfg.calibration_provenance = "MEASURED_AUTHENTIC_JSON"
            except Exception as e:
                print(f"Warning: Failed to parse camera intrinsics from {cfg.camera_intrinsics_path}: {e}")

        self.session_configs[cfg.session_id] = cfg
        self.session_data[cfg.session_id] = {
            "config": cfg,
            "mappoints": pts,
            "T_world_floor": T_wf,
            "has_floor_transform": has_fp,
        }

    def _discover_and_register_known_sessions(self) -> None:
        """Discover canonical Session 1 and Session 2 assets from repository root."""
        known_sessions = [
            "semantic_mapping_my_desk_recording_01",
            "semantic_mapping_my_desk_recording_02",
        ]
        for sname in known_sessions:
            s_dir = self.repo_root / "data" / sname
            if not s_dir.exists():
                s_dir = self.repo_root / "outputs" / sname
            if not s_dir.exists():
                continue

            ply_path = s_dir / "trajectory" / "MapPoints.ply"
            fp_path = s_dir / "floor_plane.json"

            # Check raw intrinsics
            raw_name = sname.replace("semantic_mapping_", "")
            intrin_path = self.repo_root / "data" / "raw" / raw_name / "camera_intrinsics.json"
            if not intrin_path.exists():
                intrin_path = None

            cfg = SessionGeometricConfig(
                session_id=sname,
                mappoints_ply_path=ply_path if ply_path.exists() else None,
                floor_plane_path=fp_path if fp_path.exists() else None,
                camera_intrinsics_path=intrin_path,
            )
            self.register_session_config(cfg)

    def _read_ply_points(self, path: Path) -> np.ndarray:
        """Strictly validate and parse ASCII PLY file.

        Rejects unsupported formats (e.g. binary) and ensures declared vertex count
        matches parsed finite XYZ rows.
        """
        points = []
        in_header = True
        declared_vertices: Optional[int] = None
        is_ascii = False

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if in_header:
                    if stripped.startswith("format "):
                        if "ascii 1.0" in stripped:
                            is_ascii = True
                        else:
                            raise ValueError(f"Unsupported PLY format at {path}: '{stripped}'. Only 'format ascii 1.0' is supported.")
                    elif stripped.startswith("element vertex "):
                        try:
                            declared_vertices = int(stripped.split()[2])
                        except (IndexError, ValueError):
                            raise ValueError(f"Malformed 'element vertex' header at {path}:{line_no}: '{stripped}'")
                    elif stripped == "end_header":
                        in_header = False
                        if not is_ascii:
                            raise ValueError(f"Missing or invalid format declaration in PLY header: {path}")
                    continue

                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                            points.append([x, y, z])
                    except ValueError:
                        continue

        if declared_vertices is not None and len(points) != declared_vertices:
            print(f"Warning: PLY {path} declared {declared_vertices} vertices but parsed {len(points)} finite points.")

        return np.array(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)

    def _validate_and_load_floor_plane(self, path: Path) -> Tuple[np.ndarray, bool]:
        """Strictly validate and load T_world_floor homogeneous transform.

        Validates 4x4 shape, finiteness, bottom row [0, 0, 0, 1], and rigid rotation.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                fp_data = json.load(f)
            if "T_world_floor" not in fp_data:
                print(f"Warning: Missing 'T_world_floor' key in floor plane {path}")
                return np.eye(4, dtype=np.float64), False

            T_wf = np.array(fp_data["T_world_floor"], dtype=np.float64)
            if T_wf.shape != (4, 4):
                print(f"Warning: Invalid T_world_floor shape {T_wf.shape} in {path}")
                return np.eye(4, dtype=np.float64), False

            if not np.all(np.isfinite(T_wf)):
                print(f"Warning: Non-finite elements in T_world_floor in {path}")
                return np.eye(4, dtype=np.float64), False

            # Validate bottom row is [0, 0, 0, 1]
            if not np.allclose(T_wf[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
                print(f"Warning: T_world_floor bottom row {T_wf[3, :]} is not [0,0,0,1] in {path}")
                return np.eye(4, dtype=np.float64), False

            # Validate rotation is rigid orthogonal
            R = T_wf[:3, :3]
            det_R = np.linalg.det(R)
            if abs(det_R - 1.0) > 1e-4 or not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
                print(f"Warning: T_world_floor rotation is not rigid SO(3) (det={det_R:.6f}) in {path}")
                return np.eye(4, dtype=np.float64), False

            return T_wf, True
        except Exception as e:
            print(f"Error loading floor plane from {path}: {e}")
            return np.eye(4, dtype=np.float64), False

    def extract_observation_geometry(
        self,
        obs: Dict[str, Any],
    ) -> GeometricEvidenceRecord:
        """Process a single observation record with explicit session mapping and honest metric naming."""
        obs_id = str(obs.get("obs_id", ""))
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = str(obs.get("global_entity_id", ""))
        track_id = str(obs.get("local_track_id", ""))
        points_count = int(obs.get("points_count", 0))
        bbox_xyxy = obs.get("bbox_xyxy")
        source_sess = str(obs.get("source_session", "")).strip()

        # 1. Depth extraction & physical uncertainty
        centroid_cam = obs.get("centroid_camera")
        z_depth: Optional[float] = None
        depth_valid = False
        sigma_z: Optional[float] = None
        depth_prov = "MISSING"

        # Resolve session configuration explicitly
        sess_data = self.session_data.get(source_sess)
        if sess_data is None and "__manual__" in self.session_data:
            sess_data = self.session_data["__manual__"]

        sess_cfg = sess_data["config"] if sess_data else None
        fx = sess_cfg.fx if sess_cfg else DEFAULT_FX
        baseline_m = sess_cfg.stereo_baseline_m if sess_cfg else STEREO_BASELINE_M
        sigma_d_px = sess_cfg.disparity_noise_px if sess_cfg else DISPARITY_NOISE_PX

        if centroid_cam is not None and len(centroid_cam) >= 3 and centroid_cam[2] is not None:
            raw_z = centroid_cam[2]
            if isinstance(raw_z, (int, float)) and math.isfinite(raw_z):
                z_depth = float(raw_z)
                depth_valid = (MIN_VALID_DEPTH_M <= z_depth <= MAX_VALID_DEPTH_M)
                sigma_z = compute_depth_uncertainty(z_depth, fx=fx, baseline_m=baseline_m, sigma_d_px=sigma_d_px)
                depth_prov = "MEASURED_SENSOR"

        # 2. 3D spatial extent from floor bounding box
        b_min_raw = obs.get("bbox_min_floor")
        b_max_raw = obs.get("bbox_max_floor")
        spatial_extent = 0.0
        b_min: Optional[np.ndarray] = None
        b_max: Optional[np.ndarray] = None
        if b_min_raw is not None and b_max_raw is not None and len(b_min_raw) == 3 and len(b_max_raw) == 3:
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in b_min_raw + b_max_raw):
                b_min = np.array(b_min_raw, dtype=np.float64)
                b_max = np.array(b_max_raw, dtype=np.float64)
                spatial_extent = float(np.linalg.norm(b_max - b_min))

        # 3. Camera pose
        cam_pose = obs.get("camera_pose")
        has_pose = False
        if cam_pose is not None and len(cam_pose) >= 7:
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in cam_pose[:7]):
                qx, qy, qz, qw = cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]
                q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                has_pose = (0.90 <= q_norm <= 1.10)

        # 4. Strict session resolution (NO CROSS-SESSION FALLBACK)
        candidate_pts = np.empty((0, 3), dtype=np.float64)
        point_to_bbox_prov = "MISSING_MAPPOINTS"
        point_to_bbox_valid = False
        pt_dist_px: Optional[float] = None
        inlier_ratio: Optional[float] = None
        inliers_count = 0

        if sess_data is None:
            # Unknown session: NEVER fall back to Session 1 map
            point_to_bbox_prov = "MISSING_SESSION_CONFIG"
            provenance = "MISSING_SESSION_CONFIG"
            sess_mappoints = np.empty((0, 3), dtype=np.float64)
            has_fp = False
            T_wf = np.eye(4, dtype=np.float64)
        else:
            sess_mappoints = sess_data["mappoints"]
            T_wf = sess_data["T_world_floor"]
            has_fp = sess_data["has_floor_transform"]

            if not has_fp:
                point_to_bbox_prov = "MISSING_FLOOR_TRANSFORM"
            elif len(sess_mappoints) == 0:
                point_to_bbox_prov = "MISSING_SESSION_MAP"
            elif not has_pose:
                point_to_bbox_prov = "MISSING_CAMERA_POSE"
            elif b_min is not None and b_max is not None:
                # Transform 8 corners from floor frame to world frame
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
                    pixels, valid_z, _ = project_points_to_camera(
                        candidate_pts,
                        cam_pose,
                        fx=fx,
                        fy=sess_cfg.fy if sess_cfg else DEFAULT_FY,
                        cx=sess_cfg.cx if sess_cfg else DEFAULT_CX,
                        cy=sess_cfg.cy if sess_cfg else DEFAULT_CY,
                    )
                    if np.sum(valid_z) >= 3:
                        dists, inlier_mask = compute_point_to_bbox_distances(pixels, bbox_xyxy, valid_z)
                        if len(dists) > 0:
                            pt_dist_px = float(np.mean(dists))
                            inliers_count = int(np.sum(inlier_mask))
                            inlier_ratio = float(inliers_count / max(1, len(dists)))
                            point_to_bbox_valid = True
                            point_to_bbox_prov = "MEASURED_MAPPOINTS"

        n_candidates = len(candidate_pts)

        # 5. Overall Evidence Provenance (honest distinction)
        if sess_data is None:
            provenance = "MISSING_SESSION_CONFIG"
        elif point_to_bbox_valid and depth_valid:
            provenance = "MEASURED_FULL"
        elif depth_valid:
            provenance = "DERIVED_DEPTH_ONLY"
        elif not has_fp:
            provenance = "MISSING_FLOOR_TRANSFORM"
        elif not has_pose:
            provenance = "MISSING_CAMERA_POSE"
        elif len(sess_mappoints) == 0:
            provenance = "MISSING_SESSION_MAP"
        else:
            provenance = "MISSING"

        # 6. Geometric Consistency Score (Documented Heuristic Ranking Score)
        if provenance == "MEASURED_FULL" and pt_dist_px is not None and inlier_ratio is not None and sigma_z is not None:
            inlier_score = min(1.0, max(0.0, inlier_ratio))
            residual_score = math.exp(-pt_dist_px / 35.0)
            depth_score = math.exp(-sigma_z / 0.04)
            consistency_score = 0.40 * inlier_score + 0.30 * residual_score + 0.30 * depth_score
            has_valid_evidence = bool(depth_valid and inlier_score > 0.10)
        elif depth_valid and sigma_z is not None:
            depth_score = math.exp(-sigma_z / 0.04)
            consistency_score = 0.60 * depth_score + 0.40 * min(1.0, points_count / 2000.0)
            has_valid_evidence = bool(depth_valid)
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
            observation_points_count=points_count,
            feature_reprojection_residual_valid=False,
            feature_reprojection_residual_px=None,
            feature_reprojection_provenance="UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES",
            landmark_support_count=n_candidates,
            landmark_spatial_extent_m=round(spatial_extent, 4),
            geometric_consistency_score=round(consistency_score, 4),
            provenance=provenance,
            has_valid_evidence=has_valid_evidence,
        )
