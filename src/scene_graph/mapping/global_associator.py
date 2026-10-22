"""Persistent Global Object Association, Conflict Resolution, and Geometry Fusion."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment


@dataclass
class ObjectObservationRecord:
    obs_id: str
    frame_index: int
    timestamp: float
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    centroid_camera: List[float]
    centroid_floor: Optional[List[float]] = None  # in floor_map frame [X, Y, Z]
    points_floor: Optional[np.ndarray] = None      # (N, 3) in floor_map frame
    local_track_id: Optional[str] = None
    global_entity_id: Optional[str] = None
    status: str = "VALID"
    centroid_world: Optional[List[float]] = None
    points_world: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.centroid_floor is None and self.centroid_world is not None:
            self.centroid_floor = self.centroid_world
        if self.centroid_world is None and self.centroid_floor is not None:
            self.centroid_world = self.centroid_floor
        if self.points_floor is None and self.points_world is not None:
            self.points_floor = self.points_world
        if self.points_world is None and self.points_floor is not None:
            self.points_world = self.points_floor



@dataclass
class GlobalEntity:
    entity_id: str
    class_name: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    first_frame_idx: int
    last_frame_idx: int
    is_dynamic: bool = False
    
    # Provenance tracking
    observation_ids: List[str] = field(default_factory=list)
    local_track_ids: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    class_votes: Dict[str, int] = field(default_factory=dict)
    observations: List[ObjectObservationRecord] = field(default_factory=list)
    merge_history: List[str] = field(default_factory=list)

    # Geometry in floor_map frame
    centroid_floor: Optional[np.ndarray] = None
    centroid_std_floor: Optional[float] = None  # derived positional uncertainty (meters)
    points_floor: Optional[np.ndarray] = None    # filtered point cloud (N, 3)
    bbox_min_floor: Optional[np.ndarray] = None
    bbox_max_floor: Optional[np.ndarray] = None
    dimensions: Optional[np.ndarray] = None      # [width_x, length_y, height_z]
    floor_footprint_polygon: Optional[List[List[float]]] = None  # 2D polygon in floor X-Y

    is_confirmed: bool = False
    quarantine_reason: Optional[str] = None

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence_history)) if self.confidence_history else 0.0

    @property
    def frame_span(self) -> int:
        return self.last_frame_idx - self.first_frame_idx


class GlobalObjectAssociator:
    """Maintains persistent global object identities with strict provenance and multi-view geometry fusion."""

    DYNAMIC_CLASSES = {"person"}

    # Minimum physical size thresholds (width, length, height in meters)
    MIN_DIMENSIONS = {
        "chair": (0.35, 0.35, 0.35),
        "desk": (0.60, 0.50, 0.40),
        "table": (0.60, 0.50, 0.40),
        "door": (0.35, 0.10, 0.70),
        "storage box": (0.25, 0.25, 0.15),
        "backpack": (0.20, 0.15, 0.20),
        "computer monitor": (0.30, 0.10, 0.25),
        "laptop": (0.22, 0.16, 0.05),
        "keyboard": (0.25, 0.08, 0.02),
        "pedestal fan": (0.30, 0.30, 0.60),
        "quadcopter drone": (0.15, 0.15, 0.05),
        "bottle": (0.05, 0.05, 0.15),
        "trash can": (0.20, 0.20, 0.25),
        "whiteboard": (0.40, 0.05, 0.30),
    }

    # Maximum physical size bounds (width_x, length_y, height_z in meters)
    MAX_DIMENSIONS = {
        "chair": (0.62, 0.62, 1.05),             # Standard office task/swivel chair: 55-60cm
        "desk": (1.50, 0.80, 0.85),              # Standard office workstation desk: 1.2-1.4m x 0.6-0.75m
        "table": (1.50, 0.85, 0.85),
        "door": (1.00, 0.20, 2.10),
        "storage box": (0.50, 0.45, 0.50),
        "backpack": (0.38, 0.30, 0.50),
        "computer monitor": (0.60, 0.20, 0.50),  # Widescreen 24-27" monitor on stand
        "laptop": (0.35, 0.25, 0.20),            # 14-16" laptop
        "keyboard": (0.45, 0.16, 0.08),
        "pedestal fan": (0.45, 0.45, 1.30),
        "quadcopter drone": (0.30, 0.30, 0.12),
        "bottle": (0.12, 0.12, 0.30),
        "trash can": (0.35, 0.35, 0.50),
        "whiteboard": (1.80, 0.10, 1.20),
    }

    COMPATIBLE_CLASSES: Dict[str, Set[str]] = {
        "desk": {"desk", "table", "dining table"},
        "table": {"desk", "table", "dining table"},
        "dining table": {"desk", "table", "dining table"},
        "chair": {"chair", "armchair", "seat"},
        "armchair": {"chair", "armchair", "seat"},
        "backpack": {"backpack", "bag", "handbag"},
        "bag": {"backpack", "bag", "handbag"},
        "handbag": {"backpack", "bag", "handbag"},
        "laptop": {"laptop", "computer"},
        "bottle": {"bottle", "cup"},
    }

    @classmethod
    def are_classes_compatible(cls, c1: str, c2: str) -> bool:
        if c1 == c2:
            return True
        return c2 in cls.COMPATIBLE_CLASSES.get(c1, set()) or c1 in cls.COMPATIBLE_CLASSES.get(c2, set())

    def __init__(
        self,
        association_dist_threshold_m: float = 0.65,
        min_hits_to_confirm: int = 8,
        min_frame_span: int = 0,
        max_centroid_variance_m: float = 0.30,
        voxel_size_m: float = 0.02,
    ):
        self.association_dist_threshold_m = float(association_dist_threshold_m)
        self.min_hits_to_confirm = int(min_hits_to_confirm)
        self.min_frame_span = int(min_frame_span)
        self.max_centroid_variance_m = float(max_centroid_variance_m)
        self.voxel_size_m = float(voxel_size_m)

        self.class_counters: Dict[str, int] = {}
        self.entities: Dict[str, GlobalEntity] = {}
        self.unconfirmed_entities: Dict[str, GlobalEntity] = {}

    def _next_id(self, class_name: str) -> str:
        self.class_counters[class_name] = self.class_counters.get(class_name, 0) + 1
        return f"{class_name}_{self.class_counters[class_name]:03d}"

    def update(self, frame_observations: List[ObjectObservationRecord]) -> Dict[str, str]:
        """Associate frame observations to global entities using 1-to-1 matching and class voting."""
        obs_to_entity: Dict[str, str] = {}
        if not frame_observations:
            return obs_to_entity

        all_targets = list(self.entities.values()) + list(self.unconfirmed_entities.values())

        if not all_targets:
            for obs in frame_observations:
                gid = self._next_id(obs.class_name)
                is_dyn = obs.class_name in self.DYNAMIC_CLASSES
                entity = GlobalEntity(
                    entity_id=gid,
                    class_name=obs.class_name,
                    first_seen_timestamp=obs.timestamp,
                    last_seen_timestamp=obs.timestamp,
                    first_frame_idx=obs.frame_index,
                    last_frame_idx=obs.frame_index,
                    is_dynamic=is_dyn,
                    observation_ids=[obs.obs_id],
                    local_track_ids=[obs.local_track_id] if obs.local_track_id else [],
                    confidence_history=[obs.confidence],
                    class_votes={obs.class_name: 1},
                    observations=[obs],
                    centroid_floor=np.array(obs.centroid_floor),
                    points_floor=obs.points_floor.copy(),
                    is_confirmed=False,
                )
                self.unconfirmed_entities[gid] = entity
                obs.global_entity_id = gid
                obs_to_entity[obs.obs_id] = gid
            return obs_to_entity

        # Hungarian Cost Matrix based on horizontal distance in floor plane
        cost_matrix = np.full((len(frame_observations), len(all_targets)), 1e6, dtype=np.float64)

        for i, obs in enumerate(frame_observations):
            c_obs = np.array(obs.centroid_floor)
            for j, tgt in enumerate(all_targets):
                # Never match dynamic person to static furniture
                if (obs.class_name in self.DYNAMIC_CLASSES) != tgt.is_dynamic:
                    continue

                # Strict class compatibility gate: Incompatible classes NEVER match
                if not self.are_classes_compatible(obs.class_name, tgt.class_name):
                    continue

                # Horizontal distance in floor plane
                dist_xy = math.hypot(c_obs[0] - tgt.centroid_floor[0], c_obs[1] - tgt.centroid_floor[1])
                dist_z = abs(c_obs[2] - tgt.centroid_floor[2])

                # Match within threshold
                if dist_xy <= self.association_dist_threshold_m and dist_z <= 1.2:
                    cost_matrix[i, j] = dist_xy

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_obs = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.association_dist_threshold_m:
                obs = frame_observations[r]
                tgt = all_targets[c]
                self._add_observation_to_entity(tgt, obs)
                obs.global_entity_id = tgt.entity_id
                obs_to_entity[obs.obs_id] = tgt.entity_id
                matched_obs.add(r)

        # Unmatched spawn new candidates
        for i, obs in enumerate(frame_observations):
            if i not in matched_obs:
                gid = self._next_id(obs.class_name)
                is_dyn = obs.class_name in self.DYNAMIC_CLASSES
                entity = GlobalEntity(
                    entity_id=gid,
                    class_name=obs.class_name,
                    first_seen_timestamp=obs.timestamp,
                    last_seen_timestamp=obs.timestamp,
                    first_frame_idx=obs.frame_index,
                    last_frame_idx=obs.frame_index,
                    is_dynamic=is_dyn,
                    observation_ids=[obs.obs_id],
                    local_track_ids=[obs.local_track_id] if obs.local_track_id else [],
                    confidence_history=[obs.confidence],
                    class_votes={obs.class_name: 1},
                    observations=[obs],
                    centroid_floor=np.array(obs.centroid_floor),
                    points_floor=obs.points_floor.copy(),
                    is_confirmed=False,
                )
                if (entity.observation_count >= self.min_hits_to_confirm and 
                    entity.frame_span >= self.min_frame_span and 
                    not entity.is_dynamic):
                    self.entities[gid] = entity
                    entity.is_confirmed = True
                else:
                    self.unconfirmed_entities[gid] = entity
                obs.global_entity_id = gid
                obs_to_entity[obs.obs_id] = gid


        return obs_to_entity

    def _add_observation_to_entity(self, entity: GlobalEntity, obs: ObjectObservationRecord):
        entity.observations.append(obs)
        entity.observation_ids.append(obs.obs_id)
        if obs.local_track_id and obs.local_track_id not in entity.local_track_ids:
            entity.local_track_ids.append(obs.local_track_id)
        entity.confidence_history.append(obs.confidence)
        entity.last_seen_timestamp = obs.timestamp
        entity.last_frame_idx = obs.frame_index

        # Update class votes (majority voting resolves class flicker)
        entity.class_votes[obs.class_name] = entity.class_votes.get(obs.class_name, 0) + 1
        majority_class = max(entity.class_votes.items(), key=lambda x: x[1])[0]
        entity.class_name = majority_class

        # Incremental centroid update
        n = len(entity.observations)
        entity.centroid_floor = entity.centroid_floor + (np.array(obs.centroid_floor) - entity.centroid_floor) / n

        # Accumulate raw points
        if entity.points_floor is None:
            entity.points_floor = obs.points_floor.copy()
        else:
            entity.points_floor = np.vstack((entity.points_floor, obs.points_floor))

        # Check inline promotion if requirements met
        if (entity.observation_count >= self.min_hits_to_confirm and 
            entity.frame_span >= self.min_frame_span and 
            not entity.is_dynamic):
            if entity.entity_id in self.unconfirmed_entities:
                del self.unconfirmed_entities[entity.entity_id]
                self.entities[entity.entity_id] = entity
                entity.is_confirmed = True

    def finalize_geometries(self, floor_normal: Optional[np.ndarray] = None, floor_d: Optional[float] = None):
        """Backward-compatible wrapper to filter geometries and evaluate confirmation policy."""
        self.filter_and_fuse_geometries()
        self.evaluate_confirmation_policy()


    def filter_and_fuse_geometries(self):
        """Voxel downsample and statistical outlier filter point clouds, compute consistent bounds and footprints."""
        all_entities = list(self.entities.values()) + list(self.unconfirmed_entities.values())

        for entity in all_entities:
            if entity.points_floor is None or len(entity.points_floor) == 0:
                continue

            pts = entity.points_floor

            # 1. Voxel grid downsampling
            voxel_coords = np.floor(pts / self.voxel_size_m).astype(np.int64)
            _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
            pts_down = pts[unique_indices]

            # 2. Statistical Outlier Removal (SOR)
            if len(pts_down) > 40:
                # Mean distance to k-nearest neighbors approximation
                med = np.median(pts_down, axis=0)
                dists = np.linalg.norm(pts_down - med, axis=1)
                d_med = np.median(dists)
                d_mad = np.median(np.abs(dists - d_med))
                thresh = d_med + 3.0 * max(1e-4, 1.4826 * d_mad)
                inliers = dists <= thresh
                pts_clean = pts_down[inliers]
            else:
                pts_clean = pts_down

            if len(pts_clean) < 15:
                pts_clean = pts_down

            # STORE CLEANED POINTS: The exact same points used for PLY, BBox, and Footprint!
            centroid = np.median(pts_clean, axis=0)

            # Lookup physical size constraints for class
            max_b = self.MAX_DIMENSIONS.get(entity.class_name, (1.50, 1.00, 1.20))
            min_b = self.MIN_DIMENSIONS.get(entity.class_name, (0.20, 0.20, 0.15))

            # Filter background bleed (points too far from median object center)
            max_radius = 0.5 * math.hypot(max_b[0], max_b[1]) + 0.15
            dist_xy = np.hypot(pts_clean[:, 0] - centroid[0], pts_clean[:, 1] - centroid[1])
            in_core = dist_xy <= max_radius
            pts_core = pts_clean[in_core] if np.sum(in_core) >= 10 else pts_clean

            # Compute robust 8th - 92nd percentile span to eliminate edge bleed
            q_min = np.percentile(pts_core, 8, axis=0)
            q_max = np.percentile(pts_core, 92, axis=0)
            raw_span = q_max - q_min

            # Clamp dimensions tightly to physical reality
            dim_x = float(np.clip(raw_span[0], min_b[0], max_b[0]))
            dim_y = float(np.clip(raw_span[1], min_b[1], max_b[1]))
            dim_z = float(np.clip(raw_span[2], min_b[2], max_b[2]))

            # Tight bounding box centered around verified centroid
            b_min = np.array([centroid[0] - dim_x / 2.0, centroid[1] - dim_y / 2.0, centroid[2] - dim_z / 2.0])
            b_max = np.array([centroid[0] + dim_x / 2.0, centroid[1] + dim_y / 2.0, centroid[2] + dim_z / 2.0])

            entity.points_floor = pts_core
            entity.centroid_floor = np.round(centroid, 4)
            entity.bbox_min_floor = np.round(b_min, 4)
            entity.bbox_max_floor = np.round(b_max, 4)
            entity.dimensions = np.round([dim_x, dim_y, dim_z], 4)

            # 4. Compute empirical positional uncertainty standard deviation
            if len(entity.observations) >= 2:
                obs_c = np.array([o.centroid_floor for o in entity.observations])
                std_xyz = np.std(obs_c, axis=0)
                entity.centroid_std_floor = float(np.linalg.norm(std_xyz))
            else:
                entity.centroid_std_floor = 0.05

            # 5. Compute conservative 2D floor footprint (X, Y)
            footprint_2d = pts_core[:, :2]
            if len(footprint_2d) >= 4:
                try:
                    hull = ConvexHull(footprint_2d)
                    hull_pts = footprint_2d[hull.vertices]
                    entity.floor_footprint_polygon = hull_pts.tolist()
                except Exception:
                    entity.floor_footprint_polygon = [
                        [float(b_min[0]), float(b_min[1])],
                        [float(b_max[0]), float(b_min[1])],
                        [float(b_max[0]), float(b_max[1])],
                        [float(b_min[0]), float(b_max[1])],
                    ]
            else:
                entity.floor_footprint_polygon = [
                    [float(b_min[0]), float(b_min[1])],
                    [float(b_max[0]), float(b_min[1])],
                    [float(b_max[0]), float(b_max[1])],
                    [float(b_min[0]), float(b_max[1])],
                ]

    def evaluate_confirmation_policy(self):
        """Evaluate confirmation criteria based on observation count, span, dimensions, and stability."""
        to_promote = []
        quarantined = {}

        for gid, entity in list(self.unconfirmed_entities.items()):
            # Dynamic objects (person) are never confirmed as static world furniture
            if entity.is_dynamic:
                entity.quarantine_reason = "DYNAMIC_OBJECT"
                quarantined[gid] = entity
                continue

            # Check 1: Observation count
            if entity.observation_count < self.min_hits_to_confirm:
                entity.quarantine_reason = f"INSUFFICIENT_OBSERVATIONS ({entity.observation_count} < {self.min_hits_to_confirm})"
                quarantined[gid] = entity
                continue

            # Check 2: Temporal/viewpoint frame span
            if entity.frame_span < self.min_frame_span:
                entity.quarantine_reason = f"INSUFFICIENT_FRAME_SPAN ({entity.frame_span} < {self.min_frame_span})"
                quarantined[gid] = entity
                continue

            # Check 3: Multi-view positional stability
            if entity.centroid_std_floor is not None and entity.centroid_std_floor > self.max_centroid_variance_m:
                entity.quarantine_reason = f"HIGH_CENTROID_VARIANCE ({entity.centroid_std_floor:.3f}m > {self.max_centroid_variance_m}m)"
                quarantined[gid] = entity
                continue

            # Check 4: Plausible physical dimensions
            min_dim = self.MIN_DIMENSIONS.get(entity.class_name, (0.15, 0.15, 0.15))
            if entity.dimensions is not None:
                dim = entity.dimensions
                # Check width/length and height
                if dim[0] < min_dim[0] or dim[1] < min_dim[1] or dim[2] < min_dim[2]:
                    entity.quarantine_reason = f"IMPLAUSIBLE_DIMENSIONS ({dim[0]:.2f}x{dim[1]:.2f}x{dim[2]:.2f}m < {min_dim[0]:.2f}x{min_dim[1]:.2f}x{min_dim[2]:.2f}m)"
                    quarantined[gid] = entity
                    continue

            # Passed all confirmation criteria!
            entity.is_confirmed = True
            entity.quarantine_reason = None
            to_promote.append(gid)

        for gid in to_promote:
            self.entities[gid] = self.unconfirmed_entities.pop(gid)

    def merge_coincident_entities(self, merge_distance_threshold_m: float = 0.40):
        """Merge confirmed entities that share compatible class and coincident locations."""
        confirmed_ids = list(self.entities.keys())
        merged_away = set()

        for i in range(len(confirmed_ids)):
            id_a = confirmed_ids[i]
            if id_a in merged_away or id_a not in self.entities:
                continue
            e_a = self.entities[id_a]

            for j in range(i + 1, len(confirmed_ids)):
                id_b = confirmed_ids[j]
                if id_b in merged_away or id_b not in self.entities:
                    continue
                e_b = self.entities[id_b]

                # Class compatibility gate: Incompatible classes NEVER merge
                if not self.are_classes_compatible(e_a.class_name, e_b.class_name):
                    continue

                # Check horizontal distance in floor plane
                d_xy = math.hypot(e_a.centroid_floor[0] - e_b.centroid_floor[0], e_a.centroid_floor[1] - e_b.centroid_floor[1])
                if d_xy <= merge_distance_threshold_m:
                    # Canonicalize: rewrite global_entity_id on all observations from B
                    for obs in e_b.observations:
                        obs.global_entity_id = e_a.entity_id

                    # Merge entity B into entity A
                    e_a.observations.extend(e_b.observations)
                    e_a.observation_ids.extend(e_b.observation_ids)
                    e_a.local_track_ids.extend([tid for tid in e_b.local_track_ids if tid not in e_a.local_track_ids])
                    e_a.confidence_history.extend(e_b.confidence_history)
                    for c_name, count in e_b.class_votes.items():
                        e_a.class_votes[c_name] = e_a.class_votes.get(c_name, 0) + count
                    e_a.merge_history.append(f"Merged {e_b.entity_id} at d={d_xy:.3f}m")

                    # Combine points
                    if e_b.points_floor is not None:
                        e_a.points_floor = np.vstack((e_a.points_floor, e_b.points_floor))

                    # Update majority class
                    e_a.class_name = max(e_a.class_votes.items(), key=lambda x: x[1])[0]
                    e_a.first_frame_idx = min(e_a.first_frame_idx, e_b.first_frame_idx)
                    e_a.last_frame_idx = max(e_a.last_frame_idx, e_b.last_frame_idx)

                    merged_away.add(id_b)

        for mid in merged_away:
            del self.entities[mid]

        # Re-canonicalize observation IDs and re-evaluate confirmation policy after merge
        self.canonicalize_all_records()

    def canonicalize_all_records(self):
        """Canonicalize observation global IDs, recompute geometry, and re-evaluate confirmation."""
        all_entities = {**self.entities, **self.unconfirmed_entities}
        for eid, ent in all_entities.items():
            for obs in ent.observations:
                obs.global_entity_id = eid
            ent.observation_ids = [obs.obs_id for obs in ent.observations]
        self.filter_and_fuse_geometries()
        self.evaluate_confirmation_policy()

    def compute_pairwise_distances(self) -> List[Dict[str, Any]]:
        """Compute pairwise distances and derived measurement uncertainties in floor_map frame."""
        results = []
        confirmed_list = list(self.entities.values())
        n = len(confirmed_list)
        depth_sensor_sigma_m = 0.015  # calibrated D455 depth uncertainty

        for i in range(n):
            for j in range(i + 1, n):
                e1 = confirmed_list[i]
                e2 = confirmed_list[j]

                # 3D Euclidean distance between centroids in floor_map frame
                dist_3d = float(np.linalg.norm(e1.centroid_floor - e2.centroid_floor))

                # 2D horizontal distance in floor plane
                dist_2d = float(math.hypot(
                    e1.centroid_floor[0] - e2.centroid_floor[0],
                    e1.centroid_floor[1] - e2.centroid_floor[1]
                ))

                # Footprint-to-footprint exact clearance
                if e1.bbox_min_floor is not None and e2.bbox_min_floor is not None:
                    # Check 2D bounding footprint overlap and distance
                    dx = max(0.0, max(float(e1.bbox_min_floor[0] - e2.bbox_max_floor[0]), float(e2.bbox_min_floor[0] - e1.bbox_max_floor[0])))
                    dy = max(0.0, max(float(e1.bbox_min_floor[1] - e2.bbox_max_floor[1]), float(e2.bbox_min_floor[1] - e1.bbox_max_floor[1])))
                    min_clearance = float(math.hypot(dx, dy))
                else:
                    min_clearance = dist_2d

                # DERIVED UNCERTAINTY: propagate positional sample standard deviations
                s1 = e1.centroid_std_floor or 0.03
                s2 = e2.centroid_std_floor or 0.03
                sigma_dist = math.sqrt(s1 ** 2 + s2 ** 2 + depth_sensor_sigma_m ** 2)

                results.append({
                    "entity_a": e1.entity_id,
                    "class_a": e1.class_name,
                    "entity_b": e2.entity_id,
                    "class_b": e2.class_name,
                    "centroid_distance_3d_m": round(dist_3d, 4),
                    "centroid_distance_2d_m": round(dist_2d, 4),
                    "footprint_clearance_m": round(min_clearance, 4),
                    "uncertainty_std_m": round(sigma_dist, 4),
                })

        return results

    def export_point_clouds(self, output_dir: Path):
        """Export per-object PLY point clouds and combined world_objects.ply in floor_map frame."""
        objects_dir = output_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        all_points = []
        all_colors = []

        CLASS_COLORS = {
            "chair": [65, 105, 225],            # Royal Blue
            "table": [34, 139, 34],             # Forest Green
            "desk": [46, 139, 87],              # Sea Green
            "storage box": [255, 140, 0],       # Dark Orange
            "door": [178, 34, 34],              # Firebrick
            "person": [255, 20, 147],           # Deep Pink
            "backpack": [138, 43, 226],         # Blue Violet
            "handbag": [186, 85, 211],          # Medium Orchid
            "computer monitor": [0, 206, 209],  # Dark Turquoise
            "laptop": [30, 144, 255],           # Dodger Blue
            "whiteboard": [240, 230, 140],      # Khaki
            "ceiling fan": [218, 165, 32],      # Goldenrod
            "water bottle": [0, 255, 127],      # Spring Green
        }
        DEFAULT_COLOR = [180, 180, 180]

        for entity_id, entity in self.entities.items():
            if entity.points_floor is None or len(entity.points_floor) == 0:
                continue

            pts = entity.points_floor
            color = CLASS_COLORS.get(entity.class_name, DEFAULT_COLOR)

            ply_path = objects_dir / f"{entity_id}.ply"
            with open(ply_path, "w") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                for p in pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {color[0]} {color[1]} {color[2]}\n")

            all_points.append(pts)
            all_colors.append(np.tile(color, (len(pts), 1)))

        if all_points:
            combined_pts = np.vstack(all_points)
            combined_cols = np.vstack(all_colors)
            combined_path = output_dir / "world_objects.ply"
            with open(combined_path, "w") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(combined_pts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                for p, c in zip(combined_pts, combined_cols):
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def export_semantic_map(self, output_path: Path) -> dict:
        """Export semantic registry JSON with complete provenance, bounds, and quarantine reasons."""
        entities_data = {}
        for entity_id, entity in self.entities.items():
            entities_data[entity_id] = {
                "entity_id": entity.entity_id,
                "class_name": entity.class_name,
                "is_confirmed": True,
                "is_dynamic": entity.is_dynamic,
                "observation_count": entity.observation_count,
                "observation_ids": entity.observation_ids,
                "local_track_ids": entity.local_track_ids,
                "class_votes": entity.class_votes,
                "mean_confidence": round(entity.mean_confidence, 3),
                "first_seen_timestamp": entity.first_seen_timestamp,
                "last_seen_timestamp": entity.last_seen_timestamp,
                "first_frame_idx": entity.first_frame_idx,
                "last_frame_idx": entity.last_frame_idx,
                "frame_span": entity.frame_span,
                "centroid_floor": [round(float(x), 4) for x in entity.centroid_floor] if entity.centroid_floor is not None else None,
                "centroid_std_floor_m": round(entity.centroid_std_floor, 4) if entity.centroid_std_floor is not None else None,
                "bbox_min_floor": [round(float(x), 4) for x in entity.bbox_min_floor] if entity.bbox_min_floor is not None else None,
                "bbox_max_floor": [round(float(x), 4) for x in entity.bbox_max_floor] if entity.bbox_max_floor is not None else None,
                "dimensions": [round(float(x), 4) for x in entity.dimensions] if entity.dimensions is not None else None,
                "points_count": len(entity.points_floor) if entity.points_floor is not None else 0,
                "floor_footprint_polygon": entity.floor_footprint_polygon,
                "merge_history": entity.merge_history,
            }

        unconfirmed_data = {}
        for entity_id, entity in self.unconfirmed_entities.items():
            unconfirmed_data[entity_id] = {
                "entity_id": entity.entity_id,
                "class_name": entity.class_name,
                "is_confirmed": False,
                "is_dynamic": entity.is_dynamic,
                "quarantine_reason": entity.quarantine_reason or "UNCONFIRMED",
                "observation_count": entity.observation_count,
                "observation_ids": entity.observation_ids,
                "local_track_ids": entity.local_track_ids,
                "class_votes": entity.class_votes,
                "mean_confidence": round(entity.mean_confidence, 3),
                "frame_span": entity.frame_span,
                "centroid_floor": [round(float(x), 4) for x in entity.centroid_floor] if entity.centroid_floor is not None else None,
                "dimensions": [round(float(x), 4) for x in entity.dimensions] if entity.dimensions is not None else None,
            }

        semantic_map = {
            "coordinate_frame": "floor_map (ROS REP-103: X=Forward, Y=Left, Z=Height, Z=0 on floor)",
            "confirmed_entities_count": len(entities_data),
            "unconfirmed_candidates_count": len(unconfirmed_data),
            "entities": entities_data,
            "unconfirmed_candidates": unconfirmed_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(semantic_map, f, indent=2)

        return semantic_map
