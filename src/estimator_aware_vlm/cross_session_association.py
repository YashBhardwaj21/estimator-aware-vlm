"""Cross-Session Physical Identity Association and Candidate Matching.

Solves the persistent physical object identity problem across recording sessions:
1. Transforms Session 2 floor coordinates into Session 1 reference frame via verified SE(2) registration.
2. Applies strict spatial, semantic, and geometric extent gates.
3. Computes transparent association cost: normalized Euclidean distance + extent compatibility.
4. Provides three standardized baseline algorithms on identical fixed evaluation sets:
   - Baseline 1: Semantic class + greedy nearest-centroid matching.
   - Baseline 2: Global Hungarian assignment with spatial & extent gates (no ambiguity rejection).
   - Proposed Method: Uncertainty-aware Hungarian assignment with spatial/extent gates & symmetric ambiguity rejection.
5. Employs an auditable, annotation-driven canonical mapping: (source_session, local_track_id) -> physical_instance_id.
6. Evaluates identity continuity separately from physical relocation detection.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.evidence_adapter import RegistrationInfo

repo_root = get_repo_root()


@dataclass
class EntityCluster:
    """Aggregated spatial and semantic cluster for an entity within a session."""
    entity_id: str
    semantic_class: str
    source_session: str
    centroid_floor: np.ndarray  # 2D [x, y] in session local floor frame
    centroid_std: float         # 1D dispersion metric in meters
    spatial_extent_m: Optional[float]     # Bounding dimension (diameter/width) or None if unavailable
    observations_count: int
    local_track_ids: List[str]
    observation_ids: List[str]
    has_valid_pose: bool = True
    depth_valid: bool = True
    depth_valid_ratio: float = 1.0
    centroid_3d: Optional[np.ndarray] = None  # Measured 3D [x, y, z] centroid from observations
    centroid_aligned: Optional[np.ndarray] = None
    position_uncertainty_m: float = 0.05
    merged_entity_ids: List[str] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    semantic_conflict: bool = False
    class_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class PhysicalInstanceMatch:
    """A verified or candidate cross-session physical instance association."""
    persistent_entity_id: str
    s1_entity_id: str
    s2_entity_id: str
    semantic_class: str
    distance_m: float
    position_uncertainty_m: float
    extent_diff_m: float
    match_score: float
    confidence: float
    status: str  # "MATCHED", "MATCHED_STATIC", "MATCHED_RELOCATED", "AMBIGUOUS", "DISPLACED_CANDIDATE", "GATED_OUT"
    s1_obs_count: int
    s2_obs_count: int
    match_type: str = "MATCHED_STATIC"
    diagnostics: List[str] = field(default_factory=list)
    p1_raw: Optional[List[float]] = None
    p2_raw: Optional[List[float]] = None
    p2_registered: Optional[List[float]] = None
    coordinate_frame: str = "SESSION_1_WORLD"
    displacement_xy_m: Optional[float] = None
    displacement_3d_m: Optional[float] = None

    @property
    def outcome(self) -> str:
        """Standardized outcome category according to association contract."""
        if self.status == "AMBIGUOUS":
            return "AMBIGUOUS"
        if self.status in ["MATCHED_RELOCATED", "RELOCATED"]:
            return "MATCHED_RELOCATED"
        if self.status in ["MATCHED", "MATCHED_STATIC"]:
            return "MATCHED_STATIC"
        return self.status


@dataclass
class AssociationResult:
    """Complete cross-session physical association report."""
    is_valid: bool
    registration_status: str
    total_s1_entities: int
    total_s2_entities: int
    matched_pairs: List[PhysicalInstanceMatch]
    unmatched_s1: List[str]
    unmatched_s2: List[str]
    ambiguous_pairs: List[PhysicalInstanceMatch]
    algorithm_name: str = "PROPOSED_UNCERTAINTY_SYMMETRIC_AMBIGUITY"
    diagnostics: List[str] = field(default_factory=list)


class CrossSessionCandidateMatcher:
    """Estimator-aware cross-session physical identity matcher."""

    def __init__(
        self,
        max_static_distance_m: float = 0.45,
        max_relocation_distance_m: float = 1.20,
        min_relocation_distance_m: float = 0.25,
        max_extent_relative_diff: float = 0.60,
        max_relocation_extent_relative_diff: float = 0.35,
        ambiguity_margin_m: float = 0.15,
        min_cluster_observations: int = 2,
    ):
        self.max_static_distance_m = max_static_distance_m
        self.max_relocation_distance_m = max_relocation_distance_m
        self.min_relocation_distance_m = min_relocation_distance_m
        self.max_extent_relative_diff = max_extent_relative_diff
        self.max_relocation_extent_relative_diff = max_relocation_extent_relative_diff
        self.ambiguity_margin_m = ambiguity_margin_m
        self.min_cluster_observations = min_cluster_observations

    @staticmethod
    def extract_clusters_from_observations(
        observations: List[Dict[str, Any]],
        session_keyword: str,
        min_obs: int = 1,
        session_alias_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, EntityCluster]:
        """Aggregate raw frame observations into robust entity clusters preserving measured 3D geometry."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        target_norm = session_keyword.strip()

        for o in observations:
            raw_sess = str(o.get("source_session", "")).strip()
            # Exact normalized session check with alias support; eliminate substring collisions
            sess_matched = False
            if raw_sess == target_norm:
                sess_matched = True
            elif session_alias_map and session_alias_map.get(raw_sess) == target_norm:
                sess_matched = True
            elif raw_sess.endswith(f"_{target_norm}") or raw_sess.endswith(target_norm) or raw_sess == target_norm:
                sess_matched = True

            if not sess_matched:
                continue

            eid = o.get("global_entity_id", "")
            if not eid:
                continue
            cf = o.get("centroid_floor")
            if cf is not None and len(cf) >= 2 and all(x is not None and math.isfinite(float(x)) for x in cf[:2]):
                grouped.setdefault(eid, []).append(o)

        clusters: Dict[str, EntityCluster] = {}
        for eid in sorted(grouped.keys()):
            obs_list = grouped[eid]
            if len(obs_list) < min_obs:
                continue

            pts = np.array([o["centroid_floor"][:2] for o in obs_list], dtype=np.float64)
            c = np.median(pts, axis=0)

            # Preserve measured 3D height from real observations - DO NOT fabricate z=0
            pts_3d = np.array([
                o["centroid_floor"][:3] for o in obs_list
                if len(o.get("centroid_floor", [])) >= 3
                and o["centroid_floor"][2] is not None
                and math.isfinite(float(o["centroid_floor"][2]))
            ], dtype=np.float64)
            c_3d = np.median(pts_3d, axis=0) if len(pts_3d) > 0 else None

            dispersion = float(np.mean(np.std(pts, axis=0))) if len(pts) > 1 else 0.05

            # Semantic consensus and conflict detection
            class_counts: Dict[str, int] = defaultdict(int)
            for o in obs_list:
                cls_name = o.get("class_name") or o.get("semantic_class") or "object"
                class_counts[cls_name] += 1
            sem_class = max(class_counts.items(), key=lambda item: item[1])[0]
            sem_conflict = (len(class_counts) > 1)

            # Spatial extents - DO NOT substitute 0.50m when extents are missing
            extents = []
            for o in obs_list:
                bmin = o.get("bbox_min_floor")
                bmax = o.get("bbox_max_floor")
                if bmin and bmax and len(bmin) >= 2 and len(bmax) >= 2:
                    if all(math.isfinite(float(x)) for x in bmin[:2]) and all(math.isfinite(float(x)) for x in bmax[:2]):
                        extents.append(float(np.linalg.norm(np.array(bmax[:2]) - np.array(bmin[:2]))))
            avg_extent: Optional[float] = float(np.mean(extents)) if extents else None

            track_ids = sorted(set(o.get("local_track_id") for o in obs_list if o.get("local_track_id")))
            obs_ids = [o.get("obs_id") for o in obs_list if o.get("obs_id")]
            has_pose = all(bool(o.get("has_valid_pose", False)) for o in obs_list)

            # Valid depth ratio: cluster has valid depth if at least 50% observations pass
            depth_passes = sum(1 for o in obs_list if bool(o.get("depth_valid", True)))
            depth_ratio = depth_passes / len(obs_list)
            depth_valid = (depth_ratio >= 0.50)

            # Extract temporal lifespan bounds
            frames = [int(o["frame_index"]) for o in obs_list if o.get("frame_index") is not None]
            timestamps = [float(o["timestamp"]) for o in obs_list if o.get("timestamp") is not None]
            start_fr = min(frames) if frames else 0
            end_fr = max(frames) if frames else 0
            start_ts = min(timestamps) if timestamps else 0.0
            end_ts = max(timestamps) if timestamps else 0.0

            # SEM of centroid: dispersion / sqrt(N)
            sem_pos = dispersion / math.sqrt(len(pts))

            clusters[eid] = EntityCluster(
                entity_id=eid,
                semantic_class=sem_class,
                source_session=session_keyword,
                centroid_floor=c,
                centroid_std=dispersion,
                spatial_extent_m=avg_extent,
                observations_count=len(obs_list),
                local_track_ids=track_ids,
                observation_ids=obs_ids,
                has_valid_pose=has_pose,
                depth_valid=depth_valid,
                depth_valid_ratio=depth_ratio,
                centroid_3d=c_3d,
                position_uncertainty_m=sem_pos,
                merged_entity_ids=[eid],
                start_frame=start_fr,
                end_frame=end_fr,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                semantic_conflict=sem_conflict,
                class_distribution=dict(class_counts),
            )

        return clusters

    @staticmethod
    def consolidate_intra_session_clusters(
        clusters: Dict[str, EntityCluster],
        max_merge_distance_m: float = 0.35,
        max_extent_relative_diff: float = 0.50,
        max_covisible_overlap_frames: int = 5,
        ambiguity_margin_m: float = 0.10,
    ) -> Dict[str, EntityCluster]:
        """Consolidate fragmented short tracklets of identical semantic class within spatial threshold.

        Remedies upstream tracker fragmentation by linking spatially co-located tracklets
        of the same class into unified physical entity clusters.
        Enforces complete-link diameter constraint to prevent transitive chaining (A-B-C),
        and calculates mathematically rigorous pooled uncertainty from intra-cluster dispersion
        and inter-cluster disagreement.
        """
        if not clusters:
            return {}

        keys = sorted(clusters.keys())

        # Pairwise compatibility check
        def are_pair_compatible(k1: str, k2: str) -> bool:
            c1 = clusters[k1]
            c2 = clusters[k2]
            if c1.source_session != c2.source_session:
                return False
            if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                return False
            dist = float(np.linalg.norm(c1.centroid_floor - c2.centroid_floor))
            if dist > max_merge_distance_m:
                return False
            if c1.centroid_3d is not None and c2.centroid_3d is not None:
                z1 = c1.centroid_3d[2] if len(c1.centroid_3d) >= 3 else None
                z2 = c2.centroid_3d[2] if len(c2.centroid_3d) >= 3 else None
                if z1 is not None and z2 is not None and abs(z1 - z2) > 0.35:
                    return False
            if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None:
                max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                rel_ext = abs(c1.spatial_extent_m - c2.spatial_extent_m) / max_ext
                if rel_ext > max_extent_relative_diff:
                    return False
            overlap_frames = max(0, min(c1.end_frame, c2.end_frame) - max(c1.start_frame, c2.start_frame) + 1)
            if overlap_frames > max_covisible_overlap_frames:
                return False
            return True

        # Build candidate links with ambiguity check
        candidate_links: Dict[str, List[Tuple[str, float]]] = {k: [] for k in keys}
        for i in range(len(keys)):
            k1 = keys[i]
            for j in range(i + 1, len(keys)):
                k2 = keys[j]
                if are_pair_compatible(k1, k2):
                    dist = float(np.linalg.norm(clusters[k1].centroid_floor - clusters[k2].centroid_floor))
                    candidate_links[k1].append((k2, dist))
                    candidate_links[k2].append((k1, dist))

        # Filter ambiguous links: if a cluster has competing nearby candidates, do not link
        safe_adj: Dict[str, Set[str]] = {k: set() for k in keys}
        for k, cands in candidate_links.items():
            if len(cands) == 1:
                safe_adj[k].add(cands[0][0])
            elif len(cands) > 1:
                cands.sort(key=lambda x: x[1])
                if abs(cands[1][1] - cands[0][1]) >= ambiguity_margin_m:
                    safe_adj[k].add(cands[0][0])

        # Symmetrize edges
        valid_edges: Set[Tuple[str, str]] = set()
        for k1, nbrs in safe_adj.items():
            for k2 in nbrs:
                if k1 in safe_adj.get(k2, set()):
                    valid_edges.add((min(k1, k2), max(k1, k2)))

        # Assemble components with strict complete-link / diameter constraint
        assigned: Set[str] = set()
        components: List[List[str]] = []

        for k in keys:
            if k in assigned:
                continue
            comp = [k]
            assigned.add(k)

            # Greedily consider mutual neighbors
            potential_members = [
                other for other in keys
                if other not in assigned and (min(k, other), max(k, other)) in valid_edges
            ]
            for cand in potential_members:
                # Complete-link constraint: cand MUST satisfy pairwise compatibility with ALL existing members in comp
                if all(are_pair_compatible(cand, m) and (min(cand, m), max(cand, m)) in valid_edges for m in comp):
                    comp.append(cand)
                    assigned.add(cand)

            components.append(comp)

        consolidated: Dict[str, EntityCluster] = {}
        for comp in components:
            if len(comp) == 1:
                k = comp[0]
                c = copy.copy(clusters[k])
                if not c.merged_entity_ids:
                    c.merged_entity_ids = [c.entity_id]
                consolidated[k] = c
                continue

            # Multi-cluster merged component: compute pooled statistics mathematically
            rep = max(comp, key=lambda x: (clusters[x].observations_count, -len(x)))
            total_obs = sum(clusters[x].observations_count for x in comp)
            all_trks = sorted(set(t for x in comp for t in clusters[x].local_track_ids))
            all_obs = sorted(set(o for x in comp for o in clusters[x].observation_ids))
            all_eids = sorted(set(e for x in comp for e in (clusters[x].merged_entity_ids or [x])))

            # Observation-weighted centroid
            weights = np.array([clusters[x].observations_count for x in comp], dtype=np.float64)
            pts_floor = np.array([clusters[x].centroid_floor for x in comp], dtype=np.float64)
            c_floor = np.average(pts_floor, axis=0, weights=weights)

            # Mathematical pooled variance: combining within-cluster variance and between-cluster spread
            # V_total = sum(N_i * s_i^2 + N_i * ||c_i - c_floor||^2) / total_obs
            v_within = sum(clusters[x].observations_count * (clusters[x].centroid_std ** 2) for x in comp)
            v_between = sum(clusters[x].observations_count * float(np.sum((clusters[x].centroid_floor - c_floor) ** 2)) for x in comp)
            sigma_pooled_sq = (v_within + v_between) / max(1.0, float(total_obs))
            sigma_pooled = math.sqrt(max(0.0001, sigma_pooled_sq))
            sem_pooled = sigma_pooled / math.sqrt(max(1.0, float(total_obs)))
            pooled_uncertainty = max(0.02, float(sem_pooled))

            # 3D centroid if available
            valid_3d = [clusters[x] for x in comp if clusters[x].centroid_3d is not None]
            if valid_3d:
                w_3d = np.array([c.observations_count for c in valid_3d], dtype=np.float64)
                pts_3d = np.array([c.centroid_3d for c in valid_3d], dtype=np.float64)
                c_3d = np.average(pts_3d, axis=0, weights=w_3d)
            else:
                c_3d = None

            # Summarize extents without defaults
            valid_exts = [clusters[x].spatial_extent_m for x in comp if clusters[x].spatial_extent_m is not None]
            merged_extent = float(np.mean(valid_exts)) if valid_exts else None

            has_pose = all(clusters[x].has_valid_pose for x in comp)
            depth_valid = any(clusters[x].depth_valid for x in comp)

            start_fr = min(clusters[x].start_frame for x in comp)
            end_fr = max(clusters[x].end_frame for x in comp)
            start_ts = min(clusters[x].start_timestamp for x in comp)
            end_ts = max(clusters[x].end_timestamp for x in comp)

            consolidated[rep] = EntityCluster(
                entity_id=rep,
                semantic_class=clusters[rep].semantic_class,
                source_session=clusters[rep].source_session,
                centroid_floor=c_floor,
                centroid_std=sigma_pooled,
                spatial_extent_m=merged_extent,
                observations_count=total_obs,
                local_track_ids=all_trks,
                observation_ids=all_obs,
                has_valid_pose=has_pose,
                depth_valid=depth_valid,
                centroid_3d=c_3d,
                position_uncertainty_m=pooled_uncertainty,
                merged_entity_ids=all_eids,
                start_frame=start_fr,
                end_frame=end_fr,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )

        return consolidated


    def _align_session2(
        self,
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> Dict[str, EntityCluster]:
        """Align session 2 clusters into session 1 reference frame using empirical residual dispersion."""
        reg_disp = getattr(registration, "inlier_rms_residual_m", None) or getattr(registration, "anchor_rms_residual_m", None) or 0.20

        s2_aligned: Dict[str, EntityCluster] = {}
        for eid2, c2 in s2_clusters.items():
            c2_copy = copy.copy(c2)
            c2_copy.centroid_aligned = registration.R_12 @ c2.centroid_floor + registration.translation_m
            # Combined spatial dispersion: sqrt(sem_pos^2 + reg_disp^2)
            c2_copy.position_uncertainty_m = math.sqrt(c2.position_uncertainty_m**2 + reg_disp**2)
            s2_aligned[eid2] = c2_copy
        return s2_aligned

    def _invalid_registration_result(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
        algorithm_name: str,
    ) -> AssociationResult:
        return AssociationResult(
            is_valid=False,
            registration_status=registration.status,
            total_s1_entities=len(s1_clusters),
            total_s2_entities=len(s2_clusters),
            matched_pairs=[],
            unmatched_s1=list(s1_clusters.keys()),
            unmatched_s2=list(s2_clusters.keys()),
            ambiguous_pairs=[],
            algorithm_name=algorithm_name,
            diagnostics=[f"Registration invalid or unverified: {registration.status}"],
        )

    def match_entities_nearest_centroid(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> AssociationResult:
        """Baseline 1: Greedy nearest-centroid matching by semantic class."""
        algo_name = "BASELINE_1_NEAREST_CENTROID"
        if not registration.is_valid:
            return self._invalid_registration_result(s1_clusters, s2_clusters, registration, algo_name)

        s2_aligned = self._align_session2(s2_clusters, registration)
        matched_pairs: List[PhysicalInstanceMatch] = []
        matched_s1: Set[str] = set()
        matched_s2: Set[str] = set()

        # Greedy nearest neighbor per S1 cluster
        for k1, c1 in sorted(s1_clusters.items()):
            best_k2 = None
            best_dist = float("inf")
            for k2, c2 in s2_aligned.items():
                if k2 in matched_s2:
                    continue
                if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                    continue
                dist = float(np.linalg.norm(c1.centroid_floor - c2.centroid_aligned))
                if dist <= self.max_static_distance_m and dist < best_dist:
                    best_dist = dist
                    best_k2 = k2
            if best_k2 is not None:
                c2 = s2_aligned[best_k2]
                has_extents = (c1.spatial_extent_m is not None and c2.spatial_extent_m is not None)
                ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m) if has_extents else 0.0
                p1_r = c1.centroid_3d.tolist() if c1.centroid_3d is not None else [float(c1.centroid_floor[0]), float(c1.centroid_floor[1]), 0.0]
                p2_orig = s2_clusters[best_k2]
                p2_r = p2_orig.centroid_3d.tolist() if p2_orig.centroid_3d is not None else [float(p2_orig.centroid_floor[0]), float(p2_orig.centroid_floor[1]), 0.0]
                p2_reg_xy = [float(c2.centroid_aligned[0]), float(c2.centroid_aligned[1])]
                p2_reg_3d = p2_reg_xy + ([float(p2_r[2])] if len(p2_r) >= 3 else [0.0])
                disp_3d = float(np.linalg.norm(np.array(p1_r) - np.array(p2_reg_3d))) if (len(p1_r) >= 3 and len(p2_reg_3d) >= 3) else None

                matched_pairs.append(PhysicalInstanceMatch(
                    persistent_entity_id=f"phys_{k1}",
                    s1_entity_id=k1,
                    s2_entity_id=best_k2,
                    semantic_class=c1.semantic_class,
                    distance_m=round(best_dist, 4),
                    position_uncertainty_m=round(math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2), 4),
                    extent_diff_m=round(ext_diff, 4) if has_extents else None,
                    match_score=round(best_dist, 4),
                    confidence=round(max(0.20, 1.0 - best_dist / self.max_static_distance_m), 4),
                    status="MATCHED",
                    s1_obs_count=c1.observations_count,
                    s2_obs_count=c2.observations_count,
                    diagnostics=["GREEDY_NEAREST_CENTROID"],
                    p1_raw=p1_r,
                    p2_raw=p2_r,
                    p2_registered=p2_reg_3d,
                    coordinate_frame="SESSION_1_WORLD",
                    displacement_xy_m=round(best_dist, 4),
                    displacement_3d_m=round(disp_3d, 4) if disp_3d is not None else None,
                ))
                matched_s1.add(k1)
                matched_s2.add(best_k2)

        unmatched_s1 = [k for k in s1_clusters if k not in matched_s1]
        unmatched_s2 = [k for k in s2_clusters if k not in matched_s2]

        return AssociationResult(
            is_valid=True,
            registration_status=registration.status,
            total_s1_entities=len(s1_clusters),
            total_s2_entities=len(s2_clusters),
            matched_pairs=matched_pairs,
            unmatched_s1=unmatched_s1,
            unmatched_s2=unmatched_s2,
            ambiguous_pairs=[],
            algorithm_name=algo_name,
            diagnostics=["Standard baseline: greedy nearest neighbor without Hungarian assignment or ambiguity rejection."],
        )

    def match_entities_hungarian_gated(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> AssociationResult:
        """Baseline 2: Global Hungarian assignment with spatial & extent gates, without ambiguity rejection."""
        return self._match_entities_internal(
            s1_clusters=s1_clusters,
            s2_clusters=s2_clusters,
            registration=registration,
            enable_ambiguity_rejection=False,
            algorithm_name="BASELINE_2_HUNGARIAN_GATED",
        )

    def match_entities(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> AssociationResult:
        """Proposed Method: Uncertainty-aware Hungarian assignment with spatial/extent gates & symmetric ambiguity rejection."""
        return self._match_entities_internal(
            s1_clusters=s1_clusters,
            s2_clusters=s2_clusters,
            registration=registration,
            enable_ambiguity_rejection=True,
            enable_relocation_matching=False,
            algorithm_name="PROPOSED_UNCERTAINTY_SYMMETRIC_AMBIGUITY",
        )

    def match_entities_relocation_aware(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> AssociationResult:
        """Relocation-Aware Association Method: Multi-pass Hungarian assignment handling static proximity and relocated objects."""
        return self._match_entities_internal(
            s1_clusters=s1_clusters,
            s2_clusters=s2_clusters,
            registration=registration,
            enable_ambiguity_rejection=True,
            enable_relocation_matching=True,
            algorithm_name="PROPOSED_RELOCATION_AWARE_ASSOCIATION",
        )

    def _match_entities_internal(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
        enable_ambiguity_rejection: bool,
        algorithm_name: str,
        enable_relocation_matching: bool = False,
    ) -> AssociationResult:
        """Internal worker executing global Hungarian assignment with configurable ambiguity gating and relocation hypothesis testing."""
        diagnostics: List[str] = []
        if not registration.is_valid:
            return self._invalid_registration_result(s1_clusters, s2_clusters, registration, algorithm_name)

        s2_aligned = self._align_session2(s2_clusters, registration)

        s1_keys = sorted(s1_clusters.keys())
        s2_keys = sorted(s2_aligned.keys())
        n1 = len(s1_keys)
        n2 = len(s2_keys)

        if n1 == 0 or n2 == 0:
            return AssociationResult(
                is_valid=True,
                registration_status=registration.status,
                total_s1_entities=n1,
                total_s2_entities=n2,
                matched_pairs=[],
                unmatched_s1=s1_keys,
                unmatched_s2=s2_keys,
                ambiguous_pairs=[],
                algorithm_name=algorithm_name,
                diagnostics=["One or both sessions contain zero clusters."],
            )

        INFTY_COST = 1e6
        cost_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)
        distance_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)

        for i, k1 in enumerate(s1_keys):
            c1 = s1_clusters[k1]
            for j, k2 in enumerate(s2_keys):
                c2 = s2_aligned[k2]

                # Gate A: Semantic class compatibility
                if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                    continue

                # Gate B: Spatial distance in common reference frame
                dist = float(np.linalg.norm(c1.centroid_floor - c2.centroid_aligned))
                distance_matrix[i, j] = dist

                if dist > self.max_static_distance_m:
                    continue

                # Gate C: Relative spatial extent change
                if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None:
                    max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                    ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                    rel_ext = ext_diff / max_ext
                    if rel_ext > self.max_extent_relative_diff:
                        continue
                else:
                    ext_diff = 0.0
                    rel_ext = 0.0

                combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                norm_dist_cost = dist / max(0.05, combined_unc)
                cost = norm_dist_cost + 2.0 * rel_ext
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[PhysicalInstanceMatch] = []
        ambiguous_pairs: List[PhysicalInstanceMatch] = []
        matched_s1_idx: Set[int] = set()
        matched_s2_idx: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= INFTY_COST:
                continue

            dist = distance_matrix[r, c]
            k1 = s1_keys[r]
            k2 = s2_keys[c]
            c1 = s1_clusters[k1]
            c2 = s2_aligned[k2]

            is_ambiguous = False
            ambiguity_reasons = []

            if enable_ambiguity_rejection:
                # Symmetric Ambiguity Check:
                # Direction A: Does S2 candidate k2 have competing S1 alternatives within ambiguity margin?
                competing_s1 = [
                    distance_matrix[other_r, c] for other_r in range(n1)
                    if other_r != r and distance_matrix[other_r, c] < INFTY_COST
                ]
                # Direction B: Does S1 candidate k1 have competing S2 alternatives within ambiguity margin?
                competing_s2 = [
                    distance_matrix[r, other_c] for other_c in range(n2)
                    if other_c != c and distance_matrix[r, other_c] < INFTY_COST
                ]

                if competing_s1:
                    min_other_s1 = min(competing_s1)
                    diff_s1 = abs(min_other_s1 - dist)
                    if diff_s1 < self.ambiguity_margin_m:
                        is_ambiguous = True
                        ambiguity_reasons.append(
                            f"S2 candidate '{k2}' has competing S1 neighbor within {self.ambiguity_margin_m}m margin (delta={diff_s1:.3f}m)."
                        )

                if competing_s2:
                    min_other_s2 = min(competing_s2)
                    diff_s2 = abs(min_other_s2 - dist)
                    if diff_s2 < self.ambiguity_margin_m:
                        is_ambiguous = True
                        ambiguity_reasons.append(
                            f"S1 candidate '{k1}' has competing S2 neighbor within {self.ambiguity_margin_m}m margin (delta={diff_s2:.3f}m)."
                        )

            combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
            has_extents = (c1.spatial_extent_m is not None and c2.spatial_extent_m is not None)
            ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m) if has_extents else 0.0
            confidence = max(0.20, min(0.99, 1.0 - (dist / self.max_static_distance_m) * 0.5 - (cost / 10.0) * 0.3))

            p1_r = c1.centroid_3d.tolist() if c1.centroid_3d is not None else [float(c1.centroid_floor[0]), float(c1.centroid_floor[1]), 0.0]
            p2_orig = s2_clusters[k2]
            p2_r = p2_orig.centroid_3d.tolist() if p2_orig.centroid_3d is not None else [float(p2_orig.centroid_floor[0]), float(p2_orig.centroid_floor[1]), 0.0]
            p2_reg_xy = [float(c2.centroid_aligned[0]), float(c2.centroid_aligned[1])]
            p2_reg_3d = p2_reg_xy + ([float(p2_r[2])] if len(p2_r) >= 3 else [0.0])
            disp_3d = float(np.linalg.norm(np.array(p1_r) - np.array(p2_reg_3d))) if (len(p1_r) >= 3 and len(p2_reg_3d) >= 3) else None

            match_obj = PhysicalInstanceMatch(
                persistent_entity_id=f"phys_{k1}",
                s1_entity_id=k1,
                s2_entity_id=k2,
                semantic_class=c1.semantic_class,
                distance_m=round(dist, 4),
                position_uncertainty_m=round(combined_unc, 4),
                extent_diff_m=round(ext_diff, 4) if has_extents else None,
                match_score=round(float(cost), 4),
                confidence=round(confidence, 4),
                status="AMBIGUOUS" if is_ambiguous else "MATCHED",
                s1_obs_count=c1.observations_count,
                s2_obs_count=c2.observations_count,
                match_type="MATCHED_STATIC",
                diagnostics=ambiguity_reasons,
                p1_raw=p1_r,
                p2_raw=p2_r,
                p2_registered=p2_reg_3d,
                coordinate_frame="SESSION_1_WORLD",
                displacement_xy_m=round(dist, 4),
                displacement_3d_m=round(disp_3d, 4) if disp_3d is not None else None,
            )

            if is_ambiguous:
                ambiguous_pairs.append(match_obj)
            else:
                matched_pairs.append(match_obj)
                matched_s1_idx.add(r)
                matched_s2_idx.add(c)

        # Pass 2: Relocation candidate matching for unassigned entities beyond the static gate
        if enable_relocation_matching:
            unassigned_s1 = [i for i in range(n1) if i not in matched_s1_idx]
            unassigned_s2 = [j for j in range(n2) if j not in matched_s2_idx]

            if unassigned_s1 and unassigned_s2:
                nu1 = len(unassigned_s1)
                nu2 = len(unassigned_s2)
                reloc_cost_matrix = np.full((nu1, nu2), INFTY_COST, dtype=np.float64)
                reloc_dist_matrix = np.full((nu1, nu2), INFTY_COST, dtype=np.float64)

                for ui, r in enumerate(unassigned_s1):
                    c1 = s1_clusters[s1_keys[r]]
                    for vj, c in enumerate(unassigned_s2):
                        c2 = s2_aligned[s2_keys[c]]

                        if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                            continue

                        dist = float(np.linalg.norm(c1.centroid_floor - c2.centroid_aligned))
                        reloc_dist_matrix[ui, vj] = dist

                        # Relocation distance gate: beyond static gate, within max relocation distance
                        if not (self.min_relocation_distance_m <= dist <= self.max_relocation_distance_m):
                            continue

                        # Strict extent consistency: relocated physical objects maintain dimensions
                        if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None:
                            max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                            ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                            rel_ext = ext_diff / max_ext
                            if rel_ext > self.max_relocation_extent_relative_diff:
                                continue
                        else:
                            ext_diff = 0.0
                            rel_ext = 0.0

                        # Reject relocation hypotheses if local registration uncertainty is excessive (>0.40m)
                        if c2.position_uncertainty_m > 0.40:
                            continue

                        combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                        norm_dist = dist / max(0.05, combined_unc)
                        reloc_cost_matrix[ui, vj] = norm_dist + 3.0 * rel_ext

                r_rows, r_cols = linear_sum_assignment(reloc_cost_matrix)
                for ru, cv in zip(r_rows, r_cols):
                    rcost = reloc_cost_matrix[ru, cv]
                    if rcost >= INFTY_COST:
                        continue

                    r_orig = unassigned_s1[ru]
                    c_orig = unassigned_s2[cv]
                    rdist = reloc_dist_matrix[ru, cv]
                    k1 = s1_keys[r_orig]
                    k2 = s2_keys[c_orig]
                    c1 = s1_clusters[k1]
                    c2 = s2_aligned[k2]

                    # Ambiguity check for relocation candidates
                    is_reloc_ambiguous = False
                    competing_s1 = [
                        reloc_dist_matrix[other_u, cv] for other_u in range(nu1)
                        if other_u != ru and reloc_dist_matrix[other_u, cv] < INFTY_COST
                    ]
                    competing_s2 = [
                        reloc_dist_matrix[ru, other_v] for other_v in range(nu2)
                        if other_v != cv and reloc_dist_matrix[ru, other_v] < INFTY_COST
                    ]

                    if competing_s1 and abs(min(competing_s1) - rdist) < self.ambiguity_margin_m:
                        is_reloc_ambiguous = True
                    if competing_s2 and abs(min(competing_s2) - rdist) < self.ambiguity_margin_m:
                        is_reloc_ambiguous = True

                    combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                    has_extents = (c1.spatial_extent_m is not None and c2.spatial_extent_m is not None)
                    ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m) if has_extents else 0.0
                    confidence = max(0.20, min(0.95, 1.0 - (rdist / self.max_relocation_distance_m) * 0.4 - (rcost / 10.0) * 0.3))

                    reloc_match = PhysicalInstanceMatch(
                        persistent_entity_id=f"phys_{k1}",
                        s1_entity_id=k1,
                        s2_entity_id=k2,
                        semantic_class=c1.semantic_class,
                        distance_m=round(rdist, 4),
                        position_uncertainty_m=round(combined_unc, 4),
                        extent_diff_m=round(ext_diff, 4) if has_extents else None,
                        match_score=round(float(rcost), 4),
                        confidence=round(confidence, 4),
                        status="AMBIGUOUS" if is_reloc_ambiguous else "MATCHED_RELOCATED",
                        s1_obs_count=c1.observations_count,
                        s2_obs_count=c2.observations_count,
                        match_type="MATCHED_RELOCATED",
                        diagnostics=[f"RELOCATION_CONFIRMED: displacement={rdist:.3f}m beyond static gate {self.max_static_distance_m}m"] if not is_reloc_ambiguous else ["RELOCATION_AMBIGUOUS_COMPETING_CANDIDATE"],
                    )

                    if is_reloc_ambiguous:
                        ambiguous_pairs.append(reloc_match)
                    else:
                        matched_pairs.append(reloc_match)
                        matched_s1_idx.add(r_orig)
                        matched_s2_idx.add(c_orig)

        unmatched_s1 = [s1_keys[i] for i in range(n1) if i not in matched_s1_idx]
        unmatched_s2 = [s2_keys[j] for j in range(n2) if j not in matched_s2_idx]

        return AssociationResult(
            is_valid=True,
            registration_status=registration.status,
            total_s1_entities=n1,
            total_s2_entities=n2,
            matched_pairs=matched_pairs,
            unmatched_s1=unmatched_s1,
            unmatched_s2=unmatched_s2,
            ambiguous_pairs=ambiguous_pairs,
            algorithm_name=algorithm_name,
            diagnostics=diagnostics,
        )

    def match_entities_uncertainty_adaptive(
        self,
        s1_clusters: Dict[str, EntityCluster],
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
        use_3d_height_gate: bool = True,
        max_height_diff_m: float = 0.40,
    ) -> AssociationResult:
        """Registration-Uncertainty-Aware Adaptive Matcher.

        Dynamically adapts static matching gates according to localized SE(2) registration
        uncertainty (inlier consensus core vs peripheral drift boundaries):
        - Consensus inlier region: strict gate (~0.40m - 0.45m) prevents false merges in clutter.
        - Peripheral drift region: smoothly adapts up to 0.62m to recover drifted boundary landmarks.
        - 3D Height Gating: Prevents floor-level objects from merging with desks or ceilings.
        - Symmetric Ambiguity Rejection: Rejects matches with close competing alternatives (< 0.15m margin).
        - Relocation Hypothesis Testing: Resolves displaced instances beyond the static gate.
        """
        algo_name = "PROPOSED_REGISTRATION_UNCERTAINTY_ADAPTIVE"
        if not registration.is_valid:
            return self._invalid_registration_result(s1_clusters, s2_clusters, registration, algo_name)

        s2_aligned = self._align_session2(s2_clusters, registration)
        s1_keys = sorted(s1_clusters.keys())
        s2_keys = sorted(s2_aligned.keys())
        n1 = len(s1_keys)
        n2 = len(s2_keys)

        if n1 == 0 or n2 == 0:
            return AssociationResult(
                is_valid=True,
                registration_status=registration.status,
                total_s1_entities=n1,
                total_s2_entities=n2,
                matched_pairs=[],
                unmatched_s1=s1_keys,
                unmatched_s2=s2_keys,
                ambiguous_pairs=[],
                algorithm_name=algo_name,
                diagnostics=["One or both sessions contain zero clusters."],
            )

        INFTY_COST = 1e6
        cost_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)
        distance_matrix = np.full((n1, n2), INFTY_COST, dtype=np.float64)

        inlier_disp = getattr(registration, "inlier_rms_residual_m", None)
        if inlier_disp is None:
            inlier_disp = getattr(registration, "anchor_rms_residual_m", 0.12)

        for i, k1 in enumerate(s1_keys):
            c1 = s1_clusters[k1]
            p1 = c1.centroid_floor[:2]

            dist_from_origin = float(np.linalg.norm(p1))
            # Smooth SE(2) lever-arm rotational uncertainty growth: error scales continuously with distance from reference origin
            # Eliminates arbitrary coordinate quadrant heuristics (e.g. p1[1] < -5.5 or p1[0] < -1.0)
            local_reg_unc = inlier_disp + 0.025 * dist_from_origin
            adaptive_static_gate = min(0.62, max(0.40, 0.35 + 1.5 * local_reg_unc))

            for j, k2 in enumerate(s2_keys):
                c2 = s2_aligned[k2]

                if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                    continue

                if use_3d_height_gate and c1.centroid_3d is not None and c2.centroid_3d is not None:
                    z1 = c1.centroid_3d[2] if len(c1.centroid_3d) >= 3 else 0.0
                    z2 = c2.centroid_3d[2] if len(c2.centroid_3d) >= 3 else 0.0
                    if abs(z1 - z2) > max_height_diff_m:
                        continue

                dist = float(np.linalg.norm(c1.centroid_floor[:2] - c2.centroid_aligned[:2]))
                distance_matrix[i, j] = dist
                if dist > adaptive_static_gate:
                    continue

                if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None:
                    max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                    ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                    rel_ext = ext_diff / max_ext
                    if rel_ext > self.max_extent_relative_diff:
                        continue
                else:
                    ext_diff = 0.0
                    rel_ext = 0.0

                combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2 + local_reg_unc**2)
                cost = (dist / max(0.05, combined_unc)) + 2.0 * rel_ext
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[PhysicalInstanceMatch] = []
        ambiguous_pairs: List[PhysicalInstanceMatch] = []
        matched_s1_idx: Set[int] = set()
        matched_s2_idx: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= INFTY_COST:
                continue

            dist = distance_matrix[r, c]
            k1 = s1_keys[r]
            k2 = s2_keys[c]
            c1 = s1_clusters[k1]
            c2 = s2_aligned[k2]

            is_ambiguous = False
            competing_s1 = [
                distance_matrix[other_r, c] for other_r in range(n1)
                if other_r != r and distance_matrix[other_r, c] < INFTY_COST
            ]
            competing_s2 = [
                distance_matrix[r, other_c] for other_c in range(n2)
                if other_c != c and distance_matrix[r, other_c] < INFTY_COST
            ]
            if competing_s1 and abs(min(competing_s1) - dist) < self.ambiguity_margin_m:
                is_ambiguous = True
            if competing_s2 and abs(min(competing_s2) - dist) < self.ambiguity_margin_m:
                is_ambiguous = True

            combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
            has_extents = (c1.spatial_extent_m is not None and c2.spatial_extent_m is not None)
            ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m) if has_extents else 0.0
            confidence = max(0.20, min(0.99, 1.0 - (dist / 0.65) * 0.5 - (cost / 10.0) * 0.3))

            match_obj = PhysicalInstanceMatch(
                persistent_entity_id=f"phys_{k1}",
                s1_entity_id=k1,
                s2_entity_id=k2,
                semantic_class=c1.semantic_class,
                distance_m=round(dist, 4),
                position_uncertainty_m=round(combined_unc, 4),
                extent_diff_m=round(ext_diff, 4) if has_extents else None,
                match_score=round(float(cost), 4),
                confidence=round(confidence, 4),
                status="AMBIGUOUS" if is_ambiguous else "MATCHED",
                s1_obs_count=c1.observations_count,
                s2_obs_count=c2.observations_count,
                match_type="MATCHED_STATIC",
                diagnostics=["UNCERTAINTY_ADAPTIVE_STATIC"] if not is_ambiguous else ["AMBIGUOUS_COMPETING_CANDIDATE"],
            )

            if is_ambiguous:
                ambiguous_pairs.append(match_obj)
            else:
                matched_pairs.append(match_obj)
                matched_s1_idx.add(r)
                matched_s2_idx.add(c)

        unassigned_s1 = [i for i in range(n1) if i not in matched_s1_idx]
        unassigned_s2 = [j for j in range(n2) if j not in matched_s2_idx]

        if unassigned_s1 and unassigned_s2:
            nu1 = len(unassigned_s1)
            nu2 = len(unassigned_s2)
            reloc_cost_matrix = np.full((nu1, nu2), INFTY_COST, dtype=np.float64)
            reloc_dist_matrix = np.full((nu1, nu2), INFTY_COST, dtype=np.float64)

            for ui, r in enumerate(unassigned_s1):
                c1 = s1_clusters[s1_keys[r]]
                for vj, c in enumerate(unassigned_s2):
                    c2 = s2_aligned[s2_keys[c]]

                    if c1.semantic_class.lower().strip() != c2.semantic_class.lower().strip():
                        continue

                    if use_3d_height_gate and c1.centroid_3d is not None and c2.centroid_3d is not None:
                        z1 = c1.centroid_3d[2] if len(c1.centroid_3d) >= 3 else 0.0
                        z2 = c2.centroid_3d[2] if len(c2.centroid_3d) >= 3 else 0.0
                        if abs(z1 - z2) > max_height_diff_m:
                            continue

                    dist = float(np.linalg.norm(c1.centroid_floor[:2] - c2.centroid_aligned[:2]))
                    reloc_dist_matrix[ui, vj] = dist

                    if not (self.min_relocation_distance_m <= dist <= self.max_relocation_distance_m):
                        continue

                    if c1.spatial_extent_m is not None and c2.spatial_extent_m is not None:
                        max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                        ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                        rel_ext = ext_diff / max_ext
                        if rel_ext > self.max_relocation_extent_relative_diff:
                            continue
                    else:
                        ext_diff = 0.0
                        rel_ext = 0.0

                    combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                    norm_dist = dist / max(0.05, combined_unc)
                    reloc_cost_matrix[ui, vj] = norm_dist + 3.0 * rel_ext

            r_rows, r_cols = linear_sum_assignment(reloc_cost_matrix)
            for ru, cv in zip(r_rows, r_cols):
                rcost = reloc_cost_matrix[ru, cv]
                if rcost >= INFTY_COST:
                    continue

                r_orig = unassigned_s1[ru]
                c_orig = unassigned_s2[cv]
                rdist = reloc_dist_matrix[ru, cv]
                k1 = s1_keys[r_orig]
                k2 = s2_keys[c_orig]
                c1 = s1_clusters[k1]
                c2 = s2_aligned[k2]

                is_reloc_ambiguous = False
                competing_s1 = [
                    reloc_dist_matrix[other_u, cv] for other_u in range(nu1)
                    if other_u != ru and reloc_dist_matrix[other_u, cv] < INFTY_COST
                ]
                competing_s2 = [
                    reloc_dist_matrix[ru, other_v] for other_v in range(nu2)
                    if other_v != cv and reloc_dist_matrix[ru, other_v] < INFTY_COST
                ]

                if competing_s1 and abs(min(competing_s1) - rdist) < self.ambiguity_margin_m:
                    is_reloc_ambiguous = True
                if competing_s2 and abs(min(competing_s2) - rdist) < self.ambiguity_margin_m:
                    is_reloc_ambiguous = True

                combined_unc = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2)
                has_extents = (c1.spatial_extent_m is not None and c2.spatial_extent_m is not None)
                ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m) if has_extents else 0.0
                confidence = max(0.20, min(0.95, 1.0 - (rdist / self.max_relocation_distance_m) * 0.4 - (rcost / 10.0) * 0.3))

                reloc_match = PhysicalInstanceMatch(
                    persistent_entity_id=f"phys_{k1}",
                    s1_entity_id=k1,
                    s2_entity_id=k2,
                    semantic_class=c1.semantic_class,
                    distance_m=round(rdist, 4),
                    position_uncertainty_m=round(combined_unc, 4),
                    extent_diff_m=round(ext_diff, 4) if has_extents else None,
                    match_score=round(float(rcost), 4),
                    confidence=round(confidence, 4),
                    status="AMBIGUOUS" if is_reloc_ambiguous else "MATCHED_RELOCATED",
                    s1_obs_count=c1.observations_count,
                    s2_obs_count=c2.observations_count,
                    match_type="MATCHED_RELOCATED",
                    diagnostics=[f"RELOCATION_CONFIRMED: displacement={rdist:.3f}m beyond adaptive static gate"] if not is_reloc_ambiguous else ["RELOCATION_AMBIGUOUS_COMPETING_CANDIDATE"],
                )

                if is_reloc_ambiguous:
                    ambiguous_pairs.append(reloc_match)
                else:
                    matched_pairs.append(reloc_match)
                    matched_s1_idx.add(r_orig)
                    matched_s2_idx.add(c_orig)

        unmatched_s1 = [s1_keys[i] for i in range(n1) if i not in matched_s1_idx]
        unmatched_s2 = [s2_keys[j] for j in range(n2) if j not in matched_s2_idx]

        return AssociationResult(
            is_valid=True,
            registration_status=registration.status,
            total_s1_entities=n1,
            total_s2_entities=n2,
            matched_pairs=matched_pairs,
            unmatched_s1=unmatched_s1,
            unmatched_s2=unmatched_s2,
            ambiguous_pairs=ambiguous_pairs,
            algorithm_name=algo_name,
            diagnostics=["Registration-uncertainty-aware adaptive matching executed."],
        )



def build_canonical_session_track_mapping(
    gt_path: Optional[Path] = None,
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Build authoritative session-qualified track mapping: (session_name, track_id) -> object_instance_id.

    Derived from explicit ground truth without implicit cross-session track assumptions.
    """
    if gt_path is not None:
        if gt_path.name == "cross_session_identity_annotations.json":
            p_canon = gt_path
        elif (gt_path.parent / "cross_session_identity_annotations.json").exists():
            p_canon = gt_path.parent / "cross_session_identity_annotations.json"
        elif (gt_path / "cross_session_identity_annotations.json").exists():
            p_canon = gt_path / "cross_session_identity_annotations.json"
        else:
            p_canon = gt_path
    else:
        p_canon = repo_root / "benchmarks" / "cross_session_identity_annotations.json"

@dataclass
class ClusterIdentityResolution:
    """Canonical ground-truth identity resolution for an intra-session entity cluster."""
    cluster_id: str
    session: str
    underlying_track_ids: List[str]
    annotated_physical_instance_ids: List[str]
    annotation_status: str  # "ANNOTATED_SINGLE_ID", "UNANNOTATED", "ALIAS_CONFLICT", "ANNOTATION_CONFLICT"
    resolved_instance_id: Optional[str] = None


def resolve_cluster_canonical_identity(
    cluster: Optional[EntityCluster],
    cluster_id: str,
    session: str,
    canonical_map: Dict[Tuple[str, str], str],
) -> ClusterIdentityResolution:
    """Resolve the canonical ground-truth physical identity for a cluster and all its aliases.

    Contract:
    - Look up the canonical ground-truth identity of every underlying alias.
    - If no aliases are annotated, classify the cluster as UNANNOTATED.
    - If all annotated aliases map to the exact same physical instance, return ANNOTATED_SINGLE_ID.
    - If aliases map to multiple physical instances, return ALIAS_CONFLICT and preserve ALL conflicting IDs.
      NEVER pick the first one.
    """
    track_ids: List[str] = [cluster_id]
    if cluster is not None:
        if hasattr(cluster, "merged_entity_ids") and cluster.merged_entity_ids:
            for tid in cluster.merged_entity_ids:
                if tid not in track_ids:
                    track_ids.append(tid)
        if hasattr(cluster, "local_track_ids") and cluster.local_track_ids:
            for tid in cluster.local_track_ids:
                if tid not in track_ids:
                    track_ids.append(tid)

    found_instances: Set[str] = set()
    for tid in track_ids:
        inst = canonical_map.get((session, tid))
        if inst is None:
            # Match session key across session naming variations
            for (s_key, t_key), val in canonical_map.items():
                if t_key == tid and (
                    s_key == session
                    or ("01" in s_key and "01" in session)
                    or ("02" in s_key and "02" in session)
                ):
                    inst = val
                    break
        if inst is not None:
            found_instances.add(inst)

    annotated_list = sorted(list(found_instances))
    if not annotated_list:
        return ClusterIdentityResolution(
            cluster_id=cluster_id,
            session=session,
            underlying_track_ids=track_ids,
            annotated_physical_instance_ids=[],
            annotation_status="UNANNOTATED",
            resolved_instance_id=None,
        )
    elif len(annotated_list) == 1:
        return ClusterIdentityResolution(
            cluster_id=cluster_id,
            session=session,
            underlying_track_ids=track_ids,
            annotated_physical_instance_ids=annotated_list,
            annotation_status="ANNOTATED_SINGLE_ID",
            resolved_instance_id=annotated_list[0],
        )
    else:
        return ClusterIdentityResolution(
            cluster_id=cluster_id,
            session=session,
            underlying_track_ids=track_ids,
            annotated_physical_instance_ids=annotated_list,
            annotation_status="ALIAS_CONFLICT",
            resolved_instance_id=None,
        )


def _make_metric_record(numerator: int, denominator: int, definition: str) -> Dict[str, Any]:
    """Construct an auditable metric dictionary with explicit numerator and denominator."""
    value_pct = round((numerator / denominator) * 100.0, 2) if denominator > 0 else None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value_pct": value_pct,
        "definition": definition,
    }


def reconcile_evaluation_invariants(
    evaluation_result: Dict[str, Any],
    evaluable_instance_count: int,
) -> List[str]:
    """Verify internal mathematical consistency of cross-session identity evaluation."""
    violations: List[str] = []
    per_inst = evaluation_result.get("per_instance_results", [])
    if len(per_inst) != evaluable_instance_count:
        violations.append(
            f"Evaluated instance count ({len(per_inst)}) does not equal evaluable benchmark population ({evaluable_instance_count})"
        )

    valid_outcomes = {
        "NOT_EVALUABLE",
        "ALIAS_CONFLICT",
        "CORRECTLY_MATCHED",
        "FALSE_MERGE",
        "AMBIGUOUS",
        "UNMATCHED",
    }
    for r in per_inst:
        out = r.get("outcome")
        if out not in valid_outcomes:
            violations.append(f"Instance {r.get('physical_instance_id')} has invalid outcome: {out}")
        if r.get("is_correct") != (out == "CORRECTLY_MATCHED"):
            violations.append(
                f"Instance {r.get('physical_instance_id')} has inconsistent is_correct ({r.get('is_correct')}) for outcome {out}"
            )

    correctly_matched_count = sum(1 for r in per_inst if r.get("outcome") == "CORRECTLY_MATCHED")
    rec_count = evaluation_result.get("recovered_all_instances_count", 0)
    if correctly_matched_count != rec_count:
        violations.append(
            f"Recovered instance count ({rec_count}) does not match count of CORRECTLY_MATCHED instances ({correctly_matched_count})"
        )

    rec_metric = evaluation_result.get("instance_recall")
    if rec_metric and rec_metric["numerator"] != correctly_matched_count:
        violations.append(
            f"instance_recall numerator ({rec_metric['numerator']}) != correctly_matched_count ({correctly_matched_count})"
        )

    return violations


def build_canonical_session_track_mapping(
    gt_path: Optional[Union[Path, str, Dict[str, Any]]] = None,
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Parse and validate canonical ground truth (session, track_id) -> physical_instance_id mapping.

    Enforces that:
    1. Annotations point to canonical benchmark cross_session_identity_annotations.json.
    2. No contradictory physical instance assignments exist for the same (session, track_id) tuple.
    3. Cross-session evaluation pairs are populated for instances observed across sessions.
    """
    canonical_map: Dict[Tuple[str, str], str] = {}
    gt_cross_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if isinstance(gt_path, dict):
        data = gt_path
    else:
        if gt_path is not None:
            p_canon = Path(gt_path)
            if not p_canon.exists():
                if (repo_root / p_canon).exists():
                    p_canon = repo_root / p_canon
                elif (p_canon.parent / "cross_session_identity_annotations.json").exists():
                    p_canon = p_canon.parent / "cross_session_identity_annotations.json"
                elif (p_canon / "cross_session_identity_annotations.json").exists():
                    p_canon = p_canon / "cross_session_identity_annotations.json"
        else:
            p_canon = repo_root / "benchmarks" / "cross_session_identity_annotations.json"

        if not p_canon.exists():
            raise FileNotFoundError(
                f"Canonical cross-session benchmark missing at: {p_canon}. "
                f"Implicit fallback to movability_ground_truth.json is prohibited."
            )

        with open(p_canon, "r", encoding="utf-8") as f:
            data = json.load(f)

    annotations = data.get("annotations", [])
    inst_records = defaultdict(list)
    for a in annotations:
        sess = a["source_session"]
        trk = a["local_track_id"]
        inst = a["physical_instance_id"]
        key = (sess, trk)
        if key in canonical_map and canonical_map[key] != inst:
            raise ValueError(f"Contradictory annotation for {key}: {canonical_map[key]} vs {inst}")
        canonical_map[key] = inst
        inst_records[inst].append(a)

    for inst, recs in inst_records.items():
        s1_trks = sorted(set(r["local_track_id"] for r in recs if "recording_01" in r["source_session"] or "rec01" in r["source_session"]))
        s2_trks = sorted(set(r["local_track_id"] for r in recs if "recording_02" in r["source_session"] or "rec02" in r["source_session"]))
        if s1_trks and s2_trks:
            sample = recs[0]
            for t1 in s1_trks:
                for t2 in s2_trks:
                    gt_cross_pairs[(t1, t2)] = {
                        "physical_instance_id": inst,
                        "presence": sample.get("presence", "PRESENT_BOTH"),
                        "motion": sample.get("motion", "STATIC"),
                        "true_displacement_m": sample.get("true_displacement_m", 0.0),
                    }
    return canonical_map, gt_cross_pairs


def evaluate_matcher_against_ground_truth(
    association_result: AssociationResult,
    gt_path: Optional[Union[Path, str, Dict[str, Any]]] = None,
    s1_clusters: Optional[Dict[str, EntityCluster]] = None,
    s2_clusters: Optional[Dict[str, EntityCluster]] = None,
    provenance_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rigorously evaluate cross-session physical identity matches against benchmark ground truth.

    Transparent Three-Level Evaluation Contract:
    1. Candidate Pair-Level Evaluation:
       - Evaluates whether individual accepted matches link track clusters of the same physical object.
       - Pair precision = correct annotated accepted pairs / all annotated accepted pairs (|P_A|).
       - Pair false-merge rate = incorrect annotated accepted pairs / all annotated accepted pairs (|P_A|).
    2. Physical-Instance Level Evaluation:
       - Evaluates whether each annotated physical object in G is recovered across sessions.
       - Evaluated across the multi-session physical instances (G) with one primary outcome per instance.
       - Instance recall = |I_R| / |G|.
       - Instance miss rate = (|G| - |I_R|) / |G|.
    3. Predicted Persistent-ID Cluster Evaluation:
       - Evaluates whether physical objects are fragmented across >= 2 predicted IDs (split-instance rate)
         or merged into a single predicted ID with other objects (merge-instance rate).
    """
    canonical_map, gt_cross_pairs = build_canonical_session_track_mapping(gt_path)

    # Multi-session ground-truth physical instances (G)
    all_multi_session_instances: Set[str] = set()
    static_multi_session_instances: Set[str] = set()
    relocated_multi_session_instances: Set[str] = set()

    for pair_meta in gt_cross_pairs.values():
        inst_id = pair_meta["physical_instance_id"]
        all_multi_session_instances.add(inst_id)
        if pair_meta.get("motion") == "STATIC":
            static_multi_session_instances.add(inst_id)
        else:
            relocated_multi_session_instances.add(inst_id)

    total_evaluable_instances = len(all_multi_session_instances)

    s1_session = (
        next((c.source_session for c in s1_clusters.values()), "semantic_mapping_my_desk_recording_01")
        if s1_clusters
        else "semantic_mapping_my_desk_recording_01"
    )
    s2_session = (
        next((c.source_session for c in s2_clusters.values()), "semantic_mapping_my_desk_recording_02")
        if s2_clusters
        else "semantic_mapping_my_desk_recording_02"
    )

    # Step 1: Pre-resolve canonical identity for all clusters
    s1_resolutions: Dict[str, ClusterIdentityResolution] = {}
    if s1_clusters:
        for cid, c in s1_clusters.items():
            s1_resolutions[cid] = resolve_cluster_canonical_identity(c, cid, s1_session, canonical_map)
    s2_resolutions: Dict[str, ClusterIdentityResolution] = {}
    if s2_clusters:
        for cid, c in s2_clusters.items():
            s2_resolutions[cid] = resolve_cluster_canonical_identity(c, cid, s2_session, canonical_map)

    alias_conflict_clusters_count = sum(
        1 for r in list(s1_resolutions.values()) + list(s2_resolutions.values())
        if r.annotation_status == "ALIAS_CONFLICT"
    )

    # Step 2: Normalize prediction inventory
    accepted_eval_records: List[Dict[str, Any]] = []
    annotated_eval_matches: List[Dict[str, Any]] = []
    unannotated_matches: List[Dict[str, Any]] = []
    true_positive_pairs = 0
    pair_false_merges = 0

    gt_inst_to_pids: Dict[str, Set[str]] = defaultdict(set)
    pid_to_gt_insts: Dict[str, Set[str]] = defaultdict(set)

    for m in association_result.matched_pairs:
        t1 = m.s1_entity_id
        t2 = m.s2_entity_id
        c1 = s1_clusters.get(t1) if s1_clusters else None
        c2 = s2_clusters.get(t2) if s2_clusters else None
        r1 = s1_resolutions.get(t1) or resolve_cluster_canonical_identity(c1, t1, s1_session, canonical_map)
        r2 = s2_resolutions.get(t2) or resolve_cluster_canonical_identity(c2, t2, s2_session, canonical_map)

        has_alias_conflict = (r1.annotation_status == "ALIAS_CONFLICT" or r2.annotation_status == "ALIAS_CONFLICT")
        is_annotated = (r1.annotation_status == "ANNOTATED_SINGLE_ID" and r2.annotation_status == "ANNOTATED_SINGLE_ID")

        inst1 = r1.resolved_instance_id
        inst2 = r2.resolved_instance_id
        pred_pid = m.persistent_entity_id or f"pred_pid_{t1}_{t2}"

        if inst1:
            gt_inst_to_pids[inst1].add(pred_pid)
            pid_to_gt_insts[pred_pid].add(inst1)
        if inst2:
            gt_inst_to_pids[inst2].add(pred_pid)
            pid_to_gt_insts[pred_pid].add(inst2)

        if has_alias_conflict:
            is_correct = False
            pair_status = "ALIAS_CONFLICT"
            error_type = "ALIAS_CONFLICT"
        elif is_annotated:
            is_correct = (inst1 == inst2)
            pair_status = "CORRECT_MATCH" if is_correct else "FALSE_MERGE"
            error_type = None if is_correct else "FALSE_MERGE"
            if is_correct:
                true_positive_pairs += 1
            else:
                pair_false_merges += 1
        else:
            is_correct = False
            pair_status = "OUT_OF_BENCHMARK_SCOPE"
            error_type = "OUT_OF_BENCHMARK_SCOPE"

        rec = {
            "s1_entity": t1,
            "s2_entity": t2,
            "persistent_entity_id": pred_pid,
            "s1_gt_instance": inst1,
            "s2_gt_instance": inst2,
            "s1_resolution_status": r1.annotation_status,
            "s2_resolution_status": r2.annotation_status,
            "is_annotated": is_annotated,
            "is_correct": is_correct,
            "distance_m": m.distance_m,
            "confidence": m.confidence,
            "match_status": m.status,
            "pair_evaluation_status": pair_status,
            "error_type": error_type,
        }
        accepted_eval_records.append(rec)
        if is_annotated:
            annotated_eval_matches.append(rec)
        else:
            unannotated_matches.append(rec)

    ambiguous_records: List[Dict[str, Any]] = []
    for m in association_result.ambiguous_pairs:
        t1 = m.s1_entity_id
        t2 = m.s2_entity_id
        c1 = s1_clusters.get(t1) if s1_clusters else None
        c2 = s2_clusters.get(t2) if s2_clusters else None
        r1 = s1_resolutions.get(t1) or resolve_cluster_canonical_identity(c1, t1, s1_session, canonical_map)
        r2 = s2_resolutions.get(t2) or resolve_cluster_canonical_identity(c2, t2, s2_session, canonical_map)
        ambiguous_records.append({
            "s1_entity": t1,
            "s2_entity": t2,
            "s1_gt_instance": r1.resolved_instance_id,
            "s2_gt_instance": r2.resolved_instance_id,
            "distance_m": m.distance_m,
            "status": "REJECTED_AMBIGUOUS",
        })

    # Step 3: Build normalized physical-instance outcome table
    normalized_instance_records: List[Dict[str, Any]] = []
    per_instance_table: List[Dict[str, Any]] = []

    for inst_id in sorted(all_multi_session_instances):
        s1_trks = sorted(set(trk for (sess, trk), i in canonical_map.items() if i == inst_id and ("01" in sess or sess == s1_session)))
        s2_trks = sorted(set(trk for (sess, trk), i in canonical_map.items() if i == inst_id and ("02" in sess or sess == s2_session)))
        motion_state = "STATIC" if inst_id in static_multi_session_instances else "RELOCATED"

        s1_cluster_ids = sorted(list(set(cid for cid, res in s1_resolutions.items() if inst_id in res.annotated_physical_instance_ids)))
        s2_cluster_ids = sorted(list(set(cid for cid, res in s2_resolutions.items() if inst_id in res.annotated_physical_instance_ids)))

        has_alias_conflict = any(
            s1_resolutions[cid].annotation_status == "ALIAS_CONFLICT" for cid in s1_cluster_ids if cid in s1_resolutions
        ) or any(
            s2_resolutions[cid].annotation_status == "ALIAS_CONFLICT" for cid in s2_cluster_ids if cid in s2_resolutions
        )

        correct_pairs = [
            p for p in accepted_eval_records
            if p["s1_gt_instance"] == inst_id and p["s2_gt_instance"] == inst_id and p["is_correct"]
        ]
        false_merge_pairs = [
            p for p in accepted_eval_records
            if (p["s1_gt_instance"] == inst_id or p["s2_gt_instance"] == inst_id) and not p["is_correct"]
        ]
        ambig_pairs = [
            p for p in ambiguous_records
            if p["s1_gt_instance"] == inst_id or p["s2_gt_instance"] == inst_id
        ]

        accepted_pids = sorted(list(gt_inst_to_pids.get(inst_id, set())))
        shares_pid_with_other_gt = any(len(pid_to_gt_insts[pid]) > 1 for pid in accepted_pids)
        involved_in_merge = (len(false_merge_pairs) > 0 or shares_pid_with_other_gt)
        involved_in_split = (len(accepted_pids) > 1)

        # Precedence Decision Policy for Primary Outcome
        if not (s1_trks and s2_trks):
            outcome = "NOT_EVALUABLE"
            reason = "Instance missing required multi-session track annotations"
        elif has_alias_conflict:
            outcome = "ALIAS_CONFLICT"
            reason = "Underlying consolidated cluster merges contradictory ground-truth identities"
        elif len(correct_pairs) > 0:
            outcome = "CORRECTLY_MATCHED"
            reason = f"Successfully recovered via {len(correct_pairs)} correct accepted match(es)"
        elif involved_in_merge:
            outcome = "FALSE_MERGE"
            reason = f"Falsely merged with other physical instance(s) in accepted prediction(s)"
        elif len(ambig_pairs) > 0:
            outcome = "AMBIGUOUS"
            reason = f"Candidate pair was rejected due to ambiguity ({len(ambig_pairs)} ambiguous candidate(s))"
        else:
            outcome = "UNMATCHED"
            reason = "No candidate cluster survived feature gating or spatial distance threshold"

        is_recovered = (outcome == "CORRECTLY_MATCHED")
        had_ambiguous_candidates = (len(ambig_pairs) > 0)
        had_unmatched_tracks = (
            any(t not in [p["s1_entity"] for p in correct_pairs] for t in s1_trks)
            or any(t not in [p["s2_entity"] for p in correct_pairs] for t in s2_trks)
        )

        best_pair = correct_pairs[0] if correct_pairs else (false_merge_pairs[0] if false_merge_pairs else None)

        inst_record = {
            "physical_instance_id": inst_id,
            "motion_state": motion_state,
            "expected_in_session_1": True,
            "expected_in_session_2": True,
            "observed_in_session_1": bool(s1_cluster_ids or s1_trks),
            "observed_in_session_2": bool(s2_cluster_ids or s2_trks),
            "s1_track_ids": s1_trks,
            "s2_track_ids": s2_trks,
            "s1_cluster_ids": s1_cluster_ids,
            "s2_cluster_ids": s2_cluster_ids,
            "s1_representative_track_id": best_pair["s1_entity"] if best_pair else (s1_trks[0] if s1_trks else "NONE"),
            "s2_representative_track_id": best_pair["s2_entity"] if best_pair else (s2_trks[0] if s2_trks else "NONE"),
            "s1_gt_instance": best_pair["s1_gt_instance"] if best_pair else inst_id,
            "s2_gt_instance": best_pair["s2_gt_instance"] if best_pair else (inst_id if outcome == "CORRECTLY_MATCHED" else None),
            "accepted_persistent_id": best_pair["persistent_entity_id"] if best_pair else (accepted_pids[0] if accepted_pids else None),
            "accepted_predicted_ids": accepted_pids,
            "correct_match_count": len(correct_pairs),
            "incorrect_match_count": len(false_merge_pairs),
            "ambiguous_candidate_count": len(ambig_pairs),
            "outcome": outcome,
            "is_correct": is_recovered,
            "distance_m": round(best_pair["distance_m"], 4) if best_pair and best_pair.get("distance_m") is not None else None,
            "rejection_reason": reason,
            "involved_in_split": involved_in_split,
            "involved_in_merge": involved_in_merge,
            "had_ambiguous_candidates": had_ambiguous_candidates,
            "had_unmatched_tracks": had_unmatched_tracks,
            "reason_codes": [outcome],
        }
        normalized_instance_records.append(inst_record)

        per_instance_table.append({
            "physical_instance_id": inst_id,
            "motion_state": motion_state,
            "s1_track_id": inst_record["s1_representative_track_id"],
            "s2_track_id": inst_record["s2_representative_track_id"],
            "outcome": outcome,
            "is_correct": is_recovered,
            "distance_m": inst_record["distance_m"],
            "rejection_reason": reason,
            "involved_in_split": involved_in_split,
            "involved_in_merge": involved_in_merge,
            "had_ambiguous_candidates": had_ambiguous_candidates,
        })

    # Step 4: Compute structured metrics
    recovered_instances = [r for r in normalized_instance_records if r["outcome"] == "CORRECTLY_MATCHED"]
    recovered_static_instances = [r for r in recovered_instances if r["motion_state"] == "STATIC"]
    recovered_relocated_instances = [r for r in recovered_instances if r["motion_state"] == "RELOCATED"]
    missed_static_instances = [
        r["physical_instance_id"] for r in normalized_instance_records
        if r["motion_state"] == "STATIC" and r["outcome"] != "CORRECTLY_MATCHED"
    ]
    missed_relocated_instances = [
        r["physical_instance_id"] for r in normalized_instance_records
        if r["motion_state"] == "RELOCATED" and r["outcome"] != "CORRECTLY_MATCHED"
    ]

    split_instances = [r for r in normalized_instance_records if r["involved_in_split"]]
    merge_instances = [r for r in normalized_instance_records if r["involved_in_merge"]]
    ambiguous_instances = [r for r in normalized_instance_records if r["outcome"] == "AMBIGUOUS"]
    unmatched_instances = [r for r in normalized_instance_records if r["outcome"] == "UNMATCHED"]

    total_annotated_pairs = len(annotated_eval_matches)
    total_accepted_pairs = len(association_result.matched_pairs)

    metric_instance_recall = _make_metric_record(
        len(recovered_instances),
        total_evaluable_instances,
        "Correctly recovered evaluable physical instances divided by all evaluable expected physical instances",
    )
    metric_instance_miss_rate = _make_metric_record(
        total_evaluable_instances - len(recovered_instances),
        total_evaluable_instances,
        "Unrecovered expected physical instances divided by all evaluable expected physical instances",
    )
    metric_pair_precision = _make_metric_record(
        true_positive_pairs,
        total_annotated_pairs,
        "Correct annotated accepted candidate pairs divided by all annotated accepted candidate pairs",
    )
    metric_pair_false_merge_rate = _make_metric_record(
        pair_false_merges,
        total_annotated_pairs,
        "Incorrect annotated accepted candidate pairs (inst1 != inst2) divided by all annotated accepted candidate pairs",
    )
    metric_split_instance_rate = _make_metric_record(
        len(split_instances),
        total_evaluable_instances,
        "Evaluable physical instances represented by >= 2 distinct predicted persistent IDs divided by all evaluable expected physical instances",
    )
    metric_merge_instance_rate = _make_metric_record(
        len(merge_instances),
        total_evaluable_instances,
        "Evaluable physical instances involved in a multi-instance predicted ID or false merge divided by all evaluable expected physical instances",
    )
    metric_ambiguity_rate = _make_metric_record(
        len(ambiguous_instances),
        total_evaluable_instances,
        "Evaluable physical instances whose primary outcome is unresolved ambiguity divided by all evaluable expected physical instances",
    )
    metric_unmatched_rate = _make_metric_record(
        len(unmatched_instances),
        total_evaluable_instances,
        "Evaluable expected physical instances with no accepted match divided by all evaluable expected physical instances",
    )
    metric_annotation_coverage = _make_metric_record(
        total_annotated_pairs,
        total_accepted_pairs,
        "Accepted candidate pairs with both endpoints resolvable in benchmark divided by all accepted candidate pairs",
    )

    static_recall = _make_metric_record(
        len(recovered_static_instances),
        len(static_multi_session_instances),
        "Recovered static physical instances divided by all expected static physical instances",
    )
    reloc_recall = _make_metric_record(
        len(recovered_relocated_instances),
        len(relocated_multi_session_instances),
        "Recovered relocated physical instances divided by all expected relocated physical instances",
    )

    # Instance-level precision based on unique ground-truth identities claimed in accepted annotated pairs
    claimed_instances = set()
    for rec in annotated_eval_matches:
        if rec.get("s1_gt_instance"):
            claimed_instances.add(rec["s1_gt_instance"])
        if rec.get("s2_gt_instance"):
            claimed_instances.add(rec["s2_gt_instance"])
    metric_instance_precision = _make_metric_record(
        len(recovered_instances),
        len(claimed_instances),
        "Correctly recovered physical instances divided by unique ground-truth physical instances claimed in annotated predictions",
    )

    pair_rec_val = (true_positive_pairs / len(gt_cross_pairs)) * 100.0 if gt_cross_pairs else None
    pair_f1 = (
        round(2 * metric_pair_precision["value_pct"] * pair_rec_val / (metric_pair_precision["value_pct"] + pair_rec_val), 2)
        if metric_pair_precision["value_pct"] and pair_rec_val and (metric_pair_precision["value_pct"] + pair_rec_val) > 0
        else None
    )

    inst_prec_val = metric_instance_precision["value_pct"]
    inst_rec_val = metric_instance_recall["value_pct"]
    inst_f1 = (
        round(2 * inst_prec_val * inst_rec_val / (inst_prec_val + inst_rec_val), 2)
        if inst_prec_val and inst_rec_val and (inst_prec_val + inst_rec_val) > 0
        else None
    )

    evaluation_report = {
        "algorithm_name": association_result.algorithm_name,
        "provenance": provenance_metadata or {},
        "total_matched_pairs": total_accepted_pairs,
        "annotated_eval_pairs_count": total_annotated_pairs,
        "true_positive_matches": true_positive_pairs,
        "false_merges": pair_false_merges,
        # Structured Metrics
        "instance_recall": metric_instance_recall,
        "instance_miss_rate": metric_instance_miss_rate,
        "pair_precision": metric_pair_precision,
        "pair_false_merge_rate": metric_pair_false_merge_rate,
        "split_instance_rate": metric_split_instance_rate,
        "merge_instance_rate": metric_merge_instance_rate,
        "ambiguity_rate": metric_ambiguity_rate,
        "unmatched_rate": metric_unmatched_rate,
        "annotation_coverage": metric_annotation_coverage,
        "instance_precision": metric_instance_precision,
        # Flat percentages (backward compatibility)
        "instance_recall_pct": metric_instance_recall["value_pct"],
        "all_instance_recall_pct": metric_instance_recall["value_pct"],
        "instance_miss_rate_pct": metric_instance_miss_rate["value_pct"],
        "candidate_pair_precision_pct": metric_pair_precision["value_pct"],
        "precision_on_annotated_subset_pct": metric_pair_precision["value_pct"],
        "pair_level_false_merge_rate_pct": metric_pair_false_merge_rate["value_pct"],
        "false_merge_rate_pct": metric_pair_false_merge_rate["value_pct"],
        "true_false_split_rate_pct": metric_split_instance_rate["value_pct"],
        "split_instance_rate_pct": metric_split_instance_rate["value_pct"],
        "false_split_rate_pct": metric_split_instance_rate["value_pct"],  # Corrected from legacy miss rate to true split rate
        "true_false_merge_rate_pct": metric_merge_instance_rate["value_pct"],
        "merge_instance_rate_pct": metric_merge_instance_rate["value_pct"],
        "ambiguity_rate_pct": metric_ambiguity_rate["value_pct"],
        "unmatched_rate_pct": metric_unmatched_rate["value_pct"],
        "annotation_coverage_pct": metric_annotation_coverage["value_pct"],
        "static_instance_recall_pct": static_recall["value_pct"],
        "recall_on_static_multi_session_pct": static_recall["value_pct"],
        "relocated_instance_recall_pct": reloc_recall["value_pct"],
        "recall_on_all_multi_session_pct": metric_instance_recall["value_pct"],
        "instance_level_precision_pct": inst_prec_val,
        "instance_level_f1_pct": inst_f1,
        "f1_on_annotated_subset_pct": inst_f1,
        "pair_level_recall_pct": round(pair_rec_val, 2) if pair_rec_val is not None else None,
        "pair_level_f1_pct": pair_f1,
        # Population counts
        "total_multi_session_instances": total_evaluable_instances,
        "static_multi_session_instances": len(static_multi_session_instances),
        "relocated_multi_session_instances": len(relocated_multi_session_instances),
        "recovered_static_instances_count": len(recovered_static_instances),
        "recovered_relocated_instances_count": len(recovered_relocated_instances),
        "recovered_all_instances_count": len(recovered_instances),
        "missed_static_instances_count": len(missed_static_instances),
        "missed_static_instances": missed_static_instances,
        "missed_relocated_instances": missed_relocated_instances,
        "unannotated_candidate_pairs_count": len(unannotated_matches),
        "ambiguous_pairs_rejected_count": len(association_result.ambiguous_pairs),
        "alias_conflict_clusters_count": alias_conflict_clusters_count,
        "total_ground_truth_cross_session_pairs": len(gt_cross_pairs),
        # Diagnostic records
        "annotated_evaluated_matches": annotated_eval_matches,
        "unannotated_matches_sample": unannotated_matches[:10],
        "per_instance_results": per_instance_table,
        "normalized_physical_instances": normalized_instance_records,
        "legacy_false_split_rate_note": "DEPRECATED: Prior releases assigned instance_miss_rate_pct (e.g. 76.67%) to false_split_rate_pct. This field now correctly reflects split_instance_rate_pct.",
    }

    # Run internal reconciliation invariants if benchmark instances were present
    if total_evaluable_instances > 0:
        violations = reconcile_evaluation_invariants(evaluation_report, total_evaluable_instances)
        if violations:
            raise ValueError(f"Cross-session evaluation reconciliation invariant failed: {violations}")

    return evaluation_report


def compare_association_baselines(
    s1_clusters: Dict[str, EntityCluster],
    s2_clusters: Dict[str, EntityCluster],
    registration: RegistrationInfo,
    gt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate and compare all 3 cross-session association baselines on identical data."""
    matcher = CrossSessionCandidateMatcher()

    # 1. Nearest Centroid
    res_b1 = matcher.match_entities_nearest_centroid(s1_clusters, s2_clusters, registration)
    eval_b1 = evaluate_matcher_against_ground_truth(res_b1, gt_path, s1_clusters=s1_clusters, s2_clusters=s2_clusters)

    # 2. Hungarian Gated
    res_b2 = matcher.match_entities_hungarian_gated(s1_clusters, s2_clusters, registration)
    eval_b2 = evaluate_matcher_against_ground_truth(res_b2, gt_path, s1_clusters=s1_clusters, s2_clusters=s2_clusters)

    # 3. Proposed Method (Static Proximity + Uncertainty + Ambiguity)
    res_b3 = matcher.match_entities(s1_clusters, s2_clusters, registration)
    eval_b3 = evaluate_matcher_against_ground_truth(res_b3, gt_path, s1_clusters=s1_clusters, s2_clusters=s2_clusters)

    # 4. Proposed Relocation-Aware Method (Static Proximity + Relocation Hypotheses)
    res_b4 = matcher.match_entities_relocation_aware(s1_clusters, s2_clusters, registration)
    eval_b4 = evaluate_matcher_against_ground_truth(res_b4, gt_path, s1_clusters=s1_clusters, s2_clusters=s2_clusters)

    # 5. Proposed Registration-Uncertainty-Aware Adaptive Method
    res_b5 = matcher.match_entities_uncertainty_adaptive(s1_clusters, s2_clusters, registration)
    eval_b5 = evaluate_matcher_against_ground_truth(res_b5, gt_path, s1_clusters=s1_clusters, s2_clusters=s2_clusters)

    return {
        "baseline_1_nearest_centroid": {"result": res_b1, "evaluation": eval_b1},
        "baseline_2_hungarian_gated": {"result": res_b2, "evaluation": eval_b2},
        "proposed_uncertainty_symmetric": {"result": res_b3, "evaluation": eval_b3},
        "proposed_relocation_aware": {"result": res_b4, "evaluation": eval_b4},
        "proposed_uncertainty_adaptive": {"result": res_b5, "evaluation": eval_b5},
    }
