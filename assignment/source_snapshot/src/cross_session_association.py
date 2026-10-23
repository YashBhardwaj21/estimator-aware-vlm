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

from vlm.src.evidence_adapter import RegistrationInfo

repo_root = Path(__file__).resolve().parent.parent.parent


@dataclass
class EntityCluster:
    """Aggregated spatial and semantic cluster for an entity within a session."""
    entity_id: str
    semantic_class: str
    source_session: str
    centroid_floor: np.ndarray  # 2D [x, y] in session local floor frame
    centroid_std: float         # 1D dispersion metric in meters
    spatial_extent_m: float     # Bounding dimension (diameter/width)
    observations_count: int
    local_track_ids: List[str]
    observation_ids: List[str]
    has_valid_pose: bool = True
    depth_valid: bool = True
    centroid_3d: Optional[np.ndarray] = None  # Measured 3D [x, y, z] centroid from observations
    centroid_aligned: Optional[np.ndarray] = None
    position_uncertainty_m: float = 0.05


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
    ) -> Dict[str, EntityCluster]:
        """Aggregate raw frame observations into robust entity clusters preserving measured 3D geometry."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for o in observations:
            sess = o.get("source_session", "")
            if session_keyword not in sess:
                continue
            eid = o.get("global_entity_id", "")
            if not eid:
                continue
            cf = o.get("centroid_floor")
            if cf is not None and len(cf) >= 2 and all(x is not None for x in cf[:2]):
                grouped.setdefault(eid, []).append(o)

        clusters: Dict[str, EntityCluster] = {}
        for eid, obs_list in grouped.items():
            if len(obs_list) < min_obs:
                continue

            pts = np.array([o["centroid_floor"][:2] for o in obs_list], dtype=np.float64)
            c = np.median(pts, axis=0)

            # Preserve measured 3D height from real observations
            pts_3d = np.array([o["centroid_floor"][:3] for o in obs_list if len(o.get("centroid_floor", [])) >= 3], dtype=np.float64)
            c_3d = np.median(pts_3d, axis=0) if len(pts_3d) > 0 else np.array([c[0], c[1], 0.0], dtype=np.float64)

            dispersion = float(np.mean(np.std(pts, axis=0))) if len(pts) > 1 else 0.05
            sem_class = obs_list[0].get("class_name") or obs_list[0].get("semantic_class", "object")

            extents = []
            for o in obs_list:
                bmin = o.get("bbox_min_floor")
                bmax = o.get("bbox_max_floor")
                if bmin and bmax and len(bmin) >= 2 and len(bmax) >= 2:
                    extents.append(float(np.linalg.norm(np.array(bmax[:2]) - np.array(bmin[:2]))))
            avg_extent = float(np.mean(extents)) if extents else 0.50

            track_ids = list(set(o.get("local_track_id") for o in obs_list if o.get("local_track_id")))
            obs_ids = [o.get("obs_id") for o in obs_list if o.get("obs_id")]
            has_pose = all(o.get("has_valid_pose", False) for o in obs_list)
            depth_valid = any(o.get("depth_valid", True) for o in obs_list)

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
                centroid_3d=c_3d,
                position_uncertainty_m=sem_pos,
            )

        return clusters

    def _align_session2(
        self,
        s2_clusters: Dict[str, EntityCluster],
        registration: RegistrationInfo,
    ) -> Dict[str, EntityCluster]:
        """Align session 2 clusters into session 1 reference frame using empirical residual dispersion."""
        reg_disp = getattr(registration, "inlier_rms_residual_m", 0.1198)
        if reg_disp <= 0:
            reg_disp = getattr(registration, "anchor_rms_residual_m", 0.3054)

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
                matched_pairs.append(PhysicalInstanceMatch(
                    persistent_entity_id=f"phys_{k1}",
                    s1_entity_id=k1,
                    s2_entity_id=best_k2,
                    semantic_class=c1.semantic_class,
                    distance_m=round(best_dist, 4),
                    position_uncertainty_m=round(math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2), 4),
                    extent_diff_m=round(abs(c1.spatial_extent_m - c2.spatial_extent_m), 4),
                    match_score=round(best_dist, 4),
                    confidence=round(max(0.20, 1.0 - best_dist / self.max_static_distance_m), 4),
                    status="MATCHED",
                    s1_obs_count=c1.observations_count,
                    s2_obs_count=c2.observations_count,
                    diagnostics=["GREEDY_NEAREST_CENTROID"],
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
        """Relocation-Aware Association Method: Multi-stage Hungarian assignment handling static proximity and relocated objects."""
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
                max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                rel_ext = ext_diff / max_ext
                if rel_ext > self.max_extent_relative_diff:
                    continue

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
            ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
            confidence = max(0.20, min(0.99, 1.0 - (dist / self.max_static_distance_m) * 0.5 - (cost / 10.0) * 0.3))

            match_obj = PhysicalInstanceMatch(
                persistent_entity_id=f"phys_{k1}",
                s1_entity_id=k1,
                s2_entity_id=k2,
                semantic_class=c1.semantic_class,
                distance_m=round(dist, 4),
                position_uncertainty_m=round(combined_unc, 4),
                extent_diff_m=round(ext_diff, 4),
                match_score=round(float(cost), 4),
                confidence=round(confidence, 4),
                status="AMBIGUOUS" if is_ambiguous else "MATCHED",
                s1_obs_count=c1.observations_count,
                s2_obs_count=c2.observations_count,
                match_type="MATCHED_STATIC",
                diagnostics=ambiguity_reasons,
            )

            if is_ambiguous:
                ambiguous_pairs.append(match_obj)
            else:
                matched_pairs.append(match_obj)
                matched_s1_idx.add(r)
                matched_s2_idx.add(c)

        # Stage 2: Relocation candidate matching for unassigned entities beyond the static gate
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
                        max_ext = max(c1.spatial_extent_m, c2.spatial_extent_m, 0.10)
                        ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                        rel_ext = ext_diff / max_ext
                        if rel_ext > self.max_relocation_extent_relative_diff:
                            continue

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
                    ext_diff = abs(c1.spatial_extent_m - c2.spatial_extent_m)
                    confidence = max(0.20, min(0.95, 1.0 - (rdist / self.max_relocation_distance_m) * 0.4 - (rcost / 10.0) * 0.3))

                    reloc_match = PhysicalInstanceMatch(
                        persistent_entity_id=f"phys_{k1}",
                        s1_entity_id=k1,
                        s2_entity_id=k2,
                        semantic_class=c1.semantic_class,
                        distance_m=round(rdist, 4),
                        position_uncertainty_m=round(combined_unc, 4),
                        extent_diff_m=round(ext_diff, 4),
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
        p_canon = repo_root / "vlm" / "benchmarks" / "cross_session_identity_annotations.json"

    canonical_map: Dict[Tuple[str, str], str] = {}
    gt_cross_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if not p_canon.exists():
        raise FileNotFoundError(
            f"Canonical cross-session benchmark missing at: {p_canon}. "
            f"Implicit fallback to movability_ground_truth.json is prohibited."
        )

    canonical_map: Dict[Tuple[str, str], str] = {}
    gt_cross_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

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
        s1_trks = sorted(set(r["local_track_id"] for r in recs if "recording_01" in r["source_session"]))
        s2_trks = sorted(set(r["local_track_id"] for r in recs if "recording_02" in r["source_session"]))
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
    gt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Rigorously evaluate cross-session physical identity matches against benchmark ground truth.

    Transparent Denominators:
    - Annotated Evaluation Subset: Matches where both S1 and S2 entities are annotated in GT.
    - True Positives (TP): Annotated pairs where inst1 == inst2 (correctly resolving physical identity).
    - False Merges (FM): Annotated pairs where inst1 != inst2 (merging distinct physical objects).
    - Unannotated Candidate Pairs: Matches outside GT annotation scope (reported separately, NOT counted as false positives).
    - Physical-Instance Level Recall: Evaluated against the 30 multi-session ground-truth instances
      (26 static, 4 relocated) to eliminate track Cartesian product distortion.
    """
    canonical_map, gt_cross_pairs = build_canonical_session_track_mapping(gt_path)

    # Multi-session ground-truth physical instances
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

    annotated_eval_matches = []
    unannotated_matches = []
    true_positives = 0
    false_merges = 0
    recovered_instances: Set[str] = set()
    recovered_static_instances: Set[str] = set()
    recovered_relocated_instances: Set[str] = set()

    for m in association_result.matched_pairs:
        t1 = m.s1_entity_id
        t2 = m.s2_entity_id
        inst1 = canonical_map.get(("semantic_mapping_my_desk_recording_01", t1))
        inst2 = canonical_map.get(("semantic_mapping_my_desk_recording_02", t2))

        rec = {
            "s1_entity": t1,
            "s2_entity": t2,
            "s1_gt_instance": inst1,
            "s2_gt_instance": inst2,
            "distance_m": m.distance_m,
            "confidence": m.confidence,
            "match_status": m.status,
        }

        if inst1 is not None and inst2 is not None:
            is_correct = (inst1 == inst2)
            rec["is_correct"] = is_correct
            annotated_eval_matches.append(rec)
            if is_correct:
                true_positives += 1
                recovered_instances.add(inst1)
                if inst1 in static_multi_session_instances:
                    recovered_static_instances.add(inst1)
                else:
                    recovered_relocated_instances.add(inst1)
            else:
                false_merges += 1
        else:
            rec["is_annotated"] = False
            rec["annotation_status"] = "OUT_OF_BENCHMARK_ANNOTATION_SCOPE"
            unannotated_matches.append(rec)

    total_annotated = len(annotated_eval_matches)
    missed_static_instances = sorted(static_multi_session_instances - recovered_static_instances)
    missed_relocated_instances = sorted(relocated_multi_session_instances - recovered_relocated_instances)

    n_static = max(1, len(static_multi_session_instances))
    n_reloc = max(1, len(relocated_multi_session_instances))
    n_all = max(1, len(all_multi_session_instances))
    n_gt_pairs = max(1, len(gt_cross_pairs))

    # 1. Annotated Candidate Pair-level Metrics (Denominator: Annotated Evaluated Pairs)
    precision_on_annotated = (true_positives / max(1, total_annotated)) * 100.0 if total_annotated > 0 else 0.0
    false_merge_rate = (false_merges / max(1, total_annotated)) * 100.0 if total_annotated > 0 else 0.0
    pair_recall = (true_positives / n_gt_pairs) * 100.0
    pair_f1 = (2 * precision_on_annotated * pair_recall / (precision_on_annotated + pair_recall)) if (precision_on_annotated + pair_recall) > 0 else 0.0

    # 2. Physical-Instance Level Metrics (Denominator: Ground Truth Physical Instances)
    recall_on_static = (len(recovered_static_instances) / n_static) * 100.0
    recall_on_relocated = (len(recovered_relocated_instances) / n_reloc) * 100.0
    recall_on_all = (len(recovered_instances) / n_all) * 100.0
    false_split_rate = ((n_all - len(recovered_instances)) / n_all) * 100.0

    # Instance-level precision: recovered instances vs predicted unique instances in annotated scope
    claimed_instances = set()
    for rec in annotated_eval_matches:
        if rec.get("s1_gt_instance"):
            claimed_instances.add(rec["s1_gt_instance"])
        if rec.get("s2_gt_instance"):
            claimed_instances.add(rec["s2_gt_instance"])
    instance_precision = (len(recovered_instances) / max(1, len(claimed_instances))) * 100.0 if claimed_instances else 0.0
    instance_f1 = (
        (2 * instance_precision * recall_on_all / (instance_precision + recall_on_all))
        if (instance_precision + recall_on_all) > 0 else 0.0
    )

    # 3. Per-Instance Detailed Diagnostic Table
    matched_by_pair = {(m.s1_entity_id, m.s2_entity_id): m for m in association_result.matched_pairs}
    ambiguous_set = {(m.s1_entity_id, m.s2_entity_id) for m in association_result.ambiguous_pairs}
    s1_matched_map = {m.s1_entity_id: m for m in association_result.matched_pairs}
    s2_matched_map = {m.s2_entity_id: m for m in association_result.matched_pairs}

    per_instance_table = []
    for inst_id in sorted(all_multi_session_instances):
        s1_trks = sorted(set(trk for (sess, trk), i in canonical_map.items() if i == inst_id and "recording_01" in sess))
        s2_trks = sorted(set(trk for (sess, trk), i in canonical_map.items() if i == inst_id and "recording_02" in sess))
        motion_state = "STATIC" if inst_id in static_multi_session_instances else "RELOCATED"

        # Check if any valid (s1, s2) pair was recovered
        found_match = None
        for t1 in s1_trks:
            for t2 in s2_trks:
                if (t1, t2) in matched_by_pair:
                    found_match = matched_by_pair[(t1, t2)]
                    break
            if found_match:
                break

        if found_match is not None:
            per_instance_table.append({
                "physical_instance_id": inst_id,
                "motion_state": motion_state,
                "s1_track_id": found_match.s1_entity_id,
                "s2_track_id": found_match.s2_entity_id,
                "outcome": getattr(found_match, "outcome", found_match.status),
                "is_correct": True,
                "distance_m": round(found_match.distance_m, 4),
                "rejection_reason": None,
            })
        else:
            # Analyze failure mode
            false_merge_info = None
            is_ambig = False
            for t1 in s1_trks:
                if t1 in s1_matched_map:
                    other_m = s1_matched_map[t1]
                    other_inst = canonical_map.get(("semantic_mapping_my_desk_recording_02", other_m.s2_entity_id))
                    if other_inst and other_inst != inst_id:
                        false_merge_info = f"S1 track {t1} falsely merged with {other_inst} ({other_m.s2_entity_id})"
                        break
                for t2 in s2_trks:
                    if (t1, t2) in ambiguous_set:
                        is_ambig = True

            if not false_merge_info:
                for t2 in s2_trks:
                    if t2 in s2_matched_map:
                        other_m = s2_matched_map[t2]
                        other_inst = canonical_map.get(("semantic_mapping_my_desk_recording_01", other_m.s1_entity_id))
                        if other_inst and other_inst != inst_id:
                            false_merge_info = f"S2 track {t2} falsely merged with {other_inst} ({other_m.s1_entity_id})"
                            break

            if false_merge_info:
                outcome = "FALSE_MERGE"
                reason = false_merge_info
            elif is_ambig:
                outcome = "AMBIGUOUS_REJECTED"
                reason = "Ambiguity rejection: competing nearby clusters within ambiguity margin."
            else:
                outcome = "UNMATCHED_GATE"
                reason = "Displacement exceeds gate threshold or no candidate cluster survived feature gating."

            per_instance_table.append({
                "physical_instance_id": inst_id,
                "motion_state": motion_state,
                "s1_track_id": s1_trks[0] if s1_trks else "NONE",
                "s2_track_id": s2_trks[0] if s2_trks else "NONE",
                "outcome": outcome,
                "is_correct": False,
                "distance_m": None,
                "rejection_reason": reason,
            })

    return {
        "algorithm_name": association_result.algorithm_name,
        "total_matched_pairs": len(association_result.matched_pairs),
        "annotated_eval_pairs_count": total_annotated,
        "true_positive_matches": true_positives,
        "false_merges": false_merges,
        "false_merge_rate_pct": round(false_merge_rate, 2),
        "false_split_rate_pct": round(false_split_rate, 2),
        "unannotated_candidate_pairs_count": len(unannotated_matches),
        "total_multi_session_instances": len(all_multi_session_instances),
        "static_multi_session_instances": len(static_multi_session_instances),
        "relocated_multi_session_instances": len(relocated_multi_session_instances),
        "recovered_static_instances_count": len(recovered_static_instances),
        "recovered_relocated_instances_count": len(recovered_relocated_instances),
        "recovered_all_instances_count": len(recovered_instances),
        "missed_static_instances_count": len(missed_static_instances),
        "missed_static_instances": missed_static_instances,
        "missed_relocated_instances": missed_relocated_instances,
        "total_ground_truth_cross_session_pairs": len(gt_cross_pairs),
        "static_ground_truth_cross_session_pairs": len(static_multi_session_instances),
        "relocated_ground_truth_cross_session_pairs": len(relocated_multi_session_instances),
        "precision_on_annotated_subset_pct": round(precision_on_annotated, 2),
        "candidate_pair_precision_pct": round(precision_on_annotated, 2),
        "pair_level_recall_pct": round(pair_recall, 2),
        "pair_level_f1_pct": round(pair_f1, 2),
        "recall_on_static_multi_session_pct": round(recall_on_static, 2),
        "static_instance_recall_pct": round(recall_on_static, 2),
        "relocated_instance_recall_pct": round(recall_on_relocated, 2),
        "recall_on_all_multi_session_pct": round(recall_on_all, 2),
        "all_instance_recall_pct": round(recall_on_all, 2),
        "instance_level_precision_pct": round(instance_precision, 2),
        "instance_level_f1_pct": round(instance_f1, 2),
        "f1_on_annotated_subset_pct": round(instance_f1, 2),
        "ambiguous_pairs_rejected_count": len(association_result.ambiguous_pairs),
        "annotated_evaluated_matches": annotated_eval_matches,
        "unannotated_matches_sample": unannotated_matches[:10],
        "per_instance_results": per_instance_table,
    }


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
    eval_b1 = evaluate_matcher_against_ground_truth(res_b1, gt_path)

    # 2. Hungarian Gated
    res_b2 = matcher.match_entities_hungarian_gated(s1_clusters, s2_clusters, registration)
    eval_b2 = evaluate_matcher_against_ground_truth(res_b2, gt_path)

    # 3. Proposed Method (Static Proximity + Uncertainty + Ambiguity)
    res_b3 = matcher.match_entities(s1_clusters, s2_clusters, registration)
    eval_b3 = evaluate_matcher_against_ground_truth(res_b3, gt_path)

    # 4. Proposed Relocation-Aware Method (Static Proximity + Relocation Hypotheses)
    res_b4 = matcher.match_entities_relocation_aware(s1_clusters, s2_clusters, registration)
    eval_b4 = evaluate_matcher_against_ground_truth(res_b4, gt_path)

    return {
        "baseline_1_nearest_centroid": {"result": res_b1, "evaluation": eval_b1},
        "baseline_2_hungarian_gated": {"result": res_b2, "evaluation": eval_b2},
        "proposed_uncertainty_symmetric": {"result": res_b3, "evaluation": eval_b3},
        "proposed_relocation_aware": {"result": res_b4, "evaluation": eval_b4},
    }
