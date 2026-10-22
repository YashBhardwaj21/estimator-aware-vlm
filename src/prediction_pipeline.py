"""Shared Prediction-Only Pipeline for Cross-Session Scene Graph Generation.

Decoupled Prediction Contract:
- Strictly independent of ground-truth annotations and controlled scene events.
- Emits persistent entity IDs derived entirely from prediction cluster associations.
- Computes empirical displacement and motion from registered observations and uncertainty.
- Populates explicit predicted_movability, predicted_motion, and predicted_landmark_action.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vlm.src.contextual_scene_graph_fusion import (
    ContextualSceneGraphFusion,
    ObservedMotionState,
    ContextualMovability,
)
from vlm.src.cross_session_association import AssociationPair, EntityCluster
from vlm.src.stage5_adapter import RegistrationInfo

# Semantic classes with ambiguous or high-affordance mobility
AMBIGUOUS_AFFORDANCE_CLASSES = {
    "chair",
    "stool",
    "box",
    "cart",
    "backpack",
    "bag",
    "monitor",
    "lamp",
    "laptop",
}

# Standard deterministic semantic class mobility prior baseline
CLASS_MOBILITY_PRIORS: Dict[str, Tuple[str, float, str]] = {
    "chair": ("MOVABLE", 0.90, "PRIOR_MOVABLE_FURNITURE"),
    "monitor": ("MOVABLE", 0.85, "PRIOR_PORTABLE_ELECTRONIC"),
    "desk": ("IMMOVABLE", 0.95, "PRIOR_HEAVY_DESK"),
    "table": ("IMMOVABLE", 0.95, "PRIOR_HEAVY_FURNITURE"),
    "door": ("IMMOVABLE", 0.99, "PRIOR_STRUCTURAL_ARCHITECTURAL"),
    "wall": ("IMMOVABLE", 0.99, "PRIOR_STRUCTURAL_ARCHITECTURAL"),
    "pillar": ("IMMOVABLE", 0.99, "PRIOR_STRUCTURAL_ARCHITECTURAL"),
    "floor": ("IMMOVABLE", 0.99, "PRIOR_STRUCTURAL_ARCHITECTURAL"),
    "person": ("DYNAMIC", 0.99, "PRIOR_AGENT"),
    "laptop": ("MOVABLE", 0.90, "PRIOR_PORTABLE_ELECTRONIC"),
    "box": ("MOVABLE", 0.90, "PRIOR_MOVABLE_CONTAINER"),
    "backpack": ("MOVABLE", 0.90, "PRIOR_MOVABLE_CONTAINER"),
    "cart": ("MOVABLE", 0.95, "PRIOR_WHEELED_OBJECT"),
}


def build_policy_scene_graph(
    policy_key: str,
    active_pairs: List[AssociationPair],
    s1_clusters: Dict[str, EntityCluster],
    s2_clusters: Dict[str, EntityCluster],
    reg_info: RegistrationInfo,
    s1_session: str = "semantic_mapping_my_desk_recording_01",
    s2_session: str = "semantic_mapping_my_desk_recording_02",
    observations_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a complete persistent scene graph for a given policy without GT leakage.

    Args:
        policy_key: Routing/relocation policy identifier ('R0', 'R1', 'R2', 'R3', 'R4').
        active_pairs: Cross-session association matches from Step 2.
        s1_clusters: Extracted/consolidated clusters for Session 1.
        s2_clusters: Extracted/consolidated clusters for Session 2.
        reg_info: Validated SE(2) cross-session registration.
        s1_session: Canonical Session 1 name.
        s2_session: Canonical Session 2 name.
        observations_by_id: Optional map of raw observations to extract exact timestamps/depth.
        run_id: Optional run identifier for metadata tracking.

    Returns:
        (scene_graph_dict, summary_stats_dict)
    """
    fusion = ContextualSceneGraphFusion()

    # Step 1: Ingest Session 1 Clusters as Initial Reference Entities
    for t1, c1 in s1_clusters.items():
        # Purely prediction-derived persistent identifier
        entity_id = f"pred_s1_{t1}"

        first_obs_id = (
            c1.observation_ids[0]
            if (c1.observation_ids and len(c1.observation_ids) > 0)
            else f"obs_s1_{t1}"
        )

        t_stamp = 100.0
        if observations_by_id and first_obs_id in observations_by_id:
            t_stamp = float(observations_by_id[first_obs_id].get("timestamp", 100.0))

        depth_valid = bool(getattr(c1, "depth_valid_ratio", 1.0) >= 0.50)
        centroid_3d = (
            c1.centroid_floor.tolist()
            if len(c1.centroid_floor) >= 3
            else [float(c1.centroid_floor[0]), float(c1.centroid_floor[1]), 0.0]
        )

        obs_meta = {
            "timestamp": t_stamp,
            "source_session": s1_session,
            "obs_id": first_obs_id,
            "local_track_id": t1,
            "centroid": centroid_3d,
            "spatial_extent_m": c1.spatial_extent_m,
            "depth_valid": depth_valid,
            "coordinate_frame": "floor_map",
            "is_relocation_event": False,
        }

        prior_mov, prior_conf, _ = CLASS_MOBILITY_PRIORS.get(
            c1.semantic_class, ("MOVABLE", 0.85, "DEFAULT_FALLBACK")
        )

        routed = {
            "decision": "RESOLVE_CHEAP",
            "prediction": prior_mov,
            "observed_motion": "STATIC",
            "evidence_provenance": "MEASURED_CLUSTER",
            "relocation_evaluable": False,
            "predicted_landmark_action": "RETAIN",
            "landmark_action_reason": "SINGLE_SESSION_INITIAL_OBSERVATION",
            "motion_confidence": prior_conf,
        }

        fusion.update_or_create_node(
            entity_id=entity_id,
            semantic_class=c1.semantic_class,
            routed_result=routed,
            observation_meta=obs_meta,
        )

    # Step 2: Ingest Session 2 Clusters, Fusing via Prediction Matches
    matched_s2_map: Dict[str, AssociationPair] = {
        m.s2_entity_id: m for m in active_pairs
    }

    inlier_sigma = (
        reg_info.inlier_rms_residual_m
        if (reg_info is not None and reg_info.inlier_rms_residual_m is not None)
        else (
            reg_info.anchor_rms_residual_m
            if (reg_info is not None and reg_info.anchor_rms_residual_m is not None)
            else 0.12
        )
    )

    for t2, c2 in s2_clusters.items():
        first_obs_id = (
            c2.observation_ids[0]
            if (c2.observation_ids and len(c2.observation_ids) > 0)
            else f"obs_s2_{t2}"
        )

        t_stamp = 200.0
        if observations_by_id and first_obs_id in observations_by_id:
            t_stamp = float(observations_by_id[first_obs_id].get("timestamp", 200.0))

        depth_valid = bool(getattr(c2, "depth_valid_ratio", 1.0) >= 0.50)

        # Transform Session 2 floor coordinates into Session 1 reference frame
        c2_floor_2d = np.array(c2.centroid_floor[:2], dtype=np.float64)
        if reg_info is not None and reg_info.is_valid and reg_info.R_12 is not None:
            c2_aligned_2d = reg_info.R_12 @ c2_floor_2d + reg_info.translation_m
        else:
            c2_aligned_2d = c2_floor_2d

        z_coord = float(c2.centroid_floor[2]) if len(c2.centroid_floor) >= 3 else 0.0
        aligned_centroid_3d = [
            float(c2_aligned_2d[0]),
            float(c2_aligned_2d[1]),
            z_coord,
        ]

        matched_m = matched_s2_map.get(t2)
        prior_mov, prior_conf, _ = CLASS_MOBILITY_PRIORS.get(
            c2.semantic_class, ("MOVABLE", 0.85, "DEFAULT_FALLBACK")
        )

        if matched_m is not None and matched_m.s1_entity_id in s1_clusters:
            # Fused cross-session entity: re-use S1 persistent entity ID
            t1 = matched_m.s1_entity_id
            entity_id = f"pred_s1_{t1}"
            c1 = s1_clusters[t1]

            dist = float(matched_m.distance_m)
            sigma_disp = math.sqrt(
                c1.position_uncertainty_m**2
                + c2.position_uncertainty_m**2
                + inlier_sigma**2
            )

            # Policy-conditioned motion determination
            if not (reg_info is not None and reg_info.is_valid):
                is_reloc = False
                motion_state = "UNCERTAIN"
                action = "RETAIN"
                reason = "UNVERIFIED_REGISTRATION"
            elif policy_key == "R0":
                is_reloc = False
                motion_state = "STATIC"
                action = "RETAIN"
                reason = "SEMANTIC_PRIOR_STATIC_POLICY"
            elif policy_key == "R1":
                is_reloc = bool(dist >= 0.25)
                motion_state = "DISPLACED" if is_reloc else "STATIC"
                action = "INVALIDATE" if is_reloc else "RETAIN"
                reason = (
                    "UNWEIGHTED_EUCLIDEAN_DISPLACEMENT"
                    if is_reloc
                    else "STATISTICALLY_STATIONARY"
                )
            elif policy_key == "R2":
                has_sem_ambig = (
                    c1.semantic_class in AMBIGUOUS_AFFORDANCE_CLASSES
                    or getattr(c1, "semantic_entropy", 0.0) >= 0.35
                )
                is_reloc = bool(dist >= 0.25 and has_sem_ambig)
                motion_state = "DISPLACED" if is_reloc else "STATIC"
                action = "INVALIDATE" if is_reloc else "RETAIN"
                reason = (
                    "SEMANTIC_AMBIGUITY_DISPLACEMENT"
                    if is_reloc
                    else "STATISTICALLY_STATIONARY"
                )
            else:
                # R3 & R4: Estimator-aware statistical gating
                is_reloc = bool(dist >= 0.25 and dist >= 3.0 * sigma_disp)
                motion_state = "DISPLACED" if is_reloc else "STATIC"
                action = "INVALIDATE" if is_reloc else "RETAIN"
                reason = (
                    "SIGNIFICANT_REGISTERED_DISPLACEMENT"
                    if is_reloc
                    else "STATISTICALLY_STATIONARY"
                )

            obs_meta = {
                "timestamp": t_stamp,
                "source_session": s2_session,
                "obs_id": first_obs_id,
                "local_track_id": t2,
                "centroid": aligned_centroid_3d,
                "spatial_extent_m": c2.spatial_extent_m,
                "depth_valid": depth_valid,
                "coordinate_frame": "reference_frame",
                "is_relocation_event": is_reloc,
            }

            routed = {
                "decision": "RESOLVE_CHEAP",
                "prediction": prior_mov,
                "observed_motion": motion_state,
                "world_displacement_m": dist if is_reloc else 0.0,
                "displacement_uncertainty_m": sigma_disp,
                "evidence_provenance": "MEASURED_CLUSTER_ALIGNED",
                "relocation_evaluable": True,
                "predicted_landmark_action": action,
                "landmark_action_reason": reason,
                "motion_confidence": prior_conf,
            }

            fusion.update_or_create_node(
                entity_id=entity_id,
                semantic_class=c2.semantic_class,
                routed_result=routed,
                observation_meta=obs_meta,
            )
        else:
            # Unmatched Session 2 entity
            entity_id = f"pred_s2_{t2}"
            obs_meta = {
                "timestamp": t_stamp,
                "source_session": s2_session,
                "obs_id": first_obs_id,
                "local_track_id": t2,
                "centroid": aligned_centroid_3d,
                "spatial_extent_m": c2.spatial_extent_m,
                "depth_valid": depth_valid,
                "coordinate_frame": "reference_frame",
                "is_relocation_event": False,
            }

            routed = {
                "decision": "RESOLVE_CHEAP",
                "prediction": prior_mov,
                "observed_motion": "STATIC",
                "evidence_provenance": "MEASURED_CLUSTER_UNMATCHED",
                "relocation_evaluable": False,
                "predicted_landmark_action": "RETAIN",
                "landmark_action_reason": "UNMATCHED_S2_ENTITY",
                "motion_confidence": prior_conf,
            }

            fusion.update_or_create_node(
                entity_id=entity_id,
                semantic_class=c2.semantic_class,
                routed_result=routed,
                observation_meta=obs_meta,
            )

    sg_dict = fusion.serialize_to_json()

    # Calculate summary statistics
    nodes = list(fusion.nodes.values())
    total_nodes = len(nodes)
    stationary_count = sum(
        1
        for n in nodes
        if (
            n.observed_motion == ObservedMotionState.STATIC
            or str(n.observed_motion) == "STATIC"
        )
    )
    relocated_count = sum(
        1
        for n in nodes
        if (
            n.observed_motion == ObservedMotionState.DISPLACED
            or str(n.observed_motion) == "DISPLACED"
        )
    )
    invalidated_count = sum(1 for n in nodes if n.predicted_landmark_action == "INVALIDATE")
    retained_count = sum(1 for n in nodes if n.predicted_landmark_action == "RETAIN")
    fused_cross_session_count = sum(1 for n in nodes if len(set(n.source_sessions)) >= 2)

    stats = {
        "run_id": run_id,
        "policy_key": policy_key,
        "total_nodes": total_nodes,
        "fused_cross_session_nodes": fused_cross_session_count,
        "stationary_nodes_count": stationary_count,
        "relocated_nodes_count": relocated_count,
        "invalidated_landmarks_count": invalidated_count,
        "retained_landmarks_count": retained_count,
        "provenance": "PREDICTION_DERIVED_NO_GT_LEAKAGE",
    }

    return sg_dict, stats
