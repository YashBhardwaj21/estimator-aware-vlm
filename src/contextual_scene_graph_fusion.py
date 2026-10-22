"""Stage 8: Contextual Semantic Map and Dynamic 3D Scene Graph Fusion.

Influenced by Rüdt et al. (Towards Dynamic 3D Scene Graphs):
Integrates estimator-aware evidence routing into persistent scene graph entities.

Maintains strict separation between:
1. Contextual movability: IMMOVABLE (fixed/mounted), MOVABLE (transportable furniture),
   MOBILE (self-propelled agent or carrier).
2. Observed motion: STATIC, DISPLACED, DYNAMIC, UNCERTAIN during observed interval.
3. Cross-session physical relocation: verified SE(2)/SE(3) measured change in object position.
4. Support / mounting relationship: mounted_to_structure, freestanding_on_surface, self_propelled.
5. Geometry validity: explicit missing/unavailable geometry state. Does NOT fabricate fake
   geometry or mark entities confirmed when measurements are missing.
6. Identity tracking: distinguishes source_track_ids (tracker IDs) from contributing_observation_ids (observation IDs).
7. Mathematically exact running mean without cap-induced drift.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Import repository scene graph ontology
try:
    from scene_graph.ontology.entity import (
        EntityType,
        EntityLifecycleState,
        VisibilityState,
        SemanticHypothesis,
        PersistentEntity,
    )
except ImportError:
    # Standalone fallback if invoked outside package root
    pass


class ContextualMovability(str, Enum):
    IMMOVABLE = "IMMOVABLE"
    MOVABLE = "MOVABLE"
    MOBILE = "MOBILE"
    UNKNOWN = "UNKNOWN"


class ObservedMotionState(str, Enum):
    STATIC = "STATIC"
    DISPLACED = "DISPLACED"
    DYNAMIC = "DYNAMIC"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ContextualSceneGraphNode:
    """Persistent entity node in the contextual dynamic 3D scene graph."""
    entity_id: str
    semantic_class: str
    contextual_movability: ContextualMovability = ContextualMovability.UNKNOWN
    observed_motion: ObservedMotionState = ObservedMotionState.STATIC
    physical_support: Optional[str] = None  # "mounted_to_structure", "freestanding_on_surface", "self_propelled"
    
    # 3D spatial geometry in verified reference coordinate frame
    geometry_valid: bool = False
    centroid_world: Optional[np.ndarray] = None
    spatial_extent_m: Optional[float] = None
    bounding_box_min: Optional[np.ndarray] = None
    bounding_box_max: Optional[np.ndarray] = None
    coordinate_frame: str = "floor_map"
    
    # Evidence confidence & tracking
    semantic_confidence: float = 0.50
    geometric_confidence: float = 0.0
    evidence_provenance: str = "MISSING"
    source_sessions: List[str] = field(default_factory=list)
    source_track_ids: List[str] = field(default_factory=list)
    contributing_observation_ids: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_observations_count: int = 0
    valid_geometry_observations_count: int = 0
    
    # Cross-session relocation and pose history
    relocation_evaluable: bool = False
    slam_invalidated: bool = False
    predicted_landmark_action: str = "RETAIN"  # "INVALIDATE" or "RETAIN"
    landmark_action_reason: Optional[str] = None
    motion_confidence: Optional[float] = None
    measured_displacement_m: Optional[float] = None
    displacement_uncertainty_m: Optional[float] = None
    pose_history: List[Dict[str, Any]] = field(default_factory=list)

    # Selective router metadata
    vlm_query_requested: bool = False
    vlm_inference_executed: bool = False
    vlm_decision: str = "RESOLVE_CHEAP"
    vlm_reason_codes: List[str] = field(default_factory=list)
    vlm_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        mov_str = self.contextual_movability.value if isinstance(self.contextual_movability, ContextualMovability) else str(self.contextual_movability)
        mot_str = self.observed_motion.value if isinstance(self.observed_motion, ObservedMotionState) else str(self.observed_motion)
        action_str = self.predicted_landmark_action
        if not action_str or action_str == "RETAIN":
            if self.slam_invalidated:
                action_str = "INVALIDATE"
            else:
                action_str = "RETAIN"
        reason_str = self.landmark_action_reason or ("SIGNIFICANT_REGISTERED_DISPLACEMENT" if self.slam_invalidated else "STATISTICALLY_STATIONARY")

        return {
            "entity_id": self.entity_id,
            "predicted_entity_id": self.entity_id,
            "semantic_class": self.semantic_class,
            "contextual_movability": mov_str,
            "predicted_movability": mov_str,
            "observed_motion": mot_str,
            "predicted_motion": mot_str,
            "motion_confidence": self.motion_confidence,
            "abstained": (mov_str in ["UNKNOWN", "None", None]),
            "slam_invalidated": self.slam_invalidated,
            "predicted_landmark_action": action_str,
            "landmark_action_reason": reason_str,
            "physical_support": self.physical_support,
            "geometry_valid": self.geometry_valid,
            "centroid_world": [round(float(x), 4) for x in self.centroid_world] if self.centroid_world is not None else None,
            "spatial_extent_m": round(float(self.spatial_extent_m), 4) if self.spatial_extent_m is not None else None,
            "coordinate_frame": self.coordinate_frame,
            "semantic_confidence": round(float(self.semantic_confidence), 4),
            "geometric_confidence": round(float(self.geometric_confidence), 4),
            "evidence_provenance": self.evidence_provenance,
            "source_sessions": sorted(set(self.source_sessions)),
            "source_track_ids": sorted(set(self.source_track_ids)),
            "contributing_observations_count": len(self.contributing_observation_ids),
            "total_observations_count": self.total_observations_count,
            "relocation_evaluable": self.relocation_evaluable,
            "measured_displacement_m": self.measured_displacement_m,
            "displacement_uncertainty_m": self.displacement_uncertainty_m,
            "pose_history": list(self.pose_history),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "vlm_decision": self.vlm_decision,
            "vlm_query_requested": self.vlm_query_requested,
            "vlm_inference_executed": self.vlm_inference_executed,
            "vlm_reason_codes": self.vlm_reason_codes,
        }

    def to_persistent_entity(self) -> Any:
        """Convert to authoritative PersistentEntity object for scene graph integration."""
        pose = np.eye(4, dtype=np.float64)
        if self.geometry_valid and self.centroid_world is not None:
            pose[:3, 3] = self.centroid_world

        ent_type = EntityType.OBJECT
        if self.contextual_movability == ContextualMovability.MOBILE or self.semantic_class in ["person", "robot"]:
            ent_type = EntityType.AGENT

        hypotheses = [
            SemanticHypothesis(
                label=self.semantic_class,
                confidence=float(self.semantic_confidence),
                source="contextual_router_fusion",
            )
        ]

        # Explicit lifecycle state: Do not confirm entities without valid geometry or verified semantics
        if self.geometry_valid and self.semantic_confidence >= 0.70 and self.total_observations_count >= 2:
            lifecycle = EntityLifecycleState.CONFIRMED
        else:
            lifecycle = EntityLifecycleState.TENTATIVE

        attrs = {
            "contextual_movability": self.contextual_movability.value if isinstance(self.contextual_movability, ContextualMovability) else str(self.contextual_movability),
            "observed_motion": self.observed_motion.value if isinstance(self.observed_motion, ObservedMotionState) else str(self.observed_motion),
            "physical_support": self.physical_support,
            "geometry_valid": self.geometry_valid,
            "spatial_extent_m": self.spatial_extent_m,
            "evidence_provenance": self.evidence_provenance,
            "relocation_evaluable": self.relocation_evaluable,
            "measured_displacement_m": self.measured_displacement_m,
            "displacement_uncertainty_m": self.displacement_uncertainty_m,
            "pose_history": list(self.pose_history),
            "contributing_observation_ids": list(self.contributing_observation_ids[:50]),
            "vlm_decision": self.vlm_decision,
            "vlm_reason_codes": self.vlm_reason_codes,
        }

        unc_diag = (1.0 - min(0.9, self.geometric_confidence)) if self.geometry_valid else 1.0

        return PersistentEntity(
            entity_id=self.entity_id,
            entity_type=ent_type,
            semantic_hypotheses=hypotheses,
            pose=pose,
            uncertainty=np.eye(3, dtype=np.float64) * unc_diag,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            last_observed=self.last_seen,
            visibility_state=VisibilityState.VISIBLE,
            lifecycle_state=lifecycle,
            source_track_ids=sorted(set(self.source_track_ids)),
            attributes=attrs,
        )


class ContextualSceneGraphFusion:
    """Manages persistent contextual nodes and updates them with routed evidence."""

    def __init__(self):
        self.nodes: Dict[str, ContextualSceneGraphNode] = {}

    def update_or_create_node(
        self,
        entity_id: str,
        semantic_class: str,
        routed_result: Dict[str, Any],
        observation_meta: Optional[Dict[str, Any]] = None,
    ) -> ContextualSceneGraphNode:
        """Update existing node or instantiate a new persistent scene graph node.

        Guarantees:
        - Geometric centroid and extent are preserved even if semantic movability is UNKNOWN.
        - Movability and observed physical motion remain distinct attributes.
        - Does NOT initialize fake geometry from absent measurements.
        - Running mean statistics update without cap-induced drift:
            m_k = m_{k-1} + (x_k - m_{k-1}) / k
        - source_track_ids strictly stores tracker IDs, not observation IDs.
        """
        obs_meta = observation_meta or {}
        ts_val = obs_meta.get("timestamp")
        ts = float(ts_val) if ts_val is not None else 0.0
        session = obs_meta.get("source_session", "session_01")
        obs_id = obs_meta.get("obs_id", "")
        track_id = obs_meta.get("local_track_id", "")

        # Strict geometry extraction: No fake defaults
        raw_centroid = obs_meta.get("centroid")
        if raw_centroid is None:
            raw_centroid = obs_meta.get("centroid_floor")
        if raw_centroid is None:
            raw_centroid = obs_meta.get("centroid_world")

        # Coordinate frame verification: transform session 2 local coordinates into common reference frame
        reg = obs_meta.get("registration")
        coord_frame = obs_meta.get("coordinate_frame", "floor_map")
        is_cross_session_aligned = (session == getattr(reg, "aligned_session", "") or "recording_02" in session)

        if is_cross_session_aligned and coord_frame != "reference_frame":
            if reg is not None and getattr(reg, "is_valid", False):
                if raw_centroid is not None and len(raw_centroid) >= 2 and all(x is not None for x in raw_centroid[:2]):
                    c2_floor = np.array(raw_centroid[:2], dtype=np.float64)
                    c1_aligned = reg.R_12 @ c2_floor + reg.translation_m
                    c_rest = list(raw_centroid[2:]) if len(raw_centroid) > 2 else []
                    raw_centroid = [float(c1_aligned[0]), float(c1_aligned[1])] + c_rest
                    coord_frame = "reference_frame"
            else:
                # Registration is missing or invalid: strictly reject geometry fusion to prevent unaligned coordinate corruption
                raw_centroid = None
                coord_frame = "unaligned_rejected"

        geometry_valid = False
        centroid_arr: Optional[np.ndarray] = None
        # Strict 3D geometry: require full 3D coordinates (x, y, z); do not fabricate z = 0.0
        if raw_centroid is not None and len(raw_centroid) >= 3 and all(x is not None for x in raw_centroid[:3]):
            geometry_valid = True
            centroid_arr = np.array(raw_centroid[:3], dtype=np.float64)

        extent = float(obs_meta.get("spatial_extent_m")) if obs_meta.get("spatial_extent_m") is not None else None

        pred_movability_str = routed_result.get("prediction", "UNKNOWN")
        if pred_movability_str in ["IMMOVABLE", "MOVABLE", "MOBILE"]:
            new_movability = ContextualMovability(pred_movability_str)
        else:
            new_movability = ContextualMovability.UNKNOWN

        # Extract motion state
        obs_motion_str = routed_result.get("observed_motion", "STATIC")
        if obs_motion_str in ["STATIC", "DISPLACED", "DYNAMIC", "UNCERTAIN"]:
            new_motion = ObservedMotionState(obs_motion_str)
        else:
            new_motion = ObservedMotionState.STATIC

        if entity_id not in self.nodes:
            # Instantiate new node
            node = ContextualSceneGraphNode(
                entity_id=entity_id,
                semantic_class=semantic_class,
                contextual_movability=new_movability,
                observed_motion=new_motion,
                physical_support=routed_result.get("physical_support"),
                geometry_valid=geometry_valid,
                centroid_world=centroid_arr.copy() if centroid_arr is not None else None,
                spatial_extent_m=extent,
                coordinate_frame=coord_frame,
                semantic_confidence=0.90 if new_movability != ContextualMovability.UNKNOWN else 0.40,
                geometric_confidence=0.85 if (geometry_valid and obs_meta.get("depth_valid", True)) else 0.0,
                evidence_provenance=routed_result.get("evidence_provenance", "MEASURED_FULL" if geometry_valid else "MISSING"),
                source_sessions=[session] if session else [],
                source_track_ids=[track_id] if track_id else [],
                contributing_observation_ids=[obs_id] if obs_id else [],
                first_seen=ts,
                last_seen=ts,
                total_observations_count=1,
                valid_geometry_observations_count=1 if geometry_valid else 0,
                relocation_evaluable=bool(routed_result.get("relocation_evaluable", False)),
                slam_invalidated=(
                    semantic_class == "person"
                    or new_motion in [ObservedMotionState.DYNAMIC, ObservedMotionState.DISPLACED]
                    or bool(obs_meta.get("is_relocation_event", False))
                ),
                measured_displacement_m=routed_result.get("world_displacement_m"),
                displacement_uncertainty_m=routed_result.get("displacement_uncertainty_m"),
                predicted_landmark_action=routed_result.get("predicted_landmark_action", "INVALIDATE" if (
                    semantic_class == "person"
                    or new_motion in [ObservedMotionState.DYNAMIC, ObservedMotionState.DISPLACED]
                    or bool(obs_meta.get("is_relocation_event", False))
                ) else "RETAIN"),
                landmark_action_reason=routed_result.get("landmark_action_reason"),
                motion_confidence=routed_result.get("motion_confidence"),
                vlm_query_requested=bool(routed_result.get("vlm_required", False)),
                vlm_inference_executed=bool(routed_result.get("vlm_inference_executed", False)),
                vlm_decision=routed_result.get("decision", "RESOLVE_CHEAP"),
                vlm_reason_codes=list(routed_result.get("reason_codes", [])),
                vlm_model=routed_result.get("vlm_model"),
            )
            self.nodes[entity_id] = node
            return node

        # Update existing node
        node = self.nodes[entity_id]
        node.last_seen = ts
        node.total_observations_count += 1
        if (
            semantic_class == "person"
            or new_motion in [ObservedMotionState.DYNAMIC, ObservedMotionState.DISPLACED]
            or bool(obs_meta.get("is_relocation_event", False))
        ):
            node.slam_invalidated = True
        if coord_frame == "reference_frame":
            node.coordinate_frame = "reference_frame"
        if session and session not in node.source_sessions:
            node.source_sessions.append(session)
        if track_id and track_id not in node.source_track_ids:
            node.source_track_ids.append(track_id)
        if obs_id and obs_id not in node.contributing_observation_ids:
            node.contributing_observation_ids.append(obs_id)

        # 1. Relocation-Aware Geometry and Pose History Maintenance
        # If an object is confirmed relocated (significant displacement >= 0.25m in common frame
        # accompanied by displacement evidence or inter-session relocation),
        # do NOT average old and new poses together (which creates phantom landmarks).
        # Instead, record the old pose in pose_history and update the current pose to the new measurement.
        is_relocation = False
        if node.centroid_world is not None and centroid_arr is not None:
            spatial_shift = float(np.linalg.norm(centroid_arr[:2] - node.centroid_world[:2]))
            has_relocation_signal = (
                new_motion == ObservedMotionState.DISPLACED
                or routed_result.get("observed_motion") == "DISPLACED"
                or bool(obs_meta.get("is_relocation_event", False))
            )
            if spatial_shift >= 0.25 and has_relocation_signal:
                is_relocation = True

        if geometry_valid and centroid_arr is not None:
            if is_relocation and node.centroid_world is not None:
                # Confirmed relocation: preserve previous pose in time-stamped history and invalidate old landmark
                node.slam_invalidated = True
                old_pose_record = {
                    "centroid_world": [round(float(x), 4) for x in node.centroid_world],
                    "spatial_extent_m": round(float(node.spatial_extent_m), 4) if node.spatial_extent_m is not None else None,
                    "first_seen": node.first_seen,
                    "last_seen": node.last_seen,
                    "valid_observations_count": node.valid_geometry_observations_count,
                    "coordinate_frame": node.coordinate_frame,
                    "relocation_displacement_m": round(float(np.linalg.norm(centroid_arr - node.centroid_world)), 4),
                    "slam_invalidated": True,
                    "landmark_status": "INVALIDATED_RELOCATED",
                }
                node.pose_history.append(old_pose_record)

                # Reset current pose strictly to the new measurement
                node.centroid_world = centroid_arr.copy()
                node.valid_geometry_observations_count = 1
                node.geometry_valid = True
                node.geometric_confidence = 0.85
                if extent is not None:
                    node.spatial_extent_m = extent
            else:
                # Stationary compatible observations: exact incremental running mean
                node.valid_geometry_observations_count += 1
                k = node.valid_geometry_observations_count
                if node.centroid_world is None:
                    node.centroid_world = centroid_arr.copy()
                    node.geometry_valid = True
                    node.geometric_confidence = 0.85
                else:
                    node.centroid_world = node.centroid_world + (centroid_arr - node.centroid_world) / k

                if extent is not None:
                    if node.spatial_extent_m is None:
                        node.spatial_extent_m = extent
                    else:
                        node.spatial_extent_m = node.spatial_extent_m + (extent - node.spatial_extent_m) / k

        # 2. Observed motion update
        if is_relocation or new_motion == ObservedMotionState.DISPLACED:
            node.observed_motion = ObservedMotionState.DISPLACED
            node.relocation_evaluable = True
        elif new_motion == ObservedMotionState.DYNAMIC:
            node.observed_motion = ObservedMotionState.DYNAMIC

        # 3. Contextual movability update with conflict resolution
        if new_movability != ContextualMovability.UNKNOWN:
            if node.contextual_movability == ContextualMovability.UNKNOWN:
                node.contextual_movability = new_movability
                node.semantic_confidence = 0.85
            elif node.contextual_movability == new_movability:
                node.semantic_confidence = min(0.99, node.semantic_confidence + 0.05)
            else:
                # Contradiction between observations (e.g. MOVABLE vs IMMOVABLE)
                # Decay confidence rather than violently thrashing
                node.semantic_confidence = max(0.35, node.semantic_confidence - 0.20)
                if node.semantic_confidence < 0.40:
                    node.contextual_movability = ContextualMovability.UNKNOWN

        # Update VLM routing telemetry
        if routed_result.get("vlm_required"):
            node.vlm_query_requested = True
        if routed_result.get("vlm_inference_executed"):
            node.vlm_inference_executed = True
        node.vlm_decision = routed_result.get("decision", node.vlm_decision)
        for r_code in routed_result.get("reason_codes", []):
            if r_code not in node.vlm_reason_codes:
                node.vlm_reason_codes.append(r_code)

        if routed_result.get("relocation_evaluable") is not None:
            node.relocation_evaluable = bool(routed_result["relocation_evaluable"])
        if routed_result.get("world_displacement_m") is not None:
            node.measured_displacement_m = routed_result["world_displacement_m"]
        if routed_result.get("displacement_uncertainty_m") is not None:
            node.displacement_uncertainty_m = routed_result["displacement_uncertainty_m"]
        if routed_result.get("predicted_landmark_action") is not None:
            node.predicted_landmark_action = routed_result["predicted_landmark_action"]
        if routed_result.get("landmark_action_reason") is not None:
            node.landmark_action_reason = routed_result["landmark_action_reason"]
        if routed_result.get("motion_confidence") is not None:
            node.motion_confidence = routed_result["motion_confidence"]

        return node

    def serialize_to_json(self) -> Dict[str, Any]:
        """Serialize complete persistent scene graph to JSON-compatible dictionary."""
        return {
            "total_nodes": len(self.nodes),
            "timestamp": time.time(),
            "nodes": {eid: node.to_dict() for eid, node in self.nodes.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias for serialize_to_json."""
        return self.serialize_to_json()
