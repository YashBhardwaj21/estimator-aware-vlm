from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.data.frame_packet import FramePacket
from scene_graph.geometry.point_cloud import GeometryStatus


@dataclass(frozen=True)
class ObjectSnapshot:
    track_id: str
    class_name: str
    state: str
    object_state: str
    confidence: float
    observation_count: int
    missing_count: int
    centroid_world: Optional[tuple[float, float, float]]
    velocity_world: Optional[tuple[float, float, float]]
    obb_center_world: Optional[tuple[float, float, float]]
    obb_axes_world: Optional[tuple[tuple[float, ...], ...]]
    obb_extents_world: Optional[tuple[float, float, float]]
    bbox_min_world: Optional[tuple[float, float, float]]
    bbox_max_world: Optional[tuple[float, float, float]]
    spatial_context_id: str = "world"
    is_spatial_anchor: bool = False
    label_belief: dict[str, float] = field(default_factory=dict)
    coordinate_frame: str = "world"
    localization_status: str = "localized"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "state": self.state,
            "object_state": self.object_state,
            "confidence": float(self.confidence),
            "observation_count": int(self.observation_count),
            "missing_count": int(self.missing_count),
            "spatial_context_id": self.spatial_context_id,
            "is_spatial_anchor": self.is_spatial_anchor,
            "coordinate_frame": self.coordinate_frame,
            "localization_status": self.localization_status,
            "centroid_world": list(self.centroid_world) if self.centroid_world else None,
            "velocity_world": list(self.velocity_world) if self.velocity_world else None,
            "obb_center_world": list(self.obb_center_world) if self.obb_center_world else None,
            "obb_axes_world": [list(row) for row in self.obb_axes_world] if self.obb_axes_world else None,
            "obb_extents_world": list(self.obb_extents_world) if self.obb_extents_world else None,
            "bbox_min_world": list(self.bbox_min_world) if self.bbox_min_world else None,
            "bbox_max_world": list(self.bbox_max_world) if self.bbox_max_world else None,
            "label_belief": dict(self.label_belief),
        }


@dataclass(frozen=True)
class RelationSnapshot:
    subject_id: str
    subject_class: str
    predicate: str
    object_id: str
    object_class: str
    state: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "subject_class": self.subject_class,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "object_class": self.object_class,
            "state": self.state,
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class TelemetrySnapshot:
    frame_index: int
    timestamp: float
    input_fps: float
    processing_fps: float
    total_latency_ms: float
    queue_size: int
    dropped_frames: int
    active_objects: int
    active_relations: int
    tf_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    geometry_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    relation_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "input_fps": self.input_fps,
            "processing_fps": self.processing_fps,
            "total_latency_ms": self.total_latency_ms,
            "queue_size": self.queue_size,
            "dropped_frames": self.dropped_frames,
            "active_objects": self.active_objects,
            "active_relations": self.active_relations,
            "tf_latency_ms": self.tf_latency_ms,
            "inference_latency_ms": self.inference_latency_ms,
            "geometry_latency_ms": self.geometry_latency_ms,
            "tracking_latency_ms": self.tracking_latency_ms,
            "relation_latency_ms": self.relation_latency_ms,
        }


@dataclass(frozen=True)
class SceneGraphSnapshot:
    frame_index: int
    timestamp: float
    world_T_camera: Optional[tuple[tuple[float, ...], ...]]
    objects: tuple[ObjectSnapshot, ...]
    relations: tuple[RelationSnapshot, ...]
    telemetry: TelemetrySnapshot
    spatial_contexts: dict = field(default_factory=dict)
    localization: dict = field(default_factory=dict)

    def query_objects(
        self,
        class_name: Optional[str] = None,
        state: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[ObjectSnapshot]:
        results = []
        for obj in self.objects:
            if class_name is not None and obj.class_name != class_name:
                continue
            if state is not None and obj.state != state and obj.object_state != state:
                continue
            if obj.confidence < min_confidence:
                continue
            results.append(obj)
        return results

    def query_relations(
        self,
        subject_id: Optional[str] = None,
        object_id: Optional[str] = None,
        predicate: Optional[str] = None,
        state: Optional[str] = None,
    ) -> list[RelationSnapshot]:
        results = []
        for rel in self.relations:
            if subject_id is not None and rel.subject_id != subject_id:
                continue
            if object_id is not None and rel.object_id != object_id:
                continue
            if predicate is not None and rel.predicate != predicate:
                continue
            if state is not None and rel.state != state:
                continue
            results.append(rel)
        return results

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "world_T_camera": [list(row) for row in self.world_T_camera] if self.world_T_camera else None,
            "localization": self.localization,
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations],
            "spatial_contexts": self.spatial_contexts,
            "telemetry": self.telemetry.to_dict(),
        }


def _ndarray_to_tuple3(arr) -> Optional[tuple[float, float, float]]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape[0] != 3 or not np.isfinite(a).all():
        return None
    return (float(a[0]), float(a[1]), float(a[2]))


def _ndarray_to_matrix(arr) -> Optional[tuple[tuple[float, ...], ...]]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        return None
    return tuple(tuple(float(x) for x in row) for row in a)


def create_snapshot(
    graph: TemporalSceneGraph,
    packet: FramePacket,
    *,
    input_fps: float = 0.0,
    processing_fps: float = 0.0,
    total_latency_ms: float = 0.0,
    queue_size: int = 0,
    dropped_frames: int = 0,
    tf_latency_ms: float = 0.0,
    inference_latency_ms: float = 0.0,
    geometry_latency_ms: float = 0.0,
    tracking_latency_ms: float = 0.0,
    relation_latency_ms: float = 0.0,
) -> SceneGraphSnapshot:
    active_nodes = graph.get_active_nodes()
    active_edges = graph.get_active_edges()

    class_map: dict[str, str] = {}
    objects: list[ObjectSnapshot] = []

    for node in active_nodes:
        track = node.track
        class_map[track.object_id] = track.class_name

        is_world_track = (getattr(track, "coordinate_frame", "world") == "world" and track.centroid_world is not None)

        obb_center = None
        obb_axes = None
        obb_extents = None
        bbox_min = None
        bbox_max = None

        if is_world_track and track.recent_observations:
            latest_obs = track.recent_observations[-1]
            geom = getattr(latest_obs, "object_geometry", None)
            if geom is not None and geom.status == GeometryStatus.VALID and getattr(geom, "centroid_world", None) is not None:
                obb_center = _ndarray_to_tuple3(geom.obb_center_world)
                obb_axes = _ndarray_to_matrix(geom.obb_axes_world)
                obb_extents = _ndarray_to_tuple3(geom.obb_extents_world)
                bbox_min = _ndarray_to_tuple3(geom.bbox_min_world)
                bbox_max = _ndarray_to_tuple3(geom.bbox_max_world)

        velocity = None
        if is_world_track and track.velocity_world is not None:
            velocity = _ndarray_to_tuple3(track.velocity_world)

        label_belief = getattr(track, "label_belief", {})
        if isinstance(label_belief, dict):
            label_belief_dict = {str(k): float(v) for k, v in label_belief.items()}
        else:
            label_belief_dict = {}

        centroid_world = _ndarray_to_tuple3(track.centroid_world) if is_world_track else None
        spatial_context = getattr(node, "spatial_context_id", "world" if is_world_track else "camera")
        if not is_world_track:
            spatial_context = "camera"

        objects.append(ObjectSnapshot(
            track_id=track.object_id,
            class_name=track.class_name,
            state=track.state.value,
            object_state=node.state.value,
            confidence=float(track.detection_confidence),
            observation_count=track.observation_count,
            missing_count=track.missing_count,
            centroid_world=centroid_world,
            velocity_world=velocity,
            obb_center_world=obb_center,
            obb_axes_world=obb_axes,
            obb_extents_world=obb_extents,
            bbox_min_world=bbox_min,
            bbox_max_world=bbox_max,
            spatial_context_id=spatial_context,
            is_spatial_anchor=getattr(node, "is_spatial_anchor", False),
            label_belief=label_belief_dict,
            coordinate_frame="world" if is_world_track else "camera",
            localization_status="localized" if is_world_track else "unlocalized",
        ))

    relations: list[RelationSnapshot] = []
    for edge in active_edges:
        evidence_weight = 1.0
        if hasattr(edge, "latest_evidence") and edge.latest_evidence is not None:
            evidence_weight = float(getattr(edge.latest_evidence, "confidence", 1.0))

        relations.append(RelationSnapshot(
            subject_id=edge.subject_id,
            subject_class=class_map.get(edge.subject_id, "unknown"),
            predicate=edge.predicate,
            object_id=edge.object_id,
            object_class=class_map.get(edge.object_id, "unknown"),
            state=edge.state.value,
            confidence=evidence_weight,
        ))

    telemetry = TelemetrySnapshot(
        frame_index=packet.frame_index,
        timestamp=packet.timestamp,
        input_fps=input_fps,
        processing_fps=processing_fps,
        total_latency_ms=total_latency_ms,
        queue_size=queue_size,
        dropped_frames=dropped_frames,
        active_objects=len(objects),
        active_relations=len(relations),
        tf_latency_ms=tf_latency_ms,
        inference_latency_ms=inference_latency_ms,
        geometry_latency_ms=geometry_latency_ms,
        tracking_latency_ms=tracking_latency_ms,
        relation_latency_ms=relation_latency_ms,
    )

    has_pose = packet.world_T_camera is not None and getattr(packet, "transform_valid", False) is True
    is_stale_source = getattr(packet, "transform_source", "") in ("stale_tf", "stale_slam_tf")
    age = getattr(packet, "pose_age", float("inf"))
    is_stale_age = np.isfinite(age) and (age > 0.20)
    is_stale = is_stale_source or (has_pose and is_stale_age)

    is_global_valid = (
        has_pose
        and getattr(packet, "is_persistent_world_frame", True) is True
        and not is_stale
    )
    world_T_camera = _ndarray_to_matrix(packet.world_T_camera) if is_global_valid else None

    localization_dict = {
        "status": "localized" if is_global_valid else ("stale_pose" if is_stale else "no_pose"),
        "coordinate_frame": "world" if is_global_valid else "camera",
        "transform_valid": is_global_valid,
        "transform_source": packet.transform_source,
        "pose_timestamp": packet.pose_timestamp if is_global_valid else None,
        "pose_age_sec": round(float(packet.pose_age), 4) if (is_global_valid and np.isfinite(packet.pose_age)) else None,
        "is_persistent_world_frame": getattr(packet, "is_persistent_world_frame", True),
    }

    contexts_dict = {}
    for cid, ctx in getattr(graph, "spatial_contexts", {}).items():
        contexts_dict[cid] = {
            "context_id": ctx.context_id,
            "parent_context_id": ctx.parent_context_id,
            "anchor_track_id": ctx.anchor_track_id,
            "level": ctx.level,
            "member_track_ids": list(ctx.member_track_ids),
            "bbox_min": list(ctx.bounding_box_min) if getattr(ctx, "bounding_box_min", None) is not None else None,
            "bbox_max": list(ctx.bounding_box_max) if getattr(ctx, "bounding_box_max", None) is not None else None,
        }

    return SceneGraphSnapshot(
        frame_index=packet.frame_index,
        timestamp=packet.timestamp,
        world_T_camera=world_T_camera,
        objects=tuple(objects),
        relations=tuple(relations),
        telemetry=telemetry,
        spatial_contexts=contexts_dict,
        localization=localization_dict,
    )
