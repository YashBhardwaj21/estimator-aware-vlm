from typing import List, Optional
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket, LocalizationMode
from scene_graph.perception.observation import Observation
from scene_graph.tracking.causal_tracker import CausalTracker
from scene_graph.temporal.object_state import ObjectStateMachine
from scene_graph.temporal.relation_state import RelationStateMachine
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.relations.registry import RelationRegistry

from scene_graph.relations.distance import DistanceRelationModule
from scene_graph.relations.support import SupportRelationModule
from scene_graph.relations.directional import DirectionalRelationModule
from scene_graph.relations.containment import ContainmentRelationModule
from scene_graph.relations.depth_order import DepthOrderRelationModule
from scene_graph.relations.occlusion import OcclusionRelationModule

from scene_graph.relations.inverse_algebra import derive_inverse_evidence

from scene_graph.geometry.point_cloud import (
    GeometryStatus,
    ObjectGeometry,
    compute_object_geometry,
)
from scene_graph.geometry.noise_model import create_noise_model_from_config
from scene_graph.geometry.reference_frame import CameraFrame, RelationReferenceFrame
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.geometry.provenance import GeometrySource
from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.tracking.track import Track, TrackState


class SceneGraphPipeline:
    """Core pipeline orchestrator for a single scene graph update cycle.

    Owns the tracker, relation registry, state machines, and the persistent
    world-model graph.  Each call to :meth:`update` advances the entire
    pipeline by one frame::

        FramePacket
          → geometry computation
          → CausalTracker.update
          → ObjectStateMachine.update
          → RelationRegistry.compute_all  (context → candidates → evaluation)
          → RelationStateMachine.update
          → TemporalSceneGraph.update_nodes / update_edges
          → returns TemporalSceneGraph
    """

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.geometry is None:
            raise ValueError("Geometry configuration is required.")

        if config.temporal is None or config.temporal.relation is None:
            raise ValueError("Temporal relation configuration is required.")

        self.config = config
        self.noise_model = create_noise_model_from_config(config)
        self.tracker = CausalTracker(config)

        self.registry = RelationRegistry(config)
        self._register_modules()

        self.object_state_machine = ObjectStateMachine()
        self.relation_state_machine = RelationStateMachine(
            config.temporal.relation
        )

        self.graph = TemporalSceneGraph()

    def _register_modules(self) -> None:
        self.registry.register(DistanceRelationModule(self.config))
        self.registry.register(SupportRelationModule(self.config))
        self.registry.register(DirectionalRelationModule(self.config))
        self.registry.register(ContainmentRelationModule(self.config))
        self.registry.register(DepthOrderRelationModule(self.config))
        self.registry.register(OcclusionRelationModule(self.config))

    def _compute_geometry(
        self,
        packet: FramePacket,
        observations: List[Observation],
        depth_m,
    ) -> None:

        if packet.depth is None:
            for observation in observations:
                observation.object_geometry = ObjectGeometry(
                    status=GeometryStatus.NO_DEPTH
                )
            return

        max_age = 0.20
        if hasattr(self.config, "estimation") and hasattr(self.config.estimation, "max_pose_age"):
            max_age = self.config.estimation.max_pose_age
        elif hasattr(self.config, "pose") and hasattr(self.config.pose, "max_pose_age"):
            max_age = self.config.pose.max_pose_age

        has_pose = packet.world_T_camera is not None and getattr(packet, "transform_valid", False) is True
        is_stale_source = getattr(packet, "transform_source", "") in ("stale_tf", "stale_slam_tf")
        age = getattr(packet, "pose_age", float("inf"))
        is_stale_age = np.isfinite(age) and (age > max_age)

        is_stale = is_stale_source or (has_pose and is_stale_age)
        is_global_valid = (
            has_pose
            and getattr(packet, "is_persistent_world_frame", True) is True
            and not is_stale
        )
        status = (
            GeometryStatus.VALID
            if is_global_valid
            else (GeometryStatus.STALE_POSE if is_stale else GeometryStatus.NO_POSE)
        )

        allow_fallback = getattr(packet, "metadata", {}).get("allow_camera_fallback", False)
        if not is_global_valid and packet.localization_mode == LocalizationMode.WORLD_MODE and not allow_fallback:
            for observation in observations:
                observation.object_geometry = ObjectGeometry(
                    status=status,
                    coordinate_frame="camera",
                    localization_status=status.name,
                    pose_timestamp=getattr(packet.pose_estimate, "pose_timestamp", None) if is_stale else None,
                )
            return

        effective_pose = packet.world_T_camera if is_global_valid else None

        geometry_config = self.config.geometry

        for observation in observations:
            mask = observation.get_mask()

            if mask is None:
                observation.object_geometry = ObjectGeometry(
                    status=GeometryStatus.INVALID_GEOMETRY,
                    coordinate_frame="camera",
                    localization_status="invalid_mask",
                    pose_timestamp=packet.pose_timestamp,
                )
                continue

            geom = compute_object_geometry(
                mask=mask,
                depth_m=depth_m,
                intrinsics=packet.camera_intrinsics,
                pose=effective_pose,
                min_valid_points=geometry_config.min_valid_points,
                mad_k=geometry_config.robust_depth.k,
                voxel_size_m=geometry_config.downsampling.voxel_size_m,
                measurement_noise_std_m=geometry_config.measurement_noise_std_m,
                noise_model=self.noise_model,
                min_depth_m=getattr(geometry_config, "min_depth_m", 0.10),
                max_depth_m=getattr(geometry_config, "max_depth_m", 10.0),
                spatial_outlier_sigma=getattr(geometry_config, "spatial_outlier_sigma", 3.0),
            )

            if geom is not None:
                if not is_global_valid:
                    # Camera-local fallback for local 2D/3D tracking.
                    # Strictly clear world-frame coordinates so camera coordinates NEVER pollute the persistent world graph.
                    geom.points_world = None
                    geom.points_world_sampled = None
                    geom.centroid_world = None
                    geom.bbox_min_world = None
                    geom.bbox_max_world = None
                    geom.obb_center_world = None
                    geom.obb_axes_world = None
                    geom.obb_extents_world = None
                    geom.position_covariance_world = None
                    if geom.status == GeometryStatus.VALID:
                        geom.status = status
                    geom.coordinate_frame = "camera"
                    geom.localization_status = status.name
                    geom.pose_timestamp = getattr(packet.pose_estimate, "pose_timestamp", None) if is_stale else None
                    setattr(geom, "frame", "camera")
                    setattr(geom, "source", "local_fallback")
                else:
                    geom.coordinate_frame = "world"
                    geom.localization_status = "localized"
                    geom.pose_timestamp = packet.pose_timestamp
                    setattr(geom, "frame", "world")
                    setattr(geom, "source", "slam")

            observation.object_geometry = geom

    def _build_observation_geometry(
        self,
        tracks: list[Track],
        frame_index: int,
        intrinsics: Optional[CameraIntrinsics] = None,
        world_T_camera: Optional[np.ndarray] = None,
    ) -> dict[str, ObservationGeometry]:

        geometry = {}

        for track in tracks:
            if not track.recent_observations:
                continue

            observation = track.recent_observations[-1]

            if observation.frame_index == frame_index:
                object_geometry = observation.object_geometry

                if (
                    object_geometry is None
                    or object_geometry.status not in (GeometryStatus.VALID, GeometryStatus.STALE_POSE, GeometryStatus.NO_POSE)
                ):
                    continue

                is_local = (
                    getattr(object_geometry, "coordinate_frame", "world") == "camera"
                    or getattr(object_geometry, "frame", "world") == "camera"
                    or object_geometry.centroid_world is None
                )

                if is_local:
                    pts = object_geometry.points_camera
                    centroid = object_geometry.centroid_camera
                    if centroid is None or pts is None:
                        continue

                    geometry[track.object_id] = ObservationGeometry(
                        obs_id=observation.obs_id,
                        track_id=track.object_id,
                        centroid_world=None,
                        position_covariance_world=None,
                        bbox_min_world=None,
                        bbox_max_world=None,
                        obb_center_world=None,
                        obb_axes_world=None,
                        obb_extents_world=None,
                        depth_stats=object_geometry.depth_stats,
                        points_world_sampled=None,
                        points_world=None,
                        points_camera=pts,
                        centroid_camera=centroid,
                        coordinate_frame="camera",
                        localization_status=getattr(object_geometry, "localization_status", "unlocalized"),
                        mask=observation.get_mask(),
                        valid_point_count=object_geometry.valid_point_count,
                        geometry_source=GeometrySource.OBSERVED,
                        is_in_frustum=True,
                    )
                else:
                    centroid = object_geometry.centroid_world
                    pts = object_geometry.points_world
                    if centroid is None or pts is None:
                        continue

                    geometry[track.object_id] = ObservationGeometry(
                        obs_id=observation.obs_id,
                        track_id=track.object_id,
                        centroid_world=centroid,
                        position_covariance_world=object_geometry.position_covariance_world,
                        bbox_min_world=object_geometry.bbox_min_world,
                        bbox_max_world=object_geometry.bbox_max_world,
                        obb_center_world=object_geometry.obb_center_world,
                        obb_axes_world=object_geometry.obb_axes_world,
                        obb_extents_world=object_geometry.obb_extents_world,
                        depth_stats=object_geometry.depth_stats,
                        points_world_sampled=object_geometry.points_world_sampled,
                        points_world=pts,
                        points_camera=object_geometry.points_camera,
                        centroid_camera=object_geometry.centroid_camera,
                        coordinate_frame="world",
                        localization_status="localized",
                        mask=observation.get_mask(),
                        valid_point_count=object_geometry.valid_point_count,
                        geometry_source=GeometrySource.OBSERVED,
                        is_in_frustum=True,
                    )
            else:
                # Synthesize PREDICTED geometry for active unobserved track (only if tracked in world frame)
                if (
                    track.missing_count > 0 
                    and track.state != TrackState.LOST
                    and getattr(track, "coordinate_frame", "world") == "world"
                    and track.centroid_world is not None
                ):
                    last_geom = observation.object_geometry
                    centroid = track.centroid_world
                    cov = track.position_covariance_world
                    if cov is None and last_geom is not None:
                        cov = last_geom.position_covariance_world

                    bbox_min = None
                    bbox_max = None
                    obb_center = None
                    obb_axes = None
                    obb_extents = None
                    points_world = None
                    points_sampled = None

                    if last_geom is not None and last_geom.status == GeometryStatus.VALID and getattr(last_geom, "centroid_world", None) is not None:
                        delta_pos = centroid - last_geom.centroid_world
                        if last_geom.bbox_min_world is not None:
                            bbox_min = last_geom.bbox_min_world + delta_pos
                        if last_geom.bbox_max_world is not None:
                            bbox_max = last_geom.bbox_max_world + delta_pos
                        if last_geom.obb_center_world is not None:
                            obb_center = last_geom.obb_center_world + delta_pos
                        obb_axes = last_geom.obb_axes_world
                        obb_extents = last_geom.obb_extents_world
                        if last_geom.points_world is not None:
                            points_world = last_geom.points_world + delta_pos
                        if last_geom.points_world_sampled is not None:
                            points_sampled = last_geom.points_world_sampled + delta_pos

                    is_in_frustum = getattr(track, "is_in_frustum", True)
                    if intrinsics is not None and world_T_camera is not None and centroid is not None:
                        if bbox_min is not None and bbox_max is not None:
                            is_in_frustum = bool(intrinsics.is_box_in_frustum(bbox_min, bbox_max, world_T_camera))
                        else:
                            is_in_frustum = bool(intrinsics.is_world_point_in_frustum(centroid, world_T_camera))

                    geometry[track.object_id] = ObservationGeometry(
                        obs_id=f"{observation.obs_id}_pred",
                        track_id=track.object_id,
                        centroid_world=centroid,
                        position_covariance_world=cov,
                        bbox_min_world=bbox_min,
                        bbox_max_world=bbox_max,
                        obb_center_world=obb_center,
                        obb_axes_world=obb_axes,
                        obb_extents_world=obb_extents,
                        points_world=points_world,
                        points_world_sampled=points_sampled,
                        points_camera=None,
                        centroid_camera=None,
                        coordinate_frame="world",
                        localization_status="predicted",
                        mask=None,
                        depth_stats=last_geom.depth_stats if last_geom is not None else None,
                        valid_point_count=last_geom.valid_point_count if last_geom is not None else 0,
                        geometry_source=GeometrySource.PREDICTED,
                        is_in_frustum=is_in_frustum,
                    )

        return geometry

    def update(
        self,
        packet: FramePacket,
        observations: List[Observation],
        obs_frame_index: Optional[int] = None,
        obs_timestamp: Optional[float] = None,
        obs_packet: Optional[FramePacket] = None,
    ) -> TemporalSceneGraph:

        self.graph.current_frame_index = packet.frame_index
        self.graph.current_timestamp = packet.timestamp
        depth_m = packet.depth

        if obs_packet is not None:
            self._compute_geometry(
                packet=obs_packet,
                observations=observations,
                depth_m=obs_packet.depth,
            )
        elif observations and getattr(observations[0], "object_geometry", None) is None:
            self._compute_geometry(
                packet=packet,
                observations=observations,
                depth_m=depth_m,
            )

        if obs_frame_index is not None and obs_frame_index < packet.frame_index:
            tracks = self.tracker.update_delayed(
                observations=observations,
                obs_frame_index=obs_frame_index,
                obs_timestamp=obs_timestamp if obs_timestamp is not None else packet.timestamp,
                current_frame_index=packet.frame_index,
                current_timestamp=packet.timestamp,
                camera_intrinsics=packet.camera_intrinsics,
                world_T_camera=packet.world_T_camera,
            )
        else:
            tracks = self.tracker.update(
                observations=observations,
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
                camera_intrinsics=packet.camera_intrinsics,
                world_T_camera=packet.world_T_camera,
            )

        object_states = self.object_state_machine.update(
            tracks=tracks,
            frame_index=packet.frame_index,
        )

        # 1. Relation inference operating directly from tracker tracks
        relation_tracks = [
            track
            for track in tracks
            if track.state in (
                TrackState.ACTIVE,
                TrackState.TEMPORARILY_UNOBSERVED,
            )
        ]

        all_evidences = []
        relation_states = {}

        max_age = 0.20
        if hasattr(self.config, "estimation") and hasattr(self.config.estimation, "max_pose_age"):
            max_age = self.config.estimation.max_pose_age
        elif hasattr(self.config, "pose") and hasattr(self.config.pose, "max_pose_age"):
            max_age = self.config.pose.max_pose_age

        is_stale = (
            getattr(packet, "transform_source", "") in ("stale_tf", "stale_slam_tf")
            or getattr(packet, "pose_age", 0.0) > max_age
        )
        is_global_valid = (
            packet.world_T_camera is not None
            and getattr(packet, "transform_valid", False) is True
            and getattr(packet, "is_persistent_world_frame", True) is True
            and not is_stale
        )

        effective_world_T_camera = packet.world_T_camera if is_global_valid else np.eye(4, dtype=np.float64)
        effective_relation_frame = packet.relation_frame
        if effective_relation_frame is None or not is_global_valid:
            effective_relation_frame = RelationReferenceFrame.create("camera", np.eye(4, dtype=np.float64))

        # Invariant 3: Camera-relative fallback may produce local geometry, but must never update
        # global centroids, global bounding boxes, global relations, or global-map state.
        allow_relations = is_global_valid or (packet.localization_mode == LocalizationMode.CAMERA_LOCAL_MODE)

        if allow_relations and relation_tracks:
            observation_geometry = self._build_observation_geometry(
                relation_tracks,
                packet.frame_index,
                intrinsics=packet.camera_intrinsics,
                world_T_camera=effective_world_T_camera,
            )

            if observation_geometry:
                context = FrameContext(
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    intrinsics=packet.camera_intrinsics,
                    world_T_camera=effective_world_T_camera,
                    reference_frame=effective_relation_frame,
                    camera_frame=CameraFrame.from_camera_pose(
                        effective_world_T_camera
                    ),
                    depth_image=depth_m,
                    observation_geometry=observation_geometry,
                )

                raw_evidences = self.registry.compute_all(
                    relation_tracks,
                    context,
                )

                # World model maintains canonical edges only; inverse derivation happens in query layer
                all_evidences.extend(raw_evidences)

                relation_object_ids = {
                    track.object_id
                    for track in relation_tracks
                }

                relation_states = self.relation_state_machine.update(
                    evidences=all_evidences,
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    active_object_ids=relation_object_ids,
                )

        # 2. Graph projection: Update persistent world model (nodes and edges)
        self.graph.update_nodes(
            tracks,
            object_states,
        )

        if getattr(self.registry, "last_contexts", None):
            self.graph.update_contexts(
                self.registry.last_contexts,
                self.registry.last_membership,
            )

        self.graph.update_edges(
            evidences=all_evidences,
            relation_states=relation_states,
        )

        return self.graph
