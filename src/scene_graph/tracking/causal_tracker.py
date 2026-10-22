import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from scene_graph.config import SceneGraphConfig
from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track, TrackState
from scene_graph.tracking.tracker_base import TrackerInterface
from scene_graph.tracking.track_history import TrackHistory
from scene_graph.tracking.state import KalmanState
from scene_graph.ontology.entity import VisibilityState
from scene_graph.geometry.point_cloud import GeometryStatus


Prediction = Tuple[np.ndarray, np.ndarray]


@dataclass
class TrackerCheckpoint:
    frame_index: int
    timestamp: float
    tracks: Dict[str, Track]
    next_track_id: int
    observations: List[Observation]
    camera_intrinsics: Optional[Any] = None
    world_T_camera: Optional[np.ndarray] = None


class CausalTracker(TrackerInterface):

    POSITION_DIMENSION = 3

    def __init__(self, config: SceneGraphConfig, history: Optional[TrackHistory] = None):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.tracking is None:
            raise ValueError("Tracking configuration is required.")

        self.config = config
        tracking_config = config.tracking

        self.association_threshold_m = float(tracking_config.association.max_distance_m)
        self.size_weight = float(tracking_config.association.size_weight)
        self.max_missing_seconds = float(tracking_config.occlusion.max_missing_seconds)
        self.max_missing_seconds_out_of_view = float(
            getattr(
                tracking_config.occlusion,
                "max_missing_seconds_out_of_view",
                self.max_missing_seconds * 5.0,
            )
        )
        self.min_hits_to_confirm = int(tracking_config.confirmation.min_hits)
        self.q_std = float(tracking_config.process.acceleration_std_mps2)
        self.measurement_std_m = float(tracking_config.measurement.position_std_m)
        self.initial_position_variance = float(tracking_config.initialization.position_variance_m2)
        self.initial_velocity_variance = float(tracking_config.initialization.velocity_variance_m2s2)
        self.chi2_threshold = float(chi2.ppf(tracking_config.gating.chi2_probability, df=self.POSITION_DIMENSION))
        self.history = history

        if self.association_threshold_m <= 0.0:
            raise ValueError("tracking.association.max_distance_m must be positive.")

        if self.size_weight < 0.0:
            raise ValueError("tracking.association.size_weight must be non-negative.")

        if self.max_missing_seconds < 0.0:
            raise ValueError("tracking.occlusion.max_missing_seconds must be non-negative.")

        if self.max_missing_seconds_out_of_view < self.max_missing_seconds:
            raise ValueError("max_missing_seconds_out_of_view cannot be less than max_missing_seconds.")

        if self.min_hits_to_confirm < 1:
            raise ValueError("tracking.confirmation.min_hits must be positive.")

        if self.q_std <= 0.0:
            raise ValueError("tracking.process.acceleration_std_mps2 must be positive.")

        if self.measurement_std_m <= 0.0:
            raise ValueError("tracking.measurement.position_std_m must be positive.")

        if self.initial_position_variance <= 0.0:
            raise ValueError("tracking.initialization.position_variance_m2 must be positive.")

        if self.initial_velocity_variance <= 0.0:
            raise ValueError("tracking.initialization.velocity_variance_m2s2 must be positive.")

        if not np.isfinite(self.chi2_threshold) or self.chi2_threshold <= 0.0:
            raise ValueError("tracking.gating.chi2_probability produced an invalid threshold.")

        self.measurement_covariance = np.eye(self.POSITION_DIMENSION, dtype=np.float64) * self.measurement_std_m ** 2
        self.tracks: Dict[str, Track] = {}
        self.next_track_id = 1
        self._last_frame_index: Optional[int] = None
        self._last_timestamp: Optional[float] = None
        self._checkpoints: deque[TrackerCheckpoint] = deque(maxlen=60)
        self._initial_checkpoint = TrackerCheckpoint(
            frame_index=-1,
            timestamp=0.0,
            tracks={},
            next_track_id=1,
            observations=[],
        )

    def update(
        self,
        observations: List[Observation],
        frame_index: int,
        timestamp: float,
        camera_intrinsics: Optional[Any] = None,
        world_T_camera: Optional[np.ndarray] = None,
    ) -> List[Track]:
        self._validate_frame(frame_index, timestamp)

        predictions = self._predict_tracks(timestamp)
        matched, unmatched_obs, unmatched_tracks = self._associate(observations, predictions)

        for obs_idx, track_id in matched:
            self._update_track(track_id, observations[obs_idx], frame_index, timestamp)

        for track_id in unmatched_tracks:
            self._mark_track_missing(
                track_id,
                frame_index,
                timestamp,
                camera_intrinsics=camera_intrinsics,
                world_T_camera=world_T_camera,
            )

        for obs_idx in unmatched_obs:
            obs = observations[obs_idx]
            if self._is_duplicate_detection(obs):
                continue
            self._create_track(obs, frame_index, timestamp)

        self._cleanup_lost_tracks()

        self._last_frame_index = frame_index
        self._last_timestamp = timestamp

        self._checkpoints.append(
            TrackerCheckpoint(
                frame_index=frame_index,
                timestamp=timestamp,
                tracks=copy.deepcopy(self.tracks),
                next_track_id=self.next_track_id,
                observations=observations,
                camera_intrinsics=camera_intrinsics,
                world_T_camera=world_T_camera.copy() if world_T_camera is not None else None,
            )
        )

        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def update_delayed(
        self,
        observations: List[Observation],
        obs_frame_index: int,
        obs_timestamp: float,
        current_frame_index: int,
        current_timestamp: float,
        camera_intrinsics: Optional[Any] = None,
        world_T_camera: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracker with delayed async observations and roll forward to current timestamp."""
        if obs_frame_index == current_frame_index:
            return self.update(
                observations,
                current_frame_index,
                current_timestamp,
                camera_intrinsics=camera_intrinsics,
                world_T_camera=world_T_camera,
            )

        if obs_frame_index > current_frame_index:
            raise ValueError(
                f"Delayed frame index {obs_frame_index} cannot exceed current frame index {current_frame_index}"
            )

        checkpoint_list = list(self._checkpoints)
        target_idx = None
        for i, cp in enumerate(checkpoint_list):
            if cp.frame_index == obs_frame_index:
                target_idx = i
                break

        if target_idx is None:
            return self.update(
                observations,
                current_frame_index,
                current_timestamp,
                camera_intrinsics=camera_intrinsics,
                world_T_camera=world_T_camera,
            )

        if target_idx == 0:
            prior_state = self._initial_checkpoint
        else:
            prior_state = checkpoint_list[target_idx - 1]

        subsequent_frames = []
        for i in range(target_idx, len(checkpoint_list)):
            cp = checkpoint_list[i]
            ci = cp.camera_intrinsics if cp.camera_intrinsics is not None else camera_intrinsics
            wtc = cp.world_T_camera if cp.world_T_camera is not None else world_T_camera
            if cp.frame_index == obs_frame_index:
                subsequent_frames.append((cp.frame_index, cp.timestamp, observations, ci, wtc))
            else:
                subsequent_frames.append((cp.frame_index, cp.timestamp, cp.observations, ci, wtc))

        self.tracks = copy.deepcopy(prior_state.tracks)
        self.next_track_id = prior_state.next_track_id
        self._last_frame_index = prior_state.frame_index if prior_state.frame_index >= 0 else None
        self._last_timestamp = prior_state.timestamp if prior_state.frame_index >= 0 else None

        self._checkpoints = deque(checkpoint_list[:target_idx], maxlen=self._checkpoints.maxlen)

        for fi, ts, obs, ci, wtc in subsequent_frames:
            self.update(obs, fi, ts, camera_intrinsics=ci, world_T_camera=wtc)

        if self._last_frame_index is not None and current_frame_index > self._last_frame_index:
            return self.update(
                [],
                current_frame_index,
                current_timestamp,
                camera_intrinsics=camera_intrinsics,
                world_T_camera=world_T_camera,
            )

        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def _validate_frame(self, frame_index: int, timestamp: float) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")

        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError(f"Non-monotonic frame index: {frame_index} < {self._last_frame_index}")

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(f"Non-monotonic timestamp: {timestamp} < {self._last_timestamp}")

    def _predict_tracks(self, timestamp: float) -> Dict[str, Prediction]:
        predictions: Dict[str, Prediction] = {}

        for track_id, track in self.tracks.items():
            if track.kalman_state is None:
                pos = track.centroid_world if track.coordinate_frame == "world" else track.centroid_camera
                if pos is None:
                    pos = track._initial_centroid
                position = np.asarray(pos, dtype=np.float64)
                covariance = track.position_covariance_world

                if covariance is None:
                    covariance = self.measurement_covariance.copy()
                else:
                    covariance = np.asarray(covariance, dtype=np.float64)

                predictions[track_id] = (position.copy(), covariance.copy())
                continue

            dt = timestamp - track.last_timestamp

            if dt < 0.0:
                raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

            predicted_state = copy.deepcopy(track.kalman_state)
            predicted_state.predict(dt, self.q_std)

            predictions[track_id] = (
                predicted_state.position,
                predicted_state.position_covariance,
            )
            if getattr(track, "coordinate_frame", "world") == "world":
                track.predicted_centroid_world = predicted_state.position.copy()
                track.predicted_covariance_world = predicted_state.position_covariance.copy()
            else:
                track.predicted_centroid_world = None
                track.predicted_covariance_world = None

        return predictions

    def _associate(self, observations: List[Observation], predictions: Dict[str, Prediction]) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        track_ids = sorted(predictions)

        if not observations or not track_ids:
            return [], list(range(len(observations))), track_ids

        cost_matrix = np.full((len(observations), len(track_ids)), np.inf, dtype=np.float64)

        for obs_idx, observation in enumerate(observations):
            centroid = self._get_observation_centroid(observation)
            obs_frame = self._get_observation_frame(observation)

            if centroid is None or obs_frame == "unknown":
                continue

            observation_size = self._get_observation_size(observation)
            measurement_covariance = self._get_measurement_covariance(observation)

            for track_idx, track_id in enumerate(track_ids):
                track = self.tracks[track_id]

                # INVARIANT 5: Tracking must not associate camera-relative geometry
                # with global geometry as though both use the same coordinate frame.
                if getattr(track, "coordinate_frame", "world") != obs_frame:
                    continue

                predicted_position, predicted_covariance = predictions[track_id]
                innovation = centroid - predicted_position
                distance = float(np.linalg.norm(innovation))

                if not np.isfinite(distance) or distance > self.association_threshold_m:
                    continue

                innovation_covariance = predicted_covariance + measurement_covariance

                if not self._valid_covariance(innovation_covariance):
                    continue

                try:
                    mahalanobis_sq = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
                except np.linalg.LinAlgError:
                    continue

                if not np.isfinite(mahalanobis_sq) or mahalanobis_sq > self.chi2_threshold:
                    continue

                cost = mahalanobis_sq

                if observation_size is not None and track.size_world is not None and self.size_weight > 0.0:
                    size_scale = max(float(np.linalg.norm(track.size_world)), np.finfo(float).eps)
                    normalized_size_error = float(np.linalg.norm(observation_size - track.size_world) / size_scale)
                    cost += self.size_weight * normalized_size_error

                # Soft semantic consistency cost (class name is metadata, not physical identity)
                if observation.class_name != track.class_name:
                    prob = track.label_belief.get(observation.class_name, 0.0) if hasattr(track, "label_belief") else 0.0
                    cost += 1.5 * (1.0 - prob)

                if np.isfinite(cost):
                    cost_matrix[obs_idx, track_idx] = cost

        finite_mask = np.isfinite(cost_matrix)

        if not finite_mask.any():
            return [], list(range(len(observations))), track_ids

        feasible_observations = np.flatnonzero(finite_mask.any(axis=1))
        feasible_tracks = np.flatnonzero(finite_mask.any(axis=0))

        feasible_cost_matrix = cost_matrix[np.ix_(feasible_observations, feasible_tracks)]

        feasible_finite_mask = np.isfinite(feasible_cost_matrix)
        if feasible_finite_mask.any():
            gate_penalty = max(2.0 * float(np.max(feasible_cost_matrix[feasible_finite_mask])), 1e5)
        else:
            gate_penalty = 1e5

        safe_cost_matrix = feasible_cost_matrix.copy()
        safe_cost_matrix[~feasible_finite_mask] = gate_penalty

        row_indices, col_indices = linear_sum_assignment(safe_cost_matrix)

        matched: List[Tuple[int, str]] = []
        unmatched_obs = set(range(len(observations)))
        unmatched_tracks = set(track_ids)

        for local_row, local_col in zip(row_indices, col_indices):
            row = int(feasible_observations[local_row])
            col = int(feasible_tracks[local_col])

            if not np.isfinite(cost_matrix[row, col]):
                continue

            track_id = track_ids[col]
            matched.append((row, track_id))
            unmatched_obs.remove(row)
            unmatched_tracks.remove(track_id)

        return matched, sorted(unmatched_obs), sorted(unmatched_tracks)

    def _update_track(self, track_id: str, observation: Observation, frame_index: int, timestamp: float) -> None:
        track = self.tracks[track_id]
        obs_frame = self._get_observation_frame(observation)

        # Invariant 5: Tracking must not associate camera-relative geometry with global geometry
        if getattr(track, "coordinate_frame", "world") != obs_frame:
            return

        old_state = track.state
        measurement = self._get_observation_centroid(observation)

        if measurement is None:
            return

        measurement_covariance = self._get_measurement_covariance(observation)

        if track.kalman_state is None:
            init_cov_pos = (
                measurement_covariance
                if measurement_covariance is not None
                else self.initial_position_variance
            )
            track.kalman_state = KalmanState(
                measurement,
                initial_cov_pos=init_cov_pos,
                initial_cov_vel=self.initial_velocity_variance,
            )
        else:
            dt = timestamp - track.last_timestamp

            if dt < 0.0:
                raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

            updated_state = copy.deepcopy(track.kalman_state)
            updated_state.predict(dt, self.q_std)
            updated_state.update(measurement, measurement_covariance)
            track.kalman_state = updated_state

        observation_size = self._get_observation_size(observation)

        if observation_size is not None:
            if obs_frame == "world":
                if track.size_world is None or track.observation_count <= 0:
                    track.size_world = observation_size.copy()
                else:
                    count = track.observation_count
                    track.size_world = ((count * track.size_world) + observation_size) / (count + 1)
            else:
                track.size_world = None

        geom = getattr(observation, "object_geometry", None)
        if obs_frame == "camera":
            track.centroid_camera = np.copy(measurement)
        else:
            track.centroid_camera = getattr(geom, "centroid_camera", None)

        track.pose_timestamp = getattr(geom, "pose_timestamp", None)
        track.last_observed_frame = frame_index
        track.last_timestamp = timestamp
        track.last_observed_timestamp = timestamp
        track.predicted_centroid_world = None
        track.predicted_covariance_world = None
        track.observation_count += 1
        track.missing_count = 0
        track.detection_confidence = float(observation.confidence)
        track.is_in_frustum = True
        track.visibility_state = VisibilityState.VISIBLE

        if hasattr(track, "update_label_belief"):
            track.update_label_belief(observation.class_name, observation.confidence)

        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / max(age, 1)
        track.recent_observations.append(observation)

        if track.observation_count >= self.min_hits_to_confirm:
            track.state = TrackState.ACTIVE
        elif track.state == TrackState.TEMPORARILY_UNOBSERVED:
            track.state = TrackState.CANDIDATE

        if self.history is not None:
            if old_state != track.state:
                self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, "observation_matched")

            self.history.record_observation(track_id, observation)

    def _mark_track_missing(
        self,
        track_id: str,
        frame_index: int,
        timestamp: float,
        camera_intrinsics: Optional[Any] = None,
        world_T_camera: Optional[np.ndarray] = None,
    ) -> None:
        track = self.tracks[track_id]
        old_state = track.state

        track.missing_count += 1

        age = frame_index - track.first_observed_frame + 1
        track.track_observation_ratio = track.observation_count / max(age, 1)

        missing_seconds = timestamp - track.last_timestamp

        if missing_seconds < 0.0:
            raise ValueError(f"Non-monotonic timestamp for track {track_id}: {timestamp} < {track.last_timestamp}")

        in_frustum = True
        if camera_intrinsics is not None and world_T_camera is not None:
            c_world = track.centroid_world
            if c_world is not None:
                in_frustum = bool(camera_intrinsics.is_world_point_in_frustum(c_world, world_T_camera))
        track.is_in_frustum = in_frustum
        track.visibility_state = VisibilityState.OCCLUDED if in_frustum else VisibilityState.OUT_OF_VIEW

        max_missing = self.max_missing_seconds if in_frustum else self.max_missing_seconds_out_of_view

        if missing_seconds >= max_missing:
            track.state = TrackState.LOST
            track.visibility_state = VisibilityState.UNKNOWN
        elif track.state in (TrackState.ACTIVE, TrackState.CANDIDATE):
            track.state = TrackState.TEMPORARILY_UNOBSERVED

        if self.history is not None and old_state != track.state:
            reason = "missing_threshold" if in_frustum else "out_of_view_threshold"
            self.history.record_transition(track_id, frame_index, old_state.value, track.state.value, reason)

    def _create_track(self, observation: Observation, frame_index: int, timestamp: float) -> None:
        measurement = self._get_observation_centroid(observation)
        obs_frame = self._get_observation_frame(observation)

        if measurement is None:
            return

        observation_buffer_size = self.config.tracking.history.observation_buffer_size
        observation_size = self._get_observation_size(observation)
        size_world = observation_size.copy() if (observation_size is not None and obs_frame == "world") else None

        track_id = f"track_{self.next_track_id:04d}"
        self.next_track_id += 1

        measurement_covariance = self._get_measurement_covariance(observation)
        init_cov_pos = (
            measurement_covariance
            if measurement_covariance is not None
            else self.initial_position_variance
        )

        geom = getattr(observation, "object_geometry", None)
        cam_centroid = np.copy(measurement) if obs_frame == "camera" else getattr(geom, "centroid_camera", None)

        track = Track(
            object_id=track_id,
            class_name=observation.class_name,
            state=TrackState.CANDIDATE,
            _initial_centroid=np.copy(measurement),
            last_observed_frame=frame_index,
            first_observed_frame=frame_index,
            observation_count=1,
            missing_count=0,
            detection_confidence=float(observation.confidence),
            track_observation_ratio=1.0,
            last_timestamp=timestamp,
            recent_observations=deque([observation], maxlen=observation_buffer_size),
            kalman_state=KalmanState(
                measurement,
                initial_cov_pos=init_cov_pos,
                initial_cov_vel=self.initial_velocity_variance,
            ),
            size_world=size_world,
            last_observed_timestamp=timestamp,
            is_in_frustum=True,
            visibility_state=VisibilityState.VISIBLE,
            coordinate_frame=obs_frame,
            centroid_camera=cam_centroid,
            pose_timestamp=getattr(geom, "pose_timestamp", None),
        )

        if self.min_hits_to_confirm <= 1:
            track.state = TrackState.ACTIVE

        self.tracks[track_id] = track

        if self.history is not None:
            self.history.record_transition(track_id, frame_index, "none", track.state.value, "new_track")
            self.history.record_observation(track_id, observation)

    def _is_duplicate_detection(self, observation: Observation) -> bool:
        """Check if an unmatched observation spatially duplicates an existing tracked object."""
        meas = self._get_observation_centroid(observation)
        obs_frame = self._get_observation_frame(observation)
        if meas is None or obs_frame == "unknown":
            return True

        dup_thresh = self.association_threshold_m * 0.75
        for track in self.tracks.values():
            if track.state not in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED, TrackState.CANDIDATE):
                continue
            if getattr(track, "coordinate_frame", "world") != obs_frame:
                continue
            t_center = track.centroid_world if obs_frame == "world" else track.centroid_camera
            if t_center is None:
                continue
            dist = float(np.linalg.norm(meas - t_center))
            if dist < dup_thresh:
                belief = getattr(track, "label_belief", {})
                if observation.class_name == track.class_name or belief.get(observation.class_name, 0.0) > 0.3:
                    return True
        return False

    def _cleanup_lost_tracks(self) -> None:
        lost_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.state == TrackState.LOST
        ]

        for track_id in lost_track_ids:
            del self.tracks[track_id]

    def _get_measurement_covariance(self, observation: Observation) -> np.ndarray:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is not None:
            covariance = getattr(geometry, "position_covariance_world", None)

            if covariance is not None:
                covariance = np.asarray(covariance, dtype=np.float64)

                if covariance.shape == (3, 3) and self._valid_covariance(covariance):
                    return self._symmetrize(covariance)

        return self.measurement_covariance.copy()

    @staticmethod
    def _get_observation_frame(observation: Observation) -> str:
        geometry = getattr(observation, "object_geometry", None)
        if geometry is None:
            return "unknown"
        if getattr(geometry, "status", None) == GeometryStatus.VALID and getattr(geometry, "centroid_world", None) is not None:
            return "world"
        if getattr(geometry, "centroid_camera", None) is not None:
            return "camera"
        return getattr(geometry, "coordinate_frame", getattr(geometry, "frame", "unknown"))

    @staticmethod
    def _get_observation_centroid(observation: Observation) -> np.ndarray | None:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is None:
            return None

        if getattr(geometry, "status", None) not in (None, GeometryStatus.VALID, GeometryStatus.STALE_POSE, GeometryStatus.NO_POSE):
            return None

        centroid = getattr(geometry, "centroid_world", None)
        if centroid is None:
            # Fallback to centroid_camera during local fallback so tracks and relations remain active
            centroid = getattr(geometry, "centroid_camera", None)

        if centroid is None:
            return None

        centroid = np.asarray(centroid, dtype=np.float64)

        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            return None

        return centroid

    @staticmethod
    def _get_observation_size(observation: Observation) -> np.ndarray | None:
        geometry = getattr(observation, "object_geometry", None)

        if geometry is None:
            return None

        if getattr(geometry, "status", None) not in (None, GeometryStatus.VALID, GeometryStatus.STALE_POSE, GeometryStatus.NO_POSE):
            return None

        minimum = getattr(geometry, "bbox_min_world", None)
        maximum = getattr(geometry, "bbox_max_world", None)

        if minimum is None or maximum is None:
            pts_cam = getattr(geometry, "points_camera", None)
            if pts_cam is not None and len(pts_cam) > 0:
                minimum = np.min(pts_cam, axis=0)
                maximum = np.max(pts_cam, axis=0)
            else:
                return None

        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)

        if minimum.shape != (3,) or maximum.shape != (3,):
            return None

        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            return None

        size = maximum - minimum

        if not np.isfinite(size).all() or np.any(size < 0.0):
            return None

        return size

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _valid_covariance(cls, covariance: np.ndarray) -> bool:
        covariance = np.asarray(covariance, dtype=np.float64)

        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            return False

        if not np.isfinite(covariance).all():
            return False

        covariance = cls._symmetrize(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)

        return np.isfinite(eigenvalues).all() and np.min(eigenvalues) >= -1e-10