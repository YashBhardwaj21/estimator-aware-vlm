import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.perception.observation import Observation
from scene_graph.perception.yoloe_detector import YOLOEDetector, PerceptionMode
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline
from scene_graph.relations.context import FrameContext
from scene_graph.relations.inverse_algebra import derive_inverse_evidence
from scene_graph.tracking.track import TrackState


@dataclass
class AsyncDetectionResult:
    frame_index: int
    timestamp: float
    observations: List[Observation]
    inference_time_ms: float


class FrameHistoryBuffer:
    """Thread-safe bounded circular buffer retaining recent FramePackets for historical lifting."""

    def __init__(self, capacity: int = 60):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._packets: deque[FramePacket] = deque(maxlen=capacity)
        self._by_index: Dict[int, FramePacket] = {}

    def append(self, packet: FramePacket) -> None:
        with self._lock:
            if len(self._packets) == self.capacity:
                oldest = self._packets[0]
                self._by_index.pop(oldest.frame_index, None)
            self._packets.append(packet)
            self._by_index[packet.frame_index] = packet

    def get_by_index(self, frame_index: int) -> Optional[FramePacket]:
        with self._lock:
            return self._by_index.get(frame_index)

    def get_by_timestamp(self, timestamp: float, max_dt: float = 0.05) -> Optional[FramePacket]:
        with self._lock:
            best = None
            best_dt = float("inf")
            for p in self._packets:
                dt = abs(p.timestamp - timestamp)
                if dt < best_dt:
                    best_dt = dt
                    best = p
            if best_dt <= max_dt:
                return best
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._packets)


class AsyncDetectorWorker:
    """Dedicated background worker executing object detection asynchronously."""

    def __init__(
        self,
        detector: Any,
        in_queue_size: int = 1,
        out_queue_size: int = 4,
    ):
        self.detector = detector
        self.in_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=in_queue_size)
        self.out_queue: queue.Queue[AsyncDetectionResult] = queue.Queue(maxsize=out_queue_size)
        self.stop_event = threading.Event()
        self.frames_detected = 0
        self.dropped_tasks = 0
        self.last_inference_ms = 0.0

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def submit_frame(self, packet: FramePacket) -> bool:
        """Submit frame for detection without blocking the caller."""
        try:
            self.in_queue.put_nowait(packet)
            return True
        except queue.Full:
            try:
                _ = self.in_queue.get_nowait()
                self.dropped_tasks += 1
            except queue.Empty:
                pass
            try:
                self.in_queue.put_nowait(packet)
                return True
            except queue.Full:
                self.dropped_tasks += 1
                return False

    def get_completed_detections(self) -> List[AsyncDetectionResult]:
        """Drain all completed detection results currently in the output queue."""
        results = []
        while True:
            try:
                res = self.out_queue.get_nowait()
                results.append(res)
            except queue.Empty:
                break
        return results

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if packet is None or self.stop_event.is_set():
                break

            t0 = time.monotonic()
            try:
                observations = self.detector.detect(packet)
            except Exception:
                observations = []
            dt_ms = (time.monotonic() - t0) * 1000.0
            self.last_inference_ms = dt_ms
            self.frames_detected += 1

            result = AsyncDetectionResult(
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
                observations=observations,
                inference_time_ms=dt_ms,
            )

            try:
                self.out_queue.put_nowait(result)
            except queue.Full:
                try:
                    _ = self.out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.out_queue.put_nowait(result)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.in_queue.put_nowait(None)
        except Exception:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


class AsyncOnlinePipeline:
    """Asynchronous pipeline decoupling fast tracking from slower detection."""

    def __init__(
        self,
        config: SceneGraphConfig,
        detector: Optional[Any] = None,
    ):
        self.config = config

        if detector is not None:
            self.detector = detector
        else:
            mode_str = getattr(config.perception, "mode", None)
            prompts = getattr(config.perception, "prompts", None) or getattr(config.perception, "classes", None)
            if mode_str is None:
                mode = PerceptionMode.TEXT_PROMPT if prompts else PerceptionMode.PROMPT_FREE
            else:
                mode = PerceptionMode.from_str(mode_str)

            model_path = getattr(config.perception, "model_path", None) or getattr(config.perception, "model", "models/yoloe-26m-seg.pt")
            image_size = getattr(config.perception, "image_size", 480)

            self.detector = YOLOEDetector(
                model_path=model_path,
                confidence_threshold=config.perception.confidence_threshold,
                per_class_thresholds=getattr(config.perception, "per_class_thresholds", None),
                mode=mode,
                text_prompts=list(prompts) if prompts else None,
                allowed_classes=getattr(config.perception, "allowed_classes", None),
                device=getattr(config.perception, "device", "auto"),
                image_size=image_size,
            )

        self.core = SceneGraphPipeline(config)
        history_size = getattr(config.perception, "frame_history_size", 60)
        self.history_buffer = FrameHistoryBuffer(capacity=history_size)

        max_queue = getattr(config.perception, "max_queue_size", 1)
        self.worker = AsyncDetectorWorker(self.detector, in_queue_size=max_queue, out_queue_size=4)

        self.total_camera_frames = 0
        self.total_detection_frames = 0
        self.last_observations: List[Observation] = []
        self.new_observations: Optional[List[Observation]] = None
        self.last_graph: Optional[TemporalSceneGraph] = None

        self._start_time = time.monotonic()
        self._last_camera_stamp: Optional[float] = None

    def update(self, packet: FramePacket) -> TemporalSceneGraph:
        """Process incoming sensor frame at camera rate without blocking for detector."""
        self.total_camera_frames += 1
        self._last_camera_stamp = packet.timestamp

        self.history_buffer.append(packet)
        self.worker.submit_frame(packet)

        completed_results = self.worker.get_completed_detections()

        if completed_results:
            latest_result = completed_results[-1]
            self.total_detection_frames += 1

            historical_packet = self.history_buffer.get_by_index(latest_result.frame_index)
            if historical_packet is None:
                historical_packet = self.history_buffer.get_by_timestamp(latest_result.timestamp)

            self.last_observations = latest_result.observations
            self.new_observations = latest_result.observations

            graph = self.core.update(
                packet=packet,
                observations=latest_result.observations,
                obs_frame_index=latest_result.frame_index,
                obs_timestamp=latest_result.timestamp,
                obs_packet=historical_packet,
            )
        else:
            self.new_observations = None
            graph = self.core.update(
                packet=packet,
                observations=[],
            )

        self.last_graph = graph
        return graph

    def get_telemetry(self) -> Dict[str, Any]:
        """Return operational throughput and latency statistics."""
        elapsed = max(time.monotonic() - self._start_time, 1e-3)
        return {
            "total_camera_frames": self.total_camera_frames,
            "total_detection_frames": self.total_detection_frames,
            "tracking_rate_hz": self.total_camera_frames / elapsed,
            "detection_rate_hz": self.total_detection_frames / elapsed,
            "last_detector_latency_ms": self.worker.last_inference_ms,
            "dropped_detector_tasks": self.worker.dropped_tasks,
            "history_buffer_length": len(self.history_buffer),
        }

    def stop(self) -> None:
        self.worker.stop()
