from typing import Optional, Any

from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector, PerceptionMode
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline
from scene_graph.pipeline.async_pipeline import (
    AsyncOnlinePipeline,
    AsyncDetectorWorker,
    FrameHistoryBuffer,
    AsyncDetectionResult,
)


class OnlinePipeline:
    """Online pipeline combining open-vocabulary YOLOE perception with scene-graph processing."""

    def __init__(self, config: SceneGraphConfig, async_mode: Optional[bool] = None, detector: Optional[Any] = None):
        self.config = config
        use_async = async_mode if async_mode is not None else getattr(config.perception, "async_mode", False)
        self.async_mode = use_async

        if self.async_mode:
            self._async_impl = AsyncOnlinePipeline(config, detector=detector)
            self.detector = self._async_impl.detector
            self.core = self._async_impl.core
            self.last_observations = self._async_impl.last_observations
        else:
            self._async_impl = None
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
            self.last_observations = None
            self.new_observations = None

    def update(self, packet: FramePacket):
        """Process one frame and update the scene graph."""
        if self._async_impl is not None:
            graph = self._async_impl.update(packet)
            self.last_observations = self._async_impl.last_observations
            self.new_observations = getattr(self._async_impl, "new_observations", None)
            return graph

        observations = self.detector.detect(packet)
        self.last_observations = observations
        self.new_observations = observations
        return self.core.update(packet, observations)

    def stop(self):
        """Stop background workers if running in async mode."""
        if self._async_impl is not None:
            self._async_impl.stop()

    def get_telemetry(self):
        """Return operational throughput and latency statistics if available."""
        if self._async_impl is not None:
            return self._async_impl.get_telemetry()
        return {}