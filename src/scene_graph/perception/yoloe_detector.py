"""YOLOE Detector wrapper supporting strict text-prompt, prompt-free, and visual-prompt modes."""

import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

try:
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
except ImportError:
    ultralytics = None
    YOLOE = None
    YOLOEVPSegPredictor = None

from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation_source import ObservationProducer
from scene_graph.perception.observation import Observation, encode_mask_rle

logger = logging.getLogger(__name__)

CANONICAL_23_CLASSES: Tuple[str, ...] = (
    "person",
    "chair",
    "desk",
    "table",
    "door",
    "backpack",
    "handbag",
    "mobile phone",
    "water bottle",
    "computer monitor",
    "laptop",
    "keyboard",
    "computer mouse",
    "whiteboard",
    "projector",
    "ceiling fan",
    "pedestal fan",
    "quadcopter drone",
    "drone remote controller",
    "drone battery",
    "robot",
    "toolbox",
    "storage box",
)

CANONICAL_SYNONYMS: Dict[str, str] = {
    "monitor": "computer monitor",
    "screen": "computer monitor",
    "tv": "computer monitor",
    "television": "computer monitor",
    "mouse": "computer mouse",
    "cell phone": "mobile phone",
    "phone": "mobile phone",
    "smartphone": "mobile phone",
    "bottle": "water bottle",
    "drone": "quadcopter drone",
    "uav": "quadcopter drone",
    "quadcopter": "quadcopter drone",
    "remote": "drone remote controller",
    "controller": "drone remote controller",
    "remote controller": "drone remote controller",
    "battery": "drone battery",
    "box": "storage box",
    "crate": "storage box",
    "fan": "ceiling fan",
}

OVERLAPPING_CLASS_PAIRS = (
    frozenset({"desk", "table"}),
    frozenset({"backpack", "handbag"}),
    frozenset({"ceiling fan", "pedestal fan"}),
)


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) of two xyxy bounding boxes."""
    xA = max(float(box1[0]), float(box2[0]))
    yA = max(float(box1[1]), float(box2[1]))
    xB = min(float(box1[2]), float(box2[2]))
    yB = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class PerceptionMode(Enum):
    """Supported perception operating modes for YOLOE."""
    PROMPT_FREE = "prompt_free"
    TEXT_PROMPT = "text"
    VISUAL_PROMPT = "visual"

    @classmethod
    def from_str(cls, val: str) -> "PerceptionMode":
        v = val.lower().strip()
        if v in ("prompt_free", "promptfree", "pf"):
            return cls.PROMPT_FREE
        elif v in ("text", "text_prompt", "prompt"):
            return cls.TEXT_PROMPT
        elif v in ("visual", "visual_prompt", "exemplar"):
            return cls.VISUAL_PROMPT
        raise ValueError(
            f"Unsupported perception mode: {val!r}. "
            "Expected 'prompt_free', 'text', or 'visual'."
        )


class YOLOEDetector(ObservationProducer):
    """Open-vocabulary detector supporting text-prompt, visual-prompt, and prompt-free modes.
    
    Enforces strict class allowlisting, model provenance tracking, canonical vocabulary
    normalization, and per-class confidence thresholding.
    """
    
    def __init__(
        self, 
        model_path: str = "models/yoloe-26s-seg.pt", 
        confidence_threshold: float = 0.25,
        per_class_thresholds: Optional[Dict[str, float]] = None,
        mode: Union[PerceptionMode, str] = PerceptionMode.TEXT_PROMPT,
        text_prompts: Optional[List[str]] = None,
        visual_prompt: Optional[Any] = None,
        device: str = "auto",
        allowed_classes: Optional[Union[List[str], Tuple[str, ...], Set[str]]] = None,
        image_size: Optional[int] = 480,
    ):
        """Initialize the detector with verified provenance and strict vocabulary controls."""
        if YOLOE is None:
            raise ImportError("ultralytics package is required for YOLOEDetector.")
            
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / model_path_obj
            if candidate.exists():
                model_path_obj = candidate
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {model_path}. "
                    "Automatic downloads are disabled."
                )
            
        self.model_path = str(model_path_obj)
        self.confidence_threshold = float(confidence_threshold)
        self.per_class_thresholds = dict(per_class_thresholds) if per_class_thresholds else {}
        self.device = device
        self.image_size = image_size
        self.mode = PerceptionMode.from_str(mode) if isinstance(mode, str) else mode
        
        # Backward compatibility for allowed_classes -> text_prompts
        if text_prompts is None and allowed_classes is not None:
            text_prompts = list(allowed_classes)
            self.mode = PerceptionMode.TEXT_PROMPT

        # Default text prompts to canonical 23 classes if in text mode without explicit list
        if self.mode == PerceptionMode.TEXT_PROMPT and text_prompts is None:
            text_prompts = list(CANONICAL_23_CLASSES)

        if text_prompts and self.mode == PerceptionMode.PROMPT_FREE:
            self.mode = PerceptionMode.TEXT_PROMPT

        # Strictly reject pairing text/visual prompt mode with prompt-free checkpoint (-pf.pt)
        if self.mode in (
            PerceptionMode.TEXT_PROMPT,
            PerceptionMode.VISUAL_PROMPT,
        ) and self.model_path.endswith("-pf.pt"):
            raise ValueError(
                f"{self.mode.value} mode requires a prompted YOLOE checkpoint "
                "(*-seg.pt), not a prompt-free checkpoint (*-seg-pf.pt)."
            )

        self.current_text_prompts: Optional[List[str]] = None
        self.allowed_classes: Optional[Set[str]] = None
        self.current_visual_prompt: Optional[Any] = None
        self.model = YOLOE(self.model_path)

        if self.mode == PerceptionMode.TEXT_PROMPT and text_prompts:
            self.set_text_prompts(text_prompts)
        elif self.mode == PerceptionMode.VISUAL_PROMPT and visual_prompt is not None:
            self.set_visual_prompt(visual_prompt)
        elif allowed_classes is not None:
            self.allowed_classes = {self.normalize_class_name(c) for c in allowed_classes}
                
        # Internal sequential ID counter
        self._obs_counter = 1

        # Provenance logging at initialization
        info = self.get_model_info()
        logger.info(
            f"[YOLOEDetector] Initialized | checkpoint: {info['model_path']} "
            f"(sha256: {info['model_sha256'][:8]}...) | ultralytics: {info['ultralytics_version']} | "
            f"mode: {info['mode']} | classes ({len(self.current_text_prompts or [])}): {self.current_text_prompts} | "
            f"conf: {self.confidence_threshold} | size: {self.image_size} | device: {info['device']}"
        )

    @staticmethod
    def normalize_class_name(name: str) -> str:
        """Map raw detection or synonymous label to authoritative canonical class name."""
        lowered = name.strip().lower()
        return CANONICAL_SYNONYMS.get(lowered, lowered)

    def set_text_prompts(self, prompts: List[str]) -> None:
        """Dynamically configure recognized text prompt classes using model.set_classes()."""
        if not prompts:
            return
        if self.model_path.endswith("-pf.pt"):
            raise ValueError(
                "Cannot configure text prompts on a prompt-free checkpoint (*-seg-pf.pt). "
                "Use a prompted YOLOE checkpoint (*-seg.pt)."
            )
        self.mode = PerceptionMode.TEXT_PROMPT
        self.current_text_prompts = [self.normalize_class_name(p) for p in prompts]
        self.allowed_classes = set(self.current_text_prompts)
        try:
            self.model.set_classes(self.current_text_prompts)
        except (AttributeError, AssertionError, RuntimeError) as e:
            raise RuntimeError(
                f"Model {self.model_path} failed to set runtime text prompts: {e}"
            ) from e

        # Verify model.names accurately reflects the set classes
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            active_names = {v.lower() for v in self.model.names.values()}
            missing = set(self.current_text_prompts) - active_names
            if missing:
                logger.warning(f"Some configured prompts were not found in model.names: {missing}")

    def set_visual_prompt(self, visual_prompt: Any) -> None:
        """Set the visual prompt used by YOLOE visual-prompt inference."""
        if not isinstance(visual_prompt, dict):
            raise TypeError("visual_prompt must be a dict containing 'bboxes' and 'cls'.")
        if "bboxes" not in visual_prompt or "cls" not in visual_prompt:
            raise ValueError("visual_prompt must contain 'bboxes' and 'cls'.")
        bboxes = np.asarray(visual_prompt["bboxes"], dtype=np.float32)
        cls = np.asarray(visual_prompt["cls"], dtype=np.int64)
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError("visual_prompt['bboxes'] must have shape (N, 4).")
        if cls.ndim != 1:
            raise ValueError("visual_prompt['cls'] must have shape (N,).")
        if len(bboxes) != len(cls):
            raise ValueError("visual_prompt['bboxes'] and visual_prompt['cls'] must have same length.")
        if len(bboxes) == 0:
            raise ValueError("visual_prompt must contain at least one example.")
        if np.any(cls < 0):
            raise ValueError("visual_prompt class indices must be non-negative.")
        self.mode = PerceptionMode.VISUAL_PROMPT
        self.current_visual_prompt = {
            "bboxes": bboxes,
            "cls": cls,
        }
        
    def _generate_obs_id(self) -> str:
        """Generate a sequential observation ID (e.g., obs_000001)."""
        obs_id = f"obs_{self._obs_counter:06d}"
        self._obs_counter += 1
        return obs_id
        
    def detect(
        self, 
        packet: FramePacket,
        mode: Optional[Union[PerceptionMode, str]] = None,
        text_prompts: Optional[List[str]] = None,
        visual_prompt: Optional[Any] = None,
    ) -> List[Observation]:
        """Run inference, apply canonical normalization, allowlist gating, and overlap suppression."""
        if mode is not None:
            target_mode = PerceptionMode.from_str(mode) if isinstance(mode, str) else mode
            if target_mode != self.mode:
                self.mode = target_mode

        if text_prompts is not None:
            self.set_text_prompts(text_prompts)

        if visual_prompt is not None:
            self.set_visual_prompt(visual_prompt)

        # Base confidence threshold for model predict pass
        min_conf = self.confidence_threshold
        if self.per_class_thresholds:
            min_conf = min(min_conf, min(self.per_class_thresholds.values()))

        imgsz = self.image_size if self.image_size is not None else 640
        if self.mode == PerceptionMode.VISUAL_PROMPT:
            if self.current_visual_prompt is None:
                raise RuntimeError("VISUAL_PROMPT mode requires a visual prompt.")
            results = self.model.predict(
                source=packet.rgb,
                imgsz=imgsz,
                conf=min_conf,
                visual_prompts=self.current_visual_prompt,
                predictor=YOLOEVPSegPredictor,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=packet.rgb,
                imgsz=imgsz,
                conf=min_conf,
                verbose=False,
            )
        
        observations: List[Observation] = []
        if not results or len(results) == 0:
            return observations
            
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return observations
            
        candidates = []
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if hasattr(self.model, "names") and cls_id in self.model.names:
                raw_name = str(self.model.names[cls_id]).lower()
            else:
                raw_name = f"concept_{cls_id}"
                
            canonical_name = self.normalize_class_name(raw_name)
            
            # 1. Enforce strict allowlist gating
            if self.allowed_classes is not None and canonical_name not in self.allowed_classes:
                continue

            # 2. Enforce per-class confidence thresholding
            class_thresh = self.per_class_thresholds.get(canonical_name, self.confidence_threshold)
            if conf < class_thresh:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            candidates.append({
                "index": i,
                "class_name": canonical_name,
                "confidence": conf,
                "xyxy": xyxy,
            })

        if not candidates:
            return observations

        # 3. Suppress overlapping ambiguous categories (e.g. desk vs table, backpack vs handbag)
        suppressed_indices = set()
        for i_idx, cand_a in enumerate(candidates):
            if i_idx in suppressed_indices:
                continue
            for j_idx in range(i_idx + 1, len(candidates)):
                if j_idx in suppressed_indices:
                    continue
                cand_b = candidates[j_idx]
                pair = frozenset({cand_a["class_name"], cand_b["class_name"]})
                if pair in OVERLAPPING_CLASS_PAIRS:
                    iou = compute_box_iou(cand_a["xyxy"], cand_b["xyxy"])
                    if iou > 0.55:
                        if cand_a["confidence"] >= cand_b["confidence"]:
                            suppressed_indices.add(j_idx)
                        else:
                            suppressed_indices.add(i_idx)
                            break

        for cand_idx, cand in enumerate(candidates):
            if cand_idx in suppressed_indices:
                continue

            box_idx = cand["index"]
            mask_rle = None
            if result.masks is not None and len(result.masks) > box_idx:
                mask_data = result.masks.data[box_idx].cpu().numpy()
                import cv2
                if mask_data.shape != (packet.rgb.shape[0], packet.rgb.shape[1]):
                    mask_np = cv2.resize(
                        mask_data, 
                        (packet.rgb.shape[1], packet.rgb.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_np = mask_data
                    
                mask_np = (mask_np > 0)
                mask_rle = encode_mask_rle(mask_np)
                
            obs = Observation(
                obs_id=self._generate_obs_id(),
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
                class_name=cand["class_name"],
                confidence=cand["confidence"],
                bbox_xyxy=cand["xyxy"],
                mask_rle=mask_rle,
            )
            observations.append(obs)
            
        return observations
        
    def get_model_info(self) -> dict:
        import platform
        import torch
        
        sha256 = "unknown"
        try:
            with open(self.model_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
            
        return {
            "model_path": self.model_path,
            "model_sha256": sha256,
            "ultralytics_version": ultralytics.__version__ if ultralytics else "unknown",
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "device": str(getattr(self.model, "device", self.device)),
            "confidence_threshold": self.confidence_threshold,
            "per_class_thresholds": self.per_class_thresholds,
            "mode": self.mode.value,
            "current_text_prompts": self.current_text_prompts,
            "allowed_classes": sorted(list(self.allowed_classes)) if self.allowed_classes else None,
            "image_size": self.image_size,
        }
