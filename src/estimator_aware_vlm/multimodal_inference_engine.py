"""Unified Multimodal Vision Inference Engine.

Supports:
1. Qwen2.5-VL-3B-Instruct (via llama-server OpenAI-compatible HTTP endpoint)

Provides:
- Strict live execution mode (fail-fast, zero cached fallbacks)
- Preflight health checks and multimodal smoke tests
- View mode abstraction: CROP_ONLY vs. CROP_PLUS_CONTEXT
- Honest wire telemetry: request attempted, HTTP status, response received, parse validity, schema validity, and wall-clock latency
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image

from estimator_aware_vlm.common.paths import get_repo_root

repo_root = get_repo_root()

# Default endpoints from model registry
DEFAULT_QWEN_URL = "http://127.0.0.1:8089/v1/chat/completions"

CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]

VALID_PHYSICAL_SUPPORT = {
    "mounted_to_structure",
    "freestanding_on_surface",
    "wheeled_base",
    "carried_by_agent",
    "unknown",
}

VALID_TRANSPORT_CONTEXT = {
    "stationary_environment",
    "active_carrier",
    "unknown",
}

ONTOLOGY_ALIASES = {
    "IMMOVABLE": "IMMOVABLE",
    "FIXED": "IMMOVABLE",
    "STRUCTURAL": "IMMOVABLE",
    "ARCHITECTURAL": "IMMOVABLE",
    "WALL_MOUNTED": "IMMOVABLE",
    "CEILING_MOUNTED": "IMMOVABLE",
    "MOUNTED": "IMMOVABLE",
    "MOVABLE": "MOVABLE",
    "PORTABLE": "MOVABLE",
    "RELOCATABLE": "MOVABLE",
    "FURNITURE": "MOVABLE",
    "OBJECT": "MOVABLE",
    "FREESTANDING": "MOVABLE",
    "MOBILE": "MOBILE",
    "DYNAMIC": "MOBILE",
    "SELF_PROPELLED": "MOBILE",
    "ROBOT": "MOBILE",
    "AGENT": "MOBILE",
}


def encode_image_base64(path: Path, max_dim: int = 240) -> str:
    """Resize image preserving aspect ratio and return standard base64 string."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def clean_json_response(raw_text: str) -> str:
    """Extract JSON object from raw model completion text."""
    text = raw_text.strip()
    match_fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match_fenced:
        return match_fenced.group(1).strip()
    match_braces = re.search(r"\{.*?\}", text, re.DOTALL)
    if match_braces:
        return match_braces.group(0).strip()
    if "{" in text and "}" not in text:
        return text[text.find("{"):] + "\n}"
    return text


def parse_multimodal_prediction(
    raw_text: str,
    expected_id: Optional[str] = None,
    expected_class: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Parse JSON and strictly validate schema and identity agreement.

    Returns:
        (parsed_dict, parse_valid, schema_valid)
    """
    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
        parse_valid = True
    except Exception as e:
        return (
            {
                "object_id": None,
                "object_class": None,
                "movability": "UNKNOWN",
                "confidence": 0.0,
                "abstain": True,
                "validation_error": f"json_parse_failed: {str(e)}",
            },
            False,
            False,
        )

    if not isinstance(data, dict):
        return (
            {
                "object_id": None,
                "object_class": None,
                "movability": "UNKNOWN",
                "confidence": 0.0,
                "abstain": True,
                "validation_error": "response_not_a_dictionary",
            },
            True,
            False,
        )

    errors: List[str] = []

    # 1. Identity validation (no silent overwriting or substitution)
    resp_id = str(data.get("object_id")).strip() if "object_id" in data and data["object_id"] is not None else None
    resp_cls = str(data.get("object_class")).strip() if "object_class" in data and data["object_class"] is not None else None

    if expected_id is not None:
        if resp_id is None:
            errors.append("missing_object_id")
        elif resp_id != expected_id:
            errors.append(f"object_id_mismatch (got '{resp_id}', expected '{expected_id}')")

    if expected_class is not None:
        if resp_cls is None:
            errors.append("missing_object_class")
        elif resp_cls.lower() != expected_class.lower():
            errors.append(f"object_class_mismatch (got '{resp_cls}', expected '{expected_class}')")

    # 2. Movability validation
    raw_mov = str(data.get("movability", "UNKNOWN")).strip().upper()
    abstain = bool(data.get("abstain", False))

    if raw_mov in ONTOLOGY_ALIASES:
        canonical_mov = ONTOLOGY_ALIASES[raw_mov]
    else:
        canonical_mov = "UNKNOWN"
        abstain = True
        errors.append(f"unrecognized_movability_class: '{raw_mov}'")

    # 3. Confidence validation
    raw_conf = data.get("confidence")
    confidence = None
    if raw_conf is not None:
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (ValueError, TypeError):
            errors.append(f"non_numeric_confidence: '{raw_conf}'")

    # 4. Affordance enum validations
    raw_phys = str(data.get("physical_support", "unknown")).strip().lower()
    if raw_phys not in VALID_PHYSICAL_SUPPORT:
        errors.append(f"invalid_physical_support: '{raw_phys}'")

    raw_trans = str(data.get("transport_context", "unknown")).strip().lower()
    if raw_trans not in VALID_TRANSPORT_CONTEXT:
        errors.append(f"invalid_transport_context: '{raw_trans}'")

    schema_valid = (len(errors) == 0) and ("movability" in data)

    parsed = {
        "object_id": resp_id,
        "object_class": resp_cls,
        "movability": canonical_mov,
        "raw_movability": raw_mov,
        "confidence": confidence,
        "physical_support": raw_phys if raw_phys in VALID_PHYSICAL_SUPPORT else None,
        "transport_context": raw_trans if raw_trans in VALID_TRANSPORT_CONTEXT else None,
        "short_explanation": data.get("short_explanation"),
        "abstain": abstain,
        "validation_error": "; ".join(errors) if errors else None,
    }
    return parsed, parse_valid, schema_valid


@dataclass
class WireInferenceRecord:
    """Full auditable wire execution record for a single model query."""
    physical_instance_id: str
    model_key: str
    view_mode: Literal["CROP_ONLY", "CROP_PLUS_CONTEXT", "MULTI_VIEW"]
    requested_mode: str
    actual_mode: str
    server_endpoint: str
    request_attempted: bool
    http_success: bool
    response_received: bool
    parse_valid: bool
    schema_valid: bool
    prediction_accepted: bool
    application_cache_hit: bool
    cache_hit: bool  # backward compatibility alias
    server_prompt_cache_tokens: Optional[int]
    server_load_duration_ms: Optional[float]
    img_preprocess_ms: float
    network_roundtrip_ms: float
    parsing_validation_ms: float
    total_inference_ms: float
    client_latency_ms: float  # backward compatibility alias
    raw_response: str
    parsed_prediction: str
    confidence: Optional[float]
    physical_support: Optional[str]
    transport_context: Optional[str]
    token_usage: Optional[Dict[str, Any]]
    server_timing: Optional[Dict[str, Any]]
    validation_error: Optional[str]
    transmitted_images: List[Dict[str, Any]]
    source_asset_hashes: List[str]
    model_payload_hashes: List[str]
    input_image_hashes: List[str]  # backward compatibility alias

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultimodalInferenceEngine:
    """Client for executing strict live multimodal inference against Qwen."""

    def __init__(
        self,
        model_key: str = "qwen2_5_vl_3b",
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 240.0,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ):
        self.model_key = model_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve server endpoint and model name
        self.server_url = server_url or os.environ.get("QWEN_SERVER_URL", DEFAULT_QWEN_URL)
        self.model_name = model_name or "Qwen2.5-VL-3B-Instruct"
        self.backend_type = "llama.cpp"

    def check_health(self) -> Dict[str, Any]:
        """Probe model server reachability and verify active loaded model."""
        models_url = self.server_url.replace("/chat/completions", "/models")
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", data.get("data", []))
                loaded_names = [m.get("id", m.get("name", "")) for m in models]
                model_verified = any(
                    (self.model_name and self.model_name.lower() in name.lower())
                    or "qwen2.5-vl" in name.lower()
                    for name in loaded_names
                )
                return {
                    "status": "ONLINE",
                    "url": models_url,
                    "backend": self.backend_type,
                    "loaded_models": loaded_names,
                    "model_verified": model_verified,
                    "health_only": False,
                }
        except Exception as e:
            health_url = self.server_url.replace("/v1/chat/completions", "/health")
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    # Health endpoint reachable, but model identity cannot be confirmed from /health alone
                    return {
                        "status": "ONLINE",
                        "url": health_url,
                        "code": resp.getcode(),
                        "backend": self.backend_type,
                        "model_verified": False,
                        "health_only": True,
                    }
            except Exception:
                return {
                    "status": "OFFLINE",
                    "url": self.server_url,
                    "error": str(e),
                    "model_verified": False,
                    "health_only": False,
                }

    def run_preflight_smoke_test(self, strict_identity: bool = False) -> Dict[str, Any]:
        """Verify model server reachability, model inference capability, and canonical schema response."""
        health = self.check_health()
        if health.get("status") != "ONLINE":
            raise ConnectionError(f"{self.model_key} server is OFFLINE at {self.server_url}: {health}")
        if strict_identity and not health.get("model_verified", False):
            raise RuntimeError(f"{self.model_key} server loaded model could not be verified: {health}")

        # Representative test image (32x32)
        im = Image.new("RGB", (32, 32), color=(200, 200, 200))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are a robotics perception system evaluating physical affordance. "
            "Target Instance ID: preflight_test_001\n"
            "Expected Semantic Class: chair\n"
            "Respond ONLY with a JSON object: "
            "{\"object_id\": \"preflight_test_001\", \"object_class\": \"chair\", "
            "\"movability\": \"MOVABLE\", \"physical_support\": \"freestanding_on_surface\", "
            "\"transport_context\": \"stationary_environment\", \"confidence\": 0.95, "
            "\"abstain\": false, \"short_explanation\": \"preflight validation\"}"
        )

        t0 = time.perf_counter()
        raw = ""
        usage = None
        if self.backend_type == "llama.cpp":
            payload = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                "max_tokens": 150,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                self.server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            raw = res["choices"][0]["message"]["content"]
            usage = res.get("usage")

        lat_ms = (time.perf_counter() - t0) * 1000.0

        # Validate preflight response through strict canonical schema parser
        parsed, parse_valid, schema_valid = parse_multimodal_prediction(
            raw, expected_id="preflight_test_001", expected_class="chair"
        )

        return {
            "status": "PASSED" if parse_valid else "FAILED_PARSE",
            "model_key": self.model_key,
            "latency_ms": round(lat_ms, 2),
            "parse_valid": parse_valid,
            "schema_valid": schema_valid,
            "raw_response": raw.strip(),
            "usage": usage,
        }

    def execute_live_query(
        self,
        physical_instance_id: str,
        semantic_class: Optional[str],
        prompt: str,
        crop_image_path: Path,
        scene_image_path: Optional[Path] = None,
        additional_crop_paths: Optional[List[Path]] = None,
        view_mode: Literal["CROP_ONLY", "CROP_PLUS_CONTEXT", "MULTI_VIEW"] = "CROP_PLUS_CONTEXT",
        image_hashes: Optional[List[str]] = None,
    ) -> WireInferenceRecord:
        """Execute strict live inference over the wire with dual hashing and decomposed latency."""
        request_attempted = True
        http_success = False
        response_received = False
        raw_response = ""
        server_timing = None
        token_usage = None
        validation_error = None

        t_prep_start = time.perf_counter()

        if not crop_image_path.exists():
            raise FileNotFoundError(f"Missing crop image: {crop_image_path}")

        transmitted_images: List[Dict[str, Any]] = []
        source_asset_hashes: List[str] = []
        model_payload_hashes: List[str] = []

        # 1. Process crop view (primary)
        with open(crop_image_path, "rb") as f:
            crop_asset_bytes = f.read()
        crop_asset_sha = hashlib.sha256(crop_asset_bytes).hexdigest()
        source_asset_hashes.append(crop_asset_sha)

        with Image.open(crop_image_path) as im:
            crop_dims = list(im.size)

        crop_b64 = encode_image_base64(crop_image_path, max_dim=240)
        crop_payload_sha = hashlib.sha256(crop_b64.encode("utf-8")).hexdigest()
        model_payload_hashes.append(crop_payload_sha)

        transmitted_images.append({
            "role": "crop_view",
            "view_index": 0,
            "dimensions": crop_dims,
            "source_asset_sha256": crop_asset_sha,
            "model_payload_sha256": crop_payload_sha,
        })
        images_b64 = [crop_b64]

        # 2. Process additional views if MULTI_VIEW
        if view_mode == "MULTI_VIEW" and additional_crop_paths:
            for v_idx, add_path in enumerate(additional_crop_paths, start=1):
                if add_path and add_path.exists():
                    with open(add_path, "rb") as f:
                        add_asset_bytes = f.read()
                    add_asset_sha = hashlib.sha256(add_asset_bytes).hexdigest()
                    source_asset_hashes.append(add_asset_sha)

                    with Image.open(add_path) as im:
                        add_dims = list(im.size)

                    add_b64 = encode_image_base64(add_path, max_dim=240)
                    add_payload_sha = hashlib.sha256(add_b64.encode("utf-8")).hexdigest()
                    model_payload_hashes.append(add_payload_sha)

                    transmitted_images.append({
                        "role": "crop_view",
                        "view_index": v_idx,
                        "dimensions": add_dims,
                        "source_asset_sha256": add_asset_sha,
                        "model_payload_sha256": add_payload_sha,
                    })
                    images_b64.append(add_b64)

        # 3. Process scene context if requested
        if view_mode in ("CROP_PLUS_CONTEXT", "MULTI_VIEW"):
            if scene_image_path and scene_image_path.exists():
                with open(scene_image_path, "rb") as f:
                    scene_asset_bytes = f.read()
                scene_asset_sha = hashlib.sha256(scene_asset_bytes).hexdigest()
                source_asset_hashes.append(scene_asset_sha)

                with Image.open(scene_image_path) as im:
                    scene_dims = list(im.size)

                scene_b64 = encode_image_base64(scene_image_path, max_dim=320)
                scene_payload_sha = hashlib.sha256(scene_b64.encode("utf-8")).hexdigest()
                model_payload_hashes.append(scene_payload_sha)

                transmitted_images.append({
                    "role": "scene_context",
                    "view_index": 0,
                    "dimensions": scene_dims,
                    "source_asset_sha256": scene_asset_sha,
                    "model_payload_sha256": scene_payload_sha,
                })
                images_b64.append(scene_b64)

        t_prep_end = time.perf_counter()
        img_preprocess_ms = (t_prep_end - t_prep_start) * 1000.0

        t_net_start = time.perf_counter()

        try:
            if self.backend_type == "llama.cpp":
                content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                for b64_str in images_b64:
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                    })

                payload = {
                    "messages": [{"role": "user", "content": content_items}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                req = urllib.request.Request(
                    self.server_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    http_success = (resp.getcode() == 200)
                    res = json.loads(resp.read().decode("utf-8"))
                    raw_response = res["choices"][0]["message"]["content"]
                    token_usage = res.get("usage")
                    response_received = True

        except Exception as e:
            validation_error = f"http_transport_error: {str(e)}"
            raw_response = ""

        t_net_end = time.perf_counter()
        network_roundtrip_ms = (t_net_end - t_net_start) * 1000.0

        t_parse_start = time.perf_counter()

        phys_supp = None
        trans_ctx = None

        # Parse and strictly validate response schema
        if response_received and raw_response:
            parsed, parse_valid, schema_valid = parse_multimodal_prediction(
                raw_response, physical_instance_id, semantic_class
            )
            pred = parsed.get("movability", "UNKNOWN")
            abstained = parsed.get("abstain", False) or pred == "UNKNOWN"
            pred_accepted = schema_valid and not abstained and pred in CLASSES
            conf = parsed.get("confidence")
            phys_supp = parsed.get("physical_support")
            trans_ctx = parsed.get("transport_context")
            if not schema_valid:
                validation_error = parsed.get("validation_error")
        else:
            parse_valid = False
            schema_valid = False
            pred_accepted = False
            pred = "UNKNOWN"
            conf = None

        t_parse_end = time.perf_counter()
        parsing_validation_ms = (t_parse_end - t_parse_start) * 1000.0
        total_inference_ms = img_preprocess_ms + network_roundtrip_ms + parsing_validation_ms

        server_prompt_cache = token_usage.get("prompt_tokens") if token_usage else None
        server_load_ms = server_timing.get("load_duration_ms") if server_timing else None

        return WireInferenceRecord(
            physical_instance_id=physical_instance_id,
            model_key=self.model_key,
            view_mode=view_mode,
            requested_mode="LIVE_INFERENCE",
            actual_mode="LIVE_INFERENCE",
            server_endpoint=self.server_url,
            request_attempted=request_attempted,
            http_success=http_success,
            response_received=response_received,
            parse_valid=parse_valid,
            schema_valid=schema_valid,
            prediction_accepted=pred_accepted,
            application_cache_hit=False,
            cache_hit=False,  # strictly zero local cache in live mode
            server_prompt_cache_tokens=server_prompt_cache,
            server_load_duration_ms=server_load_ms,
            img_preprocess_ms=round(img_preprocess_ms, 2),
            network_roundtrip_ms=round(network_roundtrip_ms, 2),
            parsing_validation_ms=round(parsing_validation_ms, 2),
            total_inference_ms=round(total_inference_ms, 2),
            client_latency_ms=round(total_inference_ms, 2),
            raw_response=raw_response,
            parsed_prediction=pred,
            confidence=conf,
            physical_support=phys_supp,
            transport_context=trans_ctx,
            token_usage=token_usage,
            server_timing=server_timing,
            validation_error=validation_error,
            transmitted_images=transmitted_images,
            source_asset_hashes=source_asset_hashes,
            model_payload_hashes=model_payload_hashes,
            input_image_hashes=model_payload_hashes,
        )
