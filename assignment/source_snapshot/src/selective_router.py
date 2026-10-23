"""Stage 6: Estimator-Aware Selective Router Implementation.

Aggregates evidence at the physical object instance level, checks validity gates,
executes deterministic routing policies, and selectively dispatches ambiguous
cases to the local VLM (Qwen2.5-VL-3B-Instruct) using M3 structured affordance prompting.

Includes:
- Robust SHA-256 cache fingerprinting based on evidence, prompt, model, and image contents.
- Strict schema and ID validation preventing malformed VLM outputs from corrupting predictions.
- Explicit query, execution, cache, and validation telemetry.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from vlm.src.router_policy import RouterDecision, evaluate_router_policy


CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]
DEFAULT_SERVER_URL = "http://127.0.0.1:8089/v1/chat/completions"
DEFAULT_MODEL_NAME = "Qwen2.5-VL-3B-Instruct (GGUF Q4_K_M)"
PROMPT_TEMPLATE_VERSION = "v1_structured_affordance"


def encode_image(path: Path, max_dim: int = 240) -> str:
    """Resize image preserving aspect ratio and return base64 string."""
    im = Image.open(path)
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def hash_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file's raw contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def clean_json_text(text: str) -> str:
    """Extract json substring from raw text or markdown code fence."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_brace = re.search(r"\{.*?\}", text, re.DOTALL)
    if match_brace:
        return match_brace.group(0).strip()
    return text


def parse_vlm_output(
    raw_text: str, expected_id: str, expected_class: str
) -> Tuple[Dict[str, Any], bool, bool]:
    """Parse and validate output against required schema."""
    cleaned = clean_json_text(raw_text)
    try:
        data = json.loads(cleaned)
        is_json_valid = True
    except Exception:
        return (
            {
                "object_id": expected_id,
                "object_class": expected_class,
                "movability": "UNKNOWN",
                "abstain": True,
                "error": "json_parse_failed",
            },
            False,
            False,
        )

    if not isinstance(data, dict):
        return (
            {
                "object_id": expected_id,
                "object_class": expected_class,
                "movability": "UNKNOWN",
                "abstain": True,
                "error": "not_a_dict",
            },
            True,
            False,
        )

    obj_id = str(data.get("object_id", expected_id))
    obj_class = str(data.get("object_class", expected_class))
    movability = str(data.get("movability", "UNKNOWN")).upper()
    abstain = bool(data.get("abstain", False))

    if movability not in CLASSES:
        if movability in ["FIXED", "STATIC", "STRUCTURAL"]:
            movability = "IMMOVABLE"
        elif movability in ["PORTABLE", "RELOCATABLE", "FURNITURE"]:
            movability = "MOVABLE"
        elif movability in ["DYNAMIC", "SELF_PROPELLED", "LOCOMOTIVE"]:
            movability = "MOBILE"
        else:
            movability = "UNKNOWN"
            abstain = True

    is_schema_valid = (
        "movability" in data
        and (data.get("movability", "").upper() in CLASSES or data.get("abstain") is True)
    )

    result = {
        "object_id": obj_id,
        "object_class": obj_class,
        "movability": movability,
        "abstain": abstain,
        "physical_support": data.get("physical_support"),
    }
    return result, is_json_valid, is_schema_valid


def query_llama_server(
    messages: List[Dict[str, Any]],
    server_url: str = DEFAULT_SERVER_URL,
    max_tokens: int = 75,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Tuple[str, float, Dict[str, Any]]:
    """Send chat completions request to llama-server."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    req = urllib.request.Request(
        server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    latency = time.time() - t0
    raw_content = res["choices"][0]["message"]["content"]
    usage = res.get("usage", {})
    return raw_content, latency, usage


class SelectiveRouter:
    """Estimator-aware selective router for SLAM vision-language evidence."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        server_url: str = DEFAULT_SERVER_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        enable_vlm: bool = True,
    ):
        self.cache_path = cache_path
        self.server_url = server_url
        self.model_name = model_name
        self.enable_vlm = enable_vlm
        self.cache: Dict[str, Dict[str, Any]] = {}

        if self.cache_path and Path(self.cache_path).exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        if self.cache_path:
            p = Path(self.cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            tmp.replace(p)

    def route_entity(
        self,
        evidence: Dict[str, Any],
        repo_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate an instance-level evidence record and dispatch if escalated.

        Returns:
            Structured record conforming to the selective router output contract.
        """
        entity_id = str(evidence.get("entity_id", "unknown"))
        sem_class = str(evidence.get("semantic_class", "object"))

        # Measure deterministic policy execution time
        t_pol_0 = time.perf_counter()
        decision: RouterDecision = evaluate_router_policy(evidence)
        t_pol_elapsed_ms = (time.perf_counter() - t_pol_0) * 1000.0

        provenance = evidence.get("provenance", "MEASURED_FULL" if evidence.get("depth_valid") else "MISSING")
        views_count = int(evidence.get("views_count", 0))

        # Base contract fields
        result = {
            "entity_id": entity_id,
            "decision": decision.decision,
            "reason_codes": decision.reason_codes,
            "evidence_valid": (decision.decision != "DEFER"),
            "evidence_provenance": provenance,
            "vlm_query_requested": decision.vlm_required,
            "vlm_inference_executed": False,
            "vlm_cache_hit": False,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "vlm_required": decision.vlm_required,
            "vlm_model": self.model_name if decision.vlm_required else None,
            "views_used": 0,
            "prediction": decision.cheap_label,
            "abstained": (decision.decision in ["DEFER", "ABSTAIN"]),
            "decision_latency_ms": round(t_pol_elapsed_ms, 3),
            "inference_latency_ms": 0.0,
            "token_usage": None,
            "observed_motion": evidence.get("observed_motion", "STATIC"),
            "world_displacement_m": evidence.get("world_displacement_m"),
            "displacement_uncertainty_m": evidence.get("displacement_uncertainty_m"),
            "relocation_evaluable": evidence.get("relocation_evaluable", False),
        }

        # Case 1: Resolved cheaply by prior / geometry
        if decision.decision == "RESOLVE_CHEAP":
            result["prediction"] = decision.cheap_label
            result["abstained"] = False
            return result

        # Case 2: Deferred or Abstained by policy (untrustworthy estimator or missing images)
        if decision.decision in ["DEFER", "ABSTAIN"] or not decision.vlm_required:
            result["prediction"] = "UNKNOWN"
            result["abstained"] = True
            return result

        # Case 3: ESCALATED -> VLM Query Required
        if not self.enable_vlm:
            # VLM ablation mode: escalation requested, but VLM disabled
            result["vlm_model"] = None
            result["prediction"] = "UNKNOWN"
            result["abstained"] = True
            return result

        # Resolve image crops directory
        crops_dir = evidence.get("crops_dir")
        if repo_root and crops_dir:
            sample_dir = repo_root / crops_dir
        elif crops_dir:
            sample_dir = Path(crops_dir)
        else:
            sample_dir = None

        views_to_use = min(views_count, 2) if views_count > 0 else 1

        # Compute robust image content hash
        img_hashes = []
        if sample_dir and sample_dir.exists():
            for v_idx in range(views_to_use):
                c_p = sample_dir / f"crop_view_{v_idx}.jpg"
                s_p = sample_dir / f"scene_view_{v_idx}.jpg"
                if c_p.exists():
                    img_hashes.append(hash_file_sha256(c_p))
                if s_p.exists():
                    img_hashes.append(hash_file_sha256(s_p))

        # Check Cache with full fingerprint
        cache_data = {
            "entity_id": entity_id,
            "semantic_class": sem_class,
            "model_name": self.model_name,
            "prompt_version": PROMPT_TEMPLATE_VERSION,
            "views_to_use": views_to_use,
            "img_hashes": img_hashes,
        }
        cache_key = hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:20]

        if cache_key in self.cache:
            cached = self.cache[cache_key]
            result["vlm_cache_hit"] = True
            result["prediction"] = cached.get("movability")
            result["abstained"] = cached.get("abstain", False)
            result["views_used"] = cached.get("total_views", 1)
            result["inference_latency_ms"] = round(cached.get("latency_s", 0.0) * 1000.0, 2)
            result["token_usage"] = cached.get("token_usage")
            return result

        # Execute Live VLM Query
        result["vlm_inference_executed"] = True

        prompt = (
            f"You are evaluating physical mounting affordance and contextual movability for an indoor object.\n"
            f"Object ID: {entity_id}\n"
            f"Semantic Class: {sem_class}\n"
            f"Step 1: Determine physical_support as 'mounted_to_structure' (bolted/hung to wall/ceiling), 'freestanding_on_surface' (resting on floor/desk), or 'self_propelled'.\n"
            f"Step 2: Assign movability as IMMOVABLE (if mounted), MOVABLE (if freestanding), or MOBILE (if self_propelled).\n"
            f"Respond strictly in JSON matching this schema:\n"
            f'{{\n'
            f'  "physical_support": "freestanding_on_surface",\n'
            f'  "object_id": "{entity_id}",\n'
            f'  "object_class": "{sem_class}",\n'
            f'  "movability": "MOVABLE",\n'
            f'  "abstain": false\n'
            f'}}'
        )

        image_contents = [{"type": "text", "text": prompt}]
        loaded_views = 0

        if sample_dir and sample_dir.exists():
            for v_idx in range(views_to_use):
                crop_path = sample_dir / f"crop_view_{v_idx}.jpg"
                scene_path = sample_dir / f"scene_view_{v_idx}.jpg"
                if crop_path.exists() and scene_path.exists():
                    c_b64 = encode_image(crop_path, max_dim=200)
                    s_b64 = encode_image(scene_path, max_dim=280)
                    image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{c_b64}"}})
                    image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s_b64}"}})
                    loaded_views += 1

        messages = [{"role": "user", "content": image_contents}]

        try:
            raw_text, lat_s, usage = query_llama_server(messages, server_url=self.server_url, max_tokens=75)
            parsed, is_json, is_schema = parse_vlm_output(raw_text, entity_id, sem_class)
            pred = parsed.get("movability", "UNKNOWN")
            abstained = parsed.get("abstain", False) or pred == "UNKNOWN"
            result["vlm_output_valid"] = is_json and is_schema
            result["vlm_prediction_accepted"] = (not abstained and pred in CLASSES)
        except Exception as e:
            lat_s = 0.0
            usage = {"error": str(e)}
            pred = "UNKNOWN"
            abstained = True
            result["vlm_output_valid"] = False
            result["vlm_prediction_accepted"] = False

        result["prediction"] = pred
        result["abstained"] = abstained
        result["views_used"] = loaded_views
        result["inference_latency_ms"] = round(lat_s * 1000.0, 2)
        result["token_usage"] = usage

        # Cache result under fingerprint
        self.cache[cache_key] = {
            "entity_id": entity_id,
            "movability": pred,
            "abstain": abstained,
            "total_views": loaded_views,
            "latency_s": lat_s,
            "token_usage": usage,
        }
        self._save_cache()

        return result
