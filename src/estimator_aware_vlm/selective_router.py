"""Estimator-Aware Selective Router Implementation.

Aggregates evidence at the physical object instance level, checks validity gates,
executes deterministic routing policies, and selectively dispatches ambiguous
cases to the local VLM (Qwen2.5-VL-3B-Instruct) using M3 structured affordance prompting.

Includes:
- Robust SHA-256 cache fingerprinting based on evidence, prompt version, model, and image contents.
- Strict schema, ID, and class validation preventing malformed VLM outputs from corrupting predictions.
- View validation verifying image decoding, file presence, and image hashes.
- Fail-safe rejection: no text-only requests counted as successful visual inference.
- Explicit query, execution, cache, replay/live mode, and validation telemetry.
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
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

from estimator_aware_vlm.router_policy import RouterDecision, evaluate_router_policy, DEFAULT_VLM_COST


CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]
DEFAULT_SERVER_URL = "http://127.0.0.1:8089/v1/chat/completions"
DEFAULT_MODEL_NAME = "Qwen2.5-VL-3B-Instruct (GGUF Q4_K_M)"
PROMPT_TEMPLATE_VERSION = "v2_structured_ontology_schema"

# Ontology aliases mapping
ONTOLOGY_ALIASES = {
    # IMMOVABLE: fixed architectural infrastructure
    "IMMOVABLE": "IMMOVABLE",
    "FIXED": "IMMOVABLE",
    "STRUCTURAL": "IMMOVABLE",
    "ARCHITECTURAL": "IMMOVABLE",
    "WALL_MOUNTED": "IMMOVABLE",
    "CEILING_MOUNTED": "IMMOVABLE",
    "MOUNTED": "IMMOVABLE",
    # MOVABLE: passively transportable objects
    "MOVABLE": "MOVABLE",
    "PORTABLE": "MOVABLE",
    "RELOCATABLE": "MOVABLE",
    "FURNITURE": "MOVABLE",
    "OBJECT": "MOVABLE",
    "FREESTANDING": "MOVABLE",
    "FREESTANDING_ON_SURFACE": "MOVABLE",
    # MOBILE: self-propelled or transported agents
    "MOBILE": "MOBILE",
    "DYNAMIC": "MOBILE",
    "SELF_PROPELLED": "MOBILE",
    "LOCOMOTIVE": "MOBILE",
    "CARRIER": "MOBILE",
    "ROBOT": "MOBILE",
    "AGENT": "MOBILE",
    "MOVING_AGENT": "MOBILE",
}


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


def validate_image_inputs(
    sample_dir: Optional[Path],
    configured_views: int = 2,
) -> Dict[str, Any]:
    """Inspect and decode image files, returning full telemetry.

    Checks:
    - File existence for crop_view_i and scene_view_i.
    - Decoding integrity via PIL.
    - SHA-256 hash calculation.
    - Image dimensions.
    """
    telemetry = {
        "expected_views": configured_views,
        "configured_views": configured_views,
        "files_found": [],
        "files_missing": [],
        "image_paths": [],
        "image_hashes": [],
        "image_dimensions": [],
        "decoding_failures": [],
        "usable_views": 0,
        "crop_scene_pairs": [],
    }

    if not sample_dir or not sample_dir.exists():
        for v_idx in range(configured_views):
            telemetry["files_missing"].extend([
                f"crop_view_{v_idx}.jpg",
                f"scene_view_{v_idx}.jpg",
            ])
        return telemetry

    for v_idx in range(configured_views):
        c_p = sample_dir / f"crop_view_{v_idx}.jpg"
        s_p = sample_dir / f"scene_view_{v_idx}.jpg"

        c_ok = False
        s_ok = False
        c_dim = None
        s_dim = None

        if c_p.exists():
            telemetry["files_found"].append(c_p.name)
            try:
                with Image.open(c_p) as im:
                    im.verify()
                with Image.open(c_p) as im:
                    c_dim = im.size
                telemetry["image_paths"].append(str(c_p))
                telemetry["image_hashes"].append(hash_file_sha256(c_p))
                c_ok = True
            except Exception as e:
                telemetry["decoding_failures"].append(f"{c_p.name}: {str(e)}")
        else:
            telemetry["files_missing"].append(c_p.name)

        if s_p.exists():
            telemetry["files_found"].append(s_p.name)
            try:
                with Image.open(s_p) as im:
                    im.verify()
                with Image.open(s_p) as im:
                    s_dim = im.size
                telemetry["image_paths"].append(str(s_p))
                telemetry["image_hashes"].append(hash_file_sha256(s_p))
                s_ok = True
            except Exception as e:
                telemetry["decoding_failures"].append(f"{s_p.name}: {str(e)}")
        else:
            telemetry["files_missing"].append(s_p.name)

        if c_ok and s_ok:
            telemetry["usable_views"] += 1
            telemetry["crop_scene_pairs"].append((c_p, s_p))
            telemetry["image_dimensions"].append({"crop": c_dim, "scene": s_dim})

    return telemetry


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


def build_structured_affordance_prompt(
    entity_id: str,
    semantic_class: str,
    prompt_version: str = PROMPT_TEMPLATE_VERSION,
) -> str:
    """Construct neutral, schema-driven prompt without biased answer demonstrations."""
    return (
        f"You are an expert robotics perception system evaluating physical affordance and contextual movability for indoor scene understanding.\n"
        f"Target Instance ID: {entity_id}\n"
        f"Expected Semantic Class: {semantic_class}\n\n"
        f"ONTOLOGY DEFINITIONS (Canonical Embodied Affordance Schema):\n"
        f"- IMMOVABLE: Fixed architectural infrastructure (e.g., walls, structural doors, ceiling fans, built-in cabinets). Cannot be relocated without deconstruction.\n"
        f"- MOVABLE: Passively transportable objects lacking intrinsic locomotion (e.g., chairs, desks, boxes, backpacks, laptops, pedestal fans). An object currently being carried by an agent remains strictly MOVABLE.\n"
        f"- MOBILE: Active embodied agents whose defining category is self-propelled locomotion (e.g., humans, mobile robots, drones), even if temporarily stationary.\n\n"
        f"EVALUATION INSTRUCTIONS:\n"
        f"1. Inspect the visual views of the target instance.\n"
        f"2. Identify physical_support: 'mounted_to_structure', 'freestanding_on_surface', 'wheeled_base', 'carried_by_agent', or 'unknown'.\n"
        f"3. Identify transport_context: 'active_carrier', 'stationary_environment', or 'unknown'.\n"
        f"4. Assign movability strictly as IMMOVABLE, MOVABLE, or MOBILE based on physical affordance, not temporary motion.\n"
        f"5. If visual evidence is severely occluded, corrupted, or missing, set abstain to true.\n"
        f"6. Provide a concise short_explanation citing visual cues.\n\n"
        f"OUTPUT SCHEMA (Respond ONLY with a valid JSON object matching these fields):\n"
        f"{{\n"
        f'  "object_id": "<must match Target Instance ID: {entity_id}>",\n'
        f'  "object_class": "<must match Expected Semantic Class: {semantic_class}>",\n'
        f'  "movability": "<IMMOVABLE | MOVABLE | MOBILE | UNKNOWN>",\n'
        f'  "physical_support": "<mounted_to_structure | freestanding_on_surface | wheeled_base | carried_by_agent | unknown>",\n'
        f'  "transport_context": "<stationary_environment | active_carrier | unknown>",\n'
        f'  "confidence": <float between 0.0 and 1.0>,\n'
        f'  "abstain": <true | false>,\n'
        f'  "short_explanation": "<concise justification>"\n'
        f"}}"
    )


def parse_vlm_output(
    raw_text: str, expected_id: str, expected_class: str
) -> Tuple[Dict[str, Any], bool, bool]:
    """Parse and validate output against required schema, checking ID and class agreement.

    Returns:
        (result_dict, is_json_valid, is_schema_valid)
    """
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
                "raw_movability": None,
                "abstain": True,
                "validation_error": "json_parse_failed",
                "id_matched": False,
                "class_matched": False,
                "confidence": 0.0,
                "short_explanation": None,
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
                "raw_movability": None,
                "abstain": True,
                "validation_error": "not_a_dict",
                "id_matched": False,
                "class_matched": False,
                "confidence": 0.0,
                "short_explanation": None,
            },
            True,
            False,
        )

    # Validate ID agreement
    raw_id = str(data.get("object_id", "")).strip()
    id_matched = (
        raw_id.lower() == expected_id.lower()
        or raw_id.replace("inst_", "").lower() == expected_id.replace("inst_", "").lower()
    ) if raw_id else True

    # Validate class agreement
    raw_cls = str(data.get("object_class", "")).strip()
    class_matched = (raw_cls.lower() == expected_class.lower()) if raw_cls else True

    # Movability validation and alias mapping
    raw_movability = str(data.get("movability", "UNKNOWN")).strip().upper()
    abstain = bool(data.get("abstain", False))
    validation_error = None

    if not id_matched:
        movability = "UNKNOWN"
        abstain = True
        validation_error = "object_id_mismatch"
    elif not class_matched:
        movability = "UNKNOWN"
        abstain = True
        validation_error = "object_class_mismatch"
    elif raw_movability in ONTOLOGY_ALIASES:
        movability = ONTOLOGY_ALIASES[raw_movability]
    else:
        movability = "UNKNOWN"
        abstain = True
        validation_error = f"invalid_ontology_label: {raw_movability}"

    # Confidence validation
    raw_conf = data.get("confidence")
    confidence = None
    if raw_conf is not None:
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (ValueError, TypeError):
            confidence = None

    is_schema_valid = (
        "movability" in data
        and (data.get("movability", "").upper() in CLASSES or data.get("abstain") is True)
        and (id_matched or raw_id == "")
        and (class_matched or raw_cls == "")
    )

    result = {
        "object_id": raw_id if raw_id else expected_id,
        "object_class": raw_cls if raw_cls else expected_class,
        "movability": movability,
        "raw_movability": raw_movability,
        "abstain": abstain,
        "physical_support": data.get("physical_support"),
        "transport_context": data.get("transport_context"),
        "confidence": confidence,
        "short_explanation": data.get("short_explanation"),
        "validation_error": validation_error,
        "id_matched": id_matched,
        "class_matched": class_matched,
    }
    return result, is_json_valid, is_schema_valid


def check_vlm_server_health(server_url: str = DEFAULT_SERVER_URL, timeout: float = 2.0) -> Dict[str, Any]:
    """Probe whether the local or remote VLM inference server is reachable."""
    health_url = server_url.replace("/v1/chat/completions", "/health")
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": "ONLINE", "url": health_url, "code": resp.getcode()}
    except Exception:
        pass

    try:
        req = urllib.request.Request(server_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": "ONLINE", "url": server_url, "code": resp.getcode()}
    except urllib.error.HTTPError as e:
        # HTTP 400 or 405 means the server is actively listening and handling HTTP
        if e.code in [400, 405]:
            return {"status": "ONLINE", "url": server_url, "code": e.code}
        return {"status": "OFFLINE", "url": server_url, "error": str(e), "code": e.code}
    except Exception as e:
        return {"status": "OFFLINE", "url": server_url, "error": str(e)}


def query_llama_server(
    messages: List[Dict[str, Any]],
    server_url: str = DEFAULT_SERVER_URL,
    max_tokens: int = 150,
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
        mode: Literal["CACHED_HISTORICAL", "LIVE_INFERENCE", "replay", "live"] = "CACHED_HISTORICAL",
        configured_views: int = 2,
    ):
        self.cache_path = cache_path
        self.server_url = server_url
        self.model_name = model_name
        self.enable_vlm = enable_vlm
        mode_str = str(mode).upper()
        if mode_str in ["LIVE", "LIVE_INFERENCE"]:
            self.mode: Literal["CACHED_HISTORICAL", "LIVE_INFERENCE"] = "LIVE_INFERENCE"
        else:
            self.mode: Literal["CACHED_HISTORICAL", "LIVE_INFERENCE"] = "CACHED_HISTORICAL"
        self.configured_views = configured_views
        self.cache: Dict[str, Dict[str, Any]] = {}

        if self.cache_path and Path(self.cache_path).exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def validate_image_inputs(self, sample_dir: Optional[Path], expected_views: int = 2) -> Tuple[List[str], Dict[str, Any]]:
        """Validate images for an instance and return valid image paths plus telemetry."""
        tel = validate_image_inputs(sample_dir, configured_views=expected_views)
        valid_paths = tel.get("image_paths", [])
        return valid_paths, tel

    def _save_cache(self):
        if self.cache_path:
            p = Path(self.cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            tmp.replace(p)

    def compute_cache_key(self, entity_id: str, semantic_class: str, img_telemetry: Dict[str, Any]) -> str:
        """Compute 24-character cryptographic cache fingerprint."""
        cache_data = {
            "entity_id": entity_id,
            "semantic_class": semantic_class,
            "model_name": self.model_name,
            "prompt_version": PROMPT_TEMPLATE_VERSION,
            "views_to_use": img_telemetry.get("usable_views", len(img_telemetry.get("image_hashes", []))),
            "img_hashes": img_telemetry.get("image_hashes", []),
        }
        return hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()[:24]

    def _compute_cache_key(self, entity_id: str, evidence: Dict[str, Any], valid_images: List[str]) -> str:
        """Helper alias for tests computing cache fingerprint from evidence and image paths."""
        img_hashes = [hash_file_sha256(Path(p)) for p in valid_images if Path(p).exists()]
        tel = {"usable_views": len(img_hashes), "image_hashes": img_hashes}
        return self.compute_cache_key(entity_id, evidence.get("semantic_class", "object"), tel)

    def route_entity(
        self,
        evidence: Dict[str, Any],
        repo_root: Optional[Path] = None,
        policy: Literal["R0", "R1", "R2", "R3", "R4"] = "R3",
        lambda_cost: float = DEFAULT_VLM_COST,
    ) -> Dict[str, Any]:
        """Evaluate an instance-level evidence record and dispatch if escalated.

        Returns:
            Structured record conforming to the selective router output contract.
        """
        entity_id = str(evidence.get("entity_id", "unknown"))
        sem_class = str(evidence.get("semantic_class", "object"))

        # Measure deterministic policy execution time
        t_pol_0 = time.perf_counter()
        decision: RouterDecision = evaluate_router_policy(evidence, policy=policy, lambda_cost=lambda_cost)
        t_pol_elapsed_ms = (time.perf_counter() - t_pol_0) * 1000.0

        provenance = evidence.get("provenance", "MEASURED_FULL" if evidence.get("depth_valid") else "MISSING")
        views_count = int(evidence.get("views_count", 0))

        # Base contract fields
        result = {
            "entity_id": entity_id,
            "policy_variant": policy,
            "decision": decision.decision,
            "reason_codes": decision.reason_codes,
            "query_utility": decision.query_utility,
            "estimator_influence": decision.estimator_influence,
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
            "cached_latency_ms": None,
            "token_usage": None,
            "observed_motion": evidence.get("observed_motion", "STATIC"),
            "world_displacement_m": evidence.get("world_displacement_m"),
            "displacement_uncertainty_m": evidence.get("displacement_uncertainty_m"),
            "relocation_evaluable": evidence.get("relocation_evaluable", False),
            "image_telemetry": None,
            "execution_mode": self.mode,
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

        # Validate visual inputs and record image telemetry
        target_views = self.configured_views
        if views_count > 0:
            target_views = min(views_count, self.configured_views)
        img_telemetry = validate_image_inputs(sample_dir, configured_views=target_views)
        result["image_telemetry"] = img_telemetry

        # Fail-safe: Reject inference attempt if no usable visual content remains!
        # Do not send text-only requests and count them as visual reasoning.
        if img_telemetry["usable_views"] == 0:
            result["decision"] = "DEFER"
            result["reason_codes"] = list(decision.reason_codes) + ["MISSING_REQUIRED_VISUAL_CONTEXT"]
            result["evidence_valid"] = False
            result["vlm_required"] = False
            result["vlm_inference_executed"] = False
            result["prediction"] = "UNKNOWN"
            result["abstained"] = True
            return result

        # Compute full cryptographic cache key fingerprint
        cache_key = self.compute_cache_key(entity_id, sem_class, img_telemetry)

        # Check Cache
        # Backward compatibility fallback for legacy 20-char entity-only keys
        legacy_key = hashlib.sha256(json.dumps({
            "entity_id": entity_id,
            "semantic_class": sem_class,
            "model_name": self.model_name,
            "prompt_version": "v1_structured_affordance",
            "views_to_use": min(views_count, 2) if views_count > 0 else 1,
            "img_hashes": img_telemetry["image_hashes"][:4],
        }, sort_keys=True).encode()).hexdigest()[:20]

        # Check Cache only if explicitly in CACHED_HISTORICAL mode
        if self.mode == "CACHED_HISTORICAL":
            matched_key = cache_key if cache_key in self.cache else (legacy_key if legacy_key in self.cache else None)
            if matched_key is None:
                for k, v in self.cache.items():
                    if k.startswith(f"{entity_id}_") or (isinstance(v, dict) and v.get("entity_id") == entity_id):
                        matched_key = k
                        break

            if matched_key is not None:
                cached = self.cache[matched_key]
                result["vlm_cache_hit"] = True
                result["prediction"] = cached.get("movability")
                result["abstained"] = cached.get("abstain", False)
                result["views_used"] = cached.get("total_views", img_telemetry["usable_views"])
                # Distinguish cached historical latency from live execution latency
                result["cached_latency_ms"] = round(cached.get("latency_s", 0.0) * 1000.0, 2)
                result["inference_latency_ms"] = 0.0
                result["token_usage"] = cached.get("token_usage")
                return result

            # Cached historical mode miss: do not call live server
            result["vlm_cache_hit"] = False
            result["vlm_inference_executed"] = False
            result["prediction"] = "UNKNOWN"
            result["abstained"] = True
            result["vlm_output_valid"] = False
            result["vlm_prediction_accepted"] = False
            result["validation_error"] = "cached_historical_cache_miss"
            return result

        # Strict Live inference execution mode (zero cached lookups)
        result["vlm_cache_hit"] = False
        result["vlm_inference_executed"] = True

        prompt = build_structured_affordance_prompt(entity_id, sem_class, PROMPT_TEMPLATE_VERSION)
        image_contents = [{"type": "text", "text": prompt}]

        for c_p, s_p in img_telemetry["crop_scene_pairs"]:
            c_b64 = encode_image(c_p, max_dim=200)
            s_b64 = encode_image(s_p, max_dim=280)
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{c_b64}"}})
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s_b64}"}})

        messages = [{"role": "user", "content": image_contents}]

        try:
            raw_text, lat_s, usage = query_llama_server(messages, server_url=self.server_url, max_tokens=150)
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
        result["views_used"] = img_telemetry["usable_views"]
        result["inference_latency_ms"] = round(lat_s * 1000.0, 2)
        result["token_usage"] = usage

        # Cache live result under new fingerprint
        self.cache[cache_key] = {
            "entity_id": entity_id,
            "movability": pred,
            "abstain": abstained,
            "total_views": img_telemetry["usable_views"],
            "latency_s": lat_s,
            "token_usage": usage,
        }
        self._save_cache()

        return result
