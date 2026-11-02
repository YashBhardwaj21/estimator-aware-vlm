"""Controlled VLM evaluation harness across multi-view input configurations."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Paths
ANNOTATION_DIR  = REPO / "results" / "track_annotation"
EVAL_OUTPUT_DIR = REPO / "results" / "vlm_evaluation"
RUN_DIR         = REPO / "results" / "perception_harvesting" / "run_20261101_234859"

# Server config
DEFAULT_SERVER_URL = "http://127.0.0.1:8089/v1/chat/completions"
DEFAULT_MODEL      = "Qwen2.5-VL-3B-Instruct"
REQUEST_TIMEOUT_S  = 120
MAX_IMAGE_DIM      = 336   # resize longest edge before encoding

# Label mapping
# Map annotation physical_mobility → VLM affordance class
# MOVABLE + MOVING → MOBILE (for persons) or MOVABLE (objects)
# MOVABLE + STATIONARY → MOVABLE
# STRUCTURALLY_IMMOVABLE → IMMOVABLE

VLM_CLASSES = ["IMMOVABLE", "MOVABLE", "MOBILE"]

def annotation_to_vlm_class(row: Dict[str, str]) -> Optional[str]:
    """Map annotation labels to the 3-class VLM evaluation schema."""
    mob   = row.get("physical_mobility", "")
    mot   = row.get("observed_motion", "")
    cls   = row.get("object_class", "")

    if mob == "STRUCTURALLY_IMMOVABLE":
        return "IMMOVABLE"
    if cls == "person" or (mob == "MOVABLE" and mot == "MOVING" and cls in ("person",)):
        return "MOBILE"
    # persons that are stationary
    if cls == "person":
        return "MOBILE"   # persons are always MOBILE by affordance even if temporarily still
    if mob == "MOVABLE":
        return "MOVABLE"
    return None   # UNCERTAIN annotations excluded from primary metrics


# Image utilities

def encode_image_b64(path: Path, max_dim: int = MAX_IMAGE_DIM) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Prompt templates

SYSTEM_PROMPT = (
    "You are an embodied robotics vision-language expert analyzing images from an RGB-D SLAM system. "
    "Your task is to classify the physical affordance of a detected object. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

# V1: single crop, no geometry
V1_USER_TEMPLATE = """\
The image shows a detected object (class: {semantic_class}) extracted from an indoor RGB-D recording.

Classify it into exactly one of:
- IMMOVABLE: Fixed infrastructure permanently attached to the building (walls, structural pillars, ceiling fixtures).
- MOVABLE: Passively transportable object with no self-locomotion (chair, desk, laptop, box, keyboard, plant).
- MOBILE: Self-propelled agent capable of autonomous locomotion (person, mobile robot).

Respond with STRICT JSON ONLY:
{{
  "object_id": "{track_id}",
  "object_class": "{semantic_class}",
  "affordance": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "confidence": <float 0.0–1.0>,
  "physical_support": "<how the object is supported: floor/desk/wall/unknown>",
  "explanation": "<one sentence visual justification>"
}}"""

# V2: single crop + scene context
V2_USER_TEMPLATE = """\
Two images are provided:
  Image 1: Object crop — the detected {semantic_class} extracted from the scene.
  Image 2: Full scene frame — the RGB context frame in which the object was detected.

Classify the object's physical affordance:
- IMMOVABLE: Fixed building infrastructure.
- MOVABLE: Passively transportable (chair, desk, laptop, box, keyboard, plant).
- MOBILE: Self-propelled agent (person, mobile robot).

Use both the crop detail AND the scene context for your decision.

Respond with STRICT JSON ONLY:
{{
  "object_id": "{track_id}",
  "object_class": "{semantic_class}",
  "affordance": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "confidence": <float 0.0–1.0>,
  "physical_support": "<floor/desk/wall/carried/unknown>",
  "scene_context_used": <true|false>,
  "explanation": "<one sentence visual justification>"
}}"""

# V3: multiple temporal crops, no geometry
V3_USER_TEMPLATE = """\
{n_images} images show the SAME tracked {semantic_class} at different timestamps during a recording:
{image_descriptions}

Classify its physical affordance using the temporal evidence:
- IMMOVABLE: Never moves, fixed infrastructure.
- MOVABLE: Object that could be relocated but does not self-locomote.
- MOBILE: Self-propelled agent (person, robot).

Pay attention to: position changes between frames, posture changes, motion blur,
and consistency of appearance across timestamps.

Respond with STRICT JSON ONLY:
{{
  "object_id": "{track_id}",
  "object_class": "{semantic_class}",
  "affordance": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "confidence": <float 0.0–1.0>,
  "temporal_motion_observed": <true|false>,
  "explanation": "<one sentence justification using temporal evidence>"
}}"""

# V4: multiple temporal crops + geometric context
V4_USER_TEMPLATE = """\
{n_images} images show the SAME tracked {semantic_class} at different timestamps.
{image_descriptions}

Geometric context from the SLAM estimator (ego-motion compensated):
  World-frame displacement : {world_disp_m:.3f} m
  Track lifespan           : {lifespan_s:.1f} s
  Pipeline motion label    : {pipeline_motion}
  Ego-motion compensation  : {ego_motion}

Classify the object's physical affordance using BOTH the visual evidence AND the geometric context:
- IMMOVABLE: Fixed infrastructure (world displacement likely due to estimator noise).
- MOVABLE: Can be relocated; displacement may indicate it was moved or nudged.
- MOBILE: Self-propelled agent; displacement expected from locomotion.

Respond with STRICT JSON ONLY:
{{
  "object_id": "{track_id}",
  "object_class": "{semantic_class}",
  "affordance": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "confidence": <float 0.0–1.0>,
  "geometric_context_used": <true|false>,
  "temporal_motion_observed": <true|false>,
  "explanation": "<one sentence justification referencing visual and/or geometric evidence>"
}}"""


# VLM call

def call_vlm(
    messages: List[Dict],
    server_url: str,
    model: str,
    timeout: int = REQUEST_TIMEOUT_S,
) -> Tuple[Optional[Dict], float, Optional[str]]:
    """Send chat completion request. Returns (parsed_json, latency_ms, error_str)."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
    }).encode("utf-8")

    req = urllib.request.Request(
        server_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body["choices"][0]["message"]["content"]
            usage    = body.get("usage", {})
            return {
                "raw_text"         : raw_text,
                "prompt_tokens"    : usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens"     : usage.get("total_tokens"),
            }, latency_ms, None
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return None, latency_ms, str(e)
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return None, latency_ms, str(e)


def parse_affordance(raw_text: str) -> Optional[str]:
    """Extract affordance label from raw VLM response text."""
    import re
    # Try JSON parse
    try:
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        obj = json.loads(text)
        aff = str(obj.get("affordance", "")).upper().strip()
        if aff in ("IMMOVABLE", "MOVABLE", "MOBILE"):
            return aff
    except Exception:
        pass
    # Fallback: regex scan
    for label in ("MOBILE", "MOVABLE", "IMMOVABLE"):
        if label in raw_text.upper():
            return label
    return None


def parse_confidence(raw_text: str) -> Optional[float]:
    import re
    try:
        text = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        obj = json.loads(text)
        return float(obj.get("confidence", 0.5))
    except Exception:
        return None


def parse_full_response(raw_text: str) -> Dict[str, Any]:
    import re
    result = {
        "affordance"  : None,
        "confidence"  : None,
        "explanation" : None,
        "parse_ok"    : False,
        "raw_text"    : raw_text,
    }
    try:
        text = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        obj = json.loads(text)
        aff = str(obj.get("affordance", "")).upper().strip()
        result["affordance"]  = aff if aff in ("IMMOVABLE", "MOVABLE", "MOBILE") else None
        result["confidence"]  = float(obj.get("confidence", 0.5))
        result["explanation"] = str(obj.get("explanation", ""))
        result["parse_ok"]    = result["affordance"] is not None
        result["full_json"]   = obj
    except Exception as e:
        result["parse_error"] = str(e)
        # Fallback label scan
        for label in ("MOBILE", "MOVABLE", "IMMOVABLE"):
            if label in raw_text.upper():
                result["affordance"] = label
                result["parse_ok"]   = True
                break
    return result


# Build VLM messages per configuration

def build_v1_messages(track: Dict, rep_obs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """V1: single best crop (sharpest)."""
    best = max(rep_obs, key=lambda o: o.get("sharpness", 0))
    crop_path = Path(best["crop_path"])
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop not found: {crop_path}")
    img_b64 = encode_image_b64(crop_path)

    prompt = V1_USER_TEMPLATE.format(
        track_id=track["track_id"],
        semantic_class=track["semantic_class"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text",      "text": prompt},
        ]},
    ]
    meta = {
        "config"      : "V1",
        "images_used" : [str(crop_path)],
        "obs_ids"     : [best["obs_id"]],
    }
    return messages, meta


def build_v2_messages(track: Dict, rep_obs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """V2: best crop + full RGB scene frame."""
    best = max(rep_obs, key=lambda o: o.get("sharpness", 0))
    crop_path  = Path(best["crop_path"])
    scene_path = Path(best["rgb_path"]) if best.get("rgb_path") else None

    if not crop_path.exists():
        raise FileNotFoundError(f"Crop not found: {crop_path}")
    crop_b64 = encode_image_b64(crop_path)

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
    ]
    images_used = [str(crop_path)]

    if scene_path and scene_path.exists():
        scene_b64 = encode_image_b64(scene_path, max_dim=480)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}"}})
        images_used.append(str(scene_path))

    prompt = V2_USER_TEMPLATE.format(
        track_id=track["track_id"],
        semantic_class=track["semantic_class"],
    )
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    meta = {
        "config"         : "V2",
        "images_used"    : images_used,
        "obs_ids"        : [best["obs_id"]],
        "scene_included" : len(images_used) > 1,
    }
    return messages, meta


def build_v3_messages(track: Dict, rep_obs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """V3: 3 temporal crops (early / mid / late)."""
    sorted_obs = sorted(rep_obs, key=lambda o: o["frame_index"])
    n = len(sorted_obs)
    chosen = [sorted_obs[0], sorted_obs[n // 2], sorted_obs[-1]]
    # Deduplicate if track has < 3 obs
    seen_ids: set = set()
    unique_chosen = []
    for o in chosen:
        if o["obs_id"] not in seen_ids:
            unique_chosen.append(o)
            seen_ids.add(o["obs_id"])

    content = []
    images_used = []
    descriptions = []
    for i, obs in enumerate(unique_chosen):
        cp = Path(obs["crop_path"])
        if not cp.exists():
            continue
        b64 = encode_image_b64(cp)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        images_used.append(str(cp))
        descriptions.append(f"  Image {i+1}: frame {obs['frame_index']} at t={obs['timestamp']:.2f}s")

    if not images_used:
        raise FileNotFoundError("No crop images found for V3")

    desc_str = "\n".join(descriptions)
    prompt = V3_USER_TEMPLATE.format(
        n_images=len(images_used),
        track_id=track["track_id"],
        semantic_class=track["semantic_class"],
        image_descriptions=desc_str,
    )
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    meta = {
        "config"      : "V3",
        "images_used" : images_used,
        "obs_ids"     : [o["obs_id"] for o in unique_chosen],
        "n_images"    : len(images_used),
    }
    return messages, meta


def build_v4_messages(track: Dict, rep_obs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """V4: 3 temporal crops + geometric context in prompt."""
    # Reuse V3 image selection
    sorted_obs = sorted(rep_obs, key=lambda o: o["frame_index"])
    n = len(sorted_obs)
    chosen = [sorted_obs[0], sorted_obs[n // 2], sorted_obs[-1]]
    seen_ids: set = set()
    unique_chosen = []
    for o in chosen:
        if o["obs_id"] not in seen_ids:
            unique_chosen.append(o)
            seen_ids.add(o["obs_id"])

    content = []
    images_used = []
    descriptions = []
    for i, obs in enumerate(unique_chosen):
        cp = Path(obs["crop_path"])
        if not cp.exists():
            continue
        b64 = encode_image_b64(cp)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        images_used.append(str(cp))
        descriptions.append(f"  Image {i+1}: frame {obs['frame_index']} at t={obs['timestamp']:.2f}s")

    if not images_used:
        raise FileNotFoundError("No crop images found for V4")

    desc_str = "\n".join(descriptions)
    world_disp = track.get("world_displacement_m") or 0.0
    lifespan   = track.get("lifespan_sec") or 0.0
    prompt = V4_USER_TEMPLATE.format(
        n_images=len(images_used),
        track_id=track["track_id"],
        semantic_class=track["semantic_class"],
        image_descriptions=desc_str,
        world_disp_m=float(world_disp),
        lifespan_s=float(lifespan),
        pipeline_motion=track.get("observed_motion", "UNKNOWN"),
        ego_motion="YES (estimated trajectory used)",
    )
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    meta = {
        "config"            : "V4",
        "images_used"       : images_used,
        "obs_ids"           : [o["obs_id"] for o in unique_chosen],
        "n_images"          : len(images_used),
        "world_disp_m"      : world_disp,
        "lifespan_s"        : lifespan,
        "pipeline_motion"   : track.get("observed_motion"),
    }
    return messages, meta


CONFIG_BUILDERS = {
    "V1": build_v1_messages,
    "V2": build_v2_messages,
    "V3": build_v3_messages,
    "V4": build_v4_messages,
}

# Metrics

def compute_metrics(
    predictions: List[str],
    ground_truth: List[str],
    classes: List[str] = VLM_CLASSES,
) -> Dict[str, Any]:
    """Compute balanced accuracy, per-class P/R/F1, confusion matrix."""
    from collections import Counter

    assert len(predictions) == len(ground_truth)
    n = len(predictions)
    if n == 0:
        return {"n": 0}

    # Confusion matrix
    cm: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for pred, gt in zip(predictions, ground_truth):
        if gt in cm and pred in cm:
            cm[gt][pred] += 1

    # Per-class metrics
    per_class: Dict[str, Dict] = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in classes if r != c)
        fn = sum(cm[c][r] for r in classes if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
            "support"  : tp + fn,
        }

    # Balanced accuracy = mean recall across classes with support > 0
    recalls_with_support = [
        per_class[c]["recall"]
        for c in classes
        if per_class[c]["support"] > 0
    ]
    balanced_acc = sum(recalls_with_support) / len(recalls_with_support) if recalls_with_support else 0.0

    # Macro averages
    macro_p  = sum(per_class[c]["precision"] for c in classes) / len(classes)
    macro_r  = sum(per_class[c]["recall"]    for c in classes) / len(classes)
    macro_f1 = sum(per_class[c]["f1"]        for c in classes) / len(classes)

    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / n

    return {
        "n"                : n,
        "accuracy"         : round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_precision"  : round(macro_p, 4),
        "macro_recall"     : round(macro_r, 4),
        "macro_f1"         : round(macro_f1, 4),
        "per_class"        : per_class,
        "confusion_matrix" : cm,
    }


# Main evaluation runner

def load_data(
    confidence_filter: Optional[List[str]] = None,
    split_filter: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load annotations, track manifest, representative observations.
    Returns (filtered_tracks, rep_obs_by_key).
    """
    # Load annotations
    ann_rows = list(csv.DictReader(
        open(ANNOTATION_DIR / "track_annotations.csv", encoding="utf-8")
    ))

    # Load track manifest
    manifest = json.loads((ANNOTATION_DIR / "track_manifest.json").read_text())
    manifest_by_key = {
        f"{t['sequence_id']}__{t['track_id']}": t
        for t in manifest["tracks"]
    }

    # Load evaluation split
    split_data = json.loads((ANNOTATION_DIR / "evaluation_split.json").read_text())

    # Load representative observations
    rep_obs_by_key: Dict[str, List[Dict]] = defaultdict(list)
    for line in (ANNOTATION_DIR / "representative_observations.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rep_obs_by_key[f"{o['sequence_id']}__{o['track_id']}"].append(o)

    # Apply filters
    valid_sequences: Optional[set] = None
    if split_filter:
        partition = split_data["partitions"].get(split_filter)
        if not partition:
            sys.exit(f"Unknown split: {split_filter}. Valid: {list(split_data['partitions'].keys())}")
        valid_sequences = set(partition["sequences"])
        # Never use perception_only sequences for primary evaluation
        # (they can be run explicitly with --split perception_only)
        if split_filter != "perception_only":
            perc_only = set(split_data["partitions"]["perception_only"]["sequences"])
            valid_sequences -= perc_only

    conf_filter = set(confidence_filter) if confidence_filter else None

    filtered_tracks = []
    for row in ann_rows:
        # Split filter
        if valid_sequences and row["sequence_id"] not in valid_sequences:
            continue
        # Confidence filter
        if conf_filter and row["annotation_confidence"] not in conf_filter:
            continue
        # Skip UNCERTAIN observed_motion (no unambiguous GT label)
        gt_label = annotation_to_vlm_class(row)
        if gt_label is None:
            continue

        key = f"{row['sequence_id']}__{row['track_id']}"
        track_meta = manifest_by_key.get(key, {})

        # Merge annotation row with track manifest for geometric fields
        merged = dict(row)
        merged["semantic_class"]       = row["object_class"]
        merged["world_displacement_m"] = track_meta.get("world_displacement_m")
        merged["lifespan_sec"]         = track_meta.get("lifespan_sec")
        merged["observed_motion"]      = track_meta.get("observed_motion")  # pipeline label
        merged["gt_vlm_label"]         = gt_label

        # Verify at least one crop exists
        rep_obs = rep_obs_by_key.get(key, [])
        if not rep_obs:
            continue
        valid_crops = [o for o in rep_obs if Path(o.get("crop_path", "")).exists()]
        if not valid_crops:
            continue
        merged["_valid_rep_obs"] = valid_crops

        filtered_tracks.append(merged)

    return filtered_tracks, rep_obs_by_key


def run_evaluation(
    configs: List[str],
    server_url: str,
    model: str,
    dry_run: bool,
    split_filter: Optional[str],
    confidence_filter: List[str],
    out_dir: Path,
) -> None:
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{run_tag}"
    run_dir.mkdir(parents=True)

    print(f"\nControlled VLM Evaluation")
    print(f"Server    : {server_url}")
    print(f"Model     : {model}")
    print(f"Configs   : {configs}")
    print(f"Split     : {split_filter or 'all'}")
    print(f"Confidence: {confidence_filter}")
    print(f"Dry run   : {dry_run}")
    print(f"Output    : {run_dir}\n")

    # Check server health
    if not dry_run:
        try:
            health_req = urllib.request.urlopen(
                server_url.replace("/v1/chat/completions", "/health"), timeout=5
            )
            print(f"Server health: OK ({health_req.status})\n")
        except Exception as e:
            print(f"WARNING: VLM server health check failed: {e}")
            print("Cannot proceed without VLM server. Start llama-server in WSL2 first.")
            print("\nTo start: wsl -e llama-server --model /path/to/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --port 8089 --ctx-size 4096 --n-gpu-layers 99")
            sys.exit(1)

    # Load data
    tracks, rep_obs_by_key = load_data(
        confidence_filter=confidence_filter,
        split_filter=split_filter,
    )
    print(f"Tracks eligible for evaluation: {len(tracks)}")

    if not tracks:
        sys.exit("No tracks match the filter criteria.")

    # Label distribution
    gt_dist = Counter(t["gt_vlm_label"] for t in tracks)
    print(f"GT label distribution: {dict(gt_dist)}")
    print()

    # Per-config results accumulator
    all_records: List[Dict] = []
    config_predictions: Dict[str, List] = {c: [] for c in configs}
    config_gt: Dict[str, List] = {c: [] for c in configs}
    config_latencies: Dict[str, List] = {c: [] for c in configs}
    config_token_counts: Dict[str, List] = {c: [] for c in configs}
    config_parse_failures: Dict[str, int] = {c: 0 for c in configs}
    config_server_errors: Dict[str, int] = {c: 0 for c in configs}

    total_calls = len(tracks) * len(configs)
    call_idx = 0

    for track in tracks:
        key = f"{track['sequence_id']}__{track['track_id']}"
        rep_obs = track["_valid_rep_obs"]
        gt_label = track["gt_vlm_label"]

        for config in configs:
            call_idx += 1
            builder = CONFIG_BUILDERS[config]

            record: Dict[str, Any] = {
                "run_tag"          : run_tag,
                "config"           : config,
                "sequence_id"      : track["sequence_id"],
                "track_id"         : track["track_id"],
                "semantic_class"   : track["semantic_class"],
                "gt_vlm_label"     : gt_label,
                "annotation_confidence": track["annotation_confidence"],
                "physical_mobility": track["physical_mobility"],
                "observed_motion_annotation": track["observed_motion"],
            }

            if dry_run:
                print(f"  [{call_idx}/{total_calls}] DRY-RUN {config} {track['sequence_id']}/{track['track_id']}")
                record.update({
                    "status"        : "DRY_RUN",
                    "pred_affordance": None,
                    "pred_confidence": None,
                    "latency_ms"    : 0,
                    "prompt_tokens" : 0,
                    "completion_tokens": 0,
                })
                all_records.append(record)
                continue

            # Build messages
            try:
                messages, meta = builder(track, rep_obs)
                record.update(meta)
            except Exception as e:
                print(f"  [{call_idx}/{total_calls}] BUILD ERROR {config} {key}: {e}")
                record.update({"status": "BUILD_ERROR", "error": str(e)})
                all_records.append(record)
                config_server_errors[config] += 1
                continue

            # Call VLM
            print(f"  [{call_idx}/{total_calls}] {config} {track['sequence_id']}/{track['track_id']} gt={gt_label} ...", end="", flush=True)
            response, latency_ms, error = call_vlm(messages, server_url, model)

            if error or response is None:
                print(f" ERROR ({error})")
                record.update({
                    "status"     : "SERVER_ERROR",
                    "error"      : error,
                    "latency_ms" : latency_ms,
                })
                all_records.append(record)
                config_server_errors[config] += 1
                continue

            # Parse response
            parsed = parse_full_response(response["raw_text"])
            pred   = parsed.get("affordance")

            if pred is None:
                print(f" PARSE_FAIL")
                config_parse_failures[config] += 1
            else:
                correct = "CORRECT" if pred == gt_label else "WRONG"
                print(f" pred={pred} conf={parsed.get('confidence', '?')} {correct} ({latency_ms:.0f}ms)")
                config_predictions[config].append(pred)
                config_gt[config].append(gt_label)
                config_latencies[config].append(latency_ms)
                if response.get("total_tokens"):
                    config_token_counts[config].append(response["total_tokens"])

            record.update({
                "status"            : "OK" if pred else "PARSE_FAIL",
                "pred_affordance"   : pred,
                "pred_confidence"   : parsed.get("confidence"),
                "pred_explanation"  : parsed.get("explanation"),
                "pred_parse_ok"     : parsed.get("parse_ok"),
                "raw_response"      : response["raw_text"],
                "latency_ms"        : latency_ms,
                "prompt_tokens"     : response.get("prompt_tokens"),
                "completion_tokens" : response.get("completion_tokens"),
                "total_tokens"      : response.get("total_tokens"),
                "is_correct"        : pred == gt_label if pred else None,
            })
            all_records.append(record)

    # Write per-record log
    records_path = run_dir / "inference_records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nInference records: {records_path} ({len(all_records)} entries)")

    if dry_run:
        print("\nDry-run complete. No VLM calls were made.")
        return

    # Compute metrics per config
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics_by_config: Dict[str, Dict] = {}
    for config in configs:
        preds = config_predictions[config]
        gts   = config_gt[config]
        lats  = config_latencies[config]
        toks  = config_token_counts[config]

        if not preds:
            print(f"\n{config}: No valid predictions.")
            continue

        m = compute_metrics(preds, gts)
        m["latency_ms_median"] = sorted(lats)[len(lats) // 2] if lats else None
        m["latency_ms_p95"]    = sorted(lats)[int(len(lats) * 0.95)] if lats else None
        m["latency_ms_mean"]   = sum(lats) / len(lats) if lats else None
        m["mean_tokens"]       = sum(toks) / len(toks) if toks else None
        m["parse_failures"]    = config_parse_failures[config]
        m["server_errors"]     = config_server_errors[config]
        metrics_by_config[config] = m

        print(f"\n{config}  (n={m['n']}  parse_fail={m['parse_failures']}  server_err={m['server_errors']})")
        print(f"  Accuracy         : {m['accuracy']:.3f}")
        print(f"  Balanced Accuracy: {m['balanced_accuracy']:.3f}")
        print(f"  Macro F1         : {m['macro_f1']:.3f}")
        print(f"  Macro P/R        : {m['macro_precision']:.3f} / {m['macro_recall']:.3f}")
        print(f"  Latency (median) : {m['latency_ms_median']:.0f}ms")
        print(f"  Mean tokens      : {m['mean_tokens']}")
        for cls in VLM_CLASSES:
            pc = m["per_class"][cls]
            print(f"    {cls:12s}: P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  support={pc['support']}")

    # Write metrics
    metrics_path = run_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps({
            "run_tag"          : run_tag,
            "configs_run"      : configs,
            "n_tracks"         : len(tracks),
            "split_filter"     : split_filter,
            "confidence_filter": confidence_filter,
            "gt_distribution"  : dict(gt_dist),
            "metrics"          : metrics_by_config,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nMetrics: {metrics_path}")

    # Write markdown summary
    summary_lines = [
        "# Controlled VLM Evaluation Results",
        "",
        f"Run: `{run_tag}`",
        f"Tracks evaluated: {len(tracks)}",
        f"Split: {split_filter or 'all'}",
        f"Confidence filter: {confidence_filter}",
        "",
        "## Results by Configuration",
        "",
        "| Config | n | Bal.Acc | Macro F1 | Lat.Median | Parse Fail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for config, m in metrics_by_config.items():
        summary_lines.append(
            f"| {config} | {m['n']} | {m['balanced_accuracy']:.3f} | "
            f"{m['macro_f1']:.3f} | {m.get('latency_ms_median',0):.0f}ms | {m['parse_failures']} |"
        )
    summary_lines += ["", "## Per-Class Breakdown", ""]
    for config, m in metrics_by_config.items():
        summary_lines.append(f"### {config}")
        summary_lines.append("")
        summary_lines.append("| Class | Precision | Recall | F1 | Support |")
        summary_lines.append("| --- | --- | --- | --- | --- |")
        for cls in VLM_CLASSES:
            pc = m["per_class"][cls]
            summary_lines.append(f"| {cls} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")
        summary_lines.append("")

    (run_dir / "evaluation_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"Summary : {run_dir / 'evaluation_summary.md'}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled VLM evaluation over 4 input configurations."
    )
    parser.add_argument(
        "--configs", nargs="+", default=["V1", "V2", "V3", "V4"],
        choices=["V1", "V2", "V3", "V4"],
        help="Which configurations to run (default: all four).",
    )
    parser.add_argument(
        "--split", default=None,
        choices=["development", "validation", "held_out", "perception_only"],
        help="Only evaluate sequences in this split partition.",
    )
    parser.add_argument(
        "--confidence", nargs="+", default=["HIGH", "MEDIUM"],
        choices=["HIGH", "MEDIUM", "LOW"],
        help="Annotation confidence levels to include (default: HIGH MEDIUM).",
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL,
        help=f"VLM server endpoint (default: {DEFAULT_SERVER_URL}).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model name to send in the request.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data loading and message building without calling VLM.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=EVAL_OUTPUT_DIR,
        help="Root output directory (default: results/vlm_evaluation/).",
    )
    args = parser.parse_args()

    run_evaluation(
        configs=args.configs,
        server_url=args.server_url,
        model=args.model,
        dry_run=args.dry_run,
        split_filter=args.split,
        confidence_filter=args.confidence,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

