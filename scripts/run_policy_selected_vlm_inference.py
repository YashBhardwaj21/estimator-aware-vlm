"""Policy-selected visual-language model inference module.

Executes live inference queries on observations selected by routing policies
(from results/query_policy/union_selected_queries_10pct.json).

Guarantees:
- Model evaluations performed directly via local visual-language model server.
- Verifies image existence and SHA-256 hash match against the manifest.
- Feeds observation crop and causal geometric context into Qwen2.5-VL-3B-Instruct.
- Records obs_id, image_sha256, model_id, prompt_version ("V4"), decoding params, latency, tokens, and response.
- Supports incremental checkpointing and resuming.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SERVER_URL = "http://127.0.0.1:8089/v1/chat/completions"
DEFAULT_MODEL = "Qwen2.5-VL-3B-Instruct"
DEFAULT_UNION_PATH = REPO_ROOT / "results" / "query_policy" / "union_selected_queries_10pct.json"
DEFAULT_FEATURES_PATH = REPO_ROOT / "results" / "estimator_context" / "feature_vectors.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "policy_vlm_inference"

SYSTEM_PROMPT = (
    "You are an embodied robotics vision-language expert analyzing images from an RGB-D SLAM system. "
    "Your task is to classify physical affordance and estimate observed temporal motion. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

V4_OBS_TEMPLATE = """\
Object observation (class: {semantic_class}) at sequence {sequence_id}, frame {frame_index} (t={timestamp:.2f}s):

Causal geometric context from the SLAM estimator up to this observation:
  Causal world displacement : {causal_world_disp}
  Causal path length        : {causal_path_len}
  Causal elapsed time       : {causal_elapsed_s:.1f} s
  Causal observation count  : {causal_count}
  Ego-motion compensation   : {ego_motion}

1. Classify its physical affordance:
- IMMOVABLE: Structurally fixed or permanently integrated into the environment.
- MOVABLE: Can be relocated by a person or external force.
- MOBILE: Self-propelled agent or entity (person, mobile robot, animal).

2. Estimate its observed temporal motion:
- STATIONARY: The object is static in world space at this timestamp.
- MOVING: The object is actively displacing or locomoting.

Respond with STRICT JSON ONLY:
{{
  "object_id": "{track_id}",
  "object_class": "{semantic_class}",
  "pred_affordance": "IMMOVABLE" | "MOVABLE" | "MOBILE",
  "affordance_confidence": <float 0.0-1.0>,
  "pred_observed_motion": "STATIONARY" | "MOVING",
  "motion_confidence": <float 0.0-1.0>,
  "geometric_context_used": <true|false>,
  "explanation": "<one sentence justification referencing visual and causal geometric evidence>"
}}"""


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def encode_image_b64(path: Path, max_dim: int = 336) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_vlm_response(raw_text: str) -> Dict[str, Any]:
    import re
    result = {
        "pred_affordance": None,
        "pred_observed_motion": "NOT_ASSESSED",
        "affordance_confidence": 0.5,
        "motion_confidence": 0.5,
        "explanation": "",
        "parse_ok": False,
    }
    try:
        text = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        obj = json.loads(text)

        aff = str(obj.get("pred_affordance") or obj.get("affordance", "")).upper().strip()
        if aff in ("IMMOVABLE", "MOVABLE", "MOBILE"):
            result["pred_affordance"] = aff

        mot = str(obj.get("pred_observed_motion") or obj.get("observed_motion", "")).upper().strip()
        if mot in ("STATIONARY", "MOVING"):
            result["pred_observed_motion"] = mot

        aff_c = obj.get("affordance_confidence") or obj.get("confidence")
        if aff_c is not None:
            result["affordance_confidence"] = float(aff_c)

        mot_c = obj.get("motion_confidence")
        if mot_c is not None:
            result["motion_confidence"] = float(mot_c)

        result["explanation"] = str(obj.get("explanation", ""))
        result["parse_ok"] = result["pred_affordance"] is not None
        result["full_json"] = obj
    except Exception as e:
        for label in ("MOBILE", "MOVABLE", "IMMOVABLE"):
            if label in raw_text.upper():
                result["pred_affordance"] = label
                result["parse_ok"] = True
                break
        for mot_label in ("STATIONARY", "MOVING"):
            if mot_label in raw_text.upper():
                result["pred_observed_motion"] = mot_label
                break
    return result


def call_vlm(
    messages: List[Dict],
    server_url: str,
    model: str,
    timeout: int = 120,
) -> Tuple[Optional[Dict], float, Optional[str]]:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 128,
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
            usage = body.get("usage", {})
            return {
                "raw_text": raw_text,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }, latency_ms, None
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return None, latency_ms, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Policy-Selected VLM Inference")
    parser.add_argument("--union-file", type=Path, default=DEFAULT_UNION_PATH)
    parser.add_argument("--features-file", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries (0=all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_records_path = args.out_dir / "live_policy_vlm_records.jsonl"

    print("Policy-Selected VLM Inference Execution")
    print(f"Union manifest: {args.union_file}")
    print(f"Server URL    : {args.server_url}")
    print(f"Model         : {args.model}")
    print(f"Output records: {out_records_path}")

    # Load existing records to support resumption
    existing_obs: Set[str] = set()
    if out_records_path.exists():
        with open(out_records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("inference_status") == "SUCCESS":
                        existing_obs.add(r["obs_id"])
        print(f"Resuming run: found {len(existing_obs)} existing completed records.")

    # Load feature vectors for causal geometric context
    features_by_obs: Dict[str, Dict[str, Any]] = {}
    if args.features_file.exists():
        with open(args.features_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feat = json.loads(line)
                    features_by_obs[feat["obs_id"]] = feat
        print(f"Loaded {len(features_by_obs)} causal feature vectors.")

    # Load union queries
    with open(args.union_file, "r", encoding="utf-8") as f:
        union_queries = json.load(f)
    print(f"Total union queries in manifest: {len(union_queries)}")

    to_run = [q for q in union_queries if q["obs_id"] not in existing_obs]
    if args.limit > 0:
        to_run = to_run[:args.limit]
    print(f"Queries remaining to execute: {len(to_run)}")

    if not to_run:
        print("All queries already completed.")
        return

    # Check server health
    health_url = args.server_url.replace("/v1/chat/completions", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            print(f"VLM Server health: {resp.status} OK\n")
    except Exception as e:
        sys.exit(f"Cannot connect to VLM server at {health_url}: {e}")

    success_count = 0
    fail_count = 0

    with open(out_records_path, "a", encoding="utf-8") as out_f:
        for idx, q in enumerate(to_run, start=1):
            obs_id = q["obs_id"]
            img_path = Path(q["image_path"])

            if not img_path.exists():
                print(f"[{idx}/{len(to_run)}] MISSING FILE: {img_path}")
                fail_count += 1
                continue

            # Verify SHA256 matches manifest
            actual_sha = compute_sha256(img_path)
            expected_sha = q.get("image_sha256")
            if expected_sha and actual_sha != expected_sha:
                print(f"[{idx}/{len(to_run)}] SHA256 MISMATCH for {obs_id}")
                fail_count += 1
                continue

            # Load causal features
            feat = features_by_obs.get(obs_id, {})
            wld_disp = feat.get("causal_world_displacement_m")
            path_len = feat.get("causal_path_length_m")
            elapsed_s = feat.get("causal_elapsed_time_s", 0.0)
            obs_cnt = feat.get("causal_observation_count", 1)
            ego_avail = feat.get("ego_motion_available", False)

            disp_str = f"{wld_disp:.3f} m" if wld_disp is not None else "Unavailable"
            path_str = f"{path_len:.3f} m" if path_len is not None else "Unavailable"
            ego_str = "YES (causal visual odometry)" if ego_avail else "NO (uncalibrated)"

            prompt = V4_OBS_TEMPLATE.format(
                semantic_class=q.get("semantic_class", "object"),
                sequence_id=q["sequence_id"],
                frame_index=q["frame_index"],
                timestamp=q["timestamp"],
                track_id=q["track_id"],
                causal_world_disp=disp_str,
                causal_path_len=path_str,
                causal_elapsed_s=float(elapsed_s),
                causal_count=int(obs_cnt),
                ego_motion=ego_str,
            )

            b64_img = encode_image_b64(img_path)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    {"type": "text", "text": prompt},
                ]},
            ]

            print(f"[{idx}/{len(to_run)}] Querying {obs_id} ({q['semantic_class']})... ", end="", flush=True)
            resp, lat_ms, err = call_vlm(messages, args.server_url, args.model)

            if err or not resp:
                print(f"ERROR: {err} ({lat_ms:.0f}ms)")
                rec = {
                    "obs_id": obs_id,
                    "sequence_id": q["sequence_id"],
                    "track_id": q["track_id"],
                    "frame_index": q["frame_index"],
                    "timestamp": q["timestamp"],
                    "crop_path": str(img_path),
                    "image_sha256": actual_sha,
                    "prompt_version": "V4",
                    "model_id": args.model,
                    "decoding": {"temperature": 0.0, "max_tokens": 128},
                    "latency_ms": lat_ms,
                    "inference_status": "ERROR",
                    "error": str(err),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                fail_count += 1
            else:
                parsed = parse_vlm_response(resp["raw_text"])
                pred_aff = parsed["pred_affordance"]
                pred_mot = parsed["pred_observed_motion"]
                aff_c = parsed["affordance_confidence"]
                mot_c = parsed["motion_confidence"]

                print(f"aff={pred_aff}({aff_c:.2f}) mot={pred_mot}({mot_c:.2f}) ({lat_ms:.0f}ms)")
                rec = {
                    "obs_id": obs_id,
                    "sequence_id": q["sequence_id"],
                    "track_id": q["track_id"],
                    "frame_index": q["frame_index"],
                    "timestamp": q["timestamp"],
                    "crop_path": str(img_path),
                    "image_sha256": actual_sha,
                    "prompt_version": "V4",
                    "config": "V4",
                    "model_id": args.model,
                    "decoding": {"temperature": 0.0, "max_tokens": 128},
                    "latency_ms": lat_ms,
                    "prompt_tokens": resp.get("prompt_tokens"),
                    "completion_tokens": resp.get("completion_tokens"),
                    "total_tokens": resp.get("total_tokens"),
                    "pred_affordance": pred_aff,
                    "affordance": pred_aff,
                    "pred_observed_motion": pred_mot,
                    "affordance_confidence": aff_c,
                    "confidence": aff_c,
                    "motion_confidence": mot_c,
                    "explanation": parsed["explanation"],
                    "raw_response": resp["raw_text"],
                    "inference_status": "SUCCESS",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                success_count += 1

            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    print(f"\nInference complete: {success_count} succeeded, {fail_count} failed.")
    print(f"Records saved to: {out_records_path}\n")


if __name__ == "__main__":
    main()

