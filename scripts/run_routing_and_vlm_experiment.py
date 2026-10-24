"""Comparative Routing and Live VLM Inference Experiment (Milestone 11).

Evaluates policies B0 through B6 on the 17 tracked candidate entities from fr3_walking_xyz:
- B0: Blind RETAIN (no filtering)
- B1: Blind INVALIDATE (mask all candidate tracks)
- B2: Geometry Only (mask if world displacement >= 0.30 m)
- B3: VLM All (query live VLM on 100% of detected tracks)
- B4: Random Routing (query random 5-track subset)
- B5: Confidence-Threshold Routing (query high-uncertainty / ambiguous tracks)
- B6: Estimator-Aware Selective Routing (Proposed: query where utility U(q) > 0)

Executes genuine live VLM inference with Qwen2.5-VL-3B-Instruct on port 8089,
records wire latency and tokens, and generates landmark intervention actions for Pass B.
"""

from __future__ import annotations

import base64
import json
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root


SERVER_URL = "http://127.0.0.1:8089/v1/chat/completions"


@dataclass
class VLMQueryResult:
    semantic_class: str
    contextual_movability: str
    reasoning: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    is_valid_schema: bool


def query_live_vlm(crop_path: Path, timeout: float = 75.0) -> VLMQueryResult:
    """Send target crop to live VLM server with strict schema validation."""
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop image not found: {crop_path}")

    img = cv2.imread(str(crop_path))
    if img is None:
        raise ValueError(f"Failed to decode crop image: {crop_path}")

    # Directly encode clean crop image
    h, w = img.shape[:2]
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    prompt_text = (
        "You are an expert robot perception assistant analyzing an indoor office/lab environment.\n"
        "Examine this cropped image of an object.\n"
        "Identify which relevant office object it is, such as:\n"
        "- monitor / computer screen (IMMOVABLE: mounted on desk or stand)\n"
        "- desk / table (IMMOVABLE: stationary heavy furniture)\n"
        "- chair / office chair (MOVABLE: transportable furniture on wheels or legs)\n"
        "- keyboard / mouse / laptop (MOVABLE: small tabletop accessories)\n"
        "- person / human (MOBILE: walking or moving human agent)\n\n"
        "Determine the exact semantic class and its contextual movability (IMMOVABLE, MOVABLE, or MOBILE).\n"
        "Return valid JSON strictly matching this schema:\n"
        '{"semantic_class": "<name>", "contextual_movability": "<IMMOVABLE|MOVABLE|MOBILE>", "reasoning": "<brief explanation>"}'
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.time() - t0) * 1000

    raw_text = res["choices"][0]["message"]["content"]
    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Parse JSON
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
        sem_cls = str(data.get("semantic_class", "unknown")).lower()
        mov = str(data.get("contextual_movability", "UNKNOWN")).upper()
        reason = str(data.get("reasoning", ""))
        valid = mov in ["IMMOVABLE", "MOVABLE", "MOBILE"]
    except Exception:
        sem_cls = "unknown"
        mov = "UNKNOWN"
        reason = raw_text[:100]
        valid = False

    return VLMQueryResult(
        semantic_class=sem_cls,
        contextual_movability=mov,
        reasoning=reason,
        latency_ms=round(lat_ms, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        is_valid_schema=valid,
    )


def compute_policy_decisions(
    tracks: List[Dict[str, Any]],
    vlm_query_fn: Any,
) -> Dict[str, Any]:
    """Execute B0 through B6 routing on candidate tracks."""
    random.seed(42)

    # Cache for live queries so an entity is queried at most once across policies
    vlm_cache: Dict[str, VLMQueryResult] = {}

    def get_vlm(track_id: str, crop_path_str: str) -> VLMQueryResult:
        if track_id not in vlm_cache:
            print(f"  [Live VLM Query] Querying Qwen2.5-VL for {track_id}...")
            res = vlm_query_fn(Path(crop_path_str))
            print(f"  [Live VLM Response] {track_id} -> {res.semantic_class} ({res.contextual_movability}) in {res.latency_ms:.0f} ms")
            vlm_cache[track_id] = res
        return vlm_cache[track_id]

    policies = ["B0_Blind_RETAIN", "B1_Blind_INVALIDATE", "B2_Geometry_Only", "B3_VLM_All", "B4_Random", "B5_Confidence_Threshold", "B6_Estimator_Aware"]
    results = {p: {"decisions": [], "query_count": 0, "total_latency_ms": 0.0, "masked_count": 0, "retained_count": 0} for p in policies}

    for t in tracks:
        tid = t["track_id"]
        sem_cls = t["semantic_class"]
        obs_cnt = t.get("observation_count", 0)
        w_disp = t.get("world_displacement_m", t.get("measured_displacement_m", 0.0))
        disp_unc = t.get("displacement_uncertainty_m", 0.0)
        mot = t.get("observed_motion", "STATIC")
        crop_path = t["crop_paths"][0] if t.get("crop_paths") else None

        # --- B0: Blind RETAIN ---
        results["B0_Blind_RETAIN"]["decisions"].append({
            "track_id": tid, "action": "RETAIN", "vlm_queried": False
        })
        results["B0_Blind_RETAIN"]["retained_count"] += 1

        # --- B1: Blind INVALIDATE ---
        results["B1_Blind_INVALIDATE"]["decisions"].append({
            "track_id": tid, "action": "MASK_FROM_TRACKING", "vlm_queried": False
        })
        results["B1_Blind_INVALIDATE"]["masked_count"] += 1

        # --- B2: Geometry Only (World displacement >= 0.30 m) ---
        if mot == "MOVING" or w_disp >= 0.30:
            act_b2 = "MASK_FROM_TRACKING"
            results["B2_Geometry_Only"]["masked_count"] += 1
        else:
            act_b2 = "RETAIN"
            results["B2_Geometry_Only"]["retained_count"] += 1
        results["B2_Geometry_Only"]["decisions"].append({
            "track_id": tid, "action": act_b2, "vlm_queried": False
        })

        # --- B3: VLM All (Query 100% of candidates) ---
        if crop_path:
            v_res = get_vlm(tid, crop_path)
            results["B3_VLM_All"]["query_count"] += 1
            results["B3_VLM_All"]["total_latency_ms"] += v_res.latency_ms
            # If VLM says MOBILE or (MOVABLE and observed moving): mask from SLAM
            if v_res.contextual_movability == "MOBILE" or (v_res.contextual_movability == "MOVABLE" and mot == "MOVING"):
                act_b3 = "MASK_FROM_TRACKING"
                results["B3_VLM_All"]["masked_count"] += 1
            else:
                act_b3 = "RETAIN"
                results["B3_VLM_All"]["retained_count"] += 1
        else:
            act_b3 = "RETAIN"
            results["B3_VLM_All"]["retained_count"] += 1
        results["B3_VLM_All"]["decisions"].append({
            "track_id": tid, "action": act_b3, "vlm_queried": True
        })

        # --- B4: Random Routing (Target budget: ~5 queries) ---
        is_random_query = random.random() < 0.30
        if is_random_query and crop_path:
            v_res = get_vlm(tid, crop_path)
            results["B4_Random"]["query_count"] += 1
            results["B4_Random"]["total_latency_ms"] += v_res.latency_ms
            act_b4 = "MASK_FROM_TRACKING" if v_res.contextual_movability == "MOBILE" else ("MASK_FROM_TRACKING" if mot == "MOVING" else "RETAIN")
        else:
            act_b4 = "MASK_FROM_TRACKING" if mot == "MOVING" else "RETAIN"
        if act_b4 == "MASK_FROM_TRACKING":
            results["B4_Random"]["masked_count"] += 1
        else:
            results["B4_Random"]["retained_count"] += 1
        results["B4_Random"]["decisions"].append({
            "track_id": tid, "action": act_b4, "vlm_queried": is_random_query
        })

        # --- B5: Confidence-Threshold Routing ---
        # Query when displacement is uncertain (disp_unc > 0.20 m) or borderline displaced
        needs_b5 = (0.12 <= w_disp < 0.35) or (disp_unc > 0.20)
        if needs_b5 and crop_path:
            v_res = get_vlm(tid, crop_path)
            results["B5_Confidence_Threshold"]["query_count"] += 1
            results["B5_Confidence_Threshold"]["total_latency_ms"] += v_res.latency_ms
            act_b5 = "MASK_FROM_TRACKING" if v_res.contextual_movability == "MOBILE" or mot == "MOVING" else "RETAIN"
        else:
            act_b5 = "MASK_FROM_TRACKING" if mot == "MOVING" else "RETAIN"
        if act_b5 == "MASK_FROM_TRACKING":
            results["B5_Confidence_Threshold"]["masked_count"] += 1
        else:
            results["B5_Confidence_Threshold"]["retained_count"] += 1
        results["B5_Confidence_Threshold"]["decisions"].append({
            "track_id": tid, "action": act_b5, "vlm_queried": needs_b5
        })

        # --- B6: Estimator-Aware Selective Routing (Proposed) ---
        # Utility formulation:
        # Expected SLAM loss reduction delta_L_SLAM is high when:
        # 1. Observation count is large (high potential to corrupt pose graph)
        # 2. Observed world displacement shows active motion or high dispersion (conflict with rigid assumption)
        # 3. Object category is ambiguous (e.g. chair moved vs walking human)
        obs_impact = min(1.0, obs_cnt / 100.0)
        disp_impact = min(1.0, w_disp / 0.50)
        ambiguity = 1.0 if (0.15 <= w_disp <= 0.80) else 0.2
        delta_L_slam = obs_impact * 0.40 + disp_impact * 0.40 + ambiguity * 0.20
        lambda_cost = 0.45
        c_vlm = 0.50  # Normalized query cost
        utility = delta_L_slam - lambda_cost * c_vlm

        # Escalate to live VLM only when net utility U(q) > 0
        should_query_b6 = (utility > 0.0) and (crop_path is not None)

        if should_query_b6:
            v_res = get_vlm(tid, crop_path)
            results["B6_Estimator_Aware"]["query_count"] += 1
            results["B6_Estimator_Aware"]["total_latency_ms"] += v_res.latency_ms
            # VLM Affordance Resolution:
            # 1. MOBILE (walking humans): Always MASK_FROM_TRACKING to prevent dynamic corruption.
            # 2. IMMOVABLE (monitors, desks): Always RETAIN as static reference landmarks.
            # 3. MOVABLE (chairs): MASK only if physically displaced in world frame.
            if v_res.contextual_movability == "MOBILE":
                act_b6 = "MASK_FROM_TRACKING"
            elif v_res.contextual_movability == "IMMOVABLE":
                act_b6 = "RETAIN"
            elif mot == "MOVING" and w_disp >= 0.60:
                act_b6 = "MASK_FROM_TRACKING"
            else:
                act_b6 = "RETAIN"
        else:
            # Resolve cheaply using estimator prior
            act_b6 = "MASK_FROM_TRACKING" if (mot == "MOVING" and w_disp >= 1.0) else "RETAIN"

        if act_b6 == "MASK_FROM_TRACKING":
            results["B6_Estimator_Aware"]["masked_count"] += 1
        else:
            results["B6_Estimator_Aware"]["retained_count"] += 1

        results["B6_Estimator_Aware"]["decisions"].append({
            "track_id": tid,
            "action": act_b6,
            "vlm_queried": should_query_b6,
            "utility": round(utility, 4),
            "delta_L_slam": round(delta_L_slam, 4),
        })

    return {
        "policies": results,
        "vlm_telemetry": {tid: asdict(res) for tid, res in vlm_cache.items()},
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Execute comparative routing and live VLM inference.")
    parser.add_argument("--tracks-summary", type=str, default=None, help="Path to tracks_summary.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for VLM inference results")
    parser.add_argument("--dataset-name", type=str, default="fr3_walking_xyz", help="Dataset name for reporting")
    args = parser.parse_args()

    repo_root = get_repo_root()
    summary_path = Path(args.tracks_summary) if args.tracks_summary else (repo_root / "outputs" / "perception_tracks" / "tracks_summary.json")
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Track summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    print(f"Loaded {len(tracks)} candidate tracks from {summary_path} for comparative routing.")
    print("Beginning Policy Execution (B0 through B6) with Live VLM Inference...\n")

    exp_results = compute_policy_decisions(tracks, query_live_vlm)

    # Export outputs
    out_dir = Path(args.output_dir) if args.output_dir else (repo_root / "outputs" / "vlm_inference")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "policy_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=2)

    decisions_path = out_dir / "routing_decisions.json"
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(exp_results["policies"]["B6_Estimator_Aware"], f, indent=2)

    # Print summary table
    print("\n==========================================================================================")
    print(f"                 COMPARATIVE ROUTING POLICY RESULTS ON {args.dataset_name.upper()}         ")
    print("==========================================================================================")
    print(f"{'Policy':<26} | {'VLM Calls':<10} | {'Total Latency (s)':<18} | {'Masked (Dynamic)':<18} | {'Retained (Static)'}")
    print("-" * 90)
    for pol_name, data in exp_results["policies"].items():
        lat_sec = data["total_latency_ms"] / 1000.0
        print(f"{pol_name:<26} | {data['query_count']:<10} | {lat_sec:<18.2f} | {data['masked_count']:<18} | {data['retained_count']}")
    print("==========================================================================================")
    print(f"Exported policy comparison to: {report_path}")
    print(f"Exported B6 decisions to:      {decisions_path}\n")


if __name__ == "__main__":
    main()

