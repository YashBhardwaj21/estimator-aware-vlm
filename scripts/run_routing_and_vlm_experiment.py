#!/usr/bin/env python3
"""Comparative Routing and Live VLM Inference Experiment (Stage 6).

Evaluates policies B0 through B6 on tracked candidate entities:
- B0: Blind RETAIN (no filtering, 0 queries)
- B1: Blind INVALIDATE (mask all candidate tracks, 0 queries)
- B2: Geometry Only (mask if world displacement >= 0.30 m, 0 queries)
- B3: VLM All (query VLM on 100% of candidate tracks with visual context)
- B4: Random Routing (budget-matched to B6: exactly K_B4 = K_B6 queries)
- B5: Confidence-Threshold Routing (query high-uncertainty / ambiguous tracks)
- B6: Estimator-Aware Selective Routing (Proposed: query when calibrated utility U(q) >= 0)

Outputs:
- outputs/vlm_inference/policy_comparison.json (full cross-policy evaluation with dual latency & cost)
- outputs/vlm_inference/routing_decisions.json (B6 decisions consumed by run_pass_b_slam.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from estimator_aware_vlm.calibration.counterfactual import CalibratedUtilityModel
from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.policies import (
    CachedVLMProvider,
    LiveVLMProvider,
    VLMProvider,
    run_fair_policy_comparison_suite,
)


def load_candidate_tracks(tracks_summary_path: Path) -> List[Dict[str, Any]]:
    """Load candidate tracks from perception summary."""
    if not tracks_summary_path.exists():
        raise FileNotFoundError(f"Track summary not found: {tracks_summary_path}")
    with open(tracks_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tracks" in data:
        return data["tracks"]
    return data


def resolve_vlm_provider(
    mode: str,
    cached_telemetry_path: Path,
    server_url: str = "http://127.0.0.1:8089/v1/chat/completions",
) -> VLMProvider:
    """Instantiate appropriate VLM provider based on execution mode."""
    cached_dict = {}
    if cached_telemetry_path.exists():
        try:
            with open(cached_telemetry_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                cached_dict = prev_data.get("vlm_telemetry", {})
        except Exception as e:
            print(f"Warning: Failed to load cached telemetry from {cached_telemetry_path}: {e}")

    if mode == "live":
        print(f"Initializing Live VLM Provider targeting: {server_url}")
        return LiveVLMProvider(
            server_url=server_url,
            cache_fallback=cached_dict,
        )
    else:
        print(f"Initializing Replay VLM Provider with {len(cached_dict)} pre-recorded entities from {cached_telemetry_path.name}")
        return CachedVLMProvider(cached_results=cached_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute comparative routing policies (B0–B6) with cost accounting.")
    parser.add_argument("--tracks-summary", type=str, default=None, help="Path to tracks_summary.json")
    parser.add_argument("--calibrated-model", type=str, default=None, help="Path to calibrated_utility_model.json")
    parser.add_argument("--cached-telemetry", type=str, default=None, help="Path to cached policy_comparison.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for VLM inference results")
    parser.add_argument("--dataset-name", type=str, default="fr3_walking_xyz", help="Dataset name for reporting")
    parser.add_argument("--mode", choices=["auto", "replay", "live"], default="auto", help="Execution mode")
    parser.add_argument("--lambda-cost", type=float, default=0.45, help="Lambda cost penalty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for B4 budget matching")
    args = parser.parse_args()

    repo_root = get_repo_root()

    # 1. Resolve tracks
    if args.tracks_summary:
        summary_path = Path(args.tracks_summary)
    else:
        summary_path = repo_root / "outputs" / "perception_tracks" / "tracks_summary.json"
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    tracks = load_candidate_tracks(summary_path)
    print(f"Loaded {len(tracks)} candidate tracks from {summary_path}")

    # 2. Resolve Calibrated Utility Model
    calibrated_model: Optional[CalibratedUtilityModel] = None
    calib_path = Path(args.calibrated_model) if args.calibrated_model else (repo_root / "outputs" / "calibration" / "calibrated_utility_model.json")
    if not calib_path.is_absolute():
        calib_path = repo_root / calib_path
    if calib_path.exists():
        print(f"Loading empirical calibrated utility model from: {calib_path}")
        calibrated_model = CalibratedUtilityModel.load_json(calib_path)
        print(f"  Calibrated Model: LOOCV MAE = {calibrated_model.loocv_mae:.4f} m, R2 = {calibrated_model.loocv_r2:.4f}")
    else:
        print(f"Notice: Calibrated model not found at {calib_path}. Using uncalibrated proxy fallback.")

    # 3. Resolve VLM Provider
    cached_path = Path(args.cached_telemetry) if args.cached_telemetry else (repo_root / "outputs" / "vlm_inference" / "policy_comparison.json")
    if not cached_path.is_absolute():
        cached_path = repo_root / cached_path

    mode = args.mode
    if mode == "auto":
        mode = "replay" if cached_path.exists() else "live"

    vlm_provider = resolve_vlm_provider(mode=mode, cached_telemetry_path=cached_path)

    # 4. Execute Complete Policy Suite (B0 through B6)
    print("\nBeginning Policy Execution (B0 through B6)...")
    t_start = time.perf_counter()
    policy_results = run_fair_policy_comparison_suite(
        tracks=tracks,
        vlm_provider=vlm_provider,
        calibrated_model=calibrated_model,
        lambda_cost=args.lambda_cost,
        random_seed=args.seed,
    )
    elapsed_total_s = time.perf_counter() - t_start
    print(f"Completed policy suite execution in {elapsed_total_s:.3f} s.\n")

    # 5. Extract telemetry for export
    vlm_telemetry = {}
    if isinstance(vlm_provider, CachedVLMProvider):
        vlm_telemetry = {tid: q.to_dict() for tid, q in vlm_provider.cached.items()}
    elif isinstance(vlm_provider, LiveVLMProvider):
        vlm_telemetry = {tid: q.to_dict() for tid, q in vlm_provider.session_cache.items()}

    export_payload = {
        "dataset_name": args.dataset_name,
        "total_candidate_tracks": len(tracks),
        "execution_mode": mode,
        "lambda_cost": args.lambda_cost,
        "random_seed": args.seed,
        "calibrated_utility_used": calibrated_model is not None,
        "policies": {pkey: summary.to_dict() for pkey, summary in policy_results.items()},
        "vlm_telemetry": vlm_telemetry,
    }

    # 6. Save outputs
    out_dir = Path(args.output_dir) if args.output_dir else (repo_root / "outputs" / "vlm_inference")
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "policy_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    decisions_path = out_dir / "routing_decisions.json"
    b6_dict = policy_results["B6_Estimator_Aware"].to_dict()
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(b6_dict, f, indent=2)

    # 7. Print Comprehensive Fairness and Cost Summary Table
    print("=" * 115)
    print(f"                 FAIR POLICY COMPARISON AND COST ACCOUNTING ON {args.dataset_name.upper()}         ")
    print("=" * 115)
    header = f"{'Policy':<25} | {'Queries':<9} | {'Obs Lat (s)':<13} | {'Wire Lat (s)':<13} | {'Tokens':<9} | {'Masked':<8} | {'Retained':<8} | {'Budget Matched'}"
    print(header)
    print("-" * 115)
    for pkey, summary in policy_results.items():
        obs_s = summary.total_observed_latency_ms / 1000.0
        wire_s = summary.total_hypothetical_wire_latency_ms / 1000.0
        tot_tok = summary.total_prompt_tokens + summary.total_completion_tokens
        matched = summary.budget_matched_to or "-"
        print(f"{summary.policy_name:<25} | {summary.query_count:<9} | {obs_s:<13.3f} | {wire_s:<13.2f} | {tot_tok:<9} | {summary.masked_count:<8} | {summary.retained_count:<8} | {matched}")
    print("=" * 115)
    print(f"Exported policy comparison to: {report_path}")
    print(f"Exported B6 decisions to:      {decisions_path}\n")


if __name__ == "__main__":
    main()
