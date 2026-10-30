#!/usr/bin/env python3
"""Unified Master Orchestrator: Paper-Grade Benchmark Suite.

Implements three distinct, auditable benchmark evaluation protocols:
1. 'visual-live':
   Evaluates visual physical movability classification for Qwen2.5-VL-3B
   using live over-the-wire inference. Records strict wire records, dual hashing, and latencies.
2. 'policy-replay':
   Evaluates estimator-aware router policies and downstream SLAM effects deterministically using
   immutable pre-recorded visual predictions. Strictly enforces copying of source prediction files
   into the run folder for self-contained, reproducible artifacts. Requires explicit --replay-source-dir.
3. 'router-live':
   Executes dynamic routing with live VLM queries invoked selectively by the router policy.

All protocols generate a comprehensive, tamper-evident run_manifest.json recording git commit SHA,
model digests, manifest checksums, parameters, and full SHA-256 hashes of all output artifacts.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Literal, Optional

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.evaluate_visual_classification import evaluate_visual_classification
from scripts.evaluate_router_policies import evaluate_router_policies, POLICY_KEYS
from scripts.evaluate_downstream_lifecycle import evaluate_downstream_lifecycle
from estimator_aware_vlm.statistical_analysis import run_statistical_analysis


def compute_file_sha256(path: Path) -> str:
    """Compute deterministic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash(repo_dir: Path) -> str:
    """Get current git commit hash if git repository is available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_UNTRACKED_REVISION"


def generate_run_manifest(
    run_id: str,
    protocol: str,
    output_dir: Path,
    args_dict: Dict[str, Any],
    elapsed_seconds: float,
    status: str = "COMPLETED",
    inputs_manifest: Optional[Path] = None,
    gt_manifest: Optional[Path] = None,
) -> Path:
    """Build unified, auditable run manifest recording all inputs, models, and outputs."""
    in_manifest_path = (repo_root / inputs_manifest).resolve() if inputs_manifest and not Path(inputs_manifest).is_absolute() else (Path(inputs_manifest).resolve() if inputs_manifest else (repo_root / "benchmarks" / "paper_benchmark_inputs.json"))
    gt_manifest_path = (repo_root / gt_manifest).resolve() if gt_manifest and not Path(gt_manifest).is_absolute() else (Path(gt_manifest).resolve() if gt_manifest else (repo_root / "benchmarks" / "paper_benchmark_ground_truth.json"))
    model_reg = repo_root / "benchmarks" / "model_registry.json"

    # Compute checksums for benchmark inputs
    input_manifests = {
        "inputs_json": {
            "path": str(in_manifest_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": compute_file_sha256(in_manifest_path) if in_manifest_path.exists() else None,
        },
        "ground_truth_json": {
            "path": str(gt_manifest_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": compute_file_sha256(gt_manifest_path) if gt_manifest_path.exists() else None,
        },
        "model_registry_json": {
            "path": str(model_reg.relative_to(repo_root)).replace("\\", "/"),
            "sha256": compute_file_sha256(model_reg) if model_reg.exists() else None,
        },
    }

    # Load model registry entries for inspected models
    models_info = {}
    if model_reg.exists():
        with open(model_reg, "r", encoding="utf-8") as f:
            full_reg = json.load(f)
            models_info = full_reg.get("models", {})

    # Compute checksums of all output files in output_dir recursively (excluding run_manifest itself)
    output_files: Dict[str, Dict[str, Any]] = {}
    for fpath in sorted(output_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "run_manifest.json":
            rel_p = str(fpath.relative_to(repo_root)).replace("\\", "/")
            rel_to_out = str(fpath.relative_to(output_dir)).replace("\\", "/")
            output_files[rel_to_out] = {
                "relative_path": rel_p,
                "size_bytes": fpath.stat().st_size,
                "sha256": compute_file_sha256(fpath),
            }

    manifest = {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "protocol": protocol,
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_hash(repo_root),
        "execution_parameters": args_dict,
        "input_manifests": input_manifests,
        "registered_models": models_info,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
        "output_artifacts": output_files,
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def run_paper_benchmark(
    protocol: Literal["visual-live", "policy-replay", "router-live"] = "policy-replay",
    models: List[str] = ["qwen2_5_vl_3b"],
    view_modes: List[str] = ["CROP_ONLY", "CROP_PLUS_CONTEXT"],
    task_mode: Literal["class-conditioned", "unconditioned"] = "class-conditioned",
    server_url: Optional[str] = None,
    replay_source_dir: Optional[Path] = None,
    policies: List[str] = POLICY_KEYS,
    output_dir: Optional[Path] = None,
    skip_offline: bool = True,
    inputs_manifest: Optional[Path] = None,
    ground_truth_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute paper-grade benchmark across specified protocol."""
    t0 = time.time()

    if server_url:
        os.environ["QWEN_SERVER_URL"] = server_url

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        run_id = output_dir.name
    else:
        run_id = f"paper_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = repo_root / "outputs" / "benchmark_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MASTER ORCHESTRATOR: ESTIMATOR-AWARE SELECTIVE VLM BENCHMARK")
    print(f"Run ID: {run_id}")
    print(f"Protocol: {protocol.upper()}")
    print(f"Target Models: {models}")
    print(f"View Modes: {view_modes}")
    print(f"Task Mode: {task_mode}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    results: Dict[str, Any] = {
        "run_id": run_id,
        "protocol": protocol,
        "output_dir": str(output_dir.relative_to(repo_root)).replace("\\", "/"),
    }

    # PROTOCOL 1: visual-live
    if protocol == "visual-live":
        print("\n[PROTOCOL: visual-live] Executing live multimodal visual classification...")
        vis_summary = evaluate_visual_classification(
            model_keys=models,  # type: ignore
            view_modes=view_modes,  # type: ignore
            task_mode=task_mode,
            run_dir=output_dir,
            skip_offline_models=skip_offline,
            inputs_manifest_path=inputs_manifest,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["visual_summary"] = vis_summary

    # PROTOCOL 2: policy-replay
    elif protocol == "policy-replay":
        print("\n[PROTOCOL: policy-replay] Deterministic policy evaluation from recorded predictions...")
        if replay_source_dir is None:
            raise ValueError(
                "Protocol 'policy-replay' requires an explicit --replay-source-dir containing prior prediction files. "
                "Automatic fuzzy-finding of past directories is forbidden for paper-grade reproducibility."
            )
        replay_source_dir = Path(replay_source_dir).resolve()
        if not replay_source_dir.exists() or not replay_source_dir.is_dir():
            raise FileNotFoundError(f"Replay source directory does not exist: {replay_source_dir}")

        # Verify and copy immutable prediction files into run folder
        copied_files = 0
        for pattern in ["visual_classification_predictions_*.jsonl", "visual_classification_summary.json"]:
            for src_f in replay_source_dir.glob(pattern):
                dest_f = output_dir / src_f.name
                shutil.copy2(src_f, dest_f)
                copied_files += 1
                print(f"  [COPIED] {src_f.name} -> {output_dir.name}/")

        if copied_files == 0:
            raise FileNotFoundError(
                f"No visual_classification_predictions_*.jsonl files found in replay source: {replay_source_dir}"
            )

        print("\n[STEP 1/3] Executing Axis 2: Router Policy Suite & Comparative Baselines...")
        router_summary = evaluate_router_policies(
            policy_keys=policies,
            model_key=models[0],  # type: ignore
            execution_mode="replay",
            task_mode=task_mode,
            replay_run_dir=output_dir,
            run_dir=output_dir,
            inputs_manifest_path=inputs_manifest,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["router_summary"] = router_summary

        print("\n[STEP 2/3] Executing Axis 3: Downstream SLAM Landmark Lifecycle...")
        lifecycle_summary = evaluate_downstream_lifecycle(
            router_run_dir=output_dir,
            output_run_dir=output_dir,
            inputs_manifest_path=inputs_manifest,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["lifecycle_summary"] = lifecycle_summary

        print("\n[STEP 3/3] Executing Axis 4: Statistical Analysis & Programmatic Reporting...")
        stats_summary = run_statistical_analysis(
            router_run_dir=output_dir,
            output_dir=output_dir,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["statistical_summary"] = stats_summary

    # PROTOCOL 3: router-live
    elif protocol == "router-live":
        print("\n[PROTOCOL: router-live] Executing dynamic live router evaluation...")
        print("\n[STEP 1/3] Executing Axis 2: Live Router Policies with Dynamic Dispatch...")
        router_summary = evaluate_router_policies(
            policy_keys=policies,
            model_key=models[0],  # type: ignore
            execution_mode="live",
            task_mode=task_mode,
            run_dir=output_dir,
            inputs_manifest_path=inputs_manifest,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["router_summary"] = router_summary

        print("\n[STEP 2/3] Executing Axis 3: Downstream SLAM Landmark Lifecycle...")
        lifecycle_summary = evaluate_downstream_lifecycle(
            router_run_dir=output_dir,
            output_run_dir=output_dir,
            inputs_manifest_path=inputs_manifest,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["lifecycle_summary"] = lifecycle_summary

        print("\n[STEP 3/3] Executing Axis 4: Statistical Analysis & Programmatic Reporting...")
        stats_summary = run_statistical_analysis(
            router_run_dir=output_dir,
            output_dir=output_dir,
            ground_truth_manifest_path=ground_truth_manifest,
        )
        results["statistical_summary"] = stats_summary

    else:
        raise ValueError(f"Unknown benchmark protocol: {protocol}")

    elapsed_s = time.time() - t0
    results["elapsed_seconds"] = round(elapsed_s, 2)

    # Generate unified run_manifest.json
    manifest_p = generate_run_manifest(
        run_id=run_id,
        protocol=protocol,
        output_dir=output_dir,
        args_dict={
            "protocol": protocol,
            "models": models,
            "view_modes": view_modes,
            "task_mode": task_mode,
            "replay_source_dir": str(replay_source_dir) if replay_source_dir else None,
            "policies": policies,
            "skip_offline": skip_offline,
            "inputs_manifest": str(inputs_manifest) if inputs_manifest else None,
            "ground_truth_manifest": str(ground_truth_manifest) if ground_truth_manifest else None,
        },
        elapsed_seconds=elapsed_s,
        inputs_manifest=inputs_manifest,
        gt_manifest=ground_truth_manifest,
    )
    results["run_manifest"] = str(manifest_p.relative_to(repo_root)).replace("\\", "/")

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY in {elapsed_s:.1f}s")
    print(f"Master Output Run Directory: {output_dir}")
    print(f"Unified Run Manifest: {manifest_p.name}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete paper-grade benchmark orchestrator.")
    parser.add_argument(
        "--protocol",
        default="policy-replay",
        choices=["visual-live", "policy-replay", "router-live"],
        help="Benchmark protocol: visual-live, policy-replay, or router-live",
    )
    # Backward compatibility alias
    parser.add_argument(
        "--mode",
        default=None,
        choices=["live", "replay"],
        help="Legacy alias: 'live' -> 'visual-live', 'replay' -> 'policy-replay'",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2_5_vl_3b"],
        choices=["qwen2_5_vl_3b"],
    )
    parser.add_argument(
        "--view-modes",
        nargs="+",
        default=["CROP_ONLY", "CROP_PLUS_CONTEXT"],
        choices=["CROP_ONLY", "CROP_PLUS_CONTEXT", "MULTI_VIEW"],
    )
    parser.add_argument(
        "--task-mode",
        default="class-conditioned",
        choices=["class-conditioned", "unconditioned"],
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Explicit server endpoint URL for live VLM serving (e.g. http://127.0.0.1:8089/v1/chat/completions)",
    )
    parser.add_argument(
        "--replay-source-dir",
        type=Path,
        default=None,
        help="Required directory containing source prediction .jsonl files for policy-replay protocol",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICY_KEYS,
        help="Subset of router policies to evaluate",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--inputs-manifest", type=Path, default=None, help="Path to custom inputs manifest JSON")
    parser.add_argument("--ground-truth-manifest", type=Path, default=None, help="Path to custom ground-truth manifest JSON")
    args = parser.parse_args()

    # Handle legacy --mode alias if provided
    selected_protocol = args.protocol
    if args.mode is not None:
        selected_protocol = "visual-live" if args.mode == "live" else "policy-replay"

    run_paper_benchmark(
        protocol=selected_protocol,
        models=args.models,
        view_modes=args.view_modes,
        task_mode=args.task_mode,
        server_url=args.server_url,
        replay_source_dir=args.replay_source_dir,
        policies=args.policies,
        output_dir=args.output_dir,
        skip_offline=args.skip_offline,
        inputs_manifest=args.inputs_manifest,
        ground_truth_manifest=args.ground_truth_manifest,
    )
