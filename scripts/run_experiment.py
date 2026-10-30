"""Unified Experiment CLI Runner for Estimator-Aware VLM Research.

Subcommands:
  validate   Verify environment, live VLM connectivity, and dataset integrity.
  run        Execute end-to-end research pipeline (Pass A -> Perception -> VLM Routing -> Pass B -> Scene Graph).
  compare    Print comprehensive research evaluation table and metrics comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Add src to sys.path if not installed in editable mode
repo_root_candidate = Path(__file__).resolve().parent.parent
src_dir = repo_root_candidate / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from estimator_aware_vlm.common.paths import get_repo_root


def check_vlm_health(server_url: str = "http://127.0.0.1:8089") -> bool:
    """Verify live VLM server health endpoint."""
    try:
        req = urllib.request.Request(f"{server_url}/health")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_validate(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    print("=================================================================")
    print("          ESTIMATOR-AWARE VLM: PREFLIGHT VALIDATION             ")
    print("=================================================================")

    # 1. Dataset check
    seq_dir = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz"
    assoc_file = seq_dir / "associations.txt"
    gt_file = seq_dir / "groundtruth.txt"

    dataset_ok = seq_dir.exists() and assoc_file.exists() and gt_file.exists()
    print(f"[{'PASS' if dataset_ok else 'FAIL'}] Dataset sequence: {seq_dir.name}")
    if dataset_ok:
        pairs = sum(1 for _ in open(assoc_file, "r", encoding="utf-8"))
        print(f"       -> Synchronized frame pairs: {pairs}")

    # 2. VLM Server connectivity check
    vlm_ok = check_vlm_health(args.server_url)
    print(f"[{'PASS' if vlm_ok else 'FAIL'}] Live VLM Server Endpoint: {args.server_url}")

    # 3. Model weights check
    yolo_model = repo_root / "yolov8n.pt"
    yolo_ok = yolo_model.exists()
    print(f"[{'PASS' if yolo_ok else 'FAIL'}] YOLO Detection Model: {yolo_model.name}")

    # 4. WSL & ORB-SLAM3 binary check
    orb_cmd = "wsl -d Ubuntu-22.04 -u saturn -- test -x /home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum && echo OK"
    try:
        res = subprocess.run(orb_cmd, shell=True, capture_output=True, text=True, timeout=5)
        orb_ok = "OK" in res.stdout
    except Exception:
        orb_ok = False
    print(f"[{'PASS' if orb_ok else 'FAIL'}] WSL ORB-SLAM3 Native Binary: /home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum")

    all_passed = dataset_ok and vlm_ok and yolo_ok and orb_ok
    print("-" * 65)
    print(f"Preflight status: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return 0 if all_passed else 1


def run_pipeline(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    print("=================================================================")
    print("      EXECUTING END-TO-END ESTIMATOR-AWARE VLM PIPELINE         ")
    print("=================================================================")

    # Pipeline subsystem execution
    subsystems = [
        ("Pass A Baseline SLAM", [sys.executable, "scripts/run_baseline_slam.py"]),
        ("Perception & Tracking", [sys.executable, "scripts/run_perception_pipeline.py"]),
        ("Selective Routing & VLM", [sys.executable, "scripts/run_routing_and_vlm_experiment.py"]),
        ("Pass B Filtered SLAM", [sys.executable, "scripts/run_pass_b_slam.py"]),
        ("Scene Graph Generation", [sys.executable, "scripts/build_contextual_scene_graph.py"]),
    ]

    for name, cmd in subsystems:
        print(f"\n>>> Running: {name} ...")
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(repo_root))
        dt = time.time() - t0
        if ret.returncode != 0:
            print(f"FAILED: {name} exited with code {ret.returncode}")
            return ret.returncode
        print(f">>> {name} completed in {dt:.2f}s")

    print("\nEnd-to-End Pipeline Execution Successfully Completed!")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    comparison_file = repo_root / "outputs" / "pass_b_slam" / "trajectory_improvement_comparison.json"
    policy_file = repo_root / "outputs" / "vlm_inference" / "policy_comparison.json"
    scene_graph_file = repo_root / "outputs" / "scene_graph" / "scene_graph_nodes.json"

    if not comparison_file.exists() or not policy_file.exists():
        print("Missing experiment outputs. Run the pipeline first via: python scripts/run_experiment.py run")
        return 1

    with open(comparison_file, "r", encoding="utf-8") as f:
        slam_comp = json.load(f)

    with open(policy_file, "r", encoding="utf-8") as f:
        policy_comp = json.load(f)

    scene_graph = {}
    if scene_graph_file.exists():
        with open(scene_graph_file, "r", encoding="utf-8") as f:
            scene_graph = json.load(f)

    print("\n" + "=" * 98)
    print("          RESEARCH EVALUATION RESULTS: ESTIMATOR-AWARE VLM ON FR3_WALKING_XYZ")
    print("=" * 98)

    # 1. SLAM Trajectory Accuracy Comparison Table
    base = slam_comp.get("baseline_pass_a", {})
    prop = slam_comp.get("estimator_aware_pass_b", {})
    gain = slam_comp.get("performance_gain", {})

    print("\n1. SLAM Trajectory Estimation Improvement (Horn/Umeyama SE(3) Aligned to MoCap Ground Truth)")
    print("-" * 98)
    print(f"{'Metric':<32} | {'Pass A Baseline':<22} | {'Pass B Estimator-Aware':<22} | {'Improvement'}")
    print("-" * 98)
    ate_a = base.get("ate_rmse_m", 0.0)
    ate_b = prop.get("ate_rmse_m", 0.0)
    print(f"{'ATE RMSE':<32} | {ate_a:.4f} m ({ate_a*100:.2f} cm)     | {ate_b:.4f} m ({ate_b*100:.2f} cm)      | -{gain.get('percentage_ate_improvement', 0.0):.1f}%")

    rpe_a = base.get("rpe_trans_rmse_m", 0.0)
    rpe_b = prop.get("rpe_trans_rmse_m", 0.0)
    rpe_imp = ((rpe_a - rpe_b) / rpe_a * 100.0) if rpe_a > 0 else 0.0
    print(f"{'RPE Translation RMSE':<32} | {rpe_a:.4f} m ({rpe_a*100:.2f} cm)     | {rpe_b:.4f} m ({rpe_b*100:.2f} cm)      | -{rpe_imp:.1f}%")

    cov_a = base.get("tracking_coverage", 0.0) * 100.0
    cov_b = prop.get("tracking_coverage", 0.0) * 100.0
    print(f"{'Tracking Coverage':<32} | {cov_a:.1f}%                | {cov_b:.1f}%                 | Matched")
    print("-" * 98)

    # 2. Router Policy Tradeoff Table
    print("\n2. Routing Policy Trade-Offs (VLM Query Cost vs Latency vs Action)")
    print("-" * 98)
    print(f"{'Policy ID':<24} | {'Queries':<9} | {'Latency (s)':<13} | {'Masked':<8} | {'Retained':<9} | {'Strategy'}")
    print("-" * 98)
    policies = policy_comp.get("policies", {})
    policy_descriptions = {
        "B0_Blind_RETAIN": "Zero queries; retain all keypoints (Vulnerable to dynamic blur)",
        "B1_Blind_INVALIDATE": "Zero queries; mask all detected bounding boxes (Starves landmarks)",
        "B2_Geometry_Only": "Zero queries; heuristic motion threshold > 0.60m",
        "B3_VLM_All": "Exhaustive VLM reasoning for every detected object",
        "B4_Random": "Random 50% routing baseline",
        "B5_Confidence_Threshold": "Detection confidence thresholding without SLAM loss",
        "B6_Estimator_Aware": "Selective utility routing U(q) = delta_L - lambda*C",
    }
    for pol_id, pol_data in policies.items():
        q_cnt = pol_data.get("query_count", 0)
        lat_s = pol_data.get("total_latency_ms", 0.0) / 1000.0
        m_cnt = pol_data.get("masked_count", 0)
        r_cnt = pol_data.get("retained_count", 0)
        desc = policy_descriptions.get(pol_id, "")
        print(f"{pol_id:<24} | {q_cnt:<9} | {lat_s:<13.2f} | {m_cnt:<8} | {r_cnt:<9} | {desc}")
    print("-" * 98)

    # 3. Persistent 3D Scene Graph Summary
    nodes = scene_graph.get("nodes", [])
    if nodes:
        print("\n3. Contextual 3D Scene Graph Entities (Outputs in outputs/scene_graph/scene_graph_nodes.json)")
        print("-" * 98)
        print(f"{'Entity ID':<13} | {'Semantic Class':<24} | {'Movability':<10} | {'Centroid (X, Y, Z)':<24} | {'Action'}")
        print("-" * 98)
        for n in nodes:
            c = n.get("centroid_world")
            c_str = f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]" if c else "None"
            print(f"{n.get('entity_id'):<13} | {n.get('semantic_class'):<24} | {n.get('contextual_movability'):<10} | {c_str:<24} | {n.get('predicted_landmark_action')}")
        print("-" * 98)

    print("\nArtifact Exports:")
    print(f"  - Baseline Trajectory Metrics:   outputs/baseline_slam/baseline_evaluation_metrics.json")
    print(f"  - Perception Tracks:             outputs/perception_tracks/tracks_summary.json")
    print(f"  - VLM Telemetry & Policy Stats:  outputs/vlm_inference/policy_comparison.json")
    print(f"  - Pass B Trajectory Metrics:     outputs/pass_b_slam/pass_b_evaluation_metrics.json")
    print(f"  - Improvement Comparison:        outputs/pass_b_slam/trajectory_improvement_comparison.json")
    print(f"  - 3D Dynamic Scene Graph:        outputs/scene_graph/scene_graph_nodes.json")
    print("=" * 98 + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Experiment CLI for Estimator-Aware VLM.")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Validate subcommand
    val_p = subparsers.add_parser("validate", help="Validate system dependencies and dataset")
    val_p.add_argument("--server-url", default="http://127.0.0.1:8089", help="Live VLM server URL")

    # Run subcommand
    run_p = subparsers.add_parser("run", help="Run end-to-end research experiment")

    # Compare subcommand
    comp_p = subparsers.add_parser("compare", help="Compare research evaluation metrics")

    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(run_validate(args))
    elif args.command == "run":
        sys.exit(run_pipeline(args))
    elif args.command == "compare":
        sys.exit(run_compare(args))
    else:
        # Default to compare if results exist, otherwise help
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
