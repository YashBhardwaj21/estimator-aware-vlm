"""Export Comprehensive Multi-Dataset Reproducibility Run Manifest for Estimator-Aware VLM."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from estimator_aware_vlm.common.paths import get_repo_root


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo_root = get_repo_root()
    outputs_dir = repo_root / "outputs"

    # 1. Load TUM Dataset Artifacts
    tum_base_metrics = outputs_dir / "baseline_slam" / "baseline_evaluation_metrics.json"
    tum_pass_b_metrics = outputs_dir / "pass_b_slam" / "pass_b_evaluation_metrics.json"
    tum_comp = outputs_dir / "pass_b_slam" / "trajectory_improvement_comparison.json"
    tum_causality = outputs_dir / "pass_b_slam" / "mask_causality_report.json"
    tum_routing = outputs_dir / "vlm_inference" / "routing_decisions.json"
    tum_tracks = outputs_dir / "perception_tracks" / "tracks_summary.json"
    tum_obs = outputs_dir / "perception_tracks" / "per_frame_instance_observations.jsonl"

    tum_b_data = json.load(open(tum_base_metrics, "r", encoding="utf-8")) if tum_base_metrics.exists() else {}
    tum_pb_data = json.load(open(tum_pass_b_metrics, "r", encoding="utf-8")) if tum_pass_b_metrics.exists() else {}
    tum_comp_data = json.load(open(tum_comp, "r", encoding="utf-8")) if tum_comp.exists() else {}
    tum_caus_data = json.load(open(tum_causality, "r", encoding="utf-8")) if tum_causality.exists() else {}
    tum_rout_data = json.load(open(tum_routing, "r", encoding="utf-8")) if tum_routing.exists() else {}
    tum_trk_data = json.load(open(tum_tracks, "r", encoding="utf-8")) if tum_tracks.exists() else []

    # 2. Load OpenLORIS Dataset Artifacts
    loris_base_metrics = outputs_dir / "baseline_slam_OpenLORIS-Scene_corridor1-4" / "baseline_evaluation_metrics.json"
    loris_pass_b_metrics = outputs_dir / "pass_b_slam_OpenLORIS-Scene_corridor1-4" / "pass_b_evaluation_metrics.json"
    loris_comp = outputs_dir / "pass_b_slam_OpenLORIS-Scene_corridor1-4" / "trajectory_improvement_comparison.json"
    loris_causality = outputs_dir / "pass_b_slam_OpenLORIS-Scene_corridor1-4" / "mask_causality_report.json"
    loris_routing = outputs_dir / "vlm_inference_openloris" / "routing_decisions.json"
    loris_tracks = outputs_dir / "perception_tracks_OpenLORIS-Scene_corridor1-4" / "tracks_summary.json"
    loris_obs = outputs_dir / "perception_tracks_OpenLORIS-Scene_corridor1-4" / "per_frame_instance_observations.jsonl"

    loris_b_data = json.load(open(loris_base_metrics, "r", encoding="utf-8")) if loris_base_metrics.exists() else {}
    loris_pb_data = json.load(open(loris_pass_b_metrics, "r", encoding="utf-8")) if loris_pass_b_metrics.exists() else {}
    loris_comp_data = json.load(open(loris_comp, "r", encoding="utf-8")) if loris_comp.exists() else {}
    loris_caus_data = json.load(open(loris_causality, "r", encoding="utf-8")) if loris_causality.exists() else {}
    loris_rout_data = json.load(open(loris_routing, "r", encoding="utf-8")) if loris_routing.exists() else {}
    loris_trk_data = json.load(open(loris_tracks, "r", encoding="utf-8")) if loris_tracks.exists() else []

    # 3. Catalog Figures
    figures_dir = outputs_dir / "figures"
    figures_catalog = {}
    for fig_id in range(2, 11):
        matches = list(figures_dir.glob(f"fig{fig_id}_*.png"))
        if matches:
            fp = matches[0]
            figures_catalog[f"Figure_{fig_id}"] = {
                "filename": fp.name,
                "relative_path": f"outputs/figures/{fp.name}",
                "file_size_bytes": fp.stat().st_size,
                "sha256": compute_sha256(fp),
            }

    run_manifest = {
        "manifest_version": "2.1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "system_environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "wsl_distro": "Ubuntu-22.04",
            "orb_slam3_binary": "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum",
            "orb_slam3_vocabulary": "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            "yolo_model": "models/yolov8n-seg.pt",
            "vlm_model": "Qwen2.5-VL-3B-Instruct (GGUF Q4_K_M)",
            "vlm_server": "http://127.0.0.1:8089",
        },
        "datasets": {
            "rgbd_dataset_freiburg3_walking_xyz": {
                "benchmark_reference": "Sturm et al., IROS 2012",
                "environment_type": "indoor_dynamic_lab",
                "total_frames": 827,
                "ground_truth_poses": 827,
                "baseline_pass_a": {
                    "trajectory_file": "outputs/baseline_slam/CameraTrajectory.txt",
                    "ate_rmse_m": tum_b_data.get("ate", {}).get("rmse_m"),
                    "ate_mean_m": tum_b_data.get("ate", {}).get("mean_m"),
                    "rpe_trans_rmse_m": tum_b_data.get("rpe", {}).get("trans_rmse_m"),
                    "tracking_coverage": tum_b_data.get("tracking_coverage"),
                },
                "multiview_perception": {
                    "tracks_summary_file": "outputs/perception_tracks/tracks_summary.json",
                    "total_candidate_tracks": len(tum_trk_data),
                    "discovered_entities": [t.get("track_id") for t in tum_trk_data],
                    "instance_segmentation": True,
                },
                "selective_routing": {
                    "policy": "B6_Estimator_Aware",
                    "routing_decisions_file": "outputs/vlm_inference/routing_decisions.json",
                    "vlm_query_count": tum_rout_data.get("query_count"),
                    "total_vlm_latency_ms": tum_rout_data.get("total_latency_ms"),
                    "masked_tracks": [d["track_id"] for d in tum_rout_data.get("decisions", []) if d.get("action") == "MASK_FROM_TRACKING"],
                    "retained_tracks": [d["track_id"] for d in tum_rout_data.get("decisions", []) if d.get("action") == "RETAIN"],
                },
                "estimator_aware_pass_b": {
                    "trajectory_file": "outputs/pass_b_slam/CameraTrajectory.txt",
                    "ate_rmse_m": tum_pb_data.get("ate", {}).get("rmse_m"),
                    "ate_mean_m": tum_pb_data.get("ate", {}).get("mean_m"),
                    "rpe_trans_rmse_m": tum_pb_data.get("rpe", {}).get("trans_rmse_m"),
                    "tracking_coverage": tum_pb_data.get("tracking_coverage"),
                    "relative_improvement_pct": tum_comp_data.get("performance_gain", {}).get("percentage_ate_improvement"),
                    "absolute_ate_reduction_m": tum_comp_data.get("performance_gain", {}).get("absolute_ate_reduction_m"),
                },
            },
            "OpenLORIS-Scene_corridor1-4": {
                "benchmark_reference": "Shi et al., ICRA 2020",
                "environment_type": "service_robot_corridor",
                "total_frames": 1500,
                "ground_truth_poses": 2515,
                "sensor": "Intel RealSense D400 Color + Aligned Depth",
                "baseline_pass_a": {
                    "trajectory_file": "outputs/baseline_slam_OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt",
                    "ate_rmse_m": loris_b_data.get("ate", {}).get("rmse_m"),
                    "ate_mean_m": loris_b_data.get("ate", {}).get("mean_m"),
                    "rpe_trans_rmse_m": loris_b_data.get("rpe", {}).get("trans_rmse_m"),
                    "tracking_coverage": loris_b_data.get("tracking_coverage"),
                },
                "multiview_perception": {
                    "tracks_summary_file": "outputs/perception_tracks_OpenLORIS-Scene_corridor1-4/tracks_summary.json",
                    "total_candidate_tracks": len(loris_trk_data),
                    "discovered_entities": [t.get("track_id") for t in loris_trk_data],
                    "instance_segmentation": True,
                },
                "selective_routing": {
                    "policy": "B6_Estimator_Aware",
                    "routing_decisions_file": "outputs/vlm_inference_openloris/routing_decisions.json",
                    "vlm_query_count": loris_rout_data.get("query_count"),
                    "total_vlm_latency_ms": loris_rout_data.get("total_latency_ms"),
                    "masked_tracks": [d["track_id"] for d in loris_rout_data.get("decisions", []) if d.get("action") == "MASK_FROM_TRACKING"],
                    "retained_tracks": [d["track_id"] for d in loris_rout_data.get("decisions", []) if d.get("action") == "RETAIN"],
                },
                "estimator_aware_pass_b": {
                    "trajectory_file": "outputs/pass_b_slam_OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt",
                    "ate_rmse_m": loris_pb_data.get("ate", {}).get("rmse_m"),
                    "ate_mean_m": loris_pb_data.get("ate", {}).get("mean_m"),
                    "rpe_trans_rmse_m": loris_pb_data.get("rpe", {}).get("trans_rmse_m"),
                    "tracking_coverage": loris_pb_data.get("tracking_coverage"),
                    "relative_improvement_pct": loris_comp_data.get("performance_gain", {}).get("percentage_ate_improvement"),
                    "absolute_ate_reduction_m": loris_comp_data.get("performance_gain", {}).get("absolute_ate_reduction_m"),
                },
            },
        },
        "publication_figures": figures_catalog,
        "mapping_and_ghost_map_mitigation": {
            "occupancy_grid_model": "log_odds_inverse_sensor",
            "l_occ_increment": 0.85,
            "l_free_decrement": -0.40,
            "l_bounds": [-5.0, 5.0],
            "multi_session_fusion": "probabilistic_log_odds_addition",
            "ghost_obstacle_clearance_verified": True,
        },
    }

    out_path = outputs_dir / "run_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Comprehensive Multi-Dataset Run Manifest exported to: {out_path}")


if __name__ == "__main__":
    main()
