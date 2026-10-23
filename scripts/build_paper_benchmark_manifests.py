#!/usr/bin/env python3
"""Build decoupled paper benchmark inputs and ground truth manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from PIL import Image

import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from estimator_aware_vlm.evidence_adapter import Stage5EvidenceAdapter


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_decoupled_manifests():
    src_manifest = repo_root / "benchmarks" / "router_evaluation_manifest.json"
    with open(src_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data["evaluation_instances"]
    adapter = Stage5EvidenceAdapter(repo_root=repo_root)

    inputs_list = []
    ground_truth_map = {}

    for inst in instances:
        eid = inst["entity_id"]
        scls = inst["semantic_class"]
        has_imgs = bool(inst.get("has_images", False))
        crops_dir_rel = inst.get("crops_dir")

        crop_paths = []
        scene_paths = []
        img_hashes = []
        usable_views = 0

        if has_imgs and crops_dir_rel:
            cdir = repo_root / crops_dir_rel
            if cdir.exists():
                v_count = inst.get("views_count", 1)
                for v_idx in range(v_count):
                    c_p = cdir / f"crop_view_{v_idx}.jpg"
                    s_p = cdir / f"scene_view_{v_idx}.jpg"
                    if c_p.exists() and s_p.exists():
                        try:
                            with Image.open(c_p) as im:
                                im.verify()
                            with Image.open(s_p) as im:
                                im.verify()
                            crop_paths.append(str(c_p.relative_to(repo_root)).replace("\\", "/"))
                            scene_paths.append(str(s_p.relative_to(repo_root)).replace("\\", "/"))
                            img_hashes.append(hash_file(c_p))
                            img_hashes.append(hash_file(s_p))
                            usable_views += 1
                        except Exception as e:
                            print(f"Image decode error for {eid}: {e}")

        # Extract genuine estimator evidence strictly from SLAM/pre-router evidence registration & observations
        est_evidence = adapter.build_instance_evidence(inst)
        estimator_block = {
            "observed_motion": est_evidence["observed_motion"],
            "measurement_status": est_evidence["measurement_status"],
            "measured_displacement_m": est_evidence["world_displacement_m"],
            "displacement_uncertainty_m": est_evidence["displacement_uncertainty_m"],
            "displacement_uncertainty_heuristic_m": est_evidence["displacement_uncertainty_heuristic_m"],
            "relocation_evaluable": est_evidence["relocation_evaluable"],
            "depth_valid": est_evidence["depth_valid"],
            "has_valid_pose": est_evidence["has_valid_pose"],
            "semantic_entropy": est_evidence["semantic_entropy"],
            "geometric_consistency_score": est_evidence["geometric_consistency_score"],
            "evidence_provenance": est_evidence["evidence_provenance"],
            "observation_count": est_evidence["s1_obs_count"] + est_evidence["s2_obs_count"],
            "s1_obs_count": est_evidence["s1_obs_count"],
            "s2_obs_count": est_evidence["s2_obs_count"],
            "registration_status": est_evidence["registration_status"],
        }

        # 1. Pure Inference-Visible Input Record (ZERO GT fields, zero expected decisions)
        input_record = {
            "physical_instance_id": eid,
            "semantic_class": scls,
            "source_session": "semantic_mapping_my_desk_recording_01" if inst.get("s1_obs_id") else "semantic_mapping_my_desk_recording_02",
            "source_obs_ids": [oid for oid in [inst.get("s1_obs_id"), inst.get("s2_obs_id")] if oid],
            "estimator_evidence": estimator_block,
            "visual_inputs": {
                "has_images": (usable_views > 0),
                "usable_views": usable_views,
                "crop_view_paths": crop_paths,
                "scene_view_paths": scene_paths,
                "image_sha256": img_hashes,
            },
            "split": "PILOT_BENCHMARK_FROZEN",
        }
        inputs_list.append(input_record)

        # 2. Evaluator-Only Ground Truth Record (strictly separated)
        obs_motion = inst.get("observed_motion_gt", "STATIC")
        movability = inst["contextual_movability_gt"]

        landmark_action = "RETAIN_LANDMARK"
        if movability == "MOBILE":
            landmark_action = "INVALIDATE_DYNAMIC_AGENT"
        elif obs_motion == "DISPLACED":
            landmark_action = "INVALIDATE_RELOCATED"

        ground_truth_map[eid] = {
            "physical_instance_id": eid,
            "semantic_class": scls,
            "contextual_movability_gt": movability,
            "current_motion_gt": obs_motion,
            "slam_landmark_validity_gt": landmark_action,
            "displacement_ground_truth_status": "UNAVAILABLE_NO_OPTICAL_MOCAP",
            "annotation_provenance": "PROVENANCE_UNVERIFIED_PILOT",
            "adjudication_status": "PILOT_CONSENSUS_UNVERIFIED",
        }

    inputs_manifest = {
        "manifest_title": "Estimator-Aware Selective VLM Benchmark: Inference Inputs (Inference-Visible Only)",
        "manifest_version": "1.0.0",
        "description": "Inference-visible metadata only. Contains zero ground-truth fields, zero expected decisions, and zero expected reasons.",
        "dataset_tier": "PILOT_CASE_STUDY_15_INSTANCE",
        "dataset_scope_limitations": "In-domain pilot evaluation set of 15 physical instances (3 IMMOVABLE, 9 MOVABLE, 3 MOBILE, with 14 image-eligible and only 2 image-eligible MOBILE instances). Does not demonstrate broad multi-environment generalization.",
        "instances": inputs_list,
    }

    gt_manifest = {
        "ground_truth_title": "Estimator-Aware Selective VLM Benchmark: Evaluator Ground Truth (Evaluator-Only)",
        "ground_truth_version": "1.0.0",
        "description": "Evaluator-only ground truth labels keyed strictly by physical_instance_id. Never passed to inference pipeline.",
        "labels": ground_truth_map,
    }

    out_inputs = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    out_gt = repo_root / "benchmarks" / "paper_benchmark_ground_truth.json"

    with open(out_inputs, "w", encoding="utf-8") as f:
        json.dump(inputs_manifest, f, indent=2)

    with open(out_gt, "w", encoding="utf-8") as f:
        json.dump(gt_manifest, f, indent=2)

    print(f"Generated {out_inputs} ({len(inputs_list)} instances)")
    print(f"Generated {out_gt} ({len(ground_truth_map)} ground truth records)")


if __name__ == "__main__":
    build_decoupled_manifests()
