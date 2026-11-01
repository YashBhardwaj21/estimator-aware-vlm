"""
Audit evidence inventory generation script.
Scans the local filesystem for all relevant trajectories, baselines, masks, manifests,
and raw outputs to build results/publication_campaign/run_20261101_execution_audit/audit/evidence_inventory.json.
"""

import os
import json
import hashlib
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(REPO_ROOT, "results", "publication_campaign", "run_20261101_execution_audit")
OUTPUT_JSON = os.path.join(AUDIT_DIR, "audit", "evidence_inventory.json")

def compute_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_file_metadata(rel_path):
    full_path = os.path.join(REPO_ROOT, rel_path)
    exists = os.path.exists(full_path)
    if not exists:
        return {
            "relative_path": rel_path,
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
            "line_count": 0,
            "mtime": None
        }
    size_bytes = os.path.getsize(full_path)
    sha256_hash = compute_sha256(full_path)
    line_count = 0
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            line_count = sum(1 for _ in f)
    except Exception:
        pass
    mtime = time.ctime(os.path.getmtime(full_path))
    return {
        "relative_path": rel_path,
        "exists": True,
        "size_bytes": size_bytes,
        "sha256": sha256_hash,
        "line_count": line_count,
        "mtime": mtime
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # 1. Target baselines from run_20261101_execution
    baselines_to_check = [
        "results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_xyz/CameraTrajectory.txt",
        "results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_halfsphere/CameraTrajectory.txt",
        "results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_static/CameraTrajectory.txt",
        "results/publication_campaign/run_20261101_execution/baselines/OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt",
        "results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_crowd/CameraTrajectory.txt",
        "results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_moving_obstructing_box/CameraTrajectory.txt"
    ]
    
    # 2. Genuine intervention trajectories across repo
    interventions_to_check = [
        "results/independent_intervention_expansion/run_20261101_expansion/tum/individual_person_interventions/T0_baseline_KEEP/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/tum/individual_person_interventions/T1_person_02_POLYGON_ONLY/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/tum/individual_person_interventions/T2_person_04_POLYGON_ONLY/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/tum/individual_person_interventions/T3_both_persons_joint_POLYGON/CameraTrajectory.txt",
        "results/counterfactual_interventions/run_20261031_counterfactual/tum/interventions/tum_fr3_walking_xyz_monitor_01_IMAGE_POLYGON_MASK_r01/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/openloris/corridor1-4/C0_baseline_KEEP/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/openloris/corridor1-4/C1_person_01_IMAGE_POLYGON_MASK/CameraTrajectory.txt",
        "results/independent_intervention_expansion/run_20261101_expansion/openloris/corridor1-4/C2_person_01_IMAGE_BBOX_MASK/CameraTrajectory.txt"
    ]
    
    # 3. Factorial and Telemetry evidence
    factorial_to_check = [
        "results/native_estimator_integration/run_20261101_native/step10_factorial_comparison.json",
        "results/native_estimator_integration/run_20261101_native/native_paired_factorial_matrix.csv",
        "results/native_estimator_integration/run_20261101_native/telemetry_I0_baseline_control/optimizer_summary.json",
        "results/native_estimator_integration/run_20261101_native/telemetry_I1_image_mask_only/optimizer_summary.json",
        "results/native_estimator_integration/run_20261101_native/telemetry_I2_landmark_only_integration/optimizer_summary.json",
        "results/native_estimator_integration/run_20261101_native/telemetry_I3_dual_integration/optimizer_summary.json"
    ]
    
    # 4. Campaign artifacts under audit
    campaign_artifacts = [
        "results/publication_campaign/run_20261101_execution/CAMPAIGN_EVIDENCE_REPORT.md",
        "results/publication_campaign/run_20261101_execution/tables/candidate_intervention_results.csv",
        "results/publication_campaign/run_20261101_execution/tables/canonical_per_sequence_summary.csv",
        "results/publication_campaign/run_20261101_execution/tables/policy_comparison_r0_r8.csv",
        "results/publication_campaign/run_20261101_execution/tables/factorial_integration_results.csv",
        "results/publication_campaign/run_20261101_execution/tables/d455_lifecycle_policy_comparison.csv",
        "results/publication_campaign/run_20261101_execution/vlm_inference/live_qwen_inferences.json",
        "results/publication_campaign/run_20261101_execution/edrr_validation/grouped_validation_results.json"
    ]
    
    inventory = {
        "audit_meta": {
            "auditor": "Independent Research Quality Auditor",
            "campaign_run_audited": "results/publication_campaign/run_20261101_execution",
            "audit_target": "Forensic Verification & Publication Reconciliation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "baselines_physical_trajectories": [get_file_metadata(p) for p in baselines_to_check],
        "interventions_physical_trajectories": [get_file_metadata(p) for p in interventions_to_check],
        "native_factorial_evidence": [get_file_metadata(p) for p in factorial_to_check],
        "campaign_execution_artifacts": [get_file_metadata(p) for p in campaign_artifacts]
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"Inventory saved successfully to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

