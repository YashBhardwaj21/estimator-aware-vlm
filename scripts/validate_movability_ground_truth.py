#!/usr/bin/env python3
"""Validation Gate for Movability Ground Truth and Benchmark Data.

Ensures that every physical object instance, track mapping, and observation
meets rigorous structural, geometric, and identity standards before downstream evaluation stages consume them
or downstream evaluation stages consume them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from estimator_aware_vlm.common.paths import get_repo_root


def validate_movability_benchmark(root: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate movability_ground_truth.json and track mappings."""
    benchmark_path = root / "benchmarks" / "movability_ground_truth.json"
    track_map_path = root / "benchmarks" / "track_to_instance_ground_truth.json"
    fp_catalog_path = root / "benchmarks" / "false_positive_catalog.json"

    errors: List[str] = []
    warnings: List[str] = []

    if not benchmark_path.exists():
        return False, [f"Missing benchmark file: {benchmark_path}"], {}
    if not track_map_path.exists():
        return False, [f"Missing track mapping file: {track_map_path}"], {}

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    with track_map_path.open("r", encoding="utf-8") as f:
        track_map_data = json.load(f)
        track_map: Dict[str, str] = track_map_data.get("track_to_instance", {})

    instances: List[Dict[str, Any]] = benchmark.get("physical_objects", [])
    instance_ids = [obj["object_instance_id"] for obj in instances]

    # 1. Structural checks
    if len(instance_ids) != len(set(instance_ids)):
        duplicates = [x for x in instance_ids if instance_ids.count(x) > 1]
        errors.append(f"Duplicate physical object instance IDs found: {set(duplicates)}")

    allowed_movability = {"IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"}
    allowed_motion = {"STATIC", "MOVING", "DISPLACED", "DYNAMIC", "UNCERTAIN"}

    all_associated_tracks: Set[str] = set()

    for obj in instances:
        instance_id = obj.get("object_instance_id", "UNKNOWN_ID")
        tracks = obj.get("associated_tracks", [])

        # Check duplicate tracks within instance
        if len(tracks) != len(set(tracks)):
            errors.append(f"{instance_id}: duplicate tracks listed in associated_tracks: {tracks}")

        # Check tracks across instances (no track should map to multiple physical objects)
        for trk in tracks:
            if trk in all_associated_tracks:
                errors.append(f"{instance_id}: track '{trk}' is already assigned to another instance")
            all_associated_tracks.add(trk)

            # Check track_map alignment if present
            if trk in track_map:
                mapped = track_map[trk]
                if mapped != instance_id:
                    errors.append(
                        f"{instance_id}: track '{trk}' maps to '{mapped}' in track_to_instance_ground_truth.json"
                    )

        # Check taxonomy labels
        mov = obj.get("movability_gt")
        mot = obj.get("current_motion_gt")
        if mov not in allowed_movability:
            errors.append(f"{instance_id}: invalid movability label: {mov!r}")
        if mot not in allowed_motion:
            errors.append(f"{instance_id}: invalid motion label: {mot!r}")

        # Check centroid dimensions
        centroid = obj.get("centroid_floor_m")
        if not centroid or len(centroid) != 3 or any(not isinstance(v, (int, float)) or math.isnan(v) for v in centroid):
            errors.append(f"{instance_id}: invalid centroid_floor_m coordinates: {centroid}")

        # Check dimensions
        dims = obj.get("dimensions_m")
        if not dims or len(dims) != 3 or any(not isinstance(v, (int, float)) or v <= 0 or math.isnan(v) for v in dims):
            errors.append(f"{instance_id}: invalid dimensions_m: {dims}")

    # 2. Specific Drone Identity & Verification Tests
    drone = next((o for o in instances if o["object_instance_id"] == "inst_quadcopter_drone_001"), None)
    if drone:
        drone_tracks = drone.get("associated_tracks", [])
        # Must not contain wallpaper track
        if "quadcopter drone_049" in drone_tracks:
            errors.append("inst_quadcopter_drone_001: contains wallpaper detection track 'quadcopter drone_049'")
        if "quadcopter drone_001_s2" in drone_tracks:
            errors.append("inst_quadcopter_drone_001: contains unverified track 'quadcopter drone_001_s2'")

        # Must have verified physical track
        if "quadcopter drone_001" not in drone_tracks:
            errors.append("inst_quadcopter_drone_001: missing true physical drone track 'quadcopter drone_001'")

        # Must be observed in Session 01
        sessions = drone.get("sessions_observed", [])
        if "semantic_mapping_my_desk_recording_01" not in sessions:
            errors.append(
                f"inst_quadcopter_drone_001: physical drone was observed in Session 01, but sessions_observed is {sessions}"
            )
        if "semantic_mapping_my_desk_recording_02" in sessions:
            errors.append(
                f"inst_quadcopter_drone_001: claims observation in Session 02 where only 2D screen wallpaper existed"
            )

        # Centroid must match physical workbench coordinates [-2.69, -3.30, 0.89], not monitor [2.81, -1.65, 1.04]
        c = drone.get("centroid_floor_m", [0, 0, 0])
        dist_workbench = math.sqrt((c[0] - (-2.69)) ** 2 + (c[1] - (-3.30)) ** 2)
        dist_monitor = math.sqrt((c[0] - 2.81) ** 2 + (c[1] - (-1.65)) ** 2)
        if dist_workbench > 0.35:
            errors.append(
                f"inst_quadcopter_drone_001: centroid {c} is {dist_workbench:.2f}m from physical workbench, expected <= 0.35m"
            )
        if dist_monitor < 1.0:
            errors.append(
                f"inst_quadcopter_drone_001: centroid {c} is {dist_monitor:.2f}m from computer monitor screen! Must not conflate."
            )

    # 3. Specific RC Car Identity & Displacement Tests
    rc_car = next((o for o in instances if o["object_instance_id"] == "inst_rc_car_robot_001"), None)
    if rc_car:
        rc_tracks = rc_car.get("associated_tracks", [])
        if "robot_platform_base" not in rc_tracks or "robot_001" not in rc_tracks:
            errors.append(f"inst_rc_car_robot_001: expected tracks ['robot_platform_base', 'robot_001'], got {rc_tracks}")

        # Centroid must be near S1 start location [-2.70, -3.22, 0.12] (ground floor)
        c = rc_car.get("centroid_floor_m", [0, 0, 0])
        dist_start = math.sqrt((c[0] - (-2.70)) ** 2 + (c[1] - (-3.22)) ** 2)
        if dist_start > 0.35:
            errors.append(f"inst_rc_car_robot_001: centroid {c} is {dist_start:.2f}m from S1 location [-2.70, -3.22, 0.12]")

        # Displacement must match measured physical movement (0.996m ± 0.05m), NOT synthetic 4.912m
        disp = rc_car.get("displacement_gt", 0.0)
        if abs(disp - 0.996) > 0.05:
            errors.append(f"inst_rc_car_robot_001: displacement is {disp}m, expected measured 0.996m ± 0.05m")

    # 4. Specific Human Operator Identity & Track Partition Tests
    operator = next((o for o in instances if o["object_instance_id"] == "inst_human_operator_001"), None)
    if operator:
        op_tracks = set(operator.get("associated_tracks", []))
        expected_op_tracks = {"person_003", "person_006"}
        if op_tracks != expected_op_tracks:
            errors.append(
                f"inst_human_operator_001: associated_tracks must be exactly {expected_op_tracks}, got {op_tracks}"
            )
        if "person_008" in op_tracks:
            errors.append(
                "inst_human_operator_001: contains seated background occupant track 'person_008'! Must be quarantined."
            )

        # Centroid must match operator aisle corridor [-2.52, -3.82, 0.93]
        c = operator.get("centroid_floor_m", [0, 0, 0])
        dist_aisle = math.sqrt((c[0] - (-2.52)) ** 2 + (c[1] - (-3.82)) ** 2)
        if dist_aisle > 0.35:
            errors.append(f"inst_human_operator_001: centroid {c} is {dist_aisle:.2f}m from aisle location [-2.52, -3.82, 0.93]")

        # Displacement must match intra-aisle walking (1.20m ± 0.15m), NOT cross-room jump 3.45m
        disp = operator.get("displacement_gt", 0.0)
        if abs(disp - 1.20) > 0.15:
            errors.append(f"inst_human_operator_001: displacement is {disp}m, expected 1.20m ± 0.15m")

    # 5. Quarantined Tracks Registry Verification
    quarantined_path = root / "vlm" / "benchmarks" / "quarantined_tracks.json"
    if not quarantined_path.exists():
        errors.append(f"Missing quarantined tracks file: {quarantined_path}")
    else:
        with quarantined_path.open("r", encoding="utf-8") as f:
            q_data = json.load(f)
        q_tracks = [t["track_id"] for t in q_data.get("quarantined_tracks", [])]
        if "person_008" not in q_tracks:
            errors.append("quarantined_tracks.json must contain 'person_008'")
        # Ensure none of the quarantined tracks appear in associated_tracks of any ground truth instance
        for q_trk in q_tracks:
            if q_trk in all_associated_tracks:
                errors.append(f"Quarantined track '{q_trk}' is illegally present in associated_tracks of ground truth instances!")

    # 6. False-Positive Catalog Verification
    if fp_catalog_path.exists():
        with fp_catalog_path.open("r", encoding="utf-8") as f:
            fp_data = json.load(f)
        fp_list = fp_data.get("false_positives", [])
        drone_fp = next((fp for fp in fp_list if fp["fp_id"] == "fp_monitor_drone_wallpaper_049"), None)
        if not drone_fp:
            warnings.append("false_positive_catalog.json does not contain 'fp_monitor_drone_wallpaper_049'")
        else:
            if drone_fp.get("source_session") != "semantic_mapping_my_desk_recording_02":
                errors.append("fp_monitor_drone_wallpaper_049: source_session must be 'semantic_mapping_my_desk_recording_02'")

    summary = {
        "total_instances_validated": len(instances),
        "total_associated_tracks": len(all_associated_tracks),
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }

    is_valid = (len(errors) == 0)
    return is_valid, errors, summary


def main() -> None:
    root = get_repo_root()
    is_valid, errors, summary = validate_movability_benchmark(root)

    print("=" * 65)
    print("MOVABILITY BENCHMARK VALIDATION GATE")
    print("=" * 65)
    print(f"Total instances validated : {summary['total_instances_validated']}")
    print(f"Total associated tracks   : {summary['total_associated_tracks']}")
    print(f"Status                    : {'PASSED' if is_valid else 'FAILED'}")
    print("=" * 65)

    if summary["warnings"]:
        print("\nWARNINGS:")
        for w in summary["warnings"]:
            print(f"  [WARN] {w}")

    if errors:
        print("\nERRORS DETECTED:")
        for e in errors:
            print(f"  [FAIL] {e}")
        print("\nValidation gate FAILED. Halting pipeline execution.")
        raise SystemExit(1)

    print("\nBenchmark successfully passed validation gate.")


if __name__ == "__main__":
    main()
