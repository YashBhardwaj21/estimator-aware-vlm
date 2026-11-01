"""
Script to compute policy selection overlap between R5, R6, R7 under:
1. Per-sequence ceiling (14 queries = 38.89%)
2. Global budget K = 9 (25.0%)
"""

import math
import numpy as np

# Recreate candidates exactly from scripts/run_master_publication_campaign.py
candidates = [
    # TUM XYZ
    {"candidate_id": "tum_xyz_person_02", "sequence_name": "rgbd_dataset_freiburg3_walking_xyz", "class": "person", "displacement": 1.24, "unc": 0.18, "is_dynamic": True},
    {"candidate_id": "tum_xyz_person_04", "sequence_name": "rgbd_dataset_freiburg3_walking_xyz", "class": "person", "displacement": 1.15, "unc": 0.15, "is_dynamic": True},
    {"candidate_id": "tum_xyz_monitor_01", "sequence_name": "rgbd_dataset_freiburg3_walking_xyz", "class": "monitor", "displacement": 0.012, "unc": 0.008, "is_dynamic": False},
    {"candidate_id": "tum_xyz_chair_01", "sequence_name": "rgbd_dataset_freiburg3_walking_xyz", "class": "chair", "displacement": 0.035, "unc": 0.015, "is_dynamic": False},
    {"candidate_id": "tum_xyz_desk_01", "sequence_name": "rgbd_dataset_freiburg3_walking_xyz", "class": "desk", "displacement": 0.008, "unc": 0.005, "is_dynamic": False},

    # TUM Halfsphere
    {"candidate_id": "tum_half_person_01", "sequence_name": "rgbd_dataset_freiburg3_walking_halfsphere", "class": "person", "displacement": 1.45, "unc": 0.20, "is_dynamic": True},
    {"candidate_id": "tum_half_person_02", "sequence_name": "rgbd_dataset_freiburg3_walking_halfsphere", "class": "person", "displacement": 1.30, "unc": 0.18, "is_dynamic": True},
    {"candidate_id": "tum_half_monitor_02", "sequence_name": "rgbd_dataset_freiburg3_walking_halfsphere", "class": "monitor", "displacement": 0.015, "unc": 0.009, "is_dynamic": False},
    {"candidate_id": "tum_half_chair_02", "sequence_name": "rgbd_dataset_freiburg3_walking_halfsphere", "class": "chair", "displacement": 0.028, "unc": 0.012, "is_dynamic": False},
    {"candidate_id": "tum_half_desk_02", "sequence_name": "rgbd_dataset_freiburg3_walking_halfsphere", "class": "desk", "displacement": 0.010, "unc": 0.006, "is_dynamic": False},

    # TUM Static
    {"candidate_id": "tum_stat_person_01", "sequence_name": "rgbd_dataset_freiburg3_walking_static", "class": "person", "displacement": 1.55, "unc": 0.22, "is_dynamic": True},
    {"candidate_id": "tum_stat_person_02", "sequence_name": "rgbd_dataset_freiburg3_walking_static", "class": "person", "displacement": 1.38, "unc": 0.19, "is_dynamic": True},
    {"candidate_id": "tum_stat_chair_01", "sequence_name": "rgbd_dataset_freiburg3_walking_static", "class": "chair", "displacement": 0.025, "unc": 0.011, "is_dynamic": False},
    {"candidate_id": "tum_stat_desk_01", "sequence_name": "rgbd_dataset_freiburg3_walking_static", "class": "desk", "displacement": 0.005, "unc": 0.004, "is_dynamic": False},
    {"candidate_id": "tum_stat_keyboard_01", "sequence_name": "rgbd_dataset_freiburg3_walking_static", "class": "keyboard", "displacement": 0.003, "unc": 0.003, "is_dynamic": False},

    # OpenLORIS Corridor
    {"candidate_id": "loris_person_01", "sequence_name": "OpenLORIS-Scene_corridor1-4", "class": "person", "displacement": 2.10, "unc": 0.25, "is_dynamic": True},
    {"candidate_id": "loris_person_02", "sequence_name": "OpenLORIS-Scene_corridor1-4", "class": "person", "displacement": 1.85, "unc": 0.22, "is_dynamic": True},
    {"candidate_id": "loris_trashcan_01", "sequence_name": "OpenLORIS-Scene_corridor1-4", "class": "trash can", "displacement": 0.020, "unc": 0.010, "is_dynamic": False},
    {"candidate_id": "loris_cabinet_01", "sequence_name": "OpenLORIS-Scene_corridor1-4", "class": "cabinet", "displacement": 0.008, "unc": 0.006, "is_dynamic": False},
    {"candidate_id": "loris_door_01", "sequence_name": "OpenLORIS-Scene_corridor1-4", "class": "door", "displacement": 0.012, "unc": 0.007, "is_dynamic": False},

    # Bonn Crowd
    {"candidate_id": "bonn_crowd_person_01", "sequence_name": "rgbd_bonn_crowd", "class": "person", "displacement": 1.60, "unc": 0.24, "is_dynamic": True},
    {"candidate_id": "bonn_crowd_person_02", "sequence_name": "rgbd_bonn_crowd", "class": "person", "displacement": 1.40, "unc": 0.21, "is_dynamic": True},
    {"candidate_id": "bonn_crowd_person_03", "sequence_name": "rgbd_bonn_crowd", "class": "person", "displacement": 1.25, "unc": 0.17, "is_dynamic": True},
    {"candidate_id": "bonn_crowd_box_01", "sequence_name": "rgbd_bonn_crowd", "class": "box", "displacement": 0.015, "unc": 0.008, "is_dynamic": False},
    {"candidate_id": "bonn_crowd_chair_01", "sequence_name": "rgbd_bonn_crowd", "class": "chair", "displacement": 0.030, "unc": 0.014, "is_dynamic": False},

    # Bonn Moving Box
    {"candidate_id": "bonn_box_manipulated_box", "sequence_name": "rgbd_bonn_moving_obstructing_box", "class": "box", "displacement": 0.95, "unc": 0.14, "is_dynamic": True},
    {"candidate_id": "bonn_box_person_01", "sequence_name": "rgbd_bonn_moving_obstructing_box", "class": "person", "displacement": 1.35, "unc": 0.19, "is_dynamic": True},
    {"candidate_id": "bonn_box_static_box", "sequence_name": "rgbd_bonn_moving_obstructing_box", "class": "box", "displacement": 0.005, "unc": 0.004, "is_dynamic": False},
    {"candidate_id": "bonn_box_table_01", "sequence_name": "rgbd_bonn_moving_obstructing_box", "class": "table", "displacement": 0.010, "unc": 0.006, "is_dynamic": False},
    {"candidate_id": "bonn_box_chair_01", "sequence_name": "rgbd_bonn_moving_obstructing_box", "class": "chair", "displacement": 0.022, "unc": 0.010, "is_dynamic": False},

    # D455 Desk 01
    {"candidate_id": "d455_01_mouse_01", "sequence_name": "my_desk_recording_01", "class": "mouse", "displacement": 0.12, "unc": 0.04, "is_dynamic": True},
    {"candidate_id": "d455_01_cup_01", "sequence_name": "my_desk_recording_01", "class": "cup", "displacement": 0.01, "unc": 0.008, "is_dynamic": False},
    {"candidate_id": "d455_01_monitor_01", "sequence_name": "my_desk_recording_01", "class": "monitor", "displacement": 0.005, "unc": 0.004, "is_dynamic": False},

    # D455 Desk 02
    {"candidate_id": "d455_02_mouse_02", "sequence_name": "my_desk_recording_02", "class": "mouse", "displacement": 0.14, "unc": 0.045, "is_dynamic": True},
    {"candidate_id": "d455_02_cup_02", "sequence_name": "my_desk_recording_02", "class": "cup", "displacement": 0.012, "unc": 0.009, "is_dynamic": False},
    {"candidate_id": "d455_02_keyboard_01", "sequence_name": "my_desk_recording_02", "class": "keyboard", "displacement": 0.004, "unc": 0.003, "is_dynamic": False},
]

print(f"Loaded {len(candidates)} candidates.")

# 1. Per-sequence
seq_groups = {}
for c in candidates:
    seq_groups.setdefault(c["sequence_name"], []).append(c)

per_seq = {"R5": [], "R6": [], "R7": []}
for s_name, c_list in seq_groups.items():
    k = max(1, int(math.ceil(0.25 * len(c_list))))
    per_seq["R5"].extend([x["candidate_id"] for x in sorted(c_list, key=lambda x: x["unc"], reverse=True)[:k]])
    per_seq["R6"].extend([x["candidate_id"] for x in sorted(c_list, key=lambda x: x["displacement"], reverse=True)[:k]])
    per_seq["R7"].extend([x["candidate_id"] for x in sorted(c_list, key=lambda x: (x["displacement"] / (x["unc"] + 1e-4)) * (1.2 if x["class"] == "person" else 0.8), reverse=True)[:k]])

print("--- PER-SEQUENCE BUDGET (14 queries, 38.89%) ---")
print(f"R5 count: {len(per_seq['R5'])}, R6 count: {len(per_seq['R6'])}, R7 count: {len(per_seq['R7'])}")
jacc_5_6 = len(set(per_seq["R5"]) & set(per_seq["R6"])) / len(set(per_seq["R5"]) | set(per_seq["R6"]))
jacc_6_7 = len(set(per_seq["R6"]) & set(per_seq["R7"])) / len(set(per_seq["R6"]) | set(per_seq["R7"]))
jacc_5_7 = len(set(per_seq["R5"]) & set(per_seq["R7"])) / len(set(per_seq["R5"]) | set(per_seq["R7"]))
print(f"Jaccard R5 vs R6: {jacc_5_6:.3f}, R6 vs R7: {jacc_6_7:.3f}, R5 vs R7: {jacc_5_7:.3f}")

# 2. Global budget K = 9 (25.0%)
k_global = 9
global_sel = {
    "R5": [x["candidate_id"] for x in sorted(candidates, key=lambda x: x["unc"], reverse=True)[:k_global]],
    "R6": [x["candidate_id"] for x in sorted(candidates, key=lambda x: x["displacement"], reverse=True)[:k_global]],
    "R7": [x["candidate_id"] for x in sorted(candidates, key=lambda x: (x["displacement"] / (x["unc"] + 1e-4)) * (1.2 if x["class"] == "person" else 0.8), reverse=True)[:k_global]]
}
print("\n--- GLOBAL BUDGET (9 queries, 25.00%) ---")
print("R5 candidates:", global_sel["R5"])
print("R6 candidates:", global_sel["R6"])
print("R7 candidates:", global_sel["R7"])
gjacc_5_6 = len(set(global_sel["R5"]) & set(global_sel["R6"])) / len(set(global_sel["R5"]) | set(global_sel["R6"]))
gjacc_6_7 = len(set(global_sel["R6"]) & set(global_sel["R7"])) / len(set(global_sel["R6"]) | set(global_sel["R7"]))
gjacc_5_7 = len(set(global_sel["R5"]) & set(global_sel["R7"])) / len(set(global_sel["R5"]) | set(global_sel["R7"]))
print(f"Global Jaccard R5 vs R6: {gjacc_5_6:.3f}, R6 vs R7: {gjacc_6_7:.3f}, R5 vs R7: {gjacc_5_7:.3f}")

