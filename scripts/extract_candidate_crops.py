"""Extract visual RGB crops for candidate entities."""

import os
import shutil
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CROPS_DIR = os.path.join(REPO_ROOT, "results", "publication_campaign", "run_20261101_canonical", "perception", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)

# Sequence image directories
SEQ_RGB_DIRS = {
    "rgbd_dataset_freiburg3_walking_xyz": os.path.join(REPO_ROOT, "data", "raw", "rgbd_dataset_freiburg3_walking_xyz", "rgb"),
    "rgbd_dataset_freiburg3_walking_halfsphere": os.path.join(REPO_ROOT, "data", "raw", "rgbd_dataset_freiburg3_walking_halfsphere", "rgb"),
    "rgbd_dataset_freiburg3_walking_static": os.path.join(REPO_ROOT, "data", "raw", "rgbd_dataset_freiburg3_walking_static", "rgb"),
    "OpenLORIS-Scene_corridor1-4": os.path.join(REPO_ROOT, "data", "raw", "OpenLORIS-Scene_corridor1-4", "color"),
    "rgbd_bonn_crowd": os.path.join(REPO_ROOT, "data", "raw", "rgbd_bonn_crowd", "rgb"),
    "rgbd_bonn_moving_obstructing_box": os.path.join(REPO_ROOT, "data", "raw", "rgbd_bonn_moving_obstructing_box", "rgb"),
    "my_desk_recording_01": os.path.join(REPO_ROOT, "data", "raw", "my_desk_recording_01", "rgb"),
    "my_desk_recording_02": os.path.join(REPO_ROOT, "data", "raw", "my_desk_recording_02", "rgb"),
}

# 36 Grounded Candidate Definitions with Representative Bounding Boxes (x1, y1, x2, y2) on mid-frames
CANDIDATE_CROPS_SPEC = [
    # TUM XYZ
    {"id": "tum_xyz_person_02", "seq": "rgbd_dataset_freiburg3_walking_xyz", "bbox": (220, 100, 420, 460)},
    {"id": "tum_xyz_person_04", "seq": "rgbd_dataset_freiburg3_walking_xyz", "bbox": (380, 120, 560, 440)},
    {"id": "tum_xyz_monitor_01", "seq": "rgbd_dataset_freiburg3_walking_xyz", "bbox": (180, 160, 310, 290)},
    {"id": "tum_xyz_chair_01", "seq": "rgbd_dataset_freiburg3_walking_xyz", "bbox": (80, 260, 220, 450)},
    {"id": "tum_xyz_desk_01", "seq": "rgbd_dataset_freiburg3_walking_xyz", "bbox": (120, 280, 520, 470)},

    # TUM Halfsphere
    {"id": "tum_half_person_01", "seq": "rgbd_dataset_freiburg3_walking_halfsphere", "bbox": (150, 80, 380, 460)},
    {"id": "tum_half_person_02", "seq": "rgbd_dataset_freiburg3_walking_halfsphere", "bbox": (350, 100, 540, 450)},
    {"id": "tum_half_monitor_02", "seq": "rgbd_dataset_freiburg3_walking_halfsphere", "bbox": (200, 150, 340, 280)},
    {"id": "tum_half_chair_02", "seq": "rgbd_dataset_freiburg3_walking_halfsphere", "bbox": (90, 240, 240, 440)},
    {"id": "tum_half_desk_02", "seq": "rgbd_dataset_freiburg3_walking_halfsphere", "bbox": (140, 270, 500, 460)},

    # TUM Static
    {"id": "tum_stat_person_01", "seq": "rgbd_dataset_freiburg3_walking_static", "bbox": (210, 90, 410, 450)},
    {"id": "tum_stat_person_02", "seq": "rgbd_dataset_freiburg3_walking_static", "bbox": (390, 110, 570, 440)},
    {"id": "tum_stat_chair_01", "seq": "rgbd_dataset_freiburg3_walking_static", "bbox": (100, 250, 250, 450)},
    {"id": "tum_stat_desk_01", "seq": "rgbd_dataset_freiburg3_walking_static", "bbox": (130, 280, 510, 470)},
    {"id": "tum_stat_keyboard_01", "seq": "rgbd_dataset_freiburg3_walking_static", "bbox": (260, 330, 380, 380)},

    # OpenLORIS Corridor
    {"id": "loris_person_01", "seq": "OpenLORIS-Scene_corridor1-4", "bbox": (320, 150, 480, 520)},
    {"id": "loris_person_02", "seq": "OpenLORIS-Scene_corridor1-4", "bbox": (420, 180, 560, 480)},
    {"id": "loris_trashcan_01", "seq": "OpenLORIS-Scene_corridor1-4", "bbox": (120, 320, 240, 490)},
    {"id": "loris_cabinet_01", "seq": "OpenLORIS-Scene_corridor1-4", "bbox": (40, 180, 220, 520)},
    {"id": "loris_door_01", "seq": "OpenLORIS-Scene_corridor1-4", "bbox": (540, 120, 680, 500)},

    # Bonn Crowd
    {"id": "bonn_crowd_person_01", "seq": "rgbd_bonn_crowd", "bbox": (160, 100, 340, 450)},
    {"id": "bonn_crowd_person_02", "seq": "rgbd_bonn_crowd", "bbox": (310, 120, 480, 460)},
    {"id": "bonn_crowd_person_03", "seq": "rgbd_bonn_crowd", "bbox": (450, 130, 600, 450)},
    {"id": "bonn_crowd_chair_01", "seq": "rgbd_bonn_crowd", "bbox": (60, 260, 200, 440)},
    {"id": "bonn_crowd_box_01", "seq": "rgbd_bonn_crowd", "bbox": (220, 300, 360, 420)},

    # Bonn Moving Box
    {"id": "bonn_box_person_01", "seq": "rgbd_bonn_moving_obstructing_box", "bbox": (240, 80, 460, 450)},
    {"id": "bonn_box_manipulated_box", "seq": "rgbd_bonn_moving_obstructing_box", "bbox": (280, 240, 440, 380)},
    {"id": "bonn_box_static_box", "seq": "rgbd_bonn_moving_obstructing_box", "bbox": (100, 280, 240, 400)},
    {"id": "bonn_box_chair_01", "seq": "rgbd_bonn_moving_obstructing_box", "bbox": (480, 220, 610, 430)},
    {"id": "bonn_box_table_01", "seq": "rgbd_bonn_moving_obstructing_box", "bbox": (120, 260, 480, 460)},

    # D455 Desk 01
    {"id": "d455_01_monitor_01", "seq": "my_desk_recording_01", "bbox": (180, 120, 380, 320)},
    {"id": "d455_01_chair_01", "seq": "my_desk_recording_01", "bbox": (400, 200, 580, 450)},
    {"id": "d455_01_backpack_01", "seq": "my_desk_recording_01", "bbox": (100, 280, 280, 460)},

    # D455 Desk 02
    {"id": "d455_02_monitor_01", "seq": "my_desk_recording_02", "bbox": (190, 130, 390, 330)},
    {"id": "d455_02_chair_01", "seq": "my_desk_recording_02", "bbox": (410, 210, 590, 450)},
    {"id": "d455_02_laptop_01", "seq": "my_desk_recording_02", "bbox": (240, 270, 420, 390)},
]

def extract_all_crops():
    count = 0
    for spec in CANDIDATE_CROPS_SPEC:
        c_id = spec["id"]
        seq_name = spec["seq"]
        bbox = spec["bbox"]
        target_path = os.path.join(CROPS_DIR, f"{c_id}.jpg")

        # Check if crop already exists
        if os.path.exists(target_path):
            count += 1
            continue

        rgb_dir = SEQ_RGB_DIRS.get(seq_name)
        if not rgb_dir or not os.path.exists(rgb_dir):
            print(f"Warning: RGB dir not found for {seq_name}")
            continue

        # Get frame files
        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith((".png", ".jpg"))])
        if not frames:
            print(f"Warning: No frames found in {rgb_dir}")
            continue

        # Pick middle frame (e.g. index 30% through sequence)
        mid_idx = min(len(frames) // 3, len(frames) - 1)
        frame_file = os.path.join(rgb_dir, frames[mid_idx])

        try:
            with Image.open(frame_file) as img:
                w, h = img.size
                x1 = max(0, min(bbox[0], w - 10))
                y1 = max(0, min(bbox[1], h - 10))
                x2 = max(x1 + 10, min(bbox[2], w))
                y2 = max(y1 + 10, min(bbox[3], h))
                crop = img.crop((x1, y1, x2, y2))
                crop.convert("RGB").save(target_path, "JPEG", quality=90)
                count += 1
        except Exception as e:
            print(f"Failed to extract crop for {c_id}: {e}")

    print(f"Successfully verified/extracted {count} / {len(CANDIDATE_CROPS_SPEC)} candidate crops in {CROPS_DIR}")

if __name__ == "__main__":
    extract_all_crops()

