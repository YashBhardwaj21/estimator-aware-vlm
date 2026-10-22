import cv2
import json
import numpy as np

m = cv2.imread("outputs/semantic_mapping_fused_desk_run/map.pgm", cv2.IMREAD_GRAYSCALE)
c = cv2.imread("outputs/semantic_mapping_fused_desk_run/navigation_costmap.pgm", cv2.IMREAD_GRAYSCALE)
print("map.pgm unique values and counts:")
for val, count in zip(*np.unique(m, return_counts=True)):
    print(f"  val={val}: {count} pixels ({count / m.size * 100:.2f}%)")

print("navigation_costmap.pgm unique values and counts:")
for val, count in zip(*np.unique(c, return_counts=True)):
    print(f"  val={val}: {count} pixels ({count / c.size * 100:.2f}%)")

with open("outputs/semantic_mapping_fused_desk_run/semantic_map.json") as f:
    smap = json.load(f)

print("\nSample entity dimensions and bbox sizes:")
for eid, ent in list(smap.get("entities", {}).items())[:25]:
    cls_name = ent.get("class_name")
    dims = ent.get("dimensions", [])
    b_min = ent.get("bbox_min_floor", [])
    b_max = ent.get("bbox_max_floor", [])
    dx = (b_max[0] - b_min[0]) if len(b_min) >= 2 and len(b_max) >= 2 else 0
    dy = (b_max[1] - b_min[1]) if len(b_min) >= 2 and len(b_max) >= 2 else 0
    print(f"{eid:20s} ({cls_name:15s}): dx={dx:.3f}m, dy={dy:.3f}m, dims={dims}")
