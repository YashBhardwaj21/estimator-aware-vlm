#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

def main():
    p1 = Path("outputs/semantic_mapping_my_desk_recording_01/semantic_map.json")
    p2 = Path("outputs/semantic_mapping_my_desk_recording_02/semantic_map.json")

    with open(p1, "r", encoding="utf-8") as f:
        map1 = json.load(f)
    with open(p2, "r", encoding="utf-8") as f:
        map2 = json.load(f)

    ents1 = {k: v for k, v in map1.get("entities", {}).items() if v.get("is_confirmed")}
    ents2 = {k: v for k, v in map2.get("entities", {}).items() if v.get("is_confirmed")}

    print(f"Run 1 confirmed: {len(ents1)}")
    print(f"Run 2 confirmed: {len(ents2)}")

    from collections import Counter
    c1 = Counter(v["class_name"] for v in ents1.values())
    c2 = Counter(v["class_name"] for v in ents2.values())

    print("\nRun 1 classes:", c1)
    print("\nRun 2 classes:", c2)

    # Print largest entities in Run 1 and Run 2
    print("\nTop Run 1 entities:")
    for k, v in sorted(ents1.items(), key=lambda x: x[1].get("observation_count", 0), reverse=True)[:8]:
        c = [round(x, 2) for x in v["centroid_floor"]]
        print(f"  {k} ({v['class_name']}): obs={v['observation_count']}, centroid={c}")

    print("\nTop Run 2 entities:")
    for k, v in sorted(ents2.items(), key=lambda x: x[1].get("observation_count", 0), reverse=True)[:8]:
        c = [round(x, 2) for x in v["centroid_floor"]]
        print(f"  {k} ({v['class_name']}): obs={v['observation_count']}, centroid={c}")

if __name__ == "__main__":
    main()
