#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add src and repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from scene_graph.data.synchronization import associate

def main():
    rec_dir = Path("data/raw/my_desk_recording_02")
    rgb_dir = rec_dir / "rgb"
    depth_dir = rec_dir / "depth"

    existing_rgb = set(f.name for f in rgb_dir.iterdir() if f.is_file() and f.stat().st_size > 1000)
    existing_depth = set(f.name for f in depth_dir.iterdir() if f.is_file() and f.stat().st_size > 1000)

    print(f"Valid RGB images on disk:   {len(existing_rgb)}")
    print(f"Valid Depth images on disk: {len(existing_depth)}")

    # 1. Clean rgb.txt - ensuring strictly increasing timestamps
    rgb_txt = rec_dir / "rgb.txt"
    with open(rgb_txt, "r", encoding="utf-8") as f:
        raw_rgb_lines = f.readlines()
    
    clean_rgb = []
    last_t_rgb = -float("inf")
    for line in raw_rgb_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            t = float(parts[0])
            fname = Path(parts[1]).name
            if fname in existing_rgb and t > last_t_rgb:
                clean_rgb.append((t, parts[1]))
                last_t_rgb = t

    with open(rgb_txt, "w", encoding="utf-8") as f:
        f.write("# RGB images\n# timestamp filename\n")
        for t, fn in clean_rgb:
            f.write(f"{t:.6f} {fn}\n")
    print(f"Cleaned monotonic rgb.txt entries:   {len(clean_rgb)}")

    # 2. Clean depth.txt - ensuring strictly increasing timestamps
    depth_txt = rec_dir / "depth.txt"
    with open(depth_txt, "r", encoding="utf-8") as f:
        raw_depth_lines = f.readlines()
    
    clean_depth = []
    last_t_depth = -float("inf")
    for line in raw_depth_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            t = float(parts[0])
            fname = Path(parts[1]).name
            if fname in existing_depth and t > last_t_depth:
                clean_depth.append((t, parts[1]))
                last_t_depth = t

    with open(depth_txt, "w", encoding="utf-8") as f:
        f.write("# Depth images\n# timestamp filename\n")
        for t, fn in clean_depth:
            f.write(f"{t:.6f} {fn}\n")
    print(f"Cleaned monotonic depth.txt entries: {len(clean_depth)}")

    # 3. Generate optimal monotonic 1-to-1 associations using dynamic programming
    rgb_times = [t for t, _ in clean_rgb]
    depth_times = [t for t, _ in clean_depth]

    matches = associate(rgb_times, depth_times, max_dt=0.05)
    print(f"Optimal monotonic 1-to-1 pairs:     {len(matches)}")

    assoc_txt = rec_dir / "associations.txt"
    with open(assoc_txt, "w", encoding="utf-8") as f:
        for i_rgb, i_depth in matches:
            t_r, fn_r = clean_rgb[i_rgb]
            t_d, fn_d = clean_depth[i_depth]
            f.write(f"{t_r:.6f} {fn_r} {t_d:.6f} {fn_d}\n")

    print(f"Generated {assoc_txt} successfully!")
    
    # 4. Run validation check
    from scene_graph.data.dataset_discovery import validate_recording_session
    report = validate_recording_session(rec_dir)
    print(f"Validation Status: {report['status']}")
    if report['status'] != 'PASS':
        print(f"Errors: {report.get('errors')}")
    else:
        print("ALL INTEGRITY CHECKS PASSED!")

if __name__ == "__main__":
    main()
