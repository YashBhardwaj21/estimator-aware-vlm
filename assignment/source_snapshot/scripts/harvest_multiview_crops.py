"""Harvest multi-view image crops for all benchmark objects across both sequences (Stage 3)."""

import json
from pathlib import Path
import sys

# Add project root and vlm
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.crop_harvester import MultiViewCropHarvester


def main():
    bench_path = Path("vlm/benchmarks/movability_ground_truth.json")
    if not bench_path.exists():
        raise FileNotFoundError("Run Stage 1 benchmark generation first.")

    with open(bench_path) as f:
        benchmark = json.load(f)

    out_dir = Path("vlm/outputs")
    harvester = MultiViewCropHarvester(
        output_dir=out_dir,
        max_views_per_object=3,
        min_angular_separation_deg=10.0,
        context_padding_pct=0.15
    )

    all_harvested = {}
    total_crops = 0

    for seq_name, seq_data in benchmark["sequences"].items():
        print(f"\nHarvesting multi-view crops for: {seq_name}")
        seq_raw_dir = Path(f"data/raw/{seq_name}")
        mapping_dir = Path(f"outputs/semantic_mapping_{seq_name}")

        objs = seq_data["objects"]
        harvested_objs = harvester.harvest_for_sequence(
            seq_name=seq_name,
            seq_raw_dir=seq_raw_dir,
            mapping_dir=mapping_dir,
            ground_truth_objects=objs
        )
        
        seq_crop_count = sum(o["views_collected"] for o in harvested_objs)
        total_crops += seq_crop_count
        print(f"-> Harvested {seq_crop_count} diverse crops across {len(harvested_objs)} objects in {seq_name}")
        all_harvested[seq_name] = harvested_objs

    # Save manifest
    manifest_path = out_dir / "multiview_dataset.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "1.0.0",
            "total_physical_objects": len(all_harvested.get("desk_run_20261009_161552", [])) + len(all_harvested.get("d455_rgbd_20261010_123443", [])),
            "total_crops_harvested": total_crops,
            "sequences": all_harvested
        }, f, indent=2)

    print(f"\n==================================================================")
    print(f"MULTI-VIEW CROP HARVESTING COMPLETE")
    print(f"Total Crops Extracted: {total_crops}")
    print(f"Saved Manifest: {manifest_path}")
    print(f"Crops Directory: {out_dir / 'crops'}")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
