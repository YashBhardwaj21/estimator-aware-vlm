"""Export synchronized associations file for TUM sequences."""

from pathlib import Path
from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import DatasetConfig, TUMRGBDAdapter


def main() -> None:
    repo_root = get_repo_root()
    seq_dir = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz"
    cfg = DatasetConfig(
        dataset_type="tum_rgbd",
        sequence_name="rgbd_dataset_freiburg3_walking_xyz",
        sequence_dir=seq_dir,
    )
    adapter = TUMRGBDAdapter(cfg)
    assoc_path = seq_dir / "associations.txt"
    adapter.export_associations(assoc_path)
    print(f"Successfully exported {len(adapter)} synchronized pairs to: {assoc_path}")


if __name__ == "__main__":
    main()
