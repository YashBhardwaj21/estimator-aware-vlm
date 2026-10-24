"""Generate synchronized associations.txt for OpenLORIS-Scene Corridor 1-4."""

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import TUMRGBDAdapter, load_dataset_config_from_yaml

def main():
    repo_root = get_repo_root()
    cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"
    cfg = load_dataset_config_from_yaml(cfg_path)
    adapter = TUMRGBDAdapter(cfg)
    out_file = cfg.sequence_dir / "associations.txt"
    adapter.export_associations(out_file)
    print(f"Exported {len(adapter)} synchronized associations to: {out_file}")

if __name__ == "__main__":
    main()
