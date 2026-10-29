"""Generate synchronized associations.txt for OpenLORIS-Scene Corridor 1-4."""

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import get_dataset_adapter, load_dataset_config_from_yaml

def main():
    repo_root = get_repo_root()
    cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"
    cfg = load_dataset_config_from_yaml(cfg_path)
    adapter = get_dataset_adapter(cfg)
    out_dir = repo_root / "outputs" / "associations" / cfg.sequence_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "associations.txt"
    adapter.export_associations(out_file)
    print(f"Exported {len(adapter)} synchronized associations to: {out_file}")

if __name__ == "__main__":
    main()
