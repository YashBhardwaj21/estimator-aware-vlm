"""Test suite verifying navigation decoupling and removal of autonomous navigation modules."""

from pathlib import Path
import pytest
from estimator_aware_vlm.common.paths import get_repo_root


def test_navigation_directories_removed():
    """Verify all navigation-specific directories and configs are removed."""
    root = get_repo_root()

    assert not (root / "navigation").exists(), "Root 'navigation/' directory must not exist"
    assert not (root / "configs" / "navigation").exists(), "'configs/navigation/' directory must not exist"
    assert not (root / "src" / "scene_graph" / "navigation").exists(), "'src/scene_graph/navigation/' directory must not exist"
    assert not (root / "src" / "scene_graph" / "hardware").exists(), "'src/scene_graph/hardware/' directory must not exist"


def test_no_navigation_references_in_code():
    """Verify no active Python source files contain navigation or motor controller imports."""
    root = get_repo_root()
    active_dirs = ["src", "mapping", "recording", "slam", "tools", "scripts"]

    forbidden_patterns = [
        "scene_graph.navigation",
        "motor_controller",
        "from navigation",
        "import navigation",
        "motor_interface_config",
    ]

    violations = []
    for d in active_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden_patterns:
                if pattern in text:
                    violations.append(f"{py_file.relative_to(root)} contains '{pattern}'")

    assert not violations, f"Found forbidden navigation references:\n" + "\n".join(violations)


def test_mapping_pipeline_has_no_navigation_dependency():
    """Verify run_mapping_pipeline.py does not export navigation configs or import planner."""
    root = get_repo_root()
    mapping_py = root / "mapping" / "run_mapping_pipeline.py"
    assert mapping_py.exists()
    content = mapping_py.read_text(encoding="utf-8")

    assert "navigation_config.yaml" not in content
    assert "motor_interface_config.yaml" not in content
    assert "scene_graph.navigation" not in content
