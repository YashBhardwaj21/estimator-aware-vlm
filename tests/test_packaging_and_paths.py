"""Packaging Verification Tests: Packaging, Root Resolution, and Import Integrity."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct pytest invocation without pip install -e
_src_dir = Path(__file__).resolve().parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def test_repo_root_resolution():
    """Verify get_repo_root finds repository root containing pyproject.toml."""
    from estimator_aware_vlm.common.paths import get_repo_root

    root = get_repo_root()
    assert root.exists(), f"Resolved root does not exist: {root}"
    assert (root / "pyproject.toml").exists(), f"pyproject.toml not found at root: {root}"
    assert (root / "benchmarks").exists(), f"benchmarks directory not found at root: {root}"
    assert (root / "src").exists(), f"src directory not found at root: {root}"


def test_canonical_directory_helpers():
    """Verify helper functions point to expected subdirectories."""
    from estimator_aware_vlm.common.paths import (
        get_repo_root,
        get_benchmarks_dir,
        get_configs_dir,
        get_outputs_dir,
        get_data_dir,
    )

    root = get_repo_root()
    assert get_benchmarks_dir() == root / "benchmarks"
    assert get_configs_dir() == root / "configs"
    assert get_outputs_dir() == root / "outputs"
    assert get_data_dir() == root / "data"


def test_package_imports_estimator_aware_vlm():
    """Verify estimator_aware_vlm top-level and submodules import without legacy vlm hacks."""
    import estimator_aware_vlm
    assert hasattr(estimator_aware_vlm, "__version__")
    assert estimator_aware_vlm.__version__ == "1.0.0"

    from estimator_aware_vlm.multimodal_inference_engine import MultimodalInferenceEngine
    from estimator_aware_vlm.selective_router import SelectiveRouter
    from estimator_aware_vlm.router_policy import evaluate_router_policy
    from estimator_aware_vlm.cross_session_association import CrossSessionCandidateMatcher
    from estimator_aware_vlm.contextual_scene_graph_fusion import ContextualSceneGraphFusion
    from estimator_aware_vlm.evidence_adapter import EvidenceAdapter
    from estimator_aware_vlm.statistical_analysis import run_statistical_analysis

    assert MultimodalInferenceEngine is not None
    assert SelectiveRouter is not None
    assert evaluate_router_policy is not None
    assert CrossSessionCandidateMatcher is not None
    assert ContextualSceneGraphFusion is not None
    assert EvidenceAdapter is not None
    assert run_statistical_analysis is not None


def test_package_imports_scene_graph():
    """Verify upstream scene_graph package remains cleanly importable."""
    import scene_graph
    from scene_graph.geometry.coordinate_frames import CoordinateFrameContract
    from scene_graph.tracking.causal_tracker import CausalTracker
    from scene_graph.graph.temporal_graph import TemporalSceneGraph

    assert CoordinateFrameContract is not None
    assert CausalTracker is not None
    assert TemporalSceneGraph is not None


# ---------------------------------------------------------------------------
# Structural Decoupling & Modular Architecture Verification
# ---------------------------------------------------------------------------

def test_navigation_directories_removed():
    """Verify all navigation-specific directories and configs are removed."""
    from estimator_aware_vlm.common.paths import get_repo_root
    root = get_repo_root()

    assert not (root / "navigation").exists(), "Root 'navigation/' directory must not exist"
    assert not (root / "configs" / "navigation").exists(), "'configs/navigation/' directory must not exist"
    assert not (root / "src" / "scene_graph" / "navigation").exists(), "'src/scene_graph/navigation/' directory must not exist"
    assert not (root / "src" / "scene_graph" / "hardware").exists(), "'src/scene_graph/hardware/' directory must not exist"


def test_no_navigation_references_in_code():
    """Verify no active Python source files contain navigation or motor controller imports."""
    from estimator_aware_vlm.common.paths import get_repo_root
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
    from estimator_aware_vlm.common.paths import get_repo_root
    root = get_repo_root()
    mapping_py = root / "mapping" / "run_mapping_pipeline.py"
    if mapping_py.exists():
        content = mapping_py.read_text(encoding="utf-8")
        assert "navigation_config.yaml" not in content
        assert "motor_interface_config.yaml" not in content
        assert "scene_graph.navigation" not in content
