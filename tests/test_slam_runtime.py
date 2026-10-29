"""Unit and contract tests for isolated and configurable SLAM runtime (Stage 2)."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from estimator_aware_vlm.slam import (
    EmptyTrajectoryError,
    ORBSLAM3Runner,
    SLAMExecutionError,
    SLAMRuntimeConfig,
    StaleTrajectoryError,
    create_experiment_dir,
    export_run_manifest,
    to_wsl_path,
)


def test_slam_runtime_config_defaults():
    cfg = SLAMRuntimeConfig.from_env_and_args()
    assert "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum" in cfg.orb_binary
    assert "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt" in cfg.orb_vocab
    assert cfg.wsl_distro == "Ubuntu-22.04"
    assert cfg.wsl_user == "saturn"
    assert cfg.fail_closed is True


def test_slam_runtime_config_env_overrides(monkeypatch):
    monkeypatch.setenv("ORB_SLAM3_ROOT", "/opt/orbslam3")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Custom-Ubuntu")
    monkeypatch.setenv("WSL_USER", "custom_user")
    monkeypatch.setenv("ORB_SLAM3_SETTINGS", "/configs/custom.yaml")

    cfg = SLAMRuntimeConfig.from_env_and_args()
    assert cfg.orb_binary == "/opt/orbslam3/Examples/RGB-D/rgbd_tum"
    assert cfg.orb_vocab == "/opt/orbslam3/Vocabulary/ORBvoc.txt"
    assert cfg.wsl_distro == "Custom-Ubuntu"
    assert cfg.wsl_user == "custom_user"
    assert cfg.settings_path == "/configs/custom.yaml"


def test_slam_runtime_config_cli_precedence(monkeypatch):
    monkeypatch.setenv("ORB_SLAM3_BIN", "/env/bin/rgbd_tum")
    monkeypatch.setenv("ORB_SLAM3_VOCAB", "/env/vocab.txt")

    cfg = SLAMRuntimeConfig.from_env_and_args(
        orb_binary="/cli/bin/rgbd_tum",
        orb_vocab="/cli/vocab.txt",
        settings_path="/cli/settings.yaml",
        wsl_distro="CLI-Distro",
        wsl_user="cli_user",
        timeout_sec=300,
    )
    assert cfg.orb_binary == "/cli/bin/rgbd_tum"
    assert cfg.orb_vocab == "/cli/vocab.txt"
    assert cfg.settings_path == "/cli/settings.yaml"
    assert cfg.wsl_distro == "CLI-Distro"
    assert cfg.wsl_user == "cli_user"
    assert cfg.timeout_sec == 300


def test_to_wsl_path():
    assert to_wsl_path("C:\\Users\\test\\file.txt") == "/mnt/c/Users/test/file.txt"
    assert to_wsl_path("d:/datasets/tum/rgb.txt") == "/mnt/d/datasets/tum/rgb.txt"
    assert to_wsl_path("/home/saturn/ORB_SLAM3") == "/home/saturn/ORB_SLAM3"


def test_purge_stale_trajectories(tmp_path):
    runner = ORBSLAM3Runner()
    cam_file = tmp_path / "CameraTrajectory.txt"
    kf_file = tmp_path / "KeyFrameTrajectory.txt"
    cam_file.write_text("dummy old data", encoding="utf-8")
    kf_file.write_text("dummy old kf", encoding="utf-8")

    assert cam_file.exists()
    assert kf_file.exists()

    purged = runner.purge_stale_trajectories(tmp_path)
    assert len(purged) == 2
    assert not cam_file.exists()
    assert not kf_file.exists()


def test_runner_missing_associations_fails(tmp_path):
    runner = ORBSLAM3Runner()
    fake_assoc = tmp_path / "non_existent_assoc.txt"
    with pytest.raises(FileNotFoundError, match="Associations file does not exist"):
        runner.run(
            sequence_dir=tmp_path / "seq",
            associations_file=fake_assoc,
            output_dir=tmp_path / "out",
            settings_path=tmp_path / "settings.yaml",
        )


def test_runner_fail_closed_on_nonzero_exit(tmp_path):
    assoc_file = tmp_path / "associations.txt"
    assoc_file.write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("Camera.fx: 500\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    # Mock subprocess runner that returns returncode = 1
    def mock_executor(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="Segmentation fault (core dumped)\n",
            stderr="",
        )

    runner = ORBSLAM3Runner(executor_fn=mock_executor)

    with pytest.raises(SLAMExecutionError, match="ORB-SLAM3 failed with exit code 1"):
        runner.run(
            sequence_dir=tmp_path / "seq",
            associations_file=assoc_file,
            output_dir=out_dir,
            settings_path=settings_file,
        )

    log_file = out_dir / "orbslam3_execution.log"
    assert log_file.exists()
    assert "Segmentation fault" in log_file.read_text(encoding="utf-8")


def test_runner_fail_closed_on_missing_trajectory(tmp_path):
    assoc_file = tmp_path / "associations.txt"
    assoc_file.write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("Camera.fx: 500\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    # Mock subprocess runner that succeeds with code 0 but produces no file
    def mock_executor(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="System shutdown without saving trajectory.\n",
            stderr="",
        )

    runner = ORBSLAM3Runner(executor_fn=mock_executor)

    with pytest.raises(SLAMExecutionError, match="produced no trajectory file"):
        runner.run(
            sequence_dir=tmp_path / "seq",
            associations_file=assoc_file,
            output_dir=out_dir,
            settings_path=settings_file,
        )


def test_runner_fail_closed_on_empty_trajectory(tmp_path):
    assoc_file = tmp_path / "associations.txt"
    assoc_file.write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("Camera.fx: 500\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    # Mock subprocess runner that writes empty trajectory
    def mock_executor(*args, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "CameraTrajectory.txt").write_text("# timestamp tx ty tz qx qy qz qw\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Tracking lost immediately at frame 0.\n",
            stderr="",
        )

    runner = ORBSLAM3Runner(executor_fn=mock_executor)

    with pytest.raises(EmptyTrajectoryError, match="empty or contains 0 valid poses"):
        runner.run(
            sequence_dir=tmp_path / "seq",
            associations_file=assoc_file,
            output_dir=out_dir,
            settings_path=settings_file,
        )


def test_runner_successful_trajectory_generation(tmp_path):
    assoc_file = tmp_path / "associations.txt"
    assoc_file.write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("Camera.fx: 500\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    # Mock subprocess runner that creates valid poses
    def mock_executor(*args, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        content = (
            "1.000000 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n"
            "1.033333 0.11 0.21 0.31 0.0 0.0 0.0 1.0\n"
        )
        (out_dir / "CameraTrajectory.txt").write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Tracking completed successfully. 2 keyframes.\n",
            stderr="",
        )

    runner = ORBSLAM3Runner(executor_fn=mock_executor)
    res_path = runner.run(
        sequence_dir=tmp_path / "seq",
        associations_file=assoc_file,
        output_dir=out_dir,
        settings_path=settings_file,
    )
    assert res_path.exists()
    assert runner.count_valid_trajectory_poses(res_path) == 2


def test_create_experiment_dir(tmp_path):
    exp_dir = create_experiment_dir(experiment_id="test_exp_001", base_dir=tmp_path)
    assert exp_dir.exists()
    assert exp_dir.name == "test_exp_001"

    auto_exp = create_experiment_dir(prefix="auto_test", base_dir=tmp_path)
    assert auto_exp.exists()
    assert "auto_test" in auto_exp.name


def test_export_run_manifest(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = tmp_path / "input.yaml"
    dummy_input.write_text("key: value\n", encoding="utf-8")

    dummy_output = exp_dir / "CameraTrajectory.txt"
    dummy_output.write_text("1.0 0 0 0 0 0 0 1\n", encoding="utf-8")

    manifest = export_run_manifest(
        experiment_dir=exp_dir,
        run_id="exp_001",
        pipeline_stage="Stage 2 Test",
        config={"param": 42},
        input_files={"cfg": dummy_input},
        output_artifacts={"traj": dummy_output},
        metrics={"ate": 0.015},
    )

    assert manifest.exists()
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["run_id"] == "exp_001"
    assert data["pipeline_stage"] == "Stage 2 Test"
    assert data["config"]["param"] == 42
    assert "cfg" in data["input_files"]
    assert "traj" in data["output_artifacts"]
    assert data["metrics"]["ate"] == 0.015
    assert len(data["output_artifacts"]["traj"]["sha256"]) == 64

