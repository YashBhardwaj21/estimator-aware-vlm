"""Unit tests for Data Traceability & Contracts.

Covers:
- Task 1.1: Observation ID joins and two-tier timestamp contracts.
- Task 1.2: Coordinate frame schemas and synthetic transform invariants.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from estimator_aware_vlm.cross_session_association import CrossSessionCandidateMatcher, RegistrationInfo


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def observations_file(repo_root) -> Path:
    return repo_root / "data" / "observations" / "object_observations.jsonl"


def test_observation_id_uniqueness_and_presence(observations_file):
    """Assert 100% of observations have an 'obs_id', and that (source_session, obs_id) is strictly unique."""
    assert observations_file.exists(), f"Observations file missing: {observations_file}"
    seen_session_obs_ids = set()
    s1_ids = set()
    s2_ids = set()
    total_count = 0
    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_count += 1
            obs_id = rec.get("obs_id")
            sess = rec.get("source_session")
            assert obs_id is not None and isinstance(obs_id, str) and len(obs_id) > 0, (
                f"Line {total_count} has missing or empty 'obs_id': {rec}"
            )
            assert sess is not None and isinstance(sess, str) and len(sess) > 0

            compound_key = (sess, obs_id)
            assert compound_key not in seen_session_obs_ids, f"Duplicate (session, obs_id) found: {compound_key}"
            seen_session_obs_ids.add(compound_key)

            if "01" in sess:
                assert obs_id not in s1_ids, f"Duplicate obs_id in Session 1: {obs_id}"
                s1_ids.add(obs_id)
            elif "02" in sess:
                assert obs_id not in s2_ids, f"Duplicate obs_id in Session 2: {obs_id}"
                s2_ids.add(obs_id)

    assert total_count == 7956, f"Expected 7,956 observations, found {total_count}"
    assert len(s1_ids) == 2934, f"Expected 2,934 S1 observations, found {len(s1_ids)}"
    assert len(s2_ids) == 5022, f"Expected 5,022 S2 observations, found {len(s2_ids)}"


def test_cluster_observation_ids_join_coverage(repo_root, observations_file):
    """Assert every observation ID referenced in extracted clusters joins to exactly one source record."""
    observations: List[Dict[str, Any]] = []
    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    s1_session = "semantic_mapping_my_desk_recording_01"
    s2_session = "semantic_mapping_my_desk_recording_02"

    s1_obs_by_id = {o["obs_id"]: o for o in observations if o.get("source_session") == s1_session}
    s2_obs_by_id = {o["obs_id"]: o for o in observations if o.get("source_session") == s2_session}

    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s1_session, min_obs=2
    )
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s2_session, min_obs=2
    )

    assert len(s1_clusters) > 0, "No S1 clusters extracted"
    assert len(s2_clusters) > 0, "No S2 clusters extracted"

    # Check S1 join coverage
    for cid, cluster in s1_clusters.items():
        assert len(cluster.observation_ids) >= 2, f"Cluster {cid} has < 2 obs"
        for oid in cluster.observation_ids:
            assert oid in s1_obs_by_id, f"S1 cluster {cid} references missing obs_id '{oid}'"

    # Check S2 join coverage
    for cid, cluster in s2_clusters.items():
        assert len(cluster.observation_ids) >= 2, f"Cluster {cid} has < 2 obs"
        for oid in cluster.observation_ids:
            assert oid in s2_obs_by_id, f"S2 cluster {cid} references missing obs_id '{oid}'"


def test_two_tier_timestamp_semantics_and_monotonicity(observations_file):
    """Verify raw epoch timestamps are preserved, session origins are computed,
    and tracklet timestamps are monotonically non-decreasing."""
    observations_by_session: Dict[str, List[Dict[str, Any]]] = {}
    tracklet_timestamps: Dict[str, List[float]] = {}

    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sess = rec["source_session"]
            ts = float(rec["timestamp"])
            # Assert raw epoch timestamp (ROS epoch timestamps > 1.7e9)
            assert ts > 1.7e9, f"Invalid raw epoch timestamp: {ts}"

            observations_by_session.setdefault(sess, []).append(rec)

            trk = f"{sess}::{rec.get('local_track_id', '')}"
            fr = int(rec.get("frame_index", 0))
            tracklet_timestamps.setdefault(trk, []).append((fr, ts))

    # Verify per-session origin and relative calculation
    for sess, obs_list in observations_by_session.items():
        raw_ts_list = [float(o["timestamp"]) for o in obs_list]
        origin = min(raw_ts_list)
        assert origin > 1.7e9
        session_rel_ts = [t - origin for t in raw_ts_list]
        assert min(session_rel_ts) == 0.0
        assert all(t >= 0.0 for t in session_rel_ts)
        # Verify elapsed duration is positive
        assert max(session_rel_ts) > 0.0

    # Verify monotonicity for each tracklet observation sequence (ordered by frame_index)
    for trk_id, obs_series in tracklet_timestamps.items():
        sorted_series = sorted(obs_series, key=lambda x: x[0])
        for i in range(1, len(sorted_series)):
            assert sorted_series[i][1] >= sorted_series[i - 1][1], (
                f"Monotonicity violation in tracklet {trk_id}: "
                f"frame {sorted_series[i][0]} (ts={sorted_series[i][1]}) < "
                f"frame {sorted_series[i-1][0]} (ts={sorted_series[i-1][1]})"
            )


def test_synthetic_se2_transformation_and_z_invariance():
    """Verify that planar SE(2) transforms horizontal XY while strictly preserving vertical Z."""
    theta_rad = math.radians(30.0)
    R = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad), math.cos(theta_rad)]
    ], dtype=np.float64)
    t = np.array([0.5, -0.2], dtype=np.float64)

    p_orig = np.array([1.0, 2.0, 0.75], dtype=np.float64)

    # Transform XY
    p_trans_xy = R @ p_orig[:2] + t
    p_trans_3d = np.array([p_trans_xy[0], p_trans_xy[1], p_orig[2]], dtype=np.float64)

    # Invariants
    assert abs(p_trans_3d[2] - p_orig[2]) < 1e-12, "Z coordinate must be strictly preserved"
    expected_x = math.cos(theta_rad) * 1.0 - math.sin(theta_rad) * 2.0 + 0.5
    expected_y = math.sin(theta_rad) * 1.0 + math.cos(theta_rad) * 2.0 - 0.2
    assert abs(p_trans_3d[0] - expected_x) < 1e-9
    assert abs(p_trans_3d[1] - expected_y) < 1e-9

    # Decoupled horizontal vs 3D displacement
    disp_xy = float(np.linalg.norm(p_trans_xy - p_orig[:2]))
    disp_3d = float(np.linalg.norm(p_trans_3d - p_orig))
    assert abs(disp_xy - disp_3d) < 1e-9, "When Z is identical, 2D and 3D displacements must match"


def test_double_transform_prevention_in_fusion():
    """Verify that ContextualSceneGraphFusion does not double-transform coordinates tagged as 'reference_frame'."""
    from estimator_aware_vlm.contextual_scene_graph_fusion import ContextualSceneGraphFusion

    fusion = ContextualSceneGraphFusion()
    reg = RegistrationInfo(
        is_valid=True,
        reference_session="semantic_mapping_my_desk_recording_01",
        aligned_session="semantic_mapping_my_desk_recording_02",
        rotation_deg=-15.558,
        translation_m=np.array([-0.1059, 0.0486]),
        anchor_count=10,
        robust_inlier_count=7,
        status="VERIFIED",
        R_12=np.array([[0.9633, 0.2683], [-0.2683, 0.9633]]),
        anchor_rms_residual_m=0.1198,
    )

    already_aligned_centroid = [1.50, -2.00, 0.45]
    obs_meta = {
        "timestamp": 12.5,
        "source_session": "semantic_mapping_my_desk_recording_02",
        "obs_id": "obs_test_01",
        "local_track_id": "trk_test_01",
        "centroid": list(already_aligned_centroid),
        "coordinate_frame": "reference_frame",
        "registration": reg,
    }

    routed = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
    }

    node = fusion.update_or_create_node(
        entity_id="test_entity_01",
        semantic_class="chair",
        routed_result=routed,
        observation_meta=obs_meta,
    )

    assert node.centroid_world is not None
    # Centroid world must match the input directly, without applying R_12 @ c2 + t again
    assert abs(node.centroid_world[0] - already_aligned_centroid[0]) < 1e-6
    assert abs(node.centroid_world[1] - already_aligned_centroid[1]) < 1e-6
    assert abs(node.centroid_world[2] - already_aligned_centroid[2]) < 1e-6


def test_floor_normal_included_angle_and_raw_frame_semantics():
    """Verify that the included angle between floor plane normals is ~3.49 degrees in raw sensor frames."""
    n1 = np.array([-0.0528, -0.9910, 0.1232], dtype=np.float64)
    n2 = np.array([-0.1004, -0.9913, 0.0853], dtype=np.float64)

    cos_angle = float(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle_deg = float(math.degrees(math.acos(cos_angle)))

    # Reconciled angle: ~3.49 degrees (not 2.96 degrees)
    assert 3.45 <= angle_deg <= 3.55, f"Unexpected floor normal included angle: {angle_deg:.3f}°"


def test_physical_instance_match_coordinate_provenance_schema(repo_root, observations_file):
    """Verify PhysicalInstanceMatch outputs populate p1_raw, p2_raw, p2_registered, and coordinate_frame."""
    observations: List[Dict[str, Any]] = []
    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    s1_session = "semantic_mapping_my_desk_recording_01"
    s2_session = "semantic_mapping_my_desk_recording_02"

    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s1_session, min_obs=2
    )
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s2_session, min_obs=2
    )

    reg = RegistrationInfo(
        is_valid=True,
        reference_session="semantic_mapping_my_desk_recording_01",
        aligned_session="semantic_mapping_my_desk_recording_02",
        rotation_deg=-15.558,
        translation_m=np.array([-0.1059, 0.0486]),
        anchor_count=10,
        robust_inlier_count=7,
        status="VERIFIED",
        R_12=np.array([[0.9633, 0.2683], [-0.2683, 0.9633]]),
        anchor_rms_residual_m=0.1198,
    )

    matcher = CrossSessionCandidateMatcher()
    result = matcher.match_entities(s1_clusters, s2_clusters, reg)
    assert result.is_valid
    assert len(result.matched_pairs) > 0

    for match in result.matched_pairs:
        assert match.p1_raw is not None and len(match.p1_raw) == 3
        assert match.p2_raw is not None and len(match.p2_raw) == 3
        assert match.p2_registered is not None and len(match.p2_registered) == 3
        assert match.coordinate_frame == "SESSION_1_WORLD"
        assert match.displacement_xy_m is not None and match.displacement_xy_m >= 0.0


def test_registration_anchor_subsets_and_residuals(repo_root):
    """Verify registration audit subsets: 10 candidates, 7 consensus inliers, 6 strict post-refit inliers."""
    audit_file = repo_root / "outputs" / "registration_audit.json"
    assert audit_file.exists(), f"Missing registration audit: {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchors = data["anchors"]
    assert len(anchors) == 10, f"Expected 10 candidate anchors, got {len(anchors)}"

    consensus_inliers = [a for a in anchors if a.get("is_robust_inlier") is True or a.get("pre_refit_consensus_inlier") is True]
    assert len(consensus_inliers) == 7, f"Expected 7 consensus inliers, got {len(consensus_inliers)}"

    strict_inliers = [a for a in anchors if a.get("residual_m") is not None and a["residual_m"] <= 0.20]
    assert len(strict_inliers) == 6, f"Expected 6 strict inliers <= 0.20m, got {len(strict_inliers)}"

    # Check identities: storage_box_001 exceeds 0.20m, storage_box_007 is within 0.20m
    box_001 = next(a for a in anchors if a["id1"] == "storage box_001")
    assert box_001["residual_m"] == 0.2366
    assert box_001["residual_m"] > 0.20

    box_007 = next(a for a in anchors if a["id1"] == "storage box_007")
    assert box_007["residual_m"] == 0.0971
    assert box_007["residual_m"] < 0.20


def test_registration_leave_one_out_sensitivity_exact_deltas(repo_root):
    """Verify that every LOO entry's delta_rotation_deg and delta_translation_m strictly match parameter differences."""
    audit_file = repo_root / "outputs" / "registration_audit.json"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rot0 = data["rotation_deg"]
    t0 = np.array(data["translation_m"], dtype=np.float64)

    assert abs(rot0 - (-15.558)) < 1e-3
    assert np.allclose(t0, [-0.1059, 0.0486], atol=1e-4)

    loo_list = data["leave_one_out_sensitivity"]
    assert len(loo_list) == 10, f"Expected 10 LOO entries, got {len(loo_list)}"

    for loo in loo_list:
        r = loo["rotation_deg"]
        t = np.array(loo["translation_m"], dtype=np.float64)

        expected_drot = round(abs(r - rot0), 4)
        expected_dt = round(float(np.linalg.norm(t - t0)), 4)

        assert abs(loo["delta_rotation_deg"] - expected_drot) < 1e-3, (
            f"Anchor {loo['excluded_id1']} delta_rot mismatch: stored {loo['delta_rotation_deg']} vs expected {expected_drot}"
        )
        assert abs(loo["delta_translation_m"] - expected_dt) < 1e-3, (
            f"Anchor {loo['excluded_id1']} delta_trans mismatch: stored {loo['delta_translation_m']} vs expected {expected_dt}"
        )

    # Specific check for storage_box_007 (index 10)
    loo_10 = next(x for x in loo_list if x["excluded_anchor_index"] == 10)
    assert loo_10["excluded_id1"] == "storage box_007"
    assert loo_10["rotation_deg"] == -16.006
    assert abs(loo_10["delta_rotation_deg"] - 0.448) < 1e-3, f"Expected 0.448°, got {loo_10['delta_rotation_deg']}"
    assert abs(loo_10["delta_translation_m"] - 0.0471) < 1e-3, f"Expected 0.0471m, got {loo_10['delta_translation_m']}"


def test_relocation_threshold_reclassified_as_empirical_heuristic(repo_root):
    """Verify relocation threshold is classified as empirical heuristic residual gate, not calibrated covariance."""
    from estimator_aware_vlm.evidence_adapter import load_registration_audit

    audit_file = repo_root / "outputs" / "registration_audit.json"
    reg = load_registration_audit(audit_file)

    assert reg.is_valid
    assert abs(reg.inlier_rms_residual_m - 0.1198) < 1e-3
    # 3 * RMS = ~0.3594m
    empirical_gate = 3.0 * reg.inlier_rms_residual_m
    assert abs(empirical_gate - 0.3594) < 1e-3

    # Must be explicitly marked as not a calibrated covariance
    assert reg.is_calibrated_covariance is False


def test_rgb_depth_association_skew_distribution(repo_root):
    """Verify exact RGB-D association row counts and skew distributions in associations.txt.
    
    Demonstrates that associations are NOT strictly bounded by 20ms (S1 max skew 46.254ms).
    """
    s1_assoc_file = repo_root / "data" / "raw" / "my_desk_recording_01" / "associations.txt"
    s2_assoc_file = repo_root / "data" / "raw" / "my_desk_recording_02" / "associations.txt"

    assert s1_assoc_file.exists(), f"S1 associations missing: {s1_assoc_file}"
    assert s2_assoc_file.exists(), f"S2 associations missing: {s2_assoc_file}"

    def parse_associations(p: Path) -> List[float]:
        skews = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    t_rgb = float(parts[0])
                    t_depth = float(parts[2])
                    skew_ms = abs(t_rgb - t_depth) * 1000.0
                    skews.append(skew_ms)
        return skews

    s1_skews = parse_associations(s1_assoc_file)
    s2_skews = parse_associations(s2_assoc_file)

    # Session 1: 5,108 rows
    assert len(s1_skews) == 5108, f"Expected 5,108 S1 rows, got {len(s1_skews)}"
    s1_mean = float(np.mean(s1_skews))
    s1_median = float(np.median(s1_skews))
    s1_p95 = float(np.percentile(s1_skews, 95))
    s1_max = float(np.max(s1_skews))
    s1_le_20 = sum(1 for s in s1_skews if s <= 20.0)
    s1_gt_20 = sum(1 for s in s1_skews if s > 20.0)

    assert abs(s1_mean - 13.391) < 0.05
    assert abs(s1_median - 12.318) < 0.05
    assert abs(s1_p95 - 21.034) < 0.05
    assert abs(s1_max - 46.254) < 0.05
    assert s1_le_20 == 4695
    assert s1_gt_20 == 413
    assert abs(s1_le_20 / len(s1_skews) - 0.91915) < 1e-4

    # Session 2: 5,283 rows
    assert len(s2_skews) == 5283, f"Expected 5,283 S2 rows, got {len(s2_skews)}"
    s2_mean = float(np.mean(s2_skews))
    s2_median = float(np.median(s2_skews))
    s2_p95 = float(np.percentile(s2_skews, 95))
    s2_max = float(np.max(s2_skews))
    s2_le_20 = sum(1 for s in s2_skews if s <= 20.0)
    s2_gt_20 = sum(1 for s in s2_skews if s > 20.0)

    assert abs(s2_mean - 12.539) < 0.05
    assert abs(s2_median - 12.310) < 0.05
    assert abs(s2_p95 - 12.991) < 0.05
    assert abs(s2_max - 22.140) < 0.05
    assert s2_le_20 == 5133
    assert s2_gt_20 == 150
    assert abs(s2_le_20 / len(s2_skews) - 0.97161) < 1e-4


def test_pose_provenance_and_groundtruth_contract(repo_root, observations_file):
    """Verify pose_provenance migration and groundtruth placeholder contract.
    
    100% of observations must report ESTIMATED_SLAM_TRAJECTORY (0% MEASURED_TRAJECTORY).
    groundtruth.txt files must explicitly declare synthetic identity placeholders.
    """
    total_records = 0
    estimated_count = 0
    measured_count = 0

    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_records += 1
            prov = rec.get("pose_provenance")
            if prov == "ESTIMATED_SLAM_TRAJECTORY":
                estimated_count += 1
            elif prov == "MEASURED_TRAJECTORY":
                measured_count += 1

    assert total_records == 7956
    assert estimated_count == 7956, f"Expected 7,956 ESTIMATED_SLAM_TRAJECTORY records, got {estimated_count}"
    assert measured_count == 0, f"Found {measured_count} stale MEASURED_TRAJECTORY records"

    # Verify groundtruth.txt contract
    for sess in ["my_desk_recording_01", "my_desk_recording_02"]:
        gt_file = repo_root / "data" / "raw" / sess / "groundtruth.txt"
        assert gt_file.exists()
        with open(gt_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            assert "SYNTHETIC IDENTITY PLACEHOLDER - NO GROUND TRUTH AVAILABLE" in first_line


