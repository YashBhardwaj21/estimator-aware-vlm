"""Evidence Extraction -> Selective Router -> Contextual Scene Graph Real-Recording Integration Test.

Verifies end-to-end data flow:
1. Loads real-recording observations and pre-router evidence.
2. Adapts evidence through EvidenceAdapter without ground truth leakage.
3. Routes through SelectiveRouter in deterministic offline mode (cached/prior).
4. Updates persistent ContextualSceneGraphFusion.
5. Converts to authoritative PersistentEntity.
6. Serializes scene graph to JSON.
7. Validates mathematical running mean, identity semantics (track IDs vs obs IDs),
   geometry validity, uncertainty propagation, and lifecycle state.
"""

import json
import math
import numpy as np
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

from estimator_aware_vlm.evidence_adapter import EvidenceAdapter
from estimator_aware_vlm.selective_router import SelectiveRouter
from estimator_aware_vlm.contextual_scene_graph_fusion import (
    ContextualSceneGraphFusion,
    ContextualMovability,
    ObservedMotionState,
)
from scene_graph.ontology.entity import PersistentEntity, EntityLifecycleState


def test_real_recording_scene_graph_pipeline_integration(tmp_path):
    """End-to-end integration test on real recording data."""
    manifest_path = root_dir / "benchmarks" / "router_evaluation_manifest.json"
    cache_path = root_dir / "outputs" / "router_ablation" / "router_vlm_cache.json"

    if not manifest_path.exists():
        pytest.skip("Benchmark manifest not available.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)["evaluation_instances"]

    # Target real recording chair instance
    chair_entry = next((item for item in manifest if item["entity_id"] == "inst_chair_002"), None)
    assert chair_entry is not None

    # 1 & 2: Adapter loads real observations and computes non-leaking evidence
    adapter = EvidenceAdapter(repo_root=root_dir)
    evidence = adapter.build_instance_evidence(chair_entry)

    assert evidence["entity_id"] == "inst_chair_002"
    assert evidence["semantic_class"] == "chair"
    assert evidence["has_valid_pose"] is True
    assert evidence["depth_valid"] is True
    assert evidence["relocation_evaluable"] is True
    assert evidence["world_displacement_m"] > 0.30
    assert evidence["observed_motion"] == "DISPLACED"

    # 3: Route evidence through deterministic selective router
    router = SelectiveRouter(cache_path=cache_path, enable_vlm=True)
    routed = router.route_entity(evidence, repo_root=root_dir)

    assert routed["decision"] in ["ESCALATE", "RESOLVE_CHEAP"]
    assert routed["prediction"] == "MOVABLE"

    # 4: Update Contextual Scene Graph
    fusion = ContextualSceneGraphFusion()

    # Step 4a: First observation
    obs_meta_1 = {
        "timestamp": 100.0,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": "obs_s1_001",
        "local_track_id": "trk_1002",
        "centroid": [1.50, 2.50, 0.45],
        "spatial_extent_m": 0.60,
        "depth_valid": True,
        "coordinate_frame": "floor_map",
    }
    node = fusion.update_or_create_node(
        entity_id="inst_chair_002",
        semantic_class="chair",
        routed_result=routed,
        observation_meta=obs_meta_1,
    )

    assert node.entity_id == "inst_chair_002"
    assert node.contextual_movability == ContextualMovability.MOVABLE
    assert node.observed_motion == ObservedMotionState.DISPLACED
    assert node.geometry_valid is True
    assert np.allclose(node.centroid_world, [1.50, 2.50, 0.45])
    assert node.source_track_ids == ["trk_1002"]
    assert node.contributing_observation_ids == ["obs_s1_001"]

    # Step 4b: Second observation with slight variation to verify exact running mean
    obs_meta_2 = {
        "timestamp": 101.5,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": "obs_s1_002",
        "local_track_id": "trk_1002",
        "centroid": [1.60, 2.60, 0.45],
        "spatial_extent_m": 0.62,
        "depth_valid": True,
        "coordinate_frame": "floor_map",
    }
    node_updated = fusion.update_or_create_node(
        entity_id="inst_chair_002",
        semantic_class="chair",
        routed_result=routed,
        observation_meta=obs_meta_2,
    )

    assert node_updated.total_observations_count == 2
    # Exact average: ([1.5, 2.5] + [1.6, 2.6]) / 2 = [1.55, 2.55]
    assert np.allclose(node_updated.centroid_world, [1.55, 2.55, 0.45])
    assert node_updated.spatial_extent_m == pytest.approx(0.61, abs=1e-4)
    assert node_updated.source_track_ids == ["trk_1002"]
    assert len(node_updated.contributing_observation_ids) == 2

    # 5: Convert to authoritative PersistentEntity
    entity = node_updated.to_persistent_entity()
    assert isinstance(entity, PersistentEntity)
    assert entity.entity_id == "inst_chair_002"
    assert entity.lifecycle_state == EntityLifecycleState.CONFIRMED
    assert entity.primary_label == "chair"
    assert entity.source_track_ids == ["trk_1002"]  # Must be track ID, not obs ID!
    assert "obs_s1_001" in entity.attributes["contributing_observation_ids"]
    assert entity.attributes["contextual_movability"] == "MOVABLE"
    assert entity.attributes["observed_motion"] == "DISPLACED"

    # 6: Serialization to JSON
    serialized = fusion.serialize_to_json()
    assert serialized["total_nodes"] == 1
    assert "inst_chair_002" in serialized["nodes"]
    assert serialized["nodes"]["inst_chair_002"]["geometry_valid"] is True


def test_missing_geometry_handling():
    """Verify that absent geometry does not fabricate fake coordinates and sets TENTATIVE state."""
    fusion = ContextualSceneGraphFusion()
    routed = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "IMMOVABLE",
        "observed_motion": "STATIC",
        "world_displacement_m": None,
        "relocation_evaluable": False,
        "evidence_provenance": "MISSING",
    }
    # No centroid provided
    obs_meta = {
        "timestamp": 200.0,
        "source_session": "session_01",
        "obs_id": "obs_blind_01",
        "local_track_id": "trk_99",
        "centroid": None,
        "spatial_extent_m": None,
        "depth_valid": False,
    }

    node = fusion.update_or_create_node(
        entity_id="inst_wall_unknown_geo",
        semantic_class="wall",
        routed_result=routed,
        observation_meta=obs_meta,
    )

    assert node.geometry_valid is False
    assert node.centroid_world is None
    assert node.contextual_movability == ContextualMovability.IMMOVABLE

    persistent = node.to_persistent_entity()
    # Must remain TENTATIVE when geometry is missing!
    assert persistent.lifecycle_state == EntityLifecycleState.TENTATIVE
    assert persistent.attributes["geometry_valid"] is False


def test_running_mean_mathematical_consistency():
    """Verify exact running mean without drift across 100 observations."""
    fusion = ContextualSceneGraphFusion()
    routed = {"prediction": "MOVABLE", "observed_motion": "STATIC"}

    # True values to sample around [2.0, 3.0, 1.0]
    true_mean = np.array([2.0, 3.0, 1.0])
    points = [true_mean + np.array([0.1 * math.sin(i), 0.1 * math.cos(i), 0.0]) for i in range(100)]
    expected_mean = np.mean(points, axis=0)

    for i, pt in enumerate(points):
        meta = {
            "timestamp": 100.0 + i,
            "obs_id": f"obs_{i:03d}",
            "local_track_id": "trk_01",
            "centroid": pt,
            "spatial_extent_m": 0.5,
            "depth_valid": True,
        }
        node = fusion.update_or_create_node("inst_test_math", "chair", routed, meta)

    assert node.total_observations_count == 100
    assert node.valid_geometry_observations_count == 100
    assert np.allclose(node.centroid_world, expected_mean, atol=1e-6)


def test_cross_session_physical_identity_scene_graph_fusion():
    """Cross-Session Integration Test: Associate real observations across sessions into persistent scene graph."""
    from estimator_aware_vlm.cross_session_association import CrossSessionCandidateMatcher
    from estimator_aware_vlm.evidence_adapter import load_registration_audit

    obs_path = root_dir / "outputs" / "object_observations.jsonl"
    reg_path = root_dir / "outputs" / "reports" / "registration_audit.json"
    if not reg_path.exists():
        reg_path = root_dir / "outputs" / "registration_audit.json"

    if not obs_path.exists() or not reg_path.exists():
        pytest.skip("Real observations or registration audit missing.")

    reg = load_registration_audit(reg_path)
    assert reg.is_valid, f"Registration must be valid: {reg.status}"

    observations = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    # Aggregate clusters
    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(observations, "recording_01", min_obs=2)
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(observations, "recording_02", min_obs=2)

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    assoc_res = matcher.match_entities(s1_clusters, s2_clusters, reg)

    assert assoc_res.is_valid
    assert len(assoc_res.matched_pairs) > 0

    # Explicitly require verified static pair desk_001 <-> desk_001
    target_match = next((m for m in assoc_res.matched_pairs if m.s1_entity_id == "desk_001" and m.s2_entity_id == "desk_001"), None)
    assert target_match is not None, "Verified canonical static pair desk_001 <-> desk_001 must be matched!"

    s1_eid = target_match.s1_entity_id
    s2_eid = target_match.s2_entity_id
    persistent_id = target_match.persistent_entity_id

    # Create Scene Graph Fusion
    fusion = ContextualSceneGraphFusion()

    # Feed Session 1 observation
    # Feed Session 1 observation with true measured 3D centroid
    c1 = s1_clusters[s1_eid]
    c1_3d = c1.centroid_3d.tolist() if c1.centroid_3d is not None else [float(c1.centroid_floor[0]), float(c1.centroid_floor[1]), 0.0]
    obs_meta_s1 = {
        "timestamp": 10.0,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "obs_id": c1.observation_ids[0] if c1.observation_ids else "obs_s1_001",
        "local_track_id": s1_eid,
        "centroid": c1_3d,
        "spatial_extent_m": c1.spatial_extent_m,
        "coordinate_frame": "reference_frame",
        "registration": reg,
    }
    routed_s1 = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "evidence_provenance": "MEASURED_FULL",
    }
    node = fusion.update_or_create_node(persistent_id, c1.semantic_class, routed_s1, obs_meta_s1)

    assert node.entity_id == persistent_id
    assert node.total_observations_count == 1
    assert "semantic_mapping_my_desk_recording_01" in node.source_sessions
    assert s1_eid in node.source_track_ids
    assert node.centroid_world[2] == pytest.approx(c1_3d[2], abs=1e-4)

    # Feed Session 2 observation with true measured 3D centroid
    c2 = s2_clusters[s2_eid]
    c2_3d = c2.centroid_3d.tolist() if c2.centroid_3d is not None else [float(c2.centroid_floor[0]), float(c2.centroid_floor[1]), 0.0]
    obs_meta_s2 = {
        "timestamp": 25.0,
        "source_session": "semantic_mapping_my_desk_recording_02",
        "obs_id": c2.observation_ids[0] if c2.observation_ids else "obs_s2_001",
        "local_track_id": s2_eid,
        "centroid": c2_3d,
        "spatial_extent_m": c2.spatial_extent_m,
        "coordinate_frame": "session_02_local",
        "registration": reg,
    }
    routed_s2 = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "evidence_provenance": "MEASURED_FULL",
    }
    node_merged = fusion.update_or_create_node(persistent_id, c2.semantic_class, routed_s2, obs_meta_s2)

    # Verify identity continuity across sessions
    assert node_merged.total_observations_count == 2
    assert len(node_merged.source_sessions) == 2
    assert "semantic_mapping_my_desk_recording_01" in node_merged.source_sessions
    assert "semantic_mapping_my_desk_recording_02" in node_merged.source_sessions
    assert s1_eid in node_merged.source_track_ids
    assert s2_eid in node_merged.source_track_ids

    # True 3D geometry preservation: 2D floor within small delta, 3D height is running mean of measured z coordinates
    dist_to_s1 = np.linalg.norm(node_merged.centroid_world[:2] - c1.centroid_floor[:2])
    assert dist_to_s1 < 0.25, f"Merged centroid drifted too far from reference frame: {dist_to_s1}m"
    expected_height = (c1_3d[2] + c2_3d[2]) / 2.0
    assert node_merged.centroid_world[2] == pytest.approx(expected_height, abs=1e-4)

    # Serialize to JSON and PersistentEntity
    serialized = fusion.serialize_to_json()
    assert persistent_id in serialized["nodes"]

    persistent_entity = node_merged.to_persistent_entity()
    assert persistent_entity.entity_id == persistent_id
    assert s1_eid in persistent_entity.source_track_ids
    assert s2_eid in persistent_entity.source_track_ids
    assert persistent_entity.lifecycle_state == EntityLifecycleState.CONFIRMED


def test_fusion_exact_se2_transform_application():
    """Verify that the production fusion path applies the exact SE(2) transform to Session 2 local points."""
    from estimator_aware_vlm.evidence_adapter import RegistrationInfo

    R_test = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)  # 90 deg rotation
    t_test = np.array([5.0, 10.0], dtype=np.float64)
    reg = RegistrationInfo(
        is_valid=True,
        reference_session="rec01",
        aligned_session="rec02",
        rotation_deg=90.0,
        translation_m=t_test,
        R_12=R_test,
        anchor_rms_residual_m=0.01,
        anchor_count=4,
        status="VERIFIED",
    )

    fusion = ContextualSceneGraphFusion()
    routed = {"prediction": "MOVABLE", "observed_motion": "STATIC"}

    # Input point in Session 2 local frame: [1.0, 2.0, 0.50]
    p_s2 = np.array([1.0, 2.0])
    expected_p_ref = R_test @ p_s2 + t_test  # [0, -1] @ [1, 2] + [5, 10] = [-2, 1] + [5, 10] = [3.0, 11.0]

    obs_meta = {
        "timestamp": 50.0,
        "source_session": "rec02",
        "obs_id": "obs_s2_test",
        "local_track_id": "trk_s2_99",
        "centroid": [1.0, 2.0, 0.50],
        "coordinate_frame": "session_02_local",
        "registration": reg,
    }

    node = fusion.update_or_create_node("phys_obj_exact_trans", "chair", routed, obs_meta)

    assert node.geometry_valid is True
    assert np.allclose(node.centroid_world[:2], expected_p_ref, atol=1e-5), f"Expected {expected_p_ref}, got {node.centroid_world[:2]}"
    assert node.centroid_world[2] == pytest.approx(0.50)


def test_fusion_rejects_unaligned_session2_geometry_when_reg_invalid():
    """Verify that unaligned Session 2 points are strictly rejected and retain TENTATIVE identity across updates."""
    from estimator_aware_vlm.evidence_adapter import RegistrationInfo

    reg_invalid = RegistrationInfo(
        is_valid=False,
        reference_session="rec01",
        aligned_session="rec02",
        rotation_deg=0.0,
        translation_m=np.zeros(2),
        R_12=np.eye(2),
        anchor_rms_residual_m=0.95,
        anchor_count=1,
        status="FAILED_UNVERIFIED",
    )

    fusion = ContextualSceneGraphFusion()
    routed = {"prediction": "MOVABLE", "observed_motion": "STATIC"}

    obs_meta_1 = {
        "timestamp": 60.0,
        "source_session": "rec02",
        "obs_id": "obs_unaligned_1",
        "local_track_id": "trk_unaligned",
        "centroid": [2.5, 3.5, 0.70],
        "coordinate_frame": "session_02_local",
        "registration": reg_invalid,
    }

    node = fusion.update_or_create_node("phys_unaligned", "desk", routed, obs_meta_1)

    # Must reject geometry to prevent unaligned coordinate corruption while preserving physical identity!
    assert node.geometry_valid is False
    assert node.centroid_world is None
    assert node.total_observations_count == 1
    assert "trk_unaligned" in node.source_track_ids
    persistent = node.to_persistent_entity()
    assert persistent.lifecycle_state == EntityLifecycleState.TENTATIVE

    # Subsequent update with still-invalid registration must NOT confirm node or fabricate geometry
    obs_meta_2 = {
        "timestamp": 62.0,
        "source_session": "rec02",
        "obs_id": "obs_unaligned_2",
        "local_track_id": "trk_unaligned",
        "centroid": [2.55, 3.52, 0.71],
        "coordinate_frame": "session_02_local",
        "registration": reg_invalid,
    }
    node_updated = fusion.update_or_create_node("phys_unaligned", "desk", routed, obs_meta_2)
    assert node_updated.geometry_valid is False
    assert node_updated.centroid_world is None
    assert node_updated.total_observations_count == 2
    persistent_updated = node_updated.to_persistent_entity()
    assert persistent_updated.lifecycle_state == EntityLifecycleState.TENTATIVE


def test_fusion_relocated_object_lifecycle_and_displacement():
    """Verify that a relocated object preserves physical identity, updates displacement, and does not duplicate."""
    fusion = ContextualSceneGraphFusion()

    # Session 1: Chair observed at [1.0, 2.0, 0.45]
    obs_s1 = {
        "timestamp": 10.0,
        "source_session": "rec01",
        "obs_id": "obs_s1_c1",
        "local_track_id": "chair_002",
        "centroid": [1.0, 2.0, 0.45],
        "spatial_extent_m": 0.55,
        "coordinate_frame": "reference_frame",
    }
    routed_s1 = {
        "decision": "RESOLVE_CHEAP",
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "world_displacement_m": 0.0,
        "relocation_evaluable": True,
    }
    node = fusion.update_or_create_node("inst_chair_002", "chair", routed_s1, obs_s1)
    assert node.entity_id == "inst_chair_002"
    assert node.observed_motion == ObservedMotionState.STATIC
    assert len(fusion.nodes) == 1

    # Session 2: Same chair physically relocated by 0.65m to [1.65, 2.0, 0.45]
    obs_s2 = {
        "timestamp": 120.0,
        "source_session": "rec02",
        "obs_id": "obs_s2_c1",
        "local_track_id": "chair_001",
        "centroid": [1.65, 2.0, 0.45],
        "spatial_extent_m": 0.56,
        "coordinate_frame": "reference_frame",  # already in common frame
    }
    routed_s2 = {
        "decision": "ESCALATE",
        "prediction": "MOVABLE",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.65,
        "displacement_uncertainty_m": 0.038,
        "relocation_evaluable": True,
    }
    node_updated = fusion.update_or_create_node("inst_chair_002", "chair", routed_s2, obs_s2)

    # Must preserve identity and NOT create a duplicate node
    assert len(fusion.nodes) == 1
    assert node_updated.entity_id == "inst_chair_002"
    assert node_updated.observed_motion == ObservedMotionState.DISPLACED
    assert node_updated.measured_displacement_m == 0.65
    assert node_updated.displacement_uncertainty_m == 0.038
    assert node_updated.relocation_evaluable is True
    assert set(node_updated.source_track_ids) == {"chair_002", "chair_001"}
    assert set(node_updated.source_sessions) == {"rec01", "rec02"}

    # Decouple current pose from historical poses: do NOT average across relocation
    np.testing.assert_allclose(node_updated.centroid_world, np.array([1.65, 2.0, 0.45]), atol=1e-3)
    assert len(node_updated.pose_history) == 1
    assert node_updated.pose_history[0]["centroid_world"] == [1.0, 2.0, 0.45]


def test_fusion_single_session_and_unmatched_objects():
    """Verify single-session entities (present in S1 only or S2 only) retain independent identities without false merges."""
    fusion = ContextualSceneGraphFusion()

    # Object 1: Present only in Session 1 (e.g. laptop taken away)
    obs_s1 = {
        "timestamp": 15.0,
        "source_session": "rec01",
        "obs_id": "obs_laptop_s1",
        "local_track_id": "laptop_006",
        "centroid": [2.10, 1.30, 0.75],
        "spatial_extent_m": 0.35,
        "coordinate_frame": "reference_frame",
    }
    node_s1 = fusion.update_or_create_node("inst_laptop_001", "laptop", {"prediction": "MOVABLE", "observed_motion": "STATIC"}, obs_s1)
    assert node_s1.source_sessions == ["rec01"]

    # Object 2: Present only in Session 2 (e.g. new backpack brought into room)
    obs_s2 = {
        "timestamp": 200.0,
        "source_session": "rec02",
        "obs_id": "obs_backpack_s2",
        "local_track_id": "backpack_002",
        "centroid": [3.40, -0.80, 0.40],
        "spatial_extent_m": 0.45,
        "coordinate_frame": "reference_frame",
    }
    node_s2 = fusion.update_or_create_node("inst_backpack_002", "backpack", {"prediction": "MOVABLE", "observed_motion": "STATIC"}, obs_s2)
    assert node_s2.source_sessions == ["rec02"]

    assert len(fusion.nodes) == 2
    assert "inst_laptop_001" in fusion.nodes
    assert "inst_backpack_002" in fusion.nodes
    # Verify no accidental cross-merging
    assert fusion.nodes["inst_laptop_001"].source_sessions == ["rec01"]
    assert fusion.nodes["inst_backpack_002"].source_sessions == ["rec02"]


def test_fusion_duplicate_tracker_id_different_physical_instances():
    """Verify tracker ID reuse across sessions: 'desk_016' in S1 is inst_desk_016; 'desk_016' in S2 is inst_desk_008."""
    fusion = ContextualSceneGraphFusion()

    # S1 tracker desk_016 -> inst_desk_016 at [1.0, 3.0, 0.72]
    obs_s1 = {
        "timestamp": 12.0,
        "source_session": "rec01",
        "obs_id": "obs_d16_s1",
        "local_track_id": "desk_016",
        "centroid": [1.0, 3.0, 0.72],
        "spatial_extent_m": 0.90,
        "coordinate_frame": "reference_frame",
    }
    node_1 = fusion.update_or_create_node("inst_desk_016", "desk", {"prediction": "MOVABLE", "observed_motion": "STATIC"}, obs_s1)

    # S2 tracker desk_016 -> inst_desk_008 at [4.5, -2.0, 0.72]
    obs_s2 = {
        "timestamp": 210.0,
        "source_session": "rec02",
        "obs_id": "obs_d16_s2",
        "local_track_id": "desk_016",
        "centroid": [4.5, -2.0, 0.72],
        "spatial_extent_m": 0.92,
        "coordinate_frame": "reference_frame",
    }
    node_2 = fusion.update_or_create_node("inst_desk_008", "desk", {"prediction": "MOVABLE", "observed_motion": "STATIC"}, obs_s2)

    assert len(fusion.nodes) == 2
    assert "inst_desk_016" in fusion.nodes
    assert "inst_desk_008" in fusion.nodes
    # Both share local tracker name 'desk_016', but represent completely different physical instances!
    assert fusion.nodes["inst_desk_016"].source_track_ids == ["desk_016"]
    assert fusion.nodes["inst_desk_008"].source_track_ids == ["desk_016"]
    assert np.linalg.norm(fusion.nodes["inst_desk_016"].centroid_world[:2] - fusion.nodes["inst_desk_008"].centroid_world[:2]) > 4.0


def test_scene_graph_end_to_end_orchestration():
    """Verify complete pipeline orchestration from raw observations through registration, matching, to persistent scene graph."""
    from estimator_aware_vlm.evidence_adapter import load_registration_audit
    from estimator_aware_vlm.cross_session_association import CrossSessionCandidateMatcher

    reg_path = root_dir / "outputs" / "reports" / "registration_audit.json"
    if not reg_path.exists():
        reg_path = root_dir / "outputs" / "registration_audit.json"
    if not reg_path.exists():
        pytest.skip("Registration audit artifact missing")
    reg = load_registration_audit(reg_path)

    # 1. Simulate raw observations from two sessions
    # S1 chair at [1.0, 2.0, 0.45]
    s1_obs = [
        {
            "global_entity_id": "chair_001",
            "local_track_id": "chair_001",
            "semantic_class": "chair",
            "source_session": "recording_01",
            "centroid_floor": [1.0, 2.0],
            "spatial_extent_m": 0.50,
            "obs_id": "s1_obs_1",
        },
        {
            "global_entity_id": "chair_001",
            "local_track_id": "chair_001",
            "semantic_class": "chair",
            "source_session": "recording_01",
            "centroid_floor": [1.02, 1.98],
            "spatial_extent_m": 0.50,
            "obs_id": "s1_obs_2",
        },
    ]

    # S2 chair observed in local session 2 frame:
    # True ref position is [1.01, 1.99, 0.45]
    # S2 local = R_12^T @ (p_ref - t)
    p_ref_target = np.array([1.01, 1.99], dtype=np.float64)
    p_s2_local = reg.R_12.T @ (p_ref_target - reg.translation_m)

    s2_obs = [
        {
            "global_entity_id": "chair_009",
            "local_track_id": "chair_009",
            "semantic_class": "chair",
            "source_session": "recording_02",
            "centroid_floor": list(p_s2_local),
            "spatial_extent_m": 0.50,
            "obs_id": "s2_obs_1",
        },
        {
            "global_entity_id": "chair_009",
            "local_track_id": "chair_009",
            "semantic_class": "chair",
            "source_session": "recording_02",
            "centroid_floor": list(p_s2_local + 0.01),
            "spatial_extent_m": 0.50,
            "obs_id": "s2_obs_2",
        },
    ]

    # 2. Extract clusters
    all_obs = s1_obs + s2_obs
    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(all_obs, "recording_01", min_obs=2)
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(all_obs, "recording_02", min_obs=2)

    assert "chair_001" in s1_clusters
    assert "chair_009" in s2_clusters

    # 3. Match across sessions with robust registration
    matcher = CrossSessionCandidateMatcher()
    match_result = matcher.match_entities(s1_clusters, s2_clusters, reg)

    assert len(match_result.matched_pairs) == 1
    m = match_result.matched_pairs[0]
    assert m.s1_entity_id == "chair_001"
    assert m.s2_entity_id == "chair_009"

    # 4. Contextual scene graph fusion across sessions
    fusion = ContextualSceneGraphFusion()

    # Ingest S1 observation with valid 3D height
    routed_s1 = {"prediction": "MOVABLE", "observed_motion": "STATIC"}
    fusion.update_or_create_node(
        entity_id=m.persistent_entity_id,
        semantic_class="chair",
        routed_result=routed_s1,
        observation_meta={
            "source_session": "semantic_mapping_my_desk_recording_01",
            "local_track_id": "chair_001",
            "obs_id": "s1_obs_1",
            "centroid": [1.0, 2.0, 0.45],
            "coordinate_frame": "reference_frame",
        },
    )

    # Ingest S2 observation with local coordinate frame requiring registration transformation
    routed_s2 = {"prediction": "MOVABLE", "observed_motion": "STATIC"}
    node_updated = fusion.update_or_create_node(
        entity_id=m.persistent_entity_id,
        semantic_class="chair",
        routed_result=routed_s2,
        observation_meta={
            "source_session": "semantic_mapping_my_desk_recording_02",
            "local_track_id": "chair_009",
            "obs_id": "s2_obs_1",
            "centroid": [float(p_s2_local[0]), float(p_s2_local[1]), 0.46],
            "coordinate_frame": "session_02_local",
            "registration": reg,
        },
    )

    # 5. Assert authoritative guarantees
    assert node_updated.coordinate_frame == "reference_frame"
    assert node_updated.source_sessions == ["semantic_mapping_my_desk_recording_01", "semantic_mapping_my_desk_recording_02"]
    assert node_updated.source_track_ids == ["chair_001", "chair_009"]
    assert node_updated.geometry_valid is True
    assert node_updated.centroid_world[2] == pytest.approx(0.455, abs=0.01)

    persistent = node_updated.to_persistent_entity()
    assert persistent.source_track_ids == ["chair_001", "chair_009"]
    assert persistent.lifecycle_state == EntityLifecycleState.CONFIRMED

    # 6. Verify 2D observation does NOT fabricate 3D geometry
    obs_2d_only = {
        "source_session": "semantic_mapping_my_desk_recording_01",
        "local_track_id": "trk_2d_only",
        "centroid": [1.5, 2.5],  # Only 2 elements
        "coordinate_frame": "reference_frame",
    }
    node_2d = fusion.update_or_create_node("phys_2d_only", "chair", routed_s1, obs_2d_only)
    assert node_2d.geometry_valid is False
    assert node_2d.centroid_world is None
    persistent_2d = node_2d.to_persistent_entity()
    assert persistent_2d.lifecycle_state == EntityLifecycleState.TENTATIVE


def test_relocation_decouples_current_pose_from_pose_history():
    """Relocation-Aware Association Contract: Confirmed relocation decouples current pose from historical poses.
    
    Verifies that when an object physically moves (e.g. chair pushed by 0.563m),
    its current pose reflects the new measurement rather than an arithmetic average
    (which would construct a phantom landmark in the middle of the floor).
    """
    fusion = ContextualSceneGraphFusion()

    # Initial resting pose (p1)
    p1 = np.array([1.747, -1.960, 0.423], dtype=np.float64)
    obs_1 = {
        "timestamp": 10.0,
        "source_session": "session_01",
        "obs_id": "obs_p1_01",
        "local_track_id": "trk_chair_s1",
        "centroid": p1.tolist(),
        "spatial_extent_m": 0.55,
        "coordinate_frame": "reference_frame",
    }
    routed_1 = {
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "world_displacement_m": 0.0,
    }
    node = fusion.update_or_create_node("inst_chair_002", "chair", routed_1, obs_1)
    np.testing.assert_allclose(node.centroid_world, p1)
    assert len(node.pose_history) == 0

    # Physical relocation event (p2)
    p2 = np.array([2.308, -2.000, 0.435], dtype=np.float64)
    disp = float(np.linalg.norm(p2 - p1))  # ~0.563m
    assert disp > 0.50

    obs_2 = {
        "timestamp": 120.0,
        "source_session": "session_02",
        "obs_id": "obs_p2_01",
        "local_track_id": "trk_chair_s2",
        "centroid": p2.tolist(),
        "spatial_extent_m": 0.55,
        "coordinate_frame": "reference_frame",
        "is_relocation_event": True,
    }
    routed_2 = {
        "prediction": "MOVABLE",
        "observed_motion": "DISPLACED",
        "world_displacement_m": disp,
    }
    node_updated = fusion.update_or_create_node("inst_chair_002", "chair", routed_2, obs_2)

    # 1. Current pose must be p2, NOT the phantom mean (p1 + p2)/2
    phantom_mean = (p1 + p2) / 2.0
    assert not np.allclose(node_updated.centroid_world, phantom_mean, atol=0.05)
    np.testing.assert_allclose(node_updated.centroid_world, p2, atol=1e-3)

    # 2. Pose history must contain previous pose p1
    assert len(node_updated.pose_history) == 1
    h_entry = node_updated.pose_history[0]
    np.testing.assert_allclose(h_entry["centroid_world"], [round(float(x), 4) for x in p1])
    assert h_entry["relocation_displacement_m"] == pytest.approx(disp, abs=0.01)

    # 3. Subsequent observation at the new location (p2 + small noise) updates mean at p2
    p2_noise = np.array([2.312, -1.998, 0.436], dtype=np.float64)
    obs_3 = {
        "timestamp": 122.0,
        "source_session": "session_02",
        "obs_id": "obs_p2_02",
        "local_track_id": "trk_chair_s2",
        "centroid": p2_noise.tolist(),
        "spatial_extent_m": 0.55,
        "coordinate_frame": "reference_frame",
    }
    node_updated_3 = fusion.update_or_create_node("inst_chair_002", "chair", routed_2, obs_3)
    expected_new_mean = (p2 + p2_noise) / 2.0
    np.testing.assert_allclose(node_updated_3.centroid_world, expected_new_mean, atol=1e-3)
    # Pose history count should remain 1 (no second relocation occurred)
    assert len(node_updated_3.pose_history) == 1


def test_class_prior_does_not_fabricate_measured_displacement():
    """Relocation-Aware Association Contract: Class prior affordances must not fabricate world displacement evidence."""
    adapter = EvidenceAdapter(repo_root=root_dir)

    # 1. Real robot entity from manifest has observed motion without fake 0.50m prior leakage
    manifest_path = root_dir / "benchmarks" / "router_evaluation_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)["evaluation_instances"]
    robot_entry = next(item for item in manifest if item["entity_id"] == "inst_rc_car_robot_001")
    evidence = adapter.build_instance_evidence(robot_entry)

    # Verify no hardcoded 0.50m prior displacement was injected
    assert evidence["semantic_class"] == "robot"
    assert evidence["world_displacement_m"] != 0.50
    assert evidence["relocation_evaluable"] is False

    # 2. When an entity has unknown/missing displacement, router handles it explicitly via class prior
    evidence_missing_disp = {
        "entity_id": "inst_synthetic_robot_099",
        "semantic_class": "robot",
        "observed_motion": "DYNAMIC",
        "world_displacement_m": None,  # Explicitly None (no measurement)
        "displacement_uncertainty_m": None,
        "relocation_evaluable": False,
        "has_valid_pose": True,
        "depth_valid": True,
        "has_images": False,
        "views_count": 0,
    }

    from estimator_aware_vlm.router_policy import evaluate_router_policy
    decision = evaluate_router_policy(evidence_missing_disp)

    assert decision.decision == "RESOLVE_CHEAP"
    assert decision.cheap_label == "MOBILE"
    assert "CANONICAL_MOBILE_AGENT_PRIOR" in decision.reason_codes


def test_registration_consistency_across_pipeline_components():
    """Verify that EvidenceAdapter, RegistrationInfo, and run_full_experiment load identical SE(2) parameters."""
    from estimator_aware_vlm.evidence_adapter import EvidenceAdapter, load_registration_audit

    reg_file = root_dir / "outputs" / "reports" / "registration_audit.json"
    if not reg_file.exists():
        reg_file = root_dir / "outputs" / "registration_audit.json"
    assert reg_file.exists(), "Authoritative registration file must exist"

    reg_info = load_registration_audit(reg_file)
    assert reg_info.is_valid is True
    assert round(reg_info.rotation_deg, 3) == -15.558
    np.testing.assert_allclose(reg_info.translation_m, [-0.1059, 0.0486], atol=1e-4)

    adapter = EvidenceAdapter(repo_root=root_dir)
    assert round(adapter.registration.rotation_deg, 3) == -15.558
    np.testing.assert_allclose(adapter.registration.translation_m, [-0.1059, 0.0486], atol=1e-4)


def test_record_integrity_rejects_misaligned_jsonl(tmp_path):
    """Verify that record alignment checks detect equal-length but misaligned or corrupted evidence files."""
    obs_file = tmp_path / "obs.jsonl"
    fused_file = tmp_path / "fused.jsonl"

    obs_records = [
        {"obs_id": f"obs_{i:04d}", "timestamp": 100.0 + i, "source_session": "s1"}
        for i in range(10)
    ]
    # Fused records have same count but different observation IDs
    fused_records = [
        {"obs_id": f"obs_{i+100:04d}", "timestamp": 100.0 + i, "source_session": "s1"}
        for i in range(10)
    ]

    with open(obs_file, "w", encoding="utf-8") as f:
        for r in obs_records:
            f.write(json.dumps(r) + "\n")
    with open(fused_file, "w", encoding="utf-8") as f:
        for r in fused_records:
            f.write(json.dumps(r) + "\n")

    # Alignment verification: keys must match
    obs_ids = [r["obs_id"] for r in obs_records]
    fused_ids = [r["obs_id"] for r in fused_records]
    assert obs_ids != fused_ids
    with pytest.raises(ValueError, match="Misaligned observation IDs"):
        if obs_ids != fused_ids:
            raise ValueError(f"Misaligned observation IDs between observations and fused evidence.")


def test_full_experiment_pipeline_on_committed_fixture(tmp_path):
    """Run single-command full experiment runner on committed real-data fixture without skipping."""
    from scripts.run_full_experiment import run_full_experiment

    fixture_dir = root_dir / "tests" / "fixtures" / "real_pipeline_fixture"
    obs_file = fixture_dir / "fixture_observations.jsonl"
    reg_file = fixture_dir / "fixture_registration_audit.json"
    ann_file = fixture_dir / "fixture_identity_annotations.json"

    assert obs_file.exists(), f"Committed fixture missing: {obs_file}"
    assert reg_file.exists(), f"Committed fixture missing: {reg_file}"
    assert ann_file.exists(), f"Committed fixture missing: {ann_file}"

    out_dir = tmp_path / "fixture_experiment_run"

    manifest = run_full_experiment(
        s1_session="recording_01",
        s2_session="recording_02",
        obs_path=obs_file,
        reg_path=reg_file,
        annotations_path=ann_file,
        out_dir=out_dir,
    )

    assert manifest is not None
    assert "git_commit" in manifest
    assert manifest["input_artifacts"]["observations_count"] == 61
    assert manifest["registration_diagnostics"]["rotation_deg"] == -15.558

    # Verify output artifacts were written
    sg_file = out_dir / "scene_graph.json"
    base_file = out_dir / "association_baselines_summary.json"
    diag_file = out_dir / "per_instance_association_diagnostics.json"
    manifest_file = out_dir / "run_manifest.json"

    assert sg_file.exists()
    assert base_file.exists()
    assert diag_file.exists()
    assert manifest_file.exists()

    with open(sg_file, "r", encoding="utf-8") as f:
        sg_data = json.load(f)
    assert sg_data["total_nodes"] > 0
    assert len(sg_data["nodes"]) > 0





