"""Planner for Deterministic Counterfactual SLAM Interventions.

Generates reproducible InterventionSpec plans for:
- C0 (KEEP baseline)
- C1 (IMAGE_POLYGON_MASK)
- C2 (IMAGE_BBOX_MASK)
- C3 (GRAPH_MARK_SUSPECT)
- C4 (GRAPH_TEMPORARY_EXCLUDE)
- C5 (LANDMARK_DOWNWEIGHT)
- C6 (LANDMARK_TEMPORARY_EXCLUDE)
- C7 (DUAL_INTEGRATION)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.interventions.schema import (
    InterventionAction,
    InterventionSpec,
    InterventionTargetType,
)


def create_intervention_plan(
    experiment_run_id: str,
    dataset_id: str,
    sequence_id: str,
    candidate_tracks: Sequence[str],
    actions: Sequence[InterventionAction],
    repeats: int = 1,
    config_path: Optional[str] = None,
    groundtruth_path: Optional[str] = None,
    baseline_trajectory_path: Optional[str] = None,
    observations_path: Optional[str] = None,
    action_parameters: Optional[Dict[InterventionAction, Dict[str, Any]]] = None,
) -> List[InterventionSpec]:
    """Deterministically generate an ordered list of InterventionSpec objects.

    Args:
        experiment_run_id: ID of parent run (e.g., 'run_20261031_counterfactual').
        dataset_id: Dataset identifier ('tum', 'openloris', 'd455').
        sequence_id: Sequence name ('fr3_walking_xyz', 'corridor1-4', etc.).
        candidate_tracks: List of track IDs to intervene upon.
        actions: Sequence of InterventionAction arms to generate.
        repeats: Number of independent repetitions per arm (default 1).
        config_path: Path to dataset YAML configuration.
        groundtruth_path: Path to reference ground truth trajectory.
        baseline_trajectory_path: Path to C0 baseline trajectory.
        observations_path: Path to per-frame observations JSONL.
        action_parameters: Optional dictionary mapping action to custom parameters.

    Returns:
        List of planned InterventionSpec instances.
    """
    plan: List[InterventionSpec] = []
    action_params = action_parameters or {}

    for rep in range(1, repeats + 1):
        # 1. C0 KEEP baseline is track-independent
        if InterventionAction.KEEP in actions:
            spec_id = InterventionSpec.generate_id(
                dataset_id=dataset_id,
                sequence_id=sequence_id,
                track_id=None,
                action=InterventionAction.KEEP,
                repeat_index=rep,
            )
            plan.append(
                InterventionSpec(
                    experiment_run_id=experiment_run_id,
                    intervention_id=spec_id,
                    dataset_id=dataset_id,
                    sequence_id=sequence_id,
                    track_id=None,
                    action=InterventionAction.KEEP,
                    repeat_index=rep,
                    target_type=InterventionTargetType.NONE,
                    parameters=action_params.get(InterventionAction.KEEP, {}),
                    config_path=config_path,
                    groundtruth_path=groundtruth_path,
                    baseline_trajectory_path=baseline_trajectory_path,
                    observations_path=observations_path,
                )
            )

        # 2. Targeted object interventions
        for track_id in candidate_tracks:
            for action in actions:
                if action == InterventionAction.KEEP:
                    continue  # already handled above
                
                spec_id = InterventionSpec.generate_id(
                    dataset_id=dataset_id,
                    sequence_id=sequence_id,
                    track_id=track_id,
                    action=action,
                    repeat_index=rep,
                )
                
                params = dict(action_params.get(action, {}))
                if action == InterventionAction.IMAGE_POLYGON_MASK:
                    params.setdefault("mask_type", "polygon")
                elif action == InterventionAction.IMAGE_BBOX_MASK:
                    params.setdefault("mask_type", "bbox")

                plan.append(
                    InterventionSpec(
                        experiment_run_id=experiment_run_id,
                        intervention_id=spec_id,
                        dataset_id=dataset_id,
                        sequence_id=sequence_id,
                        track_id=track_id,
                        action=action,
                        repeat_index=rep,
                        target_type=action.target_type,
                        parameters=params,
                        config_path=config_path,
                        groundtruth_path=groundtruth_path,
                        baseline_trajectory_path=baseline_trajectory_path,
                        observations_path=observations_path,
                    )
                )

    return plan

