"""Evaluation and Comparative Analysis for Counterfactual Interventions.

Computes mathematically rigorous intervention metrics:
    Delta_i(a) = L(C0) - L(C_i=a)
Where L is trajectory ATE on common timestamp support against ground truth.
If ground truth reference is unavailable (e.g. D455), trajectory metrics
remain strictly null/None, upholding zero-fabrication principles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from estimator_aware_vlm.datasets import (
    Trajectory,
    TrajectoryRole,
    parse_tum_ground_truth,
    parse_tum_trajectory,
)
from estimator_aware_vlm.evaluation.paired_metrics import (
    PairedEvaluationResult,
    evaluate_paired_trajectories_common_support,
)
from estimator_aware_vlm.evaluation.trajectory_metrics import (
    EvaluationError,
    evaluate_trajectory,
)
from estimator_aware_vlm.interventions.schema import (
    InterventionMetricRecord,
    InterventionRunResult,
    InterventionSpec,
    InterventionStatus,
)


def evaluate_intervention_trajectory(
    spec: InterventionSpec,
    intervention_traj_path: Optional[Path],
    baseline_traj_path: Optional[Path],
    groundtruth_path: Optional[Path],
    association_tolerance_sec: float = 0.02,
    execution_time_sec: float = 0.0,
    status: InterventionStatus = InterventionStatus.SUCCESS,
    notes: str = "",
) -> Tuple[InterventionMetricRecord, Optional[PairedEvaluationResult]]:
    """Evaluate an intervention trajectory against baseline and ground truth.

    Args:
        spec: Intervention specification.
        intervention_traj_path: Path to intervention CameraTrajectory.txt.
        baseline_traj_path: Path to baseline (C0) CameraTrajectory.txt.
        groundtruth_path: Path to reference groundtruth.txt (or None if unavailable).
        association_tolerance_sec: Timestamp alignment tolerance (default 20ms).
        execution_time_sec: Wall-clock run time.
        status: Execution status.
        notes: Contextual notes or explanations.

    Returns:
        Tuple of (InterventionMetricRecord, Optional[PairedEvaluationResult]).
    """
    record = InterventionMetricRecord(
        intervention_id=spec.intervention_id,
        dataset_id=spec.dataset_id,
        sequence_id=spec.sequence_id,
        track_id=spec.track_id,
        action=spec.action.value,
        repeat_index=spec.repeat_index,
        status=status.value,
        execution_time_sec=round(execution_time_sec, 2),
        notes=notes,
    )

    # If intervention failed or not implemented
    if status != InterventionStatus.SUCCESS:
        if not record.notes:
            record.notes = f"Intervention status is {status.value}."
        return record, None

    # Check trajectory existence
    if intervention_traj_path is None or not intervention_traj_path.exists():
        msg = "Intervention trajectory file not found."
        record.notes = f"{record.notes} {msg}".strip() if record.notes else msg
        if groundtruth_path is None or not groundtruth_path.exists():
            record.notes = f"{record.notes} Ground truth unavailable; trajectory ATE strictly null.".strip()
        return record, None

    # If ground truth does not exist (e.g. D455 uncalibrated), no ATE can be computed
    if groundtruth_path is None or not groundtruth_path.exists():
        msg = "Ground truth unavailable; trajectory ATE strictly null."
        record.notes = f"{record.notes} {msg}".strip() if record.notes else msg
        return record, None

    # Load trajectories
    try:
        traj_intervention = parse_tum_trajectory(
            intervention_traj_path,
            role=TrajectoryRole.ESTIMATED,
            name=f"intervention_{spec.intervention_id}",
        )
    except Exception as exc:
        record.notes = f"Failed to parse intervention trajectory: {exc}"
        record.status = InterventionStatus.FAILED.value
        return record, None

    try:
        traj_gt = parse_tum_ground_truth(groundtruth_path)
    except Exception as exc:
        record.notes = f"Failed to parse ground truth trajectory: {exc}"
        record.status = InterventionStatus.FAILED.value
        return record, None

    # Compute independent evaluation of intervention trajectory
    try:
        eval_indep = evaluate_trajectory(
            estimated=traj_intervention,
            ground_truth=traj_gt,
            max_dt_sec=association_tolerance_sec,
            sequence_name=spec.sequence_id,
        )
        record.ate_rmse_independent_m = round(float(eval_indep.ate.rmse_m), 4)
        record.rpe_trans_rmse_m = round(float(eval_indep.rpe.trans_rmse_m), 4)
        record.rpe_rot_rmse_deg = round(float(eval_indep.rpe.rot_rmse_deg), 4)
        record.tracking_coverage = round(float(eval_indep.tracking_coverage), 4)
    except Exception as exc:
        record.notes = f"Independent evaluation error: {exc}"

    # If baseline trajectory is provided, compute paired evaluation on common support
    paired_result: Optional[PairedEvaluationResult] = None
    if baseline_traj_path and baseline_traj_path.exists():
        try:
            traj_baseline = parse_tum_trajectory(
                baseline_traj_path,
                role=TrajectoryRole.ESTIMATED,
                name="baseline_c0",
            )
            paired_result = evaluate_paired_trajectories_common_support(
                traj_a=traj_baseline,
                traj_b=traj_intervention,
                ground_truth=traj_gt,
                max_dt_sec=association_tolerance_sec,
                delta_sec=1.0,
                sequence_name=spec.sequence_id,
            )

            record.ate_baseline_rmse_common_m = round(float(paired_result.ate_a_common.rmse_m), 4)
            record.ate_rmse_common_m = round(float(paired_result.ate_b_common.rmse_m), 4)
            # Delta_i(a) = L(C0) - L(C_i=a) -> positive improvement means error reduction!
            record.delta_ate_rmse_m = round(
                float(paired_result.ate_a_common.rmse_m - paired_result.ate_b_common.rmse_m), 4
            )
            record.percentage_ate_improvement = round(
                float(paired_result.percentage_ate_improvement_common), 2
            )
            record.common_poses_count = int(paired_result.num_common_poses)
            record.jaccard_support_index = round(float(paired_result.jaccard_support_index), 4)
            record.paired_ttest_pvalue = float(paired_result.ttest_pvalue)
            record.paired_wilcoxon_pvalue = (
                float(paired_result.wilcoxon_pvalue) if paired_result.wilcoxon_pvalue is not None else None
            )
            record.is_statistically_significant = bool(paired_result.is_statistically_significant)

        except Exception as exc:
            record.notes = (record.notes + f" Paired common-support evaluation error: {exc}").strip()

    return record, paired_result
