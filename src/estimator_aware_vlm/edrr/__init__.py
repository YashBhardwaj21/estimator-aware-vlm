"""Expected Downstream Regret Reduction (EDRR) and Grouped Validation Package."""

from estimator_aware_vlm.edrr.loss import (
    DownstreamSLAMLoss,
    SLAMLossWeights,
    compute_normalized_slam_loss,
)
from estimator_aware_vlm.edrr.formulation import (
    CandidateQueryProfile,
    EDRRDecision,
    compute_candidate_edrr,
    select_top_k_edrr,
)
from estimator_aware_vlm.edrr.dataset import (
    InterventionDataset,
    SingleCandidateInterventionRecord,
)
from estimator_aware_vlm.edrr.grouped_validation import (
    GroupedValidationResult,
    check_data_sufficiency,
    run_sequence_level_grouped_validation,
    MIN_SAMPLE_COUNT_THRESHOLD,
    MIN_SEQUENCE_COUNT_THRESHOLD,
)

__all__ = [
    "DownstreamSLAMLoss",
    "SLAMLossWeights",
    "compute_normalized_slam_loss",
    "CandidateQueryProfile",
    "EDRRDecision",
    "compute_candidate_edrr",
    "select_top_k_edrr",
    "InterventionDataset",
    "SingleCandidateInterventionRecord",
    "GroupedValidationResult",
    "check_data_sufficiency",
    "run_sequence_level_grouped_validation",
    "MIN_SAMPLE_COUNT_THRESHOLD",
    "MIN_SEQUENCE_COUNT_THRESHOLD",
]

