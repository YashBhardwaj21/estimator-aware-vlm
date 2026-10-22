from typing import Optional

from scene_graph.relations.evidence import (
    RelationEvidence,
    EvidenceResult,
)


from scene_graph.ontology.relation import (
    INVERSE_PREDICATES,
    SYMMETRIC_PREDICATES,
    LEGACY_PREDICATE_MAP,
    normalize_predicate,
    get_inverse_predicate,
)

INVERSE = {
    **INVERSE_PREDICATES,
    "ON": "UNDER",
    "UNDER": "ON",
    "CONTAINING": "INSIDE",
    "IN_FRONT_OF": "BEHIND",
    "OCCLUDING": "OCCLUDED_BY",
}

SYMMETRIC = set(SYMMETRIC_PREDICATES) | {"FAR"}

COMPLEMENTARY = {
    "LEFT_OF",
    "RIGHT_OF",
    "ABOVE",
    "BELOW",
    "FRONT_OF",
    "BEHIND",
    "IN_FRONT_OF",
}



def derive_inverse_evidence(
    evidence: RelationEvidence,
) -> Optional[RelationEvidence]:

    if evidence.predicate in INVERSE:
        inverse_predicate = INVERSE[evidence.predicate]
    elif evidence.predicate in SYMMETRIC:
        inverse_predicate = evidence.predicate
    else:
        raise ValueError(
            f"Unknown inverse for predicate: {evidence.predicate}"
        )

    if evidence.result == EvidenceResult.SUPPORTED:
        inverse_result = EvidenceResult.SUPPORTED
        inverse_value = evidence.value

    elif (
        evidence.result == EvidenceResult.CONTRADICTED
        and evidence.predicate in COMPLEMENTARY
    ):
        inverse_result = EvidenceResult.SUPPORTED
        inverse_value = (
            abs(evidence.value)
            if evidence.value is not None
            else None
        )

    else:
        return None

    details = evidence.details.copy()
    details["derived_from_predicate"] = evidence.predicate
    details["derived_from_result"] = evidence.result.value

    return RelationEvidence(
        predicate=inverse_predicate,
        subject_id=evidence.object_id,
        object_id=evidence.subject_id,
        frame_index=evidence.frame_index,
        timestamp=evidence.timestamp,
        result=inverse_result,
        value=inverse_value,
        threshold=evidence.threshold,
        confidence=evidence.confidence,
        reference_frame=evidence.reference_frame,
        evidence_type=f"inverse_{evidence.evidence_type}",
        details=details,
    )


def canonical_pair(
    id_a: str,
    id_b: str,
) -> tuple[str, str]:
    return (
        (id_a, id_b)
        if id_a < id_b
        else (id_b, id_a)
    )