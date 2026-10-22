from typing import List, Tuple, Optional, Dict, Any

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence
from scene_graph.relations.admissibility import AdmissibilityFilter
from scene_graph.relations.inverse_algebra import INVERSE, SYMMETRIC
from scene_graph.geometry.provenance import GeometrySource
from scene_graph.relations.hierarchy import SpatialContext, SpatialContextManager
from scene_graph.relations.candidate_generator import RelationCandidateGenerator, CandidateSummary


ALLOWED_PREDICATES = frozenset(INVERSE) | frozenset(SYMMETRIC)


class RelationRegistry:

    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.admissibility_filter = AdmissibilityFilter(config)
        self.context_manager = SpatialContextManager(
            config.relations.hierarchy if config and config.relations else None
        )
        self.candidate_generator = RelationCandidateGenerator(config)
        self._modules: List[RelationModule] = []
        self.last_contexts: Dict[str, SpatialContext] = {}
        self.last_membership: Dict[str, str] = {}
        self.last_candidate_summary: Optional[CandidateSummary] = None

    def register(self, module: RelationModule) -> None:
        self._modules.append(module)

    def _adjust_evidence_for_geometry_provenance(
        self,
        evidence: RelationEvidence,
        context: FrameContext,
    ) -> RelationEvidence:
        subject_geometry = context.observation_geometry.get(evidence.subject_id)
        object_geometry = context.observation_geometry.get(evidence.object_id)

        predicted_count = sum(
            geometry is not None
            and geometry.geometry_source != GeometrySource.OBSERVED
            for geometry in (
                subject_geometry,
                object_geometry,
            )
        )

        if predicted_count == 0:
            return evidence

        scale = self.config.relations.predicted_geometry_confidence_scale

        confidence_scale = scale ** predicted_count
        adjusted_confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    evidence.confidence * confidence_scale,
                ),
            )
        )

        details = dict(evidence.details)
        details["geometry_provenance"] = {
            "subject": (
                subject_geometry.geometry_source.value
                if subject_geometry is not None
                else None
            ),
            "object": (
                object_geometry.geometry_source.value
                if object_geometry is not None
                else None
            ),
            "predicted_endpoint_count": predicted_count,
            "confidence_scale": confidence_scale,
            "original_confidence": float(evidence.confidence),
            "adjusted_confidence": adjusted_confidence,
        }

        return RelationEvidence(
            predicate=evidence.predicate,
            subject_id=evidence.subject_id,
            object_id=evidence.object_id,
            frame_index=evidence.frame_index,
            timestamp=evidence.timestamp,
            result=evidence.result,
            value=evidence.value,
            threshold=evidence.threshold,
            confidence=adjusted_confidence,
            reference_frame=evidence.reference_frame,
            evidence_type=evidence.evidence_type,
            details=details,
        )

    @staticmethod
    def _generate_candidate_pairs(
        tracks: List[Track],
    ) -> List[Tuple[Track, Track]]:
        relation_tracks = [
            track
            for track in tracks
            if track.state in (
                TrackState.ACTIVE,
                TrackState.TEMPORARILY_UNOBSERVED,
            )
        ]

        relation_tracks.sort(key=lambda track: track.object_id)

        return [
            (subject, object_)
            for subject in relation_tracks
            for object_ in relation_tracks
            if subject.object_id != object_.object_id
        ]

    def _filter_admissible(
        self,
        pairs: List[Tuple[Track, Track]],
        predicate: str,
        context: Optional[FrameContext] = None,
    ) -> List[Tuple[Track, Track]]:
        return [
            (subject, object_)
            for subject, object_ in pairs
            if self.admissibility_filter.is_admissible(
                predicate,
                subject,
                object_,
                context=context,
            )
        ]

    def compute_all(
        self,
        tracks: List[Track],
        context: FrameContext,
    ) -> List[RelationEvidence]:
        # 1. Discover spatial contexts and assign object memberships
        contexts = self.context_manager.discover_contexts(tracks, context)
        membership = self.context_manager.assign_context_membership(tracks, contexts, context)
        self.last_contexts = contexts
        self.last_membership = membership

        # 2. Generate typed candidates before inference
        typed_candidates, summary = self.candidate_generator.generate_all_candidates(
            tracks, context, contexts, membership
        )
        self.last_candidate_summary = summary

        all_evidences: List[RelationEvidence] = []

        # 3. Route targeted candidate subsets to specialized modules
        for module in self._modules:
            predicates = tuple(module.predicates())

            invalid_predicates = [
                predicate
                for predicate in predicates
                if predicate not in ALLOWED_PREDICATES
            ]

            if invalid_predicates:
                raise ValueError(
                    f"Module {module.__class__.__name__} provided "
                    f"invalid predicates: {invalid_predicates}"
                )

            # Map predicates to their typed candidate pool
            cand_pairs: List[Tuple[Track, Track]] = []
            for pred in predicates:
                if pred in ("SUPPORTED_BY", "SUPPORTS", "ON", "UNDER"):
                    cand_pairs.extend(typed_candidates.get("structural", []))
                elif pred in ("INSIDE", "CONTAINS", "CONTAINING", "ATTACHED_TO"):
                    cand_pairs.extend(typed_candidates.get("containment", []))
                elif pred in ("NEAR", "TOUCHING", "FAR"):
                    cand_pairs.extend(typed_candidates.get("proximity", []))
                elif pred in ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "FRONT_OF", "IN_FRONT_OF", "BEHIND"):
                    cand_pairs.extend(typed_candidates.get("directional", []))
                elif pred in ("OCCLUDES", "OCCLUDED_BY", "OCCLUDING"):
                    cand_pairs.extend(typed_candidates.get("visibility", []))
                else:
                    cand_pairs.extend(self._generate_candidate_pairs(tracks))

            # Deduplicate candidate pairs for this module
            seen_pairs = set()
            unique_cand_pairs = []
            for p in cand_pairs:
                pkey = (p[0].object_id, p[1].object_id)
                if pkey not in seen_pairs:
                    seen_pairs.add(pkey)
                    unique_cand_pairs.append(p)

            admissible_pairs = set()

            for predicate in predicates:
                admissible_pairs.update(
                    self._filter_admissible(
                        unique_cand_pairs,
                        predicate,
                        context=context,
                    )
                )

            sorted_pairs = sorted(
                admissible_pairs,
                key=lambda pair: (
                    pair[0].object_id,
                    pair[1].object_id,
                ),
            )

            evidences = module.compute_pairs(
                sorted_pairs,
                context,
            )

            for evidence in evidences:
                if evidence.predicate not in ALLOWED_PREDICATES:
                    raise ValueError(
                        f"Module {module.__class__.__name__} emitted "
                        f"invalid predicate '{evidence.predicate}'."
                    )

                adjusted_evidence = self._adjust_evidence_for_geometry_provenance(
                    evidence,
                    context,
                )

                all_evidences.append(adjusted_evidence)

        return all_evidences