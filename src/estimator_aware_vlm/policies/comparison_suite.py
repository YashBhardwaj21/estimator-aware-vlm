"""Fair Policy Comparison Suite and Dual-Cost Accounting (Stage 6).

Implements rigorous, controlled comparisons across routing policies B0 through B6:
- B0: Blind RETAIN (masks 0 tracks, queries 0 VLM calls)
- B1: Blind INVALIDATE (masks 100% of candidate tracks, queries 0 VLM calls)
- B2: Geometry Only (masks tracks where camera-compensated displacement >= 0.30m, queries 0 VLM calls)
- B3: VLM All (queries live VLM on 100% of candidate tracks with visual context)
- B4: Random Routing (budget-matched to B6: exactly K_B4 = K_B6 queries sampled uniformly at random)
- B5: Confidence-Threshold Routing (queries high spatial dispersion or borderline displacement)
- B6: Estimator-Aware Selective Routing (Proposed: queries when calibrated utility U(q) >= 0)

Key Scientific Rigor Guarantees:
1. Equal Query Budgets: B4 is strictly budget-matched to B6 (K_B4 == K_B6).
2. Dual Cost Accounting: Distinguishes observed execution latency from hypothetical wire latency.
3. Decoupled Architecture: Separates the query decision from the downstream SLAM intervention action.
4. Calibrated Utility Integration: Ingests empirical LOOCV CalibratedUtilityModel for delta_L_slam.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
import urllib.request

import cv2
import numpy as np

from estimator_aware_vlm.calibration.counterfactual import (
    CalibratedUtilityModel,
    extract_track_features,
)
from estimator_aware_vlm.router_policy import (
    CANONICAL_IMMOVABLE,
    CANONICAL_MOVABLE,
    CANONICAL_MOBILE,
    AMBIGUOUS_AFFORDANCE_CLASSES,
)


@dataclass
class VLMQueryResult:
    """Wire response metadata from VLM query."""
    semantic_class: str
    contextual_movability: str  # IMMOVABLE | MOVABLE | MOBILE | UNKNOWN
    reasoning: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    is_valid_schema: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLMProvider:
    """Abstract provider for VLM inference queries."""
    def query(self, track_id: str, crop_path: Optional[str]) -> Tuple[VLMQueryResult, float]:
        """Return (VLMQueryResult, observed_latency_ms)."""
        raise NotImplementedError


class CachedVLMProvider(VLMProvider):
    """Replay provider that serves recorded VLM queries with zero network cost."""
    def __init__(
        self,
        cached_results: Dict[str, Dict[str, Any]],
        default_wire_latency_ms: float = 950.0,
        default_prompt_tokens: int = 210,
        default_completion_tokens: int = 50,
    ) -> None:
        self.cached: Dict[str, VLMQueryResult] = {}
        self.default_wire_latency_ms = default_wire_latency_ms
        self.default_prompt_tokens = default_prompt_tokens
        self.default_completion_tokens = default_completion_tokens

        for tid, data in cached_results.items():
            if isinstance(data, VLMQueryResult):
                self.cached[tid] = data
            elif isinstance(data, dict):
                self.cached[tid] = VLMQueryResult(
                    semantic_class=str(data.get("semantic_class", "unknown")),
                    contextual_movability=str(data.get("contextual_movability", "UNKNOWN")).upper(),
                    reasoning=str(data.get("reasoning", "")),
                    latency_ms=float(data.get("latency_ms", default_wire_latency_ms)),
                    prompt_tokens=int(data.get("prompt_tokens", default_prompt_tokens)),
                    completion_tokens=int(data.get("completion_tokens", default_completion_tokens)),
                    is_valid_schema=bool(data.get("is_valid_schema", True)),
                )

    def query(self, track_id: str, crop_path: Optional[str]) -> Tuple[VLMQueryResult, float]:
        t0 = time.perf_counter()
        if track_id in self.cached:
            res = self.cached[track_id]
        else:
            # Deterministic synthetic fallback for un-cached track
            res = VLMQueryResult(
                semantic_class="unknown",
                contextual_movability="UNKNOWN",
                reasoning="Not present in cache replay",
                latency_ms=self.default_wire_latency_ms,
                prompt_tokens=self.default_prompt_tokens,
                completion_tokens=self.default_completion_tokens,
                is_valid_schema=False,
            )
        observed_ms = (time.perf_counter() - t0) * 1000.0
        return res, observed_ms


class LiveVLMProvider(VLMProvider):
    """Live provider executing strict network calls against local/remote VLM endpoint."""
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8089/v1/chat/completions",
        timeout: float = 75.0,
        cache_fallback: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self.cache_fallback = CachedVLMProvider(cache_fallback or {}) if cache_fallback else None
        self.session_cache: Dict[str, VLMQueryResult] = {}

    def query(self, track_id: str, crop_path: Optional[str]) -> Tuple[VLMQueryResult, float]:
        if track_id in self.session_cache:
            return self.session_cache[track_id], 0.05

        if crop_path is None or not Path(crop_path).exists():
            if self.cache_fallback:
                return self.cache_fallback.query(track_id, crop_path)
            return VLMQueryResult(
                semantic_class="unknown",
                contextual_movability="UNKNOWN",
                reasoning="Missing crop image",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                is_valid_schema=False,
            ), 0.01

        # Attempt live query
        try:
            p = Path(crop_path)
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Failed to read image at {p}")
            h, w = img.shape[:2]
            scale = 256.0 / max(h, w)
            if scale < 1.0:
                resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = img
            _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt_text = (
                "You are an expert robot perception assistant analyzing an indoor office/lab environment.\n"
                "Examine this cropped image of an object.\n"
                "Identify which relevant office object it is, such as:\n"
                "- monitor / computer screen (IMMOVABLE: mounted on desk or stand)\n"
                "- desk / table (IMMOVABLE: stationary heavy furniture)\n"
                "- chair / office chair (MOVABLE: transportable furniture on wheels or legs)\n"
                "- keyboard / mouse / laptop (MOVABLE: small tabletop accessories)\n"
                "- person / human (MOBILE: walking or moving human agent)\n\n"
                "Determine the exact semantic class and its contextual movability (IMMOVABLE, MOVABLE, or MOBILE).\n"
                "Return valid JSON strictly matching this schema:\n"
                '{"semantic_class": "<name>", "contextual_movability": "<IMMOVABLE|MOVABLE|MOBILE>", "reasoning": "<brief explanation>"}'
            )

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 120,
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.server_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
            live_lat_ms = (time.perf_counter() - t0) * 1000.0

            raw_text = res_data["choices"][0]["message"]["content"]
            usage = res_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            try:
                parsed = json.loads(clean_text)
                sem_cls = str(parsed.get("semantic_class", "unknown")).lower()
                mov = str(parsed.get("contextual_movability", "UNKNOWN")).upper()
                reason = str(parsed.get("reasoning", ""))
                valid = mov in ["IMMOVABLE", "MOVABLE", "MOBILE"]
            except Exception:
                sem_cls = "unknown"
                mov = "UNKNOWN"
                reason = raw_text[:100]
                valid = False

            q_res = VLMQueryResult(
                semantic_class=sem_cls,
                contextual_movability=mov,
                reasoning=reason,
                latency_ms=round(live_lat_ms, 1),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_valid_schema=valid,
            )
            self.session_cache[track_id] = q_res
            return q_res, live_lat_ms

        except Exception as exc:
            if self.cache_fallback:
                return self.cache_fallback.query(track_id, crop_path)
            raise exc


@dataclass
class TrackDecisionRecord:
    """Individual candidate track routing and action decision."""
    track_id: str
    semantic_class: str
    vlm_queried: bool
    resolved_affordance: str  # IMMOVABLE | MOVABLE | MOBILE | UNKNOWN
    action: str  # RETAIN | MASK_FROM_TRACKING
    reason_codes: List[str] = field(default_factory=list)
    query_utility: Optional[float] = None
    expected_delta_L_slam: Optional[float] = None
    observed_latency_ms: float = 0.0
    hypothetical_wire_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "semantic_class": self.semantic_class,
            "vlm_queried": self.vlm_queried,
            "resolved_affordance": self.resolved_affordance,
            "action": self.action,
            "reason_codes": self.reason_codes,
            "query_utility": round(self.query_utility, 4) if self.query_utility is not None else None,
            "expected_delta_L_slam": round(self.expected_delta_L_slam, 4) if self.expected_delta_L_slam is not None else None,
            "observed_latency_ms": round(self.observed_latency_ms, 3),
            "hypothetical_wire_latency_ms": round(self.hypothetical_wire_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class PolicyExecutionSummary:
    """Summary of routing execution, fairness properties, and dual-cost accounting."""
    policy_key: str
    policy_name: str
    description: str
    total_candidates: int
    query_count: int
    query_rate: float
    budget_matched_to: Optional[str]
    masked_count: int
    retained_count: int
    total_observed_latency_ms: float
    total_hypothetical_wire_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    normalized_compute_cost: float
    decisions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_key": self.policy_key,
            "policy_name": self.policy_name,
            "description": self.description,
            "total_candidates": self.total_candidates,
            "query_count": self.query_count,
            "query_rate": round(self.query_rate, 4),
            "budget_matched_to": self.budget_matched_to,
            "masked_count": self.masked_count,
            "retained_count": self.retained_count,
            "total_observed_latency_ms": round(self.total_observed_latency_ms, 2),
            "total_hypothetical_wire_latency_ms": round(self.total_hypothetical_wire_latency_ms, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "normalized_compute_cost": round(self.normalized_compute_cost, 4),
            "decisions": self.decisions,
        }


def evaluate_b0_blind_retain(tracks: List[Dict[str, Any]]) -> PolicyExecutionSummary:
    """B0: Blind RETAIN (no tracks masked, zero VLM queries)."""
    decisions: List[Dict[str, Any]] = []
    for t in tracks:
        rec = TrackDecisionRecord(
            track_id=t["track_id"],
            semantic_class=t.get("semantic_class", "unknown"),
            vlm_queried=False,
            resolved_affordance="IMMOVABLE",
            action="RETAIN",
            reason_codes=["B0_BLIND_RETAIN_ALL"],
            observed_latency_ms=0.01,
        )
        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B0_Blind_RETAIN",
        policy_name="B0 (Blind RETAIN)",
        description="Retains 100% of candidate landmarks without queries or intervention",
        total_candidates=n,
        query_count=0,
        query_rate=0.0,
        budget_matched_to=None,
        masked_count=0,
        retained_count=n,
        total_observed_latency_ms=0.01 * n,
        total_hypothetical_wire_latency_ms=0.0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        normalized_compute_cost=0.0,
        decisions=decisions,
    )


def evaluate_b1_blind_invalidate(tracks: List[Dict[str, Any]]) -> PolicyExecutionSummary:
    """B1: Blind INVALIDATE (masks 100% of candidate tracks, zero VLM queries)."""
    decisions: List[Dict[str, Any]] = []
    for t in tracks:
        rec = TrackDecisionRecord(
            track_id=t["track_id"],
            semantic_class=t.get("semantic_class", "unknown"),
            vlm_queried=False,
            resolved_affordance="MOBILE",
            action="MASK_FROM_TRACKING",
            reason_codes=["B1_BLIND_INVALIDATE_ALL"],
            observed_latency_ms=0.01,
        )
        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B1_Blind_INVALIDATE",
        policy_name="B1 (Blind INVALIDATE)",
        description="Masks 100% of candidate landmarks without queries",
        total_candidates=n,
        query_count=0,
        query_rate=0.0,
        budget_matched_to=None,
        masked_count=n,
        retained_count=0,
        total_observed_latency_ms=0.01 * n,
        total_hypothetical_wire_latency_ms=0.0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        normalized_compute_cost=0.0,
        decisions=decisions,
    )


def evaluate_b2_geometry_only(
    tracks: List[Dict[str, Any]],
    displacement_threshold_m: float = 0.30,
) -> PolicyExecutionSummary:
    """B2: Geometry Only (masks tracks where camera-compensated displacement >= threshold)."""
    decisions: List[Dict[str, Any]] = []
    masked = 0
    retained = 0

    for t in tracks:
        w_disp = float(t.get("world_displacement_m", t.get("measured_displacement_m", 0.0)) or 0.0)
        mot = str(t.get("observed_motion", "STATIC")).upper()

        if mot == "MOVING" or w_disp >= displacement_threshold_m:
            act = "MASK_FROM_TRACKING"
            aff = "MOVABLE" if mot != "MOVING" else "MOBILE"
            reasons = ["B2_GEOMETRY_DISPLACEMENT_EXCEEDED"]
            masked += 1
        else:
            act = "RETAIN"
            aff = "IMMOVABLE"
            reasons = ["B2_GEOMETRY_SUBTHRESHOLD_STATIC"]
            retained += 1

        rec = TrackDecisionRecord(
            track_id=t["track_id"],
            semantic_class=t.get("semantic_class", "unknown"),
            vlm_queried=False,
            resolved_affordance=aff,
            action=act,
            reason_codes=reasons,
            observed_latency_ms=0.02,
        )
        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B2_Geometry_Only",
        policy_name="B2 (Geometry Only)",
        description=f"Masks tracks with world displacement >= {displacement_threshold_m}m without VLM queries",
        total_candidates=n,
        query_count=0,
        query_rate=0.0,
        budget_matched_to=None,
        masked_count=masked,
        retained_count=retained,
        total_observed_latency_ms=0.02 * n,
        total_hypothetical_wire_latency_ms=0.0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        normalized_compute_cost=0.0,
        decisions=decisions,
    )


def evaluate_b3_vlm_all(
    tracks: List[Dict[str, Any]],
    vlm_provider: VLMProvider,
) -> PolicyExecutionSummary:
    """B3: VLM All (queries 100% of candidate tracks with visual context)."""
    decisions: List[Dict[str, Any]] = []
    query_cnt = 0
    tot_obs_lat = 0.0
    tot_wire_lat = 0.0
    tot_p_tokens = 0
    tot_c_tokens = 0
    masked = 0
    retained = 0

    for t in tracks:
        tid = t["track_id"]
        sem_cls = t.get("semantic_class", "unknown")
        mot = str(t.get("observed_motion", "STATIC")).upper()
        w_disp = float(t.get("world_displacement_m", t.get("measured_displacement_m", 0.0)) or 0.0)
        crop_path = t["crop_paths"][0] if t.get("crop_paths") else None

        if crop_path:
            v_res, obs_lat = vlm_provider.query(tid, crop_path)
            query_cnt += 1
            tot_obs_lat += obs_lat
            tot_wire_lat += v_res.latency_ms
            tot_p_tokens += v_res.prompt_tokens
            tot_c_tokens += v_res.completion_tokens

            resolved_aff = v_res.contextual_movability
            if resolved_aff == "MOBILE" or (resolved_aff == "MOVABLE" and (mot == "MOVING" or w_disp >= 0.15)):
                act = "MASK_FROM_TRACKING"
                reasons = ["B3_VLM_ALL_DYNAMIC_RESOLVED", f"AFFORDANCE_{resolved_aff}"]
                masked += 1
            else:
                act = "RETAIN"
                reasons = ["B3_VLM_ALL_STATIC_RETAINED", f"AFFORDANCE_{resolved_aff}"]
                retained += 1

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=True,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                observed_latency_ms=obs_lat,
                hypothetical_wire_latency_ms=v_res.latency_ms,
                prompt_tokens=v_res.prompt_tokens,
                completion_tokens=v_res.completion_tokens,
            )
        else:
            act = "RETAIN"
            retained += 1
            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=False,
                resolved_affordance="UNKNOWN",
                action=act,
                reason_codes=["B3_MISSING_VISUAL_CONTEXT_RETAIN"],
                observed_latency_ms=0.01,
            )
            tot_obs_lat += 0.01

        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B3_VLM_All",
        policy_name="B3 (VLM All)",
        description="Queries VLM on 100% of candidate tracks with visual context",
        total_candidates=n,
        query_count=query_cnt,
        query_rate=query_cnt / max(1, n),
        budget_matched_to=None,
        masked_count=masked,
        retained_count=retained,
        total_observed_latency_ms=tot_obs_lat,
        total_hypothetical_wire_latency_ms=tot_wire_lat,
        total_prompt_tokens=tot_p_tokens,
        total_completion_tokens=tot_c_tokens,
        normalized_compute_cost=query_cnt * 0.50,
        decisions=decisions,
    )


def evaluate_b6_estimator_aware(
    tracks: List[Dict[str, Any]],
    vlm_provider: VLMProvider,
    calibrated_model: Optional[CalibratedUtilityModel] = None,
    lambda_cost: float = 0.45,
    utility_threshold: float = 0.0,
) -> PolicyExecutionSummary:
    """B6: Proposed Estimator-Aware Selective Routing (Calibrated Utility)."""
    decisions: List[Dict[str, Any]] = []
    query_cnt = 0
    tot_obs_lat = 0.0
    tot_wire_lat = 0.0
    tot_p_tokens = 0
    tot_c_tokens = 0
    masked = 0
    retained = 0

    for t in tracks:
        tid = t["track_id"]
        sem_cls = t.get("semantic_class", "unknown")
        mot = str(t.get("observed_motion", "STATIC")).upper()
        w_disp = float(t.get("world_displacement_m", t.get("measured_displacement_m", 0.0)) or 0.0)
        obs_cnt = float(t.get("observation_count", t.get("total_observations", 0)) or 0)
        crop_path = t["crop_paths"][0] if t.get("crop_paths") else None

        # 1. Compute delta_L_slam and query utility
        if calibrated_model is not None:
            feat = extract_track_features(t)
            delta_L_slam = calibrated_model.predict_delta_L(feat)
            cost_term = lambda_cost * calibrated_model.c_vlm
            utility = delta_L_slam - cost_term
            prov_tag = "CALIBRATED_EMPIRICAL"
        else:
            obs_impact = min(1.0, obs_cnt / 100.0)
            disp_impact = min(1.0, w_disp / 0.50)
            ambiguity = 1.0 if (0.15 <= w_disp <= 0.80) else 0.20
            delta_L_slam = obs_impact * 0.40 + disp_impact * 0.40 + ambiguity * 0.20
            utility = delta_L_slam - lambda_cost * 0.50
            prov_tag = "PROXY_SUPPORT"

        # 2. Decoupled Query Decision
        should_query = (utility >= utility_threshold) and (crop_path is not None)

        # 3. Affordance Resolution & Action Mapping
        if should_query:
            v_res, obs_lat = vlm_provider.query(tid, crop_path)
            query_cnt += 1
            tot_obs_lat += obs_lat
            tot_wire_lat += v_res.latency_ms
            tot_p_tokens += v_res.prompt_tokens
            tot_c_tokens += v_res.completion_tokens

            resolved_aff = v_res.contextual_movability
            if resolved_aff == "MOBILE":
                act = "MASK_FROM_TRACKING"
                reasons = ["B6_VLM_RESOLVED_MOBILE_AGENT", f"PROVENANCE_{prov_tag}"]
                masked += 1
            elif resolved_aff == "IMMOVABLE":
                act = "RETAIN"
                reasons = ["B6_VLM_RESOLVED_STATIC_LANDMARK", f"PROVENANCE_{prov_tag}"]
                retained += 1
            elif mot == "MOVING" and w_disp >= 0.60:
                act = "MASK_FROM_TRACKING"
                reasons = ["B6_VLM_RESOLVED_MOVABLE_SIGNIFICANT_DISPLACEMENT", f"PROVENANCE_{prov_tag}"]
                masked += 1
            else:
                act = "RETAIN"
                reasons = ["B6_VLM_RESOLVED_MOVABLE_STATIONARY", f"PROVENANCE_{prov_tag}"]
                retained += 1

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=True,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                query_utility=utility,
                expected_delta_L_slam=delta_L_slam,
                observed_latency_ms=obs_lat,
                hypothetical_wire_latency_ms=v_res.latency_ms,
                prompt_tokens=v_res.prompt_tokens,
                completion_tokens=v_res.completion_tokens,
            )
        else:
            tot_obs_lat += 0.02
            # Cheap local fallback without VLM
            if mot == "MOVING" and w_disp >= 1.0:
                act = "MASK_FROM_TRACKING"
                resolved_aff = "MOVABLE"
                reasons = ["B6_SUBTHRESHOLD_HEAVY_DISPLACEMENT_MASK", f"PROVENANCE_{prov_tag}"]
                masked += 1
            else:
                act = "RETAIN"
                resolved_aff = "IMMOVABLE" if sem_cls in CANONICAL_IMMOVABLE else "MOVABLE"
                reasons = ["B6_SUBTHRESHOLD_UTILITY_RETAINED", f"PROVENANCE_{prov_tag}"]
                retained += 1

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=False,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                query_utility=utility,
                expected_delta_L_slam=delta_L_slam,
                observed_latency_ms=0.02,
            )

        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B6_Estimator_Aware",
        policy_name="B6 (Estimator-Aware Selective)",
        description="Selectively queries VLM when expected SLAM loss reduction exceeds compute cost U(q) >= 0",
        total_candidates=n,
        query_count=query_cnt,
        query_rate=query_cnt / max(1, n),
        budget_matched_to=None,
        masked_count=masked,
        retained_count=retained,
        total_observed_latency_ms=tot_obs_lat,
        total_hypothetical_wire_latency_ms=tot_wire_lat,
        total_prompt_tokens=tot_p_tokens,
        total_completion_tokens=tot_c_tokens,
        normalized_compute_cost=query_cnt * 0.50,
        decisions=decisions,
    )


def evaluate_b4_random(
    tracks: List[Dict[str, Any]],
    vlm_provider: VLMProvider,
    target_budget_k: int,
    random_seed: int = 42,
) -> PolicyExecutionSummary:
    """B4: Random Routing (Strictly Budget-Matched to B6: K_B4 == K_B6)."""
    eligible_indices = [i for i, t in enumerate(tracks) if t.get("crop_paths")]
    k = min(len(eligible_indices), max(0, target_budget_k))

    rng = random.Random(random_seed)
    queried_indices: Set[int] = set(rng.sample(eligible_indices, k)) if k > 0 else set()

    decisions: List[Dict[str, Any]] = []
    query_cnt = 0
    tot_obs_lat = 0.0
    tot_wire_lat = 0.0
    tot_p_tokens = 0
    tot_c_tokens = 0
    masked = 0
    retained = 0

    for i, t in enumerate(tracks):
        tid = t["track_id"]
        sem_cls = t.get("semantic_class", "unknown")
        mot = str(t.get("observed_motion", "STATIC")).upper()
        w_disp = float(t.get("world_displacement_m", t.get("measured_displacement_m", 0.0)) or 0.0)
        crop_path = t["crop_paths"][0] if t.get("crop_paths") else None

        is_queried = (i in queried_indices) and (crop_path is not None)

        if is_queried:
            v_res, obs_lat = vlm_provider.query(tid, crop_path)
            query_cnt += 1
            tot_obs_lat += obs_lat
            tot_wire_lat += v_res.latency_ms
            tot_p_tokens += v_res.prompt_tokens
            tot_c_tokens += v_res.completion_tokens

            resolved_aff = v_res.contextual_movability
            if resolved_aff == "MOBILE":
                act = "MASK_FROM_TRACKING"
                masked += 1
                reasons = ["B4_RANDOM_QUERY_RESOLVED_MOBILE"]
            elif resolved_aff == "MOVABLE" and mot == "MOVING":
                act = "MASK_FROM_TRACKING"
                masked += 1
                reasons = ["B4_RANDOM_QUERY_RESOLVED_MOVABLE_MOVING"]
            else:
                act = "RETAIN"
                retained += 1
                reasons = ["B4_RANDOM_QUERY_RESOLVED_STATIC"]

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=True,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                observed_latency_ms=obs_lat,
                hypothetical_wire_latency_ms=v_res.latency_ms,
                prompt_tokens=v_res.prompt_tokens,
                completion_tokens=v_res.completion_tokens,
            )
        else:
            tot_obs_lat += 0.02
            # Fallback to simple geometry heuristic
            if mot == "MOVING" or w_disp >= 0.30:
                act = "MASK_FROM_TRACKING"
                resolved_aff = "MOVABLE"
                masked += 1
                reasons = ["B4_UNQUERIED_GEOMETRY_FALLBACK_MASK"]
            else:
                act = "RETAIN"
                resolved_aff = "IMMOVABLE"
                retained += 1
                reasons = ["B4_UNQUERIED_GEOMETRY_FALLBACK_RETAIN"]

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=False,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                observed_latency_ms=0.02,
            )

        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B4_Random",
        policy_name=f"B4 (Random Budget-Matched K={k})",
        description=f"Queries exactly K={k} candidates uniformly at random (budget-matched to B6)",
        total_candidates=n,
        query_count=query_cnt,
        query_rate=query_cnt / max(1, n),
        budget_matched_to="B6_Estimator_Aware",
        masked_count=masked,
        retained_count=retained,
        total_observed_latency_ms=tot_obs_lat,
        total_hypothetical_wire_latency_ms=tot_wire_lat,
        total_prompt_tokens=tot_p_tokens,
        total_completion_tokens=tot_c_tokens,
        normalized_compute_cost=query_cnt * 0.50,
        decisions=decisions,
    )


def evaluate_b5_confidence_threshold(
    tracks: List[Dict[str, Any]],
    vlm_provider: VLMProvider,
) -> PolicyExecutionSummary:
    """B5: Confidence-Threshold / Ambiguity Routing (queries high uncertainty or borderline displacement)."""
    decisions: List[Dict[str, Any]] = []
    query_cnt = 0
    tot_obs_lat = 0.0
    tot_wire_lat = 0.0
    tot_p_tokens = 0
    tot_c_tokens = 0
    masked = 0
    retained = 0

    for t in tracks:
        tid = t["track_id"]
        sem_cls = t.get("semantic_class", "unknown")
        mot = str(t.get("observed_motion", "STATIC")).upper()
        w_disp = float(t.get("world_displacement_m", t.get("measured_displacement_m", 0.0)) or 0.0)
        disp_unc = float(t.get("displacement_uncertainty_m", 0.0) or 0.0)
        crop_path = t["crop_paths"][0] if t.get("crop_paths") else None

        # Borderline displacement or high spatial dispersion
        needs_query = ((0.12 <= w_disp < 0.35) or (disp_unc > 0.08)) and (crop_path is not None)

        if needs_query:
            v_res, obs_lat = vlm_provider.query(tid, crop_path)
            query_cnt += 1
            tot_obs_lat += obs_lat
            tot_wire_lat += v_res.latency_ms
            tot_p_tokens += v_res.prompt_tokens
            tot_c_tokens += v_res.completion_tokens

            resolved_aff = v_res.contextual_movability
            if resolved_aff == "MOBILE" or (resolved_aff == "MOVABLE" and mot == "MOVING"):
                act = "MASK_FROM_TRACKING"
                masked += 1
                reasons = ["B5_CONFIDENCE_VLM_RESOLVED_DYNAMIC"]
            else:
                act = "RETAIN"
                retained += 1
                reasons = ["B5_CONFIDENCE_VLM_RESOLVED_STATIC"]

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=True,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                observed_latency_ms=obs_lat,
                hypothetical_wire_latency_ms=v_res.latency_ms,
                prompt_tokens=v_res.prompt_tokens,
                completion_tokens=v_res.completion_tokens,
            )
        else:
            tot_obs_lat += 0.02
            if mot == "MOVING" or w_disp >= 0.30:
                act = "MASK_FROM_TRACKING"
                resolved_aff = "MOVABLE"
                masked += 1
                reasons = ["B5_CONFIDENCE_FALLBACK_MASK"]
            else:
                act = "RETAIN"
                resolved_aff = "IMMOVABLE"
                retained += 1
                reasons = ["B5_CONFIDENCE_FALLBACK_RETAIN"]

            rec = TrackDecisionRecord(
                track_id=tid,
                semantic_class=sem_cls,
                vlm_queried=False,
                resolved_affordance=resolved_aff,
                action=act,
                reason_codes=reasons,
                observed_latency_ms=0.02,
            )

        decisions.append(rec.to_dict())

    n = len(tracks)
    return PolicyExecutionSummary(
        policy_key="B5_Confidence_Threshold",
        policy_name="B5 (Confidence / Ambiguity Threshold)",
        description="Queries candidates with high spatial dispersion or borderline displacement",
        total_candidates=n,
        query_count=query_cnt,
        query_rate=query_cnt / max(1, n),
        budget_matched_to=None,
        masked_count=masked,
        retained_count=retained,
        total_observed_latency_ms=tot_obs_lat,
        total_hypothetical_wire_latency_ms=tot_wire_lat,
        total_prompt_tokens=tot_p_tokens,
        total_completion_tokens=tot_c_tokens,
        normalized_compute_cost=query_cnt * 0.50,
        decisions=decisions,
    )


def run_fair_policy_comparison_suite(
    tracks: List[Dict[str, Any]],
    vlm_provider: VLMProvider,
    calibrated_model: Optional[CalibratedUtilityModel] = None,
    lambda_cost: float = 0.45,
    random_seed: int = 42,
) -> Dict[str, PolicyExecutionSummary]:
    """Execute complete fair policy comparison suite (B0 through B6).

    Executes B6 first to determine its query budget K, then budget-matches B4
    so K_B4 == K_B6. Evaluates all policies with dual latency and token accounting.
    """
    # 1. Evaluate B6 (Proposed Estimator-Aware) first to establish fair query budget
    b6_summary = evaluate_b6_estimator_aware(
        tracks=tracks,
        vlm_provider=vlm_provider,
        calibrated_model=calibrated_model,
        lambda_cost=lambda_cost,
    )
    b6_budget_k = b6_summary.query_count

    # 2. Evaluate remaining baselines with strict fairness controls
    b0_summary = evaluate_b0_blind_retain(tracks)
    b1_summary = evaluate_b1_blind_invalidate(tracks)
    b2_summary = evaluate_b2_geometry_only(tracks)
    b3_summary = evaluate_b3_vlm_all(tracks, vlm_provider=vlm_provider)
    b4_summary = evaluate_b4_random(
        tracks=tracks,
        vlm_provider=vlm_provider,
        target_budget_k=b6_budget_k,
        random_seed=random_seed,
    )
    b5_summary = evaluate_b5_confidence_threshold(tracks, vlm_provider=vlm_provider)

    return {
        "B0_Blind_RETAIN": b0_summary,
        "B1_Blind_INVALIDATE": b1_summary,
        "B2_Geometry_Only": b2_summary,
        "B3_VLM_All": b3_summary,
        "B4_Random": b4_summary,
        "B5_Confidence_Threshold": b5_summary,
        "B6_Estimator_Aware": b6_summary,
    }
