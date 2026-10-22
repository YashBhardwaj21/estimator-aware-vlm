"""Stage 6: Stage 5 to Stage 6 Evidence Adapter.

Constructs genuine, instance-level evidence records strictly from Stage 5 outputs
and recorded observations without touching ground-truth target annotations.

Features:
1. Explicit ID-resolution layer:
   Distinguishes exact observation IDs (`obs_id`), session-scoped entity IDs
   (`global_entity_id`), and tracker IDs (`local_track_id`).
2. Robust multi-session alignment:
   Validates registration audit file and anchors. If registration is missing,
   corrupted, or unverified, cross-session quantities are marked NOT_EVALUABLE
   with explicit diagnostic reasons (no hardcoded fallback numbers).
3. Rigorous uncertainty propagation:
   Combines observation cluster dispersion standard errors with anchor registration
   RMS residuals: sigma_disp = sqrt(sigma_1^2 + sigma_2^2 + sigma_reg^2).
4. Full diagnostic integrity reporting for missing inputs, duplicate IDs,
   and unverified coordinate frames.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RegistrationInfo:
    """Validated SE(2) cross-session registration parameters."""
    is_valid: bool
    reference_session: str
    aligned_session: str
    rotation_deg: float
    translation_m: np.ndarray
    R_12: np.ndarray
    anchor_rms_residual_m: float
    anchor_count: int
    status: str
    diagnostics: List[str] = field(default_factory=list)
    median_residual_m: Optional[float] = None
    robust_inlier_count: Optional[int] = None
    robust_outlier_count: Optional[int] = None
    consensus_inlier_count: Optional[int] = None
    post_refit_strict_inlier_count: Optional[int] = None
    inlier_rms_residual_m: Optional[float] = None
    post_refit_strict_inlier_rms_residual_m: Optional[float] = None
    is_calibrated_covariance: bool = False


RegistrationAudit = RegistrationInfo


def load_registration_audit(
    audit_path: Optional[Path],
    expected_ref_session: Optional[str] = None,
    expected_align_session: Optional[str] = None,
) -> RegistrationInfo:
    """Load and strictly validate the multi-session SE(2) registration audit.

    Does NOT use hardcoded fallback transforms or fallback residual metrics.
    Validates:
      1. Schema structure and explicit non-reversed session identities.
      2. Numeric finiteness (rejects NaN/Inf) and vector/matrix dimensions.
      3. Derived 2D rotation matrix orthonormality and det(R) == +1.0 within tolerance.
      4. Anchor correspondences spatial geometry (at least 3 non-collinear, non-duplicate anchors).
      5. Recomputed residuals vs stored values within documented tolerance.
      6. Separate tracking of RANSAC consensus inliers and strict post-refit threshold inliers.
      7. Exact inlier/outlier consistency (inlier + outlier == total anchors).
    """
    if audit_path is None or not Path(audit_path).exists():
        return RegistrationInfo(
            is_valid=False,
            reference_session="",
            aligned_session="",
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="REGISTRATION_FILE_MISSING",
            diagnostics=["Registration audit file does not exist at specified path."],
        )

    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return RegistrationInfo(
            is_valid=False,
            reference_session="",
            aligned_session="",
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="REGISTRATION_FILE_MALFORMED",
            diagnostics=[f"Failed to parse registration audit JSON: {str(e)}"],
        )

    if not isinstance(data, dict):
        return RegistrationInfo(
            is_valid=False,
            reference_session="",
            aligned_session="",
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="REGISTRATION_FILE_MALFORMED",
            diagnostics=["Registration audit root is not a JSON dictionary."],
        )

    reg = data.get("registration")
    if not isinstance(reg, dict):
        if "reference_session" in data and "translation_m" in data:
            reg = data
        else:
            return RegistrationInfo(
                is_valid=False,
                reference_session="",
                aligned_session="",
                rotation_deg=0.0,
                translation_m=np.zeros(2),
                R_12=np.eye(2),
                anchor_rms_residual_m=0.0,
                anchor_count=0,
                status="MISSING_REGISTRATION_BLOCK",
                diagnostics=["JSON root lacks a valid 'registration' dictionary."],
            )

    ref_sess = str(reg.get("reference_session") or "").strip()
    align_sess = str(reg.get("aligned_session") or "").strip()

    if not ref_sess or not align_sess or ref_sess == align_sess:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="SESSION_DIRECTION_INVALID",
            diagnostics=["Missing, empty, or identical reference_session and aligned_session."],
        )

    if expected_ref_session and ref_sess != expected_ref_session:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="SESSION_DIRECTION_INVALID",
            diagnostics=[f"Reference session '{ref_sess}' does not match expected '{expected_ref_session}'."],
        )

    if expected_align_session and align_sess != expected_align_session:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="SESSION_DIRECTION_INVALID",
            diagnostics=[f"Aligned session '{align_sess}' does not match expected '{expected_align_session}'."],
        )

    robust_block = data.get("robust_estimation", {})
    robust_tf = robust_block.get("robust_transform", {}) if isinstance(robust_block, dict) else {}
    if robust_tf and "rotation_deg" in robust_tf and "translation_m" in robust_tf:
        rot_raw = robust_tf.get("rotation_deg")
        trans_raw = robust_tf.get("translation_m")
    else:
        rot_raw = reg.get("rotation_deg")
        trans_raw = reg.get("translation_m")

    # Validate rotation finiteness
    if rot_raw is None or not isinstance(rot_raw, (int, float)) or not math.isfinite(float(rot_raw)):
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=0.0,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="MALFORMED_TRANSFORM_PARAMETERS",
            diagnostics=["Missing, non-numeric, or non-finite rotation_deg."],
        )
    rot_deg = float(rot_raw)

    # Validate translation finiteness & dimension
    if not isinstance(trans_raw, (list, tuple, np.ndarray)) or len(trans_raw) < 2:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="MALFORMED_TRANSFORM_PARAMETERS",
            diagnostics=["Missing or invalid translation_m 2D vector."],
        )

    try:
        t_vec = np.array(trans_raw[:2], dtype=np.float64)
        if not np.all(np.isfinite(t_vec)):
            return RegistrationInfo(
                is_valid=False,
                reference_session=ref_sess,
                aligned_session=align_sess,
                rotation_deg=rot_deg,
                translation_m=np.zeros(2),
                R_12=np.eye(2),
                anchor_rms_residual_m=0.0,
                anchor_count=0,
                status="MALFORMED_TRANSFORM_PARAMETERS",
                diagnostics=["Non-finite values (NaN/Inf) in translation_m."],
            )
    except Exception as e:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=np.zeros(2),
            R_12=np.eye(2),
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="MALFORMED_TRANSFORM_PARAMETERS",
            diagnostics=[f"Failed to parse translation_m: {str(e)}"],
        )

    # Derive rotation matrix and check orthonormality & det = +1
    rad = math.radians(rot_deg)
    R_12 = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad),  math.cos(rad)],
    ], dtype=np.float64)

    det_R = float(np.linalg.det(R_12))
    if abs(det_R - 1.0) > 1e-5:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=t_vec,
            R_12=R_12,
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="TRANSFORM_MATRIX_INCONSISTENCY",
            diagnostics=[f"Rotation matrix determinant {det_R:.6f} != +1.0"],
        )

    ortho_err = float(np.max(np.abs(R_12.T @ R_12 - np.eye(2))))
    if ortho_err > 1e-5:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=t_vec,
            R_12=R_12,
            anchor_rms_residual_m=0.0,
            anchor_count=0,
            status="TRANSFORM_MATRIX_INCONSISTENCY",
            diagnostics=[f"Rotation matrix non-orthonormal (error {ortho_err:.6e})."],
        )

    if "R_12" in reg and isinstance(reg["R_12"], list):
        try:
            stored_R = np.array(reg["R_12"], dtype=np.float64)
            if stored_R.shape == (2, 2) and float(np.max(np.abs(stored_R - R_12))) > 1e-3:
                return RegistrationInfo(
                    is_valid=False,
                    reference_session=ref_sess,
                    aligned_session=align_sess,
                    rotation_deg=rot_deg,
                    translation_m=t_vec,
                    R_12=R_12,
                    anchor_rms_residual_m=0.0,
                    anchor_count=0,
                    status="TRANSFORM_MATRIX_INCONSISTENCY",
                    diagnostics=["Stored R_12 matrix disagrees with derived rotation matrix."],
                )
        except Exception:
            pass

    # Validate anchor correspondences
    anchors = reg.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) < 3:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=t_vec,
            R_12=R_12,
            anchor_rms_residual_m=0.0,
            anchor_count=len(anchors) if isinstance(anchors, list) else 0,
            status="INSUFFICIENT_ANCHOR_CORRESPONDENCES",
            diagnostics=[f"Insufficient anchor correspondences ({len(anchors) if isinstance(anchors, list) else 0} < 3 required)."],
        )

    pts1: List[np.ndarray] = []
    pts2: List[np.ndarray] = []
    stored_residuals: List[Optional[float]] = []

    for idx, a in enumerate(anchors):
        if not isinstance(a, dict):
            return RegistrationInfo(
                is_valid=False,
                reference_session=ref_sess,
                aligned_session=align_sess,
                rotation_deg=rot_deg,
                translation_m=t_vec,
                R_12=R_12,
                anchor_rms_residual_m=0.0,
                anchor_count=len(anchors),
                status="MALFORMED_ANCHOR_RECORD",
                diagnostics=[f"Anchor at index {idx} is not a dictionary."],
            )
        c1 = a.get("c1")
        c2 = a.get("c2")
        if c1 is None or c2 is None or len(c1) < 2 or len(c2) < 2:
            return RegistrationInfo(
                is_valid=False,
                reference_session=ref_sess,
                aligned_session=align_sess,
                rotation_deg=rot_deg,
                translation_m=t_vec,
                R_12=R_12,
                anchor_rms_residual_m=0.0,
                anchor_count=len(anchors),
                status="MALFORMED_ANCHOR_RECORD",
                diagnostics=[f"Anchor at index {idx} lacks 2D coordinate vectors."],
            )
        try:
            p1 = np.array(c1[:2], dtype=np.float64)
            p2 = np.array(c2[:2], dtype=np.float64)
            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                return RegistrationInfo(
                    is_valid=False,
                    reference_session=ref_sess,
                    aligned_session=align_sess,
                    rotation_deg=rot_deg,
                    translation_m=t_vec,
                    R_12=R_12,
                    anchor_rms_residual_m=0.0,
                    anchor_count=len(anchors),
                    status="MALFORMED_ANCHOR_RECORD",
                    diagnostics=[f"Anchor at index {idx} contains non-finite coordinates."],
                )
        except Exception as e:
            return RegistrationInfo(
                is_valid=False,
                reference_session=ref_sess,
                aligned_session=align_sess,
                rotation_deg=rot_deg,
                translation_m=t_vec,
                R_12=R_12,
                anchor_rms_residual_m=0.0,
                anchor_count=len(anchors),
                status="MALFORMED_ANCHOR_RECORD",
                diagnostics=[f"Anchor at index {idx} failed coordinate conversion: {str(e)}"],
            )

        pts1.append(p1)
        pts2.append(p2)
        stored_res = a.get("residual_m")
        stored_residuals.append(float(stored_res) if stored_res is not None and isinstance(stored_res, (int, float)) else None)

    pts1_arr = np.array(pts1)
    pts2_arr = np.array(pts2)

    # Check for duplicate anchors
    for i in range(len(pts1)):
        for j in range(i + 1, len(pts1)):
            if float(np.linalg.norm(pts1_arr[i] - pts1_arr[j])) < 1e-3 or float(np.linalg.norm(pts2_arr[i] - pts2_arr[j])) < 1e-3:
                return RegistrationInfo(
                    is_valid=False,
                    reference_session=ref_sess,
                    aligned_session=align_sess,
                    rotation_deg=rot_deg,
                    translation_m=t_vec,
                    R_12=R_12,
                    anchor_rms_residual_m=0.0,
                    anchor_count=len(anchors),
                    status="DEGENERATE_ANCHOR_GEOMETRY",
                    diagnostics=[f"Anchors {i} and {j} are duplicate/coincident coordinates."],
                )

    # Spatial collinearity and rank degeneracy check
    centered1 = pts1_arr - np.mean(pts1_arr, axis=0)
    centered2 = pts2_arr - np.mean(pts2_arr, axis=0)
    s1 = np.linalg.svd(centered1, compute_uv=False)
    s2 = np.linalg.svd(centered2, compute_uv=False)
    if s1[1] < 1e-3 or (s1[0] / max(1e-9, s1[1])) > 500.0 or s2[1] < 1e-3 or (s2[0] / max(1e-9, s2[1])) > 500.0:
        return RegistrationInfo(
            is_valid=False,
            reference_session=ref_sess,
            aligned_session=align_sess,
            rotation_deg=rot_deg,
            translation_m=t_vec,
            R_12=R_12,
            anchor_rms_residual_m=0.0,
            anchor_count=len(anchors),
            status="DEGENERATE_ANCHOR_GEOMETRY",
            diagnostics=["Anchors are nearly collinear or span insufficient 2D spatial area."],
        )

    # Recompute empirical residuals from R_12 @ c2 + t_vec against c1
    recomputed_residuals: List[float] = []
    for p1, p2 in zip(pts1, pts2):
        p2_trans = R_12 @ p2 + t_vec
        res = float(np.linalg.norm(p2_trans - p1))
        recomputed_residuals.append(res)

    # Compare recomputed and stored residuals within 2 cm tolerance
    for idx, (recomp, stored) in enumerate(zip(recomputed_residuals, stored_residuals)):
        if stored is not None:
            if abs(recomp - stored) > 0.02:
                return RegistrationInfo(
                    is_valid=False,
                    reference_session=ref_sess,
                    aligned_session=align_sess,
                    rotation_deg=rot_deg,
                    translation_m=t_vec,
                    R_12=R_12,
                    anchor_rms_residual_m=0.0,
                    anchor_count=len(anchors),
                    status="STORED_RESIDUAL_DISCREPANCY",
                    diagnostics=[f"Anchor {idx} stored residual ({stored:.4f}m) disagrees with recomputed ({recomp:.4f}m) beyond 0.02m."],
                )

    rms_residual = float(np.sqrt(np.mean(np.array(recomputed_residuals)**2)))
    median_residual = float(np.median(recomputed_residuals))

    # Identify RANSAC consensus inliers vs strict post-refit inliers (<= 0.20m)
    consensus_inliers = [
        idx for idx, a in enumerate(anchors)
        if a.get("is_robust_inlier") is True or a.get("pre_refit_consensus_inlier") is True
    ]
    strict_inliers = [idx for idx, r in enumerate(recomputed_residuals) if r <= 0.20]

    consensus_inlier_count = len(consensus_inliers) if consensus_inliers else len(strict_inliers)
    post_refit_strict_count = len(strict_inliers)

    robust_inlier_residuals = [recomputed_residuals[i] for i in (consensus_inliers or strict_inliers)]
    inlier_rms = float(np.sqrt(np.mean(np.array(robust_inlier_residuals)**2))) if robust_inlier_residuals else None
    median_residual = float(np.median(robust_inlier_residuals)) if robust_inlier_residuals else float(np.median(recomputed_residuals))

    strict_inlier_residuals = [recomputed_residuals[i] for i in strict_inliers]
    post_refit_strict_rms = float(np.sqrt(np.mean(np.array(strict_inlier_residuals)**2))) if strict_inlier_residuals else None

    residuals_block = data.get("residuals", {}) if isinstance(data.get("residuals"), dict) else {}
    stored_inlier_cnt = robust_block.get("inlier_count", residuals_block.get("inlier_count"))
    stored_outlier_cnt = robust_block.get("outlier_count", residuals_block.get("outlier_count"))

    if stored_inlier_cnt is not None and stored_outlier_cnt is not None:
        if int(stored_inlier_cnt) + int(stored_outlier_cnt) != len(anchors):
            return RegistrationInfo(
                is_valid=False,
                reference_session=ref_sess,
                aligned_session=align_sess,
                rotation_deg=rot_deg,
                translation_m=t_vec,
                R_12=R_12,
                anchor_rms_residual_m=rms_residual,
                anchor_count=len(anchors),
                status="INCONSISTENT_INLIER_COUNTS",
                diagnostics=[f"Stored inlier ({stored_inlier_cnt}) + outlier ({stored_outlier_cnt}) != total anchors ({len(anchors)})."],
            )
        robust_inlier_count = int(stored_inlier_cnt)
        robust_outlier_count = int(stored_outlier_cnt)
    else:
        robust_inlier_count = consensus_inlier_count
        robust_outlier_count = len(anchors) - consensus_inlier_count

    return RegistrationInfo(
        is_valid=True,
        reference_session=ref_sess,
        aligned_session=align_sess,
        rotation_deg=rot_deg,
        translation_m=t_vec,
        R_12=R_12,
        anchor_rms_residual_m=rms_residual,
        anchor_count=len(anchors),
        status="SE2_REGISTERED_VERIFIED",
        diagnostics=[],
        median_residual_m=median_residual,
        robust_inlier_count=robust_inlier_count,
        robust_outlier_count=robust_outlier_count,
        consensus_inlier_count=consensus_inlier_count,
        post_refit_strict_inlier_count=post_refit_strict_count,
        inlier_rms_residual_m=inlier_rms,
        post_refit_strict_inlier_rms_residual_m=post_refit_strict_rms,
        is_calibrated_covariance=False,
    )



class Stage5EvidenceAdapter:
    """Resolves observation and Stage 5 fused records by explicit keys."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        audit_path: Optional[Path] = None,
        obs_path: Optional[Path] = None,
        fused_path: Optional[Path] = None,
    ):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
        self.repo_root = Path(repo_root)

        if audit_path is None:
            audit_path = self.repo_root / "vlm" / "outputs" / "stage5_registration_audit.json"
        self.registration = load_registration_audit(audit_path)

        # Primary indexing dictionaries
        self.obs_by_obs_id: Dict[str, Dict[str, Any]] = {}
        self.obs_by_session_and_entity: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.obs_by_entity_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.obs_by_track_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.fused_by_obs_id: Dict[str, Dict[str, Any]] = {}
        self.fused_by_session_and_entity: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.fused_by_entity_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Diagnostics telemetry
        self.duplicate_obs_ids: List[str] = []
        self.missing_fused_obs_ids: List[str] = []

        # Load data with canonical input path resolution
        if obs_path is None:
            canonical_cand = self.repo_root / "vlm" / "data" / "observations" / "object_observations.jsonl"
            obs_path = canonical_cand if canonical_cand.exists() else self.repo_root / "vlm" / "outputs" / "object_observations.jsonl"
        if fused_path is None:
            fused_path = self.repo_root / "vlm" / "outputs" / "stage5_fused_evidence.jsonl"

        self._load_observations(Path(obs_path))
        self._load_fused_evidence(Path(fused_path))

    def _load_observations(self, path: Path):
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue

                oid = o.get("obs_id", "")
                eid = o.get("global_entity_id", "")
                tid = o.get("local_track_id", "")
                sess = o.get("source_session", "")

                if oid:
                    if oid in self.obs_by_obs_id:
                        self.duplicate_obs_ids.append(oid)
                        self.obs_by_obs_id[oid] = None  # Sentinel: quarantined duplicate fails closed
                    else:
                        self.obs_by_obs_id[oid] = o

                if sess and eid:
                    self.obs_by_session_and_entity[(sess, eid)].append(o)
                if eid:
                    self.obs_by_entity_id[eid].append(o)
                if tid:
                    self.obs_by_track_id[tid].append(o)

    def _load_fused_evidence(self, path: Path):
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                oid = rec.get("obs_id", "")
                eid = rec.get("entity_id", "")
                sess = rec.get("source_session", "")

                if oid:
                    if oid in self.fused_by_obs_id:
                        self.fused_by_obs_id[oid] = None  # Quarantined duplicate
                    else:
                        self.fused_by_obs_id[oid] = rec
                if sess and eid:
                    self.fused_by_session_and_entity[(sess, eid)].append(rec)
                if eid:
                    self.fused_by_entity_id[eid].append(rec)

    def resolve_session_observations(
        self,
        identifier: Optional[str],
        session_keyword: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """Explicitly resolve an observation or entity identifier to its records.

        Returns:
            (observations, fused_records, diagnostics)
        """
        if not identifier:
            return [], [], ["NO_IDENTIFIER_PROVIDED"]

        obs_matches: List[Dict[str, Any]] = []
        fused_matches: List[Dict[str, Any]] = []
        diagnostics: List[str] = []

        # Strategy 1: Direct match on unique obs_id
        if identifier in self.obs_by_obs_id:
            obs = self.obs_by_obs_id[identifier]
            if obs is None:
                return [], [], [f"Observation '{identifier}' is a quarantined duplicate ID; failing closed."]
            obs_matches.append(obs)
            if identifier in self.fused_by_obs_id:
                frec = self.fused_by_obs_id[identifier]
                if frec is not None:
                    fused_matches.append(frec)
            else:
                diagnostics.append(f"Observation {identifier} found, but missing fused evidence.")

            # Validate session if requested
            if session_keyword and session_keyword not in obs.get("source_session", ""):
                diagnostics.append(
                    f"Observation {identifier} session mismatch: expected '{session_keyword}', got '{obs.get('source_session')}'."
                )
            return obs_matches, fused_matches, diagnostics

        # Strategy 2: Match by session and global_entity_id
        resolved_by_entity = False
        if session_keyword:
            for (sess, eid), o_list in self.obs_by_session_and_entity.items():
                if session_keyword in sess and eid == identifier:
                    obs_matches.extend(o_list)
                    f_list = self.fused_by_session_and_entity.get((sess, eid), [])
                    fused_matches.extend(f_list)
                    resolved_by_entity = True
                    break

        # Strategy 3: Global entity ID fallback across sessions
        if not resolved_by_entity and identifier in self.obs_by_entity_id:
            for o in self.obs_by_entity_id[identifier]:
                if session_keyword is None or session_keyword in o.get("source_session", ""):
                    obs_matches.append(o)
                    oid = o.get("obs_id")
                    if oid and oid in self.fused_by_obs_id:
                        fused_matches.append(self.fused_by_obs_id[oid])
            if obs_matches:
                resolved_by_entity = True

        # Strategy 4: Tracker ID match
        if not resolved_by_entity and identifier in self.obs_by_track_id:
            for o in self.obs_by_track_id[identifier]:
                if session_keyword is None or session_keyword in o.get("source_session", ""):
                    obs_matches.append(o)
                    oid = o.get("obs_id")
                    if oid and oid in self.fused_by_obs_id:
                        fused_matches.append(self.fused_by_obs_id[oid])
            if obs_matches:
                resolved_by_entity = True

        if not resolved_by_entity:
            diagnostics.append(f"Identifier '{identifier}' could not be resolved to any recorded observation.")

        return obs_matches, fused_matches, diagnostics

    def build_instance_evidence(self, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an instance-level evidence dictionary strictly from estimated features.

        IMPORTANT: Ground-truth benchmark target fields ('contextual_movability_gt',
        'observed_motion_gt', 'expected_decision', 'expected_reason') are NEVER
        used as routing features or passed to the policy.
        """
        entity_id = manifest_entry.get("entity_id") or manifest_entry.get("physical_instance_id") or "unknown"
        sem_class = manifest_entry.get("semantic_class", "object")
        s1_id = manifest_entry.get("s1_obs_id")
        s2_id = manifest_entry.get("s2_obs_id")

        if not s1_id and not s2_id and "source_obs_ids" in manifest_entry:
            s_ids = manifest_entry.get("source_obs_ids", [])
            if len(s_ids) >= 1:
                s1_id = s_ids[0]
            if len(s_ids) >= 2:
                s2_id = s_ids[1]

        diagnostics: List[str] = []

        # 1. Resolve Session 1 and Session 2 independently
        s1_obs, s1_fused, d1 = self.resolve_session_observations(s1_id, session_keyword="recording_01")
        s2_obs, s2_fused, d2 = self.resolve_session_observations(s2_id, session_keyword="recording_02")
        diagnostics.extend(d1)
        diagnostics.extend(d2)

        # 2. Extract 2D/3D floor centroids for both sessions
        s1_points: List[np.ndarray] = []
        for o in s1_obs:
            cf = o.get("centroid_floor")
            if cf is not None and len(cf) >= 2 and all(x is not None for x in cf[:2]):
                s1_points.append(np.array(cf[:2], dtype=np.float64))

        s2_points: List[np.ndarray] = []
        for o in s2_obs:
            cf = o.get("centroid_floor")
            if cf is not None and len(cf) >= 2 and all(x is not None for x in cf[:2]):
                s2_points.append(np.array(cf[:2], dtype=np.float64))

        # 3. Check Pose & Depth Validity (Explicit checks, no fake defaults)
        all_obs = s1_obs + s2_obs
        all_fused = s1_fused + s2_fused

        if all_obs:
            has_valid_pose = any(
                bool(o.get("has_valid_pose", False) and o.get("camera_pose") is not None and len(o.get("camera_pose", [])) >= 7)
                for o in all_obs
            )
            depth_valid = any(
                bool(o.get("centroid_camera") is not None and len(o.get("centroid_camera", [])) >= 3 and o["centroid_camera"][2] is not None and 0.20 <= o["centroid_camera"][2] <= 4.80)
                for o in all_obs
            )
        else:
            has_valid_pose = False
            depth_valid = False
            diagnostics.append("No observations available; pose and depth are invalid.")

        # 4. Cross-Session Relocation Calculation with Rigorous Uncertainty Propagation
        estimated_disp_m: Optional[float] = None
        disp_uncertainty_m: Optional[float] = None
        observed_motion = "UNKNOWN"
        measurement_status = "MEASUREMENT_UNAVAILABLE"
        relocation_evaluable = False

        if s1_points and s2_points:
            if not self.registration.is_valid:
                relocation_evaluable = False
                observed_motion = "UNKNOWN"
                measurement_status = f"REGISTRATION_INVALID_{self.registration.status}"
                diagnostics.append(f"Cross-session registration failed: {self.registration.status}")
            else:
                relocation_evaluable = True
                measurement_status = "SE2_REGISTERED_EVALUATED"
                p1_arr = np.array(s1_points)
                p2_arr = np.array(s2_points)

                c1 = np.median(p1_arr, axis=0)
                c2 = np.median(p2_arr, axis=0)

                # Transform session 2 centroid into session 1 reference frame: c2_trans = R_12 @ c2 + t_12
                c2_transformed = self.registration.R_12 @ c2 + self.registration.translation_m

                estimated_disp_m = float(np.linalg.norm(c2_transformed - c1))

                # Dispersion of observation cluster
                std1 = float(np.mean(np.std(p1_arr, axis=0))) if len(p1_arr) > 1 else 0.05
                std2 = float(np.mean(np.std(p2_arr, axis=0))) if len(p2_arr) > 1 else 0.05

                # Standard error of estimated centroids
                sem1 = std1 / math.sqrt(len(p1_arr))
                sem2 = std2 / math.sqrt(len(p2_arr))

                # Empirical registration uncertainty heuristic: residual dispersion from robust inlier set
                # Note: This is an empirical dispersion heuristic, not a calibrated Gaussian covariance.
                k_anchors = max(1, getattr(self.registration, "robust_inlier_count", self.registration.anchor_count))
                reg_disp = getattr(self.registration, "inlier_rms_residual_m", self.registration.anchor_rms_residual_m)
                reg_disp_heuristic = reg_disp / math.sqrt(k_anchors)

                # Total estimated displacement uncertainty heuristic
                disp_uncertainty_m = float(math.sqrt(sem1**2 + sem2**2 + reg_disp_heuristic**2))

                # Statistically significant displacement (minimum physical threshold 0.20m and 2.0-sigma)
                if estimated_disp_m >= max(0.20, 2.0 * disp_uncertainty_m):
                    observed_motion = "DISPLACED"
                else:
                    observed_motion = "STATIC"

        elif all_fused:
            # Single-session motion tracking within a single recording
            # Cross-session displacement is strictly unavailable; do not manufacture 0.0 or 0.04 defaults
            estimated_disp_m = None
            disp_uncertainty_m = None
            measurement_status = "SINGLE_SESSION_CROSS_DISPLACEMENT_UNAVAILABLE"
            relocation_evaluable = False

            intra_motions = [f["temporal"]["current_motion"] for f in all_fused if "temporal" in f and "current_motion" in f["temporal"]]
            if "MOVING" in intra_motions:
                observed_motion = "MOVING"
            elif "DISPLACED" in intra_motions:
                observed_motion = "DISPLACED"
            elif "STATIC" in intra_motions:
                observed_motion = "STATIC"
            else:
                observed_motion = "UNKNOWN"
            diagnostics.append("Cross-session displacement not evaluable: only single-session observations available.")

        else:
            # No geometric observations available
            estimated_disp_m = None
            disp_uncertainty_m = None
            measurement_status = "NO_OBSERVATIONS_AVAILABLE"
            observed_motion = "UNKNOWN"
            relocation_evaluable = False
            diagnostics.append("Cross-session displacement not evaluable: no valid geometric observations.")

        # 5. Extract Semantic & Geometric Aggregates from Fused Evidence
        entropy_vals = [f["semantics"]["semantic_entropy"] for f in all_fused if "semantics" in f and "semantic_entropy" in f["semantics"]]
        geo_scores = [f["geometry"]["geometric_consistency_score"] for f in all_fused if "geometry" in f and "geometric_consistency_score" in f["geometry"]]
        provenances = set(f["geometry"]["provenance"] for f in all_fused if "geometry" in f and "provenance" in f["geometry"])

        mean_entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
        mean_geo_score = float(np.mean(geo_scores)) if geo_scores else 0.0
        primary_prov = list(provenances)[0] if provenances else ("MEASURED_FULL" if depth_valid else "MISSING")

        has_images = bool(manifest_entry.get("has_images", False))
        views_count = int(manifest_entry.get("views_count", 0))
        crops_dir = manifest_entry.get("crops_dir")

        return {
            "entity_id": entity_id,
            "semantic_class": sem_class,
            "observed_motion": observed_motion,
            "measurement_status": measurement_status,
            "world_displacement_m": round(estimated_disp_m, 4) if estimated_disp_m is not None else None,
            "measured_displacement_m": round(estimated_disp_m, 4) if estimated_disp_m is not None else None,
            "displacement_uncertainty_m": round(disp_uncertainty_m, 4) if disp_uncertainty_m is not None else None,
            "displacement_uncertainty_heuristic_m": round(disp_uncertainty_m, 4) if disp_uncertainty_m is not None else None,
            "registration_residual_dispersion_m": round(reg_disp, 4) if 'reg_disp' in locals() else None,
            "centroid_dispersion_s1_m": round(std1, 4) if 'std1' in locals() else None,
            "centroid_dispersion_s2_m": round(std2, 4) if 'std2' in locals() else None,
            "is_calibrated_uncertainty": False,
            "relocation_evaluable": relocation_evaluable,
            "depth_valid": depth_valid,
            "has_valid_pose": has_valid_pose,
            "has_images": has_images,
            "views_count": views_count,
            "crops_dir": crops_dir,
            "semantic_entropy": round(mean_entropy, 4),
            "geometric_consistency_score": round(mean_geo_score, 4),
            "evidence_provenance": primary_prov,
            "s1_obs_count": len(s1_obs),
            "s2_obs_count": len(s2_obs),
            "registration_status": self.registration.status,
            "diagnostics": diagnostics,
        }
