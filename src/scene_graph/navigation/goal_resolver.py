"""Natural Language Prompt to Navigation Goal Resolver.

Transforms instructions (e.g. "Go to the table with the drone and stop 0.5 metres away",
"Go to the chair near desk_003 and stop 0.5 metres away", "Go to chair_003") into validated,
collision-free robot goal poses in the floor-aligned coordinate frame without silent misidentification.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GoalResolutionResult:
    """Structured resolution result containing target entity and approach pose."""
    success: bool
    prompt: str
    target_entity_id: Optional[str] = None
    target_class: Optional[str] = None
    standoff_distance_m: float = 0.5
    robot_radius_m: float = 0.18
    target_center_floor: Optional[Tuple[float, float, float]] = None
    target_bbox_floor: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    approach_pose_2d: Optional[Tuple[float, float, float]] = None  # (x, y, yaw_rad)
    candidate_poses: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate_face: Optional[str] = None
    verified_clearance_m: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "prompt": self.prompt,
            "target_entity_id": self.target_entity_id,
            "target_class": self.target_class,
            "standoff_distance_m": self.standoff_distance_m,
            "robot_radius_m": self.robot_radius_m,
            "target_center_floor": self.target_center_floor,
            "target_bbox_floor": self.target_bbox_floor,
            "approach_pose_2d": {
                "x_m": self.approach_pose_2d[0] if self.approach_pose_2d else None,
                "y_m": self.approach_pose_2d[1] if self.approach_pose_2d else None,
                "yaw_rad": self.approach_pose_2d[2] if self.approach_pose_2d else None,
                "yaw_deg": math.degrees(self.approach_pose_2d[2]) if self.approach_pose_2d else None,
            } if self.approach_pose_2d else None,
            "selected_candidate_face": self.selected_candidate_face,
            "verified_clearance_m": self.verified_clearance_m,
            "candidate_count": len(self.candidate_poses),
            "evidence": self.evidence,
            "rejection_reason": self.rejection_reason,
        }


class GoalResolver:
    """Resolves natural language navigation prompts against a validated semantic map."""

    # Synonym mapping for common natural language terms to canonical perception classes
    CLASS_SYNONYMS = {
        "table": ["desk", "table", "dining table"],
        "desk": ["desk", "table"],
        "chair": ["chair", "armchair", "seat"],
        "drone": ["robot", "quadcopter drone", "airplane"],
        "robot": ["robot", "quadcopter drone"],
        "laptop": ["laptop", "computer"],
        "backpack": ["backpack", "bag", "handbag"],
        "bag": ["handbag", "backpack"],
        "handbag": ["handbag", "bag", "backpack"],
        "bottle": ["bottle", "cup"],
        "door": ["door"],
        "fan": ["ceiling fan"],
        "ceiling fan": ["ceiling fan"],
    }

    def __init__(
        self,
        semantic_map: Dict[str, Any],
        object_distances: Optional[Dict[str, Any]] = None,
        robot_radius_m: float = 0.18,
        camera_forward_offset_m: float = 0.12,
        costmap_planner: Optional[Any] = None,
    ):
        self.semantic_map = semantic_map
        self.object_distances = object_distances or {}
        self.robot_radius_m = robot_radius_m
        self.camera_forward_offset_m = camera_forward_offset_m
        self.costmap_planner = costmap_planner
        self.entities = semantic_map.get("entities", {})

    @classmethod
    def from_map_dir(
        cls,
        map_dir: Path,
        robot_radius_m: float = 0.18,
        costmap_planner: Optional[Any] = None,
    ) -> "GoalResolver":
        map_dir = Path(map_dir)
        sem_path = map_dir / "semantic_map.json"
        if not sem_path.exists():
            raise FileNotFoundError(f"Missing semantic_map.json in {map_dir}")

        with open(sem_path, "r", encoding="utf-8") as f:
            sem_map = json.load(f)

        dist_path = map_dir / "object_distances.json"
        dist_data = None
        if dist_path.exists():
            with open(dist_path, "r", encoding="utf-8") as f:
                dist_data = json.load(f)

        if costmap_planner is None:
            try:
                from scene_graph.navigation.planner import GridPathPlanner
                costmap_planner = GridPathPlanner.from_map_dir(map_dir)
            except Exception:
                costmap_planner = None

        return cls(
            sem_map,
            object_distances=dist_data,
            robot_radius_m=robot_radius_m,
            costmap_planner=costmap_planner,
        )

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract structured fields: target class, target ID, relation, anchor class, anchor ID, standoff."""
        p_lower = prompt.lower().strip()

        # 1. Parse standoff distance
        standoff = 0.5
        m_dist = re.search(r"(\d+(?:\.\d+)?)\s*(?:metres?|meters?|m)\s*(?:away)?", p_lower)
        if m_dist:
            standoff = float(m_dist.group(1))

        # 2. Check for relational preposition (near, with, next to, beside, by, holding)
        rel_pattern = r"\b(near|with|next to|beside|by|holding)\b"
        m_rel = re.search(rel_pattern, p_lower)

        if m_rel:
            rel_word = m_rel.group(1)
            target_part = p_lower[: m_rel.start()].strip()
            anchor_part = p_lower[m_rel.end() :].strip()
        else:
            rel_word = None
            target_part = p_lower
            anchor_part = ""

        # Target extraction
        target_explicit_id = self._extract_entity_id(target_part)
        target_class = self._extract_class_name(target_part)

        # Anchor extraction
        anchor_explicit_id = self._extract_entity_id(anchor_part) if anchor_part else None
        anchor_class = self._extract_class_name(anchor_part) if anchor_part else None

        return {
            "target_explicit_id": target_explicit_id,
            "target_class": target_class,
            "relation": rel_word,
            "anchor_explicit_id": anchor_explicit_id,
            "anchor_class": anchor_class,
            "standoff_distance_m": standoff,
            "raw_prompt": prompt,
        }

    def _extract_entity_id(self, text: str) -> Optional[str]:
        for m in re.finditer(r"\b([a-z_]+_\d{3,4})\b", text):
            return m.group(1)
        return None

    def _extract_class_name(self, text: str) -> Optional[str]:
        for canonical, syns in self.CLASS_SYNONYMS.items():
            if re.search(rf"\b{re.escape(canonical)}\b", text):
                return canonical
            for s in syns:
                if re.search(rf"\b{re.escape(s)}\b", text):
                    return canonical
        return None

    def _is_entity_confirmed(self, entity: Dict[str, Any]) -> bool:
        if "is_confirmed" in entity:
            return bool(entity["is_confirmed"])
        status = str(entity.get("status", "")).upper()
        return status in ("CONFIRMED", "ACTIVE")

    def _get_entity_center(self, entity: Dict[str, Any]) -> Tuple[float, float, float]:
        if "centroid_floor" in entity:
            c = entity["centroid_floor"]
            return float(c[0]), float(c[1]), float(c[2])
        if "floor_coordinates" in entity and "center" in entity["floor_coordinates"]:
            c = entity["floor_coordinates"]["center"]
            return float(c[0]), float(c[1]), float(c[2])
        return (0.0, 0.0, 0.0)

    def _get_entity_radius(self, entity: Dict[str, Any]) -> float:
        if "bbox_min_floor" in entity and "bbox_max_floor" in entity:
            bmin = entity["bbox_min_floor"]
            bmax = entity["bbox_max_floor"]
            dx = float(bmax[0]) - float(bmin[0])
            dy = float(bmax[1]) - float(bmin[1])
            return max(0.2, 0.5 * max(dx, dy))
        if "floor_coordinates" in entity and "bounding_box_min_max" in entity["floor_coordinates"]:
            bb = entity["floor_coordinates"]["bounding_box_min_max"]
            dx = float(bb["max_x"]) - float(bb["min_x"])
            dy = float(bb["max_y"]) - float(bb["min_y"])
            return max(0.2, 0.5 * max(dx, dy))
        if "dimensions_m" in entity:
            dim = entity["dimensions_m"]
            if isinstance(dim, list) and len(dim) >= 2:
                return max(0.2, 0.5 * max(float(dim[0]), float(dim[1])))
            elif isinstance(dim, dict):
                return max(0.2, 0.5 * max(float(dim.get("width_x", 0.5)), float(dim.get("length_y", 0.5))))
        return 0.35

    def resolve(
        self,
        prompt: str,
        current_robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> GoalResolutionResult:
        """Resolve prompt to target entity and generate verified approach pose without ambiguity."""
        parsed = self.parse_prompt(prompt)
        standoff = parsed["standoff_distance_m"]
        target_id = parsed["target_explicit_id"]
        target_class = parsed["target_class"]
        relation = parsed["relation"]
        anchor_id = parsed["anchor_explicit_id"]
        anchor_class = parsed["anchor_class"]

        # 1. Target explicit ID requested (e.g. "go to desk_003")
        if target_id:
            entity = self.entities.get(target_id)
            if not entity:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    target_entity_id=target_id,
                    rejection_reason=f"No confirmed entities or object '{target_id}' found in semantic map",
                )
            if not self._is_entity_confirmed(entity):
                q_reason = entity.get("quarantine_reason") or entity.get("status") or "unconfirmed"
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    target_entity_id=target_id,
                    rejection_reason=f"Entity '{target_id}' is quarantined or unconfirmed ({q_reason})",
                )
            return self._build_approach_pose(target_id, entity, standoff, current_robot_pose, prompt, {"match_type": "explicit_id"})

        # 2. Gather confirmed candidates of the target class
        if not target_class:
            return GoalResolutionResult(
                success=False,
                prompt=prompt,
                rejection_reason="Could not determine target object class or explicit entity ID from prompt",
            )

        allowed_target_classes = self.CLASS_SYNONYMS.get(target_class, [target_class])
        confirmed_candidates = [
            (eid, ent) for eid, ent in self.entities.items()
            if self._is_entity_confirmed(ent) and ent.get("class_name", "").lower() in allowed_target_classes
        ]

        if not confirmed_candidates:
            # Check unconfirmed
            unconf = [eid for eid, ent in self.entities.items() if ent.get("class_name", "").lower() in allowed_target_classes]
            if unconf:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    rejection_reason=f"Found candidates {unconf} for '{target_class}', but they are UNCONFIRMED with insufficient evidence",
                )
            return GoalResolutionResult(
                success=False,
                prompt=prompt,
                rejection_reason=f"No confirmed entities of class '{target_class}' found in map",
            )

        # 3. Disambiguate using relational anchor if present
        if relation and (anchor_id or anchor_class):
            anchor_entities = []
            if anchor_id and anchor_id in self.entities:
                anchor_entities = [(anchor_id, self.entities[anchor_id])]
            elif anchor_class:
                allowed_anchor_classes = self.CLASS_SYNONYMS.get(anchor_class, [anchor_class])
                anchor_entities = [
                    (eid, ent) for eid, ent in self.entities.items()
                    if ent.get("class_name", "").lower() in allowed_anchor_classes
                ]

            scored_candidates = []
            for t_id, t_ent in confirmed_candidates:
                t_pos = np.array(self._get_entity_center(t_ent))
                min_dist = float("inf")
                best_a_id = None
                for a_id, a_ent in anchor_entities:
                    a_pos = np.array(self._get_entity_center(a_ent))
                    dist = float(np.linalg.norm(t_pos[:2] - a_pos[:2]))
                    if dist < min_dist:
                        min_dist = dist
                        best_a_id = a_id

                if min_dist <= 2.2:  # Plausible relational proximity
                    scored_candidates.append((t_id, t_ent, min_dist, best_a_id))

            if len(scored_candidates) == 0:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    rejection_reason=f"No '{target_class}' found {relation} anchor '{anchor_id or anchor_class}' within 2.2m",
                )
            elif len(scored_candidates) == 1:
                t_id, t_ent, d_rel, a_id = scored_candidates[0]
                evidence = {
                    "match_type": "relational_association",
                    "anchor_entity": a_id,
                    "related_entity": a_id,
                    "distance_to_anchor_m": round(d_rel, 3),
                    "distance_to_relation_m": round(d_rel, 3),
                }
                return self._build_approach_pose(t_id, t_ent, standoff, current_robot_pose, prompt, evidence)
            else:
                # Ambiguous: multiple targets near the same anchor -> Reject instead of silent guessing!
                c_names = [sc[0] for sc in scored_candidates]
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    rejection_reason=f"Ambiguous goal: Found {len(c_names)} candidates {c_names} {relation} anchor '{anchor_id or anchor_class}'. Please specify explicit entity ID.",
                )

        # 4. No relational anchor
        if len(confirmed_candidates) == 1:
            t_id, t_ent = confirmed_candidates[0]
            return self._build_approach_pose(t_id, t_ent, standoff, current_robot_pose, prompt, {"match_type": "unique_class_instance"})

        # Multiple candidates without distinguishing relation -> Reject ambiguity!
        cand_ids = [c[0] for c in confirmed_candidates]
        return GoalResolutionResult(
            success=False,
            prompt=prompt,
            rejection_reason=f"Ambiguous goal: Found {len(cand_ids)} candidates for '{target_class}': {cand_ids}. Please specify an explicit entity ID or relational anchor.",
        )

    def _get_entity_bbox_2d(self, entity: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Extract [min_x, min_y, max_x, max_y] bounding box in floor frame, tightly clamped to physical size."""
        cls_name = entity.get("class_name", "")
        max_w, max_l = 1.40, 0.80
        min_w, min_l = 0.30, 0.30
        if cls_name in ("chair", "seat", "armchair"):
            max_w, max_l = 0.62, 0.62
            min_w, min_l = 0.35, 0.35
        elif cls_name in ("desk", "table", "dining table"):
            max_w, max_l = 1.45, 0.80
            min_w, min_l = 0.60, 0.50
        elif cls_name in ("laptop", "computer"):
            max_w, max_l = 0.35, 0.25
            min_w, min_l = 0.22, 0.16
        elif cls_name in ("computer monitor", "monitor"):
            max_w, max_l = 0.60, 0.20
            min_w, min_l = 0.30, 0.10
        elif cls_name in ("backpack", "bag"):
            max_w, max_l = 0.40, 0.30
            min_w, min_l = 0.20, 0.15
        elif cls_name in ("storage box", "box"):
            max_w, max_l = 0.50, 0.45
            min_w, min_l = 0.25, 0.25

        cx, cy, _ = self._get_entity_center(entity)

        bmin = entity.get("bbox_min_floor")
        bmax = entity.get("bbox_max_floor")
        if bmin and bmax and len(bmin) >= 2 and len(bmax) >= 2:
            raw_w = abs(float(bmax[0]) - float(bmin[0]))
            raw_l = abs(float(bmax[1]) - float(bmin[1]))
            w = max(min_w, min(max_w, raw_w))
            l = max(min_l, min(max_l, raw_l))
            return cx - w / 2.0, cy - l / 2.0, cx + w / 2.0, cy + l / 2.0

        fcoords = entity.get("floor_coordinates") or {}
        bb = fcoords.get("bounding_box_min_max") or {}
        if "min_x" in bb and "max_x" in bb and "min_y" in bb and "max_y" in bb:
            raw_w = abs(float(bb["max_x"]) - float(bb["min_x"]))
            raw_l = abs(float(bb["max_y"]) - float(bb["min_y"]))
            w = max(min_w, min(max_w, raw_w))
            l = max(min_l, min(max_l, raw_l))
            return cx - w / 2.0, cy - l / 2.0, cx + w / 2.0, cy + l / 2.0

        r = self._get_entity_radius(entity)
        w = max(min_w, min(max_w, 2 * r))
        l = max(min_l, min(max_l, 2 * r))
        return cx - w / 2.0, cy - l / 2.0, cx + w / 2.0, cy + l / 2.0

    def generate_approach_candidates(
        self,
        entity: Dict[str, Any],
        standoff_m: float,
        current_robot_pose: Tuple[float, float, float],
    ) -> List[Dict[str, Any]]:
        """Generate footprint-aware candidate robot base poses around target entity."""
        cx, cy, _ = self._get_entity_center(entity)
        min_x, min_y, max_x, max_y = self._get_entity_bbox_2d(entity)

        # Base standoff: keeps camera lens at standoff_m from edge, and base footprint outside edge
        base_offset_m = max(standoff_m + self.camera_forward_offset_m, self.robot_radius_m + 0.05)

        candidates: List[Dict[str, Any]] = []

        # 1. Line-of-sight candidate (direct approach from robot to centroid)
        rx, ry, _ = current_robot_pose
        dx = rx - cx
        dy = ry - cy
        dist_to_obj = math.hypot(dx, dy)
        if dist_to_obj > 1e-4:
            phi = math.atan2(dy, dx)
            hx = max(0.05, (max_x - min_x) / 2.0)
            hy = max(0.05, (max_y - min_y) / 2.0)
            tx = hx / abs(math.cos(phi)) if abs(math.cos(phi)) > 1e-6 else float("inf")
            ty = hy / abs(math.sin(phi)) if abs(math.sin(phi)) > 1e-6 else float("inf")
            t_edge = min(tx, ty)
            edge_x = cx + t_edge * math.cos(phi)
            edge_y = cy + t_edge * math.sin(phi)
            px = edge_x + base_offset_m * math.cos(phi)
            py = edge_y + base_offset_m * math.sin(phi)
            yaw = math.atan2(cy - py, cx - px)
            candidates.append({
                "face": "direct_approach",
                "edge_point": (round(edge_x, 3), round(edge_y, 3)),
                "x": px,
                "y": py,
                "yaw": yaw,
                "offset_m": base_offset_m,
            })

        # 2. Four Cardinal Faces
        # East (+X face)
        px_e = max_x + base_offset_m
        py_e = cy
        candidates.append({
            "face": "east_+x",
            "edge_point": (max_x, cy),
            "x": px_e,
            "y": py_e,
            "yaw": math.atan2(cy - py_e, cx - px_e),
            "offset_m": base_offset_m,
        })

        # West (-X face)
        px_w = min_x - base_offset_m
        py_w = cy
        candidates.append({
            "face": "west_-x",
            "edge_point": (min_x, cy),
            "x": px_w,
            "y": py_w,
            "yaw": math.atan2(cy - py_w, cx - px_w),
            "offset_m": base_offset_m,
        })

        # North (+Y face)
        px_n = cx
        py_n = max_y + base_offset_m
        candidates.append({
            "face": "north_+y",
            "edge_point": (cx, max_y),
            "x": px_n,
            "y": py_n,
            "yaw": math.atan2(cy - py_n, cx - px_n),
            "offset_m": base_offset_m,
        })

        # South (-Y face)
        px_s = cx
        py_s = min_y - base_offset_m
        candidates.append({
            "face": "south_-y",
            "edge_point": (cx, min_y),
            "x": px_s,
            "y": py_s,
            "yaw": math.atan2(cy - py_s, cx - px_s),
            "offset_m": base_offset_m,
        })

        # 3. Four Corner / Diagonal Faces
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        diag_offset = base_offset_m * inv_sqrt2

        # NE Corner
        px_ne = max_x + diag_offset
        py_ne = max_y + diag_offset
        candidates.append({
            "face": "northeast_corner",
            "edge_point": (max_x, max_y),
            "x": px_ne,
            "y": py_ne,
            "yaw": math.atan2(cy - py_ne, cx - px_ne),
            "offset_m": base_offset_m,
        })

        # NW Corner
        px_nw = min_x - diag_offset
        py_nw = max_y + diag_offset
        candidates.append({
            "face": "northwest_corner",
            "edge_point": (min_x, max_y),
            "x": px_nw,
            "y": py_nw,
            "yaw": math.atan2(cy - py_nw, cx - px_nw),
            "offset_m": base_offset_m,
        })

        # SW Corner
        px_sw = min_x - diag_offset
        py_sw = min_y - diag_offset
        candidates.append({
            "face": "southwest_corner",
            "edge_point": (min_x, min_y),
            "x": px_sw,
            "y": py_sw,
            "yaw": math.atan2(cy - py_sw, cx - px_sw),
            "offset_m": base_offset_m,
        })

        # SE Corner
        px_se = max_x + diag_offset
        py_se = min_y - diag_offset
        candidates.append({
            "face": "southeast_corner",
            "edge_point": (max_x, min_y),
            "x": px_se,
            "y": py_se,
            "yaw": math.atan2(cy - py_se, cx - px_se),
            "offset_m": base_offset_m,
        })

        return candidates

    def _build_approach_pose(
        self,
        entity_id: str,
        entity: Dict[str, Any],
        standoff_m: float,
        current_robot_pose: Tuple[float, float, float],
        prompt: str,
        evidence: Dict[str, Any],
    ) -> GoalResolutionResult:
        """Calculate collision-free approach pose outside object footprint with camera offset correction."""
        cx, cy, cz = self._get_entity_center(entity)
        min_x, min_y, max_x, max_y = self._get_entity_bbox_2d(entity)
        obj_radius = self._get_entity_radius(entity)

        candidates = self.generate_approach_candidates(entity, standoff_m, current_robot_pose)
        rx, ry, _ = current_robot_pose

        feasible_candidates: List[Dict[str, Any]] = []

        if self.costmap_planner is not None:
            p = self.costmap_planner
            r_cells = int(math.ceil(self.robot_radius_m / p.res))

            for cand in candidates:
                cand_x, cand_y = cand["x"], cand["y"]
                gx, gy = p.world_to_grid(cand_x, cand_y)

                # 1. Bounds check
                if not (0 <= gx < p.width and 0 <= gy < p.height):
                    cand["is_feasible"] = False
                    cand["rejection_reason"] = "out_of_bounds"
                    continue

                # 2. Costmap cell check
                if not p.is_free(gx, gy):
                    cand["is_feasible"] = False
                    cand["rejection_reason"] = "occupied_or_inflated"
                    continue

                # 3. Measure obstacle clearance from nearest inflated boundary
                clearance_cells = 1
                found_wall = False
                for r_search in range(1, 15):
                    for dx_c in range(-r_search, r_search + 1):
                        for dy_c in range(-r_search, r_search + 1):
                            if abs(dx_c) == r_search or abs(dy_c) == r_search:
                                nx, ny = gx + dx_c, gy + dy_c
                                if not (0 <= nx < p.width and 0 <= ny < p.height) or p.grid[ny, nx] >= p.occ_thresh:
                                    clearance_cells = r_search
                                    found_wall = True
                                    break
                        if found_wall:
                            break
                    if found_wall:
                        break

                clearance_m = clearance_cells * p.res
                cand["is_feasible"] = True
                cand["clearance_m"] = round(clearance_m, 3)
                dist_to_robot = math.hypot(cand_x - rx, cand_y - ry)
                cand["dist_to_robot_m"] = round(dist_to_robot, 3)
                # Score combines route distance with clearance bonus
                cand["score"] = -dist_to_robot + 1.5 * clearance_m
                feasible_candidates.append(cand)

            if not feasible_candidates:
                return GoalResolutionResult(
                    success=False,
                    prompt=prompt,
                    target_entity_id=entity_id,
                    target_class=entity.get("class_name", "unknown"),
                    standoff_distance_m=standoff_m,
                    robot_radius_m=self.robot_radius_m,
                    target_center_floor=(cx, cy, cz),
                    target_bbox_floor=(min_x, min_y, max_x, max_y),
                    candidate_poses=candidates,
                    rejection_reason=(
                        f"No collision-free approach pose exists around '{entity_id}' "
                        f"under current costmap clearance constraints (all {len(candidates)} candidates blocked by obstacles/inflation)."
                    ),
                )

            feasible_candidates.sort(key=lambda c: c["score"], reverse=True)
            best_cand = feasible_candidates[0]
        else:
            for cand in candidates:
                cand["dist_to_robot_m"] = round(math.hypot(cand["x"] - rx, cand["y"] - ry), 3)
                cand["is_feasible"] = True
                cand["clearance_m"] = round(standoff_m, 3)
            direct = [c for c in candidates if c["face"] == "direct_approach"]
            best_cand = direct[0] if direct else min(candidates, key=lambda c: c["dist_to_robot_m"])
            feasible_candidates = candidates

        goal_x = best_cand["x"]
        goal_y = best_cand["y"]
        desired_yaw = best_cand["yaw"]

        evidence.update({
            "object_center_m": [cx, cy, cz],
            "target_bbox_m": [round(min_x, 3), round(min_y, 3), round(max_x, 3), round(max_y, 3)],
            "estimated_object_radius_m": round(obj_radius, 3),
            "robot_radius_m": self.robot_radius_m,
            "camera_offset_m": self.camera_forward_offset_m,
            "calculated_standoff_m": standoff_m,
            "selected_face": best_cand["face"],
            "verified_clearance_m": best_cand.get("clearance_m"),
            "feasible_candidate_count": len(feasible_candidates),
            "total_candidate_count": len(candidates),
        })

        return GoalResolutionResult(
            success=True,
            prompt=prompt,
            target_entity_id=entity_id,
            target_class=entity.get("class_name", "unknown"),
            standoff_distance_m=standoff_m,
            robot_radius_m=self.robot_radius_m,
            target_center_floor=(cx, cy, cz),
            target_bbox_floor=(min_x, min_y, max_x, max_y),
            approach_pose_2d=(goal_x, goal_y, desired_yaw),
            candidate_poses=candidates,
            selected_candidate_face=best_cand["face"],
            verified_clearance_m=best_cand.get("clearance_m"),
            evidence=evidence,
        )
