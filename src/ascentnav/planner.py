"""ASCENT's LLM frontier planner, vendored for one environment.

Transcribed from `relative_work/ascent/ascent/llm_planner.py` (class
`Ascent_LLM_Planner`) with the `[env]` indexing removed, the LLM behind a
plain callable, and the priors read from files this repo ships. The prompts
are byte-for-byte the reference's. Where the reference is inconsistent with
itself, the reference wins and the inconsistency is documented, not fixed:

- `get_best_frontier` returns immediately on a single frontier (:82-86),
  BEFORE the sticky/disable bookkeeping and before `_last_frontier` is
  updated. The OSG port ran the sticky rule on that path and could retire a
  floor's only frontier -- which is one of the two ways the trace diagnosis
  found it climbing stairs on same-floor episodes.
- The multi-floor prompt is DEAD in the reference run: `_explore` never
  passes `floor_num`, so the `[1]` default makes `floor_num > 1` false and
  the `-100/-200` sentinels never occur. `multi_floor=False` reproduces that;
  the code is here for when the caller passes a real floor count.
- `_disabled_frontiers` and `_best_frontier_selection_count` live on the
  OBSTACLE MAP, i.e. per floor, and the sticky reset is suppressed while
  `_neighbor_search` is set (:265). Both were missing from the port.

The three sticky-rule counters (`_last_frontier`, `_frontier_stick_step`,
`_last_frontier_distance`) are shared with `_get_close_to_stair` in the
reference (`ascent_policy.py:1015-1042`), so they are attributes here and the
agent reads them.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    FLOOR_EXP_STEP_THRESHOLD,
    INDENT_L1,
    INDENT_L2,
    MULTI_FLOOR_ASK_STEP_THRESHOLD,
    REFERENCE_ROOMS,
    REPEATED_SELECTION_THRESHOLD,
    STICKY_FRONTIER_DISTANCE_THRESHOLD,
    STICKY_FRONTIER_STEP_THRESHOLD,
)

log = logging.getLogger(__name__)

# Sentinels `_decide_frontier_with_llm` returns as the VALUE when the multi-floor
# prompt chose another storey (`llm_planner.py:243-246`); `_explore` routes them
# to the stair navigation (`ascent_policy.py:745-752`).
GO_UP = -100
GO_DOWN = -200

# `get_room_probabilities` widens the query with these (`llm_planner.py:369-372`).
SYNONYMS = {"couch": ["sofa"], "sofa": ["couch"]}


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    win = min(7, (min(a.shape[:2]) // 2) * 2 - 1)        # odd, within the image
    if a.shape != b.shape or win < 3:
        return 0.0
    score, _ = structural_similarity(a, b, full=True, win_size=win)
    return float(score)


class KnowledgeGraph:
    """`statistic_priors/knowledge_graph.json`, node-link format, read without
    networkx so the habitat env needs nothing new. Edge weights are P(room |
    object) as fractions; the reference reports them as percentages."""

    def __init__(self, links: Dict[str, Dict[str, float]], nodes: set) -> None:
        self._links = links
        self._nodes = nodes

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        with open(path) as f:
            g = json.load(f)
        links: Dict[str, Dict[str, float]] = {}
        for e in g.get("links", []):
            links.setdefault(str(e["source"]), {})[str(e["target"])] = float(e.get("weight", 0.0))
        nodes = {str(n["id"]) for n in g.get("nodes", [])}
        return cls(links, nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes

    def has_edge(self, u: str, v: str) -> bool:
        return v in self._links.get(u, {})

    def weight(self, u: str, v: str) -> float:
        return self._links[u][v]


class AscentLLMPlanner:
    def __init__(
        self,
        llm: Optional[Callable[[str], str]],
        knowledge_graph: Optional[KnowledgeGraph],
        floor_prior: Optional[dict],
        nearby_distance: float = 3.0,
        topk: int = 3,
        multi_floor: bool = False,
        stats: Optional[dict] = None,
    ) -> None:
        """`llm(prompt) -> str` must return "-1" on any failure, as ASCENT's
        `Qwen2_5Client.chat` does; the planner treats "-1" as "keep the value
        ranking" (`llm_planner.py:303-304`)."""
        self._llm = llm
        self.kg = knowledge_graph
        # {category: {str(total_floors): {str(floor): pct}}}, the output shape
        # of the reference's `get_floor_probabilities` (`llm_planner.py:392-429`),
        # produced once from the xlsx by `scripts/make_priors.py`.
        self.floor_prior = floor_prior or {}
        self.nearby_distance = float(nearby_distance)
        self.topk = int(topk)
        self.multi_floor = bool(multi_floor)
        self.stats = stats if stats is not None else {}
        self.reset()

    def reset(self) -> None:
        # `Ascent_LLM_Planner.reset` (`llm_planner.py:45-56`)
        self._force_frontier = np.zeros(2)
        self.frontier_step_list: List[int] = []
        self.vlm_response = ""
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self.multi_floor_ask_step = 0
        self.frontier_rgb_list: list = []
        self.floor_num = 1
        # Shared with the stair approach (`ascent_policy.py:1015-1042`).
        self.frontier_stick_step = 0
        self.last_frontier_distance = 0.0

    # ------------------------------------------------------------- the entry

    def get_best_frontier(
        self,
        robot_xy: np.ndarray,
        obstacle_map,
        value_map,
        object_map,
        obstacle_map_list: list,
        object_map_list: list,
        frontiers: np.ndarray,
        target: str,
        cur_floor_index: int,
        num_steps: int,
    ) -> Tuple[Optional[np.ndarray], float]:
        """`_get_best_frontier_with_llm` (`llm_planner.py:58-127`)."""
        frontiers = np.asarray(frontiers, dtype=float).reshape(-1, 2)
        # 0. One frontier: go there. No sorting, no sticky rule, no
        #    `_last_frontier` update (:82-86).
        if len(frontiers) == 1:
            return frontiers[0], 1.0

        sorted_pts, sorted_values = self._sort_frontiers_by_value(obstacle_map, value_map, frontiers)
        best_frontier, best_value = self._try_force_frontier(sorted_pts, sorted_values)

        if best_frontier is None and obstacle_map._finish_first_explore:
            nb, nv, activated = self._try_nearby_frontier(sorted_pts, sorted_values, robot_xy)
            if activated:
                obstacle_map._neighbor_search = True
                best_frontier, best_value = nb, nv
            else:
                obstacle_map._finish_first_explore = False
                obstacle_map._neighbor_search = False

        if best_frontier is None:
            best_frontier, best_value = self._decide_frontier_with_llm(
                obstacle_map, object_map, sorted_pts, sorted_values, target,
                cur_floor_index, num_steps, obstacle_map_list, object_map_list,
            )
        if best_frontier is None:
            # Every frontier was disabled. The reference would crash inside
            # `_handle_frontier_stick_and_disable`; there is nothing to pick.
            return None, 0.0

        self._handle_frontier_stick_and_disable(best_frontier, robot_xy, obstacle_map)

        self._last_value = best_value
        self._last_frontier = best_frontier
        if not obstacle_map._finish_first_explore:
            obstacle_map._finish_first_explore = True
            self._force_frontier = np.array(best_frontier, dtype=float).copy()
        return best_frontier, best_value

    # ---------------------------------------------------------- the pieces

    def _sort_frontiers_by_value(self, obstacle_map, value_map, frontiers):
        """`llm_planner.py:131-149`: value-sorted, disabled ones dropped."""
        raw_pts, raw_vals = value_map.sort_waypoints(frontiers, 0.5)
        pairs = [(pt, val) for pt, val in zip(raw_pts, raw_vals)
                 if tuple(pt) not in obstacle_map._disabled_frontiers]
        if not pairs:
            return np.zeros((0, 2)), []
        return np.array([p for p, _ in pairs]), [v for _, v in pairs]

    def _try_force_frontier(self, sorted_pts, sorted_values):
        """`llm_planner.py:157-164`: exact match on the latched frontier."""
        if np.any(self._force_frontier):
            for i, frontier in enumerate(sorted_pts):
                if np.array_equal(frontier, self._force_frontier):
                    return frontier, sorted_values[i]
        return None, None

    def _try_nearby_frontier(self, sorted_pts, sorted_values, robot_xy):
        """`llm_planner.py:166-179`: the closest one within `nearby_distance`."""
        distances = [np.linalg.norm(f - robot_xy) for f in sorted_pts]
        close = [(i, f, d) for i, (f, d) in enumerate(zip(sorted_pts, distances))
                 if d <= self.nearby_distance]
        if close:
            i = min(close, key=lambda x: x[2])[0]
            return sorted_pts[i], sorted_values[i], True
        return None, None, False

    def _decide_frontier_with_llm(self, obstacle_map, object_map, sorted_pts, sorted_values,
                                  target, cur_floor_index, num_steps,
                                  obstacle_map_list, object_map_list):
        """`llm_planner.py:181-237`."""
        if len(sorted_pts) == 0:
            return None, 0.0
        if len(sorted_pts) == 1:
            self._last_value = sorted_values[0]
            self._last_frontier = sorted_pts[0]
            return sorted_pts[0], sorted_values[0]

        # top-k candidates, each with the frame it was extracted from, de-duped
        # by SSIM > 0.75 on the greyscale frame (:196-210)
        self.frontier_step_list = []
        frontier_index_list: List[int] = []
        seen_gray: List[np.ndarray] = []
        for idx, frontier in enumerate(sorted_pts[: self.topk]):
            # A frontier the map never projected into a frame has no entry in
            # `frontier_visualization_info`; the reference tolerates the gap
            # with `except (IndexError, KeyError)` (`llm_planner.py:444-446`),
            # so it stays a candidate, without a picture.
            try:
                floor_num_steps, image_rgb = obstacle_map.extract_frontiers_with_image(frontier)
            except (KeyError, IndexError, TypeError, ValueError):
                floor_num_steps, image_rgb = obstacle_map._floor_num_steps, None
            if image_rgb is not None:
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
                if any(_ssim(g, image_gray) > 0.75 for g in seen_gray):
                    continue
                seen_gray.append(image_gray)
            self.frontier_step_list.append(floor_num_steps)
            frontier_index_list.append(idx)
            if len(self.frontier_step_list) == self.topk:
                break
        if not frontier_index_list:
            frontier_index_list = [0]
            self.frontier_step_list = [obstacle_map._floor_num_steps]

        target_object_category = str(target).split("|")[0]
        best_idx = 0
        if (self.multi_floor and self.floor_num > 1
                and num_steps - self.multi_floor_ask_step >= MULTI_FLOOR_ASK_STEP_THRESHOLD
                and obstacle_map._floor_num_steps >= FLOOR_EXP_STEP_THRESHOLD):
            self.multi_floor_ask_step = num_steps
            prompt = self._prepare_multiple_floor_prompt(
                target_object_category, cur_floor_index, obstacle_map_list, object_map_list)
            response = self._chat(prompt)
            if response == "-1":
                best_idx = self.llm_analyze_single_floor(
                    target_object_category, frontier_index_list, obstacle_map, object_map)
            else:
                current_floor = cur_floor_index + 1
                decision = self._extract_multiple_floor_decision(response, cur_floor_index)
                if decision > current_floor:
                    return sorted_pts[0], GO_UP
                if decision < current_floor:
                    return sorted_pts[0], GO_DOWN
                best_idx = self.llm_analyze_single_floor(
                    target_object_category, frontier_index_list, obstacle_map, object_map)
        else:
            best_idx = self.llm_analyze_single_floor(
                target_object_category, frontier_index_list, obstacle_map, object_map)
        return sorted_pts[best_idx], sorted_values[best_idx]

    def _handle_frontier_stick_and_disable(self, best, robot_xy, obstacle_map) -> None:
        """`llm_planner.py:256-288`, verbatim including the `_neighbor_search`
        term and the repeated-non-consecutive-selection disable."""
        if np.array_equal(self._last_frontier, best):
            if self.frontier_stick_step == 0:
                self.last_frontier_distance = float(np.linalg.norm(best - robot_xy))
                self.frontier_stick_step += 1
            else:
                current_distance = float(np.linalg.norm(best - robot_xy))
                if (abs(self.last_frontier_distance - current_distance) > STICKY_FRONTIER_DISTANCE_THRESHOLD
                        and not obstacle_map._neighbor_search):
                    self.frontier_stick_step = 0
                    self.last_frontier_distance = current_distance
                elif self.frontier_stick_step >= STICKY_FRONTIER_STEP_THRESHOLD:
                    obstacle_map._disabled_frontiers.add(tuple(best))
                    self.stats["frontier_disabled_stuck"] = self.stats.get("frontier_disabled_stuck", 0) + 1
                    self.frontier_stick_step = 0
                else:
                    self.frontier_stick_step += 1
        else:
            self.frontier_stick_step = 0
            self.last_frontier_distance = 0.0
            if tuple(best) in obstacle_map._best_frontier_selection_count:
                self._force_frontier = np.array(best, dtype=float).copy()

        key = tuple(best)
        obstacle_map._best_frontier_selection_count.setdefault(key, 0)
        if not np.array_equal(self._last_frontier, best):
            obstacle_map._best_frontier_selection_count[key] += 1
            if obstacle_map._best_frontier_selection_count[key] >= REPEATED_SELECTION_THRESHOLD:
                obstacle_map._disabled_frontiers.add(key)
                self.stats["frontier_disabled_repeated"] = self.stats.get("frontier_disabled_repeated", 0) + 1

    # ------------------------------------------------------------- the LLM

    def _chat(self, prompt: str) -> str:
        if self._llm is None:
            return "-1"
        self.stats["llm_calls"] = self.stats.get("llm_calls", 0) + 1
        try:
            out = self._llm(prompt)
        except Exception as exc:  # noqa: BLE001 - "-1" is the reference's failure value
            log.warning("LLM call failed, keeping the value ranking: %s", exc)
            self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
            return "-1"
        if out is None or out == "-1":
            self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
            return "-1"
        return str(out)

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t

    def llm_analyze_single_floor(self, target_object_category, frontier_index_list,
                                 obstacle_map, object_map) -> int:
        """`llm_planner.py:290-359`: exact parsing, 0 on any failure."""
        prompt = self._prepare_single_floor_prompt(target_object_category, obstacle_map, object_map)
        response = self._chat(prompt)
        idx = 0
        if response != "-1":
            try:
                cleaned = self._strip_fences(response).replace("\n", "").replace("\r", "")
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                log.warning("Failed to parse JSON response: %s", response[:200])
                self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
            else:
                index = d.get("Index", "N/A") if isinstance(d, dict) else "N/A"
                if index == "N/A":
                    self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
                else:
                    reason = d.get("Reason", "N/A")
                    if reason != "N/A":
                        self.vlm_response = f"## Single-floor Prompt:\nArea Index: {index}. Reason: {reason}"
                    try:
                        index_int = int(index)
                    except (TypeError, ValueError):
                        self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
                    else:
                        if 1 <= index_int <= len(frontier_index_list):
                            idx = index_int - 1
                            if idx != 0:
                                self.stats["rank_overrides"] = self.stats.get("rank_overrides", 0) + 1
                        else:
                            self.stats["rank_errors"] = self.stats.get("rank_errors", 0) + 1
        return frontier_index_list[idx]

    # ----------------------------------------------------------- the priors

    def get_room_probabilities(self, target_object_category: str) -> Dict[str, float]:
        """`llm_planner.py:361-390`."""
        if self.kg is None:
            return {}
        cats = [target_object_category] + SYNONYMS.get(target_object_category, [])
        if not any(c in self.kg for c in cats):
            return {}
        out: Dict[str, float] = {}
        for room in REFERENCE_ROOMS:
            for c in cats:
                if self.kg.has_edge(c, room):
                    out[room] = round(self.kg.weight(c, room) * 100, 1)
                    break
            else:
                out[room] = 0.0
        return out

    def get_floor_probabilities(self, target_object_category: str, total_floor: int) -> Dict[int, float]:
        """`llm_planner.py:392-429`, on the pre-tabulated json."""
        table = self.floor_prior.get(target_object_category)
        if not table:
            return {i: 0.0 for i in range(1, total_floor + 1)}
        available = sorted(int(k) for k in table)
        max_possible = max(available) if available else 0
        n = total_floor if total_floor <= max_possible else max_possible
        col = table.get(str(n), {})
        return {y: float(col.get(str(y), 0.0)) for y in range(1, n + 1)}

    # ---------------------------------------------------------- the prompts

    def _prepare_single_floor_prompt(self, target_object_category, obstacle_map, object_map) -> str:
        """`llm_planner.py:431-511`, verbatim."""
        area_descriptions = []
        self.frontier_rgb_list = []
        for i, step in enumerate(self.frontier_step_list):
            try:
                room = object_map.each_step_rooms[step] or "unknown room"
                objects = object_map.each_step_objects[step] or "no visible objects"
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                self.frontier_rgb_list.append(obstacle_map._each_step_rgb[step])
                area_descriptions.append({"area_id": i + 1, "room": room, "objects": objects})
            except (IndexError, KeyError) as e:
                log.warning("Error accessing room or objects for step %s: %s", step, e)
                continue
        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ",\n".join(
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%' for room, prob in sorted_rooms)
        area_entries = ",\n".join(
            f'{INDENT_L2}"Area {d["area_id"]}": "a {d["room"].replace("_", " ")} containing objects: {d["objects"]}"'
            for d in area_descriptions)
        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bathroom": 90.0%,\n'
            f'{INDENT_L2}"Bedroom": 10.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{INDENT_L2}"Area 1": "a bathroom containing objects: shower, towel",\n'
            f'{INDENT_L2}"Area 2": "a bedroom containing objects: bed, nightstand",\n'
            f'{INDENT_L2}"Area 3": "a garage containing objects: car",\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{area_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        return "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input,
        ])

    def _prepare_multiple_floor_prompt(self, target_object_category, cur_floor_index,
                                       obstacle_map_list, object_map_list) -> str:
        """`llm_planner.py:513-615`, verbatim."""
        current_floor = cur_floor_index + 1
        total_floors = self.floor_num
        floor_probs = self.get_floor_probabilities(target_object_category, total_floors)
        floor_prob_entries = ",\n".join(
            f'{INDENT_L2}"Floor {floor}": {prob:.1f}%' for floor, prob in floor_probs.items())
        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ",\n".join(
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%' for room, prob in sorted_rooms)
        floor_descriptions = []
        for floor in range(total_floors):
            try:
                rooms = object_map_list[floor].this_floor_rooms or {"unknown rooms"}
                objects = object_map_list[floor].this_floor_objects or {"unknown objects"}
                floor_descriptions.append({
                    "floor_id": floor + 1,
                    "status": "Current floor" if floor + 1 == current_floor else "Other floor",
                    "fully_explored": obstacle_map_list[floor]._this_floor_explored,
                    "room": ", ".join(rooms),
                    "objects": ", ".join(objects),
                })
            except Exception as e:  # noqa: BLE001 - as the reference
                log.error("Error describing floor %s: %s", floor, e)
                continue
        floor_entries = ",\n".join(
            f'{INDENT_L2}"Floor {d["floor_id"]}": "{d["status"]}. There are room types: {d["room"]}, containing objects: {d["objects"]}'
            + ('. You do not need to explore this floor again"' if d.get("fully_explored", False) else '"')
            for d in floor_descriptions)
        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{INDENT_L2}"Floor 1": 10.0%,\n'
            f'{INDENT_L2}"Floor 2": 10.0%,\n'
            f'{INDENT_L2}"Floor 3": 80.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bedroom": 80.0%,\n'
            f'{INDENT_L2}"Living room": 15.0%,\n'
            f'{INDENT_L2}"Bathroom": 5.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{INDENT_L2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",\n'
            f'{INDENT_L2}"Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel. You do not need to explore this floor again",\n'
            f'{INDENT_L2}"Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{floor_prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{floor_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        return "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely to be on the Floor 3, and the room types and object types on the Floor 1 and Floor 2 are not directly related to the target object bed, especially it do not need to explore Floor 2 again."}',
            actual_input,
        ])

    def _extract_multiple_floor_decision(self, response: str, cur_floor_index: int) -> int:
        """`llm_planner.py:621-661`: the chosen floor (1-based), or the current one."""
        current_floor = cur_floor_index + 1
        try:
            d = json.loads(self._strip_fences(response).replace("\n", "").replace("\r", ""))
            target_floor = int(d.get("Index", -1))
            reason = d.get("Reason", "N/A")
            if reason != "N/A":
                self.vlm_response = f"## Multi-floor Prompt:\nFloor Index: {target_floor}. Reason: {reason}"
            if target_floor <= 0 or target_floor > self.floor_num:
                return current_floor
            return target_floor
        except Exception as e:  # noqa: BLE001 - as the reference
            log.error("Error extracting floor decision: %s", e)
        return current_floor
