"""What a frontier's own view showed, at the moment it first appeared.

ASCENT describes a frontier to the LLM using models run on **the frame in which
that frontier was first seen** -- RAM++ for object tags and Places365 for the
room -- keyed by the step at which it appeared:

* `obstacle_map.py:421-431` stores the RGB for the step
  (`_each_step_rgb[floor_num_steps]`) and binds every genuinely new frontier to
  that step (`frontier_visualization_info[tuple(frontier)]`).
* `map_controller.py:800-830` (`_update_current_step_scene_info`) runs RAM++ and
  Places365 on that step's RGB into `each_step_objects[step]` /
  `each_step_rooms[step]`.
* `llm_planner.py:418-419, 445` reads them back and formats
  `"a bedroom containing objects: bed, nightstand"`.

OSG's `describe_area` instead queries the accumulated scene graph by position.
That answers a different question. A frontier is a frontier precisely because
what lies beyond it is unknown, so "objects already mapped near this point" and
"what the agent actually saw when looking that way" can disagree completely --
the graph answer describes the room the agent is standing in, not the opening.

This class is the missing binding. It is deliberately model-agnostic: it stores
whatever labels it is handed, so the object tagger can be OSG's existing YOLOE
detections or RAM++ without touching the frontier logic.

**Matching.** Frontiers are re-extracted from scratch every round and their
centroids drift as the costmap grows, so identity cannot be an exact
coordinate. A frontier within `match_radius_m` of one already bound is treated
as the same opening and keeps its ORIGINAL step -- which is the point: the
description must come from the view that first revealed it, not from wherever
the agent happens to be standing now.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class FrontierSemantics:
    """Per-step scene semantics, and which step each frontier was first seen at."""

    def __init__(
        self,
        match_radius_m: float = 1.0,
        max_objects: int = 8,
        fov_rad: float = np.radians(79.0),
        max_range_m: float = 5.0,
    ) -> None:
        self.match_radius_m = float(match_radius_m)
        self.max_objects = int(max_objects)
        self.fov_rad = float(fov_rad)
        self.max_range_m = float(max_range_m)
        # step -> (room, [object labels]), as seen from that step's frame.
        self._per_step: Dict[int, Tuple[str, List[str]]] = {}
        # step -> (camera xy, forward unit vector), for the visibility test.
        self._pose: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        # (xy, step) for every frontier opening bound so far this episode.
        self._bound: List[Tuple[np.ndarray, int]] = []

    def reset(self) -> None:
        self._per_step.clear()
        self._pose.clear()
        self._bound.clear()

    # ------------------------------------------------------------- recording

    def observe(
        self,
        step: int,
        room: Optional[str],
        objects: Iterable[str],
        camera_xy: Optional[np.ndarray] = None,
        heading_xy: Optional[np.ndarray] = None,
    ) -> None:
        """Record what the models saw in this step's frame, and from where."""
        seen, ordered = set(), []
        for label in objects:
            if label and label not in seen:
                seen.add(label)
                ordered.append(str(label))
            if len(ordered) >= self.max_objects:
                break
        self._per_step[int(step)] = (str(room) if room else "unknown room", ordered)
        if camera_xy is not None and heading_xy is not None:
            self._pose[int(step)] = (
                np.asarray(camera_xy, dtype=float).copy(),
                np.asarray(heading_xy, dtype=float).copy(),
            )

    def _saw(self, step: int, xy: np.ndarray) -> bool:
        """Was `xy` inside this step's camera frustum, on the ground plane?"""
        pose = self._pose.get(step)
        if pose is None:
            return False
        cam, fwd = pose
        d = xy - cam
        dist = float(np.linalg.norm(d))
        if dist < 1e-6 or dist > self.max_range_m:
            return False
        cos = float(np.dot(d / dist, fwd))
        return cos >= float(np.cos(self.fov_rad / 2.0))

    def bind(self, frontiers: Sequence, step: int) -> int:
        """Bind frontiers not seen before to the latest observed frame.

        Returns how many were new.

        Each frontier binds to the most recent observed frame that actually had
        it in view, falling back to the most recent frame of all.

        The visibility test is what makes this ASCENT's mechanism rather than a
        rough analogue. ASCENT stores the RGB inside the function that detects
        new frontiers from that frame (obstacle_map.py:421-431), so the frontier
        is necessarily within the frame it is described by. OSG runs keyframes
        on a movement threshold and frontier extraction on a step interval, so
        binding to "the latest keyframe" would instead describe whatever the
        agent happened to be facing -- measured on a 4-episode smoke, that gave
        three different frontiers the identical description "a garage containing
        objects: bench, chair, lamp, sofa", which tells the ranker nothing.

        Frontiers stay unbound while nothing has been observed at all; that
        falls back to the scene graph rather than asserting an empty room.
        """
        if not self._per_step:
            return 0
        earlier = sorted((k for k in self._per_step if k <= int(step)), reverse=True)
        if not earlier:
            return 0
        new = 0
        for f in frontiers:
            xy = np.asarray(f.centroid_xy, dtype=float)
            if self._nearest(xy) is not None:
                continue
            resolved = next((k for k in earlier if self._saw(k, xy)), earlier[0])
            self._bound.append((xy.copy(), resolved))
            new += 1
        return new

    # -------------------------------------------------------------- reading

    def describe(self, frontier) -> Optional[Tuple[str, List[str]]]:
        """`(room, objects)` from the frame that first revealed this frontier."""
        step = self._nearest(np.asarray(frontier.centroid_xy, dtype=float))
        return self._per_step.get(step) if step is not None else None

    def _nearest(self, xy: np.ndarray) -> Optional[int]:
        best, best_d = None, self.match_radius_m
        for bound_xy, step in self._bound:
            d = float(np.linalg.norm(bound_xy - xy))
            if d <= best_d:
                best, best_d = step, d
        return best

    def stats(self) -> Dict[str, int]:
        return {
            "frontier_sem_steps": len(self._per_step),
            "frontier_sem_bound": len(self._bound),
        }
