"""Looking for stairs: the periodic pitch-down, and what it accumulates.

Five methods out of `nav_agent.py`. They are one thing -- evidence about where a
staircase is, gathered by pointing the camera at the floor every so often -- and
they were sitting between the climb and the approach.

`_down_look` runs on EVERY step of every episode (`nav_agent._act_inner`), which
is why this is the one piece of the split that the pre-existing trajectory locks
already covered: on a single-floor run `down_look_every` is 0 and the method
returns at its second guard, but the call happens and the guard order is real.

`_update_value_map` deliberately stayed behind. It is not stair code -- it is the
semantic value map -- and it co-owns `_last_itm`, `_approach_itm_max` and
`_approach_itm_n` with `_act_inner`, `_start_approach` and `_recheck_rejects`.
Moving it would split the approach's telemetry across two objects to save a line
count.

The pass-through properties are the same device `climb.py` uses: named exactly as
on `NavAgent`, so the bodies moved unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE
from .state import State


class StairSense:
    """The periodic look down, and the stair evidence it accumulates."""

    def __init__(self, nav) -> None:
        self.nav = nav
        self.reset()

    def reset(self) -> None:
        self._pitch_ticks = 0
        self._last_down_look_step = -(10 ** 9)

    # -------------------------------------------------- the agent's own state

    @property
    def cfg(self):
        return self.nav.cfg

    @property
    def stats(self):
        return self.nav.stats

    @property
    def state(self):
        return self.nav.state

    @property
    def step_count(self):
        return self.nav.step_count

    @property
    def costmap(self):
        return self.nav.costmap

    @property
    def floor_layer(self):
        return self.nav.floor_layer

    @property
    def floors(self):
        return self.nav.floors

    @property
    def profiler(self):
        return self.nav.profiler

    @property
    def stair_detector(self):
        return self.nav.stair_detector

    @property
    def stair_segmenter(self):
        return self.nav.stair_segmenter

    @property
    def _down_look_every(self):
        return self.nav._down_look_every

    def _seg_stair_mask(self, frame: FrameData) -> Optional[np.ndarray]:
        if self.stair_segmenter is None:
            return None
        with self.profiler.timeit("stair_seg"):
            return self.stair_segmenter.stair_mask(frame)
    def _accumulate_down_stairs(self, frame: FrameData, layer) -> None:
        if self.stair_detector is None:
            return
        with self.profiler.timeit("stairs"):
            self.stair_detector.accumulate(
                frame, layer, None, self._seg_stair_mask(frame)
            )
    def _down_look_here(self, frame: FrameData) -> bool:
        """Is this a place worth pitching the camera down at?

        Only asked when `agent.down_look_near_stairs_m` is set. The stair map
        -- ASCENT's, pasted at episode start, plus anything this run's detector
        has confirmed -- is the evidence; with no map there is nothing to
        answer with and the look-down keeps its unconditional behaviour,
        because discovering an unknown staircase is what it is for.
        """
        near_m = float(getattr(self.cfg.agent, "down_look_near_stairs_m", 0.0) or 0.0)
        if near_m <= 0.0:
            return True
        mask = getattr(self.costmap, "stair_mask", None)
        if mask is None or not mask.any():
            return True
        rc = self.costmap.world_to_grid(frame.camera_position[list(PLANE)])
        radius = max(1, int(round(near_m / self.costmap.resolution)))
        r0, r1 = max(0, rc[0] - radius), min(mask.shape[0], rc[0] + radius + 1)
        c0, c1 = max(0, rc[1] - radius), min(mask.shape[1], rc[1] + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        yy, xx = np.ogrid[r0:r1, c0:c1]
        disk = (yy - rc[0]) ** 2 + (xx - rc[1]) ** 2 <= radius ** 2
        return bool((mask[r0:r1, c0:c1] & disk).any())
    def _down_look(self, frame: FrameData, layer, off_map: bool) -> Optional[str]:
        if self._pitch_ticks > 0:
            if not off_map:
                self._accumulate_down_stairs(frame, layer)
            self._pitch_ticks -= 1
            return "look_up"
        if self._down_look_every <= 0:
            return None
        # The look-down hunts for a staircase down; on a scene the estimator
        # knows to have one level there is nothing to find, and injecting the
        # look_down/look_up pair would perturb an otherwise single-floor run the
        # floor group should leave untouched. Gated by the same rule as the
        # undirected switch, and safe for the same reason: a schema-v2 prior map
        # seeds every storey on load, so a real multi-floor run has >= 2 levels.
        if (
            bool(getattr(self.cfg.floor, "switch_requires_second_level", False))
            and len(getattr(self.floors.estimator, "levels", {0: 0.0})) < 2
        ):
            return None
        if self.state not in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            return None
        if self.step_count - self._last_down_look_step < self._down_look_every:
            return None
        if not self._down_look_here(frame):
            self.stats["down_look_skipped_far"] = (
                self.stats.get("down_look_skipped_far", 0) + 1)
            return None
        self._last_down_look_step = self.step_count
        self._pitch_ticks += 1
        self.stats["down_look"] = self.stats.get("down_look", 0) + 1
        return "look_down"
    def _on_a_staircase(self, agent_xy: np.ndarray) -> bool:
        detector = self.stair_detector
        layer = self.floor_layer
        if detector is None or layer.up_stair_hits is None:
            return False
        mask = (
            (layer.up_stair_hits >= detector.min_hits)
            | (layer.down_stair_hits >= detector.min_hits)
        )
        if layer.disabled_stair is not None:
            mask &= ~layer.disabled_stair
        rc = layer.costmap.world_to_grid(agent_xy)
        radius = max(
            1,
            int(round(float(getattr(self.cfg.agent, "stair_exit_m", 0.5)) /
                      layer.costmap.resolution)),
        )
        r0, r1 = max(0, rc[0] - radius), min(mask.shape[0], rc[0] + radius + 1)
        c0, c1 = max(0, rc[1] - radius), min(mask.shape[1], rc[1] + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        yy, xx = np.ogrid[r0:r1, c0:c1]
        disk = (yy - rc[0]) ** 2 + (xx - rc[1]) ** 2 <= radius ** 2
        return bool((mask[r0:r1, c0:c1] & disk).any())
