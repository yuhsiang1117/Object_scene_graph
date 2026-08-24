"""Which storey the agent is on, and when to leave it.

Not part of the dynamic-scene line. This is the multi-floor work
(docs/MULTI_FLOOR.md), and it is here rather than in `nav_agent.py` because with
`floor.enabled=false` -- which is every YCB run and every default -- none of it
does anything except log, and 350 inert lines interleaved with `act()` made the
path an episode actually takes impossible to read off the file.

Nothing below is new. It is the same code, with the seams made explicit: the
policy answers questions and returns decisions, and `NavAgent` remains the only
thing that writes FSM state.

The one design point worth restating is why `costmap` lives here at all. A
single shared `Costmap2D` is what confined the pipeline to one floor: it bands
points by height above a `floor_y` latched on frame 1, so once the agent climbs,
every observation falls outside the band and is dropped -- no cells, no
frontiers, nothing to explore. `FloorStack` gives each storey its own grid, and
`costmap` is the seam: every consumer (planner, frontier extractor, room
segmenter, viewpoint planner, controller, the visualizers) still receives a
plain 2D `Costmap2D` and needs no knowledge that other floors exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..graph.priors import floor_target_evidence
from ..mapping.costmap import PLANE, Costmap2D
from ..mapping.floor_stack import FloorStack
from ..mapping.floors import FloorEstimator
from ..mapping.portals import FloorSwitchPolicy, find_portals
from ..mapping.stairs import apply_stair_mask, detect_stairs, stair_tracks


@dataclass
class PortalGoal:
    """A decision to leave this storey: where to drive, and at what height.

    The portal is only a heading -- the navmesh walks the actual stairs, and the
    floor estimator commits the new storey once the agent settles there, at
    which point FloorStack swaps in that floor's map.
    """

    goal_xy: np.ndarray
    target_y: float
    deadline_steps: int


class FloorPolicy:
    """Owns the per-storey maps, the height estimate, and the switch decision."""

    def __init__(self, cfg, stats: dict) -> None:
        self.cfg = cfg
        self.stats = stats
        fcfg = cfg.floor
        self._stairs_on = bool(fcfg.stairs)
        self._cross_floor_on = bool(fcfg.cross_floor)
        self.stack = FloorStack(
            resolution_m=cfg.mapping.resolution_m,
            room_seg_kwargs=dict(
                min_room_radius_m=cfg.scene_graph.room_min_radius_m,
                door_width_m=cfg.scene_graph.room_door_width_m,
            ),
            # The height layer is 4x the grid, so only pay for it when the
            # stair detector will actually read it. Cross-floor exploration
            # needs it to see portals, so it implies track_height too.
            track_height=self._stairs_on or self._cross_floor_on,
        )
        # Which storey the agent is on (docs/MULTI_FLOOR.md). Constructed
        # unconditionally so the estimate is always logged; whether it FEEDS
        # the costmap is gated by floor.enabled / floor.estimate_only.
        self.estimator = FloorEstimator(
            camera_height=cfg.agent.camera_height,
            level_tol_m=fcfg.level_tol_m,
            merge_m=fcfg.merge_m,
            new_level_m=fcfg.new_level_m,
            min_dwell_steps=fcfg.min_dwell_steps,
            min_horizontal_run_m=fcfg.min_horizontal_run_m,
        )
        self.switch_policy = (
            FloorSwitchPolicy(
                max_steps=cfg.agent.max_steps,
                near_frontier_m=fcfg.near_frontier_m,
                min_interval_steps=fcfg.switch_min_interval,
                no_switch_before=fcfg.no_switch_before,
                no_switch_after_frac=fcfg.no_switch_after_frac,
                use_target_evidence=fcfg.use_target_evidence,
                early_switch_step=fcfg.early_switch_step,
                min_objects_to_judge=fcfg.min_objects_to_judge,
                strong_evidence=fcfg.strong_evidence,
                evidence_patience_steps=fcfg.evidence_patience_steps,
            )
            if fcfg.cross_floor else None
        )
        self.reset()

    def reset(self) -> None:
        self._floor_y: Optional[float] = None
        # (step, floor_id, floor_height) on every committed floor change, plus
        # the first step. Surfaced per episode by eval/runner.py.
        self.floor_log: list = []
        self.floor_y_drift = 0.0
        self.stair_regions: list = []
        self.portal_log: list = []
        # Is a portal being driven to right now? The give-up net and the navmesh
        # height key both have to know: a switchback staircase barely moves in
        # (x, z) while climbing fine, and a portal goal is on ANOTHER storey, so
        # its height must not be discarded.
        self.pursuing = False
        self._portal_start_y = 0.0
        self._portal_step = 0
        self.estimator.reset()
        self.stack.reset()

    # ------------------------------------------------------------- the seam

    @property
    def costmap(self) -> Costmap2D:
        """The occupancy map of the storey the agent is on."""
        return self.stack.costmap

    @property
    def layer(self):
        return self.stack.current

    @property
    def current_id(self) -> int:
        return self.stack.current_id

    @property
    def room_labels(self) -> Optional[np.ndarray]:
        return self.stack.current.room_labels

    @room_labels.setter
    def room_labels(self, labels: Optional[np.ndarray]) -> None:
        self.stack.current.room_labels = labels

    # `levels` and `transitions` are what eval/runner.py records per episode.
    @property
    def levels(self):
        return self.estimator.levels

    @property
    def transitions(self):
        return self.estimator.transitions

    @property
    def on_stairs(self) -> bool:
        return self.estimator.on_stairs

    def height_of(self, floor_id: int) -> float:
        return self.estimator.height_of(floor_id)

    # ---------------------------------------------------------- every step

    def observe(self, frame, step: int) -> float:
        """Track the storey, and return the height to band the costmap at.

        With floor.estimate_only (the default) this only LOGS -- the costmap
        keeps using the latched _floor_y, so the estimator can be validated
        against the per-scene navmesh ground truth (scripts/scene_floors.py)
        before behaviour depends on it.
        """
        if self._floor_y is None:
            self._floor_y = float(frame.camera_position[1] - self.cfg.agent.camera_height)

        prev_floor = self.estimator.current
        floor_id = self.estimator.update(
            float(frame.camera_position[1]), step,
            xy=frame.camera_position[list(PLANE)],
        )
        if floor_id != prev_floor:
            self.end_pursuit("arrived")
        if floor_id != prev_floor or not self.floor_log:
            self.floor_log.append(
                (step, int(floor_id),
                 round(float(frame.camera_position[1]) - self.cfg.agent.camera_height, 3))
            )
        fcfg = self.cfg.floor
        floor_y = self._floor_y
        if fcfg.enabled and not fcfg.estimate_only:
            floor_y = self.estimator.height_of(floor_id)
            if fcfg.per_floor_costmap:
                # Point the stack at the agent's storey BEFORE mapping, so this
                # frame lands in that floor's own grid. While on stairs the
                # estimator freezes floor_id, so the treads keep going to the
                # floor being left rather than opening a phantom layer.
                self.stack.set_current(
                    floor_id, step=step,
                    agent_xy=frame.camera_position[list(PLANE)],
                )
        # How far the estimated floor height ever strays from the value the old
        # code latched on frame 1. On a single storey this should be ~0; larger
        # means the obstacle band is silently shifting and perturbing
        # trajectories that have nothing to do with multi-floor.
        self.floor_y_drift = max(
            self.floor_y_drift, abs(self.estimator.height_of(floor_id) - self._floor_y)
        )
        return floor_y

    def stairs_due(self, kf_count: int) -> bool:
        """Stair detection is per-keyframe and rate-limited; the caller owns the
        keyframe counter and the profiler, so it asks rather than being told."""
        return self._stairs_on and (
            kf_count % max(1, int(self.cfg.floor.stair_detect_every_kf)) == 1
        )

    def mark_stairs_traversable(self, object_layer) -> None:
        """Find steppable regions on the current storey and mark them
        traversable, so the staircase stops reading as a wall."""
        fc = self.cfg.floor
        regions = detect_stairs(
            self.costmap,
            climb_limit_m=fc.climb_limit_m,
            min_dh_m=fc.stair_min_dh_m,
            cell_m=fc.stair_cell_m,
            min_cells=fc.stair_min_cells,
            min_rise_m=fc.stair_min_rise_m,
            semantic_centers=stair_tracks(
                object_layer,
                min_obs=fc.stair_min_obs,
                min_evidence=fc.stair_min_evidence,
            ),
            require_semantic=fc.stair_require_semantic,
        )
        if not regions:
            return
        n = apply_stair_mask(self.costmap, regions, max_area_frac=fc.stair_max_area_frac)
        self.stair_regions = regions
        self.stats["stair_cells"] = self.stats.get("stair_cells", 0) + n
        self.stats["stair_regions"] = len(regions)
        self.stats["stair_regions_semantic"] = sum(1 for r in regions if r.semantic)
        self.stats["stair_max_rise_m"] = round(
            max(self.stats.get("stair_max_rise_m", 0.0), max(r.rise_m for r in regions)), 2
        )

    def pursuit_ok(self, frame, step: int, deadline: int) -> bool:
        """Should the agent keep driving to its portal instead of re-exploring?

        Held while it is still climbing (or descending) and the deadline has not
        passed. Vertical progress is the test, not horizontal: on a switchback
        staircase the (x, z) displacement over 15 steps can be small while the
        agent is making perfectly good progress, which is also why the ordinary
        give-up net must not judge a portal pursuit.
        """
        if not self.pursuing:
            return False
        if step > deadline:
            self.end_pursuit("deadline")
            return False
        climbed = abs(float(frame.camera_position[1]) - self._portal_start_y)
        if climbed >= self.cfg.floor.portal_progress_m or self.on_stairs:
            return True
        # Not moving vertically and not on stairs: the portal was unreachable or
        # the agent is stuck at the foot of it -- fall back to exploring.
        if step - self._portal_step > self.cfg.floor.portal_grace_steps:
            self.end_pursuit("no_vertical_progress")
            return False
        return True

    def end_pursuit(self, reason: str) -> None:
        self.pursuing = False
        self.stats[f"portal_end_{reason}"] = self.stats.get(f"portal_end_{reason}", 0) + 1

    def try_switch(
        self, frame, step: int, best_path_cost, scene_graph, target: str, reachable_fn
    ) -> Optional[PortalGoal]:
        """Head for another storey when this one has nothing near left.

        Returns the portal to drive to, or None to stay. The caller applies it:
        deciding to leave a floor is this policy's business, and moving the
        agent is the FSM's.
        """
        if self.switch_policy is None:
            return None
        evidence, n_objects = floor_target_evidence(
            scene_graph, self.stack.current_id, target
        )
        if not self.switch_policy.may_switch(
            step, best_path_cost, evidence=evidence, n_objects=n_objects,
            steps_on_floor=step - self.stack.current.first_step,
        ):
            return None

        floor_y = self.estimator.height_of(self.stack.current_id)
        portals = find_portals(
            self.costmap, floor_y,
            min_delta_m=self.cfg.floor.new_level_m,
            max_delta_m=self.cfg.floor.portal_max_delta_m,
            min_cells=self.cfg.floor.portal_min_cells,
        )
        self.stats["portals_seen"] = max(self.stats.get("portals_seen", 0), len(portals))
        if not portals:
            return None

        agent_xy = frame.camera_position[list(PLANE)]
        # Prefer a storey we have NOT searched, then the nearest. Nearest-only
        # let the agent bounce back onto a floor it had already given up on --
        # 4-5 transitions in some episodes, paying the travel cost each time.
        levels = self.estimator.levels

        def unvisited(p):
            return not any(
                abs(h - p.target_y) <= self.cfg.floor.level_tol_m for h in levels.values()
            )

        portals.sort(key=lambda p: (not unvisited(p),
                                    float(np.linalg.norm(p.centroid_xy - agent_xy))))
        portal = portals[0]
        if reachable_fn is not None and not reachable_fn(
            portal.centroid_xy, portal.target_y
        ):
            return None

        # The pursuit is held against same-floor frontier re-selection. While
        # the agent is on the stairs its floor id is frozen, so the costmap it
        # sees is still the floor BELOW -- and left alone, exploration picks a
        # frontier down there and walks the agent back down. Measured: three
        # episodes climbed ~1.6 m and turned around exactly this way.
        self.pursuing = True
        self._portal_start_y = float(frame.camera_position[1])
        self._portal_step = step
        self.switch_policy.note_switch(step)
        self.stats["floor_switch_attempts"] = self.stats.get("floor_switch_attempts", 0) + 1
        self.portal_log.append((
            step,
            [round(float(x), 2) for x in portal.centroid_xy],
            round(float(portal.delta_y), 2),
            portal.n_cells,
        ))
        return PortalGoal(
            goal_xy=portal.centroid_xy.copy(),
            target_y=portal.target_y,
            deadline_steps=self.cfg.floor.portal_deadline_steps,
        )

    def goal_floor_y(self, center: np.ndarray) -> Optional[float]:
        """Height to snap a 3D goal at, or None to keep the legacy behaviour of
        substituting the agent's own height.

        Snap at the goal's FLOOR, not at its ellipsoid centre: an object's
        centre sits 0.3-1.0 m above the ground, and near a mezzanine edge that
        offset is enough to snap onto the wrong storey.

        No clearance offset is added. The navmesh sits at floor height, so the
        floor height IS the right query -- and on a single floor it equals the
        agent's own standing height, which makes this a genuine no-op there.
        An earlier +0.1 m "clearance" was enough on its own to change the snap
        result and perturb single-floor trajectories.
        """
        fcfg = self.cfg.floor
        if not self.cfg.agent.navmesh_3d_goals:
            return None
        if not fcfg.enabled or not self.estimator.levels:
            return None
        return self.estimator.height_of(
            self.estimator.floor_of_height(float(center[1]))
        )

