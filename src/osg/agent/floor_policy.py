"""Which storey the agent is on, and when to leave it.

Not part of the dynamic-scene line. This is the multi-floor work
(docs/history/MULTI_FLOOR.md), and it is here rather than in `nav_agent.py` because with
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
from typing import Optional

import numpy as np

from ..graph.priors import floor_target_evidence
from ..mapping.costmap import PLANE, Costmap2D
from ..mapping.floor_stack import FloorStack
from ..mapping.floors import FloorEstimator
from ..mapping.portals import FloorSwitchPolicy, find_portals
from ..mapping.stairs import (StairRegion, apply_stair_mask, detect_stairs,
                             find_flights, mouth_xy, stair_tracks)
from ..planning.voronoi_planner import HybridVoronoiPlanner


def _parse_schedule(entries) -> dict:
    """`["120:1", "300:0"]` -> `{120: 1, 300: 0}`.

    Strings rather than a dict because Hydra's structured configs take a
    `List[str]` from the command line cleanly and an int-keyed dict does not.
    """
    schedule = {}
    for entry in entries or []:
        step, _, key = str(entry).partition(":")
        if not _:
            raise ValueError(
                f"floor.external_schedule entry {entry!r} is not 'step:key'")
        schedule[int(step)] = int(key)
    return schedule


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

    def __init__(self, cfg, stats: dict, value_map_factory=None) -> None:
        self.cfg = cfg
        self.stats = stats
        # Storeys the agent's own failed attempts have disproved. `NavAgent`
        # rebinds this to the SAME set it shares with `ExplorationStrategy`, so
        # the verdict reaches every channel that acts on it -- the floor LLM,
        # the candidate channel, the storey posterior and, since the 00800 toy
        # airplane returned to a disproved storey 51 steps after arriving,
        # the flight choice (`_drop_disproved`).
        self.disproved_floors: set = set()
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
            # The stair route's layer (`agent.climb_route`), paid for only then.
            track_top=bool(getattr(cfg.agent, "climb_route", False)),
            planner_factory=lambda: HybridVoronoiPlanner(
                collision_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
                goal_near_m=cfg.exploration.voronoi_goal_near_m,
                inflate_radius_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
            ),
            value_map_factory=value_map_factory,
        )
        # Which storey the agent is on (docs/history/MULTI_FLOOR.md). Constructed
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
        # Where `current` comes from. In external mode the estimator is still
        # CONSTRUCTED -- `_levels`, `height_of` and `floor_of_height` are read
        # all over the pipeline -- but it is never asked to decide: it is
        # written to by the switch instead. See FloorConfig.source.
        self.external = str(getattr(fcfg, "source", "estimator")) == "external"
        self._schedule = _parse_schedule(getattr(fcfg, "external_schedule", None))
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
                dwell_on_arrival=bool(getattr(fcfg, "dwell_on_arrival", False)),
            )
            if fcfg.cross_floor else None
        )
        self.reset()

    def reset(self) -> None:
        self._floor_y: Optional[float] = None
        # An external switch waiting to be applied. Queued rather than applied
        # where it arrives, because it lands mid-step from the env and the
        # costmap layer must not change underneath a control loop that has
        # already read `self.costmap`.
        self._requested_floor: Optional[int] = None
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
        # Where the current pursuit is headed, and the places a pursuit has
        # already failed from. Without the second, `find_portals` re-proposes the
        # same patch every selection round for the rest of the episode: measured
        # on 00821's cracker box, 55 switch attempts, the last 50 of them at one
        # 28-cell patch at (1.19, -7.96), for 0.17 m of ascent in 500 steps.
        self._pursuit_goal_xy = None
        self._failed_portals: list = []
        # The flight a pursuit is heading for, when the target came from the
        # height layer; the agent's climb carrot walks up its treads.
        self.pursuit_flight = None
        # Set by the agent for the duration of a climb. A half-landing is FLAT
        # ground between two flights, so standing on stair cells is not enough:
        # measured on 00808, the suppression fired on 7 frames of treads and the
        # level was still created on the landing in between.
        self.climbing = False
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

    def current(self):
        """Compatibility accessor used by the ASCENT policy facade."""
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

    def request_floor(self, key: int) -> None:
        """Someone outside the agent says it is on storey `key`.

        The operator's `/osg/floor` topic on the robot, or `external_schedule`
        in a simulated test. Applied at the top of the next `observe`, so one
        control step sees one storey.
        """
        self._requested_floor = int(key)

    def _apply_external(self, frame, step: int) -> int:
        """Make `key` the current storey, writing the same state a restored
        snapshot does.

        `graph/map_store.py:apply_map` already had to solve this exact problem
        -- point the stack at a storey the agent did not walk to -- and the
        answer is that the estimator and the stack are ONE state. Writing the
        stack alone leaves the estimator bootstrapping floor 0 on the next
        frame and silently detaching the key from the storey it names.
        """
        est = self.estimator
        if not est._levels:
            # Bootstrap, the same one `FloorEstimator.update` does on its first
            # frame ("the starting floor is floor 0"): in external mode that
            # method never runs, and every height consumer -- `height_of`, the
            # costmap band, `floor_of_height` -- needs floor 0 to exist before
            # the first switch arrives.
            est._levels[est.current] = float(
                frame.camera_position[1]) - float(self.cfg.agent.camera_height)
            est._samples[est.current] = [est._levels[est.current]]
            est._next_id = max(est._next_id, est.current + 1)
        key = self._requested_floor
        if key is None:
            key = self._schedule.get(int(step))
        self._requested_floor = None
        if key is None:
            return est.current
        key = int(key)
        if key not in est._levels:
            # The height this storey is at. On the robot the pose arriving here
            # is already lifted by `key * floor.virtual_storey_m`
            # (sim/ros2_env.py), so distinct keys get distinct heights and
            # everything keyed on height -- the costmap band, floor_of_height,
            # the prompt's floor ordering -- keeps working.
            height = float(frame.camera_position[1]) - float(self.cfg.agent.camera_height)
            est._levels[key] = height
            est._samples[key] = [height]
            est._next_id = max(est._next_id, key + 1)
        if key == est.current and key == self.stack.current_id:
            return key
        est.current = key
        # The switch is an assertion about where the agent IS, so it is never
        # mid-climb by definition.
        est.on_stairs = False
        self.stack.set_height(key, est._levels[key])
        self.stack.set_current(
            key, step=step, agent_xy=frame.camera_position[list(PLANE)])
        self._floor_y = float(est._levels[key])
        self.end_pursuit("external_switch")
        self.stats["floor_switch_external"] = (
            self.stats.get("floor_switch_external", 0) + 1)
        return key

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
        agent_xy = frame.camera_position[list(PLANE)]
        if self.external:
            # The estimator's own rule never runs: its primary signal is the
            # height trace, and the authority that called `request_floor` knows
            # better than any inference from it.
            floor_id = self._apply_external(frame, step)
            return self._floor_y_after(floor_id, prev_floor, frame, step)
        on_flight = bool(
            getattr(self.cfg.floor, "no_level_on_flight", False)
        ) and (self.climbing or self.on_flight_cells(agent_xy))
        if on_flight:
            self.stats["steps_on_a_flight"] = self.stats.get("steps_on_a_flight", 0) + 1
        floor_id = self.estimator.update(
            float(frame.camera_position[1]), step,
            xy=agent_xy, on_flight=on_flight,
        )
        self.stats["levels_suppressed_on_flight"] = int(
            getattr(self.estimator, "suppressed_levels", 0))
        return self._floor_y_after(floor_id, prev_floor, frame, step)

    def _floor_y_after(self, floor_id: int, prev_floor: int, frame, step: int) -> float:
        """Bookkeeping and the banding height, once the storey is decided.

        Shared by both sources, so an external switch is logged, counted and
        banded exactly like one the estimator committed -- `eval/record.py`
        reads `floor_log` either way and cannot tell them apart.
        """
        # Key is persistent; order is derived from these heights on demand.
        # Discovering a basement therefore changes order without renumbering
        # any track, room, cache entry, or portal edge.
        for key, height in self.estimator.levels.items():
            self.stack.set_height(key, height)
        if floor_id != prev_floor and not self.external:
            # External switches end their own pursuit, with their own reason.
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
        self.stack.current.steps_on_floor += 1
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

    # ------------------------------------------------------------- flights

    def _on_stair_mask(self, flight) -> bool:
        """Does the stair map agree this flight is a staircase?

        The mask is what a prior ASCENT pass recorded as stairs and what this
        run's own stair detector marked; `find_flights` never consults it, and
        a height layer alone cannot tell a flight from a sofa.
        """
        mask = getattr(self.costmap, "stair_mask", None)
        if mask is None:
            return False
        rc = np.asarray(flight.cells_rc)
        rows = np.clip(rc[:, 0], 0, mask.shape[0] - 1)
        cols = np.clip(rc[:, 1], 0, mask.shape[1] - 1)
        return bool(mask[rows, cols].any())

    def _destination_key(self, kind: str, floor_y: float):
        """The storey a flight of this kind would arrive at, or None.

        Nearest known level in the direction of travel -- the same rule the
        undirected branch of `try_switch` uses to pick `target_y`.
        """
        others = [(k, h) for k, h in self.known_levels().items()
                  if int(k) != int(self.stack.current_id)]
        if kind == "up":
            above = [(k, h) for k, h in others if h > float(floor_y)]
            return min(above, key=lambda kh: kh[1])[0] if above else None
        below = [(k, h) for k, h in others if h < float(floor_y)]
        return max(below, key=lambda kh: kh[1])[0] if below else None

    def _drop_disproved(self, usable: list, floor_y: float) -> list:
        """Flights back to a storey the agent's own failed attempts disproved.

        `agent.floor_disproved_after_failed_attempts` makes a storey's failed
        attempts a verdict on the STOREY, and that verdict was honoured by the
        floor LLM, the candidate channel and the storey posterior -- but not
        here, which is where it decides where to walk.

        Measured, 00800 toy airplane: two attempts failed upstairs, storey 0
        was disproved, the agent descended and ARRIVED on storey 1 at step 455
        -- then chose a `flight_up` at step 506, 51 steps later, and was back on
        the disproved storey by 699, where it spent its last attempt on a false
        positive and ended 21.5 m from a target that was downstairs all along.

        Never returns empty: a verdict may not make progress impossible, which
        is the same rule `_rank_flights` and `ExplorationStrategy` already keep.
        """
        if not self.disproved_floors:
            return usable
        kept = [f for f in usable
                if self._destination_key(f.kind, floor_y) not in self.disproved_floors]
        dropped = len(usable) - len(kept)
        if dropped and kept:
            self.stats["flights_skipped_disproved"] = (
                self.stats.get("flights_skipped_disproved", 0) + dropped)
        return kept or usable

    def known_levels(self) -> dict:
        """`{key: height}` over every storey this agent knows about.

        The ONLINE estimator's levels UNION the stack's layers, because a
        storey can be known without having been stood on: `_seed_storeys`
        creates a layer for every storey the prior obstacle map clusters, and
        `apply_map` restores the scene graph's. The estimator only ever learns
        a level by standing at that height.

        Reading the estimator alone deadlocked 00821. The agent starts at 0.13,
        the lower storey is -3.47, and `_rank_flights` discarded every
        `flight_down` because the estimator knew no level below -- so it could
        not descend to learn the level it needed in order to be allowed to
        descend. Measured: 3, 3 and 4 flights dropped in the three episodes
        that never changed storey, against 0 in the one that did (which got
        down only via a phantom mid-staircase level the estimator happened to
        commit).
        """
        levels = {int(k): float(h) for k, h in self.estimator.levels.items()}
        for key, layer in getattr(self.stack, "_layers", {}).items():
            levels.setdefault(int(key), float(getattr(layer, "floor_y", 0.0)))
        return levels

    def _rank_flights(self, usable: list, floor_y: float) -> list:
        """Drop flights that lead where no storey is, keep the rest.

        Needs TWO known storeys before it will drop anything: on a single-floor
        costmap the estimator knows one level and every flight would look
        impossible, which would take away the only way up.
        """
        levels = [float(h) for h in self.known_levels().values()]
        if len(levels) < 2:
            return usable
        gap = float(self.cfg.floor.new_level_m) * 0.5
        above = any(h > floor_y + gap for h in levels)
        below = any(h < floor_y - gap for h in levels)
        kept = [f for f in usable
                if (f.kind == "up" and above) or (f.kind == "down" and below)]
        dropped = len(usable) - len(kept)
        if dropped:
            self.stats["flights_dropped_no_storey"] = (
                self.stats.get("flights_dropped_no_storey", 0) + dropped)
        return kept or usable

    def _best_flight(self, usable: list, agent_xy):
        """Stair-map-corroborated flights first, nearest within that."""
        corroborated = [f for f in usable if self._on_stair_mask(f)]
        self.stats["flights_on_stair_mask"] = max(
            self.stats.get("flights_on_stair_mask", 0), len(corroborated))
        pool = corroborated or usable
        if corroborated:
            self.stats["flight_pick_corroborated"] = (
                self.stats.get("flight_pick_corroborated", 0) + 1)
        else:
            self.stats["flight_pick_uncorroborated"] = (
                self.stats.get("flight_pick_uncorroborated", 0) + 1)
        return min(pool, key=lambda f: float(np.linalg.norm(self.mouth(f) - agent_xy)))

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

    def pursue_route(self, frame, step: int, goal_xy, target_y: float, kind: str):
        """Start a pursuit toward where a stair ROUTE leaves this floor
        (`agent.climb_route`), when `try_switch` found no flight or portal to
        the storey asked for. Bookkept exactly as a flight pursuit, so the
        deadline, the failure memory and the climb's direction all apply; the
        flight is a stand-in with no cells, which every flight-cell carrot
        reads as "none"."""
        from types import SimpleNamespace

        goal_xy = np.asarray(goal_xy, dtype=float)
        if self._portal_failed_here(goal_xy):
            return None
        self.pursuing = True
        self.pursuit_flight = SimpleNamespace(
            kind=str(kind), cells_rc=np.zeros((0, 2), dtype=int), heights=np.zeros(0),
            n_cells=0, foot_xy=goal_xy.copy(), top_xy=goal_xy.copy(), span_m=0.0, path_m=None)
        self._pursuit_goal_xy = goal_xy.copy()
        self._portal_start_y = float(frame.camera_position[1])
        self._portal_step = step
        self.switch_policy.note_switch(step)
        self.stats["floor_switch_attempts"] = self.stats.get("floor_switch_attempts", 0) + 1
        self.stats["route_switch_attempts"] = self.stats.get("route_switch_attempts", 0) + 1
        self.portal_log.append((step, [round(float(x), 2) for x in goal_xy], f"route_{kind}", 0))
        return PortalGoal(goal_xy=goal_xy.copy(), target_y=float(target_y),
                          deadline_steps=self.cfg.floor.portal_deadline_steps)

    def end_pursuit(self, reason: str) -> None:
        self.pursuing = False
        self.stats[f"portal_end_{reason}"] = self.stats.get(f"portal_end_{reason}", 0) + 1
        # A pursuit that ended without climbing is evidence about that PLACE, not
        # just a statistic. `find_portals` is recomputed from scratch every round
        # and has no memory, so without recording the failure the same patch is
        # proposed again immediately -- which is exactly what 55 attempts at one
        # 28-cell patch looked like.
        if (
            reason in ("no_vertical_progress", "deadline")
            and self._pursuit_goal_xy is not None
            and bool(getattr(self.cfg.floor, "portal_failure_memory", False))
        ):
            self._failed_portals.append(np.asarray(self._pursuit_goal_xy, dtype=float))
            self.stats["portal_failures_remembered"] = len(self._failed_portals)
        self._pursuit_goal_xy = None
        self.pursuit_flight = None

    def _portal_failed_here(self, xy) -> bool:
        """Has a pursuit already failed from this place?"""
        if not self._failed_portals:
            return False
        radius = float(getattr(self.cfg.floor, "portal_failure_radius_m", 1.5))
        here = np.asarray(xy, dtype=float)
        return any(
            float(np.linalg.norm(here - bad)) <= radius for bad in self._failed_portals
        )

    def register_storey(self, height: float, tol_m: float = 1.0) -> int:
        """The key of the storey at `height`, created if none is within `tol_m`.

        For an observation made somewhere the agent has not stood -- pass 1's
        target tour (eval/prior_map.py) -- so it is filed under the right
        storey. Estimator and stack get the same key, as on a load
        (`map_store.apply_map`) and an external switch, so the walk arriving
        there later finds the storey rather than founding a second one. Does
        not move `current`."""
        est = self.estimator
        if est._levels:
            key = min(est._levels, key=lambda k: abs(float(est._levels[k]) - float(height)))
            if abs(float(est._levels[key]) - float(height)) <= float(tol_m):
                return int(key)
        key = max([int(est._next_id)] + [int(k) + 1 for k in self.stack._layers])
        est._levels[key] = float(height)
        est._samples[key] = [float(height)]
        est._next_id = key + 1
        self.stack.set_height(key, float(height))
        self.stats["storeys_registered"] = self.stats.get("storeys_registered", 0) + 1
        return key

    def mouth(self, flight, floor_y: Optional[float] = None) -> np.ndarray:
        """`mapping.stairs.mouth_xy` under `floor.descent_goal`."""
        mode = str(getattr(getattr(getattr(self, "cfg", None), "floor", None),
                           "descent_goal", "lowest_tread"))
        if mode != "extrapolated" or flight.kind != "down":
            return mouth_xy(flight, descent=mode)
        if floor_y is None:
            floor_y = float(self.estimator.height_of(self.stack.current_id))
        return mouth_xy(flight, descent=mode, floor_y=float(floor_y),
                        resolution=float(self.costmap.resolution))

    def flight_span_m(self, floor_y: float) -> float:
        """How tall a flight from `floor_y` may be, for `find_flights`.

        The gap to the nearest other known level when the storeys are known
        (config: floor.flight_span_from_levels), else `new_level_m`. Never
        smaller than `new_level_m`: a level a few centimetres off the current
        one -- the estimator offers those -- must not shrink the band to
        nothing.
        """
        span = float(self.cfg.floor.new_level_m)
        if not bool(getattr(self.cfg.floor, "flight_span_from_levels", False)):
            return span
        gaps = [
            abs(float(h) - float(floor_y))
            for h in self.estimator.levels.values()
            if abs(float(h) - float(floor_y)) > span * 0.5
        ]
        if not gaps:
            return span
        return max(span, min(gaps))

    def on_flight_cells(self, agent_xy) -> bool:
        """Is the agent standing on a staircase?

        The flight this pursuit chose if there is one, else any cell the stair
        mask has marked traversable on this storey. Used to refuse creating a
        new level mid-climb; a half-landing is otherwise roomy enough to become
        a storey of its own.
        """
        rc = self.costmap.world_to_grid(np.asarray(agent_xy, dtype=float))
        if not self.costmap.in_bounds(rc):
            return False
        flight = self.pursuit_flight
        if flight is not None and flight.n_cells:
            radius = max(1, int(round(0.5 / self.costmap.resolution)))
            if np.any(np.all(np.abs(flight.cells_rc - rc) <= radius, axis=1)):
                return True
        mask = self.costmap.stair_mask
        return bool(mask is not None and mask[rc[0], rc[1]])

    def _remembered_stair(self, target_floor: Optional[int]):
        """A staircase the prior map already walked, as (goal_xy, other_floor).

        `StairEdge` records both mouths of one traversal: `entry_xy` is where
        the agent stood when it committed to the new storey -- the mouth on the
        floor it LEFT -- and `exit_xy` is its landing point on the floor it
        reached. So an edge touching the current floor names a point on THIS
        floor from which the other storey is reachable, whichever way it was
        walked.

        Prefers the most-traversed edge, then the most recent, because a
        staircase pass 1 used repeatedly is the one that works.

        That reading of the two points is not what `FloorStack.set_current`
        records (both lie on the storey arrived at); `floor.prior_stair_mouth`
        "arrival" reads what it does record -- see `_remembered_arrival`.
        """
        mode = str(getattr(self.cfg.floor, "prior_stair_mouth", "edge"))
        if mode == "arrival":
            return self._remembered_arrival(target_floor)
        if mode != "edge":
            raise ValueError(f"floor.prior_stair_mouth must be 'edge' or 'arrival', got {mode!r}")
        current = int(self.stack.current_id)
        best = None
        for edge in getattr(self.stack, "stair_edges", []) or []:
            if int(edge.from_floor) == current:
                other, xy = int(edge.to_floor), edge.entry_xy
            elif int(edge.to_floor) == current:
                other, xy = int(edge.from_floor), edge.exit_xy
            else:
                continue
            if xy is None or other == current:
                continue
            if target_floor is not None and other != int(target_floor):
                continue
            rank = (int(edge.n_traversals), int(edge.step))
            if best is None or rank > best[0]:
                best = (rank, np.asarray(xy, dtype=float), other)
        return None if best is None else (best[1], best[2])

    def _remembered_arrival(self, target_floor: Optional[int]):
        """This storey's stair mouth toward `target_floor`, from where the prior
        map ARRIVED on this storey, as (goal_xy, other_floor) or None.

        `set_current` runs when the estimator commits the new storey, so an
        edge's `entry_xy` is where that traversal arrived and `exit_xy` the
        storey's first arrival -- both on `to_floor`. An edge arriving here from
        a storey on the target's side came up (or down) the staircase that goes
        there. Its `from_floor` may be a landing `drop_prior_landings` took out
        of the stack; its height is still known (`stack.dropped_levels`).

        Ranked: from the target storey itself, then the most traversed, then the
        most recent. A mouth `prior_stair_max_failures` pursuits failed beside
        is skipped. `other_floor` is the target, or with none asked for the
        nearest kept storey on the edge's side.
        """
        current = int(self.stack.current_id)
        here_y = float(self.estimator.height_of(current))
        dropped = dict(getattr(self.stack, "dropped_levels", {}) or {})

        levels = self.known_levels()

        def height(key: int) -> Optional[float]:
            if int(key) in levels:
                return float(levels[int(key)])
            return dropped.get(int(key))

        want = None
        if target_floor is not None:
            target_y = height(int(target_floor))
            if target_y is None:
                return None
            want = 1.0 if target_y > here_y else -1.0
        tol = float(self.cfg.floor.level_tol_m)
        max_fail = int(getattr(self.cfg.floor, "prior_stair_max_failures", 0) or 0)
        radius = float(getattr(self.cfg.floor, "portal_failure_radius_m", 1.5))
        best = None
        for edge in getattr(self.stack, "stair_edges", []) or []:
            if int(edge.to_floor) != current or int(edge.from_floor) == current:
                continue
            other_y = height(int(edge.from_floor))
            if other_y is None or abs(other_y - here_y) <= tol:
                continue
            side = 1.0 if other_y > here_y else -1.0
            if want is not None and side != want:
                continue
            for xy in (edge.entry_xy, edge.exit_xy):
                if xy is None:
                    continue
                xy = np.asarray(xy, dtype=float)
                if max_fail > 0 and sum(
                    float(np.linalg.norm(xy - bad)) <= radius for bad in self._failed_portals
                ) >= max_fail:
                    self.stats["prior_stair_skipped_after_failures"] = (
                        self.stats.get("prior_stair_skipped_after_failures", 0) + 1)
                    continue
                exact = target_floor is not None and int(edge.from_floor) == int(target_floor)
                rank = (exact, int(edge.n_traversals), int(edge.step))
                if best is None or rank > best[0]:
                    best = (rank, xy, side)
        if best is None:
            return None
        if target_floor is not None:
            return best[1], int(target_floor)
        kept = [(k, h) for k, h in self.known_levels().items()
                if k != current and (h - here_y) * best[2] > tol]
        if not kept:
            return None
        return best[1], min(kept, key=lambda kh: abs(kh[1] - here_y))[0]

    def _pursue_remembered(self, frame, step: int, remembered, floor_y: float,
                           reachable_fn, directed: bool, portals) -> Optional[PortalGoal]:
        """Start the pursuit of a remembered staircase (`_remembered_stair`), or
        None when the reachability check refuses it."""
        goal_xy, other_floor = remembered
        # An arrival is a point on THIS storey (`_remembered_arrival`).
        arrival = str(getattr(self.cfg.floor, "prior_stair_mouth", "edge")) == "arrival"
        if reachable_fn is not None and not reachable_fn(
            goal_xy, float(floor_y) if arrival else self.estimator.height_of(other_floor)
        ):
            self.stats["prior_stair_unreachable"] = (
                self.stats.get("prior_stair_unreachable", 0) + 1
            )
            return None
        self.pursuing = True
        self._pursuit_goal_xy = np.asarray(goal_xy, dtype=float).copy()
        self._portal_start_y = float(frame.camera_position[1])
        self._portal_step = step
        self.switch_policy.note_switch(step)
        self.stats["floor_switch_attempts"] = (
            self.stats.get("floor_switch_attempts", 0) + 1
        )
        self.stats["prior_stair_switch_attempts"] = (
            self.stats.get("prior_stair_switch_attempts", 0) + 1
        )
        if directed:
            self.stats["directed_floor_switch_attempts"] = (
                self.stats.get("directed_floor_switch_attempts", 0) + 1
            )
        self.portal_log.append((
            step, [round(float(x), 2) for x in goal_xy], "prior_stair",
            len(portals),
        ))
        if portals:
            self.stats["prior_stair_preferred_over_portal"] = (
                self.stats.get("prior_stair_preferred_over_portal", 0) + 1
            )
        return PortalGoal(
            goal_xy=np.asarray(goal_xy, dtype=float).copy(),
            target_y=float(self.estimator.height_of(other_floor)),
            deadline_steps=self.cfg.floor.portal_deadline_steps,
        )

    def try_switch(
        self, frame, step: int, best_path_cost, scene_graph, target: str, reachable_fn,
        target_floor: Optional[int] = None, presence_of=None, stair_xyz=None,
    ) -> Optional[PortalGoal]:
        """Head for another storey when this one has nothing near left.

        Returns the portal to drive to, or None to stay. The caller applies it:
        deciding to leave a floor is this policy's business, and moving the
        agent is the FSM's.

        `presence_of` is passed straight to `floor_target_evidence`, where it
        gates the target bonus on the instance still being believed. The caller
        supplies it only when `exploration.floor_evidence_by_presence` is set.
        """
        if self.switch_policy is None:
            return None
        if target_floor is not None and int(target_floor) == self.stack.current_id:
            return None
        evidence, n_objects = floor_target_evidence(
            scene_graph, self.stack.current_id, target, presence_of=presence_of,
        )
        directed = target_floor is not None
        # An undirected switch needs somewhere to go. On a single-floor scene the
        # estimator knows one level, `find_portals` still reads a tall bookcase
        # as a portal, and the agent walks off to it once its frontiers are far.
        # A schema-v2 prior map seeds every storey on load, so a real multi-floor
        # run has >= 2 levels here and is unaffected; see the config comment.
        if (
            not directed
            and bool(getattr(self.cfg.floor, "switch_requires_second_level", False))
            and len(self.estimator.levels) < 2
        ):
            return None
        # Since ARRIVAL, not since the layer was created: a floor restored from
        # the prior map has first_step 0, which made a 33-step-old arrival look
        # like 347 steps of searching and let every dwell guard through.
        steps_here = step - int(getattr(self.stack.current, "arrived_step", self.stack.current.first_step))
        if not directed and not self.switch_policy.may_switch(
            step, best_path_cost, evidence=evidence, n_objects=n_objects,
            steps_on_floor=steps_here,
        ):
            # The gate refused. Measured on outputs/crossfloor_ab, 3 of 7
            # cross-floor episodes end here every round: no storey request from
            # the posterior, the geometric rule never satisfied because a large
            # floor always has SOME near frontier, and so `try_switch` returns
            # before `find_portals` is ever called -- which is why
            # `use_prior_stairs` could not help them. They rose 0.00 to 0.19 m
            # in 500 steps without one attempt.
            #
            # A remembered staircase changes what the refusal is about. The
            # "nothing near left here" clause prices the risk of walking off
            # toward a patch of another storey that may not be a way up; a
            # traversal the prior map actually made carries no such risk.
            if not (
                bool(getattr(self.cfg.floor, "prior_stairs_override_gate", False))
                and bool(getattr(self.cfg.floor, "use_prior_stairs", False))
                and self._remembered_stair(target_floor) is not None
                and self.switch_policy.may_switch_to_known_stairs(step, steps_here)
            ):
                return None
            self.stats["prior_stair_gate_override"] = (
                self.stats.get("prior_stair_gate_override", 0) + 1
            )

        floor_y = self.estimator.height_of(self.stack.current_id)
        portals = find_portals(
            self.costmap, floor_y,
            min_delta_m=self.cfg.floor.new_level_m,
            max_delta_m=self.cfg.floor.portal_max_delta_m,
            min_cells=self.cfg.floor.portal_min_cells,
        )
        self.stats["portals_seen"] = max(self.stats.get("portals_seen", 0), len(portals))
        if bool(getattr(self.cfg.floor, "portal_failure_memory", False)):
            kept = [p for p in portals if not self._portal_failed_here(p.centroid_xy)]
            if len(kept) != len(portals):
                self.stats["portals_skipped_after_failure"] = (
                    self.stats.get("portals_skipped_after_failure", 0)
                    + (len(portals) - len(kept))
                )
            portals = kept
        remembered = (
            self._remembered_stair(target_floor)
            if bool(getattr(self.cfg.floor, "use_prior_stairs", False)) else None
        )
        # A staircase the robot has already walked certifies the transition
        # better than a flight read off the height layer
        # (`floor.prior_stairs_before_flights`): on 00800 (seed 42, static
        # 50001) the flight's foot was taken 3 m from the true one, beside the
        # treads, and both climbs from it stalled at 0.0 m, while pass 1 had
        # arrived on this storey 0.8-1.0 m from that foot.
        if (remembered is not None
                and bool(getattr(self.cfg.floor, "prior_stairs_before_flights", False))
                and bool(getattr(self.cfg.floor, "prefer_prior_stairs", False))):
            goal = self._pursue_remembered(frame, step, remembered, floor_y,
                                           reachable_fn, directed, portals)
            if goal is not None:
                self.stats["prior_stair_before_flights"] = (
                    self.stats.get("prior_stair_before_flights", 0) + 1)
                return goal
        # A flight read off the height layer is the most direct target there is:
        # its lowest tread is the foot, and a goal ON the treads is what the
        # PointNav mover will climb, slowly. Tried first, ahead of the detector's
        # `stairs` tracks (2.7 m beside the foot on 00821) and ahead of portals
        # (a sightline). A flight that failed once is out, via the same memory.
        if str(getattr(self.cfg.floor, "climb_targets", "portals")) == "flights_first":
            want = None
            if directed and self.stack.by_key(int(target_floor)) is not None:
                want = "up" if float(self.stack.by_key(int(target_floor)).floor_y) > float(floor_y) else "down"
            flights = find_flights(
                self.costmap, float(floor_y),
                new_level_m=float(self.cfg.floor.new_level_m),
                min_span_m=float(getattr(self.cfg.floor, "flight_min_span_m", 1.0)),
                min_cells=int(getattr(self.cfg.floor, "flight_min_cells", 150)),
                wide_span_m=self.flight_span_m(float(floor_y)),
                wide_mask=getattr(self.costmap, "stair_mask", None),
                order_path=bool(getattr(self.cfg.agent,
                    "climb_carrot_follow_path", False)),
            )
            self.stats["flights_seen"] = max(self.stats.get("flights_seen", 0), len(flights))
            agent_xy = frame.camera_position[list(PLANE)]
            usable = [
                f for f in flights
                if (want is None or f.kind == want) and not self._portal_failed_here(self.mouth(f))
            ]
            rank = bool(getattr(self.cfg.floor, "flights_prefer_stair_mask", False))
            if usable and rank:
                usable = self._rank_flights(usable, float(floor_y))
            # A DIRECTED switch already names its storey; an undirected one is
            # free to pick, and must not pick a storey already disproved.
            if usable and not directed:
                usable = self._drop_disproved(usable, float(floor_y))
            if usable:
                flight = (
                    self._best_flight(usable, agent_xy) if rank
                    else min(usable, key=lambda f: float(np.linalg.norm(self.mouth(f) - agent_xy)))
                )
                # Make the treads traversable on this floor's grid, so the planner
                # and the frontier extractor stop reading the flight as a wall.
                apply_stair_mask(self.costmap, [StairRegion(
                    cells_rc=flight.cells_rc, centroid_xy=flight.foot_xy,
                    n_cells=flight.n_cells, mean_dh=0.0,
                    low_y=float(flight.heights.min()), high_y=float(flight.heights.max()),
                )], max_area_frac=float(self.cfg.floor.stair_max_area_frac))
                # The mouth on THIS storey -- the top of a descent, the foot of an
                # ascent -- as `floor.descent_goal` reads a descent's top (see
                # `stairs.mouth_xy`).
                goal_xy = self.mouth(flight)
                if want is not None:
                    target_y = float(self.stack.by_key(int(target_floor)).floor_y)
                else:
                    others = [h for k, h in self.known_levels().items()
                              if k != self.stack.current_id]
                    if flight.kind == "up":
                        above = [h for h in others if h > float(floor_y)]
                        target_y = min(above) if above else float(floor_y) + float(self.cfg.floor.new_level_m)
                    else:
                        below = [h for h in others if h < float(floor_y)]
                        target_y = max(below) if below else float(floor_y) - float(self.cfg.floor.new_level_m)
                if reachable_fn is None or reachable_fn(goal_xy, float(floor_y)):
                    self.pursuing = True
                    self.pursuit_flight = flight
                    self._pursuit_goal_xy = goal_xy.copy()
                    self._portal_start_y = float(frame.camera_position[1])
                    self._portal_step = step
                    self.switch_policy.note_switch(step)
                    self.stats["floor_switch_attempts"] = self.stats.get("floor_switch_attempts", 0) + 1
                    self.stats["flight_switch_attempts"] = self.stats.get("flight_switch_attempts", 0) + 1
                    self.stats[f"flight_switch_{flight.kind}"] = self.stats.get(f"flight_switch_{flight.kind}", 0) + 1
                    if directed:
                        self.stats["directed_floor_switch_attempts"] = (
                            self.stats.get("directed_floor_switch_attempts", 0) + 1
                        )
                    self.portal_log.append((
                        step, [round(float(x), 2) for x in goal_xy], f"flight_{flight.kind}",
                        flight.n_cells,
                    ))
                    return PortalGoal(
                        goal_xy=goal_xy.copy(), target_y=float(target_y),
                        deadline_steps=self.cfg.floor.portal_deadline_steps,
                    )
        # A staircase the detector has SEEN is a place you can climb from. A
        # portal is a place you can see the next floor from, which over a
        # balcony rail is not the same thing: measured on 00821, the portal the
        # agent chased 55 times sat about 10 m from the flight the navmesh
        # actually uses. `stair_xyz` are the 3D centres of `stairs` tracks in
        # the object layer -- multi-frame, evidence-gated detections, not
        # single-frame masks -- and on this storey they are the best target
        # there is. Nearest first; the agent's climb state does the rest.
        if (
            str(getattr(self.cfg.floor, "climb_targets", "portals")) == "stairs_first"
            and stair_xyz
        ):
            here_y = float(floor_y)
            span = float(self.cfg.floor.new_level_m)
            # Which way must the flight go? A directed request says; otherwise
            # any flight will do.
            want = None
            if directed and self.stack.by_key(int(target_floor)) is not None:
                want = "up" if float(self.stack.by_key(int(target_floor)).floor_y) > here_y else "down"
            usable = []
            for xy, kind in stair_xyz:
                xy = np.asarray(xy, dtype=float)
                if want is not None and kind is not None and kind != want:
                    continue  # a flight the wrong way is not a way there
                if self._portal_failed_here(xy):
                    continue
                usable.append((xy, kind))
            if usable:
                agent_xy = frame.camera_position[list(PLANE)]
                goal_xy, kind = min(usable, key=lambda c: float(np.linalg.norm(c[0] - agent_xy)))
                if want is not None:
                    target_y = float(self.stack.by_key(int(target_floor)).floor_y)
                else:
                    others = [h for k, h in self.estimator.levels.items()
                              if k != self.stack.current_id]
                    if kind == "up":
                        above = [h for h in others if h > here_y]
                        target_y = min(above) if above else here_y + span
                    elif kind == "down":
                        below = [h for h in others if h < here_y]
                        target_y = max(below) if below else here_y - span
                    else:
                        target_y = (min(others, key=lambda h: abs(h - here_y))
                                    if others else here_y + span)
                if reachable_fn is None or reachable_fn(goal_xy, here_y):
                    self.pursuing = True
                    self._pursuit_goal_xy = np.asarray(goal_xy, dtype=float).copy()
                    self._portal_start_y = float(frame.camera_position[1])
                    self._portal_step = step
                    self.switch_policy.note_switch(step)
                    self.stats["floor_switch_attempts"] = (
                        self.stats.get("floor_switch_attempts", 0) + 1
                    )
                    self.stats["stair_track_switch_attempts"] = (
                        self.stats.get("stair_track_switch_attempts", 0) + 1
                    )
                    if directed:
                        self.stats["directed_floor_switch_attempts"] = (
                            self.stats.get("directed_floor_switch_attempts", 0) + 1
                        )
                    self.portal_log.append((
                        step, [round(float(x), 2) for x in goal_xy], "stair_track",
                        len(portals),
                    ))
                    return PortalGoal(
                        goal_xy=np.asarray(goal_xy, dtype=float).copy(),
                        target_y=float(target_y),
                        deadline_steps=self.cfg.floor.portal_deadline_steps,
                    )
        prefer = bool(getattr(self.cfg.floor, "prefer_prior_stairs", False))
        if not portals or (prefer and remembered is not None):
            # Either nothing visible to drive to, or something visible that is
            # not worth driving to.
            #
            # `find_portals` detects a patch of ANOTHER STOREY VISIBLE FROM HERE.
            # Over a balcony rail or an open stairwell that is a place you can
            # see the next floor from and cannot walk up from. Measured on
            # 00821's cracker box: 55 portal goals, every one a real 2.69 m
            # gap, the agent arrived at one (`portal_end_arrived`) and rose
            # 0.17 m in 500 steps (`portal_end_no_vertical_progress`), settling
            # into re-selecting the same 28-cell patch every five steps.
            #
            # The prior map holds something strictly better: `connectivity`
            # records where pass 1 ACTUALLY CHANGED FLOOR, which is a staircase
            # by construction rather than a sightline. `prefer_prior_stairs`
            # uses it ahead of any detected portal; with it off this stays a
            # fallback for when nothing is visible at all.
            #
            # Measured on outputs/osg_authored_15 before any of this: of 11
            # cross-floor episodes, 4 saw no portal and never attempted a
            # switch, and across the run 405 storey requests produced 12
            # attempts.
            # But the agent is not ignorant of the staircase -- the prior map
            # RECORDED the one pass 1 walked, in `connectivity`, and `apply_map`
            # restores it into `stack.stair_edges`. Nothing ever read it back.
            # A remembered mouth is also better evidence than a live portal
            # here: the undirected portal search picks the nearest height
            # artifact, and its 24 attempts moved the agent a median 0.13 m.
            if remembered is None:
                return None
            return self._pursue_remembered(frame, step, remembered, floor_y,
                                           reachable_fn, directed, portals)

        agent_xy = frame.camera_position[list(PLANE)]
        # Prefer a storey we have NOT searched, then the nearest. Nearest-only
        # let the agent bounce back onto a floor it had already given up on --
        # 4-5 transitions in some episodes, paying the travel cost each time.
        levels = self.estimator.levels

        if directed:
            target_layer = self.stack.by_key(int(target_floor))
            if target_layer is None:
                return None
            delta = float(target_layer.floor_y - floor_y)
            if abs(delta) <= self.cfg.floor.level_tol_m:
                return None
            direction = 1.0 if delta > 0.0 else -1.0
            portals = [p for p in portals if float(p.delta_y) * direction > 0.0]
            if not portals:
                return None

        def unvisited(p):
            return not any(
                abs(h - p.target_y) <= self.cfg.floor.level_tol_m for h in levels.values()
            )

        if directed:
            target_y = float(self.stack.by_key(int(target_floor)).floor_y)
            portals.sort(key=lambda p: (
                abs(float(p.target_y) - target_y),
                float(np.linalg.norm(p.centroid_xy - agent_xy)),
            ))
        else:
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
        self._pursuit_goal_xy = portal.centroid_xy.copy()
        self._portal_start_y = float(frame.camera_position[1])
        self._portal_step = step
        self.switch_policy.note_switch(step)
        self.stats["floor_switch_attempts"] = self.stats.get("floor_switch_attempts", 0) + 1
        if directed:
            self.stats["directed_floor_switch_attempts"] = (
                self.stats.get("directed_floor_switch_attempts", 0) + 1
            )
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
