"""Where to go next: unexplored space and mapped surfaces, under one index.

DualMap's reaction to a failed candidate is to take the next-highest similarity
and add the failed one to an ignore list discarded when the query ends. That is
the discrete search problem with the belief thrown away -- no model of where the
object went, no cost of getting there, no memory of what was already searched,
and no way for the retry loop to decide to go explore instead.

The classical result is that the optimal ORDER is by b(x)*d(x)/c(x) -- belief
times per-visit detection probability over cost -- and `select_frontier` already
computes exactly that index over frontiers (score / path_cost). So this does not
add an objective or a planner. It widens the candidate set to include the
surfaces an object could have been moved to, and lets the two compete:
`search_frontier_weight` is the single judgement call, naming what unmapped
space is worth against a plausible surface.

Everything an exploration round remembers lives here -- what has been searched
and how well, which frontiers are blocked and until when, which surface is being
inspected right now. None of it is FSM state, and keeping it out of `NavAgent`
is what makes this file independently readable and independently testable.

The interface is deliberately narrow in both directions. A round sees a
`WorldView` -- a snapshot of what it is allowed to read -- and returns an
`ExplorationChoice`, a description of where to go. The agent applies it. The
strategy never writes `state`, never writes `_goal_xy`, and never moves anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE, Costmap2D, nearest_free_xy
from ..mapping.frontier import Frontier, FrontierExtractor
from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading
from .search_belief import InspectionLog, build_container_candidates, select_candidate
from .selector import frontier_goal_xy, select_frontier

# WaypointController.act's default arrival tolerance. Named here because
# `frontier_reach_m` is derived from it: the two must agree or an ordinary
# arrival is classified as an unreachable stub.
FRONTIER_ARRIVAL_TOL_M = 0.2


@dataclass
class WorldView:
    """What one exploration round is allowed to see.

    Passed in per round rather than held, because every field of it changes
    under the strategy's feet -- `costmap` is a different object once the agent
    changes storey, and `goal_xy` is owned by the FSM.
    """

    frame: FrameData
    step: int
    agent_xy: np.ndarray
    costmap: Costmap2D
    scene_graph: object
    object_layer: object
    keyframes: object
    target: str
    goal_xy: Optional[np.ndarray]
    floor_id: int = 0


@dataclass
class ExplorationChoice:
    """Where the round decided to go. `kind` says which mechanism chose it, so
    a trace can tell "the search posterior sent me here" from "the frontier did"
    without inferring it from the geometry."""

    kind: str  # "frontier" | "surface"
    goal_xy: np.ndarray
    path: Optional[np.ndarray] = None
    frontier: Optional[Frontier] = None
    container_id: Optional[int] = None
    face_turns: int = 0


class ExplorationStrategy:
    def __init__(self, cfg, planner, scorer, viewpoint_planner, affinity,
                 stats: dict, profiler) -> None:
        # `cfg` is the exploration group alone: nothing here reads any other.
        self.cfg = cfg.exploration
        self.planner = planner
        self.scorer = scorer
        self.viewpoint_planner = viewpoint_planner
        self.affinity = affinity
        self.stats = stats
        self.profiler = profiler
        self.frontier_extractor = FrontierExtractor(
            min_cells=self.cfg.frontier_min_cells,
            dedup_m=self.cfg.frontier_dedup_m,
        )
        self.goal_prefer_free = bool(self.cfg.frontier_goal_free_cell)
        self.cost_prefer_free = bool(self.cfg.frontier_cost_free_cell)
        # How far from a frontier goal still counts as NOT having reached it.
        #
        # This has to be derived from the planner, not chosen: HybridVoronoi
        # returns a path ending at a graph node within `goal_near_m` (0.7 m) of
        # the goal -- it navigates the medial axis and deliberately stops near,
        # not on, the goal -- and WaypointController reports arrival within
        # `arrival_tol_m` (0.2 m) of that endpoint. A correct arrival therefore
        # leaves the agent up to 0.9 m from the frontier goal. Against a fixed
        # 0.5 m this was read as a degenerate stub: the frontier was blocked for
        # 100 rounds and `_last_giveup_pt` was set, which also suppresses the
        # all-frontiers-blocked fallback anywhere near it -- for the ordinary
        # case of having got there. Measured over 96 dynamic episodes,
        # `frontier_stub_block` fired 2.87 times per failing episode against
        # 0.39 per success.
        self.frontier_reach_m = (
            float(self.cfg.voronoi_goal_near_m) + FRONTIER_ARRIVAL_TOL_M + 0.1
        )
        self.reset()

    def reset(self) -> None:
        self.current_frontier: Optional[Frontier] = None
        # Location-keyed blacklist: frontier ids are reassigned on every
        # extraction, so blocking must be spatial to persist. [(xy, until, floor)]
        self._blocked_pts: list = []
        # Centroid of the frontier the agent most recently gave up on: excluded
        # from the "all frontiers blocked" fallback so the agent doesn't
        # immediately re-pursue the dead-end it just abandoned. Floor-scoped for
        # the same reason as the blacklist.
        self._last_giveup_pt: Optional[tuple] = None
        self._last_select_step = -100
        # What has already been searched, and how well (C3). A visit multiplies
        # a surface's belief by (1 - d) rather than zeroing it, so a place
        # glanced at from four metres stays plausible and one inspected closely
        # mostly stops being -- the distinction an ignore list cannot make.
        self.search_log = InspectionLog()
        self.search_container: Optional[int] = None
        self.search_started_step = 0
        self.surface_face_turns = 0
        self.search_log_events: List[dict] = []
        self.frontier_select_log: list = []
        self.giveup_log: list = []
        # Where and when the current pursuit was last seen to be making
        # progress. Exploration state, not FSM state: it exists only to answer
        # "is this frontier worth pushing at", and only a pursuit resets it.
        self.progress_ref_step = 0
        self.progress_ref_xy = np.zeros(2)

    # ------------------------------------------------------------- blacklist

    def block(self, f: Optional[Frontier], duration: int, step: int) -> None:
        # Blocks carry their storey. Stored unconditionally: on a single floor
        # every entry is floor 0, so the floor test below is a tautology and
        # behaviour is unchanged -- no second code path to keep in sync.
        if f is not None:
            self._blocked_pts.append((f.centroid_xy.copy(), step + duration, f.floor))

    def _blocked_ids(self, frontiers, step: int) -> set:
        self._blocked_pts = [b for b in self._blocked_pts if b[1] > step]
        active = [(xy, floor) for xy, _, floor in self._blocked_pts]
        return {
            f.id
            for f in frontiers
            if any(
                floor == f.floor and np.linalg.norm(f.centroid_xy - xy) < 0.6
                for xy, floor in active
            )
        }

    @staticmethod
    def _heading_xy(frame: FrameData) -> np.ndarray:
        """Agent forward direction on the ground plane (unit). Camera looks along
        +z (OpenCV), so world-forward = R @ [0,0,1], projected to (x, z)."""
        fwd = frame.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0])
        v = fwd[list(PLANE)]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0])
    def select(self, world: WorldView, floor_switch) -> Optional[ExplorationChoice]:
        # Extraction + top-N path planning is expensive; while waiting the
        # agent turns in place, which grows the map anyway.
        if world.step - self._last_select_step < 5:
            return None
        self._last_select_step = world.step
        # A surface is only searched once the agent has actually got there.
        # Marking it on the next selection round instead -- which fires every 5
        # steps -- spent belief on places the agent had merely set off towards,
        # so it visited seven surfaces in 500 steps and inspected none of them.
        if self.search_container is not None:
            agent_xy = world.agent_xy
            arrived = (
                world.goal_xy is not None
                and float(np.linalg.norm(agent_xy - world.goal_xy))
                <= float(self.cfg.search_arrival_m)
            )
            spent = world.step - self.search_started_step
            if not arrived and spent < int(self.cfg.search_max_steps):
                return None  # still on the way: stay committed to this surface
            self.mark_searched(world.step, arrived=arrived)
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(
                world.costmap, world.agent_xy,
                floor=world.floor_id,
            )
        if not frontiers:
            # Nothing left on this floor is the strongest possible "no near
            # frontier", so the portal gate still gets its chance.
            floor_switch(None)
            return None
        # Async scoring request (never blocks); use whatever scores exist now
        self.scorer.request(frontiers, world.scene_graph, world.target, world.keyframes)
        blocked = self._blocked_ids(frontiers, world.step)
        agent_xy = world.agent_xy
        heading_xy = self._heading_xy(world.frame)
        failed: set = set()
        with self.profiler.timeit("frontier_select"):
            best = select_frontier(
                frontiers,
                self.scorer.latest(),
                self.planner,
                world.costmap,
                agent_xy,
                unscored_prior=self.cfg.unscored_prior,
                min_path_cost_m=self.cfg.min_path_cost_m,
                top_n=self.cfg.top_n_frontiers,
                blocked=blocked,
                failed_out=failed,
                info_gain_weight=self.cfg.info_gain_weight,
                info_gain_radius_m=self.cfg.info_gain_radius_m,
                los_visibility_penalty=self.cfg.los_visibility_penalty,
                heading_xy=heading_xy,
                continuity_weight=self.cfg.continuity_weight,
                goal_prefer_free=self.goal_prefer_free,
                cost_prefer_free=self.cost_prefer_free,
            )
        by_id = {f.id: f for f in frontiers}
        for fid in failed:  # block only the candidates that actually failed
            self.block(by_id.get(fid), 50, world.step)

        surface = self._select_surface(world, best)
        if surface is not None:
            self.search_container = int(surface.ref_id)
            self.search_started_step = world.step
            self.surface_face_turns = int(self.cfg.search_face_turns)
            self.current_frontier = None
            self.stats["search_surface"] = self.stats.get("search_surface", 0) + 1
            self.search_log_events.append(
                {
                    "step": int(world.step),
                    "container_id": int(surface.ref_id),
                    "label": surface.label,
                    "prior": round(float(surface.prior), 4),
                    "path_cost": round(float(surface.path_cost or 0.0), 2),
                    "utility": round(float(surface.utility or 0.0), 5),
                }
            )
            # Returned, not driven -- but it MUST be driven. Only GOTO_FRONTIER
            # follows the goal, and an earlier version set the goal while
            # leaving the state EXPLORE: the agent never moved, re-selected the
            # same surface five steps later and scored it "never reached" each
            # time. Every C3 result before that fix was measuring selections
            # that were never acted on -- eight inspections of one desk, an
            # unchanged 2.3 m path cost, arrived=False throughout.
            return ExplorationChoice(
                kind="surface",
                goal_xy=surface.goal_xy,
                container_id=int(surface.ref_id),
                face_turns=int(self.cfg.search_face_turns),
            )
        if best is None or best.path_cost is None:
            # Every frontier was blocked (a give-up/plan-fail cascade in
            # cluttered scenes leaves nothing selectable) -- rather than turn in
            # place burning the step budget until blocks expire, fall back to the
            # best path-reachable frontier ignoring blocks, excluding only the
            # one just given up on. A frontier blocked from an earlier pose is
            # often reachable now; if it re-stalls, give-up catches it again.
            relaxed_blocked = set()
            if self._last_giveup_pt is not None:
                relaxed_blocked = {
                    f.id for f in frontiers
                    if f.floor == self._last_giveup_pt[1]
                    and np.linalg.norm(f.centroid_xy - self._last_giveup_pt[0]) < 0.6
                }
            if len(relaxed_blocked) < len(frontiers):
                best = select_frontier(
                    frontiers, self.scorer.latest(), self.planner, world.costmap,
                    agent_xy, unscored_prior=self.cfg.unscored_prior,
                    min_path_cost_m=self.cfg.min_path_cost_m,
                    top_n=self.cfg.top_n_frontiers, blocked=relaxed_blocked,
                    info_gain_weight=self.cfg.info_gain_weight,
                    info_gain_radius_m=self.cfg.info_gain_radius_m,
                    los_visibility_penalty=self.cfg.los_visibility_penalty,
                    heading_xy=heading_xy,
                    continuity_weight=self.cfg.continuity_weight,
                    goal_prefer_free=self.goal_prefer_free,
                    cost_prefer_free=self.cost_prefer_free,
                )
        # Nothing near left on this floor? Consider leaving it. Checked BEFORE
        # committing to a far frontier, because "the best thing here is 12 m
        # away" is exactly ASCENT's condition for reasoning about storeys.
        if floor_switch(None if best is None else best.path_cost):
            return None

        if best is None or best.path_cost is None:
            self.stats["select_none"] += 1
            return None
        self.stats["select_ok"] += 1
        # Per-selection trace (step, agent xy, chosen frontier xy, path cost,
        # #frontiers) for exploration-efficiency debugging. See scripts.
        self.frontier_select_log.append((
            world.step,
            [round(float(x), 2) for x in agent_xy],
            [round(float(x), 2) for x in best.centroid_xy],
            round(float(best.path_cost), 2) if best.path_cost is not None else None,
            len(frontiers),
        ))
        self.current_frontier = best
        goal_xy = frontier_goal_xy(best, world.costmap, self.goal_prefer_free)
        with self.profiler.timeit("planner"):
            result = self.planner.plan(world.costmap, agent_xy, goal_xy)
        self.stats["plan_ok" if result.success else "plan_fail"] += 1
        if not result.success:
            self.block(best, 50, world.step)
            return None
        # A fresh pursuit starts its own 15-step progress window; without this
        # the give-up timer carried over from whatever frontier was pursued (or
        # given up on) before, and could fire on the very first step of the new
        # pursuit based on stale position data. The agent resets it when it
        # applies this choice.
        return ExplorationChoice(
            kind="frontier", goal_xy=goal_xy, path=result.path, frontier=best,
        )

    def glance(self, world: WorldView) -> None:
        """A surface in plain view has been searched, without driving to it.

        Measured: a full inspection costs the agent about fifty steps -- approach,
        arrival, commitment budget -- so a 500-step episode manages seven to nine
        of them. Simulating the search order over this scene's 112 surfaces says
        the target is typically reached after 37-45 inspections but only 33-40 m
        of travel, so the budget that binds is inspections, not distance. Most of
        those surfaces are simply in view along the way; looking counts.

        A glance is weaker evidence than standing at the surface, so it retires
        belief at a lower rate -- the search log already expresses that as
        (1 - d), and a passing look gets a smaller d.
        """
        pf = world.object_layer.presence_filter
        containers = world.scene_graph.containers
        if pf is None or not containers:
            return
        d = float(self.cfg.search_glance_detect_prob)
        rng = float(self.cfg.search_glance_range_m)
        frame = world.frame
        K, T_cw = frame.intrinsics.K(), frame.T_cw
        h, w = frame.depth.shape
        for cid, node in containers.items():
            p_cam = T_cw[:3, :3] @ node.center + T_cw[:3, 3]
            z = float(p_cam[2])
            if not (0.3 <= z <= rng):
                continue
            uv = K @ p_cam
            u, v = float(uv[0] / z), float(uv[1] / z)
            if not (0 <= u < w and 0 <= v < h):
                continue
            measured = float(frame.depth[int(v), int(u)])
            if measured > 1e-3 and measured < z - 0.5:
                continue  # something solid between us and the surface
            self.search_log.searched(cid, d)

    def _select_surface(self, world: WorldView, best_frontier):
        """The best mapped surface, if it beats the best frontier on b*d/c.

        Both sides are the same index -- `select_frontier` already returns
        score/path_cost -- so the comparison is like for like, with
        search_frontier_weight naming the one judgement call: what unmapped
        space is worth against a plausible surface.
        """
        if not self.cfg.search_posterior:
            return None
        if not world.scene_graph.containers:
            return None
        agent_xy = world.agent_xy
        cands = build_container_candidates(
            world.scene_graph,
            world.target,
            self.search_log,
            detect_prob=float(self.cfg.search_detect_prob),
            last_known_xy=self._last_known_target_xy(world),
            proximity_len_m=float(self.cfg.search_proximity_len_m),
            proximity_floor=float(self.cfg.search_proximity_floor),
            surface_mass=float(self.cfg.search_surface_mass),
            plane=PLANE,
            affinity_source=self.affinity,
        )
        if not cands:
            return None
        # Drive to a pose you can STAND in, not to the middle of the furniture.
        # A container's centre is inside the desk; the follower ends wherever the
        # navmesh allows, arrival is never registered, and the surface is scored
        # as "never reached" -- a quarter credit -- so it stays top of the list
        # and gets chosen again. Measured before this fix, one episode's entire
        # search was: desk, desk, desk, desk, desk, desk, desk, desk, with its
        # prior decaying 3.20, 2.56, 2.05, 1.64 ... and an unchanged 2.5 m path
        # cost every time. Eight inspections, one surface.
        reachable = []
        for c in cands:
            view = self.viewpoint_planner.approach_viewpoint(c.goal_xy, world.costmap)
            c.goal_xy = np.asarray(
                view if view is not None else nearest_free_xy(world.costmap, c.goal_xy), dtype=float
            )
            reachable.append(c)
        cands = reachable
        # Prefer surfaces in the room the agent is already in. Simulated over
        # this scene: room-grouped order reaches the target in a median 37
        # inspections and 33 m against 45 and 40 m for a plain global argmax,
        # because crossing the house repeatedly is what the global index does
        # once the nearby surfaces are retired.
        room_bonus = float(self.cfg.search_same_room_bonus)
        if room_bonus > 1.0 and world.scene_graph.rooms:
            here = world.scene_graph.room_of_point(agent_xy)
            if here is not None:
                for c in cands:
                    node = world.scene_graph.containers.get(c.ref_id)
                    if node is not None and node.room_id == here.id:
                        c.prior *= room_bonus
        surface = select_candidate(
            cands, self.planner, world.costmap, agent_xy,
            top_n=int(self.cfg.top_n_frontiers),
            min_path_cost_m=float(self.cfg.min_path_cost_m),
        )
        if surface is None or surface.utility is None:
            return None
        beta = float(self.cfg.search_frontier_weight)
        if best_frontier is not None and best_frontier.path_cost:
            frontier_util = beta * (best_frontier.score or 0.0) / best_frontier.path_cost
            if frontier_util >= surface.utility:
                return None
        return surface

    def _last_known_target_xy(self, world: WorldView):
        """Where the target was last believed to be.

        Proximity encodes "objects are moved by someone doing a task, so short
        displacements dominate". This used to return None once absence was
        confirmed, on the reasoning that the premise had been refuted -- and
        with the proximity model of the time it measured better that way.

        It was the model that was wrong, not the premise. Confirming the object
        is not at its old POSE does not refute short displacements; the
        benchmark's in_anchor relocations move a median 0.72 m, so the object is
        usually still within a metre or two of where it was, on a neighbouring
        surface. What made keeping the term look bad was the 0.2 floor, which
        tied every distant candidate together (see SearchConfig). With
        exp(-d/1.0) and no floor, keeping the term takes the true destination
        into the top 5 in 29 of 57 in_anchor cases against 5 of 57 when it is
        dropped, and cross_anchor is unharmed at 7 of 57 either way.

        Surfaces already looked at are retired by the InspectionLog, which is
        the right instrument for "I have ruled this one out" -- a belief the
        prior should not be trying to express a second time.
        """
        best = None
        for track in world.object_layer.tracks(include_blacklisted=True):
            if str(track.label).lower().replace("_", " ") != str(world.target).lower().replace("_", " "):
                continue
            if best is None or track.presence.n_expected > best.presence.n_expected:
                best = track
        if best is None:
            return None
        return world.object_layer.center_of(best)[list(PLANE)]

    def face_surface(self, world: WorldView) -> Optional[str]:
        """Turn to look at a surface before deciding the target is not on it.

        `_mark_surface_searched` multiplies a surface's belief by (1 - 0.8) on
        arrival -- a near-decisive update -- and the frame that update rests on
        is whatever heading the navmesh follower happened to stop at. That is
        the same mistake the candidate path made and fixed: "concluding absence
        from the arrival frame abandoned a bowl that was exactly where the map
        said". Measured in condition D, the search reached the true surface five
        times and converted one of them.

        A few turns are cheap against the ~50 steps an inspection already costs,
        and they are what make the detector's silence about a surface mean
        something.
        """
        if self.search_container is None or world.goal_xy is None:
            return None
        if self.surface_face_turns <= 0:
            return None
        node = world.scene_graph.containers.get(self.search_container)
        if node is None:
            return None
        frame, agent_xy = world.frame, world.agent_xy
        if float(np.linalg.norm(agent_xy - world.goal_xy)) > float(
            self.cfg.search_arrival_m
        ):
            return None  # not there yet; nothing to look at from here

        to_surface = np.asarray(node.center, dtype=float)[list(PLANE)] - agent_xy
        if float(np.linalg.norm(to_surface)) < 1e-3:
            return None
        err = _wrap(float(np.arctan2(to_surface[1], to_surface[0]))
                    - agent_heading(frame.T_wc))
        if abs(err) <= np.radians(15.0):
            self.surface_face_turns = 0  # facing it; this frame is the evidence
            return None
        self.surface_face_turns -= 1
        self.stats["surface_face_turns"] = self.stats.get("surface_face_turns", 0) + 1
        return TURN_RIGHT if err > 0 else TURN_LEFT

    def mark_searched(self, step: int, arrived: bool = True) -> None:
        """Arriving at a surface without the target is a look that did not find
        it -- worth (1 - d), not worth zero and not worth nothing.

        Giving up on the way there is a much weaker look, and is scored as such:
        a place the agent never reached has barely been ruled out, and spending
        full belief on it would retire the very surfaces it failed to inspect.
        """
        if self.search_container is None:
            return
        d = float(self.cfg.search_detect_prob)
        if not arrived:
            d *= float(self.cfg.search_unreached_credit)
        remaining = self.search_log.searched(self.search_container, d)
        self.search_log_events.append(
            {
                "step": int(step),
                "container_id": int(self.search_container),
                "searched": True,
                "arrived": bool(arrived),
                "belief_factor": round(remaining, 4),
            }
        )
        self.search_container = None

    def note_progress(self, world: WorldView) -> None:
        """Restart the pursuit's progress window. A fresh pursuit needs its own,
        or the give-up timer carries over from whatever was pursued before and
        can fire on the new pursuit's very first step from stale position data."""
        self.progress_ref_step = world.step
        self.progress_ref_xy = world.agent_xy.copy()

    def maybe_give_up(self, world: WorldView, portal_ok: bool) -> bool:
        """No displacement for a while means an obstacle the map cannot see --
        below the obstacle band, glass, a sim collision. Abandon this frontier
        rather than push against it forever.

        Returns True if the pursuit was abandoned, in which case the caller goes
        back to exploring. A portal pursuit is exempt and judged on vertical
        progress instead: a switchback staircase barely moves in (x, z) while
        climbing perfectly well.
        """
        if world.step - self.progress_ref_step < 15:
            return False
        if portal_ok or float(np.linalg.norm(world.agent_xy - self.progress_ref_xy)) >= 0.2:
            self.note_progress(world)
            return False
        self.giveup_log.append((
            world.step,
            [round(float(x), 2) for x in self.current_frontier.centroid_xy]
            if self.current_frontier is not None else None,
            [round(float(x), 2) for x in world.agent_xy],
        ))
        self.block(self.current_frontier, 100, world.step)
        self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
        if self.current_frontier is not None:
            self._last_giveup_pt = (self.current_frontier.centroid_xy.copy(),
                                    self.current_frontier.floor)
        self.current_frontier = None
        self.note_progress(world)
        return True

    def retire_pursued(self, world: WorldView, goal: np.ndarray) -> None:
        """A pursuit that ended retires its frontier, whichever way it ended.

        The navigator returns None for arrived-or-unreachable and the FSM then
        drops GOTO_FRONTIER -> EXPLORE. Unless something blocks the frontier the
        very next selection can choose it again, and the agent freezes
        re-selecting it: give-up never fires, because it counts elapsed steps
        *inside* GOTO_FRONTIER and the re-entry resets its timer.

        This used to be guarded by the stub test -- blocking only when the agent
        was still far from the goal. That conflated two jobs, and separating
        them cost a full condition to find out: raising the reach threshold to
        the planner's true stopping radius (correct in itself, see
        `_frontier_reach_m`) removed the block for arrivals between 0.5 m and
        0.9 m and the livelock came straight back. Measured over 96 dynamic
        episodes, D against C0: frontier selections per episode 2.15 -> 20.76,
        surface inspections 3.34 -> 0.88, episodes that ever mapped the object
        at its new pose 52 -> 44, SR 0.458 -> 0.385.

        So the block is unconditional. Retiring a frontier the agent actually
        reached costs nothing -- it has been explored, which is what a frontier
        is for. The reach threshold now decides only two things: how long the
        block lasts, and whether this counts as a give-up point.
        """
        frontier = self.current_frontier
        if frontier is None:
            return
        reached = bool(np.linalg.norm(world.agent_xy - goal) <= self.frontier_reach_m)
        self.block(frontier, 50 if reached else 100, world.step)
        if reached:
            self.stats["frontier_reached"] = self.stats.get("frontier_reached", 0) + 1
            return
        # A frontier the agent could not get to. Mark it as the last give-up
        # point so the "all frontiers blocked" relaxed fallback (which
        # deliberately ignores the blacklist) does not immediately re-pursue the
        # same unreachable stub. An ordinary arrival must NOT be marked this
        # way: it would suppress the fallback everywhere near a place the agent
        # had simply finished exploring.
        self._last_giveup_pt = (frontier.centroid_xy.copy(), frontier.floor)
        self.stats["frontier_stub_block"] = self.stats.get("frontier_stub_block", 0) + 1

