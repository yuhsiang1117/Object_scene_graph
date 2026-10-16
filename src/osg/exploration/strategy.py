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
from typing import Dict, List, Optional

import numpy as np

from ..core.types import FrameData
from ..mapping.costmap import PLANE, Costmap2D, nearest_free_xy
from ..mapping.frontier import Frontier, FrontierExtractor
from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading
from .search_belief import InspectionLog, build_container_candidates, select_candidate
from .storey_belief import StoreyBelief, target_tracks
from ..core.labels import normalize_label, same_label
from .selector import frontier_goal_xy, select_frontier

# WaypointController.act's default arrival tolerance. Named here because
# `frontier_reach_m` is derived from it: the two must agree or an ordinary
# arrival is classified as an unreachable stub.
FRONTIER_ARRIVAL_TOL_M = 0.2


def container_in_view(frame: FrameData, node, max_range_m: float,
                      min_range_m: float = 0.3) -> Optional[float]:
    """Range to a container's centre if it is inside this frame and unoccluded,
    else None. One definition of "in view", shared by the passing glance and
    the close look, so the two cannot drift apart."""
    K, T_cw = frame.intrinsics.K(), frame.T_cw
    h, w = frame.depth.shape
    p_cam = T_cw[:3, :3] @ np.asarray(node.center, dtype=float) + T_cw[:3, 3]
    z = float(p_cam[2])
    if not (min_range_m <= z <= max_range_m):
        return None
    uv = K @ p_cam
    u, v = float(uv[0] / z), float(uv[1] / z)
    if not (0 <= u < w and 0 <= v < h):
        return None
    measured = float(frame.depth[int(v), int(u)])
    if measured > 1e-3 and measured < z - 0.5:
        return None  # something solid between us and the surface
    return z


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
    value_map: object = None
    #: `{layer key: (height, costmap)}` for every storey the agent knows, when
    #: the storey belief is on; None otherwise.
    storeys: Optional[dict] = None


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


# Consecutive selection rounds with no frontier before a storey counts as
# finished (nav_agent asks the operator for another one). See `select`.
EXHAUSTED_ROUNDS = 3


class ExplorationStrategy:
    def __init__(self, cfg, planner, scorer, viewpoint_planner, affinity,
                 stats: dict, profiler) -> None:
        # `cfg` is the exploration group alone: nothing here reads any other.
        self.cfg = cfg.exploration
        # The one exception, read once at construction rather than held: the bar
        # candidates are proposed at. `_anchor_untested` has to agree with
        # ObjectLayer.candidates about whether a track is still believed, and a
        # second threshold of its own would be a way for them to disagree.
        self._anchor_min_presence = float(
            getattr(getattr(getattr(cfg, "scene_graph", None), "presence", None),
                    "min_presence", 0.45)
        )
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
        # Stop anchoring the search on the last known pose once the agent has
        # BEEN there and found nothing. False keeps the anchor forever, which is
        # the shipped behaviour.
        self.drop_proximity_after_absence = bool(
            self.cfg.search_drop_proximity_after_absence
        )
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
        # The second exception, also read once: one storey's height, which is
        # what tells a storey from a landing (`exploration.storey_belief`).
        self.storey_belief: Optional[StoreyBelief] = (
            StoreyBelief(
                self.cfg,
                float(getattr(getattr(cfg, "floor", None), "new_level_m", 1.8)),
                stats,
            )
            if bool(getattr(self.cfg, "storey_belief", False)) else None
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
        self.forced_floor = None
        self._floor_arrived_step, self._floor_seen = 0, None
        if self.storey_belief is not None:
            self.storey_belief.reset()
        # What has already been searched, and how well (C3). A visit multiplies
        # a surface's belief by (1 - d) rather than zeroing it, so a place
        # glanced at from four metres stays plausible and one inspected closely
        # mostly stops being -- the distinction an ignore list cannot make.
        self.search_log = InspectionLog()
        self.search_container: Optional[int] = None
        # Set on arrival, read and cleared by the appearance pick.
        self.arrived_container: Optional[int] = None
        # Which room the surface being inspected belongs to, and how many
        # fruitless arrivals each room has absorbed. The same-room bonus is a
        # prior; this is the likelihood that updates it (see
        # `search_room_saturation`).
        self.search_room_id: Optional[int] = None
        self._room_fruitless: Dict[int, int] = {}
        # The container categories this map holds, fixed once per episode so a
        # set that grows mid-episode cannot change the prior under the agent
        # (and cannot multiply the affinity cache keys).
        self._present: Optional[set] = None
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
        self._glanced: set = set()
        self._glance_min_range: Dict[int, float] = {}
        # Surfaces that have had a real inspection -- an arrival or a close
        # look -- so the opportunistic look does not spend a detour on them.
        self._inspected: set = set()
        # A surface posterior may select another storey.  Only candidates on
        # the active floor are ever converted to viewpoints or sent to the 2D
        # planner; this field hands the other-floor decision to FloorPolicy.
        self.requested_floor: Optional[int] = None
        self.selected_search_floor: Optional[int] = None
        # A storey the AGENT has decided on -- after failed attempts disproved
        # the one it is on -- as opposed to one the posterior argmax chose.
        # Overrides the mass rule and the anchor hold in `_select_surface`,
        # because both are answers to "where might it be" and this is an
        # answer to "where it is not". Cleared on arrival.
        self.forced_floor: Optional[int] = None
        # Storeys the agent's own failed attempts have disproved (nav_agent
        # shares its set). The mass rule may not send the agent back to one:
        # measured (outputs/mf5_pass2_v12 ep1), the agent disproved the upper
        # storey, climbed down, and at the next round the posterior selected
        # the upper storey again -- it climbed back up and ended where it
        # started.
        self.disproved_floors: set = set()
        self._floor_arrived_step: int = 0
        self._floor_seen: Optional[int] = None
        # The storey the last real round found no frontier on, or None. Read
        # by nav_agent: on the robot that is the cue to ask the operator for a
        # storey nobody has mapped (agent.request_new_storey_when_exhausted).
        self._exhausted_floor: Optional[int] = None
        self._empty_rounds: int = 0
        self._empty_floor: Optional[int] = None

    def note_position(self, floor_id: int, agent_xy) -> None:
        """One step's position, for the storey belief's coverage; off, a no-op."""
        if self.storey_belief is not None:
            self.storey_belief.note(floor_id, agent_xy)

    def _knows_another_storey(self, world: WorldView) -> bool:
        """With the storey belief on and a second storey known, every storey
        an undirected switch could find is already scored by the belief, so the
        geometric gate does not get to pick one."""
        return (
            self.storey_belief is not None
            and bool(world.storeys)
            and self.storey_belief.knows_another_storey(world.storeys)
        )

    def storey_exhausted(self, floor_id: int) -> bool:
        """Did the last real round find no frontier on THIS storey?"""
        return self._exhausted_floor is not None and int(self._exhausted_floor) == int(floor_id)

    def steps_on_storey(self, step: int, floor_id: int) -> int:
        """Steps since the first real round on this storey; 0 before one."""
        if self._floor_seen is None or int(self._floor_seen) != int(floor_id):
            return 0
        return int(step) - int(self._floor_arrived_step)

    def force_select_next(self) -> None:
        """Drop the rate limit so the next `select` really runs.

        `select` fires once per `select_every` steps, which is right while the
        agent is exploring steadily and wrong immediately after an attempt
        fails: the few steps before the next candidate commit are exactly the
        ones that have to carry a selection round, because they are the only
        ones the round will get.
        """
        self._last_select_step = -10 ** 9

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
        if world.step - self._last_select_step < max(
            1, int(getattr(self.cfg, "select_every", 5))
        ):
            return None
        self._last_select_step = world.step
        if self._floor_seen is None or int(world.floor_id) != int(self._floor_seen):
            self._floor_seen, self._floor_arrived_step = int(world.floor_id), int(world.step)
        if self.forced_floor is not None and int(self.forced_floor) == int(world.floor_id):
            self.forced_floor = None       # arrived: the directive is spent
            self.requested_floor = None
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
        if self.storey_belief is not None and world.storeys and self.forced_floor is None:
            with self.profiler.timeit("storey_belief"):
                self.requested_floor = self.storey_belief.decide(
                    world.step, world.storeys,
                    target_tracks(world.object_layer, world.target), world.floor_id,
                )
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(
                world.costmap, world.agent_xy,
                floor=world.floor_id,
            )
        # "Finished" is several EMPTY rounds in a row, not one: between rounds
        # the idle agent turns in place and the grid grows, and one empty
        # extraction is what a give-up looks like (its frontier is blocked
        # for 50 steps). Measured on the robot (outputs/20260922_231133):
        # 5 frontiers at step 34, a give-up at 43, none at 64 -- the agent
        # asked for another storey -- and a frontier again from step 88.
        if not frontiers:
            same = self._empty_floor is not None and int(self._empty_floor) == int(world.floor_id)
            self._empty_rounds = self._empty_rounds + 1 if same else 1
            self._empty_floor = int(world.floor_id)
        else:
            self._empty_rounds, self._empty_floor = 0, None
        self._exhausted_floor = (
            int(world.floor_id) if self._empty_rounds >= EXHAUSTED_ROUNDS else None)
        if not frontiers and not bool(self.cfg.search_posterior):
            # Nothing left on this floor is the strongest possible "no near
            # frontier", so the portal gate still gets its chance.
            if self.storey_belief is not None and self.requested_floor is not None:
                floor_switch(None, int(self.requested_floor))
            elif not self._knows_another_storey(world):
                floor_switch(None)
            return None
        # Async scoring request (never blocks); use whatever scores exist now
        self.scorer.request(frontiers, world.scene_graph, world.target, world.keyframes)
        frontier_scores = dict(self.scorer.latest())
        # Optional ASCENT/VLFM-style semantic value.  The image-text scorer
        # paints only the active floor; applying it here keeps remote-floor
        # candidates out of local geometry and leaves the default geometric
        # path byte-for-byte unchanged when value_map is disabled.
        value_map = getattr(world, "value_map", None)
        if value_map is not None:
            value_weight = float(getattr(self.cfg, "value_weight", 1.0))
            value_argmax = bool(getattr(self.cfg, "value_argmax", False))
            value_radius = float(getattr(self.cfg, "value_radius_m", 0.5))
            for frontier in frontiers:
                value = max(
                    0.0,
                    float(value_map.value_at(frontier.centroid_xy, value_radius)),
                )
                if value <= 0.0:
                    continue
                if value_argmax:
                    frontier_scores[frontier.id] = value
                else:
                    base = frontier_scores.get(frontier.id, self.cfg.unscored_prior)
                    frontier_scores[frontier.id] = base * (1.0 + value_weight * value)
        blocked = self._blocked_ids(frontiers, world.step)
        agent_xy = world.agent_xy
        heading_xy = self._heading_xy(world.frame)
        failed: set = set()
        with self.profiler.timeit("frontier_select"):
            best = select_frontier(
                frontiers,
                frontier_scores,
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
        if self.requested_floor is not None:
            requested = int(self.requested_floor)
            try:
                switched = floor_switch(None, requested)
            except TypeError:
                # Compatibility with small external strategies/tests that use
                # the original one-argument callback.
                switched = floor_switch(None)
            if switched:
                return None
            if bool(getattr(self.cfg, "search_surface_when_floor_unreachable", False)):
                # The storey we want exists in the posterior and NOT on the
                # ground: `try_switch` found no portal to it from here. Measured
                # on outputs/osg_authored_15, that is the usual case -- 405
                # requests produced 12 directed attempts, so 97% of them found
                # nothing to drive to.
                #
                # Left alone the round returns a frontier and `_select_surface`
                # has already returned None, so a standing request switches the
                # container posterior OFF for as long as it stands -- on the
                # floor the agent is actually on, with the surfaces it can
                # actually reach. The request is a wish about another storey; it
                # should not also be a veto on searching this one.
                self.stats["floor_unreachable_fallback"] = (
                    self.stats.get("floor_unreachable_fallback", 0) + 1
                )
                surface = self._select_surface(world, best, stay_on_floor=True)
        if surface is not None:
            self.search_container = int(surface.ref_id)
            node = world.scene_graph.containers.get(int(surface.ref_id))
            self.search_room_id = int(node.room_id) if node is not None else None
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
                    "frontier_util": (
                        round(float(self._last_frontier_util), 5)
                        if getattr(self, "_last_frontier_util", None) is not None
                        else None
                    ),
                    "room": self.search_room_id,
                    "floor_key": int(surface.floor_key),
                    "room_bonus": round(
                        self._room_bonus(
                            float(self.cfg.search_same_room_bonus),
                            int(self.search_room_id),
                        ), 3,
                    ) if self.search_room_id is not None else None,
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
                    frontiers, frontier_scores, self.planner, world.costmap,
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
        if self._knows_another_storey(world):
            self.stats["undirected_switch_skipped"] = (
                self.stats.get("undirected_switch_skipped", 0) + 1)
        elif floor_switch(None if best is None else best.path_cost):
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
        for cid, node in containers.items():
            if int(getattr(node, "floor_id", getattr(node, "floor", 0))) != int(
                world.floor_id
            ):
                continue
            z = container_in_view(frame, node, rng)
            if z is None:
                continue
            # The closest range this surface was ever glanced from. A glance
            # from 3.5 m and one from 1.2 m are the same event to the belief
            # arithmetic and very different events to the detector; the record
            # keeps them apart (`glance_ranges`).
            prev = self._glance_min_range.get(int(cid))
            if prev is None or z < prev:
                self._glance_min_range[int(cid)] = float(z)
            self.search_log.searched(cid, d, floor=float(self.cfg.search_glance_floor))
            # Instrumentation only. A glance is applied per KEYFRAME, so a
            # surface the agent lingers near is multiplied many times over; how
            # many, and how much belief survives it, is not otherwise visible in
            # the record. Counted here so the question can be answered with a
            # number instead of an argument.
            self.stats["glance_updates"] = self.stats.get("glance_updates", 0) + 1
            self._glanced.add(int(cid))

    def surface_beliefs(self, world: WorldView) -> Dict[int, float]:
        """How much the search still believes in each container on this floor,
        relative to the best of them (1.0), for a decision that is not a
        selection round -- the close look asking which surface in view is worth
        its budget. Same prior, same survived factor as `_select_surface`, so
        the two cannot disagree about what a surface is worth."""
        if not self.cfg.search_posterior or not world.scene_graph.containers:
            return {}
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
            present=self._ground(world),
        )
        cands = [c for c in cands if int(c.floor_key) == int(world.floor_id)]
        if not cands:
            return {}
        peak = max(float(c.prior) for c in cands)
        if peak <= 0.0:
            return {}
        return {int(c.ref_id): float(c.prior) / peak for c in cands}

    def _storey_vote(self, world: WorldView, cands, stay_on_floor: bool):
        """The container posterior's storey vote: the surfaces on this floor,
        or None when it asked for another storey (`requested_floor`)."""
        # Choose a storey from the posterior before doing any 2D geometry.
        # Path cost is meaningful only within a floor; aggregate the remaining
        # probability mass here and delegate vertical travel to FloorPolicy.
        floor_mass: Dict[int, float] = {}
        floor_n: Dict[int, int] = {}
        for cand in cands:
            floor_mass[cand.floor_key] = floor_mass.get(cand.floor_key, 0.0) + (
                cand.prior * cand.detect_prob
            )
            floor_n[cand.floor_key] = floor_n.get(cand.floor_key, 0) + 1
        # A SUM over surfaces is a count of furniture, not a belief about where
        # the object is. Measured on 00808's prior map with the flat proximity
        # prior this line ships with, for five different targets:
        #
        #     floor 0: 78 surfaces, sum 52.5, mean 0.673
        #     floor 1: 33 surfaces, sum 22.3, mean 0.676
        #
        # The means agree to within half a percent -- the container priors carry
        # essentially NO information about which storey, which is the same
        # result docs/ARCHITECTURE.md already records for affinity ranking. The
        # sum turns that tie into a 2.4x preference for the bigger floor, so an
        # agent upstairs asks to go down every round whatever it is looking for.
        #
        # `mean` makes the score scale-free, and `floor_mass_margin` then
        # requires another storey to be decisively better before the agent pays
        # for the stairs. Together they let the posterior say "no opinion",
        # which on this evidence is the honest answer -- leaving the storey to
        # the mechanisms that do carry floor information: the stale anchor, and
        # the LLM once that anchor has been tested.
        rule = str(getattr(self.cfg, "floor_mass_rule", "sum"))
        if rule == "mean":
            score = {k: v / max(floor_n.get(k, 1), 1) for k, v in floor_mass.items()}
        else:
            score = dict(floor_mass)
        eligible = {k: v for k, v in score.items()
                    if int(k) not in self.disproved_floors or int(k) == int(world.floor_id)}
        if eligible != score:
            self.stats["floor_posterior_disproved_excluded"] = (
                self.stats.get("floor_posterior_disproved_excluded", 0) + 1)
        if not eligible:
            # Every storey with any mass is disproved: there is nowhere the
            # posterior may send the agent. Stay and explore.
            eligible = {int(world.floor_id): 0.0}
        score = eligible
        settle = int(getattr(self.cfg, "empty_storey_settle_steps", 0) or 0)
        if (
            int(world.floor_id) not in score
            and settle > 0
            and int(world.step) - int(self._floor_arrived_step) < settle
        ):
            # THIS storey has no containers and the agent only just got here.
            # That is "unmapped", not "the target is not here": containers are
            # rebuilt from tracks at keyframes, and none has happened yet.
            # Outside the window the shipped rule stands -- with nothing here,
            # anywhere else is better (test_floor_anchor_order). See the
            # config comment for the measurement.
            self.stats["floor_posterior_empty_here"] = (
                self.stats.get("floor_posterior_empty_here", 0) + 1)
            score = {int(world.floor_id): 0.0}
        selected_floor = max(score, key=lambda key: (score[key], -int(key)))
        margin = float(getattr(self.cfg, "floor_mass_margin", 0.0))
        if (
            margin > 0.0
            and int(selected_floor) != int(world.floor_id)
            and float(score.get(int(world.floor_id), 0.0)) > 0.0
            and float(score[selected_floor])
            < float(score[int(world.floor_id)]) * margin
        ):
            self.stats["floor_mass_no_opinion"] = (
                self.stats.get("floor_mass_no_opinion", 0) + 1
            )
            selected_floor = int(world.floor_id)
        self.selected_search_floor = int(selected_floor)
        self.stats["selected_search_floor"] = int(selected_floor)
        if int(selected_floor) != int(world.floor_id) and self._anchor_untested(world):
            # The map still says the object is on THIS floor and nothing has
            # tested that. Leaving now abandons the one hypothesis the prior map
            # actually supports, in favour of an argmax over how many surfaces
            # each storey happens to have. Hold the request and search here.
            self.stats["cross_floor_request_held"] = (
                self.stats.get("cross_floor_request_held", 0) + 1
            )
            selected_floor = int(world.floor_id)
        forced = getattr(self, "forced_floor", None)
        if forced is not None and int(forced) != int(world.floor_id):
            # Failed attempts have disproved this storey (agent.rearm). That
            # outranks a posterior that abstains by design (`mean` + margin)
            # and an anchor hold whose anchor has, by definition, been tested.
            selected_floor = int(forced)
            self.selected_search_floor = int(forced)
            self.stats["selected_search_floor"] = int(forced)
            self.stats["floor_forced_by_failed_attempts"] = (
                self.stats.get("floor_forced_by_failed_attempts", 0) + 1)
        if int(selected_floor) != int(world.floor_id) and not stay_on_floor:
            self.requested_floor = int(selected_floor)
            self.stats["cross_floor_search_requests"] = (
                self.stats.get("cross_floor_search_requests", 0) + 1
            )
            self.search_log_events.append({
                "step": int(world.step),
                "selected_floor": int(selected_floor),
                "current_floor": int(world.floor_id),
                "floor_mass": {
                    str(key): round(float(value), 5)
                    for key, value in sorted(floor_mass.items())
                },
            })
            return None
        if int(selected_floor) == int(world.floor_id):
            self.requested_floor = None
        cands = [c for c in cands if int(c.floor_key) == int(world.floor_id)]
        return cands

    def _select_surface(self, world: WorldView, best_frontier, stay_on_floor: bool = False):
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
        present = self._ground(world)
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
            present=present,
        )
        if not cands:
            return None
        if self.storey_belief is not None:
            # The storey is the belief's call, made in `select`; here the
            # surfaces compete on this floor only.
            cands = [c for c in cands if int(c.floor_key) == int(world.floor_id)]
        else:
            cands = self._storey_vote(world, cands, stay_on_floor)
            if cands is None:
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
            here = world.scene_graph.room_of_point(agent_xy, floor_key=world.floor_id)
            if here is not None:
                for c in cands:
                    node = world.scene_graph.containers.get(c.ref_id)
                    if node is not None and node.room_id == here.id:
                        c.prior *= self._room_bonus(room_bonus, node.room_id)
        surface = select_candidate(
            cands, self.planner, world.costmap, agent_xy,
            top_n=int(self.cfg.top_n_frontiers),
            min_path_cost_m=float(self.cfg.min_path_cost_m),
        )
        if surface is None or surface.utility is None:
            return None
        beta = float(self.cfg.search_frontier_weight)
        # Kept for the log: the number the winning surface had to beat. Without
        # it a run where the search line never hands back is indistinguishable
        # from one where there was nothing to hand back TO.
        self._last_frontier_util = None
        if best_frontier is not None and best_frontier.path_cost:
            frontier_util = beta * (best_frontier.score or 0.0) / best_frontier.path_cost
            self._last_frontier_util = float(frontier_util)
            if frontier_util >= surface.utility:
                # A surface WAS found, costed, and beaten. Distinguishable in
                # the record from "there was nothing to hand back to", which is
                # the other way `_select_surface` returns None and looks the
                # same from outside. On 00808 the posterior selected zero
                # surfaces in 16 episodes and this says whether that is because
                # no candidate survived or because exploration outbid them all.
                self.stats["search_surface_outbid"] = (
                    self.stats.get("search_surface_outbid", 0) + 1
                )
                margin = float(frontier_util) / max(float(surface.utility), 1e-9)
                self.stats["search_surface_outbid_margin_x100"] = max(
                    int(self.stats.get("search_surface_outbid_margin_x100", 0)),
                    int(round(margin * 100)),
                )
                return None
        return surface

    def _anchor_untested(self, world: WorldView) -> bool:
        """Is a target-labelled track on THIS floor still believed and unvisited?

        The condition under which leaving the storey is premature. "Untested"
        means the agent has not been to that pose (`absence_arrivals == 0`) and
        the belief is still above the bar candidates are proposed at -- exactly
        the two signals the rest of the pipeline already uses, so the floor
        question and the candidate question cannot disagree about one track.

        False whenever the feature is off, when nothing target-labelled is
        mapped on this floor, or when there is no presence filter to ask: in all
        three the storey argmax is the only opinion available and it should
        stand.
        """
        if not bool(getattr(self.cfg, "search_floor_requires_anchor_test", False)):
            return False
        layer = world.object_layer
        min_presence = float(self._anchor_min_presence)
        for track in layer.tracks(include_blacklisted=True):
            if not same_label(getattr(track, "label", ""), world.target):
                continue
            if int(getattr(track, "floor_key", world.floor_id)) != int(world.floor_id):
                continue
            if int(getattr(track, "absence_arrivals", 0)) > 0:
                continue
            presence = getattr(track, "presence", None)
            if presence is None or float(presence.p) >= min_presence:
                return True
        return False

    def _last_known_target_xy(self, world: WorldView):
        """Where the target was last believed to be -- while it is still believed.

        Proximity encodes "objects are moved by someone doing a task, so short
        displacements dominate", and for the half of the benchmark that moves a
        median 0.72 m it is worth a great deal.

        This used to return None once absence was confirmed, on the reasoning
        that the premise had been refuted, and that was removed because with the
        proximity model of the time it measured worse. The removal went one step
        too far: the premise is refuted, and the right question is WHEN.

        Measured over the 114 relocations, with the true surface counted as one
        whose footprint is within a metre of where the object landed -- close
        enough that arriving at it puts the object in view -- inspections a
        greedy search needs to reach it:

                              reaches it   median   within 10
            in_anchor   prox    23/57         2        23
            in_anchor   flat    15/57        17         5
            cross       prox    12/57        22         4
            cross       flat    20/57        16         9

        The two halves want opposite models, and a mixture cannot serve both --
        swept, both flat and two-scale, every setting lands on one frontier. But
        the agent does not have to guess which half it is in. Its own presence
        belief answers: while the map still believes the object is where it left
        it, short displacements are the live hypothesis; once that belief has
        fallen below the bar the agent proposes candidates at, the hypothesis is
        dead and what remains is coverage -- which a flat prior over
        `prior * d / cost` already is, since a constant prior orders surfaces by
        travel cost alone.

        `drop_proximity_below` is 0.0 by default, which keeps the term forever
        and is the shipped behaviour.
        """
        best = None
        for track in world.object_layer.tracks(include_blacklisted=True):
            if not same_label(track.label, world.target):
                continue
            if best is None or track.presence.n_expected > best.presence.n_expected:
                best = track
        if best is None:
            return None
        if self.drop_proximity_after_absence and best.absence_arrivals > 0:
            # The agent went to that pose and the object was not there.
            # Anchoring the search on it now is worse than not anchoring at all:
            # on the cross-anchor half, proximity reaches the true surface within
            # ten inspections 4 times in 57 against a nearest-first sweep's 9.
            #
            # The trigger is the ARRIVAL, not the belief. A first attempt used
            # `presence.p < min_presence` and fired in 56% of in_anchor episodes
            # against the 30% predicted, because presence also decays from
            # ordinary missed expectations while the agent merely walks past --
            # so it dropped the anchor on the half that needs it, and cost three
            # episodes on the first scene before the run was stopped.
            self.stats["search_proximity_dropped"] = (
                self.stats.get("search_proximity_dropped", 0) + 1
            )
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
        if int(getattr(node, "floor_id", getattr(node, "floor", 0))) != int(
            world.floor_id
        ):
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

    def _ground(self, world: WorldView):
        """The container categories this map actually contains, or None when
        grounding is off.

        Computed once per episode and cached. The LLM affinity provider is
        re-scoped to the same set, so an unknown target is ranked over the
        surfaces this house has rather than over the fourteen the codebase knows
        about -- see `affinity_grounded` for why that distinction decides
        whether scene 00848's prior is peaked or flat.
        """
        if not bool(self.cfg.affinity_grounded):
            return None
        if self._present is None:
            # From the TRACKS, not from `scene_graph.containers`. Containers are
            # rebuilt per keyframe from whatever geometry currently qualifies,
            # so at the first surface selection the container set is a subset of
            # the house -- and grounding against a subset would drop categories
            # that do exist, changing the two scenes this is meant to leave
            # untouched. The track set is the prior map, fixed from reset.
            labels = {
                normalize_label(getattr(t, "label", ""))
                for t in world.object_layer.tracks(include_blacklisted=True)
            }
            self._present = {l for l in labels if l}
            if self.affinity is not None and hasattr(self.affinity, "ground"):
                self.affinity.ground(self._present)
            self.stats["affinity_present_categories"] = len(self._present)
        return self._present

    def _room_bonus(self, bonus: float, room_id: int) -> float:
        """The same-room bonus, discounted by what the room has already failed
        to produce.

        `search_same_room_bonus` asserts "the object is in the room I am in".
        That is a reasonable prior and it was never conditioned on anything: in
        a room with 68 containers it holds every one of them above every
        frontier for the whole episode. Each fruitless arrival is evidence
        against the assertion, so it decays multiplicatively -- the same form
        the surface belief itself uses, one level up the hierarchy.

        Floored, so that by default a disappointing room becomes ordinary rather
        than actively repellent; a room the agent has searched is not worse than
        one it has never seen.
        """
        rate = float(self.cfg.search_room_saturation)
        n = int(self._room_fruitless.get(int(room_id), 0)) - int(
            self.cfg.search_room_saturation_free
        )
        if rate <= 0.0 or n <= 0:
            return bonus
        floor = float(self.cfg.search_room_saturation_floor)
        return max(floor, bonus * (1.0 - rate) ** n)

    def mark_searched(self, step: int, arrived: bool = True) -> None:
        """Arriving at a surface without the target is a look that did not find
        it -- worth (1 - d), not worth zero and not worth nothing.

        Giving up on the way there is a much weaker look, and is scored as such:
        a place the agent never reached has barely been ruled out, and spending
        full belief on it would retire the very surfaces it failed to inspect.
        """
        if self.search_container is None:
            return
        # The moment DualMap runs its local inquiry: it has reached the anchor
        # its global text match chose, and asks which nearby object looks most
        # like the query. Published rather than acted on here, because
        # committing to a candidate is the candidate policy's job; it is
        # consumed once and cleared.
        if arrived:
            self.arrived_container = int(self.search_container)
        d = float(self.cfg.search_detect_prob)
        if not arrived:
            d *= float(self.cfg.search_unreached_credit)
        remaining = self.search_log.searched(self.search_container, d)
        if arrived:
            self._inspected.add(int(self.search_container))
        # Only an arrival is evidence about the ROOM. A surface the agent gave
        # up on says nothing about its neighbours, and charging the room for it
        # would push the agent out of rooms it never actually searched.
        if arrived and self.search_room_id is not None:
            rid = int(self.search_room_id)
            self._room_fruitless[rid] = self._room_fruitless.get(rid, 0) + 1
        self.search_room_id = None
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

    @property
    def inspected(self) -> set:
        return self._inspected

    @property
    def glance_ranges(self) -> Dict[int, float]:
        return dict(self._glance_min_range)

    def close_looked(self, cid: int, step: int, detected: bool) -> None:
        """A close look ended (agent/close_look.py). Silence from a facing pose
        on the 1.5 m ring is worth what an arrival is worth; a look that found
        the target retires nothing, the candidate path takes it from here."""
        factor = None
        if not detected:
            factor = self.search_log.searched(int(cid), float(self.cfg.search_detect_prob))
        self._inspected.add(int(cid))
        self.search_log_events.append(
            {
                "step": int(step),
                "container_id": int(cid),
                "close_look": True,
                "detected": bool(detected),
                "belief_factor": None if factor is None else round(float(factor), 4),
            }
        )

    def survival_report(self) -> dict:
        """How much of the search space is still believed in.

        `retired` counts surfaces whose surviving belief factor has fallen below
        a tenth -- effectively out of the search whether or not the agent ever
        went to one.
        """
        factors = list(self.search_log.survived.values())
        return {
            "glance_containers": len(self._glanced),
            "surfaces_touched": len(factors),
            "surfaces_retired": sum(1 for f in factors if f < 0.1),
            "surface_factor_min": round(min(factors), 5) if factors else None,
            "surface_factor_median": round(sorted(factors)[len(factors) // 2], 4) if factors else None,
        }

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
        window = int(getattr(self.cfg, "giveup_after_steps", 15) or 15)
        min_m = float(getattr(self.cfg, "giveup_min_progress_m", 0.2))
        if world.step - self.progress_ref_step < window:
            return False
        if portal_ok or float(np.linalg.norm(world.agent_xy - self.progress_ref_xy)) >= min_m:
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
