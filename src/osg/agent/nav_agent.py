"""NavAgent: the full pipeline behind a single act(frame) -> action call.

State machine:
    INIT (360 scan) -> EXPLORE <-> GOTO_FRONTIER
                           |  candidate found
                           v
                  GOTO_VERIFY_VIEW -> VERIFYING --accept--> APPROACH -> STOP
                           ^                |                  |
                           |                +--reject--> blacklist, EXPLORE
                           +---------------------(retreat if visibility lost)

APPROACH walks toward the verified object one step at a time, checking with
the detector on every step: stop when the detection's bbox is large enough
(close and clearly visible) or, if a step carries the agent out of view,
retreat to the last pose that was confirmed visible and stop there. This
targets HM3D's success metric directly — distance-to-goal is measured
against the episode's view_points set (poses from which the object is
actually visible), not against the object's raw 3D position, so a
distance-only stopping rule can land just outside that set even when the
agent is standing right next to the object (see docs/DESIGN_AND_ROADMAP.md
P0->P1 history: three distance-based strategies all stalled at dtg
0.107-0.147 m).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..core.config import resolve_navigation
from ..core.profiler import Profiler
from ..core.types import Detection, FrameData
from ..exploration.async_scorer import AsyncScorer
from ..exploration.ascent_selector import FrontierCommitState, select_frontier_ascent
from ..exploration.selector import frontier_goal_xy, select_frontier
from ..graph.scene_graph import ROOM_IDS_PER_FLOOR, SceneGraph
from ..mapping.costmap import PLANE, UNKNOWN, Costmap2D
from ..mapping.floor_stack import FloorLayer, FloorStack
from ..mapping.frontier import Frontier, FrontierExtractor
from ..mapping.room_seg import VoronoiRoomSegmenter
from ..mapping.stairs import StairDetector
from ..mapping.value_map import ValueMap2D
from ..objects.object_layer import ObjectLayer
from ..perception.detector import Detector
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from ..planning.controller import WaypointController, agent_heading
from ..planning.escape import ActionHistoryEscape
from ..planning.pointnav_driver import PointNavDriver, build_pointnav
from ..planning.planner import PlanResult, StraightLinePlanner
from ..planning.voronoi_planner import HybridVoronoiPlanner
from ..verification.viewpoint import ViewpointPlanner

STOP_ACTION = "stop"
TURN_ACTION = "turn_left"
FORWARD_ACTION = "move_forward"


class State(Enum):
    INIT = "init"
    EXPLORE = "explore"
    GOTO_FRONTIER = "goto_frontier"
    GOTO_VERIFY_VIEW = "goto_verify_view"
    VERIFYING = "verifying"
    APPROACH = "approach"
    CLIMB = "climb"
    DONE = "done"


class NavAgent:
    def __init__(
        self,
        cfg,
        detector: Detector,
        scorer: AsyncScorer,
        verifier,  # always None in old-algorithm mode (no VLM verifier); kept for call-site compat
        target_category: str,
        keyframe_dir: Optional[str] = None,
        profiler: Optional[Profiler] = None,
        nav_fn=None,
        reachable_fn=None,
        image_text=None,
        ranker=None,
        room_classifier=None,
        floor_planner=None,
        pointnav=None,
        stair_segmenter=None,
    ) -> None:
        self.cfg = cfg
        # ASCENT-style forced-choice frontier ranker, or None for the existing
        # async per-frontier scorer. Built by the runner so this stays testable.
        self.ranker = ranker
        self._last_rank_step = -10_000
        # Places365 room typing. Without it RoomNode.label stays None in every
        # configuration that does not run an LLM scorer, which is all of them.
        self.room_classifier = room_classifier
        self._room_votes: list = []
        self._room_vote_cap = 500
        # Coarse level of the cascade. None keeps floor changes purely
        # geometric (stair prior x explored boost).
        self.floor_planner = floor_planner
        self._floor_goal_dir: int = 0
        self.detector = detector
        self.scorer = scorer
        self.verifier = verifier
        # Habitat-navmesh driving (old-stack alignment): nav_fn(goal_xy) returns
        # the next discrete action toward goal_xy on Habitat's navmesh, or None
        # when arrived/unreachable. When set, it replaces the costmap planner +
        # controller for all goal-following. The costmap is still built (for
        # frontier extraction / scene graph), only navigation switches.
        self._nav_fn = nav_fn
        self._reachable_fn = reachable_fn
        self.navigation = resolve_navigation(cfg.agent)
        self._use_navmesh = self.navigation == "navmesh" and nav_fn is not None
        # ASCENT's mover: sensor-only, same contract as nav_fn. Built here (not
        # in the runner) so it shares the agent's lifetime and gets reset with it.
        # Shared across episodes by the runner (a NavAgent is built per
        # episode, and the checkpoint is 34 MB -- reloading it 2000 times is
        # pure waste). Built here only when a caller did not supply one, which
        # is what the tests do.
        self.pointnav: Optional[PointNavDriver] = pointnav
        if self.navigation == "pointnav" and self.pointnav is None:
            self.pointnav = build_pointnav(cfg)
        # The one thing every goal-following call site asks: is there a mover
        # that plans for itself? navmesh and pointnav both answer yes and share
        # the FSM shape (direct-to-object approach, no viewpoint
        # pre-positioning), so the A/B between them isolates exactly the
        # privileged channels. costmap answers no and keeps planner+controller.
        self._driver = self._nav_fn if self._use_navmesh else self.pointnav
        self._direct_approach = self._driver is not None
        # Whether a frontier must be A*-reachable on the costmap before the
        # agent will pursue it. For a self-planning mover the plan is computed
        # and then thrown away -- `_follow_path` never reads `_current_path` --
        # so the gate is a costmap planner vetoing frontiers for a mover that
        # does not use the costmap. ASCENT gates on nothing: the waypoint goes
        # straight to the network (ascent_policy.py:705). Kept ON wherever the
        # planner actually drives, which is every pre-S8 configuration.
        self._frontier_gate = bool(
            getattr(cfg.agent, "frontier_reachability_gate", True)
        ) or self._driver is None
        self._straight_planner = StraightLinePlanner()
        # How often selection may run at all, and how often a pursuit already
        # under way is reconsidered. ASCENT: both every step.
        self._select_every = int(getattr(cfg.exploration, "select_every", 5))
        self._reselect_every = int(getattr(cfg.exploration, "reselect_every", 0))
        # RedNet MPCAT40 stair segmentation, shared across episodes by the
        # runner like the detector -- the checkpoint is 626 MB.
        self.stair_segmenter = stair_segmenter
        # ASCENT's stair traversal (ascent_policy.py:1075-1112). Off by default:
        # it replaces the fixed overshoot goal outright, and every pre-S31
        # cross-floor number was measured on that goal.
        self._climb_carrot = bool(getattr(cfg.agent, "climb_carrot", False))
        self._down_look_every = int(getattr(cfg.agent, "down_look_every", 0))
        self._escape = (
            ActionHistoryEscape(int(getattr(cfg.agent, "escape_window", 0)))
            if int(getattr(cfg.agent, "escape_window", 0)) > 0
            else None
        )
        # Terminal-view verification mode: skip the pre-approach best_crop VLM
        # call and instead verify the live close-up frame at the STOP decision
        # (see _do_approach). Requires a verifier; no-op when verifier is None.
        self._terminal_verify = (
            verifier is not None and bool(getattr(cfg.verification, "terminal", False))
        )
        # Continuous approach re-check (ASCENT's _double_check_goal). Uses the
        # value-map score, so it needs the image-text model, not the verifier.
        self._approach_recheck = (
            image_text is not None
            and bool(getattr(cfg.verification, "approach_recheck", False))
        )
        self._approach_recheck_thresh = float(
            getattr(cfg.verification, "approach_recheck_thresh", 0.0)
        )
        # ASCENT describes a frontier from the frame that first revealed it,
        # not from a spatial query against the accumulated map. Off by default.
        self.frontier_semantics = None
        self._frontier_desc = str(getattr(cfg.exploration, "frontier_desc", "graph"))
        if self._frontier_desc in ("frame", "frame_objects"):
            from ..exploration.frontier_semantics import FrontierSemantics
            self.frontier_semantics = FrontierSemantics(
                match_radius_m=float(
                    getattr(cfg.exploration, "frontier_desc_match_m", 1.0)),
                fov_rad=np.radians(
                    float(getattr(getattr(cfg, "eval", None), "hfov_deg", 79.0))),
                max_range_m=float(
                    getattr(getattr(cfg, "mapping", None), "max_range_m", 5.0)),
            )
        self.profiler = profiler or Profiler()
        # Debug hook: if set, called with (frame, dets) every keyframe right
        # after the detections that feed object_layer.update() are computed
        # -- lets diagnostics observe exactly what the scene graph is built
        # from without duplicating the keyframe-timing logic. None by default
        # (zero cost, never called).
        self.on_keyframe_detections = None
        # Image-text scorer behind the semantic value map; None when the value
        # map is off, so no model is loaded. Set before the floor stack, which
        # gives every floor a value map only when this exists.
        self.image_text = image_text

        # One costmap + planner per storey. With mapping.multi_floor off the
        # band is infinite, so exactly one layer ever exists and `costmap` /
        # `planner` below resolve to it -- identical to the single-map agent.
        self.floors = FloorStack(
            self._make_floor_layer,
            band_m=(
                float(getattr(cfg.mapping, "floor_band_m", 0.9))
                if getattr(cfg.mapping, "multi_floor", False)
                else float("inf")
            ),
            commit_steps=int(getattr(cfg.mapping, "floor_commit_steps", 4)),
        )
        if str(getattr(cfg.exploration, "extractor", "wfd")) == "contour":
            from ..mapping.contour_frontier import ContourFrontierExtractor

            self.frontier_extractor = ContourFrontierExtractor(
                area_thresh_m2=float(getattr(cfg.exploration, "area_thresh_m2", 1.5)),
                agent_radius_m=cfg.agent.agent_radius,
            )
        else:
            self.frontier_extractor = FrontierExtractor(
                min_cells=cfg.exploration.frontier_min_cells,
                dedup_m=cfg.exploration.frontier_dedup_m,
            )
        # ASCENT's frontier path has two independent halves: the ranking (value
        # argmax, no path cost) and the commitment (retire frontiers that are
        # chosen without being reached). `selector` picks the ranking;
        # `frontier_commit` adds the commitment to either ranking, which is what
        # makes them separable in an A/B -- measured together they came out at
        # net -3 with no way to say which half paid for it.
        self._ascent_rank = str(getattr(cfg.exploration, "selector", "utility")) == "ascent"
        self.commit_state = (
            FrontierCommitState(quantise_m=cfg.exploration.frontier_dedup_m)
            if self._ascent_rank or bool(getattr(cfg.exploration, "frontier_commit", False))
            else None
        )
        self.room_segmenter = VoronoiRoomSegmenter(
            min_room_radius_m=cfg.scene_graph.room_min_radius_m,
            door_width_m=cfg.scene_graph.room_door_width_m,
            erode_iters=int(getattr(cfg.scene_graph, "room_erode_iters", 6)),
            min_room_cells=int(getattr(cfg.scene_graph, "min_room_cells", 60)),
        )
        self.object_layer = ObjectLayer(
            assoc_score_thresh=cfg.scene_graph.assoc_score_thresh,
            assoc_depth_gate_m=cfg.scene_graph.assoc_depth_gate_m,
            assoc_category_gate=cfg.scene_graph.assoc_category_gate,
            min_obs_for_refine=cfg.scene_graph.min_obs_for_refine,
            refine_every=cfg.scene_graph.refine_every,
            refine_max_center_move_m=cfg.scene_graph.refine_max_center_move_m,
            link_dist_m=cfg.scene_graph.link_dist_m,
            min_det_score=cfg.scene_graph.min_det_score,
            min_det_bbox_px=cfg.scene_graph.min_det_bbox_px,
            confirm_baseline_m=cfg.scene_graph.confirm_baseline_m,
            repeat_view_discount=cfg.scene_graph.repeat_view_discount,
            max_range_m=cfg.mapping.max_range_m,
            fp_disable_radius_m=getattr(cfg.scene_graph, "fp_disable_radius_m", 0.5),
            cloud_stride=int(getattr(cfg.scene_graph, "cloud_stride", 4)),
            cloud_cap=int(getattr(cfg.scene_graph, "cloud_cap", 2000)),
        )
        self.scene_graph = SceneGraph()
        self.keyframes = KeyframeStore(save_dir=keyframe_dir)
        self.kf_selector = KeyframeSelector(
            cfg.scene_graph.keyframe_trans_m, cfg.scene_graph.keyframe_rot_deg
        )
        # Stair detection is what actually lets the agent leave a floor; the
        # per-floor maps are inert without it. Gated on multi_floor since a
        # stair frontier is only useful once the destination has its own map.
        self.stair_detector = (
            StairDetector(
                resolution_m=cfg.mapping.resolution_m,
                max_range_m=cfg.mapping.max_range_m,
                min_hits=int(getattr(cfg.exploration, "stair_min_hits", 1)),
                min_cells=int(getattr(cfg.exploration, "stair_min_cells", 25)),
                up_mode=str(getattr(cfg.agent, "stair_up_mode", "detector")),
            )
            if getattr(cfg.mapping, "multi_floor", False)
            else None
        )
        # Room prior over the objects already mapped near a frontier. LLM-free:
        # the knowledge graph is a joint distribution over (object, room), so
        # nearby object labels imply a room type without asking a model to name
        # one. None when disabled or the priors are missing.
        self.knowledge = self._load_knowledge()
        if self.ranker is not None and getattr(self.ranker, "kg", None) is None:
            # ASCENT states the room-to-goal priors inside the prompt rather
            # than applying them as a multiplier outside it, so the ranker needs
            # the graph even when the multiplier is off.
            self.ranker.kg = self.knowledge or self._load_knowledge(force=True)
        self.controller = WaypointController(forward_m=cfg.agent.forward_m)
        self.viewpoint_planner = ViewpointPlanner(list(cfg.verification.ring_radii_m))

        self.reset(target_category)

    def _make_floor_layer(self, key: int, floor_y: float) -> FloorLayer:
        """Everything scoped to one storey. A per-floor planner also gives each
        floor its own Voronoi skeleton cache, which is keyed on costmap shape
        and occupancy counts and would otherwise thrash on every floor change."""
        cfg = self.cfg
        layer = FloorLayer(
            key=key,
            floor_y=floor_y,
            costmap=Costmap2D(resolution=cfg.mapping.resolution_m),
            # GVG Voronoi (medial-axis) navigation ported from
            # ObjectSceneGraph_old, with a grid-A* fallback for early/tiny maps
            # (one planner so frontier selection and path planning share it).
            planner=HybridVoronoiPlanner(
                collision_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
                goal_near_m=getattr(cfg.exploration, "voronoi_goal_near_m", 0.7),
                inflate_radius_m=cfg.agent.agent_radius + cfg.mapping.inflate_margin_m,
            ),
        )
        # Per floor, sharing that floor's costmap frame, so a value read at a
        # world point always refers to the map the agent is standing on.
        if getattr(self, "image_text", None) is not None:
            layer.value_map = ValueMap2D(
                layer.costmap, max_depth_m=cfg.mapping.max_range_m
            )
        return layer
        return layer

    # The agent reads `costmap` / `planner` in a dozen places and tests reach in
    # to mutate them; routing both through the current floor keeps every one of
    # those call sites working unchanged.
    @property
    def costmap(self) -> Costmap2D:
        return self.floors.current().costmap

    @property
    def planner(self) -> HybridVoronoiPlanner:
        return self.floors.current().planner

    @property
    def selection_planner(self):
        """The planner that decides whether a frontier is worth pursuing.

        Identical to `planner` unless the reachability gate is off, in which
        case it is `StraightLinePlanner` -- see `_frontier_gate` in __init__ and
        the class docstring for why a discarded A* path must not veto a
        frontier.
        """
        return self.planner if self._frontier_gate else self._straight_planner

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.state = State.INIT
        self.step_count = 0
        if self.pointnav is not None:
            self.pointnav.reset()
        if self._escape is not None:
            self._escape.reset()
        if self.commit_state is not None:
            # Retirements and selection counts are per-episode; carrying them
            # across would retire frontiers in a scene that has never been seen.
            self.commit_state.reset()
        if self.floor_planner is not None:
            self.floor_planner.reset()
        self._floor_goal_dir = 0
        if self.ranker is not None:
            # The runner shares one ranker across episodes, so its counters are
            # cumulative unless zeroed here.
            self.ranker.reset()
        self._last_rank_step = -10_000
        self._scan_steps_left = (
            int(round(360.0 / self.cfg.agent.turn_deg)) if self.cfg.agent.initial_scan else 0
        )
        self._floor_y: Optional[float] = None
        # |current standing height - _floor_y|, updated every step. Diagnostic
        # for the cross-floor frame rejection below.
        self._off_plane_m: float = 0.0
        self._kf_count = 0
        self._current_path: Optional[np.ndarray] = None
        self._current_frontier: Optional[Frontier] = None
        # Location-keyed blacklist: frontier ids are reassigned on every
        # extraction, so blocking must be spatial to persist. [(xy, until)]
        self._blocked_frontier_pts: list = []
        # Centroid of the frontier the agent most recently gave up on: excluded
        # from the "all frontiers blocked" fallback so the agent doesn't
        # immediately re-pursue the dead-end it just abandoned.
        self._last_giveup_pt: Optional[np.ndarray] = None
        # A frontier is only genuinely "reached" if we end up within this of its
        # goal. The controller reports None (arrived) whenever the planned path
        # terminates within its arrival tolerance of the agent -- which also
        # happens for a degenerate stub path when the frontier is unreachable
        # (goal snapped to a nearby node). This threshold separates the two.
        #
        # It must not be tighter than the mover's own arrival radius, or every
        # genuine arrival is misread as a stub. pointnav reports None as soon as
        # it is within `pointnav_stop_radius` (0.9 m, VLFM's default), which is
        # outside 0.5 m: measured on a smoke run, 8 of one episode's arrivals
        # were blacklisted this way, retiring frontiers the agent had actually
        # reached and sending it back to re-explore behind itself.
        self._frontier_reach_m = 0.5
        if self.pointnav is not None:
            self._frontier_reach_m = max(
                self._frontier_reach_m, float(self.pointnav.stop_radius)
            )
        self._candidate_id: Optional[int] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._last_select_step = -100
        self._goto_deadline = 10**9
        self._progress_ref_step = 0
        self._progress_ref_xy = np.zeros(2)
        self._frontier_ref_dist: Optional[float] = None
        self._target_obj_xy: Optional[np.ndarray] = None
        # Diagnostic: the observed surface point nearest the agent at commit
        # time -- what ASCENT navigates to instead of the fitted centre
        # (object_point_cloud_map.py:127-130). Recorded, not acted on.
        self._target_cloud_xy: Optional[np.ndarray] = None
        self._agent_xy: Optional[np.ndarray] = None
        # Image-text score of the latest frame, and the best seen since the
        # current approach began -- ASCENT's _blip_cosine / _double_check_goal.
        self._last_itm: float = 0.0
        self._approach_itm_max: float = 0.0
        self._approach_itm_n: int = 0
        self._went_to_best_cam = False
        self._center_turns = 0  # centering turns spent on the current candidate
        # APPROACH state: path-goal cache is separate from _goal_xy/_current_path
        # used by GOTO_FRONTIER/GOTO_VERIFY_VIEW because APPROACH switches
        # between an "advance toward the object" goal and a "retreat to the
        # last visible pose" goal within the same episode phase.
        self._path_goal: Optional[np.ndarray] = None
        self._approach_last_good_xy: Optional[np.ndarray] = None
        self._approach_steps_left = 0
        self._approach_start_step = 0
        self._pitch_ticks = 0
        self._last_down_look_step = -10 ** 9
        self._carrot_xy: Optional[np.ndarray] = None
        self._carrot_disable_end = False
        self._climb_paused_steps = 0
        self._climb_last_dist: Optional[float] = None
        self.stats = {"plan_ok": 0, "plan_fail": 0, "select_none": 0, "select_ok": 0}
        # Step at which a target candidate first cleared the admission gates
        # (None if the episode never found one). This is the exploration-
        # efficiency signal every A/B is judged on -- unlike SR it moves on
        # almost every episode, so it stays informative at the 50-episode A/B
        # sample size where SR has a ~7pp standard error.
        self.steps_to_first_candidate: Optional[int] = None
        # Staircases that failed to carry the agent to another floor. Spatial,
        # like _blocked_frontier_pts, because stair components are re-derived
        # from the hit grids every selection round and get new ids each time.
        self._disabled_stairs: list = []
        self._climb_goal_xy = np.zeros(2)
        self._climb_fallback_xy = None
        self._climb_centroid_xy = np.zeros(2)
        self._climb_cells = None
        self._climb_cells_xy = None
        self._climb_from_key = -1
        self._climb_start_y = 0.0
        self._det_key = None
        self._det_cache: list = []
        # Closest the agent has got to the target's surface this approach;
        # the terminal rule stops when this stops improving.
        self._terminal_min_d = float("inf")
        self._terminal_last_xy = None
        self._terminal_stalls = 0
        self._climb_ref_y = 0.0
        self._climb_ref_step = 0
        self._climb_deadline = 10**9
        self.state_log = []
        self.frontier_select_log: list = []
        self.giveup_log: list = []
        # Calibration data for approach_stop_bbox_px (P1c): every bbox_px
        # observed during APPROACH, plus why the episode's approach ended.
        self.approach_bbox_log: list = []
        self.approach_stop_reason: Optional[str] = None
        # Best image-text score seen during the approach that ended in a stop.
        # Recorded even when the re-check is disarmed, to calibrate it.
        self.approach_recheck_max: Optional[float] = None
        # Approach-navigation diagnostics (for the terminal approach that ends
        # the episode): why the agent stopped short of a correctly-mapped
        # target. Split path_consumed into planner-no-path vs controller-arrived
        # and record the approach geometry. See scripts/analyze_approach.py.
        self.approach_diag: dict = {}
        self._last_follow_none_reason: Optional[str] = None
        self.kf_selector.reset()
        self.controller.reset()
        self.detector.set_vocabulary(
            [self.target.replace("_", " ")] + list(self.cfg.detector.vocabulary)
        )
        # Surface points are accumulated only for the episode target, so the
        # clouds stay bounded by one category however cluttered the scene.
        self.object_layer.keep_cloud_labels = {
            self.target, self.target.replace("_", " ")
        }

    # ------------------------------------------------------------------- act

    def act(self, frame: FrameData) -> str:
        self.step_count += 1
        self._det_key = None  # new step: the cached detections are stale
        prev_state = self.state
        with self.profiler.timeit("control_loop"):
            action = self._act_inner(frame)
        if self.state != prev_state:
            self.state_log.append((self.step_count, self.state.value))
        if self._escape is not None and self.state is not State.DONE:
            # Deliberate deviation from ASCENT, which overrides the action
            # unconditionally (ascent_policy.py:595-606): a 30-turn history
            # would there flip the agent's own terminal STOP and the episode
            # could never end. Exempt the STOP that DONE has just committed;
            # every other action, STOP included, still goes through the guard.
            #
            # An override does leave the pointnav mover's `prev_actions` holding
            # what it CHOSE rather than what the simulator executed, for one
            # step. ASCENT has the same gap (its wrapper stores the action
            # before the override, pointnav_policy.py:127), and it is
            # self-correcting: the override only fires out of a state the
            # recurrent memory has already failed to escape.
            action = self._escape(action)
        self._last_action = action
        return action

    def _act_inner(self, frame: FrameData) -> str:
        # The pointnav mover reads depth + pose directly, so it needs this
        # step's frame before any goal-following call can use it.
        if self.pointnav is not None:
            self.pointnav.observe(frame)
        # Height of the surface the agent is standing on. habitat puts the
        # sensor at [0, camera_height, 0] on an agent of that same height (see
        # sim/habitat_env.py), so this is exact, not an estimate. look_up /
        # look_down rotate the sensor about its mount and leave it unchanged.
        y_obs = float(frame.camera_position[1] - self.cfg.agent.camera_height)
        # Ground-plane position of this step, so code reached without a frame
        # in scope (_check_candidates) can still ask "nearest to me".
        self._agent_xy = frame.camera_position[list(PLANE)].copy()
        # Freeze the floor stack while traversing a staircase, matching
        # ASCENT's structure: its floor index moves only when the agent leaves
        # the stairs, never from continuous height clustering.
        layer = self.floors.observe(
            y_obs, self.step_count,
            frozen=self._floor_frozen(frame),
        )
        # The active floor's own height, so after a floor change the costmap is
        # written relative to the floor it actually belongs to. With
        # multi_floor off there is one layer whose floor_y is the start height,
        # reproducing the previous fixed-_floor_y behaviour exactly.
        self._floor_y = layer.floor_y
        self._off_plane_m = abs(y_obs - self._floor_y)
        self.stats.update(self.floors.stats())

        # Drop frames that belong to no floor's map. A frame captured between
        # storeys back-projects with rel_h reading 1-3 m, so its geometry lands
        # on the current floor's grid as obstacle/free noise and corrupts
        # frontier extraction.
        #
        # With per-floor maps the condition is "on a staircase" (the height
        # matches no known floor and has not settled into a new one). Without
        # them the only available test is a fixed height threshold, which must
        # be smaller than a storey and therefore also fires on a step or ramp
        # WITHIN a floor -- measured at 471 of 500 steps dropped on a 0.42 m
        # rise, blinding the agent for the rest of the episode. That is the
        # cost of running floor_reject_m without multi_floor.
        if getattr(self.cfg.mapping, "multi_floor", False):
            off_map = self.floors.in_transit()
        else:
            reject_m = getattr(self.cfg.mapping, "floor_reject_m", 0.0)
            off_map = reject_m > 0.0 and self._off_plane_m > reject_m
        if off_map:
            self.stats["frames_off_plane"] = self.stats.get("frames_off_plane", 0) + 1
        else:
            with self.profiler.timeit("costmap"):
                self.costmap.update(
                    frame,
                    floor_y=self._floor_y,
                    obstacle_low=self.cfg.mapping.obstacle_low_m,
                    obstacle_high=self.cfg.mapping.obstacle_high_m,
                    max_range=self.cfg.mapping.max_range_m,
                    stride=self.cfg.mapping.depth_stride,
                )
            self._update_value_map(frame, layer)
        self.controller.observe_progress(frame.T_wc, self._last_action, self.costmap, self.step_count)
        if self.controller.stuck:
            self.controller.stuck = False
            self._current_path = None  # force replan

        is_kf = self.kf_selector.is_keyframe(frame.T_wc)
        if is_kf:
            self._on_keyframe(frame)
        elif getattr(self.cfg.scene_graph, "target_every_step", False):
            # Between keyframes, still feed the object layer detections OF THE
            # TARGET CATEGORY. Keyframes are 0.25 m / 30 deg apart, so a target
            # glimpsed while crossing a doorway is missed entirely; mapping it
            # a few steps earlier is several metres of travel saved.
            #
            # Target-only on purpose: feeding the FULL detection set every step
            # would add ~500 near-duplicate observations per episode and inflate
            # ObjectTrack.evidence, invalidating the measured min_evidence /
            # confirm_baseline_m thresholds those gates were calibrated against.
            target = self.target.lower().replace("_", " ").strip()
            dets = [
                d for d in self._detections(frame)
                if d.label.lower().replace("_", " ").strip() == target
            ]
            if dets:
                with self.profiler.timeit("object_layer"):
                    self.object_layer.update(frame, dets, floor_key=layer.key)

        # Down-look probe. Handled before any state can return an action, and
        # restore always wins, so the camera can never be left tilted.
        down_look = self._down_look(frame, layer, off_map)
        if down_look is not None:
            return down_look

        return self._dispatch(frame, y_obs, layer)

    def _dispatch(self, frame: FrameData, y_obs: float, layer) -> str:
        """Decide an action, given the maps this step has already updated.

        Split from `_act_inner` so an alternative control flow can replace the
        decision without touching perception, mapping or the floor stack. The
        preamble above is shared verbatim; everything below is OSG's FSM, and
        `agent/ascent_agent.py` substitutes ASCENT's stateless dispatch for it.
        """
        if self.state == State.CLIMB:
            return self._do_climb(frame, y_obs)

        # Candidate target check happens in every state except terminal ones
        check_states = (
            (State.INIT, State.EXPLORE, State.GOTO_FRONTIER, State.GOTO_VERIFY_VIEW)
            if getattr(self.cfg.agent, "check_candidates_all_states", False)
            else (State.INIT, State.EXPLORE, State.GOTO_FRONTIER)
        )
        if self.state in check_states:
            self._check_candidates()

        if self.state == State.INIT:
            if self._scan_steps_left > 0:
                self._scan_steps_left -= 1
                return TURN_ACTION
            self.state = State.EXPLORE

        if self.state == State.EXPLORE:
            self._select_new_frontier(frame)
            if self.state == State.EXPLORE:  # nothing selectable
                return TURN_ACTION  # keep looking around; map will grow

        if self.state == State.GOTO_FRONTIER:
            # Give-up net: no displacement for a while means an obstacle the
            # map cannot see (below the obstacle band, glass, sim collision).
            # Abandon this frontier instead of pushing against it forever.
            agent_xy = frame.camera_position[list(PLANE)]
            # Arriving at a staircase hands over to CLIMB: the rest of the way
            # is a floor transition, not a costmap path, and "arrived" for a
            # stair frontier means being on the other floor.
            f = self._current_frontier
            if (
                f is not None
                and getattr(self.cfg.agent, "stair_climb_state", False)
                and getattr(f, "kind", "explore").startswith("stair_")
                and float(np.linalg.norm(agent_xy - f.centroid_xy))
                < float(getattr(self.cfg.agent, "stair_reach_m", 0.6))
            ):
                return self._enter_climb(frame, f, agent_xy)
            if not self._frontier_gate and self._frontier_consumed(f):
                # Explored away while we walked at it. Release it WITHOUT
                # blocking: it was a perfectly good frontier that simply no
                # longer exists, and blocking would suppress every other
                # frontier within 0.6 m of it too.
                self.stats["frontier_consumed"] = self.stats.get("frontier_consumed", 0) + 1
                self._current_frontier = None
                self._current_path = None
                self.state = State.EXPLORE
                self._frontier_ref_dist = None
                return self._act_inner_post_transition(frame)
            if (
                self._reselect_every > 0
                and self.step_count - self._last_select_step >= self._reselect_every
            ):
                # ASCENT re-runs its whole selection every step and re-extracts
                # the frontier list with it (`map_controller.py:528`,
                # `ascent_policy.py:684`); "commitment" there is bookkeeping
                # inside the selector, not an FSM state the agent is stuck in.
                # OSG commits for a median 23 steps, which behind a mover that
                # closes 0.046 m/step means it holds a target chosen from a map
                # two dozen steps out of date.
                self._select_new_frontier(frame)
                f = self._current_frontier
                if f is None or self.state != State.GOTO_FRONTIER:
                    return self._act_inner_post_transition(frame)
            stick_steps = int(getattr(self.cfg.agent, "frontier_stick_steps", 15))
            stick_m = float(getattr(self.cfg.agent, "frontier_stick_m", 0.2))
            if self._frontier_stalled(agent_xy, f, stick_m, stick_steps):
                self.giveup_log.append((
                    self.step_count,
                    [round(float(x), 2) for x in self._current_frontier.centroid_xy]
                    if self._current_frontier is not None else None,
                    [round(float(x), 2) for x in agent_xy],
                ))
                self._block_frontier(self._current_frontier, 100)
                self.stats["frontier_give_up"] = self.stats.get("frontier_give_up", 0) + 1
                if self._current_frontier is not None:
                    self._last_giveup_pt = self._current_frontier.centroid_xy.copy()
                self._current_frontier = None
                self._current_path = None
                self.state = State.EXPLORE
                self._progress_ref_step = self.step_count
                self._progress_ref_xy = agent_xy.copy()
                self._frontier_ref_dist = None
                return self._act_inner_post_transition(frame)
            action = self._follow_path(frame)
            if action is not None:
                return action
            self._current_frontier = None
            self.state = State.EXPLORE
            return self._act_inner_post_transition(frame)

        if self.state == State.GOTO_VERIFY_VIEW:
            # Same terminal semantics as GOTO_TARGET: with discrete actions
            # the agent rarely lands exactly on the viewpoint — verify once
            # we are near it, the path is consumed, or the deadline passes.
            agent_xy = frame.camera_position[list(PLANE)]
            near_view = (
                self._goal_xy is not None
                and np.linalg.norm(agent_xy - self._goal_xy) < 0.35
            )
            if near_view or self.step_count > self._goto_deadline:
                self.state = State.VERIFYING
            else:
                action = self._follow_path(frame)
                if action is not None:
                    return action
                self.state = State.VERIFYING

        if self.state == State.VERIFYING:
            return self._do_verification(frame)

        if self.state == State.APPROACH:
            return self._do_approach(frame)

        return STOP_ACTION

    def _recheck_rejects(self, stop_reason: str) -> bool:
        """ASCENT's _double_check_goal gate. True means abandon this target.

        Applied at EVERY exit from APPROACH that would commit STOP, not just
        the terminal one. ASCENT's approach has exactly one stop
        (ascent_policy.py:913) and it is gated -- too far returns a pointnav
        action, closing returns a forced MOVE_FORWARD, and the 100-step timeout
        returns to exploring, so nothing can stop on an unvouched target.
        Gating only the terminal rule is therefore not the same mechanism:
        measured on a 4-episode smoke with the threshold forced high, all 12
        rejections were absorbed by `path_consumed`, which stopped the agent
        0.02-0.05 m from where the gated exit would have. The gate rejected the
        target and the agent stopped on it anyway.

        Records the score at every stop whether or not the gate is armed;
        calibrating the threshold needs it from runs that rejected nothing.
        """
        self.approach_recheck_max = float(self._approach_itm_max)
        if not self._approach_recheck:
            return False
        self.stats["recheck_calls"] = self.stats.get("recheck_calls", 0) + 1
        if self._approach_itm_n == 0:
            # The score never ran during this approach -- the value map is
            # skipped on frames that fall outside the current floor's plane, and
            # an approach can be short enough to contain only those. That is an
            # absence of evidence, not evidence against; rejecting on it would
            # blacklist targets the gate never actually looked at, and would
            # score as a low reading in the calibration.
            self.stats["recheck_no_obs"] = self.stats.get("recheck_no_obs", 0) + 1
            return False
        if self._approach_itm_max >= self._approach_recheck_thresh:
            self.stats["recheck_pass"] = self.stats.get("recheck_pass", 0) + 1
            return False
        # "Might false positive, change to look for the true goal": clouds
        # wiped, region disabled, back to _explore (ascent_policy.py:915-922).
        self.stats["recheck_reject"] = self.stats.get("recheck_reject", 0) + 1
        self.stats[f"recheck_reject_{stop_reason}"] = (
            self.stats.get(f"recheck_reject_{stop_reason}", 0) + 1
        )
        if self._candidate_id is not None:
            self.object_layer.blacklist(self._candidate_id)
        self._candidate_id = None
        self._target_obj_xy = None
        self._target_cloud_xy = None
        self.state = State.EXPLORE
        return True

    def _frontier_consumed(self, frontier) -> bool:
        """Has the committed frontier already been explored away?

        ASCENT never faces this: it rebuilds the frontier list every step
        (map_controller.py:528) and re-picks from it (ascent_policy.py:684), so a
        frontier that stops being produced simply stops being a target. OSG
        commits to one `Frontier` object and walks at it until it arrives or
        stalls -- which, behind a reactive mover, means burning the 20-step stall
        window on a goal that is no longer a boundary at all.

        Re-extracting every step is what the 5-step selection throttle exists to
        avoid, so this asks the cheaper question directly of the costmap: is
        there still UNKNOWN space next to the goal? If not, the frontier has
        been consumed and the pursuit should end now rather than on a timer.
        """
        if frontier is None:
            return False
        goal = frontier_goal_xy(frontier, self.costmap)
        rc = self.costmap.world_to_grid(goal)
        # One agent-radius window: enough to survive discretisation and the
        # goal sitting a cell or two inside the explored side.
        r = max(1, int(round(float(self.cfg.agent.agent_radius) / self.costmap.resolution)))
        h, w = self.costmap.grid.shape
        r0, r1 = max(0, rc[0] - r), min(h, rc[0] + r + 1)
        c0, c1 = max(0, rc[1] - r), min(w, rc[1] + r + 1)
        if r0 >= r1 or c0 >= c1:
            return False  # off-map: not evidence of anything
        return not bool((self.costmap.grid[r0:r1, c0:c1] == UNKNOWN).any())

    def _frontier_stalled(
        self, agent_xy: np.ndarray, frontier, stick_m: float, stick_steps: int
    ) -> bool:
        """Has the pursuit of this frontier stopped getting anywhere?

        Two rules, because they catch different failures and the movers fail
        differently:

        `displacement` (default, and what every pre-S8 number was measured on)
        asks whether the AGENT MOVED. It catches an obstacle the map cannot see
        -- glass, a lip below the obstacle band -- which is how the costmap and
        navmesh arms get stuck: pushing, motionless.

        `closing` is ASCENT's rule (llm_planner.py:239-257, thresholds at
        constants.py:234-235): it asks whether the DISTANCE TO THE FRONTIER
        changed. A reactive mover with no global plan does not get stuck
        motionless; it orbits, endlessly moving and never arriving, and the
        displacement rule never fires on it. Observed directly: a pointnav smoke
        run held one frontier from step 85 to step 500 while moving the whole
        time.

        Either way the reference resets the moment real progress happens, so a
        slow approach is not punished.
        """
        if stick_steps <= 0:
            return False
        rule = str(getattr(self.cfg.agent, "frontier_stick_rule", "displacement"))
        if rule == "closing":
            goal = (
                frontier_goal_xy(frontier, self.costmap)
                if frontier is not None else self._goal_xy
            )
            if goal is None:
                return False
            dist = float(np.linalg.norm(agent_xy - goal))
            if self._frontier_ref_dist is None:
                self._frontier_ref_dist = dist
                self._progress_ref_step = self.step_count
                return False
            # ASCENT resets on a change in EITHER direction: being pushed back
            # is not the same failure as going nowhere.
            if abs(self._frontier_ref_dist - dist) > stick_m:
                self._frontier_ref_dist = dist
                self._progress_ref_step = self.step_count
                return False
            return self.step_count - self._progress_ref_step >= stick_steps

        if self.step_count - self._progress_ref_step < stick_steps:
            return False
        moved = float(np.linalg.norm(agent_xy - self._progress_ref_xy))
        self._progress_ref_step = self.step_count
        self._progress_ref_xy = agent_xy.copy()
        return moved < stick_m

    def _commit_terminal_stop(
        self, stop_reason: str, frame: FrameData, det: Optional[Detection]
    ) -> Optional[str]:
        """Run the two gates that stand between a terminal rule and STOP.

        Shared by both routes into a stop so they cannot drift apart: the
        detection-driven rules (depth / bbox / nearest_point) and the map-only
        `nearest_point` check that fires with no target in view. Returns the
        action to execute, or None to let the caller carry on approaching.
        """
        if self._recheck_rejects(stop_reason):
            return TURN_ACTION
        # Terminal-view verification: the agent is close and the target fills
        # the view -- this live close-up is the decisive frame. Ask the VLM
        # before committing STOP; a rejection means the detector locked onto a
        # false positive, so blacklist it and resume exploring rather than
        # stopping on empty/wrong space.
        #
        # Needs a detection to box, so it is skipped on the map-only route --
        # there is no live crop to show. That route is reached only when the
        # accumulated cloud says the agent is already at the object, which is
        # the evidence the verifier would otherwise be asked to second-guess.
        if self._terminal_verify and det is not None:
            self.stats["terminal_verify"] = self.stats.get("terminal_verify", 0) + 1
            if not self.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.target):
                self.stats["terminal_reject"] = self.stats.get("terminal_reject", 0) + 1
                if self._candidate_id is not None:
                    self.object_layer.blacklist(self._candidate_id)
                self._candidate_id = None
                self._target_obj_xy = None
                self._target_cloud_xy = None
                self.state = State.EXPLORE
                return TURN_ACTION
        self.state = State.DONE
        self.approach_stop_reason = stop_reason
        return STOP_ACTION

    def _do_approach(self, frame: FrameData) -> str:
        """Walk toward the verified object while it stays visible.

        Every step re-runs the detector on the current pose:
        - visible and within the target metric range (median mask depth <=
          approach_stop_depth_m) -> close and in clear view, stop. Depth is the
          primary signal (bbox area is object-size-dependent); bbox is a
          fallback for when the mask carries no valid depth.
        - visible but still too far -> record this pose as good, advance one
          more step toward the object.
        - not visible -> if a previous pose was confirmed visible, retreat
          there (a step just carried us behind an occluder the 2D costmap
          LOS check cannot see, e.g. a desk edge) and stop; otherwise the
          object was never visible from this approach at all, so keep
          advancing toward it (there is nothing better to retreat to) until
          the deadline.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        if self._approach_exhausted():
            return self._abandon_approach()
        # Track how close the agent gets to its approach goal this episode.
        if self.approach_diag and self._goal_xy is not None:
            dg = float(np.linalg.norm(agent_xy - self._goal_xy))
            cur = self.approach_diag.get("min_dist_to_goal_m")
            if cur is None or dg < cur:
                self.approach_diag["min_dist_to_goal_m"] = dg
        # The map-based terminal rule, evaluated whether or not the target is
        # visible right now. `_nearest_point_stop` measures against the
        # ACCUMULATED surface cloud, so a live detection was never one of its
        # inputs -- it was only ever an accident of where the call sat.
        #
        # ASCENT evaluates its equivalent unconditionally every step:
        # `_update_distance_on_object_map` (map_controller.py:845-866) recomputes
        # `cur_dis_to_goal` from the target cloud at ascent_policy.py:434, and
        # the stop test at :910-911 reads it with no reference to a detection.
        #
        # Navmesh mode hid the difference: ShortestPathFollower reports arrival,
        # which becomes `path_consumed` -> DONE -> STOP for 46 of 100 episodes.
        # A self-planning sensor-only mover has no such signal -- inside 1 m the
        # creep just returns move_forward forever -- so with the check gated on a
        # detection the approach can only end by seeing the target again or by
        # timing out. Measured on `outputs/s8_pointnav`: 28 episodes entered
        # APPROACH, closed to a median 0.67 m of their goal, and never stopped.
        if (
            not getattr(self.cfg.agent, "terminal_requires_detection", True)
            and getattr(self.cfg.agent, "terminal_rule", "depth") == "nearest_point"
        ):
            blind_stop = self._nearest_point_stop(agent_xy)
            if blind_stop is not None:
                action = self._commit_terminal_stop(blind_stop, frame, det=None)
                if action is not None:
                    return action

        det = self._best_target_detection(frame)

        if det is not None:
            self._approach_last_good_xy = agent_xy.copy()
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_px = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            depth = self._detection_depth(det, frame)
            # Log (step, bbox_px, depth) for terminal calibration.
            self.approach_bbox_log.append(
                (self.step_count, round(float(bbox_px), 1),
                 round(float(depth), 3) if depth is not None else None)
            )
            stop_reason: Optional[str] = None
            if getattr(self.cfg.agent, "terminal_rule", "depth") == "nearest_point":
                stop_reason = self._nearest_point_stop(agent_xy)
            elif getattr(self.cfg.agent, "approach_depth_stop", True):
                if depth is not None:
                    if depth <= self.cfg.agent.approach_stop_depth_m:
                        stop_reason = "depth"
                elif bbox_px >= self.cfg.agent.approach_stop_bbox_px:  # fallback: no valid depth
                    stop_reason = "bbox"
            if stop_reason is not None:
                action = self._commit_terminal_stop(stop_reason, frame, det)
                if action is not None:
                    return action
        elif (
            self._driver is None  # a self-planning mover knows the path; a momentary FOV loss
            # while turning along it must NOT trigger a retreat, or the agent
            # oscillates (approach -> lose detection -> retreat -> re-detect ...)
            # until the deadline. Costmap mode keeps the LOS-occlusion retreat.
            and self._approach_last_good_xy is not None
            and np.linalg.norm(agent_xy - self._approach_last_good_xy) > 0.1
        ):
            action = self._follow_to(frame, self._approach_last_good_xy)
            if action is not None:
                return action
            if self._recheck_rejects("retreat"):
                return TURN_ACTION
            self.state = State.DONE  # retreat path consumed/unreachable: stop here
            self.approach_stop_reason = "retreat"
            return STOP_ACTION

        if self.step_count > self._goto_deadline or self._approach_steps_left <= 0:
            if self._recheck_rejects("deadline"):
                return TURN_ACTION
            self.state = State.DONE
            self.approach_stop_reason = "deadline"
            return STOP_ACTION
        self._approach_steps_left -= 1
        self._last_follow_none_reason = None
        action = self._follow_to(frame, self._goal_xy)
        if action is None:  # path consumed or unreachable: as close as it gets
            if self._recheck_rejects("path_consumed"):
                return TURN_ACTION
            self.state = State.DONE
            self.approach_stop_reason = "path_consumed"
            if self.approach_diag is not None:
                self.approach_diag["path_consumed_cause"] = self._last_follow_none_reason
                if self._last_follow_none_reason == "planner_no_path":
                    self.approach_diag["plan_fail"] = self.approach_diag.get("plan_fail", 0) + 1
            return STOP_ACTION
        return action

    def _approach_exhausted(self) -> bool:
        """Has this approach outlived its step budget?

        Only asked when there is no reachability oracle. `navmesh` mode rejects
        an unreachable target up front (`_check_candidates`, `unreachable_skip`)
        and so never needs this; sensor-only there is nothing to ask, so the
        budget IS the reachability test. Guarding on `_reachable_fn is None`
        rather than on the mode keeps every navmesh number bit-identical.
        """
        if self._reachable_fn is not None:
            return False
        budget = int(getattr(self.cfg.agent, "approach_abandon_steps", 0))
        return budget > 0 and self.step_count - self._approach_start_step >= budget

    def _abandon_approach(self) -> str:
        """Give up on this target and go back to exploring.

        ASCENT's substitute for a connectivity oracle (ascent_policy.py:929-935):
        after 100 steps of not arriving, the target is treated as a false
        positive, its cells are added to `_disabled_object_map`, and the agent
        explores again -- rather than stopping where it is and failing the
        episode, which is what the deadline branch below does.

        The disable is SPATIAL, not by track id: `blacklist` alone lets the same
        object come straight back under a fresh id, which is how one smoke run
        rejected the same thing six times (docs/AB_RESULTS.md, S29).
        """
        # disable_target retires the PLACE as well as the id, reusing the same
        # `_disabled_pts` list the false-positive retraction feeds -- so a
        # re-detection is killed at track birth rather than re-committed to.
        disabled = (
            self._candidate_id is not None
            and self.object_layer.disable_target(self._candidate_id)
        )
        if not disabled and self._target_obj_xy is not None:
            self.object_layer.disable_place(self._target_obj_xy, self.target)
        self.stats["approach_abandon"] = self.stats.get("approach_abandon", 0) + 1
        self._candidate_id = None
        self._target_obj_xy = None
        self._target_cloud_xy = None
        self._goal_xy = None
        self._current_path = None
        self._path_goal = None
        self._approach_last_good_xy = None
        self.state = State.EXPLORE
        return TURN_ACTION

    def _nearest_point_stop(self, agent_xy: np.ndarray) -> Optional[str]:
        """Terminal rule measured against the object's nearest SURFACE point.

        HM3D scores geodesic distance to a view_point, and view points are
        tiled around an object's surface. Median mask depth is a poor proxy: it
        is the distance to the middle of whatever the mask covers, so a 2 m sofa
        and a chair stop at very different distances from their nearest edge.

        Stop when close enough, or when closing has stalled -- the latter is
        what handles being blocked by the object itself or by furniture in
        front of it. The distance MUST be recomputed from the live pose each
        step for that to be self-limiting.

        The stall test only counts steps in which the agent actually MOVED.
        Approaching an object involves turning to face it, and a turn leaves the
        distance unchanged; counting those as "no progress" fired the stall on
        the first turn of almost every approach. Measured: 18 of 50 episodes
        stopped that way, and median distance to goal went from 0.04 m to
        1.09 m -- a 64% -> 32% collapse in SR.
        """
        cfg = self.cfg.agent
        # `is not None`, not truthiness: track ids start at 0.
        track = (
            self.object_layer.get(self._candidate_id)
            if self._candidate_id is not None else None
        )
        if track is None:
            return None
        d = self.object_layer.nearest_point_dist_xy(
            track, agent_xy, float(getattr(cfg, "terminal_percentile", 0.0))
        )
        if d is None or d >= float(getattr(cfg, "terminal_engage_m", 1.0)):
            return None
        if d <= float(getattr(cfg, "terminal_stop_m", 0.6)):
            return "nearest_point"

        moved = (
            self._terminal_last_xy is not None
            and float(np.linalg.norm(agent_xy - self._terminal_last_xy)) > 0.05
        )
        self._terminal_last_xy = agent_xy.copy()
        if not moved:
            return None  # a turn is not a failure to close

        if abs(d - self._terminal_min_d) < float(getattr(cfg, "terminal_progress_eps", 0.1)):
            self._terminal_stalls += 1
            # One non-improving step is noise (an oblique approach barely
            # changes the distance to the nearest surface); a run of them is a
            # genuine block.
            if self._terminal_stalls >= int(getattr(cfg, "terminal_stall_steps", 3)):
                return "nearest_point_stalled"
            return None
        self._terminal_stalls = 0
        self._terminal_min_d = min(self._terminal_min_d, d)
        return None

    def _act_inner_post_transition(self, frame: FrameData) -> str:
        """Re-enter EXPLORE logic once after a state transition (no recursion
        beyond one level: EXPLORE either picks a path or turns in place)."""
        self._select_new_frontier(frame)
        if self.state == State.GOTO_FRONTIER:
            action = self._follow_path(frame)
            if action is not None:
                return action
        return TURN_ACTION

    # -------------------------------------------------------------- keyframes

    def _on_keyframe(self, frame: FrameData) -> None:
        self._kf_count += 1
        layer = self.floors.current()
        dets = self._detections(frame)
        if self.on_keyframe_detections is not None:
            self.on_keyframe_detections(frame, dets)
        with self.profiler.timeit("object_layer"):
            self.object_layer.update(frame, dets, floor_key=layer.key)
        if self.stair_detector is not None:
            with self.profiler.timeit("stairs"):
                self.stair_detector.accumulate(
                    frame, layer, dets, self._seg_stair_mask(frame)
                )
        if getattr(self.cfg.scene_graph, "fp_retraction", False):
            n = self.object_layer.retract_unconfirmed(
                frame, dets,
                half_range_m=0.5 * self.cfg.mapping.max_range_m,
                fov_rad=np.radians(float(getattr(self.cfg.eval, "hfov_deg", 79.0))),
            )
            if n:
                self.stats["fp_retract"] = self.stats.get("fp_retract", 0) + n
        layer.track_ids = {t.id for t in self.object_layer.tracks() if t.floor_key == layer.key}
        self.keyframes.add(frame)

        # Room labels are per floor: the segmentation describes one costmap, and
        # each floor keeps its own so returning to a floor does not re-segment
        # from scratch.
        if self._kf_count % self.cfg.scene_graph.room_seg_every_kf == 1:
            with self.profiler.timeit("room_seg"):
                layer.room_labels = self.room_segmenter.segment(layer.costmap)
        # Where the agent is standing, and what Places365 calls it. Votes are
        # keyed on POSITION, not on room id: the segmenter renumbers rooms on
        # every re-segmentation, so an id-keyed tally would attach a label to
        # whatever room inherited that number.
        if self.room_classifier is not None:
            with self.profiler.timeit("room_classify"):
                room_name = self.room_classifier.classify(frame.rgb)
            self._room_votes.append(
                (frame.camera_position[list(PLANE)].copy(), room_name)
            )
            del self._room_votes[: -self._room_vote_cap]
            # ASCENT's _update_current_step_scene_info (map_controller.py:800):
            # the room AND the object tags for this step both come from THIS
            # frame. `dets` is the same detector pass the object layer used, so
            # this costs nothing extra; RAM++ would slot in here in place of it.
            if self.frontier_semantics is not None:
                self.frontier_semantics.observe(
                    self.step_count, room_name, [d.label for d in dets],
                    camera_xy=frame.camera_position[list(PLANE)],
                    heading_xy=self._heading_xy(frame),
                )

        if layer.room_labels is not None:
            if layer.room_labels.shape != layer.costmap.grid.shape:
                layer.room_labels = self.room_segmenter.segment(layer.costmap)
            with self.profiler.timeit("scene_graph"):
                self.scene_graph.rebuild_floor(
                    layer.room_labels, layer.costmap, self.object_layer, floor_key=layer.key
                )
                self._label_rooms(layer)

    def _label_rooms(self, layer) -> None:
        """Give each room the majority Places365 label observed inside it.

        Majority rather than latest: a single frame taken through a doorway
        classifies the room beyond it, and one bad vote should not rename a
        room the agent has crossed twenty times.

        This is the only writer of RoomNode.label that runs in the measured
        configuration. The other one, LLMTextScorer.score, is reached only when
        exploration.frontier_text_scorer is an LLM -- which no arm in docs/AB_RESULTS uses, so
        room labels were always None and the S10 ranker saw "unknown room"
        everywhere.
        """
        from collections import Counter

        if not self._room_votes or layer.room_labels is None:
            return
        base = layer.key * ROOM_IDS_PER_FLOOR
        votes = {}
        for xy, name in self._room_votes:
            rc = layer.costmap.world_to_grid(xy)
            if not layer.costmap.in_bounds(rc):
                continue
            local = int(layer.room_labels[rc[0], rc[1]])
            if local <= 0:
                continue
            votes.setdefault(base + local, Counter())[name] += 1
        # How much of the graph carries semantics. Without this the A/B cannot
        # tell "room typing did not help" from "room typing never happened" --
        # which is exactly how S10 shipped a result about a prompt whose every
        # area read "unknown room".
        self.stats["rooms_total"] = len(self.scene_graph.rooms)
        for rid, counter in votes.items():
            room = self.scene_graph.rooms.get(rid)
            if room is not None:
                room.label = counter.most_common(1)[0][0]
        self.stats["rooms_labelled"] = sum(
            1 for r in self.scene_graph.rooms.values() if r.label
        )
        # The labels themselves, not just how many. Without these an arm that
        # regresses cannot be read: "room-typed reasoning does not help" and
        # "the classifier is wrong on HM3D renders" look identical from SR.
        self.stats["room_label_counts"] = dict(
            Counter(r.label for r in self.scene_graph.rooms.values() if r.label)
        )

    # ------------------------------------------------------------ exploration

    def _block_frontier(self, f: Optional[Frontier], duration: int) -> None:
        if f is not None:
            self._blocked_frontier_pts.append((f.centroid_xy.copy(), self.step_count + duration))

    def _blocked_ids(self, frontiers) -> set:
        active = [xy for xy, until in self._blocked_frontier_pts if until > self.step_count]
        self._blocked_frontier_pts = [
            (xy, until) for xy, until in self._blocked_frontier_pts if until > self.step_count
        ]
        return {
            f.id
            for f in frontiers
            if any(np.linalg.norm(f.centroid_xy - xy) < 0.6 for xy in active)
        }

    @staticmethod
    def _heading_xy(frame: FrameData) -> np.ndarray:
        """Agent forward direction on the ground plane (unit). Camera looks along
        +z (OpenCV), so world-forward = R @ [0,0,1], projected to (x, z)."""
        fwd = frame.T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0])
        v = fwd[list(PLANE)]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _update_value_map(self, frame: FrameData, layer: FloorLayer) -> None:
        """Score this view against the target prompt and paint it into the map.

        Strided because the score changes slowly with pose -- consecutive frames
        from a 0.25 m step see almost the same thing -- so the model does not
        need to run every step to keep the map current.
        """
        if self.image_text is None or layer.value_map is None:
            return
        stride = max(1, int(getattr(self.cfg.exploration, "value_stride", 1)))
        if self.step_count % stride:
            return
        prompt = str(
            getattr(self.cfg.exploration, "value_prompt",
                    "Seems like there is a {target} ahead.")
        ).format(target=self.target.replace("_", " "))
        with self.profiler.timeit("value_map"):
            value = float(self.image_text.score(frame.rgb, [prompt])[0])
            layer.value_map.update(frame, value)
        self.stats["value_calls"] = self.stats.get("value_calls", 0) + 1
        # ASCENT keeps this same cosine as _blip_cosine (map_controller.py:562)
        # and re-uses it to double-check the target while approaching. Free:
        # the model has already run for the value map.
        self._last_itm = value
        if self.state is State.APPROACH:
            self._approach_itm_max = max(self._approach_itm_max, value)
            self._approach_itm_n += 1

    def _load_knowledge(self, force: bool = False):
        """The (object, room) graph, or None.

        `force` bypasses the `knowledge_prior` flag. That flag turns on the
        *numeric multiplier* applied to frontier scores; the ASCENT ranker needs
        the same graph for a different purpose -- printing the room-to-goal
        probabilities into the prompt. Tying the two together would silently
        strip the priors out of the prompt in any A/B that disables the
        multiplier to isolate a variable, which is exactly how those arms are
        run.
        """
        if not force and not getattr(self.cfg.exploration, "knowledge_prior", False):
            return None
        from ..exploration.knowledge_prior import KnowledgeGraph

        try:
            return KnowledgeGraph.load(
                getattr(self.cfg.exploration, "knowledge_prior_path", None)
            )
        except OSError:
            # Priors are generated by scripts/make_priors.py and are optional;
            # exploration must not become unrunnable because they are absent.
            return None

    def _knowledge_affinity(self, f) -> Optional[float]:
        """How much the objects mapped near `f` look like the target's rooms."""
        if self.knowledge is None:
            return None
        radius = float(getattr(self.cfg.exploration, "knowledge_radius_m", 3.0))
        labels = [o.label for o in self.scene_graph.objects_near(f.centroid_xy, radius)]
        return self.knowledge.affinity(self.target, labels)

    def _rank_frontiers(self, frontiers, frontier_values, blocked, agent_xy):
        """ASCENT's forced choice among the top-k frontiers by value.

        Returns the chosen frontier, or None to leave the decision to the
        ordinary selectors -- which is also what happens on every failure, so
        the model can improve the ranking but never break selection.

        The pick must be reachable to be used. ASCENT hands its waypoint to a
        learned PointNav policy that heads toward almost anything; here a
        frontier A* cannot reach would simply be re-picked every round.
        """
        if self.ranker is None or frontier_values is None:
            return None
        every = int(getattr(self.cfg.exploration, "ranker_every_steps", 20))
        if self.step_count - self._last_rank_step < every:
            return None

        # Stair frontiers carry a prior rather than a value, and their worth is
        # a floor-level question rather than a "which room looks right" one, so
        # they are not offered to the ranker.
        cand = [
            f for f in frontiers
            if f.id not in blocked
            and getattr(f, "kind", "explore") == "explore"
            and f.id in frontier_values
        ]
        if len(cand) < 2:
            return None
        cand.sort(key=lambda f: -float(frontier_values[f.id]))
        cand = cand[: int(getattr(self.cfg.exploration, "ranker_topk", 3))]

        self._last_rank_step = self.step_count
        with self.profiler.timeit("frontier_rank"):
            idx = self.ranker.pick(cand, self.target, self.scene_graph,
                                   semantics=self.frontier_semantics,
                                   mode=self._frontier_desc)
        self.stats["rank_calls"] = self.ranker.calls
        self.stats["rank_overrides"] = self.ranker.overrides
        if getattr(self.ranker, "desc_frame", 0):
            self.stats["desc_frame"] = self.ranker.desc_frame
            self.stats["desc_differs"] = self.ranker.desc_differs

        chosen = cand[idx]
        result = self.selection_planner.plan(
            self.costmap, agent_xy, frontier_goal_xy(chosen, self.costmap)
        )
        if not result.success:
            self.stats["rank_unreachable"] = self.stats.get("rank_unreachable", 0) + 1
            return None
        chosen.path_cost = max(result.cost, self.cfg.exploration.min_path_cost_m)
        chosen.score = float(frontier_values[chosen.id])
        return chosen

    def _frontier_values(self, frontiers) -> Optional[dict]:
        """Semantic value per explore frontier, or None when the map is off.

        Stair frontiers are excluded on purpose: they carry their own prior,
        and a stairwell's similarity to "a bed is ahead" says nothing about
        whether the floor above is worth visiting.
        """
        vm = self.floors.current().value_map
        if vm is None and self.knowledge is None:
            return None
        radius = float(getattr(self.cfg.exploration, "value_radius_m", 0.5))
        prior = float(self.cfg.exploration.unscored_prior)
        kw = float(getattr(self.cfg.exploration, "knowledge_weight", 1.0))

        out = {}
        for f in frontiers:
            if getattr(f, "kind", "explore") != "explore":
                continue  # stair frontiers carry their own prior
            v = vm.value_at(f.centroid_xy, radius) if vm is not None else prior
            aff = self._knowledge_affinity(f)
            if aff is not None:
                # Multiplicative and centred on 1, so "no nearby objects" (aff
                # is None) leaves the value untouched rather than zeroing a
                # perfectly good unexplored direction.
                v *= 1.0 + kw * (aff - 0.5)
            out[f.id] = max(v, 0.0)
        return out

    def _stair_frontiers(self):
        """Staircases on this floor, as frontiers the selector can choose.

        Returns (frontiers, scores). Their score is a prior rather than
        anything semantic, scaled up once this floor is exhausted -- the whole
        question a stair frontier answers is "is this floor still worth
        searching?". stair_prior = 0 (the default) means they are never chosen,
        so the machinery costs nothing when disabled.

        "Exhausted" follows ASCENT: a floor is done when it has **no explore
        frontiers left** (`ascent_policy.py:655`, which sets
        `_this_floor_explored` on exactly that condition). It deliberately does
        NOT use a step count, and grepping `_floor_num_steps` confirms the only
        other place it appears in a decision is a stairwell-reinitialisation
        guard.

        This used to be `steps_on_floor >= floor_exp_steps` (100), which is the
        same threshold as the floor-LLM gate -- so the geometric boost and the
        semantic question fired on the same step and pointed the same way, and
        the S12 arm came out 45/50 bit-identical. ASCENT separates them on
        purpose: the LLM may move a floor at 100 steps, the geometry only once
        there is nothing left to explore.
        """
        if self.stair_detector is None:
            return [], {}
        prior = float(getattr(self.cfg.exploration, "stair_prior", 0.0))
        if prior <= 0.0:
            return [], {}

        layer = self.floors.current()
        if not layer.explored:
            layer.explored = layer.steps_on_floor >= int(
                getattr(self.cfg.exploration, "floor_exp_steps", 100)
            )
        if layer.explored:
            prior *= float(getattr(self.cfg.exploration, "stair_explored_boost", 3.0))

        out, scores = [], {}
        for det in self.stair_detector.extract(layer):
            if self._stair_disabled(det.centroid_xy):
                continue
            # Suppress the staircase the agent just arrived by: from the top of
            # a flight the way back down is the most salient "unexplored"
            # direction there is, and following it undoes the climb.
            if (
                layer.entry_xy is not None
                and float(np.linalg.norm(det.centroid_xy - layer.entry_xy)) < 1.0
            ):
                continue
            f = Frontier(
                id=self.frontier_extractor._next_id,
                centroid_xy=det.centroid_xy,
                cells=det.cells,
                size=det.n_cells,
                kind=f"stair_{det.kind}",
                floor_key=layer.key,
            )
            self.frontier_extractor._next_id += 1
            out.append(f)
            scores[f.id] = prior * self._floor_direction_boost(det.kind)
        # S33 left a hole: RedNet lifted up-stair DETECTION 10% -> 54% and
        # climb_attempt did not move at all (1 -> 1), so the gain is lost
        # somewhere between a stair component existing and the agent standing on
        # it -- and nothing recorded which. These two split that span: how often
        # a staircase was offered to the selector, and how often it won.
        self.stats["stair_frontiers_seen"] = (
            self.stats.get("stair_frontiers_seen", 0) + len(out)
        )
        if out:
            self.stats["stair_rounds"] = self.stats.get("stair_rounds", 0) + 1
        return out, scores

    def _mark_floor_explored(self, n_explore: int) -> None:
        """Has this floor been searched? ASCENT's answer is "no frontiers left"
        (ascent_policy.py:655, which sets _this_floor_explored on exactly that),
        and grepping _floor_num_steps confirms no step count feeds the decision.

        Kept out of _stair_frontiers, which returns early when there is no stair
        detector or stair_prior is 0 -- the flag is also read by the floor
        prompt, so it has to be maintained whether or not stairs are in play.

        The old rule, steps_on_floor >= floor_exp_steps (100), is the same
        threshold as the floor-LLM gate: both fired on the same step and pointed
        the same way, and the S12 arm came out 45/50 bit-identical.
        """
        layer = self.floors.current()
        rule = str(getattr(self.cfg.exploration, "stair_explored_rule", "no_frontiers"))
        if rule == "no_frontiers":
            if n_explore == 0:
                layer.explored = True
        elif not layer.explored:
            layer.explored = layer.steps_on_floor >= int(
                getattr(self.cfg.exploration, "floor_exp_steps", 100)
            )

    def _left_the_stairs(self, agent_xy) -> bool:
        """Is the agent clear of the staircase it is climbing?

        Port of `is_robot_in_stair_map_fast` (ascent/map_controller.py:181-215),
        which asks whether any stair cell lies within the agent radius. The
        comparison is done in world coordinates because the costmap can grow
        mid-climb; ASCENT can use pixel indices because its map is fixed size.

        No recorded cells means nothing to be on, so the height rule stands
        alone rather than blocking the climb forever.
        """
        if self._climb_cells_xy is None or not len(self._climb_cells_xy):
            return True
        d = float(np.linalg.norm(self._climb_cells_xy - agent_xy, axis=1).min())
        return d > float(getattr(self.cfg.agent, "stair_exit_m", 0.5))

    def _floor_frozen(self, frame) -> bool:
        """Should the floor stack be held still this step?

        Two scopes, separately measurable. `freeze_floor_in_climb` covers the
        CLIMB state, which fixed climbs ending at 72 cm of gain against a 90 cm
        threshold. `freeze_floor_on_stairs` covers standing on a staircase at
        all, which is what ASCENT actually gates on -- its floor index moves
        only when the agent leaves the stairs.
        """
        if (self.state is State.CLIMB
                and bool(getattr(self.cfg.mapping, "freeze_floor_in_climb", False))):
            return True
        if bool(getattr(self.cfg.mapping, "freeze_floor_on_stairs", False)):
            return self._on_a_staircase(frame.camera_position[list(PLANE)])
        return False

    def _on_a_staircase(self, agent_xy) -> bool:
        """Is the agent standing on ANY staircase of this floor?

        Port of `is_robot_in_stair_map_fast` (ascent/map_controller.py:181-215)
        over the floor's whole stair mask rather than one component: a bounding
        box around the agent, a circular mask, and "is any stair cell inside".

        `_left_the_stairs` answers the same question for the climb in progress
        only. That is not enough to gate the floor stack: the episode with 18
        floor switches recorded ONE climb attempt, so its oscillation happened
        during ordinary frontier navigation, with the state machine never
        involved.

        The mask threshold is the detector's own `min_hits`, so this sees
        exactly the cells a component would be built from.
        """
        det = self.stair_detector
        layer = self.floors.current()
        if det is None or layer.up_stair_hits is None:
            return False
        cm = layer.costmap
        rc = cm.world_to_grid(agent_xy)
        rad = max(1, int(round(float(getattr(self.cfg.agent, "stair_exit_m", 0.5))
                               / cm.resolution)))
        h, w = layer.up_stair_hits.shape
        r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
        c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disc = (rr - rc[0]) ** 2 + (cc - rc[1]) ** 2 <= rad * rad
        for hits in (layer.up_stair_hits, layer.down_stair_hits):
            if hits is None:
                continue
            if ((hits[r0:r1, c0:c1] >= det.min_hits) & disc).any():
                return True
        return False

    def _floor_direction_boost(self, kind: str) -> float:
        """Weight a staircase by whether it leads toward the floor the coarse
        level asked for.

        Stair frontiers compete on the same score axis as explore frontiers, so
        a direction with no weight behind it changes nothing. The opposite
        direction is damped rather than removed: the floor decision is a guess
        from a partial map, and a hard veto would strand the agent if it is
        wrong and the only staircase leads the other way.
        """
        if not self._floor_goal_dir:
            return 1.0
        boost = float(getattr(self.cfg.exploration, "floor_llm_boost", 5.0))
        wanted = "up" if self._floor_goal_dir > 0 else "down"
        return boost if kind == wanted else 1.0 / boost

    def _ask_floor(self, frame) -> None:
        """Coarse level: which storey? Sets a direction; does not plan."""
        if self.floor_planner is None:
            return
        # Which directions a staircase has actually been seen in. This is what
        # lets the question be asked before a climb rather than after it.
        dets = (self.stair_detector.extract(self.floors.current())
                if self.stair_detector is not None else [])
        has_up = any(d.kind == "up" for d in dets)
        has_down = any(d.kind == "down" for d in dets)
        with self.profiler.timeit("floor_llm"):
            direction = self.floor_planner.decide(
                self.target, self.floors, self.scene_graph, self.step_count,
                has_up=has_up, has_down=has_down,
            )
        if direction is not None:
            self._floor_goal_dir = direction
        fp = self.floor_planner
        self.stats["floor_asks"] = fp.asks
        self.stats["floor_moves"] = fp.moves
        self.stats["floor_blocked_one_floor"] = fp.blocked_one_floor
        self.stats["floor_blocked_throttle"] = fp.blocked_throttle
        self.stats["floor_blocked_too_soon"] = fp.blocked_too_soon

    def _enter_climb(self, frame: FrameData, f: Frontier, agent_xy: np.ndarray) -> str:
        """Commit to traversing `f`, a staircase the agent has just reached."""
        layer = self.floors.current()
        # Aim PAST the staircase, along the approach direction: stopping at its
        # centroid leaves the agent on the bottom step, which is neither floor.
        direction = f.centroid_xy - agent_xy
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-6 else np.array([1.0, 0.0])
        self._climb_goal_xy = f.centroid_xy + direction * float(
            getattr(self.cfg.agent, "stair_overshoot_m", 1.5)
        )
        # The overshoot lands past the flight, which for a DOWN staircase can be
        # over the void and for any staircase can be inside geometry -- the
        # navmesh then has nowhere to snap it and reports "nowhere to go" on the
        # first step. Falling back to the centroid before giving up turns those
        # instant failures into real attempts.
        self._climb_fallback_xy = f.centroid_xy.copy()
        self._climb_centroid_xy = f.centroid_xy.copy()
        self._climb_cells = f.cells
        # World coordinates, not grid indices: Costmap2D.ensure_contains
        # reallocates and shifts the origin when the map grows, which would
        # leave stored rc indices pointing at the wrong cells mid-climb.
        self._climb_cells_xy = (
            layer.costmap.grid_to_world(f.cells.astype(float))
            if f.cells is not None and f.cells.size else None
        )
        self._climb_from_key = layer.key
        self._climb_start_y = float(frame.camera_position[1] - self.cfg.agent.camera_height)
        self._climb_ref_y = self._climb_start_y
        self._climb_ref_step = self.step_count
        self._climb_deadline = self.step_count + int(
            getattr(self.cfg.agent, "climb_max_steps", 80)
        )
        self.state = State.CLIMB
        self._current_path = None
        # Per-CLIMB, not per-episode: a second staircase must not inherit the
        # first one's ratchet or its stall count.
        self._carrot_xy = None
        self._carrot_disable_end = False
        self._climb_paused_steps = 0
        self._climb_last_dist = None
        self.stats["climb_attempt"] = self.stats.get("climb_attempt", 0) + 1
        return self._do_climb(frame, self._climb_ref_y)

    def _stair_disabled(self, xy: np.ndarray) -> bool:
        return any(
            float(np.linalg.norm(xy - bad)) < 1.0 for bad in self._disabled_stairs
        )

    def _down_look(self, frame: FrameData, layer, off_map: bool) -> Optional[str]:
        """Tilt the camera down for one frame, then put it back.

        **Down stairs are pure geometry** (`mapping/stairs.py:_below_floor_points`):
        a point that back-projects below the standing floor is a hole in it, and
        no detector is involved. But a level 79-degree frustum at 0.88 m stops
        looking at the floor a couple of metres out, so a stairwell further away
        than that is not missed by the signal -- it is simply never in frame.
        Tilting down puts it there.

        This is deliberately NOT the up-stair probe. S14a already measured that
        one and killed it: up-stair recall went 19% at level pitch to **0%** at
        +30 degrees, because tilting up moves treads OUT of frame rather than
        into it. Nothing here tilts up to search; `look_up` appears only to undo
        a `look_down`.

        Pose costs nothing to get right: `sim/habitat_env.py` reads `T_wc` from
        the sensor's own state, so a tilted frame back-projects correctly with no
        pitch bookkeeping -- unlike ASCENT, which has to track `_pitch_angle` by
        hand and fold it into its transform (`ascent_policy.py:237`).

        The agent never MOVES while tilted: the probe is one `look_down`, one
        observation, one `look_up`. That keeps the mover's depth input level, and
        bounds the cost at two steps per probe.
        """
        if self._pitch_ticks > 0:
            # THIS frame is the tilted one -- the look_down was executed last
            # step -- so it is the only chance to read it.
            if not off_map:
                self._accumulate_down_stairs(frame, layer)
            # Restore unconditionally. An early return anywhere below could
            # otherwise strand the camera pointing at the floor.
            self._pitch_ticks -= 1
            return "look_up"
        if self._down_look_every <= 0:
            return None
        if self.state not in (State.INIT, State.EXPLORE, State.GOTO_FRONTIER):
            # Not during APPROACH, VERIFYING or CLIMB: those are committed
            # behaviours whose own logic reads the live frame.
            return None
        if self.step_count - self._last_down_look_step < self._down_look_every:
            return None
        self._last_down_look_step = self.step_count
        self._pitch_ticks += 1
        self.stats["down_look"] = self.stats.get("down_look", 0) + 1
        return "look_down"

    def _accumulate_down_stairs(self, frame: FrameData, layer) -> None:
        """Fold the tilted frame's below-floor geometry into the stair grid.

        Called directly rather than waiting for `_on_keyframe`, because whether a
        30-degree tilt counts as a keyframe depends on `keyframe_rot_deg` being
        30 as well -- the probe would then work or not by coincidence. Passing
        no detections keeps this to the geometric down-stair signal and skips a
        detector call on a frame pointed at the floor.
        """
        if self.stair_detector is None:
            return
        with self.profiler.timeit("stairs"):
            self.stair_detector.accumulate(frame, layer, None, self._seg_stair_mask(frame))

    def _seg_stair_mask(self, frame: FrameData) -> Optional[np.ndarray]:
        """RedNet's stair mask for this frame, or None when it is not enabled.

        Already past the >20-pixel gate ASCENT applies before trusting the
        segmenter at all (`obstacle_map.py:520`).
        """
        if self.stair_segmenter is None:
            return None
        with self.profiler.timeit("stair_seg"):
            return self.stair_segmenter.stair_mask(frame)

    def _carrot_goal(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[np.ndarray]:
        """ASCENT's carrot waypoint: steer at the farthest thing you can see.

        Port of `ascent_policy.py:1075-1112`. On a staircase the farthest depth
        return lies along the flight -- up the well, or down it -- because the
        treads and side walls are close and the far end is not. Aiming a short
        way along that bearing walks the agent THROUGH the flight, which a fixed
        goal past the staircase does not: that goal is a straight line through
        whatever wall the stairwell turns around.

        The bearing is read straight off the depth image: take every pixel at
        the maximum depth, average them, and convert the mean column to an angle
        off boresight. Then place the goal `climb_carrot_m` ahead on that
        bearing.

        Returns None when the depth image carries no usable maximum, which the
        caller treats as "just go forward" exactly as ASCENT does (:1082-1088).
        """
        depth = frame.depth
        if depth.size == 0:
            return None
        max_value = float(np.max(depth))
        if not np.isfinite(max_value):
            return None
        rows_cols = np.argwhere(depth == max_value)
        if rows_cols.size == 0:
            return None
        u = float(np.mean(rows_cols[:, 1]))

        intr = frame.intrinsics
        cx = float(intr.cx)
        hfov = 2.0 * float(np.arctan(intr.width / (2.0 * intr.fx)))
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (hfov / 2.0)
        # ASCENT subtracts the offset because its heading is CCW-positive
        # (ascent_policy.py:1095). OSG's ground-plane heading is CW-positive --
        # `turn_left` DECREASES `agent_heading`, asserted by
        # tests/unit/test_controller.py -- so a pixel right of centre, which is
        # a right turn, ADDS here. Same rotation, opposite sign convention; the
        # unit tests pin it.
        target_heading = agent_heading(frame.T_wc) + angle_offset
        dist = float(getattr(self.cfg.agent, "climb_carrot_m", 0.8))
        return agent_xy + dist * np.array([np.cos(target_heading), np.sin(target_heading)])

    def _update_carrot(self, frame: FrameData, agent_xy: np.ndarray) -> Optional[np.ndarray]:
        """The carrot, ratcheted so it only ever moves closer to the stair end.

        A bearing read from one depth frame is noisy, and on a half-landing it
        can swing back the way the agent came. ASCENT guards that by keeping the
        PREVIOUS carrot unless the new one is closer to the recorded end of the
        staircase (`ascent_policy.py:1104-1121`, L1 in map pixels; L2 in metres
        here, which orders candidates the same way without a grid to quantise
        to). The ratchet is dropped -- always take the fresh bearing -- when
        there is nothing to ratchet against yet, when the agent is essentially
        at the end already, or when the stall detector has decided the recorded
        end is not reachable (`_disable_end`, :1099-1101).

        `_climb_goal_xy` is OSG's stand-in for ASCENT's `_up/_down_stair_end`:
        both are a point placed past the flight along the approach direction.
        """
        fresh = self._carrot_goal(frame, agent_xy)
        if fresh is None:
            return self._carrot_xy
        end = self._climb_goal_xy
        near_end = (
            end is not None and float(np.linalg.norm(end - agent_xy)) <= 0.5
        )
        if self._carrot_xy is None or end is None or near_end or self._carrot_disable_end:
            self._carrot_xy = fresh
        elif np.linalg.norm(fresh - end) < np.linalg.norm(self._carrot_xy - end):
            self._carrot_xy = fresh
        return self._carrot_xy

    def _carrot_action(self, frame: FrameData, agent_xy: np.ndarray) -> str:
        """Drive at the carrot. Never reports failure.

        ASCENT forces MOVE_FORWARD whenever the network emits STOP mid-climb
        (`ascent_policy.py:1136-1139`, and again in its centroid phase at
        :1055-1058). On a staircase a STOP usually means "the treads fill my
        view", which is the one moment the agent must not stop. Ending the climb
        is left to the stall counter and the deadline, which measure whether the
        agent is actually getting anywhere.
        """
        goal = self._update_carrot(frame, agent_xy)
        if goal is None:
            return FORWARD_ACTION
        if self.pointnav is not None:
            nav = self.pointnav.step(goal)
            if nav.action is None:
                self.stats["climb_forced_forward"] = (
                    self.stats.get("climb_forced_forward", 0) + 1
                )
                return FORWARD_ACTION
            return nav.action
        action = self._follow_to(frame, goal)
        return action if action is not None else FORWARD_ACTION

    def _carrot_stalled(self, agent_xy: np.ndarray) -> bool:
        """ASCENT's climb stall rule, on distance to the staircase.

        `ascent_policy.py:1036-1046`: if the distance to the stair frontier has
        not moved by more than 0.2 m, count the step; past 15 such steps the
        recorded stair end is presumed unreachable and the ratchet is released
        (`_disable_end`), past 30 the climb is abandoned. Distance rather than
        height, because a landing mid-flight gains no height for several steps
        and a height rule cuts the climb off there.
        """
        ref = self._climb_centroid_xy
        if ref is None:
            return False
        dist = float(np.linalg.norm(agent_xy - ref))
        if self._climb_last_dist is None or abs(self._climb_last_dist - dist) > 0.2:
            self._climb_last_dist = dist
            self._climb_paused_steps = 0
        else:
            self._climb_paused_steps += 1
        if self._climb_paused_steps > 15:
            self._carrot_disable_end = True
        return self._climb_paused_steps > 30

    def _do_climb(self, frame: FrameData, y_obs: float) -> str:
        """Traverse a staircase until the floor stack commits to a new floor.

        Success is defined by the FLOOR STACK switching, not by reaching a
        waypoint: the goal of climbing is to be on another floor, and the stack
        is what decides that. The goal point is pushed past the staircase so the
        agent walks through it rather than stopping on the first tread.
        """
        agent_xy = frame.camera_position[list(PLANE)]
        layer = self.floors.current()

        # Two success conditions, and the second is load-bearing: FloorStack
        # only commits a new floor after the agent has been settled at the new
        # height for commit_steps, which happens several steps AFTER the flight
        # is cleared. Waiting for the key alone made every successful climb
        # register as a failure (climb_ok stayed 0 on episodes that demonstrably
        # changed floor and then succeeded).
        gain = abs(y_obs - self._climb_start_y)
        climbed = gain >= float(getattr(self.cfg.agent, "floor_gap_min_m", 0.9))
        if str(getattr(self.cfg.agent, "climb_exit_rule", "height")) == "topological":
            # ASCENT's rule (map_controller.py:299): the climb ends when the
            # agent is no longer on the staircase, not at a fixed height. The
            # height guard stays so stepping back off the bottom is not counted
            # as an ascent.
            climbed = climbed and self._left_the_stairs(agent_xy)
        if layer.key != self._climb_from_key or climbed:
            self.stats["climb_exit_gain_cm"] = (
                self.stats.get("climb_exit_gain_cm", 0) + int(round(gain * 100))
            )
            layer.entry_xy = agent_xy.copy()
            self.stats["climb_ok"] = self.stats.get("climb_ok", 0) + 1
            self.state = State.EXPLORE
            self._current_frontier = None
            self._current_path = None
            return self._act_inner_post_transition(frame)

        if self._climb_carrot:
            # ASCENT's traversal: a depth-derived waypoint re-aimed every step,
            # and a climb that no navigation verdict can end. `action` is never
            # None here -- a network STOP becomes a forward step
            # (ascent_policy.py:1136-1139) -- so the only exits are the floor
            # change above, the stall counter, and the deadline.
            action = self._carrot_action(frame, agent_xy)
            stalled = self._carrot_stalled(agent_xy)
        else:
            action = self._follow_to(frame, self._climb_goal_xy)
            if action is None and self._climb_fallback_xy is not None:
                # Overshoot unreachable: retry once against the staircase itself
                # before writing the attempt off.
                self._climb_goal_xy = self._climb_fallback_xy
                self._climb_fallback_xy = None
                self._current_path = None
                action = self._follow_to(frame, self._climb_goal_xy)

            # `action is None` means navigation has nothing left to do: it
            # arrived at the goal, or could not path there at all. Either way the
            # climb is over -- and since the floor check above did not fire, it
            # did not work. Ending here rather than turning in place matters: the
            # probe run burned exactly 20 steps per attempt (the stall window)
            # spinning after the follower had already given up.
            stalled = action is None or (
                self.step_count - self._climb_ref_step >= 20
                and abs(y_obs - self._climb_ref_y) < 0.15
            )
        if stalled or self.step_count > self._climb_deadline:
            # Retire this staircase by its CELLS, not its centroid. The hit grid
            # keeps accumulating, so a component grows and its centroid drifts,
            # and a centroid blacklist lets the same unusable stairwell come
            # back as a "new" staircase every selection round -- measured at 72
            # attempts in a single episode, all of the same place.
            # Retiring the component's cells removes 89% of repeat attempts but
            # measured net -1 on 50 episodes: those repeats were nearly free
            # (most failed within a single step), while permanently retiring a
            # real staircase after one bad approach costs a success. Off by
            # default; the centroid blacklist below is the part that pays.
            if (
                self.stair_detector is not None
                and self._climb_cells is not None
                and getattr(self.cfg.exploration, "stair_retire_cells", False)
            ):
                self.stair_detector.disable(self.floors.current(), self._climb_cells)
            # The CENTROID, not the goal. _stair_frontiers filters new
            # detections by centroid distance, and the goal sits
            # stair_overshoot_m (1.5 m) beyond it -- further than the 1.0 m
            # match radius -- so recording the goal meant the filter never
            # matched anything and the same stairwell was retried every round.
            self._disabled_stairs.append(self._climb_centroid_xy.copy())
            self.stats["climb_fail"] = self.stats.get("climb_fail", 0) + 1
            self.state = State.EXPLORE
            self._current_frontier = None
            self._current_path = None
            return self._act_inner_post_transition(frame)
        if abs(y_obs - self._climb_ref_y) >= 0.15:
            self._climb_ref_y, self._climb_ref_step = y_obs, self.step_count
        return action

    def _select_new_frontier(self, frame: FrameData) -> None:
        # Extraction + top-N path planning is expensive; while waiting the
        # agent turns in place, which grows the map anyway. ASCENT pays this
        # every step (`ascent_policy.py:684`) because its selection has no A* in
        # it -- see `reselect_every`.
        if self.step_count - self._last_select_step < self._select_every:
            return
        self._last_select_step = self.step_count
        with self.profiler.timeit("frontier_extract"):
            frontiers = self.frontier_extractor.extract(
                self.costmap,
                frame.camera_position[list(PLANE)],
                floor_key=self.floors.current().key,
            )
        # How many candidates each round, and how many rounds. The contour
        # extractor should produce FEWER than WFD -- it filters out frontiers
        # that only open a small pocket, and each survivor costs one of the
        # top_n path plans below. If this does not drop, the 0/1 mask
        # precondition in _filter_out_small_unexplored is not being met and the
        # area filter is silently inert.
        if self.frontier_semantics is not None:
            n_new = self.frontier_semantics.bind(frontiers, self.step_count)
            self.stats["frontier_sem_new"] = (
                self.stats.get("frontier_sem_new", 0) + n_new
            )
            self.stats.update(self.frontier_semantics.stats())
        self.stats["frontier_rounds"] = self.stats.get("frontier_rounds", 0) + 1
        self.stats["frontier_seen"] = self.stats.get("frontier_seen", 0) + len(frontiers)
        # Total frontier size, so frontier_cells/frontier_seen gives the mean.
        # Needed to tell "more, finer openings" apart from "more slivers": the
        # contour extractor has no minimum length and no dedup, where WFD drops
        # anything under frontier_min_cells and merges within frontier_dedup_m.
        self.stats["frontier_cells"] = self.stats.get("frontier_cells", 0) + sum(
            int(f.size) for f in frontiers
        )

        self._mark_floor_explored(len(frontiers))
        self._ask_floor(frame)
        stair_frontiers, stair_scores = self._stair_frontiers()
        frontiers = list(frontiers) + stair_frontiers
        if not frontiers:
            return
        # Async scoring request (never blocks); use whatever scores exist now.
        # Stair frontiers carry a fixed prior instead: a stairwell's similarity
        # to "a bed is ahead" is meaningless, and its value is entirely about
        # whether this floor is worth leaving.
        self.scorer.request(frontiers, self.scene_graph, self.target, self.keyframes)
        scores = dict(self.scorer.latest())
        scores.update(stair_scores)
        frontier_values = self._frontier_values(frontiers)
        blocked = self._blocked_ids(frontiers)
        agent_xy = frame.camera_position[list(PLANE)]
        heading_xy = self._heading_xy(frame)
        failed: set = set()
        # ASCENT's LLM decides among the top-k by value and its pick is
        # authoritative if it is reachable. Falling through to the scorers
        # otherwise keeps one commit path for every branch below.
        best = self._rank_frontiers(frontiers, frontier_values, blocked, agent_xy)
        if best is None:
            with self.profiler.timeit("frontier_select"):
                if self.commit_state is not None and not self._ascent_rank:
                    # Commitment without ASCENT's ranking: retired frontiers join
                    # the blocked set and the pick is recorded below, but
                    # score/path_cost still decides. This is what separates F3 from
                    # F2 -- the "ascent" selector changes both at once.
                    blocked = set(blocked) | {
                        f.id for f in frontiers
                        if self.commit_state.is_disabled(f.centroid_xy)
                    }
                if self._ascent_rank:
                    # ASCENT's ranking. Stair frontiers carry a prior in `scores`
                    # rather than a value-map reading, so merge the two sources --
                    # otherwise a stairwell would rank at 0 and never be chosen.
                    values = dict(frontier_values or {})
                    for fid, s in scores.items():
                        values.setdefault(fid, s)
                    best = select_frontier_ascent(
                        frontiers,
                        values,
                        self.selection_planner,
                        self.costmap,
                        agent_xy,
                        self.commit_state,
                        nearby_distance_m=float(
                            getattr(self.cfg.exploration, "nearby_distance_m", 3.0)
                        ),
                        min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                        top_n=self.cfg.exploration.top_n_frontiers,
                        blocked=blocked,
                        failed_out=failed,
                    )
                else:
                    best = select_frontier(
                        frontiers,
                        scores,
                        self.selection_planner,
                        self.costmap,
                        agent_xy,
                        unscored_prior=self.cfg.exploration.unscored_prior,
                        min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                        top_n=self.cfg.exploration.top_n_frontiers,
                        blocked=blocked,
                        failed_out=failed,
                        info_gain_weight=self.cfg.exploration.info_gain_weight,
                        info_gain_radius_m=self.cfg.exploration.info_gain_radius_m,
                        los_visibility_penalty=self.cfg.exploration.los_visibility_penalty,
                        heading_xy=heading_xy,
                        continuity_weight=self.cfg.exploration.continuity_weight,
                        frontier_values=frontier_values,
                        value_weight=float(getattr(self.cfg.exploration, "value_weight", 1.0)),
                        value_argmax=bool(
                            getattr(self.cfg.exploration, "value_argmax", False)
                        ),
                    )
        by_id = {f.id: f for f in frontiers}
        for fid in failed:  # block only the candidates that actually failed
            self._block_frontier(by_id.get(fid), 50)
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
                    if np.linalg.norm(f.centroid_xy - self._last_giveup_pt) < 0.6
                }
            # Deliberately the utility selector even on the ASCENT arm: this is a
            # deadlock breaker, not a ranking policy, and ASCENT's own retirement
            # set can empty the candidate list the same way blocks can. Reaching
            # *something* matters more here than which something.
            if len(relaxed_blocked) < len(frontiers):
                best = select_frontier(
                    frontiers, scores, self.selection_planner, self.costmap,
                    agent_xy, unscored_prior=self.cfg.exploration.unscored_prior,
                    min_path_cost_m=self.cfg.exploration.min_path_cost_m,
                    top_n=self.cfg.exploration.top_n_frontiers, blocked=relaxed_blocked,
                    info_gain_weight=self.cfg.exploration.info_gain_weight,
                    info_gain_radius_m=self.cfg.exploration.info_gain_radius_m,
                    los_visibility_penalty=self.cfg.exploration.los_visibility_penalty,
                    heading_xy=heading_xy,
                    continuity_weight=self.cfg.exploration.continuity_weight,
                )
        if best is None or best.path_cost is None:
            self.stats["select_none"] += 1
            return
        self.stats["select_ok"] += 1
        if self.commit_state is not None and not self._ascent_rank:
            # select_frontier_ascent records its own pick; the utility path has
            # to, or nothing ever accumulates and no frontier is ever retired.
            self.commit_state.observe(best, agent_xy)
        if self.commit_state is not None:
            self.stats["frontier_retired"] = len(self.commit_state.disabled)
        # Per-selection trace (step, agent xy, chosen frontier xy, path cost,
        # #frontiers) for exploration-efficiency debugging. See scripts.
        self.frontier_select_log.append((
            self.step_count,
            [round(float(x), 2) for x in agent_xy],
            [round(float(x), 2) for x in best.centroid_xy],
            round(float(best.path_cost), 2) if best.path_cost is not None else None,
            len(frontiers),
        ))
        if str(getattr(best, "kind", "explore")).startswith("stair_"):
            self.stats["stair_frontier_selected"] = (
                self.stats.get("stair_frontier_selected", 0) + 1
            )
        prev = self._current_frontier
        same_target = (
            prev is not None
            and float(np.linalg.norm(prev.centroid_xy - best.centroid_xy)) < 0.5
        )
        self._current_frontier = best
        self._plan_to(frame, frontier_goal_xy(best, self.costmap))
        if self._current_path is not None:
            if not same_target:
                self.stats["frontier_switch"] = self.stats.get("frontier_switch", 0) + 1
            self.state = State.GOTO_FRONTIER
            # A fresh pursuit starts its own 15-step progress window; without
            # this the give-up timer carried over from whatever frontier was
            # pursued (or given up on) before, and could fire on the very
            # first step of the new pursuit based on stale position data.
            if not same_target:
                # A fresh pursuit starts its own progress window. Re-selecting
                # the SAME frontier must not restart it, or the stall rule can
                # never accumulate its 20 steps and stops existing.
                self._progress_ref_step = self.step_count
                self._progress_ref_xy = agent_xy.copy()
                self._frontier_ref_dist = None
        else:
            self._block_frontier(best, 50)

    # ------------------------------------------------------------- candidates

    def _check_candidates(self) -> None:
        candidates = self.object_layer.candidates(
            self.target,
            min_obs=self.cfg.verification.min_obs,
            min_score=self.cfg.verification.min_score,
            min_bbox_px=self.cfg.verification.min_bbox_px,
            min_evidence=self.cfg.verification.min_evidence,
            step=self.step_count,
        )
        if not candidates:
            return
        if self.steps_to_first_candidate is None:
            self.steps_to_first_candidate = self.step_count
        track = candidates[0]
        self._candidate_id = track.id
        self._center_turns = 0  # fresh centering budget for this candidate
        obj_xy = self.object_layer.center_of(track)[list(PLANE)]

        # Navmesh alignment (old stack): navigate straight to the object
        # position and let Habitat's navmesh drive there, then STOP on arrival
        # -- like publishing /goal_object. No viewpoint pre-positioning.
        if self._direct_approach:
            # Don't commit to a target on a disconnected navmesh island (a
            # visible-but-unreachable object, e.g. in a sealed bathroom): the
            # agent can never get there, so blacklist it and keep exploring for
            # a reachable goal instead of stopping and failing the episode.
            #
            # Only the navmesh arm can ask this in advance. Sensor-only, there
            # is no such oracle, so `pointnav` commits and abandons on a step
            # budget instead -- see approach_abandon_steps in _do_approach.
            if self._reachable_fn is not None and not self._reachable_fn(obj_xy):
                self.object_layer.blacklist(track.id)
                self._candidate_id = None
                self.stats["unreachable_skip"] = self.stats.get("unreachable_skip", 0) + 1
                return
            # VLM verify the candidate before committing (no VERIFYING state in
            # navmesh mode). Reject -> blacklist and keep exploring; this is the
            # only FP gate in the navmesh path.
            if self.verifier is not None:
                with self.profiler.timeit("verification"):
                    ok = self.verifier.verify(track, self.target)
                if not ok:
                    # A rejection is evidence about THIS VIEW, not about the
                    # object. ASCENT re-checks its gate every step until it
                    # passes and only abandons a target at close range
                    # (map_controller.py:770-776, ascent_policy.py:910-922);
                    # OSG asked once from wherever it stood and never asked
                    # again. Measured cost: 3 of 22 episodes ended within 0.5 m
                    # of the goal unable to stop, because the only candidate
                    # they had was permanently blacklisted from across a room.
                    cooldown = int(
                        getattr(self.cfg.verification, "reject_cooldown_steps", 0)
                    )
                    if cooldown > 0:
                        self.object_layer.suppress(track.id, self.step_count + cooldown)
                    else:
                        self.object_layer.blacklist(track.id)
                    self._candidate_id = None
                    self.stats["verify_reject"] = self.stats.get("verify_reject", 0) + 1
                    return
            self._start_approach(obj_xy)
            return

        # Always pre-position at a viewpoint from which the object is visible
        # before approaching -- HM3D success requires stopping at such a pose,
        # not merely near the object's 3D center. When verification is off the
        # VERIFYING state simply skips the VLM call (see _do_verification).
        view_xy = self.viewpoint_planner.approach_viewpoint(obj_xy, self.costmap)
        if view_xy is None:
            return  # not yet observable from mapped space; keep exploring
        self._goal_xy = view_xy
        self.state = State.GOTO_VERIFY_VIEW
        self._current_path = None
        self._goto_deadline = self.step_count + 80

    def _do_verification(self, frame: FrameData) -> str:
        track = self.object_layer.get(self._candidate_id) if self._candidate_id is not None else None
        if track is None:
            self.state = State.EXPLORE
            return TURN_ACTION
        from ..planning.controller import TURN_LEFT, TURN_RIGHT, _wrap, agent_heading

        # Center-then-verify: if a VLM verifier is active and the target is
        # actually visible in the live view, bring its detection to the middle
        # of the camera before the VLM call, then verify that well-framed frame.
        if (
            self.verifier is not None
            and getattr(self.cfg.verification, "center_before_verify", True)
        ):
            det = self._best_target_detection(frame)
            if det is not None:
                bbox_cx = 0.5 * (float(det.bbox_xyxy[0]) + float(det.bbox_xyxy[2]))
                offset = float(np.arctan2(bbox_cx - frame.intrinsics.cx, frame.intrinsics.fx))
                if (
                    abs(offset) > np.radians(self.cfg.verification.center_tol_deg)
                    and self._center_turns < self.cfg.verification.center_max_turns
                ):
                    self._center_turns += 1
                    self.stats["center_turn"] = self.stats.get("center_turn", 0) + 1
                    # target right of centre (offset>0) -> turn right to centre it
                    return TURN_RIGHT if offset > 0 else TURN_LEFT
                # Centred (or out of centring budget): verify the live framed view.
                with self.profiler.timeit("verification"):
                    accepted = (
                        True if self._terminal_verify
                        else self.verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.target)
                    )
                if accepted:
                    obj_xy = self.object_layer.center_of(track)[list(PLANE)]
                    self._start_approach(obj_xy, agent_xy=frame.camera_position[list(PLANE)])
                    return self._do_approach(frame)
                self.object_layer.blacklist(track.id)
                self._candidate_id = None
                self.state = State.EXPLORE
                return TURN_ACTION
            # target not visible in the live view -> fall through to the
            # 3D-facing / best-cam recovery below.

        # Face the object first so the live view actually shows it.

        obj_xy = self.object_layer.center_of(track)[list(PLANE)]
        agent_xy = frame.camera_position[list(PLANE)]
        to_obj = obj_xy - agent_xy
        if np.linalg.norm(to_obj) > 0.05:
            err = _wrap(float(np.arctan2(to_obj[1], to_obj[0])) - agent_heading(frame.T_wc))
            if abs(err) > np.radians(20.0):
                return TURN_RIGHT if err > 0 else TURN_LEFT
        # HM3D success viewpoints require the object to actually be VISIBLE
        # from the stop pose; 2D line-of-sight misses desk-height occluders
        # (we stopped 0.12 m outside the viewpoint set). If the detector
        # cannot see the target from here, return to the pose the best
        # detection was made from — proven reachable AND proven visible
        # (ring alternatives proved unreachable and thrashed the deadline).
        if (
            not self._target_visible(frame)
            and not self._went_to_best_cam
            and track.best_cam_xy is not None
            and np.linalg.norm(track.best_cam_xy - agent_xy) > 0.35
        ):
            self._went_to_best_cam = True
            self._goal_xy = track.best_cam_xy.copy()
            self.state = State.GOTO_VERIFY_VIEW
            self._current_path = None
            self._goto_deadline = self.step_count + 60
            return self._follow_path(frame) or TURN_ACTION
        with self.profiler.timeit("verification"):
            # Pre-approach verification. Skipped (accept) when the verifier is
            # off (old-fidelity mode) OR in terminal-view mode, where the
            # decisive VLM check is deferred to the STOP moment in _do_approach.
            accepted = (
                True if (self.verifier is None or self._terminal_verify)
                else self.verifier.verify(track, self.target, live_view=frame.rgb)
            )
        if accepted:
            obj_xy = self.object_layer.center_of(track)[list(PLANE)]
            self._start_approach(obj_xy, agent_xy=frame.camera_position[list(PLANE)])
            return self._do_approach(frame)
        self.object_layer.blacklist(track.id)
        self._candidate_id = None
        self.state = State.EXPLORE
        return TURN_ACTION

    # ---------------------------------------------------------------- helpers

    def _start_approach(self, obj_xy: np.ndarray, agent_xy: Optional[np.ndarray] = None) -> None:
        if self._direct_approach:
            # Navigate to the object itself; the navmesh snaps to the nearest
            # standable point (effectively a viewpoint), like old /goal_object.
            self._goal_xy = obj_xy.copy()
        elif getattr(self.cfg.agent, "approach_navigable_goal", False) and agent_xy is not None:
            self._goal_xy = self._approach_goal_xy(obj_xy, agent_xy)
        else:
            self._goal_xy = self._nearest_free_xy(obj_xy)
        self._target_obj_xy = obj_xy.copy()
        # Diagnostic only: what ASCENT would have aimed at from here. Nothing
        # navigates to it yet.
        # The navmesh path calls this without an agent position (it has no
        # frame in scope), which is every configuration measured here -- so fall
        # back to the position recorded this step rather than silently skipping.
        here = agent_xy if agent_xy is not None else self._agent_xy
        track = (self.object_layer.get(self._candidate_id)
                 if self._candidate_id is not None else None)
        self._target_cloud_xy = (
            self.object_layer.nearest_point_xy(track, here)
            if track is not None and here is not None else None
        )
        # A fresh approach gets a fresh double-check: the score has to be earned
        # against THIS candidate, not carried over from a previous one.
        #
        # Deliberate deviation from ASCENT, which resets _double_check_goal only
        # in its episode-level reset (map_controller.py:177, alongside
        # _target_object = ""), so one latched target vouches for every later
        # one in the same episode -- its gate can only ever catch a false
        # positive approached before anything scored well. Per-approach is the
        # reading that matches the intent ("was THIS target vouched for").
        self._approach_itm_max = 0.0
        self._approach_itm_n = 0
        self.state = State.APPROACH
        self._current_path = None
        self._path_goal = None
        if self._direct_approach:
            # A self-planning mover drives the FULL distance to the object (no viewpoint
            # pre-positioning), so the short-leg cap (approach_max_steps ~= 3 m)
            # cuts the approach off while the target is still in view. Let it
            # navigate to the object, bounded only by a generous deadline.
            self._goto_deadline = self.step_count + self.cfg.agent.navmesh_approach_steps
            self._approach_steps_left = 10 ** 9
        else:
            self._goto_deadline = self.step_count + 100
            self._approach_steps_left = self.cfg.agent.approach_max_steps
        self._approach_start_step = self.step_count
        # Snapshot the terminal approach for navigation diagnostics: the goal
        # cell (nearest-free to the mapped object), how far it sits from the
        # object, the agent's start distance to it, and the costmap status of
        # the goal cell -- an UNKNOWN/OCCUPIED goal cell explains an
        # unreachable-path stop. min_dist_to_goal is filled in per step.
        self.approach_diag = {
            "goal_xy": [float(x) for x in self._goal_xy],
            "obj_xy": [float(x) for x in obj_xy],
            "goal_to_obj_m": float(np.linalg.norm(self._goal_xy - obj_xy)),
            "goal_cell": self._cell_status(self._goal_xy),
            "start_step": self.step_count,
            "min_dist_to_goal_m": None,
            "plan_fail": 0,
            "path_consumed_cause": None,
        }
        self._approach_last_good_xy = None
        self._terminal_min_d = float("inf")
        self._terminal_last_xy = None
        self._terminal_stalls = 0

    def _detections(self, frame: FrameData) -> list:
        """Detections for this frame, computed at most once.

        The detector was already being run twice on some steps -- once for the
        keyframe object-layer update and once for the terminal approach check.
        Keying on (step, frame_id, frame identity) rather than frame_id alone
        matters because tests drive _do_approach directly with fresh frames
        that all carry frame_id 0.
        """
        key = (self.step_count, frame.frame_id, id(frame))
        if self._det_key == key:
            return self._det_cache
        with self.profiler.timeit("detector"):
            dets = self.detector.detect(frame.rgb)
        self._det_key, self._det_cache = key, dets
        return dets

    def _best_target_detection(self, frame: FrameData) -> Optional[Detection]:
        """Highest-confidence detection of the target category, or None."""
        target = self.target.lower().replace("_", " ").strip()
        dets = self._detections(frame)
        matches = [
            d for d in dets
            if d.label.lower().replace("_", " ").strip() == target and d.score > 0.25
        ]
        return max(matches, key=lambda d: d.score) if matches else None

    def _target_visible(self, frame: FrameData) -> bool:
        """Does the detector see the target category in the current view?"""
        return self._best_target_detection(frame) is not None

    @staticmethod
    def _detection_depth(det: Detection, frame: FrameData) -> Optional[float]:
        """Median metric depth (m) over the detection's mask, using only valid
        depth pixels; None if too few valid samples (mask off the depth range)."""
        ys, xs = np.nonzero(det.mask)
        if ys.size == 0:
            return None
        d = frame.depth[ys, xs]
        valid = d > 1e-3
        if int(valid.sum()) < 8:
            return None
        return float(np.median(d[valid]))

    def _plan_to(
        self, frame: FrameData, goal_xy: np.ndarray, goal_tolerance_m: Optional[float] = None
    ) -> None:
        agent_xy = frame.camera_position[list(PLANE)]
        with self.profiler.timeit("planner"):
            result: PlanResult = self.selection_planner.plan(
                self.costmap, agent_xy, goal_xy, goal_tolerance_m
            )
        self._current_path = result.path if result.success else None
        self.stats["plan_ok" if result.success else "plan_fail"] += 1

    def _follow_path(self, frame: FrameData) -> Optional[str]:
        goal = (
            frontier_goal_xy(self._current_frontier, self.costmap)
            if self.state == State.GOTO_FRONTIER and self._current_frontier is not None
            else self._goal_xy
        )
        if goal is None:
            return None
        if self._driver is not None:
            # A mover that plans for itself (navmesh or pointnav). None =
            # arrived-or-cannot-get-there; if we're still far from a frontier
            # goal, block it (as the stub-block does).
            if self.pointnav is not None:
                nav = self.pointnav.step(goal)
                action, reason = nav.action, nav.reason
                if (
                    reason == "policy_stop"
                    and self.state == State.GOTO_FRONTIER
                    and not getattr(self.cfg.agent, "pointnav_stop_means_blocked", True)
                ):
                    # ASCENT treats a network STOP on an explore frontier as
                    # noise and overwrites it with one forward step, keeping the
                    # same target (ascent_policy.py:705-711). Retiring the
                    # frontier instead makes one spurious STOP cost a whole
                    # pursuit plus a 100-step block. The grind this could cause
                    # is what `escape_window` and `frontier_stick_rule: closing`
                    # are for, and both are on in this preset.
                    self.stats["pointnav_stop_forced_forward"] = (
                        self.stats.get("pointnav_stop_forced_forward", 0) + 1
                    )
                    return "move_forward"
            else:
                action = self._driver(goal)
            if (
                action is None
                and self.state == State.GOTO_FRONTIER
                and self._current_frontier is not None
                and np.linalg.norm(frame.camera_position[list(PLANE)] - goal)
                > self._frontier_reach_m
            ):
                self._block_frontier(self._current_frontier, 100)
                self._last_giveup_pt = self._current_frontier.centroid_xy.copy()
                self.stats["frontier_stub_block"] = self.stats.get("frontier_stub_block", 0) + 1
            return action
        if self._current_path is None:
            self._plan_to(frame, goal)
            if self._current_path is None:
                if self.state == State.GOTO_FRONTIER:
                    self._block_frontier(self._current_frontier, 50)
                return None
        action = self.controller.act(frame.T_wc, self._current_path)
        if action is None:
            self._current_path = None
            # Fix A: the controller returns None the moment the planned path
            # ends within its arrival tolerance of the agent. When the planner
            # hands back a degenerate stub path (goal snapped near the start
            # because the real frontier is unreachable), that reads as a false
            # "arrival": the FSM drops GOTO_FRONTIER->EXPLORE without blocking
            # the frontier, so the same one is re-selected every cycle and the
            # agent freezes in place (give-up never fires -- it only counts
            # elapsed steps *inside* GOTO_FRONTIER, and the re-entry resets its
            # timer). If we are still far from the frontier goal, this was not a
            # real arrival: block the frontier so a different one is chosen next.
            if (
                self.state == State.GOTO_FRONTIER
                and self._current_frontier is not None
                and np.linalg.norm(frame.camera_position[list(PLANE)] - goal)
                > self._frontier_reach_m
            ):
                self._block_frontier(self._current_frontier, 100)
                # Also mark it as the last give-up point so the "all frontiers
                # blocked" relaxed fallback (which deliberately ignores the
                # blacklist) doesn't immediately re-pursue this same stub.
                self._last_giveup_pt = self._current_frontier.centroid_xy.copy()
                self.stats["frontier_stub_block"] = self.stats.get("frontier_stub_block", 0) + 1
        return action

    def _follow_to(self, frame: FrameData, goal_xy: np.ndarray) -> Optional[str]:
        """Follow a path to an explicit goal, replanning when the goal
        changes (APPROACH alternates between an advance goal and a retreat
        goal within the same state, unlike the other terminal states which
        have one fixed goal for their whole visit).

        Used only by APPROACH, so tightens both the planner's and the
        controller's stopping tolerance beyond the loose defaults used for
        frontier/verify-view travel (P1f): geometry analysis against
        HM3D's actual view_points showed several near-miss episodes
        stopped within 5-8 cm (straight-line) of a real view_point, yet
        habitat's geodesic distance_to_goal still read 0.15-0.17 m --
        the default 0.2-0.3 m tolerances left slack for a short geodesic
        detour around a nearby thin obstacle to blow the 0.13 m success
        radius even when we were geometrically almost there.
        """
        if self._driver is not None:
            if self.pointnav is not None:
                # ASCENT stops consulting the network near a committed target
                # and creeps forward (ascent_policy.py:920-927); the mover's own
                # 0.9 m stop radius is far outside the success distance, so
                # letting it declare arrival here would strand every approach.
                #
                # APPROACH only. CLIMB shares this method, and there the creep
                # would suppress the `action is None` that ends a climb whose
                # goal is unreachable -- leaving it to the height-stall test and
                # the deadline, which take 20 and 80 steps respectively.
                creep = (
                    float(self.cfg.agent.pointnav_approach_creep_m)
                    if self.state is State.APPROACH else 0.0
                )
                # The arrival signal the navmesh supplied for free. Habitat's
                # follower reports "arrived" within `navmesh_goal_radius` and
                # OSG turns that into a stop for 46 of 100 episodes; PointNav
                # emits motion and nothing else, so 8 of the 20 episodes it
                # times out on are ones where it REACHED the object and had no
                # way to conclude it. 0 keeps the old signal-less behaviour.
                arrive = float(getattr(self.cfg.agent, "pointnav_arrival_m", 0.0))
                return self.pointnav(goal_xy, creep_below=creep, stop_radius=arrive)
            return self._nav_fn(goal_xy)  # navmesh drives to the object; None = arrived
        need_replan = (
            self._current_path is None
            or self._path_goal is None
            or np.linalg.norm(self._path_goal - goal_xy) > 0.05
        )
        if need_replan:
            self._plan_to(frame, goal_xy, goal_tolerance_m=self.cfg.agent.approach_goal_tolerance_m)
            self._path_goal = goal_xy.copy() if self._current_path is not None else None
            if self._current_path is None:
                # planner could not reach goal_xy (costmap disconnected /
                # goal in unknown/inflated space) -- distinct from the
                # controller reporting arrival below.
                self._last_follow_none_reason = "planner_no_path"
                return None
        action = self.controller.act(
            frame.T_wc, self._current_path, arrival_tol_m=self.cfg.agent.approach_arrival_tol_m
        )
        if action is None:
            self._current_path = None
            self._path_goal = None
            # controller consumed the path (thinks it arrived at goal_xy) --
            # a false "arrival" here means the planned path was a stub / ended
            # short of the true goal.
            self._last_follow_none_reason = "controller_arrived"
        return action

    def _approach_goal_xy(self, obj_xy: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
        """Navigable approach goal: a standoff point at approach_standoff_m from
        the object along the ray toward the agent (the side the object was
        observed from -> open, reachable space), snapped to the nearest free
        cell. Avoids the enclosed-pocket goals that _nearest_free_xy produces
        by placing the goal in front of the object rather than hard against it.
        """
        standoff = float(self.cfg.agent.approach_standoff_m)
        to_agent = agent_xy - obj_xy
        dist = float(np.linalg.norm(to_agent))
        if dist < 1e-3:
            return self._nearest_free_xy(obj_xy)
        # If the agent is already closer than the standoff, keep the goal at the
        # standoff (do not push it behind the agent past the object).
        cand = obj_xy + (to_agent / dist) * min(standoff, dist)
        return self._nearest_free_xy(cand)

    def _cell_status(self, xy: np.ndarray) -> str:
        """Costmap classification of a world point: free/occupied/unknown/oob."""
        from ..mapping.costmap import FREE, OCCUPIED, UNKNOWN

        rc = self.costmap.world_to_grid(xy)
        h, w = self.costmap.grid.shape
        if not (0 <= rc[0] < h and 0 <= rc[1] < w):
            return "oob"
        v = self.costmap.grid[rc[0], rc[1]]
        return {FREE: "free", OCCUPIED: "occupied", UNKNOWN: "unknown"}.get(int(v), str(int(v)))

    def _nearest_free_xy(self, xy: np.ndarray) -> np.ndarray:
        """Nearest FREE cell to a (possibly occupied) object position — the
        closest pose the agent can actually stand at."""
        from ..mapping.costmap import FREE

        rc = self.costmap.world_to_grid(xy)
        h, w = self.costmap.grid.shape
        best, best_d = xy, np.inf
        rad = int(1.5 / self.costmap.resolution)
        r0, r1 = max(0, rc[0] - rad), min(h, rc[0] + rad + 1)
        c0, c1 = max(0, rc[1] - rad), min(w, rc[1] + rad + 1)
        free = np.argwhere(self.costmap.grid[r0:r1, c0:c1] == FREE)
        if free.shape[0] == 0:
            return xy
        free_world = self.costmap.grid_to_world(free + np.array([r0, c0]))
        d = np.linalg.norm(free_world - xy, axis=1)
        return free_world[int(np.argmin(d))]
