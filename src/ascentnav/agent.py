"""ASCENT's navigation policy, transcribed for OSG's harness.

This is `Ascent_Policy.act` (`relative_work/ascent/ascent/ascent_policy.py`)
and the `Map_Controller` it drives, for one environment, on ASCENT's own maps
(vendored under `ascentnav/mapping/`) and ASCENT's own served models
(`scripts/serve_perception.sh`). Every rule cites the reference line it was
transcribed from. Where the reference is inconsistent with itself, the
reference wins and the inconsistency is documented -- the previous port
"fixed" several of those and measured 11 points worse.

What the rewrite closes (S71; see the plan and the two audits it came from):

  H2  the explore / no-frontier / frontier-retirement path: per-floor disabled
      frontiers, no sticky rule on a floor's last frontier, the 20-selection
      disable, stairwell reinitialisation with the tight frontier threshold,
      and the strict RedNet-AND-GroundingDINO stair mask. The port's version
      exhausted frontiers early and climbed stairs on 11 same-floor episodes,
      17.5% of all steps against the reference's 9.2%.
  H1  the lockouts against a premature, irreversible STOP: 13 opening turns
      before the goal is consulted, a gate that can only latch on a step
      AFTER `try_to_navigate` was set, a stall test that needs a second step
      inside the metre, and a 0.8 ingest bar.
  A1  the gate itself: the value map's BLIP-2 cosine >= 0.15, read on the
      NEXT step's object-map update, latched for the rest of the episode and
      never cleared by a failed approach.
  A2  a PointNav STOP during an approach is a STOP.
  A4  every target detection in the frame is ingested, with its own mask.
  A5  a failed approach burns its cells and RETURNS AN EXPLORE ACTION on the
      same step.
  A7  `ValueMap(use_max_confidence=False)`.
  A10 `filter_depth` on the depth the maps see; raw depth to the mover.
  A11 `_floor_num_steps`, which keys the prompt content and the reinit window.
  A12 maps anchored at the episode start, not the world origin.

Removed (OSG-only, no counterpart in the reference): the VLM verifier, the
commit gate, burn-on-refusal, the arrival gate, weak memory, scan-on-arrival,
the displacement escape. Kept: the behaviour recorder and every field the
runner reads.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from osg.core.types import Detection, FrameData
from osg.eval.behaviour_log import BehaviourLog
from osg.graph.scene_graph import SceneGraph
from osg.perception.stair_seg import MIN_STAIR_PIXELS
from osg.planning.escape import ActionHistoryEscape

from .constants import INITIALIZE_TURNS, STAIR_CLASS_ID
from .depth_filter import filter_depth
from .geometry import (
    EpisodeAnchor,
    camera_pitch,
    episodic_xy_heading,
    normalise_depth,
    robot_xy_heading,
    xyz_yaw_pitch_roll_to_tf_matrix,
)
from .mapping.object_point_cloud_map import ObjectPointCloudMap
from .mapping.obstacle_map import ObstacleMap
from .mapping.value_map import ValueMap
from .perception import tag_scene
from .planner import GO_DOWN, GO_UP, AscentLLMPlanner, KnowledgeGraph
from .stairs import (
    StairController,
    carrot_waypoint,
    ratchet_carrot,
    robot_on_stairs,
    stairs_in_upper_half,
)

STOP, FORWARD, LEFT, RIGHT = "stop", "move_forward", "turn_left", "turn_right"
LOOK_UP, LOOK_DOWN = "look_up", "look_down"
MAP_SIZE = 1600            # ascent/map_controller.py:35
PITCH_OFFSET_DEG = 30      # ascent config `look_down.tilt_angle`

# The reference's target strings are COCO names (`HM3D_ID_TO_NAME`, vlfm
# habitat_policies.py:28); its BLIP-2 prompt and LLM `Goal` are built from
# them and the 0.15 gate was calibrated on them. The object-map key stays the
# HM3D name so the record and the detector filter are unchanged.
HM3D_TO_COCO = {
    "chair": "chair", "bed": "bed", "toilet": "toilet",
    "tv_monitor": "tv", "sofa": "couch", "plant": "potted plant",
}

# ASCENT's Qwen system message (`model_api/qwen25_ollama.py:29-46`).
LLM_SYSTEM = (
    "You are an AI assistant with advanced spatial reasoning capabilities. "
    "Your task is to choose the optimal option to find the target object."
)

_PRIORS: dict = {}


def _load_priors(cfg) -> tuple:
    """Knowledge graph + floor table, loaded once per process."""
    if "kg" not in _PRIORS:
        kg_path = str(getattr(cfg.exploration, "knowledge_graph_path",
                              "relative_work/ascent/statistic_priors/knowledge_graph.json"))
        fp_path = str(getattr(cfg.exploration, "floor_prior_path", "data/priors/hm3d_floor_prior.json"))
        try:
            _PRIORS["kg"] = KnowledgeGraph.load(kg_path)
        except OSError:
            _PRIORS["kg"] = None
        try:
            with open(fp_path) as f:
                _PRIORS["floor"] = json.load(f)
        except OSError:
            _PRIORS["floor"] = {}
    return _PRIORS["kg"], _PRIORS["floor"]


class AscentNavAgent:
    """`Ascent_Policy` for one environment, behind OSG's `act(frame)` contract."""

    def __init__(
        self,
        cfg,
        detector,
        scorer,
        verifier,
        target_category: str,
        *,
        image_text=None,
        pointnav=None,
        ranker=None,
        room_classifier=None,
        stair_segmenter=None,
        stair_detector=None,
        ram=None,
        profiler=None,
        **_ignored,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.image_text = image_text
        self.pointnav = pointnav
        self.room_classifier = room_classifier
        self.stair_segmenter = stair_segmenter
        self.stair_detector = stair_detector
        self.ram = ram
        self.profiler = profiler

        a, e = cfg.agent, cfg.eval
        self.camera_height = float(a.camera_height)
        self.min_depth = float(getattr(e, "depth_min_m", 0.5))
        self.max_depth = float(getattr(e, "depth_max_m", 5.0))
        self.hfov = float(np.radians(e.hfov_deg))
        self.fx = self.fy = e.rgb_width / (2 * np.tan(self.hfov / 2))
        self.cx = e.rgb_width / 2.0
        self.max_steps = int(a.max_steps)
        self.stop_radius = float(getattr(a, "pointnav_stop_radius", 0.9))
        self.abandon_steps = int(getattr(a, "approach_abandon_steps", 100) or 100)
        self.gate_threshold = float(getattr(a, "blip_gate_threshold", 0.15))
        self.downstair_detector = str(getattr(a, "downstair_detector", "ascent"))
        if self.downstair_detector not in ("ascent", "lip"):
            raise ValueError(f"agent.downstair_detector must be 'ascent' or 'lip', got {self.downstair_detector!r}")
        self.initialize_turns = int(getattr(a, "initialize_turns", INITIALIZE_TURNS))
        self.escape = ActionHistoryEscape(int(getattr(a, "escape_window", 30) or 30))
        self.stair_up_mode = str(getattr(a, "stair_up_mode", "ascent"))
        passive = bool(getattr(a, "passive_stair_entry", True))
        if passive and self.stair_up_mode != "ascent":
            # The union mask fires on far more pixels than the AND; letting it
            # also pull the agent onto a staircase passively is how a same-floor
            # episode gets spent climbing.
            raise ValueError("passive_stair_entry needs stair_up_mode: ascent (the strict fusion)")
        value_model = str(getattr(cfg.exploration, "value_model", "clip"))
        if value_model == "blip2itm" and image_text is not None:
            from osg.perception.image_text import Blip2ItmScorer

            if not isinstance(image_text, Blip2ItmScorer):
                raise TypeError("ascentnav's gate reads the value map's BLIP-2 cosine; "
                                "`value_model: blip2itm` needs a Blip2ItmScorer")
        self.value_prompt = str(getattr(cfg.exploration, "value_prompt",
                                        "Seems like there is a {target} ahead."))
        self.use_max_confidence = bool(getattr(cfg.exploration, "value_use_max_confidence", False))

        self.stats: dict = {}
        self.stairs = StairController(
            agent_radius=float(a.agent_radius),
            burn_on_disable=bool(getattr(a, "stair_disable_burns_map", False)),
            passive_entry=passive, stats=self.stats,
        )
        kg, floor_prior = _load_priors(cfg)
        self.planner = AscentLLMPlanner(
            llm=self._build_llm(cfg) if str(getattr(cfg.exploration, "ranker", "none")) == "ascent" else None,
            knowledge_graph=kg, floor_prior=floor_prior,
            nearby_distance=float(getattr(cfg.exploration, "nearby_distance_m", 3.0)),
            topk=int(getattr(cfg.exploration, "ranker_topk", 3) or 3),
            multi_floor=bool(getattr(cfg.exploration, "llm_multi_floor", False)),
            stats=self.stats,
        )
        self.scene_graph = SceneGraph()
        self.reset(target_category)

    @staticmethod
    def _build_llm(cfg):
        """`Qwen2_5Client.chat(txt) -> str`, on OSG's OpenAI-compatible client.
        Returns the reply as a JSON string (the planner parses it exactly as
        the reference does) or "-1" on any failure, as the reference client."""
        from osg.llm.client import ChatClient

        client = ChatClient(
            cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
            cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
        )

        def chat(prompt: str) -> str:
            try:
                out = client.chat(LLM_SYSTEM, prompt, json_response=True, temperature=0.0)
            except Exception:  # noqa: BLE001 - the reference client returns "-1"
                return "-1"
            return json.dumps(out) if isinstance(out, dict) else str(out)

        return chat

    # ------------------------------------------------------------------ reset

    def reset(self, target_category: str) -> None:
        self.target = target_category
        self.target_coco = HM3D_TO_COCO.get(target_category.lower().replace(" ", "_"),
                                             target_category.replace("_", " "))
        self.step_count = 0
        self.anchor: Optional[EpisodeAnchor] = None
        self._floors = [self._new_floor()]
        self._floor_idx = 0
        self.stairs.reset()
        self.planner.reset()
        self.scene_graph = SceneGraph()
        if self.pointnav is not None:
            self.pointnav.reset()
        self.detector.set_vocabulary(
            [self.target.replace("_", " ")] + list(getattr(self.cfg.detector, "vocabulary", []) or [])
        )
        # `Ascent_Policy._reset` (:189-220) and `Map_Controller.reset` (:133-180)
        self._pitch_angle = 0                       # hand-tracked, degrees, + = up
        self._done_initializing = not bool(self.cfg.agent.initial_scan)
        self._initialize_step = 0
        self._try_to_navigate = False
        self._try_to_navigate_step = 0
        self._double_check_goal = False
        self._blip_cosine = 0.0
        self.cur_dis_to_goal = float("inf")
        self.min_distance_xy = float("inf")
        self.cur_frontier: Optional[np.ndarray] = None
        self._last_dets: List[Detection] = []
        self._selected_frontier: Optional[np.ndarray] = None
        self._pn_goal: Optional[np.ndarray] = None
        self._nav_goal: Optional[np.ndarray] = None
        self._last_action: Optional[str] = None
        self._last_seg = None
        self._last_stair_mask = None
        self.escape.reset()

        # Runner interface (every key `osg/eval/record.py` copies).
        self.stats.clear()
        self.state_log: list = []
        self.frontier_select_log: list = []
        self.giveup_log: list = []
        self.approach_bbox_log: list = []
        self.approach_diag: Optional[dict] = None
        self.approach_stop_reason: Optional[str] = None
        self.steps_to_first_candidate: Optional[int] = None
        self.behaviour = BehaviourLog(enabled=True)
        self.step_trace = self.behaviour.rows
        self._state = "init"

    def _new_floor(self) -> dict:
        # `Map_Controller._add_floor_map` (:229-244), with the reference's
        # constructor arguments.
        om = ObstacleMap(
            min_height=float(getattr(self.cfg.agent, "ascent_min_obstacle_h", 0.61)),
            max_height=float(getattr(self.cfg.agent, "ascent_max_obstacle_h", 0.88)),
            agent_radius=float(self.cfg.agent.agent_radius),
            area_thresh=float(getattr(self.cfg.exploration, "area_thresh_m2", 1.5)),
            hole_area_thresh=100000,
            size=MAP_SIZE,
        )
        om._downstair_detector = self.downstair_detector
        return {
            "obstacle": om,
            "value": ValueMap(value_channels=1, size=MAP_SIZE, obstacle_map=None,
                              use_max_confidence=self.use_max_confidence),
            "object": ObjectPointCloudMap(erosion_size=5, size=MAP_SIZE),
        }

    @property
    def obstacle_map(self):
        return self._floors[self._floor_idx]["obstacle"]

    @property
    def value_map(self):
        return self._floors[self._floor_idx]["value"]

    @property
    def object_map(self):
        return self._floors[self._floor_idx]["object"]

    @property
    def climb(self):
        return self.stairs

    # ------------------------------------------------- runner compatibility

    @property
    def costmap(self):
        return _CostmapView(self.obstacle_map, self.anchor)

    @property
    def object_layer(self):
        return _ObjectLayerView(self.object_map, self.target)

    def debug_panel(self, frame: FrameData) -> "np.ndarray":
        from .viz import debug_panel

        robot_xy, heading = self._pose(frame)
        return debug_panel(self, frame, self.target, robot_xy, heading, self._last_dets)

    # -------------------------------------------------------------------- act

    def act(self, frame: FrameData) -> str:
        prev = self._state
        action = self._act_inner(frame)
        if self._state != prev:
            self.state_log.append((self.step_count, self._state))
        if self._state != "done":
            # `ascent_policy.py:596-610`: 30 consecutive turns force a forward,
            # 30 consecutive forwards force a right turn.
            action = self.escape(action)
        self._last_action = action
        self.behaviour.annotate(act=action, state=self._state)
        # `:625-626`
        self.step_count += 1
        self.obstacle_map._floor_num_steps += 1
        return action

    def _pose(self, frame: FrameData):
        if self.anchor is None:
            return robot_xy_heading(frame)
        return episodic_xy_heading(frame, self.anchor)

    def _act_inner(self, frame: FrameData) -> str:
        if self.step_count >= self.max_steps - 1:          # `:612-615`
            self._state = "done"
            return STOP
        if self.anchor is None:
            self.anchor = EpisodeAnchor.from_frame(frame)   # A12

        # --- `_cache_observations` (:222-260) ------------------------------
        if self.pointnav is not None:
            self.pointnav.observe(frame)                    # raw depth: `nav_depth`
        depth_n = normalise_depth(frame.depth, self.min_depth, self.max_depth)
        depth_map = filter_depth(depth_n.astype(np.float32), blur_type=None)   # A10, `:230`
        robot_xy, heading = self._pose(frame)
        tf = xyz_yaw_pitch_roll_to_tf_matrix(
            np.array([robot_xy[0], robot_xy[1], self.camera_height]),
            heading, np.radians(-self._pitch_angle), 0.0)   # `:236-238`
        om = self.obstacle_map
        robot_px = om._xy_to_px(np.atleast_2d(robot_xy))
        zeros = np.zeros(frame.depth.shape[:2], np.uint8)

        # --- object map (`map_controller.py:723-798`) ----------------------
        img_b64 = None
        stair_det_mask = None
        dets: List[Detection] = []
        if om._floor_num_steps != 0:                        # `:734-735`, A11
            raw = self.detector.detect(frame.rgb)
            dets = self._filter_target(raw)
            self._last_dets = dets
            if self.ram is not None or self.room_classifier is not None:   # `:751`
                tag_scene(frame.rgb, om._floor_num_steps, self.object_map,
                          self.ram, self.room_classifier, stats=self.stats)
            for det in dets:                                # ALL of them, A4 (`:754-768`)
                self.object_map.update_map(
                    self.target, depth_map, det.mask.astype(np.uint8), tf,
                    self.min_depth, self.max_depth, self.fx, self.fy,
                )
                if self.steps_to_first_candidate is None:
                    self.steps_to_first_candidate = self.step_count
                # The gate, A1 / F1: the cosine is the PREVIOUS step's value
                # map score (`:771-776`, written at `:562` after this block).
                if self._try_to_navigate and not self._double_check_goal:
                    if self._blip_cosine >= self.gate_threshold:
                        self._double_check_goal = True
                        self.stats["gate_latched"] = self.stats.get("gate_latched", 0) + 1
            # the detector half of the stair fusion (`:782-789`)
            stair_det_mask = self._stair_det_mask(raw, frame.rgb)
            self.object_map.update_explored(tf, self.max_depth, 2 * np.arctan(frame.depth.shape[1] / (2 * self.fx)))
        else:
            self._last_dets = []

        # --- RedNet (`:420-431`) and the obstacle map (`map_controller.py:477-539`)
        seg = self._stair_seg(frame)
        self._last_seg, self._last_stair_mask = seg, stair_det_mask
        self.stairs.pre_update(self, om, robot_xy, robot_px)
        om.update_map(
            depth_map, tf, self.min_depth, self.max_depth, self.fx, self.fy, self.hfov,
            {}, zeros,
            stair_det_mask if stair_det_mask is not None else zeros,
            seg if seg is not None else zeros,
            self._pitch_angle, self.stairs.climb_stair_over, self.stairs.reach_stair,
            self.stairs.climb_stair_flag,
        )
        om.update_agent_traj(robot_xy, heading)
        if om._has_up_stair and self._floor_idx + 1 >= len(self._floors):     # `:531-535`
            self._floors.append(self._new_floor())
        if om._has_down_stair and self._floor_idx == 0:
            self._floors.insert(0, self._new_floor())
            self._floor_idx += 1
        self.planner.floor_num = len(self._floors)
        self.stats["n_floors"] = len(self._floors)
        om.project_frontiers_to_rgb_hush(frame.rgb)         # `:538`

        # --- value map (`map_controller.py:540-562`) -----------------------
        self._update_value_map(frame, tf, depth_map, robot_xy, heading)
        # --- `_update_distance_on_object_map` (`:845-866`) ------------------
        self.cur_dis_to_goal = self._cur_dist_to_goal(robot_xy)
        goal = self._get_target_object_location(robot_xy)    # `:380-384, :441`

        self._stair_diag(seg, stair_det_mask)
        self._trace(frame, robot_xy, heading, seg, stair_det_mask, robot_px)

        # --- dispatch (`ascent_policy.py:447-581`) --------------------------
        st = self.stairs
        if not st.climb_stair_over:
            return self._stairs_dispatch(depth_n, robot_xy, heading, robot_px, seg)
        if self._pitch_angle > 0:                            # `:559-562`
            self._state = "look_down"
            return self._look(LOOK_DOWN)
        if self._pitch_angle < 0 and not om._look_for_downstair_flag:   # `:563-566`
            self._state = "look_up"
            return self._look(LOOK_UP)
        if not self._done_initializing:                      # `:567-570`
            om._done_initializing = True
            self._state = "explore"
            return self._initialize()
        if goal is None:                                     # `:571-577`
            if om._look_for_downstair_flag:
                self._state = "look_down"
                return self._look_for_downstair(robot_xy, heading)
            self._state = "explore"
            return self._explore(depth_n, robot_xy, heading)
        self._state = "approach"                              # `:578-581`
        self._try_to_navigate = True
        return self._navigate(robot_xy, heading, goal[:2])

    # ------------------------------------------------------------ perception

    def _filter_target(self, dets: List[Detection]) -> List[Detection]:
        target = self.target.lower().replace("_", " ").strip()
        return [d for d in dets if d.label.lower().replace("_", " ").strip() == target]

    def _stair_det_mask(self, raw: List[Detection], rgb) -> Optional[np.ndarray]:
        """The detector half of ASCENT's stair fusion (`map_controller.py:782-789`).

        `ascent`: GroundingDINO `stair` >= 0.60 with MobileSAM masks, ANDed
        with RedNet by the map. `rednet`: the port's union (RedNet alone
        decides), kept as an A/B -- it is what climbed 11 same-floor episodes.
        """
        m = None
        if self.stair_detector is not None:
            m = self.stair_detector.mask(rgb)
        for d in raw:
            if d.label.lower().strip() in ("stairs", "stair", "staircase", "steps"):
                m = d.mask.astype(bool) if m is None else (m | d.mask.astype(bool))
        if self.stair_up_mode != "ascent" and self._last_seg is not None:
            rednet = self._last_seg == STAIR_CLASS_ID
            m = rednet if m is None else (m | rednet)
        return None if m is None else m.astype(np.uint8)

    def _stair_seg(self, frame: FrameData):
        if self.stair_segmenter is None:
            return None
        m = self.stair_segmenter.stair_mask(frame)
        if m is None:
            return None
        return np.where(m, STAIR_CLASS_ID, 0).astype(np.uint8)

    def _update_value_map(self, frame, tf, depth_map, robot_xy, heading) -> None:
        """`map_controller.py:540-562`: one BLIP-2 call, painted into the value
        map and kept as `_blip_cosine` for the gate."""
        if self.image_text is None:
            value = 0.0
        else:
            text = self.value_prompt.replace("{target}", self.target_coco)
            value = float(np.asarray(self.image_text.score(frame.rgb, [text])).reshape(-1)[0])
            self.stats["value_calls"] = self.stats.get("value_calls", 0) + 1
        self.value_map.update_map(np.array([value]), depth_map, tf, self.min_depth, self.max_depth, self.hfov)
        self.value_map.update_agent_traj(robot_xy, heading)
        self._blip_cosine = value

    def _cur_dist_to_goal(self, robot_xy) -> float:
        """`_update_distance_on_object_map` (`map_controller.py:845-866`)."""
        om = self.object_map
        om.update_agent_traj(robot_xy, 0.0) if hasattr(om, "update_agent_traj") else None
        if not om.has_object(self.target):
            return float("inf")
        cloud = om.get_target_cloud(self.target)
        if len(cloud) == 0:
            return float("inf")
        pos = np.array([robot_xy[0], robot_xy[1], self.camera_height])
        closest = om._get_closest_point(cloud, pos)
        return float(np.linalg.norm(closest[:2] - pos[:2]))

    def _get_target_object_location(self, robot_xy) -> Optional[np.ndarray]:
        """`ascent_policy.py:380-384`."""
        if self.object_map.has_object(self.target):
            return self.object_map.get_best_object(self.target, np.asarray(robot_xy, dtype=float))
        return None

    # --------------------------------------------------------------- looking

    def _look(self, action: str) -> str:
        """Every LOOK_UP/LOOK_DOWN the reference emits adjusts `_pitch_angle`
        by the tilt step at the point of choosing it."""
        if action == LOOK_UP:
            self._pitch_angle += PITCH_OFFSET_DEG
        else:
            self._pitch_angle -= PITCH_OFFSET_DEG
        return action

    def _initialize(self) -> str:
        """`ascent_policy.py:689-697`: TURN_LEFT until `_initialize_step > 11`,
        latching on that call -- 13 turns."""
        if self._initialize_step > self.initialize_turns - 2:
            self._done_initializing = True
            self.obstacle_map._tight_search_thresh = False
        else:
            self._initialize_step += 1
        return LEFT

    # -------------------------------------------------------------- explore

    def _explore(self, depth_n, robot_xy, heading) -> str:
        """`ascent_policy.py:699-762`."""
        om = self.obstacle_map
        frontiers = [f for f in np.atleast_2d(np.asarray(om.frontiers)).reshape(-1, 2)
                     if tuple(f) not in om._disabled_frontiers]
        self._selected_frontier = None

        if len(frontiers) == 0:                              # `:707-731`
            if (not om._reinitialize_flag and om._floor_num_steps < 50
                    and ((not om._explored_up_stair and np.size(om._up_stair_frontiers) == 0)
                         or (not om._explored_down_stair and np.size(om._down_stair_frontiers) == 0))):
                return self._handle_stairwell_reinitialization()
            om._this_floor_explored = True
            self.stats["no_frontier_steps"] = self.stats.get("no_frontier_steps", 0) + 1
            action = None
            if not om._explored_up_stair:
                action = self._navigate_stair_if_unexplored_floor(robot_xy, heading, "up")
            if action is None and not om._explored_down_stair:
                action = self._navigate_stair_if_unexplored_floor(robot_xy, heading, "down")
            if action is not None:
                return action
            self._state = "done"                              # `:725-726`
            self.approach_stop_reason = "explored_out"
            self.stats["explored_out"] = self.stats.get("explored_out", 0) + 1
            return STOP

        best, value = self.planner.get_best_frontier(          # `:735-742`
            robot_xy, om, self.value_map, self.object_map,
            [f["obstacle"] for f in self._floors], [f["object"] for f in self._floors],
            np.asarray(frontiers), self.target_coco, self._floor_idx, self.step_count,
        )
        if best is None:
            return LEFT
        if value == GO_UP:                                    # `:745-752`
            action = self._navigate_stair_if_unexplored_floor(robot_xy, heading, "up")
            if action is not None:
                return action
        elif value == GO_DOWN:
            action = self._navigate_stair_if_unexplored_floor(robot_xy, heading, "down")
            if action is not None:
                return action
        self.cur_frontier = np.asarray(best, dtype=float)
        self._selected_frontier = self.cur_frontier
        self.frontier_select_log.append((
            self.step_count, [round(float(v), 2) for v in robot_xy],
            [round(float(v), 2) for v in self.cur_frontier],
            None if value in (GO_UP, GO_DOWN) else round(float(value), 4), len(frontiers)))
        action = self._pointnav(robot_xy, heading, self.cur_frontier, stop_radius=self.stop_radius)
        if action is None or action == STOP:                  # `:759-761`
            self.stats["explore_forced_forward"] = self.stats.get("explore_forced_forward", 0) + 1
            return FORWARD
        return action

    def _handle_stairwell_reinitialization(self) -> str:
        """`ascent_policy.py:764-811`."""
        om = self.obstacle_map
        self.object_map.reset()
        self.value_map.reset()
        stash = {}
        for kind in ("up", "down"):
            if getattr(om, f"_has_{kind}_stair"):
                stash[kind] = {
                    "map": getattr(om, f"_{kind}_stair_map").copy(),
                    "start": np.array(getattr(om, f"_{kind}_stair_start")).copy(),
                    "end": np.array(getattr(om, f"_{kind}_stair_end")).copy(),
                    "frontiers": np.array(getattr(om, f"_{kind}_stair_frontiers")).copy(),
                    "explored": getattr(om, f"_explored_{kind}_stair"),
                }
        om.reset()
        for kind, d in stash.items():
            setattr(om, f"_has_{kind}_stair", True)
            setattr(om, f"_{kind}_stair_map", d["map"])
            setattr(om, f"_{kind}_stair_start", d["start"])
            setattr(om, f"_{kind}_stair_end", d["end"])
            setattr(om, f"_{kind}_stair_frontiers", d["frontiers"])
            setattr(om, f"_explored_{kind}_stair", d["explored"])
        om._reinitialize_flag = True
        om._tight_search_thresh = True
        self.stairs.climb_stair_over = True
        self.stairs.reach_stair = False
        self.stairs.reach_stair_centroid = False
        self.stairs.stair_dilate_flag = False
        self._pitch_angle = 0
        self._done_initializing = False
        self._initialize_step = 0
        self.stats["stairwell_reinit"] = self.stats.get("stairwell_reinit", 0) + 1
        return self._initialize()

    def _navigate_stair_if_unexplored_floor(self, robot_xy, heading, direction: str) -> Optional[str]:
        """`ascent_policy.py:813-849`."""
        om = self.obstacle_map
        if not getattr(om, f"_has_{direction}_stair"):
            return None
        if direction == "up":
            floor_range = range(self._floor_idx + 1, len(self._floors))
        else:
            floor_range = range(self._floor_idx - 1, -1, -1)
        if not any(not self._floors[i]["obstacle"]._this_floor_explored for i in floor_range):
            return None
        self.stairs.start_navigating(om, 1 if direction == "up" else 2)
        sf = np.asarray(self.stairs.stair_frontier).reshape(-1, 2)
        if len(sf) == 0:
            return None
        self._state = "climb"
        action = self._pointnav(robot_xy, heading, sf[0], stop_radius=0.0)   # F8
        return STOP if action is None else action            # raw, `:847`

    def _look_for_downstair(self, robot_xy, heading) -> str:
        """`ascent_policy.py:851-887`."""
        om = self.obstacle_map
        if self._pitch_angle >= 0:
            return self._look(LOOK_DOWN)
        c = np.asarray(om._potential_stair_centroid).reshape(-1, 2)
        if len(c) and float(np.linalg.norm(c[0] - np.asarray(robot_xy))) > 0.2:
            action = self._pointnav(robot_xy, heading, c[0], stop_radius=self.stop_radius)
            if action is not None and action != STOP:
                return action
        return self._reject_downstair(c)

    def _reject_downstair(self, c) -> str:
        om = self.obstacle_map
        if len(c):
            om._disabled_frontiers.add(tuple(c[0]))
        om._disabled_stair_map[om._down_stair_map == 1] = 1
        om._down_stair_map.fill(0)
        om._has_down_stair = False
        om._look_for_downstair_flag = False
        self.stats["downstair_reject"] = self.stats.get("downstair_reject", 0) + 1
        return self._look(LOOK_UP)

    # ------------------------------------------------------------- approach

    def _navigate(self, robot_xy, heading, goal) -> str:
        """`ascent_policy.py:927-990`."""
        self._nav_goal = np.asarray(goal, dtype=float)
        self._try_to_navigate_step += 1                        # `:939`
        d = self.cur_dis_to_goal
        if d < 1.0:                                            # `:961`
            if d <= 0.6 or abs(d - self.min_distance_xy) < 0.1:  # `:962`, previous-step value
                if self._double_check_goal:                    # `:963-966`
                    self._state = "done"
                    self.approach_stop_reason = "nearest_point"
                    return STOP
                return self._give_up_target("unverified", robot_xy, heading)   # `:967-975`
            self.min_distance_xy = d                           # `:977`
            return FORWARD
        action = self._pointnav(robot_xy, heading, goal, stop_radius=self.stop_radius)  # `:980`
        if action is None:
            action = STOP                                      # A2: honoured (`:989-990`)
            self.stats["policy_stop_honoured"] = self.stats.get("policy_stop_honoured", 0) + 1
        if self._try_to_navigate_step >= self.abandon_steps:   # `:981-989`
            return self._give_up_target("abandon", robot_xy, heading)
        if action == STOP:
            self._state = "done"
            self.approach_stop_reason = "policy_stop"
        return action

    def _give_up_target(self, why: str, robot_xy, heading) -> str:
        """The failure path (`:967-975`, `:981-989`): clear the cloud, burn its
        cells, and act on the exploration policy THIS step. The gate is not
        touched -- once latched it holds for the episode."""
        om = self.object_map
        self.giveup_log.append((self.step_count, why, [round(float(v), 2) for v in robot_xy]))
        om.clouds = {}
        self._try_to_navigate = False
        self._try_to_navigate_step = 0
        om._disabled_object_map[om._map == 1] = 1
        om._map.fill(0)
        self.stats[f"give_up_{why}"] = self.stats.get(f"give_up_{why}", 0) + 1
        self._state = "explore"
        return self._explore(None, robot_xy, heading)

    # ---------------------------------------------------------------- stairs

    def _stairs_dispatch(self, depth_n, robot_xy, heading, robot_px, seg) -> str:
        """`ascent_policy.py:447-557`, the `not _climb_stair_over` branch."""
        om, st = self.obstacle_map, self.stairs
        self._state = "climb"
        if st.reach_stair:
            if self._pitch_angle == 0 and st.climb_stair_flag == 2:
                return self._look(LOOK_DOWN)
            if st.climb_stair_flag == 2 and self._pitch_angle >= -30 and not st.reach_stair_centroid:
                return self._look(LOOK_DOWN)
            if om._climb_stair_paused_step < 30:
                return self._climb_stair(depth_n, robot_xy, heading, robot_px)
            # `:459-514` (F5): copy the flight to the neighbour floor's map,
            # level the camera, re-run `_initialize` on THIS floor.
            if st.climb_stair_flag == 1:
                nxt = self._floor_idx + 1
                if nxt < len(self._floors) and not self._floors[nxt]["obstacle"]._done_initializing:
                    n = self._floors[nxt]["obstacle"]
                    n._down_stair_map = om._up_stair_map.copy()
                    n._down_stair_start = np.array(om._up_stair_start).copy()
                    n._down_stair_end = np.array(om._up_stair_end).copy()
                    n._down_stair_frontiers = np.array(om._up_stair_frontiers).copy()
                    n._has_down_stair = True
            elif st.climb_stair_flag == 2:
                prv = self._floor_idx - 1
                if prv >= 0 and not self._floors[prv]["obstacle"]._done_initializing:
                    n = self._floors[prv]["obstacle"]
                    n._up_stair_map = om._down_stair_map.copy()
                    n._up_stair_start = np.array(om._down_stair_start).copy()
                    n._up_stair_end = np.array(om._down_stair_end).copy()
                    n._up_stair_frontiers = np.array(om._down_stair_frontiers).copy()
                    n._has_up_stair = True
            if self._pitch_angle > 0:
                action = self._look(LOOK_DOWN)
            elif self._pitch_angle < 0:
                action = self._look(LOOK_UP)
            else:
                om._done_initializing = False
                self._done_initializing = False        # the flag `act` dispatches on
                self._initialize_step = 0
                action = self._initialize()
            st.update_stair_state(om)
            return action
        # not yet on the stairs (`:516-556`; the `:517-530` block is dead, F4)
        if om._look_for_downstair_flag:
            return self._look_for_downstair(robot_xy, heading)
        ppm = om.pixels_per_meter
        if st.climb_stair_flag == 1 and self._pitch_angle == 0 and np.sum(om._up_stair_map) > 0:
            cells = np.argwhere(om._up_stair_map)
            d = float(np.min(np.abs(cells - robot_px[0]).sum(axis=1)))
            if d <= 2.0 * ppm and stairs_in_upper_half(None if seg is None else seg == STAIR_CLASS_ID):
                return self._look(LOOK_UP)
            return self._get_close_to_stair(robot_xy, heading)
        if st.climb_stair_flag == 2 and self._pitch_angle == 0 and np.sum(om._down_stair_map) > 0:
            cells = np.argwhere(om._down_stair_map)
            d = float(np.min(np.abs(cells - robot_px[0]).sum(axis=1)))
            if d <= 2.0 * ppm:
                return self._look(LOOK_DOWN)
            return self._get_close_to_stair(robot_xy, heading)
        return self._get_close_to_stair(robot_xy, heading)

    def _get_close_to_stair(self, robot_xy, heading) -> str:
        """`ascent_policy.py:991-1067`."""
        om, st, pl = self.obstacle_map, self.stairs, self.planner
        if st.climb_stair_flag not in (1, 2):
            self._state = "explore"
            return self._explore(None, robot_xy, heading)
        target = np.asarray(om._up_stair_frontiers if st.climb_stair_flag == 1
                            else om._down_stair_frontiers).reshape(-1, 2)
        if len(target) == 0:
            self._state = "explore"
            return self._explore(None, robot_xy, heading)
        pt = target[0]
        if np.array_equal(pl._last_frontier, pt):
            cur = float(np.linalg.norm(pt - robot_xy))
            if pl.frontier_stick_step == 0:
                pl.last_frontier_distance = cur
                pl.frontier_stick_step += 1
                st.get_close_to_stair_step += 1
            elif abs(pl.last_frontier_distance - cur) > 0.3:
                pl.frontier_stick_step = 0
                pl.last_frontier_distance = cur
            else:
                pl.frontier_stick_step += 1
                st.get_close_to_stair_step += 1
                if pl.frontier_stick_step >= 30 or st.get_close_to_stair_step >= 60:
                    st.disable_stair_and_reset(self, om, pt)
                    self._state = "explore"
                    return self._explore(None, robot_xy, heading)
        else:
            pl.frontier_stick_step = 0
            pl.last_frontier_distance = 0.0
            st.get_close_to_stair_step = 0
        pl._last_frontier = pt
        action = self._pointnav(robot_xy, heading, pt, stop_radius=0.0)
        if action is None:                                     # `:1062-1065`
            st.disable_stair_and_reset(self, om, pt)
            self._state = "explore"
            return self._explore(None, robot_xy, heading)
        return action

    def _climb_stair(self, depth_n, robot_xy, heading, robot_px) -> str:
        """`ascent_policy.py:1069-1189`."""
        om, st, pl = self.obstacle_map, self.stairs, self.planner
        target = np.asarray(om._up_stair_frontiers if st.climb_stair_flag == 1
                            else om._down_stair_frontiers).reshape(-1, 2)
        if len(target) == 0:
            self._state = "explore"
            return self._explore(None, robot_xy, heading)
        cur = float(np.linalg.norm(target[0] - robot_xy))
        if abs(pl.last_frontier_distance - cur) > 0.2:
            om._climb_stair_paused_step = 0
            pl.last_frontier_distance = cur
        else:
            om._climb_stair_paused_step += 1
        if om._climb_stair_paused_step > 15:
            om._disable_end = True
        if not st.reach_stair_centroid:                        # phase 1
            action = self._pointnav(robot_xy, heading, target[0], stop_radius=0.0)
            if action is None:
                st.reach_stair_centroid = True
                return FORWARD
            return action
        if st.climb_stair_flag == 2 and self._pitch_angle < -30:   # phase 2
            return self._look(LOOK_UP)
        # phase 3: the carrot (`:1128-1189`) on the RAW depth
        fresh = carrot_waypoint(depth_n, robot_xy, heading, self.hfov, 0.8)
        if fresh is None:
            return FORWARD
        end_px = om._up_stair_end if st.climb_stair_flag == 1 else om._down_stair_end
        carrot = ratchet_carrot(st.last_carrot_xy, fresh, end_px, robot_px,
                                lambda xy: om._xy_to_px(np.atleast_2d(xy)),
                                om.pixels_per_meter, om._disable_end)
        st.carrot_goal_xy = st.last_carrot_xy = carrot
        st.last_carrot_px = om._xy_to_px(np.atleast_2d(carrot))
        om._carrot_goal_px = st.last_carrot_px
        action = self._pointnav(robot_xy, heading, carrot, stop_radius=0.0)
        if action is None:
            self.stats["climb_forced_forward"] = self.stats.get("climb_forced_forward", 0) + 1
            return FORWARD
        return action

    # ---------------------------------------------------------------- mover

    def _pointnav(self, robot_xy, heading, goal, stop_radius: float) -> Optional[str]:
        """`ascent_policy.py:888-925`. `stop=False` at every reference call
        site makes the radius branch inert, so the driver is asked with radius
        0; a network STOP comes back as None and each caller decides (F8)."""
        if self.pointnav is None:
            return FORWARD
        goal = np.asarray(goal, dtype=float)
        self._pn_goal = goal
        world = self.anchor.to_world(goal) if self.anchor is not None else goal
        nav = self.pointnav.step(np.array([world[0], -world[1]]), stop_radius=0.0)
        return nav.action

    # ------------------------------------------------------------- recording

    def _stair_diag(self, seg, stair_mask) -> None:
        st = self.stats
        if seg is not None and int(np.count_nonzero(seg == STAIR_CLASS_ID)) > MIN_STAIR_PIXELS:
            st["stair_seg_frames"] = st.get("stair_seg_frames", 0) + 1
        if stair_mask is not None and np.any(stair_mask):
            st["stair_mask_frames"] = st.get("stair_mask_frames", 0) + 1
        om = self.obstacle_map
        st["up_stair_px"] = max(st.get("up_stair_px", 0), int(om._up_stair_map.sum()))
        st["down_stair_px"] = max(st.get("down_stair_px", 0), int(om._down_stair_map.sum()))
        if om._has_up_stair:
            st["has_up_stair_steps"] = st.get("has_up_stair_steps", 0) + 1
        if om._has_down_stair:
            st["has_down_stair_steps"] = st.get("has_down_stair_steps", 0) + 1
        for k, v in (("planner_llm_calls", "llm_calls"), ("planner_rank_errors", "rank_errors")):
            pass

    def _trace(self, frame, robot_xy, heading, seg, stair_mask, robot_px) -> None:
        """Feed the behaviour recorder. `xy` stays WORLD-frame so every analysis
        script keeps working; `xy_ep` is the episodic frame the maps use."""
        om, st = self.obstacle_map, self.stairs
        world_xy, world_heading = robot_xy_heading(frame)
        f = None
        if st.climbing and st.stair_frontier is not None and np.size(st.stair_frontier):
            f = [round(float(v), 2) for v in np.asarray(st.stair_frontier).reshape(-1, 2)[0]]
        self.behaviour.step(
            n=self.step_count, xy=world_xy, yaw=world_heading,
            height=float(frame.camera_position[1]), pitch=float(self._pitch_angle),
            state=self._state, action=None, floor=self._floor_idx,
            xy_ep=[round(float(v), 3) for v in robot_xy],
            ndet=len(self._last_dets),
            det=max((d.score for d in self._last_dets), default=0.0),
            det_px=max((float((d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]))
                        for d in self._last_dets), default=0.0),
            seg_px=int(np.count_nonzero(seg == STAIR_CLASS_ID)) if seg is not None else 0,
            stair_det_px=int(np.count_nonzero(stair_mask)) if stair_mask is not None else 0,
            explored_m2=float(om.explored_area.sum()) / (om.pixels_per_meter ** 2),
            up_px=int(om._up_stair_map.sum()), dn_px=int(om._down_stair_map.sum()),
            up_f=[round(float(v), 2) for v in np.asarray(om._up_stair_frontiers).reshape(-1, 2)[0]]
                 if np.size(om._up_stair_frontiers) else None,
            dn_f=[round(float(v), 2) for v in np.asarray(om._down_stair_frontiers).reshape(-1, 2)[0]]
                 if np.size(om._down_stair_frontiers) else None,
            sel=None if self._selected_frontier is None
                else [round(float(v), 2) for v in self._selected_frontier],
            pn_goal=None if self._pn_goal is None else [round(float(v), 2) for v in self._pn_goal],
            rho=getattr(self.pointnav, "last_rho", None) if self.pointnav else None,
            theta=getattr(self.pointnav, "last_theta", None) if self.pointnav else None,
            pn_resets=int(getattr(self.pointnav, "n_resets", 0)) if self.pointnav else 0,
            obs=int(self.object_map.has_object(self.target)),
            verified=int(self._double_check_goal), itm=round(float(self._blip_cosine), 4),
            dgoal=None if not np.isfinite(self.cur_dis_to_goal) else round(float(self.cur_dis_to_goal), 3),
            try_nav=int(self._try_to_navigate), nav_step=int(self._try_to_navigate_step),
            floor_n=int(om._floor_num_steps), nfront=int(np.atleast_2d(np.asarray(om.frontiers)).reshape(-1, 2).shape[0]),
            climb=(st.climb_stair_flag if st.climbing else 0),
            reach=int(st.reach_stair), cent=int(st.reach_stair_centroid),
            paused=int(om._climb_stair_paused_step), stair_f=f,
            on_stairs=int(robot_on_stairs(
                om._up_stair_map if st.climb_stair_flag != 2 else om._down_stair_map,
                robot_px, self.cfg.agent.agent_radius * om.pixels_per_meter)),
        )


class _CostmapView:
    """Enough of `Costmap2D` for the runner's visualiser, in WORLD coordinates."""

    def __init__(self, om, anchor: Optional[EpisodeAnchor]) -> None:
        self._om = om
        self.resolution = 1.0 / om.pixels_per_meter
        self.grid = np.where(om._map.astype(bool), 100, np.where(om.explored_area, 0, -1)).astype(np.int8)
        ep_origin = np.array([-om._episode_pixel_origin[1] * self.resolution,
                              -om._episode_pixel_origin[0] * self.resolution])
        self.origin = ep_origin if anchor is None else anchor.to_world(ep_origin)

    def world_to_grid(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, rc):
        return self.origin + (np.asarray(rc, dtype=float) + 0.5) * self.resolution

    def coverage_cells(self) -> int:
        return int((self.grid != -1).sum())


class _ObjectLayerView:
    """Enough of `ObjectLayer` for the runner's per-episode fields."""

    def __init__(self, om, target) -> None:
        self._om, self._target = om, target

    def tracks(self, include_blacklisted: bool = False):
        return []

    def get(self, _tid):
        return None
