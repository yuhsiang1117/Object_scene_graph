"""habitat-lab ObjectNav wrapper. Converts habitat observations + ground-truth
sensor pose into FrameData (OpenCV camera convention: the OpenGL camera is
rotated 180 deg about x so z points forward, y down).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.geometry import quat_to_matrix
from ..core.types import CameraIntrinsics, FrameData

# OpenGL -> OpenCV camera rotation (180 deg about x)
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def make_objectnav_config(cfg):
    """Build a habitat-lab ObjectNav config from our EvalConfig/AgentConfig."""
    import habitat
    from habitat.config.read_write import read_write
    from hydra.core.global_hydra import GlobalHydra

    # Our own @hydra.main leaves GlobalHydra initialized; habitat.get_config
    # needs to initialize its own search path. Our cfg is already resolved to
    # a plain DictConfig at this point, so clearing is safe.
    GlobalHydra.instance().clear()
    hab_cfg = habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(hab_cfg):
        task = hab_cfg.habitat.task
        sim = hab_cfg.habitat.simulator
        ds = hab_cfg.habitat.dataset

        ds.split = cfg.eval.split
        ds.data_path = cfg.eval.episodes_path.replace("{split}", cfg.eval.split)
        ds.scenes_dir = cfg.eval.scenes_dir
        # Restrict to specific scenes (e.g. single-floor only: the 2D costmap /
        # room-seg scene graph cannot handle stairs). Default ["*"] = all.
        content_scenes = cfg.eval.content_scenes
        if content_scenes:
            ds.content_scenes = list(content_scenes)

        agent = sim.agents.main_agent
        agent.sim_sensors.rgb_sensor.width = cfg.eval.rgb_width
        agent.sim_sensors.rgb_sensor.height = cfg.eval.rgb_height
        agent.sim_sensors.rgb_sensor.hfov = int(cfg.eval.hfov_deg)  # habitat wants int
        agent.sim_sensors.depth_sensor.width = cfg.eval.rgb_width
        agent.sim_sensors.depth_sensor.height = cfg.eval.rgb_height
        agent.sim_sensors.depth_sensor.hfov = int(cfg.eval.hfov_deg)
        agent.sim_sensors.depth_sensor.normalize_depth = False
        # Agent embodiment: match the HM3D ObjectNav benchmark (and the old ROS
        # system) -- a 0.88 m agent with the camera at the top. The previous
        # hardcoded 1.5 m body made the navmesh reject low-clearance areas the
        # 0.88 m benchmark agent can traverse, diverging from both the standard
        # and the workspace we compare against.
        agent.height = cfg.agent.camera_height
        agent.radius = cfg.agent.agent_radius
        cam_pos = [0.0, float(cfg.agent.camera_height), 0.0]
        agent.sim_sensors.rgb_sensor.position = cam_pos
        agent.sim_sensors.depth_sensor.position = cam_pos

        sim.forward_step_size = cfg.agent.forward_m
        sim.turn_angle = int(cfg.agent.turn_deg)

        task.measurements.success.success_distance = cfg.agent.success_distance
        hab_cfg.habitat.environment.max_episode_steps = cfg.agent.max_steps
        # Spread a fixed-size eval across scenes rather than draining one scene
        # first. Habitat groups episodes by scene and only switches after
        # max_scene_repeat_steps (default 10000), so a ~30-episode run stays
        # inside a single scene -- unrepresentative of the val split. -1 keeps
        # the habitat default.
        msre = cfg.eval.max_scene_repeat_episodes
        if msre and msre > 0:
            hab_cfg.habitat.environment.iterator_options.max_scene_repeat_episodes = msre
        hab_cfg.habitat.seed = cfg.seed
    return hab_cfg


class HabitatObjectNavEnv:
    ACTIONS = {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3, "look_up": 4, "look_down": 5}

    def __init__(self, cfg) -> None:
        import habitat

        self._hab_cfg = make_objectnav_config(cfg)
        self.env = habitat.Env(config=self._hab_cfg)
        self.intrinsics = CameraIntrinsics.from_hfov(
            cfg.eval.hfov_deg, cfg.eval.rgb_width, cfg.eval.rgb_height
        )
        self._frame_id = 0
        # Habitat-navmesh path follower, mirroring the OLD ObjectSceneGraph
        # stack (publish a goal point -> Habitat plans+drives on its own
        # navmesh) instead of the from-scratch costmap planner+controller.
        # Only used when agent.use_habitat_navmesh is set; harmless otherwise.
        self._follower = None
        self._action_name = {v: k for k, v in self.ACTIONS.items()}
        self._navmesh_goal_radius = float(cfg.agent.navmesh_goal_radius)

    def _ensure_follower(self):
        if self._follower is None:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
            # stop_on_error=False so a GreedyFollowerError reaches us instead of
            # being silently converted to a `stop` action inside habitat. The
            # value returned to the caller is None either way -- what changes is
            # that `nav_reasons` can tell a real arrival from a follower that
            # gave up. With habitat's default of True, every greedy failure was
            # indistinguishable from an arrival, which is how the 00848 kitchen
            # loop stayed invisible for four conditions.
            self._follower = ShortestPathFollower(
                self.env.sim,
                goal_radius=self._navmesh_goal_radius,
                return_one_hot=False,
                stop_on_error=False,
            )
        return self._follower

    def _goal3d(self, goal, floor_y=None) -> np.ndarray:
        """Lift a goal to a 3D navmesh query point.

        A 3-vector passes through. A 2-vector gets a height: `floor_y` when the
        caller knows which storey the goal is on, otherwise the agent's own
        height -- the historical behaviour, which silently forces every goal
        onto the agent's current floor. That substitution is why cross-floor
        targets read as unreachable (see docs/MULTI_FLOOR.md).
        """
        g = np.asarray(goal, dtype=float).ravel()
        if g.size == 3:
            return g.astype(np.float32)
        y = float(self.env.sim.get_agent_state().position[1]) if floor_y is None else float(floor_y)
        return np.array([float(g[0]), y, float(g[1])], dtype=np.float32)

    @property
    def nav_reasons(self):
        """Why `action_to_goal` returned None, counted per episode.

        Lazily created rather than set in __init__, because YCBAuthoredNavEnv
        defines its own __init__ and does not run this one -- which is exactly
        how the first version of this counter crashed a diagnostic run.
        """
        counter = getattr(self, "_nav_reasons", None)
        if counter is None:
            import collections

            counter = collections.Counter()
            self._nav_reasons = counter
        return counter

    def _path_exists(self, snapped) -> bool:
        """Whether the pathfinder finds a route from the agent to an already
        snapped goal. Separate from `is_reachable`, which snaps its own goal --
        here the snap has already happened and re-snapping would query a
        different point than the follower was given."""
        import habitat_sim

        pf = self.env.sim.pathfinder
        path = habitat_sim.ShortestPath()
        path.requested_start = pf.snap_point(self.env.sim.get_agent_state().position)
        path.requested_end = np.asarray(snapped, dtype=np.float32)
        return bool(pf.find_path(path))

    def action_to_goal(self, goal_xy, floor_y=None) -> Optional[str]:
        """Next discrete action to drive toward a goal on Habitat's navmesh, or
        None if arrived (within goal_radius) or the goal is not navigable.

        `goal_xy` is a ground-plane (x, z) pair or a full 3D point; a 2D goal is
        snapped at `floor_y`, defaulting to the agent's current height.

        None means four different things and every caller has had to treat them
        alike: the snap failed, the follower raised, the follower said stop
        because the agent is there, or the follower said stop because it cannot
        get there. `nav_reasons` counts which, per episode, because guessing
        between them has now been wrong twice -- once as "the goal snaps through
        a wall" and once as "the object is on a disconnected navmesh island".
        Neither survived measurement.
        """
        follower = self._ensure_follower()
        goal3d = self._goal3d(goal_xy, floor_y)
        snapped = self.env.sim.pathfinder.snap_point(goal3d)
        if snapped is None or bool(np.isnan(np.asarray(snapped)).any()):
            self.nav_reasons["nav_snap_failed"] += 1
            return None  # unreachable -> caller treats as "arrived" and re-decides
        try:
            a = follower.get_next_action(np.asarray(snapped, dtype=np.float32))
        except Exception:
            # The greedy follower could not produce an action sequence. That is
            # not the same as "there is no route": ask the pathfinder directly,
            # because a goal that IS geodesically reachable but that the greedy
            # follower refuses is a recoverable failure, not an arrival.
            self.nav_reasons["nav_follower_raised"] += 1
            here = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            gap = float(np.linalg.norm(here[[0, 2]] - np.asarray(snapped, dtype=float)[[0, 2]]))
            self.nav_reasons["nav_raised_m_x10"] += int(round(gap * 10))
            if self._path_exists(snapped):
                self.nav_reasons["nav_greedy_failed_but_reachable"] += 1
            else:
                self.nav_reasons["nav_greedy_failed_unreachable"] += 1
            return None
        if a is None or int(a) == self.ACTIONS["stop"]:
            # Arrived, or refused. The distance separates them: the follower
            # stops within goal_radius (0.1 m), so a "stop" issued from metres
            # away is a refusal wearing an arrival's clothes.
            here = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            gap = float(np.linalg.norm(here[[0, 2]] - np.asarray(snapped, dtype=float)[[0, 2]]))
            if gap <= max(2.0 * self._navmesh_goal_radius, 0.25):
                self.nav_reasons["nav_arrived"] += 1
            else:
                self.nav_reasons["nav_refused"] += 1
                self.nav_reasons["nav_refused_m_x10"] += int(round(gap * 10))
            return None
        return self._action_name.get(int(a))

    def is_reachable(self, goal_xy, floor_y=None) -> bool:
        """Whether a goal is on the same navmesh component as the agent (a
        geodesic path exists). Targets in sealed/disconnected rooms (closed door
        or a step in the mesh) are visible but unreachable -- the agent should
        not commit to them.

        Pass a 3D goal (or `floor_y`) for anything that may be on another
        storey: with a bare 2D goal the agent's own height is substituted, so a
        target one floor up snaps to whatever is under the agent instead and is
        reported unreachable even when the navmesh connects the two."""
        import habitat_sim

        pf = self.env.sim.pathfinder
        pos = self.env.sim.get_agent_state().position
        g = pf.snap_point(self._goal3d(goal_xy, floor_y))
        if g is None or bool(np.isnan(np.asarray(g)).any()):
            return False
        path = habitat_sim.ShortestPath()
        path.requested_start = pf.snap_point(pos)
        path.requested_end = g
        return bool(pf.find_path(path))

    # ---------------------------------------------------------------- episode

    def reset(self) -> FrameData:
        obs = self.env.reset()
        self.nav_reasons.clear()
        self._frame_id = 0
        return self._to_frame(obs)

    def step(self, action: str) -> FrameData:
        obs = self.env.step(self.ACTIONS[action])
        self._frame_id += 1
        return self._to_frame(obs)

    @property
    def episode_over(self) -> bool:
        return self.env.episode_over

    @property
    def current_episode(self):
        return self.env.current_episode

    def target_category(self) -> str:
        return str(self.current_episode.object_category)

    def metrics(self) -> dict:
        return self.env.get_metrics()

    def close(self) -> None:
        self.env.close()

    # -------------------------------------------------------------- internals

    def _to_frame(self, obs) -> FrameData:
        state = self.env.sim.get_agent_state()
        sensor = state.sensor_states.get("rgb") or state.sensor_states.get("depth")
        q = sensor.rotation  # quaternion (numpy-quaternion type: w, x, y, z)
        R_gl = quat_to_matrix(q.w, q.x, q.y, q.z)
        T_wc = np.eye(4)
        T_wc[:3, :3] = R_gl @ _GL_TO_CV
        T_wc[:3, 3] = np.asarray(sensor.position, dtype=float)

        depth = obs["depth"]
        if depth.ndim == 3:
            depth = depth[..., 0]
        return FrameData(
            frame_id=self._frame_id,
            rgb=np.ascontiguousarray(obs["rgb"][..., :3]),
            depth=depth.astype(np.float32),
            T_wc=T_wc,
            intrinsics=self.intrinsics,
        )
