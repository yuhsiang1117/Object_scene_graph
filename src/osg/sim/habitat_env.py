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
        content_scenes = getattr(cfg.eval, "content_scenes", None)
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
        agent.height = 1.5
        agent.radius = cfg.agent.agent_radius

        sim.forward_step_size = cfg.agent.forward_m
        sim.turn_angle = int(cfg.agent.turn_deg)

        task.measurements.success.success_distance = cfg.agent.success_distance
        hab_cfg.habitat.environment.max_episode_steps = cfg.agent.max_steps
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

    # ---------------------------------------------------------------- episode

    def reset(self) -> FrameData:
        obs = self.env.reset()
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
