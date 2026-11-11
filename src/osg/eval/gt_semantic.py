"""Instance ground truth for recorded keyframes, from a side simulator.

The object-layer recording (eval/object_recorder.py) holds depth, pose and
masks, but nothing about WHICH object a mask is of. A second Habitat instance
of the same scene and layout, with a depth and a semantic sensor at the
camera's exact pose, answers that offline: the majority instance id under each
recorded mask is the detection's object, and the scene's own annotation gives
every object's true extent.

Two traps, both documented where they were first hit:

  * the collector's YCB semantic ids (26..95) collide with HM3D's instance ids
    (0..N), so `semantic == id` also selects a wall -- the injected ids are
    offset by SEMANTIC_ID_OFFSET (scripts/probe_ycb_detection.py);
  * a semantic sensor ALONE renders the scene as one blob, so a colour sensor
    is attached as well.

`render_at` re-renders depth too, and `check_depth` compares it with the
recorded depth: a pose conversion that is wrong anywhere would label every
detection with the wrong object, so a run of the labeller refuses to proceed
on a frame that does not reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEMANTIC_ID_OFFSET = 50000
_CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # inverse of habitat_env._GL_TO_CV (its own inverse)


@dataclass
class GTInstance:
    id: int
    category: str
    center: np.ndarray   # (3,) world, AABB centre
    half: np.ndarray     # (3,) AABB half sizes, world axes
    source: str          # "hm3d" | "ycb"


def make_sim(layout, cfg, width: int, height: int, hfov_deg: float):
    """A simulator of `layout` with rgb + depth + semantic at the agent origin,
    so an agent state IS a camera pose."""
    import habitat_sim
    import magnum as mn
    from ..sim.ycb_env import inject_layout_objects

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(layout.scene_mesh)
    sim_cfg.scene_dataset_config_file = str(layout.scene_dataset_config)
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = True
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                       ("depth", habitat_sim.SensorType.DEPTH),
                       ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [int(height), int(width)]
        spec.position = [0.0, 0.0, 0.0]
        spec.hfov = mn.Deg(float(hfov_deg))
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.initialize_agent(0)
    for rigid in inject_layout_objects(sim, layout):
        rigid.semantic_id = SEMANTIC_ID_OFFSET + int(rigid.semantic_id)
    return sim


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (x, y, z, w)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def _observe(sim, T_wc: np.ndarray) -> dict:
    """Every sensor's observation from the OpenCV camera pose T_wc."""
    from habitat_sim.utils.common import quat_from_coeffs

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(T_wc[:3, 3], dtype=np.float32)
    state.rotation = quat_from_coeffs(_matrix_to_quat(T_wc[:3, :3] @ _CV_TO_GL).astype(np.float32))
    state.sensor_states = {}
    agent.set_state(state, reset_sensors=True)
    return sim.get_sensor_observations()


def render_rgb_at(sim, T_wc: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 colour from the OpenCV camera pose T_wc -- the frame the
    recording dropped, for figures (scripts/render_ellipsoid_fusion.py)."""
    return np.asarray(_observe(sim, T_wc)["rgb"])[..., :3].astype(np.uint8)


def render_at(sim, T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(depth, semantic) rendered from the OpenCV camera pose T_wc."""
    obs = _observe(sim, T_wc)
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth, np.asarray(obs["semantic"]).astype(np.int64)


def check_depth(rendered: np.ndarray, recorded: np.ndarray, tol_m: float = 1e-3,
                min_frac: float = 0.99, depth_range: Tuple[float, float] = (0.0, np.inf),
                min_px: int = 100) -> Tuple[bool, float]:
    """Do the two depth images agree where both are valid? Returns (ok,
    fraction within tol). The recorded image went through the run's own depth
    clipping (habitat-lab clamps to [depth_min_m, depth_max_m]), so only pixels
    strictly inside that range in the recording are compared."""
    lo, hi = float(depth_range[0]), float(depth_range[1])
    both = (rendered > 1e-3) & (recorded > max(1e-3, lo + 1e-2)) & (recorded < hi - 1e-2)
    if int(both.sum()) < int(min_px):
        # Nothing to compare: the view is clamped (a wall inside depth_min_m)
        # or invalid. Not evidence either way; the caller counts these and the
        # labeller reports how many there were. A handful of surviving pixels
        # is no evidence either: pass 2 has camera-against-a-wall keyframes
        # with ONE in-range pixel, an edge pixel whose depth is a coin toss.
        return True, float("nan")
    frac = float(np.mean(np.abs(rendered[both] - recorded[both]) < tol_m))
    return frac >= min_frac, frac


def instances(sim) -> Dict[int, GTInstance]:
    """Every annotated instance: HM3D's semantic objects and the injected
    YCB objects, by the id the semantic sensor reports."""
    out: Dict[int, GTInstance] = {}
    scene = sim.semantic_scene
    for obj in getattr(scene, "objects", []) or []:
        if obj is None:
            continue
        try:
            sid = int(obj.semantic_id)
            aabb = obj.aabb
            center = np.asarray(aabb.center, dtype=float)
            half = 0.5 * np.asarray(aabb.sizes, dtype=float)
            cat = obj.category.name() if obj.category is not None else ""
        except Exception:
            continue
        if not np.all(np.isfinite(center)) or not np.all(half > 0):
            continue
        out[sid] = GTInstance(sid, str(cat), center, half, "hm3d")
    rom = sim.get_rigid_object_manager()
    for handle in rom.get_object_handles():
        rigid = rom.get_object_by_handle(handle)
        bb = rigid.root_scene_node.cumulative_bb
        lo, hi = np.asarray(bb.min, dtype=float), np.asarray(bb.max, dtype=float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        T = np.asarray(rigid.transformation)
        world = corners @ T[:3, :3].T + T[:3, 3]
        wlo, whi = world.min(axis=0), world.max(axis=0)
        label = handle.split("::")[1] if "::" in handle else handle
        sid = int(rigid.semantic_id)
        out[sid] = GTInstance(sid, label, 0.5 * (wlo + whi), 0.5 * (whi - wlo), "ycb")
    return out


def label_masks(semantic: np.ndarray, masks: Sequence[np.ndarray],
                min_purity: float = 0.5) -> List[Tuple[Optional[int], float]]:
    """(instance id or None, purity) per mask: the majority id under the mask
    and the share of the mask it covers. Below `min_purity` the detection is
    of no single object (a blend, or mostly background) and gets None."""
    out = []
    for mask in masks:
        ids = semantic[np.asarray(mask, dtype=bool)]
        if ids.size == 0:
            out.append((None, 0.0))
            continue
        vals, counts = np.unique(ids, return_counts=True)
        k = int(np.argmax(counts))
        purity = float(counts[k]) / float(ids.size)
        out.append((int(vals[k]) if purity >= min_purity else None, purity))
    return out


def find_layouts(cfg, scenes) -> dict:
    """{(scene, layout_id): authored layout} over every layout type a scene
    has, resolved under the composed config's `ycb.*` roots."""
    from pathlib import Path

    from ..sim.ycb_layouts import discover_authored_layouts

    found = {}
    for types in (["static"], ["cross_anchor"], ["in_anchor"]):
        try:
            disc = discover_authored_layouts(
                layout_root=Path(str(cfg.ycb.layout_root)),
                data_root=Path(str(cfg.ycb.data_root)),
                scenes=sorted(scenes),
                layout_types=types,
                layout_indices=[int(v) for v in cfg.ycb.layout_indices],
                target_labels={str(k): str(v) for k, v in cfg.ycb.target_labels.items()},
                allow_incomplete=True,
                hm3d_root=Path(str(cfg.ycb.hm3d_root)),
            )
        except Exception as exc:  # a layout type a scene does not have
            print(f"  ({types[0]}: {exc})")
            continue
        for layout in disc.layouts:
            found[(layout.scene_name, layout.layout_id)] = layout
    return found
