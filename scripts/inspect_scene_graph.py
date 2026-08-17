"""Replay one episode and emit a 3D inspectable scene graph.

The existing debug output is a fixed two-panel video (`viz/debug/*.mp4`): live
RGB beside a top-down costmap. It cannot be rotated, queried, or asked the one
question that matters most right now -- *why did the agent prefer THAT object
over the real goal?* -- because the answer is a spatial relationship between an
ellipsoid, the goal view points and the storey they sit on, and a 2D projection
throws away the axis that separates them.

This writes a Rerun `.rrd` recording instead: object ellipsoids with their true
axes and orientation, one costmap plane per storey at its own height, the 3D
trajectory, portals, and the goal view points, all on a timeline you can scrub.
Nothing is displayed here (the container is headless) -- download the file and
open it in the Rerun viewer:

    pip install "rerun-sdk" "numpy<2"          # the pin matters, see below
    python scripts/inspect_scene_graph.py +experiment=scene_cvZr5TUy5C5
    rerun outputs/inspect/<tag>.rrd            # on your own machine

**The numpy pin is not optional.** Unpinned, pip resolves rerun-sdk to a build
that requires numpy>=2, and habitat-sim in this environment is on 1.26.4 and
does not support NumPy 2.0 -- installing it breaks the simulator, not the
viewer.

`--gltf` writes a single self-contained .glb instead (ellipsoids as scaled UV
spheres, storeys as textured quads), for viewing in Blender / the VS Code glTF
extension / any online viewer without adding a dependency beyond `trimesh`. It
has no timeline: it is the final state only.

Note on where the geometry comes from: `graph/serialize.to_json` exports only an
object's `center`, because `ObjectNodeView` carries no shape. The ellipsoid
axes and rotation live on `ObjectTrack.ellipsoid` in the object layer, so this
reads `agent.object_layer` directly. Logging the two side by side is deliberate
-- where the scene-graph node and the underlying track disagree is exactly where
association or Wasserstein refinement has gone wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osg.mapping.costmap import FREE, HEIGHT_AXIS, OCCUPIED, PLANE, UNKNOWN  # noqa: E402

# Stable per-label colours: the same category keeps its colour across episodes,
# so you learn the palette once. Hash-based rather than a table because the
# YOLOE vocabulary is open and a table would silently grey out new labels.
def _label_color(label: str) -> List[int]:
    h = abs(hash(label)) % (2 ** 24)
    r, g, b = (h >> 16) & 255, (h >> 8) & 255, h & 255
    # Push away from black/white so nothing vanishes against the background.
    return [80 + r % 176, 80 + g % 176, 80 + b % 176]


def _costmap_image(cm) -> np.ndarray:
    """Costmap as an RGB image: unknown grey, free white, occupied near-black."""
    img = np.zeros((*cm.grid.shape, 3), dtype=np.uint8)
    img[cm.grid == UNKNOWN] = (70, 70, 78)
    img[cm.grid == FREE] = (235, 235, 235)
    img[cm.grid == OCCUPIED] = (25, 25, 30)
    return img


def _plane_mesh(cm, y: float):
    """Two triangles spanning the costmap extent at world height `y`.

    Grid row/col map to world (x, z) via `origin + rc * resolution`, matching
    `Costmap2D.grid_to_world`, so the plane lands under the ellipsoids rather
    than beside them.
    """
    h, w = cm.grid.shape
    x0, z0 = cm.origin
    x1, z1 = x0 + h * cm.resolution, z0 + w * cm.resolution
    verts = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return verts, tris, uvs


def _goal_points(episode) -> np.ndarray:
    """Every goal view point in the episode, as (N, 3) world positions.

    These are what `distance_to_goal` is measured against, so seeing them is
    the difference between "the agent stopped 10 m away" and understanding
    which room it should have been in.
    """
    pts = []
    for goal in getattr(episode, "goals", []) or []:
        for vp in getattr(goal, "view_points", []) or []:
            agent_state = getattr(vp, "agent_state", None)
            p = getattr(agent_state, "position", None) if agent_state is not None else None
            if p is None:
                p = getattr(vp, "position", None)
            if p is not None:
                pts.append(np.asarray(p, dtype=float))
    return np.array(pts) if pts else np.zeros((0, 3))


# --------------------------------------------------------------------- rerun

def _log_static(rr, agent, episode, target: str) -> None:
    """Episode-invariant geometry: goal view points and the start pose."""
    goals = _goal_points(episode)
    if len(goals):
        rr.log(
            "world/goal_view_points",
            rr.Points3D(goals, colors=[0, 220, 90], radii=0.06,
                        labels=[f"goal:{target}"] * len(goals)),
            static=True,
        )
    start = np.asarray(episode.start_position, dtype=float)
    rr.log("world/start", rr.Points3D(start[None, :], colors=[255, 255, 0], radii=0.12),
           static=True)


def _log_step(rr, agent, frame, step: int, cam_h: float, log_floors: bool) -> None:
    rr.set_time("step", sequence=step)

    pos = np.asarray(frame.camera_position, dtype=float)
    rr.log("world/agent", rr.Points3D(pos[None, :], colors=[255, 140, 0], radii=0.10))

    # Camera frustum. T_wc maps camera -> world; Rerun wants that same
    # convention for a Transform3D, so it passes through unchanged.
    T = np.asarray(frame.T_wc, dtype=float)
    rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    K = frame.intrinsics.K()
    rr.log(
        "world/camera/image",
        rr.Pinhole(image_from_camera=K, resolution=[frame.rgb.shape[1], frame.rgb.shape[0]]),
    )
    rr.log("world/camera/image/rgb", rr.Image(frame.rgb))

    # Object ellipsoids, straight off the tracks: centre, semi-axes, rotation.
    tracks = agent.object_layer.tracks()
    if tracks:
        centers, halves, mats, colors, labels = [], [], [], [], []
        for t in tracks:
            e = t.ellipsoid
            centers.append(agent.object_layer.center_of(t))
            halves.append(np.asarray(e.axes, dtype=float))
            mats.append(np.asarray(e.R, dtype=float))
            colors.append(_label_color(t.label))
            labels.append(f"{t.label}#{t.id} n={t.n_obs} ev={t.evidence:.1f}")
        # Ellipsoids3D takes quaternions, not matrices (0.23 API). Pass a raw
        # (N, 4) xyzw array: wrapping each row in rr.Quaternion routes through a
        # batch converter that calls np.asarray(copy=...), which is numpy>=2
        # only -- under this env's pinned numpy 1.26 that raises internally,
        # rerun downgrades it to a warning, and the ROTATIONS ARE SILENTLY
        # DROPPED, rendering every ellipsoid axis-aligned. The plain array
        # avoids that path. Same trap applies to rr.RotationAxisAngle.
        from scipy.spatial.transform import Rotation

        quats = Rotation.from_matrix(np.array(mats)).as_quat().astype(np.float32)
        rr.log(
            "world/objects",
            rr.Ellipsoids3D(
                centers=np.array(centers), half_sizes=np.array(halves),
                quaternions=quats, colors=colors, labels=labels,
                fill_mode=rr.components.FillMode.MajorWireframe,
            ),
        )

    if not log_floors:
        return
    # One textured plane per storey, at that storey's own height. Upstairs
    # geometry bleeding into a downstairs grid -- the original multi-floor bug
    # -- is immediately visible here and invisible in any single top-down view.
    stack = getattr(agent, "_floor_stack", None)
    if stack is None:
        return
    for fid in stack.visited_ids():
        cm = stack.layer(fid).costmap
        y = agent.floors.height_of(fid)
        verts, tris, uvs = _plane_mesh(cm, y)
        rr.log(
            f"world/floor_{fid}/map",
            rr.Mesh3D(
                vertex_positions=verts, triangle_indices=tris,
                vertex_texcoords=uvs, albedo_texture=_costmap_image(cm),
            ),
        )


def run_rerun(cfg, out_path: Path, episode_index: int, max_steps: Optional[int],
              floor_every: int) -> dict:
    import rerun as rr

    from osg.eval.runner import build_ctf_planner, build_detector, build_scorer, build_verifier
    from osg.sim.habitat_env import HabitatObjectNavEnv
    from osg.agent.nav_agent import NavAgent
    from osg.core.profiler import Profiler

    env = HabitatObjectNavEnv(cfg)
    detector, scorer = build_detector(cfg), build_scorer(cfg)
    verifier, ctf = build_verifier(cfg), build_ctf_planner(cfg)

    for _ in range(episode_index + 1):
        frame = env.reset()
    episode, target = env.current_episode, env.target_category()
    tag = f"ep{episode.episode_id}_{Path(str(episode.scene_id)).stem.split('.')[0]}_{target}"
    out_file = out_path / f"{tag}.rrd"
    out_path.mkdir(parents=True, exist_ok=True)

    rr.init(f"osg/{tag}", spawn=False)
    rr.save(str(out_file))
    # Habitat is y-up; without this Rerun assumes z-up and every storey stacks
    # along the wrong axis.
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    agent = NavAgent(
        cfg, detector, scorer, verifier, target,
        profiler=Profiler(),
        nav_fn=env.action_to_goal if cfg.agent.use_habitat_navmesh else None,
        reachable_fn=env.is_reachable if cfg.agent.use_habitat_navmesh else None,
        ctf_planner=ctf,
    )
    _log_static(rr, agent, episode, target)

    cam_h = float(cfg.agent.camera_height)
    traj: List[np.ndarray] = []
    steps = 0
    budget = max_steps if max_steps is not None else cfg.agent.max_steps
    while not env.episode_over and steps < budget:
        action = agent.act(frame)
        # Costmap meshes are the expensive part of the recording; the map
        # changes slowly, so log them every `floor_every` steps.
        _log_step(rr, agent, frame, steps, cam_h, log_floors=(steps % floor_every == 0))
        traj.append(np.asarray(frame.camera_position, dtype=float))
        if len(traj) > 1:
            rr.log("world/trajectory", rr.LineStrips3D([np.array(traj)], colors=[255, 140, 0]))
        frame = env.step(action)
        steps += 1

    # Portals: where the agent believed another storey was reachable from.
    if agent.portal_log:
        pts = np.array([[p[1][0], agent.floors.height_of(agent.floors.current) + p[2], p[1][1]]
                        for p in agent.portal_log])
        rr.log("world/portals",
               rr.Points3D(pts, colors=[0, 160, 255], radii=0.18,
                           labels=[f"portal@step{p[0]} dy={p[2]}" for p in agent.portal_log]),
               static=True)

    m = env.metrics()
    summary = {
        "episode_id": str(episode.episode_id), "scene": Path(str(episode.scene_id)).name,
        "target": target, "steps": steps,
        "success": float(m.get("success", 0.0)), "spl": float(m.get("spl", 0.0)),
        "distance_to_goal": float(m.get("distance_to_goal", -1.0)),
        "n_tracks": len(agent.object_layer.tracks()),
        "floors_visited": agent._floor_stack.visited_ids(),
        "output": str(out_file),
    }
    env.close()
    return summary


# ---------------------------------------------------------------------- gltf

def run_gltf(cfg, out_path: Path, episode_index: int, max_steps: Optional[int]) -> dict:
    """Final-state-only .glb. No timeline, no dependency beyond trimesh."""
    import trimesh

    from osg.eval.runner import build_ctf_planner, build_detector, build_scorer, build_verifier
    from osg.sim.habitat_env import HabitatObjectNavEnv
    from osg.agent.nav_agent import NavAgent
    from osg.core.profiler import Profiler

    env = HabitatObjectNavEnv(cfg)
    detector, scorer = build_detector(cfg), build_scorer(cfg)
    verifier, ctf = build_verifier(cfg), build_ctf_planner(cfg)
    for _ in range(episode_index + 1):
        frame = env.reset()
    episode, target = env.current_episode, env.target_category()

    agent = NavAgent(
        cfg, detector, scorer, verifier, target, profiler=Profiler(),
        nav_fn=env.action_to_goal if cfg.agent.use_habitat_navmesh else None,
        reachable_fn=env.is_reachable if cfg.agent.use_habitat_navmesh else None,
        ctf_planner=ctf,
    )
    traj, steps = [], 0
    budget = max_steps if max_steps is not None else cfg.agent.max_steps
    while not env.episode_over and steps < budget:
        action = agent.act(frame)
        traj.append(np.asarray(frame.camera_position, dtype=float))
        frame = env.step(action)
        steps += 1

    scene = trimesh.Scene()
    for t in agent.object_layer.tracks():
        e = t.ellipsoid
        sphere = trimesh.creation.uv_sphere(radius=1.0, count=[12, 12])
        M = np.eye(4)
        M[:3, :3] = np.asarray(e.R, float) @ np.diag(np.asarray(e.axes, float))
        M[:3, 3] = agent.object_layer.center_of(t)
        sphere.apply_transform(M)
        sphere.visual.vertex_colors = _label_color(t.label) + [160]
        scene.add_geometry(sphere, node_name=f"{t.label}_{t.id}")

    goals = _goal_points(episode)
    for i, g in enumerate(goals):
        s = trimesh.creation.uv_sphere(radius=0.08, count=[8, 8])
        s.apply_translation(g)
        s.visual.vertex_colors = [0, 220, 90, 255]
        scene.add_geometry(s, node_name=f"goal_{i}")

    if len(traj) > 1:
        path = trimesh.load_path(np.array(traj))
        scene.add_geometry(path, node_name="trajectory")

    tag = f"ep{episode.episode_id}_{Path(str(episode.scene_id)).stem.split('.')[0]}_{target}"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{tag}.glb"
    scene.export(out_file)

    m = env.metrics()
    summary = {
        "episode_id": str(episode.episode_id), "target": target, "steps": steps,
        "success": float(m.get("success", 0.0)),
        "distance_to_goal": float(m.get("distance_to_goal", -1.0)),
        "n_tracks": len(agent.object_layer.tracks()), "output": str(out_file),
    }
    env.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay one episode into a 3D scene-graph recording.",
        epilog="Hydra overrides are passed through, e.g. "
               "`+experiment=scene_cvZr5TUy5C5 floor.semantic_stairs=true`.",
    )
    ap.add_argument("--episode-index", type=int, default=0,
                    help="index into the (filtered) episode list; default 0")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop early; useful for a quick look at a 500-step episode")
    ap.add_argument("--floor-every", type=int, default=10,
                    help="log costmap planes every N steps (they dominate file size)")
    ap.add_argument("--out", type=str, default="outputs/inspect")
    ap.add_argument("--gltf", action="store_true",
                    help="write a final-state .glb via trimesh instead of a Rerun .rrd")
    args, overrides = ap.parse_known_args()

    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    cfg_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    runner = run_gltf if args.gltf else run_rerun
    kwargs = {} if args.gltf else {"floor_every": args.floor_every}
    summary = runner(cfg, Path(args.out), args.episode_index, args.max_steps, **kwargs)

    print("\n--- scene graph inspection ---")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    if not args.gltf:
        print(f"\nOpen with:  rerun {summary['output']}")


if __name__ == "__main__":
    main()
