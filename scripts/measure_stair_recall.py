"""Does YOLOE actually see stairs on HM3D, and does the geometry gate help?

The plan's up-stair detector is "YOLOE's `stairs` class AND a geometric check".
That is only worth building a CLIMB state on top of if the detector fires at
all on these scenes -- an open-vocabulary class name is no guarantee. If recall
is near zero the up-stair branch has to become pure geometry instead, and it is
much cheaper to learn that here than after the state machine is written.

**Ground truth comes from the navmesh, not from annotations.** For a cross-floor
episode, habitat's geodesic path from the start to a goal view-point must cross
a staircase, and it is the only way up. Path waypoints where the height climbs
steeply therefore mark real stairs, with no semantic labels required:

    STAIR pose      on the path, within --near-m of a steeply-rising segment,
                    facing along the path (what the agent would see approaching)
    CONTROL pose    on the path, more than --far-m from every rising segment,
                    facing along the path (ordinary flat corridor / room)

Both sets are rendered by teleporting the camera (`get_observations_at`), never
by stepping, so the agent's own behaviour cannot bias the sample.

Reported per pose set: how often YOLOE emits a `stairs` detection (recall on
STAIR, false-positive rate on CONTROL), and how often the geometric gate from
osg.mapping.stairs accepts it -- so the gate's cost in recall and its gain in
precision are separable. The down-stair geometric signal is measured on the
same poses, since it needs no detector at all.

**Camera pitch (S14a).** ASCENT tilts the camera to find stairs: it emits
LOOK_UP when the stair class appears in the upper half of the frame within 2 m
(`ascent_policy.py:534-542`) and routes detections into the up- or down-stair
map by the sign of the pitch. OSG never pitches -- its camera sits at 0.88 m,
level, for the whole episode.

That suggests a different explanation for the 24% up-stair recall measured here
than "YOLOE is weak at stairs": standing near the foot of a flight, the treads
recede UPWARD out of a level 79-degree frame. If so it is a viewpoint problem
and no detector swap fixes it. So every pose is now rendered at each of
--pitch-deg and recall is reported per pitch.

Pitch is applied to the SENSOR node, as habitat's own look_up does, and folded
into T_wc -- otherwise the backprojected geometry silently disagrees with the
image. A sign check is printed with the results: looking up must RAISE the mean
height of the backprojected points.

Run inside the nav container:

    python scripts/measure_stair_recall.py                     # 12 episodes
    python scripts/measure_stair_recall.py --episodes 30 --dump-dir /tmp/stairs
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()

KINDS = ("stair_up", "stair_down", "control")


def _set_sensor_pitch(sim, deg: float) -> None:
    """Tilt every sensor about its local x-axis, as habitat's look_up action does.

    The sensor node is a child of the agent body, so this composes with the yaw
    passed to get_observations_at rather than replacing it.
    """
    import magnum as mn

    for sensor in sim.get_agent(0)._sensors.values():
        sensor.node.rotation = mn.Quaternion.rotation(
            mn.Deg(float(deg)), mn.Vector3.x_axis()
        )


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(float(deg))
    c, s_ = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]], dtype=float)


def _ramp_geometry(stair_geometry, pts_world, cam_xy, floor_y, HEIGHT_AXIS, PLANE,
                   h_lo=0.15, h_hi=2.0, r_lo=0.5, r_hi=4.0):
    """REJECTED candidate, kept here so the negative result stays reproducible.

    The idea: with no detector, find an up staircase as a surface whose height
    rises with distance -- a wall has no horizontal span, a table is flat.
    Measured on HM3D it scores 32% recall at 28% false-positive on flat control
    poses, i.e. it fires on ordinary corridors nearly as often as on stairs.
    Not used by osg.mapping.stairs; see docs/AB_RESULTS.md.
    """
    import numpy as np

    if pts_world.shape[0] == 0:
        return stair_geometry(pts_world, cam_xy, floor_y)
    h = pts_world[:, HEIGHT_AXIS] - floor_y
    r = np.linalg.norm(pts_world[:, list(PLANE)] - np.asarray(cam_xy), axis=1)
    sel = (h > h_lo) & (h < h_hi) & (r > r_lo) & (r < r_hi)
    return stair_geometry(pts_world[sel], cam_xy, floor_y)


def _rotation_facing(direction: np.ndarray):
    """Habitat agent rotation (quaternion) looking along `direction` (world)."""
    import quaternion  # noqa: F401  (registers the numpy dtype)

    yaw = float(np.arctan2(-direction[0], -direction[2]))  # habitat: -z forward
    return np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)


def _geodesic_path(sim, start, goal):
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(goal, dtype=np.float32)
    if not sim.pathfinder.find_path(path):
        return None
    return [np.asarray(p, dtype=float) for p in path.points]


def _rising_segments(points, rise_m: float = 0.3):
    """Indices i where the path climbs/descends more than `rise_m` from i to
    i+1. Those segments are the staircases: a geodesic path only changes floor
    that way."""
    return [
        i for i in range(len(points) - 1)
        if abs(points[i + 1][1] - points[i][1]) > rise_m
    ]


def _sample_poses(points, stair_idx, near_m, far_m, step_m=0.5):
    """Densify the path and label each sample by distance to a stair segment."""
    dense = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        seg = b - a
        n = max(1, int(np.linalg.norm(seg) / step_m))
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            dense.append((a + t * seg, seg, i))
    if not dense:
        return []

    # Each stair segment carries the direction the agent travels through it:
    # approaching a flight from the bottom (up) and from the top (down) look
    # completely different to a depth sensor, so they must be scored apart.
    stair_pts = []
    for i in stair_idx:
        up = points[i + 1][1] > points[i][1]
        stair_pts.append((points[i], up))
        stair_pts.append((points[i + 1], up))

    out = []
    for pos, seg, seg_i in dense:
        if stair_pts:
            near = min(stair_pts, key=lambda sp: float(np.linalg.norm(pos - sp[0])))
            d = float(np.linalg.norm(pos - near[0]))
            up = near[1]
        else:
            d, up = np.inf, True
        if seg_i in stair_idx or d <= near_m:
            kind = "stair_up" if up else "stair_down"
        elif d >= far_m:
            kind = "control"
        else:
            continue  # ambiguous band: neither clearly at nor clearly away
        out.append((kind, pos, seg))
    return out


def run(cfg: DictConfig, n_episodes: int, near_m: float, far_m: float,
        max_poses: int, dump_dir: str | None,
        pitches: tuple = (0.0, 30.0),
        rednet: bool = False,
        rednet_weights: str = "data/weights/rednet_semmap_mp3d_40.pth") -> None:
    from osg.core.types import CameraIntrinsics, FrameData
    from osg.eval.runner import _episode_uid, _goal_floor_gap_m, build_detector
    from osg.mapping.stairs import STAIR_LABELS, stair_geometry
    from osg.sim.habitat_env import HabitatObjectNavEnv, _GL_TO_CV
    from osg.core.geometry import backproject, quat_to_matrix
    from osg.mapping.costmap import HEIGHT_AXIS, PLANE

    env = HabitatObjectNavEnv(cfg)
    # Same up-front dataset filter run_eval applies, so `eval=dev50_mf` really
    # yields the cross-floor split rather than the head of the dataset.
    if cfg.eval.episode_ids:
        wanted = {str(e) for e in cfg.eval.episode_ids}
        kept = [ep for ep in env.env.episodes if _episode_uid(ep) in wanted]
        if kept:
            env.env.episodes = kept
    detector = build_detector(cfg)
    detector.set_vocabulary(["stairs"] + list(cfg.detector.vocabulary))
    # S33: the up-stair signal ASCENT actually uses. Built here rather than via
    # build_stair_segmenter so the probe can measure it without the agent flag
    # being on -- measuring is exactly how that flag gets decided.
    segmenter = None
    if rednet:
        from osg.perception.stair_seg import RedNetStairSegmenter
        segmenter = RedNetStairSegmenter(
            rednet_weights,
            depth_min_m=float(getattr(cfg.eval, "depth_min_m", 0.5)),
            depth_max_m=float(getattr(cfg.eval, "depth_max_m", 5.0)),
        )
    sim = env.env.sim
    intr = env.intrinsics

    counts: Counter = Counter()
    keys = [f"{k}@{p:+g}" for k in KINDS for p in pitches]
    slopes = {k: [] for k in keys}
    ramp_slopes = {k: [] for k in keys}
    mean_h = {k: [] for k in keys}
    labels_seen = Counter()
    dumped = 0
    if dump_dir:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

    for ep_i in range(n_episodes):
        env.reset()
        episode = env.current_episode
        gap = _goal_floor_gap_m(episode)
        if gap is None or gap <= 1.0:
            continue  # not a cross-floor episode: no staircase on the route

        start = np.asarray(episode.start_position, dtype=float)
        goal = None
        for g in episode.goals or []:
            for vp in getattr(g, "view_points", None) or []:
                p = np.asarray(vp.agent_state.position, dtype=float)
                if abs(p[1] - start[1]) > 1.0:
                    goal = p
                    break
            if goal is not None:
                break
        if goal is None:
            continue

        points = _geodesic_path(sim, start, goal)
        if not points or len(points) < 2:
            continue
        stair_idx = _rising_segments(points)
        if not stair_idx:
            counts["episodes_without_a_rising_segment"] += 1
            continue
        counts["episodes_used"] += 1

        poses = _sample_poses(points, stair_idx, near_m, far_m)
        rng = np.random.default_rng(cfg.seed + ep_i)
        if len(poses) > max_poses:
            poses = [poses[i] for i in rng.choice(len(poses), max_poses, replace=False)]

        for kind, pos, seg in poses:
          rot = _rotation_facing(seg)
          for pitch in pitches:
            _set_sensor_pitch(sim, pitch)
            obs = sim.get_observations_at(pos.tolist(), rot, keep_agent_at_new_pose=False)
            if obs is None:
                counts[f"{kind}_pose_unreachable"] += 1
                continue
            kind_p = f"{kind}@{pitch:+g}"
            counts[f"{kind_p}_poses"] += 1

            state = sim.get_agent_state()
            # get_observations_at restores the agent, so rebuild the pose we
            # rendered from rather than reading it back.
            # world <- body <- sensor: the pitch is a child transform of the
            # yaw, so it multiplies on the right. Getting this wrong leaves the
            # image and the backprojection describing different cameras.
            R = quat_to_matrix(rot.w, rot.x, rot.y, rot.z) @ _rot_x(pitch)
            T_wc = np.eye(4)
            T_wc[:3, :3] = R @ _GL_TO_CV
            T_wc[:3, 3] = pos + np.array([0.0, cfg.agent.camera_height, 0.0])
            depth = obs["depth"]
            depth = depth[..., 0] if depth.ndim == 3 else depth
            frame = FrameData(
                frame_id=0, rgb=np.ascontiguousarray(obs["rgb"][..., :3]),
                depth=depth.astype(np.float32), T_wc=T_wc, intrinsics=intr,
            )

            dets = detector.detect(frame.rgb)
            # Everything the detector sees, so "no stairs" is distinguishable
            # from "the detector produced nothing at all on these frames".
            for d in dets:
                labels_seen[d.label.lower().strip()] += 1
            counts[f"{kind_p}_dets_total"] += len(dets)
            stair_dets = [d for d in dets if d.label.lower().strip() in STAIR_LABELS]
            if stair_dets:
                counts[f"{kind_p}_yoloe_fired"] += 1
                best = max(stair_dets, key=lambda d: d.score)
                pts = backproject(frame.depth, intr, T_wc, mask=best.mask,
                                  stride=2, max_depth=5.0)
                geom = stair_geometry(pts, T_wc[:3, 3][list(PLANE)], float(pos[1]))
                slopes[kind_p].append(geom.slope)
                if geom.is_stair_like(0.6, 0.35, 0.30):
                    counts[f"{kind_p}_geometry_agreed"] += 1
                if dump_dir and dumped < 40:
                    import cv2
                    cv2.imwrite(
                        str(Path(dump_dir) / f"{kind}_p{pitch:+g}_{dumped:03d}_s{best.score:.2f}"
                            f"_slope{geom.slope:+.2f}.jpg"),
                        frame.rgb[..., ::-1],
                    )
                    dumped += 1

            if segmenter is not None:
                # ASCENT: fusion_stair_mask = stair_mask & (seg == STAIR_CLASS_ID),
                # believed only when the segmenter alone has >20 stair pixels
                # (obstacle_map.py:520-522). Both halves reported separately so
                # the intersection's cost in recall is visible.
                rn = segmenter.stair_mask(frame)
                if rn is not None:
                    counts[f"{kind_p}_rednet_fired"] += 1
                    # RedNet through OSG's OWN geometric gate -- the
                    # second opinion this repo already has, in place of the
                    # GroundingDINO half ASCENT intersects with.
                    rn_pts = backproject(frame.depth, intr, T_wc, mask=rn,
                                         stride=2, max_depth=5.0)
                    rn_geom = stair_geometry(rn_pts, T_wc[:3, 3][list(PLANE)], float(pos[1]))
                    if rn_geom.is_stair_like(0.6, 0.35, 0.30):
                        counts[f"{kind_p}_rednet_geometry"] += 1
                    if stair_dets:
                        fused = np.zeros_like(rn)
                        for d in stair_dets:
                            fused |= (d.mask.astype(bool) & rn)
                        if fused.any():
                            counts[f"{kind_p}_fusion_fired"] += 1

            # Both detector-free signals, on the same poses.
            allpts = backproject(frame.depth, intr, T_wc, stride=4, max_depth=5.0)
            below = int(((float(pos[1]) - allpts[:, HEIGHT_AXIS]) > 0.35).sum())
            if below >= 50:
                counts[f"{kind_p}_below_floor_geometry"] += 1
            # Sign check: looking up must raise the mean height of what the
            # camera sees. If it does not, _rot_x or the composition order is
            # inverted and every number below is describing the wrong frame.
            mean_h[kind_p].append(float(allpts[:, HEIGHT_AXIS].mean()))

            ramp = _ramp_geometry(stair_geometry, allpts, T_wc[:3, 3][list(PLANE)],
                                  float(pos[1]), HEIGHT_AXIS, PLANE)
            ramp_slopes[kind_p].append(ramp.slope)
            if ramp.is_stair_like(0.8, 0.5, 0.30):
                counts[f"{kind_p}_ramp_geometry"] += 1

    _set_sensor_pitch(sim, 0.0)  # leave the sim as we found it
    env.close()
    _report(counts, slopes, ramp_slopes, mean_h, pitches)
    print("\nevery label the detector emitted (all poses, all pitches):")
    print(f"  {dict(labels_seen.most_common(20))}")
    print(f"  total detections: {sum(labels_seen.values())}")


def _report(counts, slopes, ramp_slopes, mean_h, pitches) -> None:
    def pct(a, b):
        return f"{a:4d}/{b:<4d} {a / b:6.1%}" if b else f"{a:4d}/{0:<4d}    n/a"

    print("\n=== stair detection on HM3D ===")
    print(f"episodes used: {counts['episodes_used']}"
          f"   (skipped, no rising path segment: "
          f"{counts['episodes_without_a_rising_segment']})")

    # Sign check first: if looking up does not raise what the camera sees, the
    # pitch is being applied backwards and nothing below can be trusted.
    print("\nsign check -- mean height of backprojected points, by pitch")
    base = None
    for p in pitches:
        hs = [h for k in KINDS for h in mean_h[f"{k}@{p:+g}"]]
        m = float(np.mean(hs)) if hs else float("nan")
        print(f"  pitch {p:+5g} deg   mean point height {m:+.3f} m")
        if p == 0:
            base = m
    ups = [p for p in pitches if p > 0]
    if ups and base is not None:
        top = float(np.mean([h for k in KINDS for h in mean_h[f"{k}@{max(ups):+g}"]]))
        verdict = "OK" if top > base else "*** INVERTED -- results are meaningless ***"
        print(f"  looking up raises it by {top - base:+.3f} m   {verdict}")

    for kind in KINDS:
        print(f"\n{kind.upper()}")
        for p in pitches:
            kp = f"{kind}@{p:+g}"
            n = counts[f"{kp}_poses"]
            print(f"  pitch {p:+5g} deg  (n={n})")
            print(f"    YOLOE 'stairs' fired    {pct(counts[f'{kp}_yoloe_fired'], n)}")
            print(f"    ...and geometry agreed  {pct(counts[f'{kp}_geometry_agreed'], n)}")
            if counts[f"{kp}_rednet_fired"] or counts[f"{kp}_fusion_fired"]:
                print(f"    RedNet stair class      {pct(counts[f'{kp}_rednet_fired'], n)}")
                print(f"    ...through the geo gate {pct(counts[f'{kp}_rednet_geometry'], n)}")
                print(f"    ...fused with YOLOE     {pct(counts[f'{kp}_fusion_fired'], n)}")
            print(f"    below-floor geometry    {pct(counts[f'{kp}_below_floor_geometry'], n)}")
            print(f"    ramp geometry           {pct(counts[f'{kp}_ramp_geometry'], n)}")
            if slopes[kp]:
                sl = np.array(slopes[kp])
                print(f"    slope of fired masks    median {np.median(sl):+.2f}")

    print("\ninterpretation -- recall vs control false-positive rate")
    for kind in ("stair_up", "stair_down"):
        for p in pitches:
            kp, cp = f"{kind}@{p:+g}", f"control@{p:+g}"
            n, n_c = counts[f"{kp}_poses"], counts[f"{cp}_poses"]
            if not n:
                continue
            for sig, key in (("YOLOE", "yoloe_fired"), ("RedNet", "rednet_fired"),
                             ("RedNet+geom", "rednet_geometry"),
                             ("RedNet+YOLOE", "fusion_fired"), ("ramp", "ramp_geometry"),
                             ("below-floor", "below_floor_geometry")):
                r = counts[f"{kp}_{key}"] / n
                fp = counts[f"{cp}_{key}"] / n_c if n_c else 0.0
                print(f"  {kind:11s} pitch {p:+5g}  {sig:12s} "
                      f"recall {r:5.0%}   control FP {fp:5.0%}")

    print("\nS14a decision rule: if UP-stair YOLOE recall rises materially with"
          "\npitch, the 24% is a viewpoint problem and a look_up probe fixes it"
          "\n(S14b, cheap). If it does not, the detector is the problem and only"
          "\na dedicated segmentation model will move it (S14c, RedNet, ~200 MB).")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--pitch-deg", type=float, nargs="+", default=[0.0, 30.0],
                    help="camera pitches to render each pose at; positive = up")
    ap.add_argument("--near-m", type=float, default=1.5,
                    help="a pose this close to a rising path segment is a STAIR pose")
    ap.add_argument("--far-m", type=float, default=4.0,
                    help="a pose at least this far from every rising segment is a CONTROL pose")
    ap.add_argument("--max-poses", type=int, default=12, help="poses sampled per episode")
    ap.add_argument("--dump-dir", default=None, help="save inspected frames here")
    ap.add_argument("--rednet", action="store_true",
                    help="also measure RedNet's MPCAT40 stair class and ASCENT's "
                         "RedNet-and-detector fusion on the same poses")
    ap.add_argument("--rednet-weights", default="data/weights/rednet_semmap_mp3d_40.pth")
    ap.add_argument("-h", "--help", action="store_true")
    args, hydra_argv = ap.parse_known_args()
    if args.help:
        ap.print_help()
        print(__doc__)
        return
    sys.argv = [sys.argv[0], *hydra_argv]

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _run(cfg: DictConfig) -> None:
        run(cfg, args.episodes, args.near_m, args.far_m, args.max_poses,
            args.dump_dir, tuple(args.pitch_deg), args.rednet, args.rednet_weights)

    _run()


if __name__ == "__main__":
    main()
