#!/usr/bin/env python3
"""Presence belief drawn on the view the agent actually had.

One keyframe pose out of a saved map, re-rendered at full resolution, with
every mapped object the pose can see drawn as the ellipse the map projects
into that image and labelled with its presence score.

Nothing here is placed by hand:

* the pose is a camera pose stored on a track's own observation, so it is a
  view the agent really occupied, with the agent's intrinsics;
* the outline is `Ellipsoid.project` -- the same dual-conic projection the
  association step uses, and the one the visibility test of the presence
  filter rests on;
* `p = sigma(L)` is the log-odds saved on the track, and the miss counter
  beside it is the track's own `n_missed / n_expected`;
* a track whose ellipse falls behind rendered geometry is dropped, using the
  depth image of the same pose.

    python scripts/render_presence_view.py            # the chosen pose
    python scripts/render_presence_view.py --survey   # score poses, print them
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osg.core.paths import hm3d_scene_root
from osg.objects.ellipsoid import Ellipsoid

SCENE = "00800-TEEsavR23oF"
# The pose the survey below picked, pinned so a re-render reproduces the
# figure without re-scoring 2783 stored camera poses.
POSE = 3004
MAP = ROOT / "outputs/maps_try_steps" / f"{SCENE}.json"
OUT = ROOT / "docs/figures/presence_view.png"
CACHE = ROOT / "outputs/figures"

# The presence clamp (objects/presence.py): L in [-6, +3], so p never reaches
# 0 or 1 and a disproved track can always be revived by a sighting.
L_MIN, L_MAX = -6.0, 3.0
INK = "#26353d"
MUTED = "#5b6d75"
# red (disproved) -> amber -> green (believed); readable in print and safe for
# the common colour-vision deficiencies at these lightnesses.
CMAP = LinearSegmentedColormap.from_list(
    "presence", ["#b3352e", "#d98b2b", "#9aa63a", "#237f59"])


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# ------------------------------------------------------------------- the map

def load_tracks(path: pathlib.Path):
    blob = json.loads(path.read_text())
    tracks, poses = [], []
    for rec in blob["tracks"]:
        pres = rec.get("presence", {})
        track = dict(
            id=int(rec["id"]), label=str(rec["label"]),
            floor=int(rec.get("floor_key", 0)),
            ellipsoid=Ellipsoid(center=np.asarray(rec["center"], float),
                                axes=np.asarray(rec["axes"], float),
                                R=np.asarray(rec["R"], float).reshape(3, 3)),
            log_odds=float(pres.get("log_odds", 1.5)),
            n_missed=int(pres.get("n_missed", 0)),
            n_expected=int(pres.get("n_expected", 0)),
            evidence=float(rec.get("evidence", 0.0)),
            n_obs=len(rec.get("observations", [])),
        )
        track["p"] = sigmoid(track["log_odds"])
        tracks.append(track)
        for obs in rec.get("observations", [])[::2]:
            poses.append(dict(K=np.asarray(obs["K"], float).reshape(3, 3),
                              T_cw=np.asarray(obs["T_cw"], float).reshape(4, 4),
                              floor=track["floor"]))
    return tracks, poses


def visible(tracks, pose, depth, *, shape, margin=36, min_px=11, max_px=110,
            near=0.9, far=6.0, behind=0.30, ahead=0.55, min_evidence=1.0):
    """The tracks this pose can see, with their projected ellipses.

    `depth` is the rendered depth of the same pose, and it does two jobs: it
    drops a track that sits behind a wall, and it keeps only tracks whose
    mapped centre agrees with the surface the camera actually sees there. The
    second is what makes the drawn ellipse land on the object a reader can
    see; a track whose geometry disagrees with the render by more than half a
    metre is a mapping error, and this figure is about the belief, not about
    the ellipsoid fit.
    """
    K, T_cw = pose["K"], pose["T_cw"]
    h, w = shape
    scale_v, scale_u = depth.shape[0] / h, depth.shape[1] / w
    out = []
    for track in tracks:
        if track["floor"] != pose["floor"]:
            continue
        ell = track["ellipsoid"].project(K, T_cw)
        if ell is None:
            continue
        u, v = float(ell.mu[0]), float(ell.mu[1])
        if not (margin < u < w - margin and margin < v < h - margin):
            continue
        eig, vec = np.linalg.eigh(ell.cov)
        semi = np.sqrt(np.maximum(eig, 1e-9))
        if semi[0] < min_px or semi[1] > max_px:
            continue
        z = track["ellipsoid"].mean_depth_at(T_cw)
        if not (near < z < far):
            continue
        if track["evidence"] < min_evidence:
            continue
        z_seen = float(depth[int(v * scale_v), int(u * scale_u)])
        # The stored centre sits INSIDE the object, so it is legitimately
        # behind the surface the camera sees -- by up to the object's own
        # size. Anything further back is a different object, or a wall.
        depth_tol = behind + float(np.max(track["ellipsoid"].axes))
        if z_seen <= 0 or not (z_seen - ahead < z < z_seen + depth_tol):
            continue
        angle = float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])))
        out.append(track | dict(u=u, v=v, z=float(z), semi=semi, angle=angle))
    return out


# --------------------------------------------------------------- the renderer

class Sim:
    """One open simulator, so surveying hundreds of poses stays cheap."""

    def __init__(self, scene: str, res=(240, 320), hfov=79.0):
        import habitat_sim

        root = pathlib.Path(hm3d_scene_root())
        stem = scene.split("-", 1)[1]
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(root / f"val/{scene}/{stem}.basis.glb")
        cfg.scene_dataset_config_file = str(
            root / "hm3d_annotated_basis.scene_dataset_config.json")
        cfg.enable_physics = False
        specs = []
        for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                           ("depth", habitat_sim.SensorType.DEPTH)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = uuid, kind
            spec.resolution = [res[0], res[1]]
            spec.hfov = hfov
            spec.position = np.zeros(3, dtype=np.float32)
            specs.append(spec)
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = specs
        self._hab = habitat_sim
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))

    def look(self, T_cw: np.ndarray):
        """Render from a stored world->camera pose.

        The map stores the usual vision convention (x right, y down, z into the
        scene); a habitat sensor looks down its own -z with y up, hence the
        flip of the last two columns.
        """
        import quaternion

        T_wc = np.linalg.inv(T_cw)
        rot = T_wc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        state = self._hab.AgentState()
        state.position = T_wc[:3, 3].astype(np.float32)
        state.rotation = quaternion.from_rotation_matrix(rot)
        self.sim.get_agent(0).set_state(state)
        obs = self.sim.get_sensor_observations()
        return (np.asarray(obs["rgb"][..., :3], dtype=np.uint8),
                np.asarray(obs["depth"], dtype=np.float32))

    def close(self):
        self.sim.close()


def choose(seen, *, k=6, apart=95.0):
    """A readable subset: the extremes of belief first, none on top of another.

    The view usually projects a dozen tracks, several of them overlapping. The
    figure is about the number written beside an object, so it keeps the widest
    spread of belief it can while leaving the labels legible.
    """
    order = sorted(seen, key=lambda t: t["p"])
    ranked = []
    while order:                       # lowest, highest, next lowest, ...
        ranked.append(order.pop(0))
        if order:
            ranked.append(order.pop())
    out = []
    for track in ranked:
        if len(out) >= k:
            break
        here = np.array([track["u"], track["v"]])
        if any(np.linalg.norm(here - np.array([o["u"], o["v"]])) < apart for o in out):
            continue
        out.append(track)
    return out


def survey(tracks, poses, sim, *, shape, n=3000, seed=1):
    """Score sampled poses by how well they show a mix of beliefs."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(poses), size=min(n, len(poses)), replace=False)
    rows = []
    for i in idx:
        _, depth = sim.look(poses[i]["T_cw"])
        seen = visible(tracks, poses[i], depth, shape=shape)
        picked = choose(seen)
        low = [t for t in picked if t["p"] < 0.2]
        high = [t for t in picked if t["p"] > 0.6]
        if len(picked) < 3 or not low or not high:
            continue
        # A good view shows both ends of the belief scale on objects that
        # carry some evidence, spread across the frame.
        score = (min(len(low), 3) + min(len(high), 3)
                 + 0.02 * sum(min(t["n_expected"], 40) for t in picked) / len(picked)
                 + 0.01 * len(picked))
        rows.append((score, int(i), picked))
    rows.sort(key=lambda r: -r[0])
    return rows


# ----------------------------------------------------------------- the figure

def draw(ax, rgb, seen, *, label_px):
    ax.imshow(rgb, interpolation="bilinear")
    h, w = rgb.shape[:2]
    scale = w / float(label_px[1])           # map 640x480 pixels to the render
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Far objects first, so a near label is never hidden by a far one.
    for track in sorted(seen, key=lambda t: -t["z"]):
        color = CMAP(Normalize(0.0, 1.0)(track["p"]))
        ell = Ellipse((track["u"] * scale, track["v"] * scale),
                      2 * track["semi"][1] * scale, 2 * track["semi"][0] * scale,
                      angle=track["angle"], facecolor=color, alpha=.16,
                      edgecolor="none", zorder=4)
        ax.add_patch(ell)
        ring = Ellipse((track["u"] * scale, track["v"] * scale),
                       2 * track["semi"][1] * scale, 2 * track["semi"][0] * scale,
                       angle=track["angle"], facecolor="none", edgecolor=color,
                       linewidth=2.4, zorder=5)
        ring.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"),
                               pe.Normal()])
        ax.add_patch(ring)
        y = track["v"] * scale - track["semi"][0] * scale - 10 * scale / 2
        ax.text(track["u"] * scale, max(y, 16), f"{track['label']}   p = {track['p']:.2f}",
                ha="center", va="bottom", fontsize=11.4, color=color,
                fontweight="bold", zorder=9,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])


def legend(fig, ax, seen):
    """A colour bar for p, and the evidence behind the two extremes."""
    bar = ax.inset_axes([.026, .045, .30, .042])
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=bar,
                 orientation="horizontal")
    bar.set_xticks([0, .5, 1])
    bar.tick_params(labelsize=9.4, colors=INK, length=2, pad=2)
    bar.set_title("presence  $p=\\sigma(L)$", fontsize=10.6, color=INK,
                  fontweight="bold", pad=5)
    for spine in bar.spines.values():
        spine.set_color("#c8d4d7")

    lo = min(seen, key=lambda t: t["p"])
    hi = max(seen, key=lambda t: t["p"])
    lines = [
        f"{hi['label']}:  seen where the map says it is,  "
        f"{hi['n_missed']} misses in {hi['n_expected']} exposed views",
        f"{lo['label']}:  {lo['n_missed']} misses in {lo['n_expected']} exposed "
        f"views, so $L$ sits at the {L_MIN:g} clamp",
    ]
    ax.text(.355, .082, "\n".join(lines), transform=ax.transAxes, fontsize=10.2,
            color=INK, va="center", linespacing=1.6, zorder=9,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#c8d4d7", alpha=.94))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--map", type=pathlib.Path, default=MAP)
    ap.add_argument("--pose", type=int, default=POSE,
                    help="pose index to render; --survey re-scores them all")
    ap.add_argument("--survey", action="store_true",
                    help="print the best scoring poses and stop")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    tracks, poses = load_tracks(args.map)
    shape = (480, 640)                        # the intrinsics the map was built with

    if args.survey:
        sim = Sim(SCENE)
        rows = survey(tracks, poses, sim, shape=shape)
        print(f"{len(rows)} poses show a mix of beliefs; best:")
        for score, i, seen in rows[:6]:
            print(f"  pose {i:5d}  score {score:.1f}  "
                  f"{len(seen)} tracks  "
                  + ", ".join(f"{t['label']} {t['p']:.2f}" for t in
                              sorted(seen, key=lambda t: -t["p"])))
        sim.close()
        return
    pose_id = args.pose

    pose = poses[pose_id]
    sim = Sim(SCENE, res=(960, 1280))
    rgb, depth = sim.look(pose["T_cw"])
    seen = choose(visible(tracks, pose, depth, shape=shape))
    sim.close()
    print(f"pose {pose_id}: {len(seen)} tracks drawn")
    for t in sorted(seen, key=lambda t: -t["p"]):
        print(f"   {t['label']:<14} p={t['p']:.2f}  L={t['log_odds']:+.2f}  "
              f"missed {t['n_missed']}/{t['n_expected']}  {t['z']:.1f} m")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(9.6, 7.4), facecolor="white")
    ax = fig.add_axes((0.012, 0.012, 0.976, 0.976))
    draw(ax, rgb, seen, label_px=shape)
    legend(fig, ax, seen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        path = args.out.with_suffix(ext)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
