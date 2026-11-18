#!/usr/bin/env python3
"""Publication figure: the staircase itself, named, and the climb onto it.

`render_stair_evidence.py` argues that a staircase is *evidence* -- a mask, a
flight, a portal, an edge.  This figure answers the other half of the question
a reader asks at B.7: what are those things ON A REAL STAIRCASE, and what does
the agent DO with them.  So the two big panels are the agent's own camera at
the two mouths of one flight in `00800-TEEsavR23oF`, with the saved map's
flight cells projected into them, and the bottom row is the same staircase
from above with the two climbs this episode actually made drawn on it.

Everything coloured is measured:

* flight cells, foot, top, stair mask     `outputs/maps_try_steps/00800-*.{json,npz}`
* the mouth the agent drove to, the path, the actions, the height gained
                                          `climb_trace` of the episode named below
* the StairEdge                           `connectivity` of the saved map

The camera poses of (a) and (b) are the logged agent poses at the start of the
two climbs; the headings come from the next logged displacement, since the
trace stores position and not yaw.  Word labels on the photographs ("tread",
"landing") are annotations, and are the only hand-placed thing here.

    python scripts/render_stair_anatomy.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osg.core.paths import hm3d_scene_root
from osg.mapping.costmap import Costmap2D, FREE, OCCUPIED
from osg.mapping.stairs import find_flights

SCENE = "00800-TEEsavR23oF"
MAP = ROOT / "outputs/maps_try_steps" / f"{SCENE}.json"
RUN = ROOT / "outputs/mf5_pass2_p1500" / SCENE / "episodes.jsonl"
CACHE = ROOT / "outputs/figures"

CAMERA_HEIGHT = 0.88        # agent.camera_height
HFOV = 79.0                 # eval.hfov_deg
TILT = 30.0                 # habitat's look_down
CARROT_M = 0.8              # agent.climb_carrot_m
REACH_M = 0.6               # agent.stair_reach_m

INK = "#1f2b33"
MUTED = "#5d6b74"
ORANGE = "#e59029"
TEAL = "#126c86"
FOOT = "#bd3f3c"
TOP = "#0d8798"
DOWN = "#7660a9"
UP = "#2e7d4f"
PAPER = "#f7f9f9"


# ------------------------------------------------------------------ the map

def load_storey(path: pathlib.Path, which: str):
    """The storey of a saved map that carries the stair mask."""
    blob = json.loads(path.read_text())
    arrays = np.load(path.with_suffix(".npz"))
    floors = sorted(blob["floors"], key=lambda f: float(f["height_y"]))
    spec = floors[0] if which == "lower" else floors[-1]
    prefix = spec["prefix"]
    grid = arrays[prefix + "grid"]
    res = float(spec["resolution"])
    cm = Costmap2D(resolution=res, size_m=grid.shape[0] * res, track_height=True)
    cm.grid = grid
    cm.origin = np.asarray(arrays[prefix + "origin"], dtype=float)
    cm.height = arrays[prefix + "height"]
    cm.stair_mask = arrays[prefix + "stair_mask"].astype(bool)
    return blob, cm, floors


def pick_flight(cm, floors):
    """The flight of this storey that the stored stair mask agrees with."""
    lower, upper = float(floors[0]["height_y"]), float(floors[-1]["height_y"])
    flights = [f for f in find_flights(cm, lower, new_level_m=upper - lower,
                                       min_span_m=1.0) if f.kind == "up"]
    if not flights:
        raise SystemExit("no upward flight on this storey")

    def overlap(f):
        rc = f.cells_rc
        return int(cm.stair_mask[rc[:, 0], rc[:, 1]].sum())

    flight = max(flights, key=overlap)
    if overlap(flight) < 100:
        raise SystemExit("the best flight does not overlap the saved stair mask")
    return flight, overlap(flight)


# ------------------------------------------------------------- the episode

def climbs(run: pathlib.Path):
    """The episode's climb trace, split into its separate climbs."""
    record = json.loads(run.read_text().splitlines()[0])
    trace = record["climb_trace"]
    out, cur = [], [trace[0]]
    for a, b in zip(trace, trace[1:]):
        if b["step"] - a["step"] > 3:
            out.append(cur)
            cur = []
        cur.append(b)
    out.append(cur)
    return record, out


def heading_of(seg, i0: int = 0):
    """Yaw from the next logged displacement; the trace stores no rotation."""
    p0 = np.asarray(seg[i0]["xy"], dtype=float)
    for c in seg[i0 + 1:]:
        d = np.asarray(c["xy"], dtype=float) - p0
        if np.linalg.norm(d) > 0.05:
            return float(np.arctan2(-d[0], -d[1]))   # habitat faces -Z at yaw 0
    raise SystemExit("the climb never moved")


# ---------------------------------------------------------------- renderer

class View:
    """One RGB+depth render of the scene, with its projection."""

    def __init__(self, rgb, depth, pos, yaw, pitch, hfov):
        self.rgb, self.depth = rgb, depth
        self.pos = np.asarray(pos, dtype=float)
        self.yaw, self.pitch, self.hfov = float(yaw), float(pitch), float(hfov)
        h, w = depth.shape
        self.f = (w / 2) / np.tan(np.deg2rad(hfov) / 2)
        self.cx, self.cy = w / 2, h / 2
        cy_, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        self.rot = ry @ rx                       # world <- camera

    def project(self, pts3d):
        """Pixels, depth along the view axis, and a visibility flag."""
        cam = (np.asarray(pts3d, dtype=float) - self.pos) @ self.rot
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * cam[:, 0] / z
            v = self.cy - self.f * cam[:, 1] / z
        h, w = self.depth.shape
        inside = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        seen = np.zeros(len(z), dtype=bool)
        idx = np.nonzero(inside)[0]
        if len(idx):
            d = self.depth[v[idx].astype(int), u[idx].astype(int)]
            # Tolerant: the height layer samples a 0.05 m cell and the mesh is
            # a reconstruction, so demand only that nothing solid is in front.
            seen[idx] = (d <= 0) | (z[idx] < d + 0.25)
        return u, v, z, inside & seen


def render(scene: str, pos, yaw, pitch, *, res=(960, 1280), hfov=HFOV,
           cache: pathlib.Path = CACHE) -> View:
    key = (f"view_{scene}_{pos[0]:.2f}_{pos[1]:.2f}_{pos[2]:.2f}"
           f"_{np.rad2deg(yaw):.1f}_{np.rad2deg(pitch):.1f}_{res[0]}.npz")
    path = cache / key
    if path.exists():
        blob = np.load(path)
        return View(blob["rgb"], blob["depth"], pos, yaw, pitch, hfov)

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
        spec.position = np.array([0, 0, 0], dtype=np.float32)
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    import quaternion
    state = habitat_sim.AgentState()
    state.position = np.asarray(pos, dtype=np.float32)
    state.rotation = (quaternion.from_rotation_vector([0, yaw, 0])
                      * quaternion.from_rotation_vector([pitch, 0, 0]))
    sim.get_agent(0).set_state(state)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"][..., :3], dtype=np.uint8)
    depth = np.asarray(obs["depth"], dtype=np.float32)
    sim.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=rgb, depth=depth)
    return View(rgb, depth, pos, yaw, pitch, hfov)


# ------------------------------------------------------------------ drawing

HEIGHT_CMAP = "viridis"      # flight cells, coloured by height above the storey


def tag(ax, xy, text, *, color=INK, fs=10.0, ha="center", va="center"):
    ax.annotate(text, xy, ha=ha, va=va, fontsize=fs, color=color,
                fontweight="bold", zorder=14,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])


def callout(ax, xy, text, xytext, *, color, fs=14.5, lw=2.2, rad=0.14):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="axes fraction",
                ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold", zorder=15,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=1, shrinkB=5,
                                connectionstyle=f"arc3,rad={rad}"),
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#c8d2d6", linewidth=1.0, alpha=0.96))


def wrapped(ax, x, y, text, *, width, fs=9.0, color=INK):
    """A paragraph with a predictable height; returns the y it ends at."""
    lines = textwrap.wrap(text, width)
    ax.text(x, y, "\n".join(lines), fontsize=fs, color=color, va="top",
            ha="left", linespacing=1.46, transform=ax.transAxes)
    return y - 0.0265 * len(lines)


def photo_axes(ax, view, crop_w, *, title):
    w = int(view.rgb.shape[1] * crop_w)
    ax.imshow(view.rgb[:, :w], interpolation="bilinear")
    ax.set_xlim(0, w)
    ax.set_ylim(view.rgb.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=18.5, color=INK, fontweight="bold", pad=9)
    for spine in ax.spines.values():
        spine.set_color("#c7d0d4")
        spine.set_linewidth(1.0)


def draw_cells(ax, view, pts, rel_h, *, vmax, size=9.0):
    """Flight cells in the photograph, coloured by height above the storey."""
    u, v, _, ok = view.project(pts)
    ax.scatter(u[ok], v[ok], s=size, c=rel_h[ok], cmap=HEIGHT_CMAP, vmin=0,
               vmax=vmax, alpha=0.85, linewidths=0, zorder=6)
    return int(ok.sum())


def px(view, pt):
    u, v, _, _ = view.project(np.asarray(pt, dtype=float)[None, :])
    return float(u[0]), float(v[0])


def marker(ax, view, pt, *, color, symbol="o", ms=13):
    u, v = px(view, pt)
    ax.plot([u], [v], marker=symbol, markersize=ms, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=2.0, linestyle="none",
            zorder=12)
    return u, v


def run_slope(cells3d, floor_y, axis, foot_xy):
    """Rise per metre along the flight's axis, from the run's own cells."""
    run = cells3d[cells3d[:, 1] - floor_y > 0.9]
    along = (run[:, [0, 2]] - foot_xy) @ axis
    return float(np.polyfit(along, run[:, 1], 1)[0])


def run_point(cells3d, floor_y, frac):
    """A cell a given fraction up the ASCENDING RUN.

    Sorting the whole component by height would land on its low plateau (the
    sofa beside the foot), so only cells a stair's height above the storey
    count as the run.
    """
    run = cells3d[cells3d[:, 1] - floor_y > 0.9]
    run = run[np.argsort(run[:, 1])]
    return run[min(int(frac * len(run)), len(run) - 1)]


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/figures/F_stair_anatomy.pdf"))
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--run", default=str(RUN))
    args = ap.parse_args()

    blob, cm, floors = load_storey(pathlib.Path(args.map), "lower")
    flight, _ = pick_flight(cm, floors)
    lower_y = float(floors[0]["height_y"])
    upper_y = float(floors[-1]["height_y"])
    _, segs = climbs(pathlib.Path(args.run))
    descent = segs[0]

    xy = np.stack([cm.grid_to_world(rc.astype(float)) for rc in flight.cells_rc])
    cells3d = np.column_stack([xy[:, 0], flight.heights, xy[:, 1]])
    rel_h = flight.heights - lower_y
    vmax = float(rel_h.max())
    foot3d = np.array([flight.foot_xy[0], float(flight.heights.min()),
                       flight.foot_xy[1]])
    top3d = np.array([flight.top_xy[0], float(flight.heights.max()),
                      flight.top_xy[1]])

    # (a) one stride behind the lowest step, on the staircase's own axis, at
    # the agent's eye height and with the agent's own field of view.
    axis = flight.top_xy - flight.foot_xy
    axis /= np.linalg.norm(axis)
    side = np.array([-axis[1], axis[0]])
    a_pos = flight.foot_xy - 1.4 * axis - 0.45 * side
    a_yaw = float(np.arctan2(-axis[0], -axis[1]))
    view_a = render(SCENE, (a_pos[0], lower_y + CAMERA_HEIGHT, a_pos[1]), a_yaw, 0.0)

    # (b) a logged pose, six steps into a real descent, after its one look_down.
    b = descent[6]
    b_pos = np.asarray(b["xy"], dtype=float)
    b_yaw = heading_of(descent, 6)
    view_b = render(SCENE, (b_pos[0], float(b["standing"]) + CAMERA_HEIGHT, b_pos[1]),
                    b_yaw, -np.deg2rad(TILT))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(14.6, 6.2), dpi=220, facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.008, right=0.992, top=0.945, bottom=0.012,
                          wspace=0.035)

    # ----------------------------------------------------------------- (a)
    ax_a = fig.add_subplot(gs[0, 0])
    photo_axes(ax_a, view_a, 0.76, title="(a)  Upward traversal")
    draw_cells(ax_a, view_a, cells3d, rel_h, vmax=vmax, size=9.0)
    marker(ax_a, view_a, foot3d, color=FOOT)
    marker(ax_a, view_a, top3d, color=TOP)
    callout(ax_a, px(view_a, top3d),
            "top endpoint",
            (0.700, 0.945), color=TOP, rad=-0.10)
    callout(ax_a, px(view_a, foot3d),
            "$x_{entry}$ (up)",
            (0.420, 0.110), color=FOOT, rad=0.18)
    callout(ax_a, px(view_a, run_point(cells3d, lower_y, 0.55)),
            "stair flight",
            (0.290, 0.700), color=TEAL, rad=0.10)
    slope = run_slope(cells3d, lower_y, axis, flight.foot_xy)
    tread_xy = flight.foot_xy + axis * 0.85 - side * 0.45
    tread_y = float(flight.heights.min()) + slope * 0.85
    callout(ax_a, px(view_a, np.array([tread_xy[0], tread_y, tread_xy[1]])),
            "flat tread", (0.235, 0.425), color=ORANGE, rad=0.10)
    riser_xy = tread_xy - axis * 0.42
    callout(ax_a, px(view_a, np.array([riser_xy[0], tread_y - slope * 0.42 - 0.09,
                                       riser_xy[1]])),
            "stair mask", (0.235, 0.290),
            color=ORANGE, rad=-0.10)

    # ----------------------------------------------------------------- (b)
    ax_b = fig.add_subplot(gs[0, 1])
    photo_axes(ax_b, view_b, 0.72, title="(b)  Downward traversal")
    draw_cells(ax_b, view_b, cells3d, rel_h, vmax=vmax, size=11.0)
    callout(ax_b, px(view_b, top3d),
            "$x_{entry}$ (down)",
            (0.655, 0.865), color=TOP, rad=-0.12)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(out.with_suffix(ext), facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} (+ .pdf, .svg)")


if __name__ == "__main__":
    main()
