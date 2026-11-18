#!/usr/bin/env python3
"""Scene-based paper teaser with a real HM3D cutaway and schematic annotations.

The backdrop is an orthographic section of 00873-bxsVRursffK rendered by
``render_scene_graph_multifloor.py`` at section fraction 0.35. Its upper bedroom,
lower kitchen, and exposed staircase are real scene geometry. The cup positions,
robot, and arrows explain the dynamic cross-floor problem; they are illustrative
and must not be described as a logged episode trajectory.

    python3 scripts/render_teaser_scene.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "docs/figures/hm3d_section_00873.png"
INK = "#202a32"
MUTED = "#5c6972"
BLUE = "#2673c9"
RED = "#bf3034"
GREEN = "#168443"
RIGHT_X = 0.707
RIGHT_W = 0.275
VIEW_Y_MIN = 0.05
VIEW_Y_MAX = 0.91


def card(ax, x, y, w, h, *, face, edge, radius=0.018, lw=1.1, z=2):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, a, b, *, color, lw=3.8, scale=17, z=9, rad=0):
    # White outline keeps a route legible over the photograph.
    patch = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale,
                            connectionstyle=f"arc3,rad={rad}",
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                            zorder=z)
    patch.set_path_effects([pe.Stroke(linewidth=lw + 3.2, foreground="white"),
                            pe.Normal()])
    ax.add_patch(patch)


def ring(ax, x, y, *, color, r=0.022, lw=2.8, fill="white", z=12):
    # Adjust the data-space y radius so the marker appears round on the page.
    physical_y_ratio = 10.7 / 4.4 * (VIEW_Y_MAX - VIEW_Y_MIN)
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * physical_y_ratio,
                         facecolor=fill, edgecolor=color, linewidth=lw,
                         zorder=z))


def label(ax, x, y, text, *, color=INK, size=9.3, weight="bold", ha="left",
          box=True, z=15):
    bbox = dict(boxstyle="round,pad=0.32,rounding_size=0.22",
                facecolor="white", edgecolor="#cad4d8", linewidth=0.9,
                alpha=0.96) if box else None
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            va="center", ha=ha, zorder=z, bbox=bbox)


def cross(ax, x, y, *, color=RED, dx=0.012, dy=0.030, lw=3.6):
    for yy0, yy1 in ((-dy, dy), (dy, -dy)):
        line, = ax.plot([x - dx, x + dx], [y + yy0, y + yy1],
                        color=color, lw=lw, solid_capstyle="round", zorder=16)
        line.set_path_effects([pe.Stroke(linewidth=lw + 3.1,
                                        foreground="white"), pe.Normal()])


def cup(ax, x, y, *, color=RED, ghost=False):
    w, h = 0.014, 0.052
    ax.add_patch(Polygon([(x - w, y + h), (x + w, y + h),
                          (x + 0.73 * w, y), (x - 0.73 * w, y)],
                         facecolor="white" if ghost else color,
                         edgecolor=color, linewidth=2.0,
                         linestyle="--" if ghost else "-", zorder=14))
    ax.add_patch(Ellipse((x, y + h), 2 * w, 0.015,
                         facecolor="white" if ghost else color,
                         edgecolor=color, linewidth=1.8, zorder=15))
    ax.add_patch(Ellipse((x + w * 1.06, y + 0.029), 0.016, 0.035,
                         facecolor="none", edgecolor=color, linewidth=2.0,
                         zorder=14))


def robot(ax, x, y):
    ring(ax, x, y, color=BLUE, r=0.015, lw=2.8, fill=BLUE, z=13)
    ax.text(x, y, "R", ha="center", va="center", color="white",
            fontsize=9.3, fontweight="bold", zorder=14)


def floor_tag(ax, x, y, text):
    card(ax, x, y - 0.024, 0.100, 0.048, face="#34424a", edge="white",
         radius=0.008, lw=1.6, z=14)
    ax.text(x + 0.050, y, text, ha="center", va="center", fontsize=8.2,
            color="white", fontweight="bold", zorder=15)


def graph_node(ax, x, y, text, color):
    card(ax, x - 0.0305, y - 0.034, 0.061, 0.068,
         face="white", edge=color, radius=0.009, lw=1.3, z=5)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=INK, zorder=7)


def graph_panel(ax, y, *, stale):
    color = RED if stale else GREEN
    face = "#fff3f3" if stale else "#ecf9ef"
    title = "Outdated belief" if stale else "Belief revised"
    h = 0.325
    card(ax, RIGHT_X, y, RIGHT_W, h, face=face,
         edge="#e5c8ca" if stale else "#bfe2c7", radius=0.018, lw=1.3)
    ax.text(RIGHT_X + RIGHT_W / 2, y + h - 0.057, title,
            ha="center", fontsize=14.5, fontweight="bold", color=color,
            zorder=5)
    nodes = ("floor 2", "bedroom", "bedside\ntable", "cup") if stale else (
        "floor 1", "kitchen", "counter", "cup")
    kinds = ("FLOOR", "ROOM", "CONTAINER", "OBJECT")
    xx = [RIGHT_X + 0.038, RIGHT_X + 0.104, RIGHT_X + 0.170,
          RIGHT_X + 0.236]
    node_y = y + 0.145
    for a, b in zip(xx, xx[1:]):
        ax.plot([a + 0.029, b - 0.029], [node_y, node_y],
                color=color, lw=2.3, linestyle="--" if stale else "-", zorder=4)
    for x, node, kind in zip(xx, nodes, kinds):
        graph_node(ax, x, node_y, node, color)
        ax.text(x, y + 0.206, kind, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=color, zorder=6)
    if stale:
        cross(ax, (xx[2] + xx[3]) / 2, node_y, dx=0.008,
              dy=0.025, lw=2.8)
def scene_panel(ax, image):
    # The section is a real scene render. Black void was made transparent when
    # exporting the PNG, so both storeys float on the paper background.
    ax.imshow(image, extent=(0.020, 0.680, 0.075, 0.885),
              origin="upper", interpolation="bilinear", aspect="auto", zorder=1)
    floor_tag(ax, 0.024, 0.737, "UPSTAIRS")
    floor_tag(ax, 0.024, 0.316, "DOWNSTAIRS")

    # The red cross marks the now-empty bedroom site. The prior cup appears
    # in the scene graph, so a ghost cup and numbered badges add no information.
    robot(ax, 0.245, 0.630)
    cross(ax, 0.145, 0.655, dx=0.010, dy=0.025, lw=3.1)
    label(ax, 0.178, 0.727, "old location empty", color=RED, size=8.7)
    ax.annotate("", xy=(0.145, 0.655), xytext=(0.190, 0.700),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                mutation_scale=12), zorder=15)

    # The dashed outline sits on the visible stair treads. One leader labels
    # this flight; the thick route, drawn separately, shows the floor switch.
    card(ax, 0.406, 0.137, 0.063, 0.213, face="none", edge=GREEN,
         radius=0.006, lw=1.8, z=7).set_linestyle((0, (4, 3)))
    label(ax, 0.490, 0.435, "stairs recognized\nfrontier selects switch",
          color=GREEN, size=8.5)
    ax.annotate("", xy=(0.467, 0.340), xytext=(0.506, 0.407),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5,
                                mutation_scale=12), zorder=15)

    # The continuous route descends through the photographed treads and then
    # turns into the kitchen. Its last arrow stops just outside the cup icon.
    route = [(0.263, 0.606), (0.397, 0.541), (0.432, 0.343),
             (0.432, 0.180), (0.350, 0.218)]
    cup_approach = (0.176, 0.251)
    route_line, = ax.plot(*zip(*route), color=GREEN, lw=4.0,
                          solid_capstyle="round", solid_joinstyle="round",
                          zorder=9)
    route_line.set_path_effects([pe.Stroke(linewidth=7.2,
                                           foreground="white"), pe.Normal()])
    # Make the terminal segment the arrow itself. A separate arrow laid over
    # a complete line left a visible line tail protruding past the arrowhead.
    arrow(ax, route[-1], cup_approach, color=GREEN,
          lw=4.0, scale=17, z=10)
    cup(ax, 0.145, 0.224, color=RED)
    label(ax, 0.072, 0.159, "cup found in kitchen",
          color=GREEN, size=8.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/figures/teaser_scene.pdf")
    parser.add_argument("--scene-image", type=Path, default=SCENE)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not args.scene_image.exists():
        raise SystemExit(f"missing scene image: {args.scene_image}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    # This size gives legible type when the teaser is placed at two-column width.
    fig = plt.figure(figsize=(10.7, 4.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(VIEW_Y_MIN, VIEW_Y_MAX)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    scene_panel(ax, plt.imread(args.scene_image))
    graph_panel(ax, 0.535, stale=True)
    graph_panel(ax, 0.122, stale=False)
    ax.annotate("", xy=(RIGHT_X + RIGHT_W / 2, 0.485),
                xytext=(RIGHT_X + RIGHT_W / 2, 0.530),
                arrowprops=dict(arrowstyle="-|>", color="#7c888d",
                                lw=2.5, mutation_scale=18), zorder=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg", ".png"):
        target = args.out.with_suffix(ext)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
