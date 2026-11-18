#!/usr/bin/env python3
"""Render a two-band, module-I/O full pipeline figure for the paper.

The HM3D floor plans are real renders; raised graph nodes are a schematic
overlay illustrating containment and stair-edge semantics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
INK = "#24353f"
MUTED = "#5c707a"
LINE = "#83949a"
GRAPH = "#15788a"
BELIEF = "#7663a2"
SEARCH = "#a47637"
NAV = "#27805a"
STAIR = "#c36b3d"
NODE = {
    "building": "#425261", "storey": "#4872ab", "room": "#28a193",
    "container": "#956ab5", "object": "#d78447",
}


def box(ax, x, y, w, h, fill, stroke, lw=1.5, z=2):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=z)
    ax.add_patch(p)


def arrow(ax, a, b, color=LINE, dashed=False, z=6):
    p = FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=15, linewidth=2.0,
        color=color, linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=0, shrinkB=0, zorder=z)
    p.set_path_effects([pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()])
    ax.add_patch(p)


def route(ax, points, color=LINE, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2.0,
                linestyle=(0, (4, 3)) if dashed else "solid", zorder=6)
    arrow(ax, points[-2], points[-1], color, dashed)


def module(ax, x, y, w, h, title, io, color):
    box(ax, x, y, w, h, "white", color, lw=1.8, z=3)
    ax.text(x+w/2, y+h*.69, title, ha="center", va="center",
            fontsize=14.0 if len(title) > 18 else 15.2,
            fontweight="bold", color=color, zorder=7)
    ax.plot([x+.012, x+w-.012], [y+h*.48, y+h*.48],
            color="#e4e9eb", lw=1.0, zorder=4)
    ax.text(x+w/2, y+h*.26, io, ha="center", va="center",
            fontsize=11.0, color=INK, zorder=7)


def sphere(ax, x, y, kind, r=.0107, z=12):
    # Compensate for the wide figure so nodes are spheres rather than flat discs.
    aspect = ax.figure.get_figwidth() / ax.figure.get_figheight()
    color = np.asarray(matplotlib.colors.to_rgb(NODE[kind]))
    ax.add_patch(Ellipse((x+r*.22, y-r*.35*aspect), r*2.12, r*1.68*aspect,
                         facecolor="#25333d", alpha=.16, edgecolor="none",
                         zorder=z-.1))
    for i in range(16):
        t = i/15
        radius = r*(1-.76*t)
        shade = color*(.65+.35*t)
        shade = shade*(1-.60*t) + .60*t
        ax.add_patch(Ellipse((x-r*.24*t, y+r*.28*aspect*t),
                             radius*2, radius*2*aspect,
                             facecolor=shade, edgecolor="none", zorder=z+i*.01))
    ax.add_patch(Ellipse((x, y), 2*r, 2*r*aspect, facecolor="none",
                         edgecolor="white", linewidth=.7, zorder=z+.2))


def edge(ax, a, b, color="#5b6a73", dashed=False):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
            lw=2.2 if dashed else 1.9,
            linestyle=(0, (4, 3)) if dashed else "solid", zorder=10,
            path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])


def _perspective_coefficients(quad, bounds, source_size, output_size):
    """Map output pixels back to source pixels for Pillow's perspective warp."""
    minx, maxx, miny, maxy = bounds
    sw, sh = source_size
    ow, oh = output_size
    sources = ((0, 0), (sw-1, 0), (sw-1, sh-1), (0, sh-1))
    rows, values = [], []
    for (x, y), (sx, sy) in zip(quad, sources):
        px = (x-minx)/(maxx-minx)*(ow-1)
        py = (maxy-y)/(maxy-miny)*(oh-1)
        rows.append([px, py, 1, 0, 0, 0, -sx*px, -sx*py])
        values.append(sx)
        rows.append([0, 0, 0, px, py, 1, -sy*px, -sy*py])
        values.append(sy)
    return tuple(np.linalg.solve(np.asarray(rows), np.asarray(values)))


def slab(ax, path, quad, label):
    """Place a genuine top-down render on a thick, foreshortened floor plane."""
    im = Image.open(path).convert("RGBA")
    xs, ys = zip(*quad)
    bounds = (min(xs), max(xs), min(ys), max(ys))
    out_size = (850, 560)
    coeffs = _perspective_coefficients(quad, bounds, im.size, out_size)
    warped = im.transform(out_size, Image.PERSPECTIVE, coeffs,
                          resample=Image.BICUBIC,
                          fillcolor=(0, 0, 0, 0))
    far_left, far_right, near_right, near_left = quad
    thickness = .017
    near_left_bottom = (near_left[0], near_left[1]-thickness)
    near_right_bottom = (near_right[0], near_right[1]-thickness)
    ax.add_patch(Polygon((near_left, near_right, near_right_bottom, near_left_bottom),
                         closed=True, facecolor="#354753", edgecolor="#354753",
                         linewidth=1.2, zorder=3))
    ax.add_patch(Polygon((far_right, near_right, near_right_bottom,
                          (far_right[0], far_right[1]-thickness)),
                         closed=True, facecolor="#667984", edgecolor="#354753",
                         linewidth=1.0, zorder=3))
    ax.add_patch(Polygon(quad, closed=True, facecolor="white",
                         edgecolor="#d0d8da", linewidth=1.1, zorder=4))
    ax.imshow(warped, extent=bounds, aspect="auto", interpolation="bilinear",
              zorder=5)
    ax.add_patch(Polygon(quad, closed=True, facecolor="none",
                         edgecolor="#a9b8bd", linewidth=1.3, zorder=6))
    ax.text(near_right[0]-.006, far_right[1]+.017, label,
            ha="right", va="center", fontsize=11.3,
            color=MUTED, fontweight="bold", zorder=7)


def central_graph(ax):
    box(ax, .555, .405, .405, .505, "white", "#9fc9d1", lw=1.8)
    ax.text(.756, .875, "Persistent 3D scene graph  Gₜ", ha="center",
            va="center", fontsize=17.1, fontweight="bold", color=GRAPH, zorder=7)
    upper = FIGURES / "hm3d_topdown_00873_upper.png"
    lower = FIGURES / "hm3d_topdown_00873_lower.png"
    for p in (upper, lower):
        if not p.exists():
            raise FileNotFoundError("missing HM3D render: " + str(p))
    # Two floor planes support the left/right branches of one standing tree.
    slab(ax, upper, ((.616, .575), (.712, .575),
                     (.751, .474), (.584, .474)), "")
    slab(ax, lower, ((.794, .575), (.890, .575),
                     (.929, .474), (.762, .474)), "")

    building = [(.756, .828)]
    storeys = [(.668, .761), (.844, .761)]
    rooms = [(x, .689) for x in (.624, .712, .800, .888)]
    containers = [(x, .616) for x in
                  (.602, .646, .690, .734, .778, .822, .866, .910)]
    objects = [(float(x), .544) for x in np.linspace(.591, .921, 16)]
    levels = (building, storeys, rooms, containers, objects)
    for parents, children in zip(levels[:-1], levels[1:]):
        for i, parent in enumerate(parents):
            for child in children[2*i:2*i+2]:
                edge(ax, parent, child)

    # Stair connectivity stays on the storey level, separate from containment.
    edge(ax, storeys[0], storeys[1], STAIR, dashed=True)
    for i, (x, y) in enumerate(objects):
        ground = .494 + (.008 if i % 2 else 0)
        ax.plot([x, x], [ground, y-.009], color="#647884", alpha=.65,
                lw=.9, zorder=8)
        ax.add_patch(Ellipse((x, ground), .011, .008, facecolor="#24353f",
                             edgecolor="none", alpha=.17, zorder=7))
        ax.add_patch(Ellipse((x, ground+.008), .009, .023,
                             facecolor=NODE["object"], edgecolor=NODE["object"],
                             linewidth=.6, alpha=.30, zorder=8))
    for nodes, kind, radius in zip(
            levels, ("building", "storey", "room", "container", "object"),
            (.010, .0085, .0071, .0057, .0044)):
        for point in nodes:
            sphere(ax, *point, kind, r=radius)
    # Storey names sit beside their spheres, outside all tree edges.
    for x, name, align in ((.651, "story 2", "right"),
                            (.861, "story 1", "left")):
        ax.text(x, .771, name, ha=align, va="center", fontsize=9.5,
                color=MUTED, fontweight="bold", zorder=16)

    # Legend sits below the two images.
    ly = .435
    for kind, x in (("building", .586), ("storey", .652), ("room", .710),
                    ("container", .765), ("object", .849)):
        sphere(ax, x, ly, kind, r=.0060, z=15)
        display_name = "story" if kind == "storey" else kind
        ax.text(x+.009, ly, display_name, ha="left", va="center",
                fontsize=8.1, color=MUTED, zorder=16)
    ax.plot([.889, .902], [ly, ly], color=STAIR, lw=2.0,
            linestyle=(0, (4, 3)), zorder=16)
    ax.text(.905, ly, "stair edge", ha="left", va="center",
            fontsize=8.1, color=MUTED, zorder=16)


def check_text(fig, ax):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = [(a.get_text(), a.get_window_extent(ren)) for a in ax.texts
             if a.get_visible()]
    for i, (name, rect) in enumerate(texts):
        if rect.x0 < ax.bbox.x0-2 or rect.x1 > ax.bbox.x1+2:
            raise RuntimeError("text outside figure: " + repr(name))
        for other, rr in texts[i+1:]:
            dx = min(rect.x1, rr.x1)-max(rect.x0, rr.x0)
            dy = min(rect.y1, rr.y1)-max(rect.y0, rr.y0)
            if dx > 3 and dy > 3:
                raise RuntimeError("overlapping text: " + repr((name, other)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES/"full_pipeline.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 7.25), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    box(ax, .035, .375, .93, .585, "#f3fbfc", "none", z=0)
    box(ax, .035, .045, .93, .295, "#fffbf3", "none", z=0)
    ax.text(.070, .923, "MAPPING + BELIEF REVISION", ha="left", va="center",
            fontsize=11.7, color=GRAPH, fontweight="bold", zorder=3)
    ax.text(.070, .303, "GOAL SELECTION + NAVIGATION", ha="left", va="center",
            fontsize=11.7, color=SEARCH, fontweight="bold", zorder=3)

    box(ax, .070, .598, .130, .178, "white", "#a8bcc3", lw=1.7, z=3)
    ax.text(.135, .729, "System inputs", ha="center", va="center",
            fontsize=14.4, color=INK, fontweight="bold", zorder=7)
    ax.text(.135, .667, "RGB-D + pose\nprior graph  Gₜ₋₁", ha="center",
            va="center", fontsize=11.8, color=INK, linespacing=1.25, zorder=7)
    module(ax, .223, .711, .145, .126, "Object mapping",
           "keyframes → 3D tracks", GRAPH)
    module(ax, .223, .508, .145, .126, "Spatial mapping",
           "depth + pose → geometry", GRAPH)
    module(ax, .389, .711, .150, .126, "Presence revision",
           "tracks + evidence → pᵢ", BELIEF)
    arrow(ax, (.200, .760), (.219, .760))
    route(ax, ((.200, .630), (.210, .630), (.210, .571), (.219, .571)))
    route(ax, ((.135, .598), (.135, .476), (.551, .476)), GRAPH)
    ax.text(.463, .488, "prior graph  Gₜ₋₁", ha="center", va="center",
            fontsize=10.4, color=MUTED, zorder=7)
    arrow(ax, (.368, .774), (.385, .774))
    arrow(ax, (.539, .774), (.552, .774), BELIEF)
    arrow(ax, (.368, .571), (.552, .571), GRAPH)
    central_graph(ax)

    box(ax, .070, .107, .137, .140, "white", "#bdc8cc", lw=1.7, z=3)
    ax.text(.138, .206, "Target query", ha="center", va="center",
            fontsize=15.0, color=INK, fontweight="bold", zorder=7)
    ax.text(.138, .152, "q", ha="center", va="center",
            fontsize=14.0, color=INK, zorder=7)
    module(ax, .412, .102, .195, .146, "Search-goal selection",
           "Gₜ + q → goal g*; floor φ*", SEARCH)
    module(ax, .710, .102, .174, .146, "Multi-floor navigation",
           "goal + map → path / action", NAV)
    box(ax, .907, .120, .040, .111, "#f1f8f3", "#a9d5b8", lw=1.4, z=3)
    ax.text(.927, .176, "aₜ", ha="center", va="center",
            fontsize=14.7, color=NAV, fontweight="bold", zorder=7)
    arrow(ax, (.207, .176), (.408, .176), SEARCH)
    ax.text(.308, .202, "query", ha="center", va="center",
            fontsize=11.7, color=MUTED, zorder=7)
    arrow(ax, (.607, .176), (.706, .176), NAV)
    ax.text(.657, .203, "goal + floor", ha="center", va="center",
            fontsize=11.7, color=MUTED, zorder=7)
    arrow(ax, (.884, .176), (.903, .176), NAV)
    arrow(ax, (.586, .405), (.586, .253), GRAPH)
    ax.text(.482, .356, "revised graph + map", ha="center", va="center",
            fontsize=11.5, color=MUTED, zorder=7)
    route(ax, ((.797, .248), (.797, .360), (.871, .360), (.871, .400)),
          NAV, dashed=True)
    ax.text(.777, .352, "arrival / transition", ha="right", va="center",
            fontsize=10.7, color=NAV, zorder=7)

    check_text(fig, ax)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        path = args.out.with_suffix(ext)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
