#!/usr/bin/env python3
"""Render the simplified multi-floor policy figure for the paper.

HM3D floor textures and the stair guide are real scene assets. The switch
request and the connection between separate floor images are schematic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/figures"
INK = "#26353d"
MUTED = "#5b6d75"
GOLD = "#a67633"
GREEN = "#237f59"
STAIR = "#c37937"
BLUE = "#396eaa"
RATIO = 10.2 / 5.3
LOW_MAP = (.070, .260, .340, .450)
UP_MAP = (.590, .260, .340, .450)


def card(ax, x, y, w, h, *, face="white", edge="#c8d4d7", lw=1.25,
         radius=.012, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def label(ax, x, y, text, *, color=INK, size=10.7, weight="normal",
          ha="center", z=15, box=False):
    bbox = dict(boxstyle="round,pad=0.25,rounding_size=0.18",
                facecolor="white", edgecolor="#d7e0e2", linewidth=.8,
                alpha=.97) if box else None
    ax.text(x, y, text, ha=ha, va="center", fontsize=size,
            color=color, fontweight=weight, linespacing=1.15,
            zorder=z, bbox=bbox)


def route(ax, pts, *, color, lw=2.5, dashed=False, arrow_end=True, z=11):
    if len(pts) < 2:
        return
    body = pts[:-1] if arrow_end and len(pts) > 2 else pts
    line, = ax.plot(*zip(*body), color=color, linewidth=lw,
                    linestyle=(0, (4, 3)) if dashed else "solid",
                    solid_capstyle="round", solid_joinstyle="round", zorder=z)
    line.set_path_effects([pe.Stroke(linewidth=lw+2.2, foreground="white"),
                           pe.Normal()])
    if arrow_end:
        a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                            mutation_scale=15, color=color, linewidth=lw,
                            linestyle=(0, (4, 3)) if dashed else "solid",
                            shrinkA=0, shrinkB=0, zorder=z+1)
        a.set_path_effects([pe.Stroke(linewidth=lw+2.2, foreground="white"),
                            pe.Normal()])
        ax.add_patch(a)


def ellipse_marker(ax, x, y, text, color, *, r=.012, z=16):
    ax.add_patch(Ellipse((x, y), 2*r, 2*r*RATIO,
                         facecolor=color, edgecolor="white",
                         linewidth=1.8, zorder=z))
    label(ax, x, y, text, color="white", size=9.2, weight="bold", z=z+1)


def image_point(meta, u, v, box):
    scale = meta["resolution"] / 800.0
    left, top, right, bottom = meta["crop"]
    x0, y0, w, h = box
    x = x0 + w * ((u*scale-left)/(right-left))
    y = y0 + h * (1-(v*scale-top)/(bottom-top))
    return (x, y)


def floor_image(ax, image, box):
    x, y, w, h = box
    card(ax, x-.006, y-.009, w+.012, h+.018,
         face="white", edge="#cbd7d9", lw=1.0, radius=.007, z=3)
    ax.imshow(image, extent=(x, x+w, y, y+h), origin="upper",
              interpolation="bilinear", aspect="auto", zorder=4)


def stair_marker(ax, point):
    x, y = point
    card(ax, x-.012, y-.023, .024, .046, face="white", edge=STAIR,
         lw=1.4, radius=.003, z=18)
    # A step symbol specifies what the endpoint is; no unexplained empty rings.
    xx = [x-.007, x-.0025, x-.0025, x+.002, x+.002, x+.007]
    yy = [y-.012, y-.012, y, y, y+.012, y+.012]
    ax.plot(xx, yy, color=STAIR, lw=1.6, zorder=19)


def switch_panel(ax, upper, lower, meta, stair):
    card(ax, .035, .035, .930, .940, face="#fcfdfd", edge="#d2dee0",
         lw=1.1, radius=.020, z=0)
    card(ax, .110, .845, .780, .112, face="#fff8ee", edge="#dfc69e",
         lw=1.3, radius=.013, z=2)
    label(ax, .500, .919, "Switch request: score margin met + timing permits",
          size=12.8, weight="bold", color=GOLD)
    label(ax, .500, .875, "Hold while a credible target remains untested on the current floor",
          size=10.9, color=MUTED)

    label(ax, .240, .784, "Current floor", size=14.1, weight="bold")
    label(ax, .760, .784, "Preferred floor", size=14.1, weight="bold", color=GREEN)
    floor_image(ax, lower, LOW_MAP)
    floor_image(ax, upper, UP_MAP)
    label(ax, .240, .213, "Negative inspections lower local support",
          size=10.5, color=MUTED)
    label(ax, .760, .213, "Higher mean score → resume search",
          size=10.5, color=MUTED)

    pts = stair["points_uvy"]
    low_route = [image_point(meta,u,v,LOW_MAP) for u,v,_ in reversed(pts[6:])]
    up_route = [image_point(meta,u,v,UP_MAP) for u,v,_ in reversed(pts[:4])]
    route(ax, low_route, color=GREEN, lw=2.2, arrow_end=False, z=9)
    route(ax, up_route, color=GREEN, lw=2.2, z=9)
    entry, exit_point = low_route[-1], up_route[0]
    approach = (exit_point[0]-.017, exit_point[1]-.002)
    route(ax, [entry, (.450,.624), (.550,.624), approach],
          color=STAIR, lw=2.0, dashed=True, arrow_end=True, z=12)
    # The dashed inter-image connector is a requested crossing, not a known edge.
    label(ax, .500, .681, "Flight first", size=11.1, weight="bold", color=STAIR)
    label(ax, .500, .565, "candidate\ncrossing", size=10.3, color=MUTED)
    for point in (entry, exit_point):
        stair_marker(ax, point)
    ellipse_marker(ax, *low_route[0], "R", BLUE, r=.013, z=20)

    card(ax, .160, .076, .680, .080, face="#f0f8f3", edge="#bdd8c9",
         radius=.010, z=2)
    label(ax, .500, .116, "Crossing completed  →  record StairEdge",
          size=12.0, weight="bold", color=GREEN)


def check_text(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(a.get_text(),a.get_window_extent(renderer))
             for a in ax.texts if a.get_visible()]
    for i,(name,box) in enumerate(texts):
        if box.x0 < ax.bbox.x0-2 or box.x1 > ax.bbox.x1+2:
            raise RuntimeError(f"text outside figure: {name!r}")
        for other,bb in texts[i+1:]:
            dx = min(box.x1,bb.x1)-max(box.x0,bb.x0)
            dy = min(box.y1,bb.y1)-max(box.y0,bb.y0)
            if dx > 4 and dy > 4:
                raise RuntimeError(f"text collision: {name!r} / {other!r}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=ASSETS/"search_floor_decision.png")
    ap.add_argument("--dpi",type=int,default=300)
    args=ap.parse_args()
    meta=json.loads((ASSETS/"hm3d_topdown_00873_meta.json").read_text())
    stair=json.loads((ASSETS/"hm3d_topdown_00873_stair_path.json").read_text())
    upper=plt.imread(ASSETS/"hm3d_topdown_00873_upper.png")
    lower=plt.imread(ASSETS/"hm3d_topdown_00873_lower.png")
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,
                         "svg.fonttype":"none"})
    fig=plt.figure(figsize=(10.2,5.3),facecolor="white")
    ax=fig.add_axes((0,0,1,1))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_axis_off()
    switch_panel(ax,upper,lower,meta,stair)
    check_text(fig,ax)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in (".png",".pdf",".svg"):
        p=args.out.with_suffix(ext)
        fig.savefig(p,dpi=args.dpi,facecolor="white")
        print("wrote",p)
    plt.close(fig)


if __name__=="__main__":
    main()
