#!/usr/bin/env python3
"""Compose the three relocation patterns from actual YCB layout renders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
ASSETS = FIGURES / "ycb_layouts"
DEFAULT_LAYOUT_ROOT = ROOT.parent / "habitat-data-collector/outputs/dualmap_multifloor"
SCENE = "00873-bxsVRursffK"
INK = "#263640"
MUTED = "#62727b"
LINE = "#73838b"
BLUE = "#3875a4"
GREEN = "#16816d"


def label(ax, x, y, value, *, size=11.0, color=INK, weight="normal",
          ha="center", va="center", z=20):
    ax.text(x, y, value, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, linespacing=1.17, zorder=z)


def card(ax, x, y, w, h, *, face="white", edge="#cad6da", radius=.010,
         lw=1.1, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def arrow(ax, a, b, *, color=LINE, lw=1.8, head=12, z=15):
    patch = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head,
                            color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                            zorder=z)
    patch.set_path_effects([pe.Stroke(linewidth=lw+2.1, foreground="white"),
                            pe.Normal()])
    ax.add_patch(patch)


def read_layouts(scene: str, layout_root: Path):
    root = layout_root / scene
    files = {
        "static": root / "static_scene_config.json",
        "in_anchor": root / "dynamic_scene_config/in_anchor/layout_01.json",
        "cross_anchor": root / "dynamic_scene_config/cross_anchor/layout_01.json",
    }
    return {name: json.loads(path.read_text()) for name, path in files.items()}


def object_info(layout, handle):
    sid = next(int(k) for k, value in layout["id_handle_mapping"].items()
               if value == handle)
    item = next(value for value in layout["objects"]
                if int(value["semantic_id"]) == sid)
    return np.asarray(item["translation"], dtype=float), item["anchor"]


class RenderSet:
    def __init__(self, scene: str):
        asset = scene.split("-", 1)[0]
        self.meta = json.loads((ASSETS / f"{asset}_meta.json").read_text())
        self.images = {}
        for condition in ("static", "in_anchor", "cross_anchor"):
            for floor in ("upper", "lower"):
                self.images[(condition, floor)] = plt.imread(
                    ASSETS / f"{asset}_{condition}_{floor}.png")

    def pixel(self, world):
        resolution = float(self.meta["resolution"])
        span = float(self.meta["span_m"])
        cx, cz = self.meta["camera_center_xz"]
        left, top, _, _ = self.meta["crop"]
        return (resolution/2 + (float(world[0])-cx)*resolution/span-left,
                resolution/2 + (float(world[2])-cz)*resolution/span-top)

    def crop(self, fig, box, condition, floor, centre, target, *, span_m=2.25,
             prior=False):
        ax = fig.add_axes(box)
        image = self.images[(condition, floor)]
        ax.imshow(image, origin="upper", interpolation="bilinear")
        cx, cy = self.pixel(centre)
        tx, ty = self.pixel(target)
        radius = span_m * float(self.meta["resolution"]) / float(self.meta["span_m"]) / 2
        ax.set_xlim(cx-radius, cx+radius)
        ax.set_ylim(cy+radius, cy-radius)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#b7c7ce")
            spine.set_linewidth(1.25)
        color = BLUE if prior else GREEN
        ax.plot(tx, ty, marker="o", markersize=17, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=2.8,
                linestyle="none", zorder=20)
        ax.plot(tx, ty, marker="o", markersize=20.5, markerfacecolor="none",
                markeredgecolor="white", markeredgewidth=1.5,
                linestyle="none", zorder=19)


def panel(fig, ax, renders, layouts, *, x, title, handle, changed,
          before_floor, after_floor, description, span=2.25,
          common_view=False):
    width = .304
    card(ax, x, .290, width, .650, face="#fbfcfc", edge="#d0dcdf",
         radius=.012, lw=1.05)
    label(ax, x+.018, .897, title, size=15.0, weight="bold", ha="left")
    pretty = handle.split("_", 1)[-1].replace("_", " ")
    label(ax, x+.018, .850, pretty, size=10.5, color=MUTED, ha="left")

    before, _ = object_info(layouts["static"], handle)
    after, _ = object_info(layouts[changed], handle)
    centre_before = (before+after)/2 if common_view else before
    centre_after = (before+after)/2 if common_view else after
    left_box = (x+.015, .444, .124, .318)
    right_box = (x+.165, .444, .124, .318)
    renders.crop(fig, left_box, "static", before_floor, centre_before, before,
                 span_m=span, prior=True)
    renders.crop(fig, right_box, changed, after_floor, centre_after, after,
                 span_m=span, prior=False)
    label(ax, x+.077, .790, "prior layout", size=9.2, color=BLUE, weight="bold")
    label(ax, x+.227, .790, "relocated layout", size=9.2, color=GREEN,
          weight="bold")
    arrow(ax, (x+.141, .603), (x+.162, .603), head=10, lw=1.6)
    distance = float(np.linalg.norm(after-before))
    label(ax, x+width/2, .365,
          f"{description}\n{distance:.2f} m relocation",
          size=10.6, color=INK)


def protocol(ax):
    ax.plot([.025, .975], [.240, .240], color="#c7d2d6", linewidth=1.05)
    steps = [(.053, "1", "Explore and map"),
             (.385, "2", "Relocate YCB objects"),
             (.716, "3", "Load prior map and search")]
    for x, number, value in steps:
        label(ax, x, .167, number, size=16.0, color=BLUE, weight="bold", ha="left")
        label(ax, x+.038, .167, value, size=12.0, ha="left")
    arrow(ax, (.291, .167), (.350, .167), head=10, lw=1.5)
    arrow(ax, (.621, .167), (.680, .167), head=10, lw=1.5)
    ax.plot(.290, .073, marker="o", markersize=12, markerfacecolor="none",
            markeredgecolor=BLUE, markeredgewidth=2.3, linestyle="none")
    label(ax, .310, .073, "mapped target pose", size=9.8, color=MUTED, ha="left")
    ax.plot(.515, .073, marker="o", markersize=12, markerfacecolor="none",
            markeredgecolor=GREEN, markeredgewidth=2.3, linestyle="none")
    label(ax, .535, .073, "target in changed layout", size=9.8,
          color=MUTED, ha="left")


def render(out_dir: Path, dpi: int, scene: str, layout_root: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    layouts = read_layouts(scene, layout_root)
    renders = RenderSet(scene)
    fig = plt.figure(figsize=(14.2, 5.8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(fig, ax, renders, layouts, x=.025, title="(a)  In-anchor",
          handle="005_tomato_soup_can", changed="in_anchor",
          before_floor="lower", after_floor="lower",
          description="same dining table · same floor", span=1.85,
          common_view=True)
    panel(fig, ax, renders, layouts, x=.348, title="(b)  Cross-anchor",
          handle="019_pitcher_base", changed="cross_anchor",
          before_floor="lower", after_floor="lower",
          description="kitchen island → dining table · same floor", span=2.15)
    panel(fig, ax, renders, layouts, x=.671, title="(c)  Cross-floor",
          handle="003_cracker_box", changed="cross_anchor",
          before_floor="upper", after_floor="lower",
          description="table (Floor 2) → kitchen island (Floor 1)", span=2.15)
    protocol(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"relocation_modes.{suffix}", dpi=dpi,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    args = parser.parse_args()
    render(args.out_dir, args.dpi, args.scene, args.layout_root)
