#!/usr/bin/env python3
"""Render visibility-gated and multi-sensor presence belief for the Method.

The rays and track trajectory are schematic. The depth ordering, evidence
channels and arrival alternatives follow PresenceFilter and AbsenceSensor.
The worked log-odds steps use current default reliability values from
src/osg/core/config/scene_graph.py and verification.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/presence_belief"
INK = "#23333d"
MUTED = "#61727c"
EXPECTED = "#728da0"
OCCLUDER = "#d98b44"
MISS = "#bb4b46"
SIGHTING = "#198873"
BACKGROUND = "#c2ccd0"
VLM = "#7560a7"


def cup(ax, x: float, *, visible: bool) -> None:
    color = SIGHTING if visible else EXPECTED
    face = "#d9f1e9" if visible else "none"
    dash = "solid" if visible else (0, (3, 2))
    ax.add_patch(
        Polygon(
            [(x - 0.34, 3.62), (x + 0.34, 3.62),
             (x + 0.26, 2.52), (x - 0.26, 2.52)],
            closed=True, facecolor=face, edgecolor=color,
            linewidth=2.2, linestyle=dash, zorder=8,
        )
    )
    ax.add_patch(
        Circle(
            (x + 0.42, 3.21), 0.23, fill=False, edgecolor=color,
            linewidth=2.2, linestyle=dash, zorder=8,
        )
    )
    ax.plot([x + 0.27, x + 0.36], [3.43, 3.43],
            color=color, linewidth=2.0, linestyle=dash, zorder=8)
    ax.plot([x + 0.25, x + 0.37], [2.99, 2.99],
            color=color, linewidth=2.0, linestyle=dash, zorder=8)


def base(ax, title: str, *, visible: bool) -> None:
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.08), 7.88, 4.81,
            boxstyle="round,pad=0.015,rounding_size=0.12",
            facecolor="white", edgecolor="#d9e0e3", linewidth=1.3,
            zorder=0,
        )
    )
    ax.text(0.35, 4.53, title, fontsize=18.0, color=INK,
            fontweight="bold", va="center")
    # The faint view cone and central ray indicate the same camera view in
    # every panel. They are not a measured trajectory or a depth sample cloud.
    ax.add_patch(
        Polygon(
            [(1.03, 3.05), (7.20, 3.92), (7.20, 2.15)],
            closed=True, facecolor="#f3f7f8", edgecolor="none", zorder=1,
        )
    )
    ax.add_patch(Circle((0.93, 3.05), 0.29, facecolor=INK,
                        edgecolor="white", linewidth=1.5, zorder=10))
    ax.add_patch(Polygon([(1.14, 3.21), (1.43, 3.05), (1.14, 2.89)],
                         facecolor=INK, edgecolor="none", zorder=10))
    ax.text(0.48, 2.35, "camera", fontsize=13.0, color=MUTED)
    ax.plot([4.10, 5.85], [2.42, 2.42], color="#8c9aa0",
            linewidth=2.6, zorder=5)
    cup(ax, 4.86, visible=visible)
    if not visible:
        ax.text(4.86, 3.76, "mapped pose", fontsize=13.0,
                color=EXPECTED, ha="center", zorder=9)
    else:
        ax.text(4.86, 3.87, "matched sighting", fontsize=13.0,
                color=SIGHTING, ha="center", zorder=9)


def depth_axis(ax, measured: float, measured_color: str, compare: str) -> None:
    # Depth increases along this horizontal axis. The shaded interval is the
    # expected object; only the near side can veto an informative miss.
    ax.plot([1.08, 7.25], [1.38, 1.38], color="#71828a",
            linewidth=1.2, zorder=2)
    ax.annotate("", (7.31, 1.38), (7.06, 1.38),
                arrowprops={"arrowstyle": "-|>", "color": "#71828a",
                            "lw": 1.2})
    ax.text(7.32, 1.16, "depth", fontsize=12.0, color=MUTED, ha="right")
    ax.add_patch(Rectangle((4.27, 1.14), 1.21, 0.48,
                           facecolor="#dce8ee", edgecolor="none",
                           alpha=0.86, zorder=1))
    ax.plot([4.27, 4.27], [0.96, 1.72], color=EXPECTED,
            linewidth=1.6, linestyle=(0, (3, 2)), zorder=3)
    ax.text(4.27, 0.85, r"$z_{\rm near}$", fontsize=13.0,
            color=EXPECTED, ha="center", zorder=3)
    ax.text(4.88, 1.78, "expected object", fontsize=12.0,
            color=EXPECTED, ha="center", zorder=3)
    ax.plot([measured, measured], [1.07, 1.69], color=measured_color,
            linewidth=2.6, zorder=4)
    ax.plot([measured], [1.38], marker="o", markersize=8.0,
            color=measured_color, markeredgecolor="white",
            markeredgewidth=1.2, zorder=6)
    ax.text(measured, 2.03, compare, fontsize=13.0,
            color=measured_color, ha="center", fontweight="bold", zorder=5)


def evidence(ax, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.49, 0.19), 7.02, 0.48,
            boxstyle="round,pad=0.025,rounding_size=0.11",
            facecolor=color, edgecolor="none", alpha=0.12, zorder=3,
        )
    )
    ax.text(4.0, 0.43, text, fontsize=17.0, color=color,
            fontweight="bold", ha="center", va="center", zorder=4)


def occluded(ax) -> None:
    base(ax, "(a) Occluded pose", visible=False)
    ax.plot([1.43, 3.12], [3.05, 3.05], color=OCCLUDER,
            linewidth=2.1, zorder=7)
    ax.plot([3.48, 4.52], [3.05, 3.05], color="#a8b5bc",
            linewidth=1.5, linestyle=(0, (3, 3)), zorder=3)
    ax.add_patch(Rectangle((3.10, 2.39), 0.40, 1.37,
                           facecolor=OCCLUDER, edgecolor="#ad6631",
                           linewidth=1.1, zorder=9))
    ax.text(3.30, 4.10, "occluder", fontsize=13.0,
            color=OCCLUDER, ha="center")
    depth_axis(ax, 3.28, OCCLUDER, "nearer")
    evidence(ax, r"$E_i=0,\ Z_i=0:\quad \Delta L_i=0$", OCCLUDER)


def removed(ax) -> None:
    base(ax, "(b) Removed object", visible=False)
    ax.plot([1.43, 6.78], [3.05, 3.05], color=MISS,
            linewidth=2.1, zorder=6)
    ax.add_patch(Rectangle((6.72, 2.42), 0.43, 1.44,
                           facecolor=BACKGROUND, edgecolor="#8c9ca5",
                           linewidth=1.1, zorder=7))
    ax.text(6.94, 4.10, "background", fontsize=13.0,
            color=MUTED, ha="center")
    depth_axis(ax, 6.86, MISS, "farther")
    evidence(ax, r"$E_i=1,\ Z_i=0:\quad \Delta L_i<0$", MISS)


def redetected(ax) -> None:
    base(ax, "(c) Re-detected object", visible=True)
    ax.plot([1.43, 4.58], [3.05, 3.05], color=SIGHTING,
            linewidth=2.1, zorder=6)
    depth_axis(ax, 4.88, SIGHTING, "object depth")
    evidence(ax, r"$Z_i=1:\quad \Delta L_i>0$", SIGHTING)


def card(ax, x: float, y: float, w: float, title: str, detail: str,
         color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, 0.98,
            boxstyle="round,pad=0.025,rounding_size=0.11",
            facecolor="white", edgecolor=color, linewidth=1.8, zorder=3,
        )
    )
    ax.text(x + 0.17, y + 0.78, title, fontsize=14.0, color=color,
            fontweight="bold", va="center", zorder=4)
    ax.text(x + 0.17, y + 0.30, detail, fontsize=13.0, color=INK,
            va="center", linespacing=1.10, zorder=4)


def arrow(ax, start, end, color: str, *, dashed: bool = False) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": "-|>", "color": color, "lw": 2.2,
            "linestyle": "dashed" if dashed else "solid",
            "mutation_scale": 14,
        },
        zorder=2,
    )


def sensor_panel(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.08), 9.88, 4.81,
            boxstyle="round,pad=0.015,rounding_size=0.12",
            facecolor="white", edgecolor="#d9e0e3", linewidth=1.3,
            zorder=0,
        )
    )
    ax.text(0.36, 4.53, "(d) Multiple observers, one belief",
            fontsize=18.0, color=INK, fontweight="bold", va="center")
    card(ax, 0.43, 3.12, 3.85, "Keyframe detector",
         "clear miss; E=1\n(r=.60, q=.05)", MISS)
    card(ax, 0.43, 1.86, 3.85, "Arrival VLM crop",
         "cup judged absent\n(r=.90, q=.20)", VLM)
    card(ax, 0.43, 0.60, 3.85, "Detector scan fallback",
         "clear site; detector silent\n(r=.80, q=.05)", OCCLUDER)
    ax.add_patch(
        FancyBboxPatch(
            (6.48, 1.83), 2.94, 1.30,
            boxstyle="round,pad=0.025,rounding_size=0.14",
            facecolor="#eff5f6", edgecolor=EXPECTED, linewidth=2.0,
            zorder=4,
        )
    )
    ax.text(7.95, 2.67, r"same $\Delta L_i(r,q)$",
            fontsize=17.0, color=INK, ha="center",
            fontweight="bold", zorder=5)
    ax.text(7.95, 2.14, "one bounded state",
            fontsize=13.0, color=MUTED, ha="center", zorder=5)
    arrow(ax, (4.30, 3.60), (6.43, 2.91), MISS)
    arrow(ax, (4.30, 2.34), (6.43, 2.55), VLM)
    arrow(ax, (4.30, 1.08), (6.43, 2.02), OCCLUDER, dashed=True)
    ax.text(5.22, 0.30, "One arrival reading: VLM or fallback",
            fontsize=13.0, color=MUTED, ha="center", zorder=5)


def belief_panel(ax) -> None:
    # Illustrative arithmetic from the configured default sensor parameters.
    # An arrival uses the VLM reading OR the detector fallback, never both.
    miss = math.log((1.0 - 0.60) / (1.0 - 0.05))
    vlm_no = math.log((1.0 - 0.90) / (1.0 - 0.20))
    fallback_no = math.log((1.0 - 0.80) / (1.0 - 0.05))
    sighting = math.log(0.60 / 0.05)
    levels = [1.50, 1.50, 1.50 + miss,
              1.50 + miss + vlm_no,
              1.50 + miss + vlm_no + sighting]
    fallback_level = levels[2] + fallback_no

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#d9e0e3")
        spine.set_linewidth(1.3)
    ax.set_xlim(-0.25, 4.35)
    ax.set_ylim(-2.2, 4.0)
    ax.text(0.02, 0.95, "(e) Weighted updates of one track",
            transform=ax.transAxes, fontsize=18.0, color=INK,
            fontweight="bold", va="top", zorder=10)
    ax.set_xticks(range(5), ["mapped", "occluded\nframe",
                             "clear\nmiss", "arrival\nVLM absence",
                             "re-sighted"])
    ax.tick_params(axis="both", labelsize=12.0, colors=MUTED, length=0)
    ax.text(-0.16, 2.60, r"log-odds $L_i$",
            fontsize=11.8, color=MUTED, ha="left", va="center", zorder=10)
    ax.grid(axis="y", color="#e6ecee", linewidth=0.9)
    ax.axhline(3.0, color=EXPECTED, linestyle=(0, (5, 3)),
               linewidth=1.4, zorder=1)
    ax.text(4.29, 3.02, "positive cap +3",
            fontsize=11.0, color=EXPECTED, ha="right", va="bottom")
    gate = math.log(0.45 / 0.55)
    ax.axhline(gate, color="#b4bcc0", linestyle=(0, (3, 3)),
               linewidth=1.3, zorder=1)
    ax.text(0.08, gate + 0.05, r"candidate gate $p_i=0.45$",
            fontsize=10.9, color=MUTED, ha="left", va="bottom")

    colors = [OCCLUDER, MISS, VLM, SIGHTING]
    for j, color in enumerate(colors):
        ax.plot([j, j + 1], levels[j:j + 2],
                color=color, linewidth=2.8, zorder=4)
    ax.scatter(range(5), levels, s=100,
               color=[INK, OCCLUDER, MISS, VLM, SIGHTING],
               edgecolors="white", linewidths=1.6, zorder=6)
    ax.plot([2, 3], [levels[2], fallback_level],
            color=OCCLUDER, linewidth=2.0,
            linestyle=(0, (4, 3)), zorder=3)
    ax.scatter([3], [fallback_level], s=95,
               facecolors="white", edgecolors=OCCLUDER,
               linewidths=2.0, zorder=6)
    # A swatch in open space identifies the alternative dashed branch. Its
    # label no longer crosses the rising green re-sighting segment.
    ax.plot([2.62, 2.94], [2.36, 2.36], color=OCCLUDER,
            linewidth=2.2, linestyle=(0, (4, 3)), zorder=7)
    ax.text(3.02, 2.36, "detector fallback\n−1.56 (alternative)",
            fontsize=10.9, color=OCCLUDER, ha="left", va="center", zorder=7)
    ax.text(2.66, -1.70, "VLM: cup absent (−2.08)",
            fontsize=10.5, color=VLM, fontweight="bold",
            ha="left", va="center", zorder=7)
    ax.text(0.10, 1.71, "initial 1.50", fontsize=11.0, color=INK)
    ax.text(1.18, 1.65, "0", fontsize=11.4, color=OCCLUDER,
            fontweight="bold")
    ax.text(1.99, 1.06, "−0.87", fontsize=11.4, color=MISS,
            fontweight="bold")
    ax.text(3.92, 1.21, "+2.48", fontsize=11.4, color=SIGHTING,
            fontweight="bold", ha="right")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14.7, 8.1), facecolor="white")
    gs = fig.add_gridspec(
        2, 6, height_ratios=[1.05, 0.94],
        left=0.016, right=0.984, top=0.975, bottom=0.10,
        hspace=0.18, wspace=0.09,
    )
    occluded(fig.add_subplot(gs[0, 0:2]))
    removed(fig.add_subplot(gs[0, 2:4]))
    redetected(fig.add_subplot(gs[0, 4:6]))
    sensor_panel(fig.add_subplot(gs[1, 0:3]))
    belief_ax = fig.add_subplot(gs[1, 3:6])
    # Leave enough room for the chart's signed y ticks beside panel (d).
    box = belief_ax.get_position()
    belief_ax.set_position([box.x0 + 0.018, box.y0,
                            box.width - 0.018, box.height])
    belief_panel(belief_ax)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        target = OUT.with_suffix(suffix)
        fig.savefig(target, dpi=300, facecolor="white")
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
