#!/usr/bin/env python3
"""Render the floor-score formula figure for the paper (companion to
`render_search_floor_decision.py`, same visual language).

The figure visualises one equation:

    M_f = mean over the floor's eligible surfaces of b(x) * lam(x) * d(x)

and the rule that acts on it: take the argmax floor, and accept a switch only
when it beats the current floor by a margin (`exploration.floor_mass_margin`).

Nothing is schematic except the wording. The bars and the curve are computed by
the pipeline's own code -- `build_container_candidates` and `InspectionLog`,
the same functions `ExplorationStrategy._select_surface` calls -- on a saved
two-storey map, with the target's own mapped track as the last believed
position. The inspection sequence is the real rule (lam <- lam (1-d)) applied
to the best remaining surface on the starting floor, which is what the agent
does while it still prefers that floor.

    python scripts/render_floor_score_formula.py
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osg.exploration.search_belief import build_container_candidates, InspectionLog

ASSETS = ROOT / "docs/figures"
SCENE = "00873-bxsVRursffK"          # the scene of search_floor_decision.png
MAPS = ROOT / "outputs/maps_p1500_osg"
TARGET = "tin can"

# The configuration the multi-floor arm runs (configs/experiment/
# osg_unified_pipeline.yaml + core/config/exploration.py).
DETECT_PROB = 0.8                    # d(x), exploration.search_detect_prob
SURFACE_MASS = 1.0                   # exploration.search_surface_mass
PROX_LEN_M = 1.0                     # exploration.search_proximity_len_m
MARGIN = 1.15                        # exploration.floor_mass_margin
RULE = "mean"                        # exploration.floor_mass_rule

INK = "#26353d"
MUTED = "#5b6d75"
GOLD = "#a67633"
GREEN = "#237f59"
STAIR = "#c37937"
BLUE = "#396eaa"


# --------------------------------------------------------------- the numbers

def load_graph(scene: str, maps: Path):
    """The saved two-storey map, with its containers rebuilt per storey.

    Reuses `render_scene_graph_multifloor.load_multifloor` rather than copying
    it: that loader carries the per-storey container height band, without which
    every surface on an upper floor fails the absolute band and the score has
    nothing to average.
    """
    spec = importlib.util.spec_from_file_location(
        "mfloor", Path(__file__).with_name("render_scene_graph_multifloor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_multifloor(scene, maps)


def anchor_of(layer, target: str) -> np.ndarray:
    """Where the map last believed the target to be.

    `ExplorationStrategy._last_known_target_xy` takes the target-labelled track
    with the most expected observations; on a map just reloaded that counter is
    empty, so the most-observed track is the same choice by the same argument.
    """
    tracks = [t for t in layer.tracks(include_blacklisted=True)
              if str(t.label).lower() == target]
    if not tracks:
        raise SystemExit(f"the saved map holds no {target!r} track")
    best = max(tracks, key=lambda t: int(getattr(t, "n_obs", 0)))
    return np.asarray(layer.center_of(best), dtype=float)


def per_floor(graph, target, log, anchor_xy):
    """Every eligible surface, grouped by storey, with its b*lam*d."""
    cands = build_container_candidates(
        graph, target, log, detect_prob=DETECT_PROB, surface_mass=SURFACE_MASS,
        last_known_xy=anchor_xy, proximity_len_m=PROX_LEN_M,
        proximity_floor=0.0)
    out = {}
    for cand in cands:
        out.setdefault(int(cand.floor_key), []).append(cand)
    return out


def mean_of(cands) -> float:
    values = [c.prior * c.detect_prob for c in cands]
    return float(np.mean(values)) if values else 0.0


def sweep(graph, target, anchor_xy, here: int, other: int, n: int = 24):
    """Inspect the best remaining surface on `here`, step by step.

    Returns the per-floor score after each inspection, plus the surfaces that
    were inspected in order -- the real decay rule, not a fitted curve.
    """
    log = InspectionLog()
    means, inspected = [], []
    for k in range(n + 1):
        groups = per_floor(graph, target, log, anchor_xy)
        means.append((mean_of(groups.get(here, [])), mean_of(groups.get(other, []))))
        if k == 0:
            first = groups
        if k < n:
            pick = max(groups.get(here, []), key=lambda c: c.prior * c.detect_prob)
            log.searched(pick.ref_id, DETECT_PROB)
            inspected.append((int(pick.ref_id), str(pick.label)))
    return first, np.asarray(means), inspected


# --------------------------------------------------------------- the drawing

def card(ax, x, y, w, h, *, face="white", edge="#c8d4d7", lw=1.2, radius=.012, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                zorder=z))


def label(ax, x, y, text, *, color=INK, size=10.6, weight="normal",
          ha="center", z=15):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.25, zorder=z)


def bar_axes(fig, rect, cands, *, color, name, count_note, show_y):
    ax = fig.add_axes(rect)
    values = np.sort(np.array([c.prior * c.detect_prob for c in cands]))[::-1]
    ax.bar(np.arange(len(values)), values, width=.85, color=color, linewidth=0)
    mean = float(values.mean())
    ax.axhline(mean, color=color, lw=1.4, ls=(0, (4, 3)))
    ax.text(len(values) * .30, mean * 1.05, f"mean {mean:.3f}", color=color,
            fontsize=9.8, fontweight="bold", va="bottom", ha="left")
    ax.set_xlim(-1, len(values))
    ax.set_xlabel(f"{name}\n{count_note}", fontsize=9.8, color=color,
                  fontweight="bold", labelpad=6)
    ax.tick_params(labelsize=8.4, colors=MUTED, length=3)
    ax.set_xticks([])
    if show_y:
        ax.set_ylabel("$\\bar b(x)\\,\\lambda(x)\\,d(x)$", fontsize=10.4,
                      color=INK, labelpad=4)
    else:
        ax.set_yticklabels([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8d4d7")
    return ax, mean


def curve_axes(fig, rect, means, *, accept_k, cross_k):
    ax = fig.add_axes(rect)
    here, other = means[:, 0], means[:, 1]
    k = np.arange(len(here))
    ax.plot(k, here, color=BLUE, lw=2.4, label="current floor  $M_{\\varphi}$")
    ax.plot(k, other, color=GREEN, lw=2.4, label="other floor  $M_{\\varphi'}$")
    ax.plot(k, other / MARGIN, color=GOLD, lw=1.5, ls=(0, (4, 3)),
            label=f"what the current floor must beat\n($M_{{\\varphi'}}/{MARGIN:g}$)")
    if accept_k is not None:
        ax.axvline(accept_k, color=GOLD, lw=1.1, alpha=.55)
        ax.plot([accept_k], [other[accept_k]], marker="*", markersize=15,
                color=GOLD, markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        ax.annotate(f"switch accepted\nafter {accept_k} inspections",
                    xy=(accept_k, other[accept_k]),
                    xytext=(accept_k + 1.6, other[accept_k] * .42),
                    fontsize=9.4, color=GOLD, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4,
                                    shrinkA=1, shrinkB=4))
    if cross_k is not None:
        ax.annotate("scores cross", xy=(cross_k, here[cross_k]),
                    xytext=(cross_k - 7.4, here[cross_k] * .35), fontsize=9.2,
                    color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                                 lw=1.2, shrinkA=1, shrinkB=3))
    ax.set_xlabel("negative inspections on the current floor", fontsize=9.8,
                  color=MUTED, labelpad=5)
    ax.text(.035, .045, "$\\lambda(x)\\leftarrow\\lambda(x)\\,(1-d(x))$ per look",
            transform=ax.transAxes, fontsize=9.6, color=MUTED, va="bottom")
    ax.set_ylabel("floor score  $M_\\varphi$", fontsize=10.4, color=INK,
                  labelpad=4)
    ax.set_xlim(0, len(here) - 1)
    ax.set_ylim(0, float(max(here.max(), other.max())) * 1.18)
    ax.tick_params(labelsize=8.6, colors=MUTED, length=3)
    ax.legend(fontsize=9.0, frameon=False, loc="upper right",
              labelcolor="linecolor", handlelength=1.5, borderaxespad=.1,
              labelspacing=.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8d4d7")
    return ax


def chain(ax, y, items):
    """The gates, in the order they are applied."""
    widths = [.022 + .0063 * len(t) for t in items]
    total = sum(widths) + .026 * (len(items) - 1)
    x = (1.0 - total) / 2
    for i, (text, w) in enumerate(zip(items, widths)):
        card(ax, x, y - .032, w, .064, face="#f6faf8", edge="#c3d8cc",
             radius=.008, z=3)
        label(ax, x + w / 2, y, text, size=8.6, color=INK, z=6)
        x += w
        if i < len(items) - 1:
            ax.add_patch(FancyArrowPatch((x + .004, y), (x + .022, y),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=GREEN, lw=1.3, zorder=5))
            x += .026


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ASSETS / "floor_score_formula.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    blob, grids, costmaps, layer, graph, floors = load_graph(SCENE, MAPS)
    anchor3 = anchor_of(layer, TARGET)
    anchor_xy = anchor3[[0, 2]]
    heights = {int(f["key"]): float(f["height_y"]) for f in floors}
    # The agent searches the floor its own score prefers, so that is the floor
    # the inspections happen on. Choosing it by the argmax rather than by hand
    # keeps the figure from picking the flattering storey.
    start = per_floor(graph, TARGET, InspectionLog(), anchor_xy)
    here = max(start, key=lambda k: mean_of(start[k]))
    other = max((k for k in heights if k != here),
                key=lambda k: mean_of(start.get(k, [])))

    first, means, inspected = sweep(graph, TARGET, anchor_xy, here, other)
    accept = np.nonzero(means[:, 1] >= MARGIN * means[:, 0])[0]
    cross = np.nonzero(means[:, 1] >= means[:, 0])[0]
    accept_k = int(accept[0]) if len(accept) else None
    cross_k = int(cross[0]) if len(cross) else None

    n_here, n_other = len(first[here]), len(first[other])
    print(f"{SCENE}  target={TARGET!r}  searching phi{here} (y={heights[here]:.2f}), "
          f"anchor y={float(anchor3[1]):.2f}")
    print(f"  |S| = {n_here} / {n_other};  M = {means[0,0]:.4f} / {means[0,1]:.4f}")
    print(f"  scores cross at {cross_k}, margin {MARGIN:g}x cleared at {accept_k}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(11.0, 5.9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    card(ax, .020, .020, .960, .960, face="#fcfdfd", edge="#d2dee0",
         lw=1.1, radius=.018, z=0)

    # ------------------------------------------------------------- header
    card(ax, .060, .790, .880, .168, face="#fff8ee", edge="#dfc69e",
         lw=1.3, radius=.012, z=2)
    label(ax, .500, .907,
          "$M_\\varphi=\\dfrac{1}{|\\mathcal{S}_\\varphi|}"
          "\\sum_{x\\in\\mathcal{S}_\\varphi}\\bar b(x)\\,\\lambda(x)\\,d(x)$"
          "$\\qquad\\varphi^{\\star}=\\arg\\max_\\varphi M_\\varphi$"
          f"$\\qquad$accept a switch only if $M_{{\\varphi^{{\\star}}}}\\geq "
          f"{MARGIN:g}\\,M_{{\\varphi_{{\\mathrm{{now}}}}}}$",
          size=14.6, color=INK)
    label(ax, .198, .830, "$\\bar b(x)$  prior for the surface", size=10.0,
          color=GOLD, weight="bold")
    label(ax, .500, .830, "$\\lambda(x)$  what earlier inspections left",
          size=10.0, color=STAIR, weight="bold")
    label(ax, .805, .830, "$d(x)$  chance one inspection finds it", size=10.0,
          color=MUTED, weight="bold")

    # ------------------------------------------------- left: the surfaces
    label(ax, .262, .750, "One value per surface, averaged per floor",
          size=12.4, weight="bold")
    span = .295
    w_here = span * n_here / (n_here + n_other)
    ax_l, _ = bar_axes(fig, [.088, .330, w_here, .390], first[here], color=BLUE,
             name=f"floor $\\varphi{here}$ \u2014 the score's first choice",
             count_note=f"$|\\mathcal{{S}}|$ = {n_here} surfaces", show_y=True)
    ax_r, _ = bar_axes(fig, [.098 + w_here, .330, span - w_here, .390],
                       first[other], color=GREEN, name=f"floor $\\varphi{other}$",
                       count_note=f"$|\\mathcal{{S}}|$ = {n_other}", show_y=False)
    top = max(ax_l.get_ylim()[1], ax_r.get_ylim()[1])
    ax_l.set_ylim(0, top)
    ax_r.set_ylim(0, top)

    # ------------------------------------------------- right: the decision
    label(ax, .730, .750, "Every negative inspection lowers this floor's score",
          size=12.4, weight="bold")
    curve_axes(fig, [.560, .330, .385, .390], means, accept_k=accept_k,
               cross_k=cross_k)

    # ------------------------------------------------------------- footer
    chain(ax, .135, ["argmax over floors with surfaces",
                     f"beat here by {MARGIN:g}×",
                     "hold if a believed track here is untested",
                     "route: flight ▸ stairs track ▸ portal"])

    fig.canvas.draw()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        path = args.out.with_suffix(ext)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
