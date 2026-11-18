#!/usr/bin/env python3
"""Publication figure: a real HM3D staircase and the geometry read from it.

The orthographic staircase section is from Habitat-Sim. The map panels use one
saved OSG map of the same scene. The matched flight, portal, and StairEdge are
computed/read from that map, not placed by hand. The photo arrows identify the
visible staircase; they are annotations rather than an episode trajectory.

    python3 scripts/render_stair_evidence.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osg.mapping.costmap import Costmap2D, FREE, OCCUPIED
from osg.mapping.portals import find_portals
from osg.mapping.stairs import find_flights


SCENE = "00800-TEEsavR23oF"
MAPS = ROOT / "outputs/maps_try_steps"
SECTION = ROOT / "outputs/figures/side_00800-TEEsavR23oF_2800_0.50.npz"
INK = "#1f2b33"
MUTED = "#5d6b74"
ORANGE = "#e59029"
TEAL = "#126c86"
FOOT = "#bd3f3c"
TOP = "#0d8798"
PURPLE = "#7660a9"
LIGHT = "#eef3f4"


def load_map(scene: str, root: Path):
    path = root / f"{scene}.json"
    if not path.exists():
        path = root / scene / f"{scene}.json"
    if not path.exists():
        raise SystemExit(f"missing saved map: {path}")
    blob = json.loads(path.read_text())
    arrays = np.load(path.with_suffix(".npz"))
    floors = sorted(blob.get("floors", []), key=lambda f: float(f["height_y"]))
    if len(floors) < 2:
        raise SystemExit("the selected map needs two saved storeys")
    lower, upper = floors[0], floors[1]
    prefix = lower["prefix"]
    for suffix in ("grid", "origin", "height", "stair_mask"):
        if prefix + suffix not in arrays:
            raise SystemExit(f"map is missing {prefix + suffix}")
    grid = arrays[prefix + "grid"]
    cm = Costmap2D(resolution=float(lower["resolution"]),
                   size_m=grid.shape[0] * float(lower["resolution"]),
                   track_height=True)
    cm.grid = grid
    cm.origin = np.asarray(arrays[prefix + "origin"], dtype=float)
    cm.height = arrays[prefix + "height"]
    cm.stair_mask = arrays[prefix + "stair_mask"].astype(bool)
    return blob, cm, lower, upper


def select_evidence(blob, cm, lower, upper):
    gap = float(upper["height_y"] - lower["height_y"])
    flights = [f for f in find_flights(cm, float(lower["height_y"]),
                                      new_level_m=gap, min_span_m=1.0)
               if f.kind == "up"]
    # A completed transition may have been traversed in either direction.
    floor_keys = {int(lower["key"]), int(upper["key"])}
    edges = [e for e in blob.get("connectivity", [])
             if {int(e["from_floor"]), int(e["to_floor"])} == floor_keys]
    if not flights or not edges:
        raise SystemExit("no upward flight or completed storey edge in this map")
    edge = edges[0]
    entry = np.asarray(edge["entry_xy"], dtype=float)

    def score(flight):
        rc = flight.cells_rc
        overlap = int(cm.stair_mask[rc[:, 0], rc[:, 1]].sum())
        return overlap, -float(np.linalg.norm(flight.foot_xy - entry))

    flight = max(flights, key=score)
    overlap = score(flight)[0]
    if overlap < 100:
        raise SystemExit("the selected flight does not overlap the stored stair mask")
    portals = [p for p in find_portals(cm, float(lower["height_y"]))
               if p.delta_y > 0]
    portal = min(portals, key=lambda p: np.linalg.norm(p.centroid_xy - flight.foot_xy))
    return flight, portal, edge, overlap


def crop_bounds(cm, flight, portal, edge, pad=22):
    pts = np.vstack([flight.cells_rc,
                     cm.world_to_grid(np.asarray(portal.centroid_xy))[None, :],
                     cm.world_to_grid(np.asarray(edge["entry_xy"]))[None, :]])
    lo = np.maximum(pts.min(axis=0) - pad, 0)
    hi = np.minimum(pts.max(axis=0) + pad + 1, cm.grid.shape)
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def map_base(cm, sl):
    grid = cm.grid[sl]
    rgb = np.empty(grid.shape + (3,), float)
    rgb[:] = (0.81, 0.84, 0.85)
    rgb[grid == FREE] = (0.97, 0.97, 0.95)
    rgb[grid == OCCUPIED] = (0.20, 0.25, 0.28)
    return rgb


def show_grid(ax, rgb, *, title, subtitle):
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
              interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12.2, color=INK, fontweight="bold", pad=8)
    ax.set_xlabel(subtitle, fontsize=9.0, color=MUTED, labelpad=8)
    for spine in ax.spines.values():
        spine.set_color("#c7d0d4")
        spine.set_linewidth(1.0)


def marker_at(ax, cm, xy, bounds, *, label, color, symbol="o", offset=(8, 7)):
    r0, _, c0, _ = bounds
    rc = cm.world_to_grid(np.asarray(xy, dtype=float))
    x, y = float(rc[0] - r0), float(rc[1] - c0)
    ax.plot([x], [y], marker=symbol, markersize=11, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=1.9,
            linestyle="none", zorder=10)
    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                fontsize=9.4, color=color, fontweight="bold", zorder=11,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])


def photo_panel(ax, section_path: Path):
    if not section_path.exists():
        raise SystemExit(f"missing HM3D section cache: {section_path}")
    section = np.load(section_path)["img"]
    # Crop to the photographed flight, with both its landings still visible.
    section = section[:, : int(section.shape[1] * 0.49), :]
    alpha = np.where(section.sum(axis=2) > 16, 255, 0).astype(np.uint8)
    ax.imshow(np.dstack([section, alpha]), origin="upper",
              interpolation="bilinear", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a)  Real staircase in HM3D", fontsize=12.2,
                 fontweight="bold", color=INK, pad=8)
    ax.set_xlabel("Orthographic section; orange marks the photographed run",
                  fontsize=9.0, color=MUTED, labelpad=8)
    # Visual pointers on the photograph. Exact foot/top coordinates are shown
    # separately on the saved height map below.
    ax.annotate("upper landing", xy=(0.52, 0.55), xycoords="axes fraction",
                xytext=(0.58, 0.87), textcoords="axes fraction",
                ha="center", fontsize=9.4, color=TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.0),
                bbox=dict(boxstyle="round,pad=0.27", facecolor="white",
                          edgecolor="#cad4d8"), zorder=10)
    ax.annotate("visible risers", xy=(0.25, 0.35), xycoords="axes fraction",
                xytext=(0.64, 0.34), textcoords="axes fraction",
                ha="center", fontsize=9.4, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2),
                bbox=dict(boxstyle="round,pad=0.27", facecolor="white",
                          edgecolor="#cad4d8"), zorder=10)
    ax.annotate("lower landing", xy=(0.10, 0.10), xycoords="axes fraction",
                xytext=(0.58, 0.08), textcoords="axes fraction",
                ha="center", fontsize=9.4, color=FOOT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=FOOT, lw=2.0),
                bbox=dict(boxstyle="round,pad=0.27", facecolor="white",
                          edgecolor="#cad4d8"), zorder=10)
    for spine in ax.spines.values():
        spine.set_color("#c7d0d4")


def criterion_panel(ax, cm, lower, upper, flight):
    dh, valid = cm.height_gradient(cell_m=0.1)
    values = dh[valid & np.isfinite(dh)]
    values = values[(values >= 0) & (values <= 1.4)]
    ax.hist(values, bins=np.linspace(0, 1.4, 100), color="#acb8bf",
            alpha=0.88, log=True)
    ax.axvspan(0.03, 0.20, facecolor=ORANGE, alpha=0.24, zorder=0)
    ax.axvline(0.03, color=ORANGE, lw=1.4)
    ax.axvline(0.20, color=ORANGE, lw=1.4)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(bottom=0.8)
    ax.set_yticks([])
    ax.set_xlabel("largest height difference to 8 neighbours  (m)",
                  fontsize=9.0, color=MUTED, labelpad=7)
    ax.set_title("(b)  Geometry separates three regimes", fontsize=12.2,
                 color=INK, fontweight="bold", pad=8)
    for x, y, name, color in ((0.035, 0.95, "flat", "#56805a"),
                              (0.12, 0.84, "riser", ORANGE),
                              (0.82, 0.95, "wall", MUTED)):
        ax.text(x, y, name, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=10.0, color=color,
                fontweight="bold", zorder=5)
    rise = flight.span_m
    gap = float(upper["height_y"] - lower["height_y"])
    ax.text(0.49, 0.57,
            f"local band: 0.03–0.20 m\n"
            "per-cell test fired in 28/35\n"
            "single-floor episodes; require ≥1 m rise\n"
            f"matched flight: {rise:.2f} m / {gap:.2f} m gap",
            transform=ax.transAxes, fontsize=8.7, color=INK,
            bbox=dict(boxstyle="round,pad=0.50", facecolor="white",
                      edgecolor="#d0dadd"), zorder=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def edge_panel(ax, lower, upper, edge):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(c)  Connectivity belongs on the storey layer",
                 fontsize=12.2, color=INK, fontweight="bold", pad=8)
    for x, y, name, h in ((0.16, 0.73, "φ lower", lower["height_y"]),
                          (0.83, 0.73, "φ upper", upper["height_y"])):
        ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.11), 0.26, 0.22,
                                    boxstyle="round,pad=0.015,rounding_size=0.04",
                                    facecolor=LIGHT, edgecolor=TEAL,
                                    linewidth=1.5, zorder=3))
        ax.text(x, y, f"{name}\ny={float(h):.2f} m", ha="center",
                va="center", fontsize=10.3, color=INK, fontweight="bold", zorder=4)
    descending = int(edge["from_floor"]) == int(upper["key"])
    start, end = ((0.68, 0.73), (0.31, 0.73)) if descending else ((0.31, 0.73), (0.68, 0.73))
    ax.add_patch(FancyArrowPatch(start, end,
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.5, color=TEAL, zorder=2))
    ax.text(0.50, 0.84, "StairEdge", ha="center", fontsize=11.0,
            color=TEAL, fontweight="bold", zorder=5)
    ax.text(0.50, 0.51,
            "StairEdge = (φfrom, φto, xentry, xexit)",
            ha="center", fontsize=9.5, color=TEAL, fontweight="bold")
    ax.text(0.50, 0.30,
            f"x entry = {np.asarray(edge['entry_xy']).round(2).tolist()}\n"
            f"x exit  = {np.asarray(edge['exit_xy']).round(2).tolist()}\n"
            f"saved after transition, step {int(edge['step'])}",
            ha="center", va="center", fontsize=9.3, color=INK,
            bbox=dict(boxstyle="round,pad=0.50", facecolor="white",
                      edgecolor="#d0dadd"), zorder=4)
    ax.text(0.50, 0.07, "A route between storeys; no containment node is added.",
            ha="center", fontsize=9.2, color=MUTED, zorder=4)


def map_panels(ax_mask, ax_flight, cm, flight, portal, bounds, overlap):
    r0, r1, c0, c1 = bounds
    sl = (slice(r0, r1), slice(c0, c1))
    base = map_base(cm, sl)
    mask = cm.stair_mask[sl]
    shown = base.copy()
    shown[mask] = matplotlib.colors.to_rgb(ORANGE)
    show_grid(ax_mask, shown, title="(d)  Saved traversability stair mask",
              subtitle="orange = allowed cells; the mask has no foot or top")

    flight_mask = np.zeros(cm.grid.shape, dtype=bool)
    flight_mask[flight.cells_rc[:, 0], flight.cells_rc[:, 1]] = True
    shown = base.copy()
    shown[mask] = matplotlib.colors.to_rgb("#f0d7b4")
    shown[flight_mask[sl]] = matplotlib.colors.to_rgb(TEAL)
    show_grid(ax_flight, shown, title="(e)  Flight from intermediate heights",
              subtitle=f"blue = connected treads; {flight.n_cells} cells, {flight.span_m:.2f} m rise")
    marker_at(ax_flight, cm, flight.foot_xy, bounds,
              label="foot", color=FOOT, offset=(8, -22))
    marker_at(ax_flight, cm, flight.top_xy, bounds,
              label="top", color=TOP, offset=(7, 6))
    marker_at(ax_flight, cm, portal.centroid_xy, bounds,
              label="portal", color=PURPLE, symbol="D", offset=(-54, 15))
    ax_flight.text(0.02, 0.02, f"mask ∩ flight: {overlap} cells",
                   transform=ax_flight.transAxes, fontsize=8.6, color=INK,
                   bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                             edgecolor="#d0dadd"), zorder=15)


def legend_panel(ax, portal, n_tracks):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(f)  What each signal answers", fontsize=12.2,
                 color=INK, fontweight="bold", pad=8)
    rows = [
        (ORANGE, "Stair mask", "Where may the agent step?"),
        (TEAL, "Flight", "Where are its foot and top?"),
        (PURPLE, "Portal", "Which other storey is visible?"),
    ]
    for i, (color, term, question) in enumerate(rows):
        y = 0.82 - i * 0.23
        ax.add_patch(FancyBboxPatch((0.05, y - 0.055), 0.08, 0.11,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=color, edgecolor="none"))
        ax.text(0.17, y + 0.028, term, fontsize=10.5,
                color=color, fontweight="bold", va="center")
        ax.text(0.17, y - 0.035, question, fontsize=9.0,
                color=INK, va="center")
    ax.text(0.05, 0.18, "Goal priority", fontsize=9.5,
            color=MUTED, fontweight="bold")
    ax.text(0.05, 0.115, "flight  >  `stairs` track  >  portal",
            fontsize=9.3, color=INK, fontweight="bold")
    ax.text(0.05, 0.045,
            f"Track rate: 15% multi-floor; here: {n_tracks}.  Portal Δy={portal.delta_y:.2f} m.",
            fontsize=8.4, color=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--maps", type=Path, default=MAPS)
    parser.add_argument("--section", type=Path, default=SECTION)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/figures/F_stair_geometry.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    blob, cm, lower, upper = load_map(args.scene, args.maps)
    flight, portal, edge, overlap = select_evidence(blob, cm, lower, upper)
    bounds = crop_bounds(cm, flight, portal, edge)
    n_tracks = sum(1 for t in blob.get("tracks", [])
                   if str(t.get("label", "")).lower() in
                   ("stairs", "staircase", "stair"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14.4, 8.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.10, 1.05, 1.12],
                          height_ratios=[1.13, 1.0],
                          left=0.025, right=0.985, top=0.90, bottom=0.065,
                          wspace=0.20, hspace=0.37)
    fig.suptitle("From stair geometry to a cross-storey edge",
                 x=0.025, y=0.965, ha="left", fontsize=18.0,
                 fontweight="bold", color=INK)
    photo_panel(fig.add_subplot(gs[0, 0]), args.section)
    criterion_panel(fig.add_subplot(gs[0, 1]), cm, lower, upper, flight)
    edge_panel(fig.add_subplot(gs[0, 2]), lower, upper, edge)
    ax_mask = fig.add_subplot(gs[1, 0])
    ax_flight = fig.add_subplot(gs[1, 1])
    map_panels(ax_mask, ax_flight, cm, flight, portal, bounds, overlap)
    legend_panel(fig.add_subplot(gs[1, 2]), portal, n_tracks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        target = args.out.with_suffix(suffix)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"wrote {target}")
    plt.close(fig)
    print(f"flight={flight.n_cells} cells, span={flight.span_m:.2f} m, "
          f"mask overlap={overlap}, portal Δy={portal.delta_y:.2f} m, "
          f"edge step={edge['step']}, stairs tracks={n_tracks}")


if __name__ == "__main__":
    main()
