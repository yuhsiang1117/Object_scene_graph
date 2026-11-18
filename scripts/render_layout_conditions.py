#!/usr/bin/env python3
"""The four layout conditions of the dynamic benchmark, on a real floor plan.

static / in-anchor / cross-anchor / multi-floor, read out of the authored
layout files rather than described: each panel marks where the STATIC layout
put an object -- the position pass 1 maps and pass 2 starts out believing --
and where the relocated layout puts it, with the real displacement.

Panel examples are chosen by rule from the current layout files (see
`classify`), so a re-authored layout set changes the numbers rather than
silently invalidating them. The script prints what it chose.

    python scripts/render_layout_conditions.py
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osg.core.paths import ycb_multi_floor_root

SCENE = "00873-bxsVRursffK"
ASSETS = ROOT / "docs/figures"
OUT = ASSETS / "layout_conditions.png"

INK = "#26353d"
MUTED = "#5b6d75"
GOLD = "#a67633"
GREEN = "#237f59"
STAIR = "#c37937"
BLUE = "#396eaa"
RED = "#b3352e"
PURPLE = "#7660a9"


# ------------------------------------------------------------------ the data

def layouts(scene: str):
    root = pathlib.Path(ycb_multi_floor_root()) / scene
    static = json.loads((root / "static_scene_config.json").read_text())
    found = {}
    for kind in ("in_anchor", "cross_anchor"):
        for path in sorted((root / "dynamic_scene_config" / kind).glob("layout_*.json")):
            found[(kind, path.stem)] = json.loads(path.read_text())
    return static, found, root


def by_id(blob):
    return {int(o["semantic_id"]): o for o in blob["objects"]}


def floor_of(y: float, floors) -> int:
    return int(min(range(len(floors)), key=lambda i: abs(y - floors[i]["height"])))


def classify(static, blob, floors):
    """Every object this layout moved, tagged with the condition it realises."""
    home, now = by_id(static), by_id(blob)
    out = []
    for sid, obj in now.items():
        if sid not in home:
            continue
        a = np.asarray(home[sid]["translation"], float)
        b = np.asarray(obj["translation"], float)
        dist = float(np.linalg.norm(b - a))
        if dist < 0.02:
            continue
        anchor_a = (home[sid].get("anchor") or {}).get("object_id")
        anchor_b = (obj.get("anchor") or {}).get("object_id")
        fa, fb = floor_of(a[1], floors), floor_of(b[1], floors)
        if fa != fb:
            kind = "multi-floor"
        elif anchor_a == anchor_b and dist < 0.6:
            kind = "in-anchor"
        else:
            kind = "cross-anchor"
        out.append(dict(sid=sid, kind=kind, home=a, now=b, dist=dist,
                        anchor_home=anchor_a, anchor_now=anchor_b,
                        floor_home=fa, floor_now=fb,
                        label=static["id_handle_mapping"][str(sid)]))
    return out


def pick(cases, kind, *, inside, prefer_far=False, floor=None):
    """One example of a condition: the clearest one that fits the floor plan."""
    ok = [c for c in cases if c["kind"] == kind
          and inside(c["home"]) and inside(c["now"])
          and (floor is None or (c["floor_home"] == floor and c["floor_now"] == floor))]
    if not ok:
        raise SystemExit(f"no {kind} example inside the rendered floors")
    return max(ok, key=lambda c: c["dist"]) if prefer_far else \
        min(ok, key=lambda c: c["dist"])


def pretty(handle: str) -> str:
    """`005_tomato_soup_can` -> `tomato soup can`."""
    return handle.split("_", 1)[-1].replace("_", " ").replace("-a ", " ")


def anchor_name(anchor_id: str) -> str:
    return str(anchor_id).rsplit("_", 1)[0] if anchor_id else "?"


# --------------------------------------------------------------- the drawing

def card(ax, x, y, w, h, *, face="white", edge="#c8d4d7", lw=1.2, radius=.012, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                zorder=z))


def label(ax, x, y, text, *, color=INK, size=10.4, weight="normal",
          ha="center", va="center", z=15):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, linespacing=1.3, zorder=z)


class Plan:
    """A rendered floor, and the world -> axes mapping that goes with it."""

    def __init__(self, ax, image, meta, box):
        self.ax, self.meta, self.box = ax, meta, box
        x, y, w, h = box
        card(ax, x - .006, y - .008, w + .012, h + .016, face="white",
             edge="#cbd7d9", lw=1.0, radius=.007, z=2)
        ax.imshow(image, extent=(x, x + w, y, y + h), origin="upper",
                  interpolation="bilinear", aspect="auto", zorder=3)

    def uv(self, world):
        cx, cz = self.meta["camera_center_xz"]
        span = self.meta["span_m"]
        return (400.0 + (float(world[0]) - cx) * 800.0 / span,
                400.0 + (float(world[2]) - cz) * 800.0 / span)

    def point(self, world):
        u, v = self.uv(world)
        scale = self.meta["resolution"] / 800.0
        left, top, right, bottom = self.meta["crop"]
        x, y, w, h = self.box
        return (x + w * ((u * scale - left) / (right - left)),
                y + h * (1 - (v * scale - top) / (bottom - top)))

    def inside(self, world) -> bool:
        u, v = self.uv(world)
        scale = self.meta["resolution"] / 800.0
        left, top, right, bottom = self.meta["crop"]
        return left <= u * scale <= right and top <= v * scale <= bottom


def home_marker(ax, xy, *, size=13, color=BLUE, z=12):
    """Where the prior map says the object is: an open ring."""
    ax.plot(*xy, marker="o", markersize=size, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2.6, linestyle="none", zorder=z)


def now_marker(ax, xy, *, size=12, color=GREEN, z=12):
    """Where pass 2 will actually find it: a filled disc."""
    ax.plot(*xy, marker="o", markersize=size, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=2.0, linestyle="none", zorder=z)


def move_arrow(ax, a, b, *, color=RED, lw=2.4, rad=.18, z=11):
    arrow = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                            linewidth=lw, color=color, zorder=z,
                            linestyle=(0, (5, 3)),
                            connectionstyle=f"arc3,rad={rad}",
                            shrinkA=9, shrinkB=9)
    arrow.set_path_effects([pe.Stroke(linewidth=lw + 2.4, foreground="white"),
                            pe.Normal()])
    ax.add_patch(arrow)


def tag(ax, xy, text, *, color=INK, size=9.8, z=16):
    ax.text(*xy, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold", zorder=z,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])


def zoom(ax, plan, case, box, *, span_m=1.1):
    """A magnified crop, for a move too small to see at floor scale."""
    inset = ax.inset_axes(box)
    image = plan.ax.images[0]
    arr = image.get_array()
    meta = plan.meta
    scale = meta["resolution"] / 800.0
    left, top, right, bottom = meta["crop"]
    centre = (case["home"] + case["now"]) / 2
    u, v = plan.uv(centre)
    px = span_m * 800.0 / meta["span_m"] * scale / 2
    c, r = u * scale - left, v * scale - top
    lo_c, hi_c = int(max(c - px, 0)), int(min(c + px, arr.shape[1]))
    lo_r, hi_r = int(max(r - px, 0)), int(min(r + px, arr.shape[0]))
    inset.imshow(arr[lo_r:hi_r, lo_c:hi_c], interpolation="bilinear")
    for world, marker in ((case["home"], home_marker), (case["now"], now_marker)):
        uu, vv = plan.uv(world)
        marker(inset, (uu * scale - left - lo_c, vv * scale - top - lo_r))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(1.2)
    # A box on the plan where the crop came from, and a leader to the inset.
    x0, y0 = plan.point(case["home"])
    half = span_m / meta["span_m"] * (plan.box[2] * 800.0 * scale
                                      / (right - left)) / 2
    ax.add_patch(Rectangle((x0 - half, y0 - half), 2 * half, 2 * half,
                           facecolor="none", edgecolor=MUTED, lw=1.1, zorder=10))
    ax.add_patch(FancyArrowPatch((x0 + half, y0), (box[0], box[1] + box[3] / 2),
                                 arrowstyle="-", color=MUTED, lw=1.0,
                                 linestyle=(0, (3, 3)), zorder=10))
    return inset


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    static, found, root = layouts(args.scene)
    floors = static["authoring"]["floors"]
    asset = args.scene.split("-", 1)[0]
    meta = json.loads((ASSETS / f"hm3d_topdown_{asset}_meta.json").read_text())
    lower = plt.imread(ASSETS / f"hm3d_topdown_{asset}_lower.png")
    upper = plt.imread(ASSETS / f"hm3d_topdown_{asset}_upper.png")

    cases = []
    for (kind, name), blob in found.items():
        for case in classify(static, blob, floors):
            case["layout"] = f"{kind}/{name}"
            cases.append(case)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14.4, 9.8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    card(ax, .012, .012, .976, .976, face="#fcfdfd", edge="#d2dee0",
         lw=1.1, radius=.016, z=0)

    # ------------------------------------------------------------- header
    card(ax, .045, .898, .910, .080, face="#fff8ee", edge="#dfc69e",
         lw=1.3, radius=.010, z=1)
    label(ax, .500, .953, "Pass 1 maps the static layout   →   objects are "
          "relocated   →   pass 2 searches from that stale snapshot",
          size=13.0, weight="bold", color=GOLD)
    for x, sym, text in ((.215, "ring", "where the prior map says it is"),
                         (.525, "disc", "where pass 2 will find it"),
                         (.790, "arrow", "the authored relocation")):
        if sym == "ring":
            home_marker(ax, (x - .085, .919), size=10)
        elif sym == "disc":
            now_marker(ax, (x - .085, .919), size=9)
        else:
            move_arrow(ax, (x - .105, .919), (x - .068, .919), rad=0, lw=2.0)
        label(ax, x, .919, text, size=10.0, color=MUTED, ha="left")

    # The three same-storey panels are drawn on whichever storey has both a
    # short move and a long one to show, so the figure never mixes a case with
    # a floor plan it does not belong to.
    images = {0: lower, 1: upper}
    names = {0: "lower", 1: "upper"}
    storey = next((f for f in sorted(images)
                   if any(c["kind"] == "in-anchor" and c["floor_home"] == f
                          and c["floor_now"] == f for c in cases)
                   and any(c["kind"] == "cross-anchor" and c["floor_home"] == f
                           and c["floor_now"] == f for c in cases)), 0)
    plans = []
    boxes = [(.048, .530, .285, .330), (.360, .530, .285, .330),
             (.672, .530, .285, .330)]
    for box in boxes:
        sub = fig.add_axes((0, 0, 1, 1), frameon=False)
        sub.set_xlim(0, 1)
        sub.set_ylim(0, 1)
        sub.set_axis_off()
        sub.set_zorder(2)
        plans.append(Plan(sub, images[storey], meta, box))
    plan_lo = plans[0]

    def inside_lower(world):
        return plan_lo.inside(world)

    # (a) the static layout: every authored object on this storey
    ax_a = plans[0].ax
    home = by_id(static)
    n_here = 0
    for sid, obj in sorted(home.items()):
        pos = np.asarray(obj["translation"], float)
        if floor_of(pos[1], floors) != storey or not plans[0].inside(pos):
            continue
        home_marker(ax_a, plans[0].point(pos), size=11)
        n_here += 1
    label(ax, .190, .878, "(a)  static", size=12.6, weight="bold")
    label(ax, .190, .462, f"the layout pass 1 maps: {n_here} objects on the "
          f"{names[storey]} storey,\neach on a named piece of furniture.  In the "
          "static\ncondition pass 2 sees exactly this.", size=9.8, color=MUTED)

    # (b) in-anchor and (c) cross-anchor, on the same storey
    in_case = pick(cases, "in-anchor", inside=inside_lower, floor=storey,
                   prefer_far=True)
    cross_case = pick(cases, "cross-anchor", inside=inside_lower, floor=storey,
                      prefer_far=True)
    for i, (case, title, note) in enumerate((
            (in_case, "(b)  in-anchor",
             "same piece of furniture, a short move:\nthe map's pose is wrong "
             "by less than a metre,\nand an arrival can see the object from it."),
            (cross_case, "(c)  cross-anchor",
             "another piece of furniture on the same storey:\nthe mapped pose "
             "is empty and the object is\nsomewhere the prior map does not "
             "suggest.")), start=1):
        plan = plans[i]
        a, b = plan.point(case["home"]), plan.point(case["now"])
        move_arrow(plan.ax, a, b)
        home_marker(plan.ax, a)
        now_marker(plan.ax, b)
        label(ax, boxes[i][0] + boxes[i][2] / 2, .878, title, size=12.6,
              weight="bold")
        label(ax, boxes[i][0] + boxes[i][2] / 2, .462, note, size=9.8, color=MUTED)
        tag(ax, (boxes[i][0] + boxes[i][2] / 2, .507),
            f"{pretty(case['label'])}\n{anchor_name(case['anchor_home'])} "
            f"→ {anchor_name(case['anchor_now'])},  {case['dist']:.2f} m",
            color=RED, size=10.2)
    zoom(plans[1].ax, plans[1], in_case, (.545, .545, .088, .088))

    # (d) multi-floor: the two storeys side by side
    multi = pick(cases, "multi-floor", inside=lambda w: True)
    lo_box, up_box = (.115, .030, .320, .232), (.565, .030, .320, .232)
    sub = fig.add_axes((0, 0, 1, 1), frameon=False)
    sub.set_xlim(0, 1)
    sub.set_ylim(0, 1)
    sub.set_axis_off()
    sub.set_zorder(2)
    plan_from = Plan(sub, images[multi["floor_home"]], meta, lo_box)
    plan_to = Plan(sub, images[multi["floor_now"]], meta, up_box)
    home_marker(sub, plan_from.point(multi["home"]))
    now_marker(sub, plan_to.point(multi["now"]))
    a = plan_from.point(multi["home"])
    b = plan_to.point(multi["now"])
    move_arrow(sub, (a[0] + .012, a[1]), (b[0] - .012, b[1]), rad=-.12, lw=2.6)
    label(ax, .500, .372, "(d)  multi-floor", size=12.6, weight="bold")
    tag(ax, (.500, .330),
        f"{pretty(multi['label'])}:  {anchor_name(multi['anchor_home'])} "
        f"→ {anchor_name(multi['anchor_now'])},  {multi['dist']:.2f} m, "
        f"{abs(multi['now'][1] - multi['home'][1]):.2f} m of it vertical",
        color=RED, size=10.8)
    label(ax, .500, .295, "the search must leave the storey the map points at, "
          "so the target's floor becomes a decision of its own",
          size=9.8, color=MUTED)
    for box, floor in ((lo_box, multi["floor_home"]), (up_box, multi["floor_now"])):
        # On the panel's own axes: the floor images sit above `ax`.
        tag(sub, (box[0] + .090, box[1] + box[3] + .016),
            f"storey {floor}   y = {floors[floor]['height']:+.2f} m",
            color=MUTED, size=9.8)

    print(f"{args.scene}: layouts from {root}")
    for name, case in (("in-anchor", in_case), ("cross-anchor", cross_case),
                       ("multi-floor", multi)):
        print(f"  {name:<12} {pretty(case['label']):<20} {case['layout']:<24} "
              f"{case['dist']:.2f} m  f{case['floor_home']}->f{case['floor_now']}  "
              f"{anchor_name(case['anchor_home'])} -> {anchor_name(case['anchor_now'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        path = args.out.with_suffix(ext)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
