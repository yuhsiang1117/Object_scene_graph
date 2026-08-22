#!/usr/bin/env python3
"""Author dynamic YCB layouts whose destinations are SEMANTICALLY plausible.

The collector's relocations are not. Measured over all 114 of them, the objects
land on bed 26, desk 19, table 6, cabinet 3, nightstand 3, ... and 37 of the 53
whose destination surface is mapped land on a category the search prior does not
list for that class -- a tomato soup can is put on a bed seven times. The
existing synthesiser makes this worse by construction: it PERMUTES authored
poses between objects, so a soup can inherits whatever surface a plate was on.

A benchmark built that way cannot reward a semantic search prior, which is
awkward when the semantic search posterior is the thing under test. Ranking the
true destination among mapped surfaces, over those 114 relocations:

    affinity x proximity   top-1 19/114   median rank  5
    proximity alone        top-1 26/114   median rank  2
    affinity alone         top-1  0/114   median rank 27
    arbitrary order        top-1  2/114   median rank 25

Affinity is no better than shuffling. This script builds the other dataset: the
same scenes and the same object set, relocated onto surfaces a person plausibly
would use, so that "where does a bowl get put down" is a question the data can
actually answer either way.

Ground truth for "which surface" comes from HM3D's own annotations -- the
`.semantic.glb` vertex colours keyed by `.semantic.txt` -- NOT from our
detector's container layer. Using the agent's own perception to author the
benchmark it is scored on would be circular.

    python scripts/author_semantic_layouts.py --list-surfaces --scene 00829-QaLdnwvtxbs
    python scripts/author_semantic_layouts.py --out outputs/semantic_layouts

The source root is opened read-only and a new root is written, so the
collector's data under /datasets/habitat-data-collector is never edited.

Requires `trimesh` (authoring only -- the runtime never imports it).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# HM3D's semantic mesh is z-up; habitat's world is y-up. Verified rather than
# assumed: under this map all six of 00829's static YCB objects rest 0.02-0.15 m
# above a mesh vertex directly beneath them, which is what "resting on a
# surface" looks like. Under the sign-flipped alternative two of the six have no
# support at all beneath them and the rest float 0.56-0.81 m.
def mesh_to_habitat(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


# Which HM3D categories can hold a small object at all. Geometry still has to
# agree -- this is the category gate, and `_top_faces` applies the height and
# area one -- but a curtain is never a surface however flat a patch of it looks.
SUPPORT_CATEGORIES = {
    "table", "desk", "counter", "kitchen counter", "washbasin counter",
    "bed table", "night stand", "nightstand", "bedside table", "coffee table",
    "side table", "dining table", "console table", "tv stand", "cabinet",
    "kitchen cabinet", "shelf", "bookshelf", "chest of drawers", "dresser",
    "sideboard", "bench", "stool", "ottoman", "chair", "armchair", "sofa",
    "bed", "kitchen island", "sink counter", "vanity", "desk table",
}

# Where a person would actually put each YCB target down, best first, over
# HM3D's OWN category names. This is the whole point of the dataset: a
# relocation destination is drawn from this list, so a search prior that knows
# a soup can belongs on a counter can be rewarded for knowing it -- and one that
# does not can be shown to be no better than chance on the same episodes.
SEMANTIC_HOMES: Dict[str, List[str]] = {
    "024_bowl": ["kitchen counter", "counter", "dining table", "table",
                 "kitchen island", "desk", "sideboard", "shelf", "cabinet"],
    "029_plate": ["kitchen counter", "counter", "dining table", "table",
                  "kitchen island", "sideboard", "shelf", "cabinet"],
    "025_mug": ["kitchen counter", "counter", "desk", "dining table", "table",
                "bed table", "night stand", "nightstand", "bedside table",
                "coffee table", "side table", "shelf"],
    "005_tomato_soup_can": ["kitchen counter", "counter", "kitchen cabinet",
                            "shelf", "cabinet", "kitchen island", "sideboard",
                            "dining table", "table"],
    "003_cracker_box": ["kitchen counter", "counter", "kitchen cabinet", "shelf",
                        "cabinet", "kitchen island", "sideboard", "dining table",
                        "table"],
    "019_pitcher_base": ["kitchen counter", "counter", "kitchen island",
                         "dining table", "table", "sideboard", "shelf"],
    "021_bleach_cleanser": ["kitchen counter", "counter", "washbasin counter",
                            "sink counter", "vanity", "kitchen cabinet",
                            "cabinet", "shelf"],
    "011_banana": ["kitchen counter", "counter", "dining table", "table",
                   "kitchen island", "sideboard", "coffee table", "side table"],
}

CELL_M = 0.05          # (x, z) binning of an instance's vertices
TOP_BAND_M = 0.08      # how far below an instance's top a cell still counts as its top face
MIN_TOP_H_M = 0.25     # a surface below this is the floor or a footstool
MAX_TOP_H_M = 1.40     # above this nothing is put down casually
MIN_TOP_CELLS = 24     # 24 cells at 5 cm = 0.06 m2, the container layer's own floor
EDGE_INSET_M = 0.12    # keep the object off the lip of the surface
CLEAR_M = 0.30         # vertical clearance wanted above a placement point
REACH_M = 1.6          # a navigable point must exist within this of the placement


# ------------------------------------------------------------------ semantics

class SemanticSurfaces:
    """HM3D instances, their categories, regions, and their top faces."""

    def __init__(self, scene_dir: Path, stem: str) -> None:
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - authoring-only dependency
            raise SystemExit(
                "author_semantic_layouts needs trimesh (authoring only): pip install trimesh"
            ) from exc

        txt = scene_dir / f"{stem}.semantic.txt"
        glb = scene_dir / f"{stem}.semantic.glb"
        for path in (txt, glb):
            if not path.is_file():
                raise SystemExit(f"missing HM3D semantic annotation: {path}")

        self.by_colour: Dict[int, Tuple[int, str, str]] = {}
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",", 3)
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            self.by_colour[int(parts[1], 16)] = (
                int(parts[0]), parts[2].strip('"').strip().lower(), parts[3].strip())

        scene = trimesh.load(str(glb), process=False)
        pts, cols = [], []
        for geom in scene.geometry.values():
            colour = np.asarray(geom.visual.to_color().vertex_colors)[:, :3]
            pts.append(np.asarray(geom.vertices))
            cols.append(colour)
        raw = np.vstack(pts)
        rgb = np.vstack(cols).astype(np.uint32)
        self.points = mesh_to_habitat(raw)
        self.keys = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    def instances(self) -> List[dict]:
        """One record per annotated instance that could hold something."""
        out: List[dict] = []
        order = np.argsort(self.keys, kind="stable")
        keys = self.keys[order]
        pts = self.points[order]
        edges = np.flatnonzero(np.diff(keys)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(keys)]):
            info = self.by_colour.get(int(keys[lo]))
            if info is None:
                continue
            inst_id, category, region = info
            if category not in SUPPORT_CATEGORIES:
                continue
            face = _top_faces(pts[lo:hi])
            if face is None:
                continue
            cells, top_h = face
            out.append({
                "instance_id": inst_id, "category": category, "region": region,
                "top_h": top_h, "cells": cells,
                "centre": np.array([cells[:, 0].mean(), top_h, cells[:, 1].mean()]),
                "n_cells": len(cells),
            })
        return out


def _top_faces(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """The (x, z) cells that make up an instance's usable top, and its height.

    Per cell the highest vertex; then keep the cells within `TOP_BAND_M` of the
    instance's 90th-percentile height. The percentile rather than the max
    because a table with a lamp on it should still offer its table top, and the
    max would be the lamp.
    """
    if len(points) < 8:
        return None
    ij = np.floor(points[:, [0, 2]] / CELL_M).astype(np.int64)
    key = (ij[:, 0].astype(np.int64) << 32) ^ (ij[:, 1].astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(key, kind="stable")
    key, ys, ij = key[order], points[order, 1], ij[order]
    edges = np.flatnonzero(np.diff(key)) + 1
    tops, cell_xy = [], []
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        tops.append(ys[lo:hi].max())
        cell_xy.append(ij[lo])
    tops = np.asarray(tops)
    cell_xy = np.asarray(cell_xy, dtype=float) * CELL_M + CELL_M / 2.0
    top_h = float(np.percentile(tops, 90))
    if not (MIN_TOP_H_M <= top_h <= MAX_TOP_H_M):
        return None
    keep = np.abs(tops - top_h) <= TOP_BAND_M
    if int(keep.sum()) < MIN_TOP_CELLS:
        return None
    return cell_xy[keep], top_h


def inset_cells(cells: np.ndarray, inset_m: float) -> np.ndarray:
    """Cells at least `inset_m` from the edge of the top face.

    A point on the lip of a counter is a place an object falls off, and -- more
    to the point here -- a place the authored pose intersects the edge geometry.
    """
    if len(cells) < 4:
        return cells
    ring = max(1, int(round(inset_m / CELL_M)))
    ij = np.round(cells / CELL_M).astype(np.int64)
    have = {(int(a), int(b)) for a, b in ij}
    keep = []
    for idx, (a, b) in enumerate(ij):
        if all((int(a) + da, int(b) + db) in have
               for da in range(-ring, ring + 1) for db in range(-ring, ring + 1)):
            keep.append(idx)
    return cells[keep] if keep else cells


def clearance_ok(surfaces: SemanticSurfaces, xz: Sequence[float], top_h: float,
                 need_m: float) -> bool:
    """Nothing solid in the way of an object standing here."""
    d = np.abs(surfaces.points[:, [0, 2]] - np.asarray(xz, dtype=float))
    near = (d[:, 0] < CELL_M) & (d[:, 1] < CELL_M)
    if not near.any():
        return True
    ys = surfaces.points[near, 1]
    above = ys[(ys > top_h + 0.02) & (ys < top_h + need_m)]
    return len(above) == 0


# ----------------------------------------------------------------- placement

def base_offset(sim, handle: str, rotation: Sequence[float]) -> float:
    """How far the mesh's lowest point sits below its origin, at this rotation.

    The authored `translation` is the object's ORIGIN, and YCB origins are not
    at the base -- placing a pitcher's origin on a counter top buries half of it.
    """
    import habitat_sim
    import magnum as mn
    from osg.sim.ycb_env import _template_handle

    manager = sim.get_rigid_object_manager()
    templates = sim.get_object_template_manager()
    rigid = manager.add_object_by_template_handle(_template_handle(templates, handle))
    try:
        rigid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        rigid.translation = mn.Vector3(0.0, 0.0, 0.0)
        rigid.rotation = mn.Quaternion(
            mn.Vector3(*[float(v) for v in rotation[:3]]), float(rotation[3]))
        box = habitat_sim.geo.get_transformed_bb(
            rigid.root_scene_node.cumulative_bb,
            rigid.root_scene_node.absolute_transformation())
        return float(box.min[1]), float(max(box.size_x(), box.size_z()) / 2.0)
    finally:
        manager.remove_object_by_id(rigid.object_id)


def reachable(pathfinder, xz: Sequence[float], y: float, radius_m: float) -> bool:
    """Is there navigable floor close enough for the agent to look at this?"""
    from osg.sim.ycb_env import _finite_point

    snapped = _finite_point(pathfinder.snap_point(
        np.array([float(xz[0]), float(y), float(xz[1])], dtype=np.float32)))
    if snapped is None:
        return False
    return float(np.linalg.norm(snapped[[0, 2]] - np.asarray(xz, dtype=float))) <= radius_m


def place_on(surface: dict, surfaces: SemanticSurfaces, pathfinder, footprint_m: float,
             rng: np.random.Generator, avoid: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Choose a point on this surface's top that an object can actually occupy."""
    cells = inset_cells(surface["cells"], EDGE_INSET_M + footprint_m)
    if len(cells) == 0:
        return None
    order = rng.permutation(len(cells))
    for idx in order:
        xz = cells[idx]
        if any(float(np.linalg.norm(xz - np.asarray(a)[[0, 2]])) < 0.35 for a in avoid):
            continue
        if not clearance_ok(surfaces, xz, surface["top_h"], CLEAR_M):
            continue
        if not reachable(pathfinder, xz, surface["top_h"], REACH_M):
            continue
        return np.array([xz[0], surface["top_h"], xz[1]], dtype=float)
    return None


def choose_surface(handle: str, candidates: List[dict], static_xz: np.ndarray,
                   static_region: Optional[str], kind: str,
                   rng: np.random.Generator, taken: set) -> Optional[dict]:
    """The destination: a surface this class belongs on, at the right distance.

    `in_anchor` keeps the object in the same part of the house -- the short
    displacement that dominates real rearrangement -- and `cross_anchor` sends
    it to another region. Both draw ONLY from the class's own home categories,
    which is the property the collector's layouts do not have.
    """
    homes = SEMANTIC_HOMES.get(handle, [])
    rank = {name: i for i, name in enumerate(homes)}
    pool = []
    for surface in candidates:
        if surface["category"] not in rank:
            continue
        if surface["instance_id"] in taken:
            continue
        d = float(np.linalg.norm(surface["centre"][[0, 2]] - static_xz))
        if d < 0.6:
            continue  # that is where it already is; a relocation has to move it
        same_region = static_region is not None and surface["region"] == static_region
        if kind == "in_anchor":
            if d > 3.5:
                continue
        else:
            if d < 4.0 or same_region:
                continue
        # Best home category first, then nearest for in_anchor / furthest for cross.
        pool.append(((rank[surface["category"]], d if kind == "in_anchor" else -d), surface))
    if not pool:
        return None
    pool.sort(key=lambda item: item[0])
    # A little randomness across indices 1..3 so the three layouts differ, but
    # always from the best-affinity tier available.
    best_tier = pool[0][0][0]
    tier = [s for (r, _), s in pool if r == best_tier]
    return tier[int(rng.integers(len(tier)))] if tier else pool[0][1]
