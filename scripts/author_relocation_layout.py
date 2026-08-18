#!/usr/bin/env python3
"""Synthesise a cross-anchor dynamic layout from an authored static one.

Phase 2 needs a before/after pair to relocate between, and the collector has so
far authored only `static_scene_config.json`. Rather than block, this builds the
`after` layout by PERMUTING the authored poses: every object is moved to a pose
that a human already placed and validated, on a different anchor. Nothing is
invented -- the poses are the collector's own, so objects still rest on real
surfaces at real heights -- and the result is written through the ordinary
schema, so ycb_layouts' validator checks it like any hand-authored layout.

    python scripts/author_relocation_layout.py \
        --layout-root /datasets/habitat-data-collector/outputs/dualmap_authoring \
        --data-root /datasets/habitat-data-collector/data \
        --scene 00829-QaLdnwvtxbs --index 1

This is a STAND-IN, and it is only honest for `cross_anchor`. An `in_anchor`
layout needs a second pose on the SAME surface, which cannot be borrowed from
another object and cannot be invented without the anchor's extent -- that one
has to come from the collector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def cross_anchor_permutation(objects: List[dict], seed: int) -> Optional[Dict[int, int]]:
    """Assign each object the pose of another object on a DIFFERENT anchor.

    A perfect matching on the bipartite graph "object i may take pose j iff
    anchor(i) != anchor(j)". Backtracking is ample for the handful of objects a
    layout holds, and trying candidates in a seeded random order keeps
    successive indices from producing the same permutation.
    """
    rng = np.random.default_rng(seed)
    anchors = [obj["anchor"]["object_id"] for obj in objects]
    n = len(objects)
    order = sorted(range(n), key=lambda i: sum(1 for j in range(n) if anchors[j] != anchors[i]))
    assignment: Dict[int, int] = {}
    used: set = set()

    def solve(k: int) -> bool:
        if k == len(order):
            return True
        i = order[k]
        candidates = [j for j in range(n) if j not in used and anchors[j] != anchors[i]]
        rng.shuffle(candidates)
        for j in candidates:
            assignment[i], _ = j, used.add(j)
            if solve(k + 1):
                return True
            del assignment[i]
            used.discard(j)
        return False

    return assignment if solve(0) else None


def build_layout(static: dict, permutation: Dict[int, int], index: int) -> dict:
    objects = static["objects"]
    moved = []
    for i, obj in enumerate(objects):
        donor = objects[permutation[i]]
        moved.append(
            {
                **obj,
                "translation": list(donor["translation"]),
                "rotation": list(donor["rotation"]),
                # The object is now standing on the donor's anchor -- recording
                # anything else would make the layout lie about itself, and the
                # validator would (correctly) reject it.
                "anchor": dict(donor["anchor"]),
            }
        )
    return {
        **static,
        "objects": moved,
        "authoring": {
            "layout_type": "cross_anchor",
            "layout_index": int(index),
            "reference_static_config": "../../static_scene_config.json",
            "relocated_semantic_ids": [int(o["semantic_id"]) for o in moved],
            "synthesised_by": "scripts/author_relocation_layout.py",
            "note": (
                "poses permuted across anchors from the authored static layout; "
                "replace with collector-authored placements when available"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--index", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing layout")
    args = ap.parse_args()

    static_path = args.layout_root / args.scene / "static_scene_config.json"
    static = json.loads(static_path.read_text(encoding="utf-8"))

    permutation = cross_anchor_permutation(
        static["objects"], args.seed if args.seed is not None else args.index
    )
    if permutation is None:
        raise SystemExit(
            f"{args.scene}: no cross-anchor permutation exists -- every object would have "
            "to land on a different anchor, and this layout does not have enough of them"
        )

    out = args.layout_root / args.scene / "dynamic_scene_config" / "cross_anchor" / \
        f"layout_{args.index:02d}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_layout(static, permutation, args.index), indent=2),
                   encoding="utf-8")

    # Validate through the real loader, against the real static layout: a
    # generator that writes files the pipeline rejects is worse than none.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from osg.core.config import YCB_TARGET_LABELS
    from osg.sim.ycb_layouts import load_authored_layout

    static_layout = load_authored_layout(
        static_path, layout_root=args.layout_root, data_root=args.data_root,
        target_labels=YCB_TARGET_LABELS,
    )
    dynamic = load_authored_layout(
        out, layout_root=args.layout_root, data_root=args.data_root,
        target_labels=YCB_TARGET_LABELS, static_layout=static_layout,
    )
    print(f"wrote {out}")
    by_id = {o.semantic_id: o for o in static_layout.objects}
    for obj in dynamic.objects:
        before = by_id[obj.semantic_id]
        print(f"  {obj.label:18s} {before.anchor_object_id:12s} -> {obj.anchor_object_id:12s}"
              f"  ({np.linalg.norm(np.array(before.translation) - np.array(obj.translation)):.2f} m)")


if __name__ == "__main__":
    main()
