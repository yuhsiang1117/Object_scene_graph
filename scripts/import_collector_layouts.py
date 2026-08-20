#!/usr/bin/env python3
"""Import the collector's raw HM3D_collect layouts into the authoring schema.

The collector records dynamic layouts as bare pose lists under
`data/dualmap/HM3D_collect/<scene>/dynamic_scene_config/<type>/*.json`: no
`authoring` block, no per-object anchor, and a scene path that predates the
current mount. `ycb_layouts` needs all three, so this converts rather than
relaxing the validator -- the checks it performs (identical object sets,
identical id mapping, every object relocated, in_anchor stays on its anchor)
are exactly what makes a layout trustworthy.

    python scripts/import_collector_layouts.py --scene 00829-QaLdnwvtxbs

Anchors are DERIVED, and the file says so. In this collector's scenes every
static object sits on its own piece of furniture -- the closest pair is 2.19 m
apart -- so "one static object, one anchor" is not a guess. A dynamic layout
then inherits its object's anchor when the directory says `in_anchor`, and is
assigned a different one when it says `cross_anchor`. The directory is the
author's intent and stays authoritative; the displacement check below is a
cross-check that reports disagreement rather than overruling it.

Written to a SEPARATE layout root: `outputs/dualmap_authoring` holds a different
authoring session for the same scene, with different poses, and pairing a raw
dynamic layout against those would show objects moving from poses they were
never in.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

# The collector's own scene paths point at machines that no longer exist
# (/home/eku/..., which rebase_collector_path cannot map because it has no
# `data` component). This is the form the nav image mounts.
SCENE_BLOCK = {
    "scene_path": "/app/data/scene_datasets/hm3d/val/{scene}/{stem}.basis.glb",
    "scene_dataset_config": (
        "/app/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    ),
}

# Where in_anchor stops and cross_anchor starts, for the cross-check only.
# Measured on 00829-QaLdnwvtxbs: in_anchor moves reach 1.56 m, cross_anchor
# moves start at 2.18 m.
SAME_ANCHOR_MAX_M = 2.0


def anchor_id(handle: str) -> str:
    return f"surface_{handle}"


def scene_block(scene: str) -> Dict[str, str]:
    stem = scene.split("-", 1)[1] if "-" in scene else scene
    return {k: v.format(scene=scene, stem=stem) for k, v in SCENE_BLOCK.items()}


def dedupe_instances(objects: List[dict], mapping: Dict[str, str],
                     where: str) -> Tuple[List[dict], List[dict]]:
    """One instance per semantic id, keeping the first and reporting the rest.

    00848-ziup5kvtCCR places TWO mugs and gives both semantic id 98. The schema
    keys goals, viewpoints and relocation pairs on that id, so two objects
    wearing it is not a scene with two mugs -- it is a scene where "the mug" has
    no referent. The collector's own data is already inconsistent about it: one
    of the seven layouts omits the second mug entirely, so the object sets do
    not match across layouts either, and the importer's identical-object-sets
    check would reject the scene for that alone.

    Dropping the extra instance costs a distractor that is never a target here
    and buys a usable scene. It is recorded in the layout rather than done
    quietly, and the collector's files are not touched.
    """
    kept, dropped, seen = [], [], set()
    for obj in objects:
        sid = int(obj["semantic_id"])
        if sid in seen:
            dropped.append(obj)
            print(f"  {where}: dropping a second instance of "
                  f"{mapping[str(sid)]} (semantic id {sid}) at "
                  f"{[round(v, 2) for v in obj['translation']]}")
            continue
        seen.add(sid)
        kept.append(obj)
    return kept, dropped


def _dropped_note(dropped: List[dict], mapping: Dict[str, str]) -> List[dict]:
    return [
        {"semantic_id": int(o["semantic_id"]), "handle": mapping[str(o["semantic_id"])],
         "translation": [float(v) for v in o["translation"]],
         "reason": "duplicate semantic id; the schema allows one instance per id"}
        for o in dropped
    ]


def convert_static(raw: dict, scene: str) -> dict:
    mapping = raw["id_handle_mapping"]
    raw_objects, dropped = dedupe_instances(
        raw["objects"], mapping, "static_scene_config.json")
    objects = []
    for obj in raw_objects:
        handle = mapping[str(obj["semantic_id"])]
        objects.append(
            {
                **obj,
                "anchor": {"object_id": anchor_id(handle), "category": "surface"},
            }
        )
    return {
        "scene": scene_block(scene),
        "id_handle_mapping": mapping,
        "objects": objects,
        "authoring": {
            "layout_type": "static",
            "layout_index": None,
            "reference_static_config": None,
            "relocated_semantic_ids": [],
            "imported_by": "scripts/import_collector_layouts.py",
            "anchors": "derived: one anchor per static object (all >2 m apart)",
            "dropped_duplicate_instances": _dropped_note(dropped, mapping),
        },
    }


def convert_dynamic(raw: dict, static_raw: dict, scene: str, layout_type: str,
                    index: int, source: str) -> dict:
    mapping = static_raw["id_handle_mapping"]
    static_by_id = {o["semantic_id"]: o for o in static_raw["objects"]}
    raw_objects, dropped = dedupe_instances(raw["objects"], mapping, source)
    objects = []
    for obj in raw_objects:
        sid = obj["semantic_id"]
        handle = mapping[str(sid)]
        if layout_type == "in_anchor":
            anchor = anchor_id(handle)
        else:
            # Cross-anchor: the object is on SOMETHING else. Name it by the
            # static object it landed nearest to when it plausibly took that
            # object's surface, otherwise by itself -- either way it differs
            # from its own former anchor, which is what the type asserts.
            anchor = f"surface_moved_{handle}"
            best, best_d = None, SAME_ANCHOR_MAX_M
            for other_id, other in static_by_id.items():
                if other_id == sid:
                    continue
                d = math.dist(obj["translation"], other["translation"])
                if d < best_d:
                    best, best_d = mapping[str(other_id)], d
            if best is not None:
                anchor = anchor_id(best)
        objects.append({**obj, "anchor": {"object_id": anchor, "category": "surface"}})

    return {
        "scene": scene_block(scene),
        "id_handle_mapping": mapping,
        "objects": objects,
        "authoring": {
            "layout_type": layout_type,
            "layout_index": int(index),
            "reference_static_config": "../../static_scene_config.json",
            "relocated_semantic_ids": [int(o["semantic_id"]) for o in objects],
            "imported_by": "scripts/import_collector_layouts.py",
            "source": source,
            "dropped_duplicate_instances": _dropped_note(dropped, mapping),
            "anchors": "derived from the collector's directory type",
        },
    }


def displacement_report(static_raw: dict, raw: dict, layout_type: str) -> List[str]:
    mapping = static_raw["id_handle_mapping"]
    static_by_id = {o["semantic_id"]: o for o in static_raw["objects"]}
    lines, disagree = [], []
    seen = set()
    for obj in raw["objects"]:
        sid = obj["semantic_id"]
        if sid in seen:
            continue
        seen.add(sid)
        before = static_by_id[sid]["translation"]
        d = math.dist(before, obj["translation"])
        lines.append(f"    {mapping[str(sid)]:22s} {d:5.2f} m")
        same = d <= SAME_ANCHOR_MAX_M
        if (layout_type == "in_anchor") != same:
            disagree.append(f"{mapping[str(sid)]} moved {d:.2f} m")
    if disagree:
        lines.append(
            f"    NOTE: displacement disagrees with the declared {layout_type} for: "
            + ", ".join(disagree)
            + " (the collector's directory is taken as authoritative)"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collector-root", type=Path,
                    default=Path("/datasets/habitat-data-collector/data/dualmap/HM3D_collect"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("/datasets/habitat-data-collector/data"))
    ap.add_argument("--out-root", type=Path, default=Path("outputs/collector_layouts"))
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    src = args.collector_root / args.scene
    static_raw = json.loads((src / "static_scene_config.json").read_text(encoding="utf-8"))
    out_scene = args.out_root / args.scene
    out_scene.mkdir(parents=True, exist_ok=True)
    static_out = out_scene / "static_scene_config.json"
    static_out.write_text(json.dumps(convert_static(static_raw, args.scene), indent=2),
                          encoding="utf-8")
    print(f"wrote {static_out}")

    written = []
    for layout_type in ("in_anchor", "cross_anchor"):
        directory = src / "dynamic_scene_config" / layout_type
        if not directory.is_dir():
            continue
        for index, path in enumerate(sorted(directory.glob("*.json")), start=1):
            if index > 3:
                print(f"  skipping {path.name}: the schema allows 3 layouts per type")
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            out = out_scene / "dynamic_scene_config" / layout_type / f"layout_{index:02d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    convert_dynamic(raw, static_raw, args.scene, layout_type, index, path.name),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"wrote {out}  (from {path.name})")
            for line in displacement_report(static_raw, raw, layout_type):
                print(line)
            written.append(out)

    # Validate everything through the real loader, exactly as discovery will.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from osg.core.config import YCB_TARGET_LABELS
    from osg.sim.ycb_layouts import load_authored_layout

    static_layout = load_authored_layout(
        static_out, layout_root=args.out_root, data_root=args.data_root,
        target_labels=YCB_TARGET_LABELS,
    )
    for out in written:
        load_authored_layout(
            out, layout_root=args.out_root, data_root=args.data_root,
            target_labels=YCB_TARGET_LABELS, static_layout=static_layout,
        )
    print(f"validated {len(written) + 1} layouts against the authoring schema")


if __name__ == "__main__":
    main()
