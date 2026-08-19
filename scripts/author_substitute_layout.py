#!/usr/bin/env python3
"""Swap an undetectable YCB asset for another one, at the poses already authored.

Some of the collector's targets cannot be recognised by any open-vocabulary
detector at any resolution or under any name -- the scissors render as 315 px of
grey and score 0.00 against eight candidate labels (docs/DYNAMIC_SCENES.md). An
episode targeting one of those measures asset coverage, not dynamic-scene
handling, so it is dead weight in a 21-episode benchmark.

Rather than drop the slot, this puts a DIFFERENT object in it. The poses are
untouched: same x and z, same rotation, same anchor, same relocation structure
across static/in_anchor/cross_anchor -- so every property the benchmark rests on
(objects rest on real surfaces, dynamic layouts move the same object set) still
holds. Only the height is recomputed, because two meshes have different
origin-to-base offsets and a pose authored for a flat pair of scissors would
bury a mug to its rim.

    python scripts/author_substitute_layout.py \\
        --source outputs/collector_layouts --out outputs/substituted_layouts \\
        --scene 00829-QaLdnwvtxbs --replace 037_scissors=025_mug

The source root is opened read-only and a new root is written, so the collector's
data -- and DualMap's own authoring under `data/dualmap` -- is never edited. The
substitution is recorded in each layout's `authoring` block, so a layout can
always say what it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_replacements(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--replace wants OLD_HANDLE=NEW_HANDLE, got {item!r}")
        old, new = item.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def layout_files(scene_root: Path) -> List[Path]:
    files = [scene_root / "static_scene_config.json"]
    dynamic = scene_root / "dynamic_scene_config"
    if dynamic.is_dir():
        files.extend(sorted(dynamic.rglob("layout_*.json")))
    return [path for path in files if path.is_file()]


def _base_y(sim, handle: str, translation, rotation) -> Tuple[float, float]:
    """(bottom y, top y) of the mesh placed at this pose, in world coordinates."""
    import habitat_sim
    import magnum as mn
    from osg.sim.ycb_env import _template_handle

    manager = sim.get_rigid_object_manager()
    templates = sim.get_object_template_manager()
    rigid = manager.add_object_by_template_handle(_template_handle(templates, handle))
    rigid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    rigid.translation = mn.Vector3(*[float(v) for v in translation])
    rigid.rotation = mn.Quaternion(mn.Vector3(*[float(v) for v in rotation[:3]]),
                                   float(rotation[3]))
    node = rigid.root_scene_node
    box = habitat_sim.geo.get_transformed_bb(node.cumulative_bb,
                                             node.absolute_transformation())
    low, high = float(box.min[1]), float(box.max[1])
    manager.remove_object_by_id(rigid.object_id)
    return low, high


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="outputs/collector_layouts")
    ap.add_argument("--out", default="outputs/substituted_layouts")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--replace", action="append", default=[],
                    help="OLD_HANDLE=NEW_HANDLE, repeatable")
    ap.add_argument("--data-root", default="/datasets/habitat-data-collector/data")
    args = ap.parse_args()

    replacements = parse_replacements(args.replace)
    if not replacements:
        raise SystemExit("nothing to do: pass at least one --replace")

    source_scene = Path(args.source) / args.scene
    out_scene = Path(args.out) / args.scene
    if out_scene.resolve() == source_scene.resolve():
        raise SystemExit("refusing to write over the source layouts")
    files = layout_files(source_scene)
    if not files:
        raise SystemExit(f"no layouts under {source_scene}")

    static = json.loads(files[0].read_text(encoding="utf-8"))
    mapping = {str(k): str(v) for k, v in static["id_handle_mapping"].items()}
    replaced_ids = {int(sid) for sid, handle in mapping.items() if handle in replacements}
    if not replaced_ids:
        raise SystemExit(f"none of {sorted(replacements)} appear in this scene's mapping")

    import habitat_sim
    from osg.sim.ycb_layouts import rebase_collector_path

    data_root = Path(args.data_root)
    scene_mesh = rebase_collector_path(static["scene"]["scene_path"], data_root)
    dataset_config = rebase_collector_path(static["scene"]["scene_dataset_config"], data_root)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_mesh)
    sim_cfg.scene_dataset_config_file = str(dataset_config)
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = True
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.get_object_template_manager().load_configs(
        str(data_root / "versioned_data" / "ycb" / "configs")
    )

    written = []
    try:
        for path in files:
            blob = json.loads(path.read_text(encoding="utf-8"))
            blob["id_handle_mapping"] = {
                sid: replacements.get(handle, handle)
                for sid, handle in blob["id_handle_mapping"].items()
            }
            notes = []
            for obj in blob["objects"]:
                sid = int(obj["semantic_id"])
                if sid not in replaced_ids:
                    continue
                old_handle = mapping[str(sid)]
                new_handle = replacements[old_handle]
                translation = [float(v) for v in obj["translation"]]
                rotation = [float(v) for v in obj["rotation"]]
                surface_y, _ = _base_y(sim, old_handle, translation, rotation)
                new_low, new_high = _base_y(sim, new_handle, translation, rotation)
                # Same contact plane: lift the replacement by however far its own
                # base sits below the base the authored object rested on.
                shift = surface_y - new_low
                obj["translation"] = [translation[0], translation[1] + shift, translation[2]]
                notes.append({
                    "semantic_id": sid, "from": old_handle, "to": new_handle,
                    "surface_y": round(surface_y, 5),
                    "translation_y": [round(translation[1], 5),
                                      round(translation[1] + shift, 5)],
                    "height_m": round(new_high - new_low, 4),
                })
            authoring = dict(blob.get("authoring") or {})
            authoring["substitutions"] = notes
            authoring["substituted_from"] = str(path)
            blob["authoring"] = authoring

            target = out_scene / path.relative_to(source_scene)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(blob, indent=4) + "\n", encoding="utf-8")
            written.append(target)
            for note in notes:
                print(f"{path.name:24s} {note['from']} -> {note['to']}  "
                      f"y {note['translation_y'][0]:.3f} -> {note['translation_y'][1]:.3f} "
                      f"(surface {note['surface_y']:.3f}, height {note['height_m']:.3f} m)")
    finally:
        sim.close()

    print(f"\nwrote {len(written)} layouts under {out_scene}")


if __name__ == "__main__":
    main()
