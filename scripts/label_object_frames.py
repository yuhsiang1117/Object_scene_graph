#!/usr/bin/env python3
"""Label a recorded object-layer run with instance ground truth.

For every episode directory under `+label.root` (eval.record_object_frames_dir
of the recorded run), re-render the scene at each keyframe's camera pose in a
side simulator (osg/eval/gt_semantic.py), check that the rendered depth IS the
recorded depth, and write beside the recording:

    gt.json     {"kf": [[[instance id or null, purity], ...per detection], ...per keyframe],
                 "depth_check": [fraction within 1 mm, ...]}
    <root>/_instances/<scene>__<layout_id>.json   every instance's AABB

Compose with the preset the recording was made under so the layouts resolve:

    python scripts/label_object_frames.py +experiment=mf5_mapping_both \\
        +label.root=outputs/objrec_v2
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from osg.eval import gt_semantic as gts
    from osg.eval.object_recorder import load_keyframe

    lab = cfg.get("label", {})
    root = Path(str(lab.get("root")))
    redo = bool(lab.get("redo", False))
    min_frac = float(lab.get("min_depth_frac", 0.99))
    # final.json is written when an episode is over: a directory without it
    # is still being recorded.
    eps = sorted(d for d in root.iterdir() if (d / "final.json").exists())
    want = [str(s) for s in lab.get("scenes", [])]
    metas = {d: json.loads((d / "prior.json").read_text()).get("meta", {}) for d in eps}
    if want:
        eps = [d for d in eps if metas[d].get("scene") in want]
    layouts = gts.find_layouts(cfg, {m.get("scene") for m in metas.values()})
    inst_dir = root / "_instances"
    inst_dir.mkdir(exist_ok=True)

    sims = {}
    try:
        for d in eps:
            if (d / "gt.json").exists() and not redo:
                continue
            meta = metas[d]
            key = (meta.get("scene"), meta.get("layout_id"))
            layout = layouts.get(key)
            if layout is None:
                raise SystemExit(f"{d.name}: no layout {key}; known: {sorted(layouts)[:6]}...")
            kfs = sorted(d.glob("kf_*.npz"))
            if not kfs:
                continue
            frame0, _, _ = load_keyframe(kfs[0])
            intr = frame0.intrinsics
            hfov = math.degrees(2.0 * math.atan(intr.width / (2.0 * intr.fx)))
            if key not in sims:
                for k in list(sims):
                    sims.pop(k).close()
                sims[key] = gts.make_sim(layout, cfg, intr.width, intr.height, hfov)
                inst = gts.instances(sims[key])
                (inst_dir / f"{key[0]}__{key[1]}.json").write_text(json.dumps({
                    str(i): {"category": g.category, "center": g.center.tolist(),
                             "half": g.half.tolist(), "source": g.source}
                    for i, g in inst.items()}))
                print(f"{key}: {len(inst)} instances "
                      f"({sum(g.source == 'ycb' for g in inst.values())} ycb)", flush=True)
            sim = sims[key]
            out = {"kf": [], "depth_check": []}
            for path in kfs:
                frame, dets, _ = load_keyframe(path)
                depth, sem = gts.render_at(sim, frame.T_wc)
                ok, frac = gts.check_depth(depth, frame.depth, min_frac=min_frac,
                                           depth_range=(float(cfg.eval.depth_min_m),
                                                        float(cfg.eval.depth_max_m)))
                if not ok:
                    raise SystemExit(f"{d.name}/{path.name}: rendered depth does not reproduce "
                                     f"the recording ({frac:.4f} within 1 mm) -- pose conversion?")
                out["depth_check"].append(None if frac != frac else round(frac, 5))
                out["kf"].append([[i, round(p, 4)] for i, p in
                                  gts.label_masks(sem, [x.mask for x in dets])])
            (d / "gt.json").write_text(json.dumps(out))
            n = sum(len(k) for k in out["kf"])
            hit = sum(1 for k in out["kf"] for i, _ in k if i is not None)
            checked = [c for c in out["depth_check"] if c is not None]
            print(f"{d.name}: {len(kfs)} kf, {n} dets, {hit} labelled, "
                  f"depth match min {min(checked) if checked else float('nan'):.4f}, "
                  f"{len(kfs) - len(checked)} kf uncheckable (fully clamped)", flush=True)
    finally:
        for sim in sims.values():
            sim.close()


if __name__ == "__main__":
    main()
