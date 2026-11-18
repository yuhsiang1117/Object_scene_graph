#!/usr/bin/env python3
"""Lay a recorded pass 1 (eval.record_object_frames_dir) out as ConceptGraphs "replica" scenes.

The object-layer recording has every keyframe's depth, pose and intrinsics but not its colour
(eval/object_recorder.py drops it). This re-renders colour, and the semantic instance image the
ground truth comes from, at each keyframe's exact camera pose in a side simulator
(osg/eval/gt_semantic.py), checks the rendered depth against the recorded one as the labeller
does, and writes per scene:

    <out>/<scene>/results/frame%06d.jpg    rendered colour
    <out>/<scene>/results/depth%06d.png    the RECORDED depth, uint16 mm; pixels the run's own
                                           clamp pinned to depth_min_m / depth_max_m are 0
                                           (ConceptGraphs' "no depth"), since they are not
                                           measurements
    <out>/<scene>/semantic/sem%06d.npz     uint32 instance id (gt_semantic's ids, which pass 65535
                                           for the injected YCB objects)
    <out>/<scene>/traj.txt                 T_wc per frame, OpenCV camera, as ReplicaDataset reads it
    <out>/<scene>/frames.json              frame i -> [episode dir, keyframe index], and the camera
    <out>/dataconfig.yaml                  ConceptGraphs dataconfig for this camera

A scene's frames are its pass-1 episodes' keyframes in the order they ran
(object_replay.scene_episodes), which is one chain: each episode starts from the previous one's
map. `+export.stride` keeps every n-th keyframe of that concatenation (ConceptGraphs' own runs
use stride 5).

    python scripts/baselines/conceptgraphs/export_recording.py +experiment=mf5_mapping_both \\
        +export.rec=outputs/objrec_v2 +export.out=outputs/baselines/cg_rec +export.stride=5
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


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import cv2

    from osg.eval import gt_semantic as gts
    from osg.eval.object_recorder import load_keyframe
    from osg.eval.object_replay import read_json, scene_episodes

    ex = cfg.get("export", {})
    rec = Path(str(ex.get("rec")))
    out = Path(str(ex.get("out")))
    stride = int(ex.get("stride", 5))
    want = [str(s) for s in ex.get("scenes", [])]
    lo, hi = float(cfg.eval.depth_min_m), float(cfg.eval.depth_max_m)

    metas = {d: read_json(d / "prior.json").get("meta", {})
             for d in sorted(rec.iterdir()) if (d / "final.json").exists()}
    scenes = sorted({m.get("scene") for m in metas.values()})
    if want:
        scenes = [s for s in scenes if s in want]
    layouts = gts.find_layouts(cfg, set(scenes))
    out.mkdir(parents=True, exist_ok=True)
    camera = None

    for scene in scenes:
        eps = scene_episodes(rec, scene)
        lay = {metas[d].get("layout_id") for d in eps}
        if len(lay) != 1:
            raise SystemExit(f"{scene}: pass 1 spans layouts {sorted(lay)}")
        key = (scene, lay.pop())
        frames = [(d, k, p) for d in eps for k, p in enumerate(sorted(d.glob("kf_*.npz")))]
        frames = frames[::stride]
        dst = out / scene
        (dst / "results").mkdir(parents=True, exist_ok=True)
        (dst / "semantic").mkdir(parents=True, exist_ok=True)
        f0, _, _ = load_keyframe(frames[0][2])
        intr = f0.intrinsics
        hfov = math.degrees(2.0 * math.atan(intr.width / (2.0 * intr.fx)))
        cam = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
               "cx": intr.cx, "cy": intr.cy}
        if camera is not None and camera != cam:
            raise SystemExit(f"{scene}: camera {cam} differs from {camera}")
        camera = cam
        sim = gts.make_sim(layouts[key], cfg, intr.width, intr.height, hfov)
        index, poses, unchecked = [], [], 0
        try:
            for i, (d, k, path) in enumerate(frames):
                frame, _dets, _ = load_keyframe(path)
                obs = gts._observe(sim, frame.T_wc)
                depth = np.asarray(obs["depth"], np.float32)
                depth = depth[..., 0] if depth.ndim == 3 else depth
                ok, frac = gts.check_depth(depth, frame.depth, depth_range=(lo, hi))
                if not ok:
                    raise SystemExit(f"{d.name}/{path.name}: rendered depth does not reproduce "
                                     f"the recording ({frac:.4f} within 1 mm)")
                unchecked += frac != frac
                rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
                sem = np.asarray(obs["semantic"]).astype(np.int64)
                rd = np.asarray(frame.depth, np.float32)
                valid = (rd > lo + 1e-2) & (rd < hi - 1e-2)
                mm = np.where(valid, np.round(rd * 1000.0), 0).astype(np.uint16)
                cv2.imwrite(str(dst / "results" / f"frame{i:06d}.jpg"), rgb[..., ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dst / "results" / f"depth{i:06d}.png"), mm)
                np.savez_compressed(dst / "semantic" / f"sem{i:06d}.npz", sem=sem.astype(np.uint32))
                poses.append(np.asarray(frame.T_wc, float).reshape(-1))
                index.append([d.name, k])
        finally:
            sim.close()
        with open(dst / "traj.txt", "w") as fh:
            for row in poses:
                fh.write(" ".join(f"{v:.9g}" for v in row) + "\n")
        (dst / "frames.json").write_text(json.dumps({"stride": stride, "camera": cam,
                                                     "layout": key[1], "frames": index}))
        print(f"{scene}: {len(frames)} frames (stride {stride}) from {len(eps)} episodes, "
              f"{unchecked} uncheckable", flush=True)

    (out / "dataconfig.yaml").write_text(
        "# ConceptGraphs dataconfig for a recording laid out by export_recording.py: the run's\n"
        "# camera and millimetre depth. Loader: ReplicaDataset.\n"
        "dataset_name: 'replica'\n"
        "camera_params:\n"
        f"  image_height: {camera['height']}\n  image_width: {camera['width']}\n"
        f"  fx: {camera['fx']!r}\n  fy: {camera['fy']!r}\n"
        f"  cx: {camera['cx']!r}\n  cy: {camera['cy']!r}\n"
        "  png_depth_scale: 1000.0\n  crop_edge: 0\n")


if __name__ == "__main__":
    main()
