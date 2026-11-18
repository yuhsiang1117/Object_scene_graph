#!/usr/bin/env python3
"""What each benchmark object looks like to the agent, at fixed ranges, and
what both detectors make of it.

For one released trial per object, the object is placed by the protocol env
exactly as in the benchmark, the camera is put on the navmesh at 0.6-3 m from
it facing it (the agent's own height and optics), and the frame is rendered.
Each frame is run through our detector (with the vocabulary the agent uses for
that query) and through DualMap's (`yolov8l-world.pt`, its own class list), and
the best score of the right label covering the object's pixel is recorded.
Writes per-object PNGs (full frame with the object circled, a zoom crop), one
montage per object, and `views.json`.

    python scripts/render_asset_views.py +experiment=dualmap_protocol_osg_look_flat_anchor_v2_island_close \
        'dualmap.scenes=[00848-ziup5kvtCCR]' 'dualmap.conditions=[in_anchor]' \
        'dualmap.trial_ids=[00848-ziup5kvtCCR__in_anchor__0116__scissors,...]' \
        +render.out=outputs/asset_evidence
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

from osg.core.config import register_configs

register_configs()

RANGES_M = (0.6, 1.0, 1.5, 2.0, 3.0)
VENDOR = Path("outputs/dualmap_comparison/vendor/DualMap")
THEIR_WEIGHTS = VENDOR / "model/yolov8l-world.pt"
THEIR_CLASSES = VENDOR / "config/class_list/hm3d300_classes_ycb.txt"
THEIR_NAME = {"tin can": "soup can", "red plate": "plate", "blue plastic pitcher": "pitcher"}
# Longest extent of each YCB asset (m), from the meshes; the circle drawn.
EXTENT_M = {"scissors": 0.202, "mug": 0.117, "bowl": 0.161, "cracker box": 0.213,
            "plate": 0.261, "pitcher": 0.242, "soup can": 0.102, "banana": 0.178,
            "mustard bottle": 0.19, "coffee can": 0.14, "toy airplane": 0.23, "bleach bottle": 0.25}


def hz(a, b):
    return float(math.hypot(a[0] - b[0], a[2] - b[2]))


def covers(box, u, v, pad=8.0):
    x1, y1, x2, y2 = [float(c) for c in box]
    return x1 - pad <= u <= x2 + pad and y1 - pad <= v <= y2 + pad


def project(frame, point):
    k = frame.intrinsics
    p = frame.T_cw[:3, :3] @ np.asarray(point, dtype=float) + frame.T_cw[:3, 3]
    z = float(p[2])
    if z <= 1e-3:
        return None
    u = k.fx * p[0] / z + k.cx
    v = k.fy * p[1] / z + k.cy
    if not (0 <= u < k.width and 0 <= v < k.height):
        return None
    return float(u), float(v), z


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    from ultralytics import YOLOWorld

    from osg.perception.vocabulary import target_vocabulary
    from osg.pipeline.components import build_detector, build_env

    out = Path(str(cfg.render.out))
    out.mkdir(parents=True, exist_ok=True)
    env = build_env(cfg)
    detector = build_detector(cfg)
    their_names = [n.strip() for n in THEIR_CLASSES.read_text().splitlines() if n.strip()]
    # A swapped-in asset has no entry in DualMap's list; its rerun would add
    # one, so the test gives its detector the same name.
    for trial in env.trials:
        name = THEIR_NAME.get(trial["query"], trial["query"])
        if name not in their_names:
            their_names.append(name)
    theirs = YOLOWorld(str(THEIR_WEIGHTS))
    theirs.set_classes(their_names)

    results = []
    for _ in range(len(env.trials)):
        env.reset()
        ep = env.env.current_episode
        info = ep.info["ycb"]
        query, label = info["query"], info["agent_query"]
        target = np.asarray(info["target_position"], dtype=float)
        sim = env.env.sim
        pf = sim.pathfinder
        start = np.asarray(sim.get_agent_state().position, dtype=np.float32)
        island = int(pf.get_island(pf.snap_point(start)))
        detector.set_vocabulary(target_vocabulary(label, cfg.detector.vocabulary))
        their_label = THEIR_NAME.get(label, label)
        odir = out / query.replace(" ", "_")
        odir.mkdir(exist_ok=True)
        print(f"# {query} ({ep.episode_id}) target {np.round(target, 2)}", flush=True)
        tiles = []
        for d in RANGES_M:
            best = None
            for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                p = np.array([target[0] + d * math.cos(ang), start[1], target[2] + d * math.sin(ang)],
                             dtype=np.float32)
                s = pf.snap_point(p, island_index=island)
                if np.isnan(s).any() or abs(hz(s, target) - d) > 0.2:
                    continue
                dx, dz = target[0] - s[0], target[2] - s[2]
                rot = quat_from_angle_axis(math.atan2(-dx, -dz), np.array([0.0, 1.0, 0.0]))
                obs = sim.get_observations_at(position=s, rotation=rot, keep_agent_at_new_pose=True)
                frame = env._to_frame(obs)
                uv = project(frame, target)
                if uv is None:
                    continue
                u, v, z = uv
                dep = float(frame.depth[int(v), int(u)])
                occluded = dep > 1e-3 and dep < z - 0.25
                err = abs(hz(s, target) - d)
                cand = (occluded, err, s, rot, frame, (u, v, z))
                if best is None or cand[:2] < best[:2]:
                    best = cand
                if not occluded and err < 0.1:
                    break
            if best is None:
                print(f"  {d} m: no navigable pose", flush=True)
                continue
            occluded, err, s, rot, frame, (u, v, z) = best
            rgb = frame.rgb
            ours = detector.detect(rgb)
            our_best = max((float(det.score) for det in ours
                            if str(det.label).lower() == label.lower() and covers(det.bbox_xyxy, u, v)),
                           default=0.0)
            our_any = sorted(((float(det.score), str(det.label)) for det in ours if covers(det.bbox_xyxy, u, v)),
                             reverse=True)[:3]
            res = theirs.predict(rgb, imgsz=int(cfg.detector.imgsz), verbose=False)[0]
            their_best = max((float(c) for b, k, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                                          res.boxes.conf.tolist())
                              if their_names[int(k)] == their_label and covers(b, u, v)), default=0.0)
            their_any = sorted(((float(c), their_names[int(k)]) for b, k, c in
                                zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())
                                if covers(b, u, v)), reverse=True)[:3]
            # Which name would our detector answer to? Optional `render.names`
            # runs each as the sole vocabulary on this frame.
            name_scores = {}
            for name in list(cfg.render.get("names", []) or []):
                detector.set_vocabulary([str(name)])
                name_scores[str(name)] = round(max((float(det.score) for det in detector.detect(rgb)
                                                    if covers(det.bbox_xyxy, u, v)), default=0.0), 3)
            if name_scores:
                detector.set_vocabulary(target_vocabulary(label, cfg.detector.vocabulary))
            r_px = frame.intrinsics.fx * (0.5 * EXTENT_M.get(query, 0.15)) / z
            cv2.imwrite(str(odir / f"raw_d{d:.1f}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.circle(bgr, (int(u), int(v)), int(max(r_px, 6)), (0, 0, 255), 2)
            tag = f"{query} @ {hz(s, target):.2f} m  ours {our_best:.2f}  DualMap {their_best:.2f}"
            cv2.putText(bgr, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(bgr, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(odir / f"d{d:.1f}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            half = 128
            x0, y0 = int(max(0, u - half)), int(max(0, v - half))
            crop = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[y0:y0 + 2 * half, x0:x0 + 2 * half]
            crop = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"{hz(s, target):.1f} m", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(crop, f"{hz(s, target):.1f} m", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(crop, f"ours {our_best:.2f} DM {their_best:.2f}", (8, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(crop, f"ours {our_best:.2f} DM {their_best:.2f}", (8, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(odir / f"zoom_d{d:.1f}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            tiles.append(crop)
            row = {"query": query, "trial_id": ep.episode_id, "range_m": round(hz(s, target), 2),
                   "occluded": bool(occluded), "pixel": [round(u), round(v)], "object_radius_px": round(r_px, 1),
                   "ours_score": round(our_best, 3), "ours_top": our_any,
                   "dualmap_score": round(their_best, 3), "dualmap_top": their_any,
                   "name_scores": name_scores}
            results.append(row)
            print("  ", json.dumps(row), flush=True)
        if tiles:
            cv2.imwrite(str(odir / "montage.jpg"), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 85])
    prev = out / "views.json"
    old = json.loads(prev.read_text()) if prev.is_file() else []
    keep = [r for r in old if r["query"] not in {r2["query"] for r2 in results}]
    prev.write_text(json.dumps(keep + results, indent=1))
    print(f"wrote {prev}")


if __name__ == "__main__":
    main()
