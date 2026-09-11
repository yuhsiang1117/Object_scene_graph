#!/usr/bin/env python
"""Per-class score calibration for a target detector, at the dataset's own viewpoints.

    bash scripts/serve_perception.sh
    python scripts/calibrate_detector.py --scenes 6 --per-class 40

A single global confidence threshold prices every class the same, and detectors
do not. This places the agent at the goal VIEW POINTS the episode dataset ships
-- positions from which the object is, by construction, visible -- turns it to
face the object, and records what the detector says about the target class.
Negatives come from random navigable poses in the same scene.

Why it matters here: ASCENT thresholds target detections at `coco_threshold`
0.8, and porting that number wholesale (`ascentnav_perception`) left 13 of 19
never-committed episodes with no detection at all in 500 steps, 9 of them
`toilet`. A 60-frame probe of one scene showed why -- chair/couch/bed peak
around 0.95 while toilet/tv/potted plant peak at 0.73/0.48/0.47 -- so the flat
bar is not one bar, it is six different bars wearing the same number.

Prints a table of per-class true/false score distributions and the threshold
that maximises Youden's J, ready to paste into `detector.class_conf`.
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import json
import random
import urllib.request
from collections import defaultdict

import cv2
import numpy as np

HM3D_TO_COCO = {
    "chair": "chair", "bed": "bed", "toilet": "toilet",
    "tv_monitor": "tv", "sofa": "couch", "plant": "potted plant",
}


def dfine(url: str, rgb: np.ndarray, timeout: float = 30.0) -> dict:
    ok, buf = cv2.imencode(".jpg", rgb)
    body = json.dumps({"image": base64.b64encode(buf.tobytes()).decode()}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def best_for(resp: dict, coco_name: str) -> float:
    out = 0.0
    for ph, lg in zip(resp.get("phrases", []), resp.get("logits", [])):
        if str(ph).strip().lower() == coco_name:
            out = max(out, float(lg))
    return out


def youden(pos: np.ndarray, neg: np.ndarray) -> tuple:
    best = (0.5, -1.0)
    for thr in np.arange(0.10, 0.96, 0.01):
        j = float((pos >= thr).mean()) - float((neg >= thr).mean())
        if j > best[1]:
            best = (float(thr), j)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root",
                    default="data/datasets/objectnav/hm3d/v1/val/content")
    ap.add_argument("--scenes-dir", default="data/scene_datasets/hm3d/val")
    ap.add_argument("--url", default="http://localhost:13186/dfine")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--negatives", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim

    rng = random.Random(args.seed)
    files = sorted(glob.glob(f"{args.episodes_root}/*.json.gz"))
    rng.shuffle(files)
    pos: dict = defaultdict(list)
    neg: dict = defaultdict(list)

    for f in files[: args.scenes]:
        d = json.load(gzip.open(f))
        stem = f.split("/")[-1].replace(".json.gz", "")
        hit = glob.glob(f"{args.scenes_dir}/*-{stem}/{stem}.basis.glb")
        if not hit:
            continue
        cfg = habitat_sim.Configuration(
            habitat_sim.SimulatorConfiguration(), [habitat_sim.agent.AgentConfiguration()])
        cfg.sim_cfg.scene_id = hit[0]
        cfg.sim_cfg.gpu_device_id = 0
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.resolution, spec.hfov = "rgb", [480, 640], 79
        spec.position = [0.0, 0.88, 0.0]
        cfg.agents[0].sensor_specifications = [spec]
        sim = habitat_sim.Simulator(cfg)
        agent = sim.get_agent(0)

        def shoot(position, look_at=None) -> np.ndarray:
            st = agent.get_state()
            st.position = position
            if look_at is not None:
                v = np.asarray(look_at, float) - np.asarray(position, float)
                yaw = float(np.arctan2(-v[0], -v[2]))
                st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            agent.set_state(st)
            return sim.get_sensor_observations()["rgb"][:, :, :3]

        for key, goals in (d.get("goals_by_category") or {}).items():
            cat = key.split("_")[-1]
            coco = HM3D_TO_COCO.get(cat)
            if coco is None:
                continue
            vps = [(vp["agent_state"]["position"], g["position"])
                   for g in goals for vp in g.get("view_points", [])]
            if not vps:
                continue
            rng.shuffle(vps)
            for p, obj in vps[: args.per_class]:
                pos[cat].append(best_for(dfine(args.url, shoot(p, obj)), coco))

        for _ in range(args.negatives):
            p = sim.pathfinder.get_random_navigable_point()
            resp = dfine(args.url, shoot(p))
            for cat, coco in HM3D_TO_COCO.items():
                neg[cat].append(best_for(resp, coco))
        sim.close()

    print(f"\n{'class':12s} {'n+':>5s} {'n-':>5s} {'p50+':>7s} {'p75+':>7s} "
          f"{'p95-':>7s} {'rec@0.8':>8s} {'thr*':>6s} {'rec@thr*':>9s} {'fpr@thr*':>9s}")
    picked = {}
    for cat in sorted(pos):
        p, n = np.array(pos[cat]), np.array(neg[cat])
        if len(p) == 0 or len(n) == 0:
            continue
        thr, _ = youden(p, n)
        picked[cat] = round(thr, 2)
        print(f"{cat:12s} {len(p):5d} {len(n):5d} {np.median(p):7.2f} "
              f"{np.percentile(p, 75):7.2f} {np.percentile(n, 95):7.2f} "
              f"{float((p >= 0.8).mean()):8.2f} {thr:6.2f} "
              f"{float((p >= thr).mean()):9.2f} {float((n >= thr).mean()):9.2f}")
    # The negatives are random navigable poses, and a random pose in a house
    # genuinely contains a chair -- so `fpr` here is contaminated and Youden's J
    # is only meaningful for classes that are actually rare (toilet). What is
    # clean is the POSITIVE side: these are the dataset's own view points, so
    # recall here is recall on views the episode guarantees are of the object.
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nrecall at goal view points, by threshold")
    print("  " + f"{'class':12s}" + "".join(f"{t:>7.2f}" for t in grid))
    for cat in sorted(pos):
        p_ = np.array(pos[cat])
        print("  " + f"{cat:12s}" + "".join(f"{float((p_ >= t).mean()):7.2f}" for t in grid))

    print("\nclass_conf:")
    for cat, thr in picked.items():
        print(f"  {cat}: {thr}")


if __name__ == "__main__":
    main()
