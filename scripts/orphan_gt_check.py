"""P1i follow-up #2: door/cabinet/shelf had the highest orphan rates in
orphan_node_check.py (89%/56%/48% respectively) -- is that genuinely many
different real instances each glimpsed once (there ARE a lot of doors and
shelves in a house), or detector noise that never recurs? detector_gt_check.py
answered this question for ObjectNav *target* categories using
episode.goals[i].object_id as ground truth, but door/cabinet/shelf are never
ObjectNav targets, so there's no "goals" list to anchor on. Instead, for each
of these categories we build the ground-truth instance set directly from
every semantic-scene object whose raw HM3D category name plausibly means
that category (substring match against a small hand-picked synonym list --
see CATEGORY_SYNONYMS -- since these are architectural categories, not an
ObjectNav-defined class with a fixed goal list).

Reuses SemanticHabitatEnv/classify_frame from detector_gt_check.py (which
already initializes hydra config at import time -- do not also call
compose() here).

Usage: python scripts/orphan_gt_check.py --num-episodes 4 --max-steps 300
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from detector_gt_check import SemanticHabitatEnv, cfg, classify_frame
from osg.agent.nav_agent import NavAgent
from osg.eval.runner import _unload_ollama_models, build_detector, build_scorer, build_verifier

# Suffix match against sem_scene.objects' raw category names (not plain
# substring: "garage door opener motor"/"railing" contain "door" but aren't
# doors; "door frame" is the opening's frame, not the door itself -- neither
# ends with " door"). "clothes on shelf"/"shoes on shelf" are excluded from
# "shelf" for the same reason (items resting on a shelf, not the shelf
# structure). All judgment calls, not ground truth from the dataset itself;
# noted here so the choice is inspectable.
CATEGORIES = {
    "door": lambda name: name == "door" or name.endswith(" door"),
    "cabinet": lambda name: name == "cabinet" or name.endswith(" cabinet"),
    "shelf": lambda name: name.endswith("shelf") and "on shelf" not in name,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--categories", default=None,
                     help="comma-separated subset of door,cabinet,shelf (default: all)")
    ap.add_argument("--dump-dir", default=None,
                     help="directory to save TP and FP-hallucination crops to (labeled with GT category)")
    args = ap.parse_args()
    wanted_cats = args.categories.split(",") if args.categories else list(CATEGORIES)
    dump_dir = None
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    _unload_ollama_models(cfg)
    env = SemanticHabitatEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)

    stats = defaultdict(lambda: defaultdict(int))
    ious = defaultdict(list)
    offsets = defaultdict(list)
    matched_raw_names = defaultdict(set)

    n_total = len(env.env.episodes)
    n_run = min(args.num_episodes, n_total)
    for ep_i in range(n_run):
        frame = env.reset()
        episode = env.current_episode
        target = env.target_category()
        agent = NavAgent(cfg, detector, scorer, verifier, target, profiler=None)

        sem_scene = env.env.sim.semantic_scene
        by_semid = {o.semantic_id: o for o in sem_scene.objects}
        cat_ids = {}
        for cat in wanted_cats:
            match_fn = CATEGORIES[cat]
            ids = set()
            for o in sem_scene.objects:
                name = o.category.name()
                if match_fn(name):
                    ids.add(o.semantic_id)
                    matched_raw_names[cat].add(name)
            cat_ids[cat] = ids

        print(f"--- ep={episode.episode_id} target={target} "
              + " ".join(f"{c}={len(cat_ids[c])}" for c in wanted_cats) + " instances in scene ---")

        step = 0
        while not env.episode_over and step < args.max_steps:
            action = agent.act(frame)
            frame = env.step(action)
            step += 1
            if step % args.sample_every != 0:
                continue
            sem_ids = env.last_obs["semantic"][..., 0]
            dets = detector.detect(frame.rgb)
            for cat, ids in cat_ids.items():
                verdict, iou, offset = classify_frame(
                    sem_ids, ids, ids, dets, cat,
                    by_semid=by_semid, dump_dir=dump_dir,
                    dump_prefix=f"{cat}_ep{episode.episode_id}_step{step}",
                    full_frame_rgb=frame.rgb,
                )
                stats[cat][verdict] += 1
                if iou is not None and verdict == "TP":
                    ious[cat].append(iou)
                if offset is not None:
                    offsets[cat].append(offset)

    env.close()
    scorer.shutdown()

    print()
    print("=== matched raw HM3D category names per bucket ===")
    for cat, names in matched_raw_names.items():
        print(f"  {cat}: {sorted(names)}")

    print()
    print(f"{'category':10s} {'TP':>4s} {'FN':>4s} {'FP-wrong':>9s} {'FP-halluc':>10s} {'TN':>5s} "
          f"{'recall':>7s} {'mean_IoU':>9s} {'mean_off_px':>11s}")
    for cat in wanted_cats:
        s = stats[cat]
        tp, fn = s["TP"], s["FN"]
        fp_wrong, fp_halluc, tn = s["FP-wrong-instance"], s["FP-hallucination"], s["TN"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        mean_iou = float(np.mean(ious[cat])) if ious[cat] else float("nan")
        mean_off = float(np.mean(offsets[cat])) if offsets[cat] else float("nan")
        print(f"{cat:10s} {tp:4d} {fn:4d} {fp_wrong:9d} {fp_halluc:10d} {tn:5d} "
              f"{recall:7.3f} {mean_iou:9.3f} {mean_off:11.1f}")


if __name__ == "__main__":
    main()
