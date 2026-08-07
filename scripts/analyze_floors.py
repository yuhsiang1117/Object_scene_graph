"""Decompose a run's SR by floor structure, and audit the floor estimator.

Answers the three questions Stage 0/1 of the multi-floor work exists to answer
(docs/MULTI_FLOOR.md):

  1. **Where is the SR lost?** SR/SPL split by `floor_class` -- `same_floor`
     (the goal category has an instance on the start floor) vs `cross_floor`
     (the stairs are mandatory).

  2. **Does the floor estimator hallucinate floors?** `n_floors_seen` per
     episode against the per-scene ground truth from `scripts/scene_floors.py`.
     A count ABOVE ground truth means phantom floors -- most likely staircase
     landings, the failure mode that would make per-floor costmaps worse than
     the current collapse. Below is benign (unvisited floors).

  3. **Does YOLOE ever see stairs?** Total `n_stair_tracks`. Stage 4's semantic
     stair detection assumes these exist; if they are ~0 the geometric height
     -difference check has to carry it alone.

Usage:
    python scripts/analyze_floors.py outputs/<dir>/
    python scripts/analyze_floors.py outputs/<dir>/ --floors data/floor_classes.json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict


def load_episodes(run_dir):
    path = os.path.join(run_dir, "episodes.jsonl")
    if not os.path.exists(path):
        cands = glob.glob(os.path.join(run_dir, "**", "episodes.jsonl"), recursive=True)
        if not cands:
            sys.exit(f"no episodes.jsonl under {run_dir}")
        path = cands[0]
    return [json.loads(l) for l in open(path) if l.strip()], path


def scene_key(rec):
    """`XB4GS9ShBRE.basis.glb` -> `XB4GS9ShBRE`."""
    return str(rec.get("scene", "")).split(".")[0]


def block(title, rows, cols):
    print(f"\n{title}")
    print("  " + "".join(f"{c:>{w}}" for c, w in cols))
    for r in rows:
        print("  " + "".join(f"{v:>{w}}" for v, (_, w) in zip(r, cols)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--floors", default=None,
                    help="scene_floors.py --out JSON, for the ground-truth floor count")
    args = ap.parse_args()

    eps, path = load_episodes(args.run_dir)
    print(f"{len(eps)} episodes from {path}")

    truth = {}
    if args.floors and os.path.exists(args.floors):
        data = json.load(open(args.floors))
        for scene, meta in data.get("scenes", {}).items():
            truth[scene] = meta.get("navmesh_n_floors") or meta.get("n_floors")

    # ------------------------------------------------------ 1. SR by floor class
    by_fc = defaultdict(list)
    for r in eps:
        by_fc[r.get("floor_class", "unknown")].append(r)
    rows = []
    for fc, rs in sorted(by_fc.items()):
        n = len(rs)
        rows.append([
            fc, n,
            f"{100.0 * sum(r.get('success', 0) for r in rs) / n:.1f}",
            f"{sum(r.get('spl', 0) for r in rs) / n:.3f}",
            f"{sum(r.get('distance_to_goal', 0) for r in rs) / n:.2f}",
            f"{st.mean(r.get('floor_changes', 0) for r in rs):.2f}",
        ])
    block("SR by floor class", rows, [
        ("class", 20), ("n", 5), ("SR%", 7), ("SPL", 7), ("dtg", 7), ("fl.chg", 8)])

    # ------------------------------------- 2. estimator vs ground-truth floors
    print("\nFloor estimator audit (n_floors_seen vs scene ground truth)")
    seen_counter = Counter(r.get("n_floors_seen", 0) for r in eps)
    print("  n_floors_seen histogram:",
          ", ".join(f"{k}:{v}" for k, v in sorted(seen_counter.items())))
    if truth:
        over, under, exact, unknown = [], [], 0, 0
        for r in eps:
            gt = truth.get(scene_key(r))
            seen = r.get("n_floors_seen", 0)
            if gt is None:
                unknown += 1
            elif seen > gt:
                over.append((scene_key(r), r["episode_id"], seen, gt))
            elif seen < gt:
                under.append((scene_key(r), r["episode_id"], seen, gt))
            else:
                exact += 1
        print(f"  exact {exact} | under (unvisited floors, benign) {len(under)} "
              f"| OVER (phantom floors, harmful) {len(over)} | no ground truth {unknown}")
        if over:
            print("  phantom-floor episodes (likely staircase landings):")
            for scene, ep, seen, gt in over[:15]:
                print(f"    {scene:<16} ep{ep:<6} saw {seen}  truth {gt}")
    else:
        print("  (pass --floors to compare against scene_floors.py ground truth)")

    changed = [r for r in eps if r.get("floor_transitions", 0) > 0]
    print(f"  episodes with a committed floor transition: {len(changed)}/{len(eps)}")

    # ---------------------------------------------------- 3. stair detections
    # Multi-floor episodes only: single-floor scenes have no stairs to detect,
    # so including them understates the detection rate.
    multi_eps = [r for r in eps if truth.get(scene_key(r), 1) > 1] if truth else []
    total = sum(r.get("n_stair_tracks", 0) for r in eps)
    with_any = sum(1 for r in eps if r.get("n_stair_tracks", 0) > 0)
    print(f"\nYOLOE stair tracks: {total} across {len(eps)} episodes "
          f"({with_any} episodes with >=1, {100.0 * with_any / max(1, len(eps)):.0f}%)")
    scores = [t.get("best_score", 0) for r in eps for t in r.get("stair_tracks", [])]
    obs = [t.get("n_obs", 0) for r in eps for t in r.get("stair_tracks", [])]
    if scores:
        print(f"  best_score  median {st.median(scores):.3f}  max {max(scores):.3f}")
        print(f"  n_obs       median {st.median(obs):.1f}  max {max(obs)}")
    # The verdict is about RATE, not existence: a stair track that appears in
    # one episode in twelve cannot drive a floor-transition decision. Judge it
    # on the multi-floor episodes only -- single-floor scenes have no stairs to
    # detect, so including them understates the rate.
    pool = multi_eps if multi_eps else eps
    rate = sum(r.get("n_stair_tracks", 0) for r in pool) / max(1, len(pool))
    hit = sum(1 for r in pool if r.get("n_stair_tracks", 0) > 0) / max(1, len(pool))
    print(f"  on multi-floor episodes: {rate:.2f} tracks/episode, "
          f"{100 * hit:.0f}% of episodes see >=1")
    if hit >= 0.5:
        print("  => semantic stair detection is viable; fuse it with the dh check")
    elif hit > 0.0:
        print("  => TOO SPARSE to drive floor transitions on its own. Lead with the")
        print("     geometric dh check; use the tracks only to confirm.")
    else:
        print("  => NO stair tracks: drop the semantic half of Stage 4 and rely")
        print("     on the geometric height-difference (dh) check alone")

    # ----------------------------------------------- multi-floor scene detail
    if multi_eps:
        n = len(multi_eps)
        sr = 100.0 * sum(r.get("success", 0) for r in multi_eps) / n
        single = [r for r in eps if truth.get(scene_key(r), 1) <= 1]
        print(f"\nBy SCENE structure: multi-floor {n} eps SR {sr:.1f}%", end="")
        if single:
            print(f" | single-floor {len(single)} eps SR "
                  f"{100.0 * sum(r.get('success', 0) for r in single) / len(single):.1f}%")
        else:
            print()


if __name__ == "__main__":
    main()
