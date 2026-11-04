#!/usr/bin/env python3
"""Retrieval quality of each method's map before any navigation: over the release's 79 static
queries, is the top-ranked candidate within 1 m (horizontal) of an instance of the queried
object, and is any of the top three?

Candidates are de-duplicated as the navigator does (`static_map_nav.py`): a candidate within 1 m
of one already taken is skipped, since a map's objects are often fragments of one thing. A query
with no candidate at all (a closed-vocabulary map with no track of the label) is a miss.

    python scripts/baselines/rank_quality.py \\
        --rankings 'Khronos=outputs/baselines/khronos/map/{scene}/rankings.json' \\
        --rankings 'MD-SG=outputs/baselines/mdsg/original/{scene}.json'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.eval.dualmap_release import SCENES, STATIC_QUERIES, distance_to_target, static_target_positions  # noqa: E402

DEDUP_M = 1.0
HIT_M = 1.0


def hits(r: dict, query: str, targets) -> Tuple[int, int]:
    """(top-1 hit, any-of-top-3 hit) for one query."""
    objs = {o["uid"]: o for o in r["objects"]}
    tried, dists = [], []
    for uid, _ in r["rankings"].get(query, []):
        c = objs[uid]["center"]
        if any(np.hypot(c[0] - t[0], c[1] - t[1]) <= DEDUP_M for t in tried):
            continue
        tried.append(c)
        dists.append(distance_to_target(np.array([c[0], c[2], -c[1]]), targets)[0])  # release (z up) -> habitat
        if len(tried) == 3:
            break
    if not dists:
        return 0, 0
    return int(dists[0] <= HIT_M), int(min(dists) <= HIT_M)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankings", action="append", required=True,
                    help="'name=path' with {scene} standing for the scene id")
    args = ap.parse_args()

    print(f"{'method':24s} " + " ".join(f"{s[:5]:>11s}" for s in SCENES) + f" {'top-1':>9s} {'top-3':>9s}")
    for spec in args.rankings:
        name, _, pattern = spec.partition("=")
        h1 = h3 = n = 0
        cells = []
        for scene in SCENES:
            path = Path(pattern.format(scene=scene))
            if not path.exists():
                cells.append(f"{'--':>11s}")
                continue
            r = json.loads(path.read_text())
            s1 = s3 = 0
            for q in STATIC_QUERIES[scene]:
                a, b = hits(r, q, static_target_positions(scene, q))
                s1, s3 = s1 + a, s3 + b
            k = len(STATIC_QUERIES[scene])
            cells.append(f"{s1:>4d}/{s3:>2d}/{k:<3d}")
            h1, h3, n = h1 + s1, h3 + s3, n + k
        print(f"{name:24s} " + " ".join(cells) + f" {h1:>4d}/{n:<4d} {h3:>4d}/{n:<4d}")
    print("\nper scene: top-1 / top-3 / queries; a hit is within 1 m horizontally of an instance")


if __name__ == "__main__":
    main()
