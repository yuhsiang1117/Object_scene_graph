#!/usr/bin/env python3
"""MD-SG's prior map as a `rankings.json`, so `rank_quality.py` scores it like the baselines.

Each static query is asked of the map the way OSG's agent asks it: the detector name
`agent_query(q)` (three YCB renames, everything else unchanged), the tracks with that label that
are not blacklisted, ranked by presence then evidence (`object_layer.candidates`' presence key
with its evidence tie-break). A query whose label is not in the map gets no candidates: the map
is closed-vocabulary, and online OSG's detector would be asked for it live.
Centres go from the map's Habitat frame (y up) to the release's (z up): (x, -z, y).

    python scripts/baselines/osg_rankings.py --maps outputs/maps_released_upload/maps_released \\
        --scene 00829-QaLdnwvtxbs --out outputs/baselines/mdsg/original/00829-QaLdnwvtxbs.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.eval.dualmap_release import STATIC_QUERIES  # noqa: E402
from osg.sim.dualmap_env import agent_query  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True, help="root holding <scene>/<scene>.json")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = json.loads((args.maps / args.scene / f"{args.scene}.json").read_text())
    objects, by_label = [], {}
    for t in m["tracks"]:
        if t.get("blacklisted"):
            continue
        x, y, z = t["center"]
        uid = str(t["id"])
        a = t.get("axes", [0, 0, 0])
        objects.append({"uid": uid, "center": [x, -z, y], "dimensions": [2 * a[0], 2 * a[2], 2 * a[1]]})
        p = 1.0 / (1.0 + math.exp(-float(t.get("presence", {}).get("log_odds", 0.0))))
        by_label.setdefault(t["label"].lower().replace(" ", "_"), []).append(
            (p, float(t.get("evidence", 0.0)), uid))
    rankings, missing = {}, []
    for q in STATIC_QUERIES[args.scene]:
        tracks = sorted(by_label.get(agent_query(q).lower().replace(" ", "_"), []), reverse=True)
        if not tracks:
            missing.append(q)
        rankings[q] = [[uid, p] for p, _, uid in tracks]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"method": "MD-SG", "scene": args.scene, "map": str(args.maps / args.scene),
                                    "encoder": "label", "objects": objects, "rankings": rankings}))
    print(f"{args.scene}: {len(objects)} tracks, {len(by_label)} labels; "
          f"no track for {len(missing)}/{len(rankings)}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
