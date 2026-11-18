"""Draw a stored ASCENT map with an episode's target marked on it.

Runs produce this automatically now (`eval.obstacle_map_png`, on by default,
written to `viz/obstacle_map_<scene>.png` whenever the run read
`ycb.obstacle_map_in`). This is the same renderer as a command, for looking at
an older run or a different maps directory than the one it was run with.

    python scripts/legacy/plot_cross_anchor_map.py \\
        --maps outputs/maps_mf5_ascent --scene 00800-TEEsavR23oF \\
        --run outputs/mf5_pass2_v4 --out /tmp/map.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from osg.eval.obstacle_map_viz import plot_obstacle_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="directory of <scene>.json/.npz")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True, help="a run directory with episodes.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="",
                    help="comma-separated storey heights, bottom-up; "
                         "default: taken from the run's own floor_log")
    args = ap.parse_args()

    episodes = [json.loads(line) for line in
                (Path(args.run) / "episodes.jsonl").read_text().splitlines()
                if line.strip()]
    scoped = [e for e in episodes if str(e.get("scene", "")) == args.scene] or episodes
    heights = [float(v) for v in args.heights.split(",")] if args.heights else None
    written = plot_obstacle_map(args.maps, args.scene, scoped, args.out, heights=heights)
    print(f"wrote {written}" if written else "nothing to draw: no stored storeys")


if __name__ == "__main__":
    main()
