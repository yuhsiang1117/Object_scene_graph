"""How many rooms does the segmenter actually find, and at what parameters?

Measured on 12 dev50 episodes: median **1** room per episode, only 5/12 with more
than one, and the two episodes that ran the full 500 steps still segmented into
1-2 rooms. Half the time "which room is this frontier in" has a single answer, so
the room level of the scene graph carries no information -- and any prompt built
on it hands the model "every area is in the same room", the same shape of
failure as S10's "every area is unknown room".

The suspected mechanism is the erosion. `VoronoiRoomSegmenter` erodes free space
by `erode_iters` 3x3 passes (12 -> 0.6 m at 0.05 m/cell) to sever doorways, then
regrows. A partially explored costmap's free space is a narrow region carved
along the trajectory, so a 0.6 m erosion can destroy every core except the
widest, after which the regrow step assigns the entire map to that one core.

There is no ground-truth room annotation to score against, so the acceptance bar
is a proxy and is stated as such: **an episode that ran the full 500 steps should
segment into at least 3 rooms.** A house explored for 500 steps that reads as one
room is wrong regardless of what the true count is.

Usage:
    python scripts/run_eval.py ... eval.save_costmap=true output_dir=outputs/probe
    python scripts/measure_room_seg.py outputs/probe
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.mapping.costmap import Costmap2D  # noqa: E402
from osg.mapping.room_seg import VoronoiRoomSegmenter  # noqa: E402

ERODE = [4, 6, 8, 10, 12]
MIN_CELLS = [30, 60, 120]
FULL_BUDGET = 500


def load_costmap(path: Path):
    d = np.load(path, allow_pickle=False)
    cm = Costmap2D(resolution=float(d["resolution"]), size_m=1.0)
    cm.grid = d["grid"]
    cm.origin = d["origin"]
    return cm, int(d["steps"])


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/probe") / "costmaps"
    files = sorted(root.glob("*.npz"))
    if not files:
        sys.exit(f"no costmaps in {root} -- run with eval.save_costmap=true")

    maps = [load_costmap(f) for f in files]
    exhausted = [i for i, (_, s) in enumerate(maps) if s >= FULL_BUDGET]
    print(f"{len(maps)} costmaps, {len(exhausted)} ran the full {FULL_BUDGET} steps\n")

    print(f"{'erode':>6} {'min_cells':>10} {'median':>7} {'mean':>6} {'max':>4} "
          f"{'>1 room':>8} {'exhausted>=3':>13}")
    for erode in ERODE:
        for min_cells in MIN_CELLS:
            counts = []
            for cm, _ in maps:
                seg = VoronoiRoomSegmenter(min_room_cells=min_cells, erode_iters=erode)
                labels = seg.segment(cm)
                counts.append(len(set(np.unique(labels).tolist()) - {0}))
            multi = sum(1 for c in counts if c > 1)
            ok = (sum(1 for i in exhausted if counts[i] >= 3) if exhausted else 0)
            flag = ""
            if exhausted and ok == len(exhausted):
                flag = "  <-- passes the bar"
            print(f"{erode:>6} {min_cells:>10} {st.median(counts):>7.1f} "
                  f"{st.mean(counts):>6.1f} {max(counts):>4} "
                  f"{multi:>4}/{len(counts):<3} {ok:>6}/{len(exhausted):<6}{flag}")

    print("\nNo ground-truth room count exists, so 'at least 3 rooms after a full "
          "500-step episode' is a proxy for 'not under-segmenting', not an "
          "accuracy measure.")


if __name__ == "__main__":
    main()
