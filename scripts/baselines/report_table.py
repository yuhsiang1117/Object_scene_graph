#!/usr/bin/env python3
"""Table I of the paper -- SR by scene and condition -- for any set of methods run on the release.

Every run is a directory of `trials/<trial_id>/result.json` in the layout
`run_dualmap_released_native.py` and `scripts/baselines/static_map_nav.py` both write, one
directory per collector seed. A method's cell is the mean over its seeds, with the spread
(min-max) beside it when there is more than one; a trial a run did not finish counts as missing,
not as a failure, and the cell says how many it has.

    python scripts/baselines/report_table.py \\
        --run 'DualMap (rerun)=outputs/dualmap_official_bench/seed*' \\
        --run 'Khronos=outputs/baselines/khronos/nav/seed*' \\
        --static-map Khronos --out outputs/baselines/TABLE.md

`--static-map` marks the methods that never update their map, so their dynamic rows read as
what they are: the agent going where the static map says the object was.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from osg.eval.dualmap_release import CONDITIONS, SCENES, protocol  # noqa: E402

LABEL = {"static": "Static", "in_anchor": "Dynamic (In-anchor)", "cross_anchor": "Dynamic (Cross-anchor)"}


def load(run: Path) -> Dict[str, int]:
    out = {}
    for path in (run / "trials").glob("*/result.json"):
        result = json.loads(path.read_text())
        out[result["trial_id"]] = int(result["success"])
    return out


def cell(values: List[float]) -> str:
    if not values:
        return "--"
    m = 100 * mean(values)
    if len(values) == 1:
        return f"{m:.1f}%"
    return f"{m:.1f}% ({100 * min(values):.0f}-{100 * max(values):.0f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="'name=glob' of seed directories")
    ap.add_argument("--static-map", action="append", default=[], help="methods that never update their map")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    trials = protocol()
    by_group: Dict[Tuple[str, str], List[str]] = {}
    for t in trials:
        by_group.setdefault((t["condition"], t["scene"]), []).append(t["trial_id"])

    methods = []
    for spec in args.run:
        name, pattern = spec.split("=", 1)
        runs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / "trials").is_dir())
        if not runs:
            raise SystemExit(f"{name}: nothing matches {pattern}")
        methods.append((name, runs, [load(r) for r in runs]))

    lines = ["| Scene Type | Method | " + " | ".join(s.split("-")[0] for s in SCENES) + " | Trials | Avg. SR | Seeds |",
             "|---|---|" + "---|" * len(SCENES) + "---|---|---|"]
    notes = []
    for condition in CONDITIONS:
        for name, runs, results in methods:
            cells = []
            for scene in SCENES:
                ids = by_group[(condition, scene)]
                per_seed = [mean(r[i] for i in ids if i in r) for r in results if any(i in r for i in ids)]
                cells.append(cell(per_seed))
            ids = [i for scene in SCENES for i in by_group[(condition, scene)]]
            per_seed = [mean(r[i] for i in ids if i in r) for r in results if any(i in r for i in ids)]
            done = [sum(i in r for i in ids) for r in results]
            if any(d != len(ids) for d in done):
                notes.append(f"{name} / {condition}: trials completed per seed {done} of {len(ids)}")
            label = name + (" (static map)" if condition != "static" and name in args.static_map else "")
            lines.append(f"| {LABEL[condition]} | {label} | " + " | ".join(cells)
                         + f" | {len(ids)} | {cell(per_seed)} | {len(runs)} |")
    text = "\n".join(lines) + "\n"
    if notes:
        text += "\nIncomplete:\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    text += ("\nA cell is the mean SR over the method's seeds (collector seeds: start poses), "
             "with the min-max over seeds in brackets.\n")
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
