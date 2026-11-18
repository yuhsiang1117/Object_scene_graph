#!/usr/bin/env python3
"""What the detector was looking at when it failed to name the target.

`eval.gt_dump_dir` writes one row per keyframe in which the target was genuinely
visible, and each row carries what the detector DID say (`top_labels`) alongside
whether it named the target. That turns "seen, never named" -- the largest single
failure bucket -- from a count into a diagnosis, and it needs no GPU: the run
already paid for the inference.

Three questions, which have three different fixes:

  Is it SIZE?         missed frames are small (`gt_px`) and distant (`range_m`).
                      Fix: get closer, or raise imgsz. Nothing about vocabulary
                      will help.
  Is it COMPETITION?  another class won the same pixels at a high score. An
                      open-vocabulary head runs class-competitive NMS, so `box`
                      and `book` really do eat a cracker box. Fix: drop the
                      competing class from the vocabulary.
  Is it the NAME?     the target is big, close, centred, and nothing else claims
                      it -- the model simply does not call it that. Fix: rename,
                      and `scripts/legacy/probe_nearmiss.py` measures which name.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

EPISODE_RE = re.compile(r"^(?P<scene>\d{5}-\w+)_(?P<layout>[^_]+)_ep(?P<trial>.+)$")


def load(dump: Path) -> List[Dict[str, Any]]:
    rows = []
    for index in sorted(dump.glob("*/index.tsv")):
        match = EPISODE_RE.match(index.parent.name)
        trial = match.group("trial") if match else index.parent.name
        target = trial.rsplit("__", 1)[-1].replace("_", " ") if "__" in trial else "?"
        with index.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["trial"] = trial
                row["target"] = target
                rows.append(row)
    return rows


def num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    parser.add_argument("--out", type=Path, default=Path("outputs/osg_dualmap_tightring/GT_DUMP.md"))
    args = parser.parse_args()

    rows = load(args.dump)
    if not rows:
        raise SystemExit(f"no index.tsv under {args.dump}")
    named = [r for r in rows if r.get("named") == "1"]
    missed = [r for r in rows if r.get("named") != "1"]

    lines = [
        "# What the detector saw on the frames it failed on",
        "",
        f"{len(rows)} in-view keyframes over {len({r['trial'] for r in rows})} episodes "
        f"from `{args.dump}`. Named on {len(named)}, missed on {len(missed)} "
        f"(in-situ recall {len(named) / len(rows):.2f}).",
        "",
        "## What covered the object's pixel",
        "",
        "`gt_px` is the mask area of whatever detection covers the object's "
        "projected point -- NOT the target's own apparent size, which the dump "
        "does not record. That makes it a cleaner diagnosis than size would be:",
        "",
        "* **0 px** -- no detection mask covers the object at all. The head "
        "produced nothing there. A name change cannot help; this is size, "
        "resolution or salience.",
        "* **large, with a rival label** -- another class claimed those exact "
        "pixels and won class-competitive NMS. Dropping that class from the "
        "vocabulary is the cheap fix.",
        "",
        "| Target | Frames | Recall | Missed with NOTHING there | Missed with a rival mask | Median range (missed) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in sorted({r["target"] for r in rows}):
        sub = [r for r in rows if r["target"] == target]
        no = [r for r in sub if r.get("named") != "1"]
        empty = sum(1 for r in no if num(r, "gt_px") <= 0)
        rival = len(no) - empty
        recall = 1.0 - len(no) / len(sub)
        rng = f"{np.median([num(r, 'range_m') for r in no]):.2f} m" if no else "n/a"
        lines.append(
            f"| {target} | {len(sub)} | {recall:.2f} | {empty} | {rival} | {rng} |"
        )

    # ------------------------------------------------------------ competition
    lines += [
        "",
        "## What won the pixels instead",
        "",
        "`top_labels` on a missed frame is what the head DID fire on. A class that "
        "appears often and scores high is competing for the target's own pixels, "
        "and removing it from the vocabulary is the cheap fix -- this is how `box` "
        "and `book` were found to be eating the cracker box.",
        "",
        "| Target | Missed frames | Classes named instead (count, median score) |",
        "|---|---:|---|",
    ]
    for target in sorted({r["target"] for r in rows}):
        no = [r for r in rows if r["target"] == target and r.get("named") != "1"]
        if not no:
            continue
        tally: Dict[str, List[float]] = collections.defaultdict(list)
        for row in no:
            for entry in (row.get("top_labels") or "").split(","):
                if ":" not in entry:
                    continue
                label, _, score = entry.rpartition(":")
                try:
                    tally[label.strip()].append(float(score))
                except ValueError:
                    continue
        top = sorted(tally.items(), key=lambda kv: -len(kv[1]))[:6]
        listed = ", ".join(f"`{k}` ×{len(v)} @{np.median(v):.2f}" for k, v in top) or "nothing"
        lines.append(f"| {target} | {len(no)} | {listed} |")

    # --------------------------------------------------------- the name itself
    lines += [
        "",
        "## Frames where only the name can be the problem",
        "",
        "Missed frames that were big (>2000 px), close (<2.5 m), centred "
        "(off-axis < 0.5) and unoccluded, where nothing else scored above 0.5 on "
        "those pixels. The detector had every chance and did not call it that.",
        "",
        "| Target | Such frames | Best score the target ever got on them |",
        "|---|---:|---:|",
    ]
    for target in sorted({r["target"] for r in rows}):
        clean = []
        for row in rows:
            if row["target"] != target or row.get("named") == "1":
                continue
            if not (num(row, "gt_px") > 2000 and num(row, "range_m") < 2.5
                    and num(row, "offaxis") < 0.5 and num(row, "vis_frac") > 0.8):
                continue
            rival = max(
                (float(e.rpartition(":")[2]) for e in (row.get("top_labels") or "").split(",")
                 if ":" in e and _floatable(e.rpartition(":")[2])),
                default=0.0,
            )
            if rival < 0.5:
                clean.append(num(row, "best_score"))
        if clean:
            lines.append(f"| {target} | {len(clean)} | {max(clean):.2f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(lines))


def _floatable(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
