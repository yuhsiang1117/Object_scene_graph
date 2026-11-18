#!/usr/bin/env python3
"""Why our result on the released DualMap benchmark differs from DualMap's.

The comparison table says how far apart the two systems are. This says what the
distance is made of, and it is deliberately two separate questions because the
static and dynamic splits fail for different reasons:

  **Static.** Its queries are HM3DSem furniture classes. DualMap stores a CLIP
  embedding per mapped object, so any text query can be retrieved from its
  preloaded map. Our prior map stores tracks under a fixed vocabulary, so a
  query outside it has nothing in the prior to retrieve and has to be found by
  exploration. Splitting our static SR on that one fact is the first table.

  **Dynamic.** Its queries are the eight YCB objects, all inside our
  vocabulary, so prior coverage cannot be the explanation. The second table
  splits our failures by how far the episode got: never saw the object, saw it
  and never named it, named it and still did not arrive.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

SCENES = ("00829-QaLdnwvtxbs", "00848-ziup5kvtCCR", "00880-Nfvxx8J5NCo")


def load_runs(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prior_labels(map_root: Path) -> Dict[str, collections.Counter]:
    """What each scene's preloaded map actually holds, by label."""
    out = {}
    for scene in SCENES:
        path = map_root / scene / f"{scene}.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        out[scene] = collections.Counter(
            str(track.get("label", "")).strip().lower()
            for track in (blob.get("tracks") or [])
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osg", type=Path, default=Path("outputs/osg_dualmap_protocol"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/osg_dualmap_protocol/GAP_ANALYSIS.md")
    )
    args = parser.parse_args()

    rows = load_runs(args.osg)
    priors = prior_labels(args.maps)
    lines = [
        "# What our shortfall on the released DualMap benchmark is made of",
        "",
        f"{len(rows)} trials from {args.osg}.",
        "",
        "## Static: does our prior map hold the queried class at all?",
        "",
        "DualMap preloads a map whose objects each carry a CLIP embedding, so a "
        "text query is answered by retrieval. Ours preloads tracks labelled from a "
        "fixed detector vocabulary, so a query outside it has nothing to retrieve "
        "and the agent has to search for it. The two rows are the same system, the "
        "same scenes and the same budget, differing only in whether the query names "
        "something the prior already holds.",
        "",
        "| Static queries | Trials | Our SR |",
        "|---|---:|---:|",
    ]
    buckets = collections.Counter()
    absent_queries = collections.Counter()
    for record in rows:
        block = (record.get("authored_layout") or {}).get("dualmap") or {}
        if block.get("condition") != "static":
            continue
        scene = str(record.get("scene", ""))
        query = str(block.get("agent_query", block.get("query", ""))).lower()
        held = priors.get(scene, collections.Counter()).get(query, 0) > 0
        buckets[held, bool(block["success"])] += 1
        if not held:
            absent_queries[query] += 1
    for held, name in ((True, "named in our prior map"), (False, "absent from our prior map")):
        ok, fail = buckets[held, True], buckets[held, False]
        total = ok + fail
        if total:
            lines.append(f"| {name} | {total} | {ok}/{total} = {100 * ok / total:.0f}% |")

    lines += ["", f"Queries absent from the prior ({len(absent_queries)} distinct): "
                  + ", ".join(f"`{q}`" for q in sorted(absent_queries)), ""]

    # ------------------------------------------------------------- dynamic
    lines += ["## Dynamic: how far did the failures get?", "",
              "These queries are the eight YCB objects, every one of them inside our "
              "vocabulary, so prior coverage is not available as an explanation. "
              "`gt_*` are ground-truth instruments observed beside the run and never "
              "given to the agent.", "",
              "| Split | Failures | Never saw it | Saw it, never named it | Named it, still failed | At the step budget |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition in ("in_anchor", "cross_anchor"):
        failures = [
            r for r in rows
            if ((r.get("authored_layout") or {}).get("dualmap") or {}).get("condition") == condition
            and not ((r.get("authored_layout") or {}).get("dualmap") or {}).get("success")
        ]
        if not failures:
            continue
        never = sum(1 for r in failures if not (r.get("gt_kf_in_view") or 0))
        unnamed = sum(
            1 for r in failures
            if (r.get("gt_kf_in_view") or 0) and not (r.get("gt_kf_detected") or 0)
        )
        named = sum(1 for r in failures if (r.get("gt_kf_detected") or 0))
        budget = sum(1 for r in failures if int(r.get("steps", 0)) >= 500)
        lines.append(
            f"| {condition} | {len(failures)} | {never} | {unnamed} | {named} | {budget} |"
        )

    # Where the failures ended, which separates a search that went to the wrong
    # place from one that arrived and could not close.
    lines += ["", "## Where our failures ended", "",
              "| Split | Failures | <1 m | 1-3 m | >3 m |", "|---|---:|---:|---:|---:|"]
    for condition in ("static", "in_anchor", "cross_anchor"):
        failures = [
            ((r.get("authored_layout") or {}).get("dualmap") or {})
            for r in rows
        ]
        failures = [
            b for b in failures
            if b.get("condition") == condition and not b.get("success")
        ]
        if not failures:
            continue
        d = [float(b["final_distance_horizontal_m"]) for b in failures]
        lines.append(
            f"| {condition} | {len(d)} | {sum(1 for x in d if x < 1.0)} | "
            f"{sum(1 for x in d if 1.0 <= x <= 3.0)} | {sum(1 for x in d if x > 3.0)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
