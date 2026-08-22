#!/usr/bin/env python3
"""Read one or more YCB dynamic-benchmark runs and say what happened.

The headline SR is the least interesting number a run produces. What decides the
next experiment is *which* failure a run is making, and that needs the episode
records rather than the summary:

  * a commit to the object's REMEMBERED pose is the correct opening move of a
    dynamic episode, not an error -- scoring commits against the post-move
    position alone counts a good first attempt as a false positive, which is a
    mistake this analysis made once and now cannot make again;
  * a track committed to repeatedly is a livelock, and the count separates
    "searched and lost" from "stood in one place re-proposing the same thing";
  * distance-to-goal at the end separates a search failure from a terminal one.

    python scripts/analyze_dynamic_bench.py outputs/2026*/  --label baseline
    python scripts/analyze_dynamic_bench.py A=outputs/run_a B=outputs/run_b
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

LAYOUT_ROOT = Path("outputs/substituted_layouts")
TARGET_LABELS = {
    "003_cracker_box": "cracker box", "005_tomato_soup_can": "cylindrical can",
    "011_banana": "banana", "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle", "024_bowl": "bowl",
    "025_mug": "mug", "029_plate": "red dish",
}
NEAR_M = 0.5


def static_positions(scene: str) -> Dict[str, tuple]:
    """Where each target was BEFORE the relocation, from the static layout."""
    path = LAYOUT_ROOT / scene / "static_scene_config.json"
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    mapping = {int(k): v for k, v in blob["id_handle_mapping"].items()}
    out = {}
    for obj in blob["objects"]:
        handle = mapping[int(obj["semantic_id"])]
        label = TARGET_LABELS.get(handle)
        if label:
            out[label] = (obj["translation"][0], obj["translation"][2])
    return out


def load(paths: List[Path]) -> List[dict]:
    rows = []
    for base in paths:
        for path in sorted(base.rglob("episodes.jsonl")):
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                row["_run"] = str(path.parent)
                rows.append(row)
    return rows


def classify(row: dict, before: Dict[str, tuple]) -> str:
    pos = row["authored_layout"]["target_position"]
    now = (pos[0], pos[2])
    was = before.get(row["target"])
    kinds = set()
    for commit in row.get("goal_commit_log") or []:
        cx, cz = commit["center"][0], commit["center"][2]
        if math.hypot(cx - now[0], cz - now[1]) <= NEAR_M:
            kinds.add("correct")
        elif was is not None and math.hypot(cx - was[0], cz - was[1]) <= NEAR_M:
            kinds.add("stale")
        else:
            kinds.add("false")
    if row["success"]:
        return "succeeded"
    if not kinds:
        return "never committed"
    if "correct" in kinds:
        return "reached the right track, still failed"
    if kinds == {"stale"}:
        return "the remembered pose only, never re-found it"
    if "stale" in kinds:
        return "the remembered pose, then false positives"
    return "false positives only"


def agg(rows: List[dict]) -> tuple:
    n = len(rows)
    if not n:
        return 0, float("nan"), float("nan"), float("nan")
    spl = [r["spl"] for r in rows if math.isfinite(r["spl"])]
    dtg = [r["distance_to_goal"] for r in rows if math.isfinite(r["distance_to_goal"])]
    return (n, sum(r["success"] for r in rows) / n,
            sum(spl) / len(spl) if spl else float("nan"),
            sum(dtg) / len(dtg) if dtg else float("nan"))


def table(rows: List[dict], key, title: str, order: Optional[List[str]] = None) -> None:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    print(f"  {title}")
    print(f"    {'':26s} {'n':>3s} {'SR':>6s} {'SPL':>6s} {'d2g':>6s}")
    keys = order or sorted(groups, key=lambda k: -agg(groups[k])[1])
    for k in keys:
        if k not in groups:
            continue
        n, sr, spl, dtg = agg(groups[k])
        print(f"    {str(k):26s} {n:3d} {sr:6.3f} {spl:6.3f} {dtg:6.2f}")
    n, sr, spl, dtg = agg(rows)
    print(f"    {'ALL':26s} {n:3d} {sr:6.3f} {spl:6.3f} {dtg:6.2f}\n")


def report(name: str, rows: List[dict]) -> None:
    print("=" * 78)
    print(f"{name}: {len(rows)} episodes")
    print("=" * 78)
    table(rows, lambda r: r["authored_layout"]["layout_type"], "By condition:",
          ["in_anchor", "cross_anchor", "static"])
    table(rows, lambda r: r["scene"], "By scene:")
    table(rows, lambda r: r["target"], "By target:")

    before = {s: static_positions(s) for s in {r["scene"] for r in rows}}
    fam = collections.Counter(classify(r, before[r["scene"]]) for r in rows)
    print("  Why episodes ended as they did:")
    for k, v in fam.most_common():
        print(f"    {v:4d}  {k}")
    print()

    commits = [len(r.get("goal_commit_log") or []) for r in rows]
    commits.sort()
    live = [r for r in rows if len(r.get("goal_commit_log") or []) > 10]
    print("  Goal commits per episode (a livelock shows up here and nowhere else):")
    print(f"    median {commits[len(commits) // 2]}, max {commits[-1]}, "
          f"{len(live)} episodes over 10 (their SR "
          f"{sum(r['success'] for r in live) / len(live):.3f})" if live else
          f"    median {commits[len(commits) // 2]}, max {commits[-1]}, none over 10")
    ident = sum((r.get("agent_stats") or {}).get("verify_reject", 0) for r in rows)
    aband = sum((r.get("agent_stats") or {}).get("absence_abandon", 0) for r in rows)
    verr = sum(r.get("verify_errors", 0) or 0 for r in rows)
    vcalls = sum(r.get("verify_calls", 0) or 0 for r in rows)
    print(f"    VLM candidate rejections {ident}, absence abandons {aband}, "
          f"VLM calls {vcalls} ({verr} errors)")
    srch = [len(r.get("search_log_events") or []) for r in rows]
    arr = [sum(1 for e in (r.get("search_log_events") or []) if e.get("arrived")) for r in rows]
    print(f"    search: {sum(1 for x in srch if x)} episodes searched, "
          f"median {sorted(srch)[len(srch) // 2]} selections, "
          f"median {sorted(arr)[len(arr) // 2]} arrivals")
    print()
    buckets = collections.Counter()
    for r in rows:
        if r["success"]:
            continue
        d = r["distance_to_goal"]
        buckets["A inside 0.18 m, never stopped" if d <= 0.18 else
                "B last metre" if d <= 1.0 else
                "C 1-4 m" if d <= 4.0 else "D over 4 m / unreachable"] += 1
    print("  How far the failures ended from the goal:")
    for k in sorted(buckets):
        print(f"    {buckets[k]:4d}  {k}")
    print()


def compare(runs: Dict[str, List[dict]]) -> None:
    if len(runs) < 2:
        return
    print("=" * 78)
    print("SIDE BY SIDE")
    print("=" * 78)
    names = list(runs)
    print(f"  {'':26s}" + "".join(f"{n:>16s}" for n in names))
    def line(label, rows_by_run):
        cells = ""
        for n in names:
            rows = rows_by_run.get(n, [])
            if rows:
                _, sr, spl, _ = agg(rows)
                cells += f"{sr:8.3f}/{spl:<7.3f}"
            else:
                cells += f"{'-':>16s}"
        print(f"  {label:26s}{cells}")
    print("  (SR / SPL)")
    line("overall", {n: r for n, r in runs.items()})
    for cond in ("in_anchor", "cross_anchor"):
        line(cond, {n: [x for x in r if x["authored_layout"]["layout_type"] == cond]
                    for n, r in runs.items()})
    for scene in sorted({x["scene"] for r in runs.values() for x in r}):
        line(scene, {n: [x for x in r if x["scene"] == scene] for n, r in runs.items()})
    for tgt in sorted({x["target"] for r in runs.values() for x in r}):
        line(tgt, {n: [x for x in r if x["target"] == tgt] for n, r in runs.items()})
    print()
    print(f"  {'failure family':40s}" + "".join(f"{n:>12s}" for n in names))
    fams = {}
    for n, rows in runs.items():
        before = {s: static_positions(s) for s in {r["scene"] for r in rows}}
        fams[n] = collections.Counter(classify(r, before[r["scene"]]) for r in rows)
    for k in sorted({k for c in fams.values() for k in c}):
        print(f"  {k:40s}" + "".join(f"{fams[n][k]:12d}" for n in names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="paths, or NAME=path to label a run")
    args = ap.parse_args()
    runs: Dict[str, List[dict]] = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).name, item
        rows = load([Path(path)])
        if not rows:
            print(f"  (no episodes under {path})")
            continue
        runs[name] = rows
    for name, rows in runs.items():
        report(name, rows)
    compare(runs)


if __name__ == "__main__":
    main()
