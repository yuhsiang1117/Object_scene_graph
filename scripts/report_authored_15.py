#!/usr/bin/env python3
"""Our pipeline on the 15-scene authored benchmark, scored at 1 m to the object.

Reads the nav-pass records under `--run` (one directory per scene) and writes
one report: SR and SPL by condition under the object-distance rule, habitat's
own viewpoint rule beside it, the per-scene table, the failure funnel from the
ground-truth visibility instrument, and the multi-storey split (same-floor
against cross-floor relocations), because on this root the object often sits
a storey away from where it was.

    python scripts/report_authored_15.py --run outputs/osg_authored_15
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CONDITIONS = ("in_anchor", "cross_anchor")


def horizontal(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


def load(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "-"


def mean(values) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.mean(values):.3f}" if values else "-"


def med(values) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.median(values):.1f}" if values else "-"


def table(title: str, header: List[str], rows: List[List[str]], note: str = "") -> List[str]:
    out = ["", f"## {title}", ""]
    if note:
        out += [note, ""]
    out += ["| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def rule(r: Dict[str, Any]) -> Dict[str, Any]:
    return (r.get("authored_layout") or {}).get("object_distance") or {}


def stage(r: Dict[str, Any]) -> str:
    if rule(r).get("success"):
        return "success"
    if not int(r.get("gt_kf_in_view") or 0):
        return "never in view"
    if not int(r.get("gt_kf_detected") or 0):
        return "seen, never named"
    if not int(r.get("gt_kf_admitted") or 0):
        return "named, never admitted"
    if r.get("target_obj_xy") is None:
        return "admitted, never committed"
    target = r["authored_layout"]["target_position"]
    if horizontal((r["target_obj_xy"][0], 0.0, r["target_obj_xy"][1]), target) > 1.0:
        return "committed elsewhere"
    return "committed, never arrived"


STAGES = ["success", "never in view", "seen, never named", "named, never admitted",
          "admitted, never committed", "committed elsewhere", "committed, never arrived"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/osg_authored_15"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.run / "RESULTS.md")

    rows = [r for r in load(args.run) if r["authored_layout"].get("layout_type") in CONDITIONS]
    if not rows:
        raise SystemExit(f"no dynamic episodes under {args.run}")
    scenes = sorted({r["scene"] for r in rows})
    definition = next((rule(r).get("success_definition") for r in rows if rule(r)), "?")
    lines = ["# Our pipeline on the authored 15-scene benchmark", "",
             f"`{args.run}`: {len(rows)} dynamic episodes over {len(scenes)} scenes, layout index 1. "
             f"Success rule: {definition}. Habitat's viewpoint rule is reported beside it, "
             "not as the result."]

    # 1. headline by condition
    hdr = ["split", "episodes", "SR (1 m to object)", "SPL", "SR (habitat viewpoint)",
           "median steps", "at budget", "never in view", "target named"]
    body = []
    for c in list(CONDITIONS) + ["all"]:
        xs = [r for r in rows if c == "all" or r["authored_layout"]["layout_type"] == c]
        body.append([c, str(len(xs)),
                     pct(sum(int(rule(r).get("success", 0)) for r in xs), len(xs)),
                     mean([float(rule(r).get("spl", 0.0)) for r in xs]),
                     pct(sum(int(round(float(r.get("habitat_success") or 0))) for r in xs), len(xs)),
                     med([int(r.get("steps") or 0) for r in xs]),
                     str(sum(int(r.get("steps") or 0) >= 500 for r in xs)),
                     str(sum(not int(r.get("gt_kf_in_view") or 0) for r in xs)),
                     str(sum(int(r.get("gt_kf_detected") or 0) > 0 for r in xs))])
    lines += table("1. Success by condition", hdr, body)

    # 2. same floor vs cross floor
    body = []
    for c in CONDITIONS:
        for cls in ("same_floor", "cross_floor"):
            xs = [r for r in rows if r["authored_layout"]["layout_type"] == c
                  and str(r.get("floor_class")) == cls]
            if not xs:
                continue
            body.append([c, cls, str(len(xs)),
                         pct(sum(int(rule(r).get("success", 0)) for r in xs), len(xs)),
                         str(sum(bool(r.get("goal_floor_reached")) for r in xs)),
                         str(sum(int(r.get("floor_changes") or 0) > 0 for r in xs)),
                         str(sum(not int(r.get("gt_kf_in_view") or 0) for r in xs))])
    lines += table("2. Same-floor against cross-floor relocations",
                   ["split", "relocation", "episodes", "SR", "reached the object's floor",
                    "changed floor at least once", "never in view"], body,
                   "The authored root is multi-storey throughout; `floor_class` compares the "
                   "object's floor to the start's.")

    # 3. per scene
    body = []
    for s in scenes:
        row = [s[:5]]
        for c in CONDITIONS:
            xs = [r for r in rows if r["scene"] == s and r["authored_layout"]["layout_type"] == c]
            row.append(pct(sum(int(rule(r).get("success", 0)) for r in xs), len(xs)))
        xs = [r for r in rows if r["scene"] == s]
        row += [pct(sum(int(round(float(r.get("habitat_success") or 0))) for r in xs), len(xs)),
                str(sum(str(r.get("floor_class")) == "cross_floor" for r in xs)),
                med([int(r.get("steps") or 0) for r in xs])]
        body.append(row)
    lines += table("3. Per scene", ["scene", "in-anchor SR", "cross-anchor SR",
                                    "habitat SR (all)", "cross-floor episodes", "median steps"], body)

    # 4. per target
    body = []
    for t in sorted({r["target"] for r in rows}):
        xs = [r for r in rows if r["target"] == t]
        iv = sum(int(r.get("gt_kf_in_view") or 0) for r in xs)
        det = sum(int(r.get("gt_kf_detected") or 0) for r in xs)
        body.append([t, str(len(xs)), pct(sum(int(rule(r).get("success", 0)) for r in xs), len(xs)),
                     f"{det / iv:.2f}" if iv else "-",
                     str(sum(int(r.get("gt_kf_detected") or 0) > 0 for r in xs))])
    lines += table("4. Per target", ["target", "episodes", "SR", "in-situ recall", "episodes named"], body)

    # 5. funnel
    body = []
    counts = {c: collections.Counter(stage(r) for r in rows if r["authored_layout"]["layout_type"] == c)
              for c in CONDITIONS}
    for st in STAGES:
        body.append([st] + [str(counts[c][st]) for c in CONDITIONS])
    lines += table("5. Failure funnel", ["stage"] + list(CONDITIONS), body,
                   "Each episode lands in the first stage it failed; stages 2-4 read the "
                   "ground-truth visibility instrument.")

    # 6. mechanism counters
    def stat(r, k):
        return int((r.get("agent_stats") or {}).get(k, 0) or 0)
    body = []
    for c in CONDITIONS:
        xs = [r for r in rows if r["authored_layout"]["layout_type"] == c]
        body.append([c, str(sum(stat(r, "close_look_started") for r in xs)),
                     str(sum(stat(r, "close_look_detected") for r in xs)),
                     str(sum(stat(r, "close_look_reapproach") for r in xs)),
                     str(sum(stat(r, "absence_abandon") for r in xs)),
                     str(sum(len(r.get("goal_commit_log") or []) for r in xs)),
                     str(sum(int(r.get("llm_calls") or 0) for r in xs)),
                     str(sum(int(r.get("llm_errors") or 0) for r in xs))])
    lines += table("6. Mechanism counters", ["split", "close looks", "detected during a look",
                                             "re-approached", "absence_abandon", "goal commits",
                                             "LLM calls", "LLM errors"], body)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
