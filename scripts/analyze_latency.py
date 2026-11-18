#!/usr/bin/env python3
"""The latency table of one run: what each stage costs a control step.

The review asked for real-world latency. Every run -- simulator or robot
(`scripts/run_robot.py`, `eval.mode=ros2`) -- writes the same profiler output,
so this reads a run directory's `summary.json` (or `timing.csv`) and prints, per
pipeline stage, the per-call cost (mean / p50 / p90 / max, ms), how often it
runs per control step, and its share of control-loop time; then the control
rate itself, and the SYNCHRONOUS model calls separately, because one of them
inside a step is the step.

    python scripts/analyze_latency.py outputs/<run>              # one run
    python scripts/analyze_latency.py outputs/<run> --json lat.json

p90 and p99 exist only in runs written after they were added to the profiler
(`core/profiler.py`); older runs print "-". A number measured while other
processes share the GPU is a contended number: say so beside it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional

STAGES = [
    ("perception", ["detector", "region_proposal", "stair_seg", "gate_itm"]),
    ("object layer + scene graph", ["object_layer", "scene_graph", "candidates", "glance"]),
    ("mapping", ["costmap", "floors", "stairs", "room_seg"]),
    ("decision", ["frontier_extract", "frontier_select", "planner", "floor_switch",
                  "close_look", "search"]),
    ("mover", ["mover"]),
]
SYNC_MODEL_CALLS = ["absence_vlm", "stop_gate", "verify", "terminal_verify"]
LOOP = "control_loop"


def load_timing(run: Path) -> Dict[str, Dict[str, float]]:
    summary = run / "summary.json"
    if summary.is_file():
        timing = json.loads(summary.read_text()).get("timing") or {}
        if timing:
            return timing
    out: Dict[str, Dict[str, float]] = {}
    csv_path = run / "timing.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{run}: neither summary.json['timing'] nor timing.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["module"]] = {k: float(v) for k, v in row.items() if k != "module" and v != ""}
    return out


def _ms(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def table(timing: Dict[str, Dict[str, float]]) -> dict:
    loop = timing.get(LOOP)
    if not loop:
        raise SystemExit(f"no `{LOOP}` section: not an agent run")
    steps = int(loop["count"])
    loop_total = float(loop["total_s"])
    print("| stage | module | calls / step | mean ms | p50 ms | p90 ms | max ms | share of loop |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    rows = []
    seen = {LOOP}
    for stage, modules in STAGES:
        for m in modules:
            r = timing.get(m)
            if not r:
                continue
            seen.add(m)
            share = float(r["total_s"]) / loop_total if loop_total else 0.0
            rows.append({"stage": stage, "module": m, **r, "share": share})
            print(f"| {stage} | {m} | {r['count'] / steps:.2f} | {_ms(r.get('mean_ms'))} | "
                  f"{_ms(r.get('median_ms'))} | {_ms(r.get('p90_ms'))} | {_ms(r.get('max_ms'))} | "
                  f"{100 * share:.1f}% |")
    sync = {m: timing[m] for m in SYNC_MODEL_CALLS if m in timing}
    seen |= set(sync)
    other = {m: r for m, r in timing.items() if m not in seen}
    for m, r in sorted(other.items(), key=lambda kv: -float(kv[1]["total_s"])):
        share = float(r["total_s"]) / loop_total if loop_total else 0.0
        rows.append({"stage": "other", "module": m, **r, "share": share})
        print(f"| other | {m} | {r['count'] / steps:.2f} | {_ms(r.get('mean_ms'))} | "
              f"{_ms(r.get('median_ms'))} | {_ms(r.get('p90_ms'))} | {_ms(r.get('max_ms'))} | "
              f"{100 * share:.1f}% |")
    print()
    mean = float(loop["mean_ms"])
    print(f"control loop: {steps} steps, mean {mean:.1f} ms ({1000.0 / mean:.2f} Hz), "
          f"p50 {_ms(loop.get('median_ms'))} ms, p90 {_ms(loop.get('p90_ms'))} ms, "
          f"max {_ms(loop.get('max_ms'))} ms")
    if sync:
        print("\nsynchronous model calls (inside a control step -- the step waits for them):")
        for m, r in sync.items():
            print(f"  {m}: {int(r['count'])} calls, mean {_ms(r.get('mean_ms'))} ms, "
                  f"max {_ms(r.get('max_ms'))} ms, {100 * float(r['total_s']) / loop_total:.1f}% "
                  f"of loop time")
    return {"steps": steps, "loop": loop, "rows": rows, "sync": sync}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a run directory (summary.json / timing.csv)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = table(load_timing(args.run))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
