#!/usr/bin/env python3
"""What the VLM decided, scored against where the object really was.

    python scripts/analyze_vlm.py outputs/mfv2/osg_descent_s42 [more runs]

Reads `vlm_call_log` from each run's episodes.jsonl (written since 54c4020) and
classifies every call by the track it was about:

  TRUE   the track is within the success radius of the object, on its storey
  STALE  it is at the object's PRIOR pose (pass 1's), where the object no longer is
  FP     anything else

Identity calls (`candidate`, `candidate_live`, `terminal`) answer "is this the
target?": an accept is right on TRUE, wrong on STALE / FP. Absence calls
(`absence_*`, "is it still there?") answer the same question at a place the
agent arrived at: "still there" is right on TRUE, "bare" right on STALE / FP;
"blocked" is no answer. A call with no verdict failed (network / parse).

Per run: calls per episode by role, and a confusion table per kind. Runs
recorded before the log existed report their call COUNTS only (`verify_calls`).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

SUCCESS_M = 1.0
STOREY_M = 1.5


def load(root: Path) -> List[Dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.glob("**/episodes.jsonl"))
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _near(a, b) -> bool:
    return (math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2])) <= SUCCESS_M
            and abs(float(a[1]) - float(b[1])) <= STOREY_M)


def truth(record: Dict[str, Any], xyz) -> Optional[str]:
    layout = record.get("authored_layout") or {}
    target, prior = layout.get("target_position"), layout.get("prior_position")
    if xyz is None or target is None:
        return None
    if _near(xyz, target):
        return "TRUE"
    if prior is not None and _near(xyz, prior):
        return "STALE"
    return "FP"


def kind_of(role: str) -> str:
    return "absence" if str(role).startswith("absence") else "identity"


def report(runs: "OrderedDict[str, List[Dict[str, Any]]]") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, rows in runs.items():
        n = len(rows)
        logged = [r for r in rows if "vlm_call_log" in r]
        roles: Counter = Counter()
        table: Dict[str, Counter] = {"identity": Counter(), "absence": Counter()}
        for r in rows:
            for call in r.get("vlm_call_log") or []:
                role = str(call.get("role", "?"))
                roles[role] += 1
                verdict = call.get("verdict")
                k = kind_of(role)
                t = truth(r, call.get("xyz")) or "?"
                v = ("none" if verdict is None else
                     ("yes" if verdict is True else "no"))
                table[k][(t, v)] += 1
        calls = sum(int(r.get("verify_calls") or 0) for r in rows)
        print(f"## {name}: {n} episodes, {calls} VLM calls ({calls / max(n, 1):.2f}/episode), "
              f"{len(logged)} with a per-call log")
        if roles:
            print("   by role: " + ", ".join(f"{k} {v}" for k, v in roles.most_common()))
        for k, tab in table.items():
            if not tab:
                continue
            yes = "accept" if k == "identity" else "still there"
            no = "reject" if k == "identity" else "bare"
            print(f"\n| {k} call about | {yes} | {no} | no answer |\n|---|---:|---:|---:|")
            for t in ("TRUE", "STALE", "FP", "?"):
                row = [tab[(t, "yes")], tab[(t, "no")], tab[(t, "none")]]
                if sum(row):
                    print(f"| {t} | {row[0]} | {row[1]} | {row[2]} |")
            right = tab[("TRUE", "yes")] + tab[("STALE", "no")] + tab[("FP", "no")]
            judged = sum(v for (t, verdict), v in tab.items() if verdict != "none" and t != "?")
            print(f"\n   right {right}/{judged}"
                  + (f" = {right / judged:.0%}" if judged else ""))
        print()
        out[name] = {"episodes": n, "calls": calls, "roles": dict(roles),
                     "tables": {k: {f"{t}/{v}": c for (t, v), c in tab.items()}
                                for k, tab in table.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args(argv)
    runs = OrderedDict((r, load(Path(r))) for r in args.runs)
    out = report(runs)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
