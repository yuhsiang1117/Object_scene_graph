"""Pick verification.approach_recheck_thresh from a run's own distribution.

ASCENT's 0.15 is a BLIP-2 ITM cosine. OSG's value map is CLIP, whose cosines
live in a different range entirely -- a 4-episode smoke put three true
positives at 0.237-0.249, so copying 0.15 would accept everything and the
mechanism would be inert. The threshold has to come from OSG's own numbers.

Feed it a run made with `verification.approach_recheck=true
verification.approach_recheck_thresh=0.0`: the gate records
`approach_recheck_max` at every stop while rejecting nothing, so the run is
behaviourally identical to the baseline and doubles as the paired control.

Ground truth is the episode outcome. A success is a target the gate must not
reject; a far-commit failure -- committed to something, walked to it, ended
more than a metre from any real goal -- is one it should. Everything else
(never committed, near miss) is out of the mechanism's reach and is reported
separately rather than folded into the score.

    python scripts/calibrate_approach_recheck.py outputs/<run>/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(run: Path):
    rows = [json.loads(l) for l in (run / "episodes.jsonl").open()]
    keep, skipped = [], 0
    for r in rows:
        if r.get("approach_recheck_max") is None:
            skipped += 1  # never reached a stop the gate sees
            continue
        n = r.get("approach_recheck_n")
        # The score never ran during the approach: unjudgeable, and folding its
        # 0.0 in would fake a large low-scoring population. Runs predating the
        # counter are identified by the initialiser value they left behind --
        # a genuine CLIP cosine of exactly 0.0 does not occur.
        if n == 0 or (n is None and r["approach_recheck_max"] == 0.0):
            skipped += 1
            continue
        keep.append(r)
    return keep, skipped, len(rows)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    run = Path(sys.argv[1])
    rows, skipped, total = _load(run)
    if not rows:
        raise SystemExit(
            f"no episode in {run} recorded approach_recheck_max -- was the run "
            "made with verification.approach_recheck=true?"
        )

    good = [r for r in rows if r["success"]]
    committed_fail = [r for r in rows
                      if not r["success"]
                      and r.get("steps_to_first_candidate") is not None]
    far = [r for r in committed_fail if r["distance_to_goal"] > 1.0]
    near = [r for r in committed_fail if r["distance_to_goal"] <= 1.0]

    def dist(name, sub):
        if not sub:
            print(f"  {name:24s}   n=0")
            return
        v = sorted(r["approach_recheck_max"] for r in sub)
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        print(f"  {name:24s} n={len(v):3d}  min {v[0]:.3f}  p10 {q(.10):.3f}  "
              f"p25 {q(.25):.3f}  med {q(.50):.3f}  p75 {q(.75):.3f}  max {v[-1]:.3f}")

    print(f"\n{total} episodes, {skipped} never reached a gated stop\n")
    print("approach_recheck_max by outcome:")
    dist("successes", good)
    dist("far-commit failures", far)
    dist("near-miss failures", near)

    if not far or not good:
        print("\nnot enough of both classes to choose a threshold")
        return

    # AUC answers the question the percentiles only hint at: over all
    # (success, far-commit) pairs, how often does the success score higher?
    # 0.5 is a coin flip -- no threshold anywhere can help, because there is no
    # ordering to cut. Anything a threshold achieves at 0.5 is an artefact of
    # this sample.
    gs = [r["approach_recheck_max"] for r in good]
    fs = [r["approach_recheck_max"] for r in far]
    wins = sum((g > f) + 0.5 * (g == f) for g in gs for f in fs)
    auc = wins / (len(gs) * len(fs))
    print(f"\nAUC (success scores above far-commit) = {auc:.3f}"
          f"   [0.5 = no separation]")

    # The only threshold worth taking is one that rejects far-commits while
    # costing no successes: a rejected success is an episode thrown away.
    print("\nthreshold sweep -- 'costs' is successes the gate would have killed:")
    print(f"  {'thresh':>7s} {'rejects far':>12s} {'costs successes':>16s} {'net':>5s}")
    best = None
    lo = min(r["approach_recheck_max"] for r in rows)
    hi = max(r["approach_recheck_max"] for r in rows)
    for i in range(41):
        t = lo + (hi - lo) * i / 40
        rej = sum(1 for r in far if r["approach_recheck_max"] < t)
        cost = sum(1 for r in good if r["approach_recheck_max"] < t)
        if rej == 0 and cost == 0:
            continue
        net = rej - cost
        print(f"  {t:7.3f} {rej:12d} {cost:16d} {net:+5d}")
        if best is None or net > best[1]:
            best = (t, net, rej, cost)

    print()
    if best is None or best[1] <= 0:
        print("NO USABLE THRESHOLD: every cut that rejects a far-commit costs at\n"
              "least as many successes. The CLIP score does not separate the two\n"
              "classes on this run, so the mechanism cannot help as instrumented.")
    else:
        t, net, rej, cost = best
        print(f"best: thresh={t:.3f} -> rejects {rej} far-commits, costs {cost} "
              f"successes (net {net:+d})")
        print("A rejected far-commit is not automatically a rescue: the agent has\n"
              "to find the real goal in the steps it has left. Treat net as an\n"
              "upper bound and confirm with a paired run.")


if __name__ == "__main__":
    main()
