---
name: diagnose-run
description: Explain a finished evaluation run, or the difference between two, with the repo's analyzers instead of guesses — where episodes fail (exploration vs approach), localization vs navigation error, floor-structure splits, behaviour traces, and paired A/B or ASCENT-vs-OSG comparisons. Use after a run under outputs/ finishes, when SR changed and you need to say why, or before claiming an improvement.
---

# Diagnosing a run

The repo's currency is measured behaviour. At n = 100 SR moves ±3 points on noise, so "SR went
up" is not a finding until a paired trace metric moved with it. Every tool below reads a run's
`episodes.jsonl`; pass the run directory, or omit it where noted to use the newest run under
`outputs/`.

## 1. Where is SR lost? — start here

```bash
python scripts/analyze_stages.py outputs/<run>
```

Splits failures by the furthest FSM phase reached: never produced a candidate (exploration /
coverage / detector) vs reached APPROACH but stopped outside the success set (the last metres).
The answer decides which branch below to follow.

## 2a. Approach failures — the target was found

```bash
python scripts/analyze_localization.py outputs/<run>   # stop pose vs GT goal objects
python scripts/analyze_trackloc.py outputs/<run>       # err_loc (map) vs err_nav (driving)
python scripts/analyze_approach.py outputs/<run>       # why the terminal approach stopped short
```

Run them in that order: localization separates "stopped near a real object" from "committed to
something that is not there"; trackloc says whether the mapped track or the driving was wrong;
approach says why the controller stopped (planner had no path vs false arrival). HM3D ObjectNav
runs need the dataset at `data/datasets/objectnav/hm3d/v1/val` (`--episodes` to override).

## 2b. Multi-floor and behaviour

```bash
python scripts/analyze_floors.py outputs/<run>        # SR by floor_class, floor-estimator audit
python scripts/analyse_behaviour.py outputs/<run> [outputs/<other>]   # needs eval.behaviour_log=true
```

`analyze_floors.py` takes per-scene ground truth from `scripts/scene_floors.py`.
`analyse_behaviour.py` answers reach / sight / framing / motion and buckets every failure.

## 2c. The dynamic (moved-object) benchmarks

```bash
python scripts/analyze_dynamic_bench.py A=outputs/<run_a> B=outputs/<run_b>   # YCB authored layouts
python scripts/analyze_dynamic_failures.py --osg outputs/<run>   # eval.mode=dualmap_protocol only
```

`analyze_dynamic_bench.py` scores commits against the object's remembered pose as well as its
new one (a commit to the old pose is the correct opening move, not a false positive) and counts
livelocks. `analyze_dynamic_failures.py` puts each of DualMap's trials in the first stage it
failed: never in view → seen but not named → not admitted → no commit → committed elsewhere →
no arrival.

## 3. A/B — always paired, never two headline numbers

```bash
python scripts/compare_runs.py outputs/<baseline> outputs/<treatment> [--split multi|single] [--flips]
python scripts/compare_ascent_osg.py data/reference/ascent_behaviour_100 outputs/<run>
```

`compare_runs.py` walks both runs down the same decision funnel episode by episode (`--flips`
lists the episodes that changed outcome). `compare_ascent_osg.py` pairs an OSG run against
native ASCENT's recorded 100 episodes and prints the trace metrics that carried the S71
diagnosis (SAW episodes, never-saw STOP/timeout, climb steps on same-floor episodes, earliest
STOP, P(success | committed)); the run needs `eval.behaviour_log=true`. That ASCENT trace
(`data/reference/ascent_behaviour_100`) is **not in this checkout** — check it exists before
promising this comparison; without it, pair against an earlier OSG run with `compare_runs.py`.

Both arms of an A/B must share the episode set and use `verification=off` (see `run-eval`).

## Reporting

- Quote the analyzer's numbers with the run directory they came from.
- A change "worked" when the metric it was meant to move moved, in the paired comparison.
  If SR moved and no trace metric did, say it is within noise.
- Results worth keeping go in `docs/history/AB_RESULTS.md` (the log) with the exact command.
