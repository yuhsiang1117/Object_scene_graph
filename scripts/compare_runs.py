"""Paired A/B comparison of two runs. The single judgement tool for every stage.

At the 50-episode A/B size, SR alone is nearly useless: with SR ~= 0.5 the
standard error is ~7 percentage points, so a real +4pp effect and pure noise
look identical. Two things make the comparison readable at that size:

1. **Pairing.** Both arms run the *same* episode ids, so each episode is its own
   control. What matters is not the SR delta but which episodes flipped:
   ``gained`` (fail -> success), ``lost`` (success -> fail), ``net``. An arm
   that gains 4 and loses 1 is informative; a +2pp SR delta is not.

2. **Mechanism metrics.** Per-episode counters (frames_off_plane, climb_ok,
   steps_to_first_candidate, ...) move on nearly every episode, so they are far
   more stable than the binary success flag and show whether the mechanism did
   what it was built to do -- even when SR cannot resolve it.

Decision rule (from the plan):
    net >= +3     real effect      -> keep, enable by default
    -2..+2        noise            -> keep the code, leave the flag OFF,
                                      decide on the final full-split run
    net <= -3     regression       -> revert

Protocol guard: both runs must share the same evaluation protocol, or the
comparison is meaningless. summary.json's "config" block is diffed and any
mismatch is reported as an ERROR.

Usage:
    python scripts/compare_runs.py outputs/<baseline>/ outputs/<treatment>/
    python scripts/compare_runs.py base/ treat/ --split multi   # multi-floor only
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# Per-episode counters worth diffing. Absent keys (a stage that has not landed
# yet) are simply skipped, so this works across any pair of runs.
MECHANISM_KEYS = [
    # ASCENT's dense approach re-check. recheck_calls counts approaches that
    # reached a stop condition; reject counts those the score never vouched for.
    "recheck_calls", "recheck_pass", "recheck_reject", "recheck_no_obs",
    # Frontier descriptions sourced from the frame that revealed them.
    # frontier_sem_bound is how many openings ever got a frame description;
    # if it stays 0 the mechanism is inert whatever the SR says.
    "frontier_sem_new", "frontier_sem_bound", "frontier_sem_steps",
    # desc_differs/desc_frame is the fraction of descriptions the frame
    # source actually changed. Near 0 means the LLM read the same prompt.
    "desc_frame", "desc_differs",
    "steps_to_first_candidate",
    "frames_off_plane",
    "n_floors", "floor_switches", "cross_floor_candidate",
    "climb_ok", "climb_fail",
    # Frontier supply. frontier_seen/frontier_rounds is the mean number of
    # candidates per selection round, frontier_cells/frontier_seen their mean
    # size. Measured on a 4-episode probe: the contour extractor offers MORE
    # candidates than WFD (9.7 vs 6.1 per round), not fewer as first expected --
    # it has no minimum length and no dedup, and OSG's raycast-derived explored
    # region has a raggeder boundary than the fog-of-war cone ASCENT contours,
    # so it splits into more arcs. Read the two together: more candidates at a
    # similar mean size is finer openings, at a much smaller mean size is
    # slivers.
    "frontier_rounds", "frontier_seen", "frontier_cells", "frontier_give_up",
    "frontier_retired", "rank_calls", "rank_overrides", "rank_unreachable",
    "rooms_total", "rooms_labelled",
    "floor_asks", "floor_moves", "floor_blocked_one_floor", "floor_blocked_too_soon",
    "fp_retract", "value_calls",
    "llm_calls", "verify_calls",
    "steps", "spl", "distance_to_goal", "y_range_m",
]

# y_range_m is the height the agent ACTUALLY covered; on a multi-floor episode
# it rising toward goal_floor_gap_m is the signal that the agent started
# climbing at all. goal_floor_gap_m itself is a property of the episode and is
# identical in both arms, so it is not worth diffing.

# Protocol fields that MUST match for a comparison to mean anything.
PROTOCOL_KEYS = [
    "success_distance", "max_steps", "allow_sliding", "shuffle",
    "max_scene_repeat_steps", "dataset_version", "split", "seed",
]


def load(run_dir: Path) -> tuple[dict, dict, dict]:
    """(episodes keyed by uid, evaluation protocol, algorithm config)."""
    epf = run_dir / "episodes.jsonl"
    if not epf.exists():
        sys.exit(f"no episodes.jsonl in {run_dir}")
    episodes = {}
    for line in epf.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # `uid` is scene-qualified; older runs predate it, so fall back to
        # rebuilding it. episode_id ALONE is not unique -- habitat restarts it
        # at "0" in every per-scene content file.
        episodes[r.get("uid") or f"{r.get('scene', '?')}:{r['episode_id']}"] = r
    cfg, alg = {}, {}
    sf = run_dir / "summary.json"
    if sf.exists():
        summary = json.loads(sf.read_text())
        cfg = summary.get("config", {})
        alg = summary.get("algorithm", {})
    return episodes, cfg, alg


def show_algorithm_diff(a_alg: dict, b_alg: dict) -> None:
    """What actually differs between the two arms.

    A paired A/B only means something if exactly one thing changed. Runs
    predating the algorithm fingerprint have nothing here, and for those the
    difference is only recorded in whatever command launched them.
    """
    if not a_alg and not b_alg:
        print("[warn] neither run records its algorithm config (pre-fingerprint "
              "run); what differs between these arms is NOT recoverable from "
              "the artifacts\n")
        return
    if not a_alg or not b_alg:
        print("[warn] only one run records its algorithm config; the diff below "
              "is incomplete\n")
    keys = sorted(set(a_alg) | set(b_alg))
    diffs = [(k, a_alg.get(k), b_alg.get(k)) for k in keys if a_alg.get(k) != b_alg.get(k)]
    if not diffs:
        print("algorithm config: IDENTICAL -- any difference below is run-to-run "
              "noise, not an effect\n")
        return
    print("algorithm config differs in:")
    for k, av, bv in diffs:
        print(f"  {k}: {av!r} -> {bv!r}")
    if len(diffs) > 1:
        print("  [warn] more than one variable changed; the net below cannot be "
              "attributed to any single one")
    print()


def check_protocol(a_cfg: dict, b_cfg: dict) -> bool:
    diffs = [
        (k, a_cfg.get(k), b_cfg.get(k))
        for k in PROTOCOL_KEYS
        if k in a_cfg and k in b_cfg and a_cfg[k] != b_cfg[k]
    ]
    if diffs:
        print("ERROR: evaluation protocol differs between the two runs.")
        print("       The comparison below is NOT valid.\n")
        for k, av, bv in diffs:
            print(f"  {k}: baseline={av!r}  treatment={bv!r}")
        print()
    missing = [k for k in PROTOCOL_KEYS if k not in a_cfg or k not in b_cfg]
    if missing:
        print(f"[warn] protocol fields absent from summary.json (pre-S0 run?): "
              f"{', '.join(missing)}\n")
    return not diffs


def _is_multi_floor(rec_a: dict, rec_b: dict, floor_span: float) -> bool:
    """Does this episode require a floor change?

    Prefers the GROUND-TRUTH start-to-goal gap. Falls back to the observed
    trajectory span only for runs predating that field -- and that fallback is
    biased: an episode where the agent SHOULD have climbed but never did shows
    y_range_m ~ 0 and is misread as single-floor, which hides precisely the
    failures the multi-floor stages target.
    """
    for rec in (rec_a, rec_b):
        gap = rec.get("goal_floor_gap_m")
        if gap is not None:
            return float(gap) > floor_span
    return max(rec_a.get("y_range_m", 0.0), rec_b.get("y_range_m", 0.0)) > floor_span


def fmt_delta(x: float, better_is_low: bool = False) -> str:
    if abs(x) < 1e-9:
        return "     ="
    good = (x < 0) if better_is_low else (x > 0)
    return f"{x:+9.3f} {'+' if good else '-'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("treatment", type=Path)
    ap.add_argument("--split", choices=["all", "multi", "single"], default="all",
                    help="restrict to episodes whose goal is on another floor "
                         "(goal_floor_gap_m > --floor-span), or to those whose "
                         "goal is on the starting floor")
    ap.add_argument("--floor-span", type=float, default=1.0)
    ap.add_argument("--flips", action="store_true",
                    help="list every episode that flipped")
    args = ap.parse_args()

    a, a_cfg, a_alg = load(args.baseline)
    b, b_cfg, b_alg = load(args.treatment)
    protocol_ok = check_protocol(a_cfg, b_cfg)
    show_algorithm_diff(a_alg, b_alg)

    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("no episodes in common -- were the two runs on the same split?")
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    if only_a or only_b:
        print(f"[warn] unpaired episodes ignored: {only_a} baseline-only, "
              f"{only_b} treatment-only\n")

    if args.split != "all":
        want_multi = args.split == "multi"
        shared = [u for u in shared if _is_multi_floor(a[u], b[u], args.floor_span) == want_multi]
        if not shared:
            sys.exit(f"no {args.split}-floor episodes in the paired set")

    gained = [u for u in shared if a[u]["success"] < 0.5 <= b[u]["success"]]
    lost = [u for u in shared if b[u]["success"] < 0.5 <= a[u]["success"]]
    net = len(gained) - len(lost)
    sr_a = sum(a[u]["success"] for u in shared) / len(shared)
    sr_b = sum(b[u]["success"] for u in shared) / len(shared)

    print(f"paired episodes: {len(shared)}"
          + (f"   ({args.split}-floor only)" if args.split != "all" else ""))
    print(f"  baseline  {args.baseline}  SR {sr_a:.1%}")
    print(f"  treatment {args.treatment}  SR {sr_b:.1%}")
    print()
    print(f"  gained {len(gained):3d}   lost {len(lost):3d}   NET {net:+d}")

    verdict = ("REAL EFFECT -- keep, enable by default" if net >= 3 else
               "REGRESSION -- revert" if net <= -3 else
               "NOISE at n=%d -- keep the code, leave the flag OFF, decide on "
               "the full split" % len(shared))
    print(f"  verdict: {verdict}")
    if not protocol_ok:
        print("  (INVALID: protocol mismatch, see above)")
    print()

    if args.flips:
        for label, uids in (("gained", gained), ("lost", lost)):
            for u in uids:
                print(f"  {label:7s} {u:44s} {a[u].get('target','?'):12s} "
                      f"dtg {a[u].get('distance_to_goal', -1):6.2f} -> "
                      f"{b[u].get('distance_to_goal', -1):6.2f}")
        print()

    print("mechanism metrics (mean over paired episodes, treatment - baseline):")
    # distance_to_goal / steps are better when lower; the rest are descriptive
    # counters where "better" is context-dependent -- the sign marker is only a
    # reading aid, the number is what matters.
    lower_is_better = {"distance_to_goal", "steps", "steps_to_first_candidate"}
    for key in MECHANISM_KEYS:
        va = [a[u][key] for u in shared if a[u].get(key) is not None]
        vb = [b[u][key] for u in shared if b[u].get(key) is not None]
        if not va and not vb:
            continue
        ma = st.mean(va) if va else 0.0
        mb = st.mean(vb) if vb else 0.0
        n_a, n_b = len(va), len(vb)
        note = "" if n_a == n_b == len(shared) else f"   (n={n_a}->{n_b})"
        print(f"  {key:26s} {ma:9.3f} -> {mb:9.3f}   "
              f"{fmt_delta(mb - ma, key in lower_is_better)}{note}")


if __name__ == "__main__":
    main()
