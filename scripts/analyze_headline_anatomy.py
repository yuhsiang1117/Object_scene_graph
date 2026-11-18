#!/usr/bin/env python3
"""What the historical 0.812 / 0.458 was made of, and what has not changed since.

Two runs, one funnel. The historical run is condition N's 96 frozen episodes on the
authored layouts (`outputs/dualmap_comparison/baseline`), scored at the time by
geodesic distance to an authored viewpoint. The current run is the tight-ring run on
DualMap's released benchmark (`outputs/osg_dualmap_tightring`), scored by horizontal
distance to the object at 1 m. Both are rescored here by the same rule -- 1 m to the
object -- and pushed through the same ordered failure stages as
`analyze_dynamic_failures.py`, so the two columns can sit beside each other.

Everything a table here says about "the same then and now" is read from fields both
runs already record: the ground-truth visibility instrument (`gt_kf_*`), the commit
log, the final pose. Nothing is re-simulated.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release  # noqa: E402

STAGES = [
    ("success", "succeeded"),
    ("never_in_view", "never in view"),
    ("seen_not_named", "seen, never named"),
    ("named_not_admitted", "named, never admitted"),
    ("admitted_not_committed", "admitted, never committed"),
    ("committed_elsewhere", "committed elsewhere"),
    ("committed_no_arrival", "committed, never arrived"),
]
CONDITIONS = ("in_anchor", "cross_anchor")
# The authored benchmark named three targets for its detector; the released one asks
# for the object. Same YCB asset either way.
ALIAS = {"tin can": "soup can", "red plate": "plate", "blue plastic pitcher": "pitcher"}
SHARED = ("bowl", "plate", "cracker box", "pitcher", "soup can", "banana")
CONTAINER_CATEGORIES = {
    "table", "desk", "counter", "shelf", "cabinet", "dresser", "nightstand", "bed",
    "sofa", "stool", "bench", "oven", "washing machine", "refrigerator",
}
# "Detector wall": the object was in frame close and centred at least this many
# keyframes and the detector still never named it. Fewer than this, and the agent
# never gave the detector a proper look -- that is exposure, a search failure.
WALL_MIN_CLOSE_CENTRED = 3


def horizontal(a: Sequence[float], b: Sequence[float]) -> float:
    ax, az = (a[0], a[2]) if len(a) == 3 else (a[0], a[1])
    bx, bz = (b[0], b[2]) if len(b) == 3 else (b[0], b[1])
    return float(np.hypot(ax - bx, az - bz))


def read_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count(value: Any) -> int:
    return int(value or 0)


class Trial:
    """One episode reduced to the handful of numbers both runs can supply."""

    def __init__(self, record: Dict[str, Any], *, condition: str, target: str,
                 new_pos: Sequence[float], old_pos: Sequence[float],
                 d_new: float, goal_d_new: Optional[float], goal_d_old: Optional[float],
                 commit_errors: List[float], authored: bool, geodesic: Optional[float]):
        self.record = record
        self.condition = condition
        self.target = ALIAS.get(target, target)
        self.scene = str(record["scene"])[:5]
        self.new_pos, self.old_pos = new_pos, old_pos
        self.displacement = horizontal(new_pos, old_pos)
        self.d_new = d_new
        self.d_old = horizontal(record["final_xy"], old_pos)
        self.has_goal = goal_d_new is not None
        self.goal_ok = goal_d_new is not None and goal_d_new <= 1.0
        self.goal_stale = (goal_d_old is not None and goal_d_old <= 1.0
                           and goal_d_new is not None and goal_d_new > 1.0)
        self.commit_errors = commit_errors
        self.authored = authored
        self.geodesic = geodesic
        self.in_view = count(record.get("gt_kf_in_view"))
        self.detected = count(record.get("gt_kf_detected"))
        self.admitted = count(record.get("gt_kf_admitted"))
        self.close_centred = count(record.get("gt_kf_close_centred"))
        self.close_centred_detected = count(record.get("gt_kf_close_centred_detected"))
        self.steps = int(record.get("steps") or 0)
        self.stop = record.get("approach_stop_reason")
        stats = record.get("agent_stats") or {}
        self.absence_abandon = int(stats.get("absence_abandon", 0) or 0)

    def within(self, tol: float) -> bool:
        return self.d_new <= tol

    def stage(self, tol: float = 1.0) -> str:
        if self.within(tol):
            return "success"
        if not self.in_view:
            return "never_in_view"
        if not self.detected:
            return "seen_not_named"
        if not self.admitted:
            return "named_not_admitted"
        if not self.has_goal:
            return "admitted_not_committed"
        if not self.goal_ok:
            return "committed_elsewhere"
        return "committed_no_arrival"

    @property
    def first_commit(self) -> Optional[Dict[str, Any]]:
        log = self.record.get("goal_commit_log") or []
        return log[0] if log else None


# --------------------------------------------------------------------------- loading

def load_historical(root: Path, layout_root: Path) -> List[Trial]:
    """Condition N: authored layouts, one target per episode, success by viewpoint."""
    static_cache: Dict[str, Dict[str, Any]] = {}

    def stale_position(scene: str, semantic_id: int) -> List[float]:
        if scene not in static_cache:
            static_cache[scene] = json.loads(
                (layout_root / scene / "static_scene_config.json").read_text(encoding="utf-8")
            )
        for obj in static_cache[scene]["objects"]:
            if int(obj["semantic_id"]) == int(semantic_id):
                return list(obj["translation"])
        raise KeyError(f"{scene}: semantic id {semantic_id} not in the static layout")

    trials = []
    for record in read_rows(root):
        layout = record["authored_layout"]
        if layout.get("layout_type") not in CONDITIONS:
            continue
        new = layout["target_position"]
        old = stale_position(layout["scene"], layout["target_semantic_id"])
        goal = record.get("target_obj_xy")
        commits = [horizontal(e["center"], new) for e in (record.get("goal_commit_log") or [])
                   if e.get("center")]
        trials.append(Trial(
            record, condition=layout["layout_type"], target=record["target"],
            new_pos=new, old_pos=old,
            d_new=horizontal(record["final_xy"], new),
            goal_d_new=None if goal is None else horizontal(goal, new),
            goal_d_old=None if goal is None else horizontal(goal, old),
            commit_errors=commits, authored=bool(record["success"]),
            geodesic=(layout.get("start") or {}).get("initial_geodesic_distance"),
        ))
    return trials


def load_current(root: Path) -> List[Trial]:
    """The released benchmark: targets from the release, success by its own rule."""
    protocol = {t["trial_id"]: t for t in release.protocol()}
    trials = []
    for record in read_rows(root):
        layout = record["authored_layout"]
        block = (layout.get("dualmap") or {})
        if block.get("condition") not in CONDITIONS:
            continue
        targets = release.trial_targets(protocol[block["trial_id"]])
        new = layout["target_position"]
        olds = release.static_target_positions(layout["scene"], block["query"])
        old = min(olds, key=lambda o: horizontal(o[0], new))[0]
        goal = record.get("target_obj_xy")
        commits = [release.horizontal_distance_xz((e["center"][0], e["center"][2]), targets)
                   for e in (record.get("goal_commit_log") or []) if e.get("center")]
        trials.append(Trial(
            record, condition=block["condition"], target=block["query"],
            new_pos=new, old_pos=list(map(float, old)),
            d_new=release.horizontal_distance_xz(record["final_xy"], targets),
            goal_d_new=None if goal is None else release.horizontal_distance_xz(goal, targets),
            goal_d_old=None if goal is None else horizontal(goal, old),
            commit_errors=commits, authored=bool(block["success"]),
            geodesic=layout.get("reference_shortest_success_path_m"),
        ))
    return trials


# --------------------------------------------------------------------------- tables

def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "-"


def table(title: str, header: List[str], rows: List[List[str]], note: str = "") -> List[str]:
    out = ["", f"## {title}", ""]
    if note:
        out += [note, ""]
    out += ["| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def by_condition(trials: List[Trial], condition: str) -> List[Trial]:
    return [t for t in trials if t.condition == condition]


def median(values: Sequence[float]) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.median(values):.2f}" if values else "-"


def section_rules(runs: Dict[str, List[Trial]]) -> List[str]:
    rows = []
    for name, trials in runs.items():
        for c in CONDITIONS:
            xs = by_condition(trials, c)
            rows.append([f"{name} {c}", pct(sum(t.authored for t in xs), len(xs))]
                        + [pct(sum(t.within(tol) for t in xs), len(xs)) for tol in (1.0, 1.5, 2.0)])
    return table(
        "Success under each rule",
        ["run / split", "as scored at the time", "1.0 m to object", "1.5 m", "2.0 m"], rows,
        "The authored rule is geodesic distance to a sampled viewpoint; the released rule is "
        "horizontal distance to the object. Both runs rescored at 1 / 1.5 / 2 m to the object.",
    )


def section_do_nothing(hist: List[Trial]) -> List[str]:
    rows = []
    for c in CONDITIONS:
        xs = by_condition(hist, c)
        rows.append([c] + [pct(sum(t.displacement <= tol for t in xs), len(xs))
                           for tol in (1.0, 1.5, 2.0)]
                    + [pct(sum(t.authored for t in xs), len(xs))])
    return table(
        "A do-nothing agent on the authored layouts",
        ["split", "stands at the stale position, 1 m", "1.5 m", "2 m", "condition N as scored"],
        rows,
        "An agent that walks to where the object USED to be and stops, scored against where "
        "it is now. The authored rule's effective tolerance is about 2 m.",
    )


def section_anatomy(hist: List[Trial]) -> List[str]:
    rows = []
    for c in CONDITIONS:
        xs = [t for t in by_condition(hist, c) if t.authored]
        rows.append([
            c, str(len(xs)),
            str(sum(t.within(1.0) for t in xs)),
            str(sum(1.0 < t.d_new <= 1.5 for t in xs)),
            str(sum(t.d_new > 1.5 for t in xs)),
            str(sum(t.detected == 0 for t in xs)),
            str(sum(t.goal_stale for t in xs)),
            str(sum(t.d_old < t.d_new for t in xs)),
            median([t.d_new for t in xs]),
            f"{max(t.d_new for t in xs):.2f}" if xs else "-",
        ])
    band = []
    for c in CONDITIONS:
        xs = [t for t in by_condition(hist, c) if 1.0 < t.d_new <= 1.5]
        stops = collections.Counter(str(t.stop) for t in xs)
        band.append([c, str(len(xs)), str(sum(t.goal_ok for t in xs)),
                     ", ".join(f"{k} x{v}" for k, v in stops.most_common())])
    return table(
        "What condition N's scored successes were made of",
        ["split", "scored successes", "within 1 m", "1.0-1.5 m", "beyond 1.5 m",
         "never detected the target", "goal was the stale position",
         "ended nearer the stale position than the object", "median final m", "max final m"],
        rows,
    ) + table(
        "The 1.0-1.5 m band among them",
        ["split", "episodes", "committed goal within 1 m of the object", "stop reasons"], band,
        "Right object, right goal, stopped short: the fixed standoff the tight ring removed.",
    )


def section_funnel(runs: Dict[str, List[Trial]]) -> List[str]:
    header = ["stage"] + [f"{name} {c}" for name in runs for c in CONDITIONS]
    rows = []
    for key, label in STAGES:
        rows.append([label] + [str(sum(t.stage() == key for t in by_condition(trials, c)))
                               for trials in runs.values() for c in CONDITIONS])
    rows.append(["trials"] + [str(len(by_condition(trials, c)))
                              for trials in runs.values() for c in CONDITIONS])
    return table("The same funnel, at 1 m, for both runs", header, rows,
                 "Each trial lands in the first stage it failed; stages 1-4 read the "
                 "ground-truth visibility instrument, which the agent never sees.")


def section_conversion(runs: Dict[str, List[Trial]]) -> List[str]:
    rows = []
    for name, trials in runs.items():
        for c in CONDITIONS:
            xs = by_condition(trials, c)
            named = [t for t in xs if t.detected > 0]
            ok = [t for t in named if t.goal_ok]
            rows.append([f"{name} {c}", str(len(named)), str(len(ok)),
                         pct(sum(t.within(1.0) for t in ok), len(ok)),
                         pct(sum(t.within(1.5) for t in ok), len(ok)),
                         median([t.steps for t in xs if t.within(1.0)]),
                         median([t.steps for t in xs if not t.within(1.0)]),
                         str(sum(t.steps >= 500 for t in xs))])
    return table(
        "Conversion once the detector named the object",
        ["run / split", "named", "then committed to a correct goal", "correct goal -> within 1 m",
         "-> within 1.5 m", "median steps, successes", "median steps, failures", "at budget"],
        rows,
    )


def section_recall(runs: Dict[str, List[Trial]]) -> List[str]:
    targets = sorted({t.target for trials in runs.values() for t in trials},
                     key=lambda s: (s not in SHARED, s))
    rows = []
    for target in targets:
        row = [target]
        for trials in runs.values():
            xs = [t for t in trials if t.target == target]
            cc = sum(t.close_centred for t in xs)
            det = sum(t.close_centred_detected for t in xs)
            row += [str(len(xs)), str(cc), f"{det / cc:.2f}" if cc else "-"]
        rows.append(row)
    header = ["target"] + [f"{name} {col}" for name in runs
                           for col in ("trials", "close+centred kf", "recall")]
    return table(
        "Close, centred, in-situ recall per target",
        header, rows,
        "Keyframes with the object's centre within 3 m and inside 0.6 of the half-frame. "
        "Same detector, weights and image size in both runs.",
    )


def section_exposure(runs: Dict[str, List[Trial]]) -> List[str]:
    rows = []
    for name, trials in runs.items():
        for c in CONDITIONS:
            xs = [t for t in by_condition(trials, c) if t.stage() == "seen_not_named"]
            wall = collections.Counter(t.target for t in xs
                                       if t.close_centred >= WALL_MIN_CLOSE_CENTRED)
            expo = collections.Counter(t.target for t in xs
                                       if t.close_centred < WALL_MIN_CLOSE_CENTRED)
            rows.append([f"{name} {c}", str(len(xs)),
                         f"{sum(wall.values())} ({', '.join(f'{k} x{v}' for k, v in wall.most_common())})",
                         f"{sum(expo.values())} ({', '.join(f'{k} x{v}' for k, v in expo.most_common())})"])
    return table(
        "\"Seen, never named\": the detector, or the look?",
        ["run / split", "episodes", f"detector wall (>= {WALL_MIN_CLOSE_CENTRED} close+centred kf, 0 det)",
         "exposure (never a close, centred look)"],
        rows,
        "An object that crossed the frame at 3 m while the agent walked to a frontier is "
        "\"in view\" to the instrument and invisible to a detector whose recall past 2.5 m is 0.28.",
    )


def section_first_commit(runs: Dict[str, List[Trial]]) -> List[str]:
    rows = []
    for name, trials in runs.items():
        for c in CONDITIONS:
            xs = [t for t in by_condition(trials, c) if t.first_commit is not None]
            near_new = [t for t in xs if horizontal(t.first_commit["center"], t.new_pos) <= 1.0]
            left = [t for t in near_new if not t.within(1.0) and t.d_new > 3.0]
            rows.append([
                f"{name} {c}", str(len(xs)),
                str(sum(t.first_commit["step"] <= 2 for t in xs)),
                str(sum(horizontal(t.first_commit["center"], t.old_pos) <= 1.0 for t in xs)),
                str(len(near_new)),
                str(sum(t.within(1.0) for t in near_new)),
                str(len(left)),
                str(sum(t.absence_abandon > 0 for t in left)),
                pct(sum(sum(e > 1.0 for e in t.commit_errors) for t in xs),
                    sum(len(t.commit_errors) for t in xs)),
            ])
    return table(
        "The first commit, and what the agent did with it",
        ["run / split", "episodes with a commit", "committed by step 2",
         "first commit within 1 m of the stale position", "first commit within 1 m of the NEW position",
         "of those, succeeded", "of those, ended > 3 m away", "... after absence_abandon",
         "all commits > 1 m from the object"],
        rows,
        "In-anchor moves the object a median 0.7 m, so the stale track is already a correct "
        "goal. Arriving there, getting no detection, and leaving is the in-anchor failure.",
    )


def section_own_surface(current: List[Trial], maps_root: Path) -> List[str]:
    """Cross-anchor failures without a detection: was the object's own surface ever a goal?"""
    maps: Dict[str, Dict[int, Dict[str, Any]]] = {}

    full: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def all_maps(scene: str) -> Dict[int, Dict[str, Any]]:
        if scene not in full:
            path = maps_root / scene / f"{scene}.json"
            full[scene] = {} if not path.exists() else {
                int(t["id"]): t for t in json.loads(path.read_text(encoding="utf-8"))["tracks"]
            }
        return full[scene]

    def containers(scene: str) -> Dict[int, Dict[str, Any]]:
        if scene not in maps:
            maps[scene] = {cid: t for cid, t in all_maps(scene).items()
                           if t.get("label") in CONTAINER_CATEGORIES}
        return maps[scene]

    kinds = collections.OrderedDict((k, [0, 0, 0, 0, collections.Counter(), 0, 0])
                                    for k in ("never_in_view", "exposure", "detector wall"))
    for t in by_condition(current, "cross_anchor"):
        if t.within(1.0) or t.detected > 0:
            continue
        kind = ("never_in_view" if not t.in_view
                else "detector wall" if t.close_centred >= WALL_MIN_CLOSE_CENTRED else "exposure")
        scene = str(t.record["scene"])
        tracks = containers(scene)
        if not tracks:
            return ["", "## The object's own surface", "",
                    f"Prior map not found under `{maps_root}`; table skipped."]
        dist, near = min((horizontal(tr["center"], t.new_pos), tr) for tr in tracks.values())
        events = t.record.get("search_log_events") or []
        # Matched by distance, not id: the agent's container id is the smallest
        # track id of a linked component, so the nearest single track's id need
        # not be the id the search logs.
        all_tracks = dict(all_maps(scene))
        for cid, node in (t.record.get("containers") or {}).items():
            all_tracks[int(cid)] = {"id": int(cid), "label": node["label"], "center": node["center"]}

        def close(cid) -> bool:
            tr = all_tracks.get(int(cid)) if cid is not None else None
            return tr is not None and horizontal(tr["center"], t.new_pos) <= 1.5

        selected = any("utility" in e and close(e.get("container_id")) for e in events)
        arrived = any(e.get("arrived") and close(e.get("container_id")) for e in events)
        looked = any(int(e.get("container_id", -1)) >= 0 and close(e["container_id"])
                     for e in (t.record.get("close_look_log") or []))
        glances = [float(v) for k, v in (t.record.get("glance_ranges") or {}).items() if close(k)]
        glance = min(glances) if glances else None
        row = kinds[kind]
        row[0] += 1
        row[1] += int(dist <= 1.5)
        row[2] += int(selected)
        row[3] += int(arrived)
        row[4][str(near["label"])] += 1
        row[5] += int(looked)
        row[6] += int(glance is not None and float(glance) <= 2.5)
    rows = [[k, str(v[0]), str(v[1]), str(v[2]), str(v[3]), str(v[5]), str(v[6]),
             ", ".join(f"{lab} x{n}" for lab, n in v[4].most_common())]
            for k, v in kinds.items()]
    return table(
        "Cross-anchor failures without a detection: the object's own surface",
        ["failure kind", "episodes", "a mapped container within 1.5 m of the object",
         "that container ever selected by the search", "ever arrived at", "close-looked",
         "glanced from <= 2.5 m", "what it was"],
        rows,
        "Read from the prior map's tracks and the episode's `search_log_events`, "
        "`close_look_log` and `glance_ranges` (the last two are empty on runs that predate them); "
        "an event counts as the object's own surface when the logged container's centre is within 1.5 m "
        "of where the object landed, because the agent's container id is the smallest track id of a "
        "linked component and need not equal the nearest track's id. "
        "The search made 4-21 surface selections per episode and never chose this one.",
    )


def section_shared(runs: Dict[str, List[Trial]]) -> List[str]:
    rows = []
    for name, trials in runs.items():
        for c in CONDITIONS:
            xs = [t for t in by_condition(trials, c) if t.target in SHARED]
            rows.append([f"{name} {c}", str(len(xs))]
                        + [pct(sum(t.within(tol) for t in xs), len(xs)) for tol in (1.0, 1.5, 2.0)])
    return table(
        "Shared targets only",
        ["run / split", "trials", "1.0 m", "1.5 m", "2.0 m"], rows,
        f"Targets in both benchmarks: {', '.join(SHARED)}. Scissors and mug (released only) "
        "and bleach bottle (authored only) removed.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=Path("outputs/dualmap_comparison/baseline"))
    parser.add_argument("--historical-layouts", type=Path, default=Path("outputs/substituted_layouts"))
    parser.add_argument("--current", type=Path, default=Path("outputs/osg_dualmap_tightring"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/osg_dualmap_tightring/HEADLINE_ANATOMY.md"))
    args = parser.parse_args()

    current = load_current(args.current)
    runs: Dict[str, List[Trial]] = {}
    hist: List[Trial] = []
    if args.historical.exists() and args.historical_layouts.exists():
        hist = load_historical(args.historical, args.historical_layouts)
        runs["N"] = hist
    else:
        print(f"historical run not found at {args.historical} / {args.historical_layouts}; "
              "writing the current-run half only")
    runs["now"] = current

    lines = ["# What the old headline was made of, and what has not changed", "",
             f"Historical: `{args.historical}` ({len(hist)} dynamic trials). "
             f"Current: `{args.current}` ({len(current)} dynamic trials). "
             "Both scored here at 1 m horizontal distance to the object."]
    lines += section_rules(runs)
    if hist:
        lines += section_do_nothing(hist)
        lines += section_anatomy(hist)
    lines += section_funnel(runs)
    lines += section_shared(runs)
    lines += section_conversion(runs)
    lines += section_recall(runs)
    lines += section_exposure(runs)
    lines += section_first_commit(runs)
    lines += section_own_surface(current, args.maps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
