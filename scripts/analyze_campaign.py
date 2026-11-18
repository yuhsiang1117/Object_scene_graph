#!/usr/bin/env python3
"""Compare conditions of the dynamic YCB ladder, by what they did and why.

The headline SR is the least informative number a run produces. What decides the
next experiment is the funnel: how far each episode got before it stopped, and
which stage lost it. Three funnels are reported because three different things
can go wrong, and the fixes for them do not overlap:

  PERCEPTION -- was the object ever mapped at its NEW pose? Split first,
      because 40 of C0's 52 failures never perceived it at all and no amount of
      better stopping or ranking helps those.
  SEARCH -- did the surface posterior propose, and then ARRIVE at, the surface
      the object was actually moved to? Measured against the prior map's own
      container tracks, footprint-aware: a counter is three metres long and its
      centre can be two metres from an object resting on it.
  TERMINAL -- once committed to the real object, did the episode convert?

Runs are identified by the `+run_tag` recorded in Hydra's parallel output tree,
not by directory glob: a glob once swept one condition's third scene into
another's bucket and the numbers were wrong until it was caught.

    python scripts/analyze_campaign.py C0 D
    python scripts/analyze_campaign.py --list
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LAYOUT_ROOT = Path("outputs/substituted_layouts")
MAPS_ROOT = Path("outputs/maps_hires")
TARGET_LABELS = {
    "003_cracker_box": "cracker box", "005_tomato_soup_can": "tin can",
    "011_banana": "banana", "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle", "024_bowl": "bowl",
    "025_mug": "mug", "029_plate": "red plate",
}
NEAR_M = 0.5
SURFACE_TOL_M = 0.5


# ------------------------------------------------------------------ discovery

def run_tag_of(run_dir: Path) -> Optional[str]:
    """The `+run_tag` override, read from Hydra's parallel tree.

    The two trees are named by two SEPARATE evaluations of `${now:...}` -- the
    episodes directory from `output_dir`, the config directory by Hydra itself --
    and they can land a second apart. Matching them exactly makes such a run
    invisible: condition M's first scene wrote `20260825_163247` against a Hydra
    dir of `16-32-46`, so a third of the condition silently vanished and the
    headline read 0.450 instead of 0.633.

    A one-second skew has no other cause here, so a small window is safe: two
    eval runs cannot start within seconds of each other on one GPU. The window
    is searched nearest-first and the first tagged hit wins.
    """
    import datetime

    name = run_dir.name
    if len(name) != 15 or name[8] != "_":
        return None
    try:
        stamp = datetime.datetime.strptime(name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    for delta in (0, -1, 1, -2, 2):
        moment = stamp + datetime.timedelta(seconds=delta)
        hydra = (Path("outputs") / moment.strftime("%Y-%m-%d")
                 / moment.strftime("%H-%M-%S") / ".hydra" / "overrides.yaml")
        if not hydra.is_file():
            continue
        for line in hydra.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("+run_tag="):
                return line.split("=", 1)[1]
    return None


def discover() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = collections.defaultdict(list)
    for run in sorted(Path("outputs").glob("2026*_*")):
        if not (run / "episodes.jsonl").is_file():
            continue
        tag = run_tag_of(run)
        if tag:
            out[tag].append(run)
    return out


def load(dirs: List[Path]) -> List[dict]:
    rows = []
    for d in dirs:
        for line in (d / "episodes.jsonl").open(encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------- geometry

_static: Dict[str, Dict[str, Tuple[float, float]]] = {}


def before(scene: str) -> Dict[str, Tuple[float, float]]:
    """Where each target sat BEFORE the relocation."""
    if scene in _static:
        return _static[scene]
    path = LAYOUT_ROOT / scene / "static_scene_config.json"
    out: Dict[str, Tuple[float, float]] = {}
    if path.is_file():
        blob = json.loads(path.read_text(encoding="utf-8"))
        mapping = {int(k): v for k, v in blob["id_handle_mapping"].items()}
        for obj in blob["objects"]:
            label = TARGET_LABELS.get(mapping[int(obj["semantic_id"])])
            if label:
                out[label] = (obj["translation"][0], obj["translation"][2])
    _static[scene] = out
    return out


_containers: Dict[str, Dict[int, dict]] = {}
CONTAINER_CATEGORIES = {
    "table", "desk", "counter", "shelf", "cabinet", "dresser", "nightstand",
    "bed", "sofa", "stool", "bench", "oven", "washing machine", "refrigerator",
}


def container_tracks(scene: str) -> Dict[int, dict]:
    if scene in _containers:
        return _containers[scene]
    path = MAPS_ROOT / scene / f"{scene}.json"
    out: Dict[int, dict] = {}
    if path.is_file():
        for t in json.loads(path.read_text(encoding="utf-8"))["tracks"]:
            if str(t.get("label", "")).lower().replace("_", " ").strip() in CONTAINER_CATEGORIES:
                out[int(t["id"])] = t
    _containers[scene] = out
    return out


def footprint_distance(track: dict, xy: Tuple[float, float]) -> float:
    ax = track.get("axes") or [0.0, 0.0, 0.0]
    dx = max(0.0, abs(track["center"][0] - xy[0]) - float(ax[0]))
    dz = max(0.0, abs(track["center"][2] - xy[1]) - float(ax[2]))
    return math.hypot(dx, dz)


def truth_xy(row: dict) -> Tuple[float, float]:
    p = row["authored_layout"]["target_position"]
    return (p[0], p[2])


# ------------------------------------------------------------------- funnels

def mapped_at_new_pose(row: dict) -> bool:
    now = truth_xy(row)
    return any(math.hypot(t["center"][0] - now[0], t["center"][2] - now[1]) <= NEAR_M
               for t in (row.get("target_tracks") or []))


def committed_to_real_object(row: dict) -> bool:
    now = truth_xy(row)
    if any(math.hypot(c["center"][0] - now[0], c["center"][2] - now[1]) <= NEAR_M
           for c in (row.get("goal_commit_log") or [])):
        return True
    return bool(row["success"]) and row.get("distance_to_goal", 9e9) < NEAR_M


def search_reached_true_surface(row: dict) -> bool:
    now = truth_xy(row)
    tracks = container_tracks(row["scene"])
    for e in (row.get("search_log_events") or []):
        if not (e.get("searched") and e.get("arrived")):
            continue
        t = tracks.get(int(e["container_id"]))
        if t is not None and footprint_distance(t, now) <= SURFACE_TOL_M:
            return True
    return False


def commit_kinds(row: dict) -> set:
    now = truth_xy(row)
    was = before(row["scene"]).get(row["target"])
    kinds = set()
    for c in (row.get("goal_commit_log") or []):
        cx, cz = c["center"][0], c["center"][2]
        if math.hypot(cx - now[0], cz - now[1]) <= NEAR_M:
            kinds.add("correct")
        elif was is not None and math.hypot(cx - was[0], cz - was[1]) <= NEAR_M:
            kinds.add("stale")
        else:
            kinds.add("false")
    return kinds


def family(row: dict) -> str:
    """A commit to the REMEMBERED pose is the correct opening move of a dynamic
    episode, not a false positive -- scoring against the post-move position
    alone counts a good first attempt as an error, a mistake made once here."""
    kinds = commit_kinds(row)
    if row["success"]:
        return "succeeded"
    if not kinds:
        return "never committed"
    if "correct" in kinds:
        return "reached right track, failed"
    if kinds == {"stale"}:
        return "remembered pose only"
    if "false" in kinds and "stale" in kinds:
        return "mixed stale+false"
    return "false positives only"


FAMILIES = ["succeeded", "remembered pose only", "false positives only",
            "mixed stale+false", "reached right track, failed", "never committed"]


# -------------------------------------------------------------------- report


def redetection_split(rows: List[dict]) -> Optional[List[tuple]]:
    """Why did an episode fail to hold a track on the object at its new pose?

    Needs the ground-truth visibility fields the runner writes
    (`gt_in_view_frames`), so it is skipped for runs that predate them. The
    whole point is to separate two causes that every earlier analysis had to
    guess between:

      NEVER LOOKED    the object's new position never entered an unoccluded
                      camera frustum -- a coverage and search problem;
      LOOKED, MISSED  it did, at a range where detection is plausible, and no
                      track came of it -- a detector and admission-gate problem.

    A third bucket, LOOKED FAR, is kept separate rather than folded into either:
    seeing something at six metres is not evidence the detector had a chance.
    """
    if not any("gt_in_view_frames" in r for r in rows):
        return None
    out = collections.Counter()
    won = collections.Counter()
    for r in rows:
        now = truth_xy(r)
        tracks = r.get("target_tracks") or []
        err = min((math.hypot(t["center"][0] - now[0], t["center"][2] - now[1])
                   for t in tracks), default=None)
        if err is not None and err <= 0.25:
            key = "localized (<= 0.25 m)"
        elif int(r.get("gt_in_view_close_frames", 0)) > 0:
            key = "LOOKED at it within 3 m, MISSED"
        elif int(r.get("gt_in_view_frames", 0)) > 0:
            key = "looked, but only from beyond 3 m"
        else:
            key = "NEVER LOOKED at the new pose"
        out[key] += 1
        won[key] += int(r["success"])
    order = ["localized (<= 0.25 m)", "LOOKED at it within 3 m, MISSED",
             "looked, but only from beyond 3 m", "NEVER LOOKED at the new pose"]
    return [(k, out[k], won[k]) for k in order if out[k]]



FRAMING_BUCKETS = ("close_centred", "close_peripheral", "far_centred", "far_peripheral")


def insitu_recall(rows: List[dict]) -> Optional[dict]:
    """What the detector does on the frames the agent actually gets.

    The probe in `scripts/probe_ycb_detection.py` measures recall at AUTHORED
    viewpoints: navigable poses on rings around the object, ranked by how many
    pixels of it they see, top ten kept. That is a best case by construction.
    This is the same question asked of the frames the agent really collected --
    every keyframe where the object was in view and unoccluded, against the
    detections that keyframe produced.

    The gap between the two is the difference between "the detector can see this
    object" and "the detector saw this object", and only the second one decides
    an episode.
    """
    if not any("gt_kf_in_view" in r for r in rows):
        return None
    seen = sum(int(r.get("gt_kf_in_view", 0)) for r in rows)
    hit = sum(int(r.get("gt_kf_detected", 0)) for r in rows)
    out = {"keyframes_in_view": seen, "detected": hit,
           "recall": (hit / seen) if seen else float("nan"), "buckets": {}}
    for key in FRAMING_BUCKETS:
        n = sum(int(r.get(f"gt_kf_{key}", 0)) for r in rows)
        d = sum(int(r.get(f"gt_kf_{key}_detected", 0)) for r in rows)
        out["buckets"][key] = (n, d)
    return out


def _rate(rows, pred) -> str:
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return f"{k:3d}/{n:<3d} {k / n:5.1%}" if n else "   --"


def report(conds: Dict[str, List[dict]]) -> None:
    names = list(conds)
    W = 20

    def head(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    head("HEADLINE")
    print(f"{'':26}" + "".join(f"{n:>{W}}" for n in names))
    def line(label, fn):
        print(f"{label:26}" + "".join(f"{fn(conds[n]):>{W}}" for n in names))
    line("episodes", lambda R: f"{len(R)}")
    line("success rate", lambda R: f"{sum(r['success'] for r in R) / len(R):.3f}")
    line("SPL", lambda R: f"{sum((r['spl'] if r['spl'] == r['spl'] else 0.0) for r in R) / len(R):.3f}")
    line("median steps", lambda R: f"{sorted(r['steps'] for r in R)[len(R) // 2]}")
    line("hit the 500 cap", lambda R: f"{sum(1 for r in R if r['steps'] >= 500)}")
    line("control fps", lambda R: f"{sum(r['control_fps'] for r in R) / len(R):.2f}")

    head("BY LAYOUT TYPE")
    print(f"{'':26}" + "".join(f"{n:>{W}}" for n in names))
    for lt in ["in_anchor", "cross_anchor"]:
        def f(R, lt=lt):
            sub = [r for r in R if r["authored_layout"]["layout_type"] == lt]
            return f"{sum(r['success'] for r in sub) / len(sub):.3f} (n={len(sub)})" if sub else "--"
        print(f"{lt:26}" + "".join(f"{f(conds[n]):>{W}}" for n in names))

    head("THE THREE FUNNELS")
    print(f"{'':38}" + "".join(f"{n:>{W}}" for n in names))
    for label, pred in [
        ("PERCEPTION mapped it at the new pose", mapped_at_new_pose),
        ("SEARCH   ran at all", lambda r: bool(r.get("search_log_events"))),
        ("SEARCH   ARRIVED at the true surface", search_reached_true_surface),
        ("COMMIT   to a track on the real object", committed_to_real_object),
        ("TERMINAL succeeded", lambda r: bool(r["success"])),
    ]:
        print(f"{label:38}" + "".join(f"{_rate(conds[n], pred):>{W}}" for n in names))
    print(f"\n{'conversion once committed':38}"
          + "".join(f"{_conv(conds[n]):>{W}}" for n in names))

    if any(redetection_split(conds[n]) for n in names):
        head("WHY THE OBJECT WAS NOT HELD AT ITS NEW POSE")
        keys = ["localized (<= 0.25 m)", "LOOKED at it within 3 m, MISSED",
                "looked, but only from beyond 3 m", "NEVER LOOKED at the new pose"]
        print(f"{'':36}" + "".join(f"{n:>{W}}" for n in names))
        for key in keys:
            row = f"{key:36}"
            for n in names:
                split = redetection_split(conds[n])
                cell = "--"
                if split:
                    hit = [x for x in split if x[0] == key]
                    if hit:
                        _, cnt, w = hit[0]
                        cell = f"{cnt:3d}  SR {w / cnt:.2f}"
                row += f"{cell:>{W}}"
            print(row)
        print(f"\n{'in-view frames, median':36}"
              + "".join(f"{_median_field(conds[n], 'gt_in_view_frames'):>{W}}" for n in names))
        print(f"{'closest approach while in view':36}"
              + "".join(f"{_median_field(conds[n], 'gt_min_range_m'):>{W}}" for n in names))

    if any(insitu_recall(conds[n]) for n in names):
        head("WHAT THE DETECTOR DOES ON THE FRAMES THE AGENT ACTUALLY GETS")
        print(f"{'':34}" + "".join(f"{n:>{W}}" for n in names))
        for label, fn in (
            ("keyframes with it in view", lambda d: f"{d['keyframes_in_view']}"),
            ("of those, detected by name", lambda d: f"{d['detected']}"),
            ("IN-SITU RECALL", lambda d: f"{d['recall']:.3f}"),
        ):
            row = f"{label:34}"
            for n in names:
                d = insitu_recall(conds[n])
                row += f"{(fn(d) if d else '--'):>{W}}"
            print(row)
        print()
        for key in FRAMING_BUCKETS:
            row = f"  recall, {key:24}"
            for n in names:
                d = insitu_recall(conds[n])
                cell = "--"
                if d:
                    cnt, hit = d["buckets"][key]
                    cell = f"{hit}/{cnt} = {hit / cnt:.2f}" if cnt else "0/0"
                row += f"{cell:>{W}}"
            print(row)
        print()
        print("  Compare the authored-viewpoint probe: 0.584 at imgsz 1280 over the ten")
        print("  best viewpoints of each object. That is what the detector CAN do; the")
        print("  number above is what it DID.")

    head("FAILURE FAMILIES")
    print(f"{'':32}" + "".join(f"{n:>{W}}" for n in names))
    for fam in FAMILIES:
        print(f"{fam:32}" + "".join(
            f"{sum(1 for r in conds[n] if family(r) == fam):>{W}}" for n in names))

    for key, title in (("scene", "BY SCENE"), ("target", "BY TARGET")):
        head(title)
        keys = sorted({r[key] for n in names for r in conds[n]})
        print(f"{key:26}" + "".join(f"{n:>{W}}" for n in names))
        for k in keys:
            row = f"{k:26}"
            for n in names:
                sub = [r for r in conds[n] if r[key] == k]
                cell = "--"
                if sub:
                    cell = f"{sum(r['success'] for r in sub) / len(sub):.3f} (n={len(sub)})"
                row += f"{cell:>{W}}"
            print(row)

    head("AGENT STATS (mean per episode)")
    stats = sorted({k for n in names for r in conds[n] for k in r["agent_stats"]})
    print(f"{'':30}" + "".join(f"{n:>{W}}" for n in names))
    for s in stats:
        vals = [sum(r["agent_stats"].get(s, 0) for r in conds[n]) / len(conds[n]) for n in names]
        if max(vals) < 0.01:
            continue
        print(f"{s:30}" + "".join(f"{v:>{W}.2f}" for v in vals))


def _median_field(rows, field: str) -> str:
    vals = sorted(r[field] for r in rows if r.get(field) is not None)
    return f"{vals[len(vals) // 2]:.2f}" if vals else "--"


def _conv(rows) -> str:
    c = [r for r in rows if committed_to_real_object(r)]
    if not c:
        return "--"
    return f"{sum(r['success'] for r in c):.0f}/{len(c)} = {sum(r['success'] for r in c) / len(c):.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="run tags to compare, in order")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--alias", action="append", default=[],
                    help="TAG=dir,dir,... -- name a condition explicitly. Needed for "
                         "the untagged runs that predate +run_tag, and to exclude a "
                         "stray run that shares a tag with a real one.")
    ap.add_argument("--matched", action="store_true",
                    help="restrict every condition to the episode ids they all share")
    args = ap.parse_args()

    found = discover()
    for spec in args.alias:
        tag, _, dirs = spec.partition("=")
        found[tag] = [Path(d) for d in dirs.split(",") if d]
    if args.list or not args.tags:
        for tag, dirs in sorted(found.items()):
            n = sum(sum(1 for _ in (d / "episodes.jsonl").open()) for d in dirs)
            print(f"{tag:12} {n:4d} episodes over {len(dirs)} runs   "
                  f"{', '.join(d.name for d in dirs)}")
        return

    conds: Dict[str, List[dict]] = {}
    for tag in args.tags:
        if tag not in found:
            raise SystemExit(f"no runs tagged {tag!r}; try --list")
        conds[tag] = load(found[tag])

    if args.matched:
        shared = set.intersection(*({r["episode_id"] for r in rows} for rows in conds.values()))
        conds = {k: [r for r in v if r["episode_id"] in shared] for k, v in conds.items()}
        print(f"matched on {len(shared)} episode ids present in every condition")

    report(conds)


if __name__ == "__main__":
    main()
