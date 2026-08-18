"""SR/SPL aggregation + timing summary, and the dynamic-scene metrics.

SR alone hides the mechanism: an agent that never notices a change and one that
notices instantly both fail an episode they cannot reach in time, and both
succeed on an easy one. The metrics below measure the noticing itself
(docs/DYNAMIC_SCENES.md, Phase 2).
"""
from __future__ import annotations

from typing import Dict, List


def aggregate(episode_results: List[dict]) -> Dict[str, float]:
    n = len(episode_results)
    if n == 0:
        return {"num_episodes": 0, "success_rate": 0.0, "spl": 0.0}
    sr = sum(r.get("success", 0.0) for r in episode_results) / n
    spl = sum(r.get("spl", 0.0) for r in episode_results) / n
    dtg = sum(r.get("distance_to_goal", 0.0) for r in episode_results) / n
    steps = sum(r.get("steps", 0) for r in episode_results) / n
    return {
        "num_episodes": n,
        "success_rate": round(sr, 4),
        "spl": round(spl, 4),
        "mean_distance_to_goal": round(dtg, 3),
        "mean_steps": round(steps, 1),
    }


def per_category(episode_results: List[dict]) -> Dict[str, Dict[str, float]]:
    by_cat: Dict[str, List[dict]] = {}
    for r in episode_results:
        by_cat.setdefault(r.get("target", "unknown"), []).append(r)
    return {cat: aggregate(rs) for cat, rs in sorted(by_cat.items())}


def per_floor_class(episode_results: List[dict]) -> Dict[str, Dict[str, float]]:
    """SR/SPL split by whether the goal is on the agent's starting floor.

    Multi-floor scenes are the dominant remaining SR loss (see
    docs/MULTI_FLOOR.md); without this split a run reports one number that
    averages two very different regimes. `floor_class` is written per episode
    by eval/floors.py.
    """
    by_fc: Dict[str, List[dict]] = {}
    for r in episode_results:
        by_fc.setdefault(r.get("floor_class", "unknown"), []).append(r)
    return {fc: aggregate(rs) for fc, rs in sorted(by_fc.items())}


def per_scene(episode_results: List[dict]) -> Dict[str, Dict[str, float]]:
    by_scene: Dict[str, List[dict]] = {}
    for r in episode_results:
        by_scene.setdefault(r.get("scene", "unknown"), []).append(r)
    return {s: aggregate(rs) for s, rs in sorted(by_scene.items())}


# ------------------------------------------------------------ dynamic scenes

GHOST_RADIUS_M = 1.0
STALE_P = 0.5


def _relocation(r: dict) -> dict:
    meta = r.get("authored_layout") or {}
    reloc = meta.get("relocation") or {}
    return reloc if isinstance(reloc, dict) else {}


def belief_latency(episode_results: List[dict]) -> Dict[str, float]:
    """Steps from the relocation to the map first disbelieving the target.

    Only counts episodes where the object was actually relocated. An episode
    where the belief never flipped is reported in `flip_rate`, not folded into
    the mean as a zero -- averaging in a non-event would make a system that
    never notices look fast.
    """
    latencies: List[int] = []
    n_reloc = 0
    for r in episode_results:
        reloc = _relocation(r)
        step = reloc.get("step")
        if step is None:
            continue
        n_reloc += 1
        target = str(r.get("target", "")).lower()
        after = [
            e for e in r.get("presence_events", [])
            if int(e.get("step", -1)) >= int(step)
            and str(e.get("label", "")).lower() == target
        ]
        if after:
            latencies.append(int(min(e["step"] for e in after)) - int(step))
    if not n_reloc:
        return {}
    out = {
        "n_relocated": n_reloc,
        "flip_rate": round(len(latencies) / n_reloc, 4),
    }
    if latencies:
        ordered = sorted(latencies)
        out["mean_steps"] = round(sum(ordered) / len(ordered), 1)
        out["median_steps"] = float(ordered[len(ordered) // 2])
    return out


def stale_goal_rate(episode_results: List[dict]) -> Dict[str, float]:
    """Share of goal commitments made to an object the map had already stopped
    believing in. This is the bucket DualMap reports as false matches."""
    total = stale = 0
    for r in episode_results:
        for commit in r.get("goal_commit_log", []):
            total += 1
            if float(commit.get("p", 1.0)) < STALE_P:
                stale += 1
    if not total:
        return {}
    return {"n_commits": total, "stale_rate": round(stale / total, 4)}


def ghost_rate(episode_results: List[dict]) -> Dict[str, float]:
    """Share of relocation episodes still believing the target sits where it
    used to. A map that only ever adds evidence scores 1.0 here by construction.
    """
    import math

    n = ghosts = 0
    for r in episode_results:
        reloc = _relocation(r)
        origin = reloc.get("origin_position")
        if reloc.get("step") is None or origin is None:
            continue
        n += 1
        target = str(r.get("target", "")).lower()
        for track in r.get("target_tracks", []):
            if str(track.get("label", "")).lower() != target:
                continue
            if float(track.get("p", 0.0)) < STALE_P:
                continue
            d = math.dist(track.get("center", [0, 0, 0]), origin)
            if d <= GHOST_RADIUS_M:
                ghosts += 1
                break
    if not n:
        return {}
    return {"n_relocated": n, "ghost_rate": round(ghosts / n, 4)}


def dynamic_summary(episode_results: List[dict]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, fn in (
        ("belief_latency", belief_latency),
        ("stale_goals", stale_goal_rate),
        ("ghosts", ghost_rate),
    ):
        value = fn(episode_results)
        if value:
            out[name] = value
    return out
