"""SR/SPL aggregation + timing summary."""
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
