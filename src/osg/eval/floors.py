"""Floor classification for HM3D ObjectNav episodes and trajectories.

Habitat is y-up, so "which floor" is a question about the height axis alone.
Two different thresholds are needed and they are not the same number:

* `SAME_FLOOR_M` (0.5) -- how far apart two *standing poses* can be and still
  be the same level. Goal view points on one floor scatter by a few cm; the
  gap to the next floor is >2 m in every HM3D scene.
* `FLOOR_CHANGE_M` (1.5) -- how far the agent must actually move vertically,
  and then settle, before we call it a floor change. Deliberately larger than
  SAME_FLOOR_M so that standing mid-staircase does not count.

An episode is **cross_floor** when no goal view point lies within
`SAME_FLOOR_M` of the agent's start height: the agent cannot succeed without
changing level. Otherwise **same_floor**. This is the per-episode split; the
per-scene single/multi-floor split (which built
`configs/eval/hm3d_val_single_floor.yaml`) comes from the vertical spread of
all goal view points in a scene -- see `scripts/scene_floors.py`.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

HEIGHT_AXIS = 1

SAME_FLOOR_M = 0.5
FLOOR_CHANGE_M = 1.5
# Consecutive steps the agent must hold a new height before it counts as
# settled on another floor (at 0.25 m/step this is ~2.5 m of travel).
FLOOR_SETTLE_STEPS = 10


def navmesh_floor_heights(
    pathfinder,
    bin_m: float = 0.05,
    peak_window_m: float = 0.4,
    min_area_frac: float = 0.15,
    merge_m: float = 1.0,
) -> List[float]:
    """Floor heights of a scene from its Habitat navmesh, area-weighted.

    Slice the navmesh at `bin_m` intervals and take the navigable AREA at each
    height (`get_topdown_view`), then keep local maxima that hold more than
    `min_area_frac` of the largest slice, merging maxima closer than `merge_m`.

    Area, not vertex density: navmesh vertex counts are biased by mesh
    tessellation. And the histogram must be over area rather than a gap-cluster
    over vertex heights -- a staircase contributes vertices continuously across
    the whole vertical range, so gap-clustering vertices reports one floor for
    any scene whose stairs are navigable.

    This is the scene-level ground truth (it finds floors that hold no goal
    objects, which the goal-view-point clustering cannot). Requires habitat_sim
    and the scene meshes, so it is optional -- see scripts/scene_floors.py.
    """
    import numpy as np

    verts = np.asarray(pathfinder.build_navmesh_vertices())
    if verts.size == 0:
        return []
    ys = np.arange(verts[:, HEIGHT_AXIS].min(), verts[:, HEIGHT_AXIS].max() + bin_m, bin_m)
    area = np.array(
        [float(np.asarray(pathfinder.get_topdown_view(0.1, float(y))).sum()) for y in ys]
    )
    if area.max() <= 0:
        return []

    w = max(1, int(peak_window_m / bin_m))
    peaks: List[int] = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w), min(len(ys), i + w + 1)
        if area[i] == area[lo:hi].max() and area[i] > min_area_frac * area.max():
            peaks.append(i)

    merged: List[int] = []
    for i in peaks:
        if merged and ys[i] - ys[merged[-1]] < merge_m:
            if area[i] > area[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return [round(float(ys[i]), 3) for i in merged]


def goal_view_heights(episode) -> List[float]:
    """Heights of every goal view point of a habitat ObjectGoal episode.

    Falls back to the goal centres when an episode carries no view points (the
    v1 episodes do have them, but a stub/synthetic episode in tests may not).
    """
    heights: List[float] = []
    for goal in getattr(episode, "goals", None) or []:
        vps = getattr(goal, "view_points", None) or []
        for vp in vps:
            state = getattr(vp, "agent_state", None)
            pos = getattr(state, "position", None) if state is not None else None
            if pos is not None:
                heights.append(float(pos[HEIGHT_AXIS]))
        if not vps:
            pos = getattr(goal, "position", None)
            if pos is not None:
                heights.append(float(pos[HEIGHT_AXIS]))
    return heights


def classify_episode(
    start_y: float, goal_ys: Sequence[float], same_floor_m: float = SAME_FLOOR_M
) -> str:
    """`same_floor` / `cross_floor` / `unknown` (no goal heights available)."""
    if not goal_ys:
        return "unknown"
    if any(abs(y - start_y) <= same_floor_m for y in goal_ys):
        return "same_floor"
    return "cross_floor"


def count_floor_changes(
    ys: Iterable[float],
    change_m: float = FLOOR_CHANGE_M,
    settle_steps: int = FLOOR_SETTLE_STEPS,
) -> int:
    """How many times the trajectory settled on a new level.

    A candidate only commits after the agent has held a height more than
    `change_m` from the current reference for `settle_steps` consecutive
    samples, so climbing halfway up a staircase and coming back down counts as
    zero -- unlike a bare threshold crossing, which would count two.
    """
    ys = list(ys)
    if not ys:
        return 0
    ref = ys[0]
    n = 0
    cand: Optional[float] = None
    held = 0
    for y in ys:
        if abs(y - ref) > change_m:
            if cand is None or abs(y - cand) > SAME_FLOOR_M:
                cand, held = y, 1
            else:
                held += 1
                if held >= settle_steps:
                    ref, cand, held = y, None, 0
                    n += 1
        else:
            cand, held = None, 0
    return n


def episode_floor_fields(episode, trajectory_y: Sequence[float]) -> dict:
    """The per-episode floor block written into `episodes.jsonl`."""
    start = getattr(episode, "start_position", None)
    start_y = float(start[HEIGHT_AXIS]) if start is not None else (
        float(trajectory_y[0]) if len(trajectory_y) else 0.0
    )
    goal_ys = goal_view_heights(episode)
    return {
        "floor_class": classify_episode(start_y, goal_ys),
        "start_y": round(start_y, 3),
        "final_y": round(float(trajectory_y[-1]), 3) if len(trajectory_y) else None,
        "traj_y_range": (
            round(float(max(trajectory_y) - min(trajectory_y)), 3)
            if len(trajectory_y) else None
        ),
        "floor_changes": count_floor_changes(trajectory_y),
        "goal_y_span": round(max(goal_ys) - min(goal_ys), 3) if goal_ys else None,
    }


def runtime_floor_fields(episode, trajectory_y: Sequence[float], agent, authored: dict) -> dict:
    """Stable floor identities and cross-floor relocation telemetry.

    Heights remain in the older fields for source compatibility.  These fields
    bind them to FloorStack's persistent keys, whose numeric value never implies
    vertical order.
    """
    levels = dict(getattr(getattr(agent, "floors", None), "levels", {}) or {})

    def key_at(y: Optional[float]) -> Optional[int]:
        if y is None or not levels:
            return None
        key = min(levels, key=lambda item: abs(float(levels[item]) - float(y)))
        if abs(float(levels[key]) - float(y)) > SAME_FLOOR_M:
            return None
        return int(key)

    start = getattr(episode, "start_position", None)
    start_y = float(start[HEIGHT_AXIS]) if start is not None else (
        float(trajectory_y[0]) if trajectory_y else None
    )
    goal_ys = goal_view_heights(episode)
    goal_y = min(goal_ys, key=lambda y: abs(y - start_y)) if goal_ys and start_y is not None else (
        goal_ys[0] if goal_ys else None
    )
    relocation = authored.get("relocation") or {}
    origin = relocation.get("origin_position") if isinstance(relocation, dict) else None
    destination = relocation.get("destination_position") if isinstance(relocation, dict) else None
    prior_y = authored.get("prior_floor_y")
    if prior_y is None and origin is not None:
        prior_y = float(origin[HEIGHT_AXIS])
    destination_y = authored.get("destination_floor_y")
    if destination_y is None:
        destination_y = float(destination[HEIGHT_AXIS]) if destination is not None else goal_y

    start_floor = key_at(start_y)
    goal_floor = key_at(destination_y)
    prior_floor = key_at(prior_y)
    direction = authored.get("relocation_direction")
    if prior_floor is not None and goal_floor is not None:
        if prior_floor == goal_floor:
            direction = "same_floor"
        else:
            direction = "upward" if levels[goal_floor] > levels[prior_floor] else "downward"
    if direction is None and prior_y is not None and destination_y is not None:
        delta = float(destination_y) - float(prior_y)
        direction = (
            "same_floor" if abs(delta) <= SAME_FLOOR_M
            else "upward" if delta > 0.0 else "downward"
        )
    goal_height = (
        levels.get(goal_floor) if goal_floor is not None else destination_y
    )
    reached = None if goal_height is None else any(
        abs(float(y) - float(goal_height)) <= SAME_FLOOR_M for y in trajectory_y
    )
    exploration = getattr(agent, "exploration", None)
    selected = getattr(exploration, "selected_search_floor", None)
    stack = getattr(getattr(agent, "floors", None), "stack", None)
    stats = dict(getattr(agent, "stats", {}) or {})
    return {
        "start_floor": start_floor,
        "goal_floor": goal_floor,
        "prior_floor": prior_floor,
        "relocation_direction": direction,
        "selected_search_floor": None if selected is None else int(selected),
        "floor_switches": len(getattr(stack, "stair_edges", []) or []),
        "climb_attempts": int(
            stats.get("climb_attempt", 0) + stats.get("floor_switch_attempts", 0)
        ),
        "goal_floor_reached": reached,
    }
