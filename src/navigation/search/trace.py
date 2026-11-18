"""`agent.search.trace`: extra per-step fields for the behaviour log.

Logging only. Everything is read from state the control step already computed;
nothing here calls a model or writes to the agent.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from navigation.vendor.vlfm.utils.geometry_utils import get_point_cloud, transform_points


def weak_points(weak: Sequence[tuple], target: str, depth_n: np.ndarray, tf: np.ndarray,
                min_depth: float, max_depth: float, fx: float, fy: float,
                anchor=None) -> List[list]:
    """`[score, x, y]` for each sub-threshold box of the target class.

    The position is where the box's CENTRE lands: the median depth of the
    middle third of the box, through the same camera model and transform the
    object map uses (`ObjectPointCloudMap._extract_object_cloud`), in the WORLD
    plane the trace's `xy` is logged in when an anchor is given. `[score]`
    alone when that patch has no valid depth. No mask is asked for these boxes,
    so this is a bearing-and-range estimate, not an object cloud.
    """
    want = str(target).lower().replace("_", " ").strip()
    h, w = depth_n.shape[:2]
    out: List[list] = []
    for label, score, box in weak:
        if str(label).lower().replace("_", " ").strip() != want:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        u0, u1 = int(max(0, cx - bw / 6)), int(min(w, cx + bw / 6 + 1))
        v0, v1 = int(max(0, cy - bh / 6)), int(min(h, cy + bh / 6 + 1))
        row = [round(float(score), 3)]
        patch = depth_n[v0:v1, u0:u1]
        valid = patch[patch > 0]
        if valid.size:
            # One pixel at the box centre carrying the patch's median range.
            d = np.zeros((h, w), np.float32)
            m = np.zeros((h, w), bool)
            pv, pu = int(min(h - 1, max(0, cy))), int(min(w - 1, max(0, cx)))
            d[pv, pu] = float(np.median(valid)) * (max_depth - min_depth) + min_depth
            m[pv, pu] = True
            xyz = transform_points(tf, get_point_cloud(d, m, fx, fy))
            if len(xyz):
                xy = np.asarray(xyz[0][:2], float)
                if anchor is not None:
                    xy = anchor.to_world(xy)
                row += [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        out.append(row)
    return out


def step_fields(agent, goal: Optional[np.ndarray], weak: List[list]) -> dict:
    """The extra columns for this step's trace row."""
    om = agent.obstacle_map
    frontiers = np.atleast_2d(np.asarray(om.frontiers)).reshape(-1, 2)
    live = sum(1 for f in frontiers if tuple(f) not in om._disabled_frontiers)
    out = {
        "floor_uid": int(agent._floors[agent._floor_idx]["uid"]),
        "n_floors": len(agent._floors),
        "has_up": int(bool(om._has_up_stair)), "has_dn": int(bool(om._has_down_stair)),
        "exp_up": int(bool(om._explored_up_stair)), "exp_dn": int(bool(om._explored_down_stair)),
        "floor_done": int(bool(om._this_floor_explored)),
        "nfront_live": int(live),
    }
    if goal is not None:
        g = np.asarray(goal[:2], float)
        if agent.anchor is not None:
            g = agent.anchor.to_world(g)
        out["goal_xy"] = [round(float(g[0]), 2), round(float(g[1]), 2)]
    if weak:
        out["weak"] = weak
    return out
