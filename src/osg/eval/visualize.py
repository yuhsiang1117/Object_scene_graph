"""Top-down trajectory visualization (paper Fig. 5 style): costmap, agent
path, frontiers, object ellipsoid footprints and the target marker.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mapping.costmap import FREE, OCCUPIED, PLANE, Costmap2D

# ---------------------------------------------------------------------------
# Fast per-step debug rendering (cv2, no matplotlib): a segmentation overlay on
# the live RGB and a top-down costmap, stitched by the runner into a per-episode
# debug video when eval.debug_frames is set.
# ---------------------------------------------------------------------------
_PALETTE = [
    (0, 200, 0), (200, 120, 0), (0, 160, 220), (180, 0, 200),
    (0, 210, 210), (120, 120, 255), (60, 180, 75), (245, 130, 48),
]
_TARGET_BGR = (0, 0, 255)  # target category drawn in red


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def overlay_segmentation(rgb: np.ndarray, dets, target: str) -> np.ndarray:
    """BGR image of `rgb` with translucent YOLOE masks + boxes + label(score);
    the target category is highlighted in red."""
    import cv2

    img = np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR
    tgt = _norm(target)
    layer = img.copy()
    for i, d in enumerate(dets):
        col = _TARGET_BGR if _norm(d.label) == tgt else _PALETTE[i % len(_PALETTE)]
        layer[d.mask.astype(bool)] = col
    img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
    for i, d in enumerate(dets):
        is_t = _norm(d.label) == tgt
        col = _TARGET_BGR if is_t else _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if is_t else 1)
        txt = f"{d.label} {d.score:.2f}"
        for c, th in ((( 0, 0, 0), 3), (col, 1)):
            cv2.putText(img, txt, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, th, cv2.LINE_AA)
    return img


def render_costmap_bgr(
    costmap: Costmap2D,
    agent_xy: Optional[np.ndarray] = None,
    trajectory_xy: Optional[List[np.ndarray]] = None,
    frontiers: Optional[list] = None,
    path_xy: Optional[np.ndarray] = None,
    chosen_frontier=None,
    out_h: int = 480,
) -> np.ndarray:
    """Top-down costmap as a BGR image: unknown gray, free white, occupied dark.
    Overlays (drawn at output scale so markers stay crisp): trajectory (orange),
    the planned path to the current goal (green), all frontiers (small cyan
    dots), the CHOSEN frontier the agent is heading to (large yellow ring +
    cross), and the agent (red dot)."""
    import cv2

    grid = costmap.grid
    img = np.full((*grid.shape, 3), 140, np.uint8)  # unknown gray
    img[grid == FREE] = (245, 245, 245)
    img[grid == OCCUPIED] = (40, 40, 50)

    scale = out_h / grid.shape[0]
    img = cv2.resize(img, (int(grid.shape[1] * scale), out_h), interpolation=cv2.INTER_NEAREST)

    def to_px(xy):
        rc = costmap.world_to_grid(np.asarray(xy, float))
        return int(rc[1] * scale), int(rc[0] * scale)  # cv2 (x=col, y=row)

    if trajectory_xy:
        cv2.polylines(img, [np.array([to_px(p) for p in trajectory_xy], np.int32)],
                      False, (0, 165, 255), 1)
    if path_xy is not None and len(path_xy) > 1:
        cv2.polylines(img, [np.array([to_px(p) for p in path_xy], np.int32)],
                      False, (0, 220, 0), 2)
    for f in (frontiers or []):
        cv2.circle(img, to_px(f.centroid_xy), 3, (200, 200, 0), -1)
    if chosen_frontier is not None:
        px = to_px(chosen_frontier.centroid_xy)
        cv2.circle(img, px, 9, (0, 255, 255), 2)
        cv2.drawMarker(img, px, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    if agent_xy is not None:
        cv2.circle(img, to_px(agent_xy), 5, (0, 0, 255), -1)
    return img


def save_topdown(
    path_png: str,
    costmap: Costmap2D,
    trajectory_xy: List[np.ndarray],
    scene_graph=None,
    target_xy: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = costmap.grid
    img = np.full((*grid.shape, 3), 0.55)  # unknown gray
    img[grid == FREE] = (0.95, 0.95, 0.95)
    img[grid == OCCUPIED] = (0.15, 0.15, 0.2)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [
        costmap.origin[0],
        costmap.origin[0] + grid.shape[0] * costmap.resolution,
        costmap.origin[1],
        costmap.origin[1] + grid.shape[1] * costmap.resolution,
    ]
    # grid rows are plane axis 0, cols plane axis 1; show axis 0 on x
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=extent)

    if trajectory_xy:
        traj = np.stack(trajectory_xy)
        ax.plot(traj[:, 0], traj[:, 1], color="orange", lw=2, label="path")
        ax.plot(traj[0, 0], traj[0, 1], "o", color="tab:blue", ms=8, label="start")

    if scene_graph is not None:
        for obj in scene_graph.objects:
            xy = obj.center[list(PLANE)]
            ax.plot(xy[0], xy[1], "^", color="tab:green", ms=5)
            ax.annotate(obj.label, xy, fontsize=6, alpha=0.8)

    if target_xy is not None:
        ax.plot(target_xy[0], target_xy[1], "*", color="red", ms=14, label="target")

    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    Path(path_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
