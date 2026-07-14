"""Top-down trajectory visualization (paper Fig. 5 style): costmap, agent
path, frontiers, object ellipsoid footprints and the target marker.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..mapping.costmap import FREE, OCCUPIED, PLANE, Costmap2D


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
