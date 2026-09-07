"""Per-step debug rendering for `AscentNavAgent`.

Three panels, left to right:

    RGB (+ target detections) | obstacle map | value map

The two map panels are ASCENT's own visualisers -- `ObstacleMap.visualize`
(`obstacle_map.py:765`) and `ValueMap.visualize` (`value_map.py:202`) -- so what
is drawn is what the pipeline actually holds, not a re-derivation that could
disagree with it. Both flip vertically at the end, hence `_vis_px`.

What this file adds on top of them is the decision: which frontier was CHOSEN
this step (the vendored visualiser draws all of them identically), where the
object goal is, and what the climb is aiming at.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# BGR, matching the vendored visualiser's own choices where they overlap.
SELECTED = (0, 0, 255)      # chosen frontier -- filled red
FRONTIER = (200, 0, 0)      # candidate frontier -- blue ring (drawn by ObstacleMap)
GOAL = (0, 200, 255)        # object goal -- amber
CARROT = (42, 42, 165)      # stair carrot
AGENT = (0, 128, 0)


def _vis_px(om, xy) -> tuple:
    """World (x, y) -> pixel in the FLIPPED visualisation image."""
    px = om._xy_to_px(np.atleast_2d(np.asarray(xy, dtype=float)))[0]
    return int(px[0]), int(om._map.shape[0] - 1 - px[1])


def _explored_bbox(om, pad: int = 30):
    """Crop both maps to what has actually been seen -- the raw map is 1600 px
    of 80 m and an episode covers a few metres of it."""
    mask = (om.explored_area == 1) | (om._map == 1)
    ys, xs = np.where(mask)
    h, w = om._map.shape[:2]
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    # in the flipped frame the rows invert
    return int(x0), int(h - y1), int(x1), int(h - y0)


def _fit(img, size: int):
    """Letterbox to a square so the two map panels stay aligned frame to frame
    even as the explored region grows."""
    h, w = img.shape[:2]
    s = min(size / max(w, 1), size / max(h, 1))
    out = np.full((size, size, 3), 40, np.uint8)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def _label(img, text: str, y: int, color=(255, 255, 255), scale=0.5) -> None:
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _thin_trajectory(m) -> None:
    """The vendored `TrajectoryVisualizer` draws a 3 px path over a 20 px/m map,
    which at map scale is a 15 cm-wide green ribbon that hides the obstacles
    underneath it. Viz only; set before the first draw because the path mask is
    cached (`traj_visualizer.py:10-17`)."""
    tv = getattr(m, "_traj_vis", None)
    if tv is not None:
        tv.path_thickness = 1
        tv.agent_line_thickness = 2


def obstacle_panel(agent, robot_xy, heading) -> np.ndarray:
    om = agent.obstacle_map
    _thin_trajectory(om)
    img = om.visualize()          # explored / obstacles / stairs / all frontiers

    # Frontiers this agent has retired. The vendored visualiser greys out
    # `ObstacleMap._disabled_frontiers`, but this agent keeps its own set
    # (`_sticky`), so without this they would still be drawn as live blue rings.
    for f in getattr(agent, "_disabled_frontiers", ()) or ():
        cv2.drawMarker(img, _vis_px(om, np.array(f)), (128, 128, 128),
                       cv2.MARKER_TILTED_CROSS, 9, 2)

    # the decision the vendored visualiser cannot show: which one was picked
    sel = getattr(agent, "_selected_frontier", None)
    if sel is not None:
        cv2.circle(img, _vis_px(om, sel), 7, SELECTED, -1)
        cv2.line(img, _vis_px(om, robot_xy), _vis_px(om, sel), SELECTED, 1, cv2.LINE_AA)
    goal = getattr(agent, "_nav_goal", None)
    if goal is not None:
        cv2.drawMarker(img, _vis_px(om, goal), GOAL, cv2.MARKER_STAR, 16, 2)
    carrot = getattr(agent.climb, "carrot_xy", None) if hasattr(agent, "climb") else None
    if carrot is not None:
        cv2.circle(img, _vis_px(om, carrot), 5, CARROT, 2)

    # Heading arrow. `_xy_to_px` maps world +x to a DECREASING row (after the
    # flip, upward) and world +y to a DECREASING column (leftward), so forward
    # (cos h, sin h) becomes (-sin h, -cos h) in image axes. Getting this wrong
    # draws an arrow that looks plausible and points 90 degrees off.
    p = _vis_px(om, robot_xy)
    cv2.circle(img, p, 4, AGENT, -1)
    cv2.line(img, p, (int(p[0] - 18 * np.sin(heading)), int(p[1] - 18 * np.cos(heading))),
             AGENT, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = _explored_bbox(om)
    return img[y0:y1, x0:x1]


def value_panel(agent) -> np.ndarray:
    om = agent.obstacle_map
    _thin_trajectory(agent.value_map)
    img = agent.value_map.visualize(obstacle_map=om)
    x0, y0, x1, y1 = _explored_bbox(om)
    return img[y0:y1, x0:x1]


def rgb_panel(rgb: np.ndarray, dets, target: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for d in dets or []:
        x0, y0, x1, y1 = [int(v) for v in d.bbox_xyxy]
        cv2.rectangle(img, (x0, y0), (x1, y1), GOAL, 2)
        _label(img, f"{d.label} {d.score:.2f}", max(14, y0 - 6), GOAL, 0.45)
    return img


def debug_panel(agent, frame, target: str, robot_xy, heading, dets=None,
                size: int = 480) -> np.ndarray:
    """One video frame. Fixed output size so the writer never has to resize."""
    # The RGB keeps its own 4:3 -- letterboxing it into a square wasted a third
    # of the panel on black bars.
    rgb = cv2.resize(rgb_panel(frame.rgb, dets, target), (int(size * 4 / 3), size))
    obs = _fit(obstacle_panel(agent, robot_xy, heading), size)
    val = _fit(value_panel(agent), size)
    panel = cv2.hconcat([rgb, obs, val])
    x_obs = rgb.shape[1]

    # Dark strips so the header and the legend stay readable over a white map.
    panel[:52] = (panel[:52] * 0.25).astype(np.uint8)
    panel[-22:] = (panel[-22:] * 0.25).astype(np.uint8)

    state = getattr(agent, "_state", "?")
    sel = getattr(agent, "_selected_frontier", None)
    n_f = len(np.atleast_2d(np.asarray(agent.obstacle_map.frontiers)).reshape(-1, 2))
    _label(panel, f"{target}   step {agent.step_count}   state={state}", 20)
    _label(panel, f"frontiers={n_f}"
                  + (f"   selected=({sel[0]:.1f}, {sel[1]:.1f})" if sel is not None else "")
                  + (f"   climb dir={agent.climb.direction} reached={int(agent.climb.reached)}"
                     f" centroid={int(agent.climb.reached_centroid)}" if agent.climb.climbing else ""),
                 40)
    _label(panel, "RGB + detections", size - 7, (200, 200, 200), 0.45)
    _label(panel, "maps: +x up, +y left", 60, (180, 180, 180), 0.42)
    cv2.putText(panel, "obstacle map: blue=frontier red=selected purple=up-stair orange=down-stair",
                (x_obs + 8, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, "obstacle map: blue=frontier red=selected purple=up-stair orange=down-stair",
                (x_obs + 8, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, "value map (CLIP cosine, inferno)", (x_obs + size + 8, size - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, "value map (CLIP cosine, inferno)", (x_obs + size + 8, size - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    return panel
