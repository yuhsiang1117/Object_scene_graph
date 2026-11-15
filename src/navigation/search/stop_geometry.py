"""Where to stand for an object, given the geometry the object layer reports.

`ascentnav` never reads that geometry: it walks to the nearest point of
ASCENT's own cloud and stops inside 1 m of it (`agent.py:_navigate`). To ask
whether an ellipsoid's centre and size change where the agent ends up, the
geometry has to decide the stop, so this is the one rule every geometry is
put through -- OSG's own two pieces, composed:

    stop = centre + unit(agent - centre) * min(radius + standoff, |agent - centre|)

`radius` is the reported geometry's largest semi-axis
(`osg/agent/nav_agent.py:_horizontal_radius_m`, the viewpoint ring's radius)
and `standoff` is `agent.approach_standoff_m` along the ray toward the agent
(`osg/agent/approach.py:_standoff_goal`). A size error changes how far back
the point sits, one for one. A centre error along the ray moves it one for
one; a centre error `e` across the ray moves it `e * (1 - (radius + standoff)
/ D)` for an agent `D` away, because the ray swings with the centre -- so a
sideways error matters less the closer the agent already is.

Pure: scripts/eval_stop_geometry.py scores it offline against Habitat's
success rule and drives to it in the simulator.
"""
from __future__ import annotations

import numpy as np


def horizontal_radius(axes) -> float:
    """The largest semi-axis; 0 when there is none to read."""
    try:
        r = float(np.max(np.abs(np.asarray(axes, dtype=float))))
    except (TypeError, ValueError):
        return 0.0
    return r if np.isfinite(r) else 0.0


def stop_point(centre_xy, radius: float, agent_xy, standoff: float) -> np.ndarray:
    """The stand-off point on the agent's side of the object. An agent already
    inside the stand-off stays where it is rather than being sent backwards."""
    centre = np.asarray(centre_xy, dtype=float)[:2]
    to_agent = np.asarray(agent_xy, dtype=float)[:2] - centre
    dist = float(np.linalg.norm(to_agent))
    if dist < 1e-3:
        return centre.copy()
    return centre + to_agent / dist * min(float(radius) + float(standoff), dist)
