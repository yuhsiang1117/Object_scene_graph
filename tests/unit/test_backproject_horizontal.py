"""`core.geometry.backproject_horizontal`: only the depth pixels on a roughly
horizontal surface -- a floor or a tread, not a wall or a riser face.

A wall's column below the camera is what turned 00821's switchback landing
into a route north into the wall (`agent.climb_surfaces_horizontal`).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osg.core.geometry import backproject, backproject_horizontal

K = SimpleNamespace(fx=100.0, fy=100.0, cx=80.0, cy=60.0)
# OpenCV camera (x right, y down, z forward) 1 m above the floor, looking along
# world -z: world = R @ cam + t with y up.
T = np.eye(4)
T[:3, :3] = np.diag([1.0, -1.0, -1.0])
T[:3, 3] = [0.0, 1.0, 0.0]


def _floor_and_wall(wall_z=3.0, h=120, w=160):
    vs, _ = np.mgrid[0:h, 0:w]
    b = (vs - K.cy) / K.fy                    # downward slope of each pixel's ray
    with np.errstate(divide="ignore"):
        floor = np.where(b > 0, 1.0 / b, np.inf)
    return np.minimum(floor, wall_z).astype(np.float32)


def test_the_floor_stays_and_the_wall_goes():
    depth = _floor_and_wall()
    every = backproject(depth, K, T, max_depth=10.0)
    flat = backproject_horizontal(depth, K, T, max_depth=10.0)
    wall = np.abs(every[:, 2] + 3.0) < 1e-6
    assert wall.sum() > 1000 and (every[wall, 1] > 0.5).any()
    assert len(flat) > 1000
    assert np.allclose(flat[:, 1], 0.0, atol=1e-6)            # floor only


def test_a_tread_is_horizontal_and_its_riser_is_not():
    """A step 0.2 m up at 2 m: the tread's top stays, its riser face does not."""
    h, w = 120, 160
    vs, _ = np.mgrid[0:h, 0:w]
    b = (vs - K.cy) / K.fy
    with np.errstate(divide="ignore"):
        floor = np.where(b > 0, 1.0 / b, np.inf)          # y = 0
        tread = np.where(b > 0, 0.8 / b, np.inf)          # y = 0.2
    riser_hit = (b * 2.0 <= 1.0) & (b * 2.0 >= 0.8)       # the face at z = -2, y in [0, 0.2]
    depth = np.where(floor < 2.0, floor, np.where(riser_hit, 2.0, np.minimum(tread, 6.0)))
    flat = backproject_horizontal(depth.astype(np.float32), K, T, max_depth=10.0)
    heights = np.round(flat[:, 1], 3)
    assert set(np.unique(heights)) <= {0.0, 0.2}
    assert (heights == 0.2).sum() > 100
