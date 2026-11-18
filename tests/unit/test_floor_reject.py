"""Cross-floor frame rejection (mapping.floor_reject_m).

The costmap is a single ground plane. A frame captured a storey above the floor
the map was started on back-projects with rel_h = y - floor_y reading 1-3 m, so
the upper floor's geometry is written onto the lower floor's grid and corrupts
frontier extraction. These tests pin that such frames are dropped when the flag
is on, and that the flag defaults to the pre-existing behaviour.
"""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.exploration.async_scorer import AsyncScorer
from osg.perception.detector import StubDetector

from .conftest import make_camera, make_frame
from .test_nav_agent import _StubScorer, make_cfg

CAMERA_HEIGHT = 0.88
STOREY_M = 2.5


def _agent(floor_reject_m: float) -> NavAgent:
    cfg = make_cfg()
    cfg.mapping.floor_reject_m = floor_reject_m
    # A real NavAgent, but with an empty costmap (unlike test_nav_agent's
    # make_agent, which carves it all FREE) so coverage_cells() reflects only
    # what the frames below wrote.
    return NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair")


def _frame(intrinsics, floor_y: float, look_dz: float = 3.0, frame_id: int = 0):
    """A frame looking horizontally at a wall 3 m away, standing on `floor_y`.

    `look_dz` picks the heading. The tests below map +z first and then probe
    with a -z frame, because an upper storey only *shows* as pollution where it
    writes cells the current floor has not already mapped -- looking the same
    way from the same spot writes the identical (x, z) footprint whatever the
    height, so it is invisible to any grid probe.
    """
    eye = [0.0, floor_y + CAMERA_HEIGHT, 0.0]
    return make_frame(
        intrinsics,
        make_camera(eye, [eye[0], eye[1], eye[2] + look_dz]),
        depth_value=3.0,
        frame_id=frame_id,
    )


def test_on_plane_frame_is_mapped(intrinsics):
    agent = _agent(0.4)
    agent.act(_frame(intrinsics, floor_y=0.0))
    assert agent.costmap.coverage_cells() > 0
    assert agent.stats.get("frames_off_plane", 0) == 0


def test_off_plane_frame_is_skipped(intrinsics):
    agent = _agent(0.4)
    agent.act(_frame(intrinsics, floor_y=0.0, look_dz=3.0, frame_id=0))
    before = agent.costmap.grid.copy()

    agent.act(_frame(intrinsics, floor_y=STOREY_M, look_dz=-3.0, frame_id=1))

    assert np.array_equal(agent.costmap.grid, before), "upper storey polluted the map"
    assert agent.stats["frames_off_plane"] == 1


def test_disabled_by_default_lets_the_upper_floor_through(intrinsics):
    """floor_reject_m=0 must reproduce the behaviour from before this existed,
    and documents the failure being fixed: a storey up, the geometry is still
    inside the stale floor's [obstacle_low, obstacle_high] band (the vertical
    FOV spans +-2.25 m at 3 m), so ~3700 cells of the upper floor get written
    onto the lower floor's grid. Nothing about the height filters it out -- the
    costmap has no idea the frame came from another storey."""
    agent = _agent(0.0)
    agent.act(_frame(intrinsics, floor_y=0.0, look_dz=3.0, frame_id=0))
    before = agent.costmap.grid.copy()

    agent.act(_frame(intrinsics, floor_y=STOREY_M, look_dz=-3.0, frame_id=1))

    assert (agent.costmap.grid != before).sum() > 1000
    assert agent.stats.get("frames_off_plane", 0) == 0


def test_small_excursion_still_mapped(intrinsics):
    """A threshold / rug / ramp is not a floor change: staying within the band
    must not stop mapping, or the agent goes blind on uneven ground."""
    agent = _agent(0.4)
    agent.act(_frame(intrinsics, floor_y=0.0, look_dz=3.0, frame_id=0))
    before = agent.costmap.grid.copy()

    agent.act(_frame(intrinsics, floor_y=0.25, look_dz=-3.0, frame_id=1))

    assert (agent.costmap.grid != before).sum() > 1000, "small step stopped mapping"
    assert agent.stats.get("frames_off_plane", 0) == 0


def test_floor_y_is_the_standing_surface(intrinsics):
    """_floor_y must be the surface height, not the camera height -- the
    rejection band is metres of storey, so a 0.88 m offset would put a
    same-floor frame most of the way to the threshold."""
    agent = _agent(0.4)
    agent.act(_frame(intrinsics, floor_y=0.0))
    assert agent._floor_y == 0.0
    assert agent._off_plane_m == 0.0
