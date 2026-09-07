"""Semantic value map.

The first test is the acceptance gate for the whole stage. A sign error in the
camera-to-world construction produces a map that is mirrored or rotated: it
looks entirely plausible as a heatmap, passes any test that only checks "some
cells got written", and steers the agent away from the target. So the gate
asserts the observed region lands IN FRONT of the camera and nothing lands
behind it, for headings all the way round.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.costmap import Costmap2D
from osg.mapping.value_map import ValueMap2D

from .conftest import make_camera, make_frame

CAM_H = 0.88


def _map(res=0.1, size_m=30.0):
    cm = Costmap2D(resolution=res, size_m=size_m)
    return cm, ValueMap2D(cm, max_depth_m=5.0)


def _frame_facing(intrinsics, heading_xz, depth=4.0, eye=(0.0, CAM_H, 0.0)):
    """A frame looking along `heading_xz` (world x, z) with uniform depth."""
    d = np.asarray([heading_xz[0], 0.0, heading_xz[1]], dtype=float)
    d /= np.linalg.norm(d)
    return make_frame(intrinsics, make_camera(list(eye), list(np.asarray(eye) + 3 * d)),
                      depth_value=depth)


@pytest.mark.parametrize(
    "heading", [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.7, 0.7)]
)
def test_cone_lands_in_front_of_the_camera(intrinsics, heading):
    cm, vm = _map()
    vm.update(_frame_facing(intrinsics, heading), value=1.0)

    h = np.asarray(heading, dtype=float)
    h /= np.linalg.norm(h)
    ahead = 2.0 * h
    behind = -2.0 * h

    assert vm.value_at(ahead, radius_m=0.4) > 0.5, "nothing painted ahead"
    assert vm.value_at(behind, radius_m=0.4) == 0.0, "painted BEHIND the camera"


def test_nothing_painted_beyond_the_observed_range(intrinsics):
    """Depth 2 m means no evidence at 4 m, however confident the view is."""
    cm, vm = _map()
    vm.update(_frame_facing(intrinsics, (0.0, 1.0), depth=2.0), value=1.0)
    assert vm.value_at(np.array([0.0, 1.0]), radius_m=0.3) > 0.5
    assert vm.value_at(np.array([0.0, 4.0]), radius_m=0.3) == 0.0


def test_unobserved_space_reads_zero(intrinsics):
    cm, vm = _map()
    assert vm.value_at(np.array([1.0, 1.0])) == 0.0


def test_confidence_is_highest_on_the_optical_axis(intrinsics):
    """Angular falloff: the same surface seen head-on must outrank a glancing
    look at it, which is what stops a sweep past a doorway overwriting a good
    observation of the room beyond."""
    cm, vm = _map()
    vm.update(_frame_facing(intrinsics, (0.0, 1.0)), value=1.0)
    rc_axis = cm.world_to_grid(np.array([0.0, 3.0]))
    rc_edge = cm.world_to_grid(np.array([2.0, 3.0]))
    assert vm.conf[rc_axis[0], rc_axis[1]] > vm.conf[rc_edge[0], rc_edge[1]] > 0.0


def test_more_confident_observation_wins(intrinsics):
    """A head-on look must overwrite an earlier glancing one, not the reverse."""
    cm, vm = _map()
    # Glancing: the point sits at the edge of a view aimed elsewhere.
    vm.update(_frame_facing(intrinsics, (1.0, 1.0)), value=0.1)
    # Head-on at the same point.
    vm.update(_frame_facing(intrinsics, (0.0, 1.0)), value=0.9)
    assert vm.value_at(np.array([0.0, 3.0]), radius_m=0.3) == pytest.approx(0.9, abs=1e-5)


def test_survives_a_costmap_grow(intrinsics):
    """The value map must stay pinned to the costmap's frame: after an
    auto-grow a world point has to read the same value it did before."""
    cm, vm = _map(size_m=12.0)
    vm.update(_frame_facing(intrinsics, (0.0, 1.0)), value=0.7)
    probe = np.array([0.0, 3.0])
    before = vm.value_at(probe, radius_m=0.3)
    assert before > 0.0

    cm.ensure_contains(np.array([40.0, 40.0]), margin_m=1.0)

    assert vm.value.shape == cm.grid.shape
    assert vm.value_at(probe, radius_m=0.3) == pytest.approx(before)


def test_reset_clears(intrinsics):
    cm, vm = _map()
    vm.update(_frame_facing(intrinsics, (0.0, 1.0)), value=1.0)
    vm.reset()
    assert vm.value_at(np.array([0.0, 3.0])) == 0.0
