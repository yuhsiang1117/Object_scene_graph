"""A portal that did not lead anywhere must not be proposed again.

`find_portals` is recomputed from scratch every selection round and keeps no
state, so the patch it ranks first stays first until the map changes. Measured on
00821's cracker box: 55 switch attempts, the last 50 of them at one 28-cell patch
at (1.19, -7.96), every one reporting a correct +2.68 m gap, for 0.17 m of ascent
in 500 steps. The agent arrived once and learned nothing from it.
"""
from __future__ import annotations

import numpy as np

from osg.agent.floor_policy import FloorPolicy

from .test_nav_agent import make_cfg


def _policy(memory: bool, radius: float = 1.5):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.portal_failure_memory = memory
    cfg.floor.portal_failure_radius_m = radius
    policy = FloorPolicy(cfg, stats={})
    return policy


def _fail_a_pursuit(policy, xy, reason="no_vertical_progress"):
    policy.pursuing = True
    policy._pursuit_goal_xy = np.asarray(xy, dtype=float)
    policy.end_pursuit(reason)


def test_a_failed_pursuit_is_remembered():
    policy = _policy(True)
    _fail_a_pursuit(policy, [1.19, -7.96])
    assert policy._portal_failed_here([1.19, -7.96])
    assert policy.stats["portal_failures_remembered"] == 1


def test_a_nearby_patch_counts_as_the_same_place():
    """One opening yields several patches from several poses; re-proposing the
    next patch of the same stairwell is the same mistake."""
    policy = _policy(True, radius=1.5)
    _fail_a_pursuit(policy, [1.19, -7.96])
    assert policy._portal_failed_here([1.9, -7.5])


def test_a_different_opening_is_still_open():
    policy = _policy(True, radius=1.5)
    _fail_a_pursuit(policy, [1.19, -7.96])
    assert not policy._portal_failed_here([6.18, 1.73])


def test_a_pursuit_that_climbed_is_not_held_against_the_place():
    """Only a pursuit that failed is evidence. `end_pursuit` is also called on
    ordinary arrival, which must leave the portal usable."""
    policy = _policy(True)
    _fail_a_pursuit(policy, [1.19, -7.96], reason="arrived")
    assert not policy._portal_failed_here([1.19, -7.96])


def test_the_memory_is_off_by_default():
    policy = _policy(False)
    _fail_a_pursuit(policy, [1.19, -7.96])
    assert not policy._portal_failed_here([1.19, -7.96])


def test_reset_forgets_between_episodes():
    policy = _policy(True)
    _fail_a_pursuit(policy, [1.19, -7.96])
    policy.reset()
    assert not policy._portal_failed_here([1.19, -7.96])
