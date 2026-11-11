"""The labeller's depth self-check (osg/eval/gt_semantic.check_depth)."""
import math

import numpy as np

from osg.eval.gt_semantic import check_depth

RANGE = (0.5, 5.0)


def test_matching_depth_passes():
    rec = np.full((60, 80), 2.0, np.float32)
    ok, frac = check_depth(rec.copy(), rec, depth_range=RANGE)
    assert ok and frac == 1.0


def test_wrong_pose_fails():
    rec = np.full((60, 80), 2.0, np.float32)
    ok, frac = check_depth(rec + 0.5, rec, depth_range=RANGE)
    assert not ok and frac == 0.0


def test_fully_clamped_view_is_uncheckable():
    rec = np.full((60, 80), 0.5, np.float32)       # a wall inside depth_min_m
    ok, frac = check_depth(np.full_like(rec, 0.2), rec, depth_range=RANGE)
    assert ok and math.isnan(frac)


def test_a_few_surviving_pixels_are_no_evidence():
    # Pass 2's camera-against-a-wall keyframes: everything clamped but one edge
    # pixel, whose rendered depth differs. That is not a pose error.
    rec = np.full((60, 80), 0.5, np.float32)
    rec[10, 10] = 2.566
    rendered = np.full_like(rec, 0.15)
    ok, frac = check_depth(rendered, rec, depth_range=RANGE)
    assert ok and math.isnan(frac)
    # ...while a real view's worth of mismatching pixels still fails.
    rec[:20, :] = 2.0
    ok, frac = check_depth(rendered, rec, depth_range=RANGE)
    assert not ok
