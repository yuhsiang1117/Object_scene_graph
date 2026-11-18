"""One physical surface should be one anchor.

`container_merge_m` is a flat radius and cannot serve a 2 m bed and a 0.5 m
nightstand at once. Measured against HM3D's own semantic annotations, the flat
1.0 m leaves 00829 with 16 `bed` container nodes in a house that has ONE bed.
The extent-aware test scales itself to the object: a shadow is a 2D Gaussian,
so the Mahalanobis distance of one centre in the other's metric is "how many of
its own radii away the neighbour sits".
"""
from __future__ import annotations

import numpy as np

from osg.graph.containers import shadows_overlap


def shadow(cx, cz, rx, rz):
    """A ground shadow at (cx, cz) with semi-axes (rx, rz), as (mu, inv_cov)."""
    cov = np.diag([rx ** 2, rz ** 2])
    return (np.array([cx, cz], dtype=float), np.linalg.inv(cov))


def test_off_by_default():
    a = [shadow(0.0, 0.0, 0.5, 0.5)]
    b = [shadow(0.6, 0.0, 0.5, 0.5)]
    assert shadows_overlap(a, b, 0.0) is False


def test_two_halves_of_a_bed_merge():
    """Each half is about a metre long; their centres are a metre apart."""
    a = [shadow(0.0, 0.0, 1.0, 0.5)]
    b = [shadow(1.2, 0.0, 1.0, 0.5)]
    assert shadows_overlap(a, b, 2.0) is True


def test_two_nightstands_do_not():
    """Same gap, much smaller furniture: six radii away, not one."""
    a = [shadow(0.0, 0.0, 0.2, 0.2)]
    b = [shadow(1.2, 0.0, 0.2, 0.2)]
    assert shadows_overlap(a, b, 2.0) is False


def test_the_rule_scales_with_the_object_not_the_gap():
    """The same 1.2 m gap merges beds and separates nightstands. That is the
    whole point of the test, so it is asserted directly."""
    gap = 1.2
    big = shadows_overlap([shadow(0, 0, 1.0, 0.5)], [shadow(gap, 0, 1.0, 0.5)], 2.0)
    small = shadows_overlap([shadow(0, 0, 0.2, 0.2)], [shadow(gap, 0, 0.2, 0.2)], 2.0)
    assert big and not small


def test_the_smaller_shadow_reaching_in_is_enough():
    """A fragment of a bed is small and the bed is not; requiring BOTH
    directions would never merge a fragment into its parent."""
    parent = [shadow(0.0, 0.0, 1.2, 0.6)]
    fragment = [shadow(1.0, 0.0, 0.15, 0.15)]
    assert shadows_overlap(parent, fragment, 1.5) is True


def test_any_member_shadow_can_match():
    """A merged container carries every member's shadow, which is what makes
    the merge transitive across a bed's several fragments."""
    merged = [shadow(0.0, 0.0, 0.3, 0.3), shadow(1.0, 0.0, 0.3, 0.3)]
    third = [shadow(1.8, 0.0, 0.3, 0.3)]
    assert shadows_overlap(merged, third, 3.0) is True
    assert shadows_overlap([merged[0]], third, 3.0) is False


def test_far_apart_never_merges():
    a = [shadow(0.0, 0.0, 1.0, 1.0)]
    b = [shadow(8.0, 0.0, 1.0, 1.0)]
    assert shadows_overlap(a, b, 3.0) is False


def test_empty_shadow_lists_are_safe():
    assert shadows_overlap([], [shadow(0, 0, 1, 1)], 2.0) is False
    assert shadows_overlap([shadow(0, 0, 1, 1)], [], 2.0) is False
