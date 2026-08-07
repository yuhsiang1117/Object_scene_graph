"""Stair detection (osg.mapping.stairs).

Geometry leads and semantics only confirm, because the YOLOE `stairs` class was
measured firing on just 15% of multi-floor episodes -- see docs/MULTI_FLOOR.md.
"""
import numpy as np
import pytest

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D
from osg.mapping.stairs import apply_stair_mask, detect_stairs, stair_tracks

RES = 0.05


def blank(track_height=True):
    cm = Costmap2D(resolution=RES, size_m=20.0, track_height=track_height)
    cm.grid[:] = FREE
    return cm


def paint(cm, r0, r1, c0, c1, height_fn):
    """Write a surface height over a block of cells."""
    for r in range(r0, r1):
        for c in range(c0, c1):
            cm.height[r, c] = height_fn(r, c)


# ---------------------------------------------------------------- the signal


def test_flat_floor_has_no_stair_cells():
    cm = blank()
    paint(cm, 100, 200, 100, 200, lambda r, c: 0.0)
    assert detect_stairs(cm) == []


def test_wall_is_not_steppable():
    """A 2 m jump exceeds the climb limit -- that is what excludes walls."""
    cm = blank()
    paint(cm, 100, 200, 100, 150, lambda r, c: 0.0)
    paint(cm, 100, 200, 150, 200, lambda r, c: 2.0)
    assert detect_stairs(cm) == []


def test_ramp_is_detected():
    """A continuously sloping surface -- a ramp, not a staircase."""
    cm = blank()
    paint(cm, 100, 200, 100, 160, lambda r, c: 0.0)
    paint(cm, 100, 200, 160, 220, lambda r, c: (c - 160) * RES * (0.17 / 0.28))
    regions = detect_stairs(cm)
    assert regions, "ramp not detected"
    assert regions[0].rise_m > 1.0


@pytest.mark.xfail(
    reason="KNOWN DEFECT: the per-cell `dh > min_dh_m` test excludes flat tread "
           "INTERIORS, so a real staircase fragments into disconnected riser "
           "strips (measured: 9 components, largest 200 cells, 0 regions) and "
           "no threshold recovers it. Detection currently only works on ramps. "
           "See docs/MULTI_FLOOR.md; floor.stairs is default-off because of it.",
    strict=True,
)
def test_discrete_treads_are_detected():
    """A REAL staircase: flat 0.28 m treads separated by 0.17 m risers.

    This is the shape that matters and the one the current detector misses --
    the ramp above passes only because its surface never goes flat.
    """
    cm = blank()
    paint(cm, 100, 200, 100, 160, lambda r, c: 0.0)
    paint(cm, 100, 200, 160, 220,
          lambda r, c: (int(((c - 160) * RES) / 0.28) + 1) * 0.17)
    assert detect_stairs(cm), "real staircase not detected"


def test_threshold_is_rejected_by_min_rise():
    """A door threshold passes the per-cell test but goes nowhere vertically."""
    cm = blank()
    paint(cm, 100, 200, 100, 160, lambda r, c: 0.0)
    paint(cm, 100, 200, 160, 164, lambda r, c: 0.04)
    paint(cm, 100, 200, 164, 220, lambda r, c: 0.0)
    assert detect_stairs(cm) == []


def test_climb_limit_bounds_what_counts():
    cm = blank()
    paint(cm, 100, 200, 100, 160, lambda r, c: 0.0)
    paint(cm, 100, 200, 160, 220, lambda r, c: (c - 160) * RES * (0.17 / 0.28))
    assert detect_stairs(cm, climb_limit_m=0.2)  # ramp
    # A climb limit below the per-cell rise rejects the same geometry
    assert detect_stairs(cm, climb_limit_m=0.01) == []


def test_speckle_is_rejected_by_min_cells():
    cm = blank()
    paint(cm, 100, 200, 100, 200, lambda r, c: 0.0)
    cm.height[150, 150] = 0.1
    assert detect_stairs(cm, min_cells=200) == []


# ------------------------------------------------------------ the relabelling


def _staircase_map():
    cm = blank()
    paint(cm, 100, 200, 100, 160, lambda r, c: 0.0)
    paint(cm, 100, 200, 160, 220, lambda r, c: (c - 160) * RES * (0.17 / 0.28))
    return cm  # a ramp: the only shape the detector currently finds


def test_mask_writes_free_and_survives_reobservation():
    """The regression that makes the whole thing work: OCCUPIED is never
    cleared, so without the raycast exemption the next frame restores the wall
    across the staircase and the relabel is a silent no-op."""
    cm = _staircase_map()
    regions = detect_stairs(cm)
    assert apply_stair_mask(cm, regions) > 0
    rc = regions[0].cells_rc[0]
    assert cm.grid[rc[0], rc[1]] == FREE

    # Now stamp obstacles straight through the masked cells, as a later frame
    # observing the rising treads would.
    cam_rc = np.array([150, 50])
    obst_xy = np.stack([cm.grid_to_world(c) for c in regions[0].cells_rc[:40]])
    cm._raycast_batch(cam_rc, np.zeros((0, 2)), obst_xy)
    assert cm.grid[rc[0], rc[1]] == FREE, "stair cell was re-stamped OCCUPIED"


def test_unmasked_obstacles_are_still_stamped():
    """The exemption must be scoped to stairs, not disable obstacles at large."""
    cm = _staircase_map()
    apply_stair_mask(cm, detect_stairs(cm))
    far = cm.grid_to_world(np.array([300.0, 300.0]))
    cm._raycast_batch(np.array([150, 50]), np.zeros((0, 2)), far[None, :])
    assert cm.grid[300, 300] == OCCUPIED


def test_area_cap_limits_the_relabel():
    cm = _staircase_map()
    regions = detect_stairs(cm)
    assert apply_stair_mask(cm, regions, max_area_frac=0.0) == 0


def test_apply_is_a_noop_without_regions():
    cm = blank()
    assert apply_stair_mask(cm, []) == 0
    assert cm.stair_mask is None


# ------------------------------------------------------------------ semantics


class _Track:
    def __init__(self, label, center, n_obs=5, evidence=2.0):
        self.label, self.n_obs, self.evidence = label, n_obs, evidence
        self.center = np.asarray(center, float)


class _Layer:
    def __init__(self, tracks):
        self._t = tracks

    def tracks(self, include_blacklisted=False):
        return self._t

    def center_of(self, t):
        return t.center


def test_stair_tracks_filters_by_label_and_evidence():
    layer = _Layer([
        _Track("stairs", [1, 0, 1]),
        _Track("staircase", [2, 0, 2]),
        _Track("chair", [3, 0, 3]),
        _Track("stairs", [4, 0, 4], n_obs=1),          # too few observations
        _Track("stairs", [5, 0, 5], evidence=0.1),     # too little evidence
    ])
    assert len(stair_tracks(layer)) == 2


def test_semantic_flag_is_set_but_not_required():
    cm = _staircase_map()
    plain = detect_stairs(cm)
    assert plain and not plain[0].semantic

    near = [np.array([plain[0].centroid_xy[0], 0.5, plain[0].centroid_xy[1]])]
    confirmed = detect_stairs(cm, semantic_centers=near)
    assert confirmed and confirmed[0].semantic


def test_require_semantic_discards_uncorroborated_stairs():
    """Off by default precisely because it would drop most real staircases."""
    cm = _staircase_map()
    assert detect_stairs(cm, require_semantic=True) == []


# ------------------------------------------------------------------ plumbing


def test_height_layer_keeps_the_lowest_surface():
    """A table top must not hide the floor under it."""
    cm = blank()
    cm._record_heights(np.array([[0.0, 0.75, 0.0], [0.0, 0.02, 0.0]]))
    rc = cm.world_to_grid(np.array([0.0, 0.0]))
    assert cm.height[rc[0], rc[1]] == pytest.approx(0.02)


def test_layers_grow_with_the_grid():
    cm = blank()
    cm.stair_mask = np.zeros(cm.grid.shape, dtype=bool)
    cm.ensure_contains(np.array([50.0, 50.0]))
    assert cm.height.shape == cm.grid.shape
    assert cm.stair_mask.shape == cm.grid.shape


def test_gradient_requires_the_height_layer():
    with pytest.raises(ValueError):
        Costmap2D(resolution=RES).height_gradient()


def test_unseen_cells_are_invalid_not_zero():
    """NaN must not be read as a flat 0.0 surface, or unobserved space becomes
    a giant fake staircase edge."""
    cm = blank()
    paint(cm, 100, 120, 100, 120, lambda r, c: 0.0)
    dh, valid = cm.height_gradient()
    assert not valid.all()
    assert np.isnan(dh[~valid]).all()
