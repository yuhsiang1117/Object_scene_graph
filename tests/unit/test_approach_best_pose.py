"""Two fixes for the last half metre, both sensor-only.

`_return_to_best`: an approach stops where its terminal rule fires, which is
routinely farther from the object than somewhere it already stood. Measured
over 81 trials the agent gives up a median 0.18 m between the two.

`nearest_clear_xy`: the closing walk's advantage is a goal that is standable by
construction. The navmesh answers that from ground truth; the costmap can
answer it from the agent's own depth.
"""
from __future__ import annotations

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, Costmap2D
from osg.verification.viewpoint import nearest_clear_xy


# ------------------------------------------------------- nearest_clear_xy

def _open_map(res=0.05, size_m=12.0):
    cm = Costmap2D(resolution=res, size_m=size_m)
    cm.grid[:, :] = FREE
    return cm


def test_nearest_clear_point_is_the_nearest_one():
    cm = _open_map()
    obj = np.array([0.0, 0.0])
    got = nearest_clear_xy(cm, obj, min_clearance_m=0.15)
    assert got is not None
    # Nothing is occupied, so the nearest qualifying cell is essentially the
    # object itself -- the caller's own "no gain" test rejects that.
    assert float(np.linalg.norm(got - obj)) <= 0.15


def test_it_refuses_cells_the_agent_does_not_fit_in():
    """A slot one cell wide is FREE and is not standable."""
    cm = _open_map()
    obj = np.array([0.0, 0.0])
    # Wall off everything except a thin slot at x in [0.30, 0.35].
    for r in range(cm.grid.shape[0]):
        for c in range(cm.grid.shape[1]):
            x, z = cm.grid_to_world(np.array([r, c]))
            if 0.20 <= x <= 0.50 and abs(z) <= 1.0 and not (0.30 <= x <= 0.35):
                cm.grid[r, c] = OCCUPIED
    got = nearest_clear_xy(cm, obj, min_clearance_m=0.25, max_radius_m=0.6)
    if got is not None:
        assert not (0.30 <= float(got[0]) <= 0.35), "the slot is too narrow to stand in"


def test_it_returns_none_when_nothing_qualifies():
    cm = _open_map()
    cm.grid[:, :] = OCCUPIED
    got = nearest_clear_xy(cm, np.array([0.0, 0.0]), min_clearance_m=0.2, max_radius_m=1.0)
    assert got is None


def test_unknown_is_not_free():
    """An unmapped cell is not somewhere we have seen floor."""
    cm = Costmap2D(resolution=0.05, size_m=12.0)   # all UNKNOWN
    got = nearest_clear_xy(cm, np.array([0.0, 0.0]), min_clearance_m=0.1, max_radius_m=1.0)
    assert got is None


# ------------------------------------------------------- _return_to_best

class _Nav:
    """The three attributes `_return_to_best` reads, and a stats dict."""

    def __init__(self, cfg, obj_xy, goal_xy=None):
        self.cfg = cfg
        self._target_obj_xy = np.asarray(obj_xy, dtype=float)
        self._goal_xy = goal_xy
        self._current_path = None
        self._goto_deadline = 0
        self.step_count = 0
        self.stats = {}


def _approach(margin, obj_xy=(0.0, 0.0), follow_returns="move_forward"):
    from osg.agent.approach import ApproachPolicy
    from tests.unit.test_nav_agent import make_cfg

    cfg = make_cfg(approach_stop_at_best_m=margin)
    nav = _Nav(cfg, obj_xy)
    ap = ApproachPolicy.__new__(ApproachPolicy)
    ap.nav = nav
    ap.reset()
    ap.follow_to = lambda frame, goal: follow_returns
    return ap, nav


def test_off_by_default_never_walks_back():
    ap, nav = _approach(0.0)
    ap._note_best(np.array([0.5, 0.0]))
    assert ap._return_to_best(None, np.array([3.0, 0.0])) is None
    assert nav.stats.get("approach_best_pose_returns", 0) == 0


def test_a_worse_final_pose_walks_back_to_the_best_one():
    ap, nav = _approach(0.15)
    ap._note_best(np.array([0.8, 0.0]))      # 0.8 m from the object
    ap._note_best(np.array([1.6, 0.0]))      # drifted out
    action = ap._return_to_best(None, np.array([1.6, 0.0]))
    assert action == "move_forward"
    assert nav.stats["approach_best_pose_returns"] == 1
    assert np.allclose(nav._goal_xy, [0.8, 0.0])


def test_a_pose_within_the_margin_just_stops():
    ap, nav = _approach(0.15)
    ap._note_best(np.array([0.80, 0.0]))
    assert ap._return_to_best(None, np.array([0.90, 0.0])) is None
    assert nav.stats.get("approach_best_pose_returns", 0) == 0
    assert ap.returned


def test_it_happens_at_most_once_per_approach():
    ap, nav = _approach(0.15)
    ap._note_best(np.array([0.8, 0.0]))
    assert ap._return_to_best(None, np.array([1.6, 0.0])) == "move_forward"
    # the walk is consumed
    assert ap._return_to_best(None, np.array([0.9, 0.0])) is None
    assert ap.returned
    # and never again
    assert ap._return_to_best(None, np.array([5.0, 0.0])) is None
    assert nav.stats["approach_best_pose_returns"] == 1


def test_the_gain_is_recorded_in_centimetres():
    ap, nav = _approach(0.15)
    ap._note_best(np.array([0.8, 0.0]))
    ap._return_to_best(None, np.array([1.6, 0.0]))     # from 1.6 m
    ap._return_to_best(None, np.array([0.85, 0.0]))    # ended at 0.85 m
    assert nav.stats["approach_best_pose_gain_cm"] == 75


def test_no_committed_track_is_a_no_op():
    ap, nav = _approach(0.15)
    nav._target_obj_xy = None
    ap._note_best(np.array([1.0, 0.0]))
    assert ap._return_to_best(None, np.array([2.0, 0.0])) is None


def test_the_best_pose_does_not_survive_into_the_next_approach():
    """It is per-approach state. Carried over, the next approach compares
    against a distance measured to a DIFFERENT object and walks back to a pose
    beside the previous one."""
    ap, nav = _approach(0.15, obj_xy=(0.0, 0.0))
    ap._note_best(np.array([0.4, 0.0]))          # close to target A
    assert ap._return_to_best(None, np.array([2.0, 0.0])) == "move_forward"
    # a new approach, to a target eight metres away
    nav._target_obj_xy = np.array([8.0, 0.0])
    ap.start = None  # not called here; emulate what start() must clear
    ap.best_xy = None
    ap.returning = False
    ap.returned = False
    ap._note_best(np.array([9.0, 0.0]))
    assert ap._return_to_best(None, np.array([9.05, 0.0])) is None
    assert np.allclose(ap.best_xy, [9.0, 0.0]), "the new approach owns its own best pose"


def test_a_retarget_re_measures_the_best_pose():
    """`retarget` moves the track centre mid-approach; a cached distance would
    then be measured against a centre that no longer exists."""
    ap, nav = _approach(0.15, obj_xy=(0.0, 0.0))
    ap._note_best(np.array([1.0, 0.0]))          # 1.0 m from the old centre
    nav._target_obj_xy = np.array([2.0, 0.0])    # centre refined: now 1.0 m the other side
    assert abs(ap._best_distance() - 1.0) < 1e-9
    ap._note_best(np.array([1.8, 0.0]))          # 0.2 m from the NEW centre: better
    assert np.allclose(ap.best_xy, [1.8, 0.0])
