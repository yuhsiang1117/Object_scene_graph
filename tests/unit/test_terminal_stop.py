"""Terminal stop rule: distance to the object's nearest surface point.

HM3D scores geodesic distance to a view_point, and view points are tiled around
an object's SURFACE. The default rule uses median mask depth, which is the
distance to the middle of whatever the mask covers -- so a 2 m sofa and a chair
stop at very different distances from their near edge. These tests pin the
surface-based alternative and that it stays opt-in.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from osg.core.types import Detection
from osg.objects.linking import relink
from osg.objects.object_layer import ObjectLayer

from .conftest import draw_ellipse_mask, make_camera, make_frame

W, H = 640, 480


def _det(label="chair", cx=320, cy=240, semi=(60, 40)):
    return Detection(
        label=label, score=0.8,
        bbox_xyxy=np.array([cx - semi[0], cy - semi[1], cx + semi[0], cy + semi[1]], float),
        mask=draw_ellipse_mask(H, W, (cx, cy), semi),
    )


def _frame(intrinsics, depth=3.0):
    return make_frame(intrinsics, make_camera([0.0, 0.88, 0.0], [0.0, 0.88, 3.0]),
                      depth_value=depth)


def _layer_with_target(intrinsics, depth=3.0, cap=2000):
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0, cloud_cap=cap)
    layer.keep_cloud_labels = {"chair"}
    layer.update(_frame(intrinsics, depth), [_det()])
    return layer, layer.tracks()[0]


# ---------------------------------------------------------------- the cloud


def test_nearest_point_is_the_surface_not_the_centre(intrinsics):
    layer, track = _layer_with_target(intrinsics)
    d = layer.nearest_point_dist_xy(track, np.array([0.0, 0.0]))
    centre_d = float(np.linalg.norm(track.ellipsoid.center[[0, 2]]))
    assert d is not None and d <= centre_d + 1e-6


def test_no_cloud_for_non_target_labels(intrinsics):
    """Clouds stay bounded by accumulating them for the target only, however
    cluttered the scene."""
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0)
    layer.keep_cloud_labels = {"sofa"}
    layer.update(_frame(intrinsics), [_det("chair")])
    track = layer.tracks()[0]
    assert track.points_w is None
    assert layer.nearest_point_dist_xy(track, np.array([0.0, 0.0])) is None


def test_cloud_stays_capped(intrinsics):
    layer, _ = _layer_with_target(intrinsics, cap=200)
    for _ in range(5):
        layer.update(_frame(intrinsics), [_det()])
    assert layer.tracks()[0].points_w.shape[0] <= 200


def test_nearest_point_spans_the_linked_component(intrinsics):
    """An L-shaped sofa is two ellipsoids but one object; the near end is what
    the agent should stop at."""
    layer, track = _layer_with_target(intrinsics)
    near = copy.deepcopy(track)
    near.id = 99
    shift = np.array([0.0, 0.0, -0.5])  # half a metre closer to the camera
    near.ellipsoid.center = track.ellipsoid.center + shift
    near.points_w = track.points_w + shift
    layer._tracks[99] = near
    relink(list(layer._tracks.values()), 1.0)
    assert 99 in track.linked_ids

    alone = float(np.linalg.norm(track.points_w[:, [0, 2]], axis=1).min())
    assert layer.nearest_point_dist_xy(track, np.array([0.0, 0.0])) < alone


# ----------------------------------------------------------------- the rule


def _agent_at(intrinsics, dist_m, terminal_rule="nearest_point"):
    """A NavAgent whose committed target's surface sits `dist_m` ahead."""
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.agent.terminal_rule = terminal_rule
    cfg.agent.terminal_engage_m = 1.0
    cfg.agent.terminal_stop_m = 0.6
    cfg.agent.terminal_progress_eps = 0.1
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair")

    layer, track = _layer_with_target(intrinsics)
    track.points_w = np.array([[0.0, 0.9, float(dist_m)]])
    agent.object_layer = layer
    agent._candidate_id = track.id
    return agent


def test_stops_inside_the_stop_radius(intrinsics):
    agent = _agent_at(intrinsics, 0.5)
    assert agent._nearest_point_stop(np.array([0.0, 0.0])) == "nearest_point"


def test_keeps_going_while_still_closing(intrinsics):
    agent = _agent_at(intrinsics, 0.9)
    assert agent._nearest_point_stop(np.array([0.0, 0.0])) is None


def test_does_not_engage_beyond_the_engage_range(intrinsics):
    agent = _agent_at(intrinsics, 2.0)
    assert agent._nearest_point_stop(np.array([0.0, 0.0])) is None


def test_turning_never_counts_as_a_stall(intrinsics):
    """The bug that collapsed SR 64% -> 32%.

    Approaching an object means turning to face it, and a turn leaves the
    distance to it unchanged. Counting that as "failed to close" fired the stall
    on the first turn of nearly every approach: 18 of 50 episodes stopped that
    way and median distance to goal went from 0.04 m to 1.09 m.
    """
    agent = _agent_at(intrinsics, 0.9)
    here = np.array([0.0, 0.0])
    for _ in range(10):  # ten steps in place: turning, not stuck
        assert agent._nearest_point_stop(here) is None


def test_stops_when_closing_stalls_while_moving(intrinsics):
    """Blocked by the object itself or by furniture in front of it: the agent
    keeps moving but stops getting closer, so stop rather than push to the
    deadline. Requires a RUN of such steps -- one is noise, since an oblique
    approach barely changes the distance to the nearest surface."""
    agent = _agent_at(intrinsics, 0.9)
    # Move each step (sideways, so the distance to the target barely changes).
    outs = [agent._nearest_point_stop(np.array([0.1 * i, 0.0])) for i in range(6)]
    assert outs[0] is None, "the first moving step cannot already be a stall"
    assert "nearest_point_stalled" in outs
    assert outs.index("nearest_point_stalled") >= 3, "stalled too eagerly"


def test_rule_is_opt_in():
    """Every existing approach test relies on the median-depth rule, so the
    surface rule must stay off until an experiment selects it."""
    from osg.core.config import AgentConfig

    assert AgentConfig().terminal_rule == "depth"


def test_track_id_zero_is_handled(intrinsics):
    """Track ids start at 0, so a truthiness check on _candidate_id silently
    disables the rule for the first object mapped in an episode."""
    agent = _agent_at(intrinsics, 0.5)
    assert agent._candidate_id == 0
    assert agent._nearest_point_stop(np.array([0.0, 0.0])) is not None


# ------------------------------------------------------- outlier robustness


def test_one_stray_point_hijacks_the_minimum(intrinsics):
    """Why the surface rule measured net -8 on dev50.

    The cloud accumulates over hundreds of frames of mask noise and pose drift,
    so its minimum is set by its worst point. Here the object sits 2 m away and
    a single stray point sits at 0.3 m; the min reports 0.3 and the agent stops
    1.7 m short. Measured: 14 of 31 approaches landed within 0.1 m and the rest
    scattered out to 2.3 m -- bimodal, which a too-large threshold cannot
    produce, since that would shift the whole distribution rather than split it.
    """
    layer, track = _layer_with_target(intrinsics)
    real = np.full((200, 3), 2.0)
    real[:, 1] = 0.9
    track.points_w = np.vstack([real, [[0.3, 0.9, 0.0]]])
    here = np.array([0.0, 0.0])

    assert layer.nearest_point_dist_xy(track, here) < 0.5, "the min follows the stray"
    assert layer.nearest_point_dist_xy(track, here, percentile=5.0) > 1.5


def test_percentile_still_reports_the_near_surface(intrinsics):
    """The guard must not turn into 'distance to the object's middle' -- that is
    the median-depth rule the surface rule exists to improve on."""
    layer, track = _layer_with_target(intrinsics)
    # A 1 m deep object spanning 2.0 to 3.0 m ahead, no outliers.
    z = np.linspace(2.0, 3.0, 400)
    track.points_w = np.stack([np.zeros_like(z), np.full_like(z, 0.9), z], axis=1)
    d = layer.nearest_point_dist_xy(track, np.array([0.0, 0.0]), percentile=5.0)
    assert 2.0 <= d < 2.1, "should sit at the near face, not the centroid"


def test_percentile_defaults_to_the_exact_minimum(intrinsics):
    """Default must reproduce today's behaviour so the flag is measurable."""
    layer, track = _layer_with_target(intrinsics)
    here = np.array([0.0, 0.0])
    assert layer.nearest_point_dist_xy(track, here) == \
        layer.nearest_point_dist_xy(track, here, percentile=0.0)

    from osg.core.config import AgentConfig
    assert AgentConfig().terminal_percentile == 5.0, "the opt-in rule should default to the version that works"


def test_nearest_point_xy_returns_an_observed_point(intrinsics):
    """What ASCENT navigates to (object_point_cloud_map.py:127-130, :225): the
    observed surface point closest to the agent, not a fitted centre.

    The distinction is the whole diagnosis of the same-floor gap. A centre is
    inferred and can land somewhere never observed -- measured at a median 3.29 m
    from the nearest real instance across every far-commit failure, against
    0.30 m for successes. A cloud point is a depth return, so it is by
    construction somewhere the agent has actually seen.
    """
    layer, track = _layer_with_target(intrinsics)
    track.points_w = np.array([[0.0, 0.9, 3.0], [0.0, 0.9, 2.0], [0.0, 0.9, 5.0]])
    here = np.array([0.0, 0.0])

    p = layer.nearest_point_xy(track, here)
    assert p is not None
    # PLANE is (x, z), so the nearest of the three is the one at z = 2.
    assert float(np.linalg.norm(p - here)) == pytest.approx(2.0)
    assert layer.nearest_point_dist_xy(track, here) == pytest.approx(2.0)


def test_nearest_point_xy_without_a_cloud(intrinsics):
    layer = ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0)
    layer.keep_cloud_labels = set()
    layer.update(_frame(intrinsics), [_det("chair")])
    assert layer.nearest_point_xy(layer.tracks()[0], np.array([0.0, 0.0])) is None


def test_cloud_target_is_recorded_on_the_navmesh_path(intrinsics):
    """The navmesh branch calls _start_approach WITHOUT an agent position
    (nav_agent.py:1509, it has no frame in scope), and that branch is what every
    configuration in docs/AB_RESULTS runs.

    The first version of this diagnostic required the argument, so it recorded
    None on every episode of a 100-episode run -- four hours for an empty
    column. It now falls back to the position stored each step.
    """
    agent = _agent_at(intrinsics, 0.5)
    agent._agent_xy = np.array([0.0, 0.0])
    track = agent.object_layer.get(agent._candidate_id)
    track.points_w = np.array([[0.0, 0.9, 2.0], [0.0, 0.9, 4.0]])

    agent._start_approach(np.array([0.0, 3.0]))  # no agent_xy, as navmesh does
    assert agent._target_cloud_xy is not None
    assert float(np.linalg.norm(agent._target_cloud_xy - agent._agent_xy)) == pytest.approx(2.0)
