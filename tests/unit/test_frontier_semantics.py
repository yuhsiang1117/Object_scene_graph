"""Frontier descriptions sourced from the frame that revealed the frontier.

ASCENT binds each frontier to the step it appeared at and describes it with
RAM++/Places365 output from THAT step's RGB (obstacle_map.py:421-431 ->
map_controller.py:800-830 -> llm_planner.py:418-419). OSG previously described a
frontier by querying the accumulated scene graph around its centroid, which
describes the mapped surroundings of the point rather than the view through the
opening.

The failure mode to guard against is silent inertness: if nothing ever binds,
`describe_area` quietly falls back to the graph and the A/B reports "no
change" -- indistinguishable from the mechanism being useless. Several tests
here exist only to make that impossible.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.exploration.ascent_ranker import describe_area
from osg.exploration.frontier_semantics import FrontierSemantics


class _F:
    """The only field FrontierSemantics reads off a Frontier."""

    def __init__(self, xy) -> None:
        self.centroid_xy = np.asarray(xy, dtype=float)


def _sem(**kw) -> FrontierSemantics:
    return FrontierSemantics(**kw)


# ------------------------------------------------------------------ binding


def test_a_frontier_is_described_by_the_frame_that_revealed_it():
    s = _sem()
    s.observe(10, "bedroom", ["bed", "nightstand"])
    s.bind([_F([1.0, 1.0])], 10)
    assert s.describe(_F([1.0, 1.0])) == ("bedroom", ["bed", "nightstand"])


def test_the_original_view_survives_later_ones():
    """The whole point: a frontier keeps the description of the frame that
    first revealed it, not of wherever the agent stands later."""
    s = _sem()
    s.observe(10, "bedroom", ["bed"])
    s.bind([_F([1.0, 1.0])], 10)
    s.observe(40, "kitchen", ["oven"])
    s.bind([_F([1.05, 1.0])], 40)  # same opening, re-extracted and drifted
    assert s.describe(_F([1.05, 1.0])) == ("bedroom", ["bed"])


def test_a_drifted_centroid_is_the_same_opening():
    s = _sem(match_radius_m=1.0)
    s.observe(10, "bedroom", ["bed"])
    s.bind([_F([1.0, 1.0])], 10)
    assert s.describe(_F([1.6, 1.0])) is not None


def test_a_distant_frontier_is_a_different_opening():
    s = _sem(match_radius_m=1.0)
    s.observe(10, "bedroom", ["bed"])
    s.bind([_F([1.0, 1.0])], 10)
    assert s.describe(_F([9.0, 9.0])) is None


def test_keyframes_and_extraction_landing_on_different_steps_still_bind():
    """The bug that would have made this inert.

    ASCENT stores the RGB in the same function that detects new frontiers, so
    the two steps always coincide. OSG runs keyframes on a movement threshold
    and frontier extraction on a step interval, so they routinely do not.
    Requiring an exact step match would leave nearly every frontier unbound.
    """
    s = _sem()
    s.observe(10, "bedroom", ["bed"])
    assert s.bind([_F([1.0, 1.0])], 13) == 1, "no keyframe on the exact step"
    assert s.describe(_F([1.0, 1.0])) == ("bedroom", ["bed"])


def test_nothing_binds_before_anything_is_observed():
    """Better to fall back to the graph than to assert an empty room."""
    s = _sem()
    assert s.bind([_F([1.0, 1.0])], 5) == 0
    assert s.describe(_F([1.0, 1.0])) is None


def test_a_frontier_binds_only_once():
    s = _sem()
    s.observe(10, "bedroom", ["bed"])
    assert s.bind([_F([1.0, 1.0])], 10) == 1
    assert s.bind([_F([1.0, 1.0])], 10) == 0


# -------------------------------------------------------------- description


def test_the_rendered_string_matches_ascents_format():
    """`llm_planner.py:445`."""
    s = _sem()
    s.observe(10, "living_room", ["sofa", "tv"])
    f = _F([1.0, 1.0])
    s.bind([f], 10)
    assert (describe_area(f, None, 3.0, semantics=s)
            == "a living room containing objects: sofa, tv")


def test_an_unbound_frontier_falls_back_to_the_graph():
    """Enabling the frame source may replace a description, never delete one."""
    s = _sem()
    f = _F([1.0, 1.0])
    with_sem = describe_area(f, None, 3.0, semantics=s)
    without = describe_area(f, None, 3.0)
    assert with_sem == without


def test_objects_are_capped_and_deduplicated():
    s = _sem(max_objects=2)
    s.observe(10, "kitchen", ["oven", "oven", "sink", "fridge"])
    f = _F([0.0, 0.0])
    s.bind([f], 10)
    assert s.describe(f)[1] == ["oven", "sink"]


def test_a_view_of_nothing_is_reported_as_such():
    s = _sem()
    s.observe(10, None, [])
    f = _F([0.0, 0.0])
    s.bind([f], 10)
    assert (describe_area(f, None, 3.0, semantics=s)
            == "a unknown room containing objects: no visible objects")


# ------------------------------------------------------------------ wiring


def test_the_agent_builds_the_store_only_when_asked():
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from osg.core.config import ExplorationConfig

    from .test_nav_agent import _StubScorer, make_cfg

    # The default that matters is the real dataclass's, not the test stub's.
    assert ExplorationConfig().frontier_desc == "graph", "must default to the old source"

    off = make_cfg()
    off.exploration.frontier_desc = "graph"
    agent = NavAgent(off, StubDetector(), AsyncScorer(_StubScorer()), None, "chair")
    assert agent.frontier_semantics is None

    on = make_cfg()
    on.exploration.frontier_desc = "frame"
    agent = NavAgent(on, StubDetector(), AsyncScorer(_StubScorer()), None, "chair")
    assert agent.frontier_semantics is not None


def test_the_store_is_fed_from_the_same_detector_pass_as_the_object_layer():
    """`observe` must be reached from a real keyframe, with that frame's own
    detections -- not a second detector run, and not the scene graph."""
    import inspect

    from osg.agent.nav_agent import NavAgent

    src = inspect.getsource(NavAgent._on_keyframe)
    assert "frontier_semantics.observe" in src, (
        "the store is not fed from the keyframe handler, so it can only ever "
        "describe frontiers with stale or missing semantics"
    )
    assert "d.label for d in dets" in src, (
        "object labels must come from this frame's detections"
    )


# ------------------------------------------------------------ channel split


def test_frame_objects_keeps_the_graph_room():
    """The arm that isolates the channel RAM++ would replace.

    S27 moved both halves at once and regressed; a probe then found the graph
    has no room label in 66% of cases, so most of what "frame" did was overwrite
    an explicit "unknown room" with a guess. This mode leaves the room alone.
    """
    s = _sem()
    s.observe(10, "garage", ["sofa", "lamp"])
    f = _F([1.0, 1.0])
    s.bind([f], 10)
    d = describe_area(f, None, 3.0, semantics=s, mode="frame_objects")
    assert "garage" not in d, "the frame's room label leaked into frame_objects"
    assert d == "a unknown room containing objects: lamp, sofa"


def test_frame_mode_still_takes_both():
    s = _sem()
    s.observe(10, "garage", ["sofa", "lamp"])
    f = _F([1.0, 1.0])
    s.bind([f], 10)
    assert (describe_area(f, None, 3.0, semantics=s, mode="frame")
            == "a garage containing objects: lamp, sofa")


def test_the_agent_builds_the_store_for_frame_objects_too():
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.exploration.frontier_desc = "frame_objects"
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "chair")
    assert agent.frontier_semantics is not None
    assert agent._frontier_desc == "frame_objects"
