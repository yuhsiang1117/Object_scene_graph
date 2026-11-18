"""The stair-mouth geometry behind `scripts/stair_anchors.py`.

The tool decides which anchors count as "by the stairs", and that ranking is
what `restage_near_stairs.py` turns into an `--exclude-anchor` list -- so a
wrong mouth silently re-authors the dataset toward the wrong end of a storey.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "stair_anchors", Path(__file__).resolve().parents[2] / "scripts" / "stair_anchors.py"
)


@pytest.fixture(scope="module")
def mod():
    module = importlib.util.module_from_spec(SPEC)
    try:
        SPEC.loader.exec_module(module)
    except Exception as exc:  # habitat / collector not importable on CPU boxes
        pytest.skip(f"stair_anchors not importable here: {exc}")
    return module


def test_cluster_xy_follows_a_switchback(mod):
    """Single link, not centroid: a switchback is ONE flight, not two.

    Two parallel runs joined at the turn are within `link_m` step to step but
    their centroids are metres apart, so a centroid method splits them and
    reports two flights where the scene has one.
    """
    up = np.array([[0.0, y * 0.2, 0.0] for y in range(10)])
    up[:, 0] = np.linspace(0.0, 1.8, 10)
    back = up.copy()
    back[:, 0] = np.linspace(1.8, 0.0, 10)
    back[:, 2] = 0.8
    pts = np.vstack([up, back])
    assert len(mod.cluster_xy(pts, link_m=1.0)) == 1
    # Two runs 6 m apart are two flights.
    far = back.copy()
    far[:, 2] += 6.0
    assert len(mod.cluster_xy(np.vstack([up, far]), link_m=1.0)) == 2


def test_cluster_xy_empty(mod):
    assert mod.cluster_xy(np.zeros((0, 3)), link_m=1.0) == []


def test_flight_mouth_is_the_landing_not_the_flight(mod):
    """The mouth must be a point OF THE STOREY, not of the staircase.

    The agent steps onto the landing; a point taken from the flight itself sits
    mid-tread, which is not a place an anchor can be measured against.
    """
    lower = np.array([[x, 0.0, 0.0] for x in np.linspace(-4.0, 0.0, 20)])
    upper = np.array([[x, 3.0, 0.0] for x in np.linspace(2.0, 6.0, 20)])
    storeys = [
        {"index": 0, "height": 0.0, "points": lower},
        {"index": 1, "height": 3.0, "points": upper},
    ]
    flight = np.array([[x, y, 0.0] for x, y in zip(np.linspace(0.4, 1.6, 8),
                                                   np.linspace(0.5, 2.5, 8))])
    out = mod.flight_mouths(storeys, [flight], touch_m=1.5)
    assert len(out) == 1
    mouths = out[0]["mouths"]
    assert set(mouths) == {0, 1}
    # Nearest point of each storey to the flight: the ends facing the stairs.
    assert mouths[0]["xy"][0] == pytest.approx(0.0, abs=1e-6)
    assert mouths[1]["xy"][0] == pytest.approx(2.0, abs=1e-6)


def test_storey_too_far_from_the_flight_gets_no_mouth(mod):
    """A storey the flight does not reach must not be given one."""
    near = np.array([[0.0, 0.0, 0.0]])
    far = np.array([[40.0, 3.0, 0.0]])
    storeys = [{"index": 0, "height": 0.0, "points": near},
               {"index": 1, "height": 3.0, "points": far}]
    flight = np.array([[0.5, 1.5, 0.0]])
    out = mod.flight_mouths(storeys, [flight], touch_m=1.5)
    assert set(out[0]["mouths"]) == {0}


def test_anchor_matches_its_own_storey_by_height(mod):
    """By HEIGHT, never by the collector's floor_index.

    00878's floor detector merges its basement and ground storey into one
    index, so an index match would measure a ground-floor anchor against a
    mouth 2.8 m below it.
    """
    mouths = [(0.0, [0.0, 0.0]), (2.8, [10.0, 0.0])]
    # An anchor standing 0.45 m above the ground storey belongs to it.
    got = mod.nearest_mouth_m([3.0, 0.45, 4.0], mouths)
    assert got is not None and got[1] == 0.0
    assert got[0] == pytest.approx(5.0, abs=1e-6)
    # One on the top storey matches the top mouth even though it is further in xy.
    got = mod.nearest_mouth_m([0.0, 3.25, 0.0], mouths)
    assert got is not None and got[1] == 2.8


def test_anchor_on_no_known_storey_returns_none(mod):
    assert mod.nearest_mouth_m([0.0, 40.0, 0.0], [(0.0, [0.0, 0.0])]) is None
