"""The benchmark both systems are scored on, pinned.

A comparison is only worth reading if the two columns were produced under the
same rule. These tests hold that property in place: the trial list, the target
geometry and the success rule are one module, and the numbers already measured
for DualMap can be recomputed from it.

The geometry tests need the released dataset and are skipped without it; the
shape tests do not and always run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from osg.eval import dualmap_release as release


REFERENCE = Path("outputs/dualmap_official_bench/seed12")
released = pytest.mark.skipif(
    not release.RELEASE_ROOT.is_dir(), reason="released DualMap dataset is not mounted"
)
measured = pytest.mark.skipif(
    not (REFERENCE / "trials").is_dir(), reason="no measured DualMap run to compare against"
)


def test_success_rule_measures_to_the_box_not_the_centroid():
    """A point target and a 2 m box at the same centre are not the same target."""
    point = release.as_targets([np.array([0.0, 0.0, 0.0])])
    box = release.as_targets([np.array([0.0, 0.0, 0.0])], [np.array([2.0, 0.5, 2.0])])
    probe = [1.5, 0.0, 0.0]
    assert release.distance_to_target(probe, point)[0] == pytest.approx(1.5)
    assert release.distance_to_target(probe, box)[0] == pytest.approx(0.5)
    # Which is the whole point: the same stop fails against the centroid and
    # succeeds against the object.
    assert release.distance_to_target(probe, point)[0] > release.SUCCESS_DISTANCE_M
    assert release.distance_to_target(probe, box)[0] <= release.SUCCESS_DISTANCE_M


def test_success_rule_is_horizontal():
    """Standing under a picture is standing at it; height is not a miss."""
    target = release.as_targets([np.array([0.0, 2.0, 0.0])])
    assert release.distance_to_target([0.0, 0.0, 0.0], target)[0] == pytest.approx(0.0)
    assert release.distance_to_target([0.0, 0.0, 0.0], target)[1] == pytest.approx(2.0)


def test_nearest_instance_answers_the_query():
    """A class query is answered by any instance of the class."""
    targets = release.as_targets(
        [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
    )
    assert release.distance_to_target([9.5, 0.0, 0.0], targets)[0] == pytest.approx(0.5)


@released
def test_protocol_is_the_published_trial_population():
    trials = release.protocol()
    assert len(trials) == 186
    counts = {
        condition: sum(1 for t in trials if t["condition"] == condition)
        for condition in release.CONDITIONS
    }
    assert counts == {"static": 79, "in_anchor": 54, "cross_anchor": 53}
    assert len({t["trial_id"] for t in trials}) == len(trials)
    # The one trial the appendix does not report.
    assert not any(
        t["scene"] == "00880-Nfvxx8J5NCo"
        and t["condition"] == "cross_anchor"
        and t["layout"] == "0128-2.json"
        and t["query"] == "cracker box"
        for t in trials
    )


@released
@measured
def test_every_measured_trial_recomputes_from_this_module():
    """The DualMap run on disk is reproducible from the shared definition.

    Both its target geometry and its per-attempt distances -- so a change here
    that would silently rescore one system shows up as a failure rather than as
    a moved number in a comparison table.
    """
    checked = 0
    for trial in release.protocol():
        path = REFERENCE / "trials" / trial["trial_id"] / "result.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        targets = release.trial_targets(trial)
        assert np.allclose(
            [c for c, _ in targets], np.asarray(record["target_positions"])
        ), trial["trial_id"]
        assert np.allclose(
            [h for _, h in targets], np.asarray(record["target_half_extents"])
        ), trial["trial_id"]
        for attempt in record["attempts"]:
            horizontal, spatial = release.distance_to_target(attempt["position"], targets)
            assert horizontal == pytest.approx(attempt["distance_horizontal_m"])
            assert spatial == pytest.approx(attempt["distance_3d_m"])
            assert bool(attempt["success"]) == (horizontal <= release.SUCCESS_DISTANCE_M)
        checked += 1
    assert checked >= 180, f"only {checked} measured trials were available to check"


@released
def test_static_queries_all_resolve_to_a_target():
    """Every static query names something the release can locate."""
    for scene, queries in release.STATIC_QUERIES.items():
        for query in queries:
            targets = release.static_target_positions(scene, query)
            assert targets, f"{scene}/{query}"
            for centre, half in targets:
                assert np.isfinite(centre).all() and np.isfinite(half).all()
                assert (half >= 0).all()


def test_position_correct_is_box_relative_horizontal_and_rejects_none():
    """The stricter predicate: where the system aimed, not where it stopped."""
    # One instance, a 0.4 m half-extent in x, centred at the origin.
    targets = [(np.array([0.0, 1.0, 0.0]), np.array([0.4, 0.2, 0.1]))]

    # Distance is measured to the box, not the centroid: 1.3 m along x is 0.9 m
    # of gap, which is inside the 1 m tolerance even though the centroid is not.
    assert release.horizontal_distance_xz((1.3, 0.0), targets) == pytest.approx(0.9)
    assert release.position_correct((1.3, 0.0), targets)
    assert not release.position_correct((1.5, 0.0), targets)

    # Height is ignored entirely -- neither system records a trustworthy goal height.
    assert release.horizontal_distance_xz((0.0, 0.0), targets) == pytest.approx(0.0)

    # The nearest of several instances wins, as with distance_to_target.
    two = targets + [(np.array([10.0, 1.0, 0.0]), np.zeros(3))]
    assert release.position_correct((10.5, 0.0), two)

    # No goal is not a correct goal -- this is the case raw SR over-credits.
    assert not release.position_correct(None, targets)

    # The tolerance is the benchmark's own by default.
    assert release.position_correct((1.4, 0.0), targets, tolerance=1.5)


@released
def test_position_correct_reproduces_a_measured_dualmap_local_path():
    """A hand-checked DualMap trial, recomputed through the shared predicate.

    00829 cross_anchor 0128-1 plate: the object sits at [-0.543, 0.840, 1.490]
    and DualMap's last local-path waypoint is [-0.90, -2.06] in its own z-up
    frame, i.e. (x, z) = (-0.90, 2.06) in Habitat -- 0.67 m away, so the goal
    was correct even though the agent stopped 0.745 m out.
    """
    trial_id = "00829-QaLdnwvtxbs__cross_anchor__0128-1__plate"
    trial = next(t for t in release.protocol() if t["trial_id"] == trial_id)
    targets = release.trial_targets(trial)
    assert np.allclose([c for c, _ in targets], [[-0.5433655977249146, 0.840084969997406, 1.4897381067276]])

    waypoint = (-0.90, -2.06)  # DualMap's frame
    goal_xz = (waypoint[0], -waypoint[1])  # -> Habitat (x, z)
    assert release.horizontal_distance_xz(goal_xz, targets) == pytest.approx(0.67, abs=0.02)
    assert release.position_correct(goal_xz, targets)
