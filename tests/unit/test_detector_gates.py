"""Per-class admission gates on the detector.

One global confidence threshold prices every class the same. A false-positive
census over 900 random navigable poses says they are not the same: at imgsz
1280, moving the gate 0.30 -> 0.20 costs the tomato soup can and the banana
nothing at all in false positives and buys them 8-9 points of recall, while it
costs the cracker box EIGHTEEN false positives for six points. Applied globally
the move is +8 true positives against +32 false ones. Applied per class it is
worth making for exactly the labels that are weak to begin with.

These tests exercise the gating arithmetic without ultralytics, which is a
half-gigabyte import and a GPU away.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.perception.detector import YoloeDetector


class _Gates(YoloeDetector):
    """Just the thresholding, with the model constructor skipped."""

    def __init__(self, conf, class_conf):
        self.conf = conf
        self.class_conf = {self._normalize(k): float(v) for k, v in class_conf.items()}


def test_a_class_without_an_override_keeps_the_global_gate():
    d = _Gates(0.30, {"banana": 0.20})
    assert d._admits("cracker box", 0.25) is False
    assert d._admits("cracker box", 0.31) is True


def test_an_overridden_class_uses_its_own_gate():
    d = _Gates(0.30, {"banana": 0.20})
    assert d._admits("banana", 0.25) is True
    assert d._admits("banana", 0.19) is False


def test_labels_are_normalised_before_lookup():
    """The vocabulary is normalised on the way in; the gates have to match it or
    an override silently does nothing."""
    d = _Gates(0.30, {"Tomato_Soup_Can": 0.20})
    assert d._admits("tomato soup can", 0.22) is True


def test_inference_runs_at_the_lowest_gate_anyone_asks_for():
    """A detection ultralytics never returns cannot be admitted afterwards, so
    predict() has to be called at the floor and the per-class gates applied to
    what comes back."""
    assert _Gates(0.30, {})._floor_conf() == pytest.approx(0.30)
    assert _Gates(0.30, {"banana": 0.20})._floor_conf() == pytest.approx(0.20)
    # An override ABOVE the global gate must not raise the inference floor and
    # silently discard everything else's detections.
    assert _Gates(0.30, {"banana": 0.45})._floor_conf() == pytest.approx(0.30)


def test_a_stricter_override_is_honoured_too():
    """The mechanism is symmetric: it is a per-class threshold, not a discount."""
    d = _Gates(0.30, {"cracker box": 0.45})
    assert d._admits("cracker box", 0.40) is False
    assert d._admits("cracker box", 0.50) is True


def test_no_overrides_is_exactly_the_old_behaviour():
    d = _Gates(0.30, {})
    for score in (0.0, 0.29, 0.30, 0.99):
        assert d._admits("anything", score) == (score >= 0.30)
