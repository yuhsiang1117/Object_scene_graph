"""The second look: what it crops, what it keeps, and what it counts.

The measurement behind the module is in its docstring; these pin the mechanics
it rests on, because every one of them is a way to report a recovery that did
not happen -- a crop taken at the wrong place, a mask pasted at the wrong
offset, or a duplicate counted as an addition.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import Detection
from osg.perception.foveate import _square, _to_full_frame, merge


def _det(label, box, mask_shape=(40, 40), score=0.6):
    mask = np.zeros(mask_shape, dtype=bool)
    mask[5:15, 5:15] = True
    return Detection(label=label, score=score,
                     bbox_xyxy=np.asarray(box, dtype=float), mask=mask)


def test_the_window_is_square_and_padded():
    """A lopsided crop spends the magnification on the letterbox's grey bars,
    and a crop that ends at the object's own edge scored 0/21 in the probe --
    worse than not cropping at all."""
    x1, y1, x2, y2 = _square((100, 100, 300, 140), w=1280, h=960, pad=0.15)
    assert (x2 - x1) == (y2 - y1), "the window must be square"
    assert (x2 - x1) > 200, "and larger than the surface it is taken around"
    assert 0.5 * (x1 + x2) == 200 and 0.5 * (y1 + y2) == 120


def test_the_window_is_clipped_to_the_image():
    x1, y1, x2, y2 = _square((0, 0, 60, 60), w=100, h=100, pad=0.5)
    assert x1 >= 0 and y1 >= 0 and x2 <= 100 and y2 <= 100


def test_a_crop_detection_comes_back_in_full_frame_coordinates():
    """The detector is handed the crop and answers in the crop's coordinates.
    Pasting that back at the wrong offset puts the object somewhere the agent
    will then drive to."""
    det = _det("tin can", (5.0, 5.0, 15.0, 15.0))

    out = _to_full_frame(det, x0=200, y0=300, crop_hw=(40, 40), full_hw=(960, 1280))

    assert np.allclose(out.bbox_xyxy, [205.0, 305.0, 215.0, 315.0])
    assert out.mask.shape == (960, 1280)
    ys, xs = np.nonzero(out.mask)
    assert xs.min() == 205 and ys.min() == 305
    assert out.label == "tin can" and out.score == det.score


def test_a_mask_at_a_different_resolution_is_resized_not_misplaced():
    det = _det("tin can", (5.0, 5.0, 15.0, 15.0), mask_shape=(20, 20))

    out = _to_full_frame(det, x0=0, y0=0, crop_hw=(40, 40), full_hw=(100, 100))

    assert out.mask.shape == (100, 100)
    # The 20 px mask covered rows 5-15 of 20; at 40 px that is rows 10-30.
    ys, _ = np.nonzero(out.mask)
    assert ys.min() == 10 and ys.max() == 29


def test_only_what_the_first_pass_missed_is_counted_as_added():
    """The arm's entire claim is what the second look ADDS. A foveated
    detection that duplicates one the whole-frame pass already made is not
    evidence of anything, and counting it makes a null look like a result."""
    base = [_det("sofa", (0.0, 0.0, 100.0, 100.0))]
    duplicate = _det("sofa", (2.0, 2.0, 98.0, 98.0))
    novel = _det("tin can", (10.0, 10.0, 30.0, 30.0))

    merged, n_added = merge(base, [duplicate, novel])

    assert n_added == 1
    assert [d.label for d in merged] == ["sofa", "tin can"]


def test_the_same_box_under_a_different_label_is_not_a_duplicate():
    """`sofa` and `tin can` over the same pixels is exactly the situation the
    module exists for -- the can is INSIDE the sofa's box."""
    base = [_det("sofa", (0.0, 0.0, 100.0, 100.0))]
    inside = _det("tin can", (0.0, 0.0, 100.0, 100.0))

    _, n_added = merge(base, [inside])

    assert n_added == 1
