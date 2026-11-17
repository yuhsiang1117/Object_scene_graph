"""`agent.speed.*`: how ascentnav calls its served models (S88).

Each switch must leave every action and every map the step builds unchanged;
these tests pin that on stub models, and the 3-episode smoke pins it live.
"""
from __future__ import annotations

import threading

import numpy as np

from osg.core.types import Detection
from osg.perception.detector import DFineDetector

from .test_ascentnav_stairs import _wall_frame

SPEED_OFF = dict(sam_target_only=False, lazy_stair_detector=False, parallel_servers=False)


# ------------------------------------------------------------------ D-FINE masks


def _dfine(replies):
    d = DFineDetector()
    calls = []

    def post(url, payload):
        calls.append(url)
        if url == d.url:
            return replies
        h, w = 480, 640
        import base64
        return {"cropped_mask": base64.b64encode(np.ones(h * w, np.uint8).tobytes()).decode()}

    d._post = post
    d.set_vocabulary(["chair", "couch", "tv"])
    return d, calls


REPLY = {"boxes": [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],
         "logits": [0.9, 0.95], "phrases": ["chair", "couch"]}


def test_every_admitted_box_is_masked_by_default():
    d, calls = _dfine(REPLY)
    out = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert [x.label for x in out] == ["chair", "sofa"]
    assert calls.count(d.sam_url) == 2


def test_mask_labels_send_only_the_kept_label_to_mobilesam():
    d, calls = _dfine(REPLY)
    d.set_mask_labels(["chair"])
    out = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert [x.label for x in out] == ["chair", "sofa"], "every detection is still returned"
    assert calls.count(d.sam_url) == 1 and d.n_sam_skipped == 1
    assert out[0].mask.all(), "the kept one carries MobileSAM's mask"
    box = out[1].mask
    assert box[240:336, 320:448].all() and not box[:240].any(), "the other its box"


def test_mask_labels_match_hm3d_names_with_underscores():
    reply = {"boxes": [[0.1, 0.1, 0.3, 0.3]], "logits": [0.9], "phrases": ["tv"]}
    d, calls = _dfine(reply)
    d.set_mask_labels(["tv monitor"])
    out = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert out[0].label == "tv_monitor" and calls.count(d.sam_url) == 1


# ------------------------------------------------------------- the whole step


class _Det:
    """Finds the target at a fixed box on odd frames."""

    def __init__(self):
        self.mask_labels = None
        self.n = 0

    def set_vocabulary(self, classes):
        pass

    def set_mask_labels(self, labels):
        self.mask_labels = labels

    def detect(self, rgb):
        self.n += 1
        if int(rgb[0, 0, 0]) % 2 == 0:
            return []
        m = np.zeros(rgb.shape[:2], bool)
        m[200:280, 280:360] = True
        return [Detection(label="chair", score=0.9,
                          bbox_xyxy=np.array([280.0, 200.0, 360.0, 280.0]), mask=m),
                Detection(label="sofa", score=0.9,
                          bbox_xyxy=np.array([0.0, 0.0, 60.0, 60.0]), mask=m.copy())]


class _Value:
    def __init__(self):
        self.threads = set()

    def score(self, rgb, texts):
        self.threads.add(threading.current_thread().name)
        return np.array([0.1 + 0.01 * int(rgb[0, 0, 0])], np.float32)


class _Ram:
    def tags(self, rgb, img_b64=None):
        return [f"tag{int(rgb[0, 0, 0])}"]


class _Stair:
    def __init__(self):
        self.n = 0

    def mask(self, rgb, img_b64=None):
        self.n += 1
        m = np.zeros(rgb.shape[:2], bool)
        m[300:400, 200:400] = True
        return m


class _Seg:
    """RedNet: stairs in view on every third frame only."""

    def stair_mask(self, frame):
        if int(frame.rgb[0, 0, 0]) % 3 != 0:
            return None
        m = np.zeros(frame.rgb.shape[:2], bool)
        m[300:420, 200:420] = True
        return m


def _run(speed, n=8):
    from osg.core.config import OSGConfig
    from osg.exploration.async_scorer import AsyncScorer
    from navigation.agent import AscentNavAgent

    from .test_ascent_agent import _Driver
    from .test_nav_agent import _StubScorer

    cfg = OSGConfig()
    cfg.agent.navigation = "pointnav"
    cfg.agent.initial_scan = False
    cfg.agent.stair_up_mode = "ascent"
    for k, v in speed.items():
        setattr(cfg.agent.speed, k, v)
    det, value, stair = _Det(), _Value(), _Stair()
    a = AscentNavAgent(cfg, det, AsyncScorer(_StubScorer()), None, "chair",
                       pointnav=_Driver(), image_text=value, ram=_Ram(),
                       stair_segmenter=_Seg(), stair_detector=stair)
    a.reset("chair")
    a.obstacle_map._floor_num_steps = 60
    actions, cosines = [], []
    for i in range(n):
        f = _wall_frame([0.1 * i, 0.88, 0], [1 + 0.1 * i, 0.88, 0.2])
        f.rgb[:] = i + 1
        actions.append(a.act(f))
        cosines.append(a._blip_cosine)
    return a, det, value, stair, actions, cosines


def test_the_speed_switches_change_no_action_and_no_map():
    off = _run(SPEED_OFF)
    on = _run({})
    a0, a1 = off[0], on[0]
    assert off[4] == on[4], "actions"
    assert off[5] == on[5], "the gate's cosine"
    assert a0.object_map.each_step_objects == a1.object_map.each_step_objects
    assert np.array_equal(a0.obstacle_map._map, a1.obstacle_map._map)
    assert np.array_equal(a0.obstacle_map._up_stair_map, a1.obstacle_map._up_stair_map)
    assert np.array_equal(a0.value_map._value_map, a1.value_map._value_map)
    assert set(a0.object_map.clouds) == set(a1.object_map.clouds)
    for k in a0.object_map.clouds:
        assert np.array_equal(a0.object_map.clouds[k], a1.object_map.clouds[k])


def test_the_defaults_mask_the_target_skip_empty_rednet_frames_and_run_in_parallel():
    a, det, value, stair, _, _ = _run({})
    assert det.mask_labels == ["chair"]
    # RedNet saw stairs on frames 3 and 6 of 1..8; the detector ran on those only.
    assert stair.n == 2 and a.stats["stair_det_skipped"] == 6
    assert any(t.startswith("ascentnav-model") for t in value.threads)


def test_off_calls_the_stair_detector_on_every_detecting_step():
    _, det, value, stair, _, _ = _run(SPEED_OFF)
    assert det.mask_labels is None and stair.n == 8
    assert value.threads == {threading.main_thread().name}


def test_lazy_without_parallel_counts_the_skips():
    a, _, _, stair, _, _ = _run(dict(parallel_servers=False))
    assert stair.n == 2 and a.stats["stair_det_skipped"] == 6
