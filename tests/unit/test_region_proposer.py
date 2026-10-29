"""The proposal stage finds what the detector will not name -- within limits.

Measured (docs/history/PERCEPTION_GAP.md): a tight proposal on the object comes from
YOLOE on 0-48% of frames and from FastSAM on 72-100%, and YOLOE calls the boxes
that do cover it `mirror`, `lamp`, `chair`. But appearance can only rank the
region first for compact objects, and a threshold at 0.24 buys 85% precision at
24% recall -- so the stage is gated, capped, and hands its winner into the
ordinary detection path to be judged like any other.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.types import Detection
from osg.perception.region_proposer import RegionProposer, phrase


class _Cfg:
    imgsz, conf, iou, device = 1024, 0.25, 0.7, "cpu"
    tau, admit_score, pad_frac = 0.24, 0.5, 0.25
    min_area_px, max_area_frac, max_regions = 200.0, 0.02, 64
    only_when_unnamed, max_per_episode = True, 40
    use_for_absence = True
    require_never_named, unnamed_keyframes = True, 0


class _Boxes:
    def __init__(self, b): self.xyxy = _T(np.asarray(b, dtype=float))
    def __len__(self): return len(self.xyxy.v)


class _T:
    def __init__(self, v): self.v = v
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.v
    def __getitem__(self, i): return _T(self.v[i])   # masks are indexed on the device


class _Masks:
    def __init__(self, m): self.data = _T(np.asarray(m))


class _Result:
    def __init__(self, boxes, masks):
        self.boxes = _Boxes(boxes) if boxes is not None else None
        self.masks = _Masks(masks) if masks is not None else None


class _Model:
    def __init__(self, boxes, masks): self._r = _Result(boxes, masks)
    def predict(self, *a, **k): return [self._r]


class _Enc:
    """Scores region i by `sims[i]`; the text feature is a one-hot selector."""
    def __init__(self, sims): self.sims = np.asarray(sims, dtype=np.float32)
    def text_feature(self, text): self._t = text; return np.array([1.0], dtype=np.float32)
    def encode_images(self, crops): return self.sims[: len(crops)].reshape(-1, 1)


def _proposer(boxes, masks, sims, **over):
    cfg = _Cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    rp = RegionProposer(_Model(boxes, masks), _Enc(sims), cfg)
    rp.set_target("scissors")
    return rp


def _frame(h=960, w=1280):
    """The live frame size. The size band is a FRACTION of the frame, so a
    toy-sized frame would call an ordinary object furniture."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_the_query_is_phrased_naturally():
    assert phrase("scissors") == "a pair of scissors"
    assert phrase("tin can") == "a tin can of soup"
    assert phrase("teapot") == "a photo of a teapot"   # unknown falls back


def test_the_best_region_is_returned_as_a_detection():
    boxes = [[10, 10, 40, 40], [50, 50, 80, 80]]
    masks = [np.zeros((960, 1280), bool), np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.20, 0.31])
    det = rp.propose(_frame())
    assert isinstance(det, Detection)
    assert det.label == "scissors"
    assert np.allclose(det.bbox_xyxy, [50, 50, 80, 80]), "the higher-scoring region wins"
    assert rp.counters["region_admitted"] == 1


def test_a_best_region_below_the_threshold_is_declined():
    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.21])
    assert rp.propose(_frame()) is None
    assert rp.counters["region_below_tau"] == 1
    assert rp.counters["region_admitted"] == 0


def test_the_admitted_score_is_the_detector_scale_not_the_cosine():
    """Downstream gates are calibrated against detector scores; a 0.24 cosine
    fed into them reads as an almost-certainly-false detection."""
    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])
    det = rp.propose(_frame())
    assert det.score == pytest.approx(0.5)


def test_furniture_sized_regions_are_refused():
    """max_area_frac is what separates the object from the thing it rests on."""
    boxes = [[0, 0, 1279, 959]]         # the whole frame
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])
    assert rp.propose(_frame()) is None


def test_slivers_are_refused():
    boxes = [[10, 10, 15, 15]]          # 25 px, under min_area_px
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])
    assert rp.propose(_frame()) is None


def test_nothing_segmented_is_not_an_error():
    rp = _proposer(None, None, [])
    assert rp.propose(_frame()) is None


def test_without_a_target_it_declines():
    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    cfg = _Cfg()
    rp = RegionProposer(_Model(boxes, masks), _Enc([0.9]), cfg)
    assert rp.propose(_frame()) is None, "set_target has not been called"


def test_set_target_clears_the_counters():
    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])
    rp.propose(_frame())
    assert rp.counters["region_admitted"] == 1
    rp.set_target("mug")
    assert rp.counters["region_admitted"] == 0, "counters are per episode"


# ------------------------------------------------- the absence sensor

def test_the_absence_test_must_be_able_to_see_a_proposal():
    """`_best_target_detection` re-runs the raw detector and filters by label,
    and the close look asks it before declaring a committed track ABSENT. For
    the objects this stage exists for that is circular: the detector that could
    not name the object is asked whether it is there, says no, and the track is
    retired. Measured on in_anchor__0117__banana -- looked from 1.5 m, told
    "not detected", belief 0.95 -> 0.433, committed elsewhere, in an episode
    where the stage had admitted 16 regions."""
    import numpy as np

    from osg.core.types import Detection

    class _Agent:
        """Just the three attributes `_region_detection` touches."""
        target = "scissors"
        stats: dict = {}

        def __init__(self, rp):
            self.region_proposer = rp
            self._region_cache = None
            self._region_admits = 0
            self._region_kf = 99      # past the grace period
            self._region_named = False
            self.stats = {}

        class _Prof:
            def timeit(self, _):
                import contextlib
                return contextlib.nullcontext()

        profiler = _Prof()

    from osg.agent.nav_agent import NavAgent

    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])
    agent = _Agent(rp)

    class _Frame:
        frame_id = 7
        rgb = np.zeros((960, 1280, 3), dtype=np.uint8)

    agent._region_active = lambda: NavAgent._region_active(agent)
    got = NavAgent._region_detection(agent, _Frame())
    assert isinstance(got, Detection) and got.label == "scissors"


def test_the_frame_result_is_computed_once():
    """Several callers ask per step -- the approach stop, the close look, the
    absence sensor. Segmenting per CALLER would multiply the cost by the number
    of askers rather than by the number of frames."""
    import numpy as np

    from osg.agent.nav_agent import NavAgent

    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90])

    class _Agent:
        target = "scissors"

        def __init__(self):
            self.region_proposer = rp
            self._region_cache = None
            self._region_admits = 0
            self._region_kf = 99
            self._region_named = False
            self.stats = {}

        class _Prof:
            def timeit(self, _):
                import contextlib
                return contextlib.nullcontext()

        profiler = _Prof()

    class _Frame:
        frame_id = 7
        rgb = np.zeros((960, 1280, 3), dtype=np.uint8)

    agent, frame = _Agent(), _Frame()
    agent._region_active = lambda: NavAgent._region_active(agent)
    NavAgent._region_detection(agent, frame)
    NavAgent._region_detection(agent, frame)
    NavAgent._region_detection(agent, frame)
    assert rp.counters["region_frames"] == 1, "one segment per frame, not per caller"


def test_the_absence_fallback_can_be_switched_off():
    import numpy as np

    from osg.agent.nav_agent import NavAgent

    boxes = [[10, 10, 40, 40]]
    masks = [np.zeros((960, 1280), bool)]
    rp = _proposer(boxes, masks, [0.90], use_for_absence=False)

    class _Agent:
        target = "scissors"

        def __init__(self):
            self.region_proposer = rp
            self._region_cache = None
            self._region_admits = 0
            self._region_kf = 99
            self._region_named = False
            self.stats = {}

        class _Prof:
            def timeit(self, _):
                import contextlib
                return contextlib.nullcontext()

        profiler = _Prof()

    class _Frame:
        frame_id = 7
        rgb = np.zeros((960, 1280, 3), dtype=np.uint8)

    a = _Agent()
    a._region_active = lambda: NavAgent._region_active(a)
    assert NavAgent._region_detection(a, _Frame()) is None


def test_once_the_detector_names_the_target_the_stage_stands_down():
    """The gate that the full 107 said was wrong. Per-FRAME, the stage fires on
    a working trial because the target is absent from most individual frames;
    measured, that took the other 86 trials from 57 to 41 with 13 of the 18
    losses exhausting all three attempts. Episode-level, a detector that can see
    this object switches the fallback off."""
    from osg.agent.nav_agent import NavAgent

    class _A:
        def __init__(self, named, kf=99, admits=0):
            self.region_proposer = _proposer([[10, 10, 40, 40]],
                                             [np.zeros((960, 1280), bool)], [0.9])
            self._region_named, self._region_kf, self._region_admits = named, kf, admits

    assert NavAgent._region_active(_A(named=False)) is True
    assert NavAgent._region_active(_A(named=True)) is False, "the detector can see it"


def test_the_grace_period_holds_the_stage_off_early():
    from osg.agent.nav_agent import NavAgent

    class _A:
        def __init__(self, kf):
            rp = _proposer([[10, 10, 40, 40]], [np.zeros((960, 1280), bool)], [0.9])
            rp.cfg.unnamed_keyframes = 20
            self.region_proposer = rp
            self._region_named, self._region_kf, self._region_admits = False, kf, 0

    assert NavAgent._region_active(_A(kf=5)) is False
    assert NavAgent._region_active(_A(kf=25)) is True


def test_the_per_episode_cap_still_binds():
    from osg.agent.nav_agent import NavAgent

    class _A:
        def __init__(self, admits):
            self.region_proposer = _proposer([[10, 10, 40, 40]],
                                             [np.zeros((960, 1280), bool)], [0.9])
            self._region_named, self._region_kf, self._region_admits = False, 99, admits

    assert NavAgent._region_active(_A(admits=39)) is True
    assert NavAgent._region_active(_A(admits=40)) is False
