"""The appearance channel's arithmetic, and its promise to stay out of the way.

The last of these is the one that matters most: every scored result on the
DualMap benchmark was produced with this code absent, and runs there are exactly
reproducible, so a candidate list or a surface order that shifts by a hair when
the feature term is off would silently invalidate the comparison it exists to
serve.
"""
from __future__ import annotations

import numpy as np

from osg.exploration.search_belief import InspectionLog, build_container_candidates
from osg.objects.feature_memory import (
    admits,
    blend_label_text,
    container_children,
    container_feature_scores,
    cosine,
    feature_term,
    merge_running_mean,
    refuted,
)


def unit(*values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Track:
    def __init__(self, tid, label="thing", ft=None, **kw):
        self.id = tid
        self.label = label
        self.clip_ft = ft
        self.feature_sim = -1.0
        self.feature_admitted = False
        self.from_prior = kw.get("from_prior", False)
        self._seen_live = kw.get("seen_live", False)
        self.absence_arrivals = kw.get("absence_arrivals", 0)
        self.failed_attempts = kw.get("failed_attempts", 0)
        self.blacklisted = kw.get("blacklisted", False)
        self.disabled = kw.get("disabled", False)

    @property
    def seen_live(self):
        return self._seen_live


class _Layer:
    def __init__(self, tracks):
        self._by_id = {t.id: t for t in tracks}

    def get(self, tid):
        return self._by_id.get(int(tid))


class _Node:
    def __init__(self, cid, label, centre, top_h=0.75, area=1.0, track_ids=()):
        self.id = cid
        self.label = label
        self.center = np.asarray(centre, dtype=float)
        self.top_h = top_h
        self.area_m2 = area
        self.room_id = 0
        self.floor = 0
        self.floor_id = 0
        self.track_ids = list(track_ids)
        self.object_ids = []


class _Obj:
    def __init__(self, tid, centre, container_id=None):
        self.track_id = tid
        self.center = np.asarray(centre, dtype=float)
        self.container_id = container_id
        self.floor_id = 0


class _Graph:
    def __init__(self, containers, objects):
        self.containers = {c.id: c for c in containers}
        self.objects = list(objects)
        self.floors = {}
        self.rooms = {}


# ------------------------------------------------------------------ the term

def test_the_best_looking_surface_is_unweighted_and_the_rest_decay():
    assert feature_term(0.30, 0.30, beta=20.0, floor=0.25) == 1.0
    near = feature_term(0.29, 0.30, beta=20.0, floor=0.25)
    far = feature_term(0.20, 0.30, beta=20.0, floor=0.25)
    assert 0.25 < far < near < 1.0


def test_a_surface_with_no_feature_is_ordinary_not_impossible():
    """`floor`, never zero: half the objects in a house were never mapped, so
    silence about a surface is not evidence against it."""
    assert feature_term(None, 0.30, beta=20.0, floor=0.25) == 0.25
    assert feature_term(None, 0.30, beta=20.0, floor=0.0) == 0.0


def test_beta_zero_is_the_off_switch():
    assert feature_term(0.10, 0.30, beta=0.0, floor=0.25) == 1.0


def test_cosine_is_rounded_and_missing_features_score_minus_one():
    a, b = unit(1, 0, 0), unit(1, 0, 0)
    assert cosine(a, b) == 1.0
    assert cosine(None, b) == -1.0
    assert cosine(a, None) == -1.0
    assert cosine(a, unit(0, 1, 0)) == 0.0
    # Rounded, so a kernel that differs in the seventh decimal cannot flip an
    # admission and break a reproducible run.
    assert cosine(np.float32([1, 1e-9, 0]), unit(1, 0, 0)) == 1.0


# ------------------------------------------------------------------- merging

def test_merged_track_features_stay_unit_and_average():
    ft, n = merge_running_mean(None, 0, unit(1, 0, 0))
    assert n == 1 and np.isclose(np.linalg.norm(ft), 1.0)
    ft, n = merge_running_mean(ft, n, unit(0, 1, 0))
    assert n == 2 and np.isclose(np.linalg.norm(ft), 1.0)
    assert np.allclose(ft, unit(1, 1, 0), atol=1e-6)
    same, n_same = merge_running_mean(ft, n, None)
    assert n_same == 2 and np.allclose(same, ft)


def test_label_text_blending_is_off_at_weight_zero():
    image = unit(1, 0, 0)
    assert np.allclose(blend_label_text(image, unit(0, 1, 0), 0.0), image)
    mixed = blend_label_text(image, unit(0, 1, 0), 0.3)
    assert np.isclose(np.linalg.norm(mixed), 1.0) and mixed[1] > 0


# ---------------------------------------------------------------- admission

def test_admission_needs_the_threshold_and_holds_back_unseen_prior_tracks():
    live = _Track(1, ft=unit(1, 0, 0))
    live.feature_sim = 0.31
    assert admits(live, 0.30) is True
    live.feature_sim = 0.29
    assert admits(live, 0.30) is False

    ghost = _Track(2, from_prior=True, seen_live=False)
    ghost.feature_sim = 0.40
    assert admits(ghost, 0.30) is False
    assert admits(ghost, 0.30, admit_prior_tracks=True) is True

    seen = _Track(3, from_prior=True, seen_live=True)
    seen.feature_sim = 0.40
    assert admits(seen, 0.30) is True


def test_refutation_is_an_arrival_or_a_failed_attempt_not_a_low_belief():
    assert refuted(_Track(1)) is False
    assert refuted(_Track(2, absence_arrivals=1)) is True
    assert refuted(_Track(3, failed_attempts=1)) is True
    assert refuted(_Track(4, blacklisted=True)) is True
    assert refuted(None) is True


# ---------------------------------------------------------------- containers

def _scene():
    desk = _Node(10, "desk", (0.0, 0.0, 0.0), track_ids=[10])
    bed = _Node(20, "bed", (5.0, 0.0, 0.0), track_ids=[20])
    objects = [
        _Obj(1, (0.1, 0.8, 0.0), container_id=10),   # resting on the desk
        _Obj(2, (5.4, 0.8, 0.0), container_id=None),  # beside the bed, unbound
    ]
    return _Graph([desk, bed], objects)


def test_a_surface_inherits_the_appearance_of_what_it_holds():
    text = unit(1, 0, 0)
    tracks = [
        _Track(1, ft=unit(1, 0, 0)), _Track(2, ft=unit(0, 1, 0)),
        _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1)),
    ]
    scores = container_feature_scores(_scene(), _Layer(tracks), text)
    assert scores[10] == 1.0          # the desk, through its child
    assert scores[20] == 0.0          # the bed, on its own feature only


def test_the_child_set_widens_by_proximity_because_support_binds_few_objects():
    graph = _scene()
    bound = container_children(graph, radius_m=0.0)
    assert 2 not in bound[20]
    widened = container_children(graph, radius_m=1.0)
    assert 2 in widened[20]

    text = unit(0, 1, 0)
    tracks = [_Track(1, ft=unit(1, 0, 0)), _Track(2, ft=unit(0, 1, 0)),
              _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1))]
    assert container_feature_scores(graph, _Layer(tracks), text)[20] == 0.0
    widened_scores = container_feature_scores(graph, _Layer(tracks), text, radius_m=1.0)
    assert widened_scores[20] == 1.0


def test_a_refuted_child_stops_recommending_its_surface():
    text = unit(1, 0, 0)
    tracks = [_Track(1, ft=unit(1, 0, 0), absence_arrivals=1), _Track(2, ft=unit(0, 1, 0)),
              _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1))]
    scores = container_feature_scores(_scene(), _Layer(tracks), text)
    assert scores[10] == 0.0
    kept = container_feature_scores(_scene(), _Layer(tracks), text, skip_refuted=False)
    assert kept[10] == 1.0


def test_no_text_feature_means_no_scores_at_all():
    assert container_feature_scores(_scene(), _Layer([]), None) == {}


# ------------------------------------------------- the promise to stay away

def test_the_candidate_builder_is_untouched_when_no_features_are_supplied():
    graph = _scene()
    plain = build_container_candidates(graph, "bowl", InspectionLog())
    again = build_container_candidates(
        graph, "bowl", InspectionLog(), feature_scores=None, feature_beta=9.0,
        feature_floor=0.0,
    )
    assert [(c.ref_id, c.prior, c.goal_xy.tolist()) for c in plain] == \
           [(c.ref_id, c.prior, c.goal_xy.tolist()) for c in again]


def test_the_feature_term_can_reorder_surfaces_without_changing_the_scale():
    graph = _scene()
    plain = {c.ref_id: c.prior for c in build_container_candidates(graph, "bowl", InspectionLog())}
    fused = build_container_candidates(
        graph, "bowl", InspectionLog(), feature_scores={20: 0.31, 10: 0.20},
        feature_beta=20.0, feature_floor=0.1,
    )
    priors = {c.ref_id: c.prior for c in fused}
    assert priors[20] > priors[10]
    # The best candidate still carries the full surface mass, so the
    # search-versus-explore comparison sees the same magnitude it always did.
    assert max(priors.values()) == max(plain.values())


# ------------------------------------------------------- the local inquiry

class _Cfg:
    local_pick_on_arrival = True
    local_radius_m = 2.0
    local_min_obs = 3
    max_local_picks = 2
    max_live_crops_per_keyframe = 32
    admit_threshold = 0.0
    admit_prior_tracks = False
    prompt_template = "a photo of a {target}"


class _Encoder:
    """Text and image features chosen by the test, no model."""

    def __init__(self, text, images=None):
        self._text = text
        self._images = images or {}

    def text_feature(self, prompt):
        return self._text

    def encode_images(self, crops):
        return np.stack([self._images[id(c)] for c in crops])


def _live(tid, label, ft, xy, n_obs=5, **kw):
    t = _Track(tid, label, ft, **kw)
    t._seen_live = kw.get("seen_live", True)
    t.n_obs = n_obs
    t.ellipsoid = type("E", (), {"center": np.array([xy[0], 0.8, xy[1]])})()
    return t


def _memory(target_ft):
    from osg.objects.feature_memory import FeatureMemory

    fm = FeatureMemory(_Encoder(target_ft), _Cfg())
    fm.set_target("mug")
    return fm


def test_the_pick_takes_the_argmax_and_never_needs_a_threshold():
    """Even a poor best is returned: DualMap's matcher always answers, which is
    the whole difference from an admission bar."""
    fm = _memory(unit(1, 0, 0))
    near = [_live(1, "towel", unit(0.9, 0.44, 0), (0.2, 0.0)),
            _live(2, "chair", unit(0.2, 0.98, 0), (0.5, 0.0))]
    assert fm.best_near((0.0, 0.0), near).id == 1
    fm2 = _memory(unit(1, 0, 0))
    weak = [_live(3, "chair", unit(0.05, 0.999, 0), (0.3, 0.0))]
    assert fm2.best_near((0.0, 0.0), weak).id == 3


def test_the_pick_is_scoped_by_radius_observations_and_liveness():
    fm = _memory(unit(1, 0, 0))
    tracks = [
        _live(1, "towel", unit(1, 0, 0), (9.0, 0.0)),               # too far
        _live(2, "towel", unit(1, 0, 0), (0.3, 0.0), n_obs=2),      # too few views
        _live(3, "towel", unit(1, 0, 0), (0.3, 0.0), seen_live=False, from_prior=True),
        _live(4, "towel", unit(1, 0, 0), (0.3, 0.0), absence_arrivals=1),  # refuted
        _live(5, "desk", unit(0.7, 0.71, 0), (0.4, 0.0)),           # the only eligible one
    ]
    assert fm.best_near((0.0, 0.0), tracks).id == 5


def test_a_track_is_picked_at_most_once_and_the_budget_is_honoured():
    fm = _memory(unit(1, 0, 0))
    tracks = [_live(1, "towel", unit(1, 0, 0), (0.2, 0.0)),
              _live(2, "desk", unit(0.9, 0.44, 0), (0.3, 0.0)),
              _live(3, "rug", unit(0.8, 0.6, 0), (0.4, 0.0))]
    assert fm.best_near((0.0, 0.0), tracks).id == 1
    assert fm.best_near((0.0, 0.0), tracks).id == 2   # 1 is not offered again
    assert fm.counters["feature_local_picks"] == 2    # the caller enforces the cap


def test_no_eligible_track_means_no_pick_and_no_counter():
    fm = _memory(unit(1, 0, 0))
    assert fm.best_near((0.0, 0.0), []) is None
    assert fm.counters["feature_local_picks"] == 0


def test_the_running_mean_is_what_gets_matched():
    """DualMap matches an object's average appearance, not its best frame."""
    fm = _memory(unit(1, 0, 0))
    t = _live(1, "towel", None, (0.2, 0.0))
    t.clip_ft, t.clip_n = merge_running_mean(None, 0, unit(1, 0, 0))
    t.clip_ft, t.clip_n = merge_running_mean(t.clip_ft, t.clip_n, unit(0, 1, 0))
    assert fm.best_near((0.0, 0.0), [t]).id == 1
    assert abs(fm.sim(t.clip_ft) - float(np.dot(unit(1, 1, 0), unit(1, 0, 0)))) < 1e-3


def _sized(tid, label, ft, xy, extent_m, **kw):
    t = _live(tid, label, ft, xy, **kw)
    half = extent_m / 2.0
    t.ellipsoid = type("E", (), {
        "center": np.array([xy[0], 0.8, xy[1]]),
        "axes": np.array([half, half, half]),
    })()
    return t


def test_the_pick_will_not_choose_furniture():
    """A bed's crop matches "a photo of a mug" better than forty pixels of the
    real mug, so the bound is on geometry, not on the score."""
    class Cfg(_Cfg):
        local_max_extent_m = 0.6

    from osg.objects.feature_memory import FeatureMemory
    fm = FeatureMemory(_Encoder(unit(1, 0, 0)), Cfg())
    fm.set_target("mug")
    tracks = [
        _sized(1, "bed", unit(1, 0, 0), (0.3, 0.0), 2.0),      # perfect match, too big
        _sized(2, "towel", unit(0.6, 0.8, 0), (0.4, 0.0), 0.2),  # worse match, mug-sized
    ]
    pick = fm.best_near((0.0, 0.0), tracks)
    assert pick.id == 2
    assert fm.counters["feature_local_too_large"] == 1


def test_the_size_bound_is_off_at_zero_and_tolerates_a_missing_ellipsoid():
    class Cfg(_Cfg):
        local_max_extent_m = 0.0

    from osg.objects.feature_memory import FeatureMemory
    fm = FeatureMemory(_Encoder(unit(1, 0, 0)), Cfg())
    fm.set_target("mug")
    assert fm.best_near((0.0, 0.0), [_sized(1, "bed", unit(1, 0, 0), (0.3, 0.0), 2.0)]).id == 1
