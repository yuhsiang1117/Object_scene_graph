"""Mid-episode relocation and the dynamic-scene metrics (Phase 2).

What this protocol buys, and why it is the gate for every later number: with one
layout per episode the world only ever changes BETWEEN episodes, so an agent can
be scored on recovering from a stale map but never on noticing a change. Belief
latency is undefined -- there is no moment to measure from. These tests pin the
pairing that makes a change possible and the metrics that read it back.

Habitat-free: pairing is a scene-level join over already-validated layouts, and
the metrics are pure functions of the episode records.
"""
import pytest

from osg.eval.metrics import belief_latency, dynamic_summary, ghost_rate, stale_goal_rate
from osg.sim.ycb_layouts import RelocationPair, YCBLayoutError, relocation_pairs


class _Obj:
    def __init__(self, sid, translation, anchor="table_1"):
        self.semantic_id = sid
        self.handle = f"h{sid}"
        self.translation = tuple(translation)
        self.anchor_object_id = anchor


class _Layout:
    def __init__(self, scene, layout_type, layout_id, objects):
        self.scene_name = scene
        self.layout_type = layout_type
        self.layout_id = layout_id
        self.objects = tuple(objects)


def static(scene="s1", objs=None):
    return _Layout(scene, "static", "static", objs or [_Obj(1, (0.0, 0.5, 0.0))])


def dynamic(scene="s1", kind="cross_anchor", lid="cross_anchor_1", objs=None):
    return _Layout(scene, kind, lid, objs or [_Obj(1, (3.0, 0.5, 2.0), anchor="shelf_9")])


# ----------------------------------------------------------------- pairing


def test_each_dynamic_layout_pairs_with_its_own_scenes_static():
    a, b = static("s1"), static("s2")
    d1, d2 = dynamic("s1"), dynamic("s2", lid="in_anchor_1", kind="in_anchor")
    pairs = relocation_pairs([a, b, d1, d2])
    assert {(p.scene_name, p.kind) for p in pairs} == {("s1", "cross_anchor"), ("s2", "in_anchor")}
    for p in pairs:
        assert p.before.scene_name == p.after.scene_name, "paired across scenes"


def test_a_dynamic_layout_without_its_static_is_a_named_error():
    """Selecting only the destination leaves nothing to change FROM -- worth an
    explicit message rather than a silently static run."""
    with pytest.raises(YCBLayoutError, match="no static layout in the selection"):
        relocation_pairs([dynamic("s1")])


def test_static_only_selection_yields_no_pairs():
    assert relocation_pairs([static("s1"), static("s2")]) == ()


def test_moved_ids_compare_poses_rather_than_trusting_the_manifest():
    before = static(objs=[_Obj(1, (0.0, 0.5, 0.0)), _Obj(2, (1.0, 0.5, 0.0))])
    after = dynamic(objs=[_Obj(1, (3.0, 0.5, 2.0)), _Obj(2, (1.0, 0.5, 0.0))])
    pair = RelocationPair(before=before, after=after)
    assert pair.moved_semantic_ids() == (1,), "object 2 did not move"


def test_destination_lookup_names_the_missing_object():
    pair = RelocationPair(before=static(), after=dynamic())
    assert pair.destination_of(1).translation == (3.0, 0.5, 2.0)
    with pytest.raises(YCBLayoutError, match="not placed"):
        pair.destination_of(99)


# ----------------------------------------------------------------- metrics


def episode(*, target="mug", step=10, events=(), commits=(), tracks=(), origin=(0.0, 0.5, 0.0)):
    return {
        "target": target,
        "authored_layout": {
            "relocation": {"step": step, "origin_position": list(origin) if origin else None}
        },
        "presence_events": list(events),
        "goal_commit_log": list(commits),
        "target_tracks": list(tracks),
    }


def test_belief_latency_measures_from_the_relocation():
    r = episode(step=10, events=[{"step": 34, "label": "mug"}])
    assert belief_latency([r])["mean_steps"] == 24.0


def test_belief_latency_ignores_a_flip_that_predates_the_change():
    """A belief that collapsed before the object moved says nothing about how
    fast the map noticed THIS change."""
    r = episode(step=40, events=[{"step": 5, "label": "mug"}])
    out = belief_latency([r])
    assert out["flip_rate"] == 0.0 and "mean_steps" not in out


def test_belief_latency_reports_never_noticed_separately():
    """Folding a non-event in as a zero would make a system that never notices
    look instant."""
    out = belief_latency([episode(events=[{"step": 34, "label": "mug"}]), episode(events=[])])
    assert out["n_relocated"] == 2 and out["flip_rate"] == 0.5
    assert out["mean_steps"] == 24.0


def test_belief_latency_only_counts_the_target_category():
    r = episode(target="mug", step=10, events=[{"step": 20, "label": "bowl"}])
    assert belief_latency([r])["flip_rate"] == 0.0


def test_belief_latency_is_empty_without_a_relocation():
    assert belief_latency([{"target": "mug", "authored_layout": {}}]) == {}


def test_stale_goal_rate_counts_commitments_to_disbelieved_objects():
    r = episode(commits=[{"p": 0.9}, {"p": 0.2}, {"p": 0.05}])
    assert stale_goal_rate([r]) == {"n_commits": 3, "stale_rate": pytest.approx(2 / 3, abs=1e-4)}


def test_ghost_rate_counts_a_target_still_believed_at_the_old_pose():
    still_there = episode(tracks=[{"label": "mug", "center": [0.1, 0.5, 0.0], "p": 0.9}])
    forgotten = episode(tracks=[{"label": "mug", "center": [0.1, 0.5, 0.0], "p": 0.02}])
    moved_on = episode(tracks=[{"label": "mug", "center": [5.0, 0.5, 5.0], "p": 0.9}])
    assert ghost_rate([still_there])["ghost_rate"] == 1.0
    assert ghost_rate([forgotten])["ghost_rate"] == 0.0
    assert ghost_rate([moved_on])["ghost_rate"] == 0.0
    assert ghost_rate([still_there, forgotten])["ghost_rate"] == 0.5


def test_dynamic_summary_omits_metrics_with_no_data():
    """A static run must not sprout empty dynamic sections."""
    assert dynamic_summary([{"target": "mug", "authored_layout": {}}]) == {}
    out = dynamic_summary([episode(events=[{"step": 12, "label": "mug"}],
                                   commits=[{"p": 0.1}],
                                   tracks=[{"label": "mug", "center": [0, 0.5, 0], "p": 0.9}])])
    assert set(out) == {"belief_latency", "stale_goals", "ghosts"}


# ------------------------------------------------- offline vs live change


class _Policy:
    """The firing rule in isolation -- see YCBAuthoredNavEnv._maybe_relocate."""

    def __init__(self, enabled, at_step):
        self.enabled, self.at_step = enabled, at_step

    def should_fire(self, frame_id, pair_exists, already):
        if not self.enabled or not pair_exists or already:
            return False
        return frame_id >= self.at_step


def test_a_disabled_policy_never_fires_even_though_the_pair_is_known():
    """The two-pass protocol needs the pair for metadata (what moved, from
    where) while the world must NOT change during the episode. Firing here would
    re-apply poses the world is already in and record a live change that never
    happened, hiding the offline one from the metrics."""
    policy = _Policy(enabled=False, at_step=-1)
    assert policy.should_fire(0, pair_exists=True, already=False) is False
    assert policy.should_fire(500, pair_exists=True, already=False) is False


def test_an_enabled_policy_fires_once_at_its_step():
    policy = _Policy(enabled=True, at_step=60)
    assert policy.should_fire(59, True, False) is False
    assert policy.should_fire(60, True, False) is True
    assert policy.should_fire(61, True, already=True) is False


def test_offline_change_is_reported_as_step_zero():
    """With the map loaded from a run before the objects moved, the map is stale
    from the first step -- so the belief-latency clock starts at 0."""
    r = episode(step=0, events=[{"step": 25, "label": "mug"}])
    assert belief_latency([r])["mean_steps"] == 25.0


# ----------------------------------------- C5: arriving and seeing nothing


class _Filter:
    """The belief update in isolation -- see NavAgent._absence_at_arrival."""

    def __init__(self, p=0.82):
        import math
        self.log_odds = math.log(p / (1 - p))

    def miss(self, recall, q=0.05):
        import math
        self.log_odds += math.log((1 - recall) / (1 - q))
        return 1 / (1 + math.exp(-self.log_odds))


THRESHOLD = 0.45  # the stricter A/B setting; the default is 1.0, see below


def test_one_vlm_no_abandons_a_ghost_the_detector_alone_would_keep():
    """The behaviour C5 buys, and the arithmetic the threshold is chosen from.
    Walking to where the map said an object was and finding nothing must be an
    observation, not just a failed trip -- otherwise the map learns nothing and
    sends the agent back next episode. But cheap evidence must earn it slowly:
    the detector's silence is worth log(0.5/0.95) and a trusted VLM "no"
    log(0.15/0.98), nearly three times as much."""
    assert _Filter().miss(recall=0.5) > THRESHOLD, "one detector miss should not decide it"
    assert _Filter().miss(recall=0.85, q=0.02) < THRESHOLD, "a trusted no should decide it"


def test_the_detector_alone_needs_three_failed_approaches():
    f, p = _Filter(), 1.0
    seen = []
    for _ in range(3):
        p = f.miss(recall=0.5)
        seen.append(round(p, 3))
    assert seen[0] > THRESHOLD and seen[1] > THRESHOLD and seen[2] < THRESHOLD, seen


def test_a_belief_earned_in_this_episode_survives_one_trusted_no():
    """Absence has to be earned. An object seen repeatedly a moment ago is more
    likely occluded than gone, and must not be deleted on one inconclusive look."""
    saturated = _Filter()
    saturated.log_odds = 3.0  # PresenceConfig.l_clamp_pos
    assert saturated.miss(recall=0.85, q=0.02) > THRESHOLD


# ------------------------------------- a move that does not move is a no-op


class _Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Rigid:
    """A STATIC habitat object: assignment to translation is silently dropped,
    and reading it back returns the OLD pose -- which is why this went unnoticed."""

    def __init__(self, sid, xyz, static=True):
        self.semantic_id = sid
        self._t = _Vec(*xyz)
        self.static = static
        self.motion_type = "STATIC"
        self.rotation = None

    @property
    def translation(self):
        return self._t

    @translation.setter
    def translation(self, value):
        if self.static and self.motion_type == "STATIC":
            return  # the silent drop
        self._t = _Vec(value.x, value.y, value.z) if hasattr(value, "x") else _Vec(*value)


def test_relocation_verifies_the_move_actually_took():
    """Every mid-episode relocation before this reported six objects moved while
    the render was byte-identical: a STATIC object ignores a new translation and
    reports back the old one, so nothing in the calling code could tell."""
    import types
    import numpy as np
    import osg.sim.ycb_env as ycb_env

    rigid = _Rigid(56, (0.0, 0.9, 0.0))
    authored = types.SimpleNamespace(semantic_id=56, translation=(5.0, 0.9, 5.0),
                                     rotation=(0.0, 0.0, 0.0, 1.0))
    layout = types.SimpleNamespace(objects=[authored])

    class _MN:
        Vector3 = staticmethod(lambda *a: _Vec(*a))
        Quaternion = staticmethod(lambda *a: None)

    class _MotionType:
        KINEMATIC = "KINEMATIC"

    fake_hs = types.SimpleNamespace(
        physics=types.SimpleNamespace(MotionType=_MotionType)
    )
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "magnum":
            return _MN
        if name == "habitat_sim":
            return fake_hs
        return real_import(name, *args, **kwargs)

    import builtins
    builtins.__import__ = fake_import
    try:
        moved = ycb_env.apply_layout_transforms([rigid], layout)
    finally:
        builtins.__import__ = real_import

    assert rigid.motion_type == "KINEMATIC", "the object was never made movable"
    assert moved == [56]
    assert (rigid.translation.x, rigid.translation.z) == (5.0, 5.0)



def test_by_default_an_unseen_target_is_always_abandoned():
    """The default threshold is 1.0, and the reason is what the alternative
    actually is. Not "keep believing and look again later" but "STOP here and
    end the episode": two cross-anchor episodes arrived at an empty spot,
    dropped the belief to 0.64, and -- being above a 0.45 threshold -- stopped
    and failed with 450 steps unspent. An approach that never saw its target has
    no reason to stop at it while steps remain."""
    from osg.core.config import VerificationConfig

    assert VerificationConfig().abandon_below_p == 1.0
    assert _Filter().miss(recall=0.5) < 1.0


def test_a_sweep_that_ever_expected_the_object_counts_as_having_looked():
    """Measured: judging by the frame the sweep ENDS on -- after a full circle,
    the arrival heading again -- meant all nine cross-anchor episodes stopped at
    29-58 steps with 440+ unspent, and the re-search never ran once. Expecting
    it at any heading of the sweep is what 'looked at it' means."""
    class _Agent:
        def __init__(self, scanned): self._scan_expected = scanned
        def blocked(self, expected_now):
            # mirrors NavAgent._absence_at_arrival's gate
            return self._scan_expected == 0 and not expected_now

    assert _Agent(0).blocked(expected_now=False) is True
    assert _Agent(3).blocked(expected_now=False) is False, "the sweep saw the place"
    assert _Agent(0).blocked(expected_now=True) is False


def test_the_arrival_sweep_turns_toward_the_object_not_blindly():
    """Captured from a real absence decision on a CORRECT map: the saved frame
    was a wall and a painting, with the target's table off to the right. A blind
    360-degree sweep ends on the heading it began with -- the navmesh follower's
    arrival heading -- so the decision was taken on whatever happened to be in
    front. The VLM answered 'bare' and was right about the pixels it was shown."""
    import numpy as np
    from osg.planning.controller import _wrap

    def turn_needed(agent_xy, obj_xy, heading):
        err = _wrap(float(np.arctan2(*(np.asarray(obj_xy) - np.asarray(agent_xy))[::-1])) - heading)
        return abs(err) > np.radians(15.0), err

    # object due east, agent facing north -> must turn, and turn the short way
    need, err = turn_needed((0, 0), (1, 0), np.radians(90))
    assert need and err < 0
    # already facing it -> no turn, decide on this frame
    need, _ = turn_needed((0, 0), (1, 0), np.radians(5))
    assert not need


# ------------------------------------------- the track-creation funnel (K+1)
#
# Nine failures of the last campaign named the target at its NEW pose in 4 to 36
# keyframes and finished with the only same-label tracks in the map being the
# ones loaded from the prior -- 0.00 m from a prior track, so no new track was
# created at all rather than one created in the wrong place. A detection can be
# lost in four places between the detector and the map, and the record named
# none of them.


def _det_with_mask(label, box, score=0.9, mask_box=None):
    import numpy as np

    from osg.core.types import Detection

    x1, y1, x2, y2 = box
    mask = np.zeros((480, 640), dtype=bool)
    mx1, my1, mx2, my2 = mask_box if mask_box else box
    mask[int(my1):int(my2), int(mx1):int(mx2)] = True
    return Detection(
        label=label, score=score,
        bbox_xyxy=np.array([float(x1), float(y1), float(x2), float(y2)]),
        mask=mask,
    )


def _frame_with_depth(depth):
    import numpy as np

    from osg.core.types import CameraIntrinsics, FrameData

    k = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)
    return FrameData(
        frame_id=1,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.full((480, 640), depth, dtype=np.float32),
        T_wc=np.eye(4),
        intrinsics=k,
    )


def test_the_funnel_counts_a_detection_all_the_way_to_a_track():
    from osg.objects.object_layer import ObjectLayer

    layer = ObjectLayer(min_det_score=0.3, min_det_bbox_px=1200.0)
    layer.update(_frame_with_depth(2.0), [_det_with_mask("bowl", (280, 200, 360, 280))])
    f = layer.funnel
    assert (f["det_seen"], f["det_admitted"], f["tracks_created"]) == (1, 1, 1)
    assert f["obs_rejected"] == 0 and f["ellipsoid_rejected"] == 0


def test_a_detection_below_the_size_gate_never_reaches_the_map():
    from osg.objects.object_layer import ObjectLayer

    layer = ObjectLayer(min_det_score=0.3, min_det_bbox_px=1200.0)
    layer.update(_frame_with_depth(2.0), [_det_with_mask("bowl", (310, 230, 330, 250))])
    f = layer.funnel
    assert f["det_seen"] == 1 and f["det_admitted"] == 0 and f["tracks_created"] == 0


def test_a_detection_with_no_readable_depth_is_counted_where_it_dies():
    """Depth of zero means the sensor returned nothing under the mask. The
    detection is admitted, and then quietly discarded -- which is exactly the
    case the funnel exists to make visible."""
    from osg.objects.object_layer import ObjectLayer

    layer = ObjectLayer(min_det_score=0.3, min_det_bbox_px=1200.0)
    layer.update(_frame_with_depth(0.0), [_det_with_mask("bowl", (280, 200, 360, 280))])
    f = layer.funnel
    assert f["det_admitted"] == 1
    assert f["obs_rejected"] == 1
    assert f["tracks_created"] == 0


def test_an_arrival_that_finds_nothing_is_recorded_on_the_track():
    """`absence_arrivals` is the event the search prior keys on, and it must be
    distinct from both of its neighbours: `identity_rejections` also counts
    unreachable verdicts and failed attempts, and `presence.p` also decays from
    ordinary missed expectations while merely walking past."""
    import types

    import numpy as np

    from osg.core.config import OSGConfig
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid
    from osg.objects.presence import PresenceFilter
    from osg.verification.absence import AbsenceSensor

    cfg = OSGConfig()
    cfg.verification.absence_requires_expectation = False
    cfg.verification.absence_use_vlm = False
    sensor = AbsenceSensor(cfg, verifier=None,
                           profiler=types.SimpleNamespace(timeit=lambda n: _null()),
                           stats={})
    track = ObjectTrack(id=1, label="bowl",
                        ellipsoid=Ellipsoid(center=np.array([1.0, 0.6, 2.0]),
                                            axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)))
    assert track.absence_arrivals == 0
    sensor.observe(track, "bowl", _frame_with_depth(2.0), PresenceFilter(),
                   scan_expected=1, reason="deadline")
    assert track.absence_arrivals == 1, "the agent went there and it was gone"


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# --------------------------- absence needs the agent to have GOT there (Q+2)
#
# In navmesh mode the follower returns None for arrived-or-unreachable alike, so
# an unreachable goal ends the approach exactly as an arrival does. Measured on
# 00848: the agent commits at step 1 to a track 0.38 m from the true object, is
# told the goal is unreachable while still 6.4 m away, asks the VLM about a
# handful of pixels at that range, gets "bare", and applies it at full strength.
# The correct track collapses 0.82 -> 0.36 and the agent never approaches the
# object again. All six pitcher episodes on that scene are byte-identical.


def _sensor(max_range, verifier=None):
    import types

    from osg.core.config import OSGConfig
    from osg.verification.absence import AbsenceSensor

    cfg = OSGConfig()
    cfg.verification.absence_max_range_m = max_range
    cfg.verification.absence_requires_expectation = False
    cfg.verification.absence_use_vlm = verifier is not None
    return AbsenceSensor(cfg, verifier=verifier,
                         profiler=types.SimpleNamespace(timeit=lambda n: _null()),
                         stats={})


def _track_at(z):
    import numpy as np

    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    return ObjectTrack(id=1, label="blue plastic pitcher",
                       ellipsoid=Ellipsoid(center=np.array([0.0, 0.0, float(z)]),
                                           axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)))


def test_a_reading_from_across_the_room_is_refused():
    from osg.objects.presence import PresenceFilter

    sensor = _sensor(3.0)
    track = _track_at(6.4)
    before = track.presence.p
    assert sensor.observe(track, "blue plastic pitcher", _frame_with_depth(6.4),
                          PresenceFilter(), scan_expected=1, reason="path_consumed") is None
    assert track.presence.p == before, "the belief must not move"
    assert track.absence_arrivals == 0, "and this is not an arrival"
    assert sensor.stats["absence_too_far"] == 1


def test_a_reading_at_arm_s_length_is_still_taken():
    from osg.objects.presence import PresenceFilter

    sensor = _sensor(3.0)
    track = _track_at(1.2)
    before = track.presence.p
    verdict = sensor.observe(track, "blue plastic pitcher", _frame_with_depth(1.2),
                             PresenceFilter(), scan_expected=1, reason="depth")
    assert verdict is not None and track.presence.p < before
    assert track.absence_arrivals == 1


def test_the_range_gate_is_off_by_default():
    """0.0 keeps the shipped behaviour, so the A/B has a control."""
    from osg.core.config import OSGConfig
    from osg.objects.presence import PresenceFilter

    assert OSGConfig().verification.absence_max_range_m == 0.0
    sensor = _sensor(0.0)
    track = _track_at(6.4)
    assert sensor.observe(track, "blue plastic pitcher", _frame_with_depth(6.4),
                          PresenceFilter(), scan_expected=1, reason="path_consumed") is not None


def test_the_gate_is_checked_before_the_vlm_is_asked():
    """A call about a handful of pixels is not worth making, and its answer is
    not worth having."""
    from osg.objects.presence import PresenceFilter

    class _CountingVerifier:
        def __init__(self): self.calls = 0
        def verify_still_there(self, *a, **k):
            self.calls += 1
            return False

    v = _CountingVerifier()
    sensor = _sensor(3.0, verifier=v)
    sensor.observe(_track_at(6.4), "blue plastic pitcher", _frame_with_depth(6.4),
                   PresenceFilter(), scan_expected=1, reason="path_consumed")
    assert v.calls == 0
