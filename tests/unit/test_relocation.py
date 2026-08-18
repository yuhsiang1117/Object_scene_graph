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
