"""Target-category priors for the floor-switch decision (osg.graph.priors)."""
import pytest

from osg.graph.priors import context_categories, floor_target_evidence
from osg.mapping.portals import FloorSwitchPolicy


class _Obj:
    def __init__(self, label, floor_id=0):
        self.label, self.floor_id = label, floor_id


class _Room:
    def __init__(self, label, floor_id=0):
        self.label, self.floor_id = label, floor_id


class _SG:
    def __init__(self, objects, rooms=()):
        self.objects = objects
        self.rooms = {i: r for i, r in enumerate(rooms)}


# ------------------------------------------------------------------ evidence


def test_bathroom_context_is_evidence_for_a_toilet():
    sg = _SG([_Obj("sink"), _Obj("bathtub"), _Obj("towel"), _Obj("chair")])
    ev, n = floor_target_evidence(sg, 0, "toilet")
    assert ev == 3 and n == 4


def test_no_context_is_zero_evidence():
    sg = _SG([_Obj("chair"), _Obj("table"), _Obj("desk")])
    ev, _ = floor_target_evidence(sg, 0, "toilet")
    assert ev == 0


def test_evidence_is_per_floor():
    sg = _SG([_Obj("sink", 1), _Obj("bathtub", 1), _Obj("chair", 0)])
    assert floor_target_evidence(sg, 0, "toilet")[0] == 0
    assert floor_target_evidence(sg, 1, "toilet")[0] == 2


def test_the_target_itself_dominates():
    sg = _SG([_Obj("toilet")])
    ev, _ = floor_target_evidence(sg, 0, "toilet")
    assert ev >= 10


def test_duplicate_context_objects_count_once():
    """Three sinks are one kind of evidence, not three."""
    sg = _SG([_Obj("sink"), _Obj("sink"), _Obj("sink")])
    assert floor_target_evidence(sg, 0, "toilet")[0] == 1


def test_a_named_room_counts_as_evidence():
    sg = _SG([_Obj("chair")], rooms=[_Room("bathroom")])
    assert floor_target_evidence(sg, 0, "toilet")[0] == 1


def test_categories_without_a_usable_prior_return_none():
    """A plant appears in every room type, so its absence means nothing --
    better to say "unknown" than to invent a signal."""
    sg = _SG([_Obj("chair"), _Obj("table")])
    ev, n = floor_target_evidence(sg, 0, "plant")
    assert ev is None and n == 2
    assert context_categories("plant") == set()


def test_underscored_and_spaced_labels_match():
    sg = _SG([_Obj("tv monitor"), _Obj("cushion")])
    assert floor_target_evidence(sg, 0, "sofa")[0] == 2


# ---------------------------------------------------------------- the gate


def policy(**kw):
    kw.setdefault("use_target_evidence", True)
    return FloorSwitchPolicy(max_steps=500, **kw)


def test_absent_context_switches_early():
    """The point of the change: the geometric rule cannot fire until the floor
    is exhausted (~step 200 measured); zero evidence fires at 40."""
    p = policy()
    assert p.may_switch(step=40, best_path_cost=1.0, evidence=0, n_objects=12)


def test_geometry_alone_would_not_have_switched_there():
    p = policy()
    assert not p.may_switch(step=40, best_path_cost=1.0, evidence=None, n_objects=12)


def test_too_few_objects_means_not_looked_yet():
    """Zero evidence on an unmapped floor is ignorance, not absence."""
    p = policy()
    assert not p.may_switch(step=40, best_path_cost=1.0, evidence=0, n_objects=2)


def test_strong_evidence_holds_the_agent_on_this_floor():
    """Even with nothing near left, do not abandon the most promising storey."""
    p = policy()
    assert not p.may_switch(step=200, best_path_cost=None, evidence=3, n_objects=20)


def test_weak_evidence_still_allows_the_geometric_switch():
    p = policy()
    assert p.may_switch(step=200, best_path_cost=None, evidence=1, n_objects=20)


def test_evidence_never_overrides_the_interval_guard():
    p = policy()
    p.note_switch(100)
    assert not p.may_switch(step=120, best_path_cost=1.0, evidence=0, n_objects=20)


def test_evidence_never_overrides_the_late_cutoff():
    p = policy()
    assert not p.may_switch(step=400, best_path_cost=1.0, evidence=0, n_objects=20)


def test_flag_off_restores_the_pure_geometric_gate():
    p = policy(use_target_evidence=False)
    assert not p.may_switch(step=40, best_path_cost=1.0, evidence=0, n_objects=20)
    assert p.may_switch(step=200, best_path_cost=None, evidence=5, n_objects=20)


# ------------------------------------------- P2: engagement regression fixes


def test_one_incidental_companion_no_longer_blocks_an_early_switch():
    """The regression: requiring evidence == 0 was far too strict. Almost any
    floor has a single incidental companion object, so cross-floor switch
    attempts collapsed to 3 of 24 episodes where the looser rule gave 14."""
    p = policy()
    assert p.may_switch(step=60, best_path_cost=1.0, evidence=1, n_objects=15)


def test_strong_context_still_holds_the_agent_at_first():
    p = policy()
    assert not p.may_switch(step=60, best_path_cost=None, evidence=3,
                            n_objects=15, steps_on_floor=20)


def test_strong_context_goes_stale_after_a_long_fruitless_search():
    """A bathroom on this storey does not mean THIS storey's bathroom has the
    toilet -- without expiry the 'stay' rule suppressed switching entirely."""
    p = policy()
    assert p.may_switch(step=200, best_path_cost=None, evidence=3,
                        n_objects=15, steps_on_floor=200)


def test_the_target_itself_is_never_abandoned():
    """Evidence carries a +10 bonus when the target CATEGORY is mapped here;
    no amount of patience should make the agent leave that floor."""
    p = policy()
    assert not p.may_switch(step=300, best_path_cost=None, evidence=11,
                            n_objects=30, steps_on_floor=300)


def test_patience_can_be_disabled():
    p = policy(evidence_patience_steps=0)
    assert not p.may_switch(step=300, best_path_cost=None, evidence=3,
                            n_objects=15, steps_on_floor=300)


def test_an_unmapped_floor_still_gets_no_early_switch():
    p = policy()
    assert not p.may_switch(step=60, best_path_cost=1.0, evidence=0, n_objects=2)


# ------------------------------------------------------- grounded affinity


def test_a_complete_map_grounds_to_exactly_the_ungrounded_ranking():
    """00829 and 00880 contain all 14 container categories, so grounding cannot
    change them. This is what makes the A/B surgical: only 00848, the scene with
    no counter and no table, can move."""
    from osg.graph.containers import CONTAINER_CATEGORIES
    from osg.graph.priors import affinity_scores

    every = set(CONTAINER_CATEGORIES)
    for target in ("bowl", "cracker box", "red plate", "tin can"):
        assert affinity_scores(target, present=every) == affinity_scores(target)


def test_a_table_entry_is_filtered_to_the_surfaces_that_exist():
    """`bowl` ranks table > counter > desk > cabinet > shelf > sink. In a house
    with neither a table nor a counter the entry must not keep asserting them --
    and desk, not table, is then the best place to look."""
    from osg.graph.priors import affinity_scores

    present = {"desk", "cabinet", "shelf", "bed", "refrigerator"}
    scores = affinity_scores("bowl", present=present)
    # table and counter are buildable-but-absent, so they go; `sink` is not a
    # CONTAINER_CATEGORY at all, scores nothing either way, and stays so that
    # the surviving weights are not rescaled out from under a complete scene.
    assert "table" not in scores and "counter" not in scores
    assert max(scores, key=scores.get) == "desk"
    assert scores["desk"] == 1.0 and scores["cabinet"] < 1.0


def test_an_entry_with_nothing_left_defers_to_the_grounded_source():
    """The one case a model may speak over the static table: the table's options
    do not exist here, so it has abstained rather than been overruled."""
    from osg.graph.priors import affinity_scores

    called = {}

    def source(key):
        called["key"] = key
        return ["cabinet", "shelf"]

    # `pillow` ranks bed > sofa > bench; none of them are in this map.
    scores = affinity_scores("pillow", source=source, present={"cabinet", "shelf"})
    assert called["key"] == "pillow"
    assert max(scores, key=scores.get) == "cabinet"


def test_grounding_the_provider_scopes_its_cache_key():
    """A grounded answer must never overwrite the ungrounded one, and a complete
    scene must keep using the entry it already had -- byte for byte."""
    from osg.llm.affinity import AffinityProvider

    every = ["bed", "cabinet", "counter", "desk", "shelf", "table"]
    p = AffinityProvider(client=None, surfaces=every)
    p._cache = {"pitcher": ["counter", "table"], "pitcher@bed,cabinet,desk,shelf": ["cabinet"]}

    assert p("pitcher") == ["counter", "table"]
    p.ground(set(every))            # complete map -> no scoping at all
    assert p._scope == ""
    assert p("pitcher") == ["counter", "table"]
    p.ground({"bed", "cabinet", "desk", "shelf"})
    assert p("pitcher") == ["cabinet"]
