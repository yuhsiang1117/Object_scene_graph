"""The floor gate's "never leave a floor that has the target on it", conditioned.

`floor_target_evidence` gives a mapped instance of the target category a bonus
of 10, and `FloorSwitchPolicy.may_switch` treats anything at or above that as
absolute: return False, stay here. That is right for a map the agent built this
episode and wrong for a stale prior map, where the instance carrying the bonus
is exactly the object that has since been moved.

`presence_of` gates the bonus alone. The context categories are furniture and
stay ungated on purpose: they do not move, and their presence decays from
ordinary missed expectations while the agent merely walks past -- the defect
that made a first attempt at proximity-dropping fire on 56% of in_anchor
episodes against the 30% predicted.
"""
from __future__ import annotations

from types import SimpleNamespace

from osg.graph.priors import floor_target_evidence
from osg.mapping.portals import FloorSwitchPolicy

TARGET_PRESENT = 10


def _obj(track_id, label, floor):
    return SimpleNamespace(track_id=track_id, label=label, floor_id=floor)


def _graph(objects, rooms=None):
    return SimpleNamespace(objects=objects, rooms=rooms or {})


def test_a_believed_instance_still_carries_the_bonus():
    sg = _graph([_obj(1, "toilet", 0)])
    evidence, _ = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: True)
    assert evidence >= TARGET_PRESENT


def test_a_disbelieved_instance_no_longer_pins_the_agent():
    """The stale-map case. The toilet track is on this floor and the map is
    wrong about it; the gate must stop treating that as decisive."""
    sg = _graph([_obj(1, "toilet", 0)])
    evidence, _ = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: False)
    assert evidence < TARGET_PRESENT


def test_the_default_is_unchanged_term_for_term():
    """Left None the arithmetic is exactly what it was."""
    sg = _graph([_obj(1, "toilet", 0)])
    assert floor_target_evidence(sg, 0, "toilet") == floor_target_evidence(
        sg, 0, "toilet", presence_of=None
    )


def test_context_furniture_is_never_gated():
    """A sink does not move. Gating it would let ordinary missed expectations
    erase the room-type signal the storey decision rests on."""
    sg = _graph([_obj(1, "sink", 0), _obj(2, "bathtub", 0)])
    gated, _ = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: False)
    plain, _ = floor_target_evidence(sg, 0, "toilet")
    assert gated == plain > 0


def test_n_objects_counts_everything_regardless_of_belief():
    """`n_objects` answers "has this floor been looked at", which is about the
    mapping effort, not about any one object still being there."""
    sg = _graph([_obj(1, "toilet", 0), _obj(2, "sink", 0)])
    _, n = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: False)
    assert n == 2


def test_the_gate_reopens_once_the_instance_is_disbelieved():
    """End to end through the policy: the same floor, the same map, and the
    only difference is whether presence still believes the mapped toilet."""
    # `use_target_evidence` defaults False on the class and True in
    # `core/config/floor.py`; the shipped configuration is the one under test.
    policy = FloorSwitchPolicy(max_steps=500, no_switch_before=0, min_interval_steps=0,
                               use_target_evidence=True)
    sg = _graph([_obj(1, "toilet", 0)] + [_obj(i, "chair", 0) for i in range(2, 12)])

    believed, n = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: True)
    stale, _ = floor_target_evidence(sg, 0, "toilet", presence_of=lambda tid: False)

    assert not policy.may_switch(200, None, evidence=believed, n_objects=n,
                                 steps_on_floor=200)
    assert policy.may_switch(200, None, evidence=stale, n_objects=n,
                             steps_on_floor=200)


def test_a_floor_just_reached_is_not_left_for_having_no_map_yet():
    """Measured on 00808's yellow bottle under v13: a committed descent to the
    object's floor at step 314, the geometric clause satisfied at 347 because
    the new floor had almost no map, back upstairs by 418."""
    from osg.mapping.portals import FloorSwitchPolicy

    plain = FloorSwitchPolicy(max_steps=500, no_switch_before=50, min_interval_steps=0)
    dwell = FloorSwitchPolicy(max_steps=500, no_switch_before=50, min_interval_steps=0,
                              dwell_on_arrival=True)
    # 33 steps onto a new floor, nothing selectable yet (inside the late cutoff).
    assert plain.may_switch(300, best_path_cost=None, steps_on_floor=33)
    assert not dwell.may_switch(300, best_path_cost=None, steps_on_floor=33)
    # Having looked for the same dwell the episode start gets, it may leave.
    assert dwell.may_switch(340, best_path_cost=None, steps_on_floor=86)


def test_steps_on_a_floor_count_from_arrival_not_from_the_prior_map():
    """A storey restored from the prior map has first_step 0. Under v14 the
    agent reached the object's floor at step 314 and its dwell guard read 347
    steps on the floor at step 347, so it left at once, exactly as v13 had."""
    from osg.mapping.floor_stack import FloorStack

    stack = FloorStack(resolution_m=0.05)
    stack.layer(0, step=0); stack.layer(1, step=0)   # both "from the map"
    stack.current_id = 1
    stack.set_current(0, step=314)
    assert stack.current.first_step == 0
    assert stack.current.arrived_step == 314
