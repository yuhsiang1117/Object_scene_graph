"""`navigation/search/storey.py`: the storey belief that runs on ASCENT's maps."""
import pytest

from navigation.search.storey import (FloorSearchBelief, Storey, TraceStoreys, floor_table,
                                      unexplored_beyond)


def belief(**over):
    kw = dict(switch_ratio=1.8, detect_prob=0.85, prior_weight=0.8, frontier_area_m2=8.0)
    kw.update(over)
    return FloorSearchBelief(**kw)


def test_coverage_is_zero_unvisited_one_exhausted_and_a_ratio_between():
    b = belief()
    assert b.coverage(Storey()) == 0.0
    assert b.coverage(Storey(visited=True, explored_m2=40.0, live_frontiers=5, exhausted=True)) == 1.0
    assert b.coverage(Storey(visited=True, explored_m2=40.0, live_frontiers=5)) == pytest.approx(0.5)
    assert b.coverage(Storey(visited=True, explored_m2=0.0, live_frontiers=0)) == 0.0


def test_prior_mixes_the_table_with_uniform_and_falls_back_when_empty():
    b = belief()
    assert b.prior({1: 25.0, 2: 75.0}, 2) == pytest.approx([0.8 * 0.25 + 0.1, 0.8 * 0.75 + 0.1])
    # A zero in the table (4-storey beds: 0, 0, 50, 50) is thin statistics, not
    # an impossibility: the storey keeps the uniform share.
    assert min(b.prior({1: 0.0, 2: 0.0, 3: 50.0, 4: 50.0}, 4)) == pytest.approx(0.05)
    assert b.prior({}, 3) == pytest.approx([1 / 3] * 3)
    assert b.prior(None, 2) == pytest.approx([0.5, 0.5])
    assert b.stats["storey_prior_empty"] == 2
    # The table can be shorter than the building (`get_floor_probabilities`
    # caps at the tallest building it has statistics for).
    assert len(b.prior({1: 40.0, 2: 60.0}, 3)) == 3


def test_a_single_storey_never_asks():
    b = belief()
    here = Storey(visited=True, explored_m2=100.0, exhausted=True)
    assert b.decide(10, 0, [here], None, can_up=True, can_down=True) is None


def test_asks_for_the_other_storey_only_once_this_one_is_searched_enough():
    b = belief()
    up = Storey()
    table = {1: 50.0, 2: 50.0}
    # Equal priors: P_up / P_here = 1 / (1 - 0.85 c) reaches 1.8 at c = 0.523.
    early = Storey(visited=True, explored_m2=20.0, live_frontiers=5)      # c = 0.33
    assert b.decide(50, 0, [early, up], table, can_up=True, can_down=False) is None
    late = Storey(visited=True, explored_m2=60.0, live_frontiers=5)       # c = 0.6
    assert b.decide(150, 0, [late, up], table, can_up=True, can_down=False) == "up"


def test_a_prior_toward_the_other_storey_moves_the_switch_earlier():
    table = {1: 20.0, 2: 80.0}
    here = Storey(visited=True, explored_m2=20.0, live_frontiers=5)       # c = 0.33
    assert belief().decide(50, 0, [here, Storey()], table, can_up=True, can_down=False) == "up"
    assert belief().decide(50, 1, [Storey(), here], table, can_up=False, can_down=True) is None


def test_no_request_toward_a_side_the_caller_rules_out():
    b = belief()
    here = Storey(visited=True, explored_m2=80.0, live_frontiers=1)
    storeys = [Storey(), here, Storey()]
    assert b.decide(1, 1, storeys, None, can_up=False, can_down=False) is None
    assert b.decide(2, 1, storeys, None, can_up=False, can_down=True) == "down"
    assert b.decide(3, 1, storeys, None, can_up=True, can_down=False) == "up"


def test_request_is_held_until_this_storey_overtakes_and_dropped_on_arrival():
    b = belief()
    table = {1: 50.0, 2: 50.0}
    up = Storey()
    assert b.decide(1, 0, [Storey(visited=True, explored_m2=60.0, live_frontiers=5), up], table,
                    can_up=True, can_down=False) == "up"
    # New frontiers open on the way to the stairs: the ratio falls below 1.8
    # but the other storey still leads, so the request stands.
    more = Storey(visited=True, explored_m2=60.0, live_frontiers=9)       # c = 0.45, ratio 1.63
    assert b.decide(2, 0, [more, up], table, can_up=True, can_down=False) == "up"
    # A fresh belief at the same state would not have asked.
    assert belief().decide(2, 0, [more, up], table, can_up=True, can_down=False) is None
    # On the new storey nothing is held: it starts unsearched and keeps the agent.
    arrived = Storey(visited=True, explored_m2=3.0, live_frontiers=2)
    assert b.decide(3, 1, [more, arrived], table, can_up=False, can_down=True) is None


def test_log_records_changes_only_and_reset_clears_it():
    b = belief()
    here = Storey(visited=True, explored_m2=20.0, live_frontiers=5)
    for step in range(5):
        b.decide(step, 0, [here, Storey()], None, can_up=True, can_down=False)
    assert len(b.log) == 1 and b.log[0]["request"] is None
    assert b.stats["storey_belief_rounds"] == 5
    b.reset()
    assert b.log == []


def test_hold_follows_the_storey_not_its_index():
    b = belief()
    table = {1: 50.0, 2: 50.0}
    here = Storey(visited=True, explored_m2=60.0, live_frontiers=5)
    assert b.decide(1, 0, [here, Storey()], table, can_up=True, can_down=False, here_id=7) == "up"
    # A storey is inserted below: same storey, index 1 now. The request stands.
    more = Storey(visited=True, explored_m2=60.0, live_frontiers=9)
    three = {1: 0.0, 2: 50.0, 3: 50.0}
    assert b.decide(2, 1, [Storey(), more, Storey()], three, can_up=True, can_down=False,
                    here_id=7) == "up"


def test_floor_table_reads_what_the_planner_reads_under_the_hm3d_name():
    from navigation.planner import AscentLLMPlanner
    prior = {"bed": {"2": {"1": 41.0, "2": 59.0},
                     "3": {"1": 22.2, "2": 29.6, "3": 48.1}},
             "tv_monitor": {"2": {"1": 70.0, "2": 30.0}}}
    planner = AscentLLMPlanner.__new__(AscentLLMPlanner)
    planner.floor_prior = prior
    for n in (2, 3, 5):
        assert floor_table(prior, "bed", n) == planner.get_floor_probabilities("bed", n), n
    # No column for a one-storey building: the planner prints 0.0%, which is
    # the same "no information" the belief reads from an empty table.
    assert planner.get_floor_probabilities("bed", 1) == {1: 0.0} and floor_table(prior, "bed", 1) == {}
    assert floor_table(prior, "bed", 5) == {1: 22.2, 2: 29.6, 3: 48.1}      # the tallest column
    assert floor_table(prior, "tv_monitor", 2) == {1: 70.0, 2: 30.0}
    assert floor_table(prior, "TV Monitor", 2) == {1: 70.0, 2: 30.0}
    assert floor_table(prior, "sofa", 2) == {} and floor_table(None, "bed", 2) == {}


def _row(floor, uid, n, area=10.0, live=3, done=0):
    return {"floor": floor, "floor_uid": uid, "n_floors": n, "explored_m2": area,
            "nfront_live": live, "floor_done": done}


def test_trace_storeys_follow_appends_inserts_and_moves():
    ts = TraceStoreys()
    here, storeys = ts.update(_row(0, 0, 1))
    assert here == 0 and len(storeys) == 1 and storeys[0].visited
    here, storeys = ts.update(_row(0, 0, 2, area=20.0))            # a storey appended above
    assert [s.visited for s in storeys] == [True, False]
    here, storeys = ts.update(_row(1, 0, 3, area=30.0, live=1))    # one inserted below
    assert here == 1 and [s.visited for s in storeys] == [False, True, False]
    assert storeys[1].explored_m2 == 30.0 and storeys[1].live_frontiers == 1
    here, storeys = ts.update(_row(2, 2, 3, area=4.0))             # climbed: a new uid on top
    assert here == 2 and [s.visited for s in storeys] == [False, True, True]
    assert storeys[1].explored_m2 == 30.0                          # the storey left keeps its numbers
    here, storeys = ts.update(_row(1, 0, 3, area=31.0, done=1))    # and back down
    assert here == 1 and storeys[1].exhausted and ts.uids == [None, 0, 2]


def test_unexplored_beyond_is_the_stair_navigation_condition():
    done = Storey(visited=True, exhausted=True)
    assert unexplored_beyond([Storey(), Storey(visited=True), done], 1, "down")
    assert not unexplored_beyond([Storey(), Storey(visited=True), done], 1, "up")
    assert not unexplored_beyond([Storey(visited=True)], 0, "up")
