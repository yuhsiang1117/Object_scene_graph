"""The search posterior: turning "not here" into "then look there" (C3).

DualMap answers a failed candidate with the next-highest similarity and an
ignore list it throws away at query end -- no model of where the object went, no
cost of getting there, no memory of how well a place was already searched. These
tests pin the three things that replaces.

Habitat-free: priors are arithmetic and the selector takes a planner protocol.
"""
import numpy as np
import pytest

from osg.exploration.search_belief import (
    InspectionLog,
    SearchCandidate,
    build_container_candidates,
    container_prior,
    select_candidate,
)
from osg.graph.priors import affinity_scores, affords


class _Result:
    def __init__(self, success, cost):
        self.success, self.cost = success, cost


class _Planner:
    """Path cost = Euclidean distance, except where told otherwise."""

    def __init__(self, blocked=(), detour=None):
        self.blocked = set(blocked)
        self.detour = detour or {}
        self.calls = 0

    def plan(self, costmap, start, goal):
        self.calls += 1
        key = (round(float(goal[0]), 3), round(float(goal[1]), 3))
        if key in self.blocked:
            return _Result(False, 0.0)
        if key in self.detour:
            return _Result(True, self.detour[key])
        return _Result(True, float(np.linalg.norm(np.asarray(goal, float) - np.asarray(start, float))))


class _Node:
    def __init__(self, cid, label, center, top_h=0.75, area=0.6):
        self.id, self.label = cid, label
        self.center = np.asarray(center, float)
        self.top_h, self.area_m2 = top_h, area


class _Graph:
    def __init__(self, nodes):
        self.containers = {n.id: n for n in nodes}


# ------------------------------------------------------------------- priors


def test_a_surface_that_cannot_hold_the_class_is_not_a_worse_place_but_no_place():
    """Binary on purpose: a shelf at head height is not a slightly worse place
    to look for a bowl."""
    assert affords("bowl", 0.75, 0.6) == 1.0
    assert affords("bowl", 1.9, 0.6) == 0.0
    assert container_prior("bowl", "shelf", 1.9, 0.6, np.zeros(2)) == 0.0


def test_affinity_ranks_plausible_surfaces_above_implausible_ones():
    table = container_prior("bowl", "table", 0.75, 0.6, np.zeros(2))
    sink = container_prior("bowl", "sink", 0.75, 0.6, np.zeros(2))
    assert table > sink > 0.0


def test_an_unknown_surface_category_is_not_evidence_against():
    """No prior is not a negative prior -- a place we have never thought about
    must stay searchable."""
    assert container_prior("bowl", "sideboard", 0.75, 0.6, np.zeros(2)) > 0.0


def test_proximity_prefers_surfaces_near_where_the_object_was():
    """Objects are moved by someone doing a task, so short displacements
    dominate. Distance from the AGENT is the cost term, not this."""
    near = container_prior("bowl", "table", 0.75, 0.6, np.array([1.0, 0.0]),
                           last_known_xy=np.zeros(2))
    far = container_prior("bowl", "table", 0.75, 0.6, np.array([9.0, 0.0]),
                          last_known_xy=np.zeros(2))
    assert near > far > 0.0


def test_the_static_table_wins_over_a_model():
    """A model must not quietly rewrite a prior someone chose deliberately."""
    called = []

    def source(target):
        called.append(target)
        return ["sink"]

    assert affinity_scores("bowl", source=source)["table"] == 1.0
    assert called == []
    assert affinity_scores("kettle", source=source) == {"sink": 1.0}
    assert called == ["kettle"]


# ---------------------------------------------------------------- the index


def test_selection_is_belief_times_detection_over_cost_not_belief_alone():
    """The whole point of the index: a slightly better place far away loses to
    a decent one nearby."""
    far_good = SearchCandidate("container", 1, np.array([10.0, 0.0]), prior=1.0, detect_prob=0.8)
    near_ok = SearchCandidate("container", 2, np.array([1.0, 0.0]), prior=0.6, detect_prob=0.8)
    best = select_candidate([far_good, near_ok], _Planner(), None, np.zeros(2))
    assert best.ref_id == 2


def test_an_unreachable_candidate_is_reported_not_chosen():
    reachable = SearchCandidate("container", 1, np.array([3.0, 0.0]), prior=0.5, detect_prob=0.8)
    unreachable = SearchCandidate("container", 2, np.array([1.0, 0.0]), prior=1.0, detect_prob=0.8)
    failed = set()
    best = select_candidate([reachable, unreachable], _Planner(blocked=[(1.0, 0.0)]),
                            None, np.zeros(2), failed_out=failed)
    assert best.ref_id == 1
    assert ("container", 2) in failed


def test_cost_is_geodesic_not_straight_line():
    """A surface on the other side of a wall is not nearby, and ranking on
    Euclidean distance would defeat the point of paying for a planner."""
    through_wall = SearchCandidate("container", 1, np.array([1.0, 0.0]), prior=1.0, detect_prob=0.8)
    around = SearchCandidate("container", 2, np.array([3.0, 0.0]), prior=1.0, detect_prob=0.8)
    planner = _Planner(detour={(1.0, 0.0): 20.0})
    assert select_candidate([through_wall, around], planner, None, np.zeros(2)).ref_id == 2


def test_zero_prior_candidates_never_cost_a_plan():
    planner = _Planner()
    select_candidate(
        [SearchCandidate("container", 1, np.array([1.0, 0.0]), prior=0.0, detect_prob=0.8)],
        planner, None, np.zeros(2),
    )
    assert planner.calls == 0


# ------------------------------------------------------- searching a place


def test_a_look_multiplies_belief_rather_than_zeroing_it():
    """DualMap's ignore list is b <- 0 after one look, discarded at query end.
    A place glanced at from four metres must stay plausible."""
    log = InspectionLog()
    assert log.factor(7) == 1.0
    assert log.searched(7, 0.8) == pytest.approx(0.2)
    assert log.searched(7, 0.8) == pytest.approx(0.04)


def test_a_searched_surface_loses_to_an_unsearched_one():
    graph = _Graph([_Node(1, "table", [0.5, 0.8, 0.0]), _Node(2, "table", [0.6, 0.8, 0.0])])
    log = InspectionLog()
    before = {c.ref_id: c.prior for c in build_container_candidates(graph, "bowl", log)}
    log.searched(1, 0.8)
    after = {c.ref_id: c.prior for c in build_container_candidates(graph, "bowl", log)}
    assert after[1] < before[1]
    assert after[2] == pytest.approx(before[2])


def test_surfaces_that_cannot_hold_the_target_are_not_offered_at_all():
    graph = _Graph([_Node(1, "shelf", [0.0, 1.9, 0.0], top_h=1.9),
                    _Node(2, "table", [1.0, 0.8, 0.0])])
    ids = {c.ref_id for c in build_container_candidates(graph, "bowl", InspectionLog())}
    assert ids == {2}


def test_the_far_field_stays_ordered_by_distance():
    """The proximity term used to be clipped at a 0.2 floor, so that an object
    could not be called impossible for having moved 7 m. The intent was right,
    the mechanism was not: `max(exp(-d/L), floor)` gives every candidate past
    L*ln(1/floor) the SAME prior, so the whole far field ties and its order
    falls to whatever `sorted` does with equal keys -- track id. That is exactly
    the regime a cross-anchor relocation lives in, and it discarded the only
    signal left. Measured over 114 relocations, unclipping took the true surface
    into the top 5 in 7 of 57 cross-anchor cases against 1 of 57 clipped."""
    far = [container_prior("bowl", "table", 0.75, 0.6, np.array([d, 0.0]),
                           last_known_xy=np.zeros(2), proximity_len_m=1.0)
           for d in (6.0, 8.0, 10.0, 20.0)]
    assert far == sorted(far, reverse=True)
    assert all(a > b for a, b in zip(far, far[1:]))
    assert far[-1] > 0.0  # weak evidence is still evidence; never a hard zero


def test_a_floor_is_still_available_when_one_is_asked_for():
    """Unclipped is the default, not a prohibition: a genuine mixture is a
    reasonable thing to want, and the parameter still expresses one."""
    floored = container_prior("bowl", "table", 0.75, 0.6, np.array([20.0, 0.0]),
                              last_known_xy=np.zeros(2), proximity_len_m=4.0,
                              proximity_floor=0.2)
    assert floored == pytest.approx(1.0 * 0.2)


def test_nearer_the_last_known_pose_outranks_further_from_it():
    near = container_prior("bowl", "table", 0.75, 0.6, np.array([1.0, 0.0]),
                           last_known_xy=np.zeros(2), proximity_len_m=1.0)
    far = container_prior("bowl", "table", 0.75, 0.6, np.array([8.0, 0.0]),
                          last_known_xy=np.zeros(2), proximity_len_m=1.0)
    assert near > far


def test_absence_from_a_six_entry_list_is_not_evidence_against():
    """An unlisted category used to score 0.25 -- BELOW the lowest listed entry,
    i.e. positive evidence against -- justified by one anecdote about a bed and
    a sofa tying with a sink as places to look for a bowl.

    The benchmark's own relocations say the opposite. Of 53 whose destination
    surface is mapped, 37 land on a category CONTAINER_AFFINITY does not list
    for that class, and a bed is the single most common destination of all (26).
    Ranking the true surface over 114 relocations, affinity alone scores top-1
    0/114 and median rank 27, against 25 for shuffling the candidates. A
    six-entry hand-written list is not exhaustive and absence from it carries no
    information, so an unlisted category sits level with the lowest listed one
    rather than beneath it.

    Ties here are harmless: proximity is unclipped and decides the product.
    """
    listed_worst = container_prior("bowl", "sink", 0.75, 0.6, np.zeros(2))
    unlisted = container_prior("bowl", "bed", 0.75, 0.6, np.zeros(2))
    assert unlisted == pytest.approx(listed_worst)

    listed_best = container_prior("bowl", "table", 0.75, 0.6, np.zeros(2))
    assert listed_best > unlisted, "the ranking among listed categories survives"


def test_affinity_is_softened_to_a_tie_breaker_not_a_veto():
    """Proximity ranks the true surface at median 2; affinity alone at 27
    against 25 for arbitrary order. Affinity should separate near-equals, not
    overrule a metre of distance -- so its spread is compressed, and a nearer
    surface of a worse category still outranks a far one of a better."""
    near_bad = container_prior("bowl", "bed", 0.75, 0.6, np.array([0.5, 0.0]),
                               last_known_xy=np.zeros(2), proximity_len_m=1.0)
    far_good = container_prior("bowl", "table", 0.75, 0.6, np.array([4.0, 0.0]),
                               last_known_xy=np.zeros(2), proximity_len_m=1.0)
    assert near_bad > far_good


def test_with_no_ranking_at_all_every_surface_stays_equally_plausible():
    """Having no prior is different from having one that excludes you."""
    a = container_prior("zorb", "bed", 0.75, 0.6, np.zeros(2))
    b = container_prior("zorb", "table", 0.75, 0.6, np.zeros(2))
    assert a == b > 0.0


# ------------------------------------------------- where the approach ends

def test_a_depth_stop_lands_between_the_viewpoint_rings():
    """The arithmetic behind agent.approach_to_viewpoint. HM3D scores success
    against the nearest sampled goal viewpoint, and those sit on rings at fixed
    radii. Stopping when the target's depth reaches 1.0 m puts the agent
    radially between the 0.8 m and 1.2 m rings -- 0.2 m from either, against a
    0.18 m success radius. Measured: four of seven batch episodes ended at 0.18,
    0.19, 0.21 and 0.28 m having FOUND the object."""
    rings = [0.8, 1.2, 1.5, 2.0]
    depth_stop = 1.0
    radial_miss = min(abs(depth_stop - r) for r in rings)
    assert radial_miss > 0.18, "a depth stop at 1.0 m would be inside the success radius"

    # Standing ON a ring leaves only angular error: at 24 samples the spacing on
    # the innermost ring is 0.21 m, so the worst case is half of that.
    import numpy as np
    angular_miss = 0.5 * (2 * np.pi * rings[0] / 24)
    assert angular_miss < 0.18, "even on a ring the sampling would be too coarse"


def test_the_planner_samples_the_same_rings_the_benchmark_does():
    """The fix only works because the agent's own viewpoint rings coincide with
    the ones the episode manifest sampled its goal viewpoints on."""
    from osg.core.config import VerificationConfig, YCBAuthoredConfig
    from osg.verification.viewpoint import ViewpointPlanner

    assert list(VerificationConfig().ring_radii_m) == list(YCBAuthoredConfig().viewpoint_radii_m)
    assert ViewpointPlanner(list(VerificationConfig().ring_radii_m)).ring_radii == [0.8, 1.2, 1.5, 2.0]


def test_proximity_is_dropped_once_the_object_is_known_to_have_moved():
    """Proximity encodes "displacements are usually short". Going to the old
    place and finding nothing refutes the premise -- and the surfaces the term
    favours are the ones just ruled out. Measured over nine cross-anchor
    episodes: keeping it ranks the true destination 32nd of 112 on median and
    in the top 8 in 0 of 9; dropping it gives 20 and 3 of 9."""
    ghost = np.zeros(2)
    far = np.array([9.0, 0.0])
    with_prox = container_prior("bowl", "table", 0.75, 0.6, far, last_known_xy=ghost)
    without = container_prior("bowl", "table", 0.75, 0.6, far, last_known_xy=None)
    assert without > with_prox
    near = container_prior("bowl", "table", 0.75, 0.6, np.array([0.5, 0.0]), last_known_xy=None)
    assert near == pytest.approx(without), "with no last-known pose, distance stops mattering"


def test_repeatedly_reselecting_one_surface_is_what_an_unreachable_goal_looks_like():
    """The signature of the bug this guards: a surface whose goal cannot be
    stood in is never 'arrived at', so it earns only the quarter credit for an
    unreached look and stays top of the list. One real episode's whole search
    was desk, desk, desk, desk, desk, desk, desk, desk -- priors 3.20, 2.56,
    2.05, 1.64 ... each exactly 0.8 of the last -- at an unchanged 2.5 m path
    cost. Eight inspections, one surface."""
    log = InspectionLog()
    d, unreached_credit = 0.8, 0.25
    priors = []
    for _ in range(4):
        priors.append(round(log.factor(1), 3))
        log.searched(1, d * unreached_credit)          # never arrived
    assert priors == [1.0, 0.8, 0.64, 0.512]

    log2 = InspectionLog()
    log2.searched(2, d)                                 # arrived and looked
    assert log2.factor(2) == pytest.approx(0.2), "a real inspection must retire it"


def test_choosing_a_surface_must_also_drive_to_it():
    """The defect that invalidated every earlier C3 result: only GOTO_FRONTIER
    follows _goal_xy, and the surface selection set the goal while leaving the
    state EXPLORE. The agent never moved, re-selected the same surface five
    steps later, and scored it "never reached" each time -- eight inspections of
    one desk at an unchanged 2.3 m path cost."""
    import inspect
    from osg.agent import nav_agent

    src = inspect.getsource(nav_agent.NavAgent._select_new_frontier)
    chose = src.index("self._search_container = int(surface.ref_id)")
    after = src[chose:]
    assert "State.GOTO_FRONTIER" in after, "a chosen surface must be driven to"


def test_the_best_candidate_carries_a_fixed_mass_whatever_the_prior_looks_like():
    """Ordering and scale are used for different decisions and only one of them
    is meaningful.

    `affinity * proximity` is a relative score; its magnitude depends on how
    peaked the proximity model happens to be. But `_select_surface` compares that
    magnitude against a frontier's utility, so sharpening proximity silently
    re-tunes search-versus-explore. Measured: sharpening exp(-d/4) clipped at 0.2
    to exp(-d/1) took a pilot from 27 surface inspections over six episodes to
    ONE, because every candidate past the first few now scored below any
    frontier. Anchoring the peak decouples the two.
    """
    graph = _Graph([_Node(1, "counter", [0.0, 0.8, 0.0]),
                    _Node(2, "table", [3.0, 0.8, 0.0]),
                    _Node(3, "desk", [9.0, 0.8, 0.0])])
    for length in (1.0, 4.0):
        cands = build_container_candidates(
            graph, "bowl", InspectionLog(), last_known_xy=np.zeros(2),
            proximity_len_m=length, surface_mass=0.5,
        )
        assert max(c.prior for c in cands) == pytest.approx(0.5)
    # ...and the ordering still follows the sharper model.
    sharp = {c.ref_id: c.prior for c in build_container_candidates(
        graph, "bowl", InspectionLog(), last_known_xy=np.zeros(2),
        proximity_len_m=1.0, surface_mass=0.5)}
    assert sharp[1] > sharp[2] > sharp[3]


def test_an_inspected_surface_still_falls_away_after_normalisation():
    """Decay is applied after the peak anchor, not before. Normalising
    post-decay would restore the best survivor to full mass every round and the
    agent would never hand back to exploration."""
    graph = _Graph([_Node(1, "counter", [0.0, 0.8, 0.0]),
                    _Node(2, "table", [3.0, 0.8, 0.0])])
    log = InspectionLog()
    before = {c.ref_id: c.prior for c in build_container_candidates(
        graph, "bowl", log, last_known_xy=np.zeros(2), surface_mass=0.5)}
    log.searched(1, 0.8)
    after = {c.ref_id: c.prior for c in build_container_candidates(
        graph, "bowl", log, last_known_xy=np.zeros(2), surface_mass=0.5)}
    assert after[1] < before[1]
    assert after[2] == pytest.approx(before[2])
    assert max(after.values()) < 0.5


def test_surface_mass_scales_every_candidate_without_reordering():
    """The mass is the search-versus-explore knob; it must not touch the order."""
    graph = _Graph([_Node(1, "counter", [0.0, 0.8, 0.0]),
                    _Node(2, "table", [3.0, 0.8, 0.0]),
                    _Node(3, "desk", [9.0, 0.8, 0.0])])
    low = build_container_candidates(graph, "bowl", InspectionLog(),
                                     last_known_xy=np.zeros(2), surface_mass=0.5)
    high = build_container_candidates(graph, "bowl", InspectionLog(),
                                      last_known_xy=np.zeros(2), surface_mass=1.0)
    lo = {c.ref_id: c.prior for c in low}
    hi = {c.ref_id: c.prior for c in high}
    assert sorted(lo, key=lo.get) == sorted(hi, key=hi.get)
    for k in lo:
        assert hi[k] == pytest.approx(2.0 * lo[k])
