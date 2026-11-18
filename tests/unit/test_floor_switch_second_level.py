"""An undirected storey switch requires a second known level.

Measured on the DualMap released benchmark: with the floor group on, a
single-floor scene still hunted portals once its frontiers got far, and
`find_portals` read a tall bookcase as a portal -- 1-2 spurious floor-switch
attempts an episode that walked the agent off and perturbed a run the floor
group should leave bit-identical. A schema-v2 prior map seeds every storey into
the estimator on load, so a genuine multi-floor run has >= 2 levels here and is
unaffected; only a one-level scene, with nowhere to go, is gated.
"""
from __future__ import annotations

from types import SimpleNamespace

from osg.mapping.portals import FloorSwitchPolicy


class _Estimator:
    def __init__(self, levels):
        self._levels = dict(levels)

    @property
    def levels(self):
        return self._levels


def _floor_cfg(**over):
    base = dict(
        enabled=True, switch_requires_second_level=False, prior_stairs_override_gate=False,
        use_prior_stairs=False, new_level_m=1.8, portal_max_delta_m=4.0, portal_min_cells=20,
        climb_targets="portals", portal_failure_memory=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _policy(estimator, fcfg):
    from osg.agent.floor_policy import FloorPolicy
    pol = FloorPolicy.__new__(FloorPolicy)
    pol.estimator = estimator
    pol.stack = SimpleNamespace(current_id=0, current=SimpleNamespace(first_step=0, arrived_step=0))
    pol.switch_policy = FloorSwitchPolicy(max_steps=500, no_switch_before=0, min_interval_steps=0)
    pol.cfg = SimpleNamespace(floor=fcfg, agent=SimpleNamespace(navmesh_3d_goals=False))
    pol.stats = {}
    return pol


def test_gate_blocks_undirected_switch_with_one_level():
    pol = _policy(_Estimator({0: 0.0}), _floor_cfg(switch_requires_second_level=True))
    # Deep in the episode, floor exhausted -- the gate would otherwise open.
    out = pol.try_switch(frame=None, step=300, best_path_cost=None, scene_graph=SimpleNamespace(objects=[], rooms={}),
                         target="bowl", reachable_fn=lambda *a, **k: True, target_floor=None)
    assert out is None
    assert pol.stats.get("floor_switch_attempts", 0) == 0


def test_gate_is_off_by_default():
    """Default flag off: the one-level scene is NOT gated here (it reaches the
    portal search, which finds nothing on a None costmap and returns None -- but
    not because of this gate). The point is the early return does not fire."""
    pol = _policy(_Estimator({0: 0.0}), _floor_cfg(switch_requires_second_level=False))
    # It will raise or return inside find_portals with a None costmap; we only
    # assert the guard itself did not short-circuit by checking the flag path.
    import pytest
    with pytest.raises(Exception):
        pol.try_switch(frame=None, step=300, best_path_cost=None,
                       scene_graph=SimpleNamespace(objects=[], rooms={}),
                       target="bowl", reachable_fn=lambda *a, **k: True, target_floor=None)


def test_two_levels_passes_the_gate():
    """With a second level known the gate does not block -- it falls through to
    the portal search (which raises on a None costmap, proving it got past)."""
    pol = _policy(_Estimator({0: 0.0, 1: 3.0}), _floor_cfg(switch_requires_second_level=True))
    import pytest
    with pytest.raises(Exception):
        pol.try_switch(frame=None, step=300, best_path_cost=None,
                       scene_graph=SimpleNamespace(objects=[], rooms={}),
                       target="bowl", reachable_fn=lambda *a, **k: True, target_floor=None)
