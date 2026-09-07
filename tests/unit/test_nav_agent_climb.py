

def test_floor_is_exhausted_when_no_explore_frontiers_remain():
    """ASCENT's rule (ascent_policy.py:655), and why it replaced a step count.

    The old rule was `steps_on_floor >= floor_exp_steps` (100), which is the
    same threshold as the floor-LLM gate. Both mechanisms then fired on the same
    step and pointed the same way, and the S12 arm came out 45/50 episodes
    bit-identical to its baseline. ASCENT separates them: the LLM may move a
    floor at 100 steps, the geometry only once there is nothing left here.
    """
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.exploration.stair_prior = 0.6
    cfg.exploration.stair_explored_rule = "no_frontiers"
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "bed")
    layer = agent.floors.current()
    layer.steps_on_floor = 400  # far past the old 100-step threshold

    agent._mark_floor_explored(3)
    assert not layer.explored, "frontiers remain, so the floor is not exhausted"

    agent._mark_floor_explored(0)
    assert layer.explored


def test_step_rule_remains_available_and_is_not_the_default():
    from osg.core.config import ExplorationConfig

    assert ExplorationConfig().stair_explored_rule == "no_frontiers"


def _climb_agent(rule="topological"):
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.agent.climb_exit_rule = rule
    cfg.agent.stair_exit_m = 0.5
    return NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "bed")


def test_on_the_staircase_is_not_off_it():
    """Port of is_robot_in_stair_map_fast (ascent/map_controller.py:181-215):
    any stair cell within the exit radius means the agent is still on it."""
    import numpy as np

    agent = _climb_agent()
    agent._climb_cells_xy = np.array([[1.0, 0.0], [1.2, 0.0], [1.4, 0.0]])
    assert not agent._left_the_stairs(np.array([1.1, 0.0]))
    assert not agent._left_the_stairs(np.array([1.0, 0.4]))
    assert agent._left_the_stairs(np.array([1.0, 2.0]))


def test_no_recorded_cells_does_not_trap_the_climb():
    """Nothing to be on means the height rule stands alone, rather than a climb
    that can never end."""
    import numpy as np

    agent = _climb_agent()
    agent._climb_cells_xy = None
    assert agent._left_the_stairs(np.array([0.0, 0.0]))


def test_cells_are_stored_in_world_coordinates():
    """Costmap2D.ensure_contains reallocates and shifts the origin when the map
    grows, so grid indices captured at climb entry can point at the wrong cells
    a few steps later. ASCENT can keep pixel indices because its map is a fixed
    size; this one cannot.
    """
    import numpy as np

    from osg.mapping.frontier import Frontier

    agent = _climb_agent()
    layer = agent.floors.current()
    cells = np.array([[10, 10], [10, 11]], dtype=int)
    before = layer.costmap.grid_to_world(cells.astype(float)).copy()

    f = Frontier(id=0, centroid_xy=layer.costmap.grid_to_world(np.array([10.0, 10.0])),
                 cells=cells, size=2, kind="stair_up")
    agent._climb_cells_xy = layer.costmap.grid_to_world(f.cells.astype(float))

    # Grow the map: the same world points now live at different indices.
    layer.costmap.ensure_contains(np.array([200.0, 200.0]))
    after = layer.costmap.grid_to_world(cells.astype(float))
    assert not np.allclose(before, after), "the grow did not shift the origin"
    assert np.allclose(agent._climb_cells_xy, before), "stored cells drifted with the map"


def test_height_rule_is_the_default():
    from osg.core.config import AgentConfig

    assert AgentConfig().climb_exit_rule == "height"


def _stairs_agent(**mapping):
    from osg.agent.nav_agent import NavAgent
    from osg.exploration.async_scorer import AsyncScorer
    from osg.perception.detector import StubDetector

    from .test_nav_agent import _StubScorer, make_cfg

    cfg = make_cfg()
    cfg.mapping.multi_floor = True
    cfg.agent.stair_exit_m = 0.5
    for k, v in mapping.items():
        setattr(cfg.mapping, k, v)
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(_StubScorer()), None, "bed")
    layer = agent.floors.current()
    agent.stair_detector._ensure_grids(layer)
    return agent, layer


def _mark_stairs(agent, layer, xy, hits=5):
    import numpy as np

    rc = layer.costmap.world_to_grid(np.array(xy, dtype=float))
    layer.up_stair_hits[rc[0] - 2:rc[0] + 3, rc[1] - 2:rc[1] + 3] = hits


def test_on_a_staircase_uses_the_whole_floor_mask():
    """Port of is_robot_in_stair_map_fast over the floor's stair map, not just
    the component of a climb in progress. `_left_the_stairs` answers the
    narrower question and cannot gate the floor stack: the episode with 18 floor
    switches recorded ONE climb attempt, so its oscillation never entered CLIMB.
    """
    import numpy as np

    agent, layer = _stairs_agent()
    _mark_stairs(agent, layer, (2.0, 0.0))
    assert agent._on_a_staircase(np.array([2.0, 0.0]))
    assert agent._on_a_staircase(np.array([2.0, 0.4]))
    assert not agent._on_a_staircase(np.array([2.0, 3.0]))


def test_cells_below_the_detector_threshold_do_not_count():
    """The mask uses the detector's own min_hits, so it sees exactly the cells a
    component would be built from -- not every stray projection."""
    import numpy as np

    agent, layer = _stairs_agent()
    agent.stair_detector.min_hits = 3
    _mark_stairs(agent, layer, (2.0, 0.0), hits=1)
    assert not agent._on_a_staircase(np.array([2.0, 0.0]))
    _mark_stairs(agent, layer, (2.0, 0.0), hits=3)
    assert agent._on_a_staircase(np.array([2.0, 0.0]))


def test_the_two_freeze_scopes_are_independent():
    import numpy as np

    from osg.core.types import FrameData

    class _F:
        camera_position = np.array([2.0, 0.88, 0.0])

    agent, layer = _stairs_agent(freeze_floor_on_stairs=True)
    _mark_stairs(agent, layer, (2.0, 0.0))
    assert agent._floor_frozen(_F()), "on stairs, outside CLIMB"

    off, layer_off = _stairs_agent(freeze_floor_on_stairs=False)
    _mark_stairs(off, layer_off, (2.0, 0.0))
    assert not off._floor_frozen(_F()), "both scopes off means never frozen"


def test_both_freezes_are_off_by_default():
    from osg.core.config import MappingConfig

    cfg = MappingConfig()
    assert cfg.freeze_floor_in_climb is False
    assert cfg.freeze_floor_on_stairs is False
