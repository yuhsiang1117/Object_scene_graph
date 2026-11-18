"""The dynamic surface posterior must cross floors before planning in 2D."""
from types import SimpleNamespace

import numpy as np

from osg.agent.nav_agent import NavAgent
from osg.exploration.async_scorer import AsyncScorer
from osg.exploration.scorer import NullScorer
from osg.exploration.strategy import WorldView
from osg.graph.scene_graph import ContainerNode, FloorNode
from osg.perception.detector import StubDetector

from .conftest import make_frame
from .test_nav_agent import make_cfg


class _CountingPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, *_args):
        self.calls += 1
        raise AssertionError("a remote-floor coordinate reached the 2D planner")


class _CountingViewpoints:
    def approach_viewpoint(self, *_args):
        raise AssertionError("a remote-floor coordinate was projected into this floor")


def _agent():
    cfg = make_cfg()
    cfg.exploration.search_posterior = True
    agent = NavAgent(cfg, StubDetector(), AsyncScorer(NullScorer()), None, "bowl")
    agent.exploration.planner = _CountingPlanner()
    agent.exploration.viewpoint_planner = _CountingViewpoints()
    agent.scene_graph.floors = {
        4: FloorNode(4, 0.0),
        9: FloorNode(9, 2.7),
    }
    agent.scene_graph.containers = {
        1: ContainerNode(1, "table", [1], np.array([1.0, 0.75, 2.0]),
                         0.75, 0.6, floor=4),
        2: ContainerNode(2, "table", [2], np.array([1.0, 3.45, 2.0]),
                         3.45, 0.6, floor=9),
        3: ContainerNode(3, "counter", [3], np.array([2.0, 3.45, 2.0]),
                         3.45, 0.6, floor=9),
    }
    return agent


def _world(agent, intrinsics, floor_key=4):
    frame = make_frame(intrinsics, np.eye(4))
    return WorldView(
        frame=frame,
        step=12,
        agent_xy=np.zeros(2),
        costmap=agent.costmap,
        scene_graph=agent.scene_graph,
        object_layer=agent.object_layer,
        keyframes=agent.keyframes,
        target="bowl",
        goal_xy=None,
        floor_id=floor_key,
    )


def test_remote_floor_mass_requests_a_switch_without_same_floor_planning(intrinsics):
    agent = _agent()
    world = _world(agent, intrinsics)

    assert agent.exploration._select_surface(world, None) is None
    assert agent.exploration.requested_floor == 9
    assert agent.exploration.selected_search_floor == 9
    assert agent.exploration.planner.calls == 0


def test_glance_evidence_is_scoped_to_the_observed_floor(intrinsics):
    agent = _agent()
    world = _world(agent, intrinsics)
    world.frame = make_frame(intrinsics, np.eye(4), depth_value=5.0)
    agent.object_layer.presence_filter = SimpleNamespace()

    agent.exploration.glance(world)

    assert agent.exploration.search_log.factor(1) < 1.0
    assert agent.exploration.search_log.factor(2) == 1.0
    assert agent.exploration.search_log.factor(3) == 1.0
