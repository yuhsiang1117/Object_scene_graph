"""The prior map, drawn with the target on it, as a run artifact.

The figure answers geometric questions the episode record cannot: which storey
the target is on, which end of the staircase the agent drove to, and whether
the prior map covers the place at all. It is written at the END of a run, after
episodes.jsonl and summary.json, so the one rule it must obey is that it can
never take a finished run away -- every failure path here is silent or printed,
never raised.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from osg.eval.obstacle_map_viz import (_render, _storey_heights,
                                       save_obstacle_map_pngs)
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D


def _cfg(maps_dir="", png=True):
    return SimpleNamespace(
        ycb=SimpleNamespace(obstacle_map_in=maps_dir),
        eval=SimpleNamespace(obstacle_map_png=png),
    )


def test_a_run_that_read_no_obstacle_map_draws_nothing(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"scene": "s"}) + "\n")
    assert save_obstacle_map_pngs(_cfg(""), tmp_path) == []


def test_the_flag_turns_it_off(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"scene": "s"}) + "\n")
    assert save_obstacle_map_pngs(_cfg(str(tmp_path), png=False), tmp_path) == []


def test_a_missing_episodes_file_is_not_an_error(tmp_path):
    assert save_obstacle_map_pngs(_cfg(str(tmp_path)), tmp_path) == []


def test_a_missing_map_for_the_scene_is_not_an_error(tmp_path):
    """The commonest real case: `obstacle_map_in` points at a directory that
    has maps for other scenes but not this one."""
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"scene": "00000-nosuch", "target": "bowl"}) + "\n")
    assert save_obstacle_map_pngs(_cfg(str(tmp_path)), tmp_path) == []


def test_a_broken_snapshot_does_not_raise(tmp_path):
    """A finished run must survive a corrupt prior map."""
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"scene": "00800-x", "target": "bowl"}) + "\n")
    (tmp_path / "00800-x.json").write_text("{ this is not json")
    assert save_obstacle_map_pngs(_cfg(str(tmp_path)), tmp_path) == []


def test_unreadable_episode_lines_do_not_raise(tmp_path):
    (tmp_path / "episodes.jsonl").write_text("{not json at all\n")
    assert save_obstacle_map_pngs(_cfg(str(tmp_path)), tmp_path) == []


def test_the_render_separates_the_four_states():
    cm = Costmap2D(resolution=0.05, size_m=2.0)
    cm.grid[:, :] = UNKNOWN
    cm.grid[0, 0] = FREE
    cm.grid[0, 1] = OCCUPIED
    cm.stair_mask = np.zeros(cm.grid.shape, dtype=bool)
    cm.stair_mask[0, 2] = True
    img = _render(cm)
    assert tuple(img[0, 0]) != tuple(img[0, 1]), "free and obstacle look the same"
    assert tuple(img[0, 2]) != tuple(img[0, 0]), "stairs are not distinguishable"
    assert tuple(img[5, 5]) != tuple(img[0, 0]), "unknown reads as free"


def test_storey_heights_come_from_the_run_bottom_up():
    episodes = [
        {"floor_log": [[1, 0, 3.163], [40, 0, 3.163]]},
        {"floor_log": [[1, 1, 0.163]]},
    ]
    assert _storey_heights(episodes) == [0.163, 3.163]


def test_storey_heights_of_a_run_with_no_floor_log():
    assert _storey_heights([{"target": "bowl"}]) == []
