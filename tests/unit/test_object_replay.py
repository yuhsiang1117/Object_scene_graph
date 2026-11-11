"""A recording replayed through the unchanged layer is the online run, to the bit.

This is the acceptance test of the offline ellipsoid harness
(eval/object_recorder.py, eval/object_replay.py): every variant is measured as
a replay, so a replay that drifted from the run it replays would measure the
harness instead of the variant.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from osg.core.config import OSGConfig
from osg.eval.object_recorder import ObjectFrameRecorder, _tracks_blob
from osg.eval.object_replay import (
    chain_scene, first_difference, load_prior_into, replay_episode, tracks_equal,
)
from osg.pipeline.objects import build_object_layer

from .conftest import make_camera
from .synthetic_scene import Box, detections, frame_at


BOXES = [
    Box("table", np.array([0.0, 0.38, 0.0]), np.array([0.6, 0.38, 0.4])),
    Box("chair", np.array([1.4, 0.45, 0.6]), np.array([0.25, 0.45, 0.25])),
    Box("bowl", np.array([0.1, 0.81, 0.05]), np.array([0.09, 0.05, 0.09])),
]


def _cfg(presence=True):
    cfg = OSGConfig()
    cfg.scene_graph.presence.enabled = presence
    cfg.scene_graph.target_bypasses_gates = True   # the final arm's setting
    return cfg


def _orbit(intr, n, radius=2.6, phase=0.0):
    for k in range(n):
        a = phase + 2.0 * np.pi * k / (n * 2.5)
        pos = np.array([radius * np.cos(a), 1.25, radius * np.sin(a)])
        yield k, make_camera(pos, np.array([0.3, 0.5, 0.2]))


def _record(cfg, intr, root, tag, scene, wall, n=14, phase=0.0, prior_blob=None):
    layer = build_object_layer(cfg)
    layer.set_target("bowl")
    layer.keep_cloud_labels = {"bowl"}
    if prior_blob is not None:
        load_prior_into(layer, prior_blob, restore=True)
    layer.recorder = ObjectFrameRecorder(str(root), tag, {"scene": scene, "wall_start": wall})
    for k, T in _orbit(intr, n, phase=phase):
        frame, inst = frame_at(BOXES, intr, T, frame_id=k)
        layer.update(frame, detections(BOXES, inst, frame, bleed_px=2), floor_key=0)
    layer.recorder.close(layer)
    return layer


@pytest.fixture
def small_intr():
    from osg.core.types import CameraIntrinsics
    return CameraIntrinsics(fx=200.0, fy=200.0, cx=160.0, cy=120.0, width=320, height=240)


def test_replay_reproduces_the_recorded_run_bit_for_bit(tmp_path, small_intr):
    cfg = _cfg()
    online = _record(cfg, small_intr, tmp_path, "ep0", "s", 0.0)
    assert len(online._tracks) >= 3          # the scene was actually mapped
    replayed = replay_episode(cfg, tmp_path / "ep0")
    recorded = json.loads((tmp_path / "ep0" / "final.json").read_text())
    assert first_difference(_tracks_blob(replayed), recorded) is None
    assert replayed.funnel == online.funnel


def test_a_changed_layer_is_seen_as_changed(tmp_path, small_intr):
    cfg = _cfg()
    _record(cfg, small_intr, tmp_path, "ep0", "s", 0.0)
    cfg.scene_graph.refine_every = 1
    replayed = replay_episode(cfg, tmp_path / "ep0")
    recorded = json.loads((tmp_path / "ep0" / "final.json").read_text())
    assert not tracks_equal(_tracks_blob(replayed), recorded)


def test_chained_episodes_are_recovered_and_replay_exactly(tmp_path, small_intr):
    cfg = _cfg()
    first = _record(cfg, small_intr, tmp_path, "ep0", "s", 0.0)
    _record(cfg, small_intr, tmp_path, "ep1", "s", 1.0, phase=1.0, prior_blob=_tracks_blob(first))
    eps = [tmp_path / "ep0", tmp_path / "ep1"]
    assert chain_scene(cfg, eps) == {"ep0": None, "ep1": "ep0"}
    # ep1 replayed from ep0's REPLAYED final, through restore_tracks.
    r0 = replay_episode(cfg, eps[0])
    r1 = replay_episode(cfg, eps[1], prior=_tracks_blob(r0), restore=True)
    recorded = json.loads((eps[1] / "final.json").read_text())
    assert first_difference(_tracks_blob(r1), recorded) is None
