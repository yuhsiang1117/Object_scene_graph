"""OSG's world model, detached from OSG's FSM.

`NavAgent` builds two things on every step: a world model -- per-storey
occupancy, the object layer with its presence beliefs, the room segmentation,
the scene graph -- and a decision about what to do next. Only the second is
OSG's control flow. This class is the first half on its own, so a policy that
is NOT `NavAgent` can leave behind the map the dynamic-scene protocol reads
back (`graph/map_store.py`).

Why it exists
-------------
Stage 1 needs two artifacts per scene: ASCENT's `ObstacleMap` stack and OSG's
scene graph. Running the scene twice to get them is not just slow, it is worse:
measured on 00800, `ascentnav` reaches both storeys inside its 500-step budget
while `nav_agent` needs 1500 steps to reach the second at all, and at 500 it
ends with 213 furniture tracks and no target objects. Building the scene graph
ALONGSIDE ASCENT's navigation gets the better explorer and one pass.

It is fed AFTER the action is decided and hands nothing back, so it cannot
change what the transcription does -- `src/navigation/README.md`'s F1-F14
fidelity notes are unaffected.

What it deliberately does not do
--------------------------------
No planning, no frontiers, no verification. Those belong to whichever policy is
driving. This owns evidence only, which is exactly the set `save_map`
serialises: `_floor_stack`, `object_layer`, `costmap`.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

import numpy as np

from ..core.profiler import Profiler
from ..core.types import Detection, FrameData
from ..graph.scene_graph import ROOM_IDS_PER_FLOOR, SceneGraph
from ..mapping.costmap import PLANE, Costmap2D
from ..pipeline.objects import build_object_layer
from ..perception.keyframe import KeyframeSelector, KeyframeStore
from .floor_policy import FloorPolicy


class WorldModel:
    """The OSG map, built from frames and detections by whoever is driving.

    Exposes `_floor_stack`, `object_layer` and `costmap` under the names
    `graph/map_store.save_map` reads, so a snapshot taken from this is the same
    schema-v2 artifact a `NavAgent` pass writes -- pass 2 cannot tell which
    policy built its prior map, which is what keeps the two comparable.
    """

    def __init__(
        self,
        cfg,
        detector,
        *,
        target: str = "",
        keyframe_dir=None,
        profiler: Optional[Profiler] = None,
        room_classifier=None,
        stair_detector=None,
        stats: Optional[dict] = None,
    ) -> None:
        self.cfg = cfg
        # The object layer's OWN detector, and on this benchmark it is NOT the
        # policy's: ASCENT drives on D-FINE, which is closed-set COCO and
        # cannot name a cracker box or a pitcher, while the scene graph the
        # presence filter re-finds objects in is made of exactly those. One
        # detector per job, both reading the same frame.
        self.detector = detector
        self.profiler = profiler or Profiler(enabled=False)
        self.room_classifier = room_classifier
        self.stair_detector = stair_detector
        self.stats: dict = stats if stats is not None else {}

        self.floors = FloorPolicy(cfg, self.stats)
        self.object_layer = build_object_layer(cfg)
        self.scene_graph = SceneGraph(
            container_top_h_m=tuple(cfg.scene_graph.container_top_h_m),
            container_min_area_m2=cfg.scene_graph.container_min_area_m2,
            container_support_tol_m=cfg.scene_graph.container_support_tol_m,
            container_min_obs=cfg.scene_graph.container_min_obs,
            container_min_score=cfg.scene_graph.container_min_score,
            container_merge_m=cfg.scene_graph.container_merge_m,
            containers_floor_relative=bool(getattr(
                cfg.scene_graph, "containers_floor_relative", False)),
            container_merge_sigma=float(
                getattr(cfg.scene_graph, "container_merge_sigma", 0.0) or 0.0
            ),
        )
        self.keyframes = KeyframeStore(save_dir=keyframe_dir)
        self.kf_selector = KeyframeSelector(
            cfg.scene_graph.keyframe_trans_m, cfg.scene_graph.keyframe_rot_deg
        )
        self._kf_count = 0
        self._room_votes: list = []
        self.reset(target)

    # ------------------------------------------------------------------ state

    def reset(self, target: str) -> None:
        """Per-episode state. The object layer and scene graph are NOT cleared
        here, for the same reason `NavAgent.reset` does not clear them: one is
        constructed per episode, so the only caller is the constructor."""
        self.target = str(target or "")
        self.step_count = 0
        self._kf_count = 0
        self._room_votes = []
        self.floors.reset()
        self.object_layer.set_target(self.target)
        self.object_layer.keep_cloud_labels = {self.target}
        self.kf_selector.reset()
        self.keyframes.reset()

    @property
    def costmap(self) -> Costmap2D:
        return self.floors.costmap

    @property
    def floor_layer(self):
        return self.floors.layer

    @property
    def _floor_stack(self):  # graph/map_store.py snapshots the stack
        return self.floors.stack

    @property
    def _room_labels(self):
        return self.floors.room_labels

    @_room_labels.setter
    def _room_labels(self, labels) -> None:
        self.floors.room_labels = labels

    # ---------------------------------------------------------------- observe

    def observe(self, frame: FrameData, step: int) -> None:
        """One step of mapping: which storey, and what the depth image adds.

        Mirrors `NavAgent._act_inner`'s mapping half, including the stair
        check: while the agent is ON a staircase its depth belongs to neither
        storey, and folding it into either paints a phantom wall across the
        landing (docs/history/MULTI_FLOOR.md).
        """
        self.step_count = int(step)
        floor_y = self.floors.observe(frame, self.step_count)
        multi_floor = bool(
            getattr(self.cfg.mapping, "multi_floor", False)
            or getattr(self.cfg.floor, "per_floor_costmap", False)
        )
        reject_m = float(getattr(self.cfg.mapping, "floor_reject_m", 0.0))
        standing_y = float(frame.camera_position[1] - self.cfg.agent.camera_height)
        if multi_floor:
            off_map = self.floors.on_stairs
        else:
            off_map = reject_m > 0.0 and abs(standing_y - float(floor_y)) > reject_m
        if off_map:
            self.stats["wm_frames_off_plane"] = (
                self.stats.get("wm_frames_off_plane", 0) + 1)
            return
        with self.profiler.timeit("wm_costmap"):
            self.costmap.update(
                frame,
                floor_y=floor_y,
                obstacle_low=self.cfg.mapping.obstacle_low_m,
                obstacle_high=self.cfg.mapping.obstacle_high_m,
                max_range=self.cfg.mapping.max_range_m,
                stride=self.cfg.mapping.depth_stride,
            )

    def maybe_keyframe(self, frame: FrameData,
                       dets: Optional[List[Detection]] = None) -> bool:
        """Run the keyframe pipeline if this pose is far enough from the last.

        `dets` lets a caller hand over detections it has already paid for.
        None means this runs its own detector, which is the normal case here:
        the policy's detector answers a different question.
        """
        if not self.kf_selector.is_keyframe(frame.T_wc):
            return False
        self._on_keyframe(frame, dets)
        return True

    def _on_keyframe(self, frame: FrameData,
                     dets: Optional[List[Detection]] = None) -> None:
        self._kf_count += 1
        if dets is None:
            with self.profiler.timeit("wm_detector"):
                dets = self.detector.detect(frame.rgb)
        dets = list(dets or [])
        if self.room_classifier is not None:
            self._room_votes.append((
                frame.camera_position[list(PLANE)].copy(),
                self.room_classifier.classify(frame.rgb),
            ))
        with self.profiler.timeit("wm_object_layer"):
            self.object_layer.update(frame, dets, floor_key=self.floors.current_id)
        self.keyframes.add(frame)

        if (
            bool(getattr(self.cfg.agent, "stair_evidence_every_kf", False))
            and self.stair_detector is not None
            and not self.floors.on_stairs
        ):
            with self.profiler.timeit("wm_stairs"):
                self.stair_detector.accumulate(frame, self.floor_layer, dets, None)

        pf = self.object_layer.presence_filter
        if pf is not None:
            self.stats["wm_presence_expected"] = pf.n_expected
            self.stats["wm_presence_negative"] = pf.n_negative
            self.stats["wm_presence_positive"] = pf.n_positive

        if self.floors.stairs_due(self._kf_count):
            with self.profiler.timeit("wm_stairs"):
                self.floors.mark_stairs_traversable(self.object_layer)

        if self._kf_count % self.cfg.scene_graph.room_seg_every_kf == 1:
            with self.profiler.timeit("wm_room_seg"):
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
        if self._room_labels is not None:
            if self._room_labels.shape != self.costmap.grid.shape:
                self._room_labels = self.floor_layer.segmenter.segment(self.costmap)
            with self.profiler.timeit("wm_scene_graph"):
                self.scene_graph.rebuild_floor(
                    self._room_labels, self.costmap, self.object_layer,
                    floor_key=self.floors.current_id,
                    floor_height=self.floors.height_of(self.floors.current_id),
                )
                self._label_rooms(self.floor_layer)

    def _label_rooms(self, layer) -> None:
        """Places365's vote per room, as `NavAgent._label_rooms` does it."""
        if not self._room_votes or layer.room_labels is None:
            return
        base = layer.key * ROOM_IDS_PER_FLOOR
        votes: dict = {}
        for xy, name in self._room_votes:
            rc = layer.costmap.world_to_grid(xy)
            if not layer.costmap.in_bounds(rc):
                continue
            local = int(layer.room_labels[rc[0], rc[1]])
            if local > 0:
                votes.setdefault(base + local, Counter())[name] += 1
        self.stats["wm_rooms_total"] = len(self.scene_graph.rooms)
        for room_id, counter in votes.items():
            room = self.scene_graph.rooms.get(room_id)
            if room is not None:
                room.label = counter.most_common(1)[0][0]

    # ------------------------------------------------------------- reporting

    def summary(self) -> dict:
        """What this pass mapped -- the numbers a snapshot is judged by."""
        tracks = list(self.object_layer.tracks(include_blacklisted=True))
        return {
            "floors": len(self._floor_stack._layers),
            "tracks": len(tracks),
            "labels": sorted({str(t.label) for t in tracks}),
            "known_cells": int(sum(
                int((layer.costmap.grid != -1).sum())
                for layer in self._floor_stack._layers.values()
            )),
            "rooms": len(self.scene_graph.rooms),
            "keyframes": self._kf_count,
        }
