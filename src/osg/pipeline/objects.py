"""The object layer, built from a config.

NavAgent and WorldModel used to construct `ObjectLayer` with the same eighteen
keyword arguments, written out twice; the offline replay
(scripts/replay_object_layer.py) needs the identical object, so there is one
builder and three callers. A knob added to `scene_graph` is wired here once.
"""
from __future__ import annotations

from ..objects.object_layer import ObjectLayer
from .beliefs import build_presence_filter


def build_object_layer(cfg, feature_memory=None) -> ObjectLayer:
    sg = cfg.scene_graph
    return ObjectLayer(
        feature_memory=feature_memory,
        assoc_score_thresh=sg.assoc_score_thresh,
        assoc_depth_gate_m=sg.assoc_depth_gate_m,
        assoc_category_gate=sg.assoc_category_gate,
        min_obs_for_refine=sg.min_obs_for_refine,
        refine_every=sg.refine_every,
        refine_max_center_move_m=sg.refine_max_center_move_m,
        link_dist_m=sg.link_dist_m,
        link_max_frame_gap=sg.link_max_frame_gap,
        min_det_score=sg.min_det_score,
        min_det_bbox_px=sg.min_det_bbox_px,
        confirm_baseline_m=sg.confirm_baseline_m,
        repeat_view_discount=sg.repeat_view_discount,
        presence_filter=build_presence_filter(cfg),
        target_bypasses_gates=sg.target_bypasses_gates,
        max_range_m=cfg.mapping.max_range_m,
        fp_disable_radius_m=sg.fp_disable_radius_m,
        cloud_stride=sg.cloud_stride,
        cloud_cap=sg.cloud_cap,
        ellipsoid=getattr(sg, "ellipsoid", None),
    )
