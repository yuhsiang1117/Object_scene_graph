"""The belief models, and the switches that turn them on.

Three separate questions get three separate channels, and keeping them separate
is the whole design:

  presence    is the object still at its mapped pose? A binary Bayes filter in
              log-odds, fed by the detector and optionally by the VLM as a
              second sensor with its own measured (r, q). objects/presence.py.

  recall      P(detected | present, this view) -- the size of a negative update.
              A constant is the honest default before anything is fitted;
              scripts/fit_recall_model.py produces the logistic from a logged
              run. objects/presence.RecallModel.

  affinity    where does an object of this class get put down? A static table
              covers the categories this project has cared about, and every YCB
              target is missing from it -- a prior of "no idea" makes the search
              posterior fall back to a flat weight over every surface in the
              house, which is the undirected wandering the search exists to
              replace. graph/priors.py, llm/affinity.py.

Each is OFF by default and each is built here rather than inline in the agent,
so "which belief models is this run using, and with what constants" is one file
to read rather than three constructors buried in NavAgent.__init__.
"""
from __future__ import annotations


def build_affinity_prior(cfg):
    """Where does this class of object get put down? None unless asked for.

    graph/priors.py has a hand-written table for the categories this project
    has cared about; every YCB target is missing from it, and a prior of "no
    idea" makes the search posterior fall back to a flat weight over every
    surface in the house -- the undirected wandering C3 exists to replace.
    """
    ec = cfg.exploration
    if ec is None or not ec.affinity_llm:
        return None
    from ..graph.containers import CONTAINER_CATEGORIES
    from ..llm.affinity import AffinityProvider
    from ..llm.client import ChatClient

    client = None
    if cfg.llm.api_key:
        client = ChatClient(
            cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
            cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
        )
    return AffinityProvider(
        client, sorted(CONTAINER_CATEGORIES),
        cache_path=str(ec.affinity_cache or "") or None,
    )


def build_presence_filter(cfg):
    """None unless scene_graph.presence.enabled -- the filter must be an opt-in
    A/B, not a silent default (docs/DYNAMIC_SCENES.md, Phase 1)."""
    pc = cfg.scene_graph.presence
    if pc is None or not pc.enabled:
        return None
    from ..objects.presence import PresenceFilter, RecallModel

    recall = (
        RecallModel.load(pc.recall_model_path, constant=pc.recall_constant)
        if pc.recall_model_path
        else RecallModel(constant=pc.recall_constant)
    )
    return PresenceFilter(
        recall=recall,
        q_false_alarm=pc.q_false_alarm,
        l_clamp=pc.l_clamp,
        l_clamp_pos=pc.l_clamp_pos,
        occ_ratio_max=pc.occ_ratio_max,
        depth_tol_m=pc.depth_tol_m,
        # Expectation shares the ADMISSION threshold by construction: expecting
        # detections at a size the layer would have discarded biases the filter.
        min_area_px=cfg.scene_graph.min_det_bbox_px,
        range_m=tuple(pc.range_m),
        img_inside_frac=pc.img_inside_frac,
        min_depth_samples=pc.min_depth_samples,
        max_samples=pc.max_samples,
        max_tracks=pc.max_tracks,
        z_overlap_iou=pc.z_overlap_iou,
        log_path=pc.log_path,
    )
