"""The swappable parts of a run: detector, frontier scorer, VLM, environment.

Every one of these is an A/B axis rather than a fixed choice -- the detector is
switched by resolution and per-class gate, the scorer between an LLM and pure
geometry, the verifier on and off and between candidate-gate and absence-sensor
duty, the environment between the standard HM3D benchmark and the authored
dynamic one. Keeping their construction out of `eval/runner.py` means changing
which model a run uses never touches the file that drives the episode loop.
"""
from __future__ import annotations

from ..exploration.async_scorer import AsyncScorer
from ..exploration.llm_scorer import LLMTextScorer
from ..llm.client import ChatClient
from ..core.config import resolve_navigation, resolve_policy


def build_env(cfg):
    """The benchmark this run scores against.

    `ycb_authored` is the dynamic-scene benchmark: it discovers the collector's
    authored layouts at runtime, generates deterministic episodes from the
    placed YCB objects, and re-injects every rigid object after each Habitat
    reset. `objectnav` is the standard HM3D episode dataset.
    """
    mode = str(cfg.eval.mode)
    if mode == "ycb_authored":
        from ..sim.ycb_env import YCBAuthoredNavEnv

        return YCBAuthoredNavEnv(cfg)
    if mode == "objectnav":
        from ..sim.habitat_env import HabitatObjectNavEnv

        return HabitatObjectNavEnv(cfg)
    raise ValueError(f"unknown eval.mode: {mode}")


def build_detector(cfg):
    if cfg.detector.name == "yoloe":
        from ..perception.detector import YoloeDetector

        return YoloeDetector(
            weights=cfg.detector.weights,
            conf=cfg.detector.conf,
            class_conf=dict(cfg.detector.class_conf or {}),
            imgsz=cfg.detector.imgsz,
            half=cfg.detector.half,
            device=cfg.detector.device,
        )
    if cfg.detector.name == "stub":
        from ..perception.detector import StubDetector

        return StubDetector()
    raise ValueError(f"unknown detector: {cfg.detector.name}")


def build_scorer(cfg) -> AsyncScorer:
    # Geometric-only exploration (no LLM): nearest frontier weighted by
    # exploration range (info gain). select_frontier falls back to
    # unscored_prior for every frontier.
    mode = str(getattr(cfg.exploration, "frontier_text_scorer", "disabled"))
    if mode == "disabled":
        from ..exploration.scorer import NullScorer
        return AsyncScorer(NullScorer())
    if mode != "llm_text":
        raise ValueError(
            f"exploration.frontier_text_scorer={mode!r} is not recognised"
        )
    # Old-algorithm pipeline: text-LLM frontier ranking over the scene-graph
    # subgraphs (ObjectSceneGraph_old frontiers_ranking).
    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    inner = LLMTextScorer(client, cfg.exploration.subgraph_radius_m,
                          cfg.exploration.max_frontiers_per_call)
    return AsyncScorer(inner)


def build_ranker(cfg):
    if str(getattr(cfg.exploration, "ranker", "none")) != "ascent":
        return None
    from ..exploration.ascent_ranker import AscentFrontierRanker

    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    return AscentFrontierRanker(
        client, topk=int(cfg.exploration.ranker_topk),
        subgraph_radius_m=float(cfg.exploration.subgraph_radius_m),
    )


def build_floor_planner(cfg):
    if not bool(getattr(cfg.exploration, "floor_llm", False)):
        return None
    from ..exploration.floor_planner import FloorDecisionPlanner
    from ..exploration.knowledge_prior import FloorPrior, KnowledgeGraph

    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    try:
        floor_prior = FloorPrior.load()
    except OSError:
        floor_prior = None
    try:
        knowledge = KnowledgeGraph.load()
    except OSError:
        knowledge = None
    return FloorDecisionPlanner(
        client, floor_prior=floor_prior, kg=knowledge,
        ask_every_steps=int(cfg.exploration.floor_ask_every),
        min_steps_on_floor=int(cfg.exploration.floor_min_steps),
    )


def build_run_components(cfg) -> dict:
    """Construct heavyweight, shareable components once per evaluation run."""
    from ..perception.image_text import build_image_text_scorer
    from ..perception.room_classifier import build_room_classifier
    from ..perception.stair_seg import build_stair_segmenter

    navigation = resolve_navigation(cfg.agent)
    pointnav = None
    if navigation == "pointnav":
        from ..planning.pointnav_driver import build_pointnav

        pointnav = build_pointnav(cfg)
    return {
        "navigation": navigation,
        "policy": resolve_policy(cfg.agent),
        "pointnav": pointnav,
        "detector": build_detector(cfg),
        "scorer": build_scorer(cfg),
        "verifier": build_verifier(cfg),
        "ranker": build_ranker(cfg),
        "floor_planner": build_floor_planner(cfg),
        "room_classifier": build_room_classifier(cfg),
        "image_text": build_image_text_scorer(cfg),
        "stair_segmenter": build_stair_segmenter(cfg),
    }


def build_agent(
    cfg, components: dict, target: str, *, keyframe_dir=None, profiler=None,
    nav_fn=None, reachable_fn=None,
):
    """Build one policy behind the common ``act(frame)`` contract."""
    policy = components["policy"]
    if policy == "nav_agent":
        from ..agent.nav_agent import NavAgent

        agent_cls = NavAgent
    elif policy == "ascent":
        from ..agent.ascent_agent import AscentAgent

        agent_cls = AscentAgent
    else:
        from ascentnav.agent import AscentNavAgent

        agent_cls = AscentNavAgent
    return agent_cls(
        cfg, components["detector"], components["scorer"],
        components["verifier"], target, keyframe_dir=keyframe_dir,
        profiler=profiler, nav_fn=nav_fn, reachable_fn=reachable_fn,
        pointnav=components["pointnav"], ranker=components["ranker"],
        floor_planner=components["floor_planner"],
        room_classifier=components["room_classifier"],
        image_text=components["image_text"],
        stair_segmenter=components["stair_segmenter"],
    )


def build_verifier(cfg):
    """VLM candidate verifier, or None when verification is disabled (the
    old-algorithm terminal: viewpoint pre-position + bbox/depth stop, no VLM).

    The verifier reuses the configured LLM endpoint/key (already NVIDIA NIM in
    the matched setup) and only swaps in the vision model named by
    verification.vlm_model -- the text scorer and the vision verifier share one
    NIM account, differing only by model."""
    if not cfg.verification.enabled:
        return None
    from ..verification.verifier import VLMVerifier

    client = ChatClient(
        cfg.llm.base_url, cfg.verification.vlm_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    return VLMVerifier(
        client,
        accept_confidence=cfg.verification.accept_confidence,
        choice_mode=cfg.verification.choice_mode,
    )


def unload_ollama_models(cfg) -> None:
    """Ask ollama to release VRAM (keep_alive=0) so the one-time YOLOE text
    encoding can run on the GPU; ollama reloads lazily on the next call.

    Must cover every model ollama might be holding: exploration scoring
    (cfg.llm.*) AND the separate, larger verification model
    (cfg.verification.vlm_model) — omitting the latter left a 7B model
    resident from a prior run/benchmark and starved YOLOE's fp32 load of
    VRAM (CUDA OOM observed here on a 6 GB card)."""
    import json as _json
    import urllib.request

    host = str(cfg.llm.base_url).rsplit("/v1", 1)[0]
    models = {cfg.llm.text_model, cfg.llm.vlm_model}
    if cfg.verification.enabled:
        models.add(cfg.verification.vlm_model)
    for model in models:
        try:
            req = urllib.request.Request(
                host + "/api/generate",
                data=_json.dumps({"model": model, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass  # best-effort; ollama may be down in no-LLM ablations
