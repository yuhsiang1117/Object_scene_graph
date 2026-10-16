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
    reset. `objectnav` is the standard HM3D episode dataset. `dualmap_protocol`
    is the released DualMap benchmark scored by DualMap's own rule, paired
    trial-by-trial with a measured run of that system.
    """
    mode = str(cfg.eval.mode)
    if mode == "ycb_authored":
        from ..sim.ycb_env import YCBAuthoredNavEnv

        return YCBAuthoredNavEnv(cfg)
    if mode == "objectnav":
        from ..sim.habitat_env import HabitatObjectNavEnv

        return HabitatObjectNavEnv(cfg)
    if mode == "dualmap_protocol":
        from ..sim.dualmap_env import DualMapProtocolEnv

        return DualMapProtocolEnv(cfg)
    if mode == "ros2":
        from ..sim.ros2_env import Ros2Env

        return Ros2Env(cfg)
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
    if cfg.detector.name == "yolo_coco":
        from ..perception.detector import YoloDetector

        return YoloDetector(
            weights=cfg.detector.weights,
            conf=cfg.detector.conf,
            class_conf=dict(cfg.detector.class_conf or {}),
            imgsz=cfg.detector.imgsz,
            half=cfg.detector.half,
            device=cfg.detector.device,
        )
    if cfg.detector.name == "dfine":
        from ..perception.detector import DFineDetector

        return DFineDetector(
            url=str(getattr(cfg.detector, "url", "http://localhost:13186/dfine")),
            sam_url=str(getattr(cfg.detector, "sam_url",
                                "http://localhost:13183/mobile_sam")),
            conf=cfg.detector.conf,
            class_conf=dict(cfg.detector.class_conf or {}),
            use_sam=bool(getattr(cfg.detector, "use_sam", True)),
            timeout_s=float(getattr(cfg.detector, "timeout_s", 15.0)),
            strict=bool(getattr(cfg.detector, "strict", False)),
            # ASCENT's non-COCO branch. Empty by default, so the HM3D arms this
            # detector was measured on are byte-identical; the benchmarks whose
            # targets COCO cannot name switch it on.
            gdino_url=str(getattr(cfg.detector, "gdino_target_url", "")),
            gdino_conf=float(getattr(cfg.detector, "gdino_target_conf", 0.35)),
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


def build_run_components(cfg, env=None) -> dict:
    """Construct heavyweight, shareable components once per evaluation run.

    `env` is needed only by the `nav2` mover, whose backend is either the
    robot's navigator (reached through the env's bridge) or habitat's follower
    (reached through the env itself). Every other component is built from the
    config alone, which is why this stayed env-free for so long.
    """
    from ..perception.image_text import build_image_text_scorer
    from ..perception.room_classifier import build_room_classifier
    from ..perception.stair_seg import build_stair_segmenter

    navigation = resolve_navigation(cfg.agent)
    pointnav = None
    if navigation == "pointnav":
        from ..planning.pointnav_driver import build_pointnav

        pointnav = build_pointnav(cfg)
    elif navigation == "nav2":
        # Returned in the `pointnav` slot because that is the slot NavAgent
        # drives a goal-shaped mover from (`nav_agent.py:1574`, `:2010`): both
        # answer `step(goal_xy) -> NavStep`, so the FSM needs no branch of its
        # own for this mover.
        pointnav = build_nav2_driver(cfg, env)
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
        "feature_memory": build_feature_memory(cfg),
        "stair_segmenter": build_stair_segmenter(cfg),
        "stair_detector": build_stair_detector(cfg),
        "region_proposer": build_region_proposer(cfg),
        "ram": build_ram_tagger(cfg),
    }


def build_nav2_driver(cfg, env):
    """The `nav2` mover, over whichever navigator this run has."""
    from ..planning.nav2_backends import RosNav2Backend, SimNav2Backend
    from ..planning.nav2_driver import Nav2Driver

    if env is None:
        raise ValueError(
            "agent.navigation=nav2 needs the env: the goal is driven by a "
            "navigator, not inferred from the frame. Pass build_run_components("
            "cfg, env=env)."
        )
    if str(cfg.eval.mode) == "ros2":
        backend = RosNav2Backend(env.transport, cfg.ros2)
    else:
        backend = SimNav2Backend(
            env, arrival_m=max(2.0 * float(cfg.agent.navmesh_goal_radius), 0.25))
    # `nav_timeout_s` is a wall-clock patience, and the mover counts control
    # ticks; `step_period_s` is what one tick costs while Nav2 drives.
    period = max(float(cfg.ros2.step_period_s), 1e-3)
    return Nav2Driver(
        backend,
        stop_radius=float(cfg.agent.pointnav_stop_radius),
        goal_resend_m=float(cfg.ros2.goal_resend_m),
        timeout_steps=int(round(float(cfg.ros2.nav_timeout_s) / period)),
    )


def build_stair_detector(cfg):
    """GroundingDINO `stair` boxes + MobileSAM masks -- the detector half of
    ASCENT's stair fusion. Only under the strict fusion; the union does not
    read it."""
    if str(getattr(cfg.agent, "policy", "")) != "ascentnav":
        return None
    if str(getattr(cfg.agent, "stair_up_mode", "ascent")) != "ascent":
        return None
    from ..perception.ascent_models import GroundingDinoStairDetector

    return GroundingDinoStairDetector(
        url=str(getattr(cfg.detector, "gdino_url", "http://localhost:13184/gdino")),
        sam_url=str(getattr(cfg.detector, "sam_url", "http://localhost:13183/mobile_sam")),
        conf=float(getattr(cfg.detector, "gdino_stair_conf", 0.60)),
        timeout_s=float(getattr(cfg.detector, "timeout_s", 15.0)),
        strict=bool(getattr(cfg.detector, "strict", False)),
    )


def build_ram_tagger(cfg):
    if not bool(getattr(cfg.exploration, "ram_tags", False)):
        return None
    from ..perception.ascent_models import RamTagger

    return RamTagger(
        url=str(getattr(cfg.exploration, "ram_url", "http://localhost:13185/ram")),
        timeout_s=float(getattr(cfg.detector, "timeout_s", 15.0)),
        strict=bool(getattr(cfg.detector, "strict", False)),
    )


def probe_served_models(cfg) -> None:
    """Fail before Habitat loads if a served model the run depends on is down.
    An unreachable BLIP-2 is not a degraded run: under ASCENT's gate it is an
    agent that can never STOP."""
    from ..perception.ascent_models import probe_perception_servers

    endpoints = {}
    if str(cfg.detector.name) == "dfine":
        endpoints["dfine"] = str(cfg.detector.url)
        if bool(getattr(cfg.detector, "use_sam", True)):
            endpoints["mobile_sam"] = str(cfg.detector.sam_url)
    if str(getattr(cfg.exploration, "value_model", "clip")) == "blip2itm" and bool(cfg.exploration.value_map):
        endpoints["blip2itm"] = str(getattr(cfg.exploration, "value_blip2_url", "http://localhost:13182/blip2itm"))
    if (str(getattr(cfg.agent, "policy", "")) == "ascentnav"
            and str(getattr(cfg.agent, "stair_up_mode", "ascent")) == "ascent"):
        endpoints["gdino"] = str(getattr(cfg.detector, "gdino_url", "http://localhost:13184/gdino"))
    if bool(getattr(cfg.exploration, "ram_tags", False)):
        endpoints["ram"] = str(getattr(cfg.exploration, "ram_url", "http://localhost:13185/ram"))
    if endpoints and bool(getattr(cfg.detector, "strict", False)):
        probe_perception_servers(endpoints)


def build_agent(
    cfg, components: dict, target: str, *, keyframe_dir=None, profiler=None,
    nav_fn=None, reachable_fn=None, nearest_navigable_fn=None, climb_nav_fn=None,
):
    """Build one policy behind the common ``act(frame)`` contract."""
    policy = components["policy"]
    extra = {}
    if policy == "nav_agent":
        from ..agent.nav_agent import NavAgent

        agent_cls = NavAgent
        extra["nearest_navigable_fn"] = nearest_navigable_fn
        extra["climb_nav_fn"] = climb_nav_fn
    elif climb_nav_fn is not None:
        raise ValueError("agent.climb_oracle is implemented for the nav_agent policy only")
    elif policy == "ascent":
        from ..agent.ascent_agent import AscentAgent

        agent_cls = AscentAgent
    else:
        from navigation.agent import AscentNavAgent

        agent_cls = AscentNavAgent
        extra["world_model"] = build_world_model(
            cfg, target, keyframe_dir=keyframe_dir, profiler=profiler,
            room_classifier=components.get("room_classifier"),
            stair_detector=components.get("stair_detector"),
        )
    return agent_cls(
        cfg, components["detector"], components["scorer"],
        components["verifier"], target, keyframe_dir=keyframe_dir,
        profiler=profiler, nav_fn=nav_fn, reachable_fn=reachable_fn,
        pointnav=components["pointnav"], ranker=components["ranker"], **extra,
        floor_planner=components["floor_planner"],
        room_classifier=components["room_classifier"],
        image_text=components["image_text"],
        feature_memory=components.get("feature_memory"),
        region_proposer=components.get("region_proposer"),
        stair_segmenter=components["stair_segmenter"],
        stair_detector=components.get("stair_detector"),
        ram=components.get("ram"),
    )


def build_world_model(cfg, target, *, keyframe_dir=None, profiler=None,
                      room_classifier=None, stair_detector=None):
    """OSG's world model for a policy that does not have one, or None.

    Its detector is built here rather than reused from `components`: the
    driving policy's detector answers a different question (see
    `agent/world_model.py`).
    """
    if not bool(getattr(cfg.agent, "osg_world_model", False)):
        return None
    from ..agent.world_model import WorldModel
    from ..perception.detector import YoloeDetector

    detector = YoloeDetector(
        weights=str(getattr(cfg.agent, "osg_world_model_weights",
                            "data/weights/yoloe-11s-seg.pt")),
        conf=cfg.detector.conf if str(cfg.detector.name) == "yoloe" else 0.3,
        class_conf=dict(cfg.detector.class_conf or {}),
        imgsz=cfg.detector.imgsz,
        half=cfg.detector.half,
        device=cfg.detector.device,
    )
    detector.set_vocabulary(
        [target.replace("_", " ")] + list(getattr(cfg.detector, "vocabulary", []) or [])
    )
    return WorldModel(
        cfg, detector, target=target, keyframe_dir=keyframe_dir,
        profiler=profiler, room_classifier=room_classifier,
        stair_detector=stair_detector,
    )


def build_region_proposer(cfg):
    """The class-agnostic proposal stage, or None when it is off."""
    from ..perception.region_proposer import build_region_proposer as _build

    return _build(cfg)


def build_feature_memory(cfg):
    """The appearance matcher, or None when it is off (nothing is loaded)."""
    from ..objects.feature_memory import FeatureMemory
    from ..perception.feature_encoder import build_feature_encoder

    encoder = build_feature_encoder(cfg)
    return None if encoder is None else FeatureMemory(encoder, cfg.feature_memory)


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
        (getattr(cfg.verification, "base_url", "") or cfg.llm.base_url),
        cfg.verification.vlm_model,
        (getattr(cfg.verification, "api_key", "") or cfg.llm.api_key),
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
