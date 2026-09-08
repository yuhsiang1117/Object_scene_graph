"""Episode loop: builds the agent stack from config, runs HM3D ObjectNav
episodes, writes per-episode jsonl + aggregate summary + timing + trajectory
visualizations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..agent.nav_agent import NavAgent
from ..core.config import resolve_navigation
from ..core.profiler import Profiler
from ..exploration.async_scorer import AsyncScorer
from ..exploration.llm_scorer import LLMTextScorer
from ..llm.client import ChatClient
from ..perception.room_classifier import build_room_classifier
from ..mapping.costmap import PLANE
from .metrics import aggregate, per_category
from .visualize import overlay_segmentation, render_costmap_bgr, save_topdown


class _DebugVideo:
    """Per-episode debug video: each frame is [live RGB + YOLOE segmentation
    overlay | top-down costmap] at every step. The detector is re-run here for
    visualization only (it does not feed the object layer), so pipeline
    behaviour / SR is unchanged. Enabled by eval.debug_frames."""

    def __init__(self, cfg, out_dir: Path, episode_id) -> None:
        import cv2

        from ..mapping.costmap import PLANE as _PLANE

        self._cv2 = cv2
        self._plane = list(_PLANE)
        self._cm_w = 480
        self._h = cfg.eval.rgb_height
        self._w = cfg.eval.rgb_width + self._cm_w
        self._traj: list = []
        # Scene-qualified: habitat restarts episode_id at "0" in every per-scene
        # content file, so `ep3.mp4` collided across scenes and each split
        # silently kept only the last one.
        self._path = out_dir / "viz" / "debug" / f"{str(episode_id).replace('/', '_')}.mp4"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Opened lazily: an agent that renders its own panel decides the frame
        # size, and only it knows what that is.
        self._vw = None

    def _writer(self, size):
        if self._vw is None:
            self._vw = self._cv2.VideoWriter(
                str(self._path), self._cv2.VideoWriter_fourcc(*"mp4v"), 8, size)
        return self._vw

    def write(self, frame, agent, target: str, detector) -> None:
        cv2 = self._cv2
        # An agent that draws its own maps renders itself -- `AscentNavAgent`
        # holds ASCENT's obstacle/value maps, which the costmap path below
        # cannot show.
        if hasattr(agent, "debug_panel"):
            panel = agent.debug_panel(frame)
            self._writer((panel.shape[1], panel.shape[0])).write(panel)
            return
        agent_xy = frame.camera_position[self._plane]
        self._traj.append(agent_xy)
        dets = detector.detect(frame.rgb)  # viz-only; does not update object layer
        seg = overlay_segmentation(frame.rgb, dets, target)
        seg = cv2.resize(seg, (self._w - self._cm_w, self._h))
        cm = render_costmap_bgr(
            agent.costmap, agent_xy, self._traj,
            path_xy=getattr(agent, "_current_path", None),
            chosen_frontier=getattr(agent, "_current_frontier", None),
            out_h=self._h,
        )
        cm = cv2.resize(cm, (self._cm_w, self._h))
        panel = cv2.hconcat([seg, cm])
        cv2.putText(panel, f"{target}  step {len(self._traj)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        self._writer((self._w, self._h)).write(panel)

    def close(self) -> None:
        if self._vw is not None:
            self._vw.release()


def build_scorer(cfg) -> AsyncScorer:
    # Geometric-only exploration (no LLM): nearest frontier weighted by
    # exploration range (info gain). select_frontier falls back to
    # unscored_prior for every frontier.
    mode = str(getattr(cfg.exploration, "frontier_text_scorer", "disabled"))
    if mode == "disabled":
        from ..exploration.scorer import NullScorer
        return AsyncScorer(NullScorer())
    if mode != "llm_text":
        # The old field accepted "nearest"/"geometric"/"none" as synonyms for
        # off and silently treated anything else ("vlm", "random", a typo) as
        # on. Fail loudly instead: a stale override is exactly the kind of
        # silently-inert setting this log has been bitten by repeatedly.
        raise ValueError(
            f"exploration.frontier_text_scorer={mode!r} is not recognised; "
            "expected 'disabled' or 'llm_text' (this field was renamed from "
            "`scorer`, whose 'nearest'/'geometric'/'none' all meant 'off')"
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
    """ASCENT's forced-choice frontier ranker, or None to keep the async scorer.

    The knowledge graph is attached later by NavAgent, which already loads one;
    building a second copy here would double the file read per episode.
    """
    if str(getattr(cfg.exploration, "ranker", "none")) != "ascent":
        return None
    from ..exploration.ascent_ranker import AscentFrontierRanker

    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    return AscentFrontierRanker(
        client,
        topk=int(getattr(cfg.exploration, "ranker_topk", 3)),
        subgraph_radius_m=cfg.exploration.subgraph_radius_m,
    )


def build_floor_planner(cfg):
    """The coarse level of the cascade, or None to keep floor changes geometric."""
    if not bool(getattr(cfg.exploration, "floor_llm", False)):
        return None
    from ..exploration.floor_planner import FloorDecisionPlanner
    from ..exploration.knowledge_prior import FloorPrior, KnowledgeGraph

    client = ChatClient(
        cfg.llm.base_url, cfg.llm.text_model, cfg.llm.api_key,
        cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
    )
    # Both priors are loaded regardless of exploration.knowledge_prior: that
    # flag turns on a numeric multiplier applied outside the model, while these
    # go into the prompt as text.
    try:
        floor_prior = FloorPrior.load()
    except OSError:
        floor_prior = None
    try:
        kg = KnowledgeGraph.load()
    except OSError:
        kg = None
    return FloorDecisionPlanner(
        client, floor_prior=floor_prior, kg=kg,
        ask_every_steps=int(getattr(cfg.exploration, "floor_ask_every", 60)),
        min_steps_on_floor=int(getattr(cfg.exploration, "floor_min_steps", 100)),
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
        choice_mode=getattr(cfg.verification, "choice_mode", True),
    )


def build_detector(cfg):
    if cfg.detector.name == "yoloe":
        from ..perception.detector import YoloeDetector

        return YoloeDetector(
            weights=cfg.detector.weights,
            conf=cfg.detector.conf,
            imgsz=cfg.detector.imgsz,
            half=cfg.detector.half,
            device=cfg.detector.device,
        )
    if cfg.detector.name == "stub":
        from ..perception.detector import StubDetector

        return StubDetector()
    raise ValueError(f"unknown detector: {cfg.detector.name}")


def _unload_ollama_models(cfg) -> None:
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


def _goal_floor_gap_m(episode) -> Optional[float]:
    """Height difference between the episode START and the nearest goal
    view-point, i.e. how many storeys the agent must actually climb.

    This is the GROUND-TRUTH multi-/single-floor label, and the only correct
    one for A/B splitting. Two weaker labels are tempting and both wrong:

    * the SCENE's floor span -- a multi-storey house is full of episodes whose
      goal is on the starting floor (measured: only 15/50 of a scene-level
      "multi-floor" split actually requires a floor change);
    * the agent's OBSERVED trajectory span -- an episode where the agent should
      have climbed but never did reads as single-floor, which hides exactly the
      failures the multi-floor work targets.
    """
    ys = [
        float(vp.agent_state.position[1])
        for goal in (episode.goals or [])
        for vp in (getattr(goal, "view_points", None) or [])
    ]
    if not ys:
        return None
    start_y = float(episode.start_position[1])
    return min(abs(y - start_y) for y in ys)


def _episode_uid(episode) -> str:
    """Globally unique episode key, "<scene>:<episode_id>".

    habitat's ObjectNav loader assigns episode_id per CONTENT FILE
    (object_nav_dataset.py: `episode.episode_id = str(i)` inside the per-scene
    loop), so every scene has an episode "0". Any code that subsets, pairs, or
    joins episodes across runs must key on the scene as well -- see
    scripts/compare_runs.py."""
    return f"{str(episode.scene_id).split('/')[-1]}:{episode.episode_id}"


def _target_track_fields(agent) -> dict:
    """Snapshot the committed target track for GT-localization analysis.

    _target_obj_xy is set (in _start_approach) only once a candidate is
    accepted into APPROACH, so it is None for episodes that never committed to
    a target (pure exploration failures) -- recorded as None there."""
    obj_xy = getattr(agent, "_target_obj_xy", None)
    cloud_xy = getattr(agent, "_target_cloud_xy", None)
    cand_id = getattr(agent, "_candidate_id", None)
    track = agent.object_layer.get(cand_id) if cand_id is not None else None
    best_cam = getattr(track, "best_cam_xy", None) if track is not None else None
    return {
        "target_obj_xy": [float(x) for x in obj_xy] if obj_xy is not None else None,
        # What ASCENT would have aimed at: the observed surface point nearest
        # the agent, rather than the fitted ellipsoid centre above.
        "target_cloud_xy": [float(x) for x in cloud_xy] if cloud_xy is not None else None,
        "cand_best_cam_xy": [float(x) for x in best_cam] if best_cam is not None else None,
        "cand_best_score": float(track.best_score) if track is not None else None,
        "cand_n_obs": int(track.n_obs) if track is not None else None,
    }


def run_eval(cfg) -> dict:
    from ..sim.habitat_env import HabitatObjectNavEnv

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolved once, before anything expensive is built, so a bad or
    # self-contradictory navigation setting fails in a second rather than after
    # the simulator and three models have loaded.
    _navigation = resolve_navigation(cfg.agent)

    _unload_ollama_models(cfg)
    env = HabitatObjectNavEnv(cfg)
    detector = build_detector(cfg)
    scorer = build_scorer(cfg)
    verifier = build_verifier(cfg)
    ranker = build_ranker(cfg)
    room_classifier = build_room_classifier(cfg)
    floor_planner = build_floor_planner(cfg)
    from ..perception.image_text import build_image_text_scorer

    image_text = build_image_text_scorer(cfg)
    # Built once for the whole run, like the detector and scorer above -- a
    # NavAgent is constructed per episode and this loads a 34 MB checkpoint.
    pointnav = None
    if _navigation == "pointnav":
        from ..planning.pointnav_driver import build_pointnav

        pointnav = build_pointnav(cfg)
    from ..perception.stair_seg import build_stair_segmenter

    stair_segmenter = build_stair_segmenter(cfg)
    if verifier is not None and cfg.eval.debug_frames:
        verifier.debug_dir = str(out_dir / "verify_debug")

    # Episode-id subsetting (the dev50 / dev50_mf A/B splits). Applied by
    # FILTERING THE DATASET, not by skipping after env.reset(): habitat's
    # episode_id restarts at "0" in every per-scene content file
    # (object_nav_dataset.py: `episode.episode_id = str(i)` inside the per-file
    # loop), so a bare id is ambiguous across scenes AND reset() loads a scene
    # before we could skip it -- subsetting 50 of ~2000 episodes that way costs
    # ~2000 scene loads. Ids are therefore scene-qualified, "<scene>:<id>".
    if cfg.eval.episode_ids:
        wanted = {str(e) for e in cfg.eval.episode_ids}
        kept = [ep for ep in env.env.episodes if _episode_uid(ep) in wanted]
        if not kept:
            sample = sorted(_episode_uid(ep) for ep in env.env.episodes[:3])
            raise ValueError(
                f"eval.episode_ids matched 0 of {len(env.env.episodes)} episodes. "
                f"Ids must be scene-qualified '<scene>:<episode_id>', e.g. {sample}. "
                f"Got e.g. {sorted(wanted)[:3]}."
            )
        if len(kept) != len(wanted):
            missing = sorted(wanted - {_episode_uid(ep) for ep in kept})
            print(f"[warn] {len(missing)} requested episode ids not in split: {missing[:5]}")
        env.env.episodes = kept

    n_total = len(env.env.episodes)
    n_run = n_total if cfg.eval.num_episodes < 0 else min(cfg.eval.num_episodes, n_total)

    results = []
    episodes_file = out_dir / "episodes.jsonl"
    profiler_all = Profiler()

    for ep_i in range(n_run):
        frame = env.reset()
        episode = env.current_episode
        target = env.target_category()
        if verifier is not None:
            verifier.debug_tag = f"ep{episode.episode_id}"

        # scorer/verifier are built once and shared across every episode in
        # this run, so their call/error counters are cumulative — snapshot
        # before and report deltas below, or every episode after the first
        # would show the whole run's running total instead of its own.
        llm_calls_before, llm_errors_before = scorer.n_calls, scorer.n_errors
        llm_last_error_before = scorer.last_error
        verify_calls_before = verifier.n_calls if verifier is not None else 0
        verify_errors_before = verifier.n_errors if verifier is not None else 0

        # frontier.id/room.id both restart from 0/1 each episode (fresh
        # FrontierExtractor/RoomSegmenter per NavAgent below), but the
        # scorer's internal caches are keyed by those same small ints and
        # persist across the whole run -- without this, a new episode can
        # silently inherit a stale score/room-label from a previous,
        # unrelated scene the moment an id collides.
        scorer.reset()
        if image_text is not None:
            image_text.reset()

        profiler = Profiler()
        _policy = str(getattr(cfg.agent, "policy", "nav_agent"))
        _Agent = NavAgent
        if _policy == "ascent":
            from ..agent.ascent_agent import AscentAgent

            _Agent = AscentAgent
        elif _policy == "ascentnav":
            # ASCENT's own maps under ASCENT's control flow -- a separate
            # package, not a NavAgent subclass. See src/ascentnav/agent.py.
            from ascentnav.agent import AscentNavAgent

            _Agent = AscentNavAgent
        elif _policy != "nav_agent":
            raise ValueError(
                f"agent.policy={_policy!r} is not nav_agent | ascent | ascentnav")
        agent = _Agent(
            cfg, detector, scorer, verifier, target,
            keyframe_dir=str(out_dir / "keyframes" / f"ep{episode.episode_id}")
            if cfg.eval.save_viz else None,
            profiler=profiler,
            # The two privileged channels. Handed over ONLY in navmesh mode --
            # `pointnav` and `costmap` receive neither, which is what makes
            # their numbers comparable to ascent's (docs/AB_RESULTS.md, S8).
            nav_fn=env.action_to_goal if _navigation == "navmesh" else None,
            reachable_fn=env.is_reachable if _navigation == "navmesh" else None,
            image_text=image_text,
            ranker=ranker,
            room_classifier=room_classifier,
            floor_planner=floor_planner,
            pointnav=pointnav,
            stair_segmenter=stair_segmenter,
        )
        trajectory = [frame.camera_position[list(PLANE)]]
        # Height track, kept separately from `trajectory` (which is ground-plane
        # only): its span is how a run is split into single- vs multi-floor
        # episodes for every A/B, without needing GT scene metadata.
        traj_y = [float(frame.camera_position[1])]
        dbg = _DebugVideo(cfg, out_dir, _episode_uid(episode)) if cfg.eval.debug_frames else None
        t0 = time.time()
        steps = 0
        while not env.episode_over:
            action = agent.act(frame)  # updates agent.costmap from `frame`
            if dbg is not None:
                dbg.write(frame, agent, target, detector)
            frame = env.step(action)
            trajectory.append(frame.camera_position[list(PLANE)])
            traj_y.append(float(frame.camera_position[1]))
            steps += 1
        if dbg is not None:
            dbg.close()

        m = env.metrics()
        rec = {
            # Scene-qualified key. episode_id alone repeats across scenes (see
            # _episode_uid), so this is what compare_runs.py pairs two runs on.
            "uid": _episode_uid(episode),
            "episode_id": str(episode.episode_id),
            "scene": str(episode.scene_id).split("/")[-1],
            "target": target,
            "success": float(m.get("success", 0.0)),
            "spl": float(m.get("spl", 0.0)),
            "distance_to_goal": float(m.get("distance_to_goal", -1.0)),
            "steps": steps,
            "wall_time_s": round(time.time() - t0, 1),
            "control_fps": round(profiler.fps("control_loop"), 2),
            "llm_calls": scorer.n_calls - llm_calls_before,
            "llm_errors": scorer.n_errors - llm_errors_before,
            "llm_last_error": scorer.last_error if scorer.last_error != llm_last_error_before else None,
            "agent_stats": agent.stats,
            "state_log": agent.state_log[:40],
            "frontier_select_log": agent.frontier_select_log,
            "giveup_log": agent.giveup_log[:50],
            "approach_bbox_log": agent.approach_bbox_log,
            "approach_stop_reason": agent.approach_stop_reason,
            "approach_recheck_max": agent.approach_recheck_max,
            "approach_recheck_n": agent._approach_itm_n,
            "approach_diag": agent.approach_diag,
            "final_xy": [float(x) for x in trajectory[-1]],
            "verify_calls": (verifier.n_calls - verify_calls_before) if verifier is not None else 0,
            "verify_errors": (verifier.n_errors - verify_errors_before) if verifier is not None else 0,
            # --- A/B mechanism instrumentation -------------------------------
            # At n=50 (the A/B split size) SR has a ~7pp standard error, so a
            # single stage's effect is rarely visible in SR alone. These are the
            # per-mechanism signals each stage is actually judged on; they are
            # far more stable than SR at that sample size. Counters that a later
            # stage populates read 0/None until then -- the field is emitted
            # unconditionally so scripts/compare_runs.py can diff any two runs
            # without knowing which stages were enabled.
            #
            # Height span of the trajectory. > ~1 m means the agent changed
            # floor (storeys are ~2.5 m); this is the multi-/single-floor split
            # used by every A/B, derived from the run itself rather than GT.
            "y_range_m": round(float(max(traj_y) - min(traj_y)), 3),
            # Ground-truth storeys between start and goal. THIS is what the
            # multi-/single-floor A/B split must key on -- see
            # _goal_floor_gap_m for why y_range_m and the scene span are not.
            "goal_floor_gap_m": (
                round(gap, 3) if (gap := _goal_floor_gap_m(episode)) is not None else None
            ),
            "steps_to_first_candidate": agent.steps_to_first_candidate,
            # Per-step trace, only when debug frames are on: it is ~20 numbers a
            # step, which would bloat every normal run's episodes.jsonl.
            **({"step_trace": agent.step_trace}
               if cfg.eval.debug_frames and hasattr(agent, "step_trace") else {}),
            **{k: agent.stats.get(k, 0) for k in (
                "frontier_consumed",     # S30: explored away mid-pursuit
                "frontier_switch",                       # S34
                "stair_rounds", "stair_frontiers_seen",   # S33: where the
                "stair_frontier_selected",                # stair gain is lost
                "climb_attempt", "climb_forced_forward",  # S31
                "down_look",             # S32
                "pointnav_stop_forced_forward",  # S30
                "approach_abandon",      # S8: the sensor-only reachability test
                "unreachable_skip",      # S8: its privileged counterpart
                "frontier_give_up",
                "frames_off_plane",      # S1
                "n_floors", "floor_switches", "cross_floor_candidate",  # S2
                "climb_ok", "climb_fail",  # S3
                "fp_retract",            # S5
                "value_calls",           # S4
            )},
            # GT-localization instrumentation: the mapped 3D center (x-z) of the
            # object track the agent committed to APPROACH, plus that track's
            # best-detection camera pose and score. d(target_obj_xy, GT goal)
            # is the scene-graph localization error; d(final_xy, target_obj_xy)
            # is the residual navigation error -- together they split "stopped
            # far from goal" into mislocalized-track vs failed-nav vs false-
            # positive detection (scripts/analyze_localization.py).
            **_target_track_fields(agent),
        }
        results.append(rec)
        with open(episodes_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for name, samples in profiler._samples.items():
            for s in samples:
                profiler_all.add(name, s)

        if getattr(cfg.eval, "save_costmap", False):
            # The final occupancy grid, so room segmentation can be swept
            # offline. Sweeping in-process would need a full episode per
            # parameter setting; one 40 KB grid per episode buys the whole grid
            # search. See scripts/measure_room_seg.py.
            cm_dir = out_dir / "costmaps"
            cm_dir.mkdir(parents=True, exist_ok=True)
            cm = agent.costmap
            np.savez_compressed(
                cm_dir / f"{rec['uid'].replace('/', '_')}.npz",
                grid=cm.grid, resolution=cm.resolution, origin=cm.origin,
                steps=rec["steps"], target=target,
            )

        if cfg.eval.save_viz:
            save_topdown(
                str(out_dir / "viz" / f"ep{episode.episode_id}.png"),
                agent.costmap,
                trajectory,
                scene_graph=agent.scene_graph,
                title=f"ep {episode.episode_id} target={target} "
                f"success={rec['success']:.0f} spl={rec['spl']:.2f}",
            )
        print(
            f"[{ep_i + 1}/{n_run}] ep={episode.episode_id} target={target} "
            f"success={rec['success']:.0f} spl={rec['spl']:.3f} steps={steps} "
            f"fps={rec['control_fps']}"
        )

    summary = {
        # Full protocol fingerprint. A number is only comparable to the ASCENT
        # baseline (63% SR) if this block matches configs/experiment/
        # ascent_matched.yaml -- check it before quoting any SR. The values are
        # read back off the habitat config actually handed to the simulator, not
        # off our own cfg, so an override that failed to apply shows up here.
        "config": {
            "frontier_text_scorer": cfg.exploration.frontier_text_scorer,
            "verification": cfg.verification.enabled,
            # name alone is "yoloe" for BOTH the 11s@512 and 11l@640 configs,
            # so it cannot tell the laptop profile from the report one. Every
            # number in docs/AB_RESULTS predates this and was taken on whichever
            # weights happened to be present.
            "detector": cfg.detector.name,
            "detector_weights": str(cfg.detector.weights),
            "detector_imgsz": int(cfg.detector.imgsz),
            "dataset_version": cfg.eval.dataset_version,
            "split": cfg.eval.split,
            "success_distance": float(
                env.hab_cfg.habitat.task.measurements.success.success_distance
            ),
            "max_steps": int(env.hab_cfg.habitat.environment.max_episode_steps),
            "allow_sliding": bool(env.hab_cfg.habitat.simulator.habitat_sim_v0.allow_sliding),
            "shuffle": bool(env.hab_cfg.habitat.environment.iterator_options.shuffle),
            "max_scene_repeat_steps": int(
                env.hab_cfg.habitat.environment.iterator_options.max_scene_repeat_steps
            ),
            "max_scene_repeat_episodes": int(
                env.hab_cfg.habitat.environment.iterator_options.max_scene_repeat_episodes
            ),
            "use_habitat_navmesh": bool(getattr(cfg.agent, "use_habitat_navmesh", False)),
            "navigation": _navigation,
            "n_episodes": len(results),
            "n_episodes_in_split": n_total,
            "seed": cfg.seed,
        },
        # The ALGORITHM configuration, kept separate from the evaluation
        # protocol above. Without it a run's artifacts record how it was scored
        # but not what was actually run: two arms differing only in
        # `value_weight` or `knowledge_prior` produce byte-identical "config"
        # blocks, and the only record of which was which lives in whatever shell
        # command happened to launch it. Every A/B in docs/AB_RESULTS predates
        # this and had to be reconstructed from the command line.
        "algorithm": {
            # `detector` in the protocol block is "yoloe" for BOTH the 11s@512
            # and the 11l@640 config, so it cannot tell the laptop profile from
            # the report one. Every number in docs/AB_RESULTS predates this and
            # was taken on whichever weights the container had -- 11s@512.
            "detector_weights": str(cfg.detector.weights),
            "detector_imgsz": int(cfg.detector.imgsz),
            "extractor": str(getattr(cfg.exploration, "extractor", "wfd")),
            "area_thresh_m2": float(getattr(cfg.exploration, "area_thresh_m2", 1.5)),
            "frontier_min_cells": int(cfg.exploration.frontier_min_cells),
            "selector": str(getattr(cfg.exploration, "selector", "utility")),
            "frontier_commit": bool(getattr(cfg.exploration, "frontier_commit", False)),
            "nearby_distance_m": float(getattr(cfg.exploration, "nearby_distance_m", 3.0)),
            "info_gain_weight": float(cfg.exploration.info_gain_weight),
            "continuity_weight": float(cfg.exploration.continuity_weight),
            "los_visibility_penalty": float(cfg.exploration.los_visibility_penalty),
            "value_map": bool(getattr(cfg.exploration, "value_map", False)),
            "value_model": str(getattr(cfg.exploration, "value_model", "clip")),
            "value_weight": float(getattr(cfg.exploration, "value_weight", 1.0)),
            "value_argmax": bool(getattr(cfg.exploration, "value_argmax", False)),
            "knowledge_prior": bool(getattr(cfg.exploration, "knowledge_prior", False)),
            "knowledge_weight": float(getattr(cfg.exploration, "knowledge_weight", 1.0)),
            "select_every": int(getattr(cfg.exploration, "select_every", 5)),
            "reselect_every": int(getattr(cfg.exploration, "reselect_every", 0)),
            "frontier_desc": str(
                getattr(cfg.exploration, "frontier_desc", "graph")),
            "frontier_desc_match_m": float(
                getattr(cfg.exploration, "frontier_desc_match_m", 1.0)),
            "ranker": str(getattr(cfg.exploration, "ranker", "none")),
            "ranker_topk": int(getattr(cfg.exploration, "ranker_topk", 3)),
            "ranker_every_steps": int(getattr(cfg.exploration, "ranker_every_steps", 20)),
            "floor_llm": bool(getattr(cfg.exploration, "floor_llm", False)),
            "floor_ask_every": int(getattr(cfg.exploration, "floor_ask_every", 60)),
            "floor_min_steps": int(getattr(cfg.exploration, "floor_min_steps", 100)),
            "floor_llm_boost": float(getattr(cfg.exploration, "floor_llm_boost", 5.0)),
            "room_classifier": str(getattr(cfg.scene_graph, "room_classifier", "none")),
            "room_erode_iters": int(getattr(cfg.scene_graph, "room_erode_iters", 6)),
            "multi_floor": bool(getattr(cfg.mapping, "multi_floor", False)),
            "stair_prior": float(getattr(cfg.exploration, "stair_prior", 0.0)),
            "freeze_floor_on_stairs": bool(
                getattr(cfg.mapping, "freeze_floor_on_stairs", False)
            ),
            "freeze_floor_in_climb": bool(
                getattr(cfg.mapping, "freeze_floor_in_climb", False)
            ),
            "climb_exit_rule": str(getattr(cfg.agent, "climb_exit_rule", "height")),
            "stair_exit_m": float(getattr(cfg.agent, "stair_exit_m", 0.5)),
            "floor_gap_min_m": float(getattr(cfg.agent, "floor_gap_min_m", 0.9)),
            "stair_min_hits": int(getattr(cfg.exploration, "stair_min_hits", 1)),
            "stair_min_cells": int(getattr(cfg.exploration, "stair_min_cells", 25)),
            "stair_explored_rule": str(
                getattr(cfg.exploration, "stair_explored_rule", "no_frontiers")
            ),
            # Navigation source. This is the S8 variable and it lived only in
            # the "config" block, which compare_runs does not diff -- so an S8
            # A/B would have reported "algorithm config: IDENTICAL".
            "use_habitat_navmesh": bool(
                getattr(cfg.agent, "use_habitat_navmesh", False)),
            # The resolved mover, which is what actually drove -- `navigation`
            # and `use_habitat_navmesh` are two spellings of one setting and
            # only this says which won.
            "navigation": _navigation,
            "policy": str(getattr(cfg.agent, "policy", "nav_agent")),
            "ascent_min_obstacle_h": float(
                getattr(cfg.agent, "ascent_min_obstacle_h", 0.61)),
            "ascent_max_obstacle_h": float(
                getattr(cfg.agent, "ascent_max_obstacle_h", 0.88)),
            "pointnav_weights": str(getattr(cfg.agent, "pointnav_weights", "")),
            "pointnav_stop_radius": float(
                getattr(cfg.agent, "pointnav_stop_radius", 0.9)),
            "pointnav_arrival_m": float(getattr(cfg.agent, "pointnav_arrival_m", 0.0)),
            "pointnav_approach_creep_m": float(
                getattr(cfg.agent, "pointnav_approach_creep_m", 1.0)),
            "pointnav_depth_shape": [
                int(v) for v in getattr(cfg.agent, "pointnav_depth_shape", [224, 224])],
            "approach_abandon_steps": int(
                getattr(cfg.agent, "approach_abandon_steps", 0)),
            "escape_window": int(getattr(cfg.agent, "escape_window", 0)),
            "commit_gate": bool(getattr(cfg.agent, "commit_gate", False)),
            "scan_on_arrival": int(getattr(cfg.agent, "scan_on_arrival", 0)),
            "terminal_requires_detection": bool(
                getattr(cfg.agent, "terminal_requires_detection", True)),
            "pointnav_stop_means_blocked": bool(
                getattr(cfg.agent, "pointnav_stop_means_blocked", True)),
            "frontier_reachability_gate": bool(
                getattr(cfg.agent, "frontier_reachability_gate", True)),
            "climb_carrot": bool(getattr(cfg.agent, "climb_carrot", False)),
            "climb_carrot_m": float(getattr(cfg.agent, "climb_carrot_m", 0.8)),
            "down_look_every": int(getattr(cfg.agent, "down_look_every", 0)),
            "stair_up_mode": str(getattr(cfg.agent, "stair_up_mode", "detector")),
            "rednet_stairs": bool(getattr(cfg.agent, "rednet_stairs", False)),
            "frontier_stick_m": float(getattr(cfg.agent, "frontier_stick_m", 0.2)),
            "frontier_stick_steps": int(
                getattr(cfg.agent, "frontier_stick_steps", 15)),
            "frontier_stick_rule": str(
                getattr(cfg.agent, "frontier_stick_rule", "displacement")),
            "approach_navigable_goal": bool(
                getattr(cfg.agent, "approach_navigable_goal", False)),
            "terminal_rule": str(getattr(cfg.agent, "terminal_rule", "depth")),
            "terminal_stop_m": float(getattr(cfg.agent, "terminal_stop_m", 0.6)),
            "terminal_percentile": float(getattr(cfg.agent, "terminal_percentile", 0.0)),
            "fp_retraction": bool(getattr(cfg.scene_graph, "fp_retraction", False)),
            "verify_reject_cooldown_steps": int(
                getattr(cfg.verification, "reject_cooldown_steps", 0)),
            "verify_min_obs": int(cfg.verification.min_obs),
            "verify_min_score": float(cfg.verification.min_score),
            "verify_min_bbox_px": int(cfg.verification.min_bbox_px),
            "verify_accept_confidence": float(cfg.verification.accept_confidence),
            "verify_choice_mode": bool(getattr(cfg.verification, "choice_mode", True)),
            "verify_terminal": bool(getattr(cfg.verification, "terminal", False)),
            "verify_approach_recheck": bool(
                getattr(cfg.verification, "approach_recheck", False)),
            "verify_approach_recheck_thresh": float(
                getattr(cfg.verification, "approach_recheck_thresh", 0.0)),
            "verify_center_before_verify": bool(
                getattr(cfg.verification, "center_before_verify", True)
            ),
            "text_model": str(cfg.llm.text_model),
            "vlm_model": str(cfg.verification.vlm_model),
        },
        "metrics": aggregate(results),
        "per_category": per_category(results),
        "timing": profiler_all.report(),
        "pipeline_fps": round(profiler_all.fps("control_loop"), 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    profiler_all.write_csv(str(out_dir / "timing.csv"))
    scorer.shutdown()
    env.close()
    print(json.dumps(summary["metrics"], indent=2))
    return summary
