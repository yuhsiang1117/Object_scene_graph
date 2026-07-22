"""Structured configs registered with Hydra's ConfigStore so that typos in
yaml/CLI overrides fail fast instead of silently creating new keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ~40 common indoor categories used as the fixed detector vocabulary in
# addition to the episode target. HM3D ObjectNav v2 targets are a subset.
DEFAULT_VOCABULARY: List[str] = [
    "chair", "sofa", "armchair", "plant", "bed", "toilet", "tv_monitor",
    "table", "desk", "cabinet", "shelf", "dresser", "wardrobe", "nightstand",
    "lamp", "pillow", "cushion", "picture", "mirror", "window", "door",
    "sink", "bathtub", "shower", "towel", "counter", "stool", "bench",
    "refrigerator", "oven", "microwave", "washing machine", "stove",
    "fireplace", "stairs", "rug", "curtain", "clothes", "book", "box", "basket",
]


@dataclass
class AgentConfig:
    max_steps: int = 500
    forward_m: float = 0.25
    turn_deg: float = 30.0
    success_distance: float = 0.1  # paper mode: 0.13
    initial_scan: bool = True  # 360 deg spin at episode start to seed the map
    camera_height: float = 0.88
    agent_radius: float = 0.18
    # Terminal APPROACH phase: walk toward the verified object while a
    # detection stays visible, stopping once its bbox is large enough (a
    # borderline "object recognizable but distant" crop measured ~25k px^2
    # in verify_debug samples; this threshold asks for a noticeably closer
    # view than that before considering the approach complete).
    approach_stop_bbox_px: float = 40_000.0
    approach_max_steps: int = 12  # ~3 m of travel at forward_m=0.25
    # Tighter-than-default planner/controller stopping precision for the
    # final APPROACH segment only (P1f). HM3D success is a geodesic
    # distance to a view_point; the general 0.3 m (planner) / 0.2 m
    # (controller) tolerances used for frontier/verify-view travel left
    # enough slack that a short geodesic detour around a nearby thin
    # obstacle (wall corner, furniture edge) blew the 0.13 m success
    # radius on episodes where we were already 5-8 cm away in a straight
    # line. Kept above the 0.05 m costmap resolution to stay robust to
    # grid discretization.
    approach_goal_tolerance_m: float = 0.12
    approach_arrival_tol_m: float = 0.1


@dataclass
class DetectorConfig:
    name: str = "yoloe"
    weights: str = "data/weights/yoloe-11s-seg.pt"
    conf: float = 0.3
    imgsz: int = 512
    half: bool = True
    device: str = "cuda"
    vocabulary: List[str] = field(default_factory=lambda: list(DEFAULT_VOCABULARY))


@dataclass
class SceneGraphConfig:
    keyframe_trans_m: float = 0.25
    keyframe_rot_deg: float = 30.0
    min_obs_for_refine: int = 3
    refine_every: int = 3
    link_dist_m: float = 1.0
    near_edge_dist_m: float = 1.5
    assoc_score_thresh: float = 0.4
    assoc_depth_gate_m: float = 0.5
    # Wasserstein data association requires the detection label to match the
    # track label. The ported VOOM matcher had no label check, but for SR eval
    # (navigate to a target CATEGORY) cross-category merges corrupt labels and
    # starve target candidates -- so gate on category by default.
    assoc_category_gate: bool = True
    room_seg_every_kf: int = 10
    room_min_radius_m: float = 0.9
    # 1.2 caused universal 1-room collapse on real HM3D scans: the merge
    # condition is clearance > door_width_m/2, so a wider value RAISES the
    # threshold and preserves more boundaries. 2.0 is where a 10-episode
    # sweep on real explored costmaps saturates (matches the measured
    # 0.85m/0.934m boundary clearances in the multi-room episodes).
    room_door_width_m: float = 2.0
    # Node-creation quality gate (P1h/orphan-node follow-up): a real diagnostic
    # run showed ~228 tracks/episode with 36% never re-observed and 49% never
    # reaching min_obs_for_refine -- most of the scene graph's memory was
    # spent on throwaway single-sighting noise that every downstream consumer
    # (frontier scoring, room segmentation, relinking) still had to pay for.
    min_det_score: float = 0.35
    min_det_bbox_px: float = 1500.0
    # Evidence-score corroboration (P1i, FUS3DMaps-inspired 2026-07-19): a
    # detection that only re-matches an existing track from nearly the same
    # camera position adds little real corroborating evidence (no parallax)
    # -- it's still consistent with a one-off misdetection that happened to
    # repeat within the current keyframe's dwell. Earlier this was a hard
    # confirmed/tentative visibility gate (a track was hidden from tracks()/
    # candidates() entirely until re-observed from far enough away), but a
    # 30-episode eval showed that starved scene_graph.rebuild() of objects
    # early in exploration -- frontier scoring got "(no objects mapped yet)"
    # prompts and SR/SPL roughly halved (agent_stats: stop_reason=None,
    # select_none 0->22-24). Replaced with a soft evidence weight instead:
    # tracks are visible immediately from creation (ObjectTrack.evidence
    # accumulates every observation's det.score, discounted by
    # repeat_view_discount when the camera hasn't moved this far from the
    # track's first sighting). 0 disables the discount (every observation
    # gets full weight, matching pre-feature behavior).
    confirm_baseline_m: float = 0.15
    repeat_view_discount: float = 0.2


@dataclass
class ExplorationConfig:
    scorer: str = "vlm"  # vlm | llm_text | nearest | random
    top_n_frontiers: int = 5
    frontier_dedup_m: float = 1.0
    frontier_min_cells: int = 8
    subgraph_radius_m: float = 3.0
    images_per_frontier: int = 1  # each image costs ~1-2k ctx tokens
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5


@dataclass
class LLMConfig:
    base_url: str = "${oc.env:OLLAMA_HOST,http://localhost:11434}/v1"
    api_key: str = "ollama"
    text_model: str = "qwen2.5vl:3b"
    vlm_model: str = "qwen2.5vl:3b"
    timeout_s: float = 120.0
    max_image_px: int = 512
    # Some OpenAI-compatible providers (e.g. NVIDIA NIM vision models) return
    # malformed output when sent response_format=json_object; setting this false
    # omits that param and parses the JSON out of the plain-text reply instead.
    send_response_format: bool = True


@dataclass
class VerificationConfig:
    enabled: bool = True
    min_obs: int = 3
    # Candidate quality gates: sliver/fragment detections (a chair edge seen
    # through furniture) must not trigger the expensive approach+verify loop.
    min_score: float = 0.45
    min_bbox_px: int = 3000
    # Evidence-score gate (P1i follow-up, 2026-07-19): threshold picked from
    # a real 8-episode/1343-track measurement (scripts/orphan_node_check.py)
    # of evidence separated by whether a track ever reached candidate
    # quality -- non-candidate tracks: p75=0.84 p90=1.27; candidate-quality
    # tracks: min=0.64 p10=1.18 p25=1.56. 1.0 sits between the non-candidate
    # p75/p90 (filtering roughly 75-80% of low-evidence noise) and just
    # under the candidate p10 (sacrificing only ~6-7% of genuine candidates,
    # erring toward not rejecting real targets over aggressively filtering).
    min_evidence: float = 1.0
    ring_radii_m: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.5, 2.0])
    accept_confidence: float = 0.5
    # Verification is rare (1-3 calls/episode) and precision-critical: the 3B
    # VLM rejected clear true positives in prompt-lab tests; 7B passed all.
    vlm_model: str = "qwen2.5vl:7b"


@dataclass
class MappingConfig:
    resolution_m: float = 0.05
    obstacle_low_m: float = 0.1  # sim depth is noise-free; catch low furniture bases
    obstacle_high_m: float = 1.5
    max_range_m: float = 5.0
    depth_stride: int = 4
    inflate_margin_m: float = 0.07


@dataclass
class EvalConfig:
    split: str = "val"
    dataset_version: str = "v2"  # HM3D-semantics v0.2, 6 categories
    episodes_path: str = "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    scenes_dir: str = "data/scene_datasets/"
    num_episodes: int = -1  # -1 = all
    episode_ids: Optional[List[str]] = None
    # Restrict the eval to specific scene ids (None/["*"] = all). Used by the
    # single-floor preset since the 2D scene graph cannot represent stairs.
    content_scenes: Optional[List[str]] = None
    save_viz: bool = True
    rgb_width: int = 640
    rgb_height: int = 480
    hfov_deg: float = 79.0


@dataclass
class OSGConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: str = "outputs/${now:%Y%m%d_%H%M%S}"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=OSGConfig)
    cs.store(group="agent", name="base_default", node=AgentConfig)
    cs.store(group="detector", name="base_yoloe", node=DetectorConfig)
    cs.store(group="scene_graph", name="base_default", node=SceneGraphConfig)
    cs.store(group="exploration", name="base_vlm", node=ExplorationConfig)
    cs.store(group="llm", name="base_ollama", node=LLMConfig)
    cs.store(group="verification", name="base_on", node=VerificationConfig)
    cs.store(group="mapping", name="base_default", node=MappingConfig)
    cs.store(group="eval", name="base_hm3d", node=EvalConfig)
