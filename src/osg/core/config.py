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
    "chair", "sofa", "plant", "bed", "toilet", "tv_monitor",
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
    room_seg_every_kf: int = 10
    room_min_radius_m: float = 0.9
    room_door_width_m: float = 1.2


@dataclass
class ExplorationConfig:
    scorer: str = "vlm"  # vlm | llm_text | nearest | random
    top_n_frontiers: int = 5
    frontier_dedup_m: float = 1.0
    frontier_min_cells: int = 8
    subgraph_radius_m: float = 3.0
    images_per_frontier: int = 2
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5


@dataclass
class LLMConfig:
    base_url: str = "${oc.env:OLLAMA_HOST,http://localhost:11434}/v1"
    api_key: str = "ollama"
    text_model: str = "qwen2.5vl:3b"
    vlm_model: str = "qwen2.5vl:3b"
    timeout_s: float = 60.0
    max_image_px: int = 512


@dataclass
class VerificationConfig:
    enabled: bool = True
    min_obs: int = 2
    ring_radii_m: List[float] = field(default_factory=lambda: [0.8, 1.2, 1.5, 2.0])
    accept_confidence: float = 0.5


@dataclass
class MappingConfig:
    resolution_m: float = 0.05
    obstacle_low_m: float = 0.2
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
