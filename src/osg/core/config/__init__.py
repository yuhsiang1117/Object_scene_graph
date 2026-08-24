"""Structured configs registered with Hydra's ConfigStore so that typos in
yaml/CLI overrides fail fast instead of silently creating new keys.

One module per Hydra group, because that is the unit people override:
`exploration.search_posterior=true` on a command line and `exploration.py` in
this package are the same object. `OSGConfig` below assembles them, and every
name this package has ever exported is re-exported here, so
`from osg.core.config import ...` is unaffected by the split.

The defaults are not arbitrary. Nearly every one was chosen by an experiment and
carries the measurement in a comment beside it; `tests/unit/test_config_snapshot.py`
pins all 211 of them so a refactor cannot move one by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from .agent import AgentConfig
from .detector import DEFAULT_VOCABULARY, DetectorConfig
from .eval import EvalConfig
from .exploration import ExplorationConfig
from .floor import FloorConfig
from .llm import LLMConfig
from .mapping import MappingConfig
from .scene_graph import PresenceConfig, SceneGraphConfig
from .verification import VerificationConfig
from .ycb import YCB_TARGET_LABELS, YCBAuthoredConfig

__all__ = [
    "AgentConfig", "DetectorConfig", "EvalConfig", "ExplorationConfig",
    "FloorConfig", "LLMConfig", "MappingConfig", "OSGConfig", "PresenceConfig",
    "SceneGraphConfig", "VerificationConfig", "YCBAuthoredConfig",
    "DEFAULT_VOCABULARY", "YCB_TARGET_LABELS", "register_configs",
]


@dataclass
class OSGConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    floor: FloorConfig = field(default_factory=FloorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ycb: YCBAuthoredConfig = field(default_factory=YCBAuthoredConfig)
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
    cs.store(group="floor", name="base_default", node=FloorConfig)
    cs.store(group="eval", name="base_hm3d", node=EvalConfig)
    cs.store(group="ycb", name="base_authored", node=YCBAuthoredConfig)
