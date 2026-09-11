"""`detector` group: the open-vocabulary detector and the vocabulary it answers to.

The vocabulary is not a list of the objects in the scene -- it is the set of
query strings the detector is asked about, and which strings are in it changes
what gets found. See `osg/perception/vocabulary.py` for the target-side rule and
docs/ARCHITECTURE.md for the measurements behind the per-class gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
class DetectorConfig:
    name: str = "yoloe"
    weights: str = "data/weights/yoloe-11s-seg.pt"
    conf: float = 0.3
    # Per-class overrides of `conf`, {label: threshold}. One global gate prices
    # every class the same and they are not the same: measured over 900 random
    # navigable poses, dropping the gate 0.30 -> 0.20 costs the tomato soup can
    # and the banana NOTHING in false positives while buying them 8-9 points of
    # recall, and costs the cracker box eighteen false positives for six. See
    # YoloeDetector for the full table.
    class_conf: Dict[str, float] = field(default_factory=dict)
    imgsz: int = 512
    half: bool = True
    device: str = "cuda"
    # `dfine` only: ASCENT's models answer over HTTP (scripts/serve_perception.sh).
    url: str = "http://localhost:13186/dfine"
    sam_url: str = "http://localhost:13183/mobile_sam"
    use_sam: bool = True
    timeout_s: float = 15.0
    # GroundingDINO, the detector half of ASCENT's stair fusion
    # (`map_controller.py:700-704, 782-786`): caption "stair .", logit >= 0.60.
    gdino_url: str = "http://localhost:13184/gdino"
    gdino_stair_conf: float = 0.60
    # Raise `PerceptionUnavailable` on any served-model failure instead of
    # returning a neutral value. The ascentnav preset turns this on.
    strict: bool = False
    vocabulary: List[str] = field(default_factory=lambda: list(DEFAULT_VOCABULARY))


