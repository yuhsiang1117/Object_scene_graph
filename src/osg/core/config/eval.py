"""`eval` group: which benchmark runs, over which episodes, with what recorded.

`mode` picks the benchmark: `objectnav` is the standard HM3D dataset,
`ycb_authored` is the dynamic-scene benchmark built from authored layouts.
`attempts` is the protocol knob -- DualMap allows a query several navigation
attempts and scoring one is a stricter protocol than the system being compared
against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalConfig:
    # Navigation attempts per query. DualMap allows several: a failed attempt
    # updates the map and the agent goes again. Scoring one attempt is a
    # STRICTER protocol than the system being compared against, so this exists
    # to match theirs rather than to flatter ours. 1 keeps the old behaviour.
    attempts: int = 1
    # `objectnav` loads the standard HM3D episode dataset. `ycb_authored`
    # discovers scene-layout JSON files written by habitat-data-collector and
    # builds equivalent ObjectNav episodes from the placed YCB objects.
    mode: str = "objectnav"
    split: str = "val"
    dataset_version: str = "v2"  # HM3D-semantics v0.2, 6 categories
    episodes_path: str = "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
    scenes_dir: str = "data/scene_datasets/"
    num_episodes: int = -1  # -1 = all
    # >0 forces habitat to move to a new scene after this many episodes, so a
    # fixed-size subset spans the split instead of draining one scene first.
    # -1 = habitat default (group by scene, ~10000-step budget per scene).
    max_scene_repeat_episodes: int = -1
    episode_ids: Optional[List[str]] = None
    # Restrict the eval to specific scene ids (None/["*"] = all). Used by the
    # single-floor preset since the 2D scene graph cannot represent stairs.
    content_scenes: Optional[List[str]] = None
    save_viz: bool = True
    # Per-step debug video: for each episode write viz/debug/ep<ID>.mp4 whose
    # frames are [live RGB + YOLOE segmentation overlay | top-down costmap] at
    # every step. The detector is re-run per step FOR VISUALIZATION ONLY (it
    # does not feed the object layer -- keyframe detection is unchanged), so SR
    # is unaffected; it roughly doubles detector load, hence off by default.
    debug_frames: bool = False
    rgb_width: int = 640
    rgb_height: int = 480
    hfov_deg: float = 79.0


