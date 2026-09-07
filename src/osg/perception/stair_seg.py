"""The up-stair signal ASCENT actually uses: RedNet MPCAT40 segmentation.

OSG detects up-stairs with YOLOE's open-vocabulary `stairs` class plus a
geometric gate. Measured on 250 stair poses (S32) that fires 10.4% of the time,
and S14a's pre-registered rule says a dedicated segmentation model is what moves
it. ASCENT runs one -- RedNet, every step, `ascent_policy.py:151, 424` -- and
intersects it with a second detector before believing anything:

    fusion_stair_mask = stair_mask & (seg_mask == STAIR_CLASS_ID)     # :522
    if np.sum(seg_mask == STAIR_CLASS_ID) > 20:                        # :520

`stair_mask` there is GroundingDINO prompted with `"stair ."`
(`map_controller.py:700-704`). OSG has no GroundingDINO, so YOLOE's `stairs`
mask plays that role: same job -- an open-vocabulary detector giving a second,
independent opinion -- from the model this repo already loads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.types import FrameData

# MPCAT40 index 16, plus the +1 the wrapper adds to every argmax.
# ASCENT: constants.py:214 `STAIR_CLASS_ID = 17`.
STAIR_CLASS_ID = 17
# ASCENT will not look at the fused mask unless the segmenter alone is this
# confident there are stairs in frame (obstacle_map.py:520).
MIN_STAIR_PIXELS = 20


class RedNetStairSegmenter:
    """Per-frame boolean stair mask from RedNet.

    Loaded once per run and shared across episodes, like the detector: the
    checkpoint is 626 MB and a NavAgent is built per episode.
    """

    def __init__(
        self,
        weights_path: str,
        *,
        depth_min_m: float = 0.5,
        depth_max_m: float = 5.0,
        device: Optional[str] = None,
    ) -> None:
        import torch

        from .rednet.rednet_model import load_rednet

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RedNet weights not found at {path}. "
                "Run: python scripts/download_weights.py --rednet"
            )
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_rednet(self.device, ckpt=str(path), resize=True)
        self.model.eval()
        self.n_calls = 0

    def segment(self, frame: FrameData) -> np.ndarray:
        """MPCAT40 class ids (1..40) at the frame's own resolution."""
        import torch

        rgb = np.ascontiguousarray(frame.rgb[..., :3])
        # The wrapper divides RGB by 255 itself and normalises depth with
        # mean 0.213 / std 0.285, i.e. it expects the [0, 1] depth habitat
        # produces with normalize_depth=True. OSG keeps metres (the costmap and
        # object layer want them), so undo the difference here -- the same
        # conversion planning/pointnav_driver.py does for the mover.
        d = np.clip(np.asarray(frame.depth, dtype=np.float32),
                    self.depth_min_m, self.depth_max_m)
        d = (d - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)
        with torch.no_grad():
            rgb_t = torch.from_numpy(rgb).to(self.device)[None].float()
            depth_t = torch.from_numpy(d).to(self.device)[None, ..., None]
            pred = self.model(rgb_t, depth_t)
        self.n_calls += 1
        return pred.squeeze().cpu().numpy().astype(np.uint8)

    def stair_mask(self, frame: FrameData) -> Optional[np.ndarray]:
        """Boolean stair mask, or None when there is too little to trust.

        The `> MIN_STAIR_PIXELS` gate is ASCENT's and it is doing real work: a
        handful of stray stair pixels on a bannister or a skirting board would
        otherwise be intersected with a detector box and written into the map as
        a staircase.
        """
        seg = self.segment(frame)
        mask = seg == STAIR_CLASS_ID
        if int(mask.sum()) <= MIN_STAIR_PIXELS:
            return None
        return mask


def build_stair_segmenter(cfg):
    """None unless `agent.rednet_stairs` is on, so the import cost is opt-in."""
    if not bool(getattr(cfg.agent, "rednet_stairs", False)):
        return None
    return RedNetStairSegmenter(
        str(getattr(cfg.agent, "rednet_weights", "data/weights/rednet_semmap_mp3d_40.pth")),
        depth_min_m=float(getattr(cfg.eval, "depth_min_m", 0.5)),
        depth_max_m=float(getattr(cfg.eval, "depth_max_m", 5.0)),
    )
