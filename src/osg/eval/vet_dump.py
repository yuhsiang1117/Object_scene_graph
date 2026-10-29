"""What the STOP gate's recogniser was shown, written down (`eval.vet_dump_dir`).

The recogniser was chosen on rendered crops at <= 1.5 m (docs/history/
VERIFIER_EVAL.md), but live vetting judges a track's saved best crop from
wherever it was seen -- a different crop treatment, at two to four metres. To
choose a better recogniser offline, the set has to be the crops the agent
actually judged. Each judgement becomes one line of `<dir>/vets.jsonl` and a
PNG of the crop; live and STOP judgements also keep the full frame and the box,
so another crop treatment can be cut from the same view.

Strictly read-only: nothing here touches a track, a frame or the gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(np.asarray(rgb)[..., ::-1]),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])


def _f(x) -> Optional[list]:
    return None if x is None else [round(float(v), 4) for v in np.asarray(x, dtype=float).ravel()]


class VetDump:
    def __init__(self, root: str, episode_tag: str, target: str) -> None:
        self.root = Path(root)
        self.tag = str(episode_tag)
        self.target = str(target)
        self.n = 0

    def write(self, kind: str, step: int, track, crop: Optional[np.ndarray], verdict,
              reading: Dict[str, Any], centre, frame: Optional[np.ndarray] = None,
              bbox=None, range_m: Optional[float] = None) -> None:
        stem = f"{self.tag}/{kind}_{int(step):04d}_t{int(track.id)}_{self.n}"
        self.n += 1
        rec: Dict[str, Any] = {
            "episode_id": self.tag, "target": self.target, "kind": kind, "step": int(step),
            "track_id": int(track.id), "label": str(track.label),
            "verdict": None if verdict is None else bool(verdict), **dict(reading or {}),
            "centre": _f(centre), "range_m": range_m,
            "bbox_xyxy": _f(bbox), "best_score": round(float(getattr(track, "best_score", 0.0)), 4),
            "n_obs": int(getattr(track, "n_obs", 0)),
            "evidence": round(float(getattr(track, "evidence", 0.0)), 4),
            "presence_p": round(float(track.presence.p), 4) if getattr(track, "presence", None) is not None else None,
            "from_prior": bool(getattr(track, "from_prior", False)),
            "seen_live": bool(getattr(track, "seen_live", False)),
            "proposal_only": bool(getattr(track, "proposal_only", False)),
            "floor_key": int(getattr(track, "floor_key", 0)),
            "best_cam_xy": _f(getattr(track, "best_cam_xy", None)),
        }
        ell = getattr(track, "ellipsoid", None)
        if ell is not None:
            rec["axes"] = _f(ell.axes)
            rec["R"] = _f(ell.R)
        if crop is not None and getattr(crop, "size", 0):
            rec["crop"] = f"{stem}_crop.png"
            _png(self.root / rec["crop"], crop)
        if frame is not None and getattr(frame, "size", 0):
            rec["frame"] = f"{stem}_frame.png"
            _png(self.root / rec["frame"], frame)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "vets.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
