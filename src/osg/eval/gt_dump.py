"""The picture behind `gt_kf_in_view=14, gt_kf_detected=0`.

That pair of counters is the sharpest instrument in the runner and it still
stops one question short. It says the agent pointed a camera at the object,
unoccluded, and the detector did not name it -- but not WHY, and the two
candidate whys need opposite fixes. Either the object was a legible object in
that frame and the open-vocabulary head simply does not recognise the asset (a
vocabulary problem), or it was thirty pixels in the corner of the image behind a
chair leg and no detector was ever going to name it (a framing problem).

So for every keyframe the instrument counts as in-view, write the frame: the RGB
with the projected object marked, every RAW detection drawn beside it, and the
range / off-axis / visible-fraction the instrument used, burned into the corner.
Then look at them.

Written from `GroundTruthVisibility`, which already holds the projection, and
strictly one-way: this writes JPEGs to disk and hands the agent nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.labels import normalize_label
from .visualize import overlay_segmentation


class KeyframeDump:
    """One JPEG per in-view keyframe, plus an index.tsv of what it showed."""

    def __init__(self, out_dir: str, tag: str, target: str) -> None:
        import cv2

        self._cv2 = cv2
        self.dir = Path(out_dir) / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.n = 0
        self._index = self.dir / "index.tsv"
        self._index.write_text(
            "kf\tstep\trange_m\toffaxis\tvis_frac\tgt_px\tnamed\tbest_score"
            "\tbest_px\tadmitted\ttop_labels\n"
        )

    def write(self, frame, dets, u, v, z, fraction, offaxis,
              best: float, best_px: float, admitted: bool) -> None:
        cv2 = self._cv2
        self.n += 1
        img = overlay_segmentation(frame.rgb, dets or [], self.target)

        # The instrument's own projected centre, in the instrument's own terms:
        # a ring the eye can find at a glance even when nothing was detected,
        # which is exactly the case this file exists for.
        cx, cy = int(round(u)), int(round(v))
        cv2.circle(img, (cx, cy), 26, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 27, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

        verdict = "NAMED" if best > 0.0 else "MISSED"
        if admitted:
            verdict = "ADMITTED"
        lines = [
            f"{self.target}  [{verdict}]",
            f"range {z:.2f} m   offaxis {offaxis:.2f}   visible {fraction:.2f}",
            f"best {best:.2f} @ {best_px:.0f} px",
        ]
        for i, text in enumerate(lines):
            org = (10, 24 + 22 * i)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, th, cv2.LINE_AA)

        step = int(getattr(frame, "step", -1) or -1)
        stem = f"kf{self.n:04d}_r{z:.2f}_{verdict}"
        cv2.imwrite(str(self.dir / f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        # The clean frame as well, losslessly, with the projected pixel in the
        # name. Every "would it have been detected if..." question is then an
        # offline re-run over these instead of another 500-step episode, which
        # is the difference between a probe that takes a minute and one that
        # takes half an hour.
        raw = self.dir / "raw"
        raw.mkdir(exist_ok=True)
        cv2.imwrite(str(raw / f"{stem}_u{cx}_v{cy}.png"),
                    np.ascontiguousarray(frame.rgb[..., ::-1]))

        # What the detector DID say where the object is -- the near-miss labels
        # are the whole diagnosis when the target's own label scores zero.
        near = sorted(
            (d for d in (dets or []) if _covers(d, u, v)),
            key=lambda d: -float(d.score),
        )[:4]
        top = ",".join(f"{d.label}:{d.score:.2f}" for d in near) or "-"
        with open(self._index, "a") as f:
            f.write(f"{self.n}\t{step}\t{z:.3f}\t{offaxis:.3f}\t{fraction:.3f}\t"
                    f"{int(np.count_nonzero(_mask_at(dets, u, v)))}\t"
                    f"{int(best > 0.0)}\t{best:.3f}\t{best_px:.0f}\t"
                    f"{int(admitted)}\t{top}\n")


def _covers(det, u: float, v: float, pad: float = 8.0) -> bool:
    x1, y1, x2, y2 = [float(c) for c in det.bbox_xyxy]
    return x1 - pad <= u <= x2 + pad and y1 - pad <= v <= y2 + pad


def _mask_at(dets, u: float, v: float) -> np.ndarray:
    """Pixels of whatever the detector put where the object is (0 if nothing)."""
    for det in dets or []:
        if _covers(det, u, v) and det.mask is not None:
            return det.mask.astype(bool)
    return np.zeros((1,), dtype=bool)
