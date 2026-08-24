"""VLM target verifier: a precision gate between a candidate object track and
the terminal APPROACH+STOP, restoring the pipeline's "improvement C" that was
stripped in commit c7adbde.

The matched-config eval (verification off) showed the dominant failure mode is
the agent walking up to and STOPping on a *confidently detected but wrong*
object -- with no gate, every open-vocab detector false positive (or a
same-category track far from the episode goal) converts straight to a failed
STOP (scripts/analyze_stages.py: 37/87 far-misses stopped on a close, confident
target detection). This verifier asks a vision-language model to confirm the
crop actually shows the target category before the agent commits.

One VLM call per accepted candidate (rare: 1-3/episode). NIM's
llama-3.2-11b-vision allows at most one image per prompt, so we send the
track's single best crop (the highest-confidence view the detector produced,
which is also what tests/fixtures/verify_bench scored on) rather than the live
frame or a ring of views.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.labels import normalize_label
from ..llm.client import ChatClient
from ..llm.prompts import (
    ABSENCE_CHOICE_SYSTEM,
    ABSENCE_CHOICE_USER,
    ABSENCE_SYSTEM,
    ABSENCE_USER,
    OBJECTNAV_CATEGORIES,
    VERIFY_CHOICE_SYSTEM,
    VERIFY_CHOICE_USER,
    VERIFY_SYSTEM,
    VERIFY_USER,
)


class VLMVerifier:
    def __init__(
        self,
        client: ChatClient,
        accept_confidence: float = 0.5,
        choice_mode: bool = True,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        self.accept_confidence = accept_confidence
        # Forced-choice: make the VLM pick one category from the list rather
        # than answer a yes/no about the target (catches detector mislabels).
        self.choice_mode = choice_mode
        self.categories = list(categories) if categories else list(OBJECTNAV_CATEGORIES)
        self.n_calls = 0
        self.n_errors = 0
        self.last_error: Optional[str] = None
        # Debug: when debug_dir is set, every call saves its exact input image
        # (whole frame + red box, as sent to the VLM) and the parsed response to
        # debug_dir, indexed by debug_tag (episode id). Set by the runner.
        self.debug_dir: Optional[str] = None
        self.debug_tag: str = "ep"

    def _save_debug(self, img: np.ndarray, target: str, prompt: str,
                    response: dict, accepted: bool) -> None:
        if not self.debug_dir:
            return
        import json
        import os

        import cv2

        os.makedirs(self.debug_dir, exist_ok=True)
        stem = f"{self.debug_tag}_call{self.n_calls:03d}_{normalize_label(target).replace(' ', '')}_{'ACC' if accepted else 'REJ'}"
        cv2.imwrite(os.path.join(self.debug_dir, stem + ".jpg"), img[..., ::-1])  # RGB->BGR
        with open(os.path.join(self.debug_dir, "index.jsonl"), "a") as f:
            f.write(json.dumps({
                "tag": self.debug_tag, "call": self.n_calls, "target": target,
                "accepted": accepted, "response": response, "prompt": prompt,
                "image": stem + ".jpg",
            }) + "\n")

    @staticmethod
    def _draw_bbox(rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """Full RGB image with a red box around the candidate. Copies so the
        source frame (shared across tracks) is never mutated."""
        import cv2

        img = np.ascontiguousarray(rgb).copy()
        x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        # thickness scales with image size so the box reads at any resolution
        th = max(2, int(round(0.006 * max(h, w))))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), th)  # red in RGB
        return img

    def _ask(self, img: Optional[np.ndarray], target: str) -> bool:
        """Send one image to the VLM. Fails OPEN (accept) on any VLM/parse
        error or missing image: a transient API failure must not silently
        reject an otherwise-good candidate and send the agent back to wandering
        -- strictly worse than the no-verifier baseline. Errors are counted so
        a systematic outage shows up in the per-episode verify_errors field."""
        if img is None or img.size == 0:
            return True  # nothing to look at -> defer to detector, don't block
        if self.choice_mode:
            system, user = VERIFY_CHOICE_SYSTEM, VERIFY_CHOICE_USER.format(
                categories=", ".join(self._choice_list(target))
            )
        else:
            system, user = VERIFY_SYSTEM, VERIFY_USER.format(target=normalize_label(target))
        self.n_calls += 1
        try:
            out = self.client.chat(system, user, images=[img], json_response=True)
        except Exception as e:  # noqa: BLE001 - fail open, but record
            self.n_errors += 1
            self.last_error = repr(e)[:200]
            self._save_debug(img, target, user, {"_error": self.last_error}, True)
            return True
        conf = float(out.get("confidence", 0.0) or 0.0)
        if self.choice_mode:
            accepted = normalize_label(out.get("category", "")) == normalize_label(target) and conf >= self.accept_confidence
        else:
            accepted = bool(out.get("is_target", False)) and conf >= self.accept_confidence
        self._save_debug(img, target, user, out, accepted)
        return accepted

    def _choice_list(self, target: str) -> List[str]:
        """Category list shown to the VLM, guaranteed to contain the target."""
        cats = list(self.categories)
        if not any(normalize_label(c) == normalize_label(target) for c in cats):
            cats.append(normalize_label(target))
        return cats

    def verify_bbox(self, rgb: Optional[np.ndarray], bbox_xyxy: Optional[np.ndarray],
                    target: str) -> bool:
        """Return True if the VLM confirms the object inside `bbox_xyxy` of the
        full image `rgb` is a `target`. The whole scene is shown with a red box
        around the candidate, giving the VLM context a bare crop lacks."""
        if rgb is None or bbox_xyxy is None:
            return True
        return self._ask(self._draw_bbox(rgb, bbox_xyxy), target)

    def verify_absence(
        self,
        rgb: Optional[np.ndarray],
        region_bbox_xyxy: Optional[np.ndarray],
        categories: List[str],
        max_categories: int = 5,
    ) -> Optional[dict]:
        """Which of `categories` are inside the marked region?

        The verifier's mirror image. Confirming what IS there is worth one call;
        asking what is NOT there is what lets a map correct itself -- a single
        trusted "no" collapses a belief that would otherwise need several
        detector misses to shift.

        Returns {category: present} for the categories asked, or None if the
        call failed -- None means "no information", NOT "absent", because
        treating a network error as evidence of absence would quietly delete
        objects.

        `categories` is truncated to `max_categories`: enumerating a long list
        is where vision-language models are least reliable, and an absence you
        cannot trust is worse than no absence at all.
        """
        if rgb is None or region_bbox_xyxy is None or not categories:
            return None
        asked = [str(c) for c in categories[:max_categories]]
        img = self._draw_bbox(rgb, np.asarray(region_bbox_xyxy, dtype=float))
        self.n_calls += 1
        try:
            reply = self.client.chat(
                ABSENCE_SYSTEM,
                ABSENCE_USER.format(categories=", ".join(asked)),
                images=[img],
                json_response=True,
            )
        except Exception as exc:  # network/model failure is not evidence
            self.n_errors += 1
            self.last_error = str(exc)
            return None
        if not isinstance(reply, dict):
            self.n_errors += 1
            self.last_error = f"unexpected absence reply: {reply!r}"
            return None
        present = reply.get("present") or []
        if isinstance(present, str):
            present = [present]
        seen = {normalize_label(str(c)) for c in present}
        result = {c: normalize_label(c) in seen for c in asked}
        self._save_debug(img, ",".join(asked), ABSENCE_USER, reply,
                         accepted=any(result.values()))
        return result

    @staticmethod
    def _zoom(rgb: np.ndarray, bbox: np.ndarray, pad: float = 1.6, out_px: int = 320):
        """Crop around the region and upscale it.

        A 100 px object in a 640x480 frame is most of the reason a VLM answers
        about the scene instead of the region: measured, the same question on a
        zoomed crop went from 12/20 to 17/20.
        """
        import cv2

        h, w = rgb.shape[:2]
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * pad / 2.0
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
        sub = rgb[y1:y2, x1:x2]
        if sub.size == 0:
            return rgb, np.asarray(bbox, dtype=float)
        scale = max(1.0, float(out_px) / max(sub.shape[:2]))
        sub = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        moved = np.array([(bbox[0] - x1) * scale, (bbox[1] - y1) * scale,
                          (bbox[2] - x1) * scale, (bbox[3] - y1) * scale], dtype=float)
        return sub, moved

    def verify_still_there(
        self, rgb: Optional[np.ndarray], region_bbox_xyxy: Optional[np.ndarray],
        target: str, zoom: bool = True,
    ) -> Optional[bool]:
        """Is `target` still inside the marked region? None = no information.

        The forced-choice form of `verify_absence`, and the one worth using: a
        model asked "is a bowl present" answers about plausibility, while a model
        made to choose between "bowl", "bare" and "blocked" answers about the
        pixels. `blocked` returns None -- an obstructed view is not evidence of
        absence, and treating it as such deletes objects behind doors.
        """
        if rgb is None or region_bbox_xyxy is None:
            return None
        bbox = np.asarray(region_bbox_xyxy, dtype=float)
        img, bbox = self._zoom(rgb, bbox) if zoom else (rgb, bbox)
        img = self._draw_bbox(img, bbox)
        self.n_calls += 1
        try:
            out = self.client.chat(
                ABSENCE_CHOICE_SYSTEM,
                ABSENCE_CHOICE_USER.format(target=normalize_label(target)),
                images=[img],
                json_response=True,
            )
        except Exception as exc:
            self.n_errors += 1
            self.last_error = repr(exc)[:200]
            return None
        choice = normalize_label(str(out.get("choice", "")))
        self._save_debug(img, target, "still_there", out, accepted=choice == normalize_label(target))
        if normalize_label(target) in choice:
            return True
        if "bare" in choice:
            return False
        return None  # blocked, or an answer we cannot read

    def verify_crop(self, img: Optional[np.ndarray], target: str) -> bool:
        """Verify a pre-cropped image (fallback when no full frame + bbox)."""
        return self._ask(img, target)

    def verify(self, track, target: str, live_view: Optional[np.ndarray] = None) -> bool:
        """Pre-approach verification of a candidate track: the full frame of its
        best detection with the target boxed. Falls back to the crop if the
        full frame/bbox weren't recorded."""
        if track.best_frame_rgb is not None and track.best_bbox_xyxy is not None:
            return self.verify_bbox(track.best_frame_rgb, track.best_bbox_xyxy, target)
        img = track.best_crop if track.best_crop is not None else live_view
        return self.verify_crop(img, target)
