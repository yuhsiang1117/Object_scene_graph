"""Clients for the rest of ASCENT's served perception: GroundingDINO (stairs)
and RAM++ (per-step scene tags), plus the fail-loud contract they all share.

ASCENT runs every model as a Flask process (`relative_work/ascent/model_api/`)
and its policy calls them over HTTP; this file speaks those wire formats so
the OSG side can run ASCENT's control flow on ASCENT's models.

FAIL LOUD. Every client in ASCENT's own `model_api` swallows connection
errors and returns a neutral value -- and OSG's `Blip2ItmScorer` did the same,
returning 0. Under ASCENT's control flow a BLIP-2 cosine of 0 means the commit
gate never latches, which means the agent never STOPs, which is 0% SR with no
error anywhere. `strict=True` turns any transport failure, non-200, or
malformed reply into `PerceptionUnavailable`, and `probe_perception_servers`
checks every endpoint before Habitat is even loaded.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

import numpy as np


class PerceptionUnavailable(RuntimeError):
    """A served model did not answer. Raised instead of returning a neutral
    value, because a neutral value here silently changes what the agent does."""


def encode_rgb(rgb: np.ndarray) -> str:
    """JPEG + base64, exactly as ASCENT's `image_to_str` does it.

    No channel swap: ASCENT hands the RGB array straight to `cv2.imencode` and
    the server decodes it back with `cv2.imdecode`, so the round trip preserves
    whatever was passed. Swapping to BGR here would feed every model swapped
    images and they would all still return plausible-looking answers.
    """
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise PerceptionUnavailable("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def post_json(url: str, payload: dict, timeout_s: float, strict: bool,
              counters: Optional[dict] = None):
    """POST a JSON payload; return the decoded reply, or None (non-strict)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            if r.status != 200:
                raise PerceptionUnavailable(f"{url} answered HTTP {r.status}")
            out = json.loads(r.read())
    except PerceptionUnavailable:
        if counters is not None:
            counters["errors"] = counters.get("errors", 0) + 1
        if strict:
            raise
        return None
    except Exception as exc:  # noqa: BLE001 - transport / decode
        if counters is not None:
            counters["errors"] = counters.get("errors", 0) + 1
        if strict:
            raise PerceptionUnavailable(f"{url}: {exc}") from exc
        return None
    if counters is not None:
        counters["calls"] = counters.get("calls", 0) + 1
    return out


def probe_perception_servers(endpoints: Dict[str, str], timeout_s: float = 5.0) -> None:
    """GET every endpoint; raise naming every one that is not up.

    ASCENT's servers answer GET with `{"status": "ok", ...}`.
    """
    dead = []
    for name, url in endpoints.items():
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as r:
                if r.status != 200:
                    dead.append(f"{name} ({url}): HTTP {r.status}")
        except Exception as exc:  # noqa: BLE001
            dead.append(f"{name} ({url}): {exc}")
    if dead:
        raise PerceptionUnavailable(
            "perception servers not ready -- start them with "
            "`bash scripts/serve_perception.sh`:\n  " + "\n  ".join(dead)
        )


class GroundingDinoStairDetector:
    """The detector half of ASCENT's stair fusion.

    ASCENT appends `" stair ."` to its GroundingDINO caption on every step
    (`map_controller.py:700-704`), keeps boxes whose phrase is `stair` with
    logit >= 0.60 (`:782-786`), segments each box with MobileSAM, and the
    obstacle map ANDs the union of those masks with RedNet's stair class
    (`obstacle_map.py:520-524`). OSG's port unioned RedNet in instead, for
    recall -- and the trace diagnosis measured that union climbing 11
    same-floor episodes into the ground. Strict fusion is the reference.
    """

    def __init__(
        self,
        url: str = "http://localhost:13184/gdino",
        sam_url: str = "http://localhost:13183/mobile_sam",
        caption: str = "stair .",
        conf: float = 0.60,
        timeout_s: float = 15.0,
        strict: bool = False,
    ) -> None:
        self.url = url
        self.sam_url = sam_url
        self.caption = caption
        self.conf = float(conf)
        self.timeout_s = float(timeout_s)
        self.strict = bool(strict)
        self.counters: dict = {}

    def boxes(self, rgb: np.ndarray, img_b64: Optional[str] = None) -> List[np.ndarray]:
        """Pixel xyxy boxes of `stair` detections above `conf`."""
        img = img_b64 or encode_rgb(rgb)
        resp = post_json(self.url, {"image": img, "caption": self.caption},
                         self.timeout_s, self.strict, self.counters)
        if not resp or "boxes" not in resp:
            if self.strict:
                raise PerceptionUnavailable(f"{self.url}: malformed reply {resp!r}")
            return []
        h, w = rgb.shape[:2]
        out = []
        for box, logit, phrase in zip(resp["boxes"], resp["logits"], resp.get("phrases", [])):
            if str(phrase).strip().lower() != "stair" or float(logit) < self.conf:
                continue
            out.append(np.asarray(box, float) * np.array([w, h, w, h], float))
        return out

    def mask(self, rgb: np.ndarray, img_b64: Optional[str] = None) -> Optional[np.ndarray]:
        """Union of MobileSAM masks over the stair boxes, or None when none."""
        img = img_b64 or encode_rgb(rgb)
        h, w = rgb.shape[:2]
        union = None
        for box in self.boxes(rgb, img):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            resp = post_json(self.sam_url, {"image": img, "bbox": [x0, y0, x1, y1]},
                             self.timeout_s, self.strict, self.counters)
            if not resp or "cropped_mask" not in resp:
                if self.strict:
                    raise PerceptionUnavailable(f"{self.sam_url}: malformed reply")
                continue
            raw = np.frombuffer(base64.b64decode(resp["cropped_mask"]), dtype=np.uint8)
            if raw.size != h * w:
                if self.strict:
                    raise PerceptionUnavailable(
                        f"{self.sam_url}: mask has {raw.size} px, frame has {h * w}")
                continue
            m = raw.reshape(h, w).astype(bool)
            union = m if union is None else (union | m)
        return union


class RamTagger:
    """RAM++ open-set tags for a frame, as ASCENT feeds its LLM prompts.

    `map_controller.py:808-813`: `cur_objs = self._ram.predict(rgb)`, split on
    `|`, stripped. The server returns RAM++'s `inference()` tuple (english,
    chinese) jsonified as a list, and ASCENT's client reads `[0]`.
    """

    def __init__(self, url: str = "http://localhost:13185/ram",
                 timeout_s: float = 15.0, strict: bool = False) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)
        self.strict = bool(strict)
        self.counters: dict = {}

    def tags(self, rgb: np.ndarray, img_b64: Optional[str] = None) -> List[str]:
        img = img_b64 or encode_rgb(rgb)
        resp = post_json(self.url, {"image": img}, self.timeout_s, self.strict, self.counters)
        if resp is None:
            return []
        text = resp[0] if isinstance(resp, (list, tuple)) else resp
        if not isinstance(text, str):
            if self.strict:
                raise PerceptionUnavailable(f"{self.url}: malformed reply {resp!r}")
            return []
        return [t.strip() for t in text.split("|") if t.strip()]
