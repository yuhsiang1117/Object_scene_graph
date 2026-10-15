"""The STOP gate: a second opinion on WHAT the agent is about to stop at.

A STOP is the expensive decision -- it spends one of three attempts -- and on
the YCB benchmarks it is usually taken at the wrong object: 31 of 40 goal
commits on the multi-floor set were false positives, and 49 of 73 live commits
on DualMap's static condition. The pipeline already has a sensor for "it is not
here" (`verification/absence.py`); it has none for "that is not it". A false
positive is a real object, so the detector re-detects it on arrival, the
absence sensor has nothing to say, and the attempt is spent.

So just before a STOP that rests on a live detection of the target, the gate
asks a recogniser that is not the detector -- a VLM on the full frame with the
candidate boxed, or MobileCLIP's margin of the query over the detector's
background vocabulary -- whether the boxed object is the query. A "no" is
identity evidence: the candidate is released without an attempt, its presence
takes one reading at the gate's own (r, q) through the same filter a failed
attempt uses (`eval/attempts.py`), and its identity count rises by one, so two
rejections retire it exactly as two failed attempts would.

What it deliberately does not do:

* judge a stop with no detection behind it (a stale-anchor stop, a stop on
  the last metre of a path): there is nothing in the frame to judge, and the
  absence sensor already ran;
* treat a failed call as a "no" -- an error is no information, as for the
  absence sensor;
* reject without bound. A wrong recogniser costs a trip each time it rejects
  the real object, so a track is let stop after `stop_gate_max_per_track`
  rejections and an episode rejects at most `stop_gate_max_per_episode`.

It is called once per arrival, synchronously, like the arrival-time absence
reading -- a terminal decision, not the per-step loop that must never wait on
a model.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.labels import normalize_label
from .state import TURN_ACTION

MODES = ("off", "vlm", "clip")


class StopGate:
    def __init__(self, nav) -> None:
        self.nav = nav
        vc = nav.cfg.verification
        self.mode = str(getattr(vc, "stop_gate", "off") or "off")
        if self.mode not in MODES:
            raise ValueError(f"verification.stop_gate must be one of {MODES}, got {self.mode!r}")
        self._vlm = None
        self.reset()

    def reset(self) -> None:
        self.rejections_by_track: dict = {}
        self.n_rejected = 0
        self.log: list = []

    @property
    def on(self) -> bool:
        return self.mode != "off"

    # ------------------------------------------------------------- the check

    def check(self, frame) -> Optional[str]:
        """The action to take INSTEAD of the STOP about to be emitted, or None
        to let it stand."""
        if not self.on:
            return None
        nav, vc = self.nav, self.nav.cfg.verification
        stats = nav.stats
        track_id = nav._candidate_id
        track = nav.object_layer.get(track_id) if track_id is not None else None
        if track is None:
            return None
        if self.n_rejected >= int(vc.stop_gate_max_per_episode):
            stats["stop_gate_episode_cap"] = stats.get("stop_gate_episode_cap", 0) + 1
            return None
        if self.rejections_by_track.get(int(track_id), 0) >= int(vc.stop_gate_max_per_track):
            stats["stop_gate_track_cap"] = stats.get("stop_gate_track_cap", 0) + 1
            return None
        det = nav._best_target_detection(frame)
        if det is None:
            stats["stop_gate_nothing_to_judge"] = stats.get("stop_gate_nothing_to_judge", 0) + 1
            return None
        stats["stop_gate_calls"] = stats.get("stop_gate_calls", 0) + 1
        with nav.profiler.timeit("stop_gate"):
            verdict = self._judge(frame, det)
        entry = {
            "step": int(nav.step_count), "track_id": int(track_id),
            "label": str(det.label), "score": round(float(det.score), 3),
            "verdict": verdict,
        }
        self.log.append(entry)
        stats["stop_gate_log"] = self.log
        if verdict is None:
            stats["stop_gate_errors"] = stats.get("stop_gate_errors", 0) + 1
            return None
        if verdict:
            stats["stop_gate_accept"] = stats.get("stop_gate_accept", 0) + 1
            return None
        stats["stop_gate_reject"] = stats.get("stop_gate_reject", 0) + 1
        self._reject(track)
        return TURN_ACTION

    # ------------------------------------------------------------ recognisers

    def _judge(self, frame, det) -> Optional[bool]:
        if self.mode == "vlm":
            return self._judge_vlm(frame, det)
        return self._judge_clip(frame, det)

    def _vlm_verifier(self):
        if self._vlm is None:
            from ..llm.client import ChatClient
            from ..verification.verifier import VLMVerifier

            cfg = self.nav.cfg
            vc = cfg.verification
            client = ChatClient(
                (vc.base_url or cfg.llm.base_url),
                (vc.stop_gate_model or vc.vlm_model),
                (vc.api_key or cfg.llm.api_key),
                cfg.llm.timeout_s, cfg.llm.max_image_px, cfg.llm.send_response_format,
            )
            self._vlm = VLMVerifier(
                client, accept_confidence=vc.accept_confidence,
                choice_mode=bool(vc.stop_gate_choice),
                categories=self._background() + ["none"],
            )
        return self._vlm

    def _judge_vlm(self, frame, det) -> Optional[bool]:
        verifier = self._vlm_verifier()
        errors = verifier.n_errors
        accepted = verifier.verify_bbox(frame.rgb, det.bbox_xyxy, self.nav.target)
        # The verifier fails OPEN on an API error; an error is not an answer.
        return None if verifier.n_errors > errors else bool(accepted)

    def _judge_clip(self, frame, det) -> Optional[bool]:
        encoder = getattr(getattr(self.nav, "region_proposer", None), "encoder", None)
        if encoder is None:
            return None  # no encoder loaded on this arm: no opinion
        crop = crop_with_margin(frame.rgb, det.bbox_xyxy)
        feat = encoder.encode_image(crop)
        if feat is None:
            return None
        prompt = "a photo of a {}"
        query = encoder.encode_text([prompt.format(normalize_label(self.nav.target))])[0]
        background = encoder.encode_text([prompt.format(c) for c in self._background()])
        margin = float(feat @ query) - float(np.max(background @ feat))
        return margin > float(self.nav.cfg.verification.stop_gate_clip_margin)

    def _background(self) -> list:
        return background_for(self.nav.target, self.nav.cfg.detector.vocabulary)

    # --------------------------------------------------------------- verdict

    def _reject(self, track) -> None:
        """Identity evidence, recorded the way a failed attempt records it,
        minus the attempt."""
        nav, vc = self.nav, self.nav.cfg.verification
        presence = getattr(nav.object_layer, "presence_filter", None)
        if presence is not None:
            presence.apply_reading(track, False, float(vc.stop_gate_recall),
                                   float(vc.stop_gate_q))
            bar = float(nav.cfg.scene_graph.presence.min_presence)
            bar = min(max(bar, 1e-3), 1.0 - 1e-3)
            under_the_bar = math.log(bar / (1.0 - bar)) - 0.87  # as eval/attempts.py
            track.presence.log_odds = min(float(track.presence.log_odds), under_the_bar)
        track.identity_rejections += 1
        self.rejections_by_track[int(track.id)] = self.rejections_by_track.get(int(track.id), 0) + 1
        self.n_rejected += 1
        nav.release_candidate()
        nav.hold_explore()


def background_for(target: str, vocabulary) -> list:
    """The confusers: the vocabulary the detector runs with for this query
    (`target_vocabulary`), minus the query itself."""
    from ..perception.vocabulary import target_vocabulary

    return [normalize_label(c) for c in target_vocabulary(target, vocabulary)[1:]]


def crop_with_margin(rgb: np.ndarray, bbox, margin: float = 0.3) -> np.ndarray:
    """The box grown by `margin` of its own size each side, clipped to the frame
    -- the crop `scripts/build_verifier_set.py` saves, so offline and online
    scores are of the same thing."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1 = int(max(0, math.floor(x1 - mx))), int(max(0, math.floor(y1 - my)))
    x2, y2 = int(min(w, math.ceil(x2 + mx))), int(min(h, math.ceil(y2 + my)))
    return rgb[y1:y2, x1:x2]
