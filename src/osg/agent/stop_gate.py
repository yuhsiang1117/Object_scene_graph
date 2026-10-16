"""The STOP gate: a second opinion on WHAT the agent is about to stop at.

A STOP is the expensive decision -- it spends one of three attempts -- and on
the YCB benchmarks it is usually taken at the wrong object: 31 of 40 goal
commits on the multi-floor set were false positives, and 49 of 73 live commits
on DualMap's static condition. The pipeline already has a sensor for "it is not
here" (`verification/absence.py`); it has none for "that is not it". A false
positive is a real object, so the detector re-detects it on arrival, the
absence sensor has nothing to say, and the attempt is spent.

So just before a STOP that rests on a live detection of the target, the gate
asks a recogniser that is not the detector -- MobileCLIP's margin of the query
over a fixed confuser list, or a VLM on the full frame with the candidate
boxed -- whether the boxed object is the query. Measured offline
(scripts/eval_verifiers.py, <= 1.5 m): the margin keeps 95% of true targets
and 54% of false positives are rejected; the pipeline's VLM (llama-3.2-11b)
keeps 75% and rejects 44%, at 5-28 s a call.

A "no" is identity evidence: the candidate is released without an attempt, its presence
takes one reading at the gate's own (r, q) through the same filter a failed
attempt uses (`eval/attempts.py`), and its identity count rises by one, so two
rejections retire it exactly as two failed attempts would.

With nothing in the STOP frame -- 49 of 66 STOPs on Benchmark v2's seed 42,
because the approach ends where the object has left the view -- the shipped
gate lets the STOP stand. `stop_gate_view: approach` judges instead the most
confident sighting of this approach within `stop_gate_close_m`, the range the
recogniser was measured at; `stop_gate_unseen: reject` releases a STOP that no
such sighting backs (42 false positives and 3 true targets on those runs). A
stale-anchor stop -- a prior track never seen live -- is let through either
way: `stop_at_stale_anchor_once` takes that gamble on purpose.

What it deliberately does not do:
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
        view = str(getattr(vc, "stop_gate_view", "frame"))
        if view not in ("frame", "approach"):
            raise ValueError(f"verification.stop_gate_view must be frame|approach, got {view!r}")
        unseen = str(getattr(vc, "stop_gate_unseen", "allow"))
        if unseen not in ("allow", "reject"):
            raise ValueError(f"verification.stop_gate_unseen must be allow|reject, got {unseen!r}")
        self._vlm = None
        self.reset()

    def reset(self) -> None:
        self.rejections_by_track: dict = {}
        self.n_rejected = 0
        self.log: list = []
        self.new_approach()

    def new_approach(self) -> None:
        # (score, rgb, detection, range, step) of this approach's most
        # confident sighting within `stop_gate_close_m`.
        self.close_view: Optional[tuple] = None

    @property
    def on(self) -> bool:
        return self.mode != "off"

    def note_sighting(self, frame, det, range_m: Optional[float]) -> None:
        """The approach saw the target. Keep the best close view, for a STOP
        taken where the object has left the frame (`stop_gate_view`)."""
        vc = self.nav.cfg.verification
        if not self.on or str(getattr(vc, "stop_gate_view", "frame")) != "approach":
            return
        if range_m is None or float(range_m) > float(vc.stop_gate_close_m):
            return
        if self.close_view is not None and float(det.score) <= self.close_view[0]:
            return
        self.close_view = (float(det.score), frame.rgb.copy(), det, float(range_m),
                           int(self.nav.step_count))

    def vet_prior(self) -> Optional[dict]:
        """Judge the prior map's target-labelled tracks before the episode.

        Each pass-1 track carrying the target's label is judged on its saved
        best crop and read through the presence filter at the gate's (r, q):
        a "no" leaves a reloaded track under the commit bar, a "yes" raises
        it, and a later live sighting can undo either -- nothing is deleted.
        VLM mode is not used here: a model call per prior track before the
        first step is the stall the control loop must not take.
        """
        vc = self.nav.cfg.verification
        if self.mode != "clip" or not bool(getattr(vc, "stop_gate_vet_prior", False)):
            return None
        layer = getattr(self.nav, "object_layer", None)
        presence = getattr(layer, "presence_filter", None)
        if layer is None:
            return None
        target = normalize_label(self.nav.target)
        out = {"judged": 0, "accepted": 0, "rejected": 0, "no_crop": 0, "no_opinion": 0,
               "tracks": []}
        for track in layer.tracks(include_blacklisted=True):
            if not getattr(track, "from_prior", False) or normalize_label(track.label) != target:
                continue
            crop = getattr(track, "best_crop", None)
            if crop is None or getattr(crop, "size", 0) == 0:
                out["no_crop"] += 1
                continue
            with self.nav.profiler.timeit("stop_gate_vet"):
                verdict = self._clip_verdict(crop)
            if verdict is None:
                out["no_opinion"] += 1
                continue
            out["judged"] += 1
            out["accepted" if verdict else "rejected"] += 1
            out["tracks"].append([int(track.id), bool(verdict)])
            if presence is not None:
                presence.apply_reading(track, bool(verdict), float(vc.stop_gate_recall),
                                       float(vc.stop_gate_q))
        stats = self.nav.stats
        for k in ("judged", "accepted", "rejected"):
            stats[f"prior_vet_{k}"] = stats.get(f"prior_vet_{k}", 0) + out[k]
        return out

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
        rgb, view = getattr(frame, "rgb", None), "frame"
        if det is None and self.close_view is not None:
            _, rgb, det, _, _ = self.close_view
            view = "approach"
        if det is None:
            stale = (bool(getattr(track, "from_prior", False))
                     and not bool(getattr(track, "seen_live", False)))
            if str(getattr(vc, "stop_gate_unseen", "allow")) != "reject" or stale:
                stats["stop_gate_nothing_to_judge"] = (
                    stats.get("stop_gate_nothing_to_judge", 0) + 1)
                return None
            # Never seen within reach of the recogniser: the claim that this
            # is the target rests on a sighting from further than any
            # recogniser -- the detector included -- was measured at.
            stats["stop_gate_unseen_reject"] = stats.get("stop_gate_unseen_reject", 0) + 1
            self.log.append({"step": int(nav.step_count), "track_id": int(track_id),
                             "view": "unseen", "verdict": False})
            stats["stop_gate_log"] = self.log
            self._reject(track)
            return TURN_ACTION
        stats["stop_gate_calls"] = stats.get("stop_gate_calls", 0) + 1
        with nav.profiler.timeit("stop_gate"):
            verdict = self._judge(rgb, det)
        entry = {
            "step": int(nav.step_count), "track_id": int(track_id),
            "label": str(det.label), "score": round(float(det.score), 3),
            "view": view, "verdict": verdict,
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

    def _judge(self, rgb, det) -> Optional[bool]:
        if self.mode == "vlm":
            return self._judge_vlm(rgb, det)
        return self._judge_clip(rgb, det)

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

    def _judge_vlm(self, rgb, det) -> Optional[bool]:
        verifier = self._vlm_verifier()
        errors = verifier.n_errors
        accepted = verifier.verify_bbox(rgb, det.bbox_xyxy, self.nav.target)
        # The verifier fails OPEN on an API error; an error is not an answer.
        return None if verifier.n_errors > errors else bool(accepted)

    def _judge_clip(self, rgb, det) -> Optional[bool]:
        return self._clip_verdict(crop_with_margin(rgb, det.bbox_xyxy))

    def _clip_verdict(self, crop) -> Optional[bool]:
        encoder = getattr(getattr(self.nav, "region_proposer", None), "encoder", None)
        if encoder is None:
            return None  # no encoder loaded on this arm: no opinion
        feat = encoder.encode_image(crop)
        if feat is None:
            return None
        prompt = "a photo of a {}"
        query = encoder.encode_text([prompt.format(normalize_label(self.nav.target))])[0]
        background = encoder.encode_text([prompt.format(c) for c in self._background()])
        margin = float(feat @ query) - float(np.max(background @ feat))
        return margin > float(self.nav.cfg.verification.stop_gate_clip_margin)

    def _background(self) -> list:
        return background_for(self.nav.target, confuser_vocabulary(self.nav.cfg))

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


def generic_confusers() -> list:
    """The indoor background vocabulary plus the 80 COCO classes: a fixed,
    published list that names small household objects without naming any
    benchmark's."""
    from ..core.config.detector import DEFAULT_VOCABULARY
    from ..perception.detector import COCO_CLASSES

    out = list(DEFAULT_VOCABULARY)
    out += [c for c in COCO_CLASSES if c not in out]
    return out


def confuser_vocabulary(cfg) -> list:
    which = str(getattr(cfg.verification, "stop_gate_confusers", "generic"))
    if which == "generic":
        return generic_confusers()
    if which == "detector":
        return list(cfg.detector.vocabulary)
    raise ValueError(f"verification.stop_gate_confusers must be generic|detector, got {which!r}")


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
